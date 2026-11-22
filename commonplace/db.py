"""All SQL lives here. Plain SQL over one async engine; the schema is in alembic/."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from commonplace.config import settings
from commonplace.sources.allowlist import Publication

engine = create_async_engine(settings.database_url, pool_pre_ping=True)


@asynccontextmanager
async def tx() -> AsyncIterator[AsyncConnection]:
    """One transaction: commits on success, rolls back on any exception."""
    async with engine.begin() as conn:
        yield conn


def vec(v: list[float]) -> str:
    """pgvector's text form; cast with CAST(:x AS vector) in SQL."""
    return "[" + ",".join(map(str, v)) + "]"


async def upsert_publication(conn: AsyncConnection, pub: Publication) -> UUID:
    row = await conn.execute(
        text("""
            INSERT INTO publications (slug, name, feed_url, homepage)
            VALUES (:slug, :name, :feed_url, :homepage)
            ON CONFLICT (slug) DO UPDATE
              SET name = EXCLUDED.name, feed_url = EXCLUDED.feed_url, homepage = EXCLUDED.homepage
            RETURNING id
        """),
        pub.model_dump(),
    )
    return UUID(str(row.scalar_one()))


async def stored_canonical_ids(conn: AsyncConnection, ids: list[str]) -> set[str]:
    rows = await conn.execute(
        text("SELECT canonical_id FROM articles WHERE canonical_id = ANY(:ids)"), {"ids": ids}
    )
    return set(rows.scalars())


async def create_job(
    conn: AsyncConnection, event_id: str, publication_id: UUID, canonical_ids: list[str]
) -> None:
    await conn.execute(
        text("""
            INSERT INTO jobs (event_id, publication_id, canonical_ids)
            VALUES (:e, :p, :ids) ON CONFLICT (event_id) DO NOTHING
        """),
        {"e": event_id, "p": publication_id, "ids": canonical_ids},
    )


async def job_progress(conn: AsyncConnection, event_id: str) -> tuple[int, int] | None:
    """(done, total) for a discovery job, counted live from `articles`."""
    row = (
        await conn.execute(
            text("""
                SELECT cardinality(j.canonical_ids) AS total,
                       (SELECT count(*) FROM articles a
                        WHERE a.canonical_id = ANY(j.canonical_ids)) AS done
                FROM jobs j WHERE j.event_id = :e
            """),
            {"e": event_id},
        )
    ).one_or_none()
    return None if row is None else (row.done, row.total)


async def save_article(
    conn: AsyncConnection,
    *,
    canonical_id: str,
    publication_id: UUID,
    url: str,
    title: str,
    author: str | None,
    published_at: datetime | None,
    content_hash: str,
    body: str,
    chunks: list[str],
    vectors: list[list[float]],
    embed_tokens: int,
    embed_usd: float,
) -> UUID:
    """Upsert the article and its passages, and log the embedding cost, atomically.

    Idempotent: articles key on canonical_id, passages on (article_id, chunk_index),
    so saving the same article twice changes nothing.
    """
    article_id = UUID(
        str(
            (
                await conn.execute(
                    text("""
                        INSERT INTO articles (canonical_id, publication_id, url, title, author,
                                              published_at, content_hash, body)
                        VALUES (:cid, :pid, :url, :title, :author, :published_at, :hash, :body)
                        ON CONFLICT (canonical_id) DO UPDATE SET
                          url = EXCLUDED.url, title = EXCLUDED.title, author = EXCLUDED.author,
                          published_at = EXCLUDED.published_at,
                          content_hash = EXCLUDED.content_hash, body = EXCLUDED.body,
                          updated_at = now()
                        RETURNING id
                    """),
                    {
                        "cid": canonical_id,
                        "pid": publication_id,
                        "url": url,
                        "title": title,
                        "author": author,
                        "published_at": published_at,
                        "hash": content_hash,
                        "body": body,
                    },
                )
            ).scalar_one()
        )
    )
    await conn.execute(
        text("DELETE FROM passages WHERE article_id = :a AND chunk_index >= :n"),
        {"a": article_id, "n": len(chunks)},
    )
    if chunks:
        await conn.execute(
            text("""
                INSERT INTO passages (article_id, chunk_index, body, embedding)
                VALUES (:a, :i, :body, CAST(:e AS vector))
                ON CONFLICT (article_id, chunk_index) DO UPDATE
                  SET body = EXCLUDED.body, embedding = EXCLUDED.embedding
            """),
            [
                {"a": article_id, "i": i, "body": c, "e": vec(v)}
                for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
    await conn.execute(
        text("""
            INSERT INTO costs (kind, model, tokens, usd, article_id)
            VALUES ('embed', :m, :t, :u, :a)
        """),
        {"m": settings.embedding_model, "t": embed_tokens, "u": embed_usd, "a": article_id},
    )
    return article_id


_HIT_COLUMNS = """
    p.id AS passage_id, p.chunk_index, p.body,
    a.title, a.url, a.author, a.published_at, pub.name AS publication
"""
_HIT_JOINS = (
    "JOIN articles a ON a.id = p.article_id JOIN publications pub ON pub.id = a.publication_id"
)


async def search_sparse(conn: AsyncConnection, query: str, k: int) -> list[dict[str, Any]]:
    """Postgres full-text search, ranked by ts_rank_cd (cover density)."""
    rows = await conn.execute(
        text(f"""
            SELECT {_HIT_COLUMNS}, ts_rank_cd(p.tsv, q) AS score
            FROM passages p {_HIT_JOINS}, websearch_to_tsquery('english', :q) q
            WHERE p.tsv @@ q
            ORDER BY score DESC, p.id
            LIMIT :k
        """),
        {"q": query, "k": k},
    )
    return [dict(r._mapping) for r in rows]


async def search_dense(conn: AsyncConnection, vector: list[float], k: int) -> list[dict[str, Any]]:
    """Nearest passages by cosine distance (HNSW index)."""
    rows = await conn.execute(
        text(f"""
            SELECT {_HIT_COLUMNS}, 1 - (p.embedding <=> CAST(:v AS vector)) AS score
            FROM passages p {_HIT_JOINS}
            ORDER BY p.embedding <=> CAST(:v AS vector)
            LIMIT :k
        """),
        {"v": vec(vector), "k": k},
    )
    return [dict(r._mapping) for r in rows]


async def counts(conn: AsyncConnection) -> dict[str, int]:
    row = (
        await conn.execute(
            text("""
                SELECT (SELECT count(*) FROM articles) AS articles,
                       (SELECT count(*) FROM passages) AS passages
            """)
        )
    ).one()
    return dict(row._mapping)
