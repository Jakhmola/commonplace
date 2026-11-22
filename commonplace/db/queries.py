"""Every query in the project. Functions and endpoints call these; they never build SQL."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Row, Select, func, select, update
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import (
    Article,
    Cost,
    Cursor,
    Failure,
    Job,
    Passage,
    Publication,
    Snapshot,
)

# --- publications ---


async def upsert_publication(
    session: AsyncSession, *, slug: str, name: str, feed_url: str, homepage: str
) -> uuid.UUID:
    fields = {"name": name, "feed_url": feed_url, "homepage": homepage}
    stmt = (
        insert(Publication)
        .values(id=uuid.uuid4(), slug=slug, **fields)
        .on_conflict_do_update(index_elements=[Publication.slug], set_=fields)
        .returning(Publication.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def get_publication_by_slug(session: AsyncSession, slug: str) -> Publication | None:
    return (
        await session.execute(select(Publication).where(Publication.slug == slug))
    ).scalar_one_or_none()


async def list_publications(session: AsyncSession) -> list[Publication]:
    return list((await session.execute(select(Publication).order_by(Publication.slug))).scalars())


# --- polling cursors ---


async def get_cursor(session: AsyncSession, publication_id: uuid.UUID) -> Cursor | None:
    return await session.get(Cursor, publication_id)


async def record_poll(
    session: AsyncSession, *, publication_id: uuid.UUID, not_modified: bool
) -> None:
    """Count one poll of a feed, and whether it found the feed unchanged."""
    unchanged = 1 if not_modified else 0
    stmt = (
        insert(Cursor)
        .values(
            publication_id=publication_id,
            last_checked_at=func.now(),
            poll_count=1,
            not_modified_count=unchanged,
        )
        .on_conflict_do_update(
            index_elements=[Cursor.publication_id],
            set_={
                "last_checked_at": func.now(),
                "poll_count": Cursor.poll_count + 1,
                "not_modified_count": Cursor.not_modified_count + unchanged,
            },
        )
    )
    await session.execute(stmt)


async def save_validators(
    session: AsyncSession,
    *,
    publication_id: uuid.UUID,
    etag: str | None,
    last_modified: str | None,
) -> None:
    """Remember the feed version we have fully dealt with. The cursor row must exist."""
    cursor = await session.get(Cursor, publication_id)
    if cursor is not None:
        cursor.etag, cursor.last_modified = etag, last_modified


# --- articles and passages ---


async def get_existing_canonical_ids(session: AsyncSession, canonical_ids: list[str]) -> set[str]:
    stmt = select(Article.canonical_id).where(Article.canonical_id.in_(canonical_ids))
    return set((await session.execute(stmt)).scalars())


async def upsert_article(
    session: AsyncSession,
    *,
    canonical_id: str,
    publication_id: uuid.UUID,
    url: str,
    title: str,
    author: str | None,
    published_at: datetime,
    content_hash: str,
    text: str,
) -> uuid.UUID:
    """Keyed on `canonical_id`, the dedupe key shared by every ingestion path."""
    fields = {
        "url": url,
        "title": title,
        "author": author,
        "published_at": published_at,
        "content_hash": content_hash,
        "text": text,
    }
    stmt = (
        insert(Article)
        .values(id=uuid.uuid4(), canonical_id=canonical_id, publication_id=publication_id, **fields)
        .on_conflict_do_update(
            index_elements=[Article.canonical_id], set_={**fields, "updated_at": func.now()}
        )
        .returning(Article.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def get_article_by_canonical_id(session: AsyncSession, canonical_id: str) -> Article | None:
    return (
        await session.execute(select(Article).where(Article.canonical_id == canonical_id))
    ).scalar_one_or_none()


async def count_current_passages(session: AsyncSession, article_id: uuid.UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(Passage)
        .where(Passage.article_id == article_id, Passage.valid_to.is_(None))
    )
    return (await session.execute(stmt)).scalar_one()


async def store_passages(
    session: AsyncSession, *, article_id: uuid.UUID, passages: list[dict[str, Any]]
) -> int:
    """Make `passages` the article's current version.

    Whatever was current is closed with `valid_to = now()`, never deleted or
    overwritten, so the history of an edited article stays queryable. For a
    new article there is nothing to close.
    """
    await session.execute(
        update(Passage)
        .where(Passage.article_id == article_id, Passage.valid_to.is_(None))
        .values(valid_to=func.now())
    )
    if passages:
        await session.execute(
            insert(Passage).values(
                [
                    {
                        "id": uuid.uuid4(),
                        "article_id": article_id,
                        "chunk_index": p["chunk_index"],
                        "text": p["text"],
                        "embedding": p["embedding"],
                    }
                    for p in passages
                ]
            )
        )
    return len(passages)


async def count_articles(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(Article))).scalar_one()


async def count_passages(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(Passage))).scalar_one()


async def count_articles_for_publication(session: AsyncSession, publication_id: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(Article).where(Article.publication_id == publication_id)
    return (await session.execute(stmt)).scalar_one()


# --- jobs ---


async def create_job(
    session: AsyncSession,
    *,
    event_id: str,
    kind: str,
    canonical_ids: list[str],
    publication_id: uuid.UUID | None = None,
) -> None:
    stmt = (
        insert(Job)
        .values(
            event_id=event_id,
            kind=kind,
            # A replay that finds nothing new is finished the moment it is recorded.
            status="running" if canonical_ids else "completed",
            total=len(canonical_ids),
            done=0,
            publication_id=publication_id,
            canonical_ids=canonical_ids,
        )
        .on_conflict_do_nothing(index_elements=[Job.event_id])
    )
    await session.execute(stmt)


async def get_job(session: AsyncSession, event_id: str) -> Job | None:
    return await session.get(Job, event_id)


async def get_job_status(session: AsyncSession, event_id: str) -> dict[str, Any] | None:
    """Progress counted from the articles that actually exist, then written back.

    `ingest-article` never learns which job asked for it, so nothing increments
    a counter: `done` is how many of the job's expected articles are stored.
    The caller commits to keep the repaired `done`/`status`.
    """
    job = await get_job(session, event_id)
    if job is None:
        return None
    job.done = len(await get_existing_canonical_ids(session, job.canonical_ids))
    if job.done >= job.total:
        job.status = "completed"
    return {
        "event_id": job.event_id,
        "kind": job.kind,
        "status": job.status,
        "total": job.total,
        "done": job.done,
    }


# --- costs ---


async def insert_cost(
    session: AsyncSession,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    est_usd: float,
    article_id: uuid.UUID | None = None,
) -> None:
    session.add(
        Cost(
            article_id=article_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            est_usd=est_usd,
        )
    )


# --- search ---


def _passage_hits(score: Any) -> Select[Any]:
    """Current passages joined to their article, with `score` as `rank` (higher is better)."""
    return (
        select(
            Passage.id,
            Passage.text,
            Article.title,
            Article.author,
            Article.url,
            Article.published_at,
            score.label("rank"),
        )
        .join(Article, Article.id == Passage.article_id)
        .where(Passage.valid_to.is_(None))
    )


def _hit(row: Row[Any]) -> dict[str, Any]:
    return {
        "passage_id": str(row.id),
        "text": row.text,
        "title": row.title,
        "author": row.author,
        "url": row.url,
        "published_at": row.published_at.isoformat(),
        "rank": float(row.rank),
    }


# Each term of the query that a passage contains adds that term's rarity (inverse
# document frequency, in BM25's form) to the passage's score. A passage matches
# if it has any of the terms, so a question-shaped query still finds something,
# and a rare term outweighs any number of common ones.
# ponytail: term frequency and passage length are ignored (passages are all about
# the same size); use a real BM25 index if ranking within one term starts to matter.
_KEYWORD_SEARCH = sql_text("""
    WITH corpus AS (
        SELECT count(*)::float AS total FROM passages WHERE valid_to IS NULL
    ), terms AS (
        SELECT quote_literal(lexeme)::tsquery AS term
        FROM unnest(tsvector_to_array(to_tsvector('english', :query))) AS lexeme
    ), weighted AS (
        SELECT term, ln(1 + (total - df + 0.5) / (df + 0.5)) AS idf
        FROM corpus, terms, LATERAL (
            SELECT count(*) AS df FROM passages WHERE valid_to IS NULL AND tsv @@ term
        ) AS frequency
        WHERE df > 0
    )
    SELECT p.id, p.text, a.title, a.author, a.url, a.published_at, sum(w.idf) AS rank
    FROM weighted w
    JOIN passages p ON p.valid_to IS NULL AND p.tsv @@ w.term
    JOIN articles a ON a.id = p.article_id
    GROUP BY p.id, a.id
    ORDER BY rank DESC, p.id
    LIMIT :limit
""")


async def search_passages_bm25(
    session: AsyncSession, query: str, limit: int
) -> list[dict[str, Any]]:
    """Keyword search over the generated `tsv` column (GIN index).

    A query with double quotes is taken literally: phrases must match exactly
    and every term is required. Anything else is ranked by term rarity.
    """
    if '"' in query:
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank(Passage.tsv, tsquery)
        stmt = (
            _passage_hits(rank)
            .where(Passage.tsv.op("@@")(tsquery))
            .order_by(rank.desc(), Passage.id)
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
    else:
        rows = (await session.execute(_KEYWORD_SEARCH, {"query": query, "limit": limit})).all()
    return [_hit(row) for row in rows]


async def search_passages_dense(
    session: AsyncSession, query_embedding: list[float], limit: int
) -> list[dict[str, Any]]:
    """Nearest passages by cosine distance (HNSW index). `rank` is cosine similarity."""
    distance = Passage.embedding.cosine_distance(query_embedding)
    stmt = (
        _passage_hits(1 - distance)
        .where(Passage.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(limit)
    )
    return [_hit(row) for row in (await session.execute(stmt)).all()]


# --- snapshots ---


async def save_snapshot(
    session: AsyncSession,
    *,
    snapshot_id: str,
    question: str,
    answer_text: str,
    citations: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    retriever_variant: str,
    as_of: datetime | None,
    latency_ms: int,
) -> None:
    session.add(
        Snapshot(
            snapshot_id=snapshot_id,
            question=question,
            answer_text=answer_text,
            citations=citations,
            retrieved=retrieved,
            retriever_variant=retriever_variant,
            as_of=as_of,
            latency_ms=latency_ms,
        )
    )


async def get_snapshot(session: AsyncSession, snapshot_id: str) -> Snapshot | None:
    return await session.get(Snapshot, snapshot_id)
