"""Every query in the project. Functions and endpoints call these; they never build SQL."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Row, Select, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import Article, Cost, Job, Passage, Publication

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


async def replace_passages(
    session: AsyncSession, *, article_id: uuid.UUID, passages: list[dict[str, Any]]
) -> int:
    """Upsert by `(article_id, chunk_index)` and drop any tail left from a longer version.

    Saving the same article twice changes nothing.
    """
    await session.execute(
        delete(Passage).where(
            Passage.article_id == article_id, Passage.chunk_index >= len(passages)
        )
    )
    if not passages:
        return 0
    stmt = insert(Passage).values(
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
    stmt = stmt.on_conflict_do_update(
        index_elements=[Passage.article_id, Passage.chunk_index],
        set_={"text": stmt.excluded.text, "embedding": stmt.excluded.embedding},
    )
    await session.execute(stmt)
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
            status="running",
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


async def search_passages_bm25(
    session: AsyncSession, query: str, limit: int
) -> list[dict[str, Any]]:
    """Full-text search over the generated `tsv` column (GIN index)."""
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank(Passage.tsv, tsquery)
    stmt = (
        _passage_hits(rank)
        .where(Passage.tsv.op("@@")(tsquery))
        .order_by(rank.desc(), Passage.id)
        .limit(limit)
    )
    return [_hit(row) for row in (await session.execute(stmt)).all()]


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
