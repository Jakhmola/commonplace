"""Editing an article must supersede its passages, not overwrite or duplicate them."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import Passage
from commonplace.db.queries import (
    count_current_passages,
    search_passages_bm25,
    store_passages,
    upsert_article,
    upsert_publication,
)
from commonplace.functions.ingest import persist_article
from commonplace.schemas.events import ArticleDiscovered
from commonplace.sources.embedder import pack
from commonplace.sources.parser import ParsedArticle

VECTOR = [0.1] * 1536
URL = "https://history-test.example/p/an-edited-post"


async def _article(session: AsyncSession) -> uuid.UUID:
    publication_id = await upsert_publication(
        session,
        slug="history-test",
        name="History Test",
        feed_url="https://history-test.example/feed",
        homepage="https://history-test.example",
    )
    return await upsert_article(
        session,
        canonical_id="pub:history-test:an-edited-post",
        publication_id=publication_id,
        url=URL,
        title="An edited post",
        author=None,
        published_at=datetime(2026, 11, 17, tzinfo=UTC),
        content_hash="v1",
        text="first version",
    )


def _passages(*texts: str) -> list[dict[str, Any]]:
    return [{"chunk_index": i, "text": t, "embedding": VECTOR} for i, t in enumerate(texts)]


async def _all_rows(session: AsyncSession, article_id: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(Passage).where(Passage.article_id == article_id)
    return (await session.execute(stmt)).scalar_one()


async def _found(session: AsyncSession, query: str) -> list[str]:
    hits = await search_passages_bm25(session, query, 25)
    return sorted(h["text"] for h in hits if h["url"] == URL)  # equal ranks: order is by id


async def test_an_edit_closes_old_passages_and_search_sees_only_the_new_ones(
    db: AsyncSession,
) -> None:
    article_id = await _article(db)
    await store_passages(
        db, article_id=article_id, passages=_passages("quokkazebra one", "quokkazebra two")
    )
    assert await _found(db, "quokkazebra") == ["quokkazebra one", "quokkazebra two"]

    await store_passages(db, article_id=article_id, passages=_passages("wombatokapi rewritten"))

    assert await _all_rows(db, article_id) == 3  # history is kept
    assert await count_current_passages(db, article_id) == 1
    assert await _found(db, "quokkazebra") == []  # superseded passages are invisible
    assert await _found(db, "wombatokapi") == ["wombatokapi rewritten"]


async def test_persisting_the_same_version_twice_keeps_one_current_set(db: AsyncSession) -> None:
    # Inngest re-runs a step whose response was lost after its transaction committed.
    publication_id = await upsert_publication(
        db,
        slug="history-test",
        name="History Test",
        feed_url="https://history-test.example/feed",
        homepage="https://history-test.example",
    )
    payload = ArticleDiscovered(
        canonical_id="pub:history-test:persisted-twice",
        publication_id=publication_id,
        url=URL,
        title="Persisted twice",
        published_at=datetime(2026, 11, 17, tzinfo=UTC),
        via="manual",
    )
    article = ParsedArticle(
        title="Persisted twice",
        author=None,
        published_at=datetime(2026, 11, 17, tzinfo=UTC),
        text="one two",
        content_hash="same-hash",
    )
    chunks, vectors = ["one", "two"], [pack(VECTOR), pack(VECTOR)]

    first = await persist_article(db, payload, URL, article, chunks, vectors)
    second = await persist_article(db, payload, URL, article, chunks, vectors)

    assert first == second
    article_id = uuid.UUID(first["article_id"])
    assert await _all_rows(db, article_id) == 2
    assert await count_current_passages(db, article_id) == 2
