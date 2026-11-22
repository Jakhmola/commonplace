"""Retrieval against the real Postgres: what full-text search matches, and how hybrid degrades."""

import uuid
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.queries import search_passages_bm25, store_passages
from commonplace.retrieval import hybrid
from tests.integration.conftest import ARTICLE_URL, passages


async def _found(session: AsyncSession, query: str) -> list[str]:
    hits = await search_passages_bm25(session, query, 25)
    return [h["text"] for h in hits if h["url"] == ARTICLE_URL]


async def test_a_question_still_matches_on_its_rare_terms(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    await store_passages(
        db, article_id=article_id, passages=passages("an overview of the quokkazebra framework")
    )
    # Requiring every word of a question to appear in one passage matches almost nothing.
    question = "How does the quokkazebra framework compare to wombatokapi for agents?"
    assert await _found(db, question) == ["an overview of the quokkazebra framework"]


async def test_passages_matching_more_of_the_query_rank_higher(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    await store_passages(
        db,
        article_id=article_id,
        passages=passages("quokkazebra only", "quokkazebra and wombatokapi together"),
    )
    assert await _found(db, "quokkazebra wombatokapi") == [
        "quokkazebra and wombatokapi together",
        "quokkazebra only",
    ]


async def test_a_quoted_query_keeps_exact_phrase_matching(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    await store_passages(
        db, article_id=article_id, passages=passages("the quokkazebra framework overview")
    )
    assert await _found(db, '"quokkazebra framework"') == ["the quokkazebra framework overview"]
    assert await _found(db, '"framework quokkazebra"') == []


async def test_hybrid_falls_back_to_keyword_search_when_embedding_fails(
    db: AsyncSession, article_id: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    await store_passages(db, article_id=article_id, passages=passages("quokkazebra in practice"))

    async def provider_is_down(_: list[str]) -> dict[str, Any]:
        raise RuntimeError("embedding provider is down")

    monkeypatch.setattr(hybrid, "embed_texts", provider_is_down)
    top, candidates = await hybrid.hybrid_search(db, "quokkazebra")
    assert [c["text"] for c in top] == ["quokkazebra in practice"]
    assert {c["source"] for c in candidates} == {"bm25"}
