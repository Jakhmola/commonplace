"""Retrieval against the real Postgres: what full-text search matches, and how hybrid degrades."""

import uuid
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.queries import search_passages_bm25, store_passages
from commonplace.retrieval import graph as graph_retrieval
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
    hits = await search_passages_bm25(db, question, 5)
    # Hundreds of passages contain "agents" or "compare"; the one with the rare term wins.
    assert hits[0]["text"] == "an overview of the quokkazebra framework"
    assert hits[0]["url"] == ARTICLE_URL


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


async def test_graph_claims_come_back_as_the_passages_holding_their_quotes(
    db: AsyncSession, article_id: uuid.UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    await store_passages(
        db,
        article_id=article_id,
        passages=passages("an unrelated opening", "the quokkazebra framework scales linearly"),
    )

    async def graph_returns(**_: Any) -> list[dict[str, Any]]:
        claim = {
            "article_id": str(article_id),
            "claim_text": "Quokkazebra scales linearly",
            "claim_type": "measurement",
            "confidence": 0.9,
            "matched_topics": 1,
        }
        return [
            {**claim, "claim_id": "c-1", "quote_span": "quokkazebra framework scales linearly"},
            # a second claim quoted from the same passage adds nothing new to retrieve
            {**claim, "claim_id": "c-2", "quote_span": "framework scales linearly"},
            # a claim whose article is gone is skipped
            {**claim, "claim_id": "c-3", "article_id": str(uuid.uuid4()), "quote_span": "x" * 20},
        ]

    monkeypatch.setattr(graph_retrieval, "search_graph_by_topics", graph_returns)
    (hit,) = await graph_retrieval.search_graph_claims(db, "quokkazebra")
    assert hit["text"] == "the quokkazebra framework scales linearly"
    assert (hit["claim_id"], hit["claim_type"]) == ("c-1", "measurement")
    assert hit["url"] == ARTICLE_URL
