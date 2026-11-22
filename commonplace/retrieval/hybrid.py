"""Hybrid retrieval: keyword search, vector search and the claim graph, fused by rank.

None is enough alone. Full-text search finds exact tokens (a function name, a
version string) and misses paraphrases. Vector search understands meaning and
blurs exact tokens. The graph connects claims from different publications
through shared topics and has no notion of similarity at all. Reciprocal Rank
Fusion merges the three lists using only each passage's position in each list,
so scores that live on different scales, or do not exist, are never added.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.config import settings
from commonplace.db.queries import search_passages_bm25, search_passages_dense
from commonplace.retrieval.graph import search_graph_claims
from commonplace.sources.embedder import embed_texts

logger = logging.getLogger(__name__)


@dataclass
class RetrievedCandidate:
    passage_id: str
    text: str
    title: str
    author: str | None
    url: str
    published_at: str
    rrf_score: float = 0.0
    sparse_rank: int | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    graph_rank: int | None = None
    # The claim that led the graph to this passage, if it was reached that way.
    claim_id: str | None = None
    claim_text: str | None = None
    claim_type: str | None = None
    # Which engines found it: bm25, vector, both, graph, bm25+graph, vector+graph or all.
    source: str = "bm25"


def _source(c: RetrievedCandidate) -> str:
    text_engines = {(True, True): "both", (True, False): "bm25", (False, True): "vector"}.get(
        (c.sparse_rank is not None, c.dense_rank is not None)
    )
    if c.graph_rank is None:
        return text_engines or "bm25"
    if text_engines is None:
        return "graph"
    return "all" if text_engines == "both" else f"{text_engines}+graph"


def compute_rrf(
    sparse_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]] | None = None,
    k: int = 60,
) -> list[RetrievedCandidate]:
    """score = sum over the engines that returned the passage of 1 / (k + rank).

    A passage found by several engines collects points from each, so agreement
    between very different engines beats being the favourite of one. Only
    positions are used: the graph has no similarity score at all, and it still
    fuses cleanly. Ties break on passage id, so the same question always gives
    the same order.
    """
    candidates: dict[str, RetrievedCandidate] = {}

    def candidate(hit: dict[str, Any]) -> RetrievedCandidate:
        fields = ("passage_id", "text", "title", "author", "url", "published_at")
        return candidates.setdefault(
            hit["passage_id"], RetrievedCandidate(**{f: hit[f] for f in fields})
        )

    for rank, hit in enumerate(sparse_results, start=1):
        c = candidate(hit)
        c.sparse_rank, c.sparse_score = rank, float(hit["rank"])
        c.rrf_score += 1 / (k + rank)
    for rank, hit in enumerate(dense_results, start=1):
        c = candidate(hit)
        c.dense_rank, c.dense_score = rank, float(hit["rank"])
        c.rrf_score += 1 / (k + rank)
    # Several claims can be quoted from one passage. It takes one position in the
    # graph ranking, with the best-ranked of those claims.
    by_passage: dict[str, dict[str, Any]] = {}
    for hit in graph_results or []:
        by_passage.setdefault(hit["passage_id"], hit)
    for rank, hit in enumerate(by_passage.values(), start=1):
        c = candidate(hit)
        c.graph_rank = rank
        c.claim_id, c.claim_text, c.claim_type = (
            hit.get("claim_id"),
            hit.get("claim_text"),
            hit.get("claim_type"),
        )
        c.rrf_score += 1 / (k + rank)

    for c in candidates.values():
        c.source = _source(c)
    return sorted(candidates.values(), key=lambda c: (-c.rrf_score, c.passage_id))


async def hybrid_search(
    session: AsyncSession,
    query: str,
    limit: int = 5,
    fetch_multiplier: int = 3,
    k: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(the top `limit` passages, every fused candidate).

    Callers that only answer need the first; a snapshot keeps the second.
    """
    per_engine = max(limit * fetch_multiplier, 15)

    # The question is embedded before the session is touched, so no connection
    # is held during the network call. If the embedding provider is down the
    # search degrades to keywords only instead of failing.
    query_embedding: list[float] | None = None
    try:
        (query_embedding,) = (await embed_texts([query]))["embeddings"]
    except Exception:
        logger.exception("Could not embed the query; falling back to keyword search")

    sparse_results = await search_passages_bm25(session, query, per_engine)
    dense_results = (
        await search_passages_dense(session, query_embedding, per_engine)
        if query_embedding is not None
        else []
    )

    graph_results = await search_graph_claims(session, query, settings.graph_retrieval_limit)

    fused = [
        asdict(c)
        for c in compute_rrf(sparse_results, dense_results, graph_results, k or settings.rrf_k)
    ]
    return fused[:limit], fused


async def search(session: AsyncSession, query: str, limit: int) -> list[dict[str, Any]]:
    top, _ = await hybrid_search(session, query, limit=limit)
    return top
