"""Hybrid retrieval: keyword search and vector search, fused by rank.

Neither is enough alone. Full-text search finds exact tokens (a function
name, a version string) and misses paraphrases; vector search understands
meaning and blurs exact tokens. Reciprocal Rank Fusion merges the two lists
using only each passage's position in each list, so two scores that live on
different scales are never added together.
"""

import asyncio
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.config import settings
from commonplace.db.queries import search_passages_bm25, search_passages_dense
from commonplace.sources.embedder import embed_texts


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
    source: str = "bm25"  # "bm25", "vector" or "both"


def compute_rrf(
    sparse_results: list[dict[str, Any]], dense_results: list[dict[str, Any]], k: int = 60
) -> list[RetrievedCandidate]:
    """score = sum over the engines that returned the passage of 1 / (k + rank).

    A passage both engines found collects points twice, so agreement between
    two very different engines beats being the favourite of one. Ties break
    on passage id, so the same question always gives the same order.
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
        c.source = "both" if c.sparse_rank is not None else "vector"

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
    # One session cannot run two queries at once. The slow part is the network
    # call that embeds the question, so that overlaps the full-text query and
    # the vector query runs when the embedding is back.
    sparse_results, embedded = await asyncio.gather(
        search_passages_bm25(session, query, per_engine), embed_texts([query])
    )
    (query_embedding,) = embedded["embeddings"]
    dense_results = await search_passages_dense(session, query_embedding, per_engine)

    fused = [asdict(c) for c in compute_rrf(sparse_results, dense_results, k or settings.rrf_k)]
    return fused[:limit], fused


async def search(session: AsyncSession, query: str, limit: int) -> list[dict[str, Any]]:
    top, _ = await hybrid_search(session, query, limit=limit)
    return top
