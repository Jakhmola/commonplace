"""The retrieval benchmark: the same questions put to each retriever.

Everything here runs against the live corpus. A case names the articles that
answer its question; a retriever is scored on whether those articles are among
the first `k` distinct articles it returns. Scoring is per article, not per
passage, because three passages of one long post should not crowd out the
second article a question needs.
"""

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.config import settings
from commonplace.db.queries import search_passages_bm25, search_passages_dense
from commonplace.evals.metrics import mrr, recall_at_k
from commonplace.retrieval.graph import search_graph_claims
from commonplace.retrieval.hybrid import compute_rrf

VARIANTS = ("sparse_bm25", "dense_pgvector", "hybrid_2way_rrf_k60", "hybrid_3way_rrf_k60")

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Passages fetched per engine. Deeper than `k`, so that `k` distinct articles
# can be found even when one article contributes several passages.
_DEPTH_PER_K = 4


def load_benchmark(path: str | Path | None = None) -> dict[str, Any]:
    file = Path(path or settings.eval_benchmark_file)
    data = yaml.safe_load((file if file.is_absolute() else _REPO_ROOT / file).read_text())
    return {"suite_name": data["suite_name"], "k": int(data["k"]), "cases": list(data["cases"])}


def missing_expected(cases: list[dict[str, Any]], stored_ids: set[str]) -> list[str]:
    """Expected articles that are not in the corpus. A case that points at one
    cannot be answered by any retriever, which is a fact about the corpus, not
    a retrieval miss.
    """
    expected = {cid for case in cases for cid in case["expected_canonical_ids"]}
    return sorted(expected - stored_ids)


async def retrieve_variant(
    session: AsyncSession, variant: str, question: str, query_embedding: list[float], k: int
) -> list[dict[str, Any]]:
    """The passages one retrieval variant returns for a question, best first."""
    depth = k * _DEPTH_PER_K
    if variant == "sparse_bm25":
        return await search_passages_bm25(session, question, depth)
    if variant == "dense_pgvector":
        return await search_passages_dense(session, query_embedding, depth)

    sparse = await search_passages_bm25(session, question, depth)
    dense = await search_passages_dense(session, query_embedding, depth)
    graph = (
        await search_graph_claims(session, question, depth)
        if variant == "hybrid_3way_rrf_k60"
        else None
    )
    return [asdict(c) for c in compute_rrf(sparse, dense, graph, settings.rrf_k)]


def score_case(case: dict[str, Any], hits: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """How one variant did on one case, at article level."""
    articles = list(dict.fromkeys(h["canonical_id"] for h in hits if h.get("canonical_id")))[:k]
    expected = list(case["expected_canonical_ids"])
    return {
        "case_id": case["case_id"],
        "category": case.get("category"),
        "expected_canonical_ids": expected,
        "retrieved_canonical_ids": articles,
        "recall_at_k": recall_at_k(articles, expected, k),
        "mrr": mrr(articles, expected, k),
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarise(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Averages over all cases and per category."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in case_results:
        by_category[result["category"]].append(result)

    def averages(results: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(results),
            "recall_at_k": _mean([r["recall_at_k"] for r in results]),
            "mrr": _mean([r["mrr"] for r in results]),
        }

    return {
        **averages(case_results),
        # Needing articles from different feeds is the case the graph was added for.
        "cross_publication_recall": _mean(
            [r["recall_at_k"] for r in by_category.get("cross_publication", [])]
        ),
        "by_category": {name: averages(results) for name, results in sorted(by_category.items())},
    }
