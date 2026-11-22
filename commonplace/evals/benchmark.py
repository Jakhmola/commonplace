"""The retrieval benchmark: the same questions put to each retriever.

Everything here runs against the live corpus, and the hybrid variants run
through the very function that `/search` and `/ask` use. A case names the
articles that answer its question; a retriever is scored on whether those
articles are among the first `k` passages it returns, because those passages
are what would reach the prompt. Three passages of one long post count as one
article, and they also take three of the `k` places: a retriever that fills
the prompt from a single post is scored as doing so.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.config import settings
from commonplace.db.queries import search_passages_bm25, search_passages_dense
from commonplace.evals.metrics import mrr, recall_at_k
from commonplace.retrieval.hybrid import hybrid_search

VARIANTS = ("sparse_bm25", "dense_pgvector", "hybrid_2way_rrf_k60", "hybrid_3way_rrf_k60")

_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def scorable_cases(cases: list[dict[str, Any]], missing: list[str]) -> list[dict[str, Any]]:
    """The cases as they can be scored on this corpus: expected articles that
    are not stored are left out, and a case with none left is dropped.
    """
    gone = set(missing)
    kept = []
    for case in cases:
        expected = [cid for cid in case["expected_canonical_ids"] if cid not in gone]
        if expected:
            kept.append({**case, "expected_canonical_ids": expected})
    return kept


async def retrieve_variant(
    session: AsyncSession, variant: str, question: str, query_embedding: list[float], k: int
) -> list[dict[str, Any]]:
    """The `k` passages one retrieval variant would put in a prompt, best first."""
    if variant == "sparse_bm25":
        return await search_passages_bm25(session, question, k)
    if variant == "dense_pgvector":
        return await search_passages_dense(session, query_embedding, k)
    top, _ = await hybrid_search(
        session,
        question,
        limit=k,
        query_embedding=query_embedding,
        use_graph=variant == "hybrid_3way_rrf_k60",
    )
    return top


def score_case(case: dict[str, Any], hits: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """How one variant did on one case: which articles its first `k` passages come from."""
    shown = hits[:k]
    articles = list(dict.fromkeys(h["canonical_id"] for h in shown if h.get("canonical_id")))
    expected = list(case["expected_canonical_ids"])
    return {
        "case_id": case["case_id"],
        "category": case.get("category"),
        "expected_canonical_ids": expected,
        "retrieved_canonical_ids": articles,
        "recall_at_k": recall_at_k(articles, expected, k),
        "mrr": mrr(articles, expected, k),
        # How many of those passages the claim graph found (alone or with another engine).
        "graph_hits": sum(1 for h in shown if h.get("graph_rank") is not None),
    }


def suite_is_degraded(*, graph_status: str, three_way_graph_hits: int) -> bool:
    """True when the three-way variant ran without its graph leg. Graph retrieval
    fails soft (an unreachable Memgraph returns nothing), so without this the
    three-way row would silently repeat the two-way row.
    """
    return graph_status != "ok" or three_way_graph_hits == 0


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

    graph_hits = sum(r.get("graph_hits", 0) for r in case_results)

    return {
        **averages(case_results),
        "graph_hits": graph_hits,
        # Needing articles from different feeds is the case the graph was added for.
        "cross_publication_recall": _mean(
            [r["recall_at_k"] for r in by_category.get("cross_publication", [])]
        ),
        "by_category": {name: averages(results) for name, results in sorted(by_category.items())},
    }
