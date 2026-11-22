"""Retrieval eval against the running API.

For each query and mode, finds the rank of the first passage from the expected
article in the top k. Reports hit@k and MRR, overall and per query kind.

    uv run python scripts/eval_retrieval.py [--k 5] [--modes sparse,dense]
"""

import argparse
import sys
from pathlib import Path

import httpx
import yaml

from commonplace.sources.urls import canonical_id

QUERIES = Path(__file__).resolve().parents[1] / "evals" / "queries_v1.yaml"


def first_rank(hits: list[dict[str, str]], expect: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if canonical_id(h["url"]) == expect:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", default="sparse,dense")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    queries = yaml.safe_load(QUERIES.read_text())["queries"]
    modes = args.modes.split(",")
    ranks: dict[str, list[int | None]] = {m: [] for m in modes}
    with httpx.Client(base_url=args.api, timeout=30) as c:
        for m in modes:
            for item in queries:
                r = c.get("/search", params={"q": item["q"], "mode": m, "k": args.k})
                r.raise_for_status()
                ranks[m].append(first_rank(r.json()["results"], item["expect"]))

    def score(rs: list[int | None]) -> str:
        hit = sum(r is not None for r in rs) / len(rs)
        mrr = sum(1 / r for r in rs if r) / len(rs)
        return f"{hit:.2f} | {mrr:.2f}"

    kinds = sorted({q["kind"] for q in queries})
    print(
        f"| mode | all hit@{args.k} | all MRR | "
        + " | ".join(f"{k} hit@{args.k} | {k} MRR" for k in kinds)
        + " |"
    )
    print("|---" * (3 + 2 * len(kinds)) + "|")
    for m in modes:
        cells = [score(ranks[m])]
        for k in kinds:
            cells.append(
                score([r for r, q in zip(ranks[m], queries, strict=True) if q["kind"] == k])
            )
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\nPer query (rank of expected article, - = not in top k):")
    for i, q in enumerate(queries):
        cols = "  ".join(f"{m}={ranks[m][i] or '-'}" for m in modes)
        print(f"  [{q['kind'][:4]}] {cols}  {q['q']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
