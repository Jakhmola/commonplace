"""Evaluation results in Postgres: scoring an answer, and the report built from the scores."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import EvalResult
from commonplace.db.queries import (
    get_latest_eval_report,
    get_snapshot,
    get_superseded_ids,
    persist_article_claims,
    save_snapshot,
    search_passages_bm25,
    store_passages,
    upsert_eval_result,
    upsert_eval_run,
)
from tests.integration.conftest import passages


async def _snapshot(session: AsyncSession, snapshot_id: str, question: str = "A question?") -> None:
    await save_snapshot(
        session,
        snapshot_id=snapshot_id,
        question=question,
        answer_text="An answer [1].",
        citations=[{"citation_idx": 1, "passage_id": "p1"}],
        retrieved=[{"passage_id": "p1", "in_prompt": True}],
        retriever_variant="hybrid_3way_rrf_k60",
        as_of=datetime(2026, 11, 17, tzinfo=UTC),
        latency_ms=10,
    )
    await session.flush()


def _score(snapshot_id: str, **overrides: Any) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "deterministic_passed": True,
        "checks": {"citation_bounds_valid": True},
        "faithfulness_verdict": "PASS",
        "faithfulness_critique": "Every statement is supported.",
        "disagreement_verdict": "NOT_APPLICABLE",
        "disagreement_critique": "One author.",
        "failure_codes": [],
        "composite_score": 1.0,
        **overrides,
    }


async def test_scoring_an_answer_sets_its_confidence_and_is_idempotent(db: AsyncSession) -> None:
    await _snapshot(db, "snap_evaltest_1")
    assert (await get_snapshot(db, "snap_evaltest_1")).confidence is None  # type: ignore[union-attr]

    first = await upsert_eval_result(db, **_score("snap_evaltest_1"))
    # the evaluator ran again (a retried step, or a re-evaluation with a better judge)
    second = await upsert_eval_result(
        db,
        **_score(
            "snap_evaltest_1",
            faithfulness_verdict="FAIL",
            faithfulness_critique="[1] does not say this.",
            failure_codes=["FM-4"],
            composite_score=0.0,
        ),
    )
    assert first == second  # one row per snapshot
    rows = (
        await db.execute(
            select(func.count())
            .select_from(EvalResult)
            .where(EvalResult.snapshot_id == "snap_evaltest_1")
        )
    ).scalar_one()
    assert rows == 1

    snapshot = await get_snapshot(db, "snap_evaltest_1")
    assert snapshot is not None
    await db.refresh(snapshot)
    assert float(snapshot.confidence) == 0.0  # type: ignore[arg-type]


async def test_the_report_counts_failure_modes_and_pass_rates(db: AsyncSession) -> None:
    before = await get_latest_eval_report(db)
    for n, score in enumerate(
        [
            _score("snap_evaltest_a"),
            _score(
                "snap_evaltest_b",
                faithfulness_verdict="FAIL",
                faithfulness_critique="The figure is not in [2].",
                failure_codes=["FM-4"],
                composite_score=0.0,
            ),
            _score(
                "snap_evaltest_c",
                deterministic_passed=False,
                failure_codes=["FM-1", "FM-4"],
                faithfulness_verdict="FAIL",
                composite_score=0.0,
            ),
            _score("snap_evaltest_d", faithfulness_verdict="SKIPPED", composite_score=1.0),
        ]
    ):
        await _snapshot(db, score["snapshot_id"], question=f"Question {n}?")
        await upsert_eval_result(db, **score)
    await upsert_eval_run(
        db,
        run_id="run_evaltest",
        suite_name="golden_v1",
        variants_summary={"sparse_bm25": {"recall_at_k": 0.5, "mrr": 0.4}},
        case_results=[{"case_id": "c1"}],
    )

    report = await get_latest_eval_report(db, limit=3)
    answers = report["answers"]
    assert answers["evaluated"] == before["answers"]["evaluated"] + 4
    for code, added in (("FM-4", 2), ("FM-1", 1)):
        assert report["failure_modes"][code] == before["failure_modes"].get(code, 0) + added
    # a skipped judge is left out of its pass rate instead of counting either way
    assert 0.0 <= answers["faithfulness_pass_rate"] <= 1.0
    assert answers["faithfulness_judged"] == before["answers"]["faithfulness_judged"] + 3

    # the most recent failures come first, with the critique that explains them
    assert [f["snapshot_id"] for f in report["recent_failures"][:2]] == [
        "snap_evaltest_c",
        "snap_evaltest_b",
    ]
    assert report["recent_failures"][1]["critique"] == "The figure is not in [2]."
    assert len(report["recent_failures"]) <= 3
    assert report["latest_run"]["run_id"] == "run_evaltest"
    assert report["latest_run"]["variants_summary"]["sparse_bm25"]["mrr"] == 0.4


async def test_superseded_passages_and_claims_are_found(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    await store_passages(db, article_id=article_id, passages=passages("quokkazebra old version"))
    (old_passage,) = [
        h["passage_id"]
        for h in await search_passages_bm25(db, "quokkazebra", 25)
        if h["canonical_id"] == "pub:integration-test:a-seeded-post"
    ]
    old_claim = {
        "claim_id": str(uuid.uuid4()),
        "text": "Old claim",
        "quote_span": "quokkazebra old version",
        "claim_type": "opinion",
        "topics": [{"slug": "quokkazebra", "name": "Quokkazebra"}],
    }
    now = datetime(2026, 11, 23, tzinfo=UTC)
    await persist_article_claims(db, article_id=article_id, claims=[old_claim], now=now)
    ids = [old_passage, old_claim["claim_id"]]
    assert await get_superseded_ids(db, passage_ids=[old_passage], claim_ids=ids[1:]) == set()

    # the article is edited: both the passage and the claim are closed
    await store_passages(db, article_id=article_id, passages=passages("quokkazebra new version"))
    await persist_article_claims(db, article_id=article_id, claims=[], now=now)
    assert await get_superseded_ids(db, passage_ids=[old_passage], claim_ids=ids[1:]) == set(ids)
    # ids that are not uuids (or not known) are ignored, not an error
    assert await get_superseded_ids(db, passage_ids=["p1", "nope"], claim_ids=[]) == set()


async def test_a_score_records_which_judge_gave_it(db: AsyncSession) -> None:
    await _snapshot(db, "snap_evaltest_judge")
    await upsert_eval_result(db, **_score("snap_evaltest_judge"), judge_version="a-model:1a2b3c4d")
    report = await get_latest_eval_report(db)
    assert report["answers"]["judge_versions"]["a-model:1a2b3c4d"] == 1
