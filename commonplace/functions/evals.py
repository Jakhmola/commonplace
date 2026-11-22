"""Evaluation as durable functions.

`evaluate-answer` (on cp/answer.produced) scores every answer shortly after it
is given, without making the person who asked wait for it:

1. `load-snapshot`        what was asked, shown and answered
2. `score-deterministic`  the checks in code (free, always run)
3. `judge-faithfulness`   one judge call (memoized)
4. `judge-disagreement`   one judge call, only if two authors were in the prompt
5. `persist-eval-scores`  `eval_results`, and the snapshot's confidence
6. `emit`                 cp/answer.evaluated

Each judge is its own step for the usual reason: if the second one fails, a
retry does not pay for the first again.
"""

from dataclasses import asdict
from typing import Any

import inngest

from commonplace.agents.core import run_agent
from commonplace.config import settings
from commonplace.db.queries import get_snapshot, get_superseded_ids, upsert_eval_result
from commonplace.db.session import get_session
from commonplace.evals.deterministic import run_deterministic_checks
from commonplace.evals.judges import (
    NOT_APPLICABLE,
    SKIPPED,
    BinaryVerdict,
    disagreement_judge,
    faithfulness_judge,
    format_judge_input,
    requires_disagreement_check,
)
from commonplace.inngest_client import client
from commonplace.schemas.events import (
    ANSWER_EVALUATED,
    ANSWER_PRODUCED,
    AnswerEvaluated,
    AnswerProduced,
    event,
)


def composite_score(
    deterministic_passed: bool, faithfulness_verdict: str, disagreement_verdict: str
) -> float:
    """One number per answer, stored as the snapshot's confidence.

    0.0 if a check in code or the faithfulness judge failed: the answer cannot
    be relied on. 0.5 if it is faithful but hides a disagreement. 1.0 otherwise.
    A judge that did not run (skipped, or not applicable) neither lowers the
    score nor vouches for the answer.
    """
    if not deterministic_passed or faithfulness_verdict == "FAIL":
        return 0.0
    return 0.5 if disagreement_verdict == "FAIL" else 1.0


@client.create_function(
    fn_id="evaluate-answer",
    trigger=inngest.TriggerEvent(event=ANSWER_PRODUCED),
    concurrency=[inngest.Concurrency(limit=settings.concurrency_eval_runs)],
    retries=2,
)
async def evaluate_answer(ctx: inngest.Context) -> dict[str, object]:
    payload = AnswerProduced.model_validate(ctx.event.data)

    async def load_snapshot() -> dict[str, Any] | None:
        async with get_session() as session:
            snapshot = await get_snapshot(session, payload.snapshot_id)
        if snapshot is None:
            return None
        return {
            "question": snapshot.question,
            "answer_text": snapshot.answer_text,
            "citations": snapshot.citations,
            "retrieved": snapshot.retrieved,
        }

    snapshot = await ctx.step.run("load-snapshot", load_snapshot)
    if snapshot is None:
        return {"snapshot_id": payload.snapshot_id, "status": "skipped_missing_snapshot"}
    citations: list[dict[str, Any]] = snapshot["citations"]
    retrieved: list[dict[str, Any]] = snapshot["retrieved"]
    trace = {
        "question": snapshot["question"],
        "retrieved": retrieved,
        "answer_text": snapshot["answer_text"],
        "prompt_limit": settings.retrieval_limit,
    }

    async def score_deterministic() -> dict[str, Any]:
        async with get_session() as session:
            superseded = await get_superseded_ids(
                session,
                passage_ids=[c["passage_id"] for c in citations if c.get("passage_id")],
                claim_ids=[c["claim_id"] for c in citations if c.get("claim_id")],
            )
        result = run_deterministic_checks(
            answer_text=snapshot["answer_text"],
            citations=citations,
            retrieved=retrieved,
            prompt_limit=settings.retrieval_limit,
            superseded_ids=frozenset(superseded),
        )
        return asdict(result)

    checks = await ctx.step.run("score-deterministic", score_deterministic)
    failure_codes: list[str] = list(checks["failure_codes"])

    async def judge(agent: Any, step_id: str) -> dict[str, str]:
        """One judge as one step. Without a key nothing runs and the verdict says so."""
        if not settings.anthropic_api_key:
            return {"verdict": SKIPPED, "critique": "No ANTHROPIC_API_KEY: the judge did not run."}
        verdict = await run_agent(ctx.step, agent, format_judge_input(**trace), step_id)
        assert isinstance(verdict, BinaryVerdict)
        if verdict.verdict == "FAIL" and verdict.failure_code not in ("NONE", *failure_codes):
            failure_codes.append(verdict.failure_code)
        return {"verdict": verdict.verdict, "critique": verdict.critique}

    faithfulness = await judge(faithfulness_judge, "judge-faithfulness")
    if requires_disagreement_check(retrieved, settings.retrieval_limit):
        disagreement = await judge(disagreement_judge, "judge-disagreement")
    else:
        disagreement = {
            "verdict": NOT_APPLICABLE,
            "critique": "Fewer than two authors were in the prompt: nobody to disagree with.",
        }

    score = composite_score(checks["passed"], faithfulness["verdict"], disagreement["verdict"])

    async def persist() -> str:
        async with get_session() as session:
            eval_id = await upsert_eval_result(
                session,
                snapshot_id=payload.snapshot_id,
                deterministic_passed=checks["passed"],
                checks=checks["checks"],
                faithfulness_verdict=faithfulness["verdict"],
                faithfulness_critique=faithfulness["critique"],
                disagreement_verdict=disagreement["verdict"],
                disagreement_critique=disagreement["critique"],
                failure_codes=failure_codes,
                composite_score=score,
            )
            await session.commit()
        return str(eval_id)

    eval_id = await ctx.step.run("persist-eval-scores", persist)

    evaluated = AnswerEvaluated(
        snapshot_id=payload.snapshot_id,
        composite_score=score,
        deterministic_passed=checks["passed"],
        faithfulness_verdict=faithfulness["verdict"],
        disagreement_verdict=disagreement["verdict"],
        failure_codes=failure_codes,
    )
    await ctx.step.send_event("emit", event(ANSWER_EVALUATED, evaluated))
    return {"eval_id": eval_id, **evaluated.model_dump(mode="json")}
