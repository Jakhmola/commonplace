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

`run-eval-suite` (on cp/eval.run) compares the retrieval variants on the
golden benchmark.
"""

from dataclasses import asdict
from typing import Any

import inngest

from commonplace.agents.core import Agent
from commonplace.agents.providers import ModelRefusal, call_llm
from commonplace.config import settings
from commonplace.db.queries import (
    get_existing_canonical_ids,
    get_snapshot,
    get_superseded_ids,
    upsert_eval_result,
    upsert_eval_run,
)
from commonplace.db.session import get_session
from commonplace.evals.benchmark import (
    VARIANTS,
    load_benchmark,
    missing_expected,
    retrieve_variant,
    scorable_cases,
    score_case,
    suite_is_degraded,
    summarise,
)
from commonplace.evals.deterministic import run_deterministic_checks
from commonplace.evals.judges import (
    NOT_APPLICABLE,
    SKIPPED,
    BinaryVerdict,
    disagreement_judge,
    faithfulness_judge,
    format_judge_input,
    judge_version,
    requires_disagreement_check,
)
from commonplace.graph.client import check_memgraph_health
from commonplace.inngest_client import client
from commonplace.schemas.events import (
    ANSWER_EVALUATED,
    ANSWER_PRODUCED,
    EVAL_RUN_REQUESTED,
    AnswerEvaluated,
    AnswerProduced,
    EvalRunRequested,
    event,
)
from commonplace.sources.embedder import embed_texts, pack, unpack


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


async def judge_once(agent: Agent[BinaryVerdict], judge_input: str) -> dict[str, Any]:
    """Run one judge on one trace: `{verdict, critique, failure_code, usage}`.

    If the provider refuses the request, the verdict is recorded as skipped
    rather than raised: asking again would be refused again, and one refusal
    must not leave the answer without any score at all.
    """
    try:
        result = await call_llm(
            system_prompt=agent.system_prompt,
            user_input=judge_input,
            model=agent.model,
            max_tokens=agent.max_tokens,
            effort=agent.effort,
            output_schema=BinaryVerdict,
        )
    except ModelRefusal as exc:
        return {
            "verdict": SKIPPED,
            "critique": f"The judge was refused by the provider: {exc}",
            "failure_code": "NONE",
        }
    verdict = BinaryVerdict.model_validate(result["output"])
    # `usage` at the top level is what the cost middleware looks for.
    return {**verdict.model_dump(), "usage": result["usage"]}


async def score_answer(
    step: Any,
    *,
    question: str,
    answer_text: str,
    citations: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    prompt_limit: int,
) -> dict[str, Any]:
    """Every evaluator applied to one answer, each as its own step.

    `{checks, faithfulness, disagreement, failure_codes, composite_score,
    judge_usd, judge_version}`. Used for answers from `/ask` and for research reports alike:
    `retrieved` is everything that was found, with `in_prompt` on what the
    writer was shown.
    """

    async def score_deterministic() -> dict[str, Any]:
        async with get_session() as session:
            superseded = await get_superseded_ids(
                session,
                passage_ids=[c["passage_id"] for c in citations if c.get("passage_id")],
                claim_ids=[c["claim_id"] for c in citations if c.get("claim_id")],
            )
        result = run_deterministic_checks(
            answer_text=answer_text,
            citations=citations,
            retrieved=retrieved,
            prompt_limit=prompt_limit,
            superseded_ids=frozenset(superseded),
        )
        return asdict(result)

    checks = await step.run("score-deterministic", score_deterministic)
    failure_codes: list[str] = list(checks["failure_codes"])
    judge_input = format_judge_input(
        question=question, retrieved=retrieved, answer_text=answer_text, prompt_limit=prompt_limit
    )
    judge_usd = 0.0

    async def judge(agent: Agent[BinaryVerdict], step_id: str) -> dict[str, Any]:
        """One judge as one step. Without a key nothing runs and the verdict says so."""
        nonlocal judge_usd
        if not settings.anthropic_api_key:
            return {"verdict": SKIPPED, "critique": "No ANTHROPIC_API_KEY: the judge did not run."}
        verdict: dict[str, Any] = await step.run(step_id, judge_once, agent, judge_input)
        judge_usd += (verdict.pop("usage", None) or {}).get("est_usd", 0.0)
        if verdict["verdict"] == "FAIL" and verdict["failure_code"] not in ("NONE", *failure_codes):
            failure_codes.append(verdict["failure_code"])
        return verdict

    faithfulness = await judge(faithfulness_judge, "judge-faithfulness")
    if requires_disagreement_check(retrieved, prompt_limit):
        disagreement = await judge(disagreement_judge, "judge-disagreement")
    else:
        disagreement = {
            "verdict": NOT_APPLICABLE,
            "critique": "Fewer than two authors were in the prompt: nobody to disagree with.",
        }
    return {
        "checks": checks,
        "faithfulness": faithfulness,
        "disagreement": disagreement,
        "failure_codes": failure_codes,
        "composite_score": composite_score(
            checks["passed"], faithfulness["verdict"], disagreement["verdict"]
        ),
        "judge_usd": round(judge_usd, 6),
        "judge_version": judge_version(),
    }


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
    scores = await score_answer(
        ctx.step,
        question=snapshot["question"],
        answer_text=snapshot["answer_text"],
        citations=snapshot["citations"],
        retrieved=snapshot["retrieved"],
        prompt_limit=settings.retrieval_limit,
    )
    checks, faithfulness, disagreement = (
        scores["checks"],
        scores["faithfulness"],
        scores["disagreement"],
    )
    failure_codes: list[str] = scores["failure_codes"]
    score: float = scores["composite_score"]

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
                judge_version=scores["judge_version"],
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


@client.create_function(
    fn_id="run-eval-suite",
    trigger=inngest.TriggerEvent(event=EVAL_RUN_REQUESTED),
    # One suite at a time: runs would otherwise compete for the same database
    # and make each other's timings meaningless.
    concurrency=[inngest.Concurrency(limit=1)],
    retries=2,
)
async def run_eval_suite(ctx: inngest.Context) -> dict[str, object]:
    """cp/eval.run -> run-eval-suite.

    Puts the golden questions to each retrieval variant against the live
    corpus and stores how each did. One step per variant, so a failure in the
    third does not repeat the first two.
    """
    payload = EvalRunRequested.model_validate(ctx.event.data or {})

    async def load() -> dict[str, Any]:
        benchmark = load_benchmark()
        expected = {cid for case in benchmark["cases"] for cid in case["expected_canonical_ids"]}
        async with get_session() as session:
            stored = await get_existing_canonical_ids(session, sorted(expected))
        return {
            **benchmark,
            "missing_expected": missing_expected(benchmark["cases"], stored),
            "graph_status": await check_memgraph_health(),
        }

    benchmark = await ctx.step.run("load-benchmark", load)
    # A case is scored only on expected articles that are actually stored.
    cases = scorable_cases(benchmark["cases"], benchmark["missing_expected"])
    k: int = benchmark["k"]

    async def embed_questions() -> dict[str, Any]:
        result = await embed_texts([case["question"] for case in cases])
        return {"vectors": [pack(v) for v in result["embeddings"]], "usage": result["usage"]}

    embedded = await ctx.step.run("embed-questions", embed_questions)

    variants_summary: dict[str, Any] = {}
    case_results: list[dict[str, Any]] = []
    for variant in VARIANTS:

        async def evaluate(variant: str = variant) -> list[dict[str, Any]]:
            results = []
            async with get_session() as session:
                for case, vector in zip(cases, embedded["vectors"], strict=True):
                    hits = await retrieve_variant(
                        session, variant, case["question"], unpack(vector), k
                    )
                    results.append({"variant": variant, **score_case(case, hits, k)})
            return results

        results = await ctx.step.run(f"evaluate-variant-{variant}", evaluate)
        variants_summary[variant] = summarise(results)
        case_results.extend(results)

    degraded = suite_is_degraded(
        graph_status=benchmark["graph_status"],
        three_way_graph_hits=variants_summary["hybrid_3way_rrf_k60"]["graph_hits"],
    )

    async def persist() -> str:
        summary = {
            "variants": variants_summary,
            "k": k,
            "cases_scored": len(cases),
            "missing_expected": benchmark["missing_expected"],
            "graph_status": benchmark["graph_status"],
            # True: the three-way variant ran without its graph leg and repeats the two-way one.
            "degraded": degraded,
        }
        async with get_session() as session:
            run_id = await upsert_eval_run(
                session,
                run_id=str(ctx.event.id),
                suite_name=payload.suite_name,
                variants_summary=summary,
                case_results=case_results,
            )
            await session.commit()
        return run_id

    run_id = await ctx.step.run("persist-suite-summary", persist)
    return {
        "run_id": run_id,
        "suite_name": payload.suite_name,
        "variants": variants_summary,
        "missing_expected": benchmark["missing_expected"],
        "degraded": degraded,
    }
