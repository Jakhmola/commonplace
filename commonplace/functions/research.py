"""The research agent: several searches for a question one search would not answer.

`run-research-agent` (on cp/research.requested) is a loop written directly on
steps:

1. `record-research-start`   the run is marked as running
2. `plan-subqueries`         the planner proposes the first searches
3. `hop-<n>-retrieve`        each search runs through the three-way retrieval;
                             new passages join a deduplicated pool
4. `hop-<n>-reflect`         the critic reads the pool: enough, or what next?
5. `synthesize-report`       the report is written from the best of the pool
6. `score-deterministic`, `judge-faithfulness`, `judge-disagreement`
                             the report is scored like any other answer
7. `persist-research-run`    everything is stored
8. `emit`                    cp/research.completed

A loop that a model controls needs bounds that the model does not. There are
three, all in plain code (`agents/roles/researcher.py`): a cap on hops, a cap
on dollars, and convergence (a hop that found nothing new ends the run). The
dollar cap bounds the searching: once it is reached nothing more is searched
or reflected on, and the report is written from what was gathered.

Every model call is its own step, so a retry never pays twice, and the loop
can be killed and resumed at any point without losing a hop.
"""

from datetime import UTC, datetime
from typing import Any

import inngest

from commonplace.agents.core import run_agent_step
from commonplace.agents.roles.researcher import (
    CoverageReflection,
    ResearchPlan,
    coverage_critic,
    format_plan_input,
    format_reflection_input,
    format_report_input,
    fresh_queries,
    merge_hop_candidates,
    query_planner,
    research_synthesizer,
    stop_reason_after_hop,
)
from commonplace.citations import NO_PASSAGES, citation, cited
from commonplace.config import settings
from commonplace.db.queries import (
    fail_research_run,
    finish_research_run,
    start_research_run,
)
from commonplace.db.session import get_session
from commonplace.functions.evals import score_answer
from commonplace.inngest_client import client
from commonplace.retrieval.hybrid import hybrid_search
from commonplace.schemas.events import (
    RESEARCH_COMPLETED,
    RESEARCH_REQUESTED,
    ResearchCompleted,
    ResearchRequested,
    event,
)


async def retrieve_hop(queries: list[str]) -> dict[str, Any]:
    """One hop: every query through the three-way search, the best passages of each."""
    candidates: list[dict[str, Any]] = []
    async with get_session() as session:
        for query in queries:
            top, _ = await hybrid_search(session, query, limit=settings.research_passages_per_hop)
            candidates.extend(top)
    return {"candidates": candidates}


async def gather_evidence(
    step: Any, *, question: str, max_iterations: int, max_usd: float, has_llm: bool
) -> dict[str, Any]:
    """Plan, then search and reflect until one of the bounds says stop.

    Returns `{plan, hops, pool, stop_reason, iterations, spent_usd}`. Without a
    model there is nobody to plan or reflect: the question is searched once.
    """
    spent = 0.0
    if has_llm:
        planned = await run_agent_step(
            step, query_planner, format_plan_input(question), "plan-subqueries"
        )
        plan = ResearchPlan.model_validate(planned["output"]).model_dump()
        spent += planned["usage"]["est_usd"]
    else:
        plan = {
            "rationale": "No ANTHROPIC_API_KEY: the question is searched as it was asked.",
            "subqueries": [question],
            "target_topics": [],
        }

    per_hop = settings.research_max_subqueries_per_hop
    queries = fresh_queries(plan["subqueries"], [], per_hop) or [question]
    pool: list[dict[str, Any]] = []
    executed: list[str] = []
    hops: list[dict[str, Any]] = []
    iteration = 0
    while True:
        iteration += 1
        found = await step.run(f"hop-{iteration}-retrieve", retrieve_hop, queries)
        executed.extend(queries)
        pool, added = merge_hop_candidates(pool, found["candidates"])
        hop: dict[str, Any] = {
            "iteration": iteration,
            "queries": queries,
            "retrieved": len(found["candidates"]),
            "added": added,
            "pool_size": len(pool),
        }
        hops.append(hop)

        stop_reason = stop_reason_after_hop(
            iteration=iteration,
            max_iterations=max_iterations,
            spent_usd=spent,
            max_usd=max_usd,
            added=added,
        )
        if stop_reason is not None:
            break
        if not has_llm:
            stop_reason = "no_followup_queries"
            break

        reflected = await run_agent_step(
            step,
            coverage_critic,
            format_reflection_input(
                question=question,
                executed=executed,
                pool=pool,
                limit=settings.research_report_passages,
            ),
            f"hop-{iteration}-reflect",
        )
        reflection = CoverageReflection.model_validate(reflected["output"])
        spent += reflected["usage"]["est_usd"]
        hop["reflection"] = reflection.model_dump(exclude={"followup_queries"})

        if reflection.is_sufficient:
            stop_reason = "sufficient_coverage"
            break
        queries = fresh_queries(reflection.followup_queries, executed, per_hop)
        if not queries:
            stop_reason = "no_followup_queries"
            break
        if spent >= max_usd:
            # The reflection itself used up the budget: what it asked for is not searched.
            stop_reason = "budget_exceeded"
            break

    return {
        "plan": plan,
        "hops": hops,
        "pool": pool,
        "stop_reason": stop_reason,
        "iterations": iteration,
        "spent_usd": round(spent, 6),
    }


async def write_report(
    step: Any, *, question: str, passages: list[dict[str, Any]], has_llm: bool
) -> tuple[str, list[dict[str, Any]], float]:
    """(report, its citations, what writing it cost)."""
    if not passages:
        return NO_PASSAGES, [], 0.0
    if not has_llm:
        report = (
            f"Retrieved {len(passages)} relevant passages. "
            "Set ANTHROPIC_API_KEY to get a written report with inline citations."
        )
        return report, [citation(n, p) for n, p in enumerate(passages, start=1)], 0.0

    written = await run_agent_step(
        step, research_synthesizer, format_report_input(question, passages), "synthesize-report"
    )
    if written["stop_reason"] == "refusal":
        report = "The model declined to write this report."
    else:
        report = str(written["output"] or "") or (
            f"The model returned no report (stop reason: {written['stop_reason']})."
        )
    return report, cited(report, passages), written["usage"]["est_usd"]


async def mark_failed(ctx: inngest.Context) -> None:
    """Runs when a research run has failed for good, so nobody polls a run that
    will never finish.
    """
    failure: dict[str, Any] = dict(ctx.event.data or {})
    research_id = ((failure.get("event") or {}).get("data") or {}).get("research_id")
    if not research_id:
        return  # the request itself was malformed: there is no run to mark
    error = failure.get("error")
    message = (error.get("message") or error.get("name")) if isinstance(error, dict) else error

    async def mark() -> None:
        async with get_session() as session:
            await fail_research_run(
                session, research_id=str(research_id), error=str(message or "unknown error")
            )
            await session.commit()

    await ctx.step.run("mark-research-failed", mark)


@client.create_function(
    fn_id="run-research-agent",
    trigger=inngest.TriggerEvent(event=RESEARCH_REQUESTED),
    concurrency=[inngest.Concurrency(limit=settings.concurrency_research_runs)],
    on_failure=mark_failed,
    retries=2,
)
async def run_research_agent(ctx: inngest.Context) -> dict[str, object]:
    payload = ResearchRequested.model_validate(ctx.event.data)
    max_iterations = payload.max_iterations or settings.research_max_iterations
    max_usd = payload.max_usd or settings.research_max_usd
    has_llm = bool(settings.anthropic_api_key)

    async def record_start() -> str:
        async with get_session() as session:
            await start_research_run(
                session,
                research_id=payload.research_id,
                event_id=str(ctx.event.id),
                question=payload.question,
                status="running",
            )
            await session.commit()
        # Returned from a step so that it is the same on every replay.
        return datetime.now(UTC).isoformat()

    started_at = await ctx.step.run("record-research-start", record_start)

    gathered = await gather_evidence(
        ctx.step,
        question=payload.question,
        max_iterations=max_iterations,
        max_usd=max_usd,
        has_llm=has_llm,
    )
    pool: list[dict[str, Any]] = gathered["pool"]
    shown = pool[: settings.research_report_passages]
    for passage in shown:
        passage["in_prompt"] = True  # what the report writer saw, for the evaluators

    report, citations, report_usd = await write_report(
        ctx.step, question=payload.question, passages=shown, has_llm=has_llm
    )
    scores = await score_answer(
        ctx.step,
        question=payload.question,
        answer_text=report,
        citations=citations,
        retrieved=pool,
        prompt_limit=settings.research_report_passages,
    )
    total_usd = round(gathered["spent_usd"] + report_usd + scores["judge_usd"], 6)
    status = "budget_exceeded" if gathered["stop_reason"] == "budget_exceeded" else "completed"

    async def persist() -> int:
        latency_ms = int(
            (datetime.now(UTC) - datetime.fromisoformat(started_at)).total_seconds() * 1000
        )
        async with get_session() as session:
            await finish_research_run(
                session,
                research_id=payload.research_id,
                status=status,
                stop_reason=gathered["stop_reason"],
                plan=gathered["plan"],
                hops=gathered["hops"],
                report_text=report,
                citations=citations,
                retrieved=pool,
                eval_summary={
                    "deterministic_passed": scores["checks"]["passed"],
                    "checks": scores["checks"]["checks"],
                    "faithfulness": scores["faithfulness"],
                    "disagreement": scores["disagreement"],
                    "failure_codes": scores["failure_codes"],
                    "judge_version": scores["judge_version"],
                    "limits": {"max_iterations": max_iterations, "max_usd": max_usd},
                    "search_usd": gathered["spent_usd"],
                },
                confidence=scores["composite_score"],
                iterations_used=gathered["iterations"],
                total_cost_usd=total_usd,
                latency_ms=latency_ms,
            )
            await session.commit()
        return latency_ms

    latency_ms = await ctx.step.run("persist-research-run", persist)

    completed = ResearchCompleted(
        research_id=payload.research_id,
        status=status,
        stop_reason=gathered["stop_reason"],
        iterations_used=gathered["iterations"],
        total_cost_usd=total_usd,
        confidence=scores["composite_score"],
        failure_codes=scores["failure_codes"],
    )
    await ctx.step.send_event("emit", event(RESEARCH_COMPLETED, completed))
    return {**completed.model_dump(mode="json"), "latency_ms": latency_ms}
