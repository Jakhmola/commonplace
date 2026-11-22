"""Unit tests for the research agent, the weekly digest and the production packaging.

No Docker or keys needed.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import inngest
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from commonplace import operations
from commonplace.agents import core as agents_core
from commonplace.agents.roles.researcher import (
    CoverageReflection,
    ResearchPlan,
    coverage_critic,
    digest_writer,
    format_reflection_input,
    fresh_queries,
    merge_hop_candidates,
    query_planner,
    research_synthesizer,
    stop_reason_after_hop,
)
from commonplace.citations import format_context
from commonplace.functions import FUNCTIONS, research
from commonplace.functions.digest import (
    citations_in_bounds,
    digest_evidence,
    digest_id_for,
    digest_window,
    format_digest_input,
    generate_weekly_digest,
    plain_digest,
    shared_topics,
)
from commonplace.functions.research import run_research_agent
from commonplace.main import DigestRequest, ResearchRequest
from commonplace.mcp import tools
from commonplace.mcp.server import mcp
from commonplace.schemas.events import (
    DIGEST_PUBLISHED,
    DIGEST_REQUESTED,
    RESEARCH_COMPLETED,
    RESEARCH_REQUESTED,
    DigestRequested,
    ResearchCompleted,
    ResearchRequested,
)
from commonplace.schemas.ids import research_id

# --- ids and events ---


def test_research_ids_are_prefixed_and_unique() -> None:
    ids = {research_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("res_") and len(i) == 16 for i in ids)


def test_research_and_digest_events() -> None:
    assert (RESEARCH_REQUESTED, RESEARCH_COMPLETED) == (
        "cp/research.requested",
        "cp/research.completed",
    )
    assert (DIGEST_REQUESTED, DIGEST_PUBLISHED) == ("cp/digest.requested", "cp/digest.published")
    requested = ResearchRequested(research_id="res_abc", question="q")
    assert (requested.max_iterations, requested.max_usd) == (None, None)  # defaults live in config
    assert DigestRequested().lookback_days == 7
    completed = ResearchCompleted(
        research_id="res_abc",
        status="completed",
        stop_reason="converged",
        iterations_used=2,
        total_cost_usd=0.21,
        confidence=None,  # the judges may not have run
        failure_codes=[],
    )
    assert completed.confidence is None


# --- the evidence pool ---


def _hit(passage_id: str, score: float, **extra: Any) -> dict[str, Any]:
    source = {"title": "A post", "author": "Ann", "url": "https://example.com/p/a-post"}
    return {"passage_id": passage_id, "rrf_score": score, "text": passage_id, **source, **extra}


def test_merging_a_hop_keeps_the_best_score_and_counts_what_is_new() -> None:
    pool = [_hit("p1", 0.030), _hit("p2", 0.020)]
    hop = [
        _hit("p2", 0.045, claim_id="c1", claim_text="a claim", claim_type="opinion"),
        _hit("p3", 0.025),
        _hit("p1", 0.010),  # found again, lower down: the earlier score stands
    ]
    merged, added = merge_hop_candidates(pool, hop)
    assert added == 1
    assert [(p["passage_id"], p["rrf_score"]) for p in merged] == [
        ("p2", 0.045),
        ("p1", 0.030),
        ("p3", 0.025),
    ]
    assert merged[0]["claim_id"] == "c1"  # the graph's claim travels with the passage


def test_a_claim_is_not_lost_when_the_passage_is_found_again_without_it() -> None:
    pool = [_hit("p1", 0.02, claim_id="c1", claim_text="a claim", claim_type="measurement")]
    merged, added = merge_hop_candidates(pool, [_hit("p1", 0.05)])
    assert added == 0
    assert (merged[0]["rrf_score"], merged[0]["claim_id"]) == (0.05, "c1")


def test_merging_is_deterministic_on_ties_and_leaves_its_inputs_alone() -> None:
    pool = [_hit("pb", 0.02)]
    merged, _ = merge_hop_candidates(pool, [_hit("pa", 0.02)])
    assert [p["passage_id"] for p in merged] == ["pa", "pb"]
    assert pool == [_hit("pb", 0.02)]


# --- when the loop stops ---


def _after(**overrides: Any) -> str | None:
    hop = {"iteration": 2, "max_iterations": 3, "spent_usd": 0.10, "max_usd": 0.75, "added": 4}
    return stop_reason_after_hop(**(hop | overrides))


def test_a_hop_that_adds_nothing_ends_the_run() -> None:
    assert _after(added=0) == "converged"
    # The first hop finding nothing is not convergence: there is nothing to converge on.
    assert _after(iteration=1, added=0) is None
    assert _after(added=1) is None


def test_the_loop_stops_at_the_iteration_and_budget_caps() -> None:
    assert _after(iteration=3) == "max_iterations_reached"
    assert _after(iteration=1, max_iterations=1) == "max_iterations_reached"
    assert _after(spent_usd=0.75) == "budget_exceeded"
    assert _after(spent_usd=0.80) == "budget_exceeded"
    assert _after(spent_usd=0.74) is None
    # Convergence is reported in preference to the caps: it says more about the run.
    assert _after(iteration=3, added=0, spent_usd=1.0) == "converged"


def test_repeated_follow_ups_are_not_run_again() -> None:
    executed = ["How do open models compare?", "inference cost"]
    proposed = ["  how do OPEN models   compare? ", "Inference cost", "who leads adoption"]
    assert fresh_queries(proposed, executed, limit=3) == ["who leads adoption"]
    assert fresh_queries(["inference cost"], executed, limit=3) == []  # -> no_followup_queries
    assert fresh_queries(["a", "b", "A", "c", "d"], [], limit=3) == ["a", "b", "c"]
    assert fresh_queries(["", "   "], [], limit=3) == []


# --- the roles ---


def test_plans_and_reflections_have_bounded_shapes() -> None:
    plan = ResearchPlan(rationale="two angles", subqueries=["a", "b"])
    assert plan.target_topics == []
    with pytest.raises(ValidationError):
        ResearchPlan(rationale="nothing to search for", subqueries=[])
    reflection = CoverageReflection(rationale="enough", is_sufficient=True)
    assert (reflection.missing_aspects, reflection.followup_queries) == ([], [])
    assert query_planner.output_schema is ResearchPlan
    assert coverage_critic.output_schema is CoverageReflection
    # The report and the digest are prose with [n] citations, not structured output.
    assert research_synthesizer.output_schema is None and digest_writer.output_schema is None


def test_evidence_is_numbered_with_claims_and_types() -> None:
    passages: list[dict[str, Any]] = [
        {"title": "T1", "author": "Ann", "url": "https://a/1", "text": "first text"},
        {
            "title": "T2",
            "author": None,
            "url": "https://b/2",
            "text": "second text",
            "claim_text": "X beats Y",
            "claim_type": "measurement",
        },
    ]
    context = format_context(passages)
    assert context.index("[1]") < context.index("first text") < context.index("[2]")
    assert "Attributed claim (measurement): X beats Y" in context
    assert "Author: Unknown" in context


def test_the_critic_sees_what_was_searched_and_only_the_top_of_the_pool() -> None:
    pool = [
        {"title": f"T{n}", "author": "Ann", "url": f"https://a/{n}", "text": f"text-{n}"}
        for n in range(1, 5)
    ]
    shown = format_reflection_input(
        question="What changed?", executed=["first query", "second query"], pool=pool, limit=2
    )
    assert "What changed?" in shown
    assert "first query" in shown and "second query" in shown
    assert "text-2" in shown and "text-3" not in shown
    assert "4 passages" in shown  # it is told how much it is not seeing


# --- the loop, with a fake step and a fake model ---


class _Step:
    """Runs each step at once, like Inngest does the first time through."""

    def __init__(self) -> None:
        self.ids: list[str] = []

    async def run(self, step_id: str, fn: Any, *args: Any) -> Any:
        self.ids.append(step_id)
        return await fn(*args)


class _Model:
    """Answers the planner with `plan` and each reflection with the next of `reflections`."""

    def __init__(self, plan: list[str], reflections: list[dict[str, Any]], usd: float = 0.05):
        self.plan, self.reflections, self.usd = plan, list(reflections), usd
        self.calls: list[str] = []

    async def __call__(self, *, output_schema: Any, **_: Any) -> dict[str, Any]:
        self.calls.append(output_schema.__name__)
        if output_schema is ResearchPlan:
            output: dict[str, Any] = {"rationale": "r", "subqueries": self.plan}
        else:
            output = {"rationale": "r", "is_sufficient": False, **self.reflections.pop(0)}
        usage = {"model": "m", "input_tokens": 1, "output_tokens": 1, "est_usd": self.usd}
        return {"output": output, "usage": usage, "stop_reason": "end_turn"}


def _searching(monkeypatch: pytest.MonkeyPatch, found: dict[str, list[str]]) -> None:
    """Each query finds the passage ids listed for it (default: one passage named after it)."""

    async def retrieve_hop(queries: list[str]) -> dict[str, Any]:
        ids = [pid for q in queries for pid in found.get(q, [f"p-{q}"])]
        return {"candidates": [_hit(pid, 0.03) for pid in ids]}

    monkeypatch.setattr(research, "retrieve_hop", retrieve_hop)


async def _gather(
    monkeypatch: pytest.MonkeyPatch, model: _Model, **bounds: Any
) -> tuple[dict[str, Any], _Step]:
    monkeypatch.setattr(agents_core, "call_llm", model)
    step = _Step()
    limits = {"max_iterations": 3, "max_usd": 0.75, "has_llm": True} | bounds
    return await research.gather_evidence(step, question="the question", **limits), step


async def test_a_run_stops_when_the_critic_is_satisfied(monkeypatch: pytest.MonkeyPatch) -> None:
    _searching(monkeypatch, {})
    model = _Model(["a", "b"], [{"is_sufficient": True}])
    found, step = await _gather(monkeypatch, model)
    assert (found["stop_reason"], found["iterations"]) == ("sufficient_coverage", 1)
    assert [p["passage_id"] for p in found["pool"]] == ["p-a", "p-b"]
    assert found["spent_usd"] == pytest.approx(0.10)  # one plan and one reflection
    assert step.ids == ["plan-subqueries", "hop-1-retrieve", "hop-1-reflect"]
    assert found["hops"][0]["queries"] == ["a", "b"] and found["hops"][0]["added"] == 2


async def test_a_hop_that_finds_nothing_new_ends_without_another_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _searching(monkeypatch, {"a": ["p1"], "again, differently": ["p1"]})
    model = _Model(["a"], [{"followup_queries": ["again, differently"]}])
    found, step = await _gather(monkeypatch, model)
    assert (found["stop_reason"], found["iterations"]) == ("converged", 2)
    assert model.calls == ["ResearchPlan", "CoverageReflection"]  # nobody was asked after hop 2
    assert step.ids[-1] == "hop-2-retrieve"


async def test_a_critic_that_repeats_itself_does_not_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _searching(monkeypatch, {})
    model = _Model(["a", "b"], [{"followup_queries": ["A", " b "]}])
    found, _ = await _gather(monkeypatch, model)
    assert (found["stop_reason"], found["iterations"]) == ("no_followup_queries", 1)


async def test_the_iteration_cap_holds_against_a_critic_that_always_wants_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _searching(monkeypatch, {})
    model = _Model(["a"], [{"followup_queries": [f"more-{n}"]} for n in range(10)])
    found, _ = await _gather(monkeypatch, model, max_iterations=2)
    assert (found["stop_reason"], found["iterations"]) == ("max_iterations_reached", 2)
    assert model.calls == ["ResearchPlan", "CoverageReflection"]


async def test_the_budget_stops_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    _searching(monkeypatch, {})
    model = _Model(["a"], [{"followup_queries": [f"more-{n}"]} for n in range(10)], usd=0.30)
    found, step = await _gather(monkeypatch, model, max_usd=0.50)
    # plan 0.30, reflection 0.30: over the cap, so the follow-up it asked for is not run
    assert (found["stop_reason"], found["iterations"]) == ("budget_exceeded", 1)
    assert found["spent_usd"] == pytest.approx(0.60)
    assert "hop-2-retrieve" not in step.ids


async def test_without_a_model_the_question_is_searched_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _searching(monkeypatch, {})
    model = _Model([], [])
    found, step = await _gather(monkeypatch, model, has_llm=False)
    assert model.calls == []
    assert found["plan"]["subqueries"] == ["the question"]
    assert (found["stop_reason"], found["iterations"], found["spent_usd"]) == (
        "no_followup_queries",
        1,
        0.0,
    )
    assert step.ids == ["hop-1-retrieve"]


# --- the function, the API and the tools ---


def test_research_runs_on_request_with_a_concurrency_cap() -> None:
    config = run_research_agent.get_config("http://localhost:8000").main
    (trigger,) = config.triggers
    assert getattr(trigger, "event", None) == "cp/research.requested"
    (limit,) = config.concurrency or []
    assert (limit.key, limit.limit) == (None, 2)
    assert run_research_agent in FUNCTIONS


def test_research_bounds_are_validated() -> None:
    assert ResearchRequest(question="q").max_iterations is None
    assert ResearchRequest(question="q", max_iterations=5, max_usd=5).max_iterations == 5
    for bad in (
        {"max_iterations": 0},
        {"max_iterations": 6},
        {"max_usd": 0},
        {"max_usd": -1},
        {"max_usd": 5.01},
        {"question": ""},
    ):
        with pytest.raises(ValidationError):
            ResearchRequest(**({"question": "q"} | bad))


async def test_queueing_research_returns_an_id_to_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[inngest.Event] = []
    recorded: list[dict[str, Any]] = []

    async def send(event: inngest.Event) -> list[str]:
        sent.append(event)
        return ["evt-research"]

    async def record(**fields: Any) -> None:
        recorded.append(fields)

    monkeypatch.setattr(operations.client, "send", send)
    monkeypatch.setattr(operations, "_record_research", record)
    handle = await operations.queue_research("How do the authors differ?", max_iterations=2)

    rid = handle["research_id"]
    assert rid.startswith("res_") and handle["status"] == "queued"
    assert handle["event_id"] == "evt-research"
    assert sent[0].name == "cp/research.requested"
    assert sent[0].data == {
        "research_id": rid,
        "question": "How do the authors differ?",
        "max_iterations": 2,
        "max_usd": None,
    }
    # The row exists before the function starts, so the id can be polled at once.
    assert recorded == [
        {"research_id": rid, "event_id": "evt-research", "question": "How do the authors differ?"}
    ]


async def test_an_unknown_research_id_is_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def nothing(_: str) -> None:
        return None

    monkeypatch.setattr(operations, "_load_research", nothing)
    with pytest.raises(ToolError, match="No research run"):
        await tools.cp_research_result("res_does_not_exist")


# --- the weekly digest ---


def test_the_digest_window_is_whole_days_and_stable_within_a_day() -> None:
    monday_morning = datetime(2026, 11, 23, 8, 0, tzinfo=UTC)
    start, end = digest_window(monday_morning, 7)
    # the seven complete days before today: last Monday up to, not including, this one
    assert (start, end) == (
        datetime(2026, 11, 16, tzinfo=UTC),
        datetime(2026, 11, 23, tzinfo=UTC),
    )
    assert digest_window(datetime(2026, 11, 23, 23, 59, tzinfo=UTC), 7) == (start, end)
    assert digest_window(monday_morning, 1) == (datetime(2026, 11, 22, tzinfo=UTC), end)


def test_the_same_window_always_gets_the_same_digest_id() -> None:
    start, end = digest_window(datetime(2026, 11, 23, 8, 0, tzinfo=UTC), 7)
    assert digest_id_for(start, end) == digest_id_for(start, end)
    assert digest_id_for(start, end) != digest_id_for(start - timedelta(days=7), start)
    assert digest_id_for(start, end) != digest_id_for(end - timedelta(days=1), end)


def _claims() -> list[dict[str, Any]]:
    def claim(n: int, kind: str, publication: str, topics: list[str]) -> dict[str, Any]:
        return {
            "claim_id": f"c{n}",
            "claim_text": f"claim {n}",
            "claim_type": kind,
            "topics": topics,
            "article_id": f"a{n}",
            "title": f"Post {n}",
            "author": f"Author {n}",
            "url": f"https://example.com/p/{n}",
            "publication": publication,
        }

    return [
        claim(1, "measurement", "Alpha", ["rag", "evals"]),
        claim(2, "opinion", "Beta", ["rag"]),
        claim(3, "prediction", "Alpha", ["evals", "agents"]),
        claim(4, "opinion", "Gamma", ["rag"]),
    ]


def test_digest_evidence_is_numbered_from_one() -> None:
    evidence = digest_evidence(_claims())
    assert [e["n"] for e in evidence] == [1, 2, 3, 4]
    assert evidence[0] == {
        "n": 1,
        "claim_id": "c1",
        "claim_text": "claim 1",
        "claim_type": "measurement",
        "title": "Post 1",
        "author": "Author 1",
        "publication": "Alpha",
        "url": "https://example.com/p/1",
        "topics": ["rag", "evals"],
    }


def test_shared_topics_are_those_several_publications_wrote_about() -> None:
    shared = shared_topics(digest_evidence(_claims()))
    # "evals" came up twice but only in Alpha; "agents" once
    assert shared == [
        {"topic": "rag", "publications": ["Alpha", "Beta", "Gamma"], "claims": [1, 2, 4]}
    ]


def test_a_digest_citing_past_its_evidence_is_caught() -> None:
    evidence = digest_evidence(_claims())
    assert citations_in_bounds("Two things happened [1][4].", evidence)
    assert not citations_in_bounds("Something else [5].", evidence)
    assert not citations_in_bounds("Nothing is cited here.", evidence)  # a digest must cite
    assert citations_in_bounds("Nothing to report.", [])


def test_without_a_model_the_digest_lists_the_claims_by_type() -> None:
    evidence = digest_evidence(_claims())
    listing = plain_digest(evidence)
    assert citations_in_bounds(listing, evidence)
    measured, argued, predicted = (
        listing.index("## What was measured"),
        listing.index("## What authors argued"),
        listing.index("## Predictions to watch"),
    )
    assert measured < listing.index("claim 1") < argued < listing.index("claim 2") < predicted
    assert "claim 4 (Author 4, Post 4) [4]" in listing
    assert plain_digest([]) == "No claims were extracted from the articles in this window."


def test_the_writer_is_shown_numbered_claims_and_the_shared_topics() -> None:
    evidence = digest_evidence(_claims())
    shown = format_digest_input("Week of 2026-11-16", evidence, shared_topics(evidence))
    assert '[2] (opinion) claim 2 - Author 2, Beta, "Post 2"' in shown
    assert "rag: Alpha, Beta, Gamma (claims 1, 2, 4)" in shown


def test_the_digest_runs_on_mondays_and_on_request() -> None:
    config = generate_weekly_digest.get_config("http://localhost:8000").main
    triggers = {getattr(t, "cron", None) or getattr(t, "event", None) for t in config.triggers}
    assert triggers == {"0 8 * * 1", "cp/digest.requested"}
    assert generate_weekly_digest in FUNCTIONS and len(FUNCTIONS) == 9


def test_digest_requests_are_bounded() -> None:
    assert DigestRequest().lookback_days is None
    for bad in (0, 91):
        with pytest.raises(ValidationError):
            DigestRequest(lookback_days=bad)


async def test_requesting_a_digest_returns_a_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[inngest.Event] = []

    async def send(event: inngest.Event) -> list[str]:
        sent.append(event)
        return ["evt-digest"]

    monkeypatch.setattr(operations.client, "send", send)
    handle = await operations.queue_digest(14)
    assert (sent[0].name, sent[0].data) == ("cp/digest.requested", {"lookback_days": 14})
    assert (handle["event_id"], handle["status"]) == ("evt-digest", "queued")


async def test_no_digest_yet_is_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def nothing() -> None:
        return None

    monkeypatch.setattr(operations, "_load_latest_digest", nothing)
    with pytest.raises(ToolError, match="No digest"):
        await tools.cp_weekly_digest()


async def test_the_server_registers_thirteen_tools() -> None:
    names = {tool.name for tool in await mcp.list_tools()}
    assert len(names) == 13
    assert {"cp_research", "cp_research_result", "cp_weekly_digest"} <= names
