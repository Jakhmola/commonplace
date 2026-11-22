"""Unit tests for the research agent, the weekly digest and the production packaging.

No Docker or keys needed.
"""

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
from commonplace.functions.research import run_research_agent
from commonplace.main import ResearchRequest
from commonplace.mcp import tools
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
