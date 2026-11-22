"""Unit tests for the research agent, the weekly digest and the production packaging.

No Docker or keys needed.
"""

from typing import Any

import pytest
from pydantic import ValidationError

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
    return {"passage_id": passage_id, "rrf_score": score, "text": passage_id, **extra}


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
