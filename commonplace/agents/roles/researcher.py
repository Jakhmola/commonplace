"""The research agent's roles, and the plain code that keeps its loop honest.

Three roles, each one model call with one job: a planner turns a question into
a few searches, a critic reads what the searches found and says whether it is
enough, a synthesizer writes the report. A fourth role writes the weekly digest.

What decides when the loop ends is not a model. `stop_reason_after_hop` and
`fresh_queries` are ordinary functions: an iteration cap, a dollar cap, and a
check that a hop actually found something new.
"""

from typing import Any

from pydantic import BaseModel, Field

from commonplace.agents.core import Agent
from commonplace.agents.prompts import (
    COVERAGE_CRITIC_SYSTEM_PROMPT,
    DIGEST_WRITER_SYSTEM_PROMPT,
    RESEARCH_PLANNER_SYSTEM_PROMPT,
    RESEARCH_SYNTHESIZER_SYSTEM_PROMPT,
)
from commonplace.citations import format_context
from commonplace.config import settings


class ResearchPlan(BaseModel):
    rationale: str = Field(description="Why these searches, in a sentence or two")
    subqueries: list[str] = Field(
        min_length=1, description="One to three searches to run first, most important first"
    )
    target_topics: list[str] = Field(
        default_factory=list,
        description="Technical topics a complete answer would have to cover",
    )


class CoverageReflection(BaseModel):
    rationale: str = Field(description="What the evidence covers and what it does not")
    is_sufficient: bool = Field(
        description="True if the question can be answered well from the evidence gathered so far"
    )
    missing_aspects: list[str] = Field(
        default_factory=list, description="Parts of the question the evidence does not address"
    )
    followup_queries: list[str] = Field(
        default_factory=list,
        description="Up to two new searches for what is missing; none if the evidence is enough",
    )


def _role[T: BaseModel](
    name: str, role: str, system_prompt: str, output_schema: type[T] | None = None
) -> Agent[T]:
    return Agent(
        name=name,
        role=role,
        system_prompt=system_prompt,
        model=settings.research_model,
        output_schema=output_schema,
        max_tokens=settings.research_max_tokens,
        effort=settings.research_effort,
    )


query_planner: Agent[ResearchPlan] = _role(
    "query-planner",
    "Breaks a question into the first few searches",
    RESEARCH_PLANNER_SYSTEM_PROMPT,
    ResearchPlan,
)
coverage_critic: Agent[CoverageReflection] = _role(
    "coverage-critic",
    "Decides whether the evidence gathered so far is enough, or what to search next",
    COVERAGE_CRITIC_SYSTEM_PROMPT,
    CoverageReflection,
)
research_synthesizer: Agent[Any] = _role(
    "research-synthesizer",
    "Writes the cited report from the gathered evidence",
    RESEARCH_SYNTHESIZER_SYSTEM_PROMPT,
)
digest_writer: Agent[Any] = _role(
    "digest-writer",
    "Writes the weekly briefing from the week's claims",
    DIGEST_WRITER_SYSTEM_PROMPT,
)

_CLAIM_FIELDS = ("claim_id", "claim_text", "claim_type")


def merge_hop_candidates(
    pool: list[dict[str, Any]], new: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """(the pool with one hop's results merged in, how many passages were new).

    One entry per passage. A passage found again keeps its best score, and a
    claim the graph attached to it on any hop stays attached. The order is by
    score, then passage id, so the same hops always give the same pool.
    """
    by_id = {p["passage_id"]: dict(p) for p in pool}
    added = 0
    for hit in new:
        known = by_id.get(hit["passage_id"])
        if known is None:
            by_id[hit["passage_id"]] = dict(hit)
            added += 1
            continue
        claim = {f: known.get(f) or hit.get(f) for f in _CLAIM_FIELDS if f in known or f in hit}
        best = hit if hit["rrf_score"] > known["rrf_score"] else known
        by_id[hit["passage_id"]] = {**best, **claim}
    merged = sorted(by_id.values(), key=lambda p: (-p["rrf_score"], p["passage_id"]))
    return merged, added


def _normalised(query: str) -> str:
    return " ".join(query.lower().split())


def fresh_queries(candidates: list[str], executed: list[str], limit: int) -> list[str]:
    """The proposed searches that have not been run yet, in order, at most `limit`.

    A critic that keeps proposing what was already searched gets an empty list
    back, and the loop ends instead of paying for the same hop again.
    """
    seen = {_normalised(q) for q in executed}
    fresh: list[str] = []
    for query in candidates:
        key = _normalised(query)
        if key and key not in seen:
            seen.add(key)
            fresh.append(query.strip())
    return fresh[:limit]


def stop_reason_after_hop(
    *, iteration: int, max_iterations: int, spent_usd: float, max_usd: float, added: int
) -> str | None:
    """Why the loop must end after this hop, or None if it may reflect and go on.

    Checked before the critic is asked, so a run that has to stop anyway does
    not pay for one more opinion.
    """
    if added == 0 and iteration > 1:
        return "converged"
    if iteration >= max_iterations:
        return "max_iterations_reached"
    if spent_usd >= max_usd:
        return "budget_exceeded"
    return None


def format_plan_input(question: str) -> str:
    return (
        f"Question: {question}\n\nPlan at most {settings.research_max_subqueries_per_hop} searches."
    )


def format_reflection_input(
    *, question: str, executed: list[str], pool: list[dict[str, Any]], limit: int
) -> str:
    """What the critic reads: the question, every search run so far, and the best
    `limit` passages of the pool, numbered as the report would cite them.
    """
    searches = "\n".join(f"- {q}" for q in executed)
    return (
        f"Question: {question}\n\n"
        f"Searches already run:\n{searches}\n\n"
        f"Evidence gathered so far ({len(pool)} passages; the best {min(limit, len(pool))} "
        f"are shown):\n\n{format_context(pool[:limit])}"
    )


def format_report_input(question: str, passages: list[dict[str, Any]]) -> str:
    return (
        f"Passages:\n\n{format_context(passages)}\n\n"
        f"Question: {question}\n\n"
        "Write the report from the passages, citing them inline."
    )
