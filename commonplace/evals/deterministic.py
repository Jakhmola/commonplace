"""Level 1: what plain code can say about an answer.

These checks cost nothing and take microseconds, so they run first and on
every answer. They cover the failures that are structural: a citation that
points nowhere, a source that has been superseded, an answer that neither
cites nor abstains. Whether a cited sentence is actually supported by its
passage takes reading, and is left to the judges.
"""

from dataclasses import dataclass
from typing import Any

from commonplace.citations import NO_PASSAGES, cited_numbers

# How an answer says "the library does not cover this". The first is what
# `/ask` returns by itself when nothing was retrieved or no key is set; the
# others are how the model is asked to say it.
_ABSTAIN_MARKERS = (
    NO_PASSAGES.lower(),
    "set anthropic_api_key",
    "passages do not cover",
    "passages don't cover",
    "passages do not address",
    "passages do not mention",
    "passages do not contain",
)


@dataclass(frozen=True)
class DeterministicEvalResult:
    passed: bool
    checks: dict[str, bool]
    failure_codes: list[str]
    score: float  # the share of checks that passed


def is_abstain_answer(answer_text: str) -> bool:
    lowered = " ".join(answer_text.lower().split())
    return any(marker in lowered for marker in _ABSTAIN_MARKERS)


def prompt_passages(retrieved: list[dict[str, Any]], prompt_limit: int = 5) -> list[dict[str, Any]]:
    """The passages the model was actually shown, in the order it saw them.

    A snapshot keeps every candidate. The ones that reached the prompt are
    flagged; snapshots from before the flag existed fall back to the first
    `prompt_limit`.
    """
    if any("in_prompt" in r for r in retrieved):
        return [r for r in retrieved if r.get("in_prompt")]
    return retrieved[:prompt_limit]


def run_deterministic_checks(
    *,
    answer_text: str,
    citations: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    prompt_limit: int = 5,
    superseded_ids: frozenset[str] = frozenset(),
) -> DeterministicEvalResult:
    """Four assertions over one answer and what it was built from.

    - `citation_bounds_valid` (FM-1): every `[n]` in the text is the number of a
      passage that was in the prompt, and is listed among the citations.
    - `citations_subset_of_retrieved` (FM-1): every citation points at the
      retrieved passage with that number.
    - `no_superseded_sources_cited` (FM-2): no cited passage or claim is in
      `superseded_ids` (looked up by the caller, so this stays pure).
    - `abstain_or_cite_discipline` (FM-3): with nothing in the prompt the answer
      must abstain; with passages it must either cite or abstain.
    """
    shown = prompt_passages(retrieved, prompt_limit)
    numbers = cited_numbers(answer_text)
    listed = {int(c["citation_idx"]): c for c in citations if "citation_idx" in c}

    bounds = all(1 <= n <= len(shown) and n in listed for n in numbers)
    subset = all(
        1 <= n <= len(shown) and c.get("passage_id") == shown[n - 1].get("passage_id")
        for n, c in listed.items()
    ) and len(listed) == len(citations)
    fresh = not any(
        c.get("passage_id") in superseded_ids or c.get("claim_id") in superseded_ids
        for c in citations
    )
    abstained = is_abstain_answer(answer_text)
    if shown:
        discipline = abstained or bool(numbers)
    else:
        discipline = abstained and not numbers and not citations

    checks = {
        "citation_bounds_valid": bounds,
        "citations_subset_of_retrieved": subset,
        "no_superseded_sources_cited": fresh,
        "abstain_or_cite_discipline": discipline,
    }
    failure_codes = [
        code
        for code, ok in (("FM-1", bounds and subset), ("FM-2", fresh), ("FM-3", discipline))
        if not ok
    ]
    return DeterministicEvalResult(
        passed=all(checks.values()),
        checks=checks,
        failure_codes=failure_codes,
        score=round(sum(checks.values()) / len(checks), 4),
    )
