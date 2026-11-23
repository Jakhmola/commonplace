"""How answers cite, in one place: used when an answer is produced and when it is checked.

Kept apart from the answering code so that the evaluators can use the same
parser without importing the functions they evaluate.
"""

import re
from typing import Any

# What `/ask` says by itself when retrieval comes back empty.
NO_PASSAGES = "No relevant passages were found in the knowledge base for this question."

_BRACKETS = re.compile(r"\[([\d\s,\-\u2013]+)\]")
# Six digits is far beyond any passage number, and keeps int() away from absurd input.
_PART = re.compile(r"\s*(\d{1,6})\s*(?:[-\u2013]\s*(\d{1,6})\s*)?")


def cited_numbers(answer_text: str) -> set[int]:
    """Passage numbers an answer refers to: [1], [1][2], [1, 2] and [1-3] all count.

    Brackets that are not citations, such as a date [2024-01-15], are ignored.
    A range that runs backwards or is implausibly wide is not expanded, but its
    two ends still count, so a check on citation bounds sees them.
    """
    numbers: set[int] = set()
    for group in _BRACKETS.findall(answer_text):
        parts = [_PART.fullmatch(part) for part in group.split(",")]
        if not all(parts):
            continue
        for part in parts:
            assert part is not None
            low = int(part.group(1))
            high = int(part.group(2) or low)
            numbers.update(range(low, high + 1) if 0 <= high - low < 50 else (low, high))
    return numbers


def format_context(passages: list[dict[str, Any]]) -> str:
    """Passages as numbered blocks; the numbers are what the answer cites.

    A passage the claim graph led to also shows that claim and what kind of
    statement it is, so the model can tell a measurement from an opinion.
    """
    blocks = []
    for n, p in enumerate(passages, start=1):
        author = p.get("author") or "Unknown"
        lines = [f"[{n}] (Title: {p['title']} | Author: {author} | URL: {p['url']})"]
        if p.get("claim_text"):
            lines.append(f"Attributed claim ({p.get('claim_type') or 'claim'}): {p['claim_text']}")
        lines.append(p["text"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def citation(n: int, passage: dict[str, Any]) -> dict[str, Any]:
    citation = {
        "citation_idx": n,
        "passage_id": passage["passage_id"],
        "title": passage["title"],
        "author": passage.get("author"),
        "url": passage["url"],
    }
    if passage.get("claim_id"):
        citation |= {"claim_id": passage["claim_id"], "claim_type": passage.get("claim_type")}
    return citation


def cited(answer_text: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Citations for the passages the answer actually refers to."""
    numbers = cited_numbers(answer_text)
    return [citation(n, p) for n, p in enumerate(passages, start=1) if n in numbers]
