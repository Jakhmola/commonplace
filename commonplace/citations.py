"""How answers cite, in one place: used when an answer is produced and when it is checked.

Kept apart from the answering code so that the evaluators can use the same
parser without importing the functions they evaluate.
"""

import re

# What `/ask` says by itself when retrieval comes back empty.
NO_PASSAGES = "No relevant passages were found in the knowledge base for this question."

_BRACKETS = re.compile(r"\[\s*(\d+(?:\s*[,\-\u2013]\s*\d+)*)\s*\]")


def cited_numbers(answer_text: str) -> set[int]:
    """Passage numbers an answer refers to: [1], [1][2], [1, 2] and [1-3] all count."""
    numbers: set[int] = set()
    for group in _BRACKETS.findall(answer_text):
        for part in group.split(","):
            first, _, last = part.replace("\u2013", "-").partition("-")
            low, high = int(first), int(last or first)
            if high - low < 50:  # a range, not two unrelated numbers far apart
                numbers.update(range(low, high + 1))
    return numbers
