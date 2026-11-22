"""The claim extractor: reads one article, proposes claims with supporting quotes."""

import re

from commonplace.agents.core import Agent
from commonplace.agents.prompts import CLAIM_EXTRACTOR_SYSTEM_PROMPT
from commonplace.config import settings
from commonplace.graph.ontology import TOPIC_ALIASES
from commonplace.schemas.entities import ClaimExtractionBatch, ClaimType, RawExtractedClaim

claim_extractor: Agent[ClaimExtractionBatch] = Agent(
    name="claim-extractor",
    role="Extracts attributed claims, each with a verbatim supporting quote",
    system_prompt=CLAIM_EXTRACTOR_SYSTEM_PROMPT,
    model=settings.extraction_model,
    output_schema=ClaimExtractionBatch,
    max_tokens=settings.extraction_max_tokens,
    effort=settings.extraction_effort,
)


def format_extractor_prompt(
    *, title: str, author: str | None, publication_name: str, article_text: str
) -> str:
    return (
        f"Publication: {publication_name}\n"
        f"Title: {title}\n"
        f"Author: {author or 'Unknown'}\n\n"
        f"Article text:\n{article_text}"
    )


# Known topics as they appear in running text: "fine-tuning" also matches "fine tuning",
# and a trailing "s" is allowed so plurals match.
_TOPIC_MENTIONS = [
    (re.compile(r"\b" + "[ -]".join(map(re.escape, alias.split("-"))) + r"s?\b"), name)
    for alias, (_slug, name) in TOPIC_ALIASES.items()
    if len(alias) > 2
]
_MEASURED = re.compile(r"\d+(?:\.\d+)?\s?(?:%|x\b|ms\b|seconds?\b|tokens?\b|usd\b|\$)")
_PREDICTED = re.compile(r"\b(?:will|going to|within \w+ years?|by 20\d\d|eventually)\b")


def heuristic_extract_claims(article_text: str, *, max_claims: int = 5) -> ClaimExtractionBatch:
    """A stand-in for the model when no API key is set.

    Takes sentences that mention a known topic and uses each sentence as both
    the claim and its quote, so everything downstream (guardrail, storage,
    graph) can be exercised without a paid call. The claims are crude; the
    quotes are real.
    """
    claims: list[RawExtractedClaim] = []
    for sentence in re.split(r"(?<=[.!?])\s+", article_text):
        sentence = sentence.strip()
        lowered = sentence.lower()
        entities = list(
            dict.fromkeys(n for pattern, n in _TOPIC_MENTIONS if pattern.search(lowered))
        )
        if len(sentence) < 25 or not entities:
            continue
        claim_type: ClaimType = "opinion"
        if _MEASURED.search(lowered):
            claim_type = "measurement"
        elif _PREDICTED.search(lowered):
            claim_type = "prediction"
        claims.append(
            RawExtractedClaim(
                claim_text=sentence,
                quote_span=sentence,
                claim_type=claim_type,
                entities=entities[:5],
                confidence=0.5,
            )
        )
        if len(claims) >= max_claims:
            break
    return ClaimExtractionBatch(claims=claims)
