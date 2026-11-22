"""The quote guardrail: a claim is kept only if its quote is really in the article.

A language model asked to quote will sometimes paraphrase, tidy punctuation, or
join two sentences with an ellipsis, and the result reads as a quotation. This
check is plain string matching, costs nothing, and runs before any claim
touches Postgres or the graph. What fails it is dropped and counted.
"""

import re
import unicodedata
import uuid

from commonplace.graph.ontology import normalize_topic, normalize_topics
from commonplace.schemas.entities import RawExtractedClaim, ValidatedClaim
from commonplace.schemas.ids import claim_id

MIN_QUOTE_SPAN_CHARS = 15

# Models retype curly quotes and long dashes as their keyboard forms. That is a
# difference of typography, not of words, so both sides are folded before
# comparing. Each replacement is one character for one, which keeps positions
# in the folded text aligned with the original.
_TYPOGRAPHY = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
    }
)


def _collapse(text: str) -> str:
    """Unicode-normalised, with every run of whitespace as one space."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def find_verbatim(
    article_text: str, quote_span: str, *, min_chars: int = MIN_QUOTE_SPAN_CHARS
) -> str | None:
    """The article's own text for `quote_span`, or None if the quote is not in the article.

    Line breaks and typographic variants are ignored when matching. The return
    value is cut from the article, so what gets stored is the source's wording
    and punctuation, not the model's retyping of it.
    """
    quote = _collapse(quote_span)
    if len(quote) < min_chars:
        return None
    article = _collapse(article_text)
    start = article.translate(_TYPOGRAPHY).find(quote.translate(_TYPOGRAPHY))
    return article[start : start + len(quote)] if start >= 0 else None


def validate_and_normalize_claims(
    article_id: uuid.UUID | str,
    content_hash: str,
    article_text: str,
    raw_claims: list[RawExtractedClaim],
    *,
    fallback_topic: str | None = None,
) -> tuple[list[ValidatedClaim], int]:
    """(claims that passed, how many were dropped).

    A claim is dropped when its quote is not verbatim, or when none of its
    topics survives canonicalisation and there is no fallback. A repeat of a
    claim already kept is skipped without counting as a drop.
    """
    fallback = normalize_topic(fallback_topic) if fallback_topic else None
    kept: dict[str, ValidatedClaim] = {}
    dropped = 0
    for raw in raw_claims:
        quote = find_verbatim(article_text, raw.quote_span)
        topics = normalize_topics(raw.entities) or ([fallback] if fallback else [])
        if quote is None or not topics:
            dropped += 1
            continue
        text = _collapse(raw.claim_text)
        cid = str(claim_id(article_id, content_hash, quote, text))
        kept.setdefault(
            cid,
            ValidatedClaim(
                claim_id=cid,
                article_id=str(article_id),
                text=text,
                quote_span=quote,
                claim_type=raw.claim_type,
                confidence=raw.confidence,
                topics=topics,
            ),
        )
    return list(kept.values()), dropped
