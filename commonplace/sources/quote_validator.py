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
# comparing.
_TYPOGRAPHY = {
    0x2018: "'",  # left single quote
    0x2019: "'",  # right single quote, apostrophe
    0x201C: '"',  # left double quote
    0x201D: '"',  # right double quote
    0x2010: "-",  # hyphen
    0x2011: "-",  # non-breaking hyphen
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2212: "-",  # minus sign
    0x00A0: " ",  # no-break space
}


def _fold(text: str) -> tuple[str, list[int]]:
    """The form of `text` used for comparing, and where each of its characters came from.

    Folding normalises Unicode (a superscript 2 becomes "2"), maps typographic
    variants to their keyboard forms and collapses every run of whitespace to
    one space. The second value gives, for each character of the folded text,
    the index in `text` of the character it was made from, which is what lets
    a match found in the folded text be cut out of the original.
    """
    folded: list[str] = []
    origin: list[int] = []
    space_at: int | None = None  # a run of whitespace waiting to become one space
    for index, original in enumerate(text):
        for char in unicodedata.normalize("NFKC", original).translate(_TYPOGRAPHY):
            if char.isspace():
                if folded and space_at is None:
                    space_at = index
                continue
            if space_at is not None:
                folded.append(" ")
                origin.append(space_at)
                space_at = None
            folded.append(char)
            origin.append(index)
    return "".join(folded), origin


def fold(text: str) -> str:
    """`text` as it is compared: Unicode-normalised, typography folded, whitespace collapsed."""
    return _fold(text)[0]


def find_verbatim(
    article_text: str, quote_span: str, *, min_chars: int = MIN_QUOTE_SPAN_CHARS
) -> str | None:
    """The article's own text for `quote_span`, or None if the quote is not in the article.

    Line breaks and typographic variants are ignored when matching. The return
    value is cut from the article as it was written, so what gets stored is
    the source's characters (its superscripts, its curly quotes), not the
    model's retyping of them. Only line breaks inside it are replaced by spaces.
    """
    quote = fold(quote_span)
    if len(quote) < min_chars:
        return None
    folded, origin = _fold(article_text)
    start = folded.find(quote)
    if start < 0:
        return None
    first, last = origin[start], origin[start + len(quote) - 1]
    return re.sub(r"\s+", " ", article_text[first : last + 1])


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
        text = " ".join(raw.claim_text.split())
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
