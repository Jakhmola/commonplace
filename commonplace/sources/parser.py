"""Article page -> clean text and metadata."""

import contextlib
import hashlib
from datetime import UTC, datetime

import trafilatura
from pydantic import BaseModel


class ParsedArticle(BaseModel):
    title: str
    author: str | None
    published_at: datetime
    text: str
    content_hash: str


def parse_html(html: str, url: str, *, fallback_published_at: datetime) -> ParsedArticle:
    """Main text with nav, subscribe prompts and footer removed.

    The page's own date wins; `fallback_published_at` (the feed's date) is
    used only when the page has none.
    """
    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False, favor_precision=True
    )
    if not text:
        raise ValueError(f"no article text found at {url!r}")

    meta = trafilatura.extract_metadata(html, default_url=url)
    published_at = fallback_published_at
    if meta and meta.date:
        with contextlib.suppress(ValueError):
            published_at = datetime.fromisoformat(meta.date).replace(tzinfo=UTC)

    return ParsedArticle(
        title=(meta.title if meta and meta.title else url),
        author=meta.author if meta else None,
        published_at=published_at,
        text=text,
        # The hash of the extracted text is what "did this article change?" compares.
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
