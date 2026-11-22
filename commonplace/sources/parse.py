import hashlib
from datetime import UTC, datetime

import trafilatura
from pydantic import BaseModel


class ArticleText(BaseModel):
    title: str
    author: str | None
    published_at: datetime | None
    text: str
    content_hash: str


def extract_article(html: str, url: str) -> ArticleText:
    """Main text and metadata from an article page, with nav/subscribe/footer removed."""
    text = trafilatura.extract(html, url=url, include_comments=False, favor_precision=True)
    if not text:
        raise ValueError(f"no article text found at {url!r}")
    meta = trafilatura.extract_metadata(html, default_url=url)
    published_at = None
    if meta and meta.date:
        try:
            published_at = datetime.fromisoformat(meta.date).replace(tzinfo=UTC)
        except ValueError:
            published_at = None
    return ArticleText(
        title=(meta.title if meta and meta.title else url),
        author=meta.author if meta else None,
        published_at=published_at,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
