import calendar
from datetime import UTC, datetime

import feedparser
from pydantic import BaseModel

from commonplace.sources.urls import clean_url


class FeedEntry(BaseModel):
    """What we take from a feed: a pointer to the article, never its content.

    Feeds may carry the full post, a summary, or nothing; ingest always fetches
    the page itself so every article is parsed the same way.
    """

    url: str
    title: str
    author: str | None
    published_at: datetime


def parse_feed(raw: bytes) -> list[FeedEntry]:
    feed = feedparser.parse(raw)
    entries = []
    for e in feed.entries:
        when = e.get("published_parsed") or e.get("updated_parsed")
        if not e.get("link") or when is None:
            continue
        entries.append(
            FeedEntry(
                url=clean_url(e.link),
                title=e.get("title", e.link),
                author=e.get("author"),
                published_at=datetime.fromtimestamp(calendar.timegm(when), UTC),
            )
        )
    return sorted(entries, key=lambda x: x.published_at, reverse=True)
