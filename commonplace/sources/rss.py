"""Feeds give us pointers to articles, never the articles themselves.

A feed may carry the full post, a summary, or nothing. Ingest always fetches
the page, so every article is parsed the same way.
"""

import calendar
from datetime import UTC, datetime

import feedparser
import httpx
from pydantic import BaseModel

from commonplace.config import settings
from commonplace.schemas.ids import clean_url


class FeedEntry(BaseModel):
    title: str
    url: str
    author: str | None = None
    published_at: datetime


async def fetch_feed(feed_url: str) -> bytes:
    async with httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent},
        timeout=settings.request_timeout_s,
        follow_redirects=True,
    ) as client:
        response = await client.get(feed_url)
        response.raise_for_status()
        return response.content


def parse_feed(raw: bytes) -> list[FeedEntry]:
    """Entries in feed order. Unusable entries are skipped; an unusable feed is an error."""
    feed = feedparser.parse(raw)
    if feed.bozo and not feed.entries:
        raise ValueError(f"Feed did not parse: {feed.get('bozo_exception')}")

    entries = []
    for item in feed.entries:
        link, title = item.get("link"), item.get("title")
        when = item.get("published_parsed") or item.get("updated_parsed")
        if not link or not title or when is None:
            continue
        entries.append(
            FeedEntry(
                title=title,
                url=clean_url(link),  # feeds append ?utm_source=... to every link
                author=item.get("author"),
                published_at=datetime.fromtimestamp(calendar.timegm(when), UTC),
            )
        )
    return entries
