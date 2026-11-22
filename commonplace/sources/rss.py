"""Feeds give us pointers to articles, never the articles themselves.

A feed may carry the full post, a summary, or nothing. Ingest always fetches
the page, so every article is parsed the same way.
"""

import calendar
from dataclasses import dataclass
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


@dataclass
class FeedResponse:
    status: int  # 200 or 304
    content: bytes | None  # None when the feed has not changed
    etag: str | None
    last_modified: str | None


async def fetch_feed_conditional(
    feed_url: str, etag: str | None = None, last_modified: str | None = None
) -> FeedResponse:
    """Ask for the feed only if it changed since the validators we hold.

    A server that supports conditional requests answers 304 with no body, so
    polling an unchanged feed costs one small round trip and no parsing. A
    server that does not simply answers 200 every time.
    """
    conditions = {}
    if etag:
        conditions["If-None-Match"] = etag
    if last_modified:
        conditions["If-Modified-Since"] = last_modified

    async with httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent},
        timeout=settings.request_timeout_s,
        follow_redirects=True,
    ) as client:
        response = await client.get(feed_url, headers=conditions)

    if response.status_code == 304:
        return FeedResponse(304, None, etag, last_modified)
    response.raise_for_status()
    return FeedResponse(
        200,
        response.content,
        response.headers.get("ETag") or etag,
        response.headers.get("Last-Modified") or last_modified,
    )


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
