"""Polite page fetch: our User-Agent, robots.txt respected, timeout and size cap."""

from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from commonplace.config import settings


class FetchError(Exception):
    """robots.txt says no, or the body is too big. Fetching again will not help."""


@dataclass
class FetchedPage:
    url: str  # after redirects
    html: str
    status_code: int


_robots: dict[str, RobotFileParser] = {}  # ponytail: per-process cache, never expires


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent},
        timeout=settings.request_timeout_s,
        follow_redirects=True,
    )


async def _robots_for(url: str) -> RobotFileParser:
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in _robots:
        parser = RobotFileParser()
        async with _client() as client:
            response = await client.get(f"{origin}/robots.txt")
        if response.status_code >= 500:
            response.raise_for_status()  # transient: let the step retry
        parser.parse(response.text.splitlines() if response.status_code == 200 else [])
        _robots[origin] = parser
    return _robots[origin]


async def fetch_article(url: str) -> FetchedPage:
    if not (await _robots_for(url)).can_fetch(settings.user_agent, url):
        raise FetchError(f"robots.txt disallows fetching {url!r}")

    async with _client() as client, client.stream("GET", url) as response:
        response.raise_for_status()
        body = bytearray()
        async for part in response.aiter_bytes():
            body += part
            if len(body) > settings.max_article_bytes:
                raise FetchError(f"{url!r} is larger than {settings.max_article_bytes} bytes")
        return FetchedPage(
            url=str(response.url),
            html=bytes(body).decode(response.encoding or "utf-8", errors="replace"),
            status_code=response.status_code,
        )
