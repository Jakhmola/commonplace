"""Polite page fetch: our User-Agent, robots.txt respected, timeout and size cap."""

from collections.abc import Callable
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


def _client(check_url: Callable[[str], None] | None = None) -> httpx.AsyncClient:
    async def check(request: httpx.Request) -> None:
        if check_url is not None:
            check_url(str(request.url))

    return httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent},
        timeout=settings.request_timeout_s,
        follow_redirects=True,
        # Runs before every request, including each redirect hop.
        event_hooks={"request": [check]},
    )


async def _robots_for(url: str) -> RobotFileParser:
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in _robots:
        parser = RobotFileParser()
        async with _client() as client:
            response = await client.get(f"{origin}/robots.txt")
        if response.status_code >= 500 or response.status_code == 429:
            response.raise_for_status()  # transient: let the step retry, cache nothing
        parser.parse(response.text.splitlines() if response.status_code == 200 else [])
        _robots[origin] = parser
    return _robots[origin]


async def fetch_article(url: str, check_url: Callable[[str], None] | None = None) -> FetchedPage:
    """Fetch one page. `check_url` is called with every URL about to be requested,
    redirects included, and may raise to stop the fetch before it leaves.
    """
    if not (await _robots_for(url)).can_fetch(settings.user_agent, url):
        raise FetchError(f"robots.txt disallows fetching {url!r}")

    async with _client(check_url) as client, client.stream("GET", url) as response:
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
