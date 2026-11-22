"""Polite HTTP GET: our User-Agent, robots.txt respected, timeout and size cap."""

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from commonplace.config import settings


class FetchRefused(Exception):
    """robots.txt says no, or the body is too big. Retrying will not help."""


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
        rp = RobotFileParser()
        async with _client() as c:
            r = await c.get(f"{origin}/robots.txt")
        if r.status_code >= 500:
            r.raise_for_status()  # transient, let the step retry
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        _robots[origin] = rp
    return _robots[origin]


async def fetch(url: str) -> tuple[str, bytes]:
    """Returns (final_url, body)."""
    if not (await _robots_for(url)).can_fetch(settings.user_agent, url):
        raise FetchRefused(f"robots.txt disallows {url!r}")
    async with _client() as c, c.stream("GET", url) as r:
        r.raise_for_status()
        body = bytearray()
        async for part in r.aiter_bytes():
            body += part
            if len(body) > settings.max_page_bytes:
                raise FetchRefused(f"{url!r} is larger than {settings.max_page_bytes} bytes")
        return str(r.url), bytes(body)
