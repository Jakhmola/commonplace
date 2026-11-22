"""URL and id helpers. Pure functions: every writer builds ids through these.

The canonical id is the dedupe key for the whole pipeline. If two links to
the same post ever produced two ids, the post would be ingested (and paid
for) twice.
"""

import secrets
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """One spelling per page: https, lowercase host, no port, query, fragment or trailing slash."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    return urlunsplit(("https", host, parts.path.rstrip("/") or "/", "", ""))


def clean_url(url: str) -> str:
    """The form we actually fetch: the link as published, minus tracking query and fragment."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme or "https", parts.netloc.lower(), parts.path, "", ""))


def canonical_id(publication_slug: str, url: str) -> str:
    """`pub:<publication>:<post-slug>`.

    The slug comes from the allowlist entry, not from the URL's host, so a
    publication reachable under two domains still yields one id per post.
    """
    segments = [s for s in urlsplit(normalize_url(url)).path.split("/") if s]
    if not segments:
        raise ValueError(f"URL has no path segment to use as a post slug: {url!r}")
    return f"pub:{publication_slug}:{segments[-1]}"


def snapshot_id() -> str:
    """`snap_<12 url-safe chars>`: names one answer, and everything recorded about it later."""
    return f"snap_{secrets.token_urlsafe(9)}"
