"""URL and id helpers. Pure functions: every writer builds ids through these.

The canonical id is the dedupe key for the whole pipeline. If two links to
the same post ever produced two ids, the post would be ingested (and paid
for) twice.
"""

import secrets
import uuid
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


_CLAIM_NAMESPACE = uuid.UUID("3f0d2a6e-58c1-4b7a-9e24-c1d7a905b6f3")


def claim_id(
    article_id: uuid.UUID | str, content_hash: str, quote_span: str, claim_text: str
) -> uuid.UUID:
    """A deterministic id for one claim in one version of an article.

    Extracting the same article version again yields the same ids, so a replay
    converges on the same rows in Postgres and the same nodes in the graph. A
    new version of the article (new content hash) yields new ids.
    """
    quote = " ".join(quote_span.split()).lower()
    claim = " ".join(claim_text.split()).lower()
    return uuid.uuid5(_CLAIM_NAMESPACE, f"{article_id}:{content_hash}:{quote}:{claim}")
