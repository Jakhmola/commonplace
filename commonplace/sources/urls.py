from urllib.parse import urlsplit, urlunsplit


def host_of(url: str) -> str:
    host = (urlsplit(url.strip()).hostname or "").lower()
    return host.removeprefix("www.")


def canonical_id(url: str) -> str:
    """The dedupe key for anything with a URL: `host/path`.

    Scheme, `www.`, query string (utm and share params), fragment and trailing
    slash are all dropped, so every link variant of a post collapses to one id.
    """
    path = urlsplit(url.strip()).path.rstrip("/")
    return f"{host_of(url)}{path}"


def clean_url(url: str) -> str:
    """The fetchable form of a link: tracking query and fragment removed."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme or "https", parts.netloc.lower(), parts.path, "", ""))
