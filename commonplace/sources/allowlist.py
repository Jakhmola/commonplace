"""publications.yaml is the allowlist: nothing outside its hosts is ever fetched.

Article URLs are checked as well as feeds, because a feed can link anywhere.
"""

from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel

from commonplace.config import settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


class PublicationEntry(BaseModel):
    slug: str
    name: str
    feed_url: str
    homepage: str
    notes: str | None = None
    # Other places the same publication lives (a second domain, an archive page).
    archive_urls: list[str] | None = None
    sitemap_url: str | None = None

    def hosts(self) -> set[str]:
        urls = [self.homepage, self.feed_url, *(self.archive_urls or [])]
        if self.sitemap_url:
            urls.append(self.sitemap_url)
        return {_host(u) for u in urls} - {""}


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def load_publications(path: str | Path | None = None) -> list[PublicationEntry]:
    file = Path(path or settings.publications_file)
    if not file.is_absolute():
        file = _REPO_ROOT / file  # independent of the process's working directory
    raw = yaml.safe_load(file.read_text())
    return [PublicationEntry.model_validate(entry) for entry in raw["publications"]]


def find_publication_for_url(
    url: str, publications: list[PublicationEntry]
) -> PublicationEntry | None:
    """Exact host match only: no wildcards, so lookalike hosts and unlisted subdomains fail."""
    host = _host(url)
    return next((pub for pub in publications if host and host in pub.hosts()), None)


def assert_allowed(url: str, publications: list[PublicationEntry]) -> PublicationEntry:
    pub = find_publication_for_url(url, publications)
    if pub is None:
        raise ValueError(f"URL is outside the allowlist (publications.yaml): {url!r}")
    return pub
