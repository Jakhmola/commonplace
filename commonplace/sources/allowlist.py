"""Only feeds listed in publications.yaml are ever fetched.

Article URLs are checked too: a feed could link anywhere, and we only follow
links that stay on the publication's own host.
"""

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from commonplace.sources.urls import host_of

PUBLICATIONS_FILE = Path(__file__).resolve().parents[2] / "publications.yaml"


class NotAllowed(Exception):
    pass


class Publication(BaseModel):
    slug: str
    name: str
    feed_url: str
    homepage: str


@cache
def load_publications(path: Path = PUBLICATIONS_FILE) -> tuple[Publication, ...]:
    raw = yaml.safe_load(path.read_text())
    return tuple(Publication.model_validate(p) for p in raw["publications"])


def find_by_feed(
    feed_url: str, pubs: tuple[Publication, ...] | list[Publication]
) -> Publication | None:
    host = host_of(feed_url)
    return next((p for p in pubs if host_of(p.feed_url) == host), None)


def assert_allowed(url: str, pubs: tuple[Publication, ...] | list[Publication]) -> Publication:
    host = host_of(url)
    for p in pubs:
        if host in (host_of(p.homepage), host_of(p.feed_url)):
            return p
    raise NotAllowed(f"{url!r} is not on an allowlisted publication host")
