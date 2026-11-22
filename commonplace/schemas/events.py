"""Event names and payloads: the only contract between functions.

Producers send `event(NAME, Model(...))`; consumers start with
`Model.model_validate(ctx.event.data)`.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

import inngest
from pydantic import BaseModel

PUBLICATION_ADDED = "cp/publication.added"
ARTICLE_DISCOVERED = "cp/article.discovered"
ARTICLE_INGESTED = "cp/article.ingested"

DiscoverySource = Literal["rss", "backfill", "manual"]


class PublicationAdded(BaseModel):
    """API -> discovery. Start ingesting one allowlisted feed."""

    feed_url: str
    name: str | None = None


class ArticleDiscovered(BaseModel):
    """Discovery -> ingest. One article to fetch."""

    canonical_id: str
    publication_id: UUID
    url: str
    title: str
    author: str | None = None
    published_at: datetime
    via: DiscoverySource


class ArticleIngested(BaseModel):
    """Ingest -> whoever listens. The article's passages are stored and searchable."""

    canonical_id: str
    article_id: UUID
    content_hash: str
    passage_count: int


def event(name: str, payload: BaseModel) -> inngest.Event:
    return inngest.Event(name=name, data=payload.model_dump(mode="json"))
