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
FEED_POLL = "cp/feed.poll"
ARTICLE_DISCOVERED = "cp/article.discovered"
ARTICLE_INGESTED = "cp/article.ingested"
ARTICLE_UPDATED = "cp/article.updated"
ANSWER_PRODUCED = "cp/answer.produced"

DiscoverySource = Literal["rss", "backfill", "manual", "poll"]


class PublicationAdded(BaseModel):
    """API -> discovery. Start ingesting one allowlisted feed."""

    feed_url: str
    name: str | None = None


class FeedPoll(BaseModel):
    """Anyone -> poller. Check feeds now instead of waiting for the schedule."""

    publication: str | None = None  # a slug, or None for every feed


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


class ArticleUpdated(BaseModel):
    """Ingest -> whoever listens. An article changed: its old passages were superseded."""

    canonical_id: str
    article_id: UUID
    content_hash: str
    passage_count: int


class AnswerProduced(BaseModel):
    """Ask -> whoever grades answers. Points at a stored snapshot."""

    snapshot_id: str
    retriever_variant: str


def event(name: str, payload: BaseModel) -> inngest.Event:
    return inngest.Event(name=name, data=payload.model_dump(mode="json"))
