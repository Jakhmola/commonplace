"""Event names and payloads: the only contract between functions.

Producers send `Model(...).model_dump(mode="json")`; consumers start with
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


class PublicationAdded(BaseModel):
    feed_url: str


class ArticleDiscovered(BaseModel):
    canonical_id: str
    publication_id: UUID
    url: str
    title: str
    author: str | None = None
    published_at: datetime
    via: Literal["rss", "manual"]


class ArticleIngested(BaseModel):
    canonical_id: str
    article_id: UUID
    content_hash: str
    passages: int


def event(name: str, payload: BaseModel) -> inngest.Event:
    return inngest.Event(name=name, data=payload.model_dump(mode="json"))
