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
CLAIMS_EXTRACTED = "cp/claims.extracted"
ANSWER_PRODUCED = "cp/answer.produced"
ANSWER_EVALUATED = "cp/answer.evaluated"
EVAL_RUN_REQUESTED = "cp/eval.run"

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


class ClaimsExtracted(BaseModel):
    """Extraction -> whoever listens. How many claims an article yielded, and how
    many the quote guardrail threw away.
    """

    article_id: UUID
    canonical_id: str
    claim_count: int
    dropped_spans: int = 0


class AnswerProduced(BaseModel):
    """Ask -> whoever grades answers. Points at a stored snapshot."""

    snapshot_id: str
    retriever_variant: str


class AnswerEvaluated(BaseModel):
    """Evaluator -> whoever listens. How one answer scored."""

    snapshot_id: str
    composite_score: float
    deterministic_passed: bool
    faithfulness_verdict: str
    disagreement_verdict: str
    failure_codes: list[str]


class EvalRunRequested(BaseModel):
    """Anyone -> benchmark suite. Compare the retrieval variants on the golden questions."""

    suite_name: str = "golden_v1"


def event(name: str, payload: BaseModel) -> inngest.Event:
    return inngest.Event(name=name, data=payload.model_dump(mode="json"))
