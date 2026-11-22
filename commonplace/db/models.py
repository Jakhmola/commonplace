"""Tables. Alembic compares migrations against this metadata (`alembic check`).

Every write goes through an upsert in `queries.py`, so replaying any step is safe.
"""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from commonplace.config import settings


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[uuid.UUID] = _pk()
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    feed_url: Mapped[str] = mapped_column(String, nullable=False)
    homepage: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool] = mapped_column(default=True, nullable=False)


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[uuid.UUID] = _pk()
    canonical_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    publication_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("publications.id"), nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    author: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="ingested")
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()


class Passage(Base):
    __tablename__ = "passages"
    __table_args__ = (
        # Unique among current passages only: superseded versions keep their chunk index.
        Index(
            "uq_passages_article_chunk",
            "article_id",
            "chunk_index",
            unique=True,
            postgresql_where=text("valid_to IS NULL"),
        ),
        Index("ix_passages_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = _pk()
    article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("articles.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimensions), nullable=True
    )
    # Derived from `text` by Postgres (generated, stored). The app never writes it.
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True), nullable=True
    )
    # A passage is current while valid_to is NULL. Searches only see current passages.
    valid_from: Mapped[datetime] = _now()
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Cursor(Base):
    """Per-feed polling state (conditional requests). Filled in by the freshness poller."""

    __tablename__ = "cursors"

    publication_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("publications.id"), primary_key=True
    )
    etag: Mapped[str | None] = mapped_column(String, nullable=True)
    last_modified: Mapped[str | None] = mapped_column(String, nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    poll_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    not_modified_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Job(Base):
    """One row per triggering event, so the API can answer "how far along is it?"."""

    __tablename__ = "jobs"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    publication_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("publications.id"), nullable=True
    )
    # The articles this job expects. Progress is counted from these, so a
    # publication's older articles never make a new job look finished.
    canonical_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    created_at: Mapped[datetime] = _now()


class Cost(Base):
    __tablename__ = "costs"

    id: Mapped[uuid.UUID] = _pk()
    article_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("articles.id"), nullable=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    est_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0)
    at: Mapped[datetime] = _now()


class Snapshot(Base):
    """One row per answer: the question, the answer, and every candidate passage
    the retriever surfaced (not only the cited ones) with its rank in each engine.

    With only the cited passages stored there is no telling "the right passage was
    never retrieved" from "it was retrieved and ignored", and those need different fixes.
    """

    __tablename__ = "snapshots"
    __table_args__ = (Index("ix_snapshots_created_at", "created_at"),)

    snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    retrieved: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    retriever_variant: Mapped[str] = mapped_column(String, nullable=False)
    # Publication date of the newest source the answer rests on.
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Written by the answer evaluators; NULL until an answer has been graded.
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _now()


class Failure(Base):
    """A run that failed for good, as reported by Inngest's own failure event.

    Real failures are worth keeping: they say which articles never made it in,
    and they are the raw material for judging the system later.
    """

    __tablename__ = "failures"

    id: Mapped[uuid.UUID] = _pk()
    function_id: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    # The event that started the failed run.
    event_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = _now()
