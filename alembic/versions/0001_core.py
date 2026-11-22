"""core tables: publications, articles, passages, cursors, jobs, costs

Revision ID: 0001
Revises:
"""

import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _now(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "publications",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("slug", sa.String(), nullable=False, unique=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("feed_url", sa.String(), nullable=False),
        sa.Column("homepage", sa.String(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
    )
    op.create_table(
        "articles",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("canonical_id", sa.String(), nullable=False, unique=True),
        sa.Column("publication_id", sa.UUID(), sa.ForeignKey("publications.id"), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("author", sa.String(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        _now("created_at"),
        _now("updated_at"),
    )
    op.create_table(
        "passages",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("article_id", sa.UUID(), sa.ForeignKey("articles.id"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=1536), nullable=True),
        # The app never writes this column: Postgres derives it from `text`.
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=True,
        ),
        _now("valid_from"),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("uq_passages_article_chunk", "passages", ["article_id", "chunk_index"], unique=True)
    op.create_index("ix_passages_tsv", "passages", ["tsv"], postgresql_using="gin")
    # HNSW can be built on an empty table (IVFFlat needs data to train on).
    op.execute(
        "CREATE INDEX ix_passages_embedding_hnsw ON passages USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_table(
        "cursors",
        sa.Column("publication_id", sa.UUID(), sa.ForeignKey("publications.id"), primary_key=True),
        sa.Column("etag", sa.String(), nullable=True),
        sa.Column("last_modified", sa.String(), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("poll_count", sa.Integer(), nullable=False),
        sa.Column("not_modified_count", sa.Integer(), nullable=False),
    )
    op.create_table(
        "jobs",
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("done", sa.Integer(), nullable=False),
        sa.Column("publication_id", sa.UUID(), sa.ForeignKey("publications.id"), nullable=True),
        sa.Column("canonical_ids", postgresql.ARRAY(sa.String()), nullable=False),
        _now("created_at"),
    )
    op.create_table(
        "costs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("article_id", sa.UUID(), sa.ForeignKey("articles.id"), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("est_usd", sa.Numeric(10, 6), nullable=False),
        _now("at"),
    )


def downgrade() -> None:
    for table in ("costs", "jobs", "cursors", "passages", "articles", "publications"):
        op.drop_table(table)
