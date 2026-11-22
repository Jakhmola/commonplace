"""claims: quote-verified assertions extracted from articles

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "claims",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("article_id", sa.UUID(), sa.ForeignKey("articles.id"), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("quote_span", sa.Text(), nullable=False),
        sa.Column("claim_type", sa.String(), nullable=False),
        sa.Column("entities", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column(
            "valid_from", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes_id", sa.UUID(), sa.ForeignKey("claims.id"), nullable=True),
    )
    op.create_index(
        "ix_claims_article_active",
        "claims",
        ["article_id"],
        postgresql_where=sa.text("valid_to IS NULL"),
    )


def downgrade() -> None:
    op.drop_table("claims")
