"""failures table; passages unique only among current versions

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "failures",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("function_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("event_data", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
    )
    # A superseded passage keeps its chunk index, so uniqueness can only hold
    # among the passages that are current.
    op.drop_index("uq_passages_article_chunk", table_name="passages")
    op.create_index(
        "uq_passages_article_chunk",
        "passages",
        ["article_id", "chunk_index"],
        unique=True,
        postgresql_where=sa.text("valid_to IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_passages_article_chunk", table_name="passages")
    op.create_index(
        "uq_passages_article_chunk", "passages", ["article_id", "chunk_index"], unique=True
    )
    op.drop_table("failures")
