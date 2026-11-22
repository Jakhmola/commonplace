"""research_runs and digests

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("research_id", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stop_reason", sa.String(), nullable=True),
        sa.Column("plan", postgresql.JSONB(), nullable=False),
        sa.Column("hops", postgresql.JSONB(), nullable=False),
        sa.Column("report_text", sa.Text(), nullable=True),
        sa.Column("citations", postgresql.JSONB(), nullable=False),
        sa.Column("retrieved", postgresql.JSONB(), nullable=False),
        sa.Column("eval_summary", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("iterations_used", sa.Integer(), nullable=False),
        sa.Column("total_cost_usd", sa.Numeric(10, 6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        _created_at(),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "digests",
        sa.Column("digest_id", sa.UUID(), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("summary_markdown", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=False),
        sa.Column("stats", postgresql.JSONB(), nullable=False),
        _created_at(),
    )


def downgrade() -> None:
    op.drop_table("digests")
    op.drop_table("research_runs")
