"""eval_results and eval_runs

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "eval_results",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.String(),
            sa.ForeignKey("snapshots.snapshot_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("deterministic_passed", sa.Boolean(), nullable=False),
        sa.Column("checks", postgresql.JSONB(), nullable=False),
        sa.Column("faithfulness_verdict", sa.String(), nullable=False),
        sa.Column("faithfulness_critique", sa.Text(), nullable=False),
        sa.Column("disagreement_verdict", sa.String(), nullable=False),
        sa.Column("disagreement_critique", sa.Text(), nullable=False),
        sa.Column("failure_codes", postgresql.JSONB(), nullable=False),
        sa.Column("composite_score", sa.Numeric(5, 4), nullable=False),
        _created_at(),
    )
    op.create_table(
        "eval_runs",
        sa.Column("run_id", sa.String(), primary_key=True),
        sa.Column("suite_name", sa.String(), nullable=False),
        sa.Column("variants_summary", postgresql.JSONB(), nullable=False),
        sa.Column("case_results", postgresql.JSONB(), nullable=False),
        _created_at(),
    )


def downgrade() -> None:
    op.drop_table("eval_runs")
    op.drop_table("eval_results")
