"""eval_results.judge_version

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("eval_results", sa.Column("judge_version", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("eval_results", "judge_version")
