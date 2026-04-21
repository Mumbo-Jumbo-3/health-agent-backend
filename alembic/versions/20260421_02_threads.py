"""create threads table for user-owned thread index

Revision ID: 20260421_02
Revises: 20260421_01
Create Date: 2026-04-21 00:00:01
"""

from alembic import op
import sqlalchemy as sa


revision = "20260421_02"
down_revision = "20260421_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "threads",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("thread_id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.clerk_user_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_threads_user_id_updated_at",
        "threads",
        ["user_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_threads_user_id_updated_at", table_name="threads")
    op.drop_table("threads")
