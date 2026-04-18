"""create shared_conversations table

Revision ID: 20260417_01
Revises: 20260416_01
Create Date: 2026-04-17 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260417_01"
down_revision = "20260416_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shared_conversations",
        sa.Column("share_id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("first_message", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("share_id"),
    )
    op.create_index("ix_shared_conversations_thread_id", "shared_conversations", ["thread_id"])


def downgrade() -> None:
    op.drop_index("ix_shared_conversations_thread_id", table_name="shared_conversations")
    op.drop_table("shared_conversations")
