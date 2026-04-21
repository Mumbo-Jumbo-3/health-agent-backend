"""create users table and add ownership to shared_conversations

Revision ID: 20260421_01
Revises: 20260417_01
Create Date: 2026-04-21 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260421_01"
down_revision = "20260417_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("clerk_user_id", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("clerk_user_id"),
    )

    # shared_conversations was pre-launch; drop any existing unowned rows,
    # then add the NOT NULL user_id column and FK.
    op.execute("DELETE FROM shared_conversations")
    op.add_column(
        "shared_conversations",
        sa.Column("user_id", sa.Text(), nullable=False),
    )
    op.create_foreign_key(
        "fk_shared_conversations_user_id",
        "shared_conversations",
        "users",
        ["user_id"],
        ["clerk_user_id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_shared_conversations_user_id",
        "shared_conversations",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_shared_conversations_user_id", table_name="shared_conversations")
    op.drop_constraint(
        "fk_shared_conversations_user_id",
        "shared_conversations",
        type_="foreignkey",
    )
    op.drop_column("shared_conversations", "user_id")
    op.drop_table("users")
