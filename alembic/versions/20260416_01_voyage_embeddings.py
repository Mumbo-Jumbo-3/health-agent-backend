"""switch embedding column to 1024 dims for voyage-3-large

Revision ID: 20260416_01
Revises: 20260409_01
Create Date: 2026-04-16 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "20260416_01"
down_revision = "20260409_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM agent_resource_chunks")
    op.drop_column("agent_resource_chunks", "embedding")
    op.add_column(
        "agent_resource_chunks",
        sa.Column("embedding", Vector(1024), nullable=False),
    )


def downgrade() -> None:
    op.execute("DELETE FROM agent_resource_chunks")
    op.drop_column("agent_resource_chunks", "embedding")
    op.add_column(
        "agent_resource_chunks",
        sa.Column("embedding", Vector(3072), nullable=False),
    )
