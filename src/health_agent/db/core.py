from functools import lru_cache

from pgvector.psycopg import register_vector
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import sessionmaker

from health_agent.config import Settings


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgres://")
    if database_url.startswith("postgresql://") and "+psycopg" not in database_url:
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def _require_database_url(settings: Settings) -> str:
    if not settings.database_url.strip():
        raise RuntimeError("DATABASE_URL must be set for Postgres-backed resource storage.")
    return normalize_database_url(settings.database_url)


@lru_cache(maxsize=8)
def _create_cached_engine(database_url: str) -> Engine:
    engine = create_engine(database_url, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _register_vector(dbapi_connection, _connection_record):
        register_vector(dbapi_connection)

    return engine


def get_engine(settings: Settings) -> Engine:
    return _create_cached_engine(_require_database_url(settings))


def get_session_factory(settings: Settings) -> sessionmaker:
    return sessionmaker(bind=get_engine(settings), autoflush=False, expire_on_commit=False)
