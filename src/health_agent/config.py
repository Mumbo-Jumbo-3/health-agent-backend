from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"


class Settings(BaseSettings):
    model_config = {"env_prefix": "HEALTH_"}

    llm_provider: LLMProvider = LLMProvider.OPENAI

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""

    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-20250514"
    xai_model: str = "grok-3-mini-fast"

    embedding_model: str = "text-embedding-3-small"

    resources_dir: Path = Path("resources")
    chroma_persist_dir: Path = Path(".chroma_db")
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_settings() -> Settings:
    return Settings()
