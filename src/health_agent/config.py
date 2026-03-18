from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"


class Settings(BaseSettings):
    model_config = {}

    llm_provider: LLMProvider = LLMProvider.XAI

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""

    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-20250514"
    xai_model: str = "grok-4-1-fast-reasoning"

    embedding_model: str = "text-embedding-3-small"

    trusted_x_accounts: list[str] = ["helios_movement", "grimhood", "aestheticprimal"]

    resources_dir: Path = Path("resources")
    chroma_persist_dir: Path = Path(".chroma_db")
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_settings() -> Settings:
    return Settings()
