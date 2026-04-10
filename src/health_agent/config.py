from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {}

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    database_url: str = ""

    trusted_xai_model: str = "grok-4-1-fast-reasoning"
    unrestricted_xai_model: str = "grok-4-1-fast-reasoning"
    anthropic_synthesis_model: str = "claude-sonnet-4-20250514"

    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = 12
    retrieval_k: int = 10
    keyword_k: int = 30
    retrieval_fetch_k: int = 80
    keyword_weight: float = 0.4
    vector_weight: float = 0.6
    reranker_score_threshold: float = 0.05

    trusted_x_accounts: list[str] = [
        "helios_movement",
        "grimhood",
        "aestheticprimal",
        "hubermanlab",
        "foundmyfitness",
        "outdoctrination",
    ]

    resources_dir: Path = Path("resources")
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_settings() -> Settings:
    return Settings()
