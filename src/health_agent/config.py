from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {}

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""

    trusted_xai_model: str = "grok-4-1-fast-reasoning"
    unrestricted_xai_model: str = "grok-4-1-fast-reasoning"
    anthropic_synthesis_model: str = "claude-sonnet-4-20250514"

    embedding_model: str = "text-embedding-3-large"

    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = 8
    retrieval_k: int = 10
    bm25_k: int = 20
    retrieval_fetch_k: int = 40
    bm25_weight: float = 0.4
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
    chroma_persist_dir: Path = Path(".chroma_db")
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_settings() -> Settings:
    return Settings()
