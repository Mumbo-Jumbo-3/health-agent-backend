import json
import time
from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from health_agent.config import Settings


def get_retriever(settings: Settings, k: int = 4) -> VectorStoreRetriever:
    from chromadb import PersistentClient
    from langchain_chroma import Chroma

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    client = PersistentClient(path=str(settings.chroma_persist_dir))
    vectorstore = Chroma(
        collection_name="health_docs",
        embedding_function=embeddings,
        client=client,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def needs_reindex(settings: Settings) -> bool:
    timestamp_file = settings.chroma_persist_dir / ".last_ingest"
    if not timestamp_file.exists():
        return True

    last_ingest = json.loads(timestamp_file.read_text())["timestamp"]

    resource_path = settings.resources_dir
    files = list(resource_path.glob("**/*.txt")) + list(resource_path.glob("**/*.md"))
    if not files:
        return False

    latest_mtime = max(f.stat().st_mtime for f in files)
    return latest_mtime > last_ingest


def mark_indexed(settings: Settings) -> None:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    timestamp_file = settings.chroma_persist_dir / ".last_ingest"
    timestamp_file.write_text(json.dumps({"timestamp": time.time()}))
