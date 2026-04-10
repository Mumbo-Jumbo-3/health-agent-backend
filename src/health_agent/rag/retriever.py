import json
import time
from pathlib import Path

from langchain_core.documents import Document

from health_agent.config import Settings
from health_agent.models import get_embeddings_model

_bm25_cache: dict[str, object] = {}
_reranker_cache: object | None = None


def get_vectorstore(settings: Settings):
    from chromadb import PersistentClient
    from langchain_chroma import Chroma

    client = PersistentClient(path=str(settings.chroma_persist_dir))
    return Chroma(
        collection_name="health_docs",
        embedding_function=get_embeddings_model(settings),
        client=client,
    )


def _ingest_timestamp(settings: Settings) -> float:
    """Read the last-ingest timestamp, or 0 if none exists."""
    timestamp_file = settings.chroma_persist_dir / ".last_ingest"
    if timestamp_file.exists():
        return json.loads(timestamp_file.read_text())["timestamp"]
    return 0.0


def get_bm25_retriever(settings: Settings):
    from langchain_community.retrievers import BM25Retriever

    # Key on persist dir + ingest timestamp so the cache self-invalidates
    # whenever the index is rebuilt — no manual clearing needed.
    cache_key = f"{settings.chroma_persist_dir}@{_ingest_timestamp(settings)}"
    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    # Evict stale entries for this persist dir
    prefix = f"{settings.chroma_persist_dir}@"
    for k in [k for k in _bm25_cache if k.startswith(prefix)]:
        del _bm25_cache[k]

    vectorstore = get_vectorstore(settings)
    all_data = vectorstore.get(include=["documents", "metadatas"])

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    retriever = BM25Retriever.from_documents(docs, k=settings.bm25_k)
    _bm25_cache[cache_key] = retriever
    return retriever


def rerank_documents(
    query: str, docs: list[Document], settings: Settings
) -> list[Document]:
    if not docs:
        return docs

    global _reranker_cache
    if _reranker_cache is None:
        from langchain_community.document_compressors import FlashrankRerank

        _reranker_cache = FlashrankRerank(
            model=settings.reranker_model,
            top_n=settings.reranker_top_k,
        )

    reranked = list(_reranker_cache.compress_documents(docs, query))
    return [
        doc
        for doc in reranked
        if doc.metadata.get("relevance_score", 0) >= settings.reranker_score_threshold
    ]


def clear_bm25_cache() -> None:
    _bm25_cache.clear()


def needs_reindex(settings: Settings) -> bool:
    timestamp_file = settings.chroma_persist_dir / ".last_ingest"
    if not timestamp_file.exists():
        return True

    last_ingest = json.loads(timestamp_file.read_text())

    resource_path = settings.resources_dir
    files = list(resource_path.glob("**/*.txt")) + list(resource_path.glob("**/*.md"))

    current_files = sorted(str(f.relative_to(resource_path)) for f in files)
    stored_files = last_ingest.get("files", [])

    # Detect additions, deletions, and renames
    if current_files != stored_files:
        return True

    if not files:
        return False

    latest_mtime = max(f.stat().st_mtime for f in files)
    return latest_mtime > last_ingest["timestamp"]


def mark_indexed(settings: Settings) -> None:
    resource_path = settings.resources_dir
    files = list(resource_path.glob("**/*.txt")) + list(resource_path.glob("**/*.md"))

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    timestamp_file = settings.chroma_persist_dir / ".last_ingest"
    timestamp_file.write_text(json.dumps({
        "timestamp": time.time(),
        "files": sorted(str(f.relative_to(resource_path)) for f in files),
    }))
