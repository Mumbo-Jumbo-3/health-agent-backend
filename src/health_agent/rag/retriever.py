import math

from langchain_core.documents import Document
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from health_agent.config import Settings
from health_agent.db import AgentResource, AgentResourceChunk, get_session_factory
from health_agent.db.models import EMBEDDING_DIMENSIONS
from health_agent.models import get_embeddings_model
from health_agent.rag.resources import filesystem_resource_manifest

_reranker_cache: object | None = None


def _database_resource_manifest(settings: Settings) -> dict[str, str]:
    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(select(AgentResource.source_path, AgentResource.content_hash)).all()
    return {source_path: content_hash for source_path, content_hash in rows}


def _chunk_to_document(chunk: AgentResourceChunk) -> Document:
    metadata = {
        "source": chunk.source,
        "source_path": chunk.source_path,
        "title": chunk.title,
        "author": chunk.author,
        "header_path": chunk.header_path,
    }
    for key in ("h1", "h2", "h3"):
        value = getattr(chunk, key)
        if value:
            metadata[key] = value
    return Document(page_content=chunk.content, metadata=metadata)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _maximal_marginal_relevance(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    lambda_mult: float,
    k: int,
) -> list[int]:
    if not candidate_embeddings or k <= 0:
        return []

    query_similarities = [
        _cosine_similarity(query_embedding, candidate_embedding)
        for candidate_embedding in candidate_embeddings
    ]
    selected = [max(range(len(candidate_embeddings)), key=query_similarities.__getitem__)]
    remaining = set(range(len(candidate_embeddings))) - set(selected)

    while remaining and len(selected) < min(k, len(candidate_embeddings)):
        best_index = None
        best_score = float("-inf")
        for candidate_index in remaining:
            diversity_penalty = max(
                _cosine_similarity(
                    candidate_embeddings[candidate_index],
                    candidate_embeddings[selected_index],
                )
                for selected_index in selected
            )
            mmr_score = (
                lambda_mult * query_similarities[candidate_index]
                - (1 - lambda_mult) * diversity_penalty
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = candidate_index

        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)

    return selected


def query_vector_chunks(query: str, settings: Settings) -> list[Document]:
    if not settings.database_url.strip():
        raise RuntimeError("DATABASE_URL must be set for vector retrieval.")
    if settings.embedding_dimensions != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "EMBEDDING_DIMENSIONS does not match the current database schema. "
            "Expected 3072 for text-embedding-3-large."
        )

    query_embedding = get_embeddings_model(settings).embed_query(query)
    distance = AgentResourceChunk.embedding.cosine_distance(query_embedding).label("distance")
    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(
            select(AgentResourceChunk, distance)
            .order_by(distance.asc(), AgentResourceChunk.chunk_index.asc())
            .limit(settings.retrieval_fetch_k)
        ).all()

    if not rows:
        return []

    candidate_chunks = [row[0] for row in rows]
    candidate_embeddings = [list(chunk.embedding) for chunk in candidate_chunks]
    selected_indices = _maximal_marginal_relevance(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        lambda_mult=0.7,
        k=settings.retrieval_k,
    )
    return [_chunk_to_document(candidate_chunks[index]) for index in selected_indices]


def _weighted_tsvector():
    title_vector = func.setweight(func.to_tsvector("english", func.coalesce(AgentResourceChunk.title, "")), "A")
    header_vector = func.setweight(
        func.to_tsvector("english", func.coalesce(AgentResourceChunk.header_path, "")),
        "B",
    )
    content_vector = func.setweight(
        func.to_tsvector("english", func.coalesce(AgentResourceChunk.content, "")),
        "C",
    )
    return title_vector.op("||")(header_vector).op("||")(content_vector)


def query_keyword_chunks(query: str, settings: Settings) -> list[Document]:
    if not settings.database_url.strip():
        raise RuntimeError("DATABASE_URL must be set for keyword retrieval.")

    weighted_tsvector = _weighted_tsvector()
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(weighted_tsvector, tsquery).label("rank")

    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(
            select(AgentResourceChunk, rank)
            .where(weighted_tsvector.op("@@")(tsquery))
            .order_by(rank.desc(), AgentResourceChunk.chunk_index.asc())
            .limit(settings.keyword_k)
        ).all()

    return [_chunk_to_document(row[0]) for row in rows]


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


def needs_reindex(settings: Settings) -> bool:
    if not settings.database_url.strip():
        return True

    current_manifest = filesystem_resource_manifest(settings.resources_dir)
    try:
        stored_manifest = _database_resource_manifest(settings)
    except SQLAlchemyError:
        return True

    return current_manifest != stored_manifest
