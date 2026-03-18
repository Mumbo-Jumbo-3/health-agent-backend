"""
Module-level compiled graph for use with `langgraph dev`.
Handles RAG indexing on startup and exposes the compiled graph.
"""

from health_agent.config import get_settings
from health_agent.graph import build_graph
from health_agent.rag.ingest import ingest_resources
from health_agent.rag.retriever import mark_indexed, needs_reindex

_settings = get_settings()

if needs_reindex(_settings):
    print("Indexing resources...")
    result = ingest_resources(_settings)
    if result is not None:
        mark_indexed(_settings)

graph = build_graph(_settings)
