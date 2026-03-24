"""Module-level compiled graph for use with `langgraph dev`."""

from health_agent.config import get_settings
from health_agent.graph import build_graph
from health_agent.rag.ingest import ingest_resources
from health_agent.rag.retriever import mark_indexed, needs_reindex

settings = get_settings()
if needs_reindex(settings):
    result = ingest_resources(settings)
    if result is not None:
        mark_indexed(settings)

graph = build_graph(settings)
