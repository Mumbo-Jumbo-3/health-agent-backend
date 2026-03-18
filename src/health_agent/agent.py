"""Module-level compiled graph for use with `langgraph dev`."""

from health_agent.config import get_settings
from health_agent.graph import build_graph

graph = build_graph(get_settings())
