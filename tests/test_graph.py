from health_agent.config import LLMProvider, Settings
from health_agent.graph import build_graph


def test_graph_compiles():
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    graph = build_graph(settings)
    assert graph is not None


def test_graph_has_expected_nodes():
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    graph = build_graph(settings)
    node_names = set(graph.get_graph().nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names
