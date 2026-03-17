import pytest

from health_agent.tools import search_health_docs


@pytest.mark.integration
def test_search_health_docs_returns_results():
    """Integration test: requires OpenAI API key and ingested resources."""
    result = search_health_docs.invoke({"query": "nutrition"})
    assert isinstance(result, str)
    assert len(result) > 0
