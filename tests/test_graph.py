import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from health_agent.config import LLMProvider, Settings
from health_agent.graph import InitialAnalysis, build_graph


def _get_node_func(node_name, mock_llm=None):
    """Build graph with a mock LLM and return the named node function."""
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    if mock_llm is None:
        mock_llm = MagicMock()
    with patch("health_agent.graph.get_chat_model") as mock_gcm:
        mock_gcm.return_value = mock_llm
        graph = build_graph(settings)
    return graph.get_graph().nodes[node_name].data.func


def test_graph_compiles():
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    graph = build_graph(settings)
    assert graph is not None


def test_graph_has_expected_nodes():
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    graph = build_graph(settings)
    node_names = set(graph.get_graph().nodes.keys())
    assert "grok_initial" in node_names
    assert "rag_retrieve" in node_names
    assert "synthesize" in node_names


# ---------------------------------------------------------------------------
# Finding 1: synthesize must include the original user question
# ---------------------------------------------------------------------------


class TestSynthesizeIncludesQuestion:
    """Verify that the synthesis LLM receives the user's original question."""

    def _run_synthesize(self, messages, initial_response, rag_context):
        """Call synthesize node and return the messages sent to the LLM."""
        captured = {}

        def capture_invoke(msgs, **kwargs):
            captured["messages"] = msgs
            return AIMessage(content="synthesized answer")

        mock_llm = MagicMock()
        mock_llm.invoke = capture_invoke
        synthesize_fn = _get_node_func("synthesize", mock_llm)

        state = {
            "messages": messages,
            "initial_response": initial_response,
            "rag_context": rag_context,
        }
        synthesize_fn(state)
        return captured["messages"]

    def test_first_turn_includes_question(self):
        """On the first turn, messages has one item; the question must still appear."""
        user_msg = HumanMessage(content="What are the benefits of magnesium?")
        sent = self._run_synthesize(
            messages=[user_msg],
            initial_response="Magnesium helps with sleep...",
            rag_context="Magnesium is essential...",
        )

        human_msgs = [m for m in sent if isinstance(m, HumanMessage)]
        assert len(human_msgs) == 1
        assert "What are the benefits of magnesium?" in human_msgs[0].content
        assert "## Initial Analysis" in human_msgs[0].content
        assert "## Retrieved Documents" in human_msgs[0].content

    def test_multi_turn_includes_question(self):
        """On subsequent turns, conversation history is included AND the current question."""
        history = [
            HumanMessage(content="Tell me about vitamin D"),
            AIMessage(content="Vitamin D is important for..."),
            HumanMessage(content="How does it interact with magnesium?"),
        ]
        sent = self._run_synthesize(
            messages=history,
            initial_response="Vitamin D and magnesium are synergistic...",
            rag_context="Vitamin D absorption...",
        )

        human_msgs = [m for m in sent if isinstance(m, HumanMessage)]
        enriched = human_msgs[-1]
        assert "How does it interact with magnesium?" in enriched.content
        assert "## Initial Analysis" in enriched.content
        assert "## Retrieved Documents" in enriched.content

    def test_question_appears_before_analysis(self):
        """The user question section should come before the analysis."""
        user_msg = HumanMessage(content="magnesium question")
        sent = self._run_synthesize(
            messages=[user_msg],
            initial_response="analysis",
            rag_context="docs",
        )

        human_msgs = [m for m in sent if isinstance(m, HumanMessage)]
        content = human_msgs[0].content
        q_pos = content.index("## User Question")
        a_pos = content.index("## Initial Analysis")
        assert q_pos < a_pos


# ---------------------------------------------------------------------------
# Finding 2: grok_initial JSON parsing resilience
# ---------------------------------------------------------------------------


class TestGrokInitialJsonParsing:
    """Verify that grok_initial handles various JSON response formats."""

    def _make_grok_fn(self, llm_content):
        """Build graph with an LLM that returns the given content."""
        mock_llm = MagicMock()

        def fake_with_config(cfg):
            configured = MagicMock()

            def fake_bind(**kwargs):
                bound = MagicMock()
                bound.invoke = lambda msgs, **kw: AIMessage(content=llm_content)
                return bound

            configured.bind = fake_bind
            return configured

        mock_llm.with_config = fake_with_config
        return _get_node_func("grok_initial", mock_llm)

    def test_clean_json(self):
        """Standard JSON string response."""
        content = json.dumps({
            "initial_response": "Magnesium helps sleep.",
            "refined_queries": ["magnesium sleep", "magnesium benefits"],
        })
        fn = self._make_grok_fn(content)
        result = fn({"messages": [HumanMessage(content="test")]})

        assert result["initial_response"] == "Magnesium helps sleep."
        assert result["refined_queries"] == ["magnesium sleep", "magnesium benefits"]

    def test_markdown_wrapped_json(self):
        """JSON wrapped in markdown code fences."""
        inner = json.dumps({
            "initial_response": "Answer here.",
            "refined_queries": ["q1"],
        })
        content = f"```json\n{inner}\n```"
        fn = self._make_grok_fn(content)
        result = fn({"messages": [HumanMessage(content="test")]})

        assert result["initial_response"] == "Answer here."

    def test_json_with_preamble(self):
        """JSON with extra text before/after — regex extraction fallback."""
        inner = json.dumps({
            "initial_response": "Response text.",
            "refined_queries": ["q1", "q2"],
        })
        content = f"Here is the analysis:\n{inner}\nHope this helps!"
        fn = self._make_grok_fn(content)
        result = fn({"messages": [HumanMessage(content="test")]})

        assert result["initial_response"] == "Response text."
        assert result["refined_queries"] == ["q1", "q2"]

    def test_completely_malformed_degrades_gracefully(self):
        """Total garbage — should degrade gracefully, not crash."""
        content = "I cannot produce valid JSON but here is my answer about magnesium."
        fn = self._make_grok_fn(content)
        result = fn({"messages": [HumanMessage(content="test")]})

        # Falls back to treating content as the response; refined_queries
        # defaults to [original_message] so RAG retrieval still runs.
        assert "magnesium" in result["initial_response"]
        assert result["refined_queries"] == ["test"]

    def test_grok_render_tags_stripped(self):
        """Citation tags in initial_response should be stripped."""
        content = json.dumps({
            "initial_response": (
                'Magnesium helps with sleep <grok:render type="cite" '
                'data-source="x">@hubermanlab</grok:render> and recovery.'
            ),
            "refined_queries": ["q1"],
        })
        fn = self._make_grok_fn(content)
        result = fn({"messages": [HumanMessage(content="test")]})

        assert "<grok:render" not in result["initial_response"]
        assert "Magnesium helps with sleep" in result["initial_response"]
        assert "and recovery." in result["initial_response"]

    def test_list_content_blocks(self):
        """When x_search is used, content is a list of blocks."""
        inner = json.dumps({
            "initial_response": "From X posts: magnesium is great.",
            "refined_queries": ["magnesium twitter"],
        })
        content = [
            {"type": "tool_use", "name": "x_search", "input": {}},
            {"type": "text", "text": inner},
        ]
        fn = self._make_grok_fn(content)
        result = fn({"messages": [HumanMessage(content="test")]})

        assert result["initial_response"] == "From X posts: magnesium is great."
        assert result["refined_queries"] == ["magnesium twitter"]

    def test_grok_render_in_content_then_json(self):
        """Citation tags embedded in the raw content before JSON parsing."""
        tagged = json.dumps({
            "initial_response": (
                'According to <grok:render type="cite" data-source="x">'
                "@hubermanlab</grok:render>, magnesium aids sleep."
            ),
            "refined_queries": ["q1"],
        })
        # Wrap in grok:render at the content level too
        content = (
            '<grok:render type="wrapper">ignore</grok:render>\n' + tagged
        )
        fn = self._make_grok_fn(content)
        result = fn({"messages": [HumanMessage(content="test")]})

        assert "<grok:render" not in result["initial_response"]


# ---------------------------------------------------------------------------
# Option A verification: with_structured_output loses bind() kwargs
# ---------------------------------------------------------------------------


class TestOptionAFailure:
    """Document why with_structured_output(json_mode) can't be used with x_search."""

    def test_with_structured_output_loses_tools_binding(self):
        """Calling .bind(tools=[...]).with_structured_output(json_mode) drops
        the tools kwargs — this is why we use manual parsing (Option B)."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", api_key="fake-key")

        x_search_tool = {"type": "x_search", "allowed_x_handles": ["hubermanlab"]}
        bound = llm.bind(tools=[x_search_tool])

        structured = bound.with_structured_output(InitialAnalysis, method="json_mode")
        bound_llm = structured.first

        # The tools kwarg is NOT preserved — this is the known limitation
        assert "tools" not in bound_llm.kwargs
        assert "response_format" in bound_llm.kwargs


# ---------------------------------------------------------------------------
# Parallel topology verification
# ---------------------------------------------------------------------------


def test_graph_has_parallel_topology():
    """Verify grok_initial and rag_retrieve fan out from START, fan in to synthesize."""
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    graph = build_graph(settings)
    graph_data = graph.get_graph()

    # Both nodes should have an edge from __start__
    start_targets = {e.target for e in graph_data.edges if e.source == "__start__"}
    assert "grok_initial" in start_targets
    assert "rag_retrieve" in start_targets

    # Both nodes should have an edge to synthesize
    synthesize_sources = {e.source for e in graph_data.edges if e.target == "synthesize"}
    assert "grok_initial" in synthesize_sources
    assert "rag_retrieve" in synthesize_sources

    # synthesize should lead to __end__
    end_sources = {e.source for e in graph_data.edges if e.target == "__end__"}
    assert "synthesize" in end_sources


# ---------------------------------------------------------------------------
# rag_retrieve uses original query, not refined_queries
# ---------------------------------------------------------------------------


class TestRagRetrieveUsesOriginalQuery:
    """Verify rag_retrieve searches with the user's original message."""

    def test_calls_vectorstore_and_bm25_with_original_query(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore.max_marginal_relevance_search.return_value = []

        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = []

        with patch("health_agent.graph.get_vectorstore", return_value=mock_vectorstore), \
             patch("health_agent.graph.get_bm25_retriever", return_value=mock_bm25):
            fn = _get_node_func("rag_retrieve")
            state = {
                "messages": [HumanMessage(content="benefits of magnesium")],
                "refined_queries": ["should be ignored"],
                "initial_response": "",
                "rag_context": "",
            }
            result = fn(state)

        # Vector search called with original query
        mock_vectorstore.max_marginal_relevance_search.assert_called_once()
        call_args = mock_vectorstore.max_marginal_relevance_search.call_args
        assert call_args[0][0] == "benefits of magnesium"

        # BM25 called with original query
        mock_bm25.invoke.assert_called_once_with("benefits of magnesium")

        # No documents → fallback context
        assert result["rag_context"] == "No relevant documents found."
