import logging
from hashlib import sha256

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from health_agent.config import Settings
from health_agent.models import get_chat_model
from health_agent.rag.retriever import get_bm25_retriever, get_vectorstore, rerank_documents
from health_agent.state import AgentState


logger = logging.getLogger(__name__)


def _content_id(doc: Document) -> str:
    """Stable fingerprint including source for correct provenance."""
    source = doc.metadata.get("source", "")
    return sha256(f"{source}:{doc.page_content}".encode()).hexdigest()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    k: int = 60,
) -> list[Document]:
    """Fuse multiple ranked lists using weighted Reciprocal Rank Fusion."""
    doc_scores: dict[str, tuple[float, Document]] = {}
    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results):
            doc_id = _content_id(doc)
            score = weight / (k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + score, doc_scores[doc_id][1])
            else:
                doc_scores[doc_id] = (score, doc)
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs]


class InitialAnalysis(BaseModel):
    initial_response: str
    refined_queries: list[str]


def build_graph(settings: Settings):
    llm = get_chat_model(settings)
    accounts = ", ".join(f"@{a}" for a in settings.trusted_x_accounts)

    initial_system = f"""You are a knowledgeable health and wellness assistant. Given the user's \
health question, do two things:

1. **initial_response**: Answer the query using search tools. \
Prioritize content from these trusted X/Twitter accounts: {accounts}. \
Cite or reference their posts where relevant.

2. **refined_queries**: Generate 3-4 keyword-rich semantic search queries that approach \
the user's question from different angles (e.g., biochemistry, lifestyle factors, \
dietary connections, symptoms). Optimize for retrieving diverse, relevant documents \
from a health and wellness vector store."""

    synthesis_system = f"""You are a knowledgeable health and wellness assistant. You will receive \
two sections: an initial analysis and retrieved documents from a wellness resource library.

Your job is to weave both into a single, cohesive narrative response. Integrate the document \
information naturally — do not list sources separately. Cite sources inline using [Source: filename] \
when referencing document content and [Source: X/Twitter account] when referencing X/Twitter posts.

Disclaimers are provided elsewhere. No need to remind users to consult healthcare professionals for personal medical advice."""

    def grok_initial(state: AgentState):
        import json as _json
        import re as _re

        last_message = state["messages"][-1]

        # Bind xAI's native X/Twitter search tool so Grok can pull live posts.
        # Use bind() instead of bind_tools() because langchain-core doesn't
        # recognize "x_search" as a well-known tool type and would try to
        # convert it to a function schema. bind() passes it through directly
        # and _use_responses_api() will route to the Responses API.
        # Tag with "nostream" so the messages-tuple stream mode skips this
        # intermediate LLM call and only the final synthesize response is
        # streamed to the frontend.
        search_llm = llm.with_config({"tags": ["nostream"]}).bind(
            tools=[
                {
                    "type": "x_search",
                    "allowed_x_handles": settings.trusted_x_accounts,
                }
            ]
        )

        json_prompt = (
            initial_system
            + "\n\nRespond with ONLY a JSON object with keys "
            '"initial_response" and "refined_queries". No other text.'
        )
        raw = search_llm.invoke(
            [SystemMessage(content=json_prompt), last_message]
        )

        # raw.content may be a list of content blocks when tools are used
        if isinstance(raw.content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw.content
                if not (isinstance(block, dict) and block.get("type") == "tool_use")
            ).strip()
        else:
            content = raw.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # Strip <grok:render> citation tags returned by the Responses API
        content = _re.sub(r"<grok:render[^>]*>.*?</grok:render>", "", content, flags=_re.DOTALL)

        try:
            parsed = _json.loads(content)
        except _json.JSONDecodeError:
            # Try to extract a JSON object from mixed content
            match = _re.search(r"\{[\s\S]*\}", content)
            if match:
                parsed = _json.loads(match.group())
            else:
                # Graceful degradation: treat entire response as the answer,
                # fall back to the original query so RAG retrieval still runs
                logger.warning("Failed to parse JSON from initial analysis; using original query for RAG")
                parsed = {"initial_response": content, "refined_queries": [last_message.content]}

        result = InitialAnalysis(**parsed)

        # Also strip citation tags from the initial_response before passing downstream
        clean_response = _re.sub(
            r"<grok:render[^>]*>.*?</grok:render>", "", result.initial_response, flags=_re.DOTALL
        )

        return {
            "initial_response": clean_response,
            "refined_queries": result.refined_queries,
        }

    def rag_retrieve(state: AgentState):
        vectorstore = get_vectorstore(settings)
        bm25_retriever = get_bm25_retriever(settings)
        result_lists = []
        weights = []

        for query in state["refined_queries"]:
            # Vector search (MMR for diversity)
            result_lists.append(vectorstore.max_marginal_relevance_search(
                query,
                k=settings.retrieval_k,
                fetch_k=settings.retrieval_fetch_k,
                lambda_mult=0.7,
            ))
            weights.append(settings.vector_weight)
            # BM25 keyword search
            result_lists.append(bm25_retriever.invoke(query))
            weights.append(settings.bm25_weight)

        # Single RRF pass across all queries — documents relevant to
        # multiple queries accumulate scores instead of being deduped away.
        all_docs = reciprocal_rank_fusion(result_lists, weights)

        if not all_docs:
            return {"rag_context": "No relevant documents found."}

        # Rerank against the original user query
        original_query = state["messages"][-1].content
        reranked = rerank_documents(original_query, all_docs, settings)

        chunks = []
        for doc in reranked:
            source = doc.metadata.get("source", "unknown")
            chunks.append(f"[Source: {source}]\n{doc.page_content}")

        return {"rag_context": "\n---\n".join(chunks)}

    def synthesize(state: AgentState):
        original_question = state["messages"][-1].content
        user_content = (
            f"## User Question\n{original_question}\n\n"
            f"## Initial Analysis\n{state['initial_response']}\n\n"
            f"## Retrieved Documents\n{state['rag_context']}"
        )

        messages = [
            SystemMessage(content=synthesis_system),
            *state["messages"][:-1],
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("grok_initial", grok_initial)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("grok_initial")
    graph.add_edge("grok_initial", "rag_retrieve")
    graph.add_edge("rag_retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
