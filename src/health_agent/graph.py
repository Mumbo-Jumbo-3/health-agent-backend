import json
import logging
import re
from hashlib import sha256

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from health_agent.config import Settings
from health_agent.models import (
    get_claude_synthesis_model,
    get_trusted_grok_model,
    get_unrestricted_grok_model,
)
from health_agent.rag.retriever import query_keyword_chunks, query_vector_chunks, rerank_documents
from health_agent.state import AgentState


logger = logging.getLogger(__name__)

STATUS_SUCCESS = "success"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"

NO_RAG_RESULTS = "No relevant documents found."


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


class TrustedSearchAnalysis(BaseModel):
    initial_response: str
    refined_queries: list[str]


class UnrestrictedSearchAnalysis(BaseModel):
    initial_response: str


def _strip_grok_render_tags(content: str) -> str:
    return re.sub(r"<grok:render[^>]*>.*?</grok:render>", "", content, flags=re.DOTALL)


def _extract_raw_content(raw: AIMessage) -> str:
    if isinstance(raw.content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw.content
            if not (isinstance(block, dict) and block.get("type") == "tool_use")
        ).strip()
    else:
        content = str(raw.content).strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return _strip_grok_render_tags(content).strip()


def _parse_json_content(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            return json.loads(match.group())
        raise


def _normalize_queries(queries: list[str], original_query: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = query.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)

    if not normalized:
        return [original_query]
    return normalized


def _search_status(content: str) -> str:
    return STATUS_SUCCESS if content.strip() else STATUS_EMPTY


def _run_search_retrieval(
    queries: list[str],
    settings: Settings,
) -> list[Document]:
    if not queries:
        return []

    result_lists: list[list[Document]] = []
    weights: list[float] = []
    query_weight_divisor = max(len(queries), 1)

    for query in queries:
        vector_results = query_vector_chunks(query, settings)
        keyword_results = query_keyword_chunks(query, settings)

        result_lists.extend([vector_results, keyword_results])
        weights.extend([
            settings.vector_weight / query_weight_divisor,
            settings.keyword_weight / query_weight_divisor,
        ])

    fused = reciprocal_rank_fusion(result_lists, weights)
    return fused[: settings.retrieval_fetch_k]


def _docs_status(docs: list[Document]) -> str:
    return STATUS_SUCCESS if docs else STATUS_EMPTY


def _format_rag_context(docs: list[Document]) -> str:
    if not docs:
        return NO_RAG_RESULTS
    return "\n---\n".join(doc.page_content for doc in docs)


def build_graph(settings: Settings):
    trusted_grok = get_trusted_grok_model(settings)
    unrestricted_grok = get_unrestricted_grok_model(settings)
    claude = get_claude_synthesis_model(settings)
    accounts = ", ".join(f"@{a}" for a in settings.trusted_x_accounts)

    trusted_search_system = f"""You are a knowledgeable health and wellness assistant.
Use X Search results to answer the user's question while prioritizing these trusted
accounts: {accounts}.

Return ONLY a JSON object with:
- "initial_response": a practical analysis grounded in relevant posts from those accounts
- "refined_queries": 3-4 keyword-rich semantic search queries for a wellness RAG library

Do not include any text outside the JSON object."""

    unrestricted_search_system = """You are a knowledgeable health and wellness assistant.
Use unrestricted X Search results to answer the user's question.

Return ONLY a JSON object with:
- "initial_response": a practical analysis grounded in relevant posts from X Search

Do not include any text outside the JSON object."""

    synthesis_system = """You are a knowledgeable health and wellness assistant.
You will receive evidence from three channels:
1. A RAG system trained on the entirety of Dr. Ray Peat's work
2. Trusted X accounts
3. Unrestricted X Search

Write a comprehensive, practical response that prioritizes those sources in that order.
If evidence conflicts, prefer the higher-priority source and briefly explain the conflict.
If a channel is empty or failed, briefly note that its evidence was limited or unavailable.
Keep the response integrated rather than source-separated, but include a brief hierarchy note
that the Ray Peat RAG corpus and trusted accounts were weighted above broader X findings.

Disclaimers are provided elsewhere. Do not mention filenames or internal implementation details."""

    def trusted_grok_search(state: AgentState):
        last_message = state["messages"][-1]
        search_llm = trusted_grok.with_config({"tags": ["nostream"]}).bind(
            tools=[
                {
                    "type": "x_search",
                    "allowed_x_handles": settings.trusted_x_accounts,
                }
            ]
        )

        try:
            raw = search_llm.invoke(
                [SystemMessage(content=trusted_search_system), last_message]
            )
            content = _extract_raw_content(raw)
            try:
                parsed = _parse_json_content(content)
            except json.JSONDecodeError:
                logger.warning("Trusted Grok returned malformed JSON; using content fallback")
                parsed = {
                    "initial_response": content,
                    "refined_queries": [str(last_message.content)],
                }

            result = TrustedSearchAnalysis(**parsed)
            cleaned_response = _strip_grok_render_tags(result.initial_response).strip()
            refined_queries = _normalize_queries(result.refined_queries, str(last_message.content))
            return {
                "trusted_search_response": cleaned_response,
                "trusted_refined_queries": refined_queries,
                "trusted_search_status": _search_status(cleaned_response),
            }
        except Exception:
            logger.exception("Trusted Grok search failed")
            return {
                "trusted_search_response": "Trusted-account X search failed.",
                "trusted_refined_queries": [str(last_message.content)],
                "trusted_search_status": STATUS_ERROR,
            }

    def unrestricted_grok_search(state: AgentState):
        last_message = state["messages"][-1]
        search_llm = unrestricted_grok.with_config({"tags": ["nostream"]}).bind(
            tools=[{"type": "x_search"}]
        )

        try:
            raw = search_llm.invoke(
                [SystemMessage(content=unrestricted_search_system), last_message]
            )
            content = _extract_raw_content(raw)
            try:
                parsed = _parse_json_content(content)
            except json.JSONDecodeError:
                logger.warning("Unrestricted Grok returned malformed JSON; using content fallback")
                parsed = {"initial_response": content}

            result = UnrestrictedSearchAnalysis(**parsed)
            cleaned_response = _strip_grok_render_tags(result.initial_response).strip()
            return {
                "unrestricted_search_response": cleaned_response,
                "unrestricted_search_status": _search_status(cleaned_response),
            }
        except Exception:
            logger.exception("Unrestricted Grok search failed")
            return {
                "unrestricted_search_response": "Unrestricted X search failed.",
                "unrestricted_search_status": STATUS_ERROR,
            }

    def rag_retrieve_base(state: AgentState):
        original_query = str(state["messages"][-1].content)
        try:
            docs = _run_search_retrieval([original_query], settings)
            return {
                "base_rag_docs": docs,
                "base_rag_status": _docs_status(docs),
            }
        except Exception:
            logger.exception("Base RAG retrieval failed")
            return {
                "base_rag_docs": [],
                "base_rag_status": STATUS_ERROR,
            }

    def rag_retrieve_enrich(state: AgentState):
        original_query = str(state["messages"][-1].content)
        candidate_queries = [
            query
            for query in state["trusted_refined_queries"]
            if query.strip().lower() != original_query.strip().lower()
        ]

        if not candidate_queries:
            return {
                "enrich_rag_docs": [],
                "enrich_rag_status": STATUS_EMPTY,
            }

        try:
            docs = _run_search_retrieval(candidate_queries, settings)
            return {
                "enrich_rag_docs": docs,
                "enrich_rag_status": _docs_status(docs),
            }
        except Exception:
            logger.exception("Enriched RAG retrieval failed")
            return {
                "enrich_rag_docs": [],
                "enrich_rag_status": STATUS_ERROR,
            }

    def rag_merge(state: AgentState):
        original_query = str(state["messages"][-1].content)
        try:
            merged_docs = reciprocal_rank_fusion(
                [state["base_rag_docs"], state["enrich_rag_docs"]],
                [1.0, 1.0],
            )
            reranked_docs = rerank_documents(original_query, merged_docs, settings)
            rag_status = _docs_status(reranked_docs)
            if rag_status == STATUS_EMPTY and (
                state["base_rag_status"] == STATUS_ERROR
                and state["enrich_rag_status"] == STATUS_ERROR
            ):
                rag_status = STATUS_ERROR

            return {
                "merged_rag_docs": reranked_docs,
                "rag_status": rag_status,
                "rag_context": _format_rag_context(reranked_docs),
            }
        except Exception:
            logger.exception("RAG merge failed")
            return {
                "merged_rag_docs": [],
                "rag_status": STATUS_ERROR,
                "rag_context": NO_RAG_RESULTS,
            }

    def claude_synthesize(state: AgentState):
        original_question = str(state["messages"][-1].content)
        user_content = (
            f"## User Question\n{original_question}\n\n"
            "## Evidence Priority\n"
            "1. RAG system trained on the entirety of Dr. Ray Peat's work\n"
            "2. Trusted X accounts\n"
            "3. Unrestricted X Search\n\n"
            "## Branch Status\n"
            f"- Trusted X Search: {state['trusted_search_status']}\n"
            f"- Unrestricted X Search: {state['unrestricted_search_status']}\n"
            f"- Base RAG: {state['base_rag_status']}\n"
            f"- Enriched RAG: {state['enrich_rag_status']}\n"
            f"- RAG Aggregate: {state['rag_status']}\n\n"
            f"## Trusted X Analysis\n{state['trusted_search_response']}\n\n"
            f"## Unrestricted X Analysis\n{state['unrestricted_search_response']}\n\n"
            f"## Retrieved Documents\n{state['rag_context']}"
        )

        messages = [
            SystemMessage(content=synthesis_system),
            *state["messages"][:-1],
            HumanMessage(content=user_content),
        ]

        response = claude.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("trusted_grok_search", trusted_grok_search)
    graph.add_node("unrestricted_grok_search", unrestricted_grok_search)
    graph.add_node("rag_retrieve_base", rag_retrieve_base)
    graph.add_node("rag_retrieve_enrich", rag_retrieve_enrich)
    graph.add_node("rag_merge", rag_merge)
    graph.add_node("claude_synthesize", claude_synthesize)

    graph.add_edge(START, "trusted_grok_search")
    graph.add_edge(START, "unrestricted_grok_search")
    graph.add_edge(START, "rag_retrieve_base")

    graph.add_edge("trusted_grok_search", "rag_retrieve_enrich")
    graph.add_edge(["rag_retrieve_base", "rag_retrieve_enrich"], "rag_merge")

    graph.add_edge(
        ["trusted_grok_search", "unrestricted_grok_search", "rag_merge"],
        "claude_synthesize",
    )

    graph.add_edge("claude_synthesize", END)

    return graph.compile()
