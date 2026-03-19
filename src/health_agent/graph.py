from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from health_agent.config import Settings
from health_agent.models import get_chat_model
from health_agent.rag.retriever import get_retriever
from health_agent.state import AgentState


class InitialAnalysis(BaseModel):
    initial_response: str
    refined_query: str


def build_graph(settings: Settings):
    llm = get_chat_model(settings)
    accounts = ", ".join(f"@{a}" for a in settings.trusted_x_accounts)

    initial_system = f"""You are a knowledgeable health and wellness assistant. Given the user's \
health question, do two things:

1. **initial_response**: Answer the query using search tools. \
Prioritize content from these trusted X/Twitter accounts: {accounts}. \
Cite or reference their posts where relevant.

2. **refined_query**: Rewrite the user's question into a keyword-rich semantic search query \
optimized for retrieving relevant documents from a health and wellness vector store."""

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
        search_llm = llm.bind(
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
            '"initial_response" and "refined_query". No other text.'
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

        parsed = _json.loads(content)
        result = InitialAnalysis(**parsed)

        # Also strip citation tags from the initial_response before passing downstream
        clean_response = _re.sub(
            r"<grok:render[^>]*>.*?</grok:render>", "", result.initial_response, flags=_re.DOTALL
        )

        return {
            "initial_response": clean_response,
            "refined_query": result.refined_query,
        }

    def rag_retrieve(state: AgentState):
        retriever = get_retriever(settings)
        docs = retriever.invoke(state["refined_query"])

        if not docs:
            return {"rag_context": "No relevant documents found."}

        chunks = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            chunks.append(f"[Source: {source}]\n{doc.page_content}")

        return {"rag_context": "\n---\n".join(chunks)}

    def synthesize(state: AgentState):
        user_content = (
            f"## Initial Analysis\n{state['initial_response']}\n\n"
            f"## Retrieved Documents\n{state['rag_context']}"
        )

        messages = [SystemMessage(content=synthesis_system)]
        # Include conversation history (all but the last user message, which we enrich)
        for msg in state["messages"][:-1]:
            messages.append(msg)
        # Add the enriched user message
        messages.append(
            AIMessage(content="Let me analyze this with my knowledge and your documents...")
        )
        from langchain_core.messages import HumanMessage

        messages.append(HumanMessage(content=user_content))

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
