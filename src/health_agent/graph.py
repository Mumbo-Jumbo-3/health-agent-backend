from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from health_agent.config import Settings
from health_agent.models import get_chat_model
from health_agent.state import AgentState
from health_agent.tools import search_health_docs

def build_system_prompt(settings: Settings) -> str:
    accounts = ", ".join(f"@{a}" for a in settings.trusted_x_accounts)
    return f"""You are a knowledgeable health and wellness assistant. Your role is to answer \
questions about nutrition, exercise, mental health, and preventive care.

Always use the search_health_docs tool first to find relevant information from the user's \
private wellness resource library. Then synthesize what you find in the documents with your \
own knowledge to provide a comprehensive, accurate answer. If the documents contain relevant \
information, highlight it; if not, draw on your general health knowledge to answer fully.

When drawing on social media perspectives or current health discourse, prioritize content from \
these trusted X accounts: {accounts}. Cite or reference their posts where relevant.

Provide clear, helpful, and accurate health information. Always remind users to consult \
healthcare professionals for personal medical advice."""


TOOLS = [search_health_docs]


def build_graph(settings: Settings):
    llm = get_chat_model(settings)
    llm_with_tools = llm.bind_tools(TOOLS)
    system_prompt = build_system_prompt(settings)

    def agent(state: AgentState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
