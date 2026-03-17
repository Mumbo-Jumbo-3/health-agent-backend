from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from health_agent.config import Settings
from health_agent.models import get_chat_model
from health_agent.state import AgentState
from health_agent.tools import search_health_docs

SYSTEM_PROMPT = """You are a knowledgeable health and wellness assistant. Your role is to answer \
questions about nutrition, exercise, mental health, and preventive care using information from \
the user's private wellness resource library.

Always use the search_health_docs tool to find relevant information before answering. \
Base your answers on the retrieved documents. If the documents don't contain relevant \
information, say so honestly rather than making up answers.

Provide clear, helpful, and accurate health information. Always remind users to consult \
healthcare professionals for personal medical advice."""

TOOLS = [search_health_docs]


def build_graph(settings: Settings):
    llm = get_chat_model(settings)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent(state: AgentState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
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
