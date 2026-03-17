from langchain_core.tools import tool

from health_agent.config import get_settings
from health_agent.rag.retriever import get_retriever


@tool
def search_health_docs(query: str) -> str:
    """Search the health and wellness knowledge base for information about nutrition, exercise, mental health, and preventive care."""
    settings = get_settings()
    retriever = get_retriever(settings)
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n---\n".join(results)
