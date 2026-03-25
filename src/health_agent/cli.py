from typing import Optional

import typer

from health_agent.config import LLMProvider, get_settings

app = typer.Typer(help="Health Agent — RAG-powered wellness assistant")


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-index even if resources are unchanged"),
):
    """Ingest wellness resources from the resources directory."""
    from health_agent.rag.ingest import ingest_resources
    from health_agent.rag.retriever import mark_indexed, needs_reindex

    settings = get_settings()

    if not force and not needs_reindex(settings):
        print("Index is up to date — skipping ingestion. Use --force to rebuild.")
        return

    result = ingest_resources(settings)
    if result is not None:
        mark_indexed(settings)


@app.command()
def chat(
    provider: Optional[LLMProvider] = typer.Option(None, help="LLM provider to use"),
):
    """Start an interactive chat session with the health agent."""
    from langchain_core.messages import HumanMessage

    from health_agent.graph import build_graph
    from health_agent.rag.ingest import ingest_resources
    from health_agent.rag.retriever import mark_indexed, needs_reindex

    settings = get_settings()
    if provider:
        settings.llm_provider = provider

    if needs_reindex(settings):
        print("Indexing resources...")
        result = ingest_resources(settings)
        if result is not None:
            mark_indexed(settings)

    graph = build_graph(settings)
    messages = []

    print(f"Health Agent ({settings.llm_provider.value}) — type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": messages})
        messages = result["messages"]

        ai_message = messages[-1]
        print(f"\nAssistant: {ai_message.content}\n")


if __name__ == "__main__":
    app()
