import typer

from health_agent.config import get_settings

app = typer.Typer(help="Health Agent — RAG-powered wellness assistant")


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-index even if resources are unchanged"),
):
    """Ingest wellness resources from the resources directory."""
    from health_agent.rag.ingest import ingest_resources
    from health_agent.rag.retriever import needs_reindex

    settings = get_settings()
    if not settings.database_url.strip():
        raise typer.BadParameter("DATABASE_URL must be set before running ingest.")

    if not force and not needs_reindex(settings):
        print("Index is up to date — skipping ingestion. Use --force to rebuild.")
        return

    result = ingest_resources(settings)
    print(
        "Ingest complete: "
        f"{result.added_resources} added, "
        f"{result.updated_resources} updated, "
        f"{result.deleted_resources} deleted, "
        f"{result.chunk_rows_written} chunks written."
    )


@app.command()
def chat():
    """Start an interactive chat session with the health agent."""
    from langchain_core.messages import HumanMessage

    from health_agent.graph import build_graph
    from health_agent.rag.retriever import needs_reindex

    settings = get_settings()

    if not settings.database_url.strip():
        print(
            "DATABASE_URL is not configured. RAG retrieval will be unavailable until Postgres "
            "is configured and ingested.\n"
        )
    elif needs_reindex(settings):
        print(
            "RAG index is missing or stale. Run `health-agent ingest` to rebuild it. "
            "Continuing without automatic ingestion.\n"
        )

    graph = build_graph(settings)
    messages = []

    print("Health Agent (Grok search + RAG + Claude synthesis) — type 'quit' to exit\n")

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
