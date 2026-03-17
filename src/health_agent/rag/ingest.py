from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from health_agent.config import Settings


def ingest_resources(settings: Settings):
    from chromadb import PersistentClient
    from langchain_chroma import Chroma

    resource_path = settings.resources_dir
    files = list(resource_path.glob("**/*.txt")) + list(resource_path.glob("**/*.md"))

    if not files:
        print("No .txt or .md files found in resources directory.")
        return None

    docs: list[Document] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": f.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    persist_dir = str(settings.chroma_persist_dir)
    client = PersistentClient(path=persist_dir)

    # Clear existing collection for idempotent re-indexing
    try:
        client.delete_collection("health_docs")
    except ValueError:
        pass

    vectorstore = Chroma(
        collection_name="health_docs",
        embedding_function=embeddings,
        client=client,
    )
    vectorstore.add_documents(chunks)

    print(f"Ingested {len(files)} file(s) into {len(chunks)} chunks.")
    return vectorstore
