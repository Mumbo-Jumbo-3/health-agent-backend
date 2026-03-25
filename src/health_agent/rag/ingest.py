import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from health_agent.config import Settings

_AUTHOR_PREFIXES = {
    "peat_": "Dr. Ray Peat",
    "grimhood_": "Grimhood",
    "ferman_": "George Ferman",
}


def _extract_title(text: str) -> str:
    """Extract the first H1 heading, or fall back to the first non-empty line."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped.removeprefix("# ").strip()
        if stripped:
            return stripped[:120]
    return "Untitled"


def _extract_author(text: str, filename: str) -> str:
    """Extract author from content patterns or filename prefix."""
    # Check for explicit author lines
    for line in text.splitlines()[:20]:
        stripped = line.strip().lower()
        if "author:" in stripped:
            return line.split(":", 1)[1].strip().strip("*#_ ")
        if re.search(r"written by\s+", stripped):
            match = re.search(r"written by\s+(.+)", line, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip("*#_ ")

    # Fall back to filename prefix
    for prefix, author in _AUTHOR_PREFIXES.items():
        if filename.startswith(prefix):
            return author

    return ""


def chunk_document(file_path: Path, settings: Settings) -> list[Document]:
    """Chunk a single file using markdown-aware splitting with contextual headers."""
    text = file_path.read_text(encoding="utf-8")
    filename = file_path.name

    title = _extract_title(text)
    author = _extract_author(text, filename)
    base_metadata = {"source": filename, "title": title, "author": author}

    # Stage A: markdown header splitting for .md files
    if file_path.suffix == ".md":
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True,
        )
        md_chunks = md_splitter.split_text(text)
    else:
        md_chunks = [Document(page_content=text, metadata={})]

    # Stage B: sub-split large sections, respecting header prefix budget
    final_chunks: list[Document] = []
    for chunk in md_chunks:
        meta = {**base_metadata, **chunk.metadata}

        # Build contextual header path
        header_parts = [meta[h] for h in ("h1", "h2", "h3") if meta.get(h)]
        header_path = " > ".join(header_parts)
        prefix = f"{header_path}\n\n" if header_path else ""

        body = chunk.page_content
        target_size = settings.chunk_size - len(prefix)

        if len(prefix) + len(body) > settings.chunk_size:
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max(target_size, 200),
                chunk_overlap=settings.chunk_overlap,
            )
            for sub_text in sub_splitter.split_text(body):
                final_chunks.append(Document(page_content=prefix + sub_text, metadata=meta))
        else:
            final_chunks.append(Document(page_content=prefix + body, metadata=meta))

    return final_chunks


def ingest_resources(settings: Settings):
    from chromadb import PersistentClient
    from langchain_chroma import Chroma

    from health_agent.rag.retriever import clear_bm25_cache

    resource_path = settings.resources_dir
    files = list(resource_path.glob("**/*.txt")) + list(resource_path.glob("**/*.md"))

    if not files:
        print("No .txt or .md files found in resources directory.")
        return None

    chunks: list[Document] = []
    for f in files:
        chunks.extend(chunk_document(f, settings))

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    persist_dir = settings.chroma_persist_dir
    staging_dir = persist_dir.with_name(persist_dir.name + "_staging")

    # Clean up any leftover staging dir from a prior failed attempt
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # Build the new index in a staging directory
    client = PersistentClient(path=str(staging_dir))
    vectorstore = Chroma(
        collection_name="health_docs",
        embedding_function=embeddings,
        client=client,
    )
    # ChromaDB has a max batch size of 5461; add in batches
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])
    del vectorstore  # drop reference so client refcount can reach zero
    client.close()  # release SQLite handles before directory swap

    # Atomic swap: live index is untouched until this point
    backup_dir = persist_dir.with_name(persist_dir.name + "_old")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if persist_dir.exists():
        persist_dir.rename(backup_dir)
    staging_dir.rename(persist_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    clear_bm25_cache()

    # Re-open from the final location so the caller gets a usable store
    live_client = PersistentClient(path=str(persist_dir))
    vectorstore = Chroma(
        collection_name="health_docs",
        embedding_function=embeddings,
        client=live_client,
    )

    print(f"Ingested {len(files)} file(s) into {len(chunks)} chunks.")
    return vectorstore
