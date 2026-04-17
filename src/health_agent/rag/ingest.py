from dataclasses import dataclass
from hashlib import sha256
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sqlalchemy import delete, select

from health_agent.config import Settings
from health_agent.db import AgentResource, AgentResourceChunk, get_session_factory
from health_agent.db.models import EMBEDDING_DIMENSIONS, utc_now
from health_agent.models import get_embeddings_model
from health_agent.rag.resources import filesystem_resource_manifest, resource_files

_AUTHOR_PREFIXES = {
    "peat_": "Dr. Ray Peat",
    "grimhood_": "Grimhood",
    "ferman_": "George Ferman",
}


@dataclass
class IngestStats:
    added_resources: int = 0
    updated_resources: int = 0
    deleted_resources: int = 0
    chunk_rows_written: int = 0


@dataclass
class ResourceFileRecord:
    file_path: Path
    source_path: str
    source_name: str
    raw_content: str
    title: str
    author: str
    content_hash: str


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
        meta["header_path"] = header_path or None
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


def _hash_text(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _resource_record(file_path: Path, resource_path: Path) -> ResourceFileRecord:
    raw_content = file_path.read_text(encoding="utf-8")
    return ResourceFileRecord(
        file_path=file_path,
        source_path=str(file_path.relative_to(resource_path)),
        source_name=file_path.name,
        raw_content=raw_content,
        title=_extract_title(raw_content),
        author=_extract_author(raw_content, file_path.name),
        content_hash=_hash_text(raw_content),
    )


def _embed_texts(texts: list[str], settings: Settings, batch_size: int = 128) -> list[list[float]]:
    if settings.embedding_dimensions != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "EMBEDDING_DIMENSIONS does not match the current database schema. "
            "Expected 1024 for voyage-3-large."
        )

    embeddings_model = get_embeddings_model(settings)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embeddings_model.embed_documents(texts[start : start + batch_size]))
    return vectors


def ingest_resources(settings: Settings) -> IngestStats:
    resource_path = settings.resources_dir
    files = resource_files(resource_path)
    records = [_resource_record(file_path, resource_path) for file_path in files]
    desired_manifest = filesystem_resource_manifest(resource_path)

    session_factory = get_session_factory(settings)
    stats = IngestStats()

    with session_factory() as session, session.begin():
        existing_resources = {
            resource.source_path: resource
            for resource in session.execute(select(AgentResource)).scalars()
        }

        missing_paths = sorted(set(existing_resources) - set(desired_manifest))
        if missing_paths:
            delete_result = session.execute(
                delete(AgentResource).where(AgentResource.source_path.in_(missing_paths))
            )
            stats.deleted_resources = delete_result.rowcount or len(missing_paths)

        for record in records:
            current = existing_resources.get(record.source_path)
            if current is not None and current.content_hash == record.content_hash:
                continue

            now = utc_now()
            if current is None:
                current = AgentResource(
                    source_path=record.source_path,
                    source_name=record.source_name,
                    title=record.title,
                    author=record.author,
                    raw_content=record.raw_content,
                    content_hash=record.content_hash,
                    created_at=now,
                    updated_at=now,
                )
                session.add(current)
                session.flush()
                stats.added_resources += 1
            else:
                current.source_name = record.source_name
                current.title = record.title
                current.author = record.author
                current.raw_content = record.raw_content
                current.content_hash = record.content_hash
                current.updated_at = now
                session.execute(
                    delete(AgentResourceChunk).where(AgentResourceChunk.resource_id == current.id)
                )
                stats.updated_resources += 1

            chunks = chunk_document(record.file_path, settings)
            embeddings = _embed_texts([chunk.page_content for chunk in chunks], settings)
            chunk_rows = []
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_rows.append(
                    AgentResourceChunk(
                        resource_id=current.id,
                        chunk_index=index,
                        source_path=record.source_path,
                        source=record.source_name,
                        title=chunk.metadata["title"],
                        author=chunk.metadata["author"],
                        h1=chunk.metadata.get("h1"),
                        h2=chunk.metadata.get("h2"),
                        h3=chunk.metadata.get("h3"),
                        header_path=chunk.metadata.get("header_path"),
                        content=chunk.page_content,
                        content_hash=_hash_text(chunk.page_content),
                        embedding=embedding,
                        created_at=now,
                        updated_at=now,
                    )
                )

            session.add_all(chunk_rows)
            stats.chunk_rows_written += len(chunk_rows)

    return stats
