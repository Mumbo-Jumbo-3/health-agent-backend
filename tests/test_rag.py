from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from health_agent.config import Settings


def test_chunking_splits_documents(test_settings: Settings):
    """Chunking produces expected number of splits for sample content."""
    resources = test_settings.resources_dir
    files = list(resources.glob("**/*.txt")) + list(resources.glob("**/*.md"))

    docs = [
        Document(page_content=f.read_text(), metadata={"source": f.name})
        for f in files
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=test_settings.chunk_size,
        chunk_overlap=test_settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    assert len(chunks) > len(docs), "Should produce more chunks than original docs"
    for chunk in chunks:
        assert len(chunk.page_content) <= test_settings.chunk_size + 50  # allow minor overflow
        assert "source" in chunk.metadata


def test_chunking_preserves_metadata(test_settings: Settings):
    """Chunk metadata retains the source filename."""
    doc = Document(page_content="A" * 500, metadata={"source": "test.md"})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=test_settings.chunk_size,
        chunk_overlap=test_settings.chunk_overlap,
    )
    chunks = splitter.split_documents([doc])

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.md"
