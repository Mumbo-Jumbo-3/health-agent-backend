from health_agent.config import Settings
from health_agent.rag.ingest import chunk_document


def test_markdown_chunking_preserves_headers(tmp_resources, test_settings: Settings):
    """Markdown header metadata is captured in chunk metadata."""
    chunks = chunk_document(tmp_resources / "nutrition.md", test_settings)

    h1_values = {c.metadata.get("h1") for c in chunks}
    h2_values = {c.metadata.get("h2") for c in chunks}
    assert "Nutrition Guide" in h1_values
    assert "Hydration" in h2_values


def test_contextual_header_prepend(tmp_resources, test_settings: Settings):
    """Chunk page_content starts with the header path for context."""
    chunks = chunk_document(tmp_resources / "nutrition.md", test_settings)

    # Find a chunk from the Electrolytes subsection
    electrolyte_chunks = [c for c in chunks if "electrolyte" in c.page_content.lower()]
    assert electrolyte_chunks
    # Should have the header path prepended
    content = electrolyte_chunks[0].page_content
    assert "Nutrition Guide" in content
    assert "Hydration" in content


def test_title_and_author_extraction(tmp_resources, test_settings: Settings):
    """Title and author are extracted into chunk metadata."""
    chunks = chunk_document(tmp_resources / "peat_thyroid.md", test_settings)

    assert all(c.metadata["title"] == "Thyroid and Metabolism" for c in chunks)
    assert all(c.metadata["author"] == "Dr. Ray Peat" for c in chunks)


def test_txt_files_skip_markdown_splitting(tmp_resources, test_settings: Settings):
    """Plain text files are chunked without markdown header splitting."""
    chunks = chunk_document(tmp_resources / "exercise.txt", test_settings)

    assert len(chunks) >= 1
    assert all(c.metadata["source"] == "exercise.txt" for c in chunks)


def test_chunking_preserves_source_metadata(tmp_resources, test_settings: Settings):
    """Every chunk retains the source filename in metadata."""
    chunks = chunk_document(tmp_resources / "nutrition.md", test_settings)

    assert len(chunks) > 0
    assert all(c.metadata["source"] == "nutrition.md" for c in chunks)


def test_author_from_filename_prefix(tmp_resources, test_settings: Settings):
    """Author is inferred from filename prefix when no explicit author line."""
    # grimhood_ prefix file
    (tmp_resources / "grimhood_test.md").write_text(
        "# Test Article\n\nSome content about health."
    )
    chunks = chunk_document(tmp_resources / "grimhood_test.md", test_settings)
    assert all(c.metadata["author"] == "Grimhood" for c in chunks)
