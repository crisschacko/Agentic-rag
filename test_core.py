from agentic_rag.ingestion import chunk_documents
from agentic_rag.schemas import Document


def test_chunking_produces_overlap_and_content():
    docs = [Document(text="abcdefghijklmnopqrstuvwxyz", source="test.pdf", page=1)]
    chunks = chunk_documents(docs, chunk_size=10, overlap=2)
    assert chunks
    assert chunks[0].source == "test.pdf"
    assert "abcdefghij" in chunks[0].text


def test_invalid_overlap_is_rejected():
    docs = [Document(text="abc", source="test.pdf", page=1)]
    try:
        chunk_documents(docs, chunk_size=10, overlap=10)
    except ValueError:
        return
    assert False, "Expected ValueError"
