from pathlib import Path
from pypdf import PdfReader
from .schemas import Document


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, source=path.name, page=page_number))
    return docs


def load_pdfs(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    docs: list[Document] = []
    for path in sorted(directory.glob("*.pdf")):
        docs.extend(load_pdf(path))
    if not docs:
        raise FileNotFoundError(f"No PDF files found in {directory}")
    return docs


def chunk_documents(documents: list[Document], chunk_size: int = 1000, overlap: int = 200) -> list[Document]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Document] = []
    step = chunk_size - overlap
    for doc in documents:
        for start in range(0, len(doc.text), step):
            text = doc.text[start:start + chunk_size].strip()
            if text:
                chunks.append(Document(text=text, source=doc.source, page=doc.page, metadata=doc.metadata.copy()))
    return chunks
