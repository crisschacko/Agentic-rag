from pathlib import Path
import json
import numpy as np
import faiss
from .schemas import Document, RetrievedDocument
from .embeddings import Embedder

class VectorRetriever:
    def __init__(self, embedder: Embedder, documents: list[Document]):
        self.embedder = embedder
        self.documents = documents
        vectors = embedder.encode([d.text for d in documents]).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 4) -> list[RetrievedDocument]:
        q = self.embedder.encode([query]).astype("float32")
        scores, indices = self.index.search(q, min(k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            d = self.documents[int(idx)]
            results.append(RetrievedDocument(d.text, d.source, d.page, float(score)))
        return results

    def save(self, output_dir: str | Path):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output / "index.faiss"))
        metadata = [
            {"text": d.text, "source": d.source, "page": d.page, "metadata": d.metadata}
            for d in self.documents
        ]
        (output / "documents.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
