import argparse
from agentic_rag.config import settings
from agentic_rag.ingestion import load_pdfs, chunk_documents
from agentic_rag.embeddings import Embedder
from agentic_rag.retrieval import VectorRetriever

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default='data')
    p.add_argument('--output-dir', default='artifacts/index')
    args = p.parse_args()
    docs = load_pdfs(args.input_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    retriever = VectorRetriever(Embedder(settings.embedding_model), chunks)
    retriever.save(args.output_dir)
    print(f'Indexed {len(chunks)} chunks from {len(docs)} source pages.')

if __name__ == '__main__':
    main()
