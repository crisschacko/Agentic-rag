import argparse
from pathlib import Path
from .config import settings
from .ingestion import load_pdfs, chunk_documents
from .embeddings import Embedder
from .retrieval import VectorRetriever
from .llm import OpenAIBackend
from .verification import EvidenceVerifier
from .agent import AgenticRAG

def build_agent(input_dir: str):
    docs = load_pdfs(input_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    embedder = Embedder(settings.embedding_model)
    retriever = VectorRetriever(embedder, chunks)
    llm = OpenAIBackend(settings.openai_api_key, settings.openai_model)
    verifier = EvidenceVerifier(llm)
    return AgenticRAG(retriever, verifier, llm, settings.max_attempts, settings.top_k)

def main():
    parser = argparse.ArgumentParser(description="Evidence-grounded Agentic RAG QA")
    parser.add_argument("question")
    parser.add_argument("--input-dir", default="data")
    args = parser.parse_args()
    agent = build_agent(args.input_dir)
    result = agent.run(args.question)
    print("\nANSWER:\n" + result.answer)
    print(f"\nDecision: {result.decision}; attempts: {result.attempts}")
    for i, source in enumerate(result.sources, 1):
        print(f"Source {i}: {source.source}, page {source.page}, score {source.distance:.4f}")

if __name__ == "__main__":
    main()
