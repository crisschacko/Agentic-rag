from dataclasses import asdict
import pandas as pd
import time
from .agent import AgenticRAG

DEFAULT_QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "Why is RAG useful for reducing hallucinations?",
    "What are the main components of a RAG system?",
    "What is Agentic RAG?",
    "How does Agentic RAG differ from traditional RAG?",
    "What is the role of retrieval in RAG?",
    "What is self-reflection in retrieval-augmented systems?",
    "What is corrective retrieval?",
    "What are some challenges of RAG systems?",
    "What are possible future directions for RAG?",
]

def run_agentic_evaluation(agent: AgenticRAG, questions=None, output: str | None = None):
    rows = []
    for question in questions or DEFAULT_QUESTIONS:
        start = time.perf_counter()
        result = agent.run(question)
        elapsed = time.perf_counter() - start
        rows.append({"question": question, "answer": result.answer, "decision": result.decision, "attempts": result.attempts, "latency_seconds": elapsed, "queries": " || ".join(result.query_history)})
    frame = pd.DataFrame(rows)
    if output:
        frame.to_csv(output, index=False)
    return frame
