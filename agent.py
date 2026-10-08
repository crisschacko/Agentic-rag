from .schemas import AgentResult
from .retrieval import VectorRetriever
from .verification import EvidenceVerifier
from .llm import OpenAIBackend

class AgenticRAG:
    def __init__(self, retriever: VectorRetriever, verifier: EvidenceVerifier, llm: OpenAIBackend, max_attempts: int = 2, top_k: int = 4):
        self.retriever = retriever
        self.verifier = verifier
        self.llm = llm
        self.max_attempts = max_attempts
        self.top_k = top_k

    def _context(self, docs):
        return "\n\n".join(f"[Source {i+1}] {d.text}" for i, d in enumerate(docs))

    def _rewrite(self, question: str, previous_query: str, context: str) -> str:
        prompt = f"""Create a better research retrieval query.
Original question: {question}
Previous query: {previous_query}
Retrieved evidence: {context}
Return only the improved query."""
        return self.llm.generate(prompt).strip()

    def run(self, question: str) -> AgentResult:
        query = question
        history = []
        last_docs = []
        decision = "INSUFFICIENT"
        for attempt in range(1, self.max_attempts + 1):
            history.append(query)
            docs = self.retriever.search(query, self.top_k)
            last_docs = docs
            context = self._context(docs)
            decision = self.verifier.verify(question, context)
            if decision == "SUFFICIENT":
                prompt = f"""Answer the question using ONLY the retrieved research evidence below.

Question: {question}

Research evidence:
{context}

Give a concise, factual answer. Do not invent information."""
                answer = self.llm.generate(prompt)
                return AgentResult(question, answer, decision, attempt, history, docs)
            if attempt < self.max_attempts:
                query = self._rewrite(question, query, context)
        return AgentResult(question, "The available research evidence is insufficient to answer reliably.", decision, self.max_attempts, history, last_docs)
