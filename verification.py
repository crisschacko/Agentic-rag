from .llm import OpenAIBackend

class EvidenceVerifier:
    def __init__(self, llm: OpenAIBackend):
        self.llm = llm

    def verify(self, question: str, context: str) -> str:
        prompt = f"""You are an evidence verification agent.

Question:
{question}

Retrieved research:
{context}

Determine whether the retrieved research contains enough information to answer the question accurately.
Reply with exactly one word: SUFFICIENT or INSUFFICIENT.
"""
        result = self.llm.generate(prompt).upper()
        return "SUFFICIENT" if "SUFFICIENT" in result and "INSUFFICIENT" not in result else "INSUFFICIENT"
