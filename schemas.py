from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedDocument:
    text: str
    source: str
    page: int | None
    distance: float

@dataclass
class AgentResult:
    question: str
    answer: str
    decision: str
    attempts: int
    query_history: list[str]
    sources: list[RetrievedDocument]
