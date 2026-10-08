# Architecture

## Core pipeline

### 1. Ingestion
`ingestion.py` reads PDF research material and converts pages into source-aware documents.

### 2. Chunking
Documents are divided into overlapping chunks. The default configuration follows the later notebook workflow: chunk size 1000 characters and overlap 200.

### 3. Embeddings
`sentence-transformers/all-MiniLM-L6-v2` is the default semantic embedding model used by the supplied notebook.

### 4. Retrieval
FAISS performs vector similarity search. Each result retains source metadata and distance.

### 5. Verification
The verifier asks a language model whether retrieved evidence is sufficient for the original question. The expected decision is exactly `SUFFICIENT` or `INSUFFICIENT`.

### 6. Adaptive retrieval
If evidence is insufficient, the agent rewrites the query and performs another retrieval attempt. The clean implementation caps the number of attempts to keep the experiment bounded.

### 7. Generation
The final answer prompt explicitly requires the answer to use only retrieved research evidence and to avoid unsupported claims.

## Relation to the supplied research corpus

The corpus distinguishes conventional RAG from Agentic RAG by emphasizing adaptive retrieval, document evaluation, corrective actions and self-reflection. CRAG uses a three-way corrective action mechanism, while Self-RAG uses reflection tokens. This project implements the simpler evidence-verification/adaptive-retrieval workflow described in the supplied draft rather than claiming to reproduce CRAG or Self-RAG training.
