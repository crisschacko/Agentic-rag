import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_PATH = DATA_DIR / "Reference corpus.pdf"

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = int(os.getenv("TOP_K", "4"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "2"))

client = None
embedding_model = None
documents = []
index = None


def get_openai_client():
    global client

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        client = OpenAI(api_key=api_key)

    return client


def load_corpus():
    global documents

    if documents:
        return documents

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Research corpus not found: {PDF_PATH}"
        )

    reader = PdfReader(str(PDF_PATH))

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = (page.extract_text() or "").strip()

        if text:
            documents.append(
                {
                    "text": text,
                    "source": PDF_PATH.name,
                    "page": page_number
                }
            )

    if not documents:
        raise RuntimeError(
            "No readable text was extracted from the research corpus."
        )

    return documents


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return embedding_model


def build_index():
    global index

    load_corpus()

    if index is not None:
        return index

    model = get_embedding_model()

    texts = [
        document["text"]
        for document in documents
    ]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(vectors)

    return index


def retrieve(query, k=TOP_K):
    load_corpus()
    vector_index = build_index()
    model = get_embedding_model()

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = vector_index.search(
        query_vector,
        min(k, len(documents))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        if idx < 0:
            continue

        document = documents[int(idx)]

        results.append(
            {
                "text": document["text"],
                "source": document["source"],
                "page": document["page"],
                "score": float(score)
            }
        )

    return results


def generate(prompt):
    response = get_openai_client().responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text.strip()


def verify_evidence(question, context):
    prompt = f"""
You are an evidence verification agent.

Question:
{question}

Retrieved research evidence:
{context}

Determine whether the retrieved research contains enough
information to answer the question accurately.

Reply with exactly one word:

SUFFICIENT

or

INSUFFICIENT
"""

    result = generate(prompt).upper()

    if (
        "SUFFICIENT" in result
        and "INSUFFICIENT" not in result
    ):
        return "SUFFICIENT"

    return "INSUFFICIENT"


def rewrite_query(question, previous_query, context):
    prompt = f"""
Create a better research retrieval query.

Original question:
{question}

Previous query:
{previous_query}

Retrieved evidence:
{context}

Return only the improved retrieval query.
"""

    return generate(prompt).strip()


def run_agentic_rag(question):
    query = question
    query_history = []

    last_sources = []
    decision = "INSUFFICIENT"

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        query_history.append(query)

        sources = retrieve(
            query,
            TOP_K
        )

        last_sources = sources

        context = "\n\n".join(
            f"[Source {i + 1}]\n{source['text']}"
            for i, source in enumerate(sources)
        )

        decision = verify_evidence(
            question,
            context
        )

        if decision == "SUFFICIENT":

            answer_prompt = f"""
Answer the question using ONLY the
retrieved research evidence below.

Question:
{question}

Research evidence:
{context}

Give a concise and factual answer.

Do not invent information that is
not supported by the evidence.
"""

            answer = generate(
                answer_prompt
            )

            return {
                "question": question,
                "answer": answer,
                "decision": decision,
                "attempts": attempt,
                "query_history": query_history,
                "sources": last_sources
            }

        if attempt < MAX_ATTEMPTS:

            query = rewrite_query(
                question,
                query,
                context
            )

    return {
        "question": question,
        "answer": (
            "The available research evidence "
            "is insufficient to answer reliably."
        ),
        "decision": decision,
        "attempts": MAX_ATTEMPTS,
        "query_history": query_history,
        "sources": last_sources
    }


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "agentic-rag"
        }
    )


@app.route(
    "/api/ask",
    methods=["POST"]
)
def ask():

    data = request.get_json(
        silent=True
    ) or {}

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:
        return jsonify(
            {
                "error": "Question is required."
            }
        ), 400

    try:

        result = run_agentic_rag(
            question
        )

        return jsonify(result)

    except Exception as error:

        return jsonify(
            {
                "error": "Agentic RAG execution failed.",
                "details": str(error)
            }
        ), 500


if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
