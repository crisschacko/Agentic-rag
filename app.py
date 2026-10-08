import os
import re
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from pypdf import PdfReader


app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_PATH = DATA_DIR / "Reference corpus.pdf"

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna"
)

TOP_K = int(
    os.getenv("TOP_K", "4")
)

MAX_ATTEMPTS = int(
    os.getenv("MAX_ATTEMPTS", "2")
)

# Enable useful error information in Render logs.
logging.basicConfig(
    level=logging.INFO
)


# --------------------------------------------------
# Global resources
# --------------------------------------------------

client = None
documents = []


# --------------------------------------------------
# OpenAI client
# --------------------------------------------------

def get_openai_client():

    global client

    if client is None:

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "OPENAI_API_KEY is not configured in Render."
            )

        client = OpenAI(
            api_key=api_key
        )

    return client


# --------------------------------------------------
# Research corpus loading
# --------------------------------------------------

def load_corpus():

    global documents

    if documents:
        return documents

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"Research corpus not found: {PDF_PATH}"
        )

    app.logger.info(
        "Loading research corpus: %s",
        PDF_PATH
    )

    reader = PdfReader(
        str(PDF_PATH)
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = (
            page.extract_text() or ""
        ).strip()

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

    app.logger.info(
        "Loaded %d research pages.",
        len(documents)
    )

    return documents


# --------------------------------------------------
# Lightweight tokenization
# --------------------------------------------------

def tokenize(text):

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]{2,}\b",
            text.lower()
        )
    )


# --------------------------------------------------
# Lightweight retrieval
# --------------------------------------------------

def retrieve(
    query,
    k=TOP_K
):

    docs = load_corpus()

    query_words = tokenize(
        query
    )

    if not query_words:
        return []

    scored = []

    for document in docs:

        document_words = tokenize(
            document["text"]
        )

        overlap = (
            query_words
            .intersection(
                document_words
            )
        )

        if not overlap:

            score = 0.0

        else:

            score = (
                len(overlap)
                / max(
                    len(query_words),
                    1
                )
            )

        scored.append(
            (
                score,
                document
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    results = []

    for score, document in scored[:k]:

        results.append(
            {
                "text": document["text"],
                "source": document["source"],
                "page": document["page"],
                "score": float(score)
            }
        )

    app.logger.info(
        "Retrieved %d sources for query.",
        len(results)
    )

    return results


# --------------------------------------------------
# LLM generation
# --------------------------------------------------

def generate(prompt):

    app.logger.info(
        "Sending request to OpenAI."
    )

    response = (
        get_openai_client()
        .responses.create(
            model=MODEL_NAME,
            input=prompt
        )
    )

    return (
        response.output_text
        .strip()
    )


# --------------------------------------------------
# Evidence verification
# --------------------------------------------------

def verify_evidence(
    question,
    context
):

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

    result = generate(
        prompt
    ).upper().strip()

    if result == "SUFFICIENT":

        return "SUFFICIENT"

    return "INSUFFICIENT"


# --------------------------------------------------
# Query rewriting
# --------------------------------------------------

def rewrite_query(
    question,
    previous_query,
    context
):

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

    return generate(
        prompt
    ).strip()


# --------------------------------------------------
# Agentic RAG
# --------------------------------------------------

def run_agentic_rag(
    question
):

    query = question

    query_history = []

    last_sources = []

    decision = "INSUFFICIENT"

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        app.logger.info(
            "Agentic attempt %d/%d",
            attempt,
            MAX_ATTEMPTS
        )

        query_history.append(
            query
        )

        sources = retrieve(
            query,
            TOP_K
        )

        last_sources = sources

        if sources:

            context = "\n\n".join(
                f"[Source {i + 1}]\n"
                f"{source['text']}"
                for i, source
                in enumerate(
                    sources
                )
            )

        else:

            context = (
                "No relevant research evidence "
                "was retrieved."
            )

        decision = verify_evidence(
            question,
            context
        )

        app.logger.info(
            "Evidence decision: %s",
            decision
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


# --------------------------------------------------
# Web interface
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify(
        {
            "status": "healthy",
            "service": "agentic-rag"
        }
    )


# --------------------------------------------------
# Ask endpoint
# --------------------------------------------------

@app.route(
    "/api/ask",
    methods=["POST"]
)
def ask():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    question = (
        data.get(
            "question",
            ""
        )
        .strip()
    )

    if not question:

        return jsonify(
            {
                "error":
                    "Question is required."
            }
        ), 400

    try:

        result = run_agentic_rag(
            question
        )

        return jsonify(
            result
        )

    except Exception as error:

        # IMPORTANT:
        # This prints the complete traceback
        # to the Render application logs.
        app.logger.exception(
            "Agentic RAG execution failed"
        )

        return jsonify(
            {
                "error":
                    "Agentic RAG execution failed.",
                "details":
                    str(error)
            }
        ), 500


# --------------------------------------------------
# Local development
# --------------------------------------------------

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
