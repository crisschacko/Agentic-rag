import os

from flask import Flask, jsonify, request

from agentic_rag.cli import build_agent


app = Flask(__name__)

_agent = None


def get_agent():
    global _agent

    if _agent is None:
        _agent = build_agent("data")

    return _agent


@app.route("/")
def home():
    return jsonify({
        "project": "Agentic RAG",
        "status": "online",
        "description": "Evidence-grounded Agentic Retrieval-Augmented Generation system",
        "version": "0.1.0"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "agentic-rag"
    })


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "Question is required."
        }), 400

    try:
        result = get_agent().run(question)

        return jsonify({
            "question": result.question,
            "answer": result.answer,
            "decision": result.decision,
            "attempts": result.attempts,
            "query_history": result.query_history,
            "sources": [
                {
                    "source": source.source,
                    "page": source.page,
                    "score": source.distance
                }
                for source in result.sources
            ]
        })

    except Exception as error:
        return jsonify({
            "error": "Agentic RAG execution failed.",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
