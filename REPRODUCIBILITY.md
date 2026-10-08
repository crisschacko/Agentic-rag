# Reproducibility

## Environment

Recommended: Python 3.10+ with a virtual environment.

## Data

Place research PDFs under `data/`. The supplied 19-page reference corpus is already included as `data/reference_corpus.pdf`.

## Indexing

```bash
python scripts/build_index.py --input-dir data --output-dir artifacts/index
```

The index contains FAISS vectors plus JSON metadata. It can be rebuilt at any time from the PDFs.

## API configuration

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Do not commit `.env`.

## Reproducibility limits

LLM outputs can vary by model version and runtime. Retrieval results can vary if the embedding model, chunking parameters, source corpus or FAISS configuration changes. Record these settings with experiment outputs.
