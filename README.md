# Lumbr

**Evaluating retrieval strategies and fine-tuning for a RAG assistant over company-internal knowledge.**
CISC 6000 Deep Learning, Group 10: Patricia Angeles, John Chester, Jieun Jeong.

Lumbr answers employee questions over [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
(512K synthetic documents from the fictional company Redwood Inference) with citations.
The study compares six retrievers (BM25, small/large dense, fine-tuned small dense,
hybrid, hybrid + reranker) with the generator held fixed. The full proposal is in
`Group10-Lumbr_RAG_Assistant.docx`.

## Getting started

**Read [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md) first.** It covers the GitHub token,
Colab Secrets, the shared Drive folder, and the bootstrap cell every notebook starts with.

Then open [`notebooks/00_setup_and_download.ipynb`](https://colab.research.google.com/github/JohnAChester/lumbr/blob/main/notebooks/00_setup_and_download.ipynb)
in Colab and run it top to bottom.

## Layout

```
src/lumbr/          shared Python code: import this from notebooks
  paths.py          the one place that defines where data lives (Drive vs. local disk)
  data.py           HF release: questions + cross-check copy of the documents (pinned revision)
  sources.py        structured source JSON from the benchmark's GitHub repo (our source of truth)
  schema.py         the passage schema: ids, header, validation, JSONL io
  chunk.py          documents -> passages (<= 512 tokens for both R2 and R3 tokenizers)
  sample.py         the v0 1K-doc passage sample (#3)
  survey.py         per-source field survey (#2)
notebooks/
  00_template.ipynb             copy this to start a new notebook
  00_setup_and_download.ipynb   one-time data download + sanity checks
  01_john_inspect_sources.ipynb build raw/erb_json.parquet, coverage checks, survey (#2)
  02_john_chunk_sample.ipynb    build processed/sample_passages.jsonl (#3)
docs/
  COLAB_SETUP.md    team setup guide
  SCHEMA.md         passage schema (#1): read this before building on the passages
  SOURCES.md        per-source findings, metadata mapping, data quirks (#2)
tests/              pytest (`pip install -e .[dev]`, then `pytest`)
```

**Models (D3):** R2 = `BAAI/bge-base-en-v1.5`, R3 = `Qwen/Qwen3-Embedding-4B`, R4 = R2 fine-tuned.

Notebooks are named `NN_<track>_<topic>.ipynb` and each has a single owner.

| Track | Owner | Drive folders written |
|---|---|---|
| Preprocessing, chunking, generator, demo | John | `processed/`, `runs/` |
| Retrieval (R1–R3, R5), indexes, evaluation | Patricia | `indexes/`, `runs/` |
| Synthetic Q&A, hard negatives, R4 fine-tuning | Jieun | `synth/`, `models/` |
| Reranker (R6) | everyone | `indexes/r6*`, `runs/` |

## Data rule

Benchmark **questions** must never be used for training. R4 trains only on synthetic
questions generated from corpus passages. The 100-question dev split is used for early
stopping and tuning, and the 400 test questions only for final reporting.
