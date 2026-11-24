# Lumbr demo search service

Serves the "First Day at Redwood" demo (`personal_site`, `/projects/lumbr-enterprise-rag/demo`): retrieval,
the similarity filter, and the study's own sources + chat messages from `lumbr.generate` (prompt v4,
small-to-big). The site sends those messages to the LLM and parses the answer.
Design: `docs/superpowers/specs/2026-11-23-interface-skeleton-design.md` (#18).

**What it retrieves with depends on the artifacts:**
- **B0** (the demo's setup, `docs/superpowers/specs/2026-11-24-b0-backend-design.md`): R1 BM25 (the study's
  Lucene index via Pyserini, Java 21+) and R4 dense, top 50 each, fused with the study's RRF (k = 60, R1 first).
  `config.json` has `"lexical": "bm25"`. Built from the study's outputs by notebook 19.
- **Dense only** (R2), the v0 sample build below, used for local development. No Java needed.

The similarity filter always reads the dense top cosine.

## Endpoints

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | none | `{ok, retriever, passages, threshold, prompt_sha256}` |
| `POST /retrieve` | `{query (1–300 chars), k=5 (1–20), context_ids=[] (≤ 8), history=[] (≤ 6 {q, a}, ≤ 600 chars each)}` | `{retriever, top_score, passed, prompt_sha256, hit_ids, sources, messages}` |
| `POST /passages` | `{ids (≤ 16)}` | `{passages}` (unknown ids omitted) |

Everything but `/health` needs `Authorization: Bearer $LUMBR_BACKEND_KEY`. Bad input is a 422.

## Build the artifacts (once, offline)

Needs `processed/sample_passages.jsonl` from Drive in `data/drive/processed/`.

```bash
pip install -e ".[dev,demo,demo-build]"
python demo/build/build_artifacts.py --passages data/drive/processed/sample_passages.jsonl --out demo/backend/artifacts
python demo/build/pick_threshold.py --artifacts demo/backend/artifacts
pytest -m artifacts demo/backend/tests     # ONNX encoder matches sentence-transformers
```

`demo/backend/artifacts/` is git-ignored (index, SQLite store, ONNX model, `config.json`).

## Run locally

```bash
LUMBR_BACKEND_KEY=dev LUMBR_ARTIFACTS=demo/backend/artifacts \
  uvicorn app.main:app --app-dir demo/backend --port 8787
```

Use `127.0.0.1`, not `localhost`: on Windows, `localhost` tries IPv6 first and adds about 2 s per call.
Then in `personal_site/.env.local`: `LUMBR_BACKEND_URL=http://127.0.0.1:8787` and `LUMBR_BACKEND_KEY=dev`,
plus the LLM variables, and `npm run dev`.

## Content for the site

```bash
python demo/build/build_questions.py --artifacts demo/backend/artifacts --out demo/content/questions.json
python demo/build/export_parse_cases.py   # demo/content/parse_cases.json, copied to the site's tests
```

## Full B0 artifacts (Colab, notebook 19)

`notebooks/19_john_b0_artifacts.ipynb` runs `demo/build/build_b0_artifacts.py` on the study's R4 shards, R4
model, R1 Lucene index and passages; `demo/build/check_parity.py` (every dev question through this service's
`Retriever`, compared with the study's `r1_r4_rrf` dev run: recall@10 must match 0.766); `pick_threshold.py`;
then uploads to a private Hugging Face dataset repo. Serving it needs `pip install -e ".[demo,demo-b0]"` and
Java 21+.
