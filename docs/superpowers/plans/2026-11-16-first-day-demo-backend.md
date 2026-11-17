# First Day at Redwood — Backend & Content (lumbr repo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small Python retrieval service on Vercel (dense retriever, int8 index, similarity filter) plus the offline scripts that produce the demo's `questions.json` and its tuned threshold.

**Architecture:** A FastAPI app in `demo/backend/` deployed as its own Vercel project (root dir `demo/backend`). It loads an ONNX query encoder, an int8-quantized passage matrix and a SQLite passage store from `artifacts/`, fetched at build time from a private Hugging Face Hub repo (never committed). Offline scripts in `demo/build/` produce those artifacts and the content bundle. Develop on the v0 sample; swap to the study's chosen retriever and subsample later.

**Tech Stack:** Python 3.12, FastAPI, onnxruntime, tokenizers, numpy, sqlite3; offline: sentence-transformers, optimum (ONNX export), huggingface_hub, pandas.

**Spec:** `docs/superpowers/specs/2026-11-16-first-day-demo-design.md`. **Companion plan:** `docs/superpowers/plans/2026-11-16-first-day-demo-site.md` (consumes the HTTP contract and `questions.json`).

## Global Constraints

- Vercel Hobby functions: **2 GB RAM / 1 vCPU**; Python bundle 500 MB, up to 5 GB with large functions (`VERCEL_SUPPORT_LARGE_FUNCTIONS=1`). Resident memory after load must stay **≤ 1.6 GB**.
- Index stored **int8** (≈ 500 MB for 640K × 768). Model and index files are **never committed**; they live in a private HF Hub repo and are fetched at build.
- The similarity filter compares the **dense cosine score** only (never BM25/RRF). Threshold tuned on **dev questions only** + an off-topic set. **Test questions are never read.**
- Passages follow `docs/SCHEMA.md` v1 (`passage_id = {doc_id}#{chunk_idx:03d}`).
- Query encoding must match the study's R2/R4 encoding exactly (same model, same query prefix, normalization).
- Every endpoint except `/health` requires `Authorization: Bearer $LUMBR_BACKEND_KEY`.

## HTTP contract (authoritative; the site plan consumes it)

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | — | `{"ok": true, "retriever": "r2", "passages": 4612, "threshold": 0.62}` |
| `POST /retrieve` | `{"query": str (1–300 chars after strip), "k": int = 5 (1–20)}` | `{"retriever": str, "top_score": float, "passed": bool, "passages": [Passage]}` |
| `POST /passages` | `{"ids": [str] (≤ 16)}` | `{"passages": [Passage]}` — unknown ids omitted, `score: null` |

`Passage = {passage_id, doc_id, source, title, channel, author, date, header, body, score}`. Validation errors → 422; bad/missing key → 401.

## Review Focus

1. **Missing dev split** — `build_questions` must fail loudly, never fall back to all 500 questions (test in Task B4).
2. **Malformed or unknown passage ids** in `/passages` — omitted, never a 500 (test in Task B2).
3. **Whitespace-only or 301-char queries** — 422, not an empty search (test in Task B2).
4. **Memory on cold start** with the full index — measured; > 1.6 GB RSS blocks the swap and triggers the HF Space fallback (verification in Task B5).
5. **Query encoding drift** from the study (missing prefix, unnormalized) — demo scores would silently disagree with the study; pinned by a parity test (Task B3).

---

## File Structure

```
demo/backend/
  app/main.py              FastAPI app (Vercel entrypoint)
  lumbr_demo/encoder.py    Encoder protocol, OnnxEncoder, HashEncoder (tests)
  lumbr_demo/index.py      Int8Index
  lumbr_demo/store.py      PassageStore (SQLite)
  lumbr_demo/service.py    Retriever (contract logic)
  lumbr_demo/config.py     env + constants (RETRIEVER, QUERY_PREFIX, THRESHOLD, paths)
  fetch_artifacts.py       build step: download artifacts/ from HF Hub
  requirements.txt, requirements-dev.txt, vercel.json
  tests/                   pytest
demo/build/
  export_onnx.py  build_index.py  upload_artifacts.py  tune_threshold.py
  draft_dms.py    build_questions.py
demo/content/
  colleague_dms.csv  offtopic.txt
```

---

### Task B1: Index and passage store

**Files:** Create `demo/backend/lumbr_demo/{index,store,encoder,config}.py`, `demo/backend/requirements*.txt`; Test `demo/backend/tests/test_index.py`, `test_store.py`

**Interfaces — Produces:**
```python
class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...   # (n, d) float32, L2-normalized
class HashEncoder:  # deterministic bag-of-words hashing into d dims, normalized; tests only
    def __init__(self, d: int = 64): ...
class Int8Index:
    @classmethod
    def build(cls, vecs: np.ndarray) -> "Int8Index": ...      # vecs normalized float32
    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...  # (rows, scores) desc
    def save(self, d: Path) -> None: ...; @classmethod
    def load(cls, d: Path, mmap: bool = True) -> "Int8Index": ...
class PassageStore:
    @classmethod
    def build(cls, jsonl: Path, out: Path) -> list[str]: ...  # returns passage_ids in row order
    def __init__(self, path: Path): ...
    def by_rows(self, rows: list[int]) -> list[dict]: ...
    def by_ids(self, ids: list[str]) -> list[dict]: ...       # unknown ids omitted, input order kept
```
Quantization: per-row symmetric — `scale = max|v|`, `q = round(v / scale * 127)` as int8, store `scale` float32; score = `(q @ query) * scale / 127`. Files: `vectors.int8.npy`, `scales.f32.npy`, `passages.sqlite` (table `p(row INTEGER PRIMARY KEY, passage_id TEXT UNIQUE, json TEXT)`).

- [ ] **Step 1: Failing tests.** 1,000 random normalized 768-d vectors: `Int8Index` top-10 overlaps exact float32 top-10 by ≥ 9/10 on average over 50 queries, and |score − exact cosine| < 0.01. Save/load round-trip gives identical results. `PassageStore` built from a 5-passage jsonl fixture: `by_rows([2,0])` order kept; `by_ids(["x#000", known])` returns only the known one.
- [ ] **Step 2:** `pytest demo/backend/tests -q` → FAIL.
- [ ] **Step 3:** Implement (numpy only; argpartition for top-k).
- [ ] **Step 4:** Tests PASS.
- [ ] **Step 5:** Commit `feat(demo): int8 index and SQLite passage store`.

---

### Task B2: Retriever service and FastAPI app

**Files:** Create `lumbr_demo/service.py`, `app/main.py`, `vercel.json`; Test `tests/test_api.py`

**Interfaces — Consumes:** B1. **Produces:** `Retriever(encoder, index, store, threshold: float, name: str)` with `retrieve(query: str, k: int) -> dict` and `passages(ids: list[str]) -> dict` matching the contract; `create_app(retriever: Retriever, key: str) -> FastAPI`; module-level `app = create_app(load_from_artifacts(), os.environ["LUMBR_BACKEND_KEY"])` loaded lazily on first request.

- [ ] **Step 1: Failing tests** (FastAPI `TestClient`, `HashEncoder`, 5-passage fixture): no/wrong bearer → 401; `/health` without auth → 200 with `passages: 5`; `/retrieve` with a query copying a fixture body → `passed: true`, that passage first, `len(passages) == k`; nonsense query with threshold 0.9 → `passed: false`; `"   "` and 301 chars → 422 (Review Focus 3); `/passages` with `["bogus", "dsid_zz#999", known]` → only known, `score is None` (Review Focus 2); 17 ids → 422.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement; `vercel.json` keys `app/main.py` (read the Vercel Python/FastAPI docs for the current entrypoint config first).
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(demo): retrieval API per contract`.

---

### Task B3: ONNX encoder, v0 artifacts, first deploy

**Files:** Create `lumbr_demo/encoder.py::OnnxEncoder`, `demo/build/{export_onnx,build_index,upload_artifacts}.py`, `demo/backend/fetch_artifacts.py`; Test `tests/test_encoder_parity.py` (marked `slow`, needs the model)

**Interfaces — Produces:** `OnnxEncoder(model_dir: Path, query_prefix: str)`; `build_index.py --passages <jsonl> --model <name|path> --out <dir>` (embeds with sentence-transformers on GPU/CPU, or `--from-faiss <path>` to reconstruct the study's vectors so they match exactly); `upload_artifacts.py --dir <dir> --repo <hf-repo>`; `fetch_artifacts.py` downloads `artifacts/` using `HF_TOKEN`, `LUMBR_ARTIFACT_REPO`, `LUMBR_ARTIFACT_REV`.

- [ ] **Step 1:** Confirm the study's R2 query encoding (prefix, pooling, normalization) from Patricia's retrieval code/notebook and set `QUERY_PREFIX` in `config.py`; note the source in a comment.
- [ ] **Step 2: Failing parity test:** for 20 queries, `OnnxEncoder` vs `SentenceTransformer("BAAI/bge-base-en-v1.5")` with the same prefix: cosine ≥ 0.999 each (Review Focus 5).
- [ ] **Step 3:** Export fp32 ONNX (`export_onnx.py`), implement `OnnxEncoder` (tokenizers + onnxruntime, CLS pooling per bge, normalize). Parity test PASS.
- [ ] **Step 4:** Build v0 artifacts from `lumbr-data/processed/sample_passages.jsonl`, upload to a private HF repo, create the Vercel project (root `demo/backend`, build runs `fetch_artifacts.py`, env `LUMBR_BACKEND_KEY`, `HF_TOKEN`, `LUMBR_ARTIFACT_REPO`, `LUMBR_ARTIFACT_REV`, `LUMBR_THRESHOLD=0.5` provisional), deploy.
- [ ] **Step 5:** Verify: `curl $URL/health`; `/retrieve` for a v0 gold question returns a gold doc in the top 5; record cold-start time, warm p50 latency over 20 calls, and peak RSS (log `resource.getrusage` at load) in `demo/backend/README.md`.
- [ ] **Step 6:** Commit `feat(demo): ONNX encoder, artifact pipeline, v0 deploy`. Hand the URL + key to the site plan (Task 10 env).

---

### Task B4: Content bundle (`questions.json`)

**Blocked on:** the study's dev split file `lumbr-data/processed/dev_question_ids.txt` (Patricia).

**Files:** Create `demo/build/{draft_dms,build_questions}.py`, `demo/content/colleague_dms.csv`; Test `tests/demo/test_build_questions.py` (repo `tests/`, small fixture parquet)

**Interfaces — Produces:** `questions.json` exactly matching the site's `DemoQuestion` type (site plan Task 1), `cached: null`.
- `draft_dms.py`: for each dev question, picks `colleague` (author of the first gold doc; INF / no-author → a frequent corpus author, seeded) and drafts a casual DM rewrite via the LLM API; writes `colleague_dms.csv` (`question_id,colleague,dm,reviewed`) for **John to review by hand** (`reviewed` column set to `y`).
- `build_questions.py --dev-ids <txt> --questions <parquet> --passages <jsonl> --dms <csv> --out <json>`: `infoNotFound` from the benchmark category; `goldSources` = gold doc ids + header of each doc's first passage.

- [ ] **Step 1:** Confirm the column names for question text, gold answer, answer facts and category in `questions.parquet` (only `question_id`, `question_type`, `expected_doc_ids` are confirmed so far); pin them in a `COLUMNS` dict.
- [ ] **Step 2: Failing tests:** missing `--dev-ids` file → exits non-zero with a message naming the file (Review Focus 1); output contains exactly the dev ids and no others; rows with `reviewed != "y"` → exits non-zero listing them; INF questions have `goldSources == []` and `infoNotFound: true`.
- [ ] **Step 3:** Implement both scripts.
- [ ] **Step 4:** Tests PASS; run on real data → 100 entries.
- [ ] **Step 5:** Commit `feat(demo): content bundle builder`; copy `questions.json` to `personal_site/src/data/lumbr-demo/`.

---

### Task B5: Threshold tuning and the real retriever swap

**Blocked on:** study's retriever choice and subsample index (~Nov 23).

**Files:** Create `demo/build/tune_threshold.py`, `demo/content/offtopic.txt` (~40 hand-written off-topic/abusive prompts, no real people); Test `tests/demo/test_tune_threshold.py`

**Interfaces — Produces:** `choose_threshold(answerable: list[float], offtopic: list[float], min_block: float = 0.9) -> float` — the lowest score threshold that blocks ≥ `min_block` of off-topic prompts; `tune_threshold.py` writes `runs/<YYYY-MM-DD>-demo-threshold/{report.md,threshold.json}` with the chosen value, off-topic block rate, answerable pass rate, and INF pass rate (descriptive).

- [ ] **Step 1: Failing test:** a prompt is blocked iff `score < threshold` (same rule as `passed = top_score >= threshold`). With `m = ceil(min_block * len(offtopic))`, the threshold is the smallest float above the m-th lowest off-topic score (`np.nextafter`). Assert `choose_threshold([0.8, 0.7, 0.75], [0.3, 0.4, 0.35, 0.6], 0.75)` is `> 0.4` and `== pytest.approx(0.4, abs=1e-9)`; with `min_block=1.0` it is `> 0.6`; the report flags when the threshold also blocks any answerable score.
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Build artifacts for the chosen CPU-feasible retriever (R2 or R4 dense; for R5 also do Task B6) on the subsample (`processed/passages.jsonl` filtered by `subsample_doc_ids.txt`), preferring `--from-faiss` so vectors equal the study's; measure int8 Recall@10 on dev vs the study's float number and record it.
- [ ] **Step 4:** Run `tune_threshold.py` against the new index; set `LUMBR_THRESHOLD`, bump `LUMBR_ARTIFACT_REV`, enable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` if the bundle exceeds 500 MB, redeploy.
- [ ] **Step 5:** Verify peak RSS ≤ 1.6 GB and warm p50 < 1.5 s (Review Focus 4). If RSS is over, deploy the same app to a free HF Space instead and point the site's `LUMBR_BACKEND_URL` there.
- [ ] **Step 6:** Commit `feat(demo): tuned threshold, production retriever`; send the int8 Recall@10, threshold report and setup name to the site's `about.ts` (site Task 10).

---

### Task B6: Hybrid retrieval + cross-encoder reranker

> Updated 2026-11-16: the team (not this plan) is building the cross-encoder reranker. This task only **loads the team's finished reranker** into the demo service; it never trains or designs one. If it's served, `top_score`/`passed` come from the reranker score and B5's threshold is tuned on that score. Measure memory with it loaded; over 1.6 GB → HF Space fallback.

**Files:** Modify `lumbr_demo/service.py`; Create `lumbr_demo/bm25.py`; Test `tests/test_hybrid.py`

- [ ] **Step 1: Failing test:** with fixture passages, `HybridRetriever.retrieve` fuses BM25 and dense rankings by RRF (k = 60, matching the study's R5) and still reports `top_score` = the **dense** top cosine and `passed` from it.
- [ ] **Step 2:** FAIL → implement with `bm25s` (index saved in `artifacts/`) → PASS.
- [ ] **Step 3:** If R6: rerank the fused top-20 (not 50) with the study's reranker only if p50 stays < 3 s on the deployed function, else document "R5 served; R6 reranker omitted on CPU" in `about.ts`.
- [ ] **Step 4:** Commit `feat(demo): hybrid retrieval`.
