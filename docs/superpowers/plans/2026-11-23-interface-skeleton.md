# Lumbr Interface Skeleton (#18) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dev question typed into the demo's Lumbr chat goes through real R2 retrieval over the v0 sample and the study's generator (prompt v4, small-to-big, `[n]` citations), and comes back with working citation chips.

**Architecture:** A FastAPI search service in `lumbr/demo/backend/` imports `lumbr.generate` and returns the top-5 hits, their small-to-big sources and the finished chat messages. The site (`personal_site`, branch `john/lumbr-demo`) sends those messages to the hosted LLM through its existing spend/deadline code and parses the answer with a TypeScript port of `parse_answer`, pinned by cases exported from Python.

**Tech Stack:** Python 3.12+ (local venv is 3.14), FastAPI, numpy, onnxruntime, tokenizers, SQLite; offline: sentence-transformers, optimum. Site: Next.js 16, TypeScript, vitest.

**Spec:** `docs/superpowers/specs/2026-11-23-interface-skeleton-design.md`

## Global Constraints

- R2 = `BAAI/bge-base-en-v1.5`; queries encoded as `"Represent this sentence for searching relevant passages:" + q` (no space), CLS pooling, L2-normalised, max 512 tokens; passages encode `p["text"]` with no prefix (`lumbr.index_dense.R2`).
- Context and messages come only from `lumbr.generate` (`build_sources(hits, by_id, neighbours=1)`, `build_messages`, `load_prompt`). No copy of the prompt in either repo.
- Hits per turn ≤ 8 (`MAX_CONTEXT_PASSAGES`); k = 5 per query.
- LLM: temperature 0, `max_tokens` 512.
- Only dev questions are read (`splits/dev_question_ids.txt`); test questions never.
- Model, index and store files are never committed: they live in `demo/backend/artifacts/` (git-ignored).
- Importing the service must not import `pandas` or `torch`.
- Commits end with the session's attribution lines.

## Review Focus

1. **Follow-up whose retrieval fails but has earlier context.** Expect it to be answered from the earlier hits only, not the canned line (Task 2 test `test_not_passed_with_context_uses_earlier_hits`).
2. **History answers carrying old `[n]` markers.** Expect them stripped before they reach the model (Task 2 `test_history_citations_stripped`).
3. **Model output `[1, 3]`, `[7]` (out of range), `[2](url)`.** Expect `[1][3]` chips, `[7]` removed, the link left alone (Task 7 `parse_cases` + `displayText` tests).
4. **An abstention with inserted words ("…can't find that specific information…").** Expect `kind: "canned"` on the site, matching Python (Task 7 shared cases).
5. **Malformed `context_ids` / oversized history from a client.** Expect 422 from the service, never a 500 (Task 2 `test_validation_errors`).

---

## File Structure

```
lumbr/
  demo/backend/
    lumbr_demo/__init__.py
    lumbr_demo/encoder.py     Encoder protocol, HashEncoder (tests), OnnxEncoder
    lumbr_demo/index.py       DenseIndex (float32, normalised)
    lumbr_demo/store.py       PassageStore (SQLite) + build_store
    lumbr_demo/service.py     Retriever: hits -> sources -> messages; strip_citations
    lumbr_demo/artifacts.py   load_retriever(artifacts_dir) from config.json
    app/main.py               create_app(retriever, key); module-level lazy app
    requirements.txt          fastapi, numpy, onnxruntime, tokenizers (+ lumbr --no-deps at deploy)
    README.md                 build + run locally
    tests/                    pytest (fixtures, HashEncoder)
  demo/build/
    build_artifacts.py        sample passages -> embeddings.npy, ids.json, passages.sqlite, onnx/, config.json
    pick_threshold.py         provisional threshold from the 11 answerable dev questions
    build_questions.py        questions.json for the 15 demo questions
    export_parse_cases.py     parse_cases.json for the site
  demo/content/colleague_dms.csv
personal_site/ (branch john/lumbr-demo)
  src/lib/lumbr-demo/types.ts                Source type
  src/lib/lumbr-demo/server/retrieval.ts     new /retrieve contract + fake
  src/lib/lumbr-demo/server/llm.ts           chat(messages); study prompt removed
  src/lib/lumbr-demo/server/judge.ts         adapt to chat(messages)
  src/lib/lumbr-demo/server/ask.ts           rewired pipeline
  src/lib/lumbr-demo/citations.ts            parseAnswer + displayText (replaces numberCitations)
  src/data/lumbr-demo/questions.json         the 15 real questions
  tests/lumbr-demo/parse_cases.json          exported from lumbr
```

---

### Task 1: Store, index, encoder protocol (lumbr)

**Files:**
- Create: `demo/backend/lumbr_demo/{__init__,store,index,encoder}.py`, `demo/backend/requirements.txt`, `demo/backend/tests/conftest.py`, `demo/backend/tests/test_store_index.py`
- Modify: `pyproject.toml` (add `demo = ["fastapi>=0.110", "httpx", "numpy", "onnxruntime", "tokenizers"]` extra; `[tool.pytest.ini_options] pythonpath = ["demo/backend"]`), `.github/workflows/*.yml` (install `.[dev,demo]`), `.gitignore` (`demo/backend/artifacts/`)

**Interfaces — Produces:**
- `build_store(passages: Iterable[dict], path: Path) -> int` (rows written; table `passages(passage_id PK, doc_id, chunk_idx, json)`, index on `(doc_id, chunk_idx)`)
- `PassageStore(path: Path)`: `.get(ids: list[str]) -> dict[str, dict]` (unknown ids omitted); `.with_neighbours(hit_ids: list[str], neighbours: int = 1) -> dict[str, dict]` (hits + chunks `chunk_idx ± neighbours` of the same doc); `.count() -> int`
- `DenseIndex(matrix: np.ndarray, ids: list[str])`: `.search(q: np.ndarray, k: int) -> list[tuple[str, float]]` (cosine, descending; ties by id); `DenseIndex.load(dir: Path)` reads `embeddings.npy` + `ids.json`
- `class Encoder(Protocol): def encode(self, text: str) -> np.ndarray` (1-D float32, unit norm); `HashEncoder(dim=64)` deterministic bag-of-words for tests

- [ ] **Step 1: Write failing tests** in `test_store_index.py`, using `tests/fixtures/` passages (`schema` v1 rows) built into a tmp SQLite:
  - `test_store_get_omits_unknown` — `get(["<real id>", "dsid_nope#000", "garbage"])` returns only the real id.
  - `test_with_neighbours_includes_adjacent_chunks` — for a hit at `chunk_idx` 1 of a 3-chunk doc, keys = `#000, #001, #002`; at `chunk_idx` 0, `#000, #001`.
  - `test_index_search_orders_by_cosine` — 3 unit vectors, query equal to row 2 → row 2 first with score ≈ 1.0, `k` respected.
  - `test_hash_encoder_unit_norm_and_deterministic`.
- [ ] **Step 2:** `pip install -e ".[dev,demo]"`, then `pytest demo/backend/tests -q` → FAIL (module not found).
- [ ] **Step 3: Implement** the four modules (stdlib `sqlite3`, numpy only).
- [ ] **Step 4:** `pytest demo/backend/tests -q` → PASS; `pytest -q` → full suite passes.
- [ ] **Step 5: Commit** `Demo backend: passage store, dense index, encoder protocol (#18)`.

### Task 2: Retriever service and API (lumbr)

**Files:**
- Create: `demo/backend/lumbr_demo/service.py`, `demo/backend/lumbr_demo/artifacts.py`, `demo/backend/app/main.py`, `demo/backend/tests/test_service.py`, `demo/backend/tests/test_api.py`

**Interfaces — Consumes:** Task 1; `lumbr.generate.build_sources`, `build_messages`, `load_prompt`.
**Produces:**
- `strip_citations(text: str) -> str` — removes `generate._CITE` matches and collapses doubled spaces.
- `Retriever(encoder: Encoder, index: DenseIndex, store: PassageStore, prompt: dict, threshold: float, name: str)`
  - `.retrieve(query: str, k: int = 5, context_ids: list[str] = [], history: list[dict] = []) -> dict` returning
    `{retriever, top_score, passed, prompt_sha256, hit_ids, sources, messages}`, where each source is the `build_sources` dict plus `passage_id` (= `passage_ids[0]`), `source, title, channel, author, date` (from that passage's row) and `score` (its hit score, `null` if it came from `context_ids`).
  - `.passages(ids: list[str]) -> list[dict]` (Passage shape, `score: None`).
- `create_app(retriever: Retriever, key: str) -> FastAPI` with `GET /health` → `{ok, retriever, passages, threshold, prompt_sha256}`, `POST /retrieve`, `POST /passages`.
- Validation (422): `query` 1–300 chars after strip; `k` 1–20; `context_ids` ≤ 8, each matching `^dsid_[0-9a-z]+#\d{3}$`; `history` ≤ 6 items of `{q, a}` strings ≤ 600 chars; `/passages` `ids` ≤ 16. Bad/missing `Authorization: Bearer <key>` → 401 on all but `/health`.
- Module-level `app` built lazily on first request from `load_retriever(Path(os.environ.get("LUMBR_ARTIFACTS", "artifacts")))` and `os.environ["LUMBR_BACKEND_KEY"]` 
- `load_retriever(artifacts: Path) -> Retriever` in `artifacts.py` — reads `config.json` (raises `ValueError` on `threshold: null`), then builds `OnnxEncoder` (imported inside the function; the class arrives in Task 3), `DenseIndex.load`, `PassageStore`, `load_prompt()`. Test `test_load_retriever_refuses_null_threshold` (tmp dir with a config only).

Hits rule (spec decision 5): `passed = top_score >= threshold`. Passed → top-k ids, then unseen `context_ids` that exist in the store, cut to 8. Not passed with valid `context_ids` → those only (`passed` stays false). Neither → `sources: [], messages: []`.
Messages (decision 6): `m = build_messages(query, sources, prompt)`; with history, return `[m[0], *pairs, m[1]]` where `pairs` = `{"role":"user","content":h["q"]}, {"role":"assistant","content":strip_citations(h["a"])}` per turn, oldest first.

- [ ] **Step 1: Write failing tests** (HashEncoder over the fixture store; threshold chosen so a query copied from a fixture passage passes and `"zzzz qqqq"` doesn't):
  - `test_first_turn_messages_match_study` — `messages == generate.build_messages(query, generate.build_sources(hit_ids, store.with_neighbours(hit_ids), 1), load_prompt())` and `prompt_sha256 == load_prompt()["sha256"]`.
  - `test_sources_carry_passage_fields` — every source has `passage_id == passage_ids[0]`, a non-empty `title`, a float `score`.
  - `test_not_passed_without_context_returns_nothing` — `passed is False`, `sources == []`, `messages == []`.
  - `test_not_passed_with_context_uses_earlier_hits` — sources built only from the context ids; their `score is None`.
  - `test_context_merged_and_capped_at_8` — 5 hits + 6 distinct context ids → `len(hit_ids) == 8`, query hits first, no duplicates.
  - `test_history_citations_stripped` — history answer `"It is 10 MiB [1][2]."` appears as `"It is 10 MiB."`; system message unchanged; last message is the study user message.
  - `test_service_import_is_light` — subprocess `python -c "import app.main, sys; assert 'pandas' not in sys.modules and 'torch' not in sys.modules"` exits 0.
  - API (`fastapi.testclient`, retriever injected): `test_health_open`, `test_auth_required` (401 without/with wrong key), `test_validation_errors` (whitespace query, 301 chars, k=0, k=21, 9 context ids, malformed id, 7 history items, 601-char answer, 17 ids on `/passages` → all 422), `test_passages_omits_unknown`.
- [ ] **Step 2:** `pytest demo/backend/tests -q` → FAIL.
- [ ] **Step 3: Implement** `service.py` and `app/main.py` (pydantic models for the bodies).
- [ ] **Step 4:** `pytest -q` → PASS.
- [ ] **Step 5: Commit** `Demo backend: /retrieve returns study sources and messages (#18)`.

### Task 3: ONNX encoder, artifacts, threshold (lumbr)

**Files:**
- Create: `demo/build/build_artifacts.py`, `demo/build/pick_threshold.py`, `demo/backend/tests/test_encoder_parity.py`, `demo/backend/README.md`
- Modify: `demo/backend/lumbr_demo/encoder.py` (add `OnnxEncoder`), `pyproject.toml` (extra `demo-build = ["sentence-transformers>=3.0", "optimum[onnxruntime]"]`; register pytest marker `artifacts`)

**Interfaces — Produces:**
- `OnnxEncoder(model_dir: Path, query_prefix: str, max_len: int = 512)` — tokenizers + onnxruntime; CLS token of `last_hidden_state`, L2-normalised.
- `build_artifacts.py --passages data/drive/processed/sample_passages.jsonl --out demo/backend/artifacts` writes `embeddings.npy` (float32, N×768, sentence-transformers `normalize_embeddings=True`, `max_seq_length=512`, encoding `p["text"]`), `ids.json`, `passages.sqlite`, `onnx/` (model + tokenizer), `config.json` = `{retriever: "r2", model, model_revision, query_prefix, n_passages, threshold: null, built_at, git_commit}`.
- `pick_threshold.py --artifacts demo/backend/artifacts` encodes the 11 dev questions whose gold docs are all in the store (read `splits/dev_question_ids.txt` + `data.load_questions()`), sets `threshold = round(min(top_score) - 0.02, 4)` in `config.json`, prints each question's top score, and prints the 4 `info_not_found` scores for information.

- [ ] **Step 1: Write failing test** `test_encoder_parity.py` (marked `artifacts`, skipped with a clear reason only when `demo/backend/artifacts/onnx` is absent): for 20 fixed queries (first 20 dev question texts), cosine(OnnxEncoder, SentenceTransformer with the prefix) ≥ 0.999 each, and the top-5 ids over `DenseIndex.load` are identical.
- [ ] **Step 2:** `pip install -e ".[dev,demo,demo-build]"`; run `python demo/build/build_artifacts.py ...`. Expected: `n_passages: 5233`, 1000 docs.
- [ ] **Step 3: Implement** `OnnxEncoder` (export via `optimum`'s ONNX exporter, task `feature-extraction`); run `pytest -m artifacts demo/backend/tests -q` → PASS.
- [ ] **Step 4:** run `pick_threshold.py`; record the threshold and the 11 + 4 scores in the commit message.
- [ ] **Step 5:** smoke: `LUMBR_BACKEND_KEY=dev uvicorn app.main:app --app-dir demo/backend --port 8787`; `curl /health` shows 5233 passages and a numeric threshold; `POST /retrieve` with one answerable dev question returns `passed: true` and the gold doc among `sources`.
- [ ] **Step 6:** README: build, threshold, run commands. **Commit** `Demo backend: ONNX R2 encoder, sample artifacts, provisional threshold (#18)`.

### Task 4: Content and parse cases for the site (lumbr)

**Files:**
- Create: `demo/build/build_questions.py`, `demo/content/colleague_dms.csv`, `demo/build/export_parse_cases.py`, `tests/demo/test_build_questions.py`, `tests/demo/test_parse_cases.py`

**Interfaces — Produces:**
- `select_questions(questions: pd.DataFrame, dev_ids: set[str], store_doc_ids: set[str]) -> pd.DataFrame` — dev only; keeps rows whose non-empty gold set ⊆ store docs, plus `question_type == "info_not_found"`.
- `to_demo(row, dm: str, colleague: str, headers: dict[str, str]) -> dict` — the site's `DemoQuestion`: `id=question_id, category=question_type, infoNotFound, colleague, dm, question, answerFacts, goldAnswer, goldSources=[{doc_id, header}], cached: None`.
- `build_questions.py --artifacts demo/backend/artifacts --out <path>`; exits non-zero with a message if `splits/dev_question_ids.txt` is missing or a selected question has no DM row.
- `colleague_dms.csv` columns `question_id,colleague,dm`: 15 rows, casual one-line DMs drafted by Claude; colleague = the gold doc's author when it has one, otherwise an author from the sample. **John reviews before Task 9.**
- `export_parse_cases.py --out <path>` writes `{prompt_sha256, abstain, cases: [{text, n_sources, citations: [n...], invalid_citations, abstained}]}` from `generate.parse_answer`, covering: `[2]`, `[1][3]`, `[1, 3]`, out-of-range `[7]`, `[2](http://x)`, `[n]` literal, the abstain line exact, with "that specific" inserted, with 3 inserted words (not abstained), curly apostrophe, lowercase, answer-then-abstain-line (not abstained), empty text.

- [ ] **Step 1: Write failing tests:** `test_missing_split_fails_loudly`; `test_selects_only_dev` (a test-split id with gold in the store is excluded); `test_high_level_without_gold_excluded`; `test_info_not_found_included_with_no_sources`; `test_parse_cases_match_python` (re-run `parse_answer` on every exported case and compare).
- [ ] **Step 2:** `pytest tests/demo -q` → FAIL.
- [ ] **Step 3: Implement**, draft the 15 DMs, run both scripts (`questions.json` to `demo/content/questions.json`, git-ignored; `parse_cases.json` to `demo/content/parse_cases.json`, committed).
- [ ] **Step 4:** `pytest -q` → PASS; `questions.json` has 15 entries (11 with gold sources, 4 INF).
- [ ] **Step 5: Commit** `Demo content: 15 dev questions, colleague DMs, parse cases for the site (#18)`.

### Task 5: Retrieval client on the new contract (personal_site)

**Files:** Modify `src/lib/lumbr-demo/types.ts`, `src/lib/lumbr-demo/server/retrieval.ts`; Test `tests/lumbr-demo/retrieval.test.ts` (new)

**Interfaces — Produces:**
- `type Source = Passage & { n: number; passage_ids: string[]; window_ids: string[] }`
- `type ChatMessage = { role: "system" | "user" | "assistant"; content: string }`
- `RetrieveResult = { topScore: number; passed: boolean; hitIds: string[]; sources: Source[]; messages: ChatMessage[] }`
- `RetrievalClient.retrieve(q: string, k: number, contextIds: string[], history: {q: string; a: string}[]): Promise<RetrieveResult>`; `passages` and `warm` unchanged.
- `fakeRetrieval` returns each matched passage as its own source (`n` from 1) and simple `[system, user]` messages (fake only; dev).

- [ ] **Step 1: Write failing tests:** `httpRetrieval` posts `{query, k, context_ids, history}` with the bearer key and maps snake_case to `hitIds/sources/messages` (stub `fetch`); `fakeRetrieval` returns numbered sources and two messages when passed.
- [ ] **Step 2:** `npm test -- retrieval` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `npm test -- retrieval` → PASS (other suites may fail until Task 7; note them).
- [ ] **Step 5: Commit** `lumbr-demo: retrieval client for sources + messages contract (#18)`.

### Task 6: Citation parsing port (personal_site)

**Files:** Modify `src/lib/lumbr-demo/citations.ts`; Create `tests/lumbr-demo/parse_cases.json` (copied from lumbr Task 4); Modify `tests/lumbr-demo/citations.test.ts`

**Interfaces — Produces:**
- `parseAnswer(text: string, nSources: number, abstainLine: string): { citations: number[]; invalidCitations: number[]; abstained: boolean }` — same regex as Python (`/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g`), same abstention rule (`_normalise`, `_words`, `_starts_with_line`, 2 extra words).
- `displayText(text: string, nSources: number): string` — `[1, 3]` → `[1][3]`; out-of-range numbers dropped (an all-invalid bracket disappears); collapse doubled spaces; trim.
- `citationsFor(nums: number[], sources: Source[]): Citation[]` — `{n, doc_id, passage: source}` in first-cited order.
- `numberCitations` is deleted.

- [ ] **Step 1: Write failing tests:** every case in `parse_cases.json` gives the same `citations`, `invalidCitations` (`invalid_citations`), `abstained`; `parse_cases.json`'s `abstain` equals `CANNED_LINE`; `displayText("A [1, 3]. B [7].", 3) === "A [1][3]. B."`; `displayText` leaves `[2](http://x)` untouched.
- [ ] **Step 2:** `npm test -- citations` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `npm test -- citations` → PASS.
- [ ] **Step 5: Commit** `lumbr-demo: parseAnswer port pinned to the study's parse cases (#18)`.

### Task 7: LLM client and ask pipeline (personal_site)

**Files:** Modify `server/llm.ts`, `server/judge.ts`, `server/ask.ts`; Tests `llm.test.ts`, `judge.test.ts`, `ask.test.ts`

**Interfaces — Consumes:** Tasks 5–6. **Produces:**
- `LlmClient.chat(messages: ChatMessage[], signal?: AbortSignal): Promise<{ text: string; usd: number }>`; `openAiCompatible` sends `temperature: 0, max_tokens: 512`.
- `GENERATOR_SYSTEM` and `generatorUser` deleted; judge calls `chat([{role:"system", content: JUDGE_SYSTEM}, {role:"user", content: judgeUser(...)}])`.
- `fakeLlm` answers with the first source's first sentence + ` [1]`, or `CANNED_LINE` when the user message has no sources.
- `askLumbr` flow: one `retrieval.retrieve(text, 5, contextIds, history)` inside the deadline → `sources.length === 0` → canned (no LLM call) → `llm.chat(messages)` → `parseAnswer(raw, sources.length, CANNED_LINE)` → abstained → `kind: "canned"` → else `{kind: "answer", text: displayText(raw, n), citations: citationsFor(...), contextIds: hitIds}`. Spend, deadline, cached fallback, warm-on-error unchanged.

- [ ] **Step 1: Write failing tests** (update existing): `ask` sends the backend's messages unchanged to the LLM; no LLM call when `sources` is empty; abstention with inserted words → `canned`; `contextIds` = `hitIds`; follow-up passes `contextIds` and `history` to `retrieve`; `llm` request body has `max_tokens: 512`, `temperature: 0`; judge still returns verdicts with the new signature.
- [ ] **Step 2:** `npm test` → FAIL in those cases.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `npm test`, `npm run lint`, `npx tsc --noEmit` → all clean.
- [ ] **Step 5: Commit** `lumbr-demo: ask pipeline uses the study's messages and parsing (#18)`.

### Task 8: Real questions and end-to-end check (both repos)

**Files:** Modify `personal_site/src/data/lumbr-demo/questions.json`, `personal_site/.env.example` (`LUMBR_BACKEND_URL=http://localhost:8787`), `lumbr/docs/DEMO_HANDOFF.md`, `lumbr/README.md` (layout lines for `demo/`)

- [ ] **Step 1:** After John approves the DMs, copy `demo/content/questions.json` into the site; `npm test` (fixture tests on `questions.json` shape) → PASS.
- [ ] **Step 2:** Run the service (Task 3 Step 5) and `npm run dev` with John's `.env.local` (OpenRouter vars) + `LUMBR_BACKEND_URL`/`LUMBR_BACKEND_KEY`.
- [ ] **Step 3: Verify (done-when):** in `/projects/lumbr-enterprise-rag/demo`, ask 3 answerable dev questions in Lumbr → cited answers whose chips open the right sources (gold doc among them); ask 1 `info_not_found` question → the canned line; one follow-up → answered with context. Record question ids, answers and timings.
- [ ] **Step 4:** Update `DEMO_HANDOFF.md` (status, how to run locally, new contract, threshold value, open items: deploy, full index B5). **Commit** in each repo.

### Task 9 (optional, gated on John's accounts): Deploy the service

- [ ] Private HF repo for artifacts; `fetch_artifacts.py` build step; Vercel project rooted at `demo/backend` with `lumbr` installed `--no-deps` from the same commit; measure cold start, RSS (< 1.6 GB) and `/retrieve` latency; set `LUMBR_BACKEND_URL` for Preview. Stop and ask John before creating any account resource.
