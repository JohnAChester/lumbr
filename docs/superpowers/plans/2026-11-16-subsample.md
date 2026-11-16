# Indexed Subsample (#5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the committed doc-ID list `splits/subsample_doc_ids.txt` (~140–150K docs) plus a manifest.

**Architecture:** `src/lumbr/subsample.py` has three independent steps. Each writes its result to Drive
`processed/`, so they can run in different Colab sessions: BM25 look-alikes (CPU), R2 look-alikes (GPU), and
assembly. Corpus I/O is kept separate from pure functions, which are unit-tested on toy data with no network,
GPU or tokenizer.

**Tech Stack:** pandas, pyarrow, scikit-learn (HashingVectorizer), scipy.sparse, numpy, sentence-transformers (GPU step only).

**Spec:** `docs/superpowers/specs/2026-11-16-subsample-design.md`

## Global Constraints
- BM25: k1=0.9, b=0.4, lowercase, English stopwords, no stemming, HashingVectorizer n_features=2**24, alternate_sign=False, norm=None.
- R2: `BAAI/bge-base-en-v1.5`, max_seq_length 512, fp16, normalized; query prefix `"Represent this sentence for searching relevant passages: "`.
- Top 50 per question from each method; all 500 questions.
- Random fill: 100,000 docs, proportional to each source's share of the full corpus, min 10 per source, seed 42.
- Document text for matching: HF `title + "\n" + content`.
- Outputs: `splits/subsample_doc_ids.txt` (sorted, one per line, trailing newline) and `splits/subsample_manifest.json`.
- Every subsample ID must exist in `sources.load_index()` (the erb_json parquet that #4 chunks).

## Review Focus
1. A gold or look-alike ID that is missing from erb_json (HF and GitHub copies differ): drop it from the list and report it in the manifest, never fail silently.
2. Unanswerable questions (empty `expected_doc_ids`) still get look-alikes, and contribute no gold.
3. A source too small for its quota: take what exists, without crashing.
4. Re-running assembly with the same inputs gives a byte-identical ID file (determinism).
5. BM25 when a question has only stopwords or unknown terms: return 50 IDs anyway (ties broken by doc order) or fewer, never crash.

---

### Task 1: BM25 look-alikes

**Files:** Create `src/lumbr/subsample.py`, `tests/test_subsample.py`. Modify `pyproject.toml` (extra `subsample = ["scikit-learn>=1.3", "scipy", "sentence-transformers>=3.0"]`).

**Interfaces:**
- Produces: `bm25_index(texts: Iterable[str]) -> BM25Index` (dataclass holding the CSR doc×term weight matrix, transposed to CSC for column slicing).
- `BM25Index.top_k(queries: list[str], k: int) -> list[list[int]]` returns row indices, best first.
- `bm25_lookalikes(k=50) -> dict[str, list[str]]` (corpus I/O: question_id → doc_ids; writes `processed/subsample_lookalikes_bm25.json`).

- [ ] Test `test_bm25_ranks_term_overlap`: corpus `["gpu kv cache eviction", "lunch menu friday", "kv cache warmup on gpu nodes"]`; `top_k(["kv cache eviction"], 2)[0] == [0, 2]`.
- [ ] Test `test_bm25_prefers_short_docs_on_equal_tf` (length normalisation): a doc with the same term once in 3 words beats one with it once in 30 words.
- [ ] Test `test_bm25_stopword_only_query_does_not_crash`: `top_k(["the and of"], 2)` returns a list of length ≤ 2.
- [ ] Run, see them fail; implement; run, see them pass; commit.

### Task 2: R2 look-alikes with a running top-k

**Interfaces:**
- Produces: `running_topk(query_emb: np.ndarray, doc_slices: Iterable[tuple[int, np.ndarray]], k: int) -> list[list[int]]`. Slices are `(offset, emb)`; it returns global row indices, best first.
- `r2_lookalikes(k=50, slice_size=50_000, batch_size=256) -> dict[str, list[str]]` (GPU, corpus I/O; writes `processed/subsample_lookalikes_r2.json`).

- [ ] Test `test_running_topk_matches_full_sort`: 1,000 random docs in slices of 128 → same top 10 as `argsort` on the full matrix.
- [ ] Test `test_running_topk_k_larger_than_corpus`: 5 docs, k=10 → 5 results.
- [ ] Fail → implement (merge per slice with `np.argpartition`) → pass → commit.

### Task 3: Assembly

**Interfaces:**
- Consumes: the two look-alike dicts; questions (`question_id`, `expected_doc_ids`); the index DataFrame (`doc_id`, `source_type`).
- Produces: `assemble(gold: dict[str, list[str]], lookalikes: dict[str, dict[str, list[str]]], index: pd.DataFrame, n_random=100_000, min_per_source=10, seed=42) -> tuple[list[str], dict]` (sorted IDs, manifest).
- `build_subsample() -> dict` (corpus I/O: reads the Drive JSONs, writes both files in `splits/`).

- [ ] Test `test_includes_all_gold_and_lookalikes`.
- [ ] Test `test_random_fill_is_proportional_with_minimum` (toy index of 3 sources at 90/9/1 %, n_random=100 → counts within ±1 of the shares, smallest ≥ min).
- [ ] Test `test_random_fill_excludes_selected_and_is_deterministic`.
- [ ] Test `test_ids_missing_from_index_are_dropped_and_reported` (manifest `missing_from_corpus`).
- [ ] Test `test_small_source_takes_what_exists`.
- [ ] Test `test_manifest_counts_and_overlap` (`by_part`, `by_source`, `lookalike_overlap`, `per_question`).
- [ ] Fail → implement → pass → commit.

### Task 4: Notebooks and docs

- [ ] `notebooks/04_john_subsample_bm25.ipynb` (CPU): bootstrap, then `bm25_lookalikes()`, then spot-check that gold docs appear in their own question's top 50 (recall@50 as a sanity number).
- [ ] `notebooks/05_john_subsample_r2.ipynb` (GPU): bootstrap, `pip install -e .[subsample]`, `r2_lookalikes()`, `build_subsample()`, show the manifest, then git add/commit/push `splits/` on a branch.
- [ ] Update README layout and `docs/SCHEMA.md` (the file table points at the subsample). Full test suite passes. Commit.
