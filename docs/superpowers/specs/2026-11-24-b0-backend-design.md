# B0 in the demo search service (#26)

**Date:** 2026-11-24 · **Branch:** `john/26-b0-backend` · **Approved:** John, 2026-11-24 (with Pyserini)

## Why

The demo serves one setup. By the rule in `2026-11-24-demo-setup-decision-rule.md` (PR #60), that setup is
**B0 = R1 BM25 + R4 dense, fused with RRF** (dev recall@10 0.766). The search service today is dense-only
R2 on the v0 sample, so it has to change to serve the study's B0 on the full corpus.

## Goals and non-goals

- **Goal:** the service returns exactly the study's B0 ranking for a question. Parity is checked on all
  100 dev questions against the study's saved `r1_r4_rrf` dev run.
- **Goal:** no change to the API, the prompt, small-to-big, history or the site.
- **Non-goal:** any reranker. Notebook 18 showed MiniLM doesn't help, and bge-reranker-v2-m3 is too slow on CPU.
- **Non-goal:** setup switching in the service (team decision: one setup).

## Approaches considered (BM25 on the server)

| | Pyserini/Lucene (chosen) | bm25s, pure Python |
|---|---|---|
| Matches the study | Exactly: the same Lucene index, k1 = 0.9, b = 0.4, the same analyzer | Different tokenizer, so results drift |
| Server needs | Java 21, about 1–1.5 GB for the JVM | Nothing extra |

John chose Pyserini. The server is a VM (AWS Lightsail), not a Vercel function, so Java is cheap. Rough
memory: R4 matrix in float32 is 2.4 GB, the JVM about 1–1.5 GB, plus Python and ONNX, so about 4 GB in all.
That means **the 8 GB Lightsail plan**.

## Design

- **`Retriever`** takes an optional `lexical` searcher. With one, it:
  - takes the top 50 from BM25 and the top 50 from R4;
  - fuses them with `lumbr.index_rrf.fuse(..., k=60, top_n=50)`, with **R1 first**, as notebook 17 did.
    Order matters: two RRF sums can differ only by float rounding, which depends on the order.
  - returns the top k fused hits. A source's `score` is its RRF score.

  Without a lexical searcher it stays dense-only, as the sample build serves today.
- **The similarity filter reads R4's top cosine**, not the RRF score, which only reflects rank. The threshold
  is re-picked with `pick_threshold.py` on the full artifacts (dev only).
- **`lexical.LuceneBM25`** wraps Pyserini's `LuceneSearcher` with its defaults, the same as the study's R1.
  It's imported lazily, so the API module stays light and the tests don't need Java.
- **`DenseIndex.search`** now does a partial sort, so it's fast on 772K passages. The results and the
  tie-break by id are unchanged.
- **Artifacts** (`demo/build/build_b0_artifacts.py`, notebook 19) come from the study's own outputs, with
  nothing re-embedded:

  | Artifact | Source |
  |---|---|
  | `embeddings.npy` (float16 on disk, float32 in memory, as the study's `load_index` casts) + `ids.json` | R4 shards |
  | `passages.sqlite` | `passages.jsonl` |
  | `bm25/` | the R1 Lucene index |
  | `onnx/` | R4 model exported for queries |
  | `config.json` | `"retriever": "B0"`, `"lexical": "bm25"` |

  The script checks that the index and the store hold the same passage ids.
- **Parity** (`demo/build/check_parity.py`): served recall@10 against the study's, per-question top-10
  agreement, and query latency.
  - Small differences may come from the query encoder (ONNX fp32 on CPU versus the study's fp16 on GPU).
  - Anything bigger blocks deploying.
- **Hosting:** artifacts go to a private Hugging Face dataset repo. The server pulls them and serves with
  `.[demo,demo-b0]` and Java 21.

## Left to the builder

- The HF repo name (John's account).
- The Lightsail setup script, a separate step that John approves before anything billable is created.
