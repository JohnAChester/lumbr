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

John chose Pyserini. The server is a VM (AWS Lightsail), not a Vercel function, so Java is cheap.

**Memory target: the 4 GB plan ($24/month), not the 8 GB one ($44/month)** (John, 2026-11-24). The demo is
always on, so the backend loads data lean without giving up any of B0's recall:

| Part | How it's held | Estimate |
|---|---|---|
| R4 vectors (772K × 768) | FAISS float16 index (`dense_sq_fp16.faiss`, built from `embeddings.npy` on first load), memory-mapped | ~1.1 GB of reclaimable OS cache; ~35 MB the process holds |
| BM25 | Lucene memory-maps its index; JVM heap capped with `-Xmx` (`LUMBR_JVM_HEAP`, default `1g`) | ≤ 1 GB of heap, plus cache |
| Passage store | SQLite on disk, a few rows read per query | small |
| R4 query encoder | ONNX in fp32 (int8 would change the query vectors) | ~0.5 GB |

That's about 2–2.5 GB of memory the process holds, plus cache the OS can drop.

**float16, not int8, for the vectors:**
- Casting float16 to float32 is exact, as the study's `load_index` did, so rankings match the study's.
- int8 would save about another 0.6 GB of cache, but changes the scores, so it could cost recall.
- Plan: add int8 only if the 4 GB measurement is tight, and only after the parity check confirms recall.

**Measured (Colab parity, numpy chunked search, 2026-11-24):** recall@10 0.7657 = the study's; 98 of 100 top-10
lists identical and all 100 the same set; process-held memory 1.7 GB (includes pandas, which the server
doesn't load); search 1.86 s median. The chunked numpy cast was the slow part. On a laptop with 2 threads,
**FAISS float16 halves the dense search (0.47 s vs 0.9 s) with the identical top 50**, and memory-mapped it holds
35 MB, so the server uses FAISS.

**Measured with FAISS (Colab parity, full corpus, 2026-11-24): recall@10 0.7657 = the study's, 100 of 100 top-10
lists identical, search 0.24 s median / 0.34 s p95, process-held memory 1.58 GB (pandas included), mapped
cache 1.74 GB.** So the server targets the 4 GB plan; confirm memory and latency on the server itself.

**Measuring:** `check_parity.py` reports peak RSS, anonymous memory (what the process really holds) and
file-backed memory (reclaimable mapped pages) after all 100 dev queries. The plan is chosen from those numbers.

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
