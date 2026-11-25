# Latency per stage

GPU: NVIDIA L4 · git: ae4990d

- gen: `Qwen/Qwen3.5-9B`
- r2: `BAAI/bge-base-en-v1.5`
- r3: `Qwen/Qwen3-Embedding-4B`
- r4: `BAAI/bge-base-en-v1.5 (fine-tuned)`
- rerank: `BAAI/bge-reranker-v2-m3`

## Per-stage latency (single-query, median and p95)

| Stage | Median ms | p95 ms | n |
|---|---|---|---|
| bm25 | 22.2 | 47.2 | 47 |
| r2_encode | 12.8 | 16.3 | 47 |
| r2_search | 81.4 | 86.4 | 47 |
| r3_encode | 59.5 | 66.9 | 47 |
| r3_search | 273.9 | 404.9 | 47 |
| r4_encode | 12.8 | 13.3 | 47 |
| r4_search | 81.1 | 84.0 | 47 |
| rrf | 0.1 | 0.1 | 47 |
| rerank | 632.4 | 716.4 | 47 |
| gen | 10517.1 | 31571.4 | 47 |

> Reranker: BAAI/bge-reranker-v2-m3 fp16, top-50 per question, batch_size=64

> Generator: Qwen3.5-9B, temperature=0, max_tokens=768, thinking off; single-query vLLM calls (not batched throughput)

> generation timed with R5 (BM25+Qwen3-Embedding RRF) top-5 context; every setup passes the same number of passages, so per-setup generation cost is assumed equal

## End-to-end latency per setup

> Retrieval and E2E = **median of per-question totals** (not sum of per-stage medians).

| Setup | Retrieval median ms | Retrieval p95 ms | E2E median ms | E2E p95 ms |
|---|---|---|---|---|
| R1 | 22.2 | 47.2 | 10565.6 | 31595.1 |
| R2 | 94.5 | 100.9 | 10619.4 | 31664.9 |
| R3 | 333.5 | 465.7 | 10982.9 | 31936.0 |
| R4 | 93.8 | 96.9 | 10612.0 | 31666.7 |
| R5 | 361.2 | 486.6 | 11031.4 | 31959.6 |
| R1+R4 | 117.5 | 147.6 | 10660.5 | 31690.3 |
| R1+R6 | 666.1 | 735.7 | 11221.9 | 32281.7 |
| R3+R6 | 972.6 | 1130.5 | 11639.2 | 32622.6 |
| R4+R6 | 726.6 | 810.4 | 11268.3 | 32353.3 |
| R5+R6 | 1006.0 | 1165.1 | 11687.7 | 32646.3 |
| R1+R4+R6 | 760.0 | 829.8 | 11316.8 | 32377.0 |

Retrieval = sum of retrieval-stage times per question, then median/p95 across questions. E2E adds single-query generation time per question before aggregating.
