# Test retrieval table (376 scorable test questions)

From notebook 16 (`16_patricia_test_eval.ipynb`), run on an L4 on 2026-11-25. Run folders: `runs/*_test_2026-11-25`.
Its eval files (`runs/20261125_0429*_test_at{10,50}.json`) give the same values.
Of the 400 test questions, 376 are scored; the other 24 have no gold documents. The test split is for final
reporting only: nothing here was tuned on it.

| run | recall@10 | nDCG@10 | MRR@10 | hit@10 | extra@5 | extra@10 | recall@50 | lat_med_ms | lat_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 (BM25) | 0.6884 | 0.5784 | 0.5731 | 0.7447 | 4.18 | 9.01 | 0.8134 | - | - |
| R2 (BGE-base) | 0.5059 | 0.4164 | 0.4254 | 0.5665 | 4.37 | 9.27 | 0.6190 | - | - |
| R3 (Qwen3) | 0.6335 | 0.5188 | 0.5146 | 0.6915 | 4.26 | 9.11 | 0.7365 | - | - |
| R4 (fine-tuned BGE) | 0.6288 | 0.5193 | 0.5189 | 0.6862 | 4.23 | 9.08 | 0.7321 | - | - |
| R5 (R1+R3 RRF) | 0.7531 | 0.6191 | 0.6060 | 0.8005 | 4.12 | 8.95 | 0.8562 | - | - |
| R1+R4 RRF (k=60) | 0.7210 | 0.6225 | 0.6231 | 0.7739 | 4.09 | 8.94 | 0.8347 | - | - |
| R1 + R6 | 0.7413 | 0.6719 | 0.6837 | 0.7952 | 4.05 | 8.94 | 0.8134 | 621.8 | 703.8 |
| R3 + R6 | 0.7000 | 0.6269 | 0.6386 | 0.7527 | 4.11 | 9.03 | 0.7365 | 583.9 | 675.7 |
| R4 + R6 | 0.7003 | 0.6288 | 0.6417 | 0.7580 | 4.11 | 8.99 | 0.7321 | 570.1 | 665.1 |
| R5 + R6 | 0.7888 | 0.7084 | 0.7132 | 0.8378 | 3.99 | 8.89 | 0.8562 | 612.7 | 691.2 |
| R1+R4 + R6 | 0.7819 | 0.7011 | 0.7063 | 0.8298 | 4.01 | 8.89 | 0.8347 | 609.5 | 690.9 |

- **extra@k**: retrieved documents in the top k that aren't gold. Without the benchmark's "valid" judge this is an upper bound (`src/lumbr/eval.py`).
- **recall@50** for the `+ R6` rows is their first-stage input's: the reranker only reorders that run's top 50.
- **Latency** is the reranker alone, per question, on an L4 (bge-reranker-v2-m3, 50 passages). First-stage retrieval
  wasn't timed; per-stage latency for every setup is #25.
- R2 + R6 was run on dev only (`results/dev_table.md`), not on test.
