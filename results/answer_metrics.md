# Answer metrics — test split

**Main runs**

| Run | n | Correctness | Completeness | Overall | Abstention | False-refusal |
|---|---|---|---|---|---|---|
| R1 (BM25) | 400 | 45.8% | 48.6% | 40.3% | 93.8% | 31.5% |
| R2 (BGE-base) | 400 | 34.8% | 38.9% | 31.1% | 100.0% | 44.0% |
| R3 (Qwen3-4B) | 400 | 42.5% | 45.6% | 36.6% | 100.0% | 35.4% |
| R4 (fine-tuned BGE) | 400 | 43.8% | 45.1% | 38.4% | 100.0% | 34.4% |
| R5 (R1+R3 RRF) | 400 | 50.5% | 52.2% | 43.7% | 87.5% | 26.8% |
| R5+R6 (reranked) | 400 | 60.0% | 59.7% | 52.5% | 93.8% | 20.1% |
| R1+R4 (B0) | 400 | 53.5% | 52.5% | 45.6% | 93.8% | 25.8% |

**Ablations: hits only, no neighbouring passages**

| Run | n | Correctness | Completeness | Overall | Abstention | False-refusal |
|---|---|---|---|---|---|---|
| R1 hits-only | 400 | 38.5% | 43.2% | 33.2% | 93.8% | 37.5% |
| R5+R6 hits-only | 400 | 53.0% | 54.7% | 46.0% | 93.8% | 23.2% |

**conflicting_info aside**

| Run | n | Correctness | Completeness | Overall | Abstention | False-refusal |
|---|---|---|---|---|---|---|
| R1 (BM25) (conflicting_info) | 16 | 81.2% | 78.9% | 70.2% | — | 0.0% |
| R2 (BGE-base) (conflicting_info) | 16 | 68.8% | 70.8% | 57.9% | — | 12.5% |
| R3 (Qwen3-4B) (conflicting_info) | 16 | 50.0% | 65.6% | 44.9% | — | 18.8% |
| R4 (fine-tuned BGE) (conflicting_info) | 16 | 62.5% | 67.4% | 53.0% | — | 0.0% |
| R5 (R1+R3 RRF) (conflicting_info) | 16 | 87.5% | 78.7% | 73.8% | — | 0.0% |
| R5+R6 (reranked) (conflicting_info) | 16 | 81.2% | 74.8% | 68.0% | — | 0.0% |
| R1+R4 (B0) (conflicting_info) | 16 | 81.2% | 76.7% | 68.0% | — | 0.0% |
| R1 hits-only (conflicting_info) | 16 | 75.0% | 74.5% | 64.0% | — | 0.0% |
| R5+R6 hits-only (conflicting_info) | 16 | 75.0% | 77.2% | 66.1% | — | 6.2% |

n = answers judged. Correctness, Completeness, and Overall are means over all judged answers, abstentions included; Overall = Completeness when the answer is judged correct, else 0 (the benchmark's combined score). Abstention = share of info\_not\_found questions where the system abstained (want high). False-refusal = share of answerable questions where it abstained (want low). — = undefined (no questions of that kind in the row).
