# Citation support (ALCE citation recall) — test split

Sample: the same 100 test questions for both runs, drawn (seed 23) from the questions both runs answered rather than abstained on.

| Run | Answers | Cited sentences judged | Supported | Citation support | 95% CI |
|---|---|---|---|---|---|
| R1+R4 (B0) | 100 | 340 | 313 | 92.1% | 89.3% to 94.9% |
| R5+R6 (reranked) | 100 | 348 | 313 | 89.9% | 85.6% to 93.5% |

Citation support = share of cited sentences whose cited sources, together, support the sentence (judge: claude-sonnet-5-5, ALCE-style yes/no prompt; sentences under 20 characters skipped). The judge sees each cited source as the generator did: header plus the whole small-to-big window. Other runs were not checked (API budget).
