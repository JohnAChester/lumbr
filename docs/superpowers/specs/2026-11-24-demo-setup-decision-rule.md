# Which setup the demo serves: the rule, fixed before the test runs (#26, #22)

**Date:** 2026-11-24 · **Status:** draft, for the team to agree before #22 runs · **Outcome (dev): B0 is served, see "Result"**

## Why write it down first

The demo serves one retrieval setup (team decision, 2026-11-23). We choose it using dev **and** test
together, because the choice gets better with more questions: dev has 94 scorable questions, test about
four times that. Writing the rule down before #22 runs means nobody can pick the rule after seeing the
results.

Choosing on test doesn't make the choice worse. It only inflates one number: the shipped setup's own
test score, which was also used to choose it. Every other setup's test score stays a clean estimate. The
reporting rules below deal with that one number.

## Candidates

Only setups that can run on the hosting are candidates.

| Candidate | What it is | Dev recall@10 | Query-time models |
|---|---|---|---|
| **B0** | BM25 + fine-tuned dense, RRF (R1+R4) | 0.766 | R4 encoder |
| **B-mini@d** | B0 + small reranker (`ms-marco-MiniLM-L6-v2`) at depth d | notebook 18 | R4 encoder + MiniLM |
| **A-mini@d** | BM25 + small reranker at depth d | notebook 18 | MiniLM |
| A0 | BM25 alone (R1) | 0.665 | none |

Source: `results/dev_table.md`, and notebook 18 for the small-reranker rows.

**B0 is served first.** It fits Vercel Hobby and doesn't depend on #22.

**Out:**
- **R3**, because its 4B embedder can't run on the hosting.
- **R5 and R5 + rerank**, because R5 merges R1 with R3, so it needs the same 4B embedder for every query.
- **The study's reranker** (bge-reranker-v2-m3, at any depth). It would be the best on dev (R1+R4 + rerank, 0.819), but it's
  far too slow on CPU. On John's laptop, with 2 cores and 10 passages of about 300 tokens, it took **17 s** per
  question (median), and the model alone needs 2.5 GB of memory. A larger CPU or int8 won't make it the six times
  faster it needs to be. Only a GPU would, at roughly $380/month always on.
- **bge-reranker-base**, at 15 s for 20 passages on one core.

**Small reranker speed** (MiniLM-L6, laptop CPU, per question):

| Depth | 1 core | 2 cores |
|---|---|---|
| 10 | 1.0 s | 0.7 s |
| 20 | 2.4 s | 1.7 s |
| 50 | 6.1 s | 4.3 s |

Model memory is about 0.1 GB. Depths 10 and 20 can pass the gate. Depth 50 can't.

## The rule

1. **Hosting gate.** Each candidate is measured on the real host with the full index:
   - **Memory:** peak memory under 1.8 GB on Vercel (the limit is 2 GB).
   - **Speed:** warm retrieval at p95 within **3 s**, reranker included. The site gives the search service
     8 s and the whole question 12 s, and the LLM needs most of the rest.
   - **Rerank depth:** a reranker candidate uses the deepest of 20 and 10 that passes. Each one is a new
     setup, measured on dev in notebook 18. If it's served, #22 also has to run it on test.
   - **Host:** a candidate that passes only on the Hugging Face Space fallback counts as failing,
     because a 30–60 s wake doesn't fit a 12 s question.
2. **Score.** Rank the candidates that pass by **recall@10 on dev and test pooled**, using every scorable
   question. #22 has to log test retrieval for every candidate that passes the gate.
3. **Tie-break.** If the top two are within **2 points** of recall@10, ship the faster one (lower p95).
   Otherwise ship the top one.
4. **No tuning on test.** Prompt, abstention rule, threshold and rerank depth are all set on dev. If test
   results lead to changing any of them, test is no longer held out for the study either.

## Reporting

- The study reports test scores for every setup side by side. Those numbers are unaffected by this rule.
- The demo's About panel shows the served setup's **dev** score and its **test** score, with a note that
  the test score was also used to choose it. It never shows the test score on its own.
- The demo's other differences from the study are stated on the About panel: int8 index, rerank depth and
  host.

## What happens before #22

Nothing here waits for #22:
- Build the full index for B0 (BM25 + R4, int8).
- Deploy B0 to Vercel and measure the gate there.
- Run notebook 18 on dev. If a MiniLM candidate is close to the study reranker and passes the gate on
  Vercel, it becomes the candidate to beat.

When #22 lands, apply the rule. Switching between candidates is a config change.

## Result: notebook 18 (dev, 2026-11-24)

| Setup | recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| R1 (BM25) | 0.665 | 0.594 | 0.611 |
| **B0: R1+R4 RRF** | **0.766** | **0.650** | **0.648** |
| R1 + MiniLM @10 / @20 / @50 | 0.655 / 0.681 / 0.723 | 0.576 / 0.572 / 0.594 | 0.586 / 0.569 / 0.582 |
| R1+R4 + MiniLM @10 / @20 / @50 | 0.746 / 0.766 / 0.774 | 0.623 / 0.623 / 0.624 | 0.622 / 0.611 / 0.608 |
| R1 + study reranker @50 | 0.802 | 0.700 | 0.695 |
| R1+R4 + study reranker @50 | 0.819 | 0.721 | 0.718 |

- **The small reranker doesn't help B0.** At the depths that pass the speed gate (10 and 20), it matches or loses
  recall@10, and it lowers nDCG and MRR at every depth: it puts the right document lower in the list.
  MiniLM was trained on web search (MS MARCO) and doesn't transfer to this corpus. Depth 50 gains 0.8 points of
  recall, which is within noise on 94 questions, and it's too slow anyway.
- **So B0 is the demo setup.** It's the only candidate that passes the gate and isn't dominated. Under the rule, a
  MiniLM variant within 2 points loses the tie-break to the faster B0, and A0 trails by 10 points. #22 can't change
  this choice, so the demo build doesn't wait for it. #22 still reports B0's test score.
- **Hosting:** the full corpus (~2–3 GB of passage text plus the indexes) doesn't fit a Vercel function, so the
  search service moves to a small always-on AWS Lightsail server (2 vCPU, 4–8 GB). The site stays on Vercel.

## Open for the team

- Is recall@10 the right score, or should it be answer correctness from #23's judge once that runs on
  both splits?
- #22 says "all six setups". It also has to cover the candidates in this table that pass the gate.
- #21 still names "BM25 + reranker" and "BM25 + dense + reranker" as the demo pipelines. With the study's
  reranker out, those become the MiniLM variants, or B0 if they don't hold up.
