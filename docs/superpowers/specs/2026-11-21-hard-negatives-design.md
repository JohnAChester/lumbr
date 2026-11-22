# Hard negatives for R4 (#13, #15): false-negative guards

## Intent
R4 is R2 (`BAAI/bge-base-en-v1.5`) fine-tuned with MultipleNegativesRankingLoss on synthetic
(question, positive, hard negatives) triples (#13, #15). The loss pushes each question *away* from its
negatives, so a negative that actually answers the question (a **false negative**) trains R4 against the
right passage. A noisy positive is a weak wrong signal; a false negative is a strong one aimed at the
passage we want ranked first. This spec adds two guards against false negatives to the #13 mining step,
and applies the first one to #15's in-batch negatives too.

## Why it matters here
- **Sibling chunks look like the best negatives.** Every passage's indexed `text` starts with its
  document's header (source, channel, title, author, date), so the other chunks of the same document
  share many terms and land near the top of BM25 and R2 results. A miner that keeps "the top-k
  non-positives" picks them first.
- **Siblings often hold the answer.** Hand review of 100 random questions from
  `synth/qgen_2026-11-18/filtered.jsonl` (2026-11-21): 6 were *unanswered* by their own chunk (Qwen turned
  a question asked in the passage into a training question). In 3 of the 6 the answer is in the next
  chunk of the same document (#8, #83, #94 in the review), and all 6 point to the right document. Mined
  naively, those answering chunks become negatives.
- **The corpus has deliberate near-duplicates.** The subsample (#5) pulls in BM25 and R2 look-alikes,
  including conflicting and near-duplicate copies of documents. Those are false negatives in *other*
  documents, which the same-document rule doesn't catch.
- **Scoring is per document** (`docs/SCHEMA.md`): a hit counts if any passage of a gold document comes
  back, and generation expands each hit to `chunk_idx ± 1` (D16). Teaching R4 to separate chunks of one
  document buys nothing in our evaluation and can only hurt it (it may also teach R4 to discount the
  header, which is a strong document-level signal).
- **R4 is judged against R2.** Bad mining shows up as "fine-tuning didn't help", so getting this wrong
  would undermine the R4 vs R2 comparison in the report.

## Goals / non-goals
- Goal: no negative from the question's own document, mined or in-batch.
- Goal: no negative the cross-encoder rates about as relevant as the positive.
- Goal: record how many candidates each guard removes, for the report.
- Non-goal: changing #12's question filter. Questions that their own chunk doesn't answer stay in for
  now; at document level they still point to the right document.

## Design
1. **Guard 1: exclude the question's own document.** When mining (#13), drop every candidate whose
   `schema.doc_id_of(passage_id)` equals the positive's document. This covers the positive itself and all of
   its siblings. The 4 doc IDs shared by two files (`docs/SOURCES.md`) are already one document after
   chunking, so they need no special case.
2. **Guard 1 in training (#15): no two examples from one document in a batch.** With
   MultipleNegativesRankingLoss every other example's positive is an in-batch negative. #12 keeps up to 2
   questions per passage, so without this a batch can hold two questions with the *same* positive.
   Sentence-Transformers' `BatchSamplers.NO_DUPLICATES` only catches identical texts, not siblings, so use a
   batch sampler keyed on the positive's `doc_id`.
3. **Guard 2: positive-aware cross-encoder threshold.** #13 already plans to "drop candidates the
   cross-encoder scores as relevant". Make the cutoff relative to the positive: drop a candidate if
   `CE(q, candidate) >= 0.95 * CE(q, positive)` (the "TopK-PercPos" rule from NV-Retriever). Scores from
   `bge-reranker-v2-m3` run high (over half of the positives score above 0.94 in #12), so a fixed absolute
   cutoff would either keep near-duplicates or throw away most hard negatives. A relative one adapts per
   question.
   - Use the same model and settings as #12 (`BAAI/bge-reranker-v2-m3`, fp16, `max_length=640`) so
     `CE(q, positive)` can be taken from the `score` field of `filtered.jsonl` without rescoring.
   - Check the 0.95 margin by hand on ~50 candidates just above and just below it before the full run; adjust
     if dropped candidates are mostly true negatives, or kept ones mostly answer the question.
4. **Order:** guard 1 first (free, no model), then guard 2 on what's left, then keep the top N per question
   by retriever rank.
5. **Stats to save** next to the triples: candidates per question before and after each guard, the share
   removed by guard 1 vs guard 2, and the number of questions left with fewer than N negatives.

## Left to the builder
Candidate depth per retriever (#13 says top BM25 and R2 results), N negatives per question, how BM25 and R2
candidates are merged, and the exact batch-sampler implementation.
