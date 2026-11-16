# Indexed subsample (#5): design

## Intent
Pick the ~140–150K of 512K documents that every retriever indexes, so the study fits Colab compute while keeping
the benchmark's hard cases: every gold document, and each question's near-duplicates and conflicting copies.
The doc-ID list is committed so all three of us index exactly the same documents. Proposal: "Corpus and preprocessing".

## Goals / non-goals / anti-goals
- Goal: a fixed, reproducible ID list at `splits/subsample_doc_ids.txt` (the input `passages.build_passages()` already expects).
- Goal: record which method contributed each look-alike, so the report can describe the subsample.
- Non-goal: R3 look-alikes (too costly to embed 512K docs with a 4B model); noted as a limitation.
- Anti-goal: a subsample tilted for or against one retriever family. That's why both BM25 and R2 pick look-alikes
  (issue comment on #5).
- Anti-goal: using benchmark questions for training. They only select documents here.

## Approaches
- A. BM25-only look-alikes: CPU only, simplest, but the distractors are hard for BM25 only. Rejected for that bias.
- **B. BM25 and R2, top 50 from each (chosen):** costs one short GPU run.

## Design
1. **Text per document:** the HF release's `title + "\n" + content` (flat text, one row per doc). It's already
   on Drive, and doc-level matching doesn't need our structured rendering. *(my call)*
2. **BM25 (CPU):** BM25 (k1=0.9, b=0.4, the Pyserini/Anserini defaults) over all documents. Lowercased,
   English stopwords, no stemming. Query = question text. Top 50 per question → `processed/subsample_lookalikes_bm25.json`.
   It's computed with scikit-learn's `HashingVectorizer` (2^24 buckets) plus SciPy, streaming the corpus into one
   sparse matrix of about 2 GB. `bm25s` was the first choice, but it holds every document's token list in Python
   memory, which won't fit for 512K long documents. *(my call)*
3. **R2 (GPU):** `BAAI/bge-base-en-v1.5`. Embeds each document's first 512 tokens in fp16, normalized; queries
   use the BGE query instruction. Exact cosine top 50 (a 512K × 768 matrix fits in GPU memory) →
   `processed/subsample_lookalikes_r2.json`. Embeddings are not kept: R2's real index is built on passages in #10. *(my call)*
4. **Which questions:** all 500, including unanswerable ones. They have no gold documents, but their look-alikes
   are the near-misses an assistant must abstain on. *(my call)*
5. **Assembly (CPU):** gold ∪ BM25 look-alikes ∪ R2 look-alikes, then random distractors from the rest. The number
   of random distractors per source is proportional to that source's share of the full corpus (100K total,
   seed 42, at least 10 per source). The look-alikes reflect what questions ask about; the random fill keeps
   the overall source mix realistic. *(my call)*
6. **Outputs (committed):** `splits/subsample_doc_ids.txt` (sorted, one per line) and `splits/subsample_manifest.json`:
   - counts by part and by source;
   - overlap between the BM25 and R2 look-alikes;
   - per-question look-alike lists;
   - seed, data revisions and code version.
7. **Checks:** every gold ID and every look-alike is in the list, and every ID exists in `erb_json.parquet`
   (so #4 can chunk it). The total lands in ~140–150K.
8. **Code:** `src/lumbr/subsample.py` (BM25, R2 and assembly as separate functions that resume from the Drive
   JSONs). Notebook `04_john_subsample_bm25.ipynb` (CPU) writes the BM25 list. Notebook
   `05_john_subsample_r2.ipynb` (GPU) writes the R2 list, then assembles and commits the ID list. Splitting them
   keeps the long BM25 step off paid GPU time. Adds a `subsample` extra (`scikit-learn`, `scipy`,
   `sentence-transformers`).
9. **R2 runs in slices:** documents are embedded in slices of 50K, keeping a running top 50 per question,
   so the full embedding matrix never has to be held at once.

## Left to the builder
Batch sizes, progress output, and how the notebooks split cells.
