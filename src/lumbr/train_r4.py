"""Fine-tune R4 (bge-base-en-v1.5) on the synthetic questions from #13 (issue #15).

Three pieces, used by notebooks/15_jieun_finetune_r4.ipynb:

  load_train(path)      train.jsonl → (Dataset of anchor/positive/negative_1..3, doc keys per row)
  DocTrainer            SentenceTransformerTrainer whose batches never repeat a document
  DevRecall             evaluator: encode the whole corpus, dev Recall@10 / Recall@50

Why the sampler: MultipleNegativesRankingLoss treats every other row's passages in the
batch as negatives. If two rows share a document (two questions on one passage, or a
sibling chunk), the shared document becomes a false negative, which is what guard 1 in
#13 removed. Spec: docs/superpowers/specs/2026-11-21-hard-negatives-design.md.
"""

from __future__ import annotations

import numpy as np

from lumbr import schema

try:  # needs the [train] extra; the sampler tests only use doc_batches
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.evaluation import SentenceEvaluator
    from sentence_transformers.sampler import DefaultBatchSampler
except ImportError:
    SentenceTransformerTrainer = SentenceEvaluator = DefaultBatchSampler = object

N_NEG = 3


def load_train(path, query_prefix: str, n_neg: int = N_NEG, limit: int | None = None):
    """Read train.jsonl from #13. Keeps rows with exactly n_neg hard negatives.

    Returns (dataset, doc_keys): a datasets.Dataset with columns anchor, positive,
    negative_1..n_neg (the order MNRL expects), and one frozenset of doc_ids per row
    (positive + negatives) for DocBatchSampler.
    """
    from datasets import Dataset

    cols = {"anchor": [], "positive": [], **{f"negative_{i+1}": [] for i in range(n_neg)}}
    doc_keys = []
    for r in schema.read_jsonl(path):
        if len(r["hard_negatives"]) != n_neg:
            continue
        cols["anchor"].append(query_prefix + r["question"])   # same prefix as at search time
        cols["positive"].append(r["positive"])
        for i, text in enumerate(r["hard_negatives"]):
            cols[f"negative_{i+1}"].append(text)
        ids = [r["positive_id"], *r["hard_negative_ids"]]
        doc_keys.append(frozenset(schema.doc_id_of(p) for p in ids))
        if limit and len(doc_keys) >= limit:
            break
    return Dataset.from_dict(cols), doc_keys


def doc_batches(doc_keys: list[frozenset], batch_size: int, seed: int):
    """Shuffle rows, then fill batches greedily, skipping a row whose docs are already in
    the batch (it waits for a later batch). Every row is used exactly once."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(doc_keys)).tolist()
    while order:
        batch, seen, later = [], set(), []
        for i in order:
            if seen & doc_keys[i]:
                later.append(i)
                continue
            batch.append(i)
            seen |= doc_keys[i]
            if len(batch) == batch_size:
                yield batch
                batch, seen = [], set()
        if batch:
            yield batch          # partial batch at the end of a pass
        order = later


class DocBatchSampler(DefaultBatchSampler):
    """doc_batches() as a sentence-transformers batch sampler; new shuffle each epoch."""

    def __init__(self, dataset, batch_size, drop_last, doc_keys, seed=0, **kwargs):
        super().__init__(dataset, batch_size=batch_size, drop_last=drop_last, seed=seed)
        self.doc_keys = doc_keys

    def __iter__(self):
        yield from doc_batches(self.doc_keys, self.batch_size, self.seed + self.epoch)

    def __len__(self):
        return -(-len(self.doc_keys) // self.batch_size)


class DocTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer that batches with DocBatchSampler."""

    def __init__(self, *args, doc_keys, **kwargs):
        self.doc_keys = doc_keys
        super().__init__(*args, **kwargs)

    def get_batch_sampler(self, dataset, batch_size, drop_last, valid_label_columns=None,
                          generator=None, seed=0):
        return DocBatchSampler(dataset, batch_size, drop_last, doc_keys=self.doc_keys, seed=seed)


class DevRecall(SentenceEvaluator):
    """Evaluator for SentenceTransformerTrainer: dev Recall@10 and Recall@50 over the whole corpus.

    Encodes every passage with the current model (bf16 on GPU), searches the dev questions
    (with the query prefix), and scores with eval.evaluate_run, the same function as R1-R3.
    Model selection uses Recall@10 only; the 400 test questions are never touched.
    """

    def __init__(self, passage_ids, passage_texts, question_ids, questions, query_prefix,
                 questions_df, batch_size: int = 256):
        super().__init__()
        self.greater_is_better = True
        self.primary_metric = "dev_recall@10"
        self.passage_ids, self.passage_texts = passage_ids, passage_texts
        self.question_ids, self.questions = question_ids, questions
        self.query_prefix, self.questions_df = query_prefix, questions_df
        self.batch_size = batch_size

    def search(self, model, k: int = 50):
        import torch

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            p = model.encode(self.passage_texts, batch_size=self.batch_size,
                             normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=True)
            q = model.encode([self.query_prefix + t for t in self.questions],
                             normalize_embeddings=True, convert_to_numpy=True)
        scores = q.astype(np.float32) @ p.astype(np.float32).T      # cosine (normalized)
        top = np.argsort(-scores, axis=1)[:, :k]
        return {qid: [(self.passage_ids[j], float(scores[i, j])) for j in top[i]]
                for i, qid in enumerate(self.question_ids)}

    def __call__(self, model, output_path=None, epoch=-1, steps=-1) -> dict:
        from lumbr import eval as ev

        run = self.search(model)
        out = {}
        for k in (10, 50):
            res = ev.evaluate_run(run, tag="r4", split="dev", k=k, questions_df=self.questions_df,
                                  split_ids=self.question_ids, write=False)
            out[f"dev_recall@{k}"] = res["overall"][f"recall@{k}"]
        print(f"[dev] epoch {epoch}: Recall@10 {out['dev_recall@10']:.3f}  "
              f"Recall@50 {out['dev_recall@50']:.3f}")
        return out
