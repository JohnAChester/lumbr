"""Retrieval evaluation: Recall@k, nDCG@k, MRR@k, extra_docs, gold_unindexed (issue #7).

evaluate_run(run, tag="r2", split="dev") is the main entry point.

A "run" maps question_id -> [(passage_id, score), ...] sorted descending by score.
Passage IDs are converted to doc IDs before scoring: a document counts as retrieved
if any of its passages appears; the highest passage score is used.

**Which questions are scored**
The official ERB evaluator scores questions that have at least one expected_doc_id
(470 of 500). In practice this means high_level (10) and info_not_found (20) are
excluded — confirmed from questions.parquet: those two categories have empty
expected_doc_ids lists; all other eight categories have at least one gold doc.
Dev after exclusion: 94; test: 376.

**extra_docs_at_5 / extra_docs_at_10**
Upper bound on the benchmark's "invalid extra documents" metric (Sec. 5,
arXiv 2605.05253). The paper defines invalid extras as retrieved docs that are
neither gold nor judged "valid but not required" by an LLM judge. Without that
judge, we count *all* non-gold docs in the top-k, so our value is an upper bound.
Computed at k=5 (what the RAG generator sees) and k=10.

**gold_unindexed** (sanity check only)
Mean number of gold doc IDs per question that have no passage in the index —
a theoretical Recall ceiling. Included in results only when indexed_doc_ids is
provided; 0.0 otherwise.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from lumbr import data, paths, schema

_SPLITS_DIR = paths.REPO_ROOT / "splits"


# ---------------------------------------------------------------- passage → doc

def passages_to_docs(
    run: dict[str, list[tuple[str, float]]],
) -> dict[str, dict[str, float]]:
    """Map a passage-level run to doc-level (max passage score per doc per query)."""
    out: dict[str, dict[str, float]] = {}
    for qid, hits in run.items():
        doc_scores: dict[str, float] = {}
        for pid, score in hits:
            did = schema.doc_id_of(pid)
            if did not in doc_scores or score > doc_scores[did]:
                doc_scores[did] = score
        out[qid] = doc_scores
    return out


# ---------------------------------------------------------------- helpers

def _load_split_ids(split: str) -> list[str]:
    splits_dir = Path(os.environ.get("LUMBR_SPLITS_DIR", _SPLITS_DIR))
    path = splits_dir / f"{split}_question_ids.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run notebook 06 to generate and commit the split first, "
            "or set LUMBR_SPLITS_DIR to a directory containing the split files."
        )
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]


def _mean_extra_docs(
    qids: Iterable[str],
    qrels_dict: dict[str, dict[str, int]],
    run_docs: dict[str, dict[str, float]],
    k: int,
) -> float:
    """Mean count of top-k retrieved docs not in the gold set (upper-bound noise metric)."""
    total = 0
    n = 0
    for qid in qids:
        gold = set(qrels_dict[qid])
        top_k = {d for d, _ in sorted(run_docs.get(qid, {}).items(), key=lambda x: -x[1])[:k]}
        total += len(top_k - gold)
        n += 1
    return total / max(n, 1)


def _eval_subset(
    qrels_dict: dict[str, dict[str, int]],
    run_dict: dict[str, dict[str, float]],
    metrics: list[str],
) -> dict[str, float]:
    """Run ranx.evaluate on a subset; return plain Python floats."""
    from ranx import Qrels, Run
    from ranx import evaluate as rx_eval

    scores = rx_eval(Qrels(qrels_dict), Run(run_dict), metrics)
    return {k: float(v) for k, v in scores.items()}


# ---------------------------------------------------------------- main

def evaluate_run(
    run: dict[str, list[tuple[str, float]]],
    *,
    tag: str = "",
    split: str = "dev",
    k: int = 10,
    questions_df: pd.DataFrame | None = None,
    split_ids: Sequence[str] | None = None,
    indexed_doc_ids: set[str] | None = None,
    runs_dir: Path = paths.RUNS,
    write: bool = True,
) -> dict:
    """Evaluate a passage-level retrieval run against benchmark gold doc IDs.

    Args:
        run: {question_id: [(passage_id, score), ...]}
        tag: short retriever label for the output file, e.g. "r2" or "bm25"
        split: "dev" or "test"
        k: cutoff for Recall, nDCG, and MRR (extra_docs is always at 5 and 10)
        questions_df: pre-loaded questions DataFrame; loads from Drive if None
        split_ids: question IDs for this split; reads splits/{split}_question_ids.txt if None
        indexed_doc_ids: doc IDs with at least one passage; populates gold_unindexed
            sanity check when provided, otherwise gold_unindexed is 0.0
        runs_dir: directory where the JSON result file is written
        write: if False, skip writing to disk (useful in tests)

    Returns:
        {
            "tag": str, "split": str, "k": int, "n_questions": int,
            "overall": {
                "recall@k": float, "ndcg@k": float, "mrr@k": float,
                "extra_docs_at_5": float,   # upper-bound noise, see module docstring
                "extra_docs_at_10": float,
                "gold_unindexed": float,    # sanity check: 0.0 if indexed_doc_ids not given
            },
            "by_category": {
                category: {
                    "recall@k": float, "ndcg@k": float, "mrr@k": float,
                    "extra_docs_at_5": float, "extra_docs_at_10": float,
                    "gold_unindexed": float, "n": int,
                }, ...
            },
        }
    """
    if questions_df is None:
        questions_df = data.load_questions()
    if split_ids is None:
        split_ids = _load_split_ids(split)

    split_set = set(split_ids)
    metrics = [f"recall@{k}", f"ndcg@{k}", f"mrr@{k}"]

    # Score only questions with at least one gold doc (official ERB criterion).
    qs = questions_df[
        questions_df["question_id"].isin(split_set)
        & questions_df["expected_doc_ids"].apply(len).gt(0)
    ].copy()

    run_docs = passages_to_docs(run)

    qrels_dict: dict[str, dict[str, int]] = {}
    gold_unindexed_per_q: dict[str, int] = {}
    for _, row in qs.iterrows():
        qid = str(row["question_id"])
        gold = set(row["expected_doc_ids"])
        qrels_dict[qid] = {did: 1 for did in gold}
        gold_unindexed_per_q[qid] = (
            len(gold - indexed_doc_ids) if indexed_doc_ids is not None else 0
        )

    run_dict: dict[str, dict[str, float]] = {
        qid: run_docs.get(qid, {}) for qid in qrels_dict
    }

    overall = _eval_subset(qrels_dict, run_dict, metrics)
    qids_all = list(qrels_dict.keys())
    overall["extra_docs_at_5"] = _mean_extra_docs(qids_all, qrels_dict, run_docs, 5)
    overall["extra_docs_at_10"] = _mean_extra_docs(qids_all, qrels_dict, run_docs, 10)
    overall["gold_unindexed"] = (
        sum(gold_unindexed_per_q.values()) / max(len(gold_unindexed_per_q), 1)
    )

    by_category: dict[str, dict] = {}
    for cat, grp in qs.groupby("question_type"):
        qids = [str(q) for q in grp["question_id"]]
        cat_qrels = {q: qrels_dict[q] for q in qids}
        cat_run = {q: run_dict.get(q, {}) for q in qids}
        cat_scores = _eval_subset(cat_qrels, cat_run, metrics)
        cat_scores["extra_docs_at_5"] = _mean_extra_docs(qids, qrels_dict, run_docs, 5)
        cat_scores["extra_docs_at_10"] = _mean_extra_docs(qids, qrels_dict, run_docs, 10)
        cat_scores["gold_unindexed"] = (
            sum(gold_unindexed_per_q.get(q, 0) for q in qids) / len(qids)
        )
        cat_scores["n"] = len(qids)
        by_category[str(cat)] = cat_scores

    result: dict = {
        "tag": tag,
        "split": split,
        "k": k,
        "n_questions": len(qs),
        "overall": overall,
        "by_category": by_category,
    }

    if write:
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{tag or 'run'}_{split}_at{k}.json"
        (runs_dir / fname).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[eval] {runs_dir / fname}")

    return result
