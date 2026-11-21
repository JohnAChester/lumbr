"""Retrieval evaluation: Recall@k, nDCG@k, MRR@k, invalid-extra-docs (issue #7).

evaluate_run(run, tag="r2", split="dev") is the main entry point.

A "run" maps question_id -> [(passage_id, score), ...] sorted descending by score.
Passage IDs are converted to doc IDs before scoring: a document counts as retrieved
if any of its passages appears in the top k; the highest passage score is used.

Two categories are excluded from all retrieval metrics because retrieval success
cannot be judged from retrieved documents alone:
  high_level       — questions that require synthesising the whole corpus
  info_not_found   — questions whose answer is explicitly absent from the corpus
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from lumbr import data, paths, schema

# Categories excluded from every retrieval metric.
EXCLUDED_CATEGORIES: frozenset[str] = frozenset({"high_level", "info_not_found"})

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
    path = _SPLITS_DIR / f"{split}_question_ids.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run notebook 06 to generate and commit the split first."
        )
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]


def _eval_subset(
    qrels_dict: dict[str, dict[str, int]],
    run_dict: dict[str, dict[str, float]],
    metrics: list[str],
) -> dict[str, float]:
    """Run ranx.evaluate on the given subset; return a plain Python float dict."""
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
        k: cutoff for Recall, nDCG, and MRR
        questions_df: pre-loaded questions DataFrame; loads from Drive if None
        split_ids: question IDs for this split; reads splits/{split}_question_ids.txt if None
        indexed_doc_ids: doc IDs with at least one passage; used to count gold docs
            that have no passages in the index (invalid_extra_docs). Pass None to skip.
        runs_dir: directory where the JSON result file is written
        write: if False, skip writing to disk (useful in tests)

    Returns:
        {
            "tag": str, "split": str, "k": int, "n_questions": int,
            "overall": {
                "recall@k": float, "ndcg@k": float, "mrr@k": float,
                "invalid_extra_docs": float,  # mean gold docs absent from index
            },
            "by_category": {
                category: {
                    "recall@k": float, "ndcg@k": float, "mrr@k": float,
                    "invalid_extra_docs": float, "n": int,
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

    qs = questions_df[
        questions_df["question_id"].isin(split_set)
        & ~questions_df["question_type"].isin(EXCLUDED_CATEGORIES)
    ].copy()

    run_docs = passages_to_docs(run)

    # Build qrels and per-question invalid_extra_docs count.
    qrels_dict: dict[str, dict[str, int]] = {}
    invalid: dict[str, int] = {}
    for _, row in qs.iterrows():
        qid = str(row["question_id"])
        gold = set(row["expected_doc_ids"])
        qrels_dict[qid] = {did: 1 for did in gold}
        invalid[qid] = (
            len(gold - indexed_doc_ids) if indexed_doc_ids is not None else 0
        )

    run_dict: dict[str, dict[str, float]] = {
        qid: run_docs.get(qid, {}) for qid in qrels_dict
    }

    # Overall metrics.
    overall = _eval_subset(qrels_dict, run_dict, metrics)
    overall["invalid_extra_docs"] = (
        sum(invalid.values()) / max(len(invalid), 1)
    )

    # Per-category metrics.
    by_category: dict[str, dict] = {}
    for cat, grp in qs.groupby("question_type"):
        qids = set(str(q) for q in grp["question_id"])
        cat_qrels = {q: qrels_dict[q] for q in qids}
        cat_run = {q: run_dict.get(q, {}) for q in qids}
        cat_scores = _eval_subset(cat_qrels, cat_run, metrics)
        cat_scores["n"] = len(qids)
        cat_scores["invalid_extra_docs"] = (
            sum(invalid.get(q, 0) for q in qids) / len(qids)
        )
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
