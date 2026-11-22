"""Retrieval error-analysis helpers (issue #10).

Three entry points:
  category_recall   — per-category recall@k comparison table (DataFrame)
  semantic_analysis — per-question breakdown for one question category
  fusion_overlap    — exclusive-hit counts for R5 fusion upper-bound estimate
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from lumbr import eval as ev


# ---------------------------------------------------------------- internal helpers

def _top_k_docs(run_docs: dict[str, dict[str, float]], qid: str, k: int) -> set[str]:
    return {d for d, _ in sorted(run_docs.get(qid, {}).items(), key=lambda x: -x[1])[:k]}


def _scored_docs(run_docs: dict[str, dict[str, float]], qid: str, k: int) -> list[tuple[str, float]]:
    return sorted(run_docs.get(qid, {}).items(), key=lambda x: -x[1])[:k]


def _evaluable_qs(questions_df: pd.DataFrame, split_ids: Sequence[str]) -> pd.DataFrame:
    split_set = set(split_ids)
    return questions_df[
        questions_df["question_id"].isin(split_set)
        & questions_df["expected_doc_ids"].apply(len).gt(0)
    ]


# ---------------------------------------------------------------- public API

def category_recall(
    runs: dict[str, dict[str, list[tuple[str, float]]]],
    questions_df: pd.DataFrame,
    split_ids: Sequence[str],
    ks: tuple[int, ...] = (10, 50),
) -> pd.DataFrame:
    """Per-category recall@k for each run. Returns a tidy DataFrame.

    Args:
        runs: {label: {question_id: [(passage_id, score), ...]}}
        ks:   cutoffs to compute (default: 10 and 50)

    Returns DataFrame with columns: category, n, {label}_hit@{k}, ...
    """
    qs = _evaluable_qs(questions_df, split_ids)
    run_docs = {label: ev.passages_to_docs(run) for label, run in runs.items()}

    def _recall_row(subset: pd.DataFrame) -> dict:
        row: dict = {"n": len(subset)}
        for k in ks:
            for label, rdocs in run_docs.items():
                hits = sum(
                    1 for _, qrow in subset.iterrows()
                    if set(qrow["expected_doc_ids"]) & _top_k_docs(rdocs, str(qrow["question_id"]), k)
                )
                row[f"{label}_hit@{k}"] = round(hits / max(len(subset), 1), 4)
        return row

    rows = []
    for cat, grp in qs.groupby("question_type"):
        rows.append({"category": str(cat), **_recall_row(grp)})
    rows.append({"category": "OVERALL", **_recall_row(qs)})
    return pd.DataFrame(rows)


def rank_gold(
    run_docs: dict[str, dict[str, float]],
    qid: str,
    gold: set[str],
    k: int = 50,
) -> dict[str, int | None]:
    """Return {doc_id: 1-based rank} for each gold doc; None if not in top-k."""
    ranked = _scored_docs(run_docs, qid, k)
    id_to_rank = {did: i + 1 for i, (did, _) in enumerate(ranked)}
    return {did: id_to_rank.get(did) for did in gold}


def semantic_analysis(
    r1_run: dict[str, list[tuple[str, float]]],
    r2_run: dict[str, list[tuple[str, float]]],
    questions_df: pd.DataFrame,
    split_ids: Sequence[str],
    doc_titles: dict[str, str],
    *,
    category: str = "semantic",
    k_rank: int = 50,
) -> list[dict]:
    """For each question in `category`, return a comparison dict.

    Each dict contains:
      question_id, question, gold_doc_ids,
      r1_ranks {doc_id: rank|None}, r2_ranks {doc_id: rank|None},
      r1_hit_at_10 (bool), r2_hit_at_10 (bool),
      r2_top3: [{doc_id, passage_id, score, title, is_gold}, ...]
    """
    qs = _evaluable_qs(questions_df, split_ids)
    qs = qs[qs["question_type"] == category]

    r1_docs = ev.passages_to_docs(r1_run)
    r2_docs = ev.passages_to_docs(r2_run)

    out = []
    for _, qrow in qs.iterrows():
        qid  = str(qrow["question_id"])
        gold = set(qrow["expected_doc_ids"])

        r1_ranks = rank_gold(r1_docs, qid, gold, k_rank)
        r2_ranks = rank_gold(r2_docs, qid, gold, k_rank)

        r2_top3 = []
        for pid, score in r2_run.get(qid, [])[:3]:
            did = pid.split("#", 1)[0]
            r2_top3.append({
                "doc_id":     did,
                "passage_id": pid,
                "score":      round(score, 4),
                "title":      doc_titles.get(did, "(no title)"),
                "is_gold":    did in gold,
            })

        out.append({
            "question_id":  qid,
            "question":     qrow["question"],
            "gold_doc_ids": sorted(gold),
            "r1_ranks":     r1_ranks,
            "r2_ranks":     r2_ranks,
            "r1_hit_at_10": any(v is not None and v <= 10 for v in r1_ranks.values()),
            "r2_hit_at_10": any(v is not None and v <= 10 for v in r2_ranks.values()),
            "r2_top3":      r2_top3,
        })
    return out


def fusion_overlap(
    run_a: dict[str, list[tuple[str, float]]],
    run_b: dict[str, list[tuple[str, float]]],
    questions_df: pd.DataFrame,
    split_ids: Sequence[str],
    *,
    k: int = 10,
    label_a: str = "r1",
    label_b: str = "r2",
) -> dict:
    """Count questions where A@k hits gold but B misses, and vice versa.

    Returns:
        {
          "both":          int,  # gold in top-k for both
          "{a}_only":      int,  # gold in A@k, not B@k → fusion gain from B
          "{b}_only":      int,  # gold in B@k, not A@k → fusion gain from A
          "neither":       int,
          "total":         int,
          "oracle_hit@{k}": float,  # (both + a_only + b_only) / total
          "by_category":   {cat: same dict (without by_category), ...}
        }
    """
    qs = _evaluable_qs(questions_df, split_ids)
    a_docs = ev.passages_to_docs(run_a)
    b_docs = ev.passages_to_docs(run_b)

    def _counts(subset: pd.DataFrame) -> dict:
        both = a_only = b_only = neither = 0
        for _, qrow in subset.iterrows():
            qid  = str(qrow["question_id"])
            gold = set(qrow["expected_doc_ids"])
            a_hit = bool(gold & _top_k_docs(a_docs, qid, k))
            b_hit = bool(gold & _top_k_docs(b_docs, qid, k))
            if   a_hit and b_hit:  both    += 1
            elif a_hit:            a_only  += 1
            elif b_hit:            b_only  += 1
            else:                  neither += 1
        n = max(len(subset), 1)
        return {
            "both":                   both,
            f"{label_a}_only":        a_only,
            f"{label_b}_only":        b_only,
            "neither":                neither,
            "total":                  len(subset),
            f"oracle_hit@{k}":         round((both + a_only + b_only) / n, 4),
        }

    result = _counts(qs)
    result["by_category"] = {
        str(cat): _counts(grp) for cat, grp in qs.groupby("question_type")
    }
    return result
