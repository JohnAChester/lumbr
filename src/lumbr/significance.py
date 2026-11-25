"""Paired bootstrap and permutation significance tests for retrieval (issue #19).

Four functions:
  per_question_recall(run, questions_df, test_ids, k=10) -> np.ndarray
      Per-question recall@k sorted deterministically by question_id.

  bootstrap_mean_diff(scores_a, scores_b, *, n_resamples=10000, seed=42)
      Vectorised paired bootstrap CI on mean(A − B).

  permutation_p(scores_a, scores_b, *, n_resamples=10000, seed=42)
      Two-sided sign-flip permutation p-value.

  holm_correction(p_values) -> list[float]
      Holm-Bonferroni step-down adjustment (preserves input order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lumbr import eval as ev


def per_question_recall(
    run: dict[str, list[tuple[str, float]]],
    questions_df: pd.DataFrame,
    test_ids: list[str],
    k: int = 10,
) -> np.ndarray:
    """Per-question recall@k sorted by question_id for deterministic paired alignment.

    recall@k per question = (# gold docs in top-k) / (# gold docs).
    Only questions in test_ids with at least one gold doc are included —
    matches the 376 scorable test questions in ERB.

    Returns:
        1-D float64 array of length = scorable questions in test_ids, sorted by qid.
    """
    test_set = set(test_ids)
    qs = questions_df[
        questions_df["question_id"].isin(test_set)
        & questions_df["expected_doc_ids"].apply(len).gt(0)
    ].sort_values("question_id").reset_index(drop=True)

    run_docs = ev.passages_to_docs(run)

    scores: list[float] = []
    for _, row in qs.iterrows():
        qid  = str(row["question_id"])
        gold = set(row["expected_doc_ids"])
        top_k = {
            d for d, _ in
            sorted(run_docs.get(qid, {}).items(), key=lambda x: -x[1])[:k]
        }
        scores.append(len(gold & top_k) / len(gold))

    return np.array(scores, dtype=np.float64)


def bootstrap_mean_diff(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Vectorised paired bootstrap CI on mean difference (A − B).

    Draws n_resamples × n question-index arrays with replacement and
    computes the mean paired difference for each resample.

    Args:
        scores_a, scores_b: equal-length per-question score arrays (sorted
            by question_id so they align element-wise).
        n_resamples: number of bootstrap draws (default 10,000).
        seed: RNG seed for reproducibility.

    Returns:
        (mean_diff, ci_lower_2.5pct, ci_upper_97.5pct)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length "
            f"({len(scores_a)} vs {len(scores_b)})"
        )
    n    = len(scores_a)
    diff = scores_a - scores_b
    obs  = float(diff.mean())

    rng        = np.random.default_rng(seed)
    idx        = rng.integers(0, n, (n_resamples, n))
    boot_means = diff[idx].mean(axis=1)

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return obs, ci_lo, ci_hi


def permutation_p(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired sign-flip permutation p-value.

    For each resample, independently flips the sign of each per-question
    difference (±1 Rademacher variables), then measures the mean.
    p-value = proportion of resamples where |perm_mean| ≥ |obs_mean| − eps.

    Args:
        scores_a, scores_b: equal-length per-question score arrays.
        n_resamples: number of sign-flip draws (default 10,000).
        seed: RNG seed for reproducibility.

    Returns:
        p-value in [0, 1]
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length "
            f"({len(scores_a)} vs {len(scores_b)})"
        )
    n    = len(scores_a)
    diff = scores_a - scores_b
    obs  = float(diff.mean())

    rng        = np.random.default_rng(seed)
    signs      = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    perm_means = (signs * diff).mean(axis=1)

    return float((np.abs(perm_means) >= abs(obs) - 1e-10).mean())


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction for multiple testing.

    Preserves the input order in the output.

    Algorithm:
      1. Sort p-values ascending.
      2. adj_p_(i) = p_(i) × (m − rank_0based)   (rank is 0-indexed)
      3. Take running maximum so monotonicity holds.
      4. Clip to 1.0.

    Args:
        p_values: raw p-values in any order.

    Returns:
        Holm-adjusted p-values in the same order as input, clipped to [0, 1].
    """
    n       = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj         = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted
