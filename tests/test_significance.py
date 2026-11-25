"""Unit tests for src/lumbr/significance.py."""

import numpy as np
import pytest

from lumbr.significance import bootstrap_mean_diff, holm_correction, permutation_p


def test_bootstrap_identical_runs_zero_diff():
    scores = np.array([0.5] * 20)
    mean_diff, ci_lo, ci_hi = bootstrap_mean_diff(scores, scores)
    assert mean_diff == pytest.approx(0.0)
    assert ci_lo    == pytest.approx(0.0)
    assert ci_hi    == pytest.approx(0.0)


def test_permutation_identical_runs_p_one():
    scores = np.array([0.5] * 20)
    p = permutation_p(scores, scores)
    assert p == pytest.approx(1.0)


def test_bootstrap_clear_difference_ci_excludes_zero():
    a = np.ones(50)
    b = np.zeros(50)
    mean_diff, ci_lo, ci_hi = bootstrap_mean_diff(a, b)
    assert ci_lo > 0.0, f"CI lower bound should be > 0 but got {ci_lo}"
    assert mean_diff == pytest.approx(1.0)


def test_permutation_clear_difference_small_p():
    a = np.ones(50)
    b = np.zeros(50)
    p = permutation_p(a, b)
    assert p < 0.001, f"Expected p < 0.001 for a clear difference but got {p}"


def test_holm_correction_basic():
    # 3 raw p-values: [0.04, 0.01, 0.05]
    # Sorted ascending: 0.01 (idx 1), 0.04 (idx 0), 0.05 (idx 2)
    # rank 0: adj = 0.01 * 3 = 0.03; running_max = 0.03  → adjusted[1] = 0.03
    # rank 1: adj = 0.04 * 2 = 0.08; running_max = 0.08  → adjusted[0] = 0.08
    # rank 2: adj = 0.05 * 1 = 0.05; running_max = 0.08  → adjusted[2] = 0.08
    adj = holm_correction([0.04, 0.01, 0.05])
    assert adj[0] == pytest.approx(0.08)
    assert adj[1] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.08)
