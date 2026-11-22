"""RRF fusion (issue #10): hand-computed scores, edge cases, save_run contract."""

import json
import math

import pytest

from lumbr import index_rrf


# ---------------------------------------------------------------- helpers

def _run(qid: str, pids: list[str]) -> dict[str, list[tuple[str, float]]]:
    """Fake run with descending dummy scores (scores don't matter for fuse())."""
    return {qid: [(pid, float(len(pids) - i)) for i, pid in enumerate(pids)]}


# ---------------------------------------------------------------- fuse: hand-computed

def test_fuse_hand_computed_scores():
    """k=1: p2(5/6) > p3(7/12) > p1(1/2) > p4(1/4)."""
    run_a = {"q1": [("p1", 3.0), ("p2", 2.0), ("p3", 1.0)]}
    run_b = {"q1": [("p2", 3.0), ("p3", 2.0), ("p4", 1.0)]}
    result = index_rrf.fuse([run_a, run_b], k=1)
    pids = [pid for pid, _ in result["q1"]]
    scores = [s for _, s in result["q1"]]

    assert pids == ["p2", "p3", "p1", "p4"]
    assert math.isclose(scores[0], 5 / 6, rel_tol=1e-6)   # 1/2 + 1/3
    assert math.isclose(scores[1], 7 / 12, rel_tol=1e-6)  # 1/4 + 1/3
    assert math.isclose(scores[2], 1 / 2, rel_tol=1e-6)   # 1/2 only
    assert math.isclose(scores[3], 1 / 4, rel_tol=1e-6)   # 1/4 only


def test_fuse_passage_in_one_run_only():
    """p3 at rank 1 in run B beats p2 at rank 2 in run A (k=1)."""
    run_a = {"q1": [("p1", 2.0), ("p2", 1.0)]}
    run_b = {"q1": [("p3", 1.0)]}
    result = index_rrf.fuse([run_a, run_b], k=1)
    pids = [pid for pid, _ in result["q1"]]
    # p3 score = 1/2; p1 score = 1/2; p2 score = 1/3
    # p1 and p3 tie at 1/2; p2 at 1/3
    scores_by_pid = dict(result["q1"])
    assert math.isclose(scores_by_pid["p3"], 0.5, rel_tol=1e-6)
    assert math.isclose(scores_by_pid["p2"], 1 / 3, rel_tol=1e-6)
    assert scores_by_pid["p3"] > scores_by_pid["p2"]


def test_fuse_tied_passages_both_present():
    """Symmetric ranking: p1 and p2 get the same RRF score."""
    run_a = {"q1": [("p1", 2.0), ("p2", 1.0)]}
    run_b = {"q1": [("p2", 2.0), ("p1", 1.0)]}
    result = index_rrf.fuse([run_a, run_b], k=1)
    scores_by_pid = dict(result["q1"])
    assert math.isclose(scores_by_pid["p1"], scores_by_pid["p2"], rel_tol=1e-9)
    assert len(scores_by_pid) == 2


def test_fuse_default_k_60():
    """Default k=60: p1 rank-1 in both runs → 1/61 + 1/61; p2 rank-2 in A only → 1/62."""
    run_a = {"q1": [("p1", 2.0), ("p2", 1.0)]}
    run_b = {"q1": [("p1", 2.0)]}
    result = index_rrf.fuse([run_a, run_b])
    scores_by_pid = dict(result["q1"])
    assert math.isclose(scores_by_pid["p1"], 1 / 61 + 1 / 61, rel_tol=1e-6)
    assert math.isclose(scores_by_pid["p2"], 1 / 62, rel_tol=1e-6)


# ---------------------------------------------------------------- fuse: edge cases

def test_fuse_top_n_truncation():
    """Only the first top_n passages from each run are considered."""
    run_a = {"q1": [(f"p{i}", float(10 - i)) for i in range(10)]}
    run_b = {"q1": []}
    result = index_rrf.fuse([run_a, run_b], k=60, top_n=3)
    pids_seen = {pid for pid, _ in result["q1"]}
    # Passages beyond rank 3 from run_a must be absent.
    assert pids_seen <= {"p0", "p1", "p2"}
    assert len(result["q1"]) == 3


def test_fuse_scores_descending():
    run_a = _run("q1", ["p1", "p2", "p3", "p4", "p5"])
    run_b = _run("q1", ["p3", "p1", "p5", "p2", "p4"])
    result = index_rrf.fuse([run_a, run_b], k=60)
    scores = [s for _, s in result["q1"]]
    assert scores == sorted(scores, reverse=True)


def test_fuse_query_missing_from_one_run():
    """A query absent from one run still produces output from the other."""
    run_a = {"q1": [("p1", 1.0)]}
    run_b = {}  # q1 missing
    result = index_rrf.fuse([run_a, run_b], k=60)
    assert "q1" in result
    assert result["q1"][0][0] == "p1"


def test_fuse_multiple_queries_are_independent():
    run_a = {
        "q1": [("p1", 2.0), ("p2", 1.0)],
        "q2": [("p3", 1.0)],
    }
    run_b = {
        "q1": [("p2", 2.0), ("p1", 1.0)],
        "q2": [("p4", 1.0)],
    }
    result = index_rrf.fuse([run_a, run_b], k=60)
    assert set(result.keys()) == {"q1", "q2"}
    q2_pids = {pid for pid, _ in result["q2"]}
    assert q2_pids == {"p3", "p4"}


# ---------------------------------------------------------------- save_run

def test_save_run_creates_files(tmp_path):
    run = {"q1": [("p1", 0.9), ("p2", 0.8)]}
    subdir = index_rrf.save_run(
        run, tmp_path,
        tag="r5_rrf", split="dev", k_rrf=60,
        input_run_dirs=["r1_bm25_dev_2026-11-01", "r3_qwen_dev_2026-11-21"],
        date="2026-11-22",
    )
    assert (subdir / "run.json").exists()
    assert (subdir / "meta.json").exists()
    assert subdir.name == "r5_rrf_dev_2026-11-22"


def test_save_run_meta_has_required_fields(tmp_path):
    run = {"q1": [("p1", 0.5)]}
    subdir = index_rrf.save_run(
        run, tmp_path,
        k_rrf=30,
        input_run_dirs=["r1_dir", "r3_dir"],
        date="2026-11-22",
    )
    meta = json.loads((subdir / "meta.json").read_text())
    assert meta["k_rrf"] == 30
    assert meta["input_runs"] == ["r1_dir", "r3_dir"]
    assert meta["split"] == "dev"
    assert "git_commit" in meta
    assert "date" in meta


def test_save_run_roundtrip(tmp_path):
    run = {"q1": [("p1", 0.9), ("p2", 0.8)], "q2": [("p3", 0.7)]}
    subdir = index_rrf.save_run(
        run, tmp_path,
        k_rrf=60,
        input_run_dirs=["r1_dir", "r3_dir"],
        date="2026-11-22",
    )
    loaded = json.loads((subdir / "run.json").read_text())
    # JSON round-trip converts tuples to lists
    assert set(loaded.keys()) == {"q1", "q2"}
    assert loaded["q1"][0] == ["p1", 0.9]
    assert len(loaded["q2"]) == 1
