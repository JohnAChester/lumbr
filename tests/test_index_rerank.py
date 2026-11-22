"""Cross-encoder reranking (R6, issue #10): fake cross-encoder, no GPU, no download."""

import json

import numpy as np
import pytest

from lumbr import index_rerank, paths


# ---------------------------------------------------------------- helpers

class _ReverseFakeCrossEncoder:
    """Assigns scores in ascending order: last pair gets the highest score.

    For a run with candidates [p0, p1, p2], predict returns [0.0, 1.0, 2.0].
    After sorting descending: p2 is first, p0 is last — a full reversal.
    """

    def predict(self, pairs, *, batch_size=64, show_progress_bar=False):
        return np.arange(len(pairs), dtype=np.float32)


class _ConstantFakeCrossEncoder:
    """Returns a constant score for every pair (all passages tie)."""

    def predict(self, pairs, *, batch_size=64, show_progress_bar=False):
        return np.ones(len(pairs), dtype=np.float32)


def _run(qid: str, pids: list[str]) -> dict[str, list[tuple[str, float]]]:
    return {qid: [(pid, float(len(pids) - i)) for i, pid in enumerate(pids)]}


def _lookup(pids: list[str]) -> dict[str, str]:
    return {pid: f"text for {pid}" for pid in pids}


def _write_passages(passages: list[dict], path) -> None:
    with open(path, "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")


# ---------------------------------------------------------------- rerank: basic

def test_rerank_returns_top_n_passages():
    pids = [f"p{i}" for i in range(10)]
    run = _run("q1", pids)
    result, _ = index_rerank.rerank(
        run, {"q1": "query"}, _lookup(pids),
        top_n=5, progress=False, _model=_ReverseFakeCrossEncoder(),
    )
    assert len(result["q1"]) == 5


def test_rerank_fewer_than_top_n_returns_all():
    pids = ["p0", "p1", "p2"]
    run = _run("q1", pids)
    result, _ = index_rerank.rerank(
        run, {"q1": "query"}, _lookup(pids),
        top_n=5, progress=False, _model=_ReverseFakeCrossEncoder(),
    )
    assert len(result["q1"]) == 3


def test_rerank_scores_descending():
    pids = [f"p{i}" for i in range(6)]
    run = _run("q1", pids)
    result, _ = index_rerank.rerank(
        run, {"q1": "query"}, _lookup(pids),
        progress=False, _model=_ReverseFakeCrossEncoder(),
    )
    scores = [s for _, s in result["q1"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_changes_order_from_original():
    """_ReverseFakeCrossEncoder flips ranking: original last passage becomes first."""
    pids = ["p0", "p1", "p2"]
    run = _run("q1", pids)
    result, _ = index_rerank.rerank(
        run, {"q1": "query"}, _lookup(pids),
        progress=False, _model=_ReverseFakeCrossEncoder(),
    )
    assert result["q1"][0][0] == "p2"
    assert result["q1"][-1][0] == "p0"


# ---------------------------------------------------------------- rerank: missing queries

def test_rerank_raises_on_missing_query():
    """run contains a qid not present in queries — must raise before any inference."""
    run = {"q1": [("p0", 1.0)], "q_unknown": [("p1", 1.0)]}
    lookup = {"p0": "text", "p1": "text"}
    with pytest.raises(ValueError, match="missing from queries"):
        index_rerank.rerank(
            run, {"q1": "only q1 present"}, lookup,
            progress=False, _model=_ConstantFakeCrossEncoder(),
        )


def test_rerank_missing_query_error_lists_ids():
    """Error message must include the offending question IDs."""
    run = {f"q{i}": [("p0", 1.0)] for i in range(7)}
    lookup = {"p0": "text"}
    with pytest.raises(ValueError) as exc_info:
        index_rerank.rerank(
            run, {"q0": "only one"}, lookup,
            progress=False, _model=_ConstantFakeCrossEncoder(),
        )
    msg = str(exc_info.value)
    assert "6" in msg          # 6 missing out of 7
    assert "missing from queries" in msg


# ---------------------------------------------------------------- rerank: missing passages

def test_rerank_raises_on_missing_passage():
    """Default allow_missing=False: raises ValueError if any pid is absent."""
    run = {"q1": [("p0", 2.0), ("p1", 1.0)]}
    lookup = {"p0": "some text"}  # p1 deliberately missing
    with pytest.raises(ValueError, match="missing from passage_lookup"):
        index_rerank.rerank(
            run, {"q1": "query"}, lookup,
            progress=False, _model=_ConstantFakeCrossEncoder(),
        )


def test_rerank_allow_missing_does_not_raise():
    """allow_missing=True: missing passages get empty string; no exception raised."""
    run = {"q1": [("p0", 2.0), ("p1", 1.0)]}
    lookup = {"p0": "some text"}  # p1 missing
    result, _ = index_rerank.rerank(
        run, {"q1": "query"}, lookup,
        allow_missing=True, progress=False, _model=_ConstantFakeCrossEncoder(),
    )
    pids = {pid for pid, _ in result["q1"]}
    assert "p0" in pids
    assert "p1" in pids


# ---------------------------------------------------------------- rerank: edge cases

def test_rerank_multiple_queries():
    n_q = 3
    run = {
        f"q{i}": [(f"p{i}_{j}", float(3 - j)) for j in range(3)]
        for i in range(n_q)
    }
    queries = {f"q{i}": f"query {i}" for i in range(n_q)}
    lookup = {f"p{i}_{j}": f"text {i} {j}" for i in range(n_q) for j in range(3)}
    result, stats = index_rerank.rerank(
        run, queries, lookup,
        progress=False, _model=_ReverseFakeCrossEncoder(),
    )
    assert set(result.keys()) == {f"q{i}" for i in range(n_q)}
    for qid, hits in result.items():
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------- rerank: latency stats

def test_rerank_latency_stats_keys():
    pids = ["p0", "p1"]
    run = _run("q1", pids)
    _, stats = index_rerank.rerank(
        run, {"q1": "q"}, _lookup(pids),
        progress=False, _model=_ConstantFakeCrossEncoder(),
    )
    assert "median_ms" in stats
    assert "p95_ms" in stats
    assert "n_questions" in stats


def test_rerank_latency_n_questions_matches_run():
    run = {f"q{i}": [(f"p{i}", 1.0)] for i in range(5)}
    queries = {f"q{i}": "q" for i in range(5)}
    lookup = {f"p{i}": "t" for i in range(5)}
    _, stats = index_rerank.rerank(
        run, queries, lookup,
        progress=False, _model=_ConstantFakeCrossEncoder(),
    )
    assert stats["n_questions"] == 5


# ---------------------------------------------------------------- build_passage_lookup

def test_build_passage_lookup_only_fetches_needed(tmp_path):
    passages = [{"passage_id": f"doc{i}#000", "text": f"text {i}"} for i in range(5)]
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    run = {"q1": [("doc0#000", 1.0), ("doc2#000", 0.8)]}
    lookup, missing = index_rerank.build_passage_lookup(run, pfile)

    assert set(lookup.keys()) == {"doc0#000", "doc2#000"}
    assert lookup["doc0#000"] == "text 0"
    assert lookup["doc2#000"] == "text 2"
    assert missing == set()


def test_build_passage_lookup_all_needed_found(tmp_path):
    passages = [{"passage_id": f"doc{i}#000", "text": f"text {i}"} for i in range(5)]
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    run = {"q1": [(f"doc{i}#000", float(5 - i)) for i in range(5)]}
    lookup, missing = index_rerank.build_passage_lookup(run, pfile)

    assert len(lookup) == 5
    assert missing == set()
    for i in range(5):
        assert lookup[f"doc{i}#000"] == f"text {i}"


def test_build_passage_lookup_reports_missing(tmp_path):
    """IDs not present in passages.jsonl appear in the returned missing set."""
    passages = [{"passage_id": f"doc{i}#000", "text": f"text {i}"} for i in range(3)]
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    # Request doc3#000 and doc4#000 which don't exist in the file.
    run = {"q1": [(f"doc{i}#000", 1.0) for i in range(5)]}
    lookup, missing = index_rerank.build_passage_lookup(run, pfile)

    assert len(lookup) == 3
    assert missing == {"doc3#000", "doc4#000"}


# ---------------------------------------------------------------- save_run

def test_save_run_creates_files(tmp_path):
    run = {"q1": [("p0", 0.9), ("p1", 0.8)]}
    stats = {"median_ms": 12.3, "p95_ms": 20.1, "n_questions": 1}
    subdir = index_rerank.save_run(
        run, tmp_path,
        tag="r1_r6", split="dev",
        input_run_dir="r1_bm25_dev_2026-11-01",
        latency_stats=stats,
        date="2026-11-22",
    )
    assert (subdir / "run.json").exists()
    assert (subdir / "meta.json").exists()
    assert subdir.name == "r1_r6_dev_2026-11-22"


def test_save_run_meta_has_required_fields(tmp_path):
    run = {"q1": [("p0", 0.5)]}
    stats = {"median_ms": 5.0, "p95_ms": 8.0, "n_questions": 1}
    subdir = index_rerank.save_run(
        run, tmp_path,
        tag="r3_r6", split="dev",
        input_run_dir="r3_dev_2026-11-21",
        latency_stats=stats,
        date="2026-11-22",
    )
    meta = json.loads((subdir / "meta.json").read_text())
    assert meta["model"] == index_rerank.RERANKER_MODEL
    assert meta["input_run"] == "r3_dev_2026-11-21"
    assert meta["latency_stats"] == stats
    assert meta["split"] == "dev"
    assert "git_commit" in meta
    assert "date" in meta


def test_save_run_roundtrip(tmp_path):
    run = {"q1": [("p0", 0.9), ("p1", 0.7)], "q2": [("p2", 0.5)]}
    stats = {"median_ms": 10.0, "p95_ms": 15.0, "n_questions": 2}
    subdir = index_rerank.save_run(
        run, tmp_path,
        tag="r5_r6", split="dev",
        input_run_dir="r5_rrf_dev_2026-11-22",
        latency_stats=stats,
        date="2026-11-22",
    )
    loaded = json.loads((subdir / "run.json").read_text())
    assert set(loaded.keys()) == {"q1", "q2"}
    assert loaded["q1"][0] == ["p0", 0.9]
    assert len(loaded["q2"]) == 1
