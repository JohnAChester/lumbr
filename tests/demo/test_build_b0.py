"""Full B0 artifacts from the study's own outputs: R4 shards, the R1 Lucene index, passages (spec b0-backend)."""
import json

import numpy as np
import pytest

import build_b0_artifacts as b0
import check_parity as cp


def _shard(d, no, ids, dim=4, seed=0):
    m = np.random.default_rng(seed + no).standard_normal((len(ids), dim)).astype(np.float16)
    np.save(d / f"emb_{no:04d}.npy", m)
    (d / f"emb_{no:04d}.ids").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return m


def test_concat_shards_keeps_shard_order_and_row_order(tmp_path):
    a = _shard(tmp_path, 0, ["p1", "p2"])
    b = _shard(tmp_path, 1, ["p3"])
    m, ids = b0.concat_shards(tmp_path)
    assert ids == ["p1", "p2", "p3"]
    assert m.dtype == np.float16 and np.array_equal(m, np.vstack([a, b]))


def test_concat_shards_refuses_unfinished_shard(tmp_path):
    _shard(tmp_path, 0, ["p1"])
    np.save(tmp_path / "emb_0001.npy", np.zeros((1, 4), np.float16))  # no .ids sidecar: never finished
    with pytest.raises(ValueError, match="emb_0001"):
        b0.concat_shards(tmp_path)


def test_concat_shards_refuses_row_count_mismatch(tmp_path):
    np.save(tmp_path / "emb_0000.npy", np.zeros((2, 4), np.float16))
    (tmp_path / "emb_0000.ids").write_text("p1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2 rows"):
        b0.concat_shards(tmp_path)


def test_concat_shards_refuses_duplicate_ids(tmp_path):
    _shard(tmp_path, 0, ["p1"])
    _shard(tmp_path, 1, ["p1"])
    with pytest.raises(ValueError, match="duplicate"):
        b0.concat_shards(tmp_path)


def test_write_dense_round_trips_through_the_service_index(tmp_path):
    from lumbr_demo.index import DenseIndex
    m = np.eye(3, dtype=np.float16)
    b0.write_dense(m, ["a", "b", "c"], tmp_path)
    assert np.load(tmp_path / "embeddings.npy").dtype == np.float16  # half the download
    idx = DenseIndex.load(tmp_path)
    assert idx.matrix.dtype == np.float16 and idx.search(np.array([0, 1, 0], np.float32), 1)[0][0] == "b"


def test_check_ids_match_store(tmp_path):
    b0.check_same_ids(["a", "b"], ["b", "a"])
    with pytest.raises(ValueError, match="1 in the index only"):
        b0.check_same_ids(["a", "b", "c"], ["a", "b"])


def test_copy_lucene_needs_a_finished_index(tmp_path):
    src, dst = tmp_path / "r1", tmp_path / "out" / "bm25"
    src.mkdir()
    with pytest.raises(FileNotFoundError, match="segments"):
        b0.copy_lucene(src, dst)
    (src / "segments_1").write_bytes(b"x")
    (src / "_0.cfs").write_bytes(b"y")
    b0.copy_lucene(src, dst)
    assert sorted(p.name for p in dst.iterdir()) == ["_0.cfs", "segments_1"]


def test_b0_config():
    c = b0.b0_config(n_passages=10, n_docs=3, model_path="/m/r4_full/final", r4_meta={"x": 1})
    assert c["retriever"] == "B0" and c["lexical"] == "bm25" and c["threshold"] is None
    assert c["query_prefix"] == "Represent this sentence for searching relevant passages:"
    assert c["fusion"] == {"method": "rrf", "k": 60, "first_stage_k": 50, "order": ["r1", "r4"]}
    json.dumps(c)


# ---------------------------------------------------------------- parity with the study's saved run

def test_compare_top_k():
    served = {"q1": [("a", 1), ("b", 1), ("c", 1)], "q2": [("x", 1), ("y", 1)]}
    study = {"q1": [("a", 9), ("b", 8), ("c", 7)], "q2": [("y", 9), ("z", 8)], "q3": [("k", 1)]}
    got = cp.compare_top_k(served, study, k=2)
    assert got == {"n": 2, "identical": 1, "same_set": 1, "mean_overlap": pytest.approx(0.75),
                   "missing": ["q3"], "differ": ["q2"]}


def test_memory_report_splits_anonymous_from_file_backed():
    status = ("Name:\tpython\nVmHWM:\t 3145728 kB\nVmRSS:\t 2097152 kB\n"
              "RssAnon:\t 1048576 kB\nRssFile:\t 1048576 kB\nRssShmem:\t 0 kB\n")
    assert cp.memory_report(status) == {"peak_rss_mb": 3072, "rss_mb": 2048, "anon_mb": 1024, "file_mb": 1024}


def test_memory_report_without_proc():
    assert cp.memory_report(None) == {}
