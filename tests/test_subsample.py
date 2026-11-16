"""Subsample selection (#5): BM25 and R2 look-alikes, and assembly. Toy data only (no corpus, GPU or network)."""

import numpy as np
import pandas as pd
import pytest

from lumbr import subsample


# ---------------------------------------------------------------- BM25

def test_bm25_ranks_term_overlap():
    idx = subsample.bm25_index(["gpu kv cache eviction", "lunch menu friday", "kv cache warmup on gpu nodes"])
    assert idx.top_k(["kv cache eviction"], 2)[0] == [0, 2]


def test_bm25_prefers_short_docs_on_equal_tf():
    filler = " ".join(f"word{i}" for i in range(30))
    idx = subsample.bm25_index([f"{filler} canary", "alpha beta canary"])
    assert idx.top_k(["canary"], 2)[0] == [1, 0]


def test_bm25_stopword_only_query_does_not_crash():
    idx = subsample.bm25_index(["gpu kv cache", "lunch menu"])
    assert len(idx.top_k(["the and of"], 2)[0]) <= 2


def test_bm25_returns_only_docs_that_match():
    idx = subsample.bm25_index(["gpu kv cache", "lunch menu"])
    assert idx.top_k(["gpu"], 10)[0] == [0]


# ---------------------------------------------------------------- R2 running top-k

def test_running_topk_matches_full_sort():
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(1000, 16)).astype(np.float32)
    qs = rng.normal(size=(5, 16)).astype(np.float32)
    slices = ((s, docs[s:s + 128]) for s in range(0, 1000, 128))
    got = subsample.running_topk(qs, slices, k=10)
    want = [list(np.argsort(-(docs @ q))[:10]) for q in qs]
    assert got == want


def test_running_topk_k_larger_than_corpus():
    docs = np.eye(5, dtype=np.float32)
    got = subsample.running_topk(np.ones((1, 5), np.float32), [(0, docs[:3]), (3, docs[3:])], k=10)
    assert sorted(got[0]) == [0, 1, 2, 3, 4]


def test_dedupe_keeps_order_and_drops_repeats():
    assert subsample.dedupe(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


# ---------------------------------------------------------------- assembly

def _index(counts):
    rows = [(f"dsid_{src}{i:05d}", src) for src, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["doc_id", "source_type"])


IDX = _index({"a": 9000, "b": 900, "c": 100})
GOLD = {"q1": ["dsid_a00001"], "q2": []}   # q2 is unanswerable
LOOK = {"bm25": {"q1": ["dsid_a00002", "dsid_b00001"], "q2": ["dsid_c00001"]},
        "r2": {"q1": ["dsid_b00001", "dsid_a00003"], "q2": []}}


def _assemble(**kw):
    args = dict(gold=GOLD, lookalikes=LOOK, index=IDX, n_random=100, min_per_source=5, seed=42)
    args.update(kw)
    return subsample.assemble(**args)


def test_includes_all_gold_and_lookalikes():
    ids, _ = _assemble()
    assert {"dsid_a00001", "dsid_a00002", "dsid_a00003", "dsid_b00001", "dsid_c00001"} <= set(ids)
    assert ids == sorted(ids) and len(ids) == len(set(ids))


def test_random_fill_is_proportional_with_minimum():
    _, m = _assemble()
    assert m["random_by_source"] == {"a": 90, "b": 9, "c": 5}   # 90% / 9% / 1% of 100, c raised to the minimum


def test_random_fill_excludes_selected_and_is_deterministic():
    ids1, m = _assemble()
    ids2, _ = _assemble()
    assert ids1 == ids2
    assert m["by_part"]["random"] == 104 and m["by_part"]["total"] == len(ids1) == 5 + 104


def test_ids_missing_from_index_are_dropped_and_reported():
    look = {"bm25": {"q1": ["dsid_zzz"], "q2": []}, "r2": {"q1": [], "q2": []}}
    ids, m = _assemble(lookalikes=look)
    assert "dsid_zzz" not in ids
    assert m["missing_from_corpus"] == ["dsid_zzz"]


def test_small_source_takes_what_exists():
    ids, m = _assemble(index=_index({"a": 9000, "b": 900, "c": 3}), gold={"q1": [], "q2": []},
                       lookalikes={"bm25": {}, "r2": {}})
    assert m["random_by_source"]["c"] == 3


def test_manifest_counts_and_overlap():
    _, m = _assemble()
    assert m["by_part"]["gold"] == 1
    assert m["by_part"]["lookalikes"] == 4          # look-alikes that are not gold
    assert m["lookalike_overlap"] == {"bm25": 3, "r2": 2, "both": 1}
    assert m["per_question"]["q1"] == {"gold": ["dsid_a00001"], "bm25": ["dsid_a00002", "dsid_b00001"],
                                       "r2": ["dsid_b00001", "dsid_a00003"]}
    assert sum(m["by_source"].values()) == m["by_part"]["total"]
