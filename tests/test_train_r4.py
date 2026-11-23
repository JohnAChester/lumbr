"""DocBatchSampler tests (issue #15): no document appears in two rows of one batch."""

import json

import pytest

from lumbr.train_r4 import doc_batches, load_train


def keys(*docs):
    return frozenset(docs)


def test_every_row_used_once():
    doc_keys = [keys(f"d{i}") for i in range(103)]
    batches = list(doc_batches(doc_keys, batch_size=10, seed=0))
    rows = [i for b in batches for i in b]
    assert sorted(rows) == list(range(103))
    assert all(len(b) <= 10 for b in batches)


def test_no_shared_doc_in_a_batch():
    # rows 0-9 all share doc "x" (e.g. two questions on one passage, or sibling chunks)
    doc_keys = [keys("x", f"n{i}") for i in range(10)] + [keys(f"d{i}") for i in range(40)]
    for seed in range(5):
        for b in doc_batches(doc_keys, batch_size=8, seed=seed):
            seen = [d for i in b for d in doc_keys[i]]
            assert len(seen) == len(set(seen)), b
        assert sorted(i for b in doc_batches(doc_keys, 8, seed) for i in b) == list(range(50))


def test_negative_doc_counts_too():
    # row 1's positive doc is row 0's hard-negative doc → must not share a batch
    doc_keys = [keys("a", "b"), keys("b", "c")]
    assert list(doc_batches(doc_keys, batch_size=2, seed=0)) in ([[0], [1]], [[1], [0]])


def test_new_shuffle_per_seed():
    doc_keys = [keys(f"d{i}") for i in range(50)]
    first = lambda s: next(doc_batches(doc_keys, 10, s))
    assert first(0) != first(1)
    assert first(0) == first(0)


def test_load_train_keeps_full_rows(tmp_path):
    pytest.importorskip("datasets")
    rows = [
        {"question": "q1", "positive_id": "A#000", "positive": "pa",
         "hard_negative_ids": ["B#000", "C#001", "D#002"], "hard_negatives": ["pb", "pc", "pd"]},
        {"question": "q2", "positive_id": "E#000", "positive": "pe",
         "hard_negative_ids": ["F#000"], "hard_negatives": ["pf"]},          # fewer than 3: dropped
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds, doc_keys = load_train(path, query_prefix="P:")
    assert ds.column_names == ["anchor", "positive", "negative_1", "negative_2", "negative_3"]
    assert ds[0]["anchor"] == "P:q1" and ds[0]["negative_3"] == "pd"
    assert doc_keys == [frozenset({"A", "B", "C", "D"})]
