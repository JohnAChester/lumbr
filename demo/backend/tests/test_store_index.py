import numpy as np

from lumbr_demo.encoder import HashEncoder
from lumbr_demo.index import DenseIndex


def test_store_counts_rows(store, passages):
    assert store.count() == len(passages)


def test_store_get_omits_unknown(store):
    got = store.get(["dsid_aaa#000", "dsid_nope#000", "garbage"])
    assert list(got) == ["dsid_aaa#000"]
    assert got["dsid_aaa#000"]["title"] == "retention-policy"


def test_with_neighbours_includes_adjacent_chunks(store):
    assert sorted(store.with_neighbours(["dsid_aaa#001"])) == ["dsid_aaa#000", "dsid_aaa#001", "dsid_aaa#002"]
    assert sorted(store.with_neighbours(["dsid_aaa#000"])) == ["dsid_aaa#000", "dsid_aaa#001"]
    assert sorted(store.with_neighbours(["dsid_aaa#000"], neighbours=0)) == ["dsid_aaa#000"]


def test_with_neighbours_skips_unknown_hits(store):
    assert store.with_neighbours(["dsid_nope#000"]) == {}


def test_index_search_orders_by_cosine():
    m = np.eye(3, dtype=np.float32)
    idx = DenseIndex(m, ["a", "b", "c"])
    hits = idx.search(np.array([0, 1, 0], dtype=np.float32), k=2)
    assert hits[0][0] == "b" and abs(hits[0][1] - 1.0) < 1e-6
    assert len(hits) == 2


def test_index_save_load_roundtrip(tmp_path):
    m = np.eye(3, dtype=np.float32)
    DenseIndex(m, ["a", "b", "c"]).save(tmp_path)
    idx = DenseIndex.load(tmp_path)
    assert idx.ids == ["a", "b", "c"] and idx.matrix.dtype == np.float32


def test_hash_encoder_unit_norm_and_deterministic():
    enc = HashEncoder()
    a, b = enc.encode("retention policy"), enc.encode("retention policy")
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1.0) < 1e-6 and a.dtype == np.float32
