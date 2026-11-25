import pytest
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


# ---------------------------------------------------------------- memory-lean loading (4 GB server)

def _fp16_index(tmp_path, n=1000, dim=8, seed=1):
    import json as _json

    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float16)
    ids = [f"dsid_{i:05d}#000" for i in range(n)]
    np.save(tmp_path / "embeddings.npy", m)
    (tmp_path / "ids.json").write_text(_json.dumps(ids), encoding="utf-8")
    return m, ids


def test_load_memory_maps_the_vectors(tmp_path):
    from lumbr_demo.index import DenseIndex
    _fp16_index(tmp_path)
    idx = DenseIndex.load(tmp_path)
    assert isinstance(idx.matrix, np.memmap) and idx.matrix.dtype == np.float16


def test_chunked_fp16_search_equals_fp32_in_memory(tmp_path):
    """fp16 -> fp32 is exact; only the BLAS summation order varies with the chunk shape (last-bit
    differences, as between numpy and the study's FAISS). The ranking must not change."""
    from lumbr_demo.index import DenseIndex
    m, ids = _fp16_index(tmp_path)
    q = np.random.default_rng(2).standard_normal(8).astype(np.float32)
    want = DenseIndex(m.astype(np.float32), ids).search(q, 50)
    for chunk in (7, 333, 1000, 5000):
        got = DenseIndex.load(tmp_path, chunk_rows=chunk).search(q, 50)
        assert [pid for pid, _ in got] == [pid for pid, _ in want], chunk
        assert np.allclose([s for _, s in got], [s for _, s in want], rtol=1e-6, atol=0), chunk


# ---------------------------------------------------------------- FAISS fp16, memory-mapped (B0 server)

def test_faiss_index_matches_numpy_and_is_built_once(tmp_path):
    from lumbr_demo.index import FAISS_FILE, FaissDenseIndex
    m, ids = _fp16_index(tmp_path, n=3000, dim=16)
    q = np.random.default_rng(3).standard_normal(16).astype(np.float32)
    want = DenseIndex(m.astype(np.float32), ids).search(q, 50)
    idx = FaissDenseIndex.load(tmp_path)
    assert (tmp_path / FAISS_FILE).exists()
    got = idx.search(q, 50)
    assert [pid for pid, _ in got] == [pid for pid, _ in want]
    assert np.allclose([s for _, s in got], [s for _, s in want], rtol=1e-5, atol=1e-6)
    built = (tmp_path / FAISS_FILE).stat().st_mtime_ns
    FaissDenseIndex.load(tmp_path)
    assert (tmp_path / FAISS_FILE).stat().st_mtime_ns == built  # reused, not rebuilt


def test_faiss_index_k_larger_than_index(tmp_path):
    from lumbr_demo.index import FaissDenseIndex
    _fp16_index(tmp_path, n=5, dim=4)
    got = FaissDenseIndex.load(tmp_path).search(np.ones(4, np.float32), 10)
    assert len(got) == 5 and len({p for p, _ in got}) == 5


def test_faiss_index_refuses_stale_file(tmp_path):
    """A .faiss file with a different row count than ids.json would silently map scores to wrong ids."""
    import json as _json

    from lumbr_demo.index import FaissDenseIndex
    _fp16_index(tmp_path, n=50, dim=4)
    FaissDenseIndex.load(tmp_path)
    (tmp_path / "ids.json").write_text(_json.dumps([f"x{i}" for i in range(49)]), encoding="utf-8")
    with pytest.raises(ValueError, match="49"):
        FaissDenseIndex.load(tmp_path)
