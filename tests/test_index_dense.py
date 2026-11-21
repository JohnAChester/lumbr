"""Dense embedding and FAISS index (#10): toy corpus, no GPU, no network required.

All tests inject _FakeModel so no sentence-transformers model is downloaded.
faiss-cpu must be installed (`pip install -e ".[retrieval]"`).
"""

import hashlib
import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lumbr import index_dense, schema


# ---------------------------------------------------------------- helpers

class _FakeModel:
    """Deterministic per-text normalized embedder; no GPU, no network."""

    dim = 32

    def encode(self, texts, *, batch_size=256, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False):
        embs = []
        for t in texts:
            # Stable hash so the same text always gets the same vector.
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            e = rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(e)
            embs.append(e / (norm if norm > 0 else 1.0))
        return np.array(embs, dtype=np.float32)


def _make_passages(n: int) -> list[dict]:
    passages = []
    for i in range(n):
        pid    = schema.make_passage_id(f"dsid_{i:032x}", 0)
        header = f"[slack] t{i}"
        body   = f"body {i} " * (1 + i % 10)
        passages.append({
            "passage_id": pid,
            "doc_id": f"dsid_{i:032x}",
            "source": "slack",
            "title": f"t{i}",
            "channel": None, "author": None, "date": None, "created": None,
            "chunk_idx": 0, "n_chunks": 1,
            "n_tokens": 10 + i % 50,   # varying lengths for the sort test
            "header": header, "body": body,
            "text": schema.join_text(header, body),
        })
    return passages


def _write_passages(passages: list[dict], path) -> None:
    with open(path, "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")


def _embed(tmp_path, passages, shard_size=50, n=None):
    """Convenience: write passages to a JSONL, run embed_passages, return index_root."""
    if n is not None:
        passages = passages[:n]
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)
    iroot = tmp_path / "indexes"
    lroot = tmp_path / "local"
    index_dense.embed_passages(
        pfile, index_dense.R2,
        index_root=iroot, local_root=lroot,
        shard_size=shard_size, batch_size=32, progress=False,
        _model=_FakeModel(),
    )
    return pfile, iroot, lroot


# ---------------------------------------------------------------- embed_passages: basic

def test_embed_produces_correct_passage_count(tmp_path):
    passages = _make_passages(120)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    shard_dir = iroot / "r2" / index_dense.EMB_SHARD_DIR
    assert len(index_dense._done_ids(shard_dir)) == 120


def test_embed_creates_right_number_of_shards(tmp_path):
    passages = _make_passages(120)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    shard_dir = iroot / "r2" / index_dense.EMB_SHARD_DIR
    assert len(index_dense._emb_shards(shard_dir)) == 3   # ceil(120 / 50)


def test_embed_return_value_equals_total_passages(tmp_path):
    passages = _make_passages(70)
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)
    iroot, lroot = tmp_path / "indexes", tmp_path / "local"
    n = index_dense.embed_passages(
        pfile, index_dense.R2,
        index_root=iroot, local_root=lroot,
        shard_size=50, batch_size=32, progress=False,
        _model=_FakeModel(),
    )
    assert n == 70


# ---------------------------------------------------------------- sort by length

def test_passages_within_shard_sorted_by_length_descending(tmp_path):
    passages = _make_passages(50)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    shard_dir = iroot / "r2" / index_dense.EMB_SHARD_DIR
    ids_file = shard_dir / "emb_0000.ids"
    ids = [ln for ln in ids_file.read_text().splitlines() if ln]
    p_by_id = {p["passage_id"]: p for p in passages}
    tokens = [p_by_id[pid]["n_tokens"] for pid in ids]
    assert tokens == sorted(tokens, reverse=True)


# ---------------------------------------------------------------- resumability

def test_resume_after_partial_crash(tmp_path):
    passages = _make_passages(120)
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)
    iroot, lroot = tmp_path / "indexes", tmp_path / "local"
    model = _FakeModel()

    # First full run, then remove shards 1 and 2 to simulate mid-run crash.
    index_dense.embed_passages(
        pfile, index_dense.R2,
        index_root=iroot, local_root=lroot,
        shard_size=50, batch_size=32, progress=False, _model=model,
    )
    shard_dir = iroot / "r2" / index_dense.EMB_SHARD_DIR
    for s in index_dense._emb_shards(shard_dir)[1:]:
        s.unlink()
        s.with_suffix(".ids").unlink()

    assert len(index_dense._done_ids(shard_dir)) == 50

    # Resume: should encode the remaining 70 passages and return 120 total.
    n = index_dense.embed_passages(
        pfile, index_dense.R2,
        index_root=iroot, local_root=lroot,
        shard_size=50, batch_size=32, progress=False, _model=model,
    )
    assert n == 120
    assert len(index_dense._done_ids(shard_dir)) == 120


def test_rerun_on_complete_corpus_is_noop(tmp_path):
    passages = _make_passages(50)
    _, iroot, lroot = _embed(tmp_path, passages, shard_size=50)
    pfile = tmp_path / "passages.jsonl"
    n = index_dense.embed_passages(
        pfile, index_dense.R2,
        index_root=iroot, local_root=lroot,
        shard_size=50, batch_size=32, progress=False, _model=_FakeModel(),
    )
    # Return value still equals total passages, but no new shards were written.
    assert n == 50
    shard_dir = iroot / "r2" / index_dense.EMB_SHARD_DIR
    assert len(index_dense._emb_shards(shard_dir)) == 1


def test_partial_shard_without_sidecar_is_redone(tmp_path):
    """A .npy without its .ids sidecar (crash after copy, before sidecar) is re-encoded."""
    passages = _make_passages(50)
    _, iroot, lroot = _embed(tmp_path, passages, shard_size=50)
    shard_dir = iroot / "r2" / index_dense.EMB_SHARD_DIR
    (shard_dir / "emb_0000.ids").unlink()

    index_dense.embed_passages(
        tmp_path / "passages.jsonl", index_dense.R2,
        index_root=iroot, local_root=lroot,
        shard_size=50, batch_size=32, progress=False, _model=_FakeModel(),
    )
    assert len(index_dense._done_ids(shard_dir)) == 50


# ---------------------------------------------------------------- build_index + search

def test_build_index_produces_valid_index(tmp_path):
    passages = _make_passages(80)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    n = index_dense.build_index(index_dense.R2, index_root=iroot)
    assert n == 80

    idir = index_dense.index_dir(index_dense.R2, iroot)
    assert (idir / index_dense.INDEX_FILE).exists()
    assert (idir / index_dense.IDS_FILE).exists()
    ids = [ln for ln in (idir / index_dense.IDS_FILE).read_text().splitlines() if ln]
    assert len(ids) == 80


def test_search_returns_k_hits(tmp_path):
    passages = _make_passages(80)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    index_dense.build_index(index_dense.R2, index_root=iroot)

    results = index_dense.search(
        ["kv cache eviction", "lunch menu"],
        index_dense.R2, k=5,
        index_root=iroot, _model=_FakeModel(),
    )
    assert len(results) == 2
    for hits in results:
        assert len(hits) == 5
        assert all(isinstance(pid, str) and isinstance(s, float) for pid, s in hits)


def test_search_scores_are_descending(tmp_path):
    passages = _make_passages(50)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    index_dense.build_index(index_dense.R2, index_root=iroot)

    results = index_dense.search(
        ["query"], index_dense.R2, k=20,
        index_root=iroot, _model=_FakeModel(),
    )
    scores = [s for _, s in results[0]]
    assert scores == sorted(scores, reverse=True)


def test_search_k_larger_than_corpus_returns_all(tmp_path):
    passages = _make_passages(30)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    index_dense.build_index(index_dense.R2, index_root=iroot)

    results = index_dense.search(
        ["query"], index_dense.R2, k=50,
        index_root=iroot, _model=_FakeModel(),
    )
    assert len(results[0]) == 30   # corpus has only 30 passages


def test_build_index_raises_on_incomplete_shard(tmp_path):
    passages = _make_passages(50)
    _, iroot, _ = _embed(tmp_path, passages, shard_size=50)
    shard_dir = iroot / "r2" / index_dense.EMB_SHARD_DIR
    (shard_dir / "emb_0000.ids").unlink()  # simulate incomplete shard

    with pytest.raises(ValueError, match="no .ids sidecar"):
        index_dense.build_index(index_dense.R2, index_root=iroot)


# ---------------------------------------------------------------- model configs

def test_r2_and_r3_configs_have_distinct_tags_and_prefixes():
    assert index_dense.R2.index_tag != index_dense.R3.index_tag
    assert index_dense.R2.query_prefix != index_dense.R3.query_prefix
    assert index_dense.R2.dtype == "fp16"
    assert index_dense.R3.dtype == "bf16"
