"""B0 = R1 (BM25) + R4 (dense) fused with the study's RRF (spec 2026-11-24-b0-backend-design)."""
import json
import re

import numpy as np
import pytest

from conftest import ANSWERABLE, JUNK, THRESHOLD
from lumbr import index_rrf


class OverlapLexical:
    """Test stand-in for Lucene BM25: score = number of query words in the passage text."""

    def __init__(self, passages):
        self.docs = {p["passage_id"]: set(re.findall(r"[a-z0-9]+", p["text"].lower())) for p in passages}

    def search(self, text, k):
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        scored = [(pid, float(len(words & ws))) for pid, ws in self.docs.items() if words & ws]
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:k]


@pytest.fixture
def fused(store, passages):
    from lumbr.generate import load_prompt
    from lumbr_demo.encoder import HashEncoder
    from lumbr_demo.index import DenseIndex
    from lumbr_demo.service import Retriever
    enc = HashEncoder()
    index = DenseIndex(np.stack([enc.encode(p["text"]) for p in passages]), [p["passage_id"] for p in passages])
    return Retriever(enc, index, store, load_prompt(), THRESHOLD, "B0", lexical=OverlapLexical(passages))


# ---------------------------------------------------------------- dense index top-k

def test_dense_search_matches_full_sort_including_ties():
    from lumbr_demo.index import DenseIndex
    rng = np.random.default_rng(0)
    m = rng.standard_normal((500, 16)).astype(np.float32)
    m[10] = m[3]  # exact ties: broken by id
    m[250] = m[3]
    ids = [f"dsid_{i:04d}#000" for i in range(500)][::-1]
    q = m[3] / np.linalg.norm(m[3])
    scores = m @ q
    want = sorted(range(500), key=lambda i: (-scores[i], ids[i]))[:20]
    got = DenseIndex(m, ids).search(q, 20)
    assert [pid for pid, _ in got] == [ids[i] for i in want]
    assert got[0][1] == pytest.approx(float(scores[want[0]]))


def test_dense_search_k_larger_than_index():
    from lumbr_demo.index import DenseIndex
    got = DenseIndex(np.eye(3, dtype=np.float32), ["a", "b", "c"]).search(np.array([1, 0, 0], np.float32), 10)
    assert [pid for pid, _ in got] == ["a", "b", "c"]


# ---------------------------------------------------------------- fusion

def test_hits_are_the_study_rrf_of_both_first_stages(fused):
    from lumbr_demo.service import FIRST_STAGE_K, RRF_K
    dense = fused.index.search(fused.encoder.encode(ANSWERABLE), FIRST_STAGE_K)
    lexical = fused.lexical.search(ANSWERABLE, FIRST_STAGE_K)
    want = index_rrf.fuse([{"q": lexical}, {"q": dense}], k=RRF_K, top_n=FIRST_STAGE_K)["q"]
    r = fused.retrieve(ANSWERABLE, k=5)
    assert r["passed"] is True
    assert r["hit_ids"] == [pid for pid, _ in want[:5]]
    assert [s["score"] for s in r["sources"]] == pytest.approx(
        [dict(want)[s["passage_id"]] for s in r["sources"]])


def test_study_constants():
    from lumbr_demo.service import FIRST_STAGE_K, RRF_K
    assert (FIRST_STAGE_K, RRF_K) == (50, 60)  # notebook 13 / 17: top 50 per run, k = 60


def test_filter_uses_the_dense_top_score(fused, passages):
    """A question BM25 matches but the dense side doesn't is still off-topic."""
    r = fused.retrieve("retention " + JUNK)  # one shared word: lexical hits, weak cosine
    top_dense = fused.index.search(fused.encoder.encode("retention " + JUNK), 1)[0][1]
    assert r["top_score"] == pytest.approx(top_dense)
    assert r["passed"] is (top_dense >= THRESHOLD)


def test_junk_still_returns_nothing(fused):
    r = fused.retrieve(JUNK)
    assert r["passed"] is False and r["hit_ids"] == [] and r["sources"] == []


def test_dense_only_retriever_unchanged(retriever):
    """No lexical searcher: R2-style dense retrieval, as the sample artifacts serve today."""
    r = retriever.retrieve(ANSWERABLE, k=5)
    dense = retriever.index.search(retriever.encoder.encode(ANSWERABLE), 5)
    assert r["hit_ids"] == [pid for pid, _ in dense]


# ---------------------------------------------------------------- Lucene wrapper and loading

class _Hit:
    def __init__(self, docid, score):
        self.docid, self.score = docid, score


class _FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, q, k):
        self.calls.append((q, k))
        return [_Hit("dsid_aaa#000", 7.5), _Hit("dsid_bbb#001", 3.25)]


def test_lucene_bm25_returns_id_score_pairs():
    from lumbr_demo.lexical import LuceneBM25
    s = _FakeSearcher()
    assert LuceneBM25(searcher=s).search("audit logs", 50) == [("dsid_aaa#000", 7.5), ("dsid_bbb#001", 3.25)]
    assert s.calls == [("audit logs", 50)]


def test_load_retriever_builds_b0_when_config_names_a_lexical_index(tmp_path, passages, monkeypatch):
    from lumbr_demo import artifacts, encoder, lexical
    from lumbr_demo.encoder import HashEncoder
    from lumbr_demo.index import DenseIndex
    from lumbr_demo.store import build_store
    enc = HashEncoder()
    DenseIndex(np.stack([enc.encode(p["text"]) for p in passages]), [p["passage_id"] for p in passages]).save(tmp_path)
    build_store(passages, tmp_path / "passages.sqlite")
    (tmp_path / "config.json").write_text(json.dumps(
        {"retriever": "B0", "threshold": 0.3, "query_prefix": "", "lexical": "bm25"}), encoding="utf-8")
    opened = []
    monkeypatch.setattr(encoder, "OnnxEncoder", lambda d, prefix: enc)
    monkeypatch.setattr(lexical, "LuceneBM25", lambda index_dir: opened.append(index_dir) or OverlapLexical(passages))
    r = artifacts.load_retriever(tmp_path)
    assert r.name == "B0" and r.lexical is not None and opened == [tmp_path / "bm25"]
    from lumbr_demo.index import FaissDenseIndex
    assert isinstance(r.index, FaissDenseIndex)  # B0 serves the memory-mapped FAISS fp16 index


def test_rrf_ties_go_to_the_bm25_order(store):
    """BM25 ranks a over b and dense ranks b over a. Their RRF sums differ only by float rounding, which
    depends on the order the runs are passed; the service must pass them as the study did, R1 first."""
    from lumbr.generate import load_prompt
    from lumbr_demo.index import DenseIndex
    from lumbr_demo.service import Retriever

    class Fixed:
        def __init__(self, hits):
            self.hits = hits

        def search(self, _q, _k):
            return self.hits

        def encode(self, _t):
            return np.array([1.0], np.float32)

    a, b = "dsid_aaa#000", "dsid_bbb#000"
    index = DenseIndex(np.array([[0.5], [0.9]], np.float32), [a, b])  # dense: b first
    r = Retriever(Fixed(None), index, store, load_prompt(), 0.1, "B0", lexical=Fixed([(a, 9.0), (b, 1.0)]))
    lex_run, dense_run = {"q": [(a, 9.0), (b, 1.0)]}, {"q": index.search(np.array([1.0], np.float32), 2)}
    study = [pid for pid, _ in index_rrf.fuse([lex_run, dense_run], k=60, top_n=50)["q"]]
    swapped = [pid for pid, _ in index_rrf.fuse([dense_run, lex_run], k=60, top_n=50)["q"]]
    assert study != swapped  # the order matters for this pair, so the test can tell them apart
    assert r.retrieve("anything", k=2)["hit_ids"] == study


def test_lucene_caps_the_jvm_heap_before_the_jvm_starts(monkeypatch, tmp_path):
    """The 4 GB server needs a bounded JVM: -Xmx goes to pyjnius before Pyserini starts the JVM."""
    import sys
    import types

    from lumbr_demo.lexical import LuceneBM25
    options = []
    jc = types.SimpleNamespace(vm_running=False, add_options=lambda *o: options.extend(o))
    opened = []

    class FakeSearcher:
        def __init__(self, path):
            opened.append(path)

    lucene = types.ModuleType("pyserini.search.lucene")
    lucene.LuceneSearcher = FakeSearcher
    monkeypatch.setitem(sys.modules, "jnius_config", jc)
    monkeypatch.setitem(sys.modules, "pyserini", types.ModuleType("pyserini"))
    monkeypatch.setitem(sys.modules, "pyserini.search", types.ModuleType("pyserini.search"))
    monkeypatch.setitem(sys.modules, "pyserini.search.lucene", lucene)
    monkeypatch.setenv("LUMBR_JVM_HEAP", "768m")
    LuceneBM25(tmp_path / "bm25")
    assert options == ["-Xmx768m"] and opened == [str(tmp_path / "bm25")]


def test_lucene_heap_default_is_1g(monkeypatch, tmp_path):
    import sys
    import types

    from lumbr_demo.lexical import LuceneBM25
    options = []
    monkeypatch.setitem(sys.modules, "jnius_config",
                        types.SimpleNamespace(vm_running=False, add_options=lambda *o: options.extend(o)))
    lucene = types.ModuleType("pyserini.search.lucene")
    lucene.LuceneSearcher = lambda path: None
    for name in ("pyserini", "pyserini.search"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "pyserini.search.lucene", lucene)
    monkeypatch.delenv("LUMBR_JVM_HEAP", raising=False)
    LuceneBM25(tmp_path)
    assert options == ["-Xmx1g"]
