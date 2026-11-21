"""BM25 / Pyserini tests (issue #8): corpus preparation, search plumbing, run I/O.

No Java or Pyserini installation required: search tests inject _FakeSearcher
instead of LuceneSearcher, and prepare_corpus / run I/O are pure Python.
"""

import json
from pathlib import Path

import pytest

from lumbr import index_bm25, schema


# ---------------------------------------------------------------- helpers

def _make_passages(n: int) -> list[dict]:
    passages = []
    for i in range(n):
        pid    = schema.make_passage_id(f"dsid_{i:032x}", 0)
        header = f"[slack] t{i}"
        body   = f"body {i} " * (1 + i % 5)
        passages.append({
            "passage_id": pid,
            "doc_id":     f"dsid_{i:032x}",
            "source": "slack", "title": f"t{i}",
            "channel": None, "author": None, "date": None, "created": None,
            "chunk_idx": 0, "n_chunks": 1, "n_tokens": 20 + i,
            "header": header, "body": body,
            "text": schema.join_text(header, body),
        })
    return passages


def _write_passages(passages: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")


class _FakeHit:
    """Minimal stand-in for a Pyserini SimpleSearcher hit."""
    def __init__(self, docid: str, score: float):
        self.docid = docid
        self.score = score


class _FakeSearcher:
    """Inject into search() to test plumbing without Java."""
    def __init__(self, results: dict[str, list[_FakeHit]]):
        # keys are raw query strings
        self._results = results

    def batch_search(
        self,
        queries: list[str],
        qids: list[str],
        k: int = 50,
        threads: int = 1,
    ) -> dict[str, list[_FakeHit]]:
        return {qid: self._results.get(q, [])[:k] for qid, q in zip(qids, queries)}


# ---------------------------------------------------------------- prepare_corpus

def test_prepare_corpus_creates_docs_jsonl(tmp_path):
    passages = _make_passages(5)
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    corpus_dir = tmp_path / "corpus"
    index_bm25.prepare_corpus(pfile, corpus_dir)

    assert (corpus_dir / index_bm25.CORPUS_FILE).exists()


def test_prepare_corpus_returns_passage_count(tmp_path):
    passages = _make_passages(7)
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    n = index_bm25.prepare_corpus(pfile, tmp_path / "corpus")
    assert n == 7


def test_prepare_corpus_id_equals_passage_id(tmp_path):
    passages = _make_passages(3)
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    index_bm25.prepare_corpus(pfile, tmp_path / "corpus")
    lines = (tmp_path / "corpus" / index_bm25.CORPUS_FILE).read_text().splitlines()
    ids_out  = [json.loads(l)["id"]       for l in lines]
    ids_orig = [p["passage_id"]            for p in passages]
    assert ids_out == ids_orig


def test_prepare_corpus_contents_equals_text(tmp_path):
    passages = _make_passages(3)
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    index_bm25.prepare_corpus(pfile, tmp_path / "corpus")
    lines = (tmp_path / "corpus" / index_bm25.CORPUS_FILE).read_text().splitlines()
    for line, p in zip(lines, passages):
        assert json.loads(line)["contents"] == p["text"]


def test_prepare_corpus_overwrites_existing(tmp_path):
    # First run writes 5 passages; second run writes 3 — final file has 3 lines.
    pfile = tmp_path / "passages.jsonl"
    corpus_dir = tmp_path / "corpus"

    _write_passages(_make_passages(5), pfile)
    index_bm25.prepare_corpus(pfile, corpus_dir)

    _write_passages(_make_passages(3), pfile)
    index_bm25.prepare_corpus(pfile, corpus_dir)

    lines = (corpus_dir / index_bm25.CORPUS_FILE).read_text().splitlines()
    assert len(lines) == 3


def test_prepare_corpus_creates_missing_directory(tmp_path):
    pfile = tmp_path / "passages.jsonl"
    _write_passages(_make_passages(2), pfile)

    deep = tmp_path / "a" / "b" / "corpus"
    index_bm25.prepare_corpus(pfile, deep)
    assert (deep / index_bm25.CORPUS_FILE).exists()


def test_prepare_corpus_valid_json_per_line(tmp_path):
    passages = _make_passages(10)
    pfile = tmp_path / "passages.jsonl"
    _write_passages(passages, pfile)

    index_bm25.prepare_corpus(pfile, tmp_path / "corpus")
    for line in (tmp_path / "corpus" / index_bm25.CORPUS_FILE).read_text().splitlines():
        obj = json.loads(line)
        assert set(obj.keys()) == {"id", "contents"}


# ---------------------------------------------------------------- stage_index

def test_stage_index_copies_from_drive_to_local(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    drive_idx  = drive_root / index_bm25.INDEX_TAG
    drive_idx.mkdir(parents=True)
    (drive_idx / "segments_1").write_text("fake segment")
    (drive_idx / "other.si").write_text("fake si")

    result = index_bm25.stage_index(drive_root, local_root)

    assert result == local_root / index_bm25.INDEX_TAG
    assert (result / "segments_1").exists()
    assert (result / "other.si").exists()


def test_stage_index_skips_if_already_staged(tmp_path):
    # If local already has a segments_* file, stage_index returns immediately.
    # Drive is absent — proves no Drive access happens.
    local_root = tmp_path / "local"
    local_idx  = local_root / index_bm25.INDEX_TAG
    local_idx.mkdir(parents=True)
    (local_idx / "segments_1").write_text("already staged")

    result = index_bm25.stage_index(tmp_path / "drive_absent", local_root)
    assert result == local_idx


# ---------------------------------------------------------------- search (mocked)

def test_search_returns_hits_for_each_query():
    pid_a = schema.make_passage_id("dsid_" + "a" * 32, 0)
    pid_b = schema.make_passage_id("dsid_" + "b" * 32, 0)
    searcher = _FakeSearcher({
        "who broke prod?": [_FakeHit(pid_a, 12.5), _FakeHit(pid_b, 9.1)],
    })
    run = index_bm25.search(
        [("q1", "who broke prod?")],
        _searcher=searcher,
    )
    assert "q1" in run
    assert len(run["q1"]) == 2
    assert run["q1"][0] == (pid_a, 12.5)
    assert run["q1"][1] == (pid_b, 9.1)


def test_search_scores_are_floats():
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    searcher = _FakeSearcher({"q": [_FakeHit(pid, 7)]})   # int score from fake
    run = index_bm25.search([("q1", "q")], _searcher=searcher)
    _, score = run["q1"][0]
    assert isinstance(score, float)


def test_search_respects_k():
    hits = [_FakeHit(schema.make_passage_id(f"dsid_{i:032x}", 0), float(10 - i))
            for i in range(10)]
    searcher = _FakeSearcher({"many": hits})
    run = index_bm25.search([("q1", "many")], k=3, _searcher=searcher)
    assert len(run["q1"]) == 3


def test_search_multiple_queries():
    pid_a = schema.make_passage_id("dsid_" + "a" * 32, 0)
    pid_b = schema.make_passage_id("dsid_" + "b" * 32, 0)
    searcher = _FakeSearcher({
        "query one": [_FakeHit(pid_a, 10.0)],
        "query two": [_FakeHit(pid_b, 8.0)],
    })
    run = index_bm25.search(
        [("q1", "query one"), ("q2", "query two")],
        _searcher=searcher,
    )
    assert set(run.keys()) == {"q1", "q2"}
    assert run["q1"][0][0] == pid_a
    assert run["q2"][0][0] == pid_b


def test_search_empty_result_for_missing_query():
    searcher = _FakeSearcher({})
    run = index_bm25.search([("q1", "nothing here")], _searcher=searcher)
    assert run["q1"] == []


# ---------------------------------------------------------------- run file I/O

def test_write_run_creates_file(tmp_path):
    run = {"q1": [("dsid_" + "a" * 32 + "#000", 10.0)]}
    path = tmp_path / "run.json"
    index_bm25.write_run(run, path)
    assert path.exists()


def test_read_run_roundtrip(tmp_path):
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    original = {"q1": [(pid, 10.5), (schema.make_passage_id("dsid_" + "b" * 32, 0), 7.2)]}
    path = tmp_path / "run.json"
    index_bm25.write_run(original, path)
    loaded = index_bm25.read_run(path)
    assert loaded == original


def test_read_run_returns_tuples(tmp_path):
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    path = tmp_path / "run.json"
    index_bm25.write_run({"q1": [(pid, 1.0)]}, path)
    loaded = index_bm25.read_run(path)
    assert isinstance(loaded["q1"][0], tuple)


def test_write_run_creates_parent_dirs(tmp_path):
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    deep = tmp_path / "a" / "b" / "run.json"
    index_bm25.write_run({"q1": [(pid, 1.0)]}, deep)
    assert deep.exists()


# ---------------------------------------------------------------- save_run

def test_save_run_creates_subdir(tmp_path):
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    subdir = index_bm25.save_run(
        {"q1": [(pid, 10.0)]}, tmp_path,
        tag="r1_bm25", split="dev", date="2026-01-01",
    )
    assert subdir.exists() and subdir.is_dir()


def test_save_run_subdir_name_includes_tag_split_date(tmp_path):
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    subdir = index_bm25.save_run(
        {"q1": [(pid, 1.0)]}, tmp_path,
        tag="r1_bm25", split="dev", date="2026-01-01",
    )
    assert subdir.name == "r1_bm25_dev_2026-01-01"


def test_save_run_writes_run_json(tmp_path):
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    run = {"q1": [(pid, 5.0)]}
    subdir = index_bm25.save_run(
        run, tmp_path, tag="r1_bm25", split="dev", date="2026-01-01",
    )
    assert (subdir / "run.json").exists()
    assert index_bm25.read_run(subdir / "run.json") == run


def test_save_run_writes_meta_json_with_required_fields(tmp_path):
    pid = schema.make_passage_id("dsid_" + "a" * 32, 0)
    subdir = index_bm25.save_run(
        {"q1": [(pid, 1.0)]}, tmp_path,
        tag="r1_bm25", split="dev", k=50, date="2026-01-01",
    )
    meta = json.loads((subdir / "meta.json").read_text())
    for field in ("retriever", "tag", "split", "k", "k1", "b",
                  "pyserini_version", "git_commit", "date"):
        assert field in meta, f"missing field: {field}"
    assert meta["retriever"] == "bm25"
    assert meta["tag"]       == "r1_bm25"
    assert meta["split"]     == "dev"
    assert meta["k"]         == 50
    assert meta["date"]      == "2026-01-01"
