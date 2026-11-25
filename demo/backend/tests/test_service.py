import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ANSWERABLE, JUNK
from lumbr import generate
from lumbr_demo.artifacts import load_retriever
from lumbr_demo.service import strip_citations

BACKEND = Path(__file__).resolve().parents[1]


def test_first_turn_messages_match_study(retriever, store):
    r = retriever.retrieve(ANSWERABLE)
    assert r["passed"] is True and len(r["hit_ids"]) == 5
    prompt = generate.load_prompt()
    sources = generate.build_sources(r["hit_ids"], store.with_neighbours(r["hit_ids"]), 1)
    assert r["messages"] == generate.build_messages(ANSWERABLE, sources, prompt)
    assert r["prompt_sha256"] == prompt["sha256"]
    assert r["hit_ids"][0] == "dsid_bbb#001"


def test_sources_carry_passage_fields(retriever):
    for s in retriever.retrieve(ANSWERABLE)["sources"]:
        assert s["passage_id"] == s["passage_ids"][0]
        assert s["title"] and s["source"] and isinstance(s["score"], float)
        assert {"n", "doc_id", "window_ids", "header", "body", "channel", "author", "date"} <= set(s)


def test_not_passed_without_context_returns_nothing(retriever):
    r = retriever.retrieve(JUNK)
    assert r["passed"] is False and r["sources"] == [] and r["messages"] == [] and r["hit_ids"] == []


def test_not_passed_with_context_uses_earlier_hits(retriever):
    r = retriever.retrieve(JUNK, context_ids=["dsid_ccc#000", "dsid_nope#000"])
    assert r["passed"] is False
    assert r["hit_ids"] == ["dsid_ccc#000"]
    assert [s["doc_id"] for s in r["sources"]] == ["dsid_ccc"]
    assert r["sources"][0]["score"] is None
    assert len(r["messages"]) == 2


def test_context_merged_and_capped_at_8(retriever, passages):
    first = retriever.retrieve(ANSWERABLE)["hit_ids"]
    context = [p["passage_id"] for p in passages if p["passage_id"] not in first][:6] + first[:1]
    r = retriever.retrieve(ANSWERABLE, context_ids=context)
    assert len(r["hit_ids"]) == 8 and len(set(r["hit_ids"])) == 8
    assert r["hit_ids"][:5] == first


def test_history_citations_stripped(retriever):
    history = [{"q": "What's the per-file limit?", "a": "It is 10 MiB [1][2]."}]
    r = retriever.retrieve(ANSWERABLE, history=history)
    plain = retriever.retrieve(ANSWERABLE)["messages"]
    assert r["messages"][0] == plain[0] and r["messages"][-1] == plain[-1]
    assert r["messages"][1:3] == [{"role": "user", "content": "What's the per-file limit?"},
                                  {"role": "assistant", "content": "It is 10 MiB."}]


def test_strip_citations_forms():
    assert strip_citations("A [1, 3] b [2](http://x) c [4].") == "A b [2](http://x) c."


def test_passages_shape(retriever):
    got = retriever.passages(["dsid_aaa#000", "dsid_nope#000"])
    assert [p["passage_id"] for p in got] == ["dsid_aaa#000"] and got[0]["score"] is None


def test_load_retriever_refuses_null_threshold(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"threshold": None}), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_retriever(tmp_path)


def test_service_import_is_light():
    code = ("import sys, app.main; bad = [m for m in ('pandas', 'torch') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=BACKEND).returncode == 0
