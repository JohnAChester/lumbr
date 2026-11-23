"""Answer generator tests (issue #14). No GPU, vLLM or Drive: the model is a fake chat function."""

import json
from pathlib import Path

import pytest

from lumbr import generate, schema


# ---------------------------------------------------------------- helpers

def _doc(doc_id: str, n_chunks: int, bodies: list[str] | None = None) -> list[dict]:
    """Schema-shaped passages for one document. Headers differ per chunk so tests can tell them apart."""
    out = []
    for i in range(n_chunks):
        header = f"[slack] #eng · T · A · 2026-01-0{i + 1}"
        body = bodies[i] if bodies else f"{doc_id} chunk {i}"
        out.append({
            "passage_id": schema.make_passage_id(doc_id, i), "doc_id": doc_id,
            "source": "slack", "title": "T", "channel": "eng", "author": "A",
            "date": f"2026-01-0{i + 1}", "created": None,
            "chunk_idx": i, "n_chunks": n_chunks, "n_tokens": 10,
            "header": header, "body": body, "text": schema.join_text(header, body),
        })
    return out


def _by_id(*docs: list[dict]) -> dict[str, dict]:
    return {p["passage_id"]: p for d in docs for p in d}


def _pid(doc_id: str, i: int) -> str:
    return schema.make_passage_id(doc_id, i)


# ---------------------------------------------------------------- Task 1: context

def test_load_passages_keeps_only_requested_docs(tmp_path):
    path = tmp_path / "passages.jsonl"
    schema.write_jsonl(_doc("a", 3) + _doc("b", 2) + _doc("c", 1), path)
    got = generate.load_passages({"a", "c"}, path)
    assert set(got) == {_pid("a", 0), _pid("a", 1), _pid("a", 2), _pid("c", 0)}


def test_single_hit_gets_neighbours():
    by_id = _by_id(_doc("a", 5))
    [src] = generate.build_sources([_pid("a", 1)], by_id)
    assert src["n"] == 1
    assert src["doc_id"] == "a"
    assert src["passage_ids"] == [_pid("a", 1)]
    assert src["window_ids"] == [_pid("a", 0), _pid("a", 1), _pid("a", 2)]


def test_window_clipped_at_document_edges():
    by_id = _by_id(_doc("a", 5), _doc("b", 1))
    s1, s2 = generate.build_sources([_pid("a", 0), _pid("b", 0)], by_id)
    assert s1["window_ids"] == [_pid("a", 0), _pid("a", 1)]
    assert s2["window_ids"] == [_pid("b", 0)]


def test_adjacent_hits_merge_into_one_block():
    by_id = _by_id(_doc("a", 6), _doc("x", 5))
    sources = generate.build_sources([_pid("a", 1), _pid("x", 2), _pid("a", 3)], by_id)
    assert len(sources) == 2
    s1, s2 = sources
    assert s1["doc_id"] == "a" and s1["n"] == 1
    assert s1["window_ids"] == [_pid("a", i) for i in range(5)]
    assert s1["passage_ids"] == [_pid("a", 1), _pid("a", 3)]
    assert s2["doc_id"] == "x" and s2["n"] == 2
    all_ids = s1["window_ids"] + s2["window_ids"]
    assert len(all_ids) == len(set(all_ids))


def test_blocks_in_rank_order_by_best_hit():
    by_id = _by_id(_doc("a", 10), _doc("b", 1))
    sources = generate.build_sources([_pid("a", 4), _pid("b", 0), _pid("a", 0)], by_id)
    assert [s["n"] for s in sources] == [1, 2, 3]
    assert [s["passage_ids"] for s in sources] == [[_pid("a", 4)], [_pid("b", 0)], [_pid("a", 0)]]


def test_body_drops_carry_and_uses_first_header():
    bodies = ["Alice: can we ship Friday?", "Alice: can we ship Friday?\nBob: yes, Friday works."]
    by_id = _by_id(_doc("a", 2, bodies))
    [src] = generate.build_sources([_pid("a", 1)], by_id)
    assert src["body"].count("can we ship Friday?") == 1
    assert "Bob: yes, Friday works." in src["body"]
    assert src["header"] == by_id[_pid("a", 0)]["header"]


def test_missing_hit_raises():
    with pytest.raises(KeyError):
        generate.build_sources([_pid("zzz", 0)], _by_id(_doc("a", 2)))
