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


# ---------------------------------------------------------------- Task 2: prompt

def test_prompt_file_is_self_consistent():
    prompt = generate.load_prompt()
    for key in ("version", "system", "user_template", "source_template", "abstain"):
        assert key in prompt
    assert prompt["version"] == 1
    assert len(prompt["sha256"]) == 64
    assert prompt["abstain"] in prompt["system"]


def test_build_messages_labels_sources_in_order():
    prompt = generate.load_prompt()
    sources = generate.build_sources([_pid("b", 0), _pid("a", 0)], _by_id(_doc("a", 1), _doc("b", 1)))
    system, user = generate.build_messages("When do we ship?", sources, prompt)
    assert system == {"role": "system", "content": prompt["system"]}
    assert user["role"] == "user"
    first = f"[1] {sources[0]['header']}\n{sources[0]['body']}"
    second = f"[2] {sources[1]['header']}\n{sources[1]['body']}"
    assert user["content"].index(first) < user["content"].index(second)
    assert user["content"].endswith("When do we ship?")


def test_prompt_hash_changes_with_content(tmp_path):
    copy = tmp_path / "answer.json"
    text = generate.PROMPT_PATH.read_text(encoding="utf-8")
    copy.write_text(text, encoding="utf-8")
    original = generate.load_prompt(copy)["sha256"]
    copy.write_text(text.replace("Lumbr", "Lumbr!", 1), encoding="utf-8")
    assert generate.load_prompt(copy)["sha256"] != original


def test_prompt_hash_ignores_line_endings(tmp_path):
    text = generate.PROMPT_PATH.read_bytes().replace(b"\r\n", b"\n")
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(text)
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    assert generate.load_prompt(lf)["sha256"] == generate.load_prompt(crlf)["sha256"]


# ---------------------------------------------------------------- Task 3: citations + abstention

def _three_sources() -> list[dict]:
    return generate.build_sources([_pid("a", 0), _pid("b", 0), _pid("c", 0)],
                                  _by_id(_doc("a", 1), _doc("b", 1), _doc("c", 1)))


def test_parse_citation_forms_and_invalid():
    sources = _three_sources()
    got = generate.parse_answer("A is true [1]. B too [1, 3]. C [3][2]. D [7].", sources,
                                generate.load_prompt())
    assert [c["n"] for c in got["citations"]] == [1, 3, 2]
    assert got["citations"][1] == {"n": 3, "passage_ids": [_pid("c", 0)], "doc_id": "c"}
    assert got["invalid_citations"] == [7]
    assert got["abstained"] is False


def test_abstention_exact():
    prompt = generate.load_prompt()
    got = generate.parse_answer(prompt["abstain"], _three_sources(), prompt)
    assert got["abstained"] is True
    assert got["citations"] == []


def test_abstention_normalised():
    text = "  i’m SORRY, but I can’t find  that information in our database. "
    got = generate.parse_answer(text, _three_sources(), generate.load_prompt())
    assert got["abstained"] is True


def test_partial_answer_is_not_abstention():
    text = "The launch is in May [2]. I'm sorry, but I can't find that information in our database."
    got = generate.parse_answer(text, _three_sources(), generate.load_prompt())
    assert got["abstained"] is False
    assert [c["n"] for c in got["citations"]] == [2]


def test_non_citation_brackets_ignored():
    got = generate.parse_answer("Use the [beta] flag [1].", _three_sources(), generate.load_prompt())
    assert [c["n"] for c in got["citations"]] == [1]
    assert got["invalid_citations"] == []
