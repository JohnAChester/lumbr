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


# ---------------------------------------------------------------- Task 4: generate + save

class _FakeChat:
    """Answers each prompt with the scripted text for the question it contains; records every call."""

    def __init__(self, replies: dict[str, str]):
        self.replies = replies
        self.calls: list[list[list[dict]]] = []

    def __call__(self, batch):
        self.calls.append(batch)
        out = []
        for messages in batch:
            question = messages[-1]["content"].rsplit("Question: ", 1)[1]
            out.append({"text": self.replies[question], "prompt_tokens": 100, "completion_tokens": 7})
        return out


def _run_fixture():
    by_id = _by_id(_doc("a", 3), _doc("b", 2))
    run = {"q1": [(_pid("a", 1), 9.0), (_pid("b", 0), 8.0)],
           "q2": [(_pid("b", 1), 5.0)]}
    return by_id, run


def test_generate_end_to_end_with_fake_chat():
    prompt = generate.load_prompt()
    by_id, run = _run_fixture()
    chat = _FakeChat({"Who owns A?": "Alice owns it [1].", "What about B?": prompt["abstain"]})
    records, wall_s = generate.generate(chat, [("q1", "Who owns A?"), ("q2", "What about B?")],
                                        run, by_id, prompt)
    assert [r["question_id"] for r in records] == ["q1", "q2"]
    r1, r2 = records
    assert r1["citations"][0]["doc_id"] == "a" and r1["abstained"] is False
    assert r1["sources"][0] == {"n": 1, "doc_id": "a", "passage_ids": [_pid("a", 1)],
                                "window_ids": [_pid("a", 0), _pid("a", 1), _pid("a", 2)]}
    assert r1["prompt_tokens"] == 100 and r1["completion_tokens"] == 7
    assert r2["abstained"] is True
    assert len(chat.calls) == 1 and len(chat.calls[0]) == 2
    assert chat.calls[0][0][-1]["content"].endswith("Who owns A?")
    assert wall_s >= 0


def test_only_top_k_hits_used():
    by_id = _by_id(*[_doc(f"d{i}", 1) for i in range(8)])
    run = {"q": [(_pid(f"d{i}", 0), 10.0 - i) for i in range(8)]}
    chat = _FakeChat({"Q?": "x [1]."})
    [rec], _ = generate.generate(chat, [("q", "Q?")], run, by_id, generate.load_prompt(), k=5)
    used = [pid for s in rec["sources"] for pid in s["passage_ids"]]
    assert used == [_pid(f"d{i}", 0) for i in range(5)]


def test_question_without_hits_abstains_without_model_call():
    prompt = generate.load_prompt()
    by_id, run = _run_fixture()
    chat = _FakeChat({"Who owns A?": "Alice [1]."})
    records, _ = generate.generate(chat, [("q1", "Who owns A?"), ("missing", "Nothing?")],
                                   run, by_id, prompt)
    rec = records[1]
    assert rec["answer"] == prompt["abstain"] and rec["abstained"] is True
    assert rec["sources"] == [] and rec["prompt_tokens"] == 0 and rec["completion_tokens"] == 0
    assert len(chat.calls[0]) == 1

    lonely = _FakeChat({})
    [only], _ = generate.generate(lonely, [("missing", "Nothing?")], run, by_id, prompt)
    assert only["abstained"] is True and lonely.calls == []


def test_needed_doc_ids():
    run = {"q1": [(_pid("a", 0), 3.0), (_pid("b", 0), 2.0), (_pid("c", 0), 1.0)],
           "q2": [(_pid("z", 0), 1.0)]}
    assert generate.needed_doc_ids(run, ["q1"], k=2) == {"a", "b"}


def _records():
    prompt = generate.load_prompt()
    by_id, run = _run_fixture()
    chat = _FakeChat({"Who owns A?": "Alice [1] [9].", "What about B?": prompt["abstain"]})
    records, _ = generate.generate(chat, [("q1", "Who owns A?"), ("q2", "What about B?")],
                                   run, by_id, prompt)
    return records


def test_save_writes_answers_and_meta(tmp_path):
    records = _records()
    out = generate.save_answers(records, tmp_path, {"model": "m"}, date="2026-11-23")
    assert out == tmp_path / "gen_r1_dev_2026-11-23"
    lines = (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "m"
    assert meta["n_questions"] == 2 and meta["n_abstained"] == 1 and meta["n_invalid_citations"] == 1
    assert meta["date"] == "2026-11-23" and meta["git_commit"]


def test_save_refuses_to_overwrite(tmp_path):
    records = _records()
    out = generate.save_answers(records, tmp_path, {}, date="2026-11-23")
    before = (out / "answers.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        generate.save_answers(records[:1], tmp_path, {}, date="2026-11-23")
    assert (out / "answers.jsonl").read_bytes() == before


# ---------------------------------------------------------------- final review fixes

def test_abstention_without_final_period():
    prompt = generate.load_prompt()
    text = prompt["abstain"].rstrip(".")
    assert generate.parse_answer(text, _three_sources(), prompt)["abstained"] is True


def test_finish_reason_recorded_and_truncations_counted(tmp_path):
    prompt = generate.load_prompt()
    by_id, run = _run_fixture()

    def chat(batch):
        return [{"text": "Alice owns it [1", "prompt_tokens": 100, "completion_tokens": 512,
                 "finish_reason": "length"} for _ in batch]

    records, _ = generate.generate(chat, [("q1", "Who owns A?"), ("missing", "Nothing?")],
                                   run, by_id, prompt)
    assert records[0]["finish_reason"] == "length"
    assert records[1]["finish_reason"] is None
    out = generate.save_answers(records, tmp_path, {}, date="2026-11-23")
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["n_truncated"] == 1


def test_chat_output_count_mismatch_raises():
    prompt = generate.load_prompt()
    by_id, run = _run_fixture()
    with pytest.raises(ValueError):
        generate.generate(lambda batch: [], [("q1", "Who owns A?")], run, by_id, prompt)
    with pytest.raises(ValueError):
        generate.generate(lambda batch: batch * 2, [("q1", "Who owns A?")], run, by_id, prompt)


# ---------------------------------------------------------------- duplicate hits

def test_duplicate_hits_skipped_and_counted(tmp_path):
    by_id = _by_id(*[_doc(f"d{i}", 1) for i in range(7)])
    hits = [_pid("d0", 0), _pid("d1", 0), _pid("d0", 0), _pid("d2", 0),
            _pid("d3", 0), _pid("d1", 0), _pid("d4", 0), _pid("d5", 0)]
    run = {"q": [(pid, 10.0 - i) for i, pid in enumerate(hits)]}
    [rec], _ = generate.generate(_FakeChat({"Q?": "x [1]."}), [("q", "Q?")], run, by_id,
                                 generate.load_prompt(), k=5)
    used = [pid for s in rec["sources"] for pid in s["passage_ids"]]
    assert used == [_pid(f"d{i}", 0) for i in range(5)]
    assert rec["n_duplicate_hits"] == 2
    out = generate.save_answers([rec], tmp_path, {}, date="2026-11-23")
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["n_questions_with_duplicate_hits"] == 1


def test_needed_doc_ids_skips_duplicates():
    run = {"q": [(_pid("a", 0), 3.0), (_pid("a", 0), 2.0), (_pid("b", 0), 1.0)]}
    assert generate.needed_doc_ids(run, ["q"], k=2) == {"a", "b"}


def test_save_records_paths_git_commit(tmp_path, monkeypatch):
    from lumbr import paths
    monkeypatch.setattr(paths, "git_commit", lambda: "abc1234")
    out = generate.save_answers(_records(), tmp_path, {}, date="2026-11-23")
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["git_commit"] == "abc1234"


# ---------------------------------------------------------------- run folder naming + early check

def test_run_tag_includes_prompt_version():
    assert generate.run_tag("r1", {"version": 2}) == "gen_r1_p2"


def test_answers_dir_raises_early_if_taken(tmp_path):
    path = generate.answers_dir(tmp_path, "gen_r1_p1", "dev", date="2026-11-23")
    assert path == tmp_path / "gen_r1_p1_dev_2026-11-23" and not path.exists()
    generate.save_answers(_records(), tmp_path, {}, tag="gen_r1_p1", date="2026-11-23")
    with pytest.raises(FileExistsError):
        generate.answers_dir(tmp_path, "gen_r1_p1", "dev", date="2026-11-23")
    assert generate.answers_dir(tmp_path, "gen_r1_p2", "dev", date="2026-11-23").name == "gen_r1_p2_dev_2026-11-23"


# ---------------------------------------------------------------- load_passages speed + edges

def test_load_passages_skips_unneeded_lines_without_parsing(tmp_path):
    path, good = tmp_path / "passages.jsonl", tmp_path / "good.jsonl"
    schema.write_jsonl(_doc("a", 2), good)
    # An unneeded line before the needed ones: it only breaks if it gets JSON-parsed.
    path.write_text('{"passage_id": "z#000", "doc_id": "z", this line is not valid JSON\n'
                    + good.read_text(encoding="utf-8"), encoding="utf-8")
    got = generate.load_passages({"a"}, path)
    assert set(got) == {_pid("a", 0), _pid("a", 1)}


def test_load_passages_stops_once_all_docs_complete(tmp_path):
    path = tmp_path / "passages.jsonl"
    schema.write_jsonl(_doc("a", 2), path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json at all\n")  # would raise if reached
    assert set(generate.load_passages({"a"}, path)) == {_pid("a", 0), _pid("a", 1)}


def test_load_passages_missing_doc_scans_whole_file(tmp_path):
    path = tmp_path / "passages.jsonl"
    schema.write_jsonl(_doc("a", 1) + _doc("b", 2), path)
    assert set(generate.load_passages({"b", "nope"}, path)) == {_pid("b", 0), _pid("b", 1)}


def test_load_passages_falls_back_when_format_differs(tmp_path):
    path = tmp_path / "passages.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for p in _doc("a", 2):
            f.write(json.dumps(p, separators=(",", ":")) + "\n")  # no spaces: the quick check can't match
    assert set(generate.load_passages({"a"}, path)) == {_pid("a", 0), _pid("a", 1)}


# ---------------------------------------------------------------- round 2: abstention = starts with the line

def test_abstention_with_cited_explanation_counts():
    prompt = generate.load_prompt()
    text = prompt["abstain"] + " [3] mentions the rollout but not the budgets. [1] is about something else."
    got = generate.parse_answer(text, _three_sources(), prompt)
    assert got["abstained"] is True
    assert [c["n"] for c in got["citations"]] == [3, 1]  # still recorded, for the explanation


def test_abstention_line_later_in_answer_is_not_abstention():
    prompt = generate.load_prompt()
    text = "Retention is 90 days. " + prompt["abstain"]  # uncited, but it answered first
    assert generate.parse_answer(text, _three_sources(), prompt)["abstained"] is False


def test_rescore_records_reapplies_parsing():
    prompt = generate.load_prompt()
    old = {"question_id": "q", "answer": prompt["abstain"] + " [1] is related only.",
           "abstained": False, "citations": [], "invalid_citations": [],
           "sources": [{"n": 1, "doc_id": "a", "passage_ids": ["a#000"], "window_ids": ["a#000"]}]}
    [new] = generate.rescore_records([old], prompt)
    assert new["abstained"] is True
    assert new["citations"] == [{"n": 1, "passage_ids": ["a#000"], "doc_id": "a"}]
    assert old["abstained"] is False  # input untouched


def test_summarize_round():
    recs = [
        {"question_id": "q1", "abstained": True, "invalid_citations": [], "finish_reason": "stop",
         "sources": [{"doc_id": "g1"}]},                     # abstained with gold present
        {"question_id": "q2", "abstained": False, "invalid_citations": [7], "finish_reason": "length",
         "sources": [{"doc_id": "x"}]},                      # answered without gold
        {"question_id": "q3", "abstained": True, "invalid_citations": [], "finish_reason": "stop",
         "sources": []},                                     # info_not_found, abstained
    ]
    types = {"q1": "basic", "q2": "basic", "q3": "info_not_found"}
    gold = {"q1": {"g1"}, "q2": {"g2"}, "q3": set()}
    s = generate.summarize(recs, types, gold)
    assert s == {"n": 3, "abstained": 2, "info_not_found_abstained": "1/1",
                 "answerable_abstained_gold_present": 1, "answerable_answered_gold_missing": 1,
                 "answers_with_invalid_citations": 1, "truncated": 1}
