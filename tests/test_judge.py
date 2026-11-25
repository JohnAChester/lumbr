"""Unit and resumability tests for src/lumbr/judge.py and judge_table.py.

No network calls: all judge calls go through _FakeClient.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumbr import judge as jg
from lumbr import judge_citation as jc
from lumbr import judge_table as jt


# ---------------------------------------------------------------- helpers / fixtures

class _FakeClient:
    """Drop-in for JudgeClient: returns scripted responses; can raise after N calls."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        raise_after: int | None = None,
        raise_exc: Exception | None = None,
    ):
        """responses: {content_prefix: response_text}; default: always "yes"/"aligned".
        raise_after: raise raise_exc (default OutOfCredits) after this many calls.
        """
        self._responses  = responses or {}
        self._raise_after = raise_after
        self._raise_exc   = raise_exc or jg.OutOfCredits("fake out of credits")
        self.calls: list[list[dict]] = []

    def call_one(self, messages: list[dict], max_tokens: int = 256) -> str:
        self.calls.append(messages)
        if self._raise_after is not None and len(self.calls) > self._raise_after:
            raise self._raise_exc
        # Match by content prefix
        content = messages[0]["content"]
        for prefix, resp in self._responses.items():
            if content.startswith(prefix) or prefix in content:
                return resp
        # Default: JSON "yes" for correctness, plain "yes" for facts (the fact prompt has "## Statement")
        if "## Statement" in content:
            return "yes"
        return '{"reason": "matches", "aligned": "yes"}'

    # Minimal batch stubs (not under test here — batch tests use _FakeBatchClient below)
    def submit_batch(self, requests): return "fake-batch-id"
    def batch_status(self, batch_id): return "ended"
    def batch_results(self, batch_id): return iter([])


class _FakeBatchClient(_FakeClient):
    """Fake client with controllable batch results."""

    def __init__(self, batch_results: list[dict], **kw):
        super().__init__(**kw)
        self._batch_results = batch_results
        self._submitted: list[dict] = []

    def submit_batch(self, requests):
        self._submitted = requests
        return "batch-001"

    def batch_status(self, batch_id):
        return "ended"

    def batch_results(self, batch_id):
        return iter(self._batch_results)


def _q(qid="q1", question_type="basic"):
    return {
        "question": "What is X?",
        "question_type": question_type,
        "gold_answer": "X is 42.",
        "answer_facts": ["X is 42", "X is positive"],
        "expected_doc_ids": ["dsid_abc"],
    }


def _rec(qid="q1", answer="X is 42 [1].", abstained=False):
    return {
        "question_id": qid,
        "answer": answer,
        "abstained": abstained,
        "sources": [{"n": 1, "doc_id": "dsid_abc",
                     "passage_ids": ["dsid_abc#000"], "window_ids": ["dsid_abc#000"]}],
    }


# ---------------------------------------------------------------- citation stripping

def test_strip_citations_basic():
    assert jg.strip_citations("X is 42 [1]. Also [2][3].") == "X is 42. Also."


def test_strip_citations_comma_form():
    assert jg.strip_citations("See [1, 3] and [2].") == "See and."


def test_strip_citations_preserves_markdown_links():
    assert jg.strip_citations("See [docs](http://x) [2].") == "See [docs](http://x)."


def test_strip_citations_no_citations():
    assert jg.strip_citations("No citations here.") == "No citations here."


# ---------------------------------------------------------------- prompt formatting & hashing

def test_correctness_messages_format():
    msgs, h = jg.correctness_messages("Q?", "Gold", "Candidate")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert "Q?" in content and "Gold" in content and "Candidate" in content
    assert len(h) == 16  # 16-char hex hash


def test_correctness_messages_hash_changes_with_content():
    _, h1 = jg.correctness_messages("Q?", "Gold", "A")
    _, h2 = jg.correctness_messages("Q?", "Gold", "B")
    assert h1 != h2


def test_fact_messages_format():
    msgs, h = jg.fact_messages("Answer text.", "fact statement")
    assert "Answer text." in msgs[0]["content"]
    assert "fact statement" in msgs[0]["content"]
    assert len(h) == 16


# ---------------------------------------------------------------- response parsers

def test_parse_correctness_yes():
    assert jg.parse_correctness_response('{"reason": "ok", "aligned": "yes"}') is True


def test_parse_correctness_no():
    assert jg.parse_correctness_response('{"reason": "off", "aligned": "no"}') is False


def test_parse_correctness_fallback_text():
    assert jg.parse_correctness_response('"aligned": "yes"') is True
    assert jg.parse_correctness_response('"aligned": "no"') is False


def test_parse_fact_yes():
    assert jg.parse_fact_response("yes") is True
    assert jg.parse_fact_response("Yes, the answer is consistent.") is True


def test_parse_fact_no():
    assert jg.parse_fact_response("no") is False
    assert jg.parse_fact_response("No.") is False


# ---------------------------------------------------------------- combined score (ERB formula)

def test_combined_score_correct():
    assert jg.combined_score(True,  0.5) == pytest.approx(0.5)
    assert jg.combined_score(True,  1.0) == pytest.approx(1.0)


def test_combined_score_incorrect():
    assert jg.combined_score(False, 0.9) == pytest.approx(0.0)
    assert jg.combined_score(False, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------- cache

def test_cache_roundtrip(tmp_path):
    cache = jg.JudgeCache(tmp_path / "cache.jsonl")
    assert not cache.has("run1", "q1", "correct", -1, "h1")
    cache.put("run1", "q1", "correct", -1, "h1", True)
    assert cache.has("run1", "q1", "correct", -1, "h1")
    assert cache.get("run1", "q1", "correct", -1, "h1")["result"] is True


def test_cache_persists_across_reload(tmp_path):
    p = tmp_path / "cache.jsonl"
    c1 = jg.JudgeCache(p)
    c1.put("run1", "q1", "correct", -1, "h1", True)
    c1.put("run1", "q1", "fact",    0,  "h2", False)
    c2 = jg.JudgeCache(p)
    assert c2.has("run1", "q1", "correct", -1, "h1")
    assert c2.has("run1", "q1", "fact",    0,  "h2")
    assert not c2.has("run1", "q1", "fact", 1, "h3")


def test_cache_survives_corrupt_last_line(tmp_path):
    p = tmp_path / "cache.jsonl"
    c = jg.JudgeCache(p)
    c.put("run1", "q1", "correct", -1, "h1", True)
    with open(p, "a") as f:
        f.write("{corrupt\n")   # simulate a crash mid-write
    c2 = jg.JudgeCache(p)
    assert c2.has("run1", "q1", "correct", -1, "h1")


# ---------------------------------------------------------------- dry run

def test_dry_run_counts(tmp_path):
    cache = jg.JudgeCache(tmp_path / "cache.jsonl")
    qs    = {"q1": _q("q1")}  # 1 correctness + 2 facts = 3 calls
    recs  = [_rec("q1")]
    report = jg.dry_run(["run1"], {"run1": recs}, qs, cache)
    assert report["run1"]["pending_correct"] == 1
    assert report["run1"]["pending_facts"]   == 2
    assert report["run1"]["estimated_calls"] == 3
    assert report["run1"]["estimated_tokens"] > 0


def test_dry_run_respects_cache(tmp_path):
    cache = jg.JudgeCache(tmp_path / "cache.jsonl")
    qs    = {"q1": _q("q1")}
    recs  = [_rec("q1")]
    cand  = jg.strip_citations(recs[0]["answer"])
    _, chash = jg.correctness_messages(qs["q1"]["question"], qs["q1"]["gold_answer"], cand)
    cache.put("run1", "q1", "correct", -1, chash, True)
    report = jg.dry_run(["run1"], {"run1": recs}, qs, cache)
    assert report["run1"]["pending_correct"] == 0
    assert report["run1"]["pending_facts"]   == 2


def test_dry_run_respects_limit(tmp_path):
    cache = jg.JudgeCache(tmp_path / "cache.jsonl")
    qs    = {"q1": _q("q1"), "q2": _q("q2")}
    recs  = [_rec("q1"), _rec("q2")]
    report = jg.dry_run(["run1"], {"run1": recs}, qs, cache, limit=1)
    assert report["run1"]["estimated_calls"] == 3   # only q1


# ---------------------------------------------------------------- score_run_direct

def test_score_run_direct_calls_and_caches(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient()
    qs     = {"q1": _q("q1")}
    recs   = [_rec("q1")]
    n = jg.score_run_direct("run1", recs, qs, cache, client)
    # 1 correctness + 2 facts
    assert n == 3
    assert len(client.calls) == 3
    assert cache.has("run1", "q1", "correct", -1,
                     jg._hash(jg._WHOLISTIC_PROMPT.format(
                         question="What is X?",
                         gold_answer="X is 42.",
                         candidate_answer=jg.strip_citations(recs[0]["answer"])
                     )))


def test_score_run_direct_skips_cached(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient()
    qs     = {"q1": _q("q1")}
    recs   = [_rec("q1")]
    jg.score_run_direct("run1", recs, qs, cache, client)
    client.calls.clear()
    n2 = jg.score_run_direct("run1", recs, qs, cache, client)
    assert n2 == 0
    assert len(client.calls) == 0   # nothing re-called


def test_score_run_direct_max_calls_stops_cleanly(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient()
    qs     = {"q1": _q("q1"), "q2": _q("q2")}
    recs   = [_rec("q1"), _rec("q2")]
    with pytest.raises(jg.BudgetExhausted):
        jg.score_run_direct("run1", recs, qs, cache, client, max_calls=2)
    done = cache.count_done("run1")
    assert done == 2  # exactly 2 cached, budget hit


def test_score_run_direct_out_of_credits_stops_cleanly(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient(raise_after=1)   # crash on 2nd call
    qs     = {"q1": _q("q1")}
    recs   = [_rec("q1")]
    with pytest.raises(jg.OutOfCredits):
        jg.score_run_direct("run1", recs, qs, cache, client)
    # 1st call succeeded and was cached; 2nd raised before caching
    assert cache.count_done("run1") == 1


# ---------------------------------------------------------------- resumability: re-run matches full run

def test_resume_matches_uninterrupted(tmp_path):
    """Partial run + resume should equal a complete uninterrupted run."""
    full_cache    = jg.JudgeCache(tmp_path / "full.jsonl")
    partial_cache = jg.JudgeCache(tmp_path / "partial.jsonl")
    full_client   = _FakeClient()
    partial1      = _FakeClient(raise_after=2)
    partial2      = _FakeClient()

    qs   = {"q1": _q("q1"), "q2": _q("q2")}
    recs = [_rec("q1"), _rec("q2")]

    # Full run
    jg.score_run_direct("run1", recs, qs, full_cache, full_client)

    # Interrupted run + resume
    with pytest.raises(jg.OutOfCredits):
        jg.score_run_direct("run1", recs, qs, partial_cache, partial1)
    jg.score_run_direct("run1", recs, qs, partial_cache, partial2)

    # Results must match
    full_js    = jg.build_judgements(["run1"], qs, {"run1": recs}, full_cache)
    partial_js = jg.build_judgements(["run1"], qs, {"run1": recs}, partial_cache)
    assert {r["question_id"]: (r["correct"], r["facts"]) for r in full_js} == \
           {r["question_id"]: (r["correct"], r["facts"]) for r in partial_js}


# ---------------------------------------------------------------- batch resumability

def test_batch_collect_succeeded_results(tmp_path):
    batch_path = tmp_path / "batches.jsonl"
    cache      = jg.JudgeCache(tmp_path / "cache.jsonl")
    qs         = {"q1": _q("q1")}
    recs       = [_rec("q1")]

    # Build the custom_id as score_run would
    cand = jg.strip_citations(recs[0]["answer"])
    msgs_c, chash = jg.correctness_messages(qs["q1"]["question"], qs["q1"]["gold_answer"], cand)
    cid = jg._make_custom_id("run1", "q1", "correct", -1)

    batch_results = [{"custom_id": cid, "type": "succeeded",
                      "text": '{"reason":"ok","aligned":"yes"}'}]
    client = _FakeBatchClient(batch_results)

    bid = jg.submit_batch_run("run1", recs, qs, cache, client, batch_path)
    assert bid is not None

    n_new, failed = jg.collect_batches(cache, client, batch_path)
    assert n_new >= 1
    assert len(failed) == 0
    assert cache.has("run1", "q1", "correct", -1, chash)


def test_batch_errored_requests_are_returned(tmp_path):
    batch_path = tmp_path / "batches.jsonl"
    cache      = jg.JudgeCache(tmp_path / "cache.jsonl")
    qs         = {"q1": _q("q1")}
    recs       = [_rec("q1")]
    cid        = jg._make_custom_id("run1", "q1", "correct", -1)

    batch_results = [{"custom_id": cid, "type": "errored", "text": None}]
    client = _FakeBatchClient(batch_results)

    jg.submit_batch_run("run1", recs, qs, cache, client, batch_path)
    n_new, failed = jg.collect_batches(cache, client, batch_path)
    assert n_new == 0
    assert len(failed) == 1


def test_batch_no_double_submit_on_rerun(tmp_path):
    """Collecting a batch should not re-submit already-cached results."""
    batch_path = tmp_path / "batches.jsonl"
    cache      = jg.JudgeCache(tmp_path / "cache.jsonl")
    qs         = {"q1": _q("q1")}
    recs       = [_rec("q1")]
    cand       = jg.strip_citations(recs[0]["answer"])
    _, chash   = jg.correctness_messages(
        qs["q1"]["question"], qs["q1"]["gold_answer"], cand)
    cid        = jg._make_custom_id("run1", "q1", "correct", -1)

    batch_results = [{"custom_id": cid, "type": "succeeded",
                      "text": '{"reason":"ok","aligned":"yes"}'}]
    client = _FakeBatchClient(batch_results)

    jg.submit_batch_run("run1", recs, qs, cache, client, batch_path)
    n1, _ = jg.collect_batches(cache, client, batch_path)
    n2, _ = jg.collect_batches(cache, client, batch_path)  # re-run
    assert n1 >= 1
    assert n2 == 0   # already cached, nothing new


# ---------------------------------------------------------------- build_judgements

def test_build_judgements_format(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient()
    qs     = {"q1": _q("q1")}
    recs   = [_rec("q1")]
    jg.score_run_direct("run1", recs, qs, cache, client)
    js = jg.build_judgements(["run1"], qs, {"run1": recs}, cache)
    assert len(js) == 1
    row = js[0]
    assert set(row) == {"run", "question_id", "correct", "facts"}
    assert isinstance(row["correct"], bool)
    assert isinstance(row["facts"], list)
    assert all(isinstance(f, bool) for f in row["facts"])
    assert len(row["facts"]) == len(qs["q1"]["answer_facts"])


def test_build_judgements_skips_uncached(tmp_path):
    cache = jg.JudgeCache(tmp_path / "cache.jsonl")
    qs    = {"q1": _q("q1")}
    recs  = [_rec("q1")]
    js    = jg.build_judgements(["run1"], qs, {"run1": recs}, cache)
    assert js == []  # nothing cached


def test_save_load_judgements_roundtrip(tmp_path):
    rows = [
        {"run": "run1", "question_id": "q1", "correct": True,  "facts": [True, False]},
        {"run": "run1", "question_id": "q2", "correct": False, "facts": [False, False]},
    ]
    p = tmp_path / "judgements.jsonl"
    jg.save_judgements(rows, p)
    loaded = jg.load_judgements(p)
    assert loaded == rows


# ---------------------------------------------------------------- aggregate

def test_aggregate_hand_computed(tmp_path):
    """Hand-computed example:
      q1: correct=True,  facts=[True, True]  → comp=1.0, overall=1.0
      q2: correct=False, facts=[True, False] → comp=0.5, overall=0.0
      q3: correct=True,  facts=[False]       → comp=0.0, overall=0.0
    Mean: correctness=2/3≈0.6667, completeness≈0.5, overall≈1/3≈0.3333
    """
    qs = {
        "q1": {**_q("q1"), "answer_facts": ["f1", "f2"]},
        "q2": {**_q("q2"), "answer_facts": ["f1", "f2"]},
        "q3": {**_q("q3"), "answer_facts": ["f1"]},
    }
    recs = [_rec("q1"), _rec("q2"), _rec("q3")]

    # Fake client that returns scripted verdicts
    class _ScriptedClient:
        RESPONSES = {
            # question_id → (correct, fact_results)
            "q1": (True,  [True,  True]),
            "q2": (False, [True,  False]),
            "q3": (True,  [False]),
        }
        def __init__(self): self._counters = {}
        def call_one(self, messages, max_tokens=256):
            # Detect metric from prompt content
            content = messages[0]["content"]
            # Find qid by matching question text in prompt
            for qid, (corr, facts) in self.RESPONSES.items():
                if f"What is X?" in content:
                    # Pick based on which call number we're on for this qid
                    key = qid
                    break
            else:
                return "yes"
            # Use INDIVIDUAL_FACT_VALIDATOR_PROMPT marker
            if "Statement" in content:
                idx = self._counters.get(key, 0)
                self._counters[key] = idx + 1
                return "yes" if self.RESPONSES[key][1][idx] else "no"
            return '{"reason":"ok","aligned":"yes"}' if self.RESPONSES[key][0] else '{"reason":"off","aligned":"no"}'

    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient(responses={
        # Will use default "yes" / "aligned yes" — override in aggregate test below
    })
    # Directly populate cache with hand-computed values
    for tag, (correct, facts) in [("run1", (True,  [True,  True])),
                                   ("run1", (False, [True,  False])),
                                   ("run1", (True,  [False]))]:
        pass  # will do below

    # Populate cache manually for deterministic results
    for qid, (corr, fact_results) in [
        ("q1", (True,  [True,  True])),
        ("q2", (False, [True,  False])),
        ("q3", (True,  [False])),
    ]:
        q    = qs[qid]
        cand = jg.strip_citations(recs[["q1","q2","q3"].index(qid)]["answer"])
        _, chash = jg.correctness_messages(q["question"], q["gold_answer"], cand)
        cache.put("run1", qid, "correct", -1, chash, corr)
        for i, fact in enumerate(q["answer_facts"]):
            _, fhash = jg.fact_messages(cand, fact)
            cache.put("run1", qid, "fact", i, fhash, fact_results[i])

    agg = jg.aggregate("run1", qs, recs, cache)
    assert agg is not None
    assert agg["n"] == 3
    assert agg["correctness"]  == pytest.approx(2/3, abs=0.001)
    assert agg["completeness"] == pytest.approx(0.5, abs=0.001)
    assert agg["overall"]      == pytest.approx(1/3, abs=0.001)


# ---------------------------------------------------------------- Part 2: abstention

def _recs_with_types(specs):
    """specs: list of (qid, question_type, abstained)."""
    qs = {qid: {**_q(qid), "question_type": qt} for qid, qt, _ in specs}
    recs = [_rec(qid, abstained=ab) for qid, _, ab in specs]
    return recs, qs


def test_abstention_metrics_basic():
    recs, qs = _recs_with_types([
        ("q1", "info_not_found", True),
        ("q2", "info_not_found", False),
        ("q3", "basic",          False),
        ("q4", "basic",          True),
    ])
    m = jt.abstention_metrics(recs, qs)
    assert m["n_info_not_found"]   == 2
    assert m["n_answerable"]       == 2
    assert m["n_abstained_inf"]    == 1
    assert m["n_abstained_ans"]    == 1
    assert m["abstention_rate"]    == pytest.approx(0.5)
    assert m["false_refusal_rate"] == pytest.approx(0.5)


def test_abstention_metrics_all_abstained():
    recs, qs = _recs_with_types([
        ("q1", "info_not_found", True),
        ("q2", "info_not_found", True),
    ])
    m = jt.abstention_metrics(recs, qs)
    assert m["abstention_rate"]    == pytest.approx(1.0)
    assert m["false_refusal_rate"] == pytest.approx(0.0)


def test_abstention_metrics_none_abstained():
    recs, qs = _recs_with_types([
        ("q1", "basic", False),
        ("q2", "basic", False),
    ])
    m = jt.abstention_metrics(recs, qs)
    assert m["abstention_rate"]    == pytest.approx(0.0)
    assert m["false_refusal_rate"] == pytest.approx(0.0)


def test_abstention_metrics_by_category():
    recs, qs = _recs_with_types([
        ("q1", "info_not_found",  True),
        ("q2", "conflicting_info", False),
        ("q3", "conflicting_info", True),
    ])
    by_cat = jt.abstention_metrics_by_category(recs, qs)
    assert "info_not_found"  in by_cat
    assert "conflicting_info" in by_cat
    assert by_cat["info_not_found"]["abstention_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------- Part 3: results table

def _abst(abstention_rate=0.8, false_refusal=0.05, n=100):
    return {
        "n_total": n, "n_info_not_found": 20, "n_answerable": n - 20,
        "n_abstained_inf": int(0.8 * 20), "n_abstained_ans": int(false_refusal * (n - 20)),
        "abstention_rate": abstention_rate, "false_refusal_rate": false_refusal,
    }


def _agg(correctness=0.75, completeness=0.80, overall=0.60, n=100, complete=True):
    return {"complete": complete, "n": n,
            "correctness": correctness, "completeness": completeness, "overall": overall}


def test_build_table_contains_key_columns():
    cfg = [{"label": "R1 (BM25)", "agg": _agg(), "abst": _abst()}]
    md  = jt.build_answer_metrics_table(cfg)
    assert "Correctness" in md
    assert "Completeness" in md
    assert "Overall" in md
    assert "Abstention" in md
    assert "False-refusal" in md
    assert "R1 (BM25)" in md


def test_build_table_incomplete_marker():
    cfg = [{"label": "R1", "agg": _agg(complete=False), "abst": _abst()}]
    md  = jt.build_answer_metrics_table(cfg)
    assert "R1 *" in md


def test_build_table_none_agg_shows_dashes():
    cfg = [{"label": "R1", "agg": None, "abst": _abst()}]
    md  = jt.build_answer_metrics_table(cfg)
    assert "—" in md


def test_build_table_section_breaks():
    cfg = [
        {"label": "R1", "agg": _agg(), "abst": _abst()},
        {"label": "R5", "agg": _agg(), "abst": _abst()},
    ]
    md = jt.build_answer_metrics_table(cfg, section_breaks={"R5": "Fusion"})
    assert "**Fusion**" in md


def test_build_table_each_section_is_its_own_table():
    """A heading between rows ends a markdown table, so each section repeats the header."""
    cfg = [{"label": l, "agg": _agg(), "abst": _abst()} for l in ("R1", "R5", "A1")]
    md = jt.build_answer_metrics_table(cfg, section_breaks={"R1": "Main", "A1": "Ablations"})
    lines = md.splitlines()
    assert md.count("| Run | n |") == 2
    for heading in ("**Main**", "**Ablations**"):
        i = lines.index(heading)
        assert lines[i + 1] == "" and lines[i + 2].startswith("| Run | n |")
    # Every row sits directly under a header or another row (no blank line inside a table).
    for i, ln in enumerate(lines):
        if ln.startswith(("| R1 ", "| R5 ", "| A1 ")):
            assert lines[i - 1].startswith("|")


def test_build_table_partial_note_only_when_partial():
    full = jt.build_answer_metrics_table([{"label": "R1", "agg": _agg(), "abst": _abst()}])
    part = jt.build_answer_metrics_table(
        [{"label": "R1", "agg": _agg(complete=False), "abst": _abst()}])
    assert "not fully judged" not in full and "not fully judged" in part


def test_build_table_rate_over_no_questions_is_a_dash():
    abst = {**_abst(), "n_info_not_found": 0, "abstention_rate": 0.0}
    md = jt.build_answer_metrics_table([{"label": "CI", "agg": _agg(), "abst": abst}])
    row = next(l for l in md.splitlines() if l.startswith("| CI"))
    assert row.endswith("| — | 5.0% |")


def test_write_answer_metrics_table(tmp_path):
    p  = tmp_path / "results" / "answer_metrics.md"
    md = jt.build_answer_metrics_table(
        [{"label": "R1", "agg": _agg(), "abst": _abst()}])
    jt.write_answer_metrics_table(md, p)
    assert p.exists()
    assert "R1" in p.read_text()


# ---------------------------------------------------------------- Part 4: ALCE citation check

def _passage(pid, text):
    doc = pid.rsplit("#", 1)[0]
    return {"passage_id": pid, "doc_id": doc, "source": "slack", "title": "T",
            "channel": "eng", "author": "A", "date": "2026-01-01", "created": None,
            "chunk_idx": int(pid[-3:]), "n_chunks": 2, "n_tokens": 10,
            "header": f"[slack] T", "body": text, "text": f"[slack] T\n\n{text}"}


def _rec_with_sources(qid="q1",
                      answer="The document confirms the answer is correct [1]. This other fact is also true.",
                      abstained=False):
    return {
        "question_id": qid,
        "answer": answer,
        "abstained": abstained,
        "sources": [
            {"n": 1, "doc_id": "dsid_abc", "passage_ids": ["dsid_abc#000"],
             "window_ids": ["dsid_abc#000"]}
        ],
    }


def test_sentence_passages_extracts_cited_sentences():
    by_pid = {"dsid_abc#000": _passage("dsid_abc#000", "passage body")}
    rec = _rec_with_sources()
    sents = jc._sentence_passages(rec["answer"], rec["sources"], by_pid)
    assert len(sents) >= 1
    assert all(len(s["passage_texts"]) > 0 for s in sents)
    # Uncited sentence should not appear
    assert all("Sentence two" not in s["sentence"] or s["passage_texts"] for s in sents)


def test_sentence_passages_skips_short_sentences():
    by_pid = {"dsid_abc#000": _passage("dsid_abc#000", "body")}
    rec = {**_rec_with_sources(), "answer": "Ok [1]. This is a real sentence here. [1]"}
    sents = jc._sentence_passages(rec["answer"], rec["sources"], by_pid)
    # "Ok" < 20 chars → skipped
    assert all(len(s["sentence"]) >= 20 for s in sents)


def test_score_citation_support_calls_and_caches(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient(responses={"You are a citation": "yes"})
    by_pid = {"dsid_abc#000": _passage("dsid_abc#000", "The answer is 42.")}
    recs   = [_rec_with_sources("q1",
              "X is exactly forty-two because of physics. [1] More context supports this claim. [1]")]
    n = jc.score_citation_support("run1", recs, by_pid, cache, client)
    assert n >= 1
    # Re-run should make no new calls
    n2 = jc.score_citation_support("run1", recs, by_pid, cache, client)
    assert n2 == 0


def test_score_citation_support_skips_abstained(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _FakeClient()
    by_pid = {"dsid_abc#000": _passage("dsid_abc#000", "body")}
    recs   = [_rec_with_sources("q1", abstained=True)]
    n = jc.score_citation_support("run1", recs, by_pid, cache, client)
    assert n == 0


def test_citation_support_aggregate_none_when_nothing_cached(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    by_pid = {"dsid_abc#000": _passage("dsid_abc#000", "body")}
    recs   = [_rec_with_sources()]
    result = jc.citation_support_aggregate("run1", recs, by_pid, cache)
    assert result is None


def _window_rec(qid="qst_0001", answer="The launch moved to March per the notes. [1]", abstained=False):
    """One source whose hit is #001, with neighbours #000 and #002 in its window."""
    return {"question_id": qid, "answer": answer, "abstained": abstained,
            "sources": [{"n": 1, "doc_id": "dsid_abc", "passage_ids": ["dsid_abc#001"],
                         "window_ids": ["dsid_abc#000", "dsid_abc#001", "dsid_abc#002"]}]}


def _window_pids():
    return {f"dsid_abc#00{i}": {**_passage(f"dsid_abc#00{i}", body), "n_chunks": 3}
            for i, body in enumerate(["Intro text.", "Hit text.", "Launch moved to March."])}


def test_alce_judge_sees_the_whole_window_not_only_the_hit():
    # The supporting text is in a neighbour (#002): the generator saw it, so the judge must too.
    sents = jc._sentence_passages(_window_rec()["answer"], _window_rec()["sources"], _window_pids())
    assert len(sents) == 1 and len(sents[0]["passage_texts"]) == 1
    text = sents[0]["passage_texts"][0]
    assert text.startswith("[slack] T\n")
    assert "Intro text." in text and "Hit text." in text and "Launch moved to March." in text


def test_alce_unclear_reply_is_not_cached_as_no(tmp_path):
    cache = jg.JudgeCache(tmp_path / "c.jsonl")
    client = _FakeClient(responses={"You are a citation": "Not entirely; the passage says..."})
    assert jc.score_citation_support("run1", [_window_rec()], _window_pids(), cache, client) == 1
    assert jc.citation_support_counts("run1", [_window_rec()], _window_pids(), cache) == \
        {"supported": 0, "judged": 0, "checks": 1, "answers": 1}


def test_alce_custom_ids_do_not_collide_with_fact_ids():
    a = jg._make_custom_id("gen_r1_r4_p7", "qst_0001", "alce", 0)
    assert a != jg._make_custom_id("gen_r1_r4_p7", "qst_0001", "fact", 0)
    assert jg.CUSTOM_ID_RE.match(a) and jg._parse_custom_id(a) == ("gen_r1_r4_p7", "qst_0001", "alce", 0)


def test_alce_sample_is_shared_seeded_and_answered_only():
    recs = {"a": [_window_rec(f"qst_{i:04d}", abstained=(i % 5 == 0)) for i in range(50)],
            "b": [_window_rec(f"qst_{i:04d}", abstained=(i % 7 == 0)) for i in range(50)]}
    s = jc.alce_sample(recs, ["a", "b"], n=10, seed=1)
    assert len(s) == 10 and s == jc.alce_sample(recs, ["a", "b"], n=10, seed=1)
    assert all(int(q[-4:]) % 5 and int(q[-4:]) % 7 for q in s)


def test_alce_batch_round_trip_with_strict_settle(tmp_path):
    cache, bp = jg.JudgeCache(tmp_path / "c.jsonl"), tmp_path / "b.jsonl"
    recs = [_window_rec("qst_0001"),
            _window_rec("qst_0002", "First claim about the launch date here. [1] Second claim "
                                    "about the March launch here. [1]")]
    pids, client = _window_pids(), _FakeBatchClient([])
    pending = jc.alce_pending("run1", recs, pids, cache)
    assert len(pending) == 3 and jc.alce_cost(pending)["usd"] < 0.01
    assert jc.submit_alce_batch("run1", recs, pids, cache, client, bp) == "batch-001"
    assert jc.alce_batch_ids(bp) == {"batch-001"}
    replies = {"run1-qst_0001-a0": "yes", "run1-qst_0002-a0": "No.",
               "run1-qst_0002-a1": "The passage mentions March, so..."}
    client._batch_results = [{"custom_id": c, "type": "succeeded", "text": t} for c, t in replies.items()]
    n_new, failed = jg.collect_batches(cache, client, bp, batch_ids=jc.alce_batch_ids(bp))
    assert n_new == 2 and [r["custom_id"] for r in failed] == ["run1-qst_0002-a1"]
    assert jg.rejudge_direct(failed, {"run1": recs}, {}, cache, _StrictClient()) == (0, failed)
    strict = _StrictClient(verdict=True)
    assert jc.rejudge_alce(failed, {"run1": recs}, pids, cache, strict) == (1, [])
    assert strict.strict_calls == ["alce"]
    assert jc.citation_support_counts("run1", recs, pids, cache) == \
        {"supported": 2, "judged": 3, "checks": 3, "answers": 2}
    assert jc.submit_alce_batch("run1", recs, pids, cache, client, bp) is None   # all cached


def test_collect_only_named_batches(tmp_path):
    cache, bp, client, qs, recs = _batch_setup(
        tmp_path, {"c": '{"aligned": "yes"}', "f0": "yes", "f1": "no"})
    assert jg.collect_batches(cache, client, bp, batch_ids=set()) == (0, [])
    assert jg.collect_batches(cache, client, bp)[0] == 3


def test_ragas_faithfulness_skipped_by_default(tmp_path):
    cache  = jg.JudgeCache(tmp_path / "cache.jsonl")
    by_pid = {"dsid_abc#000": _passage("dsid_abc#000", "body")}
    result = jc.ragas_faithfulness("run1", [_rec_with_sources()], by_pid, cache,
                                   _FakeClient(), ragas=False)
    assert result is None


# ---------------------------------------------------------------- unusable replies (Sonnet 5.5)

class _Block:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


class _Msg:
    def __init__(self, content, stop_reason="end_turn"):
        self.content, self.stop_reason = content, stop_reason


def test_response_text_joins_text_blocks_and_skips_others():
    msg = _Msg([_Block("thinking"), _Block("text", '{"aligned": '), _Block("text", '"yes"}')])
    assert jg.response_text(msg) == '{"aligned": "yes"}'


@pytest.mark.parametrize("msg", [
    _Msg([], stop_reason="refusal"),
    _Msg([_Block("text", "yes")], stop_reason="refusal"),
    _Msg([_Block("text", '{"reason": "cut')], stop_reason="max_tokens"),
    _Msg([_Block("thinking")]),
    _Msg([_Block("text", "   ")]),
])
def test_response_text_none_when_nothing_usable(msg):
    assert jg.response_text(msg) is None


def test_judge_settings_send_no_temperature():
    assert jg.JUDGE_MODEL == "claude-sonnet-5-5"
    assert jg.JUDGE_THINKING == {"type": "between_tools"}


def test_direct_unusable_reply_is_not_cached_and_retried(tmp_path):
    class _NoText(_FakeClient):
        def call_one(self, messages, max_tokens=256):
            self.calls.append(messages)
            return None

    cache = jg.JudgeCache(tmp_path / "cache.jsonl")
    client = _NoText()
    n = jg.score_run_direct("run1", [_rec("q1")], {"q1": _q("q1")}, cache, client)
    assert n == 3                      # 1 correctness + 2 facts, all counted against the budget
    _, chash = jg.correctness_messages("What is X?", "X is 42.", jg.strip_citations("X is 42 [1]."))
    assert not cache.has("run1", "q1", "correct", -1, chash)
    # A later run with a working client fills them in.
    jg.score_run_direct("run1", [_rec("q1")], {"q1": _q("q1")}, cache, _FakeClient())
    assert cache.has("run1", "q1", "correct", -1, chash)


def test_batch_results_marks_unusable_succeeded_reply_as_no_text():
    from types import SimpleNamespace as NS

    def res(cid, typ, msg=None):
        return NS(custom_id=cid, result=NS(type=typ, message=msg))

    sdk_results = [res("a", "succeeded", _Msg([_Block("text", "yes")])),
                   res("b", "succeeded", _Msg([], stop_reason="refusal")),
                   res("c", "errored")]
    client = jg.JudgeClient.__new__(jg.JudgeClient)   # skip __init__: no SDK or key needed
    client._client = NS(beta=NS(messages=NS(batches=NS(results=lambda bid: iter(sdk_results)))))
    out = list(client.batch_results("batch-1"))
    assert out == [{"custom_id": "a", "type": "succeeded", "text": "yes"},
                   {"custom_id": "b", "type": "no_text", "text": None},
                   {"custom_id": "c", "type": "errored", "text": None}]


# ---------------------------------------------------------------- batch custom_ids

def test_custom_ids_match_the_batches_api_pattern():
    # The real run tags and question ids: the API rejected the old "run|||qid|||..." ids.
    for run in ("gen_r1_p7", "gen_r5_r6_p7_n0", "gen_r1_r4_p7"):
        for metric, i in (("correct", -1), ("fact", 0), ("fact", 45)):
            cid = jg._make_custom_id(run, "qst_0123", metric, i)
            assert jg.CUSTOM_ID_RE.match(cid), cid
            assert jg._parse_custom_id(cid) == (run, "qst_0123", metric, i)


def test_custom_ids_unique_across_runs_and_facts():
    ids = {jg._make_custom_id(r, q, m, i)
           for r in ("gen_r1_p7", "gen_r1_p7_n0") for q in ("qst_0001", "qst_0002")
           for m, i in (("correct", -1), ("fact", 0), ("fact", 1))}
    assert len(ids) == 2 * 2 * 3


@pytest.mark.parametrize("run,qid", [("gen_r1_p7_qwen3.5-9b", "qst_0001"),   # "." and "-"
                                     ("gen_r1_p7", "q|1"), ("x" * 60, "qst_0001")])
def test_custom_id_rejects_what_the_api_would(run, qid):
    with pytest.raises(ValueError):
        jg._make_custom_id(run, qid, "correct", -1)


def test_submitted_batch_ids_are_all_valid(tmp_path):
    client = _FakeBatchClient([])
    qs = {"qst_0001": _q("qst_0001")}
    jg.submit_batch_run("gen_r5_r6_p7_n0", [_rec("qst_0001")], qs,
                        jg.JudgeCache(tmp_path / "c.jsonl"), client, tmp_path / "b.jsonl")
    assert client._submitted and all(jg.CUSTOM_ID_RE.match(r["custom_id"]) for r in client._submitted)


# ---------------------------------------------------------------- no double submission

class _SlowBatchClient(_FakeBatchClient):
    """Batches stay in_progress until .finish() is called; counts submissions."""
    def __init__(self, **kw):
        super().__init__([], **kw)
        self.status, self.n_submits = "in_progress", 0
    def submit_batch(self, requests):
        self.n_submits += 1
        self._submitted = requests
        return f"batch-{self.n_submits}"
    def batch_status(self, batch_id):
        return self.status


def test_rerun_while_batch_in_progress_does_not_resubmit(tmp_path):
    cache, bp = jg.JudgeCache(tmp_path / "c.jsonl"), tmp_path / "b.jsonl"
    qs, recs = {"qst_0001": _q("qst_0001")}, [_rec("qst_0001")]
    client = _SlowBatchClient()
    assert jg.submit_batch_run("run1", recs, qs, cache, client, bp) == "batch-1"
    jg.collect_batches(cache, client, bp)                         # still in progress
    assert jg.submit_batch_run("run1", recs, qs, cache, client, bp) is None
    assert client.n_submits == 1


def test_failed_requests_resubmitted_once_batch_has_ended(tmp_path):
    cache, bp = jg.JudgeCache(tmp_path / "c.jsonl"), tmp_path / "b.jsonl"
    qs, recs = {"qst_0001": _q("qst_0001")}, [_rec("qst_0001")]
    client = _SlowBatchClient()
    jg.submit_batch_run("run1", recs, qs, cache, client, bp)
    client.status = "ended"                                        # ended, but nothing succeeded
    assert jg.submit_batch_run("run1", recs, qs, cache, client, bp) == "batch-2"


def test_status_error_counts_as_in_flight(tmp_path):
    cache, bp = jg.JudgeCache(tmp_path / "c.jsonl"), tmp_path / "b.jsonl"
    qs, recs = {"qst_0001": _q("qst_0001")}, [_rec("qst_0001")]
    client = _SlowBatchClient()
    jg.submit_batch_run("run1", recs, qs, cache, client, bp)
    client.batch_status = lambda bid: (_ for _ in ()).throw(RuntimeError("network"))
    assert jg.submit_batch_run("run1", recs, qs, cache, client, bp) is None


def test_other_runs_batches_do_not_block(tmp_path):
    cache, bp = jg.JudgeCache(tmp_path / "c.jsonl"), tmp_path / "b.jsonl"
    qs, recs = {"qst_0001": _q("qst_0001")}, [_rec("qst_0001")]
    client = _SlowBatchClient()
    jg.submit_batch_run("run1", recs, qs, cache, client, bp)
    assert jg.submit_batch_run("run2", recs, qs, cache, client, bp) == "batch-2"


# ---------------------------------------------------------------- strict verdicts (507 "other" replies)

@pytest.mark.parametrize("text,want", [
    ("yes", True), ("Yes.", True), ("**Yes**", True), ("> no", False), ("No, it doesn't.", False),
    ("`yes`", True),
    ("Not quite: the answer omits it.", None), ("Note: the answer says 42.", None),
    ("None of the answer covers it.", None), ("The answer states X, so yes.", None),
    ("Looking at the statement... no.", None), ("", None),
])
def test_parse_fact_response_reads_only_a_first_word_yes_or_no(text, want):
    assert jg.parse_fact_response(text) is want


@pytest.mark.parametrize("text,want", [
    ('{"reason": "ok", "aligned": "yes"}', True),
    ('```json\n{"reason": "off", "aligned": "no"}\n```', False),
    ('Here is my verdict: {"reason": "ok", "aligned": "Yes"}', True),
    ('{"reason": "ok"}', None), ("aligned, mostly", None), ("yes", None),
])
def test_parse_correctness_response_needs_an_aligned_field(text, want):
    assert jg.parse_correctness_response(text) is want


def test_cache_drop_survives_reload(tmp_path):
    cache = jg.JudgeCache(tmp_path / "c.jsonl")
    cache.put("run1", "q1", "fact", 0, "h", False)
    cache.drop("run1", "q1", "fact", 0, "h")
    assert not cache.has("run1", "q1", "fact", 0, "h")
    assert not jg.JudgeCache(tmp_path / "c.jsonl").has("run1", "q1", "fact", 0, "h")
    cache.put("run1", "q1", "fact", 0, "h", True)           # re-adding after a drop works
    assert jg.JudgeCache(tmp_path / "c.jsonl").get("run1", "q1", "fact", 0, "h")["result"] is True


def _batch_setup(tmp_path, texts):
    """Submit run1/qst_0001 (1 correctness + 2 facts) and script the batch replies by custom_id suffix."""
    cache, bp = jg.JudgeCache(tmp_path / "c.jsonl"), tmp_path / "b.jsonl"
    qs, recs = {"qst_0001": _q("qst_0001")}, [_rec("qst_0001")]
    client = _FakeBatchClient([])
    jg.submit_batch_run("run1", recs, qs, cache, client, bp)
    client._batch_results = [
        {"custom_id": r["custom_id"], "type": "succeeded",
         "text": texts[r["custom_id"].rsplit("-", 1)[1]]}
        for r in client._submitted]
    return cache, bp, client, qs, recs


def test_collect_treats_no_verdict_as_failed_not_no(tmp_path):
    cache, bp, client, qs, recs = _batch_setup(
        tmp_path, {"c": '{"aligned": "yes"}', "f0": "yes", "f1": "The answer says 42, so yes."})
    n_new, failed = jg.collect_batches(cache, client, bp)
    assert n_new == 2 and [r["custom_id"] for r in failed] == ["run1-qst_0001-f1"]


def test_reparse_fixes_verdicts_cached_by_the_old_parser(tmp_path):
    cache, bp, client, qs, recs = _batch_setup(
        tmp_path, {"c": '{"aligned": "yes"}', "f0": "Note: it says 42.", "f1": "yes"})
    sub = {r["custom_id"]: r for r in client._submitted}
    # What the old parser cached: "Note..." -> False, and f1 wrongly False too.
    for cid in ("run1-qst_0001-f0", "run1-qst_0001-f1"):
        r = sub[cid]
        cache.put("run1", "qst_0001", "fact", r["fact_idx"], r["prompt_hash"], False)
    n_changed, n_dropped, unusable = jg.reparse_batches(cache, client, bp)
    assert (n_changed, n_dropped) == (2, 1)       # correctness added + f1 flipped; f0 dropped
    assert [r["custom_id"] for r in unusable] == ["run1-qst_0001-f0"]
    f1 = sub["run1-qst_0001-f1"]
    assert cache.get("run1", "qst_0001", "fact", 1, f1["prompt_hash"])["result"] is True
    assert jg.reparse_batches(cache, client, bp)[:2] == (0, 0)   # idempotent


class _StrictClient(_FakeClient):
    def __init__(self, verdict=True):
        super().__init__()
        self.verdict, self.strict_calls = verdict, []
    def call_strict(self, messages, metric, max_tokens=1024):
        self.strict_calls.append(metric)
        return self.verdict


def test_rejudge_direct_settles_and_tags_strict_verdicts(tmp_path):
    cache, bp, client, qs, recs = _batch_setup(
        tmp_path, {"c": '{"aligned": "yes"}', "f0": "Explaining at length...", "f1": "no"})
    _, _, unusable = jg.reparse_batches(cache, client, bp)
    strict = _StrictClient(verdict=True)
    n, still = jg.rejudge_direct(unusable, {"run1": recs}, qs, cache, strict)
    assert (n, still, strict.strict_calls) == (1, [], ["fact"])
    f0 = next(r for r in client._submitted if r["custom_id"].endswith("-f0"))
    entry = cache.get("run1", "qst_0001", "fact", 0, f0["prompt_hash"])
    assert entry["result"] is True and entry["model"] == jg.STRICT_MODEL_TAG
    # A later reparse keeps the strict verdict instead of dropping it again.
    assert jg.reparse_batches(cache, client, bp)[2] == []
    assert cache.has("run1", "qst_0001", "fact", 0, f0["prompt_hash"])


def test_rejudge_direct_leaves_unsettled_and_hash_mismatches(tmp_path):
    cache, bp, client, qs, recs = _batch_setup(
        tmp_path, {"c": "no verdict here", "f0": "yes", "f1": "hmm"})
    _, _, unusable = jg.reparse_batches(cache, client, bp)
    n, still = jg.rejudge_direct(unusable, {"run1": recs}, qs, cache, _StrictClient(verdict=None))
    assert n == 0 and len(still) == 2
    changed = [{**recs[0], "answer": "a different answer"}]   # answer changed since submission
    n, still = jg.rejudge_direct(unusable, {"run1": changed}, qs, cache, _StrictClient())
    assert n == 0 and len(still) == 2
