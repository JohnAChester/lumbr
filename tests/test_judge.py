"""Unit and resumability tests for src/lumbr/judge.py.

No network calls: all judge calls go through _FakeClient.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumbr import judge as jg


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
        # Default: return JSON "yes" for correctness or plain "yes" for facts
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
