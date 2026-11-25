"""The parse cases committed for the site still match lumbr.generate.parse_answer."""
import json
from pathlib import Path

import export_parse_cases as epc
from lumbr import generate

CASES = Path(__file__).resolve().parents[2] / "demo" / "content" / "parse_cases.json"


def test_parse_cases_match_python():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    prompt = generate.load_prompt()
    assert data["prompt_sha256"] == prompt["sha256"] and data["abstain"] == prompt["abstain"]
    assert len(data["cases"]) >= 12
    for c in data["cases"]:
        got = generate.parse_answer(c["text"], epc.sources(c["n_sources"]), prompt)
        assert [x["n"] for x in got["citations"]] == c["citations"], c["text"]
        assert got["invalid_citations"] == c["invalid_citations"], c["text"]
        assert got["abstained"] == c["abstained"], c["text"]


def test_cases_cover_both_outcomes():
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    assert any(c["abstained"] for c in cases) and any(not c["abstained"] for c in cases)
    assert any(c["invalid_citations"] for c in cases)


def test_exports_the_refusal_patterns():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    assert data["refusal_patterns"] == list(generate.REFUSAL_PATTERNS)
    assert data["abstain_rule"] == generate.ABSTAIN_RULE


def test_cases_cover_the_lead_rule():
    texts = {c["text"]: c["abstained"] for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    for t, want in epc.LEAD_CASES:
        assert texts[t] is want, t
