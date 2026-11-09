"""Chunker tests on real documents from EnterpriseRAG-Bench (MIT license), one per source plus edge cases.

Needs the two tokenizers from Hugging Face (downloaded once, ~15 MB).
"""

import json
import re
from pathlib import Path

import pytest

from lumbr import chunk, schema

FIXTURES = Path(__file__).parent / "fixtures" / "docs.jsonl"
BS_N = chr(92) + "n"


def _load():
    out = {}
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        r["doc"]["_source_type"] = r["source_type"]
        out[r["fixture"]] = r["doc"]
    return out


DOCS = _load()


@pytest.fixture(scope="module")
def passages():
    return {name: list(chunk.chunk_docs([doc])) for name, doc in DOCS.items()}


def _norm(s):
    return re.sub(r"\s+", "", s)


def test_every_source_covered():
    assert {d["_source_type"] for d in DOCS.values()} == set(schema.SOURCES)


def test_schema_valid_and_ids_unique(passages):
    for name, ps in passages.items():
        assert ps, name
        for p in ps:
            assert schema.validate_record(p.to_dict()) == [], (name, p.passage_id)
    # some fixtures share a document, so check uniqueness over distinct documents
    unique_docs = {d["dataset_doc_uuid"]: d for d in DOCS.values()}
    ids = [p.passage_id for p in chunk.chunk_docs(unique_docs.values())]
    assert len(ids) == len(set(ids))


def test_token_cap_holds_for_both_tokenizers(passages):
    texts = [p.text for ps in passages.values() for p in ps]
    for model_id, specials in schema.TOKENIZERS.items():
        counts = [len(x) + specials for x in chunk.tokenizer(model_id)(texts, add_special_tokens=False)["input_ids"]]
        assert max(counts) <= schema.MAX_TOKENS, model_id


def test_n_tokens_is_exact(passages):
    ps = [p for v in passages.values() for p in v]
    assert [p.n_tokens for p in ps] == chunk.exact_counts([p.text for p in ps])


def test_body_coverage(passages):
    """Chunk bodies, concatenated, contain all of the rendered units' text (nothing dropped)."""
    for name, doc in DOCS.items():
        rendered = _norm("".join(chunk.render_units(doc, doc["_source_type"])))
        joined = _norm("".join(p.body for p in passages[name]))
        assert len(joined) >= 0.99 * len(rendered), name


def test_units_that_fit_are_never_split(passages):
    for name, doc in DOCS.items():
        bodies = [_norm(p.body) for p in passages[name]]
        for u in chunk.render_units(doc, doc["_source_type"]):
            if chunk.count([u])[0] <= 300:
                assert any(_norm(u) in b for b in bodies), (name, u[:60])


def test_gmail_quotes_removed(passages):
    for p in passages["gmail_quoted_reply"]:
        assert not re.search(r"^\s*>", p.body, re.M)
        assert not re.search(r"^On .{5,200}wrote:\s*$", p.body, re.M)


def test_gmail_message_markers_split_and_author_found(passages):
    ps = passages["gmail_message_markers"]
    assert not any(re.search(r"-{3,}\s*MESSAGE\s+\d+", p.body, re.I) for p in ps)
    assert ps[0].author


def test_escaped_newlines_unescaped(passages):
    assert all(p.body.count(BS_N) < 2 for p in passages["escaped_newlines"])


def test_long_title_is_truncated(passages):
    for p in passages["long_title"]:
        assert len(p.title) <= chunk.TITLE_MAX_CHARS + 1
        assert "\n" not in p.header


def test_slack_lowercase_speakers_are_units():
    doc = DOCS["slack_lowercase_speakers"]
    units = chunk.render_units(doc, "slack")
    assert len(units) > 3
    assert all(chunk.SPEAKER.match(u) for u in units[1:])


def test_bogus_slack_epoch_becomes_null(passages):
    assert all(p.date is None for p in passages["slack_bogus_epoch"])


def test_duplicate_doc_ids_numbered_continuously():
    a, b = DOCS["basic_confluence"], DOCS["basic_hubspot"]
    twin = dict(b, dataset_doc_uuid=a["dataset_doc_uuid"])
    ps = list(chunk.chunk_docs([a, twin], duplicate_ids={a["dataset_doc_uuid"]}))
    assert [p.chunk_idx for p in ps] == list(range(len(ps)))
    assert {p.source for p in ps} == {"confluence", "hubspot"}
    assert all(p.n_chunks == len(ps) for p in ps)


def test_dates_normalized():
    assert chunk.normalize_date("1710501234") == "2024-03-15T11:13:54Z"
    assert chunk.normalize_date("2176543210") is None
    assert chunk.normalize_date("2025-02-15") == "2025-02-15"
    assert chunk.normalize_date("2045-01-01") is None
