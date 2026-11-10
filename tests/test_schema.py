import json

import pytest

from lumbr import schema


def _record(**over):
    header = schema.make_header("slack", "eng-infra", channel="eng-infra", author="Alex", date="2024-03-15T11:20:34Z")
    body = "Alex: Quick cost comp across regions."
    d = dict(
        passage_id="dsid_207184bb522446f7b7307262d56b6940#000",
        doc_id="dsid_207184bb522446f7b7307262d56b6940",
        source="slack", title="eng-infra", channel="eng-infra", author="Alex",
        date="2024-03-15T11:20:34Z", created="2024-03-15T11:20:34Z",
        chunk_idx=0, n_chunks=2, n_tokens=30,
        header=header, body=body, text=schema.join_text(header, body),
    )
    d.update(over)
    return d


def test_header_formats():
    assert schema.make_header("slack", "eng-infra", "eng-infra", "Alex", "2024-03-15T11:20:34Z") == \
        "[slack] #eng-infra · Alex · 2024-03-15"
    assert schema.make_header("confluence", "ADR-017", "architecture", "Ethan Park", None) == \
        "[confluence] architecture · ADR-017 · Ethan Park"
    assert schema.make_header("gmail", "Re: Beta plan") == "[gmail] Re: Beta plan"


def test_ids_round_trip():
    pid = schema.make_passage_id("dsid_abc", 7)
    assert pid == "dsid_abc#007"
    assert schema.doc_id_of(pid) == "dsid_abc"


def test_valid_record_passes():
    assert schema.validate_record(_record()) == []


@pytest.mark.parametrize("over, msg", [
    ({"n_tokens": 513}, "n_tokens"),
    ({"passage_id": "dsid_207184bb522446f7b7307262d56b6940#1"}, "passage_id"),
    ({"source": "teams"}, "unknown source"),
    ({"text": "no header"}, "text must equal"),
    ({"author": 3}, "author must be"),
    ({"chunk_idx": 2}, "chunk_idx out of range"),
])
def test_invalid_records_caught(over, msg):
    errs = schema.validate_record(_record(**over))
    assert any(msg in e for e in errs), errs


def test_validate_jsonl_catches_duplicates(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(_record()) + "\n" + json.dumps(_record()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        schema.validate_jsonl(p)


def test_validate_jsonl_summary(tmp_path):
    p = tmp_path / "x.jsonl"
    schema.write_jsonl([_record(), _record(chunk_idx=1, passage_id="dsid_207184bb522446f7b7307262d56b6940#001")], p)
    assert schema.validate_jsonl(p) == {"passages": 2, "docs": 1, "schema_version": 1}
