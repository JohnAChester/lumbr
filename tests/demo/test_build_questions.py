import pandas as pd
import pytest

import build_questions as bq


def _qs() -> pd.DataFrame:
    return pd.DataFrame([
        {"question_id": "q1", "question_type": "basic", "question": "Q1?", "expected_doc_ids": ["d1"],
         "gold_answer": "A1", "answer_facts": ["f1"]},
        {"question_id": "q2", "question_type": "basic", "question": "Q2?", "expected_doc_ids": ["d1", "dx"],
         "gold_answer": "A2", "answer_facts": ["f2"]},
        {"question_id": "q3", "question_type": "high_level", "question": "Q3?", "expected_doc_ids": [],
         "gold_answer": "A3", "answer_facts": ["f3"]},
        {"question_id": "q4", "question_type": "info_not_found", "question": "Q4?", "expected_doc_ids": [],
         "gold_answer": "The answer must state ...", "answer_facts": ["f4"]},
        {"question_id": "t1", "question_type": "basic", "question": "T1?", "expected_doc_ids": ["d1"],
         "gold_answer": "A", "answer_facts": ["f"]},
    ])


def test_missing_split_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="dev split"):
        bq.load_dev_ids(tmp_path / "nope.txt")


def test_selects_only_dev():
    got = bq.select_questions(_qs(), {"q1", "q2", "q3", "q4"}, {"d1"})
    assert "t1" not in set(got.question_id)


def test_partial_gold_excluded():
    got = bq.select_questions(_qs(), {"q1", "q2"}, {"d1"})
    assert list(got.question_id) == ["q1"]


def test_high_level_without_gold_excluded():
    assert "q3" not in set(bq.select_questions(_qs(), {"q3", "q4"}, {"d1"}).question_id)


def test_info_not_found_included_with_no_sources():
    row = bq.select_questions(_qs(), {"q4"}, {"d1"}).iloc[0]
    q = bq.to_demo(row, dm="hey quick one", colleague="Ana", headers={})
    assert q["infoNotFound"] is True and q["goldSources"] == [] and q["cached"] is None


def test_to_demo_shape():
    row = bq.select_questions(_qs(), {"q1"}, {"d1"}).iloc[0]
    q = bq.to_demo(row, dm="yo", colleague="Ben", headers={"d1": "[slack] h"})
    assert q == {"id": "q1", "category": "basic", "infoNotFound": False, "colleague": "Ben", "dm": "yo",
                 "question": "Q1?", "answerFacts": ["f1"], "goldAnswer": "A1",
                 "goldSources": [{"doc_id": "d1", "header": "[slack] h"}], "cached": None}


def test_missing_dm_row_fails():
    sel = bq.select_questions(_qs(), {"q1", "q4"}, {"d1"})
    with pytest.raises(SystemExit, match="q4"):
        bq.build(sel, dms={"q1": ("Ben", "yo")}, headers={"d1": "h"})
