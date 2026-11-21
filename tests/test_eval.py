"""Retrieval eval tests (issue #7): hand-computed expected values, no Drive required.

Fixture layout (5 questions, 3 evaluated):

  q1  basic           expected=[DOC_A]  gold at rank 1  → Recall=1, nDCG=1,         MRR=1
  q2  basic           expected=[DOC_B]  gold at rank 2  → Recall=1, nDCG=1/log2(3), MRR=1/2
  q3  basic           expected=[DOC_C]  not retrieved   → Recall=0, nDCG=0,          MRR=0
  q4  high_level      expected=[]       excluded (no gold docs)
  q5  info_not_found  expected=[]       excluded (no gold docs)

Filtering is by len(expected_doc_ids) > 0, not by category name, so q4 and q5 are
excluded regardless of their question_type label.

Hand-computed expected values (3 included questions):
  mean Recall@10   = (1+1+0)/3  = 2/3
  mean nDCG@10     = (1 + 1/log2(3) + 0)/3  ≈ 0.5436
  mean MRR@10      = (1+1/2+0)/3 = 1/2

  Run after passages_to_docs (10 unique docs per question):
    q1 top-5: DOC_A + 4 irrelevant  →  extra_docs_at_5  = 4
    q2 top-5: 1 irrel + DOC_B + 3   →  extra_docs_at_5  = 4
    q3 top-5: 5 irrelevant           →  extra_docs_at_5  = 5
    mean extra_docs_at_5  = (4+4+5)/3 = 13/3

    q1 top-10: DOC_A + 9 irrelevant  →  extra_docs_at_10 = 9
    q2 top-10: DOC_B + 9 irrelevant  →  extra_docs_at_10 = 9
    q3 top-10: 10 irrelevant          →  extra_docs_at_10 = 10
    mean extra_docs_at_10 = (9+9+10)/3 = 28/3

  gold_unindexed (sanity check, indexed = {DOC_A, DOC_B}, DOC_C absent):
    q1=0, q2=0, q3=1  →  mean = 1/3
"""

import json
import math

import pandas as pd
import pytest

ranx = pytest.importorskip("ranx")

from lumbr import eval as ev, schema

# ---------------------------------------------------------------- shared fixtures

DOC_A = "dsid_" + "a" * 32   # q1 gold
DOC_B = "dsid_" + "b" * 32   # q2 gold
DOC_C = "dsid_" + "c" * 32   # q3 gold (never retrieved)
DOC_D = "dsid_" + "d" * 32   # q4 gold — but expected=[], so excluded
DOC_E = "dsid_" + "e" * 32   # q5 gold — but expected=[], so excluded
IRREL = ["dsid_" + f"{i:032x}" for i in range(1, 11)]   # 10 irrelevant docs


def _questions() -> pd.DataFrame:
    """Five questions; q4 and q5 have empty expected_doc_ids and must be excluded."""
    return pd.DataFrame([
        {"question_id": "q1", "question_type": "basic",           "expected_doc_ids": [DOC_A]},
        {"question_id": "q2", "question_type": "basic",           "expected_doc_ids": [DOC_B]},
        {"question_id": "q3", "question_type": "basic",           "expected_doc_ids": [DOC_C]},
        {"question_id": "q4", "question_type": "high_level",      "expected_doc_ids": []},
        {"question_id": "q5", "question_type": "info_not_found",  "expected_doc_ids": []},
    ])


def _run() -> dict[str, list[tuple[str, float]]]:
    """Passage-level run matching the structure described in the module docstring."""
    pid = schema.make_passage_id
    return {
        # q1: gold passage at rank 1, 9 irrelevant
        "q1": [(pid(DOC_A, 0), 1.0)]
              + [(pid(d, 0), 0.5 - 0.05 * i) for i, d in enumerate(IRREL[:9])],
        # q2: 1 irrelevant at rank 1, gold at rank 2, 8 more irrelevant
        "q2": [(pid(IRREL[0], 0), 1.0), (pid(DOC_B, 0), 0.9)]
              + [(pid(d, 0), 0.4 - 0.04 * i) for i, d in enumerate(IRREL[1:9])],
        # q3: 10 irrelevant passages, gold (DOC_C) absent
        "q3": [(pid(d, 0), 1.0 - 0.09 * i) for i, d in enumerate(IRREL)],
        # q4 / q5: results exist in the run but questions are excluded from eval
        "q4": [(pid(DOC_D, 0), 1.0)],
        "q5": [(pid(DOC_E, 0), 1.0)],
    }


SPLIT_IDS = ["q1", "q2", "q3", "q4", "q5"]

# hand-computed constants used in multiple assertions
_EXP_RECALL        = 2 / 3
_EXP_NDCG          = (1.0 + 1 / math.log2(3)) / 3   # ≈ 0.5436
_EXP_MRR           = 0.5
_EXP_EXTRA_AT_5    = 13 / 3    # (4+4+5)/3
_EXP_EXTRA_AT_10   = 28 / 3   # (9+9+10)/3
_EXP_GOLD_UNINDEXED = 1 / 3   # DOC_C (q3) absent from {DOC_A, DOC_B}


def _result(indexed_doc_ids=None):
    return ev.evaluate_run(
        _run(),
        questions_df=_questions(),
        split_ids=SPLIT_IDS,
        indexed_doc_ids=indexed_doc_ids,
        write=False,
    )


# ---------------------------------------------------------------- passages_to_docs

def test_passages_to_docs_keeps_max_score_per_doc():
    # Two chunks from DOC_A: keep the higher score (0.9 not 0.7).
    run = {
        "q1": [
            (schema.make_passage_id(DOC_A, 0), 0.9),
            (schema.make_passage_id(DOC_A, 1), 0.7),
            (schema.make_passage_id(DOC_B, 0), 0.8),
        ]
    }
    assert ev.passages_to_docs(run) == {"q1": {DOC_A: 0.9, DOC_B: 0.8}}


def test_passages_to_docs_empty_run():
    assert ev.passages_to_docs({}) == {}


def test_passages_to_docs_single_chunk_per_doc():
    run = {"q1": [(schema.make_passage_id(DOC_A, 0), 0.5)]}
    assert ev.passages_to_docs(run) == {"q1": {DOC_A: 0.5}}


# ---------------------------------------------------------------- question filtering

def test_empty_expected_docs_excluded_from_n_questions():
    # q4 (high_level, []) and q5 (info_not_found, []) have no gold docs → excluded.
    assert _result()["n_questions"] == 3


def test_filtering_uses_expected_doc_ids_not_category_name():
    # Give q4 a non-empty expected_doc_ids — it must now be INCLUDED despite high_level type.
    qs = _questions().copy()
    qs.loc[qs["question_id"] == "q4", "expected_doc_ids"] = qs.loc[
        qs["question_id"] == "q4", "expected_doc_ids"
    ].apply(lambda _: [DOC_D])
    result = ev.evaluate_run(
        _run(), questions_df=qs, split_ids=SPLIT_IDS, write=False,
    )
    assert result["n_questions"] == 4


def test_excluded_categories_absent_from_by_category():
    cats = _result()["by_category"]
    assert "high_level" not in cats
    assert "info_not_found" not in cats


# ---------------------------------------------------------------- overall ranx metrics

def test_overall_recall_at_10():
    assert _result()["overall"]["recall@10"] == pytest.approx(_EXP_RECALL, abs=1e-6)


def test_overall_ndcg_at_10():
    assert _result()["overall"]["ndcg@10"] == pytest.approx(_EXP_NDCG, abs=1e-6)


def test_overall_mrr_at_10():
    assert _result()["overall"]["mrr@10"] == pytest.approx(_EXP_MRR, abs=1e-6)


# ---------------------------------------------------------------- extra_docs noise metric

def test_overall_extra_docs_at_5():
    # q1: 4 non-gold in top-5; q2: 4; q3: 5  →  mean = 13/3
    assert _result()["overall"]["extra_docs_at_5"] == pytest.approx(_EXP_EXTRA_AT_5, abs=1e-6)


def test_overall_extra_docs_at_10():
    # q1: 9 non-gold in top-10; q2: 9; q3: 10  →  mean = 28/3
    assert _result()["overall"]["extra_docs_at_10"] == pytest.approx(_EXP_EXTRA_AT_10, abs=1e-6)


def test_extra_docs_at_5_less_than_at_10():
    r = _result()
    assert r["overall"]["extra_docs_at_5"] < r["overall"]["extra_docs_at_10"]


def test_extra_docs_zero_when_all_gold_retrieved():
    # Single question: gold doc is the only retrieved doc → 0 extra at k=1 or k=10.
    pid = schema.make_passage_id
    run = {"q1": [(pid(DOC_A, 0), 1.0)]}
    qs = pd.DataFrame([
        {"question_id": "q1", "question_type": "basic", "expected_doc_ids": [DOC_A]},
    ])
    result = ev.evaluate_run(run, questions_df=qs, split_ids=["q1"], write=False)
    assert result["overall"]["extra_docs_at_5"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- per-category

def test_per_category_basic_recall():
    cats = _result()["by_category"]
    assert "basic" in cats
    assert cats["basic"]["recall@10"] == pytest.approx(_EXP_RECALL, abs=1e-6)


def test_per_category_basic_mrr():
    assert _result()["by_category"]["basic"]["mrr@10"] == pytest.approx(_EXP_MRR, abs=1e-6)


def test_per_category_basic_n():
    assert _result()["by_category"]["basic"]["n"] == 3


def test_per_category_basic_extra_docs_at_5():
    # All included questions are "basic" → same as overall
    cats = _result()["by_category"]
    assert cats["basic"]["extra_docs_at_5"] == pytest.approx(_EXP_EXTRA_AT_5, abs=1e-6)


# ---------------------------------------------------------------- gold_unindexed sanity check

def test_gold_unindexed_counts_unindexed_gold():
    # DOC_C (q3's gold) absent from indexed set → mean = (0+0+1)/3 = 1/3
    r = _result(indexed_doc_ids={DOC_A, DOC_B})
    assert r["overall"]["gold_unindexed"] == pytest.approx(_EXP_GOLD_UNINDEXED, abs=1e-6)


def test_gold_unindexed_zero_when_all_indexed():
    r = _result(indexed_doc_ids={DOC_A, DOC_B, DOC_C})
    assert r["overall"]["gold_unindexed"] == 0.0


def test_gold_unindexed_zero_when_not_provided():
    assert _result()["overall"]["gold_unindexed"] == 0.0


def test_gold_unindexed_propagates_to_category():
    r = _result(indexed_doc_ids={DOC_A, DOC_B})
    assert r["by_category"]["basic"]["gold_unindexed"] == pytest.approx(
        _EXP_GOLD_UNINDEXED, abs=1e-6
    )


# ---------------------------------------------------------------- write to runs/

def test_write_creates_json_file(tmp_path):
    ev.evaluate_run(
        _run(),
        questions_df=_questions(),
        split_ids=SPLIT_IDS,
        tag="r2",
        split="dev",
        runs_dir=tmp_path,
        write=True,
    )
    files = list(tmp_path.glob("*_r2_dev_at10.json"))
    assert len(files) == 1
    written = json.loads(files[0].read_text())
    assert written["tag"] == "r2"
    assert written["split"] == "dev"
    assert written["k"] == 10
    assert written["overall"]["recall@10"] == pytest.approx(_EXP_RECALL, abs=1e-6)
    assert written["overall"]["extra_docs_at_5"] == pytest.approx(_EXP_EXTRA_AT_5, abs=1e-6)


def test_write_false_creates_no_file(tmp_path):
    ev.evaluate_run(
        _run(),
        questions_df=_questions(),
        split_ids=SPLIT_IDS,
        runs_dir=tmp_path,
        write=False,
    )
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- result structure

def test_result_keys():
    r = _result()
    assert set(r.keys()) == {"tag", "split", "k", "n_questions", "overall", "by_category"}


def test_overall_keys():
    keys = set(_result()["overall"].keys())
    assert keys == {
        "recall@10", "ndcg@10", "mrr@10",
        "extra_docs_at_5", "extra_docs_at_10",
        "gold_unindexed",
    }


def test_scores_are_python_floats():
    # Must be JSON-serialisable without a custom encoder.
    json.dumps(_result())
