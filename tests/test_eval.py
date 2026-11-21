"""Retrieval eval tests (issue #7): hand-computed expected values, no Drive required.

All expected metric values are derived from first principles:
  q1 — gold at rank 1: Recall=1, nDCG=1,         MRR=1
  q2 — gold at rank 2: Recall=1, nDCG=1/log2(3), MRR=1/2
  q3 — gold not found: Recall=0, nDCG=0,          MRR=0
  q4 — high_level (excluded): never counted
  --
  mean Recall@10 = (1+1+0)/3  = 2/3
  mean nDCG@10   = (1 + 1/log2(3) + 0)/3
  mean MRR@10    = (1+0.5+0)/3 = 1/2
"""

import json
import math

import pandas as pd
import pytest

ranx = pytest.importorskip("ranx")

from lumbr import eval as ev, schema

# ---------------------------------------------------------------- shared data

DOC_A = "dsid_" + "a" * 32   # q1 gold
DOC_B = "dsid_" + "b" * 32   # q2 gold
DOC_C = "dsid_" + "c" * 32   # q3 gold (never retrieved)
DOC_D = "dsid_" + "d" * 32   # q4 gold (high_level — excluded)
IRREL = ["dsid_" + f"{i:032x}" for i in range(1, 11)]  # 10 irrelevant docs


def _questions() -> pd.DataFrame:
    return pd.DataFrame([
        {"question_id": "q1", "question_type": "basic",      "expected_doc_ids": [DOC_A]},
        {"question_id": "q2", "question_type": "basic",      "expected_doc_ids": [DOC_B]},
        {"question_id": "q3", "question_type": "basic",      "expected_doc_ids": [DOC_C]},
        {"question_id": "q4", "question_type": "high_level", "expected_doc_ids": [DOC_D]},
    ])


def _run() -> dict[str, list[tuple[str, float]]]:
    """Passage-level run with the structure described in the module docstring."""
    pid = schema.make_passage_id
    return {
        # q1: gold passage at rank 1
        "q1": [(pid(DOC_A, 0), 1.0)]
              + [(pid(d, 0), 0.5 - 0.05 * i) for i, d in enumerate(IRREL[:9])],
        # q2: irrelevant at rank 1, gold at rank 2
        "q2": [(pid(IRREL[0], 0), 1.0), (pid(DOC_B, 0), 0.9)]
              + [(pid(d, 0), 0.4 - 0.04 * i) for i, d in enumerate(IRREL[1:9])],
        # q3: 10 irrelevant passages, gold (DOC_C) absent
        "q3": [(pid(d, 0), 1.0 - 0.09 * i) for i, d in enumerate(IRREL)],
        # q4: gold at rank 1 but high_level — excluded from all metrics
        "q4": [(pid(DOC_D, 0), 1.0)],
    }


SPLIT_IDS = ["q1", "q2", "q3", "q4"]

# hand-computed expected values
_EXP_RECALL = 2 / 3                            # (1+1+0)/3
_EXP_NDCG   = (1.0 + 1 / math.log2(3)) / 3   # (1 + 0.6309…)/3 ≈ 0.5436
_EXP_MRR    = 0.5                              # (1+0.5+0)/3


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


# ---------------------------------------------------------------- overall metrics

def _result(indexed_doc_ids=None):
    return ev.evaluate_run(
        _run(),
        questions_df=_questions(),
        split_ids=SPLIT_IDS,
        indexed_doc_ids=indexed_doc_ids,
        write=False,
    )


def test_overall_recall_at_10():
    # (1+1+0)/3
    assert _result()["overall"]["recall@10"] == pytest.approx(_EXP_RECALL, abs=1e-6)


def test_overall_ndcg_at_10():
    # q1: nDCG=1.0, q2: nDCG=1/log2(3), q3: nDCG=0
    assert _result()["overall"]["ndcg@10"] == pytest.approx(_EXP_NDCG, abs=1e-6)


def test_overall_mrr_at_10():
    # q1: 1/1, q2: 1/2, q3: 0  → mean=0.5
    assert _result()["overall"]["mrr@10"] == pytest.approx(_EXP_MRR, abs=1e-6)


# ---------------------------------------------------------------- excluded categories

def test_high_level_not_counted_in_n_questions():
    # 4 questions in split_ids but q4 is high_level → n_questions=3
    assert _result()["n_questions"] == 3


def test_excluded_categories_absent_from_by_category():
    cats = _result()["by_category"]
    assert "high_level" not in cats
    assert "info_not_found" not in cats


# ---------------------------------------------------------------- per-category

def test_per_category_basic_recall():
    cats = _result()["by_category"]
    assert "basic" in cats
    assert cats["basic"]["recall@10"] == pytest.approx(_EXP_RECALL, abs=1e-6)


def test_per_category_basic_n():
    assert _result()["by_category"]["basic"]["n"] == 3


def test_per_category_basic_mrr():
    assert _result()["by_category"]["basic"]["mrr@10"] == pytest.approx(_EXP_MRR, abs=1e-6)


# ---------------------------------------------------------------- invalid_extra_docs

def test_invalid_extra_docs_counts_unindexed_gold():
    # DOC_C (q3's gold) is not in indexed_doc_ids → 1 invalid for q3, 0 for q1/q2
    # mean = (0+0+1)/3 = 1/3
    r = _result(indexed_doc_ids={DOC_A, DOC_B})
    assert r["overall"]["invalid_extra_docs"] == pytest.approx(1 / 3, abs=1e-6)


def test_invalid_extra_docs_zero_when_all_indexed():
    r = _result(indexed_doc_ids={DOC_A, DOC_B, DOC_C, DOC_D})
    assert r["overall"]["invalid_extra_docs"] == 0.0


def test_invalid_extra_docs_zero_when_not_provided():
    r = _result(indexed_doc_ids=None)
    assert r["overall"]["invalid_extra_docs"] == 0.0


def test_invalid_extra_docs_propagates_to_category():
    # Same 1/3 expected in by_category["basic"]
    r = _result(indexed_doc_ids={DOC_A, DOC_B})
    assert r["by_category"]["basic"]["invalid_extra_docs"] == pytest.approx(1 / 3, abs=1e-6)


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
    assert keys == {"recall@10", "ndcg@10", "mrr@10", "invalid_extra_docs"}


def test_scores_are_python_floats():
    # Must be JSON-serialisable without custom encoder.
    r = _result()
    json.dumps(r)   # raises TypeError for numpy scalars
