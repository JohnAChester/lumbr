"""Unit tests for latency.py (issue #25).  No GPU, no models, no Drive.

Coverage:
  median_ms / p95_ms on hand-computed lists
  warmup_exclude drops exactly n_warmup entries
  warmup_qids returns first-n question IDs
  e2e_ms sums named stage medians
  save_stage_timings / load_stage_timings roundtrip, resumability, corruption
  build_setup_rows: per-question sums, warmup exclusion, missing stage skipped,
                    median-of-sums ≠ sum-of-medians (key regression guard)
  format_latency_md: required sections, new setup-row format, gen_context_note
  write_latency_csv: header, retrieval + e2e median/p95 columns
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lumbr.latency import (
    build_setup_rows,
    e2e_ms,
    format_latency_md,
    load_stage_timings,
    median_ms,
    p95_ms,
    save_stage_timings,
    warmup_exclude,
    warmup_qids,
    write_latency_csv,
)


# ---------------------------------------------------------------- median_ms / p95_ms

def test_median_odd():
    assert median_ms([10.0, 20.0, 30.0]) == pytest.approx(20.0)


def test_median_even():
    assert median_ms([10.0, 20.0, 30.0, 40.0]) == pytest.approx(25.0)


def test_p95_hand_computed():
    # 20 values 1..20 ms; p95 = 19.05
    vals = [float(i) for i in range(1, 21)]
    assert p95_ms(vals) == pytest.approx(19.05, rel=1e-3)


def test_p95_all_same():
    assert p95_ms([5.0] * 10) == pytest.approx(5.0)


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median_ms([])


def test_p95_empty_raises():
    with pytest.raises(ValueError):
        p95_ms([])


# ---------------------------------------------------------------- warmup_exclude (list form)

def test_warmup_exclude_drops_first_n():
    t = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert warmup_exclude(t, 2) == [3.0, 4.0, 5.0]


def test_warmup_exclude_zero():
    t = [1.0, 2.0, 3.0]
    assert warmup_exclude(t, 0) == [1.0, 2.0, 3.0]


def test_warmup_exclude_negative_raises():
    with pytest.raises(ValueError):
        warmup_exclude([1.0], -1)


def test_warmup_exclude_all():
    t = [1.0, 2.0]
    assert warmup_exclude(t, 2) == []


# ---------------------------------------------------------------- warmup_qids (set form)

def test_warmup_qids_returns_first_n():
    ids = ["a", "b", "c", "d", "e"]
    assert warmup_qids(ids, 2) == {"a", "b"}


def test_warmup_qids_zero_returns_empty():
    assert warmup_qids(["a", "b", "c"], 0) == set()


def test_warmup_qids_full_list():
    ids = ["x", "y"]
    assert warmup_qids(ids, 2) == {"x", "y"}


# ---------------------------------------------------------------- e2e_ms (sum of stage medians)

def test_e2e_sums_named_stages():
    medians = {"bm25": 10.0, "r3": 20.0, "rrf": 5.0, "rerank": 50.0, "gen": 300.0}
    assert e2e_ms(medians, ["bm25", "r3", "rrf"]) == pytest.approx(35.0)


def test_e2e_single_stage():
    medians = {"bm25": 12.5}
    assert e2e_ms(medians, ["bm25"]) == pytest.approx(12.5)


def test_e2e_missing_stage_raises():
    with pytest.raises(KeyError):
        e2e_ms({"bm25": 10.0}, ["bm25", "r2"])


# ---------------------------------------------------------------- checkpoint I/O

def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "timings.jsonl"
    save_stage_timings("bm25", ["q1", "q2", "q3"], [10.0, 11.0, 12.0], p)
    loaded = load_stage_timings(p)
    assert loaded == {"bm25": {"q1": 10.0, "q2": 11.0, "q3": 12.0}}


def test_load_missing_file_returns_empty(tmp_path):
    assert load_stage_timings(tmp_path / "nonexistent.jsonl") == {}


def test_save_multiple_stages(tmp_path):
    p = tmp_path / "t.jsonl"
    save_stage_timings("bm25",      ["q1", "q2"], [1.0, 2.0], p)
    save_stage_timings("r2_encode", ["q1", "q2"], [3.0, 4.0], p)
    loaded = load_stage_timings(p)
    assert loaded["bm25"]      == {"q1": 1.0, "q2": 2.0}
    assert loaded["r2_encode"] == {"q1": 3.0, "q2": 4.0}


def test_later_save_overwrites_stage(tmp_path):
    p = tmp_path / "t.jsonl"
    save_stage_timings("bm25", ["q1"],        [1.0],      p)
    save_stage_timings("bm25", ["q1", "q2"],  [9.0, 9.1], p)
    loaded = load_stage_timings(p)
    assert loaded["bm25"] == {"q1": 9.0, "q2": 9.1}


def test_load_skips_corrupt_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    valid = json.dumps({"stage": "bm25", "question_ids": ["q1"], "timings_ms": [5.0]})
    p.write_text(valid + "\nNOT_JSON\n", encoding="utf-8")
    loaded = load_stage_timings(p)
    assert loaded == {"bm25": {"q1": 5.0}}


# ---------------------------------------------------------------- build_setup_rows

# Minimal timings for all retrieval stages in SETUP_STAGES
_T = {
    "bm25":      {"q1": 10.0, "q2": 12.0, "q3":  8.0},
    "r2_encode": {"q1": 15.0, "q2": 14.0, "q3": 16.0},
    "r2_search": {"q1":  5.0, "q2":  6.0, "q3":  4.0},
    "r3_encode": {"q1": 80.0, "q2": 85.0, "q3": 75.0},
    "r3_search": {"q1":  3.0, "q2":  3.0, "q3":  3.0},
    "r4_encode": {"q1": 16.0, "q2": 15.0, "q3": 17.0},
    "r4_search": {"q1":  5.5, "q2":  5.0, "q3":  6.0},
    "rrf":       {"q1":  3.0, "q2":  3.0, "q3":  3.0},
    "rerank":    {"q1": 600.0, "q2": 605.0, "q3": 595.0},
    "gen":       {"q1": 300.0, "q2": 310.0, "q3": 290.0},
}
_WUP = set()   # no warmup in fixture


def test_build_setup_rows_r1():
    rows = {r["setup"]: r for r in build_setup_rows(_T, _WUP)}
    # R1 = ["bm25"]; per-question sums [10, 12, 8]; median = 10
    assert rows["R1"]["retrieval"]["median_ms"] == pytest.approx(10.0)
    assert rows["R1"]["retrieval"]["n"] == 3


def test_build_setup_rows_r1_e2e():
    rows = {r["setup"]: r for r in build_setup_rows(_T, _WUP)}
    # E2E: bm25 + gen per question: [310, 322, 298]; median = 310
    assert rows["R1"]["e2e"]["median_ms"] == pytest.approx(310.0)


def test_build_setup_rows_r5():
    rows = {r["setup"]: r for r in build_setup_rows(_T, _WUP)}
    # R5 = bm25 + r3_encode + r3_search + rrf
    # q1: 10+80+3+3=96, q2: 12+85+3+3=103, q3: 8+75+3+3=89 → median=96
    assert rows["R5"]["retrieval"]["median_ms"] == pytest.approx(96.0)


def test_build_setup_rows_missing_stage_skipped():
    # Only bm25 available → only R1 appears
    partial = {"bm25": {"q1": 10.0, "q2": 12.0}}
    rows = {r["setup"]: r for r in build_setup_rows(partial, set())}
    assert "R1" in rows
    assert "R2" not in rows
    assert "R5" not in rows


def test_build_setup_rows_warmup_excluded():
    t = {"bm25": {"warmup1": 999.0, "q1": 10.0, "q2": 12.0}}
    rows = {r["setup"]: r for r in build_setup_rows(t, {"warmup1"})}
    # Only q1 and q2 contribute; warmup1 is excluded
    assert rows["R1"]["retrieval"]["n"] == 2
    assert rows["R1"]["retrieval"]["median_ms"] == pytest.approx(11.0)


def test_build_setup_rows_median_of_sums_ne_sum_of_medians():
    """
    The correct metric is median(per-question sums), not sum(per-stage medians).

    For R2 = [r2_encode, r2_search] with anti-correlated stage times:
      q1: encode=1,  search=10  → sum=11
      q2: encode=10, search=1   → sum=11
      q3: encode=50, search=50  → sum=100

    median-of-sums  = median([11, 11, 100]) = 11   ← correct
    sum-of-medians  = median([1,10,50]) + median([10,1,50]) = 10 + 10 = 20  ← wrong
    """
    t = {
        "r2_encode": {"q1": 1.0,  "q2": 10.0, "q3": 50.0},
        "r2_search": {"q1": 10.0, "q2": 1.0,  "q3": 50.0},
    }
    rows = {r["setup"]: r for r in build_setup_rows(t, set())}
    assert "R2" in rows
    assert rows["R2"]["retrieval"]["median_ms"] == pytest.approx(11.0)
    assert rows["R2"]["retrieval"]["median_ms"] != pytest.approx(20.0)


def test_build_setup_rows_gen_no_e2e_without_gen_stage():
    t = {"bm25": {"q1": 10.0}}
    rows = {r["setup"]: r for r in build_setup_rows(t, set())}
    assert rows["R1"]["e2e"] is None


def test_build_setup_rows_abstained_gen_excluded_from_e2e():
    # gen=0.0 means the question abstained (no model call) — excluded from E2E
    t = {
        "bm25": {"q1": 10.0, "q2": 12.0},
        "gen":  {"q1": 300.0, "q2": 0.0},   # q2 abstained
    }
    rows = {r["setup"]: r for r in build_setup_rows(t, set())}
    # E2E should include only q1 (q2 is excluded because gen=0)
    assert rows["R1"]["e2e"]["n"] == 1
    assert rows["R1"]["e2e"]["median_ms"] == pytest.approx(310.0)


# ---------------------------------------------------------------- format_latency_md

_STAGE_ROWS = [
    {"stage": "bm25",    "median_ms": 10.1,  "p95_ms": 15.0,  "n": 47},
    {"stage": "rerank",  "median_ms": 605.0, "p95_ms": 650.0, "n": 47},
]
_SETUP_ROWS = [
    {
        "setup":     "R1",
        "retrieval": {"median_ms": 10.1, "p95_ms": 15.0, "n": 47},
        "e2e":       {"median_ms": 310.1, "p95_ms": 380.0, "n": 47},
    },
    {
        "setup":     "R1+R6",
        "retrieval": {"median_ms": 615.1, "p95_ms": 700.0, "n": 47},
        "e2e":       None,
    },
]


def test_format_md_has_stage_section():
    md = format_latency_md(_STAGE_ROWS, _SETUP_ROWS, gpu_name="L4")
    assert "Per-stage latency" in md
    assert "bm25" in md
    assert "rerank" in md


def test_format_md_has_setup_section():
    md = format_latency_md(_STAGE_ROWS, _SETUP_ROWS, gpu_name="L4")
    assert "End-to-end" in md
    assert "R1+R6" in md


def test_format_md_includes_gpu_and_commit():
    md = format_latency_md(_STAGE_ROWS, _SETUP_ROWS, gpu_name="NVIDIA L4", git_commit="abc1234")
    assert "NVIDIA L4" in md
    assert "abc1234" in md


def test_format_md_no_e2e_shown_as_dash():
    md = format_latency_md(_STAGE_ROWS, _SETUP_ROWS, gpu_name="L4")
    # R1+R6 has e2e=None → both E2E columns should show "—"
    assert "—" in md


def test_format_md_setup_shows_p95():
    md = format_latency_md(_STAGE_ROWS, _SETUP_ROWS, gpu_name="L4")
    # The setup table has p95 columns
    assert "p95" in md.lower()
    assert "380.0" in md   # R1 e2e p95
    assert "15.0" in md    # R1 retrieval p95


def test_format_md_generator_note_appears():
    md = format_latency_md(
        _STAGE_ROWS, _SETUP_ROWS,
        gpu_name="L4",
        generator_note="single-query vLLM calls, 768 tokens max",
    )
    assert "single-query vLLM calls" in md


def test_format_md_gen_context_note_appears():
    note = "generation timed with R5 context; every setup passes the same number of passages"
    md = format_latency_md(_STAGE_ROWS, _SETUP_ROWS, gpu_name="L4", gen_context_note=note)
    assert "R5 context" in md
    assert "same number of passages" in md


def test_format_md_median_of_sums_note_present():
    md = format_latency_md(_STAGE_ROWS, _SETUP_ROWS, gpu_name="L4")
    assert "per-question" in md


# ---------------------------------------------------------------- write_latency_csv

def test_write_latency_csv_header_and_rows(tmp_path):
    p = tmp_path / "results" / "latency.csv"
    rows = [
        {
            "setup":     "R1",
            "retrieval": {"median_ms": 10.123, "p95_ms": 12.456, "n": 47},
            "e2e":       {"median_ms": 310.456, "p95_ms": 350.789, "n": 47},
        },
        {
            "setup":     "R5+R6",
            "retrieval": {"median_ms": 693.0, "p95_ms": 750.0, "n": 47},
            "e2e":       None,
        },
    ]
    write_latency_csv(rows, p)
    assert p.exists()
    parsed = list(csv.DictReader(p.read_text(encoding="utf-8").splitlines()))
    assert parsed[0]["setup"] == "R1"
    assert float(parsed[0]["retrieval_median_ms"]) == pytest.approx(10.123, rel=1e-4)
    assert float(parsed[0]["retrieval_p95_ms"])    == pytest.approx(12.456, rel=1e-4)
    assert float(parsed[0]["e2e_median_ms"])       == pytest.approx(310.456, rel=1e-4)
    assert float(parsed[0]["e2e_p95_ms"])          == pytest.approx(350.789, rel=1e-4)
    assert parsed[1]["e2e_median_ms"] == ""
    assert parsed[1]["e2e_p95_ms"]   == ""


def test_write_latency_csv_creates_parent_dir(tmp_path):
    p = tmp_path / "deep" / "nested" / "latency.csv"
    rows = [{
        "setup":     "R1",
        "retrieval": {"median_ms": 5.0, "p95_ms": 7.0, "n": 10},
        "e2e":       {"median_ms": 305.0, "p95_ms": 350.0, "n": 10},
    }]
    write_latency_csv(rows, p)
    assert p.exists()
