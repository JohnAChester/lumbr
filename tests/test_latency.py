"""Unit tests for latency.py (issue #25).  No GPU, no models, no Drive.

Coverage:
  median_ms / p95_ms on hand-computed lists
  warmup_exclude drops exactly n_warmup entries
  e2e_ms sums named stages correctly
  load_stage_timings / save_stage_timings roundtrip and resumability
  build_setup_rows computes retrieval + e2e correctly
  format_latency_md contains required sections
  write_latency_csv produces correct CSV header + rows
"""

from __future__ import annotations

import csv
import io
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


# ---------------------------------------------------------------- warmup_exclude

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


# ---------------------------------------------------------------- e2e_ms

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
    save_stage_timings("bm25", [10.0, 11.0, 12.0], p)
    loaded = load_stage_timings(p)
    assert loaded == {"bm25": [10.0, 11.0, 12.0]}


def test_load_missing_file_returns_empty(tmp_path):
    assert load_stage_timings(tmp_path / "nonexistent.jsonl") == {}


def test_save_multiple_stages(tmp_path):
    p = tmp_path / "t.jsonl"
    save_stage_timings("bm25", [1.0, 2.0], p)
    save_stage_timings("r2",   [3.0, 4.0], p)
    loaded = load_stage_timings(p)
    assert loaded["bm25"] == [1.0, 2.0]
    assert loaded["r2"]   == [3.0, 4.0]


def test_later_save_overwrites_stage(tmp_path):
    p = tmp_path / "t.jsonl"
    save_stage_timings("bm25", [1.0], p)
    save_stage_timings("bm25", [9.0, 9.1], p)
    loaded = load_stage_timings(p)
    assert loaded["bm25"] == [9.0, 9.1]


def test_load_skips_corrupt_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"stage":"bm25","timings_ms":[5.0]}\nNOT_JSON\n', encoding="utf-8")
    loaded = load_stage_timings(p)
    assert loaded == {"bm25": [5.0]}


# ---------------------------------------------------------------- build_setup_rows

_MEDIANS = {
    "bm25":   10.0,
    "r2":     15.0,
    "r3":     80.0,
    "r4":     16.0,
    "rrf":     3.0,
    "rerank": 600.0,
}


def test_build_setup_rows_r1():
    rows = {r["setup"]: r for r in build_setup_rows(_MEDIANS)}
    assert rows["R1"]["retrieval_ms"] == pytest.approx(10.0)
    assert rows["R1"]["e2e_ms"] is None  # no gen_median


def test_build_setup_rows_r5():
    rows = {r["setup"]: r for r in build_setup_rows(_MEDIANS)}
    # R5 = bm25 + r3 + rrf
    assert rows["R5"]["retrieval_ms"] == pytest.approx(10.0 + 80.0 + 3.0)


def test_build_setup_rows_e2e_with_gen():
    rows = {r["setup"]: r for r in build_setup_rows(_MEDIANS, gen_median=300.0)}
    assert rows["R1"]["e2e_ms"] == pytest.approx(10.0 + 300.0)
    assert rows["R5+R6"]["e2e_ms"] == pytest.approx(10.0 + 80.0 + 3.0 + 600.0 + 300.0)


def test_build_setup_rows_missing_stage_skipped():
    # Only BM25 available → only setups that need only "bm25" appear
    partial = {"bm25": 10.0}
    rows = {r["setup"]: r for r in build_setup_rows(partial)}
    assert "R1" in rows
    assert "R2" not in rows
    assert "R5" not in rows


# ---------------------------------------------------------------- format_latency_md

_STAGE_ROWS = [
    {"stage": "bm25",   "median_ms": 10.1, "p95_ms": 15.0, "n": 47},
    {"stage": "rerank", "median_ms": 605.0, "p95_ms": 650.0, "n": 47},
]
_SETUP_ROWS = [
    {"setup": "R1",    "retrieval_ms": 10.1, "e2e_ms": 310.1},
    {"setup": "R1+R6", "retrieval_ms": 615.1, "e2e_ms": 915.1},
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
    rows = [{"setup": "R1", "retrieval_ms": 10.0, "e2e_ms": None}]
    md = format_latency_md(_STAGE_ROWS, rows, gpu_name="L4")
    assert "—" in md


def test_format_md_generator_note_labelled_batched():
    md = format_latency_md(
        _STAGE_ROWS, _SETUP_ROWS,
        gpu_name="L4",
        generator_note="batched 400 q, 60 q/min",
    )
    assert "batched throughput" in md


# ---------------------------------------------------------------- write_latency_csv

def test_write_latency_csv_header_and_rows(tmp_path):
    p = tmp_path / "results" / "latency.csv"
    rows = [
        {"setup": "R1",   "retrieval_ms": 10.123, "e2e_ms": 310.456},
        {"setup": "R5+R6","retrieval_ms": 693.0,  "e2e_ms": None},
    ]
    write_latency_csv(rows, p)
    assert p.exists()
    reader = csv.DictReader(p.read_text(encoding="utf-8").splitlines())
    parsed = list(reader)
    assert parsed[0]["setup"] == "R1"
    assert float(parsed[0]["retrieval_ms"]) == pytest.approx(10.123, rel=1e-4)
    assert float(parsed[0]["e2e_ms"]) == pytest.approx(310.456, rel=1e-4)
    assert parsed[1]["e2e_ms"] == ""


def test_write_latency_csv_creates_parent_dir(tmp_path):
    p = tmp_path / "deep" / "nested" / "latency.csv"
    write_latency_csv([{"setup": "R1", "retrieval_ms": 5.0, "e2e_ms": 305.0}], p)
    assert p.exists()
