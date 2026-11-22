"""Stratified dev/test split (#6): toy DataFrame, no network or Drive needed."""

import json

import pandas as pd
import pytest

from lumbr import split


def _make_df(categories: list[tuple[str, int]]) -> pd.DataFrame:
    """Build a minimal questions DataFrame with the given (category, count) pairs."""
    rows = []
    for cat_i, (cat, n) in enumerate(categories):
        for j in range(n):
            rows.append({"question_id": f"q_{cat_i:02d}_{j:03d}", "question_type": cat})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- counts

def test_exact_counts_benchmark_shape():
    df = _make_df([("specific", 300), ("multi_hop", 100), ("high_level", 60), ("info_not_found", 40)])
    result = split.make_split(df)
    assert len(result["dev"]) == split.DEV_SIZE
    assert len(result["test"]) == split.TEST_SIZE
    assert len(set(result["dev"]) & set(result["test"])) == 0


def test_covers_all_questions():
    df = _make_df([("a", 200), ("b", 200), ("c", 100)])
    result = split.make_split(df)
    assert sorted(result["dev"] + result["test"]) == sorted(df.question_id.tolist())


# ---------------------------------------------------------------- stratification

def test_every_category_has_dev_rep():
    df = _make_df([("specific", 300), ("multi_hop", 100), ("high_level", 60), ("info_not_found", 40)])
    result = split.make_split(df)
    dev_set = set(result["dev"])
    dev_cats = df[df.question_id.isin(dev_set)].question_type.unique()
    for cat in df.question_type.unique():
        assert cat in dev_cats, f"category {cat!r} has no dev examples"


def test_category_proportions_are_approximate():
    df = _make_df([("a", 300), ("b", 100), ("c", 100)])
    result = split.make_split(df)
    dev_set = set(result["dev"])
    dev_df = df[df.question_id.isin(dev_set)]
    # 'a' is 60% of corpus, so it should be roughly 60% of dev (±5 tolerance)
    a_dev = (dev_df.question_type == "a").sum()
    assert abs(a_dev - 60) <= 5


# ---------------------------------------------------------------- determinism

def test_same_seed_gives_same_split():
    df = _make_df([("a", 200), ("b", 200), ("c", 100)])
    r1 = split.make_split(df, seed=42)
    r2 = split.make_split(df, seed=42)
    assert r1["dev"] == r2["dev"]
    assert r1["test"] == r2["test"]


def test_different_seeds_differ():
    df = _make_df([("a", 200), ("b", 200), ("c", 100)])
    r1 = split.make_split(df, seed=42)
    r2 = split.make_split(df, seed=99)
    assert r1["dev"] != r2["dev"]


# ---------------------------------------------------------------- manifest

def test_manifest_counts_add_up():
    df = _make_df([("specific", 300), ("multi_hop", 200)])
    result = split.make_split(df)
    m = result["manifest"]
    assert m["dev"] == len(result["dev"])
    assert m["test"] == len(result["test"])
    assert m["total"] == len(df)
    for cat, counts in m["by_category"].items():
        assert counts["dev"] + counts["test"] == counts["total"], cat


# ---------------------------------------------------------------- write / load

def test_write_and_load_roundtrip(tmp_path):
    df = _make_df([("a", 200), ("b", 200), ("c", 100)])
    result = split.make_split(df)
    split.write_split(result, splits_dir=tmp_path)

    loaded = split.load_split.__wrapped__(tmp_path) if hasattr(split.load_split, "__wrapped__") \
        else _load_from(tmp_path)
    assert loaded["dev"] == result["dev"]
    assert loaded["test"] == result["test"]


def _load_from(splits_dir) -> dict:
    """load_split but from an arbitrary directory (for testing write_split)."""
    dev = [l for l in (splits_dir / "dev_question_ids.txt").read_text().splitlines() if l]
    test = [l for l in (splits_dir / "test_question_ids.txt").read_text().splitlines() if l]
    return {"dev": dev, "test": test}


def test_manifest_file_is_valid_json(tmp_path):
    df = _make_df([("a", 300), ("b", 200)])
    split.write_split(split.make_split(df), splits_dir=tmp_path)
    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["dev"] == split.DEV_SIZE
    assert manifest["total"] == 500


# ---------------------------------------------------------------- error handling

def test_wrong_column_raises():
    df = _make_df([("a", 100)])
    with pytest.raises(ValueError, match="not in DataFrame"):
        split.make_split(df, category_col="nonexistent")


def test_load_split_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "DEV_IDS_FILE", tmp_path / "dev_question_ids.txt")
    with pytest.raises(FileNotFoundError, match="notebook 06"):
        split.load_split()
