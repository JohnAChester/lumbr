"""Dev/test split of the 500 benchmark questions (issue #6).

100 dev + 400 test, stratified by question category, fixed seed. Dev is for tuning,
early stopping, and demo prompts; test is never touched until Phase 5.

The committed files in splits/ are the source of truth — load_split() reads those.
make_split() / write_split() are run once from notebook 06 to generate them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lumbr import paths

SPLITS_DIR = paths.REPO_ROOT / "splits"
DEV_IDS_FILE = SPLITS_DIR / "dev_question_ids.txt"
TEST_IDS_FILE = SPLITS_DIR / "test_question_ids.txt"
MANIFEST_FILE = SPLITS_DIR / "split_manifest.json"

DEV_SIZE = 100
TEST_SIZE = 400
SEED = 42


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_split(df: pd.DataFrame, category_col: str = "question_type",
               seed: int = SEED) -> dict:
    """Stratified DEV_SIZE/TEST_SIZE split of the questions DataFrame by category.

    Uses largest-remainder (Hamilton) allocation so the total is exactly DEV_SIZE.
    Returns {"dev": [question_id, ...], "test": [...], "manifest": {...}}.
    Same seed always produces the same split.
    """
    if category_col not in df.columns:
        raise ValueError(
            f"{category_col!r} not in DataFrame; available: {list(df.columns)}"
        )

    sizes = df.groupby(category_col, sort=True).size()
    total = int(sizes.sum())

    # Largest-remainder allocation: give each category floor(n * dev_frac) dev slots,
    # then distribute any remaining slots to the largest fractional remainders.
    exact = sizes * DEV_SIZE / total
    alloc = exact.apply(int)
    gap = DEV_SIZE - int(alloc.sum())
    fracs = (exact - alloc).sort_values(ascending=False)
    alloc[fracs.index[:gap]] += 1

    dev_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for cat in sizes.index:
        grp = df[df[category_col] == cat].sort_values("question_id")
        n = int(alloc[cat])
        dev_grp = grp.sample(n=n, random_state=seed) if n > 0 else grp.iloc[:0]
        dev_parts.append(dev_grp)
        test_parts.append(grp.drop(dev_grp.index))

    dev_df = pd.concat(dev_parts)
    test_df = pd.concat(test_parts)

    dev_ids = sorted(dev_df.question_id.tolist())
    test_ids = sorted(test_df.question_id.tolist())

    dev_counts = dev_df.groupby(category_col).size().to_dict()
    test_counts = test_df.groupby(category_col).size().to_dict()

    manifest = {
        "dev": len(dev_ids),
        "test": len(test_ids),
        "total": total,
        "category_col": category_col,
        "seed": seed,
        "by_category": {
            cat: {
                "total": int(sizes[cat]),
                "dev": int(dev_counts.get(cat, 0)),
                "test": int(test_counts.get(cat, 0)),
            }
            for cat in sorted(sizes.index)
        },
    }
    return {"dev": dev_ids, "test": test_ids, "manifest": manifest}


def write_split(result: dict, splits_dir: Path = SPLITS_DIR) -> None:
    """Write split results to splits/. Commit those files afterwards."""
    splits_dir.mkdir(exist_ok=True)
    _write_ids(splits_dir / DEV_IDS_FILE.name, result["dev"])
    _write_ids(splits_dir / TEST_IDS_FILE.name, result["test"])
    (splits_dir / MANIFEST_FILE.name).write_text(
        json.dumps(result["manifest"], indent=2) + "\n", encoding="utf-8"
    )


def load_split() -> dict[str, list[str]]:
    """Load the committed split. Returns {"dev": [...], "test": [...]}."""
    if not DEV_IDS_FILE.exists():
        raise FileNotFoundError(
            f"{DEV_IDS_FILE} not found. Run notebook 06 to generate and commit the split."
        )
    return {"dev": _read_ids(DEV_IDS_FILE), "test": _read_ids(TEST_IDS_FILE)}
