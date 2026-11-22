"""Reciprocal Rank Fusion (R5): fuse BM25 and dense runs at passage level (issue #10).

RRF formula (Cormack 2009): score(p) = Σ_i  1 / (k + rank_i(p))
where rank_i is 1-based position in run i, and k=60 is the default smoothing constant.

Usage:
  from lumbr import index_rrf
  r5_run = index_rrf.fuse([r1_run, r3_run], k=60, top_n=50)
  index_rrf.save_run(r5_run, paths.RUNS, k_rrf=60,
                     input_run_dirs=[R1_RUN_DIR, R3_RUN_DIR])
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

from lumbr import paths


def fuse(
    runs: list[dict[str, list[tuple[str, float]]]],
    k: int = 60,
    top_n: int = 50,
) -> dict[str, list[tuple[str, float]]]:
    """RRF over passage-level runs.

    For each query, considers the top_n passages from each input run,
    accumulates 1/(k + rank) per passage across runs, and returns top_n
    fused passages sorted descending by score.

    Args:
        runs:  list of passage-level runs; each maps qid → [(pid, score), ...]
        k:     smoothing constant (default 60; higher k de-emphasises rank advantage)
        top_n: how many passages to consider from each run and to return

    Returns:
        passage-level run in the same format as the inputs
    """
    all_qids: set[str] = set().union(*(r.keys() for r in runs))
    out: dict[str, list[tuple[str, float]]] = {}
    for qid in all_qids:
        rrf: dict[str, float] = {}
        for run in runs:
            for rank, (pid, _) in enumerate(run.get(qid, [])[:top_n], start=1):
                rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (k + rank)
        out[qid] = sorted(rrf.items(), key=lambda x: -x[1])[:top_n]
    return out


def save_run(
    run: dict[str, list[tuple[str, float]]],
    runs_dir: Path,
    *,
    tag: str = "r5_rrf",
    split: str = "dev",
    k_rrf: int = 60,
    input_run_dirs: list[str | Path],
    date: str | None = None,
) -> Path:
    """Save a fused run to a dated subfolder of runs_dir.

    Subfolder: `{tag}_{split}_{date}` (e.g. `r5_rrf_dev_2026-11-21`).
    Writes:
      run.json   — passage-level hits (same format as other runs)
      meta.json  — tag, k_rrf, input_runs (folder names), git_commit, split, date

    Returns the subdir Path.
    """
    today = date or datetime.date.today().isoformat()
    subdir = runs_dir / f"{tag}_{split}_{today}"
    subdir.mkdir(parents=True, exist_ok=True)

    run_path = subdir / "run.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f)
    print(f"[{tag}] run saved → {run_path}")

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    meta = {
        "retriever":   tag,
        "k_rrf":       k_rrf,
        "input_runs":  [Path(d).name for d in input_run_dirs],
        "split":       split,
        "git_commit":  git_sha,
        "date":        today,
    }
    meta_path = subdir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{tag}] meta  → {meta_path}")
    return subdir
