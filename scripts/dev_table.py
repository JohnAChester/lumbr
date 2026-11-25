"""Rebuild the canonical dev retrieval table from the pinned run folders on Drive.

Usage (CPU, no GPU; needs ranx, which is in the `retrieval` extra):
    LUMBR_DRIVE="/path/to/lumbr-data" python scripts/dev_table.py
    # or without installing extras:
    LUMBR_DRIVE="/path/to/lumbr-data" uv run --with ranx python scripts/dev_table.py

Reads results/dev_runs.json (setup -> run folder), scores every run with the same
eval.evaluate_run call at k=10 and k=50, runs integrity checks, and writes
results/dev_table.md and results/dev_table.csv. Nothing on Drive is modified.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from lumbr import eval as ev
from lumbr import index_rrf, paths, schema
from lumbr.index_bm25 import read_run

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
EXPECTED_N = 94  # dev questions with at least one gold doc
MAX_DEPTH = 100  # 50 for standard runs; 100 for the depth-100 experiment

problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)
        print(f"  FAIL  {msg}")


def main() -> int:
    manifest = json.loads((RESULTS / "dev_runs.json").read_text())
    questions = pd.read_parquet(paths.RAW / "questions.parquet")
    dev_ids = ev._load_split_ids("dev")
    indexed = set((paths.PROCESSED / "subsample_doc_ids.txt").read_text().split())

    runs: dict[str, dict] = {}
    metas: dict[str, dict] = {}
    for setup, folder in manifest["runs"].items():
        d = paths.RUNS / folder
        runs[setup] = read_run(d / "run.json")
        metas[setup] = json.loads((d / "meta.json").read_text())

    print("Integrity checks")
    for setup, run in runs.items():
        check(set(run) == set(dev_ids), f"{setup}: queries != 100 dev IDs ({len(run)} present)")
        check(not (set(run) & set(ev._load_split_ids("test"))), f"{setup}: contains test questions")
        check(all(len(h) <= MAX_DEPTH for h in run.values()), f"{setup}: a query has > {MAX_DEPTH} hits")
        check(
            all(all(a[1] >= b[1] for a, b in zip(h, h[1:])) for h in run.values()),
            f"{setup}: hits not sorted by score",
        )
        stray = {schema.doc_id_of(p) for h in run.values() for p, _ in h} - indexed
        check(not stray, f"{setup}: {len(stray)} retrieved docs outside the indexed subsample")

    # Reranked runs must be a reordering of their input's top 50.
    for setup, meta in metas.items():
        src = meta.get("input_run")
        if not src:
            continue
        base = next((s for s, f in manifest["runs"].items() if f == src), None)
        check(base is not None, f"{setup}: input_run {src} is not a pinned folder")
        if base:
            # Compare against the input's top-n, where n is the reranked list's length
            # (50 for the standard runs, up to 100 for the depth-100 experiment).
            same = all(
                {p for p, _ in hits} == {p for p, _ in runs[base][q][: len(hits)]}
                for q, hits in runs[setup].items()
            )
            check(same, f"{setup}: passage set differs from {base} top-n (not a pure rerank)")

    # Every fused run must equal a fresh RRF of the input runs named in its meta.json.
    folder_to_setup = {f: s for s, f in manifest["runs"].items()}
    for setup, meta in metas.items():
        inputs = meta.get("input_runs")
        if not inputs:
            continue
        missing = [f for f in inputs if f not in folder_to_setup]
        check(not missing, f"{setup}: input runs not pinned: {missing}")
        if missing:
            continue
        saved = runs[setup]
        # meta.json doesn't record top_n, so recover it from the saved run's depth.
        top_n = max(len(h) for h in saved.values())
        fresh = index_rrf.fuse(
            [runs[folder_to_setup[f]] for f in inputs], k=meta.get("k_rrf", 60), top_n=top_n
        )
        check(
            all([p for p, _ in fresh[q]] == [p for p, _ in saved[q]] for q in saved),
            f"{setup}: saved run != fresh RRF of {inputs}",
        )

    print("\nScoring")
    rows = []
    for setup, run in runs.items():
        r = {}
        for k in (10, 50):
            res = ev.evaluate_run(
                run, split="dev", k=k, questions_df=questions, split_ids=dev_ids,
                indexed_doc_ids=indexed, write=False,
            )
            check(res["n_questions"] == EXPECTED_N, f"{setup}@{k}: n_questions={res['n_questions']}")
            check(res["overall"]["gold_unindexed"] == 0, f"{setup}@{k}: gold docs not indexed")
            r[k] = res["overall"]
        rows.append({
            "setup": setup,
            "run_folder": manifest["runs"][setup],
            "git_commit": metas[setup].get("git_commit", "missing"),
            "recall@10": round(r[10]["recall@10"], 3),
            "ndcg@10": round(r[10]["ndcg@10"], 3),
            "mrr@10": round(r[10]["mrr@10"], 3),
            "recall@50": round(r[50]["recall@50"], 3),
        })

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "dev_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = list(rows[0])
    md = [
        "# Dev retrieval table (94 scorable dev questions)",
        "",
        "Generated by `scripts/dev_table.py` from the folders pinned in `results/dev_runs.json`.",
        "Do not edit by hand; rerun the script.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
        *("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows),
        "",
    ]
    (RESULTS / "dev_table.md").write_text("\n".join(md))

    print(pd.DataFrame(rows).drop(columns=["run_folder"]).to_string(index=False))
    print(f"\n{len(problems)} check(s) failed" if problems else "\nAll checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
