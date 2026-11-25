"""Provisional similarity threshold (spec #18, decision 11): lowest top score among the dev questions
whose gold documents are all in the store, minus 0.02. Reads dev questions only; writes config.json.

    python demo/build/pick_threshold.py --artifacts demo/backend/artifacts
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo" / "backend"))

from lumbr import data  # noqa: E402
from lumbr_demo.encoder import OnnxEncoder  # noqa: E402
from lumbr_demo.index import DenseIndex  # noqa: E402

MARGIN = 0.02
DEV_IDS = ROOT / "splits" / "dev_question_ids.txt"


def store_doc_ids(artifacts: Path) -> set[str]:
    con = sqlite3.connect(artifacts / "passages.sqlite")
    try:
        return {d for (d,) in con.execute("SELECT DISTINCT doc_id FROM passages")}
    finally:
        con.close()


def choose_threshold(answerable_scores: list[float], margin: float = MARGIN) -> float:
    return round(min(answerable_scores) - margin, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    args = ap.parse_args()
    if not DEV_IDS.exists():
        sys.exit(f"missing {DEV_IDS}: the dev split is required (never fall back to all questions)")
    dev = set(DEV_IDS.read_text(encoding="utf-8").split())
    config_path = args.artifacts / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    enc = OnnxEncoder(args.artifacts / "onnx", config["query_prefix"])
    index = DenseIndex.load(args.artifacts)
    docs = store_doc_ids(args.artifacts)

    qs = data.load_questions()
    qs = qs[qs.question_id.isin(dev)]
    answerable = qs[qs.expected_doc_ids.map(lambda d: len(d) > 0 and set(d) <= docs)]
    inf = qs[qs.question_type == "info_not_found"]

    scores = {}
    for _, q in answerable.iterrows():
        scores[q.question_id] = index.search(enc.encode(q.question), 1)[0][1]
        print(f"answerable {q.question_id} {scores[q.question_id]:.4f}")
    for _, q in inf.iterrows():
        print(f"info_not_found {q.question_id} {index.search(enc.encode(q.question), 1)[0][1]:.4f} (not used)")
    config["threshold"] = choose_threshold(list(scores.values()))
    config["threshold_from"] = {"questions": sorted(scores), "margin": MARGIN}
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"threshold {config['threshold']} from {len(scores)} answerable dev questions")


if __name__ == "__main__":
    main()
