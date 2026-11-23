"""The demo's colleague questions (spec #18, decision 12): dev questions answerable from the served store,
plus the dev info_not_found questions, in the site's DemoQuestion shape. Test questions are never read.

    python demo/build/build_questions.py --artifacts demo/backend/artifacts \\
        --dms demo/content/colleague_dms.csv --out demo/content/questions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV_IDS = ROOT / "splits" / "dev_question_ids.txt"


def load_dev_ids(path: Path = DEV_IDS) -> set[str]:
    if not Path(path).exists():
        sys.exit(f"missing {path}: the dev split is required (never fall back to all questions)")
    return set(Path(path).read_text(encoding="utf-8").split())


def select_questions(questions, dev_ids: set[str], store_doc_ids: set[str]):
    """Dev questions whose (non-empty) gold docs are all in the store, plus dev info_not_found questions."""
    dev = questions[questions.question_id.isin(dev_ids)]
    answerable = dev.expected_doc_ids.map(lambda d: len(d) > 0 and set(d) <= store_doc_ids)
    inf = dev.question_type == "info_not_found"
    return dev[answerable | inf].sort_values("question_id").reset_index(drop=True)


def to_demo(row, dm: str, colleague: str, headers: dict[str, str]) -> dict:
    inf = row.question_type == "info_not_found"
    return {
        "id": row.question_id, "category": row.question_type, "infoNotFound": bool(inf),
        "colleague": colleague, "dm": dm, "question": row.question,
        "answerFacts": [str(f) for f in row.answer_facts], "goldAnswer": row.gold_answer,
        "goldSources": [] if inf else [{"doc_id": d, "header": headers[d]} for d in row.expected_doc_ids],
        "cached": None,
    }


def build(selected, dms: dict[str, tuple[str, str]], headers: dict[str, str]) -> list[dict]:
    missing = [q for q in selected.question_id if q not in dms]
    if missing:
        sys.exit(f"no colleague DM for {', '.join(missing)}: add rows to colleague_dms.csv")
    return [to_demo(r, dm=dms[r.question_id][1], colleague=dms[r.question_id][0], headers=headers)
            for r in selected.itertuples(index=False)]


def store_headers(store: Path) -> dict[str, str]:
    """doc_id -> header of its first chunk."""
    con = sqlite3.connect(store)
    try:
        return {d: json.loads(j)["header"] for d, j in con.execute(
            "SELECT doc_id, json FROM passages WHERE chunk_idx = 0")}
    finally:
        con.close()


def read_dms(path: Path) -> dict[str, tuple[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["question_id"]: (r["colleague"], r["dm"]) for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--dms", type=Path, default=ROOT / "demo" / "content" / "colleague_dms.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from lumbr import data
    dev = load_dev_ids()
    headers = store_headers(args.artifacts / "passages.sqlite")
    selected = select_questions(data.load_questions(), dev, set(headers))
    out = build(selected, read_dms(args.dms), headers)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_inf = sum(q["infoNotFound"] for q in out)
    print(f"{len(out)} questions ({len(out) - n_inf} answerable, {n_inf} info_not_found) -> {args.out}")


if __name__ == "__main__":
    main()
