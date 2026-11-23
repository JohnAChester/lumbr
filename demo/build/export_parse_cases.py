"""Answer-parsing cases for the site's TypeScript parseAnswer (spec #18, decision 7).

Each case is run through lumbr.generate.parse_answer; the site's tests must give the same results.

    python demo/build/export_parse_cases.py --out demo/content/parse_cases.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumbr import generate

ROOT = Path(__file__).resolve().parents[2]


def sources(n: int) -> list[dict]:
    """n placeholder sources: parse_answer only needs n, passage_ids and doc_id."""
    return [{"n": i, "doc_id": f"dsid_{i}", "passage_ids": [f"dsid_{i}#000"]} for i in range(1, n + 1)]


def texts(abstain: str) -> list[tuple[str, int]]:
    curly = abstain.replace("'", chr(0x2019))
    words = abstain.split()
    # "... can't find that specific detailed information ..." (2 inserted words: still an abstention)
    inserted2 = " ".join(words[:7] + ["specific", "detailed"] + words[7:])
    inserted3 = " ".join(words[:7] + ["specific", "detailed", "pricing"] + words[7:])
    return [
        ("The limit is 10 MiB [2].", 3),
        ("Both apply [1][3].", 3),
        ("Both apply [1, 3].", 3),
        ("It ships in May [7].", 5),
        ("See [2](http://example.com) for details [1].", 2),
        ("Cite like [n] and [n][m].", 5),
        ("Mixed [1, 9] here.", 2),
        ("Repeated [1] and again [1].", 1),
        (abstain, 5),
        (inserted2, 5),
        (inserted3, 5),
        (curly, 5),
        (abstain.lower(), 5),
        (abstain + " The sources do cover the rollout date [2].", 5),
        ("The rollout date is May 4 [2]. " + abstain, 5),
        ("", 0),
        ("No citations at all.", 3),
    ]


def build() -> dict:
    prompt = generate.load_prompt()
    cases = []
    for text, n in texts(prompt["abstain"]):
        got = generate.parse_answer(text, sources(n), prompt)
        cases.append({"text": text, "n_sources": n, "citations": [c["n"] for c in got["citations"]],
                      "invalid_citations": got["invalid_citations"], "abstained": got["abstained"]})
    return {"prompt_sha256": prompt["sha256"], "abstain": prompt["abstain"], "cases": cases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "content" / "parse_cases.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
