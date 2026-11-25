"""Answer-parsing cases for the site's TypeScript parseAnswer (spec #18, decision 7).

Each case is run through lumbr.generate.parse_answer; the site's tests must give the same results.
The file also carries the abstention rule's refusal patterns (generate.REFUSAL_PATTERNS), which the
site's parser reads at runtime instead of keeping its own copy (abstention spec 2026-11-23).

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


# The lead rule (abstain_rule 2): text, then whether it abstains. All use 5 sources.
LEAD_CASES: list[tuple[str, bool]] = [
    ("Based on the provided sources, there is no record of a Teams integration. "
     "Source [3] describes the Slack bot [3].", True),                      # own-words refusal that cites
    ("Here are the requirements:\n* There is no limit on retries [1].\n* Timeouts are 30 s [2].", False),
    ("The sources do not specify the queue-depth threshold. The interleaving ratio is 0.6 [1].", True),
    ("I can’t answer that from these sources [2].", True),             # curly apostrophe
    ("\n\nThere is no mention of that in the sources.", True),              # leading blank lines
    ("There is no rate limit on Dedicated [2].", True),                     # negative fact: known trade-off
    ("The limit is 10 MiB [2]. I cannot answer the second part.", False),   # refusal after the lead
]


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
    ] + [(text, 5) for text, _ in LEAD_CASES]


def build() -> dict:
    prompt = generate.load_prompt()
    cases = []
    for text, n in texts(prompt["abstain"]):
        got = generate.parse_answer(text, sources(n), prompt)
        cases.append({"text": text, "n_sources": n, "citations": [c["n"] for c in got["citations"]],
                      "invalid_citations": got["invalid_citations"], "abstained": got["abstained"]})
    return {"prompt_sha256": prompt["sha256"], "abstain": prompt["abstain"],
            "abstain_rule": generate.ABSTAIN_RULE, "refusal_patterns": list(generate.REFUSAL_PATTERNS),
            "cases": cases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "content" / "parse_cases.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
