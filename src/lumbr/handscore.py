"""Hand-scoring 50 test answers and the agreement between raters and the judge (#24).

All three of us score the same 50 answers, blind to which setup wrote them, on the benchmark's
own two answer metrics (methodology.md, "Metrics Based Evaluation"):

- **correct**: is the answer aligned with the gold answer? (the benchmark's wholistic correctness
  judge: core aspects addressed, no conflict with the gold answer, quantities match; extra detail
  and style don't count against it). `yes` or `no`.
- **facts**: which of the gold answer facts the answer contains or implies (the benchmark's
  per-fact completeness check), written as their numbers, e.g. `1,3`. Blank means none.

As the benchmark does, citations are stripped before scoring so they don't bias the rater.

The sample is 50 distinct test questions, stratified by question type (largest remainder, so
every type appears), each paired with one of the seven main runs, spread evenly across them. One
answer per question keeps raters from comparing two setups' answers side by side.

Agreement: Cohen's kappa for each pair of raters and for the judge against each rater, Fleiss'
kappa across the three raters. Correctness gives one label per item; completeness gives one label
per (item, fact).
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Hashable, Iterable, Sequence

N_ITEMS = 50
SEED = 24  # the issue number; never change it after the sheets are handed out
RATERS = ("john", "patricia", "jieun")

# The seven main test runs (#22): the six setups and B0, small-to-big on. The two hits-only
# ablations are left out: they share questions and retrievers with r1 / r5_r6.
RUNS = ("gen_r1_p7", "gen_r2_p7", "gen_r3_p7", "gen_r4_p7", "gen_r5_p7", "gen_r5_r6_p7",
        "gen_r1_r4_p7")

SHEET_COLUMNS = ("item", "question_type", "question", "gold_answer", "gold_facts", "answer",
                 "correct", "facts", "notes")
KEY_COLUMNS = ("item", "question_id", "run", "question_type", "abstained")

# Same forms parse_answer accepts ([2], [1][3], [1, 3]), plus the space a removed citation leaves.
_CITE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\](?!\()")


# ---------------------------------------------------------------- the sample

def allocate(type_counts: dict[str, int], n: int = N_ITEMS) -> dict[str, int]:
    """Items per question type, proportional to the test split (largest remainder), at least 1 each."""
    if n < len(type_counts):
        raise ValueError(f"{n} items cannot cover {len(type_counts)} question types")
    total = sum(type_counts.values())
    quotas = {t: n * c / total for t, c in type_counts.items()}
    alloc = {t: max(1, math.floor(q)) for t, q in quotas.items()}
    # Hand out what's left by largest remainder; ties broken by name so the result is stable.
    by_remainder = sorted(type_counts, key=lambda t: (-(quotas[t] - math.floor(quotas[t])), t))
    i = 0
    while sum(alloc.values()) < n:
        t = by_remainder[i % len(by_remainder)]
        if alloc[t] < type_counts[t]:
            alloc[t] += 1
        i += 1
    while sum(alloc.values()) > n:  # only when the minimum of 1 pushed the total over
        t = max((t for t in alloc if alloc[t] > 1), key=lambda t: (alloc[t] - quotas[t], t))
        alloc[t] -= 1
    return alloc


def draw_sample(question_types: dict[str, str], runs: Sequence[str] = RUNS, n: int = N_ITEMS,
                seed: int = SEED) -> list[dict]:
    """The items to score: [{item, question_id, run, question_type}], in the order raters see them.

    question_types maps each test question ID to its type. Questions are drawn per type, then
    shuffled, then dealt to the runs in turn, so every run gets n // len(runs) or one more.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = {}
    for qid in sorted(question_types):
        by_type.setdefault(question_types[qid], []).append(qid)
    alloc = allocate({t: len(q) for t, q in by_type.items()}, n)
    chosen = [qid for t in sorted(by_type) for qid in rng.sample(by_type[t], alloc[t])]
    rng.shuffle(chosen)
    run_order = list(runs)
    rng.shuffle(run_order)
    return [{"item": f"h{i + 1:02d}", "question_id": qid, "run": run_order[i % len(run_order)],
             "question_type": question_types[qid]} for i, qid in enumerate(chosen)]


def strip_citations(text: str) -> str:
    """The answer without its [n] citations, as the benchmark's judge sees it."""
    return _CITE.sub("", text).strip()


def number_facts(facts: Iterable[str]) -> str:
    return "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))


def build_sheets(sample: list[dict], questions: dict[str, dict],
                 answers: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    """(blind sheet rows, key rows). The sheet never names the run; the key does.

    questions: question_id -> {question, question_type, gold_answer, answer_facts}.
    answers: run -> question_id -> answer record (generate.py's answers.jsonl rows).
    """
    sheet, key = [], []
    for s in sample:
        q = questions[s["question_id"]]
        rec = answers[s["run"]][s["question_id"]]
        sheet.append({"item": s["item"], "question_type": q["question_type"],
                      "question": q["question"], "gold_answer": q["gold_answer"],
                      "gold_facts": number_facts(q["answer_facts"]),
                      "answer": strip_citations(rec["answer"]),
                      "correct": "", "facts": "", "notes": ""})
        key.append({"item": s["item"], "question_id": s["question_id"], "run": s["run"],
                    "question_type": q["question_type"], "abstained": rec["abstained"]})
    return sheet, key


def write_csv(rows: list[dict], path: Path, columns: Sequence[str]) -> None:
    """utf-8-sig so Excel and Google Sheets both open it with the right characters."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; score sheets are never overwritten")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- reading filled sheets

def parse_correct(value: str) -> bool:
    v = value.strip().lower()
    if v in ("yes", "y", "1", "true"):
        return True
    if v in ("no", "n", "0", "false"):
        return False
    raise ValueError(f"correct must be yes or no, got {value!r}")


def parse_facts(value: str, n_facts: int) -> set[int]:
    """'1, 3' -> {1, 3}. Blank or 'none' -> set(). Numbers outside 1..n_facts raise."""
    v = value.strip().lower()
    if v in ("", "none", "-"):
        return set()
    out = {int(x) for x in re.split(r"[\s,;]+", v) if x}
    bad = sorted(x for x in out if not 1 <= x <= n_facts)
    if bad:
        raise ValueError(f"fact numbers {bad} outside 1..{n_facts}")
    return out


def labels_from_sheet(rows: list[dict]) -> tuple[dict[str, bool], dict[tuple[str, int], bool]]:
    """(item -> correct, (item, fact number) -> supported) from one rater's filled sheet.

    Raises on a blank or malformed `correct`, naming the item, so a half-filled sheet can't be
    scored by accident.
    """
    correct, facts = {}, {}
    for r in rows:
        try:
            correct[r["item"]] = parse_correct(r["correct"])
            n_facts = len([ln for ln in r["gold_facts"].splitlines() if ln.strip()])
            supported = parse_facts(r["facts"], n_facts)
        except ValueError as e:
            raise ValueError(f"item {r['item']}: {e}") from None
        for k in range(1, n_facts + 1):
            facts[(r["item"], k)] = k in supported
    return correct, facts


# ---------------------------------------------------------------- agreement

def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters' labels on the same items. 1.0 when both are constant and equal."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in ca) / (n * n)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> float:
    """Fleiss' kappa; ratings[i] is every rater's label for item i (same number of raters each)."""
    if not ratings:
        raise ValueError("no items")
    m = len(ratings[0])
    if m < 2 or any(len(r) != m for r in ratings):
        raise ValueError("every item needs the same number (>= 2) of ratings")
    n = len(ratings)
    totals: Counter = Counter()
    p_i = 0.0
    for r in ratings:
        c = Counter(r)
        totals.update(c)
        p_i += (sum(v * v for v in c.values()) - m) / (m * (m - 1))
    p_bar = p_i / n
    p_e = sum((v / (n * m)) ** 2 for v in totals.values())
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def agreement(labels: dict[str, dict[Hashable, bool]]) -> dict:
    """Pairwise Cohen's kappa (and raw agreement) for every pair of label sets, plus Fleiss' kappa
    over all of them. labels: name -> item -> label; only items every set labelled are used."""
    names = list(labels)
    items = sorted(set.intersection(*(set(v) for v in labels.values())), key=str)
    if not items:
        raise ValueError("no item is labelled by every rater")
    pairs = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            a = [labels[x][k] for k in items]
            b = [labels[y][k] for k in items]
            pairs[f"{x}-{y}"] = {"kappa": round(cohen_kappa(a, b), 3),
                                 "agree": round(sum(p == q for p, q in zip(a, b)) / len(a), 3)}
    out = {"n": len(items), "pairs": pairs}
    if len(names) > 2:
        out["fleiss"] = round(fleiss_kappa([[labels[nm][k] for nm in names] for k in items]), 3)
    return out
