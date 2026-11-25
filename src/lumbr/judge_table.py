"""Abstention metrics and answer-metrics results table (issue #23, Parts 2 & 3).

Part 2 — abstention (no API calls):
  abstention_metrics(records, questions) → {abstention_rate, false_refusal_rate, n_*}

Part 3 — results table:
  build_answer_metrics_table(run_configs, ...) → markdown string
  write_answer_metrics_table(md, path)

The nine test runs (#22): six setups, B0, two hits-only ablations.
Rows in the final table are grouped:  main runs → ablations → conflicting_info aside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from lumbr import judge as jg

# ---------------------------------------------------------------- Part 2: abstention

INFO_NOT_FOUND = "info_not_found"


def abstention_metrics(
    records: list[dict],
    questions: dict[str, dict],
) -> dict:
    """Compute abstention and false-refusal rates from answer records.

    No API calls: uses only the `abstained` field written by generate.parse_answer.

    Definitions:
      abstention_rate    = abstained / total info_not_found questions
      false_refusal_rate = abstained / total answerable questions
                           (question_type != info_not_found)

    Returns:
        {
          "n_total":            int,
          "n_info_not_found":   int,
          "n_answerable":       int,
          "n_abstained_inf":    int,   # abstained on info_not_found
          "n_abstained_ans":    int,   # abstained on answerable
          "abstention_rate":    float, # INF abstention (want high)
          "false_refusal_rate": float, # answerable abstention (want low)
        }
    """
    n_inf = n_ans = ab_inf = ab_ans = 0
    for rec in records:
        qid  = rec["question_id"]
        qt   = questions.get(qid, {}).get("question_type", "")
        abst = bool(rec.get("abstained", False))
        if qt == INFO_NOT_FOUND:
            n_inf  += 1
            ab_inf += abst
        else:
            n_ans  += 1
            ab_ans += abst

    return {
        "n_total":            n_inf + n_ans,
        "n_info_not_found":   n_inf,
        "n_answerable":       n_ans,
        "n_abstained_inf":    ab_inf,
        "n_abstained_ans":    ab_ans,
        "abstention_rate":    round(ab_inf / n_inf, 4) if n_inf else 0.0,
        "false_refusal_rate": round(ab_ans / n_ans, 4) if n_ans else 0.0,
    }


def abstention_metrics_by_category(
    records: list[dict],
    questions: dict[str, dict],
) -> dict[str, dict]:
    """abstention_metrics per question_type. Useful for the conflicting_info aside."""
    by_type: dict[str, list[dict]] = {}
    for rec in records:
        qt = questions.get(rec["question_id"], {}).get("question_type", "unknown")
        by_type.setdefault(qt, []).append(rec)
    return {qt: abstention_metrics(recs, questions) for qt, recs in by_type.items()}


# ---------------------------------------------------------------- Part 3: results table

def _fmt(val, *, pct: bool = False, missing: str = "—") -> str:
    if val is None:
        return missing
    if pct:
        return f"{val * 100:.1f}%"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _row(
    label: str,
    agg: dict | None,
    abst: dict,
    *,
    incomplete_marker: str = " *",
) -> str:
    """One markdown table row."""
    if agg is None:
        corr = comp = ov = "—"
        marker = ""
    else:
        marker = incomplete_marker if not agg.get("complete", True) else ""
        corr   = _fmt(agg.get("correctness"),   pct=True)
        comp   = _fmt(agg.get("completeness"),  pct=True)
        ov     = _fmt(agg.get("overall"),       pct=True)

    n     = agg["n"] if agg else abst["n_total"]
    # A rate over zero questions is undefined, not 0% (e.g. the conflicting_info subset has
    # no info_not_found questions, so it has no abstention rate).
    abstf = _fmt(abst["abstention_rate"] if abst.get("n_info_not_found", 1) else None, pct=True)
    frr   = _fmt(abst["false_refusal_rate"] if abst.get("n_answerable", 1) else None, pct=True)

    return f"| {label}{marker} | {n} | {corr} | {comp} | {ov} | {abstf} | {frr} |"


def build_answer_metrics_table(
    run_configs: Sequence[dict],
    *,
    section_breaks: dict[str, str] | None = None,
) -> str:
    """Build the answer-metrics markdown table.

    Each entry in run_configs:
        {
          "label":      str,          # display name, e.g. "R1 (BM25)"
          "agg":        dict | None,  # from judge.aggregate()
          "abst":       dict,         # from abstention_metrics()
          "note":       str | None,   # optional footnote text
        }

    section_breaks: {label: heading_text} inserts a heading before that row.
    Returns the full markdown table as a string.
    """
    sb = section_breaks or {}
    header = ["| Run | n | Correctness | Completeness | Overall | Abstention | False-refusal |",
              "|---|---|---|---|---|---|---|"]
    any_partial = any(c.get("agg") and not c["agg"].get("complete", True) for c in run_configs)
    lines: list[str] = ["# Answer metrics — test split", ""]
    if any_partial:
        lines += [r"> \* = run not fully judged; numbers are partial.", ""]
    # Each section is its own table: a heading line between rows would end a markdown table,
    # and the rows after it would render as plain text.
    started = False
    for cfg in run_configs:
        label = cfg["label"]
        if label in sb:
            if started:
                lines.append("")
            lines += [f"**{sb[label]}**", ""] + header
            started = True
        elif not started:
            lines += header
            started = True
        lines.append(_row(label, cfg.get("agg"), cfg["abst"]))
    notes = [f"- {c['label']}: {c['note']}" for c in run_configs if c.get("note")]
    if notes:
        lines += [""] + notes
    lines += [
        "",
        "n = answers judged. Correctness, Completeness, and Overall are means over all judged "
        "answers, abstentions included; Overall = Completeness when the answer is judged "
        "correct, else 0 (the benchmark's combined score). Abstention = share of "
        r"info\_not\_found questions where the system abstained (want high). False-refusal = "
        "share of answerable questions where it abstained (want low). — = undefined (no "
        "questions of that kind in the row).",
        "",
    ]
    return "\n".join(lines)


def write_answer_metrics_table(md: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    print(f"[judge] results table → {path}")
