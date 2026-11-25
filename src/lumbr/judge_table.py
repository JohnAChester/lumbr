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
    abstf = _fmt(abst["abstention_rate"],    pct=True)
    frr   = _fmt(abst["false_refusal_rate"], pct=True)

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
    lines: list[str] = [
        "# Answer metrics — test split\n",
        "> \\* = run not fully judged; numbers are partial.\n",
        "| Run | n | Correctness | Completeness | Overall | Abstention | False-refusal |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in run_configs:
        label = cfg["label"]
        if label in sb:
            lines.append(f"\n**{sb[label]}**\n")
        lines.append(_row(label, cfg.get("agg"), cfg["abst"]))
        if cfg.get("note"):
            lines.append(f"> {label}: {cfg['note']}")
    lines.append(
        "\nCorrectness, Completeness, and Overall are macro-averages across all "
        "judged answers (abstentions included). Abstention = share of "
        "info\\_not\\_found questions where the system abstained (want high). "
        "False-refusal = share of answerable questions where it abstained (want low).\n"
    )
    return "\n".join(lines)


def write_answer_metrics_table(md: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    print(f"[judge] results table → {path}")
