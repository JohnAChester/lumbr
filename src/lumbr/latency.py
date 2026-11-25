"""Per-stage latency helpers for issue #25.

API
---
  time_stage(fn, *args, synchronize=False, **kwargs) -> (ms, result)
  median_ms(timings) -> float
  p95_ms(timings)   -> float
  warmup_exclude(timings, n_warmup) -> list[float]        # list form (progress prints)
  warmup_qids(sample_ids, n_warmup) -> set[str]           # question IDs to exclude
  e2e_ms(stage_medians, stages)     -> float              # sum of per-stage medians

  save_stage_timings(stage, question_ids, timings_ms, path)
  load_stage_timings(path) -> dict[str, dict[str, float]] # {stage: {qid: ms}}

  build_setup_rows(stage_timings, warmup_qids_set) -> list[dict]
  format_latency_md(stage_rows, setup_rows, **kwargs) -> str
  write_latency_csv(setup_rows, path)

Stage names used in the notebook (encode and FAISS search timed separately):
  "bm25"       BM25 (Pyserini) single-query search
  "r2_encode"  bge-base-en-v1.5 single-query GPU encode
  "r2_search"  bge-base FAISS search (CPU)
  "r3_encode"  Qwen3-Embedding-4B single-query GPU encode
  "r3_search"  Qwen3-Embedding FAISS search (CPU)
  "r4_encode"  fine-tuned bge-base single-query GPU encode
  "r4_search"  fine-tuned bge-base FAISS search (CPU)
  "rrf"        RRF fusion of two passage-level runs (pure Python)
  "rerank"     bge-reranker-v2-m3 reranking top-50 passages
  "gen"        single-query vLLM generation (Qwen3.5-9B, thinking off,
               timed with R5 context; assumed equal across all setups)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------- timing

def time_stage(fn: Callable, *args: Any, synchronize: bool = False, **kwargs: Any) -> tuple[float, Any]:
    """Call fn(*args, **kwargs) and return (elapsed_ms, result).

    When synchronize=True, calls torch.cuda.synchronize() before and after to
    drain/flush queued CUDA work.  Pass synchronize=True for GPU encode stages.
    """
    if synchronize:
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if synchronize:
        import torch
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, result


# ---------------------------------------------------------------- statistics

def median_ms(timings: list[float]) -> float:
    if not timings:
        raise ValueError("timings list is empty")
    return float(np.median(timings))


def p95_ms(timings: list[float]) -> float:
    if not timings:
        raise ValueError("timings list is empty")
    return float(np.percentile(timings, 95))


def warmup_exclude(timings: list[float], n_warmup: int) -> list[float]:
    """Drop the first n_warmup entries from a plain list (used for in-cell progress prints)."""
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
    return timings[n_warmup:]


def warmup_qids(sample_ids: list[str], n_warmup: int) -> set[str]:
    """Return the first n_warmup question IDs as the warm-up set to exclude from stats."""
    return set(sample_ids[:n_warmup])


def e2e_ms(stage_medians: dict[str, float], stages: list[str]) -> float:
    """Sum the per-stage median latencies for the given stages.

    Note: for the main results table use build_setup_rows, which computes the
    median of per-question sums rather than the sum of per-stage medians.
    """
    return sum(stage_medians[s] for s in stages)


# ---------------------------------------------------------------- checkpoint I/O

def save_stage_timings(
    stage: str,
    question_ids: list[str],
    timings_ms: list[float],
    path: Path,
) -> None:
    """Append one stage's per-question timings to the JSONL checkpoint.

    question_ids[i] corresponds to timings_ms[i].
    Append-only: load_stage_timings takes the last entry per stage, so
    re-saving after re-timing is safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": stage,
                             "question_ids": question_ids,
                             "timings_ms": timings_ms}) + "\n")


def load_stage_timings(path: Path) -> dict[str, dict[str, float]]:
    """Load per-stage per-question timings from a JSONL checkpoint.

    Each line: {"stage": str, "question_ids": [str, ...], "timings_ms": [float, ...]}.
    Returns {stage: {qid: ms}}.  Later entries for the same stage win.
    Returns {} if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["stage"]] = dict(zip(rec["question_ids"], rec["timings_ms"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return out


# ---------------------------------------------------------------- setup compositions

# Retrieval stages for each setup (encode and FAISS search are separate).
SETUP_STAGES: dict[str, list[str]] = {
    "R1":       ["bm25"],
    "R2":       ["r2_encode", "r2_search"],
    "R3":       ["r3_encode", "r3_search"],
    "R4":       ["r4_encode", "r4_search"],
    "R5":       ["bm25", "r3_encode", "r3_search", "rrf"],
    "R1+R4":    ["bm25", "r4_encode", "r4_search", "rrf"],
    "R1+R6":    ["bm25", "rerank"],
    "R3+R6":    ["r3_encode", "r3_search", "rerank"],
    "R4+R6":    ["r4_encode", "r4_search", "rerank"],
    "R5+R6":    ["bm25", "r3_encode", "r3_search", "rrf", "rerank"],
    "R1+R4+R6": ["bm25", "r4_encode", "r4_search", "rrf", "rerank"],
}


def build_setup_rows(
    stage_timings: dict[str, dict[str, float]],
    warmup_qids_set: set[str],
) -> list[dict]:
    """Build per-setup latency rows using per-question sums.

    For each setup, finds the common question IDs present in all retrieval
    stages (excluding warmup), sums each question's stage times, then computes
    the median and p95 of those per-question totals.

    E2E adds the "gen" stage when available (generation timed once with R5
    context; every setup passes the same number of passages, so the per-setup
    generation cost is assumed equal).

    Returns:
        [{"setup": str,
          "retrieval": {"median_ms": float, "p95_ms": float, "n": int},
          "e2e":       {"median_ms": float, "p95_ms": float, "n": int} | None}]
    """
    rows = []
    for label, stages in SETUP_STAGES.items():
        if not all(s in stage_timings for s in stages):
            continue

        # Intersection of questions present in all retrieval stages, minus warmup
        qids: set[str] | None = None
        for s in stages:
            avail = set(stage_timings[s]) - warmup_qids_set
            qids = avail if qids is None else qids & avail
        if not qids:
            continue

        sorted_qids = sorted(qids)
        ret_sums = [sum(stage_timings[s][q] for s in stages) for q in sorted_qids]

        # E2E: add gen times where available
        e2e: dict | None = None
        if "gen" in stage_timings:
            gen_avail = set(stage_timings["gen"]) - warmup_qids_set
            e2e_qids  = sorted(qids & gen_avail)
            # Exclude abstained questions (recorded as 0.0)
            e2e_sums  = [sum(stage_timings[s][q] for s in stages) + stage_timings["gen"][q]
                         for q in e2e_qids if stage_timings["gen"][q] > 0]
            if e2e_sums:
                e2e = {"median_ms": median_ms(e2e_sums),
                       "p95_ms":    p95_ms(e2e_sums),
                       "n":         len(e2e_sums)}

        rows.append({
            "setup":     label,
            "retrieval": {"median_ms": median_ms(ret_sums),
                          "p95_ms":    p95_ms(ret_sums),
                          "n":         len(ret_sums)},
            "e2e":       e2e,
        })
    return rows


# ---------------------------------------------------------------- report helpers

def format_latency_md(
    stage_rows: list[dict],
    setup_rows: list[dict],
    *,
    gpu_name: str = "L4",
    model_ids: dict[str, str] | None = None,
    git_commit: str = "unknown",
    reranker_note: str | None = None,
    generator_note: str | None = None,
    gen_context_note: str | None = None,
) -> str:
    """Build the latency report markdown.

    stage_rows: [{"stage": str, "median_ms": float, "p95_ms": float, "n": int}]
    setup_rows: build_setup_rows() output — per-question median+p95 for retrieval and E2E
    gen_context_note: explains which context was used for generation timing
    """
    model_ids = model_ids or {}
    lines: list[str] = [
        "# Latency per stage\n",
        f"GPU: {gpu_name} · git: {git_commit}\n",
    ]
    if model_ids:
        for tag, mid in sorted(model_ids.items()):
            lines.append(f"- {tag}: `{mid}`")
        lines.append("")

    lines += [
        "## Per-stage latency (single-query, median and p95)\n",
        "| Stage | Median ms | p95 ms | n |",
        "|---|---|---|---|",
    ]
    for r in stage_rows:
        lines.append(
            f"| {r['stage']} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | {r['n']} |"
        )

    if reranker_note:
        lines.append(f"\n> Reranker: {reranker_note}")
    if generator_note:
        lines.append(f"\n> Generator: {generator_note}")
    if gen_context_note:
        lines.append(f"\n> {gen_context_note}")

    lines += [
        "\n## End-to-end latency per setup\n",
        "> Retrieval and E2E = **median of per-question totals** (not sum of per-stage medians).\n",
        "| Setup | Retrieval median ms | Retrieval p95 ms | E2E median ms | E2E p95 ms |",
        "|---|---|---|---|---|",
    ]
    for r in setup_rows:
        ret = r["retrieval"]
        e2e = r.get("e2e")
        e2e_med = f"{e2e['median_ms']:.1f}" if e2e else "—"
        e2e_p95 = f"{e2e['p95_ms']:.1f}"   if e2e else "—"
        lines.append(
            f"| {r['setup']} | {ret['median_ms']:.1f} | {ret['p95_ms']:.1f}"
            f" | {e2e_med} | {e2e_p95} |"
        )

    lines.append(
        "\nRetrieval = sum of retrieval-stage times per question, then median/p95 across questions. "
        "E2E adds single-query generation time per question before aggregating.\n"
    )
    return "\n".join(lines)


def write_latency_csv(setup_rows: list[dict], path: Path) -> None:
    """Write latency.csv for issue #27.

    Columns: setup, retrieval_median_ms, retrieval_p95_ms, e2e_median_ms, e2e_p95_ms
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("setup,retrieval_median_ms,retrieval_p95_ms,e2e_median_ms,e2e_p95_ms\n")
        for r in setup_rows:
            ret = r["retrieval"]
            e2e = r.get("e2e")
            e2e_med = f"{e2e['median_ms']:.3f}" if e2e else ""
            e2e_p95 = f"{e2e['p95_ms']:.3f}"   if e2e else ""
            f.write(
                f"{r['setup']},{ret['median_ms']:.3f},{ret['p95_ms']:.3f}"
                f",{e2e_med},{e2e_p95}\n"
            )
    print(f"[latency] csv → {path}")
