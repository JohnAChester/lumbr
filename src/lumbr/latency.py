"""Per-stage latency helpers for issue #25.

API
---
  time_stage(fn, *args, synchronize=False, **kwargs) -> (ms, result)
  median_ms(timings) -> float
  p95_ms(timings)   -> float
  warmup_exclude(timings, n_warmup) -> list[float]
  e2e_ms(stage_medians, stages)     -> float

  load_stage_timings(path)             -> dict[str, list[float]]
  save_stage_timings(stage, timings, path)

  format_latency_md(stage_rows, setup_rows, **kwargs) -> str
  write_latency_csv(setup_rows, path)

Stage names used in the notebook:
  "bm25"    BM25 (Pyserini) single-query search
  "r2"      bge-base encode + FAISS search
  "r3"      Qwen3-Embedding-4B encode + FAISS search
  "r4"      fine-tuned bge-base encode + FAISS search
  "rrf"     RRF fusion of two passage-level runs (pure Python)
  "rerank"  bge-reranker-v2-m3 reranking top-50 passages
  "gen"     single-query vLLM generation (Qwen3.5-9B, thinking off)
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

    When synchronize=True, calls torch.cuda.synchronize() before and after to drain/flush
    any queued CUDA work.  Pass synchronize=True for all GPU stages (dense encode, rerank, gen).
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
    """Drop the first n_warmup entries (warm-up phase excluded from stats)."""
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
    return timings[n_warmup:]


def e2e_ms(stage_medians: dict[str, float], stages: list[str]) -> float:
    """Sum the median latencies for the named stages."""
    return sum(stage_medians[s] for s in stages)


# ---------------------------------------------------------------- checkpoint I/O

def load_stage_timings(path: Path) -> dict[str, list[float]]:
    """Load per-stage timings from a JSONL checkpoint.

    Each line: {"stage": str, "timings_ms": [float, ...]}.
    Returns {stage: timings_ms}.  Later entries for the same stage win (re-save is safe).
    Returns {} if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["stage"]] = rec["timings_ms"]
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def save_stage_timings(stage: str, timings: list[float], path: Path) -> None:
    """Append one stage's raw timings to the JSONL checkpoint (append-only; never overwrites).

    load_stage_timings takes the last entry per stage, so calling this again after re-timing
    is safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": stage, "timings_ms": timings}) + "\n")


# ---------------------------------------------------------------- report helpers

# E2E stage compositions for every setup.
# Key = display label; value = list of stage keys to sum.
SETUP_STAGES: dict[str, list[str]] = {
    "R1":       ["bm25"],
    "R2":       ["r2"],
    "R3":       ["r3"],
    "R4":       ["r4"],
    "R5":       ["bm25", "r3", "rrf"],
    "R1+R4":    ["bm25", "r4", "rrf"],
    "R1+R6":    ["bm25", "rerank"],
    "R3+R6":    ["r3", "rerank"],
    "R4+R6":    ["r4", "rerank"],
    "R5+R6":    ["bm25", "r3", "rrf", "rerank"],
    "R1+R4+R6": ["bm25", "r4", "rrf", "rerank"],
}


def build_setup_rows(
    stage_medians: dict[str, float],
    *,
    gen_median: float | None = None,
) -> list[dict]:
    """Build setup rows for format_latency_md / write_latency_csv.

    Each row: {"setup": str, "retrieval_ms": float, "e2e_ms": float | None}.
    e2e_ms adds gen_median when available, otherwise None.
    """
    rows = []
    for label, stages in SETUP_STAGES.items():
        if not all(s in stage_medians for s in stages):
            continue
        ret = e2e_ms(stage_medians, stages)
        total = ret + gen_median if gen_median is not None else None
        rows.append({"setup": label, "retrieval_ms": ret, "e2e_ms": total})
    return rows


def format_latency_md(
    stage_rows: list[dict],
    setup_rows: list[dict],
    *,
    gpu_name: str = "L4",
    model_ids: dict[str, str] | None = None,
    git_commit: str = "unknown",
    reranker_note: str | None = None,
    generator_note: str | None = None,
) -> str:
    """Build the latency report markdown.

    stage_rows: [{"stage": str, "median_ms": float, "p95_ms": float, "n": int}]
    setup_rows: [{"setup": str, "retrieval_ms": float, "e2e_ms": float | None}]
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
        lines.append(f"\n> Generator: {generator_note} (**batched throughput** — not per-query)")

    lines += [
        "\n## End-to-end latency per setup\n",
        "| Setup | Retrieval ms | E2E (+ gen) ms |",
        "|---|---|---|",
    ]
    for r in setup_rows:
        e2e = f"{r['e2e_ms']:.1f}" if r.get("e2e_ms") is not None else "—"
        lines.append(f"| {r['setup']} | {r['retrieval_ms']:.1f} | {e2e} |")

    lines.append(
        "\nRetrieval = sum of per-stage medians (no generation). "
        "E2E adds the single-query generation median.\n"
    )
    return "\n".join(lines)


def write_latency_csv(setup_rows: list[dict], path: Path) -> None:
    """Write latency.csv (columns: setup, retrieval_ms, e2e_ms) for issue #27."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("setup,retrieval_ms,e2e_ms\n")
        for r in setup_rows:
            e2e = f"{r['e2e_ms']:.3f}" if r.get("e2e_ms") is not None else ""
            f.write(f"{r['setup']},{r['retrieval_ms']:.3f},{e2e}\n")
    print(f"[latency] csv → {path}")
