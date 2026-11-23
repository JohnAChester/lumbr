"""Cross-encoder reranking (R6) with BAAI/bge-reranker-v2-m3 (issue #10).

Two-step pipeline:
  1. build_passage_lookup(run, passages_path) — scan passages.jsonl once,
     collect only the IDs present in the run (keeps peak memory low for 772 K passages).
  2. rerank(run, queries, passage_lookup, ...) — score (query, passage) pairs
     with the cross-encoder and return top_n reranked passages per query.

Returns latency stats (median and p95 ms per question) alongside the run;
save them in meta.json via save_run().

L4 wall-clock (bge-reranker-v2-m3 fp16, 94 dev questions × 50 passages,
batch_size=64): < 2 min per input run.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import numpy as np

from lumbr import paths, schema

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------- model loading

def load_model(*, device: str | None = None):
    """Load bge-reranker-v2-m3 as a CrossEncoder in fp16.

    Falls back to CPU when CUDA is unavailable. Requires sentence-transformers.
    """
    import torch
    from sentence_transformers import CrossEncoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1024 covers a ≤512-token passage plus the question with room to spare,
    # avoiding the model's 8 K default which pads every batch to full context length.
    return CrossEncoder(
        RERANKER_MODEL,
        max_length=1024,
        device=device,
        automodel_args={"torch_dtype": torch.float16},
    )


# ---------------------------------------------------------------- passage lookup

def build_passage_lookup(
    run: dict[str, list[tuple[str, float]]],
    passages_path: Path | str,
) -> tuple[dict[str, str], set[str]]:
    """Single-pass scan of passages_path; returns only IDs present in run.

    Building the lookup from the run rather than loading all 772 K passages caps
    peak memory at ~tens of MB instead of several GB.

    Returns:
        (lookup, missing_ids)
        lookup:      {passage_id: text} for every ID that was found
        missing_ids: set of IDs requested by run that were not found in the file
    """
    needed = {pid for hits in run.values() for pid, _ in hits}
    lookup: dict[str, str] = {}
    for p in schema.read_jsonl(passages_path):
        pid = p["passage_id"]
        if pid in needed:
            lookup[pid] = p["text"]
        if len(lookup) >= len(needed):
            break
    missing = needed - set(lookup)
    return lookup, missing


# ---------------------------------------------------------------- reranking

def rerank(
    run: dict[str, list[tuple[str, float]]],
    queries: dict[str, str],
    passage_lookup: dict[str, str],
    *,
    top_n: int = 50,
    batch_size: int = 64,
    progress: bool = True,
    allow_missing: bool = False,
    _model=None,  # inject a fake cross-encoder in tests; None loads the real model
) -> tuple[dict[str, list[tuple[str, float]]], dict]:
    """Rerank each query's top_n passages using a cross-encoder.

    Args:
        run:            passage-level run: {qid: [(pid, score), ...]}
        queries:        {qid: question_text}
        passage_lookup: {pid: text} — built by build_passage_lookup
        top_n:          passages to rerank and return per query
        batch_size:     cross-encoder inference batch size (tune for VRAM)
        progress:       show tqdm progress bar
        allow_missing:  if False (default), raise ValueError when any pid in run
                        is absent from passage_lookup; if True, substitute empty
                        string (the cross-encoder will score it very low)
        _model:         inject a fake encoder in tests; None loads the real model

    Returns:
        (reranked_run, stats)
        reranked_run: same format as run, sorted descending by cross-encoder score
        stats: {"median_ms": float, "p95_ms": float, "n_questions": int}
    """
    model = _model if _model is not None else load_model()

    # Fail fast: check for missing query texts before spending time on inference.
    missing_qids = set(run) - set(queries)
    if missing_qids:
        sample = sorted(missing_qids)[:5]
        raise ValueError(
            f"{len(missing_qids):,} of {len(run):,} question IDs in run are missing from "
            f"queries: {sample}{'...' if len(missing_qids) > 5 else ''}. "
            "Pass a queries dict containing all question IDs present in run."
        )

    # Fail fast: check for missing passages before spending time on inference.
    all_pids = {pid for hits in run.values() for pid, _ in hits}
    missing_pids = all_pids - set(passage_lookup)
    if missing_pids and not allow_missing:
        raise ValueError(
            f"{len(missing_pids):,} of {len(all_pids):,} passage IDs are missing from "
            "passage_lookup. Call build_passage_lookup first (assert missing == set()), "
            "or pass allow_missing=True to substitute empty strings."
        )

    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(run), unit="query", desc="rerank")

    reranked: dict[str, list[tuple[str, float]]] = {}
    latencies_ms: list[float] = []

    for qid, hits in run.items():
        candidates = hits[:top_n]
        q_text = queries[qid]  # presence verified upfront
        pairs = [(q_text, passage_lookup.get(pid, "")) for pid, _ in candidates]

        if pairs:
            t0 = time.perf_counter()
            scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            scored = [(pid, float(s)) for (pid, _), s in zip(candidates, scores)]
        else:
            latencies_ms.append(0.0)
            scored = []

        reranked[qid] = sorted(scored, key=lambda x: -x[1])[:top_n]

        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    arr = np.array(latencies_ms)
    stats = {
        "median_ms":   round(float(np.median(arr)), 3),
        "p95_ms":      round(float(np.percentile(arr, 95)), 3),
        "n_questions": len(run),
    }
    return reranked, stats


# ---------------------------------------------------------------- run file I/O

def save_run(
    run: dict[str, list[tuple[str, float]]],
    runs_dir: Path,
    *,
    tag: str,
    split: str = "dev",
    input_run_dir: Path | str,
    latency_stats: dict,
    date: str | None = None,
) -> Path:
    """Save a reranked run to a dated subfolder of runs_dir.

    Subfolder: `{tag}_{split}_{date}` (e.g. `r1_r6_dev_2026-11-21`).
    Writes:
      run.json   — passage-level hits
      meta.json  — model, input_run, latency_stats, git_commit, split, date

    Returns the subdir Path.
    """
    today = date or datetime.date.today().isoformat()
    subdir = runs_dir / f"{tag}_{split}_{today}"
    subdir.mkdir(parents=True, exist_ok=True)

    run_path = subdir / "run.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f)
    print(f"[{tag}] run saved → {run_path}")

    git_sha = paths.git_commit()

    meta = {
        "retriever":     tag,
        "model":         RERANKER_MODEL,
        "input_run":     Path(input_run_dir).name,
        "latency_stats": latency_stats,
        "split":         split,
        "git_commit":    git_sha,
        "date":          today,
    }
    meta_path = subdir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{tag}] meta  → {meta_path}")
    return subdir
