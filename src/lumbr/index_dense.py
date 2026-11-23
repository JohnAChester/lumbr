"""Dense embedding and FAISS index for R2, R3, and R4 retrieval (issue #10).

Three independent steps — run them in order:

  embed_passages(passages_path, config, ...)   stream-encode to shards on Drive
  build_index(config, ...)                     concatenate shards → FAISS flat-IP index
  search(queries, config, ...)                 load index, encode queries, return top-k

Resumable: each .npy shard is written to local disk and copied to Drive atomically;
the .ids sidecar written last marks completion. On restart, finished shards are
skipped and encoding continues from the next one. A disconnect loses at most one
in-progress shard (~500 MB fp16 for 50 K passages).

A100 wall-clock (R3, 772 K passages, 227 M tokens):
  227e6 tokens × 2 × 4e9 params = 1.82×10¹⁸ FLOPs
  A100 fp16 peak = 312 TFLOPs; at 35–45% utilisation ≈ 3.6–4.6 h encoding
  + ~15 min Drive I/O (16 × ~500 MB shards) + ~8 min model download
  → 4–6 hours total on a fresh A100 (use the throughput cell to project your run).
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lumbr import data, paths, schema

# ---------------------------------------------------------------- model configs

@dataclass(frozen=True)
class ModelConfig:
    name: str                    # HuggingFace model ID (also used as lineage label in meta.json)
    index_tag: str               # subdir under indexes/: "r2", "r3", or "r4"
    query_prefix: str            # prepended to every query at search time (not to passages)
    dtype: str                   # "fp16" (R2/R4/BERT) or "bf16" (R3/Qwen3)
    model_path: str | None = None  # local/Drive path; overrides `name` in load_model when set


R2 = ModelConfig(
    name="BAAI/bge-base-en-v1.5",
    index_tag="r2",
    # Source: https://huggingface.co/BAAI/bge-base-en-v1.5
    # Queries use this instruction; passages need no prefix.
    query_prefix="Represent this sentence for searching relevant passages:",
    dtype="fp16",
)

R3 = ModelConfig(
    name="Qwen/Qwen3-Embedding-4B",
    index_tag="r3",
    # Source: https://huggingface.co/Qwen/Qwen3-Embedding-4B
    # Format: f"Instruct: {task}\nQuery:{query}" — no space after "Query:".
    # Passages are encoded without any prefix.
    query_prefix=(
        "Instruct: Given a web search query, retrieve relevant passages that answer the query"
        "\nQuery:"
    ),
    dtype="bf16",
)

# R4 — fine-tuned from bge-base-en-v1.5 (Jieun's hard-negative fine-tune, issue #10).
# Weights live at paths.MODELS / "r4_full" / "final" on Drive; pooling and query prefix
# are identical to R2. `name` records the base for lineage; `model_path` is what load_model loads.
R4 = ModelConfig(
    name="BAAI/bge-base-en-v1.5",
    index_tag="r4",
    query_prefix="Represent this sentence for searching relevant passages:",
    dtype="fp16",
    model_path=str(paths.MODELS / "r4_full" / "final"),
)

# ---------------------------------------------------------------- paths / constants

EMB_SHARD_DIR = "emb_shards"
INDEX_FILE    = "passages.index"    # FAISS binary
IDS_FILE      = "passage_ids.txt"   # passage IDs in FAISS row order
MANIFEST_FILE = "embed_manifest.json"

SHARD_PASSAGES = 50_000   # passages per embedding shard (~150–500 MB per shard)


def index_dir(config: ModelConfig, root: Path = paths.INDEXES) -> Path:
    return root / config.index_tag


# ---------------------------------------------------------------- model loading

def load_model(config: ModelConfig):
    """Load SentenceTransformer in the right dtype. Requires sentence-transformers."""
    import torch
    from sentence_transformers import SentenceTransformer

    dtype = torch.float16 if config.dtype == "fp16" else torch.bfloat16
    model_kwargs: dict = {"torch_dtype": dtype}

    # Qwen3-Embedding is decoder-only: flash_attention_2 when flash_attn is installed,
    # otherwise SDPA (built into PyTorch ≥ 2.0) — both outperform eager attention.
    if config.dtype == "bf16":
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"

    model = SentenceTransformer(_model_name_or_path(config), model_kwargs=model_kwargs)
    model.max_seq_length = 512  # passages are pre-chunked to ≤512 tokens

    # Decoder-only models (Qwen3) use last-token pooling: pad on the left so the
    # final real token is always at position −1, not buried under padding.
    if config.dtype == "bf16":
        model.tokenizer.padding_side = "left"

    return model


def _model_name_or_path(config: ModelConfig) -> str:
    """Return the path/ID to pass to SentenceTransformer: local path if set, else HF ID."""
    return config.model_path if config.model_path is not None else config.name


# ---------------------------------------------------------------- shard helpers

def _emb_shards(shard_dir: Path) -> list[Path]:
    return sorted(shard_dir.glob("emb_*.npy"))


def _done_ids(shard_dir: Path) -> set[str]:
    """passage_ids in finished shards (a shard counts only once its .ids sidecar exists)."""
    done: set[str] = set()
    for shard in _emb_shards(shard_dir):
        ids_file = shard.with_suffix(".ids")
        if ids_file.exists():
            done.update(ln for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln)
    return done


def _next_shard_no(shard_dir: Path) -> int:
    shards = _emb_shards(shard_dir)
    return 0 if not shards else int(shards[-1].stem.split("_")[1]) + 1


# ---------------------------------------------------------------- embedding

def _flush_shard(
    buf: list[dict],
    no: int,
    shard_dir: Path,
    local_dir: Path,
    model,
    batch_size: int,
) -> int:
    """Encode one buffer of passages, write .npy + .ids atomically. Returns count."""
    # Sort longest-first so each batch has similar-length passages → less padding.
    buf_sorted = sorted(buf, key=lambda p: -p["n_tokens"])
    texts = [p["text"] for p in buf_sorted]
    ids   = [p["passage_id"] for p in buf_sorted]

    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float16)  # fp16 halves Drive footprint; cast back to fp32 at build time

    name = f"emb_{no:04d}"
    local_npy = local_dir / f"{name}.npy"
    np.save(local_npy, emb)
    data.copy_atomic(local_npy, shard_dir / f"{name}.npy")
    local_npy.unlink()
    # Sidecar written last — its presence marks the shard as finished.
    (shard_dir / f"{name}.ids").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return len(ids)


def embed_passages(
    passages_path: Path | str,
    config: ModelConfig,
    *,
    index_root: Path = paths.INDEXES,
    local_root: Path = paths.LOCAL_ROOT,
    shard_size: int = SHARD_PASSAGES,
    batch_size: int = 256,
    n_total: int | None = None,  # passage count for the progress bar; counted from file if None
    progress: bool = True,
    _model=None,  # inject a fake encoder in tests; None loads the real model
) -> int:
    """Encode passages.jsonl into float32 embeddings, checkpointing shards to Drive.

    Returns the total number of passages encoded (including already-finished ones).
    Safe to re-run after a disconnect: finished shards are skipped automatically.

    Args:
        n_total: total passage count shown in the progress bar. When None and
                 progress=True, the file is scanned once to count lines so the bar
                 displays real ETA. Pass the known count to skip that scan.
    """
    idir = index_dir(config, index_root)
    shard_dir = idir / EMB_SHARD_DIR
    local_dir = local_root / config.index_tag / EMB_SHARD_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Drop any .npy whose sidecar was never written (copy completed but crash before .ids).
    for shard in _emb_shards(shard_dir):
        if not shard.with_suffix(".ids").exists():
            shard.unlink()

    done    = _done_ids(shard_dir)
    next_no = _next_shard_no(shard_dir)
    model   = _model if _model is not None else load_model(config)

    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar_total = n_total
        if bar_total is None:
            with open(passages_path, "rb") as _f:
                bar_total = sum(1 for _ in _f)
        bar = tqdm(
            total=bar_total,
            initial=len(done),
            unit="passage",
            desc=f"embed {config.index_tag}",
        )

    buf: list[dict] = []
    total = len(done)

    for p in schema.read_jsonl(passages_path):
        if p["passage_id"] in done:
            continue
        buf.append(p)
        if len(buf) >= shard_size:
            n = _flush_shard(buf, next_no, shard_dir, local_dir, model, batch_size)
            if bar is not None:
                bar.update(n)
            total += n
            next_no += 1
            buf = []

    if buf:
        n = _flush_shard(buf, next_no, shard_dir, local_dir, model, batch_size)
        if bar is not None:
            bar.update(n)
        total += n

    if bar is not None:
        bar.close()
    return total


# ---------------------------------------------------------------- index building

def load_index(
    config: ModelConfig,
    index_root: Path = paths.INDEXES,
) -> tuple:
    """Build IndexFlatIP in memory from embedding shards; return (index, passage_ids).

    Reads shards in sorted order (same order as build_index), casting fp16 → fp32
    per shard to cap peak RAM. Verifies the resulting ID list against passage_ids.txt
    if that file already exists (written by a previous build_index call).
    """
    import faiss

    idir = index_dir(config, index_root)
    shard_dir = idir / EMB_SHARD_DIR
    shards = _emb_shards(shard_dir)
    if not shards:
        raise ValueError(f"No embedding shards in {shard_dir}; run embed_passages() first.")

    all_ids: list[str] = []
    parts: list[np.ndarray] = []
    for shard in shards:
        ids_file = shard.with_suffix(".ids")
        if not ids_file.exists():
            raise ValueError(f"{shard.name} has no .ids sidecar — shard is incomplete.")
        ids = [ln for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln]
        emb = np.load(shard).astype(np.float32)   # cast per-shard to cap peak RAM
        if len(ids) != len(emb):
            raise ValueError(f"{shard.name}: {len(emb)} embeddings but {len(ids)} IDs")
        all_ids.extend(ids)
        parts.append(emb)

    matrix = np.concatenate(parts, axis=0)
    dim = matrix.shape[1]

    # Cross-check against passage_ids.txt if it was written by a previous build_index.
    saved_path = idir / IDS_FILE
    if saved_path.exists():
        saved = [ln for ln in saved_path.read_text(encoding="utf-8").splitlines() if ln]
        if saved != all_ids:
            raise ValueError(
                f"Shard IDs ({len(all_ids)}) do not match {IDS_FILE} ({len(saved)}) — "
                "delete passage_ids.txt or re-run embed_passages."
            )

    idx = faiss.IndexFlatIP(dim)
    idx.add(matrix)
    return idx, all_ids


def build_index(
    config: ModelConfig,
    index_root: Path = paths.INDEXES,
    *,
    write_index: bool = False,
) -> int:
    """Concatenate all embedding shards into a FAISS flat-IP index.

    Always writes passage_ids.txt (small text file needed by search and load_index).
    Writes passages.index only when write_index=True — omit on Drive to avoid an
    8+ GB upload that may be cut by an auto-shutdown.

    Returns the number of passages indexed.
    """
    idx, all_ids = load_index(config, index_root)

    idir = index_dir(config, index_root)
    (idir / IDS_FILE).write_text("\n".join(all_ids) + "\n", encoding="utf-8")

    if write_index:
        import faiss
        faiss.write_index(idx, str(idir / INDEX_FILE))

    print(f"[{config.index_tag}] FAISS index: {len(all_ids):,} passages, dim={idx.d}")
    return len(all_ids)


# ---------------------------------------------------------------- search

def search(
    queries: list[str],
    config: ModelConfig,
    k: int = 50,
    *,
    index_root: Path = paths.INDEXES,
    _model=None,  # inject a fake encoder in tests
) -> list[list[tuple[str, float]]]:
    """Encode queries and search the FAISS index.

    Returns top-k (passage_id, score) per query, sorted descending by score.
    score is cosine similarity (passages and queries are L2-normalised before indexing).
    """
    idir       = index_dir(config, index_root)
    index_file = idir / INDEX_FILE
    if index_file.exists():
        import faiss
        idx = faiss.read_index(str(index_file))
        ids = [ln for ln in (idir / IDS_FILE).read_text(encoding="utf-8").splitlines() if ln]
    else:
        idx, ids = load_index(config, index_root)

    model   = _model if _model is not None else load_model(config)
    q_texts = [config.query_prefix + q for q in queries]
    q_emb   = model.encode(
        q_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    scores, indices = idx.search(q_emb, k)
    return [
        [(ids[i], float(s)) for i, s in zip(row_i, row_s) if i >= 0]
        for row_i, row_s in zip(indices, scores)
    ]


# ---------------------------------------------------------------- run file I/O

def save_run(
    run: dict[str, list[tuple[str, float]]],
    runs_dir: Path,
    config: ModelConfig,
    *,
    split: str = "dev",
    k: int = 50,
    date: str | None = None,
) -> Path:
    """Save a dense retrieval run + metadata to a dated subfolder of runs_dir.

    Subfolder: `{index_tag}_{split}_{date}` (e.g. `r2_dev_2026-11-21`).
    Writes:
      run.json   — passage-level hits (same format as index_bm25.write_run)
      meta.json  — model, query_prefix, dtype, sentence_transformers version,
                   git_commit, split, k, date

    Returns the subdir Path.
    """
    today = date or datetime.date.today().isoformat()
    subdir = runs_dir / f"{config.index_tag}_{split}_{today}"
    subdir.mkdir(parents=True, exist_ok=True)

    run_path = subdir / "run.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f)
    print(f"[{config.index_tag}] run saved → {run_path}")

    try:
        st_ver = importlib.metadata.version("sentence-transformers")
    except importlib.metadata.PackageNotFoundError:
        st_ver = "unknown"

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    meta = {
        "retriever":             config.index_tag,
        "model":                 config.name,
        "query_prefix":          config.query_prefix,
        "dtype":                 config.dtype,
        "split":                 split,
        "k":                     k,
        "sentence_transformers": st_ver,
        "git_commit":            git_sha,
        "date":                  today,
    }
    if config.model_path is not None:
        meta["model_path"] = config.model_path
    with open(subdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{config.index_tag}] meta  → {subdir / 'meta.json'}")
    return subdir
