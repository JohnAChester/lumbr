"""Build the full B0 artifacts (R1 BM25 + R4 dense, RRF) from the study's own outputs, recomputing nothing.

Inputs, all on Drive after the study's notebooks:
  --passages    processed/passages.jsonl (the subsample, ~770K passages)
  --r4-shards   indexes/r4/emb_shards (fp16, normalised, one .ids sidecar per shard; notebook 15)
  --r4-model    models/r4_full/final (Jieun's fine-tuned bge-base; exported to ONNX for queries)
  --bm25-index  the staged R1 Lucene index (lumbr.index_bm25.stage_index; notebook 07)

Writes to --out: embeddings.npy (fp16) + ids.json, passages.sqlite, onnx/, bm25/, config.json (threshold
null until pick_threshold.py runs). Then run check_parity.py against the study's dev run.

    python demo/build/build_b0_artifacts.py --passages ... --r4-shards ... --r4-model ... \\
        --bm25-index ... --out /content/b0_artifacts
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo" / "backend"))

import numpy as np  # noqa: E402

from lumbr import paths, schema  # noqa: E402
from lumbr.index_dense import R4  # noqa: E402


def concat_shards(shard_dir: Path) -> tuple[np.ndarray, list[str]]:
    """All finished shards in sorted order (as index_dense.load_index reads them), rows in file order."""
    blocks, ids = [], []
    for npy in sorted(Path(shard_dir).glob("emb_*.npy")):
        sidecar = npy.with_suffix(".ids")
        if not sidecar.exists():
            raise ValueError(f"{npy.name} has no .ids sidecar: the shard never finished")
        m = np.load(npy)
        shard_ids = [ln for ln in sidecar.read_text(encoding="utf-8").splitlines() if ln]
        if len(shard_ids) != m.shape[0]:
            raise ValueError(f"{npy.name}: {m.shape[0]} rows but {len(shard_ids)} ids")
        blocks.append(m.astype(np.float16, copy=False))
        ids.extend(shard_ids)
    if not blocks:
        raise FileNotFoundError(f"no emb_*.npy shards in {shard_dir}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(set(ids))} duplicate passage ids across shards")
    return np.vstack(blocks), ids


def write_dense(matrix: np.ndarray, ids: list[str], out: Path) -> None:
    """fp16 on disk (half the download and half the page cache); DenseIndex scores it as fp32 chunks, as the study cast it."""
    np.save(Path(out) / "embeddings.npy", matrix.astype(np.float16, copy=False))
    (Path(out) / "ids.json").write_text(json.dumps(ids), encoding="utf-8")


def check_same_ids(index_ids: list[str], store_ids: list[str]) -> None:
    a, b = set(index_ids), set(store_ids)
    if a != b:
        raise ValueError(f"{len(a - b)} in the index only, {len(b - a)} in the store only")


def copy_lucene(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    if not any(src.glob("segments_*")):
        raise FileNotFoundError(f"no Lucene segments_* in {src}: stage or build the R1 index first")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def b0_config(n_passages: int, n_docs: int, model_path: str, r4_meta: dict) -> dict:
    return {
        "retriever": "B0", "lexical": "bm25", "dense_index": "faiss_sq_fp16", "threshold": None,
        "query_prefix": R4.query_prefix, "model": R4.name, "model_path": model_path, "r4_index_meta": r4_meta,
        "fusion": {"method": "rrf", "k": 60, "first_stage_k": 50, "order": ["r1", "r4"]},
        "n_passages": n_passages, "n_docs": n_docs,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": paths.git_commit(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--passages", "--r4-shards", "--r4-model", "--bm25-index", "--out"):
        ap.add_argument(a, type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from build_artifacts import export_onnx
    from lumbr_demo.store import build_store
    from sentence_transformers import SentenceTransformer

    matrix, ids = concat_shards(args.r4_shards)
    print(f"[b0] R4 embeddings: {matrix.shape} from {args.r4_shards}")
    write_dense(matrix, ids, args.out)
    del matrix

    n_docs: set[str] = set()
    store_ids: list[str] = []

    def rows():
        for p in schema.read_jsonl(args.passages):
            n_docs.add(p["doc_id"])
            store_ids.append(p["passage_id"])
            yield p
    n = build_store(rows(), args.out / "passages.sqlite")
    check_same_ids(ids, store_ids)
    print(f"[b0] store: {n:,} passages, {len(n_docs):,} docs")

    copy_lucene(args.bm25_index, args.out / "bm25")
    print("[b0] R1 Lucene index copied")

    st = SentenceTransformer(str(args.r4_model), device="cpu")
    st.max_seq_length = 512  # as index_dense.load_model
    export_onnx(st, args.out / "onnx")
    print("[b0] R4 query encoder exported to ONNX")

    meta_path = args.r4_shards.parent / "meta.json"
    r4_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    config = b0_config(n, len(n_docs), str(args.r4_model), r4_meta)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
