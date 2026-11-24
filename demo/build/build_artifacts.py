"""Build the demo search service's artifacts from a passages JSONL (spec #18, decisions 9-10).

Writes embeddings.npy + ids.json (R2, sentence-transformers, normalised, float32), passages.sqlite,
onnx/ (the query encoder + tokenizer) and config.json (threshold null until pick_threshold.py runs).

    python demo/build/build_artifacts.py --passages data/drive/processed/sample_passages.jsonl \\
        --out demo/backend/artifacts
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo" / "backend"))

import numpy as np  # noqa: E402

from lumbr import paths  # noqa: E402
from lumbr.index_dense import R2  # noqa: E402
from lumbr_demo.index import DenseIndex  # noqa: E402
from lumbr_demo.store import build_store  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_onnx(st_model, out: Path) -> None:
    """The transformer under the SentenceTransformer, as ONNX with dynamic batch and length."""
    import torch

    class ByName(torch.nn.Module):
        """Passes inputs by keyword: newer transformers changed BertModel.forward's positional order."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state

    out.mkdir(parents=True, exist_ok=True)
    model = ByName(st_model[0].auto_model.eval())
    tok = st_model.tokenizer
    tok.save_pretrained(out)
    sample = tok(["hello world"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dyn = {n: {0: "batch", 1: "seq"} for n in names} | {"last_hidden_state": {0: "batch", 1: "seq"}}
    with torch.no_grad():
        torch.onnx.export(model, tuple(sample[n] for n in names), str(out / "model.onnx"),
                          input_names=names, output_names=["last_hidden_state"], dynamic_axes=dyn,
                          opset_version=17, dynamo=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--onnx-only", action="store_true",
                    help="keep the existing embeddings and store; redo the ONNX export and config")
    args = ap.parse_args()

    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    rows = read_jsonl(args.passages)
    args.out.mkdir(parents=True, exist_ok=True)
    revision = model_info(R2.name).sha
    st = SentenceTransformer(R2.name, revision=revision, device="cpu")
    st.max_seq_length = 512  # as index_dense.load_model
    if args.onnx_only:
        if DenseIndex.load(args.out).ids != [p["passage_id"] for p in rows]:
            sys.exit("existing embeddings do not match --passages; rebuild without --onnx-only")
        n = len(rows)
    else:
        # Sorted longest-first like index_dense, so batches pad less; rows keep their file order in the index.
        order = sorted(range(len(rows)), key=lambda i: -rows[i].get("n_tokens", 0))
        emb_sorted = st.encode([rows[i]["text"] for i in order], batch_size=args.batch_size,
                               normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        emb = np.empty_like(emb_sorted, dtype=np.float32)
        emb[order] = emb_sorted
        DenseIndex(emb, [p["passage_id"] for p in rows]).save(args.out)
        n = build_store(rows, args.out / "passages.sqlite")
    export_onnx(st, args.out / "onnx")
    config = {"retriever": "r2", "model": R2.name, "model_revision": revision, "query_prefix": R2.query_prefix,
              "n_passages": n, "n_docs": len({p["doc_id"] for p in rows}), "threshold": None,
              "passages_file": args.passages.name,
              "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "git_commit": paths.git_commit()}
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
