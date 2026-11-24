"""Build a Retriever from an artifacts directory (config.json, embeddings, store, ONNX model).

config.json "lexical" names the Lucene BM25 index directory inside the artifacts (B0); without it the
service is dense-only, as the v0 sample build is. "dense_index" picks the dense search: "faiss_sq_fp16"
(memory-mapped, built from embeddings.npy on first load) or "numpy"; it defaults to FAISS for B0 and numpy
otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

from lumbr.generate import load_prompt

from .index import DenseIndex, FaissDenseIndex
from .service import Retriever
from .store import PassageStore


def load_retriever(artifacts: Path) -> Retriever:
    artifacts = Path(artifacts)
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    if config.get("threshold") is None:
        raise ValueError("config.json has no threshold yet: run demo/build/pick_threshold.py")
    from . import encoder, lexical  # onnxruntime and the JVM load only when serving
    lex = lexical.LuceneBM25(artifacts / config["lexical"]) if config.get("lexical") else None
    dense = config.get("dense_index") or ("faiss_sq_fp16" if lex else "numpy")
    index = FaissDenseIndex.load(artifacts) if dense == "faiss_sq_fp16" else DenseIndex.load(artifacts)
    return Retriever(encoder.OnnxEncoder(artifacts / "onnx", config["query_prefix"]), index,
                     PassageStore(artifacts / "passages.sqlite"), load_prompt(), float(config["threshold"]),
                     config["retriever"], lexical=lex)
