"""Build a Retriever from an artifacts directory (config.json, embeddings, store, ONNX model).

config.json "lexical" names the Lucene BM25 index directory inside the artifacts (B0); without it the
service is dense-only, as the v0 sample build is.
"""
from __future__ import annotations

import json
from pathlib import Path

from lumbr.generate import load_prompt

from .index import DenseIndex
from .service import Retriever
from .store import PassageStore


def load_retriever(artifacts: Path) -> Retriever:
    artifacts = Path(artifacts)
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    if config.get("threshold") is None:
        raise ValueError("config.json has no threshold yet: run demo/build/pick_threshold.py")
    from . import encoder, lexical  # onnxruntime and the JVM load only when serving
    lex = lexical.LuceneBM25(artifacts / config["lexical"]) if config.get("lexical") else None
    return Retriever(encoder.OnnxEncoder(artifacts / "onnx", config["query_prefix"]), DenseIndex.load(artifacts),
                     PassageStore(artifacts / "passages.sqlite"), load_prompt(), float(config["threshold"]),
                     config["retriever"], lexical=lex)
