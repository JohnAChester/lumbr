"""Build a Retriever from an artifacts directory (config.json, embeddings, store, ONNX model)."""
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
    from .encoder import OnnxEncoder  # onnxruntime loads only when serving
    encoder = OnnxEncoder(artifacts / "onnx", config["query_prefix"])
    return Retriever(encoder, DenseIndex.load(artifacts), PassageStore(artifacts / "passages.sqlite"),
                     load_prompt(), float(config["threshold"]), config["retriever"])
