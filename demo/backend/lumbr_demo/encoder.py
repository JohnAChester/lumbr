"""Query encoders. OnnxEncoder (added with the artifacts) serves R2; HashEncoder is for tests."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, text: str) -> np.ndarray:
        """1-D float32 unit vector."""


class HashEncoder:
    """Deterministic bag-of-words hashing encoder, so tests need no model."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


class OnnxEncoder:
    """R2 query encoder on CPU: the study's prefix + text, CLS token of the last hidden state, L2-normalised."""

    def __init__(self, model_dir, query_prefix: str, max_len: int = 512):
        from pathlib import Path

        import onnxruntime as ort
        from tokenizers import Tokenizer
        model_dir = Path(model_dir)
        self.prefix = query_prefix
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_len)
        self.tokenizer.no_padding()
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, text: str) -> np.ndarray:
        enc = self.tokenizer.encode(self.prefix + text)  # no space after the prefix, as in index_dense
        feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([enc.type_ids], dtype=np.int64)}
        hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        v = hidden[0, 0].astype(np.float32)
        return v / np.linalg.norm(v)
