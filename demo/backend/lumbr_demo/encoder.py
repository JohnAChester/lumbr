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
