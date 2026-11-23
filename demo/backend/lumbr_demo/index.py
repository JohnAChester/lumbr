"""Exact cosine search over unit-normalised float32 vectors (the v0 sample is ~5K passages)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(self, matrix: np.ndarray, ids: list[str]):
        if matrix.shape[0] != len(ids):
            raise ValueError(f"{matrix.shape[0]} vectors but {len(ids)} ids")
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.ids = list(ids)

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k (id, cosine) pairs, best first; ties broken by id."""
        scores = self.matrix @ np.asarray(q, dtype=np.float32)
        order = sorted(range(len(self.ids)), key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        np.save(directory / "embeddings.npy", self.matrix)
        (directory / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        directory = Path(directory)
        return cls(np.load(directory / "embeddings.npy"), json.loads((directory / "ids.json").read_text(encoding="utf-8")))
