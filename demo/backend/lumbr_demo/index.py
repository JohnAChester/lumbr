"""Exact cosine search over unit-normalised vectors.

Memory-lean for a 4 GB server: `load` memory-maps embeddings.npy (fp16 for the full B0 build, ~1.2 GB) instead
of reading it into RAM, and `search` scores it in chunks, casting each chunk to float32 first. fp16 -> fp32 is
exact, as the study's index_dense.load_index cast; scores can differ from the study's FAISS only in the last bit
(summation order), so rankings match except at near-exact ties.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHUNK_ROWS = 65_536  # rows scored per step: 65K x 768 float32 is ~200 MB of temporary memory


class DenseIndex:
    def __init__(self, matrix: np.ndarray, ids: list[str], chunk_rows: int = CHUNK_ROWS):
        if matrix.shape[0] != len(ids):
            raise ValueError(f"{matrix.shape[0]} vectors but {len(ids)} ids")
        self.matrix = matrix  # kept as given: an ndarray or a read-only np.memmap
        self.ids = list(ids)
        self.chunk_rows = chunk_rows

    def _scores(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        if self.matrix.dtype == np.float32 and not isinstance(self.matrix, np.memmap):
            return self.matrix @ q
        out = np.empty(len(self.ids), dtype=np.float32)
        for start in range(0, len(self.ids), self.chunk_rows):
            stop = min(start + self.chunk_rows, len(self.ids))
            out[start:stop] = np.asarray(self.matrix[start:stop], dtype=np.float32) @ q
        return out

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k (id, cosine) pairs, best first; ties broken by id.

        A partial sort keeps this fast on the full corpus (~770K passages): every candidate
        scoring at least the k-th best score is kept, so ties at the cut are still broken by id.
        """
        scores = self._scores(q)
        n = len(self.ids)
        if k < n:
            kth = np.partition(scores, n - k)[n - k]
            candidates = np.flatnonzero(scores >= kth)
        else:
            candidates = np.arange(n)
        order = sorted(candidates.tolist(), key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        np.save(directory / "embeddings.npy", self.matrix)
        (directory / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, chunk_rows: int = CHUNK_ROWS) -> "DenseIndex":
        directory = Path(directory)
        matrix = np.load(directory / "embeddings.npy", mmap_mode="r")
        return cls(matrix, json.loads((directory / "ids.json").read_text(encoding="utf-8")), chunk_rows)


FAISS_FILE = "dense_sq_fp16.faiss"


class FaissDenseIndex:
    """The B0 server's dense index: a FAISS fp16 scalar-quantizer index, memory-mapped from disk.

    FAISS decodes fp16 to fp32 in SIMD while scoring (exact, like the study's fp16 -> fp32 cast), so it is about
    twice as fast as the chunked numpy cast and holds almost no process memory (the mapped file is OS cache).
    The .faiss file is built from embeddings.npy on first load (seconds) and reused after that.
    """

    def __init__(self, index, ids: list[str]):
        if index.ntotal != len(ids):
            raise ValueError(f"{index.ntotal} vectors in the FAISS index but {len(ids)} ids")
        self.index, self.ids = index, list(ids)

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k (id, cosine) pairs, best first."""
        scores, rows = self.index.search(np.asarray(q, dtype=np.float32).reshape(1, -1), min(k, len(self.ids)))
        return [(self.ids[r], float(s)) for s, r in zip(scores[0], rows[0]) if r >= 0]

    @staticmethod
    def build(directory: Path, chunk_rows: int = CHUNK_ROWS) -> Path:
        import faiss
        directory = Path(directory)
        m = np.load(directory / "embeddings.npy", mmap_mode="r")
        index = faiss.IndexScalarQuantizer(m.shape[1], faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        for start in range(0, m.shape[0], chunk_rows):
            index.add(np.asarray(m[start:start + chunk_rows], dtype=np.float32))
        out = directory / FAISS_FILE
        tmp = out.with_name(out.name + ".partial")
        faiss.write_index(index, str(tmp))
        tmp.replace(out)
        return out

    @classmethod
    def load(cls, directory: Path) -> "FaissDenseIndex":
        import faiss
        directory = Path(directory)
        path = directory / FAISS_FILE
        if not path.exists():
            cls.build(directory)
        index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP_IFC)
        return cls(index, json.loads((directory / "ids.json").read_text(encoding="utf-8")))
