"""R1 BM25 for the demo: the study's own Lucene index, searched with Pyserini (needs Java 21+).

Pyserini's LuceneSearcher defaults (BM25 k1=0.9, b=0.4, its default English analyzer) are the ones the
study's R1 runs used (lumbr.index_bm25), so the demo serves the same R1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Lexical(Protocol):
    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        """Top-k (passage_id, score) pairs, best first."""


class LuceneBM25:
    def __init__(self, index_dir: Path | None = None, *, searcher=None):
        if searcher is None:
            from pyserini.search.lucene import LuceneSearcher  # starts the JVM; only when serving
            searcher = LuceneSearcher(str(index_dir))
        self.searcher = searcher

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        return [(hit.docid, float(hit.score)) for hit in self.searcher.search(text, k)]
