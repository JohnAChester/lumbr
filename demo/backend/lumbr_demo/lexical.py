"""R1 BM25 for the demo: the study's own Lucene index, searched with Pyserini (needs Java 21+).

Pyserini's LuceneSearcher defaults (BM25 k1=0.9, b=0.4, its default English analyzer) are the ones the
study's R1 runs used (lumbr.index_bm25), so the demo serves the same R1.

Memory: Lucene memory-maps its index files (OS page cache, not JVM heap), so the heap can stay small. It is
capped with -Xmx (env LUMBR_JVM_HEAP, default 1g) before the JVM starts, so the 4 GB server keeps room for
the OS cache, the dense vectors and the ONNX encoder.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

DEFAULT_HEAP = "1g"


class Lexical(Protocol):
    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        """Top-k (passage_id, score) pairs, best first."""


def _cap_jvm_heap() -> None:
    import jnius_config  # pyjnius, installed with pyserini; the JVM starts on the first jnius import
    if not jnius_config.vm_running:
        jnius_config.add_options(f"-Xmx{os.environ.get('LUMBR_JVM_HEAP', DEFAULT_HEAP)}")


class LuceneBM25:
    def __init__(self, index_dir: Path | None = None, *, searcher=None):
        if searcher is None:
            _cap_jvm_heap()
            from pyserini.search.lucene import LuceneSearcher  # starts the JVM; only when serving
            searcher = LuceneSearcher(str(index_dir))
        self.searcher = searcher

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        return [(hit.docid, float(hit.score)) for hit in self.searcher.search(text, k)]
