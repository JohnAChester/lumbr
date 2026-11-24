"""One Lumbr turn: retrieve, apply the similarity filter, build the study's sources and messages.

With a lexical searcher this is B0: the top FIRST_STAGE_K of BM25 (R1) and of the dense index (R4), fused
with the study's own RRF (lumbr.index_rrf.fuse). The similarity filter always reads the dense top cosine:
RRF scores only reflect rank, so they can't tell an off-topic question from a real one.

Context and messages come only from lumbr.generate, so the demo's prompt is the study's (spec #18, decision 4-6).
"""
from __future__ import annotations

import re

from lumbr import generate, index_rrf

from .encoder import Encoder
from .index import DenseIndex
from .lexical import Lexical
from .store import PassageStore

MAX_HITS = 8  # the site's MAX_CONTEXT_PASSAGES
FIRST_STAGE_K = 50  # passages taken from each run before fusion (notebooks 13, 17)
RRF_K = 60  # RRF smoothing constant, frozen on dev (notebook 13)
PASSAGE_FIELDS = ("source", "title", "channel", "author", "date")


def strip_citations(text: str) -> str:
    """`text` without [n] citation markers (numbering restarts every turn)."""
    out = generate._CITE.sub("", text)
    out = re.sub(r" {2,}", " ", out)
    return re.sub(r" ([.,;:!?])", r"\1", out).strip()


def _passage(row: dict, score: float | None) -> dict:
    return {"passage_id": row["passage_id"], "doc_id": row["doc_id"],
            **{f: row.get(f) for f in PASSAGE_FIELDS},
            "header": row["header"], "body": row["body"], "score": score}


class Retriever:
    def __init__(self, encoder: Encoder, index: DenseIndex, store: PassageStore, prompt: dict,
                 threshold: float, name: str, lexical: Lexical | None = None):
        self.encoder, self.index, self.store, self.prompt = encoder, index, store, prompt
        self.threshold, self.name, self.lexical = threshold, name, lexical

    def _search(self, query: str, k: int) -> tuple[list[tuple[str, float]], float]:
        """(top-k hits with their scores, the dense top cosine for the filter)."""
        q = self.encoder.encode(query)
        if self.lexical is None:
            hits = self.index.search(q, k)
            return hits, (hits[0][1] if hits else 0.0)
        dense = self.index.search(q, FIRST_STAGE_K)
        lexical = self.lexical.search(query, FIRST_STAGE_K)
        # R1 first, as notebook 17 fused [r1_run, r4_run]: the order breaks RRF ties.
        fused = index_rrf.fuse([{"q": lexical}, {"q": dense}], k=RRF_K, top_n=FIRST_STAGE_K)["q"]
        return fused[:k], (dense[0][1] if dense else 0.0)

    def retrieve(self, query: str, k: int = 5, context_ids: list[str] = (), history: list[dict] = ()) -> dict:
        hits, top = self._search(query, k)
        passed = top >= self.threshold
        scores = dict(hits) if passed else {}
        hit_ids = [pid for pid, _ in hits] if passed else []
        known = self.store.get([c for c in dict.fromkeys(context_ids) if c not in hit_ids])
        hit_ids = (hit_ids + list(known))[:MAX_HITS]

        sources, messages = [], []
        if hit_ids:
            by_id = self.store.with_neighbours(hit_ids)
            for s in generate.build_sources(hit_ids, by_id, generate.NEIGHBOURS):
                first = by_id[s["passage_ids"][0]]
                sources.append({**_passage(first, scores.get(first["passage_id"])), **s})
            study = generate.build_messages(query, sources, self.prompt)
            turns = [m for h in history for m in ({"role": "user", "content": h["q"]},
                                                  {"role": "assistant", "content": strip_citations(h["a"])})]
            messages = [study[0], *turns, study[1]]
        return {"retriever": self.name, "top_score": top, "passed": passed,
                "prompt_sha256": self.prompt["sha256"], "hit_ids": hit_ids,
                "sources": sources, "messages": messages}

    def passages(self, ids: list[str]) -> list[dict]:
        return [_passage(row, None) for row in self.store.get(list(ids)).values()]
