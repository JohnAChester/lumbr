"""Answer generator (issue #14): Lumbr's cited answer, held fixed across all retrieval setups.

Design: docs/superpowers/specs/2026-11-23-answer-generator-design.md

  load_passages(doc_ids, path)        only the passages of the documents the top-k hits touch
  build_sources(hit_ids, by_id)       small-to-big (D16): each hit + chunk_idx ± 1, merged per document,
                                      labelled [1]..[n] in retrieval rank order
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lumbr import chunk, schema

K = 5           # passages per question the generator sees
NEIGHBOURS = 1  # chunks added on each side of a hit (D16)
PROMPT_PATH = Path(__file__).parent / "prompts" / "answer.json"


# ---------------------------------------------------------------- context

def load_passages(doc_ids: set[str], passages_path: Path | str) -> dict[str, dict]:
    """{passage_id: passage} for every passage of `doc_ids`, in one streaming pass.

    Stops early once every requested document has all of its `n_chunks` passages.
    """
    by_id: dict[str, dict] = {}
    remaining: dict[str, int] = {}
    for p in schema.read_jsonl(passages_path):
        d = p["doc_id"]
        if d not in doc_ids:
            continue
        by_id[p["passage_id"]] = p
        left = remaining.setdefault(d, p["n_chunks"]) - 1
        remaining[d] = left
        if left == 0 and len(remaining) == len(doc_ids) and not any(remaining.values()):
            break
    return by_id


def build_sources(hit_ids: list[str], by_id: dict[str, dict],
                  neighbours: int = NEIGHBOURS) -> list[dict]:
    """The generator's sources: one block per merged window, in rank order of its best hit.

    Each block: {"n", "doc_id", "passage_ids" (its hits, rank order), "window_ids" (document order),
    "header" (first window passage's), "body" (window bodies joined without context carry)}.
    A missing hit raises KeyError; a missing neighbour is skipped.
    """
    # windows[doc_id] = list of [lo, hi, best_rank, hit_ids]
    windows: dict[str, list[list]] = {}
    for rank, pid in enumerate(hit_ids):
        hit = by_id[pid]
        lo = max(0, hit["chunk_idx"] - neighbours)
        hi = min(hit["n_chunks"] - 1, hit["chunk_idx"] + neighbours)
        windows.setdefault(hit["doc_id"], []).append([lo, hi, rank, [pid]])

    blocks = []
    for doc_id, wins in windows.items():
        wins.sort(key=lambda w: w[0])
        merged = [wins[0]]
        for lo, hi, rank, pids in wins[1:]:
            cur = merged[-1]
            if lo <= cur[1] + 1:
                cur[1] = max(cur[1], hi)
                cur[2] = min(cur[2], rank)
                cur[3] = cur[3] + pids
            else:
                merged.append([lo, hi, rank, pids])
        for lo, hi, rank, pids in merged:
            window = [by_id[q] for i in range(lo, hi + 1)
                      if (q := schema.make_passage_id(doc_id, i)) in by_id]
            blocks.append((rank, doc_id, sorted(pids, key=hit_ids.index), window))

    blocks.sort(key=lambda b: b[0])
    sources = []
    for n, (_, doc_id, pids, window) in enumerate(blocks, start=1):
        parts = [window[0]["body"]]
        parts += [chunk.drop_carry(a["body"], b["body"]) for a, b in zip(window, window[1:])]
        sources.append({
            "n": n, "doc_id": doc_id, "passage_ids": pids,
            "window_ids": [p["passage_id"] for p in window],
            "header": window[0]["header"], "body": "\n".join(parts),
        })
    return sources


# ---------------------------------------------------------------- prompt

def load_prompt(path: Path | str = PROMPT_PATH) -> dict:
    """The prompt file's keys plus `sha256`, the fingerprint recorded in every run's meta.json.

    The hash is taken with line endings normalised to LF, so a Windows (CRLF) checkout and a
    Colab (LF) checkout of the same prompt get the same hash.
    """
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    prompt = json.loads(raw.decode("utf-8"))
    prompt["sha256"] = hashlib.sha256(raw).hexdigest()
    return prompt


def build_messages(question: str, sources: list[dict], prompt: dict) -> list[dict]:
    """[system, user] chat messages: the numbered sources, then the question."""
    blocks = "\n\n".join(prompt["source_template"].format(**s) for s in sources)
    user = prompt["user_template"].format(sources=blocks, question=question)
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user}]
