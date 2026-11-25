"""Synthetic schema-shaped passages for the demo backend tests (no benchmark data in git)."""
from __future__ import annotations

import pytest

from lumbr import schema

DOCS = {
    "dsid_aaa": ("confluence", "retention-policy", "eng-infra", "Ethan Park", [
        "The retention policy keeps audit logs for 400 days in the control plane.",
        "Retention is evaluated nightly by the policy engine.",
        "Exceptions need approval from the compliance team.",
    ]),
    "dsid_bbb": ("slack", "upload-limits", "api-team", "Priya Natarajan", [
        "Multipart uploads default to a 10 MiB per-file limit.",
        "The total request size limit is 50 MiB.",
    ]),
    "dsid_ccc": ("jira", "gpu-quota", None, "Marcus Lin", [
        "GPU quota requests go through the capacity board every Tuesday.",
    ]),
    "dsid_ddd": ("github", "release-notes", None, "Sean Gallagher", [
        "Release 1.4 adds streaming responses.",
        "Release 1.5 fixes idempotency key handling.",
        "Release 1.6 adds batch embeddings.",
        "Release 1.7 removes the legacy router.",
        "Release 1.8 adds tracing headers.",
    ]),
}


def make_passages() -> list[dict]:
    rows = []
    for doc_id, (source, title, channel, author, chunks) in DOCS.items():
        for i, body in enumerate(chunks):
            header = f"[{source}] {channel or '-'} | {title} | {author} | chunk {i}"
            rows.append({
                "passage_id": schema.make_passage_id(doc_id, i), "doc_id": doc_id, "source": source,
                "title": title, "channel": channel, "author": author, "date": "2026-01-30",
                "chunk_idx": i, "n_chunks": len(chunks), "header": header, "body": body,
                "text": f"{header}\n{body}",
            })
    return rows


@pytest.fixture
def passages() -> list[dict]:
    return make_passages()


@pytest.fixture
def store(tmp_path, passages):
    from lumbr_demo.store import PassageStore, build_store
    path = tmp_path / "passages.sqlite"
    build_store(passages, path)
    return PassageStore(path)


THRESHOLD = 0.2
ANSWERABLE = "total request size limit 50 MiB multipart uploads"
JUNK = "zzzz qqqq"


@pytest.fixture
def retriever(store, passages):
    import numpy as np

    from lumbr.generate import load_prompt
    from lumbr_demo.encoder import HashEncoder
    from lumbr_demo.index import DenseIndex
    from lumbr_demo.service import Retriever
    enc = HashEncoder()
    index = DenseIndex(np.stack([enc.encode(p["text"]) for p in passages]), [p["passage_id"] for p in passages])
    return Retriever(enc, index, store, load_prompt(), THRESHOLD, "test")
