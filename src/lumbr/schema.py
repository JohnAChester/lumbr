"""The passage schema: one JSON object per line in `processed/*.jsonl`.

This is the contract between chunking (John), retrieval/eval (Patricia), and
synthetic questions/fine-tuning (Jieun). See docs/SCHEMA.md for the human-readable version.
Change it only through a PR that both teammates approve.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1

# n_tokens = max over these tokenizers of the count INCLUDING special tokens (value = specials added).
# R2/R4 (bge-base, BERT) has a hard 512-position limit; R3 (Qwen3) splits digits, so it is the
# larger count for ~1% of passages. Capping both at 512 means no retriever ever truncates a passage.
TOKENIZERS = {
    "BAAI/bge-base-en-v1.5": 2,     # [CLS] ... [SEP]
    "Qwen/Qwen3-Embedding-4B": 1,   # ... <|endoftext|>
}
MAX_TOKENS = 512

SOURCES = (
    "slack", "gmail", "linear", "jira", "hubspot",
    "google_drive", "fireflies", "github", "confluence",
)
HEADER_SEP = " · "


@dataclass
class Passage:
    passage_id: str          # "{doc_id}#{chunk_idx:03d}"
    doc_id: str              # "dsid_<32 hex>", matches benchmark gold IDs
    source: str              # one of SOURCES
    title: str               # document title
    channel: str | None      # where it lives: Slack channel, Jira project, Confluence space, ...
    author: str | None       # who wrote it (message sender for Gmail/Slack chunks)
    date: str | None         # ISO-8601, most recent activity this chunk reflects
    created: str | None      # ISO-8601, document creation
    chunk_idx: int           # 0-based position within the document
    n_chunks: int            # total chunks for this document
    n_tokens: int            # max over TOKENIZERS of `text`'s count incl. special tokens; <= MAX_TOKENS
    header: str              # one-line metadata header
    body: str                # chunk content WITHOUT the header
    text: str                # header + "\n\n" + body  <- the field everyone indexes

    def to_dict(self) -> dict:
        return asdict(self)


FIELD_NAMES = [f.name for f in fields(Passage)]
_OPTIONAL = {"channel", "author", "date", "created"}
_INTS = {"chunk_idx", "n_chunks", "n_tokens"}


# ---------------------------------------------------------------- ids and text

def make_passage_id(doc_id: str, chunk_idx: int) -> str:
    return f"{doc_id}#{chunk_idx:03d}"


def doc_id_of(passage_id: str) -> str:
    """Map a passage back to its document (used for doc-level scoring)."""
    return passage_id.split("#", 1)[0]


def make_header(source: str, title: str, channel: str | None = None,
                author: str | None = None, date: str | None = None) -> str:
    """e.g. `[slack] #eng-infra · Alex · 2024-03-15` or `[gmail] Re: Beta plan · Vivek Kulkarni · 2025-05-15`.

    Missing parts are left out, never guessed. The date shows the day only; the full value is in `date`.
    """
    loc = (f"#{channel}" if source == "slack" else channel) if channel else None
    parts = [loc, title if title and title != channel else None, author, date[:10] if date else None]
    return (f"[{source}] " + HEADER_SEP.join(p for p in parts if p)).rstrip()


def join_text(header: str, body: str) -> str:
    return f"{header}\n\n{body}"


# ---------------------------------------------------------------- io + validation

def write_jsonl(passages: Iterable[Passage | dict], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in passages:
            d = p.to_dict() if isinstance(p, Passage) else p
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def validate_record(d: dict, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Return a list of problems with one passage record (empty list = valid)."""
    errs = []
    missing = [k for k in FIELD_NAMES if k not in d]
    extra = [k for k in d if k not in FIELD_NAMES]
    if missing:
        errs.append(f"missing fields {missing}")
    if extra:
        errs.append(f"unknown fields {extra}")
    if missing:
        return errs
    for k in FIELD_NAMES:
        v = d[k]
        if k in _INTS:
            if not isinstance(v, int):
                errs.append(f"{k} must be int")
        elif k in _OPTIONAL:
            if v is not None and not isinstance(v, str):
                errs.append(f"{k} must be str or null")
        elif not isinstance(v, str):
            errs.append(f"{k} must be str")
    if errs:
        return errs
    if d["source"] not in SOURCES:
        errs.append(f"unknown source {d['source']!r}")
    if not d["doc_id"].startswith("dsid_"):
        errs.append("doc_id must start with dsid_")
    if d["passage_id"] != make_passage_id(d["doc_id"], d["chunk_idx"]):
        errs.append("passage_id must equal f'{doc_id}#{chunk_idx:03d}'")
    if not 0 <= d["chunk_idx"] < d["n_chunks"]:
        errs.append("chunk_idx out of range")
    if d["text"] != join_text(d["header"], d["body"]):
        errs.append("text must equal header + '\\n\\n' + body")
    if not d["body"].strip():
        errs.append("empty body")
    if not 0 < d["n_tokens"] <= max_tokens:
        errs.append(f"n_tokens {d['n_tokens']} outside (0, {max_tokens}]")
    return errs


def validate_jsonl(path: str | Path, max_tokens: int = MAX_TOKENS, max_errors: int = 20) -> dict:
    """Validate a passage file. Raises ValueError listing the first problems; returns summary stats."""
    seen: set[str] = set()
    problems: list[str] = []
    n = 0
    for i, d in enumerate(read_jsonl(path)):
        n += 1
        for e in validate_record(d, max_tokens):
            problems.append(f"line {i + 1}: {e}")
        pid = d.get("passage_id")
        if pid in seen:
            problems.append(f"line {i + 1}: duplicate passage_id {pid}")
        seen.add(pid)
        if len(problems) >= max_errors:
            break
    if problems:
        raise ValueError(f"{path}: {len(problems)}+ problems\n  " + "\n  ".join(problems))
    docs = {doc_id_of(p) for p in seen}
    return {"passages": n, "docs": len(docs), "schema_version": SCHEMA_VERSION}
