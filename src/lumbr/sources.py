"""Structured source documents from the EnterpriseRAG-Bench GitHub repo.

The Hugging Face parquet only has `doc_id, source_type, title, content`. The GitHub repo
has the full structured JSON for every document (author, channel, dates, message lists,
comments, ...). That JSON is our single source of truth for passage text and metadata.

`build_json_parquet()` streams the repo tarball at a pinned commit and writes every
document into ONE parquet file (`raw/erb_json.parquet`). It never extracts the ~512K
small files to disk, because Drive handles many small files badly. Build it once; after
that everyone reads the parquet.

Each JSON document carries:
  dataset_doc_uuid      == the benchmark doc_id (`dsid_...`)
  title_field_name      which key holds the title
  content_field_names   which keys hold the body, IN ORDER (varies within a source!)
"""

from __future__ import annotations

import json
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator

from lumbr import paths
from lumbr.data import copy_atomic

GH_REPO = "onyx-dot-app/EnterpriseRAG-Bench"
# Pinned so all three of us use the exact same documents even if upstream changes.
GH_SHA = "d36685e273713975ee20299bbf1ab64165575b3c"
TARBALL_URL = f"https://codeload.github.com/{GH_REPO}/tar.gz/{GH_SHA}"
SOURCES_PREFIX = "generated_data/sources/"

JSON_PARQUET = "erb_json.parquet"
COLUMNS = ["doc_id", "source_type", "path", "json"]

SOURCE_TYPES = [
    "slack", "gmail", "linear", "jira", "hubspot",
    "google_drive", "fireflies", "github", "confluence",
]


# ---------------------------------------------------------------- build (once)

def iter_tarball_docs(url: str = TARBALL_URL, limit: int | None = None) -> Iterator[tuple[str, str, bytes]]:
    """Stream (source_type, path, raw_json_bytes) for every source JSON in the repo tarball.

    `path` is relative to `generated_data/sources/`, e.g. `slack/1710501234-region-placement-costs.json`.
    """
    n = 0
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            rel = member.name.split("/", 1)[-1]  # drop the "EnterpriseRAG-Bench-<sha>/" folder
            if not rel.startswith(SOURCES_PREFIX):
                continue
            rel = rel[len(SOURCES_PREFIX):]
            f = tar.extractfile(member)
            if f is None:
                continue
            yield rel.split("/", 1)[0], rel, f.read()
            n += 1
            if limit is not None and n >= limit:
                return


def build_json_parquet(force: bool = False, limit: int | None = None, batch_size: int = 20_000) -> Path:
    """Stream the pinned tarball into `paths.RAW/erb_json.parquet` (built locally, then copied to Drive).

    Takes about 10 minutes and needs no GPU. Returns the Drive path. `limit` is for quick tests only.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths.ensure_dirs()
    dst = paths.RAW / JSON_PARQUET
    if dst.exists() and dst.stat().st_size > 0 and not force and limit is None:
        print(f"[ok] {JSON_PARQUET} already on Drive ({dst.stat().st_size / 1e9:.2f} GB)")
        return dst

    tmp = paths.LOCAL_RAW / (JSON_PARQUET + ".partial")
    schema = pa.schema([(c, pa.string()) for c in COLUMNS])
    counts: Counter[str] = Counter()
    skipped: list[str] = []
    rows: dict[str, list] = {c: [] for c in COLUMNS}
    t0 = time.time()

    def flush(writer):
        if rows["doc_id"]:
            writer.write_table(pa.table(rows, schema=schema))
            for c in COLUMNS:
                rows[c].clear()

    print(f"[download] streaming {GH_REPO} @ {GH_SHA[:8]} ...")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for source, rel, raw in iter_tarball_docs(limit=limit):
            try:
                doc = json.loads(raw)
                doc_id = doc["dataset_doc_uuid"]
            except (ValueError, KeyError, TypeError):
                skipped.append(rel)
                continue
            rows["doc_id"].append(doc_id)
            rows["source_type"].append(source)
            rows["path"].append(rel)
            rows["json"].append(raw.decode("utf-8"))
            counts[source] += 1
            if len(rows["doc_id"]) >= batch_size:
                flush(writer)
                total = sum(counts.values())
                print(f"  {total:>7,} docs  ({time.time() - t0:.0f}s)")
        flush(writer)

    print(f"[ok] {sum(counts.values()):,} docs in {time.time() - t0:.0f}s; skipped {len(skipped)} non-document JSON files")
    for s in SOURCE_TYPES:
        print(f"     {s:<13} {counts.get(s, 0):>7,}")
    if skipped:
        print("     skipped examples:", skipped[:5])

    if limit is not None:
        return tmp  # test run: leave it local, never overwrite the real file on Drive
    local = paths.LOCAL_RAW / JSON_PARQUET
    tmp.replace(local)
    print(f"  copying to Drive -> {dst}")
    copy_atomic(local, dst)
    return dst


# ---------------------------------------------------------------- read

def stage_local(force: bool = False) -> Path:
    """Copy `erb_json.parquet` from Drive to local disk for fast reads this session."""
    src = paths.RAW / JSON_PARQUET
    dst = paths.LOCAL_RAW / JSON_PARQUET
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run sources.build_json_parquet() first.")
    if force or not dst.exists() or dst.stat().st_size != src.stat().st_size:
        print(f"[stage] staging {JSON_PARQUET} to local disk ...")
        copy_atomic(src, dst)
    return dst


def load_index():
    """Every document's `doc_id, source_type, path` (no JSON bodies) as a DataFrame. Small and fast."""
    import pandas as pd

    return pd.read_parquet(stage_local(), columns=["doc_id", "source_type", "path"])


def duplicate_ids() -> set[str]:
    """doc_ids shared by two different files (4 in the corpus; identical in the HF release).

    Chunking treats both files as ONE document, so either counts as retrieving that gold ID.
    """
    idx = load_index()
    return set(idx.loc[idx.doc_id.duplicated(), "doc_id"])


def iter_docs(source_type: str | None = None, doc_ids: set[str] | None = None,
              batch_size: int = 10_000) -> Iterator[dict]:
    """Yield parsed JSON documents, optionally filtered by source and/or a set of doc_ids.

    Streams in batches, so memory stays small even over all 512K documents.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(stage_local())
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
        cols = batch.to_pydict()
        for doc_id, src, raw in zip(cols["doc_id"], cols["source_type"], cols["json"]):
            if source_type is not None and src != source_type:
                continue
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            doc = json.loads(raw)
            doc["_source_type"] = src
            yield doc


def get_docs(doc_ids) -> dict[str, dict]:
    """Fetch specific documents by doc_id. One pass over the file, so batch your lookups."""
    wanted = set(doc_ids)
    return {d["dataset_doc_uuid"]: d for d in iter_docs(doc_ids=wanted)}


# ---------------------------------------------------------------- helpers used by chunking

def title_of(doc: dict) -> str:
    """The document's title, using its own `title_field_name`."""
    v = doc.get(doc.get("title_field_name") or "title")
    return str(v).strip() if v is not None else ""


def content_fields(doc: dict) -> list[tuple[str, object]]:
    """(field_name, value) for each body field, in the document's own order. Missing fields are skipped."""
    return [(f, doc[f]) for f in doc.get("content_field_names", []) if f in doc and doc[f] not in (None, "", [])]
