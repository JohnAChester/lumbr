"""The full passage file for the indexed subsample (issue #4).

Reads the subsample doc-ID list (#5), chunks those documents with `chunk.chunk_docs`, and
writes `processed/passages.jsonl` plus a manifest. 139,119 docs / 772,960 passages: about 30 min on a
High-RAM Colab CPU runtime (4+ h estimated on a standard one), so the run is resumable:

- Passages go out in shards of ~SHARD_DOCS documents to Drive `processed/passages_shards/`.
  Each shard is written to local disk first and copied to Drive atomically, so a shard on
  Drive is always complete. Its doc IDs are listed in a sidecar `.ids` file.
- On restart, documents already in a finished shard are skipped and new shards continue
  the numbering. Re-running after a disconnect just picks up where it stopped.
- When every document is done, the shards are concatenated into `passages.jsonl`,
  validated against the schema, and the manifest is written.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Iterable, Iterator

from lumbr import chunk, data, paths, schema, sources

# Committed by #5 (the subsample) so every teammate indexes exactly the same documents.
SUBSAMPLE_IDS = paths.REPO_ROOT / "splits" / "subsample_doc_ids.txt"

PASSAGES = "passages.jsonl"
MANIFEST = "passages_manifest.json"
SHARD_DIR = "passages_shards"
SHARD_DOCS = 5_000


def read_ids(path: str | Path) -> set[str]:
    ids = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()}
    ids.discard("")
    if not ids:
        raise ValueError(f"{path} lists no doc IDs")
    return ids


def _shards(shard_dir: Path) -> list[Path]:
    return sorted(shard_dir.glob("shard_*.jsonl"))


def done_ids(shard_dir: Path) -> set[str]:
    """doc IDs covered by finished shards (a shard counts only once its `.ids` sidecar exists)."""
    out: set[str] = set()
    for s in _shards(shard_dir):
        ids_file = s.with_suffix(".ids")
        if ids_file.exists():
            out |= read_ids(ids_file)
    return out


def _group_by_doc(passages: Iterable[schema.Passage]) -> Iterator[tuple[str, list[schema.Passage]]]:
    """Group a passage stream into (doc_id, passages). chunk_docs keeps a document's passages together."""
    cur, buf = None, []
    for p in passages:
        if p.doc_id != cur and buf:
            yield cur, buf
            buf = []
        cur = p.doc_id
        buf.append(p)
    if buf:
        yield cur, buf


def write_shards(doc_ids: set[str], shard_dir: Path, local_dir: Path,
                 chunk_fn: Callable[[set[str]], Iterable[schema.Passage]],
                 shard_docs: int = SHARD_DOCS, progress: bool = True) -> int:
    """Chunk every doc in `doc_ids` not already in a finished shard. Returns the number of new shards.

    `chunk_fn(ids)` yields passages for those ids (grouped per document); passed in so tests can fake it.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    # A shard copied to Drive whose `.ids` sidecar never got written (crash in between) is
    # unfinished: drop it, its docs are redone below.
    for s in _shards(shard_dir):
        if not s.with_suffix(".ids").exists():
            s.unlink()
    todo = doc_ids - done_ids(shard_dir)
    if not todo:
        return 0
    existing = _shards(shard_dir)
    next_no = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0

    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(doc_ids), initial=len(doc_ids) - len(todo), unit="doc")

    new = 0
    buf: list[schema.Passage] = []
    ids: list[str] = []

    def flush():
        nonlocal new, buf, ids
        name = f"shard_{next_no + new:04d}"
        local = local_dir / f"{name}.jsonl"
        schema.write_jsonl(buf, local)
        data.copy_atomic(local, shard_dir / local.name)
        local.unlink()
        # Sidecar last: it marks the shard as finished.
        (shard_dir / f"{name}.ids").write_text("\n".join(ids) + "\n", encoding="utf-8")
        new += 1
        buf, ids = [], []

    for doc_id, ps in _group_by_doc(chunk_fn(todo)):
        buf.extend(ps)
        ids.append(doc_id)
        if bar:
            bar.update(1)
        if len(ids) >= shard_docs:
            flush()
    if ids:
        flush()
    if bar:
        bar.close()
    return new


def merge_shards(shard_dir: Path, out: Path) -> None:
    """Concatenate the shards in order into one JSONL file."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for s in _shards(shard_dir):
            with open(s, "rb") as src:
                shutil.copyfileobj(src, f)
    tmp.replace(out)


def summarize(path: Path) -> dict:
    by_source: dict[str, int] = {}
    docs_by_source: dict[str, set] = {}
    tokens = []
    for p in schema.read_jsonl(path):
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
        docs_by_source.setdefault(p["source"], set()).add(p["doc_id"])
        tokens.append(p["n_tokens"])
    tokens.sort()
    return {
        "passages_by_source": dict(sorted(by_source.items())),
        "docs_by_source": {k: len(v) for k, v in sorted(docs_by_source.items())},
        "n_tokens": {"p10": tokens[len(tokens) // 10], "median": tokens[len(tokens) // 2],
                     "p90": tokens[int(len(tokens) * 0.9)], "max": tokens[-1],
                     "total": sum(tokens)},
    }


def build_passages(ids_path: str | Path = SUBSAMPLE_IDS, shard_docs: int = SHARD_DOCS) -> dict:
    """Chunk the subsample (resumable) and write `processed/passages.jsonl` + manifest on Drive."""
    from lumbr.sample import code_version

    paths.ensure_dirs()
    ids = read_ids(ids_path)
    known = set(sources.load_index().doc_id)
    unknown = ids - known
    if unknown:
        raise ValueError(f"{len(unknown)} doc IDs in {ids_path} are not in the corpus, e.g. {sorted(unknown)[:3]}")
    dups = sources.duplicate_ids()

    shard_dir = paths.PROCESSED / SHARD_DIR
    write_shards(ids, shard_dir, paths.LOCAL_ROOT / SHARD_DIR,
                 lambda todo: chunk.chunk_docs(sources.iter_docs(doc_ids=todo), dups & todo),
                 shard_docs=shard_docs)

    # Every doc must be covered exactly once across shards.
    covered = [d for s in _shards(shard_dir) for d in read_ids(s.with_suffix(".ids"))]
    if len(covered) != len(set(covered)):
        raise AssertionError("a doc appears in two shards; delete the newer shard and re-run")
    missing = ids - set(covered)
    if missing:  # a doc that produced no passages (empty body) never reaches a shard
        print(f"[warn] {len(missing)} docs produced no passages, e.g. {sorted(missing)[:3]}")

    local = paths.LOCAL_ROOT / PASSAGES
    merge_shards(shard_dir, local)
    stats = schema.validate_jsonl(local)
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        **stats,
        **summarize(local),
        "docs_requested": len(ids),
        "docs_without_passages": sorted(missing),
        "subsample_ids": Path(ids_path).name,
        "source_repo": f"{sources.GH_REPO}@{sources.GH_SHA}",
        "tokenizers": list(schema.TOKENIZERS),
        "max_tokens": schema.MAX_TOKENS,
        "code_version": code_version(),
    }
    data.copy_atomic(local, paths.PROCESSED / PASSAGES)
    (paths.PROCESSED / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "docs_without_passages"}, indent=2))
    return manifest
