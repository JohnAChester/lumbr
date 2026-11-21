"""BM25 retrieval via Pyserini / Lucene (R1, issue #8).

Three steps — run them in order:

  prepare_corpus(passages_path, corpus_dir)   write {"id","contents"} JSONL on local disk
                                              (corpus is a build artefact — never copied to Drive)
  build_index(corpus_dir, ...)                run Pyserini's Lucene indexer on local disk,
                                              then copy the finished index to Drive indexes/r1/
  stage_index(...)                            copy r1 index from Drive to local disk
                                              (run once per session before searching)
  search(queries, ...)                        BM25 top-k per question via batch_search
                                              (auto-stages index from Drive on first call)

The run dict returned by search() is directly compatible with eval.evaluate_run().
Use save_run() to persist results to a dated subfolder of runs/ with a metadata sidecar.

Pyserini requires Java 21+. On Colab:
  !apt-get install -y -qq openjdk-21-jdk-headless
  import os; os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
  # Then: pip install pyserini
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from lumbr import paths, schema

INDEX_TAG        = "r1"
CORPUS_FILE      = "docs.jsonl"  # inside corpus_dir; Pyserini JsonCollection format
K                = 50             # default passages retrieved per question
_INDEX_THREADS   = 4              # Pyserini indexing threads
_SEARCH_THREADS  = 4              # batch_search threads


def index_dir(root: Path = paths.INDEXES) -> Path:
    return root / INDEX_TAG


# ---------------------------------------------------------------- corpus preparation

def prepare_corpus(
    passages_path: Path | str,
    corpus_dir: Path,
) -> int:
    """Convert passages.jsonl to Pyserini {"id","contents"} JSONL on local disk.

    Writes one line per passage:  {"id": passage_id, "contents": text}

    Keep corpus_dir on local disk only — it is a build artefact and does not need to
    survive the session. Returns the total number of passages written. Atomic overwrite.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out = corpus_dir / CORPUS_FILE
    tmp = out.with_suffix(".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for p in schema.read_jsonl(passages_path):
            f.write(json.dumps({"id": p["passage_id"], "contents": p["text"]},
                               ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(out)
    print(f"[r1] corpus: {n:,} passages → {out}")
    return n


# ---------------------------------------------------------------- index building

def _copy_index_to_drive(local_idx: Path, drive_idx: Path) -> None:
    """Copy a Lucene index directory from local disk to Drive atomically.

    Writes to a sibling .tmp directory first, then renames (both on Drive), so a
    partial copy never looks like a finished index.
    """
    drive_idx.parent.mkdir(parents=True, exist_ok=True)
    tmp = drive_idx.parent / (drive_idx.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(local_idx, tmp)
    if drive_idx.exists():
        shutil.rmtree(drive_idx)
    tmp.rename(drive_idx)


def build_index(
    corpus_dir: Path,
    index_root: Path = paths.INDEXES,
    local_root: Path = paths.LOCAL_ROOT,
    threads: int = _INDEX_THREADS,
    force: bool = False,
) -> None:
    """Build Pyserini Lucene index on local disk, then copy to Drive indexes/r1/.

    - Builds into `local_root/r1_build/` for fast I/O (Drive is ~3–5× slower).
    - Copies the finished index to `index_root/r1/` on Drive for persistence.
    - Skips if Drive already has the index (detected by Lucene segment files);
      pass force=True to rebuild.
    - Wall-clock: ~15–20 min for 772 K passages on Colab CPU.
    """
    drive_idx = index_dir(index_root)
    if not force and drive_idx.exists() and any(drive_idx.glob("segments_*")):
        print(f"[r1] index already on Drive at {drive_idx} — pass force=True to rebuild")
        return
    local_build = local_root / "r1_build"
    local_build.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection",  "JsonCollection",
        "--input",       str(corpus_dir),
        "--index",       str(local_build),
        "--generator",   "DefaultLuceneDocumentGenerator",
        "--threads",     str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    # Stream stderr live (Pyserini/Java logs go there) and keep a rolling tail
    # so failures surface a readable message instead of a bare non-zero exit code.
    stderr_tail: list[str] = []
    with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True) as proc:
        assert proc.stderr is not None
        for line in proc.stderr:
            print(line, end="", flush=True)
            stderr_tail.append(line.rstrip())
            if len(stderr_tail) > 30:
                stderr_tail.pop(0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[r1] Pyserini indexer failed (exit {proc.returncode}).\n"
            + "\n".join(stderr_tail)
        )
    print(f"[r1] index built locally → {local_build}")
    _copy_index_to_drive(local_build, drive_idx)
    print(f"[r1] index copied to Drive → {drive_idx}")


# ---------------------------------------------------------------- index staging

def stage_index(
    index_root: Path = paths.INDEXES,
    local_root: Path = paths.LOCAL_ROOT,
) -> Path:
    """Copy the r1 index from Drive to local disk if not already staged.

    Reading a large Lucene index directly from the Drive FUSE mount is slow. Staging
    to the Colab VM's local disk once per session makes batch_search ~3–5× faster.
    Returns the local index path.
    """
    local_idx = local_root / INDEX_TAG
    if local_idx.exists() and any(local_idx.glob("segments_*")):
        print(f"[r1] index already staged at {local_idx}")
        return local_idx
    drive_idx = index_dir(index_root)
    if not drive_idx.exists():
        raise FileNotFoundError(
            f"{drive_idx} not found — run build_index() first."
        )
    print(f"[r1] staging index from Drive → {local_idx} ...")
    if local_idx.exists():
        shutil.rmtree(local_idx)
    shutil.copytree(drive_idx, local_idx)
    print(f"[r1] staged")
    return local_idx


# ---------------------------------------------------------------- search

def search(
    queries: list[tuple[str, str]],
    index_root: Path = paths.INDEXES,
    k: int = K,
    threads: int = _SEARCH_THREADS,
    *,
    _searcher=None,
    _local_root: Path | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """BM25 search over the Lucene index using batch_search.

    Args:
        queries: [(question_id, query_text), ...] — use the raw question field
        index_root: parent of the r1/ index on Drive (used to stage to local disk)
        k: passages to retrieve per question (top-k by BM25 score)
        threads: parallel search threads passed to batch_search
        _searcher: inject a fake LuceneSearcher in tests (skips Java / Drive)
        _local_root: override local staging directory in tests

    Returns:
        {question_id: [(passage_id, score), ...]} sorted descending by score.
        Compatible with eval.evaluate_run() directly.
    """
    if _searcher is None:
        from pyserini.search.lucene import LuceneSearcher
        local_idx = stage_index(index_root, _local_root or paths.LOCAL_ROOT)
        _searcher = LuceneSearcher(str(local_idx))

    if not queries:
        return {}

    qids        = [qid  for qid, _    in queries]
    query_texts = [text for _,   text in queries]
    hits_dict   = _searcher.batch_search(query_texts, qids, k=k, threads=threads)
    return {
        qid: [(hit.docid, float(hit.score)) for hit in hits_dict.get(qid, [])]
        for qid in qids
    }


# ---------------------------------------------------------------- run file I/O

def write_run(run: dict[str, list[tuple[str, float]]], path: Path) -> None:
    """Persist a run dict to JSON for later scoring or inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f)
    print(f"[r1] run saved → {path}")


def read_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load a run written by write_run(); restores (passage_id, score) tuples."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {qid: [tuple(hit) for hit in hits] for qid, hits in raw.items()}


def save_run(
    run: dict[str, list[tuple[str, float]]],
    runs_dir: Path,
    *,
    tag: str = "r1_bm25",
    split: str = "dev",
    k: int = K,
    k1: float = 0.9,
    b: float = 0.4,
    date: str | None = None,
) -> Path:
    """Save run JSON + metadata to a dated subfolder of runs_dir. Never overwrites.

    Subfolder: `{tag}_{split}_{date}` (e.g. `r1_bm25_dev_2026-11-21`).
    Writes two files:
      run.json   — passage-level retrieval results (readable by read_run / evaluate_run)
      meta.json  — retriever, BM25 hyperparams (k1/b are Pyserini defaults 0.9/0.4),
                   pyserini_version, git_commit, split, k, date

    Returns the subdir Path.
    """
    today = date or datetime.date.today().isoformat()
    subdir = runs_dir / f"{tag}_{split}_{today}"
    subdir.mkdir(parents=True, exist_ok=True)

    write_run(run, subdir / "run.json")

    try:
        pyserini_ver = importlib.metadata.version("pyserini")
    except importlib.metadata.PackageNotFoundError:
        pyserini_ver = "unknown"

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    meta = {
        "retriever":        "bm25",
        "tag":              tag,
        "split":            split,
        "k":                k,
        "k1":               k1,
        "b":                b,
        "pyserini_version": pyserini_ver,
        "git_commit":       git_sha,
        "date":             today,
    }
    with open(subdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[r1] run + meta → {subdir}")
    return subdir
