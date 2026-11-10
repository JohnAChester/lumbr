"""The v0 passage sample for teammates to build against (issue #3).

~1K documents = every gold document of 50 random questions + a random fill stratified by
source (at least 10 per source). Gold docs are included so Patricia can run BM25 + the eval
script end to end on a tiny index, and Jieun can prototype question generation.
"""

from __future__ import annotations

import json
import subprocess

from lumbr import chunk, data, paths, schema, sources

SAMPLE_PASSAGES = "sample_passages.jsonl"
SAMPLE_DOC_IDS = "sample_doc_ids.txt"
SAMPLE_MANIFEST = "sample_manifest.json"


def code_version() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def choose_doc_ids(n_docs: int = 1000, n_questions: int = 50, min_per_source: int = 10,
                   seed: int = 42) -> tuple[set[str], list[str]]:
    """Returns (doc_ids, question_ids used for gold docs)."""
    qs = data.load_questions()
    with_gold = qs[qs.expected_doc_ids.map(len) > 0]
    chosen = with_gold.sample(n_questions, random_state=seed)
    gold = {d for ids in chosen.expected_doc_ids for d in ids}

    idx = sources.load_index().drop_duplicates("doc_id")
    pool = idx[~idx.doc_id.isin(gold)]
    shares = idx.source_type.value_counts(normalize=True)
    fill = n_docs - len(gold)
    picked = []
    for src, share in shares.items():
        k = max(min_per_source, round(share * fill))
        picked.append(pool[pool.source_type == src].sample(k, random_state=seed))
    rand_ids = {d for df in picked for d in df.doc_id}
    return gold | rand_ids, sorted(chosen.question_id)


def build_sample(n_docs: int = 1000, n_questions: int = 50, seed: int = 42) -> dict:
    """Chunk the sample and write passages, doc IDs and a manifest to Drive `processed/`."""
    paths.ensure_dirs()
    ids, qids = choose_doc_ids(n_docs, n_questions, seed=seed)
    dups = sources.duplicate_ids() & ids

    local = paths.LOCAL_ROOT / SAMPLE_PASSAGES
    n = schema.write_jsonl(chunk.chunk_docs(sources.iter_docs(doc_ids=ids), dups), local)
    stats = schema.validate_jsonl(local)

    by_source: dict[str, int] = {}
    tokens = []
    for p in schema.read_jsonl(local):
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
        tokens.append(p["n_tokens"])
    tokens.sort()
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "passages": n,
        "docs": stats["docs"],
        "passages_by_source": dict(sorted(by_source.items())),
        "n_tokens": {"p10": tokens[len(tokens) // 10], "median": tokens[len(tokens) // 2],
                     "p90": tokens[int(len(tokens) * 0.9)], "max": tokens[-1]},
        "gold_question_ids": qids,
        "seed": seed,
        "source_repo": f"{sources.GH_REPO}@{sources.GH_SHA}",
        "tokenizers": list(schema.TOKENIZERS),
        "max_tokens": schema.MAX_TOKENS,
        "code_version": code_version(),
    }
    data.copy_atomic(local, paths.PROCESSED / SAMPLE_PASSAGES)
    (paths.PROCESSED / SAMPLE_DOC_IDS).write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    (paths.PROCESSED / SAMPLE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "gold_question_ids"}, indent=2))
    return manifest
