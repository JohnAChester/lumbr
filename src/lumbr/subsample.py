"""The indexed subsample (issue #5): which ~140-150K of the 512K documents every retriever indexes.

  subsample = every gold doc
            + top 50 BM25 look-alikes per question        (bm25_lookalikes, CPU)
            + top 50 R2 (bge-base) look-alikes per question (r2_lookalikes, GPU)
            + 100K random distractors, proportional to each source's share of the corpus

Look-alikes come from BOTH a sparse and a dense retriever so the distractors are not hard for one family only
(see the comment on issue #5 and docs/superpowers/specs/2026-11-16-subsample-design.md). Each step saves its
result to Drive `processed/`, so the steps can run in different Colab sessions; `build_subsample()` assembles
them into the committed `splits/subsample_doc_ids.txt` + `splits/subsample_manifest.json`.

Benchmark questions are used only to choose documents, never for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from lumbr import paths

K = 50
N_RANDOM = 100_000
MIN_PER_SOURCE = 10
SEED = 42

BM25_K1, BM25_B = 0.9, 0.4   # Pyserini/Anserini defaults
HASH_FEATURES = 2 ** 24
R2_MODEL = "BAAI/bge-base-en-v1.5"
R2_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

LOOKALIKES_BM25 = "subsample_lookalikes_bm25.json"
LOOKALIKES_R2 = "subsample_lookalikes_r2.json"
SPLITS = paths.REPO_ROOT / "splits"
IDS_FILE = SPLITS / "subsample_doc_ids.txt"
MANIFEST_FILE = SPLITS / "subsample_manifest.json"


def _topk_desc(scores: np.ndarray, k: int, positive_only: bool = False) -> list[int]:
    """Indices of the k largest scores, best first; ties go to the lower index."""
    idx = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    if len(idx) > k:
        part = np.argpartition(-scores[idx], k - 1)[:k]
        cut = scores[idx][part].min()
        idx = idx[scores[idx] >= cut]   # keep every tie at the cut, then order and trim
    order = np.lexsort((idx, -scores[idx]))
    return [int(i) for i in idx[order][:k]]


# ---------------------------------------------------------------- BM25

def _vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(n_features=HASH_FEATURES, alternate_sign=False, norm=None,
                             lowercase=True, stop_words="english", dtype=np.float32)


@dataclass
class BM25Index:
    weights: object   # scipy CSR, docs x hashed terms, BM25 term weights

    def top_k(self, queries: list[str], k: int = K, batch: int = 50) -> list[list[int]]:
        """Row indices of the k best-scoring docs per query (score > 0 only), best first."""
        q = _vectorizer().transform(queries)
        q.data[:] = 1.0   # each distinct query term counts once
        out = []
        for s in range(0, len(queries), batch):
            scores = (self.weights @ q[s:s + batch].T).toarray()
            out.extend(_topk_desc(scores[:, j], k, positive_only=True) for j in range(scores.shape[1]))
        return out


def bm25_index(texts: Iterable[str], batch: int = 20_000) -> BM25Index:
    """Stream texts into a hashed term-count matrix and turn the counts into BM25 weights."""
    import scipy.sparse as sp

    vec, parts, buf = _vectorizer(), [], []
    for t in texts:
        buf.append(t)
        if len(buf) >= batch:
            parts.append(vec.transform(buf))
            buf = []
    if buf:
        parts.append(vec.transform(buf))
    x = sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0].tocsr()

    n_docs = x.shape[0]
    dl = np.asarray(x.sum(axis=1)).ravel()
    avgdl = dl.mean() if n_docs else 1.0
    df = np.bincount(x.indices, minlength=x.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)   # Lucene's BM25 idf
    rows = np.repeat(np.arange(n_docs), np.diff(x.indptr))
    tf = x.data
    norm = BM25_K1 * (1 - BM25_B + BM25_B * dl[rows] / avgdl)
    x.data = (idf[x.indices] * tf * (BM25_K1 + 1) / (tf + norm)).astype(np.float32)
    return BM25Index(x)


def dedupe(ids: Iterable[str]) -> list[str]:
    """Drop repeats, keep first-seen order (the HF release lists a few doc_ids twice)."""
    return list(dict.fromkeys(ids))


# ---------------------------------------------------------------- corpus I/O

def iter_hf_docs(batch_size: int = 20_000) -> Iterable[tuple[str, str, str]]:
    """Stream (doc_id, source_type, title + newline + content) from the HF release without loading it whole."""
    import pyarrow.parquet as pq

    from lumbr import data

    pf = pq.ParquetFile(data.stage_local()["documents"])
    for b in pf.iter_batches(batch_size=batch_size, columns=["doc_id", "source_type", "title", "content"]):
        c = b.to_pydict()
        for doc_id, src, title, content in zip(c["doc_id"], c["source_type"], c["title"], c["content"]):
            yield doc_id, src, f"{title or ''}\n{content or ''}"


def _questions():
    from lumbr import data

    qs = data.load_questions()
    return list(qs.question_id), list(qs.question)


def _save(name: str, lookalikes: dict[str, list[str]], meta: dict) -> None:
    paths.ensure_dirs()
    tmp = paths.PROCESSED / (name + ".partial")
    tmp.write_text(json.dumps({"meta": meta, "lookalikes": lookalikes}), encoding="utf-8")
    tmp.replace(paths.PROCESSED / name)


def load_lookalikes(name: str) -> dict[str, list[str]]:
    return json.loads((paths.PROCESSED / name).read_text(encoding="utf-8"))["lookalikes"]


def bm25_lookalikes(k: int = K) -> dict[str, list[str]]:
    """Top-k BM25 documents per question over the full corpus. CPU: about 2 min and 7 GB peak RAM in a local test."""
    from tqdm.auto import tqdm

    ids: list[str] = []

    def texts():
        for doc_id, _, text in tqdm(iter_hf_docs(), total=512_000, unit="doc", desc="BM25 index"):
            ids.append(doc_id)
            yield text

    index = bm25_index(texts())
    qids, questions = _questions()
    top = index.top_k(questions, k + 5)   # a little slack for duplicate doc_ids
    out = {q: dedupe(ids[i] for i in rows)[:k] for q, rows in zip(qids, top)}
    _save(LOOKALIKES_BM25, out, {"k": k, "k1": BM25_K1, "b": BM25_B, "n_docs": len(ids),
                                 "hash_features": HASH_FEATURES})
    return out


# ---------------------------------------------------------------- R2

def running_topk(query_emb: np.ndarray, doc_slices: Iterable[tuple[int, np.ndarray]], k: int) -> list[list[int]]:
    """Exact top-k by dot product over docs that arrive in slices `(row_offset, embeddings)`; best first."""
    nq = query_emb.shape[0]
    best_s = np.full((nq, 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, 0), dtype=np.int64)
    for offset, emb in doc_slices:
        s = query_emb @ emb.T
        cand_s = np.concatenate([best_s, s], axis=1)
        cand_i = np.concatenate([best_i, np.broadcast_to(np.arange(offset, offset + emb.shape[0]), s.shape)], axis=1)
        if cand_s.shape[1] > k:
            keep = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
            cand_s = np.take_along_axis(cand_s, keep, axis=1)
            cand_i = np.take_along_axis(cand_i, keep, axis=1)
        best_s, best_i = cand_s, cand_i
    out = []
    for s, i in zip(best_s, best_i):
        order = np.lexsort((i, -s))
        out.append([int(x) for x in i[order]])
    return out


def r2_lookalikes(k: int = K, slice_size: int = 50_000, batch_size: int = 256) -> dict[str, list[str]]:
    """Top-k R2 (bge-base) documents per question, embedding each doc's first 512 tokens. GPU, ~10-30 min."""
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm.auto import tqdm

    model = SentenceTransformer(R2_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = 512
    if torch.cuda.is_available():
        model.half()
    qids, questions = _questions()
    q = model.encode([R2_QUERY_PREFIX + x for x in questions], normalize_embeddings=True,
                     batch_size=batch_size, convert_to_numpy=True).astype(np.float32)
    ids: list[str] = []

    def slices():
        buf: list[str] = []
        bar = tqdm(total=512_000, unit="doc", desc="R2 embed")
        for doc_id, _, text in iter_hf_docs():
            ids.append(doc_id)
            buf.append(text)
            if len(buf) == slice_size:
                yield _embed_slice(model, buf, len(ids) - len(buf), batch_size)
                bar.update(len(buf))
                buf = []
        if buf:
            yield _embed_slice(model, buf, len(ids) - len(buf), batch_size)
            bar.update(len(buf))
        bar.close()

    top = running_topk(q, slices(), k + 5)
    out = {qid: dedupe(ids[i] for i in rows)[:k] for qid, rows in zip(qids, top)}
    _save(LOOKALIKES_R2, out, {"k": k, "model": R2_MODEL, "max_seq_length": 512, "n_docs": len(ids)})
    return out


def _embed_slice(model, texts: list[str], offset: int, batch_size: int) -> tuple[int, np.ndarray]:
    emb = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, convert_to_numpy=True)
    return offset, emb.astype(np.float32)


# ---------------------------------------------------------------- assembly

def assemble(gold: dict[str, list[str]], lookalikes: dict[str, dict[str, list[str]]], index,
             n_random: int = N_RANDOM, min_per_source: int = MIN_PER_SOURCE, seed: int = SEED) -> tuple[list[str], dict]:
    """Gold + look-alikes + a random fill proportional to each source's corpus share.

    `index` is a DataFrame with `doc_id, source_type` for every document #4 can chunk; IDs not in it are
    dropped and listed under `missing_from_corpus`. Returns (sorted doc IDs, manifest).
    """
    idx = index.drop_duplicates("doc_id")
    known = set(idx.doc_id)
    gold_ids = {d for ds in gold.values() for d in ds}
    look_ids = {m: {d for ds in per_q.values() for d in ds} for m, per_q in lookalikes.items()}
    wanted = gold_ids | set().union(*look_ids.values())
    missing = sorted(wanted - known)
    selected = wanted & known

    shares = idx.source_type.value_counts(normalize=True).sort_index()
    pool = idx[~idx.doc_id.isin(selected)].sort_values("doc_id")
    random_ids, random_by_source = [], {}
    for src, share in shares.items():
        cand = pool[pool.source_type == src]
        n = min(len(cand), max(min_per_source, round(share * n_random)))
        random_ids += list(cand.doc_id.sample(n, random_state=seed))
        random_by_source[src] = n

    ids = sorted(selected | set(random_ids))
    src_of = dict(zip(idx.doc_id, idx.source_type))
    by_source: dict[str, int] = {}
    for d in ids:
        by_source[src_of[d]] = by_source.get(src_of[d], 0) + 1
    methods = sorted(lookalikes)
    manifest = {
        "by_part": {"gold": len(gold_ids & known), "lookalikes": len((wanted - gold_ids) & known),
                    "random": len(random_ids), "total": len(ids)},
        "by_source": dict(sorted(by_source.items())),
        "random_by_source": random_by_source,
        "lookalike_overlap": {**{m: len(look_ids[m] & known) for m in methods},
                              "both": len(set.intersection(*(look_ids[m] for m in methods)) & known) if methods else 0},
        "missing_from_corpus": missing,
        "params": {"n_random": n_random, "min_per_source": min_per_source, "seed": seed},
        "per_question": {q: {"gold": list(gold.get(q, [])), **{m: list(lookalikes[m].get(q, [])) for m in methods}}
                         for q in sorted(set(gold) | {q for m in methods for q in lookalikes[m]})},
    }
    return ids, manifest


def build_subsample() -> dict:
    """Assemble from the two look-alike files on Drive and write `splits/` (commit it afterwards)."""
    from lumbr import data, sources
    from lumbr.sample import code_version

    qs = data.load_questions()
    gold = {q: list(ds) for q, ds in zip(qs.question_id, qs.expected_doc_ids)}
    look = {"bm25": load_lookalikes(LOOKALIKES_BM25), "r2": load_lookalikes(LOOKALIKES_R2)}
    ids, manifest = assemble(gold, look, sources.load_index())
    manifest["versions"] = {"hf": f"{data.HF_REPO}@{data.HF_REVISION}",
                            "source_repo": f"{sources.GH_REPO}@{sources.GH_SHA}",
                            "code_version": code_version(), "k": K, "bm25": {"k1": BM25_K1, "b": BM25_B},
                            "r2_model": R2_MODEL}
    SPLITS.mkdir(exist_ok=True)
    IDS_FILE.write_text("\n".join(ids) + "\n", encoding="utf-8")
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_question"}, indent=1)[:3000])
    return manifest
