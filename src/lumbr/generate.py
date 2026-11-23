"""Answer generator (issue #14): Lumbr's cited answer, held fixed across all retrieval setups.

Design: docs/superpowers/specs/2026-11-23-answer-generator-design.md

  load_passages(doc_ids, path)        only the passages of the documents the top-k hits touch
  build_sources(hit_ids, by_id)       small-to-big (D16): each hit + chunk_idx ± 1, merged per document,
                                      labelled [1]..[n] in retrieval rank order
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Iterable

from lumbr import chunk, paths, schema

K = 5           # passages per question the generator sees
NEIGHBOURS = 1  # chunks added on each side of a hit (D16)
PROMPT_PATH = Path(__file__).parent / "prompts" / "answer.json"
# A batch of chat message lists in; one {"text", "prompt_tokens", "completion_tokens", "finish_reason"}
# per input out, in order. finish_reason "length" means the answer was cut off at max_tokens.
ChatFn = Callable[[list[list[dict]]], list[dict]]
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# ---------------------------------------------------------------- context

_DOC_ID_MARK = '"doc_id": "'  # how schema.write_jsonl (json.dumps defaults) writes the field


def _line_doc_id(line: str) -> str | None:
    """The doc_id read straight from a JSONL line's text, or None if the line isn't in the usual format."""
    i = line.find(_DOC_ID_MARK)
    if i < 0:
        return None
    start = i + len(_DOC_ID_MARK)
    end = line.find('"', start)
    return line[start:end] if end > start else None


def load_passages(doc_ids: set[str], passages_path: Path | str) -> dict[str, dict]:
    """{passage_id: passage} for every passage of `doc_ids`, in one streaming pass.

    Only lines whose doc_id is needed get JSON-parsed (the doc_id is read from the text first;
    a line in an unexpected format is parsed in full, so it is never skipped by mistake).
    Stops early once every requested document has all of its `n_chunks` passages.
    """
    by_id: dict[str, dict] = {}
    remaining: dict[str, int] = {}
    with open(passages_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            quick = _line_doc_id(line)
            if quick is not None and quick not in doc_ids:
                continue
            p = json.loads(line)
            d = p["doc_id"]
            if d not in doc_ids:
                continue
            by_id[p["passage_id"]] = p
            remaining[d] = remaining.setdefault(d, p["n_chunks"]) - 1
            if len(remaining) == len(doc_ids) and not any(remaining.values()):
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


# ---------------------------------------------------------------- answer parsing

def _normalise(text: str) -> str:
    return " ".join(text.lower().replace(chr(0x2019), "'").replace(chr(0x2018), "'").split())


def parse_answer(text: str, sources: list[dict], prompt: dict) -> dict:
    """Citations mapped back to IDs, out-of-range citation numbers, and whether the answer abstains.

    Accepts [2], [1][3] and [1, 3]. An answer abstains when it starts with the abstention line
    (normalised, final period optional), even if it then explains what the sources do cover and cites
    them. A partial answer gives its answer first, so it does not count as an abstention.
    """
    by_n = {s["n"]: s for s in sources}
    citations: list[dict] = []
    invalid: list[int] = []
    for m in _CITE.finditer(text):
        for n in (int(x) for x in m.group(1).split(",")):
            if n in by_n:
                if all(c["n"] != n for c in citations):
                    citations.append({"n": n, "passage_ids": by_n[n]["passage_ids"],
                                      "doc_id": by_n[n]["doc_id"]})
            elif n not in invalid:
                invalid.append(n)
    line = _normalise(prompt["abstain"]).rstrip(".!")  # a dropped final period still counts
    abstained = _normalise(text).startswith(line)
    return {"citations": citations, "invalid_citations": invalid, "abstained": abstained}


# ---------------------------------------------------------------- generation

def make_vllm_chat(llm, sampling_params) -> ChatFn:
    """ChatFn over a vLLM `LLM` (offline batch chat). Needs vLLM, so it is not unit-tested."""
    def chat(batch: list[list[dict]]) -> list[dict]:
        outs = llm.chat(batch, sampling_params, use_tqdm=True)
        return [{"text": o.outputs[0].text,
                 "prompt_tokens": len(o.prompt_token_ids),
                 "completion_tokens": len(o.outputs[0].token_ids),
                 "finish_reason": o.outputs[0].finish_reason} for o in outs]
    return chat


def top_k_hits(hits: list[tuple[str, float]], k: int = K) -> tuple[list[str], int]:
    """The first k distinct passage IDs of a run's hit list, and how many repeats were skipped on the way.

    Every setup then gets k different passages, even if a fused or reranked run lists one twice;
    the count is kept so such a retriever bug still shows in the run's meta.json.
    """
    seen: list[str] = []
    skipped = 0
    for pid, _ in hits:
        if len(seen) == k:
            break
        if pid in seen:
            skipped += 1
        else:
            seen.append(pid)
    return seen, skipped


def needed_doc_ids(run: dict[str, list[tuple[str, float]]], question_ids: Iterable[str],
                   k: int = K) -> set[str]:
    """Doc IDs of the top-k hits of these questions (what `load_passages` must load)."""
    return {schema.doc_id_of(pid) for q in question_ids for pid in top_k_hits(run.get(q, []), k)[0]}


def generate(chat: ChatFn, questions: list[tuple[str, str]],
             run: dict[str, list[tuple[str, float]]], by_id: dict[str, dict], prompt: dict,
             k: int = K, neighbours: int = NEIGHBOURS) -> tuple[list[dict], float]:
    """Answer every question from its top-k hits in one batched `chat` call.

    Returns (records in question order, wall-clock seconds of the chat call). A question with no
    hits gets the abstention line without a model call.
    """
    prepared, batch = [], []
    for qid, question in questions:
        hit_ids, n_dupes = top_k_hits(run.get(qid, []), k)
        sources = build_sources(hit_ids, by_id, neighbours)
        prepared.append((qid, question, sources, n_dupes))
        if sources:
            batch.append(build_messages(question, sources, prompt))

    start = time.perf_counter()
    results = chat(batch) if batch else []
    wall_s = time.perf_counter() - start
    if len(results) != len(batch):
        raise ValueError(f"chat returned {len(results)} outputs for {len(batch)} prompts")
    outputs = iter(results)

    records = []
    for qid, question, sources, n_dupes in prepared:
        out = next(outputs) if sources else {"text": prompt["abstain"], "prompt_tokens": 0,
                                             "completion_tokens": 0, "finish_reason": None}
        records.append({
            "question_id": qid, "question": question, "answer": out["text"],
            **parse_answer(out["text"], sources, prompt),
            "sources": [{key: s[key] for key in ("n", "doc_id", "passage_ids", "window_ids")}
                        for s in sources],
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "finish_reason": out.get("finish_reason"), "n_duplicate_hits": n_dupes,
        })
    return records, wall_s


def run_tag(retriever: str, prompt: dict) -> str:
    """Run folder tag naming the retriever and the prompt version, e.g. `gen_r1_p2`."""
    return f"gen_{retriever}_p{prompt['version']}"


def answers_dir(runs_dir: Path, tag: str, split: str = "dev", date: str | None = None) -> Path:
    """`{tag}_{split}_{date}` under runs_dir. Raises FileExistsError if it already exists.

    Call it before generating, so a name clash stops the notebook before the GPU run, not after.
    """
    today = date or datetime.date.today().isoformat()
    out = Path(runs_dir) / f"{tag}_{split}_{today}"
    if out.exists():
        raise FileExistsError(f"{out} already exists. Runs are never overwritten: bump the prompt "
                              "version, or delete that folder by hand if it was a bad run.")
    return out


def save_answers(records: list[dict], runs_dir: Path, meta: dict, *, tag: str = "gen_r1",
                 split: str = "dev", date: str | None = None) -> Path:
    """Write `{tag}_{split}_{date}/answers.jsonl` + `meta.json`. Raises FileExistsError, never overwrites."""
    today = date or datetime.date.today().isoformat()
    git_sha = paths.git_commit()  # before anything is written, so a failed save leaves no folder
    out = answers_dir(runs_dir, tag, split, today)
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    full_meta = {**meta, "git_commit": git_sha, "date": today, "n_questions": len(records),
                 "n_abstained": sum(r["abstained"] for r in records),
                 "n_invalid_citations": sum(len(r["invalid_citations"]) for r in records),
                 "n_truncated": sum(r.get("finish_reason") == "length" for r in records),
                 "n_questions_with_duplicate_hits": sum(r.get("n_duplicate_hits", 0) > 0 for r in records)}
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)
    print(f"[gen] {len(records)} answers + meta -> {out}")
    return out


# ---------------------------------------------------------------- comparing rounds

def rescore_records(records: list[dict], prompt: dict) -> list[dict]:
    """Copies of saved answer records with citations and abstention re-parsed by the current rules.

    Lets an earlier run be compared with a new one under the same scoring, without regenerating.
    """
    return [{**r, **parse_answer(r["answer"], r["sources"], prompt)} for r in records]


def summarize(records: list[dict], question_types: dict[str, str],
              gold_docs: dict[str, set[str]]) -> dict:
    """One round's headline numbers, for comparing prompt versions side by side.

    "answerable" means every question type except info_not_found. A gold doc counts as present when
    any of the question's expected doc IDs is among the sources the generator saw.
    """
    def gold_present(r):
        return bool(gold_docs[r["question_id"]] & {s["doc_id"] for s in r["sources"]})

    nf = [r for r in records if question_types[r["question_id"]] == "info_not_found"]
    answerable = [r for r in records if question_types[r["question_id"]] != "info_not_found"]
    return {
        "n": len(records),
        "abstained": sum(r["abstained"] for r in records),
        "info_not_found_abstained": f"{sum(r['abstained'] for r in nf)}/{len(nf)}",
        "answerable_abstained_gold_present": sum(r["abstained"] and gold_present(r) for r in answerable),
        "answerable_answered_gold_missing": sum(not r["abstained"] and not gold_present(r) for r in answerable),
        "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in records),
        "truncated": sum(r.get("finish_reason") == "length" for r in records),
    }
