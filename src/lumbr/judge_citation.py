"""ALCE-style citation check and RAGAS faithfulness (issue #23, Part 4).

Both share the same JudgeCache and resumability logic as Part 1.

ALCE citation check
-------------------
For each cited sentence of a non-abstained answer, ask the judge whether the sources it cites,
together, support it: ALCE's citation recall (Gao et al. 2023). The judge sees each cited source
exactly as the generator did (header plus the whole small-to-big window, generate.build_sources),
not only the hit passages, so a sentence backed by a neighbouring passage isn't marked unsupported.

Budget path (#23): one seeded sample of answered questions, the same for every run checked, sent
as a Message Batch; replies without a clear yes/no are re-asked with the reply forced to JSON.

  qids    = alce_sample(records_by_run, ["gen_r1_r4_p7", "gen_r5_r6_p7"], n=100)
  pending = alce_pending(tag, records, by_pid, cache, qids)      # alce_cost(pending) first
  submit_alce_batch(tag, records, by_pid, cache, client, batch_path, qids)
  jg.collect_batches(cache, client, batch_path, batch_ids=alce_batch_ids(batch_path))
  rejudge_alce(unusable, records_by_run, by_pid, cache, client)
  citation_support_counts(tag, records, by_pid, cache, qids)

RAGAS faithfulness
------------------
Requires `ragas` to be installed; run only when `ragas=True` is passed. Not used for #23.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Sequence

from lumbr import chunk
from lumbr import judge as jg

# Same sentence-end split as generate.py (not imported to avoid circular dep)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITE         = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")

# Prompt for ALCE-style citation support (adapted from ALCE: Gao et al. 2023)
_ALCE_PROMPT = """
You are a citation verifier. Given a sentence and one or more source passages, determine whether the passages together support the claim made in the sentence.

A passage supports the sentence if its content is consistent with and provides evidence for the sentence's claim. The passages do not need to use the same words, but the factual claim must be grounded in them.

## Sentence
{sentence}

## Source Passages
{passages}

CRITICAL: Output only "yes" if the passages support the sentence, or "no" if they do not.
""".strip()

# Batch prices for the judge model, $ per million tokens (Sonnet 5.5 at the 50% batch discount).
BATCH_PRICE_IN, BATCH_PRICE_OUT = 1.0, 5.0


def _source_text(src: dict, by_pid: dict[str, dict]) -> str | None:
    """A source block's text as the generator showed it: header, then the window's bodies merged
    without the context each chunk carries over from the one before (generate.build_sources).

    Falls back to the hit passages for records without `window_ids`. Missing passages are
    skipped; None if none of them are loaded.
    """
    window = [by_pid[p] for p in (src.get("window_ids") or src.get("passage_ids", []))
              if p in by_pid]
    if not window:
        return None
    parts = [window[0]["body"]]
    parts += [chunk.drop_carry(a["body"], b["body"]) for a, b in zip(window, window[1:])]
    return window[0]["header"] + "\n" + "\n".join(parts)


def _sentence_passages(text: str, sources: list[dict], by_pid: dict[str, dict]) -> list[dict]:
    """Sentences with inline citations mapped to the cited sources' texts.

    Returns [{"sentence": str, "passage_texts": [str], "n_list": [int]}] (one text per cited
    source) for every sentence that:
      - has at least one valid citation (n in sources)
      - is at least 20 chars after stripping citations
    """
    by_n = {s["n"]: s for s in sources}
    pieces = _SENTENCE_END.split(text.strip())
    out: list[dict] = []

    for i, piece in enumerate(pieces):
        cite_ns: list[int] = []
        for m in _CITE.finditer(piece):
            for n in (int(x) for x in m.group(1).split(",")):
                if n in by_n and n not in cite_ns:
                    cite_ns.append(n)
        # Also check if next piece starts with bare citations (after-sentence placement)
        if i + 1 < len(pieces):
            nxt = pieces[i + 1].lstrip()
            if _CITE.match(nxt):
                for m in _CITE.finditer(nxt):
                    for n in (int(x) for x in m.group(1).split(",")):
                        if n in by_n and n not in cite_ns:
                            cite_ns.append(n)
        if not cite_ns:
            continue
        clean = _CITE.sub("", piece).strip()
        if len(clean) < 20:
            continue

        passage_texts = [t for n in cite_ns if (t := _source_text(by_n[n], by_pid))]
        if passage_texts:
            out.append({"sentence": clean, "passage_texts": passage_texts, "n_list": cite_ns})
    return out


def _alce_messages(sentence: str, passage_texts: Sequence[str]) -> tuple[list[dict], str]:
    passages_block = "\n\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(passage_texts))
    filled = _ALCE_PROMPT.format(sentence=sentence, passages=passages_block)
    return [{"role": "user", "content": filled}], jg._hash(filled)


def _checks(rec: dict, by_pid: dict[str, dict]) -> list[tuple[int, list[dict], str]]:
    """(sentence index, messages, prompt hash) for each cited sentence; none if abstained."""
    if rec.get("abstained"):
        return []
    sents = _sentence_passages(rec.get("answer", ""), rec.get("sources", []), by_pid)
    return [(idx, *_alce_messages(sp["sentence"], sp["passage_texts"]))
            for idx, sp in enumerate(sents)]


def _in_sample(records: list[dict], qids: set[str] | None) -> list[dict]:
    return [r for r in records if qids is None or r["question_id"] in qids]


# ---------------------------------------------------------------- sample, cost, batch

def alce_sample(records_by_run: dict[str, list[dict]], run_tags: Sequence[str],
                n: int = 100, seed: int = 23) -> set[str]:
    """n question ids answered (not abstained) by every run in run_tags, drawn with a fixed seed.

    One sample for all runs, so their citation support is compared on the same questions.
    """
    answered: set[str] | None = None
    for tag in run_tags:
        ids = {r["question_id"] for r in records_by_run[tag] if not r.get("abstained")}
        answered = ids if answered is None else answered & ids
    pool = sorted(answered or ())
    return set(random.Random(seed).sample(pool, min(n, len(pool))))


def alce_pending(run_tag: str, records: list[dict], by_pid: dict[str, dict],
                 cache: jg.JudgeCache, qids: set[str] | None = None,
                 skip_ids: set[str] = frozenset()) -> list[dict]:
    """Citation checks not yet in the cache, as batch requests (with their manifest fields)."""
    out = []
    for rec in _in_sample(records, qids):
        qid = rec["question_id"]
        for idx, msgs, h in _checks(rec, by_pid):
            cid = jg._make_custom_id(run_tag, qid, "alce", idx)
            if cache.has(run_tag, qid, "alce", idx, h) or cid in skip_ids:
                continue
            out.append({"custom_id": cid, "messages": msgs, "metric": "alce",
                        "fact_idx": idx, "prompt_hash": h, "qid": qid})
    return out


def alce_cost(pending: list[dict]) -> dict:
    """Batch cost estimate, deliberately on the high side: ~3.5 characters per token in plus
    per-request overhead, ~10 tokens out (a yes/no)."""
    chars = sum(len(m["content"]) for p in pending for m in p["messages"])
    tok_in = int(chars / 3.5) + 30 * len(pending)
    tok_out = 10 * len(pending)
    usd = tok_in / 1e6 * BATCH_PRICE_IN + tok_out / 1e6 * BATCH_PRICE_OUT
    return {"n_requests": len(pending), "tokens_in": tok_in, "usd": round(usd, 2)}


def submit_alce_batch(run_tag: str, records: list[dict], by_pid: dict[str, dict],
                      cache: jg.JudgeCache, client, batch_path: Path,
                      qids: set[str] | None = None) -> str | None:
    """Submit this run's pending citation checks as one Message Batch; None if nothing pending.

    Saves the manifest (kind "alce") like judge.submit_batch_run, and skips requests still
    processing in an earlier batch, so re-running never pays twice.
    """
    in_flight = jg.in_flight_custom_ids(client, batch_path, run_tag)
    pending = alce_pending(run_tag, records, by_pid, cache, qids, skip_ids=in_flight)
    if not pending:
        return None
    batch_id = client.submit_batch(pending)
    record = {
        "batch_id": batch_id, "run": run_tag, "kind": "alce",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requests": [{"custom_id": p["custom_id"], "run": run_tag, "qid": p["qid"],
                      "metric": "alce", "fact_idx": p["fact_idx"],
                      "prompt_hash": p["prompt_hash"]} for p in pending],
    }
    batch_path = Path(batch_path)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[alce] Batch {batch_id}: {len(pending)} requests submitted for {run_tag}.")
    return batch_id


def alce_batch_ids(batch_path: Path) -> set[str]:
    """The ids of the saved citation-check batches (to collect only those)."""
    return {b["batch_id"] for b in jg._read_manifest(batch_path) if b.get("kind") == "alce"}


def rejudge_alce(reqs: list[dict], records_by_run: dict[str, list[dict]],
                 by_pid: dict[str, dict], cache: jg.JudgeCache,
                 client) -> tuple[int, list[dict]]:
    """Re-ask citation checks the batch couldn't settle, with the reply forced to yes/no JSON.

    Like judge.rejudge_direct: rebuilds each prompt and checks its hash, so a changed answer or
    passage is never scored under the old key. Returns (n_settled, still_unusable).
    """
    by_qid = {run: {r["question_id"]: r for r in recs} for run, recs in records_by_run.items()}
    n_done, still = 0, []
    for req in reqs:
        if req["metric"] != "alce":
            continue
        run, qid, idx, h = req["run"], req["qid"], req["fact_idx"], req["prompt_hash"]
        if cache.has(run, qid, "alce", idx, h):
            continue
        rec = by_qid.get(run, {}).get(qid)
        found = [c for c in (_checks(rec, by_pid) if rec else []) if c[0] == idx and c[2] == h]
        if not found:
            still.append(req)
            continue
        verdict = client.call_strict(found[0][1], "alce")
        if verdict is None:
            still.append(req)
            continue
        cache.put(run, qid, "alce", idx, h, verdict, model=jg.STRICT_MODEL_TAG)
        n_done += 1
    return n_done, still


# ---------------------------------------------------------------- direct path and results

def score_citation_support(
    run_tag: str,
    records: list[dict],
    by_pid: dict[str, dict],
    cache: jg.JudgeCache,
    client,
    *,
    limit: int | None = None,
    max_calls: int | None = None,
    qids: set[str] | None = None,
) -> int:
    """Score citation support with direct calls (small tests; the batch path costs half).

    Cache metric = "alce"; fact_idx = per-sentence index within the answer. A reply without a
    clear yes/no is not cached (it is retried next time), never counted as "no".
    Returns number of new API calls made.
    """
    to_score = _in_sample(records, qids)
    to_score = to_score[:limit] if limit else to_score
    n_calls = 0

    for rec in to_score:
        qid = rec["question_id"]
        for idx, msgs, phash in _checks(rec, by_pid):
            if cache.has(run_tag, qid, "alce", idx, phash):
                continue
            try:
                result = jg.parse_fact_response(client.call_one(msgs) or "")
            except jg.OutOfCredits as exc:
                jg._print_stop(run_tag, n_calls, exc)
                raise
            n_calls += 1
            if result is not None:
                cache.put(run_tag, qid, "alce", idx, phash, result)
            if max_calls and n_calls >= max_calls:
                raise jg.BudgetExhausted(max_calls)

    return n_calls


def citation_support_counts(run_tag: str, records: list[dict], by_pid: dict[str, dict],
                            cache: jg.JudgeCache, qids: set[str] | None = None) -> dict:
    """{supported, judged, checks, answers}: cited sentences supported / judged so far / to
    judge, and the (non-abstained) answers in the sample they come from."""
    supported = judged = checks = answers = 0
    for rec in _in_sample(records, qids):
        answers += not rec.get("abstained")
        for idx, _, h in _checks(rec, by_pid):
            checks += 1
            entry = cache.get(run_tag, rec["question_id"], "alce", idx, h)
            if entry is not None:
                judged += 1
                supported += bool(entry["result"])
    return {"supported": supported, "judged": judged, "checks": checks, "answers": answers}


def citation_support_aggregate(
    run_tag: str,
    records: list[dict],
    by_pid: dict[str, dict],
    cache: jg.JudgeCache,
    qids: set[str] | None = None,
) -> float | None:
    """Share of cited sentences the judge found supported.  None if none judged."""
    c = citation_support_counts(run_tag, records, by_pid, cache, qids)
    return round(c["supported"] / c["judged"], 4) if c["judged"] else None


# ---------------------------------------------------------------- RAGAS faithfulness (optional)

def ragas_faithfulness(
    run_tag: str,
    records: list[dict],
    by_pid: dict[str, dict],
    cache: jg.JudgeCache,
    client,
    *,
    ragas: bool = False,
    limit: int | None = None,
) -> float | None:
    """RAGAS faithfulness score for the run.

    Skipped (returns None) unless ragas=True and the `ragas` package is installed.
    Stores result in cache as metric="ragas_faith", fact_idx=-1.
    Uses the per-answer approach: faithfulness = mean over non-abstained answers.
    """
    if not ragas:
        return None

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness as ragas_faith_metric
        from datasets import Dataset
    except ImportError:
        print("[judge] ragas or datasets not installed; skipping faithfulness.")
        return None

    to_score = records[:limit] if limit else records
    scores: list[float] = []

    for rec in to_score:
        if rec.get("abstained"):
            continue
        qid = rec["question_id"]
        # Check cache first
        _, phash = jg.fact_messages(rec.get("answer", ""), "ragas_faith_sentinel")
        entry = cache.get(run_tag, qid, "ragas_faith", -1, phash)
        if entry is not None:
            scores.append(float(entry["result"]))
            continue

        # Build contexts from cited passages
        sources = rec.get("sources", [])
        contexts: list[str] = []
        for src in sources:
            for pid in src.get("window_ids", src.get("passage_ids", [])):
                p = by_pid.get(pid)
                if p:
                    contexts.append(p.get("text", p.get("body", "")))

        if not contexts:
            continue

        ds = Dataset.from_dict({
            "question":  [rec.get("question", "")],
            "answer":    [jg.strip_citations(rec.get("answer", ""))],
            "contexts":  [contexts],
        })
        try:
            result = evaluate(ds, metrics=[ragas_faith_metric])
            faith  = float(result["faithfulness"])
            cache.put(run_tag, qid, "ragas_faith", -1, phash, faith)
            scores.append(faith)
        except Exception as exc:
            print(f"[judge] RAGAS error on {qid}: {exc}")

    return round(sum(scores) / len(scores), 4) if scores else None
