"""ALCE-style citation check and RAGAS faithfulness (issue #23, Part 4).

Both share the same JudgeCache and resumability logic as Part 1.

ALCE citation check
-------------------
For each sentence in a non-abstained answer that carries citations, ask the
judge whether the cited passages support it.  Reports the share of supported
cited sentences.

Metric added to the per-run results table as `citation_support` (float).

RAGAS faithfulness
------------------
Requires `ragas` to be installed; run only when `ragas=True` is passed.
Uses the LlamaIndex integration so it works with the same Claude client.
Reports faithfulness (0–1) per run; stored in the cache as metric="ragas_faith".

Usage:
  # ALCE
  n = score_citation_support(run_tag, records, by_pid, cache, client)
  support = citation_support_aggregate(run_tag, records, cache)

  # RAGAS (optional)
  faith = ragas_faithfulness(run_tag, records, by_pid, cache, client, ragas=True)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

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


def _sentence_passages(text: str, sources: list[dict], by_pid: dict[str, dict]) -> list[dict]:
    """Sentences with inline citations mapped to passage texts.

    Returns [{"sentence": str, "passage_texts": [str], "n_list": [int]}]
    for every sentence that:
      - has at least one valid citation (n in sources)
      - is at least 20 chars after stripping citations
    """
    by_n = {s["n"]: s for s in sources}
    chunks = _SENTENCE_END.split(text.strip())
    out: list[dict] = []

    for i, chunk in enumerate(chunks):
        cite_ns: list[int] = []
        for m in _CITE.finditer(chunk):
            for n in (int(x) for x in m.group(1).split(",")):
                if n in by_n and n not in cite_ns:
                    cite_ns.append(n)
        # Also check if next chunk starts with bare citations (after-sentence placement)
        if i + 1 < len(chunks):
            nxt = chunks[i + 1].lstrip()
            if _CITE.match(nxt):
                for m in _CITE.finditer(nxt):
                    for n in (int(x) for x in m.group(1).split(",")):
                        if n in by_n and n not in cite_ns:
                            cite_ns.append(n)
        if not cite_ns:
            continue
        clean = _CITE.sub("", chunk).strip()
        if len(clean) < 20:
            continue

        # Collect passage texts for cited source blocks
        passage_texts: list[str] = []
        for n in cite_ns:
            src = by_n[n]
            for pid in src.get("passage_ids", []):
                p = by_pid.get(pid)
                if p:
                    passage_texts.append(p.get("text", p.get("body", "")))

        if passage_texts:
            out.append({"sentence": clean, "passage_texts": passage_texts, "n_list": cite_ns})
    return out


def _alce_messages(sentence: str, passage_texts: Sequence[str]) -> tuple[list[dict], str]:
    passages_block = "\n\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(passage_texts))
    filled = _ALCE_PROMPT.format(sentence=sentence, passages=passages_block)
    return [{"role": "user", "content": filled}], jg._hash(filled)


def score_citation_support(
    run_tag: str,
    records: list[dict],
    by_pid: dict[str, dict],
    cache: jg.JudgeCache,
    client,
    *,
    limit: int | None = None,
    max_calls: int | None = None,
) -> int:
    """Score ALCE citation support for all cited sentences in the run.

    Cache metric = "alce"; fact_idx = per-sentence index within the answer.
    Returns number of new API calls made.
    """
    to_score = records[:limit] if limit else records
    n_calls = 0

    for rec in to_score:
        if rec.get("abstained"):
            continue
        qid     = rec["question_id"]
        sources = rec.get("sources", [])
        sents   = _sentence_passages(rec.get("answer", ""), sources, by_pid)

        for idx, sp in enumerate(sents):
            msgs, phash = _alce_messages(sp["sentence"], sp["passage_texts"])
            if cache.has(run_tag, qid, "alce", idx, phash):
                continue
            try:
                text   = client.call_one(msgs)
                result = text.strip().lower().startswith("yes")
                cache.put(run_tag, qid, "alce", idx, phash, result)
                n_calls += 1
            except jg.OutOfCredits as exc:
                jg._print_stop(run_tag, n_calls, exc)
                raise
            if max_calls and n_calls >= max_calls:
                raise jg.BudgetExhausted(max_calls)

    return n_calls


def citation_support_aggregate(
    run_tag: str,
    records: list[dict],
    by_pid: dict[str, dict],
    cache: jg.JudgeCache,
) -> float | None:
    """Share of cited sentences the judge found supported.  None if none judged."""
    supported = total = 0
    for rec in records:
        if rec.get("abstained"):
            continue
        qid     = rec["question_id"]
        sources = rec.get("sources", [])
        sents   = _sentence_passages(rec.get("answer", ""), sources, by_pid)
        for idx, sp in enumerate(sents):
            _, phash = _alce_messages(sp["sentence"], sp["passage_texts"])
            entry = cache.get(run_tag, qid, "alce", idx, phash)
            if entry is None:
                continue
            total     += 1
            supported += bool(entry["result"])
    return round(supported / total, 4) if total else None


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
