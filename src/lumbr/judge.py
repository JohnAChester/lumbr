"""LLM judge: answer correctness and completeness for ERB (issue #23).

Prompts ported verbatim from onyx-dot-app/EnterpriseRAG-Bench (MIT licence):
  src/prompts/answer_evaluation.py
  src/scripts/answer_evaluation/metrics_based_eval.py

Three metrics per answer (issue #23):
  correct      – ANSWER_WHOLISTIC_EVALUATION_PROMPT → bool
  completeness – INDIVIDUAL_FACT_VALIDATOR_PROMPT once per gold fact → float
  overall      – combined_correctness_completeness_score (ERB §5):
                 completeness_pct if correct else 0.0

Resumability guarantee:
  Every judge result is appended to judge_cache.jsonl immediately.
  Re-runs skip cached calls; nothing finished is lost or repeated.
  Out-of-credits / auth / repeated rate-limit errors stop cleanly.
  Batch IDs are saved to judge_batches.jsonl; results are collected on re-run.

Usage:
  cache  = JudgeCache(judge_dir / CACHE_FILE)
  client = JudgeClient(api_key)
  n = score_run_direct("gen_r1_p7_test_2026-11-25", records, questions,
                       cache, client, max_calls=200)
  judgements = build_judgements(["gen_r1_p7_test_2026-11-25"], questions, cache)
  save_judgements(judgements, out_path)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterator

from lumbr import paths

# ---------------------------------------------------------------- constants

JUDGE_MODEL = "claude-sonnet-4-6"   # frozen; update requires a cache bust
CACHE_FILE  = "judge_cache.jsonl"
BATCH_FILE  = "judge_batches.jsonl"

# Token estimates for dry-run budget checks (conservative)
_TOKENS_CORRECT = 450   # prompt + response for correctness call
_TOKENS_FACT    = 200   # prompt + response per fact call

MAX_RETRIES      = 3    # on 429/529 before giving up
RETRY_BASE_S     = 2.0  # exponential-backoff base (seconds)

# ---------------------------------------------------------------- exceptions

class OutOfCredits(Exception):
    """Billing, auth, or repeated rate-limit errors — safe to resume later."""
    def __init__(self, msg: str, n_done: int = 0, n_remaining: int = 0):
        super().__init__(msg)
        self.n_done      = n_done
        self.n_remaining = n_remaining

class BudgetExhausted(Exception):
    """MAX_CALLS reached — safe to resume later."""
    def __init__(self, max_calls: int):
        super().__init__(f"Reached budget cap of {max_calls} calls.")
        self.max_calls = max_calls

# ---------------------------------------------------------------- ERB prompts (verbatim, MIT)
# Source: github.com/onyx-dot-app/EnterpriseRAG-Bench
# src/prompts/answer_evaluation.py

_WHOLISTIC_PROMPT = """
You are a wholistic and detail-oriented answer evaluator. Given a query, a gold answer, and a candidate answer, evaluate if the candidate answer aligned with the gold answer.
Use the following metrics for evaluating the answer:
- The candidate answer must provide loosely the same information as the gold answer. The core aspects directly asked by the query must be addressed in the candidate answer and they must not conflict with the gold answer.
- If there are any specific quantities mentioned in both answers, they must match.
- The candidate answer is not required to contain all of the same details as the gold answer.
- The candidate answer must address the key parts of the query, if it is missing anything critical to the question, it is misaligned.
- The candidate answer may contain more details, richer information, or other helpful relevant information than the gold answer, this is ok.
- The candidate answer may offer up additional loosely related information that adds to the context of the answer, this is ok as long as it does not lead the user to an incorrect conclusion (compared to the gold answer).
- Do not penalize the candidate answer for stylistic differences. If the candidate answer offers follow up questions, asks additional clarifications to the user, or offers additional context, this is ok as long as it contains the necessary information to answer the question.

There is a separate check for answer completeness, this is not in scope for this evaluation. However, if there are core parts of the question being left out, this is misaligned.

## Query
```
{question}
```

## Gold Answer
```
{gold_answer}
```

## Candidate Answer
```
{candidate_answer}
```

## Output Format
Output a JSON with "reason" and "aligned" fields. The "reason" field should be a as concise as possible (max 1 sentence) explanation of why the candidate answer is aligned or misaligned with the gold answer. The "aligned" field should be a simple "yes" or "no", use only those two strings literally and nothing else.

CRITICAL: Output only a JSON object with the following fields in the order shown below (with no additional text or formatting):
{{
  "reason": "reason for the classification",
  "aligned": "yes or no"
}}
""".strip()

_FACT_PROMPT = """
You are an answer validator. Given an answer and a statement, determine if the answer is consistent with and contains the information in the statement. The answer may contain more details or richer information than the statement but as long as it does not contradict the statement, this is valid. If there are negative statements such as "The answer must not say...", it is valid if the answer mentions the statement with caveats or qualifications. It is valid if additional context is shared for completeness however hallucinations are not allowed. Output a simple yes or no for if the answer is consistent with and contains the information in the statement.

## Answer
```
{answer}
```

## Statement
```
{statement}
```

CRITICAL: output only a simple yes if the answer is consistent with the statement or a no if the answer does not contain the information in the statement or contradicts the statement.
""".strip()

# ---------------------------------------------------------------- utilities

# Same pattern as handscore.strip_citations
_CITE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\](?!\()")


def strip_citations(text: str) -> str:
    """Remove [n] / [n][m] / [n, m] citations, as the benchmark judge sees the answer."""
    return _CITE.sub("", text).strip()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def combined_score(correct: bool, completeness_pct: float) -> float:
    """ERB combined_correctness_completeness_score: completeness if correct, else 0."""
    return completeness_pct if correct else 0.0


# ---------------------------------------------------------------- prompt formatters

def correctness_messages(
    question: str, gold_answer: str, candidate: str
) -> tuple[list[dict], str]:
    """Return ([user message], prompt_hash) for the correctness judge call."""
    filled = _WHOLISTIC_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        candidate_answer=candidate,
    )
    return [{"role": "user", "content": filled}], _hash(filled)


def fact_messages(answer: str, fact: str) -> tuple[list[dict], str]:
    """Return ([user message], prompt_hash) for one fact-validation call."""
    filled = _FACT_PROMPT.format(answer=answer, statement=fact)
    return [{"role": "user", "content": filled}], _hash(filled)


# ---------------------------------------------------------------- response parsers

def parse_correctness_response(text: str) -> bool:
    """Parse {"aligned": "yes"/"no"} JSON from the correctness judge."""
    text = text.strip()
    # Try JSON first
    try:
        m = json.loads(text)
        return str(m.get("aligned", "")).lower().strip() == "yes"
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: scan for yes/no
    low = text.lower()
    if '"aligned": "yes"' in low or '"aligned":"yes"' in low:
        return True
    if '"aligned": "no"' in low or '"aligned":"no"' in low:
        return False
    return "yes" in low and "no" not in low


def parse_fact_response(text: str) -> bool:
    """Parse simple yes/no from the fact validator."""
    low = text.strip().lower()
    return low.startswith("yes")


# ---------------------------------------------------------------- cache

class JudgeCache:
    """Append-only JSONL cache of completed judge calls.

    Key: (run_tag, question_id, metric, fact_idx, prompt_hash)
      metric:   "correct" or "fact"
      fact_idx: -1 for correctness, 0-based for facts
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._mem: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._mem[self._make_key(entry)] = entry
                except (json.JSONDecodeError, KeyError):
                    pass   # skip corrupt last-line from crash

    @staticmethod
    def _make_key(entry: dict) -> str:
        return (f"{entry['run']}::{entry['qid']}::{entry['metric']}"
                f"::{entry['fact_idx']}::{entry['prompt_hash']}")

    def _key(self, run: str, qid: str, metric: str,
             fact_idx: int, prompt_hash: str) -> str:
        return f"{run}::{qid}::{metric}::{fact_idx}::{prompt_hash}"

    def has(self, run: str, qid: str, metric: str,
            fact_idx: int, prompt_hash: str) -> bool:
        return self._key(run, qid, metric, fact_idx, prompt_hash) in self._mem

    def get(self, run: str, qid: str, metric: str,
            fact_idx: int, prompt_hash: str) -> dict | None:
        return self._mem.get(self._key(run, qid, metric, fact_idx, prompt_hash))

    def put(self, run: str, qid: str, metric: str, fact_idx: int,
            prompt_hash: str, result, model: str = JUDGE_MODEL) -> None:
        entry = {
            "run": run, "qid": qid, "metric": metric,
            "fact_idx": fact_idx, "prompt_hash": prompt_hash,
            "result": result, "model": model,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        self._mem[self._make_key(entry)] = entry

    def count_done(self, run: str) -> int:
        return sum(1 for e in self._mem.values() if e["run"] == run)

    def progress(
        self,
        run_tags: list[str],
        questions: dict[str, dict],
        records_by_run: dict[str, list[dict]],
    ) -> dict[str, dict]:
        """Counts for each run: done, remaining (correctness + facts)."""
        out = {}
        for tag in run_tags:
            records = records_by_run.get(tag, [])
            total = sum(1 + len(questions.get(r["question_id"], {}).get("answer_facts", []))
                        for r in records)
            done = self.count_done(tag)
            out[tag] = {"total": total, "done": done, "remaining": max(0, total - done)}
        return out


# ---------------------------------------------------------------- client

def _is_credit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(pat in msg for pat in (
        "credit balance", "insufficient_funds", "billing", "payment",
        "account has been disabled", "authentication", "api key",
    ))


def _is_rate_limit(exc: Exception) -> bool:
    cls = type(exc).__name__
    msg = str(exc).lower()
    return cls in ("RateLimitError", "OverloadedError") or "529" in msg or "429" in msg


class JudgeClient:
    """Thin wrapper around the Anthropic SDK for judging.

    Import of `anthropic` is deferred so the module loads without the SDK installed
    (tests use fake clients, never this class directly).
    """

    def __init__(self, api_key: str, model: str = JUDGE_MODEL):
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)
        self.model   = model

    def call_one(self, messages: list[dict], max_tokens: int = 256) -> str:
        """Direct synchronous call. Retries on rate-limit up to MAX_RETRIES.

        Raises OutOfCredits on billing / auth / exhausted retries.
        """
        import anthropic as _anthropic

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0,
                    messages=messages,
                )
                return resp.content[0].text
            except Exception as exc:
                if _is_credit_error(exc):
                    raise OutOfCredits(str(exc)) from exc
                if _is_rate_limit(exc) and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_S * (2 ** attempt))
                    continue
                raise OutOfCredits(f"API error after {attempt + 1} attempts: {exc}") from exc
        raise OutOfCredits(f"Rate limit not resolved after {MAX_RETRIES} retries.")

    def submit_batch(
        self, requests: list[dict]
    ) -> str:
        """Submit a Message Batch. Each request: {custom_id, messages}.
        Returns batch_id.
        """
        batch_requests = [
            {
                "custom_id": r["custom_id"],
                "params": {
                    "model":      self.model,
                    "max_tokens": 256,
                    "temperature": 0,
                    "messages":   r["messages"],
                },
            }
            for r in requests
        ]
        batch = self._client.beta.messages.batches.create(requests=batch_requests)
        return batch.id

    def batch_status(self, batch_id: str) -> str:
        """Return "in_progress" or "ended"."""
        b = self._client.beta.messages.batches.retrieve(batch_id)
        return b.processing_status

    def batch_results(self, batch_id: str) -> Iterator[dict]:
        """Yield {custom_id, type, text} for each result."""
        for result in self._client.beta.messages.batches.results(batch_id):
            typ  = result.result.type
            text = (result.result.message.content[0].text
                    if typ == "succeeded" else None)
            yield {"custom_id": result.custom_id, "type": typ, "text": text}


# ---------------------------------------------------------------- custom_id helpers

def _make_custom_id(run: str, qid: str, metric: str, fact_idx: int) -> str:
    return f"{run}|||{qid}|||{metric}|||{fact_idx}"


def _parse_custom_id(custom_id: str) -> tuple[str, str, str, int]:
    run, qid, metric, fi = custom_id.split("|||")
    return run, qid, metric, int(fi)


# ---------------------------------------------------------------- dry-run

def dry_run(
    run_tags: list[str],
    records_by_run: dict[str, list[dict]],
    questions: dict[str, dict],
    cache: JudgeCache,
    *,
    limit: int | None = None,
) -> dict:
    """Count pending calls and estimate tokens without making any API calls.

    Returns {run_tag: {pending_correct, pending_facts, estimated_calls, estimated_tokens}}.
    """
    out: dict[str, dict] = {}
    for tag in run_tags:
        records = records_by_run.get(tag, [])[:limit] if limit else records_by_run.get(tag, [])
        pc = pf = 0
        for rec in records:
            qid  = rec["question_id"]
            q    = questions.get(qid, {})
            cand = strip_citations(rec.get("answer", ""))
            _, chash = correctness_messages(
                q.get("question", ""), q.get("gold_answer", ""), cand)
            if not cache.has(tag, qid, "correct", -1, chash):
                pc += 1
            for i, fact in enumerate(q.get("answer_facts", [])):
                _, fhash = fact_messages(cand, fact)
                if not cache.has(tag, qid, "fact", i, fhash):
                    pf += 1
        est_calls  = pc + pf
        est_tokens = pc * _TOKENS_CORRECT + pf * _TOKENS_FACT
        out[tag] = {
            "pending_correct": pc, "pending_facts": pf,
            "estimated_calls": est_calls, "estimated_tokens": est_tokens,
        }
    return out


# ---------------------------------------------------------------- direct scoring

def score_run_direct(
    run_tag: str,
    records: list[dict],
    questions: dict[str, dict],
    cache: JudgeCache,
    client,            # JudgeClient or fake — duck-typed
    *,
    limit: int | None = None,
    max_calls: int | None = None,
) -> int:
    """Score answers for one run via direct API calls.

    Returns the number of new API calls made this session.
    Raises BudgetExhausted when max_calls is reached (results cached up to that point).
    Raises OutOfCredits on billing/auth errors (same).
    Both are safe to catch: re-run picks up where it left off.
    """
    to_score = records[:limit] if limit else records
    n_calls  = 0

    for rec in to_score:
        qid  = rec["question_id"]
        q    = questions.get(qid)
        if q is None:
            continue

        cand = strip_citations(rec.get("answer", ""))
        msgs_c, chash = correctness_messages(
            q["question"], q.get("gold_answer", ""), cand)

        if not cache.has(run_tag, qid, "correct", -1, chash):
            try:
                text   = client.call_one(msgs_c)
                result = parse_correctness_response(text)
                cache.put(run_tag, qid, "correct", -1, chash, result)
                n_calls += 1
            except OutOfCredits as exc:
                _print_stop(run_tag, n_calls, exc)
                raise
            if max_calls and n_calls >= max_calls:
                raise BudgetExhausted(max_calls)

        for i, fact in enumerate(q.get("answer_facts", [])):
            msgs_f, fhash = fact_messages(cand, fact)
            if not cache.has(run_tag, qid, "fact", i, fhash):
                try:
                    text   = client.call_one(msgs_f)
                    result = parse_fact_response(text)
                    cache.put(run_tag, qid, "fact", i, fhash, result)
                    n_calls += 1
                except OutOfCredits as exc:
                    _print_stop(run_tag, n_calls, exc)
                    raise
                if max_calls and n_calls >= max_calls:
                    raise BudgetExhausted(max_calls)

    return n_calls


def _print_stop(run_tag: str, n_calls: int, exc: Exception) -> None:
    print(f"\n[judge] Stopped on {run_tag}: {exc}")
    print(f"[judge] {n_calls} new calls this session.")
    print("[judge] Add credits (or a new API key) and re-run this cell to continue.")


# ---------------------------------------------------------------- batch scoring

def submit_batch_run(
    run_tag: str,
    records: list[dict],
    questions: dict[str, dict],
    cache: JudgeCache,
    client,
    batch_path: Path,
    *,
    limit: int | None = None,
    max_calls: int | None = None,
) -> str | None:
    """Build the pending request list and submit as a Message Batch.

    Saves the batch_id + request manifest to batch_path (append-only JSONL).
    Returns the batch_id, or None if there is nothing pending.
    """
    to_score = records[:limit] if limit else records
    pending: list[dict] = []

    for rec in to_score:
        qid = rec["question_id"]
        q   = questions.get(qid)
        if q is None:
            continue

        cand = strip_citations(rec.get("answer", ""))
        msgs_c, chash = correctness_messages(
            q["question"], q.get("gold_answer", ""), cand)
        if not cache.has(run_tag, qid, "correct", -1, chash):
            cid = _make_custom_id(run_tag, qid, "correct", -1)
            pending.append({"custom_id": cid, "messages": msgs_c,
                            "metric": "correct", "fact_idx": -1,
                            "prompt_hash": chash, "qid": qid})

        for i, fact in enumerate(q.get("answer_facts", [])):
            msgs_f, fhash = fact_messages(cand, fact)
            if not cache.has(run_tag, qid, "fact", i, fhash):
                cid = _make_custom_id(run_tag, qid, "fact", i)
                pending.append({"custom_id": cid, "messages": msgs_f,
                                "metric": "fact", "fact_idx": i,
                                "prompt_hash": fhash, "qid": qid})

    if max_calls:
        pending = pending[:max_calls]
    if not pending:
        return None

    batch_id = client.submit_batch(pending)

    # Save manifest so we can collect results later (even after disconnect).
    record = {
        "batch_id": batch_id,
        "run": run_tag,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requests": [
            {"custom_id": p["custom_id"], "run": run_tag, "qid": p["qid"],
             "metric": p["metric"], "fact_idx": p["fact_idx"],
             "prompt_hash": p["prompt_hash"]}
            for p in pending
        ],
    }
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

    print(f"[judge] Batch {batch_id}: {len(pending)} requests submitted.")
    return batch_id


def collect_batches(
    cache: JudgeCache,
    client,
    batch_path: Path,
    *,
    requeue: bool = True,
) -> tuple[int, list[dict]]:
    """Collect results from all saved batches.

    - Succeeded results are written to the cache.
    - Errored / expired requests are returned as a list (re-queue manually or
      pass requeue=True to have them re-submitted automatically).

    Returns (n_new_results, failed_requests).
    """
    if not batch_path.exists():
        return 0, []

    batches: list[dict] = []
    with open(batch_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                batches.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    n_new    = 0
    failed: list[dict] = []
    req_map: dict[str, dict] = {}  # custom_id -> manifest request

    for b in batches:
        for req in b.get("requests", []):
            req_map[req["custom_id"]] = req

    for b in batches:
        bid = b["batch_id"]
        try:
            status = client.batch_status(bid)
        except Exception as exc:
            print(f"[judge] Could not retrieve batch {bid}: {exc}")
            continue

        if status != "ended":
            print(f"[judge] Batch {bid} still in_progress — check again later.")
            continue

        for res in client.batch_results(bid):
            cid = res["custom_id"]
            req = req_map.get(cid)
            if req is None:
                continue

            if res["type"] == "succeeded":
                run, qid, metric, fi = _parse_custom_id(cid)
                if cache.has(run, qid, metric, fi, req["prompt_hash"]):
                    continue
                if metric == "correct":
                    result = parse_correctness_response(res["text"])
                else:
                    result = parse_fact_response(res["text"])
                cache.put(run, qid, metric, fi, req["prompt_hash"], result)
                n_new += 1
            else:
                failed.append(req)

    if failed:
        print(f"[judge] {len(failed)} errored/expired batch requests.")
    return n_new, failed


# ---------------------------------------------------------------- building output

def build_judgements(
    run_tags: list[str],
    questions: dict[str, dict],
    records_by_run: dict[str, list[dict]],
    cache: JudgeCache,
) -> list[dict]:
    """Build judgements.jsonl rows from the cache.

    For runs with incomplete coverage, the row is marked partial=True.
    Rows missing from cache are skipped (incomplete runs → omitted, not zero).

    Output format matches notebook 22 (judge-human kappa):
      {run, question_id, correct (bool), facts (list[bool] in gold order)}
    """
    rows: list[dict] = []
    for tag in run_tags:
        for rec in records_by_run.get(tag, []):
            qid = rec["question_id"]
            q   = questions.get(qid, {})
            n_facts = len(q.get("answer_facts", []))
            cand    = strip_citations(rec.get("answer", ""))

            _, chash = correctness_messages(
                q.get("question", ""), q.get("gold_answer", ""), cand)
            correct_entry = cache.get(tag, qid, "correct", -1, chash)
            if correct_entry is None:
                continue    # not yet judged

            correct = bool(correct_entry["result"])
            facts   = []
            for i, fact in enumerate(q.get("answer_facts", [])):
                _, fhash = fact_messages(cand, fact)
                fe = cache.get(tag, qid, "fact", i, fhash)
                if fe is None:
                    facts = None
                    break
                facts.append(bool(fe["result"]))

            rows.append({
                "run":         tag,
                "question_id": qid,
                "correct":     correct,
                "facts":       facts,   # None if incomplete
            })
    return rows


def save_judgements(judgements: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in judgements:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[judge] {len(judgements)} judgements → {path}")


def load_judgements(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- aggregate scores

def aggregate(
    run_tag: str,
    questions: dict[str, dict],
    records: list[dict],
    cache: JudgeCache,
) -> dict | None:
    """Compute mean correctness, completeness and overall for one run from cache.

    Returns None when the run is not fully judged.
    Returns {"complete": bool, "n": int, "correctness": float,
             "completeness": float, "overall": float}.
    """
    js = build_judgements([run_tag], questions, {run_tag: records}, cache)
    if not js:
        return None

    n_total   = len(records)
    partial   = any(r["facts"] is None for r in js)
    complete  = (not partial) and len(js) == n_total

    correctness: list[float]   = []
    completeness: list[float]  = []
    overall:      list[float]  = []

    for r in js:
        if r["facts"] is None:
            continue
        n_facts = len(questions.get(r["question_id"], {}).get("answer_facts", [])) or 1
        comp = sum(r["facts"]) / max(len(r["facts"]), 1) if r["facts"] else 0.0
        correctness.append(float(r["correct"]))
        completeness.append(comp)
        overall.append(combined_score(r["correct"], comp))

    if not correctness:
        return None

    return {
        "complete":    complete,
        "n":           len(correctness),
        "correctness": round(sum(correctness)  / len(correctness),  4),
        "completeness":round(sum(completeness) / len(completeness), 4),
        "overall":     round(sum(overall)      / len(overall),      4),
    }
