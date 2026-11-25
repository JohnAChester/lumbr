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

JUDGE_MODEL = "claude-sonnet-5-5"   # frozen; update requires a cache bust
# Current Claude models reject `temperature`, so it isn't sent. Sonnet 5.5 thinks by default and the
# thinking counts toward max_tokens; "between_tools" turns it off for these tool-free yes/no calls.
JUDGE_THINKING = {"type": "between_tools"}
JUDGE_MAX_TOKENS = 256
# Fallback for replies the plain call can't settle: same prompt, reply forced to this JSON.
STRICT_MAX_TOKENS = 1024
STRICT_MODEL_TAG  = "claude-sonnet-5-5+json-schema"   # cache "model" field for these verdicts
STRICT_SCHEMAS = {
    "correct": {"type": "object", "additionalProperties": False, "required": ["reason", "aligned"],
                "properties": {"reason": {"type": "string"},
                               "aligned": {"type": "string", "enum": ["yes", "no"]}}},
    "fact":    {"type": "object", "additionalProperties": False, "required": ["supported"],
                "properties": {"supported": {"type": "string", "enum": ["yes", "no"]}}},
}
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

# First word of a reply, past leading markdown and punctuation ("**Yes**", "> No", "`yes`").
_FIRST_WORD = re.compile(r"^[\s*_`#>\"'(\[]*([a-z]+)")
_ALIGNED    = re.compile(r'"aligned"\s*:\s*"\s*(yes|no)\b')


def _yes_no(word: str | None) -> bool | None:
    return True if word == "yes" else False if word == "no" else None


def parse_verdict(metric: str, text: str | None) -> bool | None:
    """The verdict for either metric, or None when the reply has none."""
    if text is None:
        return None
    return parse_correctness_response(text) if metric == "correct" else parse_fact_response(text)


def parse_correctness_response(text: str) -> bool | None:
    """{"aligned": "yes"/"no"} from the correctness judge -> True/False; None if there's no verdict.

    None (not False) for anything else, so an unparseable reply is retried, never scored "no".
    """
    text = text.strip()
    try:
        m = json.loads(text)
        if isinstance(m, dict):
            return _yes_no(str(m.get("aligned", "")).strip().lower())
    except (json.JSONDecodeError, ValueError):
        pass
    hit = _ALIGNED.search(text.lower())   # JSON inside a code fence or after a sentence
    return _yes_no(hit.group(1)) if hit else None


def parse_fact_response(text: str) -> bool | None:
    """The fact validator's yes/no -> True/False, read from the first word only; None otherwise.

    The prompt asks for a bare yes or no. A reply that explains first ("The answer states...")
    or starts "Not"/"Note"/"None" has no clear verdict: None, so it's retried rather than
    counted as "no" (which `startswith("no")` used to do for "Not..." and "Note...").
    """
    m = _FIRST_WORD.match(text.strip().lower())
    return _yes_no(m.group(1)) if m else None


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
                    if entry.get("dropped"):          # tombstone written by drop()
                        self._mem.pop(self._make_key(entry), None)
                    else:
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

    def drop(self, run: str, qid: str, metric: str, fact_idx: int, prompt_hash: str) -> None:
        """Forget a cached verdict (append-only: writes a tombstone that _load honours)."""
        entry = {"run": run, "qid": qid, "metric": metric, "fact_idx": fact_idx,
                 "prompt_hash": prompt_hash, "dropped": True,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        self._mem.pop(self._make_key(entry), None)

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

    def _create(self, **params):
        """messages.create with the judge's model and thinking. Retries on rate-limit up to
        MAX_RETRIES; raises OutOfCredits on billing / auth / exhausted retries."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._client.messages.create(
                    model=self.model, thinking=JUDGE_THINKING, **params)
            except Exception as exc:
                if _is_credit_error(exc):
                    raise OutOfCredits(str(exc)) from exc
                if _is_rate_limit(exc) and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_S * (2 ** attempt))
                    continue
                raise OutOfCredits(f"API error after {attempt + 1} attempts: {exc}") from exc
        raise OutOfCredits(f"Rate limit not resolved after {MAX_RETRIES} retries.")

    def call_one(self, messages: list[dict], max_tokens: int = JUDGE_MAX_TOKENS) -> str | None:
        """Direct synchronous call. Returns the reply text, or None when there is no usable
        text (see response_text). Raises OutOfCredits on billing / auth / exhausted retries."""
        return response_text(self._create(max_tokens=max_tokens, messages=messages))

    def call_strict(self, messages: list[dict], metric: str,
                    max_tokens: int = STRICT_MAX_TOKENS) -> bool | None:
        """Same prompt, but the reply is forced to STRICT_SCHEMAS[metric] (structured outputs).

        For the few replies the plain call can't settle (explained instead of answering, or ran
        past max_tokens). The benchmark's prompt text is unchanged; only the reply format is
        enforced. Returns True/False, or None if there's still no verdict (e.g. a refusal).
        """
        resp = self._create(max_tokens=max_tokens, messages=messages,
                            output_config={"format": {"type": "json_schema",
                                                      "schema": STRICT_SCHEMAS[metric]}})
        text = response_text(resp)
        if text is None:
            return None
        try:
            reply = json.loads(text)
        except json.JSONDecodeError:
            return None
        key = "aligned" if metric == "correct" else "supported"
        return _yes_no(str(reply.get(key, "")).strip().lower())

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
                    "max_tokens": JUDGE_MAX_TOKENS,
                    "thinking":   JUDGE_THINKING,
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
        """Yield {custom_id, type, text} for each result.

        A request that succeeded but has no usable text (a refusal, or cut off at max_tokens)
        comes back as type "no_text", so collect_batches treats it as failed, not as a "no".
        """
        for result in self._client.beta.messages.batches.results(batch_id):
            typ  = result.result.type
            text = response_text(result.result.message) if typ == "succeeded" else None
            if typ == "succeeded" and text is None:
                typ = "no_text"
            yield {"custom_id": result.custom_id, "type": typ, "text": text}


def response_text(message) -> str | None:
    """The reply's text blocks joined, or None when there is nothing to parse.

    None for a refusal (stop_reason "refusal", content may be empty), a reply cut off at
    max_tokens, or a reply with no text block. The parsers read anything unparseable as "no",
    so these must never reach them: they would be cached as a wrong verdict and never retried.
    """
    if getattr(message, "stop_reason", None) in ("refusal", "max_tokens"):
        return None
    parts = [b.text for b in (message.content or []) if getattr(b, "type", None) == "text"]
    text = "".join(parts).strip()
    return text or None


# ---------------------------------------------------------------- custom_id helpers

# The Batches API only accepts custom_ids matching this (no "|", ".", spaces; at most 64 chars).
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _make_custom_id(run: str, qid: str, metric: str, fact_idx: int) -> str:
    """`{run}-{qid}-c` for correctness, `{run}-{qid}-f{i}` for fact i, e.g. `gen_r1_p7-qst_0123-f3`.

    Run tags and question IDs use only letters, digits and "_", so "-" separates them safely.
    Raises ValueError for anything the API would reject, before any batch is submitted.
    """
    cid = f"{run}-{qid}-" + ("c" if metric == "correct" else f"f{fact_idx}")
    if "-" in run or "-" in qid or not CUSTOM_ID_RE.match(cid):
        raise ValueError(f"custom_id {cid!r} is not valid for the Batches API "
                         "(run tag and question id: letters, digits and _ only; 64 chars max)")
    return cid


def _parse_custom_id(custom_id: str) -> tuple[str, str, str, int]:
    run, qid, kind = custom_id.split("-")
    return (run, qid, "correct", -1) if kind == "c" else (run, qid, "fact", int(kind[1:]))


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
                text = client.call_one(msgs_c)
                n_calls += 1
                # None (no reply text, or no verdict in it): left uncached, retried next run.
                verdict = parse_correctness_response(text) if text is not None else None
                if verdict is not None:
                    cache.put(run_tag, qid, "correct", -1, chash, verdict)
            except OutOfCredits as exc:
                _print_stop(run_tag, n_calls, exc)
                raise
            if max_calls and n_calls >= max_calls:
                raise BudgetExhausted(max_calls)

        for i, fact in enumerate(q.get("answer_facts", [])):
            msgs_f, fhash = fact_messages(cand, fact)
            if not cache.has(run_tag, qid, "fact", i, fhash):
                try:
                    text = client.call_one(msgs_f)
                    n_calls += 1
                    verdict = parse_fact_response(text) if text is not None else None
                    if verdict is not None:
                        cache.put(run_tag, qid, "fact", i, fhash, verdict)
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

def in_flight_custom_ids(client, batch_path: Path, run_tag: str) -> set[str]:
    """custom_ids in this run's saved batches that haven't ended yet (status other than "ended").

    A batch whose status can't be read counts as in flight: skipping it costs one more re-run,
    resubmitting it could pay for the same requests twice.
    """
    if not Path(batch_path).exists():
        return set()
    ids: set[str] = set()
    with open(batch_path, encoding="utf-8") as f:
        for line in f:
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            if b.get("run") != run_tag:
                continue
            try:
                ended = client.batch_status(b["batch_id"]) == "ended"
            except Exception:
                ended = False
            if not ended:
                ids.update(r["custom_id"] for r in b.get("requests", []))
    return ids


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

    Requests already in one of this run's batches that hasn't ended are skipped: they aren't
    cached until that batch is collected, so without this, re-running the full-run cell while a
    batch is processing would submit (and pay for) the same requests again.
    """
    in_flight = in_flight_custom_ids(client, batch_path, run_tag)
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
        cid = _make_custom_id(run_tag, qid, "correct", -1)
        if not cache.has(run_tag, qid, "correct", -1, chash) and cid not in in_flight:
            pending.append({"custom_id": cid, "messages": msgs_c,
                            "metric": "correct", "fact_idx": -1,
                            "prompt_hash": chash, "qid": qid})

        for i, fact in enumerate(q.get("answer_facts", [])):
            msgs_f, fhash = fact_messages(cand, fact)
            cid = _make_custom_id(run_tag, qid, "fact", i)
            if not cache.has(run_tag, qid, "fact", i, fhash) and cid not in in_flight:
                pending.append({"custom_id": cid, "messages": msgs_f,
                                "metric": "fact", "fact_idx": i,
                                "prompt_hash": fhash, "qid": qid})

    if in_flight:
        print(f"[judge] {run_tag}: {len(in_flight)} requests still processing in an earlier batch; not resubmitted.")
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
                # The manifest is the source of truth for what each request was.
                run, qid, metric, fi = req["run"], req["qid"], req["metric"], req["fact_idx"]
                if cache.has(run, qid, metric, fi, req["prompt_hash"]):
                    continue
                result = parse_verdict(metric, res["text"])
                if result is None:            # ran, but no yes/no verdict in the reply
                    failed.append(req)
                    continue
                cache.put(run, qid, metric, fi, req["prompt_hash"], result)
                n_new += 1
            else:
                failed.append(req)

    if failed:
        print(f"[judge] {len(failed)} batch requests without a usable verdict "
              "(errored, expired, cut off, or no yes/no): rejudge_direct() retries them.")
    return n_new, failed


# ---------------------------------------------------------------- building output

def _read_manifest(batch_path: Path) -> list[dict]:
    if not Path(batch_path).exists():
        return []
    out = []
    with open(batch_path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def reparse_batches(cache: JudgeCache, client, batch_path: Path) -> tuple[int, int, list[dict]]:
    """Re-read every saved batch's replies with the strict parsers and correct the cache.

    For verdicts cached by an earlier, looser parser (which scored "Not...", "Note...", and
    replies that explained instead of answering as "no"). Results stay downloadable for a while
    after a batch ends, so this costs nothing. Verdicts already settled by rejudge_direct
    (STRICT_MODEL_TAG) are kept.

    Returns (n_changed, n_dropped, unusable): changed = cached verdict replaced by the strict
    parse; dropped = cached verdict removed because the reply has no clear yes/no; unusable =
    manifest requests to pass to rejudge_direct.
    """
    n_changed = n_dropped = 0
    unusable: dict[str, dict] = {}
    for b in _read_manifest(batch_path):
        if client.batch_status(b["batch_id"]) != "ended":
            print(f"[judge] batch {b['batch_id']} hasn't ended; skipped.")
            continue
        reqs = {r["custom_id"]: r for r in b.get("requests", [])}
        for res in client.batch_results(b["batch_id"]):
            req = reqs.get(res["custom_id"])
            if req is None:
                continue
            key = (req["run"], req["qid"], req["metric"], req["fact_idx"], req["prompt_hash"])
            cached = cache.get(*key)
            if cached and cached.get("model") == STRICT_MODEL_TAG:
                unusable.pop(res["custom_id"], None)
                continue
            verdict = (parse_verdict(req["metric"], res["text"])
                       if res["type"] == "succeeded" else None)
            if verdict is None:
                if cached:
                    cache.drop(*key)
                    n_dropped += 1
                unusable[res["custom_id"]] = req
            else:
                unusable.pop(res["custom_id"], None)   # a later batch settled it
                if cached is None or cached["result"] != verdict:
                    cache.put(*key, verdict)
                    n_changed += 1
    return n_changed, n_dropped, list(unusable.values())


def rejudge_direct(reqs: list[dict], records_by_run: dict[str, list[dict]],
                   questions: dict[str, dict], cache: JudgeCache,
                   client) -> tuple[int, list[dict]]:
    """Re-ask unusable requests one by one with client.call_strict (forced yes/no JSON).

    Rebuilds each prompt from the answer records and checks its hash against the manifest, so a
    changed answer or prompt is never scored under the old key. Returns (n_settled,
    still_unusable). Raises OutOfCredits like the other paths; re-running continues where it
    stopped, since settled verdicts are cached.
    """
    by_qid = {run: {r["question_id"]: r for r in recs} for run, recs in records_by_run.items()}
    n_done, still = 0, []
    for req in reqs:
        run, qid, metric, fi, h = (req["run"], req["qid"], req["metric"],
                                   req["fact_idx"], req["prompt_hash"])
        if cache.has(run, qid, metric, fi, h):
            continue
        rec, q = by_qid.get(run, {}).get(qid), questions.get(qid)
        if rec is None or q is None:
            still.append(req)
            continue
        cand = strip_citations(rec.get("answer", ""))
        if metric == "correct":
            msgs, hh = correctness_messages(q["question"], q.get("gold_answer", ""), cand)
        else:
            msgs, hh = fact_messages(cand, q["answer_facts"][fi])
        if hh != h:
            still.append(req)
            continue
        verdict = client.call_strict(msgs, metric)
        if verdict is None:
            still.append(req)
            continue
        cache.put(run, qid, metric, fi, h, verdict, model=STRICT_MODEL_TAG)
        n_done += 1
    return n_done, still


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
