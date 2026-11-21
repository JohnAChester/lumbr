# Passage schema (v1)

The contract between chunking (John, #3/#4), retrieval and eval (Patricia, #7/#8/#10), and synthetic
questions and fine-tuning (Jieun, #9/#11/#15). Code: [`src/lumbr/schema.py`](../src/lumbr/schema.py).
**Sign-off = an approved PR from both Patricia and Jieun (deadline Nov 11).** Any later change needs a new PR that both approve.

## Format

One JSON object per line (`.jsonl`, UTF-8). Files on Drive:

| file | contents |
|---|---|
| `lumbr-data/processed/sample_passages.jsonl` | **v0 sample (#3)**: 1,000 docs, ~4.6K passages, 12 MB. Includes every gold doc of 50 questions (listed in `sample_manifest.json`). |
| `lumbr-data/processed/passages.jsonl` | full subsample (#4): 139,119 docs, 772,960 passages (227M tokens), built at `e4980ad` |

## Fields

| field | type | meaning |
|---|---|---|
| `passage_id` | str | `{doc_id}#{chunk_idx:03d}`, unique |
| `doc_id` | str | `dsid_…`. **Matches the benchmark's gold `expected_doc_ids`.** Use `schema.doc_id_of(passage_id)` for doc-level scoring. |
| `source` | str | `slack`, `gmail`, `linear`, `jira`, `hubspot`, `google_drive`, `fireflies`, `github`, `confluence` |
| `title` | str | document title (max 150 chars) |
| `channel` | str or null | where it lives: Slack channel, mailbox, project, space, repo, … ([mapping](SOURCES.md#metadata-mapping-d6d7-implemented-in-chunkmeta_keys)) |
| `author` | str or null | who wrote it (thread starter for Slack/Gmail) |
| `date` | str or null | ISO-8601, most recent activity (`YYYY-MM-DD` or full timestamp) |
| `created` | str or null | ISO-8601, document creation |
| `chunk_idx` | int | 0-based position within the document |
| `n_chunks` | int | number of chunks in the document. Neighbours are `chunk_idx ± 1` (for D16 "small-to-big" generation). |
| `n_tokens` | int | **the larger of R2's and R3's token counts for `text`, including special tokens; always ≤ 512** |
| `header` | str | one-line metadata header |
| `body` | str | chunk content without the header |
| `text` | str | `header + "\n\n" + body`. **The field everyone indexes and embeds by default.** |

`null` means unknown, and is never guessed. Missing metadata is simply left out of the header.

## Example (a real passage)

```json
{
  "passage_id": "dsid_f983fca1510e4cf492c0f39a4984d80c#008",
  "doc_id": "dsid_f983fca1510e4cf492c0f39a4984d80c",
  "source": "jira",
  "title": "Unicode normalization mismatch in idempotency keys causing duplicated tool executions across Python/TS/Go SDKs",
  "channel": "customer-support",
  "author": "Amelia Chen",
  "date": "2026-03-12",
  "created": "2026-03-10",
  "chunk_idx": 8,
  "n_chunks": 9,
  "n_tokens": 97,
  "header": "[jira] customer-support · Unicode normalization mismatch in idempotency keys causing duplicated tool executions across Python/TS/Go SDKs · Amelia Chen · 2026-03-12",
  "body": "- Add telemetry to detect non-ASCII idempotency usage and alert when percent-encoding/folding is observed at edge.\n- Follow up with FastQuery in 3 days to confirm absence of duplicates after prod rollout.",
  "text": "[jira] customer-support · Unicode normalization … · Amelia Chen · 2026-03-12\n\n- Add telemetry … after prod rollout."
}
```

## Token budget

- **Cap: 512 tokens including special tokens, under BOTH tokenizers**: `BAAI/bge-base-en-v1.5` (R2/R4, hard
  512-position limit, adds `[CLS]`/`[SEP]`) and `Qwen/Qwen3-Embedding-4B` (R3, adds `<|endoftext|>`).
  **No retriever ever truncates a passage.** Qwen splits numbers into single digits, so it is the larger count for
  about 1% of passages. That's why both are checked.
- Chunks are packed toward about 350 tokens (median about 310). Natural units (an email, a message, a comment, a
  section) are only split when a single unit is over the cap.
- Issue #4 asks for "at most ~256 tokens". It was written before this schema; **the ~350 target and 512 cap here are
  what was signed off and what the code does.**
- Reranker (`bge-reranker-v2-m3`): use `max_length=640` so query + passage always fits.
- Other tokenizers (Llama, the reranker's) count slightly differently, so use `n_tokens` as an estimate for them.

## Using it

```python
from lumbr import paths, schema

for p in schema.read_jsonl(paths.PROCESSED / "sample_passages.jsonl"):
    p["passage_id"], p["text"]          # what to index
    schema.doc_id_of(p["passage_id"])   # doc-level scoring: a doc counts if any of its passages is retrieved

schema.validate_jsonl(path)             # raises on any schema problem; returns {"passages", "docs", ...}
```

Pyserini (R1) wants `{"id", "contents"}`:

```python
import json
with open("pyserini/docs.jsonl", "w") as f:
    for p in schema.read_jsonl(paths.PROCESSED / "sample_passages.jsonl"):
        f.write(json.dumps({"id": p["passage_id"], "contents": p["text"]}) + "\n")
```

### Small-to-big: neighbours at generation time (D16)

Search and scoring stay at passage level: every index holds single passages, and retrieval metrics are computed on what
the search returns. Only **after** search, when building the generator's context, each hit is expanded to its
neighbours `chunk_idx - 1 .. chunk_idx + 1` from the same document. That recovers context cut at a chunk boundary
(a question at the end of one chunk, its answer at the start of the next) without making the indexed chunks bigger.

```python
by_id = {p["passage_id"]: p for p in schema.read_jsonl(path)}

def expand(hit: dict, k: int = 1) -> list[dict]:
    """The hit plus up to k neighbours on each side, in document order."""
    lo, hi = max(0, hit["chunk_idx"] - k), min(hit["n_chunks"] - 1, hit["chunk_idx"] + k)
    return [by_id[schema.make_passage_id(hit["doc_id"], i)] for i in range(lo, hi + 1)]
```

When two hits from the same document are adjacent, merge their windows so no passage is given to the generator twice,
and cite the hit's own `passage_id`. Neighbours use `body` (not `text`) after the first, so the header isn't repeated.
Slack, Gmail and Fireflies passages start with a short **context carry** from the previous chunk (see below), so when
joining consecutive passages, drop it with `chunk.drop_carry(prev["body"], cur["body"])`:

```python
from lumbr import chunk

def merged_body(window: list[dict]) -> str:
    """Consecutive passages of one document as one text, without repeated carry."""
    parts = [window[0]["body"]] + [chunk.drop_carry(a["body"], b["body"]) for a, b in zip(window, window[1:])]
    return "\n".join(parts)
```

`by_id` above holds every passage in memory: fine for the sample, a few GB for the full file, so the generator may
want a lighter lookup (e.g. an offset index into the JSONL).

Use `body` in place of `text` for the **with/without-header ablation** (D5). It also gives a BM25 number comparable to the
paper's, whose baselines had no metadata.

## How documents become passages

1. **Units.** Each document's `content_field_names` are rendered in order into natural units:
   - Slack: one unit per message.
   - Gmail: one unit per email. A quoted earlier email is removed and leaves a one-line stub,
     `[quoted reply removed: On <date>, <sender> wrote]` (or `[quoted reply removed]` when there is no `On … wrote:` line),
     so a reply like "Thursday works" still shows that it answers someone. Forwarded emails are kept.
   - Fireflies: one unit per transcript turn. A turn starts at `Name:` with an optional timestamp in any of the
     corpus's formats: `[00:15]`, `09:00`, `00:00:02 -`, `05:41 - 06:25`, `[00:00] [00:05]`, `[S00:18]`
     (`chunk.TRANSCRIPT_SPEAKER`). All but 14 of 10,085 transcripts split into turns. Slack keeps the stricter
     `chunk.SPEAKER`, because the looser pattern matches log lines inside Slack messages.
   - Lists (comments, acceptance criteria, …): one unit per item.
   - Prose: paragraphs, with headings kept attached.
   - Non-body fields get a label, e.g. `Acceptance criteria:`.
2. **Packing.** Consecutive units are packed greedily toward about 350 tokens and never past 512. An oversized single unit is
   split at paragraphs, then lines, then sentences, then words.
   - **Context carry (Slack, Gmail, Fireflies).** Every chunk after the first starts with up to 60 tokens repeated
     from the end of the previous chunk: as many whole messages/emails/turns as fit, or else the longest tail of whole
     lines or sentences of the last one; never part of a sentence. Conversational replies refer back ("yes, let's push
     it to Friday"), so this keeps the question and its answer in the same chunk for **search**; small-to-big
     (above) only helps after search. Room for the carry is reserved during packing, so chunks still aim at ~350
     tokens. Other sources have section structure and get no carry. Expect roughly 10% more passages (measured on
     the test fixtures). `chunk.drop_carry` removes it when joining neighbours.
   - **Continued list labels.** When a chunk starts partway through a list field, the field's label is repeated as
     `Acceptance criteria (continued):` so the items still make sense on their own.
3. **Cleaning.** Literal `\n` escapes are unescaped, titles are cut to 150 chars, and implausible dates become null. Details in [SOURCES.md](SOURCES.md).
4. **Never** lowercase, stem, dedupe or expand acronyms.
5. The 4 doc IDs shared by two files are chunked as one document (chunks numbered continuously).

The carry, quote stubs and continued labels were planned here at sign-off and added in #4; no field changed.
"Window transcripts" in issue #4 is met by the carry: windows aligned to speaker turns, overlapping by ≤ 60 tokens.
`sample_passages.jsonl` (#3) predates them; rebuild it with notebook 02 so the sample matches the full file.
