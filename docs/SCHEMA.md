# Passage schema (v1)

The contract between chunking (John, #3/#4), retrieval and eval (Patricia, #7/#8/#10), and synthetic
questions and fine-tuning (Jieun, #9/#11/#15). Code: [`src/lumbr/schema.py`](../src/lumbr/schema.py).
**Sign-off = an approved PR from both Patricia and Jieun (deadline Nov 11).** Any later change needs a new PR that both approve.

## Format

One JSON object per line (`.jsonl`, UTF-8). Files on Drive:

| file | contents |
|---|---|
| `lumbr-data/processed/sample_passages.jsonl` | **v0 sample (#3)**: 1,000 docs, ~4.6K passages, 12 MB. Includes every gold doc of 50 questions (listed in `sample_manifest.json`). |
| `lumbr-data/processed/passages.jsonl` | full subsample (#4): ~140K docs, ~640K passages |

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

Use `body` in place of `text` for the **with/without-header ablation** (D5). It also gives a BM25 number comparable to the
paper's, whose baselines had no metadata.

## How documents become passages

1. **Units.** Each document's `content_field_names` are rendered in order into natural units:
   - Slack: one unit per message.
   - Gmail: one unit per email, with quoted replies removed.
   - Fireflies: one unit per transcript turn.
   - Lists (comments, acceptance criteria, …): one unit per item.
   - Prose: paragraphs, with headings kept attached.
   - Non-body fields get a label, e.g. `Acceptance criteria:`.
2. **Packing.** Consecutive units are packed greedily toward about 350 tokens and never past 512. An oversized single unit is
   split at paragraphs, then lines, then sentences, then words.
3. **Cleaning.** Literal `\n` escapes are unescaped, titles are cut to 150 chars, and implausible dates become null. Details in [SOURCES.md](SOURCES.md).
4. **Never** lowercase, stem, dedupe or expand acronyms.
5. The 4 doc IDs shared by two files are chunked as one document (chunks numbered continuously).

Planned for #4 (no schema change): context carry of ≤ 60 tokens for conversational sources, Gmail quote stubs,
and repeating a list field's label when its items continue into the next chunk.
