"""Turn structured source JSON into passages that follow the schema (issues #3, #4).

Algorithm (docs/SCHEMA.md has the full rules):
  1. render_units(): split each document into natural units (a Slack message, an email,
     a comment, a markdown section/paragraph, a transcript turn), in content-field order.
  2. pack(): greedily pack consecutive units into chunks of about TARGET_TOKENS. A unit that
     fits within the cap is never split; an oversized unit is split at paragraphs, then
     sentences, then words.
  3. Prepend the metadata header. The whole `text` is <= MAX_TOKENS including special tokens,
     for BOTH R2's tokenizer (bge-base) and R3's (Qwen3-Embedding-4B), so neither truncates.

Packing sums per-unit counts (conservative: max of both tokenizers + 1 per join); every
finished chunk is then re-counted exactly, and re-packed tighter in the rare case it is over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterable, Iterator

from lumbr import schema, sources

TARGET_TOKENS = 350   # pack up to about this many tokens (including header)
TAIL_MIN = 50         # a final chunk smaller than this is merged into the previous one
TITLE_MAX_CHARS = 150 # titles are cut to this length (see short_title)
PLAUSIBLE_YEARS = (2020, 2029)  # the company's timeline; dates outside this are generator noise
_EPOCH_MIN = datetime(PLAUSIBLE_YEARS[0], 1, 1, tzinfo=timezone.utc).timestamp()
_EPOCH_MAX = datetime(PLAUSIBLE_YEARS[1], 12, 31, tzinfo=timezone.utc).timestamp()
SPECIAL = 2           # most special tokens either tokenizer adds ([CLS] + [SEP])

# Fields whose text is the document body itself: rendered without a label.
MAIN_FIELDS = {"description", "body", "content", "text", "messages", "thread", "thread_text",
               "transcript", "emails", "content_continued"}

SPEAKER = re.compile(r"^\s*(\[[\d:]+\]\s*)?[A-Za-z@][\w.'’()\-@ ]{0,40}?:\s")
EMAIL_START = re.compile(r"\n(?=From: )")
# "-----MESSAGE 1-----", "--- Message 1 of 4 ---"
MESSAGE_MARKER = re.compile(r"^\s*-{3,}\s*MESSAGE\s+\d+[^\n]{0,20}?-{3,}\s*$", re.M | re.I)
QUOTE_INTRO = re.compile(r"^\s*(On .{5,200}wrote:|-{2,}\s*Original Message\s*-{2,})\s*$", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# (channel, author, date, created) keys per source, in fallback order. See docs/SOURCES.md.
META_KEYS = {
    "slack":        (["channel"], ["participants"], ["last_message_ts", "thread_ts"], ["first_message_ts", "thread_ts"]),
    "gmail":        (["mailbox_owner"], [], ["last_email_at"], ["first_email_at"]),
    "linear":       (["project", "team"], ["creator"], ["updated_at", "created_at"], ["created_at"]),
    "jira":         (["project"], ["reporter"], ["updated_at", "created_at"], ["created_at"]),
    "hubspot":      (["stage"], ["owner"], ["updated_at", "last_activity_at"], ["created_at"]),
    "google_drive": (["team", "drive_area"], ["owner"], ["last_modified", "created_at"], ["created_at"]),
    "fireflies":    (["call_type"], ["redwood_owner"], ["recorded_at"], ["recorded_at"]),
    "github":       (["repo"], ["author"], ["updated_at", "merged_at", "created_at"], ["created_at"]),
    "confluence":   (["space"], ["author"], ["last_updated", "created_at"], ["created_at"]),
}


# ---------------------------------------------------------------- tokenizer

@lru_cache(maxsize=None)
def tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def _raw_counts(texts: list[str]) -> list[list[int]]:
    """Per tokenizer in schema.TOKENIZERS, token counts without special tokens."""
    return [[len(ids) for ids in tokenizer(m)(texts, add_special_tokens=False)["input_ids"]]
            for m in schema.TOKENIZERS]


def count(texts: list[str]) -> list[int]:
    """Conservative packing count (no special tokens): the max over R2's and R3's tokenizers,
    +1 for the newline that joins units. Qwen splits numbers into single digits, so it is
    the larger count for about 1% of passages (ID-, timestamp- and table-heavy text)."""
    if not texts:
        return []
    return [max(c) + 1 for c in zip(*_raw_counts(texts))]


def exact_counts(texts: list[str]) -> list[int]:
    """Exact `n_tokens`: max over both tokenizers of the full count INCLUDING special tokens."""
    raw = _raw_counts(texts)
    return [max(c + schema.TOKENIZERS[m] for c, m in zip(cs, schema.TOKENIZERS)) for cs in zip(*raw)]


# ---------------------------------------------------------------- metadata

def _first(doc: dict, keys: list[str]):
    for k in keys:
        v = doc.get(k)
        if isinstance(v, list):
            v = v[0] if v else None
        if v not in (None, ""):
            return str(v).strip()
    return None


def normalize_date(v: str | None) -> str | None:
    """Epoch seconds/ms -> ISO-8601 UTC; ISO strings and YYYY-MM-DD pass through unchanged.

    About half of Slack threads carry made-up epochs (years 2038+). Dates outside
    PLAUSIBLE_YEARS become null rather than showing a wrong date in the header.
    """
    if not v:
        return None
    if v.isdigit() and len(v) in (10, 13):
        ts = int(v) / (1000 if len(v) == 13 else 1)
        if not (_EPOCH_MIN <= ts <= _EPOCH_MAX):
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    m = re.match(r"(\d{4})-\d{2}-\d{2}", v)
    if m and not (PLAUSIBLE_YEARS[0] <= int(m.group(1)) <= PLAUSIBLE_YEARS[1]):
        return None
    return v


ESC_NEWLINE = chr(92) + "n"   # the two characters backslash + n
ESC_TAB = chr(92) + "t"
ESC_QUOTE = chr(92) + '"'


def clean_text(s: str) -> str:
    """Undo literal escape sequences: 2-30% of docs (by source) contain backslash-n instead of newlines."""
    if s.count(ESC_NEWLINE) >= 2:
        s = s.replace(ESC_NEWLINE, "\n").replace(ESC_TAB, "\t").replace(ESC_QUOTE, '"')
    return s


def short_title(title: str, max_chars: int = TITLE_MAX_CHARS) -> str:
    """First line of the title, cut at a word boundary. A few docs point `title_field_name`
    at a long text field (titles up to ~8.7K chars), which would swamp every chunk's header."""
    t = title.strip().split("\n", 1)[0].strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-—") + "…"


def metadata(doc: dict, source: str) -> dict:
    ch, au, dt, cr = META_KEYS[source]
    return {
        "title": short_title(sources.title_of(doc)),
        "channel": _first(doc, ch),
        "author": _first(doc, au),
        "date": normalize_date(_first(doc, dt)),
        "created": normalize_date(_first(doc, cr)),
    }


# ---------------------------------------------------------------- units

@dataclass
class Unit:
    text: str
    n: int = 0          # tokens without specials
    speaker: str | None = None


def _humanize(field: str) -> str:
    return field.replace("_", " ").strip().capitalize()


def split_speaker_lines(text: str) -> list[str]:
    """A new unit starts at each `Name: ...` / `[mm:ss] Name: ...` line; other lines continue it."""
    units, cur = [], []
    for line in text.split("\n"):
        if SPEAKER.match(line) and cur:
            units.append("\n".join(cur).strip())
            cur = []
        cur.append(line)
    if cur:
        units.append("\n".join(cur).strip())
    return [u for u in units if u]


def split_blocks(text: str) -> list[str]:
    """Paragraphs separated by blank lines; a heading-only block is attached to the block after it."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: list[str] = []
    pending = ""
    for b in blocks:
        if re.fullmatch(r"#{1,6}\s.*", b):  # a lone heading line
            pending = f"{pending}\n{b}".strip()
            continue
        out.append(f"{pending}\n{b}".strip() if pending else b)
        pending = ""
    if pending:
        out.append(pending)
    return out


def strip_quoted_reply(email: str) -> str:
    """Remove the quoted copy of earlier emails (`On ... wrote:` / `Original Message` and `>` lines).

    Forwarded messages are kept: their content does not appear elsewhere in the thread.
    """
    kept = []
    for line in email.split("\n"):
        if QUOTE_INTRO.match(line):
            break  # everything after the quote intro is the quoted copy
        if line.lstrip().startswith(">"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _emails(value) -> list[str]:
    """One unit per email. Handles a list of emails, or one string split on `From:` / `-----MESSAGE n-----`."""
    items = value if isinstance(value, list) else EMAIL_START.split(value)
    out = []
    for x in items:
        for e in MESSAGE_MARKER.split(clean_text(str(x))):
            e = strip_quoted_reply(e)
            if e:
                out.append(e)
    return out


def render_units(doc: dict, source: str) -> list[str]:
    """The document's body as an ordered list of natural units (plain strings)."""
    units: list[str] = []
    for field, value in sources.content_fields(doc):
        if source == "gmail" and field in MAIN_FIELDS:
            parts = _emails(value)
        elif source == "slack" or (source == "fireflies" and field == "transcript"):
            parts = split_speaker_lines(clean_text(value if isinstance(value, str) else "\n".join(map(str, value))))
        elif isinstance(value, list):
            parts = [f"- {clean_text(x)}".strip() for x in map(str, value) if str(x).strip()]
        elif isinstance(value, dict):
            parts = [f"{k}: {clean_text(str(v))}" for k, v in value.items()]
        else:
            parts = split_blocks(clean_text(str(value)))
        if parts and field not in MAIN_FIELDS:
            parts[0] = f"{_humanize(field)}:\n{parts[0]}"
        units.extend(parts)
    return units


# ---------------------------------------------------------------- packing

def _split_oversized(text: str, n: int, budget: int) -> list[Unit]:
    """Split one unit that is over budget: paragraphs -> lines -> sentences -> words."""
    for sep_re, joiner in ((r"\n\s*\n", "\n\n"), (r"\n", "\n"), (SENTENCE_END.pattern, " "), (r"\s+", " ")):
        pieces = [p for p in re.split(sep_re, text) if p.strip()]
        if len(pieces) < 2:
            continue
        counts = count(pieces)
        out: list[Unit] = []
        cur, cur_n = [], 0
        for p, c in zip(pieces, counts):
            if c > budget:  # still too big on its own: recurse at a finer level
                if cur:
                    out.append(Unit(joiner.join(cur), cur_n))
                    cur, cur_n = [], 0
                out.extend(_split_oversized(p, c, budget))
                continue
            if cur and cur_n + c > budget:
                out.append(Unit(joiner.join(cur), cur_n))
                cur, cur_n = [], 0
            cur.append(p)
            cur_n += c
        if cur:
            out.append(Unit(joiner.join(cur), cur_n))
        return out
    # a single unbreakable token run (e.g. a giant URL): hard-cut by characters
    if budget < 1 or len(text) < 2:
        raise ValueError(f"cannot split {text[:40]!r} into budget {budget}")
    half = len(text) // 2
    a, b = text[:half], text[half:]
    ca, cb = count([a, b])
    return _split_oversized(a, ca, budget) if ca > budget else [Unit(a, ca)] + (
        _split_oversized(b, cb, budget) if cb > budget else [Unit(b, cb)])


def pack(units: list[Unit], budget: int, target: int) -> list[list[Unit]]:
    """Greedy packing. Never exceeds `budget`; starts a new chunk once `target` would be passed."""
    flat: list[Unit] = []
    for u in units:
        flat.extend(_split_oversized(u.text, u.n, budget) if u.n > budget else [u])
    chunks: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_n = 0
    for u in flat:
        if cur and (cur_n + u.n > budget or (cur_n + u.n > target and cur_n >= TAIL_MIN)):
            chunks.append(cur)
            cur, cur_n = [], 0
        cur.append(u)
        cur_n += u.n
    if cur:
        if chunks and cur_n < TAIL_MIN and sum(x.n for x in chunks[-1]) + cur_n <= budget:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- documents -> passages

def chunk_document(doc: dict, source: str | None = None) -> list[dict]:
    """Chunk one JSON document. Returns schema dicts with chunk_idx/n_chunks/passage_id still unset."""
    source = source or doc["_source_type"]
    meta = metadata(doc, source)
    texts = render_units(doc, source)
    if not texts:
        return []
    if source == "slack":  # thread starter = first speaker
        m = SPEAKER.match(texts[0])
        if m:
            meta["author"] = texts[0][: m.end()].strip().rstrip(":").split("]")[-1].strip()
    elif source == "gmail":  # thread starter = first sender's display name
        m = re.search(r"^From:\s*\"?([^<\"\n]*?)\"?\s*(?:<([^>\n]+)>)?\s*$", texts[0], re.M)
        if m:
            name, addr = m.group(1).strip(), m.group(2)
            meta["author"] = (name if name and "@" not in name else (addr or name).split("@")[0]) or None
    header = schema.make_header(source, meta["title"], meta["channel"], meta["author"], meta["date"])
    n_header = count([header])[0]
    budget = schema.MAX_TOKENS - SPECIAL - n_header
    target = max(TAIL_MIN, TARGET_TOKENS - SPECIAL - n_header)
    units = [Unit(t, c) for t, c in zip(texts, count(texts))]

    def bodies(groups):
        return ["\n".join(u.text for u in g) for g in groups]

    groups = pack(units, budget, target)
    exact = exact_counts([schema.join_text(header, b) for b in bodies(groups)])
    # Packing counts are conservative but not exact for Qwen's BPE; re-pack any chunk that is over.
    for _ in range(5):
        over = [i for i, n in enumerate(exact) if n > schema.MAX_TOKENS]
        if not over:
            break
        new_groups, new_exact = [], []
        for i, (g, n) in enumerate(zip(groups, exact)):
            if i not in over:
                new_groups.append(g)
                new_exact.append(n)
                continue
            tighter = budget - (n - schema.MAX_TOKENS) - 8
            sub = pack(g, tighter, min(target, tighter))
            new_groups.extend(sub)
            new_exact.extend(exact_counts([schema.join_text(header, b) for b in bodies(sub)]))
        groups, exact = new_groups, new_exact
    out = []
    for body, n in zip(bodies(groups), exact):
        out.append({"doc_id": doc["dataset_doc_uuid"], "source": source, **meta, "n_tokens": n,
                    "header": header, "body": body, "text": schema.join_text(header, body)})
    return out


def finalize(doc_id: str, parts: list[dict]) -> list[schema.Passage]:
    """Number chunks (continuously across files that share a doc_id) and assert the token cap."""
    passages = []
    for i, p in enumerate(parts):
        if p["n_tokens"] > schema.MAX_TOKENS:
            raise AssertionError(f"{doc_id} chunk {i}: {p['n_tokens']} tokens > {schema.MAX_TOKENS}")
        passages.append(schema.Passage(
            passage_id=schema.make_passage_id(doc_id, i), chunk_idx=i, n_chunks=len(parts), **p))
    return passages


def chunk_docs(docs: Iterable[dict], duplicate_ids: set[str] = frozenset()) -> Iterator[schema.Passage]:
    """Chunk a stream of documents. Files that share a doc_id (see docs/SOURCES.md) are merged at the end."""
    held: dict[str, list[dict]] = {}
    for doc in docs:
        doc_id = doc["dataset_doc_uuid"]
        parts = chunk_document(doc)
        if doc_id in duplicate_ids:
            held.setdefault(doc_id, []).extend(parts)
            continue
        if parts:
            yield from finalize(doc_id, parts)
    for doc_id, parts in held.items():
        if parts:
            yield from finalize(doc_id, parts)
