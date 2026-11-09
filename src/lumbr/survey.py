"""Per-source surveys of the raw JSON (issue #2): which keys exist, how often, what they look like.

Used by notebooks/01_john_inspect_sources.ipynb to produce docs/SOURCES.md.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from lumbr import sources

# Keys that every document has and that carry no content (bookkeeping from the generator).
BOOKKEEPING = {"dataset_doc_uuid", "title_field_name", "content_field_names", "original_location", "_source_type"}


def sample_by_source(n_per_source: int = 2000, seed: int = 42) -> dict[str, list[dict]]:
    """Reservoir-sample up to n documents per source in one pass over the parquet."""
    rng = random.Random(seed)
    res: dict[str, list[dict]] = defaultdict(list)
    seen: Counter[str] = Counter()
    for doc in sources.iter_docs():
        s = doc["_source_type"]
        seen[s] += 1
        if len(res[s]) < n_per_source:
            res[s].append(doc)
        else:
            j = rng.randrange(seen[s])
            if j < n_per_source:
                res[s][j] = doc
    return dict(res)


def _kind(v) -> str:
    if isinstance(v, list):
        inner = {type(x).__name__ for x in v[:5]}
        return f"list[{'|'.join(sorted(inner)) or '?'}]"
    return type(v).__name__


def _example(v, width: int = 70) -> str:
    if isinstance(v, list):
        v = v[0] if v else ""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " ".join(s.split())
    return s[:width] + ("…" if len(s) > width else "")


def key_table(docs: list[dict]) -> list[dict]:
    """One row per key: fill rate, type, whether it's a content field, and an example value."""
    n = len(docs)
    fill: Counter[str] = Counter()
    content: Counter[str] = Counter()
    kinds: dict[str, Counter] = defaultdict(Counter)
    example: dict[str, str] = {}
    for d in docs:
        cf = set(d.get("content_field_names", []))
        for k, v in d.items():
            if k in BOOKKEEPING or v in (None, "", []):
                continue
            fill[k] += 1
            kinds[k][_kind(v)] += 1
            if k in cf:
                content[k] += 1
            example.setdefault(k, _example(v))
    rows = []
    for k, c in fill.most_common():
        rows.append({
            "key": k,
            "fill": c / n,
            "type": kinds[k].most_common(1)[0][0],
            "content": content[k] / c if c else 0.0,
            "example": example[k],
        })
    return rows


def content_variants(docs: list[dict], top: int = 5) -> list[tuple[tuple[str, ...], int]]:
    return Counter(tuple(d.get("content_field_names", [])) for d in docs).most_common(top)


def title_fields(docs: list[dict]) -> Counter:
    return Counter(d.get("title_field_name") for d in docs)


def markdown_report(samples: dict[str, list[dict]], counts: dict[str, int] | None = None,
                    min_fill: float = 0.05) -> str:
    """Render the survey as Markdown (pasted into docs/SOURCES.md)."""
    out = []
    for s in sources.SOURCE_TYPES:
        docs = samples.get(s, [])
        if not docs:
            continue
        total = f"{counts[s]:,} docs; " if counts else ""
        out.append(f"### {s}\n\n{total}surveyed {len(docs):,}. "
                   f"Title field: {dict(title_fields(docs))}.\n")
        out.append("Content-field variants (most common):\n")
        for v, c in content_variants(docs):
            out.append(f"- {c / len(docs):.0%} `{list(v)}`")
        out.append("\n| key | fill | type | content? | example |\n|---|---|---|---|---|")
        for r in key_table(docs):
            if r["fill"] < min_fill:
                continue
            ex = r["example"].replace("|", "\\|")
            out.append(f"| `{r['key']}` | {r['fill']:.0%} | {r['type']} | "
                       f"{'yes' if r['content'] > 0.5 else ('some' if r['content'] else '')} | {ex} |")
        out.append("")
    return "\n".join(out)
