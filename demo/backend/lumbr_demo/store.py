"""SQLite passage store: lookup by id, and a hit plus its neighbouring chunks (small-to-big, D16)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from lumbr import schema


def build_store(passages: Iterable[dict], path: Path) -> int:
    """Write passages to a new SQLite file at `path`. Returns the number of rows."""
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    with con:
        con.execute("CREATE TABLE passages (passage_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,"
                    " chunk_idx INTEGER NOT NULL, json TEXT NOT NULL)")
        con.executemany("INSERT INTO passages VALUES (?, ?, ?, ?)",
                        ((p["passage_id"], p["doc_id"], p["chunk_idx"], json.dumps(p)) for p in passages))
        con.execute("CREATE INDEX doc_chunk ON passages (doc_id, chunk_idx)")
    n = con.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    con.close()
    return n


class PassageStore:
    def __init__(self, path: Path):
        # Read-only; check_same_thread=False because the web server may call from worker threads.
        self._con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, check_same_thread=False)

    def count(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def get(self, ids: list[str]) -> dict[str, dict]:
        """Passages by id, in the order asked; unknown ids are omitted."""
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = dict(self._con.execute(f"SELECT passage_id, json FROM passages WHERE passage_id IN ({marks})", ids))
        return {i: json.loads(rows[i]) for i in ids if i in rows}

    def with_neighbours(self, hit_ids: list[str], neighbours: int = 1) -> dict[str, dict]:
        """The hits plus chunks chunk_idx ± neighbours of the same document. Unknown hits are skipped."""
        out: dict[str, dict] = {}
        for pid, hit in self.get(hit_ids).items():
            want = [schema.make_passage_id(hit["doc_id"], i)
                    for i in range(hit["chunk_idx"] - neighbours, hit["chunk_idx"] + neighbours + 1) if i >= 0]
            out.update(self.get([w for w in want if w not in out]))
        return out
