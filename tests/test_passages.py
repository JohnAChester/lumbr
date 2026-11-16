"""Resumable sharding for the full passage build (#4), with a fake chunker (no tokenizers needed)."""

import pytest

from lumbr import passages, schema


def _fake_passages(doc_id, n):
    out = []
    for i in range(n):
        header = f"[slack] t{doc_id}"
        body = f"body {i}"
        out.append(schema.Passage(passage_id=schema.make_passage_id(doc_id, i), doc_id=doc_id,
                                  source="slack", title="t", channel=None, author=None, date=None,
                                  created=None, chunk_idx=i, n_chunks=n, n_tokens=5, header=header,
                                  body=body, text=schema.join_text(header, body)))
    return out


IDS = {f"dsid_{i:032x}" for i in range(23)}


def fake_chunk(todo, fail_after=None):
    for k, doc_id in enumerate(sorted(todo)):
        if fail_after is not None and k == fail_after:
            raise RuntimeError("runtime disconnected")
        yield from _fake_passages(doc_id, 1 + k % 3)


def _all_ids(shard_dir):
    return [d for s in sorted(shard_dir.glob("shard_*.jsonl")) for d in passages.read_ids(s.with_suffix(".ids"))]


def test_shards_cover_every_doc_once(tmp_path):
    n = passages.write_shards(IDS, tmp_path / "drive", tmp_path / "local", fake_chunk,
                              shard_docs=5, progress=False)
    assert n == 5  # 23 docs / 5 per shard
    assert sorted(_all_ids(tmp_path / "drive")) == sorted(IDS)
    # rerun is a no-op
    assert passages.write_shards(IDS, tmp_path / "drive", tmp_path / "local", fake_chunk,
                                 shard_docs=5, progress=False) == 0


def test_resume_after_interruption(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    with pytest.raises(RuntimeError):
        passages.write_shards(IDS, drive, local, lambda t: fake_chunk(t, fail_after=12),
                              shard_docs=5, progress=False)
    assert len(passages.done_ids(drive)) == 10  # two full shards survived; the partial one did not
    passages.write_shards(IDS, drive, local, fake_chunk, shard_docs=5, progress=False)
    ids = _all_ids(drive)
    assert sorted(ids) == sorted(IDS) and len(ids) == len(set(ids))

    out = tmp_path / "passages.jsonl"
    passages.merge_shards(drive, out)
    stats = schema.validate_jsonl(out)
    assert stats["docs"] == len(IDS)


def test_shard_without_sidecar_is_redone(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    passages.write_shards(IDS, drive, local, fake_chunk, shard_docs=5, progress=False)
    (drive / "shard_0004.ids").unlink()  # crash after the shard copy, before its sidecar
    passages.write_shards(IDS, drive, local, fake_chunk, shard_docs=5, progress=False)
    out = tmp_path / "passages.jsonl"
    passages.merge_shards(drive, out)
    assert schema.validate_jsonl(out)["docs"] == len(IDS)  # no duplicate passage_ids


def test_doc_passages_never_split_across_shards(tmp_path):
    passages.write_shards(IDS, tmp_path / "d", tmp_path / "l", fake_chunk, shard_docs=4, progress=False)
    for s in sorted((tmp_path / "d").glob("shard_*.jsonl")):
        ps = list(schema.read_jsonl(s))
        for doc_id in {p["doc_id"] for p in ps}:
            mine = [p for p in ps if p["doc_id"] == doc_id]
            assert len(mine) == mine[0]["n_chunks"]
