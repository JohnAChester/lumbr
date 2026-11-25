"""data.stage_local works in a fresh process that never called paths.ensure_dirs (e.g. a demo/build script)."""
from lumbr import data, paths


def test_stage_local_creates_the_local_folder(tmp_path, monkeypatch):
    raw, local = tmp_path / "drive" / "raw", tmp_path / "work" / "raw"  # local does not exist yet
    raw.mkdir(parents=True)
    for key in data.FILES:
        (raw / f"{key}.parquet").write_bytes(b"parquet-bytes-" + key.encode())
    monkeypatch.setattr(paths, "RAW", raw)
    monkeypatch.setattr(paths, "LOCAL_RAW", local)
    out = data.stage_local()
    for key in data.FILES:
        assert out[key] == local / f"{key}.parquet"
        assert out[key].read_bytes() == b"parquet-bytes-" + key.encode()
