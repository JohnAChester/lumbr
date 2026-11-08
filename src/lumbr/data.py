"""Download and load the EnterpriseRAG-Bench dataset.

The raw parquet files are downloaded once from Hugging Face at a pinned revision,
kept on Drive (paths.RAW) as the team's canonical copy, and copied to local disk
(paths.LOCAL_RAW) each session for fast reads.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from lumbr import paths

HF_REPO = "onyx-dot-app/EnterpriseRAG-Bench"
# Pinned so all three of us use the exact same data even if upstream changes.
HF_REVISION = "69916e31c68aa5963c00248fd7f0bc12d04fd235"

FILES = {
    "documents": "data/documents/test.parquet",  # ~1.41 GB, 512K docs
    "questions": "data/questions/test.parquet",  # ~0.4 MB, 500 questions
}


def _local_name(key: str) -> str:
    return f"{key}.parquet"


def download_raw(force: bool = False) -> dict[str, Path]:
    """Ensure both parquet files exist on Drive (paths.RAW). Returns their Drive paths.

    Downloads to local disk first, then copies to Drive. Writing large files to the
    Drive mount in one sequential copy is more reliable than downloading straight into it.
    """
    from huggingface_hub import hf_hub_download

    paths.ensure_dirs()
    out = {}
    for key, remote in FILES.items():
        dst = paths.RAW / _local_name(key)
        if dst.exists() and not force:
            print(f"[ok] {key}: already on Drive ({dst.stat().st_size / 1e9:.2f} GB)")
            out[key] = dst
            continue
        print(f"[download] {key}: downloading {remote} @ {HF_REVISION[:8]} ...")
        tmp = hf_hub_download(
            repo_id=HF_REPO,
            filename=remote,
            repo_type="dataset",
            revision=HF_REVISION,
        )
        print(f"  copying to Drive -> {dst}")
        shutil.copyfile(tmp, dst)
        out[key] = dst
    return out


def stage_local(force: bool = False) -> dict[str, Path]:
    """Copy raw parquet files from Drive to local disk for fast reads this session."""
    out = {}
    for key in FILES:
        src = paths.RAW / _local_name(key)
        dst = paths.LOCAL_RAW / _local_name(key)
        if not src.exists():
            raise FileNotFoundError(f"{src} missing; run data.download_raw() first.")
        if force or not dst.exists() or dst.stat().st_size != src.stat().st_size:
            print(f"[stage] staging {key} to local disk ...")
            shutil.copyfile(src, dst)
        out[key] = dst
    return out


def load_questions():
    """All 500 benchmark questions as a pandas DataFrame."""
    import pandas as pd

    return pd.read_parquet(stage_local()["questions"])


def load_documents(columns: list[str] | None = None):
    """Benchmark documents as a pandas DataFrame.

    Columns: doc_id, source_type, title, content. Pass `columns` to load only what you
    need. `content` is by far the largest column.
    """
    import pandas as pd

    return pd.read_parquet(stage_local()["documents"], columns=columns)
