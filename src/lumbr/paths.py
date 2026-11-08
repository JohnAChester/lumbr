"""Shared filesystem layout for Lumbr.

Every notebook and script gets paths from here, so the whole team reads and writes
the same locations. On Colab:

- DRIVE_ROOT: persistent, shared team storage (Google Drive folder `lumbr-data`)
- LOCAL_ROOT: fast scratch on the Colab VM's local disk (wiped at session end)

Off Colab (e.g. a laptop), both default to `<repo>/data/...`. Override either one with
the LUMBR_DRIVE / LUMBR_LOCAL environment variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

REPO_ROOT = Path(__file__).resolve().parents[2]

if IN_COLAB:
    _default_drive = Path("/content/drive/MyDrive/lumbr-data")
    _default_local = Path("/content/work")
else:
    _default_drive = REPO_ROOT / "data" / "drive"
    _default_local = REPO_ROOT / "data" / "local"

DRIVE_ROOT = Path(os.environ.get("LUMBR_DRIVE", _default_drive))
LOCAL_ROOT = Path(os.environ.get("LUMBR_LOCAL", _default_local))

# Persistent team storage (Drive)
RAW = DRIVE_ROOT / "raw"              # pinned EnterpriseRAG-Bench parquet files
PROCESSED = DRIVE_ROOT / "processed"  # passages, splits, subsample ID lists
INDEXES = DRIVE_ROOT / "indexes"      # BM25 / FAISS indexes per setup (r1..r6)
SYNTH = DRIVE_ROOT / "synth"          # synthetic Q&A + hard negatives for R4
MODELS = DRIVE_ROOT / "models"        # our fine-tuned checkpoints only
RUNS = DRIVE_ROOT / "runs"            # eval outputs, one dated subfolder per run

# Fast local scratch (Colab VM disk)
LOCAL_RAW = LOCAL_ROOT / "raw"

DRIVE_DIRS = [RAW, PROCESSED, INDEXES, SYNTH, MODELS, RUNS]
LOCAL_DIRS = [LOCAL_ROOT, LOCAL_RAW]


def ensure_dirs() -> None:
    """Create the standard folders if missing. Safe to call every session."""
    if IN_COLAB and not DRIVE_ROOT.parent.exists():
        raise RuntimeError(
            f"{DRIVE_ROOT.parent} not found. Mount Drive first (bootstrap cell)."
        )
    if IN_COLAB and not DRIVE_ROOT.exists():
        raise RuntimeError(
            f"{DRIVE_ROOT} not found. John: create the 'lumbr-data' folder in My Drive. "
            "Teammates: add a shortcut to it in My Drive (see docs/COLAB_SETUP.md)."
        )
    for d in DRIVE_DIRS + LOCAL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def show() -> None:
    """Print the resolved paths and whether each one exists."""
    print(f"IN_COLAB   = {IN_COLAB}")
    print(f"HF_HOME    = {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    for name in ["DRIVE_ROOT", "RAW", "PROCESSED", "INDEXES", "SYNTH", "MODELS", "RUNS",
                 "LOCAL_ROOT", "LOCAL_RAW"]:
        p = globals()[name]
        print(f"{'[ok]' if p.exists() else '[--]'} {name:<10} {p}")
