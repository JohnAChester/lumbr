# Team Setup: GitHub + Colab + Shared Storage

Do this once per person (about 10 minutes). After that, every notebook starts with the
same bootstrap cell, which keeps everyone on the same code and the same data.

**How the pieces fit together**

| What | Where it lives | Why |
|---|---|---|
| Code (`src/lumbr/`, notebooks) | GitHub: `JohnAChester/lumbr` | Version control and one source of truth |
| Data, indexes, checkpoints | Google Drive: shared folder `lumbr-data/` | Survives Colab sessions; the whole team can read it |
| Model weights (Llama, Qwen, BGE, ...) | Hugging Face Hub, downloaded fresh each session | Fast to download; too big for Drive |
| Working copies during a session | Colab local disk (`/content/work`) | Much faster than reading from Drive |

---

## 1. GitHub access from Colab (token)

The repo is public, so *cloning and pulling* need no token. You need a token only to
**push** from Colab.

### 1a. Create a token (each person, on their own GitHub account)

1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)**
   → **Generate new token (classic)**.
2. Note: `lumbr-colab`. Expiration: **Dec 31, 2026**.
3. Scope: check **`public_repo`** only.
4. Copy the token (`ghp_...`). You won't see it again.

> Use a **classic** token. Fine-grained tokens can't reach a repo owned by *another
> person's* personal account, so Patricia's and Jieun's fine-grained tokens would not work
> on `JohnAChester/lumbr`. Classic tokens with `public_repo` scope work for every collaborator.

### 1b. Store it in Colab Secrets

In any Colab notebook, click the **🔑 key icon** in the left sidebar and add these secrets.
Turn on **"Notebook access"** for each one.

| Name | Value |
|---|---|
| `GH_TOKEN` | your `ghp_...` token |
| `GH_USER` | your GitHub username |
| `GH_EMAIL` | the email on your GitHub account |
| `HF_TOKEN` | a Hugging Face **read** token (huggingface.co → Settings → Access Tokens) |

Secrets belong to your Google account, not to a notebook, so you only set them once.
**Never paste a token into a cell.** Notebooks get committed, and the token would end up
on public GitHub.

`HF_TOKEN` is needed for gated models. **Llama 3.1 8B Instruct** requires clicking "Agree
and access" on its Hugging Face model page first (approval is usually quick).

### 1c. Saving work back to GitHub

There are two ways, depending on what you changed:

- **Notebooks:** Colab menu **File → Save a copy in GitHub** → repo `JohnAChester/lumbr`,
  branch `main`, same path. Before saving, run **Edit → Clear all outputs**. Outputs make
  diffs huge and cause merge conflicts.
- **`.py` files edited in Colab** (in `/content/lumbr`): run from a cell:
  ```python
  !cd /content/lumbr && git add -A && git commit -m "Describe the change" && git push
  ```
  The bootstrap cell has already set up your credentials, so `git push` just works.

**The Colab VM is wiped when the session ends.** Anything not pushed (code) or written to
Drive (data) is lost.

### Team git rules

- **One owner per notebook.** Name them `notebooks/NN_<track>_<topic>.ipynb`. Don't edit
  someone else's notebook. Copy it instead.
- Shared logic goes in `src/lumbr/` as `.py` files, not copy-pasted between notebooks.
  Notebooks import it.
- `git pull` before you start. The bootstrap cell does this for you.
- Never commit data, model weights, or tokens. `.gitignore` already blocks data files.

---

## 2. Shared storage on Google Drive

John's Drive (40 GB) holds the team's data in one folder. **Drive quota is per person,
not pooled**: a file counts against the quota of whoever *created* it, even inside a
folder someone else shared.

### 2a. John: create and share the folder (once)

1. In Drive, create the folder **`lumbr-data`** at the top level of My Drive.
2. Right-click → **Share** → add Patricia and Jieun as **Editor**.

### 2b. Patricia and Jieun: add a shortcut (once)

1. Open **Shared with me** in Drive.
2. Right-click `lumbr-data` → **Organize → Add shortcut** → choose **My Drive**.

After that, the folder appears at the **same path for all three of us** in Colab:

```
/content/drive/MyDrive/lumbr-data
```

That's what makes the shared code work. `src/lumbr/paths.py` uses this one path.

> If your school Google account offers **Shared Drives**, a Shared Drive is even better:
> files there count against no one's personal quota. In that case, change `DRIVE_ROOT` in
> `src/lumbr/paths.py` (or set the `LUMBR_DRIVE` environment variable) to its mount path.

### 2c. Folder layout

```
lumbr-data/
  raw/          # EnterpriseRAG-Bench parquet files (pinned revision), never edited
  processed/    # chunked passages, dev/test split, subsample ID lists
  indexes/      # BM25 (Pyserini) and FAISS indexes, one subfolder per setup (r1 ... r6)
  synth/        # synthetic Q&A pairs + hard negatives for R4 fine-tuning
  models/       # fine-tuned R4 checkpoints (only ours, never base models)
  runs/         # evaluation outputs: retrieved lists, generated answers, judge scores
```

Expected total: **about 10–15 GB**, well under 40 GB.

### 2d. Rules for Drive

- **Copy to local disk, then work.** Reading thousands of small files or a large index
  directly from `/content/drive` is very slow. Copy what you need into `/content/work` at the
  start of a session and write *finished* outputs back to Drive. `lumbr.data` helpers do
  this for the raw data.
- **Write once, don't overwrite.** Put each run in its own dated subfolder, e.g.
  `runs/r2_bge-small_2026-11-13/`, so two people never write to the same file.
- **Only the owner of a track writes to its folder.** Patricia: `indexes/`. Jieun:
  `synth/` and `models/`. John: `processed/` and `runs/` for generator runs.

---

## 3. Don't store LLMs or base models in Drive

Base model weights (**Llama 3.1 8B, Qwen2.5 7B / Qwen3-8B, Qwen3-Embedding-4B,
bge-reranker-v2-m3, BGE/E5**) are **downloaded from Hugging Face each session**:

- Together they're **about 50 GB**, more than the whole Drive.
- Downloading from Hugging Face on Colab is fast, often faster than reading the same file
  from the Drive mount.
- They go into the Hugging Face cache on the Colab local disk (`/content/hf_cache`). The
  bootstrap cell sets `HF_HOME` so they never land on Drive by accident.

**What *does* go to Drive:** only things we created that would be expensive to rebuild:
processed passages, embeddings/indexes, synthetic training data, and our fine-tuned R4
checkpoints (small, about 0.1–0.5 GB each).

---

## 4. Opening cells for every notebook

Copy these two cells to the top of **every** notebook, unchanged.
`notebooks/00_template.ipynb` already has them, so you can also start from a copy of it.

### Cell 1: bootstrap (mount Drive, sync repo, git credentials, install)

```python
# === Lumbr bootstrap: run first, every session. Do not edit. ===
import os, sys, subprocess
from pathlib import Path
from google.colab import drive, userdata

drive.mount("/content/drive")

REPO_DIR = Path("/content/lumbr")
REPO_URL = "https://github.com/JohnAChester/lumbr.git"

# Git credentials come from Colab Secrets and are written to the VM's home dir (wiped at session end).
gh_user, gh_token = userdata.get("GH_USER"), userdata.get("GH_TOKEN")
cred_file = Path.home() / ".git-credentials"
cred_file.write_text(f"https://{gh_user}:{gh_token}@github.com\n")
cred_file.chmod(0o600)
for k, v in [("credential.helper", "store"),
             ("user.name", gh_user),
             ("user.email", userdata.get("GH_EMAIL")),
             ("pull.ff", "only")]:
    subprocess.run(["git", "config", "--global", k, v], check=True)

# Clone on first run; otherwise pull the latest main.
if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)

# Model weights go to local disk, never Drive. HF_TOKEN unlocks gated models (Llama).
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# Install project deps and make `import lumbr` work without a runtime restart.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR / "src"))
```

### Cell 2: project paths + auto-reload

```python
%load_ext autoreload
%autoreload 2          # picks up changes to src/lumbr/*.py without restarting

from lumbr import paths, data
paths.ensure_dirs()
paths.show()
```

`autoreload` means that after you `git pull` a teammate's change to `src/lumbr/`, your
next cell uses the new code.

**If `git pull` fails** with "Not possible to fast-forward": you have uncommitted edits in
`/content/lumbr`. Either commit and push them, or discard them with
`!git -C /content/lumbr reset --hard origin/main`.

---

## 5. Checklist

- [ ] Classic GitHub token with `public_repo` created
- [ ] Colab Secrets set: `GH_TOKEN`, `GH_USER`, `GH_EMAIL`, `HF_TOKEN` (notebook access on)
- [ ] Accepted the Llama 3.1 license on Hugging Face
- [ ] `lumbr-data` visible at `/content/drive/MyDrive/lumbr-data` (John: created + shared; others: shortcut added)
- [ ] Ran `notebooks/00_setup_and_download.ipynb` once: bootstrap succeeds and `paths.show()` prints `[ok]` on every line
