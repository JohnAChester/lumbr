"""paths.git_commit(): the commit every run's meta.json records."""

import subprocess

from lumbr import paths


def _head() -> str:
    return subprocess.run(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_git_commit_is_repo_head_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # outside the repo, like the demo backend
    assert paths.git_commit() == _head()


def test_git_commit_unknown_without_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", no_git)
    assert paths.git_commit() == "unknown"


def test_git_commit_unknown_when_git_fails(monkeypatch):
    def fails(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")
    monkeypatch.setattr(subprocess, "run", fails)
    assert paths.git_commit() == "unknown"
