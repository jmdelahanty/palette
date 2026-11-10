"""Behavior of scripts/git-hooks/pre-push against a throwaway repository.

The real census takes ~25 s, so the temp repository carries a tiny stand-in
``fisheye.diagnostics.zarr_storage_census`` with the same ``--check`` /
``--write`` / ``--repo-root`` interface. The stand-in logs each invocation so
the tests can assert when the hook skips the census entirely.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK = REPO_ROOT / "scripts" / "git-hooks" / "pre-push"
ZERO = "0" * 40

FAKE_CENSUS = '''
import argparse, os, sys
from pathlib import Path

OUT = Path("docs/diagnostics/zarr_fake_census.json")

def expected(root):
    files = sorted(p.relative_to(root).as_posix() for p in root.glob("src/**/*.py"))
    return "\\n".join(files) + "\\n"

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path)
parser.add_argument("--check", action="store_true")
parser.add_argument("--write", action="store_true")
args = parser.parse_args()
with open(os.environ["FAKE_CENSUS_LOG"], "a") as log:
    log.write("write\\n" if args.write else "check\\n")
target = args.repo_root / OUT
if args.write:
    target.write_text(expected(args.repo_root))
elif not target.is_file() or target.read_text() != expected(args.repo_root):
    raise SystemExit("stale Zarr census artifacts: " + OUT.as_posix())
'''

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    pkg = repo / "src" / "fisheye" / "diagnostics"
    pkg.mkdir(parents=True)
    (repo / "docs" / "diagnostics").mkdir(parents=True)
    for init in (repo / "src" / "fisheye", pkg):
        (init / "__init__.py").write_text("")
    (pkg / "zarr_storage_census.py").write_text(FAKE_CENSUS)
    py = repo / "scripts" / "py"
    py.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    (repo / "README.md").write_text("readme\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.hooksPath", "/dev/null")
    log = tmp_path / "census.log"
    log.write_text("")
    env = dict(os.environ, FAKE_CENSUS_LOG=str(log))
    env.pop("PALETTE_SKIP_PRE_PUSH", None)
    subprocess.run(
        [str(py), "-m", "fisheye.diagnostics.zarr_storage_census", "--repo-root",
         str(repo), "--write"],
        env={**env, "PYTHONPATH": str(repo / "src")}, check=True,
    )
    base = _commit(repo, "base with fresh census")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    _git(repo, "checkout", "-q", "-b", "feature")
    log.write_text("")
    return repo


def _run_hook(repo: Path, sha: str, *, skip: bool = False):
    env = dict(os.environ, FAKE_CENSUS_LOG=str(repo.parent / "census.log"))
    env.pop("PALETTE_SKIP_PRE_PUSH", None)
    if skip:
        env["PALETTE_SKIP_PRE_PUSH"] = "1"
    line = f"refs/heads/feature {sha} refs/heads/feature {ZERO}\n"
    return subprocess.run(
        ["bash", str(HOOK), "origin", "git@example.com:x.git"],
        cwd=repo, input=line, env=env, capture_output=True, text=True,
    )


def _census_calls(repo: Path) -> list[str]:
    return (repo.parent / "census.log").read_text().split()


def test_branch_deletion_is_skipped(repo: Path) -> None:
    result = subprocess.run(
        ["bash", str(HOOK), "origin", "url"], cwd=repo,
        input=f"(delete) {ZERO} refs/heads/feature {ZERO}\n",
        env=dict(os.environ, FAKE_CENSUS_LOG=str(repo.parent / "census.log")),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert _census_calls(repo) == []


def test_push_without_census_inputs_skips_the_census(repo: Path) -> None:
    (repo / "README.md").write_text("changed\n")
    (repo / "notes.txt").write_text("not python\n")
    sha = _commit(repo, "docs only")
    result = _run_hook(repo, sha)
    assert result.returncode == 0, result.stderr
    assert _census_calls(repo) == []


def test_fresh_census_passes(repo: Path) -> None:
    (repo / "src" / "fisheye" / "new.py").write_text("x = 1\n")
    subprocess.run(
        [str(repo / "scripts" / "py"), "-m", "fisheye.diagnostics.zarr_storage_census",
         "--repo-root", str(repo), "--write"],
        env=dict(os.environ, PYTHONPATH=str(repo / "src"),
                 FAKE_CENSUS_LOG=str(repo.parent / "census.log")),
        check=True,
    )
    sha = _commit(repo, "code plus regenerated census")
    (repo.parent / "census.log").write_text("")
    result = _run_hook(repo, sha)
    assert result.returncode == 0, result.stderr
    assert _census_calls(repo) == ["check"]


def test_stale_census_is_regenerated_and_push_refused(repo: Path) -> None:
    (repo / "src" / "fisheye" / "new.py").write_text("x = 1\n")
    sha = _commit(repo, "code without census")
    result = _run_hook(repo, sha)
    assert result.returncode == 1
    assert "docs/diagnostics/zarr_fake_census.json" in result.stderr
    assert "Commit these files and push again" in result.stderr
    assert _census_calls(repo) == ["check", "write"]
    # The hook never commits; the regenerated file is left for the pusher.
    assert _git(repo, "rev-parse", "HEAD") == sha
    assert "zarr_fake_census.json" in _git(repo, "status", "--porcelain")

    fixed = _commit(repo, "regenerate census")
    assert _run_hook(repo, fixed).returncode == 0


def test_stale_non_head_push_leaves_working_tree_alone(repo: Path) -> None:
    (repo / "src" / "fisheye" / "new.py").write_text("x = 1\n")
    stale = _commit(repo, "code without census")
    _git(repo, "checkout", "-q", "main")
    result = _run_hook(repo, stale)
    assert result.returncode == 1
    assert "not the checked-out HEAD" in result.stderr
    assert _git(repo, "status", "--porcelain") == ""


def test_escape_hatch_skips_everything(repo: Path) -> None:
    (repo / "src" / "fisheye" / "new.py").write_text("x = 1\n")
    sha = _commit(repo, "code without census")
    result = _run_hook(repo, sha, skip=True)
    assert result.returncode == 0
    assert _census_calls(repo) == []
