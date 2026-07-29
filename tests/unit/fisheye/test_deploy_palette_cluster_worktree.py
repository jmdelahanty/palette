from __future__ import annotations

from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "deploy_palette_cluster_worktree.sh"
)
BRANCH = "agent/palette/cluster-deploy-test"


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _configure_identity(repository: Path) -> None:
    _git("config", "user.name", "Palette Tests", cwd=repository)
    _git("config", "user.email", "palette-tests@example.invalid", cwd=repository)


def _build_fixture(tmp_path: Path) -> dict[str, Path | str]:
    remote = tmp_path / "remote.git"
    source_main = tmp_path / "source-main"
    source_worktree = tmp_path / "source-linked-worktree"
    groups_repo = tmp_path / "groups" / "palette"
    deploy_root = tmp_path / "groups" / "palette-worktrees"
    ssh_key = tmp_path / "dummy-key"

    _git("init", "--bare", "-q", str(remote))
    _git("init", "-q", str(source_main))
    _configure_identity(source_main)
    scripts_dir = source_main / "scripts"
    scripts_dir.mkdir()
    scripts_py = scripts_dir / "py"
    scripts_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    scripts_py.chmod(0o755)
    (source_main / "README.md").write_text("fixture\n", encoding="utf-8")
    _git("add", ".", cwd=source_main)
    _git("commit", "-qm", "fixture base", cwd=source_main)
    _git("branch", "-M", "sun", cwd=source_main)
    _git("remote", "add", "origin", str(remote), cwd=source_main)
    _git("push", "-q", "-u", "origin", "sun", cwd=source_main)

    _git(
        "worktree",
        "add",
        "-q",
        "-b",
        BRANCH,
        str(source_worktree),
        cwd=source_main,
    )
    _configure_identity(source_worktree)
    (source_worktree / "feature.txt").write_text(
        "commit-pinned cluster deployment\n",
        encoding="utf-8",
    )
    _git("add", "feature.txt", cwd=source_worktree)
    _git("commit", "-qm", "add deployment fixture", cwd=source_worktree)
    commit = _git("rev-parse", "HEAD", cwd=source_worktree)

    groups_repo.parent.mkdir()
    _git("clone", "-q", "--branch", "sun", str(remote), str(groups_repo))
    ssh_key.write_text("test-only\n", encoding="utf-8")
    return {
        "remote": remote,
        "source_main": source_main,
        "source_worktree": source_worktree,
        "groups_repo": groups_repo,
        "deploy_root": deploy_root,
        "ssh_key": ssh_key,
        "commit": commit,
    }


def _helper_command(fixture: dict[str, Path | str]) -> list[str]:
    return [
        "bash",
        str(SCRIPT),
        "--source-repo",
        str(fixture["source_worktree"]),
        "--groups-repo",
        str(fixture["groups_repo"]),
        "--deploy-root",
        str(fixture["deploy_root"]),
        "--ssh-key",
        str(fixture["ssh_key"]),
        "--skip-host-verify",
    ]


def _last_output_value(output: str, key: str) -> str:
    prefix = f"{key}="
    values = [
        line[len(prefix) :] for line in output.splitlines() if line.startswith(prefix)
    ]
    assert values
    return values[-1]


def test_helper_deploys_linked_worktree_without_switching_shared_checkout(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    source_worktree = Path(fixture["source_worktree"])
    groups_repo = Path(fixture["groups_repo"])
    remote = Path(fixture["remote"])
    commit = str(fixture["commit"])
    assert (source_worktree / ".git").is_file()
    shared_branch_before = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=groups_repo)
    shared_head_before = _git("rev-parse", "HEAD", cwd=groups_repo)

    result = subprocess.run(
        _helper_command(fixture),
        check=True,
        text=True,
        capture_output=True,
    )

    deployment = Path(_last_output_value(result.stdout, "palette_repo"))
    assert _last_output_value(result.stdout, "status") == "deployed"
    assert deployment.name == f"cluster-deploy-test-{commit[:8]}"
    assert _git("rev-parse", "HEAD", cwd=deployment) == commit
    assert _git("status", "--porcelain", cwd=deployment) == ""
    assert _git("rev-parse", "--abbrev-ref", "HEAD", cwd=deployment) == "HEAD"
    assert _git("rev-parse", "--abbrev-ref", "HEAD", cwd=groups_repo) == (
        shared_branch_before
    )
    assert _git("rev-parse", "HEAD", cwd=groups_repo) == shared_head_before
    assert _git(f"--git-dir={remote}", "rev-parse", f"refs/heads/{BRANCH}") == commit
    deploy_git_dir = Path(_git("rev-parse", "--git-dir", cwd=deployment))
    assert (deploy_git_dir / "locked").is_file()

    repeated = subprocess.run(
        _helper_command(fixture),
        check=True,
        text=True,
        capture_output=True,
    )
    assert _last_output_value(repeated.stdout, "status") == "already_deployed"
    assert Path(_last_output_value(repeated.stdout, "palette_repo")) == deployment


def test_helper_dry_run_does_not_push_or_create_destination(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    command = [*_helper_command(fixture), "--dry-run"]

    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )

    deployment = Path(_last_output_value(result.stdout, "palette_repo"))
    assert _last_output_value(result.stdout, "status") == "planned"
    assert not deployment.exists()
    remote_ref = subprocess.run(
        [
            "git",
            f"--git-dir={fixture['remote']}",
            "rev-parse",
            "--verify",
            f"refs/heads/{BRANCH}",
        ],
        text=True,
        capture_output=True,
    )
    assert remote_ref.returncode != 0


def test_helper_rejects_dirty_source_worktree(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    source_worktree = Path(fixture["source_worktree"])
    (source_worktree / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")

    result = subprocess.run(
        _helper_command(fixture),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "source worktree must be clean" in result.stderr
    assert not Path(fixture["deploy_root"]).exists()
