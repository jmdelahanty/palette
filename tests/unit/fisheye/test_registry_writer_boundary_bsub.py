from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[3]
CITRUS_SCRIPT = REPO / "scripts" / "submit_citrus_session_import_bsub.sh"
PROJECTION_SCRIPT = REPO / "scripts" / "submit_registry_zarr_projection_refresh_bsub.sh"


def _run_citrus(
    tmp_path: Path,
    *options: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PALETTE_REGISTRY_WRITER_HOST", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [
            "bash",
            str(CITRUS_SCRIPT),
            "--session-dir",
            str(tmp_path / "missing-session"),
            "--log-dir",
            str(tmp_path / "citrus-logs"),
            *(() if "--run-id" in options else ("--run-id", "writer-boundary")),
            "--dry-run",
            *options,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


@pytest.mark.parametrize(
    "flags", [["--register"], ["--registry", "/r.sqlite"], ["--writer-host", "writer01"]]
)
def test_citrus_job_mode_registration_is_retired(tmp_path: Path, flags) -> None:
    result = _run_citrus(tmp_path, *flags)

    assert result.returncode == 2
    assert "job-mode registration is retired" in result.stderr
    assert not (tmp_path / "citrus-logs").exists()


def test_citrus_job_never_touches_the_registry(tmp_path: Path) -> None:
    env = {
        "PALETTE_REGISTRY_WRITER_HOST": "writer01",
        "PALETTE_REGISTRY_WRITER_LOCK_PATH": str(tmp_path / "writer.lock"),
        "PALETTE_REGISTRY_SHADOW_TEMP_ROOT": str(tmp_path / "shadows"),
        "PALETTE_REGISTRY_SHADOW_BACKUP_DIR": str(tmp_path / "backups"),
        "PALETTE_REGISTRY": str(tmp_path / "registry.sqlite"),
    }
    for options in ([], ["--no-register"]):
        result = _run_citrus(tmp_path, "--run-id", f"run{len(options)}", *options, env_overrides=env)
        assert result.returncode == 0, result.stderr
        assert "hname==" not in result.stdout
        assert "registration=writer_host_only" in result.stdout
    for job_script in (tmp_path / "citrus-logs").glob("**/run_citrus_session_import.sh"):
        subprocess.run(["bash", "-n", str(job_script)], check=True)
        job = job_script.read_text(encoding="utf-8")
        assert "--register" not in job and "--registry" not in job
        assert "PALETTE_REGISTRY" not in job and "REGISTER" not in job


def test_projection_refresh_dry_run_renders_without_submission(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    registry.touch()
    zarr = tmp_path / "analysis.zarr"
    zarr.mkdir()
    (zarr / "zarr.json").write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            str(PROJECTION_SCRIPT),
            "--run-id",
            "projection-dry-run",
            "--zarr-path",
            str(zarr),
            "--registry",
            str(registry),
            "--palette-repo",
            str(REPO),
            "--source-repo",
            str(REPO),
            "--output-root",
            str(tmp_path / "projection-logs"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "mode=render-only" in result.stdout
    assert "operation=dry-run" in result.stdout
    job_script = (
        tmp_path
        / "projection-logs"
        / "projection-dry-run"
        / "run_registry_zarr_projection_refresh.sh"
    )
    subprocess.run(["bash", "-n", str(job_script)], check=True)
    job = job_script.read_text(encoding="utf-8")
    assert "cmd+=(--dry-run)" in job
    assert "OPERATION=dry-run" in job
    assert "BACKUP_STATUS=none" in job


def test_projection_refresh_apply_fails_closed(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(PROJECTION_SCRIPT),
            "--apply",
            "--run-id",
            "projection-apply-blocked",
            "--zarr-path",
            str(tmp_path / "analysis.zarr"),
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--palette-repo",
            str(REPO),
            "--source-repo",
            str(REPO),
            "--output-root",
            str(tmp_path / "projection-logs"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "--apply is disabled" in result.stderr
    assert not (tmp_path / "projection-logs").exists()


@pytest.mark.parametrize(
    "flag", ["--recording-type", "--recording-subtype", "--behavior-mode", "--recording-only"]
)
def test_citrus_launcher_refuses_operator_context_flags(tmp_path: Path, flag: str) -> None:
    # Recording context comes only from the producer's transfer snapshot.
    extra = [flag] if flag == "--recording-only" else [flag, "free"]
    result = _run_citrus(tmp_path, *extra)
    assert result.returncode == 2
    assert f"Unknown arg: {flag}" in result.stderr


def test_citrus_launcher_job_runs_transfer_v2_only(tmp_path: Path) -> None:
    result = _run_citrus(tmp_path)
    assert result.returncode == 0, result.stderr
    job = next((tmp_path / "citrus-logs").glob("**/run_citrus_session_import.sh")).read_text()
    assert "--recording-type" not in job and "--behavior-mode" not in job
    assert "diagnostics" not in job
