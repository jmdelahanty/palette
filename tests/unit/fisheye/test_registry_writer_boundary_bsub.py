from __future__ import annotations

import os
from pathlib import Path
import subprocess

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
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--log-dir",
            str(tmp_path / "citrus-logs"),
            "--run-id",
            "writer-boundary",
            "--dry-run",
            *options,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def test_citrus_register_requires_designated_writer_host(tmp_path: Path) -> None:
    result = _run_citrus(tmp_path)

    assert result.returncode == 2
    assert "--register requires --writer-host" in result.stderr


def test_citrus_register_pins_host_and_exports_shadow_configuration(
    tmp_path: Path,
) -> None:
    env = {
        "PALETTE_REGISTRY_WRITER_LOCK_PATH": str(tmp_path / "writer.lock"),
        "PALETTE_REGISTRY_SHADOW_TEMP_ROOT": str(tmp_path / "shadows"),
        "PALETTE_REGISTRY_SHADOW_BACKUP_DIR": str(tmp_path / "backups"),
    }
    result = _run_citrus(tmp_path, "--writer-host", "writer01", env_overrides=env)

    assert result.returncode == 0, result.stderr
    assert "hname==writer01" in result.stdout
    assert "span\\[hosts=1\\]" in result.stdout
    job_script = next(
        (tmp_path / "citrus-logs").glob("**/run_citrus_session_import.sh")
    )
    subprocess.run(["bash", "-n", str(job_script)], check=True)
    job = job_script.read_text(encoding="utf-8")
    assert "export PALETTE_REGISTRY_WRITER_HOST=writer01" in job
    assert (
        f"export PALETTE_REGISTRY_WRITER_LOCK_PATH={env['PALETTE_REGISTRY_WRITER_LOCK_PATH']}"
        in job
    )
    assert (
        f"export PALETTE_REGISTRY_SHADOW_TEMP_ROOT={env['PALETTE_REGISTRY_SHADOW_TEMP_ROOT']}"
        in job
    )
    assert (
        f"export PALETTE_REGISTRY_SHADOW_BACKUP_DIR={env['PALETTE_REGISTRY_SHADOW_BACKUP_DIR']}"
        in job
    )
    assert "REGISTER=1" in job


def test_citrus_no_register_dry_run_does_not_need_writer_host(tmp_path: Path) -> None:
    result = _run_citrus(tmp_path, "--no-register")

    assert result.returncode == 0, result.stderr
    assert "hname==" not in result.stdout
    job_script = next(
        (tmp_path / "citrus-logs").glob("**/run_citrus_session_import.sh")
    )
    subprocess.run(["bash", "-n", str(job_script)], check=True)
    job = job_script.read_text(encoding="utf-8")
    assert "REGISTER=0" in job
    assert 'if [[ "${REGISTER}" == "1" ]]' in job


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
