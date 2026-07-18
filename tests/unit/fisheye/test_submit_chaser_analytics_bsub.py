from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "submit_chaser_analytics_bsub.sh"


def _build_clean_palette_checkout(path: Path) -> None:
    scripts_dir = path / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "py", scripts_dir / "py")
    (path / "src").symlink_to(REPO / "src", target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Palette Tests",
            "-c",
            "user.email=palette-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def _read_config(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        for key, value in [line.split("=", 1)]
    }


def _render(
    *,
    palette_repo: Path,
    zarr_path: Path,
    log_dir: Path,
    run_id: str,
    selectors: tuple[str, ...] = (),
) -> tuple[dict[str, str], str]:
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--zarr",
            str(zarr_path),
            "--palette-repo",
            str(palette_repo),
            "--log-dir",
            str(log_dir),
            "--run-id",
            run_id,
            "--skip-movement",
            "--skip-stimulus-epoch",
            "--skip-detection-occupancy",
            "--skip-chaser-distance",
            "--dry-run",
            *selectors,
        ],
        cwd=palette_repo,
        check=True,
        text=True,
        capture_output=True,
    )
    run_dir = log_dir / f"chaser_analytics_{run_id}"
    return (
        _read_config(run_dir / "config.env"),
        (run_dir / "run_one_zarr.sh").read_text(encoding="utf-8"),
    )


def test_skipped_producers_resolve_authoritative_inputs_per_recording(
    tmp_path: Path,
) -> None:
    palette_repo = tmp_path / "palette-checkout"
    _build_clean_palette_checkout(palette_repo)
    zarr_path = tmp_path / "recording" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    config, job_script = _render(
        palette_repo=palette_repo,
        zarr_path=zarr_path,
        log_dir=tmp_path / "logs",
        run_id="authoritative_inputs",
    )

    for key in (
        "TRACK_RUN",
        "SWIM_BOUT_RUN",
        "BOUT_KINEMATICS_RUN",
        "EPOCH_RUN",
        "CHASER_DISTANCE_RUN",
    ):
        assert config[key] == "latest"
        assert config[f"{key}_SELECTION"] == "authoritative_latest_complete"
    assert "resolve_authoritative_run_name" in job_script
    assert 'chasers = parent[resolved]["chasers"]' in job_script
    assert config["SPEED_LEVEL"] == "''"
    assert 'if [[ -n "$SPEED_LEVEL" ]]' in job_script
    assert 'epoch_speed_args+=(--speed-level "$SPEED_LEVEL")' in job_script


def test_explicit_reused_inputs_override_authoritative_resolution(
    tmp_path: Path,
) -> None:
    palette_repo = tmp_path / "palette-checkout"
    _build_clean_palette_checkout(palette_repo)
    zarr_path = tmp_path / "recording" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    selectors = (
        "--track-run",
        "track_explicit",
        "--swim-bout-run",
        "bouts_explicit",
        "--bout-kinematics-run",
        "bout_kinematics_explicit",
        "--epoch-run",
        "epochs_explicit",
        "--chaser-distance-run",
        "distance_explicit",
    )
    config, _job_script = _render(
        palette_repo=palette_repo,
        zarr_path=zarr_path,
        log_dir=tmp_path / "logs",
        run_id="explicit_inputs",
        selectors=selectors,
    )

    assert config["TRACK_RUN"] == "track_explicit"
    assert config["SWIM_BOUT_RUN"] == "bouts_explicit"
    assert config["BOUT_KINEMATICS_RUN"] == "bout_kinematics_explicit"
    assert config["EPOCH_RUN"] == "epochs_explicit"
    assert config["CHASER_DISTANCE_RUN"] == "distance_explicit"
    for key in (
        "TRACK_RUN_SELECTION",
        "SWIM_BOUT_RUN_SELECTION",
        "BOUT_KINEMATICS_RUN_SELECTION",
        "EPOCH_RUN_SELECTION",
        "CHASER_DISTANCE_RUN_SELECTION",
    ):
        assert config[key] == "explicit_reuse"
