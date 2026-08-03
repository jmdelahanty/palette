from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


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
        env={**os.environ, "PALETTE_PYTHON": sys.executable},
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

    assert config["PROTOCOL_PROFILE"].endswith("chaser_event_windows_v1.yaml")

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
    assert "export PALETTE_DISABLE_REGISTRY_WRITES=1" in job_script
    assert '"registry_write_mode": "deferred_to_serial_finalizer"' in job_script
    run_dir = tmp_path / "logs" / "chaser_analytics_authoritative_inputs"
    finalizer = (run_dir / "run_registry_finalizer.sh").read_text(encoding="utf-8")
    assert "fisheye.analysis_workflows.registry_finalize" in finalizer
    assert "--stage-selector eye_angles=latest" in finalizer


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


def test_profile_enable_adds_dependency_and_renders_generic_steps(
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
        run_id="enabled_escape_events",
        selectors=("--enable-chaser-module", "chaser_escape_events"),
    )
    run_dir = tmp_path / "logs" / "chaser_analytics_enabled_escape_events"
    profiles = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))

    assert config["RUN_CHASER_BOUT_RESPONSE"] == "1"
    assert config["RUN_CHASER_ESCAPE_EVENTS"] == "1"
    assert config["RUN_CHASER_RADIAL_OCCUPANCY"] == "0"
    assert profiles["analysis_selection"]["explicit_enable"] == [
        "chaser_escape_events"
    ]
    selected = profiles["analysis_selection"]["selected_module_ids"]
    assert selected.index("chaser_bout_response") < selected.index(
        "chaser_escape_events"
    )
    assert "fisheye.analysis.chaser_bout_response" in job_script
    assert "fisheye.analysis.chaser_escape_events" in job_script
    assert "fisheye.utils.run_goodcopbadcop" not in job_script


def test_goodcopbadcop_v2_preset_selects_full_generic_profile(
    tmp_path: Path,
) -> None:
    palette_repo = tmp_path / "palette-checkout"
    _build_clean_palette_checkout(palette_repo)
    zarr_path = tmp_path / "recording" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    config, _job_script = _render(
        palette_repo=palette_repo,
        zarr_path=zarr_path,
        log_dir=tmp_path / "logs",
        run_id="goodcopbadcop_v2",
        selectors=("--preset", "goodcopbadcop_v2"),
    )
    run_dir = tmp_path / "logs" / "chaser_analytics_goodcopbadcop_v2"
    profiles = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))

    assert config["CHASER_ANALYSIS_PROFILE"].endswith(
        "chaser_behavior_full_v2.yaml"
    )
    assert profiles["analysis_profile"]["profile_id"] == "chaser_behavior_full_v2"
    assert all(
        config[key] == "1"
        for key in (
            "RUN_CHASER_BOUT_RESPONSE",
            "RUN_CHASER_ESCAPE_EVENTS",
            "RUN_CHASER_RADIAL_OCCUPANCY",
            "RUN_CHASER_RESPONSE_REGIMES",
        )
    )
    assert config["CHASER_BOUT_RESPONSE_COMPONENT"] == (
        "chaser_bout_response_v2_20260717"
    )


def test_profile_selection_rejects_explicitly_disabled_dependency(
    tmp_path: Path,
) -> None:
    palette_repo = tmp_path / "palette-checkout"
    _build_clean_palette_checkout(palette_repo)
    zarr_path = tmp_path / "recording" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--zarr",
            str(zarr_path),
            "--palette-repo",
            str(palette_repo),
            "--log-dir",
            str(tmp_path / "logs"),
            "--run-id",
            "disabled_dependency",
            "--enable-chaser-module",
            "chaser_escape_events",
            "--disable-chaser-module",
            "chaser_bout_response",
            "--dry-run",
        ],
        cwd=palette_repo,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PALETTE_PYTHON": sys.executable},
    )

    assert result.returncode != 0
    assert "requires explicitly disabled" in result.stderr
