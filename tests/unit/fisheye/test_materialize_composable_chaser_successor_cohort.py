from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils import materialize_composable_chaser_successor_cohort as cohort


REPO = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = REPO / "scripts/submit_composable_chaser_successors_bsub.sh"


def _write_group(path: Path, attrs: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _record(attrs: dict[str, object], key: str, value: dict[str, object]) -> None:
    attrs[key] = value
    attrs[f"{key}_sha256"] = canonical_json_sha256(value)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    recording_id = "2026-08-10T17-20-55Z_arena_1_goodbatbadbat"
    recording = tmp_path / recording_id
    archive = recording / "zarr" / f"{recording_id}_analysis.zarr"
    raw_h5 = recording / "raw" / f"{recording_id}.h5"
    raw_h5.parent.mkdir(parents=True)
    raw_h5.write_bytes(b"sealed raw h5 fixture")
    stat = raw_h5.stat()
    _write_group(
        archive,
        {
            "recording_id": recording_id,
            "source_h5_path": str(raw_h5),
            "source_h5_size_bytes": stat.st_size,
            "source_h5_fingerprint": "fixture-stat",
            "source_h5_fingerprint_strategy": "stat_v1",
        },
    )
    _write_group(
        archive / "analysis/stimulus_runs/stimulus_canonical_v1_fixture",
        {"schema_id": "palette.stimulus_run"},
    )

    geometry_run = "arena_geometry_selection_fixture"
    _write_group(
        archive / "analysis/arena_geometry_selection",
        {"latest": geometry_run, "latest_complete": geometry_run},
    )
    geometry_attrs: dict[str, object] = {"selection_id": geometry_run}
    _record(
        geometry_attrs,
        "selection_record",
        {
            "selected_candidate": {
                "arena_binding": {
                    "arena_id": "arena_1",
                    "camera_serial": "2010093",
                }
            }
        },
    )
    _write_group(
        archive / "analysis/arena_geometry_selection" / geometry_run,
        geometry_attrs,
    )
    physical_attrs: dict[str, object] = {}
    _record(
        physical_attrs,
        "source_camera_physical_authority",
        {"camera_id": "2010093", "pixels_per_mm": 50.0},
    )
    _write_group(
        archive / "analysis/calibration/coordinate_frames",
        physical_attrs,
    )

    for run_name, provider in (
        (cohort.KEYPOINT_PROXY_RUN, "keypoint"),
        (cohort.DETECTION_PROXY_RUN, "detection"),
    ):
        attrs: dict[str, object] = {}
        _record(
            attrs,
            "chaser_input_provenance_proxy_manifest",
            {
                "recording_id": recording_id,
                "run_name": run_name,
                "provider": provider,
            },
        )
        _write_group(
            archive / "analysis/chaser_input_provenance_proxy_runs" / run_name,
            attrs,
        )

    motion_run, bout_run = cohort.MOTION_BOUT_PAIRS[0]
    _write_group(
        archive / "analysis/track_kinematics_runs/provider" / motion_run,
        {"recording_id": recording_id},
    )
    _write_group(
        archive / "analysis/swim_bout_runs" / bout_run,
        {"recording_id": recording_id},
    )

    snapshot = tmp_path / "registry.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "dataset_id": "fixture-dataset",
                    "recording_id": recording_id,
                    "zarr_path": str(archive),
                    "protocol_name": "goodbatbadbat",
                    "protocol_hash": "a" * 64,
                    "arena_id": "arena_1",
                    "camera_id": "2010093",
                }
            ]
        ),
        encoding="utf-8",
    )
    return archive, raw_h5, snapshot


def _clean_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "palette-deployment"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    py = scripts / "py"
    shutil.copy2(REPO / "scripts/py", py)
    os.chmod(py, 0o755)
    (repo / "src").symlink_to(REPO / "src", target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
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
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return repo, commit


def test_plan_freezes_exact_recording_inputs(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = cohort.plan_cohort_task(
        snapshot,
        operations_root=tmp_path / "operations",
    )

    assert task["recording_count"] == 1
    assert task["status_counts"] == {"ready": 1}
    assert task["runnable_task_indices"] == [1]
    assert cohort.load_cohort_task(task)["task_sha256"] == task["task_sha256"]
    entry = task["entries"][0]
    assert entry["canonical_stimulus_run"] == "stimulus_canonical_v1_fixture"
    assert entry["keypoint_proxy"]["run_name"] == cohort.KEYPOINT_PROXY_RUN
    assert entry["detection_proxy"]["run_name"] == cohort.DETECTION_PROXY_RUN
    assert entry["motion_and_bouts"]["motion_run_path"].endswith(
        cohort.MOTION_BOUT_PAIRS[0][0]
    )
    assert len(entry["input_group_bindings"]) == 7
    assert entry["existing_output_group_paths"] == []
    assert task["safety"] == cohort.EXPECTED_SAFETY


def test_task_digest_rejects_mutation(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = cohort.plan_cohort_task(snapshot, operations_root=tmp_path / "operations")
    task["entries"][0]["recording_id"] = "changed"

    with pytest.raises(cohort.ComposableChaserCohortError, match="digest is stale"):
        cohort.load_cohort_task(task)


def test_task_successor_freezes_receipt_bound_plot_recipes(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = cohort.plan_cohort_task(
        snapshot, operations_root=tmp_path / "operations"
    )
    successor = cohort.successor_cohort_task(original)

    assert successor["schema_version"] == 2
    assert successor["selection_policy"]["successor_of_task_sha256"] == original[
        "task_sha256"
    ]
    entry = successor["entries"][0]
    assert entry["relative_frame_validation"]["mode"] == (
        cohort.RELATIVE_FRAME_VALIDATION_MODE
    )
    assert entry["output_run_names"]["spatial_occupancy"] == (
        cohort.SPATIAL_OCCUPANCY_RECEIPT_BOUND_RUN
    )
    assert entry["output_run_names"]["spatial_plot_bundle"] == (
        cohort.SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME
    )
    assert entry["spatial_occupancy_successor"]["mode"] == (
        "materialize_missing_receipt_bound_v2"
    )
    assert entry["output_run_names"]["dashboard_bundle"] == (
        cohort.DASHBOARD_RECIPE_BUNDLE_NAME
    )
    assert entry["output_run_names"]["detailed_bundle"] == (
        cohort.DETAILED_RECIPE_BUNDLE_NAME
    )
    assert cohort.load_cohort_task(successor)["task_sha256"] == successor[
        "task_sha256"
    ]


def test_task_successor_reuses_existing_exact_spatial_science(tmp_path: Path) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = cohort.plan_cohort_task(
        snapshot, operations_root=tmp_path / "operations"
    )
    recording_id = original["entries"][0]["recording_id"]
    _write_group(
        archive
        / "analysis/chaser_spatial_occupancy_runs"
        / cohort.SPATIAL_OCCUPANCY_RUN,
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "selection": "none",
            "palette_run_name": cohort.SPATIAL_OCCUPANCY_RUN,
            "recording_id": recording_id,
        },
    )

    successor = cohort.successor_cohort_task(original)

    entry = successor["entries"][0]
    assert entry["output_run_names"]["spatial_occupancy"] == (
        cohort.SPATIAL_OCCUPANCY_RUN
    )
    assert entry["spatial_occupancy_successor"] == {
        "mode": "reuse_existing_exact_complete_v1",
        "exact_run_name": cohort.SPATIAL_OCCUPANCY_RUN,
        "plot_bundle": cohort.SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME,
    }


def test_run_one_rejects_changed_frozen_input_metadata(tmp_path: Path) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = cohort.plan_cohort_task(snapshot, operations_root=tmp_path / "operations")
    motion_path = Path(
        task["entries"][0]["motion_and_bouts"]["motion_run_path"]
    )
    (archive / motion_path / "zarr.json").write_text("{}", encoding="utf-8")

    with pytest.raises(cohort.ComposableChaserCohortError, match="metadata changed"):
        cohort.run_one(
            task,
            task_index=1,
            palette_repo=tmp_path / "unused",
            palette_commit="0" * 40,
            scratch_root=tmp_path / "scratch",
            receipt_root=tmp_path / "receipts",
        )


def test_run_one_dry_run_renders_complete_serial_chain(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = cohort.plan_cohort_task(snapshot, operations_root=tmp_path / "operations")
    repo, commit = _clean_repo(tmp_path)

    result = cohort.run_one(
        task,
        task_index=1,
        palette_repo=repo,
        palette_commit=commit,
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        apply=False,
    )

    assert result["status"] == "planned_no_writes"
    assert result["palette_commit"] == commit
    assert [stage["stage"] for stage in result["stages"]] == [
        "semantic_stimulus",
        "semantic_epoch_v1",
        "semantic_epoch_v2",
        "semantic_selection",
        "keypoint_relative_frame",
        "detection_relative_frame",
        "composable_successors",
        "keypoint_radial_near_field",
        "detection_radial_near_field",
        "spatial_occupancy",
        "spatial_occupancy_plots",
        "dashboard_plots",
        "detailed_plots",
    ]
    assert all(stage["mode"] == "planned_no_write" for stage in result["stages"])
    assert result["safety"] == cohort.EXPECTED_SAFETY
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "receipts").exists()


def test_receipt_bound_successor_dry_run_passes_targeted_receipts(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = cohort.plan_cohort_task(
        snapshot, operations_root=tmp_path / "operations"
    )
    task = cohort.successor_cohort_task(original)
    repo, commit = _clean_repo(tmp_path)

    result = cohort.run_one(
        task,
        task_index=1,
        palette_repo=repo,
        palette_commit=commit,
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        apply=False,
    )

    names = [stage["stage"] for stage in result["stages"]]
    assert names[6:8] == [
        "keypoint_relative_frame_validation_receipt",
        "detection_relative_frame_validation_receipt",
    ]
    for source in (
        "semantic_selection",
        "keypoint_radial",
        "detection_radial",
        "controller",
        "bout",
        "escape",
        "spatial_occupancy",
    ):
        assert f"{source}_exact_child_validation_receipt" in names
    spatial = next(
        stage for stage in result["stages"] if stage["stage"] == "spatial_occupancy"
    )
    detailed = next(
        stage for stage in result["stages"] if stage["stage"] == "detailed_plots"
    )
    dashboard = next(
        stage for stage in result["stages"] if stage["stage"] == "dashboard_plots"
    )
    for stage in (spatial, detailed):
        assert "--keypoint-relative-frame-receipt" in stage["command"]
        assert "--detection-relative-frame-receipt" in stage["command"]
        keypoint_receipt = stage["command"][
            stage["command"].index("--keypoint-relative-frame-receipt") + 1
        ]
        assert commit in keypoint_receipt
    for option in (
        "--semantic-selection-receipt",
        "--keypoint-radial-receipt",
        "--detection-radial-receipt",
    ):
        assert option in spatial["command"]
        assert option not in detailed["command"]
    for option in (
        "--controller-validation-receipt",
        "--bout-validation-receipt",
        "--escape-validation-receipt",
    ):
        assert option in dashboard["command"]
        assert option in detailed["command"]
    assert dashboard["command"][
        dashboard["command"].index("--bundle-name") + 1
    ] == cohort.DASHBOARD_RECIPE_BUNDLE_NAME
    occupancy_plot = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "spatial_occupancy_plots"
    )
    assert occupancy_plot["command"][
        occupancy_plot["command"].index("--bundle-name") + 1
    ] == cohort.SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME
    assert "--source-validation-receipt" in occupancy_plot["command"]


def test_reused_plot_receipt_is_content_verified(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    output.write_bytes(b"figure bytes")
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "recording_id": "recording-1",
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "outputs": [
            {
                "path": str(output),
                "sha256": cohort._sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
        ],
    }
    receipt["payload_sha256"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert cohort._validated_plot_receipt(
        receipt_path, recording_id="recording-1"
    )
    output.write_bytes(b"changed")
    with pytest.raises(cohort.ComposableChaserCohortError, match="differs"):
        cohort._validated_plot_receipt(receipt_path, recording_id="recording-1")


def test_bsub_submitter_renders_pinned_array_without_submission(
    tmp_path: Path,
) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = cohort.plan_cohort_task(snapshot, operations_root=tmp_path / "operations")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    repo, commit = _clean_repo(tmp_path)
    run_root = tmp_path / "runs"

    completed = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--task",
            str(task_path),
            "--palette-repo",
            str(repo),
            "--palette-commit",
            commit,
            "--run-root",
            str(run_root),
            "--run-id",
            "fixture",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PALETTE_PYTHON": os.environ.get("PALETTE_PYTHON", "")},
    )

    assert "mode=dry_run_no_submission" in completed.stdout
    assert "recording_count=1" in completed.stdout
    assert "array_indices=1-1" in completed.stdout
    assert "selected_recording_count=1" in completed.stdout
    run_dir = run_root / "composable_chaser_successors_fixture"
    job_script = (run_dir / "run_one_recording.sh").read_text(encoding="utf-8")
    assert "LSB_JOBINDEX" in job_script
    assert "materialize_composable_chaser_successor_cohort run-one" in job_script
    assert f"--palette-commit {commit}" in job_script
    submission = (run_dir / "submission.env").read_text(encoding="utf-8")
    assert "selector_eligible=false" in submission
    assert "production_authority=false" in submission
    assert "registry_update=false" in submission
    assert "submit_requested=0" in submission
