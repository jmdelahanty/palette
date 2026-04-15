from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import audit_swim_bladder_tuning_metadata as mod


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_analysis_metadata_json(
    path: Path,
    *,
    camera_id: str,
    swim_entry: dict[str, object] | None,
) -> None:
    payload = {
        "attributes": {
            "camera_metadata": json.dumps({"device_serial_number": camera_id}),
        }
    }
    if swim_entry is not None:
        payload["attributes"]["subject_mask_tuning"] = {
            "version": "2.0",
            "components": {"swim_bladder": swim_entry},
        }
    _write_json(path / "analysis_metadata" / "zarr.json", payload)


def _make_training_zarr(
    base: Path,
    name: str,
    *,
    camera_id: str,
    swim_entry: dict[str, object] | None,
    crop_run: str | None = None,
    keypoint_run: str | None = None,
    keypoint_group: str = "refined_keypoints_runs",
) -> Path:
    zarr_path = base / name
    _make_analysis_metadata_json(zarr_path, camera_id=camera_id, swim_entry=swim_entry)
    if crop_run:
        (zarr_path / "crop_runs" / crop_run).mkdir(parents=True, exist_ok=True)
    if keypoint_run:
        (zarr_path / keypoint_group / keypoint_run).mkdir(parents=True, exist_ok=True)
    return zarr_path


def test_scan_zarr_path_classifies_source_like(tmp_path: Path) -> None:
    zarr_path = _make_training_zarr(
        tmp_path,
        "recording_training.zarr",
        camera_id="2010093",
        swim_entry={
            "method": "polar_boundary_center_seed",
            "tuned_timestamp": "2026-04-01T17:11:12.804583+00:00",
            "context": {
                "storage_component_name": "swim_bladder",
                "crop_run": "crop_001",
                "keypoint_run": "refined_keypoints_001",
                "keypoint_source": "refined_keypoints_runs",
                "roi_index": 20,
            },
        },
        crop_run="crop_001",
        keypoint_run="refined_keypoints_001",
    )

    row = mod._scan_zarr_path(  # noqa: SLF001
        zarr_path,
        zarr_use_filter="training",
        camera_ids=set(),
    )

    assert row.status == "source_like"
    assert row.camera_id == "2010093"
    assert row.local_crop_exists is True
    assert row.local_keypoint_exists is True
    assert "crop_run" in row.context_keys


def test_scan_zarr_path_classifies_clean_propagated(tmp_path: Path) -> None:
    zarr_path = _make_training_zarr(
        tmp_path,
        "recording_training.zarr",
        camera_id="2010094",
        swim_entry={
            "method": "polar_boundary_center_seed",
            "tuned_timestamp": "2026-04-01T17:11:12.804583+00:00",
            "context": {
                "storage_component_name": "swim_bladder",
            },
            "propagated_by_camera": True,
            "propagated_component": "swim_bladder",
        },
    )

    row = mod._scan_zarr_path(  # noqa: SLF001
        zarr_path,
        zarr_use_filter="training",
        camera_ids={"2010094"},
    )

    assert row.status == "clean_propagated"
    assert row.propagated_by_camera is True
    assert row.context_keys == ()


def test_scan_zarr_path_classifies_stale_source_context(tmp_path: Path) -> None:
    zarr_path = _make_training_zarr(
        tmp_path,
        "recording_training.zarr",
        camera_id="2010095",
        swim_entry={
            "method": "polar_boundary_center_seed",
            "tuned_timestamp": "2026-04-01T17:11:12.804583+00:00",
            "context": {
                "storage_component_name": "swim_bladder",
                "crop_run": "crop_001",
                "keypoint_run": "refined_keypoints_001",
                "keypoint_source": "refined_keypoints_runs",
            },
        },
    )

    row = mod._scan_zarr_path(  # noqa: SLF001
        zarr_path,
        zarr_use_filter="training",
        camera_ids={"2010095"},
    )

    assert row.status == "stale_source_context"
    assert row.local_crop_exists is False
    assert row.local_keypoint_exists is False


def test_main_strict_succeeds_for_one_source_and_one_clean_propagated(tmp_path: Path, capsys) -> None:
    _make_training_zarr(
        tmp_path,
        "recording_a_training.zarr",
        camera_id="2010096",
        swim_entry={
            "method": "polar_boundary_center_seed",
            "tuned_timestamp": "2026-04-01T17:11:12.804583+00:00",
            "context": {
                "storage_component_name": "swim_bladder",
                "crop_run": "crop_001",
                "keypoint_run": "refined_keypoints_001",
                "keypoint_source": "refined_keypoints_runs",
            },
        },
        crop_run="crop_001",
        keypoint_run="refined_keypoints_001",
    )
    _make_training_zarr(
        tmp_path,
        "recording_b_training.zarr",
        camera_id="2010096",
        swim_entry={
            "method": "polar_boundary_center_seed",
            "tuned_timestamp": "2026-04-01T17:11:12.804583+00:00",
            "context": {"storage_component_name": "swim_bladder"},
            "propagated_by_camera": True,
            "propagated_component": "swim_bladder",
        },
    )

    rc = mod.main([str(tmp_path), "--recursive", "--camera-id", "2010096", "--strict"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "camera_id=2010096 total=2 source_like=1 clean_propagated=1" in out
