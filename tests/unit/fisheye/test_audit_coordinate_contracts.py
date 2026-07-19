from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import fisheye.utils.audit_coordinate_contracts as coordinate_audit

from fisheye.utils.audit_coordinate_contracts import audit_registry
from fisheye.utils.audit_coordinate_contracts import open_registry_readonly
from fisheye.utils.audit_coordinate_contracts import summarize
from fisheye.utils.audit_coordinate_contracts import write_csv
from fisheye.utils.audit_coordinate_contracts import write_jsonl
from fisheye.utils.audit_coordinate_contracts import write_markdown
from fisheye.utils.audit_coordinate_contracts import write_summary


def _write_node(
    root: Path,
    relative_path: str = ".",
    *,
    node_type: str = "group",
    attributes: dict[str, object] | None = None,
    shape: list[int] | None = None,
) -> Path:
    path = root if relative_path == "." else root / relative_path
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "zarr_format": 3,
        "node_type": node_type,
        "attributes": attributes or {},
    }
    if node_type == "array":
        payload.update(
            {
                "shape": shape or [1],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": shape or [1]},
                },
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": [],
            }
        )
    (path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ensure_groups(root: Path, relative_path: str) -> None:
    parts = Path(relative_path).parts[:-1]
    for index in range(1, len(parts) + 1):
        path = "/".join(parts[:index])
        metadata = root / path / "zarr.json"
        if not metadata.exists():
            _write_node(root, path)


def _write_array(
    root: Path,
    relative_path: str,
    *,
    attributes: dict[str, object] | None = None,
    shape: list[int] | None = None,
) -> None:
    _ensure_groups(root, relative_path)
    _write_node(
        root,
        relative_path,
        node_type="array",
        attributes=attributes,
        shape=shape,
    )


def _replace_group_attrs(root: Path, relative_path: str, attributes: dict[str, object]) -> None:
    _write_node(root, relative_path, attributes=attributes)


def _make_registry(path: Path, rows: list[dict[str, object]]) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_use TEXT,
                artifact_kind TEXT,
                status TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO datasets (
                dataset_id, recording_id, zarr_path, zarr_use, artifact_kind, status
            ) VALUES (
                :dataset_id, :recording_id, :zarr_path, :zarr_use, :artifact_kind, :status
            )
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _complete_xy_descriptor(*, space_id: str = "source_camera_image_px", units: str = "px") -> dict[str, object]:
    return {
        "schema_id": "palette.coordinate_descriptor",
        "schema_version": 1,
        "space_id": space_id,
        "geometry_type": "points_xy",
        "components": ["x", "y"],
        "component_units": [units, units],
        "origin": "top_left",
        "positive_directions": {"x": "right", "y": "down"},
        "reference_extent": {
            "width": 640,
            "height": 480,
            "units": "px",
            "authority": "/raw_video/images_full.shape[-2:]",
        },
        "pixel_convention": "pixel_center",
        "row_identity": {"mode": "track_frame_indices", "array_ref": "../frame_indices"},
        "source_camera_overlay": "direct",
        "lineage_refs": [{"ref": "/coordinate_records/source_selection"}],
    }


def _dataset_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if record["record_type"] == "coordinate_dataset"]


def _surface_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if record["record_type"] == "coordinate_surface"]


def test_registry_is_query_only_and_every_row_is_preserved(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _write_node(zarr_path)
    track_path = "analysis/track_kinematics_runs/offline/run/tracks/id_0/positions_px"
    _write_array(
        zarr_path,
        track_path,
        shape=[3, 2],
        attributes={
            "coordinate_descriptor": _complete_xy_descriptor(),
            "row_identity_ref": "frame_indices",
            "source_ref": "analysis/refined_detect_runs/source/instances",
            "source_camera_overlay_suitable": True,
        },
    )
    _write_array(
        zarr_path,
        "analysis/track_kinematics_runs/offline/run/tracks/id_0/frame_indices",
        shape=[3],
    )

    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "good",
                "recording_id": "rec-1",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "analysis",
                "status": "active",
            },
            {
                "dataset_id": "unreachable",
                "recording_id": "rec-2",
                "zarr_path": str(tmp_path / "does-not-exist.zarr"),
                "zarr_use": "analysis",
                "artifact_kind": "analysis",
                "status": "active",
            },
            {
                "dataset_id": "missing-path",
                "recording_id": "rec-3",
                "zarr_path": None,
                "zarr_use": "training",
                "artifact_kind": "training",
                "status": "missing",
            },
        ],
    )

    conn = open_registry_readonly(registry)
    try:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE datasets SET status = 'changed'")
    finally:
        conn.close()

    records = audit_registry(registry)
    datasets = _dataset_rows(records)
    assert [record["dataset_id"] for record in datasets] == ["good", "missing-path", "unreachable"]
    assert datasets[0]["status"] == "compatible"
    assert datasets[1]["status"] == "missing_or_unreadable"
    assert datasets[2]["status"] == "missing_or_unreadable"

    surfaces = _surface_rows(records)
    assert len(surfaces) == 1
    assert surfaces[0]["surface_type"] == "track_positions_px"
    assert surfaces[0]["status"] == "compatible"


def test_surface_families_and_fail_closed_classifications(tmp_path: Path) -> None:
    zarr_path = tmp_path / "families.zarr"
    _write_node(zarr_path)

    simple_arrays = {
        "analysis/refined_online_runs/online/positions_px": "refined_online_positions_px",
        "analysis/detect_runs/detect/bbox_norm_coords": "detect_bbox",
        "analysis/refined_detect_runs/refined/instances/bbox_img_xyxy": "refined_detect_bbox",
        "analysis/crop_runs/crop/bbox_img_xyxy": "crop_geometry",
        "analysis/crop_runs/crop/roi_coordinates_full": "crop_geometry",
        "analysis/crop_runs/crop/roi_coordinates_ds": "crop_geometry",
        "analysis/keypoints_runs/kp/keypoints_roi": "keypoint_roi",
        "analysis/refined_keypoints_runs/kp/keypoints_img": "keypoint_source_image",
        "analysis/refined_keypoints_runs/kp/keypoints_norm": "keypoint_normalized",
        "analysis/subject_shape_runs/shape/centroid_xy": "subject_shape_geometry",
        "analysis/subject_shape_runs/shape/centerline_spline_xy": "subject_shape_geometry",
        "analysis/refined_subject_masks_runs/mask/masks_roi": "subject_mask_geometry",
        "analysis/new_geometry_stage/foo_bbox_vertices": "unclassified_geometry_candidate",
    }
    for relative_path in simple_arrays:
        _write_array(zarr_path, relative_path, shape=[2, 2])

    _ensure_groups(zarr_path, "analysis/stimulus_runs/stim/tracking_data/chaser_states/field")
    _write_node(
        zarr_path,
        "analysis/stimulus_runs/stim/tracking_data/chaser_states",
        attributes={
            "coordinate_frame": "arena_relative_canvas_px",
            "units": "px",
            "coordinate_origin": "top_left_of_active_arena",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
            "reference_width": 800,
            "reference_height": 600,
            "pixel_convention": "continuous",
            "geometry_convention": "xy_point",
            "source_ref": "stimulus_runtime",
            "source_camera_overlay_suitable": False,
            "field_names": ["stimulus_frame_num", "chaser_pos_x", "chaser_pos_y"],
        },
    )
    _write_array(
        zarr_path,
        "analysis/stimulus_runs/stim/tracking_data/chaser_states/chaser_pos_x",
        shape=[2],
    )
    _write_array(
        zarr_path,
        "analysis/stimulus_runs/stim/tracking_data/chaser_states/target_clamped_pos_y",
        shape=[2],
    )

    _write_array(
        zarr_path,
        "analysis/calibration/homography_matrix",
        shape=[3, 3],
        attributes={"calibration_ref": "calibration/acquisition.json"},
    )

    # The historical online track-mm writer is an explicit recomputation class;
    # the audit does not guess from the numeric range.
    online_run = "analysis/track_kinematics_runs/online/run"
    _ensure_groups(zarr_path, f"{online_run}/tracks/id_0/positions_mm/value")
    _replace_group_attrs(
        zarr_path,
        online_run,
        {
            "method": "track_kinematics_online",
            "coordinate_space": "texture",
            "pixel_to_mm": 12.0,
            "pixels_per_mm_projector": 12.0,
            "position_source_path": "analysis/refined_online_detect_runs/online",
        },
    )
    _write_array(zarr_path, f"{online_run}/tracks/id_0/positions_mm", shape=[2, 2])
    _write_array(zarr_path, f"{online_run}/tracks/id_0/frame_indices", shape=[2])

    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "all-families",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "analysis",
                "status": "active",
            }
        ],
    )
    surfaces = _surface_rows(audit_registry(registry))
    surface_types = {str(record["surface_type"]) for record in surfaces}
    assert set(simple_arrays.values()) <= surface_types
    assert "stimulus_chaser" in surface_types
    assert "calibration_homography" in surface_types
    assert "track_positions_mm" in surface_types
    assert sum(record["surface_type"] == "stimulus_chaser" for record in surfaces) == 3

    chaser = next(record for record in surfaces if record["surface_type"] == "stimulus_chaser")
    assert chaser["status"] == "compatible_via_explicit_legacy_rule"

    homography = next(record for record in surfaces if record["surface_type"] == "calibration_homography")
    assert homography["status"] == "ambiguous_fail_closed"
    assert "HOMOGRAPHY_DIRECTION_MISSING" in homography["issue_codes"]

    online_mm = next(record for record in surfaces if record["surface_type"] == "track_positions_mm")
    assert online_mm["status"] == "recompute_required"
    assert "ONLINE_MM_CONVERSION_RECOMPUTATION_REQUIRED" in online_mm["issue_codes"]

    unclassified = next(
        record for record in surfaces if record["surface_type"] == "unclassified_geometry_candidate"
    )
    assert unclassified["status"] == "ambiguous_fail_closed"
    assert "UNCLASSIFIED_GEOMETRY_CANDIDATE" in unclassified["issue_codes"]


def test_offline_crop_camera_reconstruction_requires_numerical_validation(tmp_path: Path) -> None:
    zarr_path = tmp_path / "offline.zarr"
    _write_node(zarr_path)
    run = "analysis/track_kinematics_runs/offline/run"
    _ensure_groups(zarr_path, f"{run}/tracks/id_0/positions_px/value")
    _replace_group_attrs(
        zarr_path,
        run,
        {
            "provenance": {
                "parameters": {"coordinate_space": "camera"},
                "inputs": {
                    "position_source_kind": "crop_rows",
                    "position_source_path": "crop_runs/crop",
                },
            }
        },
    )
    _write_array(zarr_path, f"{run}/tracks/id_0/positions_px", shape=[2, 2])
    _write_array(zarr_path, f"{run}/tracks/id_0/frame_indices", shape=[2])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "offline",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "analysis",
                "status": "active",
            }
        ],
    )
    surface = next(
        record
        for record in _surface_rows(audit_registry(registry))
        if record["surface_type"] == "track_positions_px"
    )
    assert surface["status"] == "numerical_validation_required"
    assert "OFFLINE_CROP_SOURCE_RECONSTRUCTION_NUMERICAL_VALIDATION_REQUIRED" in surface["issue_codes"]


def test_outputs_are_deterministic_and_resume_preserves_complete_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "empty.zarr"
    _write_node(zarr_path)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "empty",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "analysis",
                "status": "active",
            }
        ],
    )
    records = audit_registry(registry)
    summary = summarize(records)
    assert summary["dataset_row_count"] == 1
    assert summary["surface_count"] == 0

    first_jsonl = tmp_path / "first.jsonl"
    second_jsonl = tmp_path / "second.jsonl"
    csv_path = tmp_path / "inventory.csv"
    markdown_path = tmp_path / "inventory.md"
    summary_path = tmp_path / "summary.json"
    write_jsonl(first_jsonl, records)
    resumed = audit_registry(registry, resume_jsonl=first_jsonl)
    write_jsonl(second_jsonl, resumed)
    write_csv(csv_path, records)
    write_markdown(markdown_path, records, summary)
    write_summary(summary_path, summary)

    assert resumed == records
    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()
    assert csv_path.read_text(encoding="utf-8").startswith("record_type,dataset_key")
    assert "# Coordinate contract inventory" in markdown_path.read_text(encoding="utf-8")
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary

    # Root metadata is unchanged, but a child surface changed.  Resume must
    # invalidate from the full metadata inventory digest rather than silently
    # reusing the registry/root fingerprint.
    _write_array(
        zarr_path,
        "analysis/detect_runs/new/bbox_norm_coords",
        shape=[1, 4],
    )
    changed = audit_registry(registry, resume_jsonl=first_jsonl)
    assert summarize(changed)["surface_count"] == 1
    assert changed != records


def test_source_change_during_scan_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "changing.zarr"
    _write_node(zarr_path)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "changing",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "analysis",
                "status": "active",
            }
        ],
    )
    first_snapshot = list(coordinate_audit.iter_metadata_nodes(zarr_path))
    second_snapshot = [replace(first_snapshot[0], attributes={"changed": True})]
    snapshots = iter((first_snapshot, second_snapshot))

    def _changing_nodes(_path: Path):
        yield from next(snapshots)

    monkeypatch.setattr(coordinate_audit, "iter_metadata_nodes", _changing_nodes)
    dataset = _dataset_rows(audit_registry(registry))[0]
    assert dataset["status"] == "missing_or_unreadable"
    assert dataset["scan_complete"] is False
    assert "SOURCE_CHANGED_DURING_SCAN" in dataset["issue_codes"]
