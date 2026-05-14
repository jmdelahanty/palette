from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.registry.db import Registry
from fisheye.utils import accept_detect_review as mod


def _make_zarr(path: Path, *, with_group: str = "interpolated") -> Path:
    root = zarr.open_group(store=path, mode="w")
    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    if with_group:
        group = run.create_group(with_group)
        group.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
        group.create_array(
            "bbox_norm_coords",
            data=np.asarray(
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.4, 0.4, 0.1, 0.1],
                ],
                dtype=np.float64,
            ),
        )
        group.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    return path


def _make_curated_zarr(path: Path) -> Path:
    root = zarr.open_group(store=path, mode="w")
    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    run.attrs["source_detect_run"] = "detect_1"
    run.create_array("refined_row_ids", data=np.asarray([0], dtype=np.int64))
    run.create_array("frame_indices", data=np.asarray([0], dtype=np.int32))
    run.create_array("entity_ids", data=np.asarray([0], dtype=np.int32))
    run.create_array(
        "bbox_img_xyxy",
        data=np.asarray([[1.0, 1.0, 4.0, 4.0]], dtype=np.float64),
    )
    run.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
    )
    run.create_array("status_codes", data=np.asarray([0], dtype=np.int8))
    run.create_array("source_kind_codes", data=np.asarray([1], dtype=np.int8))
    run.create_array("review_state_codes", data=np.asarray([1], dtype=np.int8))
    run.create_array("keypoints_state_codes", data=np.asarray([0], dtype=np.int8))
    run.create_array("subject_mask_state_codes", data=np.asarray([0], dtype=np.int8))
    run.create_array("eye_mask_state_codes", data=np.asarray([0], dtype=np.int8))
    run.create_array("swim_bladder_state_codes", data=np.asarray([0], dtype=np.int8))
    return path


def test_accept_detect_review_writes_status_and_latest(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
    rc = mod.main(
        [
            str(zarr_path),
            "--state",
            "approved",
            "--method",
            "manual",
            "--intended-use",
            "training",
            "--reviewer",
            "operator1",
        ]
    )
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_detect_runs"]
    run = parent["refined_1"]
    status = dict(run.attrs["detect_review_status"])
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert "timestamp_utc" in status
    assert status["timestamp"] == status["timestamp_utc"]
    assert status["resolved_group"] == "interpolated"
    assert parent.attrs["detect_review_status_latest"] == "refined_1"
    profile_parent = root["analysis/detection_profile_runs"]
    profile_run = profile_parent.attrs["latest"]
    profile_group = profile_parent[profile_run]
    assert profile_group.attrs["source_detection_path"] == "refined_detect_runs/refined_1/interpolated"
    assert profile_group.attrs["source_detection_type"] == "interpolated"
    assert profile_group.attrs["fingerprint_status"] == "complete"
    assert len(profile_group.attrs["source_fingerprint"]) == 64
    summary = profile_group.attrs["profile_summary"]
    assert summary["source"]["review_state"] == "approved"
    assert summary["source"]["review_intended_use"] == "training"
    assert len(summary["source"]["content_hash"]) == 64


def test_accept_detect_review_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
    rc = mod.main([str(zarr_path), "--dry-run", "--json"])
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    run = root["refined_detect_runs"]["refined_1"]
    assert "detect_review_status" not in run.attrs


def test_accept_detect_review_strict_requires_reviewer_for_approved(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
    rc = mod.main(
        [
            str(zarr_path),
            "--strict",
            "--state",
            "approved",
            "--intended-use",
            "training",
            "--json",
        ]
    )
    assert rc == 1


def test_accept_detect_review_target_group_missing_fails(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
    rc = mod.main([str(zarr_path), "--target-group", "manual", "--json"])
    assert rc == 1


def test_accept_detect_review_json_output_contains_expected_fields(tmp_path: Path, capsys) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
    rc = mod.main(
        [
            str(zarr_path),
            "--state",
            "approved",
            "--intended-use",
            "full_recording",
            "--reviewer",
            "operator2",
            "--json",
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["refined_run"] == "refined_1"
    assert payload["resolved_group"] == "interpolated"
    assert payload["state"] == "approved"
    assert payload["intended_use"] == "full_recording"


def test_accept_detect_review_prefers_curated_root_when_present(tmp_path: Path) -> None:
    zarr_path = _make_curated_zarr(tmp_path / "curated.zarr")
    rc = mod.main(
        [
            str(zarr_path),
            "--state",
            "approved",
            "--intended-use",
            "training",
            "--reviewer",
            "operator3",
        ]
    )
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    status = dict(root["refined_detect_runs"]["refined_1"].attrs["detect_review_status"])
    assert status["resolved_group"] == "refined"
    profile_group = root["analysis/detection_profile_runs"][
        root["analysis/detection_profile_runs"].attrs["latest"]
    ]
    assert profile_group.attrs["source_detection_path"] == "refined_detect_runs/refined_1"


def test_accept_detect_review_syncs_detection_profile_registry(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "registered.zarr", with_group="interpolated")
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_detect_review",
        session_uuid="session_detect_review",
        zarr_path=zarr_path,
        recording_id="recording_detect_review",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.close()

    rc = mod.main(
        [
            str(zarr_path),
            "--state",
            "approved",
            "--method",
            "manual",
            "--intended-use",
            "training",
            "--reviewer",
            "operator4",
            "--registry",
            str(registry_path),
        ]
    )
    assert rc == 0

    registry = Registry(registry_path)
    try:
        rows = registry.query_detection_data_profile_latest(dataset_ids=["dataset_detect_review"])
    finally:
        registry.close()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["detection_type"] == "interpolated"
    assert row["detection_path"] == "refined_detect_runs/refined_1/interpolated"
    assert row["recording_id"] == "recording_detect_review"
    assert row["zarr_use"] == "training"
    assert row["coverage_percent"] == 100.0
