from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.registry.db import Registry
from fisheye.shared.zarr.detect_frame_decisions import (
    DETECT_FRAME_DECISION_FAMILY,
    FRAME_REVIEW_CONTRACT_ATTR,
    FRAME_REVIEW_CONTRACT_ID,
)
from fisheye.shared.zarr_run_completion import mark_run_complete
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


def _make_selector_ineligible_training_candidate(
    path: Path,
    *,
    positive_frames: tuple[int, ...] = (0, 1),
) -> Path:
    root = zarr.open_group(store=path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_artifact_status": "awaiting_detection_review",
            "stage_selector_eligible": False,
        }
    )
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.zeros((2, 8, 8), dtype=np.uint8),
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([10, 20], dtype=np.int64),
    )
    parent = root.create_group("refined_detect_runs")
    run = parent.create_group("refined_1")
    run.attrs.update(
        {
            "stage_selector_eligible": False,
            FRAME_REVIEW_CONTRACT_ATTR: FRAME_REVIEW_CONTRACT_ID,
        }
    )
    instances = run.create_group("instances")
    frames = np.asarray(positive_frames, dtype=np.int32)
    row_count = int(frames.shape[0])
    counts = np.bincount(frames, minlength=2).astype(np.int32)
    offsets = np.zeros(3, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    instances.create_array(
        "refined_row_ids",
        data=np.arange(row_count, dtype=np.int64),
    )
    instances.create_array("frame_indices", data=frames)
    instances.create_array("frame_offsets", data=offsets)
    instances.create_array("frame_counts", data=counts)
    instances.create_array(
        "bbox_img_xyxy",
        data=np.repeat(
            np.asarray([[2.0, 2.0, 4.0, 4.0]], dtype=np.float32),
            row_count,
            axis=0,
        ),
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.repeat(
            np.asarray([[0.375, 0.375, 0.25, 0.25]], dtype=np.float32),
            row_count,
            axis=0,
        ),
    )
    instances.create_array(
        "source_kind_codes",
        data=np.ones(row_count, dtype=np.uint8),
    )
    instances.create_array(
        "manual_edit_flags",
        data=np.zeros(row_count, dtype=np.bool_),
    )
    instances.create_array(
        "source_detect_row_index",
        data=np.arange(row_count, dtype=np.int64),
    )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name="refined_1",
        run_provenance={"schema": "test"},
    )
    run.attrs["stage_selector_eligible"] = False
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
    assert "detect_review_status_latest" not in parent.attrs
    assert parent.attrs["authoritative_run"] == "refined_1"
    assert status["authoritative_approval"]["status"] == "ok"
    profile_parent = root["analysis/detection_profile_runs"]
    profile_run = profile_parent.attrs["latest"]
    profile_group = profile_parent[profile_run]
    assert (
        profile_group.attrs["source_detection_path"]
        == "refined_detect_runs/refined_1/interpolated"
    )
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


def test_accept_detect_review_strict_requires_reviewer_for_approved(
    tmp_path: Path,
) -> None:
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


def test_accept_detect_review_json_output_contains_expected_fields(
    tmp_path: Path, capsys
) -> None:
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
    status = dict(
        root["refined_detect_runs"]["refined_1"].attrs["detect_review_status"]
    )
    assert status["resolved_group"] == "refined"
    profile_group = root["analysis/detection_profile_runs"][
        root["analysis/detection_profile_runs"].attrs["latest"]
    ]
    assert (
        profile_group.attrs["source_detection_path"] == "refined_detect_runs/refined_1"
    )


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
        rows = registry.query_detection_data_profile_latest(
            dataset_ids=["dataset_detect_review"]
        )
    finally:
        registry.close()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["detection_type"] == "interpolated"
    assert row["detection_path"] == "refined_detect_runs/refined_1/interpolated"
    assert row["recording_id"] == "recording_detect_review"
    assert row["zarr_use"] == "training"
    assert row["coverage_percent"] == 100.0


def test_accept_detect_review_approved_is_fail_closed_when_run_is_incomplete(
    tmp_path: Path,
) -> None:
    zarr_path = _make_zarr(tmp_path / "strict.zarr", with_group="interpolated")
    root = zarr.open_group(store=zarr_path, mode="a")
    # Strict completion epoch: unmarked runs are no longer legacy-complete, so
    # the authoritative approval path must block and the CLI must not write
    # detect_review_status.
    root["refined_detect_runs"].attrs["palette_completion_epoch"] = 1

    rc = mod.main(
        [
            str(zarr_path),
            "--state",
            "approved",
            "--intended-use",
            "training",
            "--reviewer",
            "operator5",
        ]
    )
    assert rc == 1

    reopened = zarr.open_group(store=zarr_path, mode="r")
    parent = reopened["refined_detect_runs"]
    assert "detect_review_status" not in parent["refined_1"].attrs
    assert "authoritative_run" not in parent.attrs
    assert "detect_review_status_latest" not in parent.attrs


def test_accept_detect_review_non_approved_state_skips_authoritative_approval(
    tmp_path: Path,
) -> None:
    zarr_path = _make_zarr(tmp_path / "pending.zarr", with_group="interpolated")
    rc = mod.main([str(zarr_path), "--state", "pending"])
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_detect_runs"]
    status = dict(parent["refined_1"].attrs["detect_review_status"])
    assert status["state"] == "pending"
    assert "authoritative_approval" not in status
    assert "authoritative_run" not in parent.attrs
    assert "detect_review_status_latest" not in parent.attrs


def test_selector_ineligible_candidate_approval_materializes_frame_receipt(
    tmp_path: Path,
) -> None:
    zarr_path = _make_selector_ineligible_training_candidate(
        tmp_path / "candidate.zarr"
    )

    rc = mod.main(
        [
            str(zarr_path),
            "--strict",
            "--state",
            "approved",
            "--intended-use",
            "training",
            "--reviewer",
            "operator6",
            "--selector-ineligible-candidate",
            "--json",
        ]
    )

    assert rc == 0
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["refined_detect_runs"]
    run = parent["refined_1"]
    status = dict(run.attrs["detect_review_status"])
    receipt = dict(status["selector_ineligible_candidate_receipt"])
    assert status["state"] == "approved"
    assert status["authority_scope"] == "selector_ineligible_training_candidate"
    assert status["authoritative_approval"]["status"] == (
        "deferred_selector_ineligible"
    )
    assert receipt["status"] == "complete"
    assert receipt["frame_count"] == 2
    assert receipt["positive_frame_count"] == 2
    assert receipt["negative_frame_count"] == 0
    assert len(receipt["frame_decision_digest"]) == 64
    assert receipt["parent_selectors_updated"] is False
    decisions = root[f"{DETECT_FRAME_DECISION_FAMILY}/refined_1"]
    np.testing.assert_array_equal(decisions["decision_codes"][:], [0, 0])
    assert "authoritative_run" not in parent.attrs
    assert "latest" not in parent.attrs
    assert root.attrs["training_artifact_status"] == "review_complete"
    assert "analysis" not in root


def test_selector_ineligible_candidate_rejects_unreviewed_empty_frame(
    tmp_path: Path,
) -> None:
    zarr_path = _make_selector_ineligible_training_candidate(
        tmp_path / "unreviewed.zarr",
        positive_frames=(0,),
    )

    rc = mod.main(
        [
            str(zarr_path),
            "--strict",
            "--state",
            "approved",
            "--intended-use",
            "training",
            "--reviewer",
            "operator7",
            "--selector-ineligible-candidate",
            "--json",
        ]
    )

    assert rc == 1
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_review_status" not in root["refined_detect_runs/refined_1"].attrs
    assert DETECT_FRAME_DECISION_FAMILY not in root
