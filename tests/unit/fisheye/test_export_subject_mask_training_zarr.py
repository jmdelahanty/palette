"""Tests for merged subject-mask-training export and validation."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.utils import check_training_registry as registry_view
from fisheye.utils import validate_subject_mask_training_zarr as validate_cli
from fisheye.utils.export_subject_mask_training_zarr import (
    SubjectMergeSourceSpec,
    export_merged_subject_mask_training_zarr,
    export_merged_subject_mask_training_zarr_from_sources,
    validate_merged_subject_mask_training_zarr,
)


def _write_source_subject_zarr(
    path: Path,
    *,
    dataset_id: str,
    session_uuid: str,
    label_schema_id: str = "subject_v1_lr",
    frame_start: int = 100,
    include_body: bool = False,
    include_swim_bladder: bool = False,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["dataset_id"] = dataset_id
    root.attrs["session_uuid"] = session_uuid

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.create_array(
        "roi_images",
        data=np.zeros((4, 16, 16), dtype=np.uint8),
        chunks=(2, 16, 16),
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "crop_bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "frame_indices",
        data=np.arange(frame_start, frame_start + 4, dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "detection_source",
        data=np.array([0, 1, 0, 0], dtype=np.int8),
        chunks=(4,),
    )
    crop.create_array(
        "source_refined_row_ids",
        data=np.arange(frame_start + 10, frame_start + 14, dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "source_detect_row_index",
        data=np.array([frame_start + 20, -1, frame_start + 22, frame_start + 23], dtype=np.int32),
        chunks=(4,),
    )

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_masks_001"
    subject = subject_parent.create_group("subject_masks_001")
    subject.attrs["source_crop_run"] = "crop_001"
    subject.attrs["label_schema_id"] = label_schema_id
    if label_schema_id == "subject_v1_lr":
        labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
        available = np.array(
            [include_body, True, True, include_swim_bladder],
            dtype=np.bool_,
        )
        masks = np.zeros((4, 4, 16, 16), dtype=np.uint8)
        masks[:, 1, 4:7, 4:7] = 1
        masks[:, 2, 4:7, 9:12] = 1
        if include_body:
            masks[:, 0, 2:10, 2:14] = 1
        if include_swim_bladder:
            masks[:, 3, 5:8, 7:10] = 1
    else:
        labels = ["subject_body", "eyes_union", "swim_bladder"]
        available = np.array(
            [include_body, True, include_swim_bladder],
            dtype=np.bool_,
        )
        masks = np.zeros((4, 3, 16, 16), dtype=np.uint8)
        masks[:, 1, 4:7, 4:7] = 1
        masks[:, 1, 4:7, 9:12] = 1
        if include_body:
            masks[:, 0, 2:10, 2:14] = 1
        if include_swim_bladder:
            masks[:, 2, 5:8, 7:10] = 1

    subject.attrs["mask_labels"] = labels
    subject.create_array("masks_roi", data=masks, chunks=(2, len(labels), 16, 16))
    subject.create_array("available_channels", data=available, chunks=(len(labels),))


def _write_refined_subject_source(
    path: Path,
    *,
    run_name: str = "refined_subject_masks_001",
    review_state: str = "approved",
) -> None:
    root = zarr.open_group(str(path), mode="a")
    subject = root["subject_mask_runs/subject_masks_001"]
    parent = root.require_group("refined_subject_masks_runs")
    parent.attrs["latest"] = run_name
    refined = parent.create_group(run_name)
    labels = list(subject.attrs["mask_labels"])
    refined.attrs.update(
        {
            "source_subject_mask_run": "subject_masks_001",
            "source_crop_run": "crop_001",
            "label_schema_id": subject.attrs["label_schema_id"],
            "mask_labels": labels,
            "method": "manual_refined_subject_masks",
        }
    )
    masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    masks[:, 0, 1:15, 1:15] = 1
    refined.create_array("masks_roi", data=masks, chunks=subject["masks_roi"].chunks)
    refined.create_array(
        "available_channels",
        data=np.asarray(subject["available_channels"][:], dtype=np.bool_),
        chunks=subject["available_channels"].chunks,
    )
    available = np.asarray(subject["available_channels"][:], dtype=np.bool_)
    review_payload = {
        label: {
            "state": review_state,
            "method": "manual",
            "intended_use": "training",
            "reviewer": "pytest",
            "timestamp_utc": "2026-04-03T00:00:00+00:00",
        }
        for index, label in enumerate(labels)
        if index < int(available.shape[0]) and bool(available[index])
    }
    refined.attrs["component_review_statuses"] = review_payload
    refined.attrs["refined_subject_mask_review_status"] = {
        "state": review_state,
        "method": "manual",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp_utc": "2026-04-03T00:00:00+00:00",
    }


def _replace_refined_masks_with_rle(path: Path, *, run_name: str = "refined_subject_masks_001") -> np.ndarray:
    root = zarr.open_group(str(path), mode="a")
    refined = root[f"refined_subject_masks_runs/{run_name}"]
    masks = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    labels = [str(label) for label in refined.attrs["mask_labels"]]
    del refined["masks_roi"]
    write_component_rle_mask_store_from_dense(
        refined,
        masks,
        component_names=labels,
        encode_row_chunk_size=2,
    )
    return masks


def test_export_merged_subject_mask_training_zarr_then_validate(tmp_path: Path) -> None:
    source_path = tmp_path / "source_subject_lr.zarr"
    out_path = tmp_path / "merged_subject_lr.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_a",
        session_uuid="subject_source_a",
    )

    summary = export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        subject_label_schema="subject_v1_lr",
        overwrite=True,
    )

    assert summary["total_samples"] == 4
    assert summary["channels"] == 4
    assert summary["coverage_class"] == "eyes_only"
    assert summary["source_subject_mask_run"] == "subject_masks_001"
    assert summary["source_crop_run"] == "crop_001"

    recheck = validate_merged_subject_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_schema_id="subject_v1_lr",
    )
    assert recheck["total_samples"] == 4

    root = zarr.open_group(str(out_path), mode="r")
    source_index = root["source_index"]
    assert source_index.attrs["label_origin_codebook"] == {
        "unknown": 0,
        "auto": 1,
        "manual_review": 2,
        "manual_training": 3,
        "interpolated": 4,
        "synthetic": 5,
    }
    assert source_index.attrs["supervision_mode_codebook"] == {
        "no_supervision": 0,
        "dense": 1,
        "explicit_negative": 2,
        "box_only": 3,
    }
    assert np.asarray(root["source_index/source_refined_row_ids"][:], dtype=np.int64).tolist() == [
        110,
        111,
        112,
        113,
    ]
    assert np.asarray(root["source_index/source_detect_row_index"][:], dtype=np.int64).tolist() == [
        120,
        -1,
        122,
        123,
    ]
    assert root["source_index/source_crop_run"][:].tolist() == ["crop_001"]
    assert root["source_index/source_mask_store_encoding"][:].tolist() == ["dense_uint8"]
    assert root["source_index/source_mask_storage_surface"][:].tolist() == ["masks_roi"]
    assert recheck["channels"] == 4

    latest = str(root["subject_mask_runs"].attrs["latest"])
    run = root[f"subject_mask_runs/{latest}"]
    crop_latest = str(root["crop_runs"].attrs["latest"])
    crop_run = root[f"crop_runs/{crop_latest}"]
    assert crop_run.attrs["source_crop_run"] == "crop_001"
    assert crop_run.attrs["source_crop_runs"] == ["crop_001"]
    assert crop_run.attrs["source_zarr_path"] == str(source_path.resolve())
    assert crop_run.attrs["source_zarr_paths"] == [str(source_path.resolve())]
    target_valid = np.asarray(run["target_valid_channels"][:], dtype=np.bool_)
    assert target_valid.shape == (4, 4)
    assert target_valid[:, 0].tolist() == [False, False, False, False]
    assert target_valid[:, 1].tolist() == [True, True, True, True]
    assert target_valid[:, 2].tolist() == [True, True, True, True]
    assert target_valid[:, 3].tolist() == [False, False, False, False]

    export_meta = root.attrs["training_export"]
    assert export_meta["source_subject_mask_run"] == "subject_masks_001"
    assert export_meta["source_subject_mask_runs"] == ["subject_masks_001"]
    assert export_meta["source_crop_run"] == "crop_001"
    assert export_meta["source_crop_runs"] == ["crop_001"]
    assert export_meta["channel_supervision_summary"]["contains_only_eye_masks"] is True
    assert export_meta["channel_supervision_summary"]["supervised_row_counts"] == {
        "subject_body": 0,
        "eye_left": 4,
        "eye_right": 4,
        "swim_bladder": 0,
    }


def test_export_subject_mask_training_reads_refined_subject_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source_subject_refined.zarr"
    out_path = tmp_path / "merged_subject_refined.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_refined",
        session_uuid="subject_source_refined",
        include_body=True,
        include_swim_bladder=True,
    )
    _write_refined_subject_source(source_path)

    summary = export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        source_stage_group="refined_subject_masks_runs",
        subject_run="refined_subject_masks_001",
        subject_label_schema="subject_v1_union",
        overwrite=True,
    )

    assert summary["total_samples"] == 4
    assert summary["channels"] == 3
    assert summary["source_mask_store_encoding"] == "dense_uint8"
    assert summary["source_mask_storage_surface"] == "masks_roi"
    assert summary["coverage_class"] == "dense_all_components"
    assert summary["source_subject_mask_run"] == "refined_subject_masks_001"

    recheck = validate_merged_subject_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_schema_id="subject_v1_union",
    )
    assert recheck["channels"] == 3

    root = zarr.open_group(str(out_path), mode="r")
    source_index = root["source_index"]
    assert source_index["source_stage_group"][:].tolist() == ["refined_subject_masks_runs"]
    latest = str(root["subject_mask_runs"].attrs["latest"])
    run = root[f"subject_mask_runs/{latest}"]
    assert run.attrs["source_mask_stage"] == "refined_subject_masks_runs"
    assert run.attrs["source_subject_mask_run"] == "refined_subject_masks_001"
    assert root.attrs["training_export"]["source_stage"] == "refined_subject_masks_runs"
    assert root.attrs["training_export"]["source_stage_groups"] == ["refined_subject_masks_runs"]
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    valid = np.asarray(run["target_valid_channels"][:], dtype=np.bool_)
    assert valid.tolist() == [[True, True, True]] * 4
    assert int(np.sum(masks[:, 0])) > 0
    assert int(np.sum(masks[:, 1])) > 0
    assert int(np.sum(masks[:, 2])) > 0


def test_export_subject_mask_training_reads_compact_refined_subject_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source_subject_refined_compact.zarr"
    out_path = tmp_path / "merged_subject_refined_compact.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_refined_compact",
        session_uuid="subject_source_refined_compact",
        include_body=True,
        include_swim_bladder=True,
    )
    _write_refined_subject_source(source_path)
    expected_source_masks = _replace_refined_masks_with_rle(source_path)

    summary = export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        source_stage_group="refined_subject_masks_runs",
        subject_run="refined_subject_masks_001",
        subject_label_schema="subject_v1_union",
        overwrite=True,
    )

    assert summary["total_samples"] == 4
    assert summary["channels"] == 3
    assert summary["source_mask_store_encoding"] == "component_rle_v1"
    assert summary["source_mask_storage_surface"] == "mask_rle"

    root = zarr.open_group(str(out_path), mode="r")
    latest = str(root["subject_mask_runs"].attrs["latest"])
    run = root[f"subject_mask_runs/{latest}"]
    assert "masks_roi" in run
    assert "mask_rle" not in run
    assert run.attrs["mask_storage_format"] == "dense_uint8"
    assert run.attrs["mask_storage_surface"] == "masks_roi"
    assert run.attrs["mask_store_encoding"] == "dense_uint8"
    assert run.attrs["source_mask_store_encoding"] == "component_rle_v1"
    assert run.attrs["source_mask_store_encodings"] == ["component_rle_v1"]
    assert root.attrs["training_export"]["mask_storage_format"] == "dense_uint8"
    assert root.attrs["training_export"]["mask_storage_surface"] == "masks_roi"
    assert root.attrs["training_export"]["source_mask_store_encodings"] == ["component_rle_v1"]
    assert root.attrs["training_export"]["source_mask_storage_surface"] == "mask_rle"
    assert root.attrs["training_export"]["source_mask_storage_surfaces"] == ["mask_rle"]
    assert root["source_index/source_crop_run"][:].tolist() == ["crop_001"]
    assert root["source_index/source_mask_store_encoding"][:].tolist() == ["component_rle_v1"]
    assert root["source_index/source_mask_storage_surface"][:].tolist() == ["mask_rle"]

    exported_masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    expected_union = np.zeros((4, 3, 16, 16), dtype=np.uint8)
    expected_union[:, 0] = expected_source_masks[:, 0]
    expected_union[:, 1] = np.maximum(expected_source_masks[:, 1], expected_source_masks[:, 2])
    expected_union[:, 2] = expected_source_masks[:, 3]
    assert np.array_equal(exported_masks, expected_union)


def test_validate_subject_mask_training_rejects_compact_training_mask_surface(tmp_path: Path) -> None:
    source_path = tmp_path / "source_subject_refined_compact_invalid.zarr"
    out_path = tmp_path / "merged_subject_refined_compact_invalid.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_refined_compact_invalid",
        session_uuid="subject_source_refined_compact_invalid",
        include_body=True,
        include_swim_bladder=True,
    )
    _write_refined_subject_source(source_path)

    export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        source_stage_group="refined_subject_masks_runs",
        subject_run="refined_subject_masks_001",
        subject_label_schema="subject_v1_union",
        overwrite=True,
    )

    root = zarr.open_group(str(out_path), mode="a")
    latest = str(root["subject_mask_runs"].attrs["latest"])
    root[f"subject_mask_runs/{latest}"].create_group("mask_rle")

    with pytest.raises(ValueError, match="compact mask_rle is analysis-only"):
        validate_merged_subject_mask_training_zarr(out_path)


def test_export_subject_mask_training_rejects_unapproved_refined_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source_subject_refined_pending.zarr"
    out_path = tmp_path / "merged_subject_refined_pending.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_refined_pending",
        session_uuid="subject_source_refined_pending",
        include_body=True,
        include_swim_bladder=True,
    )
    _write_refined_subject_source(source_path, review_state="pending")

    with pytest.raises(ValueError, match="component 'subject_body' is not approved"):
        export_merged_subject_mask_training_zarr(
            source_path,
            out_path,
            source_stage_group="refined_subject_masks_runs",
            subject_run="refined_subject_masks_001",
            subject_label_schema="subject_v1_union",
            overwrite=True,
        )

    summary = export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        source_stage_group="refined_subject_masks_runs",
        subject_run="refined_subject_masks_001",
        subject_label_schema="subject_v1_union",
        overwrite=True,
        allow_unapproved_refined=True,
    )
    assert summary["total_samples"] == 4


def test_validate_subject_mask_training_zarr_cli(tmp_path: Path, capsys) -> None:
    source_path = tmp_path / "source_subject_lr.zarr"
    out_path = tmp_path / "merged_subject_lr.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_a",
        session_uuid="subject_source_a",
    )
    export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        subject_label_schema="subject_v1_lr",
        overwrite=True,
    )

    assert validate_cli.main(
        [
            str(out_path),
            "--expected-input-format",
            "gray",
            "--expected-label-schema-id",
            "subject_v1_lr",
            "--expected-total-samples",
            "4",
        ]
    ) == 0
    stdout = capsys.readouterr().out
    assert "Validation passed." in stdout
    assert '"label_schema_id": "subject_v1_lr"' in stdout


def test_validate_subject_mask_training_zarr_rejects_bad_codebook(tmp_path: Path) -> None:
    source_path = tmp_path / "source_subject_lr.zarr"
    out_path = tmp_path / "merged_subject_lr.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_a",
        session_uuid="subject_source_a",
    )
    export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        subject_label_schema="subject_v1_lr",
        overwrite=True,
    )
    root = zarr.open_group(str(out_path), mode="a")
    root["source_index"].attrs["label_origin_codebook"] = {"unknown": 0}

    try:
        validate_merged_subject_mask_training_zarr(out_path)
    except ValueError as exc:
        assert "label_origin_codebook" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("Expected bad label_origin_codebook to fail validation.")


def test_export_subject_mask_lr_to_union_collapses_eyes_and_preserves_unsupervised_channels(tmp_path: Path) -> None:
    source_path = tmp_path / "source_subject_lr.zarr"
    out_path = tmp_path / "merged_subject_union.zarr"
    _write_source_subject_zarr(
        source_path,
        dataset_id="subject_source_a",
        session_uuid="subject_source_a",
    )

    summary = export_merged_subject_mask_training_zarr(
        source_path,
        out_path,
        subject_label_schema="subject_v1_union",
        overwrite=True,
    )

    assert summary["channels"] == 3
    assert summary["coverage_class"] == "eyes_only"

    recheck = validate_merged_subject_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_schema_id="subject_v1_union",
    )
    assert recheck["channels"] == 3

    root = zarr.open_group(str(out_path), mode="r")
    latest = str(root["subject_mask_runs"].attrs["latest"])
    masks = np.asarray(root[f"subject_mask_runs/{latest}/masks_roi"][:], dtype=np.uint8)
    valid = np.asarray(root[f"subject_mask_runs/{latest}/target_valid_channels"][:], dtype=np.bool_)
    assert masks.shape == (4, 3, 16, 16)
    assert valid[:, 0].tolist() == [False, False, False, False]
    assert valid[:, 1].tolist() == [True, True, True, True]
    assert valid[:, 2].tolist() == [False, False, False, False]
    assert int(np.sum(masks[:, 1])) > 0


def test_subject_mask_export_registry_marks_set_as_eyes_only(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    source_a = tmp_path / "source_a.zarr"
    source_b = tmp_path / "source_b.zarr"
    out_path = tmp_path / "merged_subject_set.zarr"
    _write_source_subject_zarr(
        source_a,
        dataset_id="subject_source_a",
        session_uuid="subject_source_a",
        frame_start=100,
    )
    _write_source_subject_zarr(
        source_b,
        dataset_id="subject_source_b",
        session_uuid="subject_source_b",
        frame_start=200,
    )

    registry = Registry(registry_path)
    registry.upsert_dataset(
        "subject_source_a",
        session_uuid="subject_source_a",
        zarr_path=source_a,
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "subject_source_b",
        session_uuid="subject_source_b",
        zarr_path=source_b,
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_training_set(
        set_id="subject_masks_my_set_v001",
        name="subject mask set",
        task_type="subject_masks",
        query_filter={"task_type": "subject_masks"},
        dataset_ids=["subject_source_a", "subject_source_b"],
        invocation={"task_type": "subject_masks"},
    )
    registry.close()

    summary = export_merged_subject_mask_training_zarr_from_sources(
        source_specs=[
            SubjectMergeSourceSpec(source_zarr=source_a),
            SubjectMergeSourceSpec(source_zarr=source_b),
        ],
        out_zarr=out_path,
        subject_label_schema="subject_v1_lr",
        overwrite=True,
        registry=registry_path,
        training_set_id="subject_masks_my_set_v001",
        training_set_name="subject mask set",
    )
    assert summary["total_samples"] == 8
    assert summary["contains_only_eye_masks"] is True

    db = Registry(registry_path)
    set_row = db.conn.execute(
        "SELECT task_type, invocation_json, dataset_ids_json FROM training_sets WHERE set_id = ?",
        ("subject_masks_my_set_v001",),
    ).fetchone()
    assert set_row is not None
    assert set_row["task_type"] == "subject_masks"
    set_invocation = json.loads(set_row["invocation_json"])
    assert set_invocation["subject_mask_training_summary"]["coverage_class"] == "eyes_only"
    assert set_invocation["subject_mask_training_summary"]["contains_only_eye_masks"] is True
    dataset_ids = sorted(json.loads(set_row["dataset_ids_json"]))
    assert dataset_ids == sorted(
        [
            "subject_source_a",
            "subject_source_b",
            "subject_masks_my_set_v001_merged",
        ]
    )

    set_rows = registry_view._load_set_rows(db, "subject_masks_my_set_v001", limit=10)  # noqa: SLF001
    assert len(set_rows) == 1
    assert set_rows[0].task_type == "subject_masks"
    assert set_rows[0].data_summary == "eyes only"
    db.close()
