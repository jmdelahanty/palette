"""Tests for merged eye-mask-training Zarr validation and export scaffold."""

from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import export_eye_mask_training_zarr as export_mod
from fisheye.utils.export_eye_mask_training_zarr import (
    export_merged_eye_mask_training_zarr,
    validate_merged_eye_mask_training_zarr,
)
from fisheye.shared.detect_reason_codec import decode_reason_bytes


def _write_valid_merged_eye_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    root.attrs["training_task"] = "eye_masks"
    root.attrs["training_export"] = {"input_format": "gray", "label_mode": "lr"}

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "merged_export_smoke"
    crop = crop_parent.create_group("merged_export_smoke")
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
        data=np.arange(4, dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "detection_source",
        data=np.array([0, 1, 0, 0], dtype=np.int8),
        chunks=(4,),
    )

    eye_parent = root.create_group("eye_masks_runs")
    eye_parent.attrs["latest"] = "merged_export_smoke"
    eye = eye_parent.create_group("merged_export_smoke")

    masks = np.zeros((4, 2, 16, 16), dtype=np.uint8)
    masks[0, 0, 4:8, 4:8] = 1
    masks[0, 1, 4:8, 9:13] = 1
    masks[1, 0, 4:8, 4:8] = 1
    masks[1, 1, 4:8, 9:13] = 1
    masks[2, 0, 4:8, 4:8] = 1
    masks[2, 1, 4:8, 9:13] = 1
    masks[3, 0, 4:8, 4:8] = 1
    masks[3, 1, 4:8, 9:13] = 1
    eye.create_array("masks_roi", data=masks, chunks=(2, 2, 16, 16))

    ellipse_params = np.full((4, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.array(
        [
            [True, True],
            [True, True],
            [True, True],
            [False, False],
        ],
        dtype=np.bool_,
    )
    ellipse_params[0, 0] = np.array([6.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[0, 1] = np.array([11.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[1, 0] = np.array([6.0, 6.0, 9.0, 6.0, 5.0], dtype=np.float32)
    ellipse_params[1, 1] = np.array([11.0, 6.0, 9.0, 6.0, 5.0], dtype=np.float32)
    ellipse_params[2, 0] = np.array([6.0, 6.0, 7.5, 5.5, 10.0], dtype=np.float32)
    ellipse_params[2, 1] = np.array([11.0, 6.0, 7.5, 5.5, 10.0], dtype=np.float32)
    eye.create_array("ellipse_params", data=ellipse_params, chunks=(4, 2, 5))
    eye.create_array("ellipse_success", data=ellipse_success, chunks=(4, 2))
    eye.create_array("eye_separation", data=np.array([5.0, 5.0, 5.0, np.nan], dtype=np.float32), chunks=(4,))
    eye.create_array("frame_indices", data=np.arange(4, dtype=np.int64), chunks=(4,))
    eye.create_array("detection_source", data=np.array([0, 1, 0, 0], dtype=np.int8), chunks=(4,))
    export_mod.write_reason_columns(
        eye,
        np.asarray(["clean", "clean", "manual_correction|clean", "incomplete"], dtype=object),
        chunk_size=4,
        include_reason_text=True,
        overwrite=True,
    )

    splits = root.create_group("splits")
    splits.create_array("train_indices", data=np.array([0, 1], dtype=np.int64), chunks=(2,))
    splits.create_array("val_indices", data=np.array([2], dtype=np.int64), chunks=(1,))
    splits.create_array("test_indices", data=np.array([3], dtype=np.int64), chunks=(1,))

    source = root.create_group("source_index")
    source.create_array(
        "source_dataset_idx",
        data=np.array([0, 0, 0, 0], dtype=np.int32),
        chunks=(4,),
    )
    source.create_array(
        "source_frame_idx",
        data=np.array([10, 11, 12, 13], dtype=np.int64),
        chunks=(4,),
    )
    source.create_array(
        "source_roi_idx",
        data=np.arange(4, dtype=np.int64),
        chunks=(4,),
    )
    export_mod._write_string_array(source, "source_dataset_id", ["dataset_a"])
    export_mod._write_string_array(source, "source_zarr_path", ["/tmp/source_a.zarr"])


def _write_source_eye_zarr(
    path: Path,
    *,
    dataset_id: str | None = None,
    session_uuid: str | None = None,
    frame_start: int = 100,
    ellipse_success: np.ndarray | None = None,
    reason_labels: list[str] | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    if dataset_id is not None:
        root.attrs["dataset_id"] = str(dataset_id)
    if session_uuid is not None:
        root.attrs["session_uuid"] = str(session_uuid)
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
        data=np.array([frame_start + 0, frame_start + 1, frame_start + 2, frame_start + 3], dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "detection_source",
        data=np.array([0, 1, 0, 0], dtype=np.int8),
        chunks=(4,),
    )

    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"
    refined = refined_parent.create_group("refined_eye_masks_001")
    refined.attrs["source_crop_run"] = "crop_001"
    refined.attrs["method"] = "traditional_eye_segmentation"
    refined.attrs["eye_labels"] = ["eye_left", "eye_right"]
    refined.attrs["source_keypoints_run"] = "refined_keypoints_001"
    refined.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    refined.create_array(
        "masks_roi",
        data=np.zeros((4, 2, 16, 16), dtype=np.uint8),
        chunks=(2, 2, 16, 16),
    )
    ellipse_params = np.full((4, 2, 5), np.nan, dtype=np.float32)
    ellipse_params[0, 0] = np.array([6.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[0, 1] = np.array([11.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[1, 0] = np.array([6.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[1, 1] = np.array([11.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    if ellipse_success is None:
        ellipse_success_data = np.array([[True, True], [True, True], [False, False], [False, False]], dtype=np.bool_)
    else:
        ellipse_success_data = np.asarray(ellipse_success, dtype=np.bool_)
    for row_idx in range(int(ellipse_success_data.shape[0])):
        for eye_idx in range(int(ellipse_success_data.shape[1])):
            if not bool(ellipse_success_data[row_idx, eye_idx]):
                continue
            if np.all(np.isfinite(ellipse_params[row_idx, eye_idx])):
                continue
            center_x = 6.0 if eye_idx == 0 else 11.0
            ellipse_params[row_idx, eye_idx] = np.array([center_x, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    refined.create_array("ellipse_params", data=ellipse_params, chunks=(4, 2, 5))
    refined.create_array(
        "ellipse_success",
        data=ellipse_success_data,
        chunks=(4, 2),
    )
    refined.create_array("eye_separation", data=np.array([5.0, 5.0, np.nan, np.nan], dtype=np.float32), chunks=(4,))
    metrics = refined.create_group("metrics")
    if reason_labels is None:
        reason_labels = ["clean", "clean", "incomplete", "incomplete"]
    export_mod.write_reason_columns(
        metrics,
        np.asarray(reason_labels, dtype=object),
        chunk_size=4,
        include_reason_text=True,
        overwrite=True,
    )


def test_validate_merged_eye_mask_training_zarr_passes(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_eye_ok.zarr"
    _write_valid_merged_eye_zarr(zarr_path)

    summary = validate_merged_eye_mask_training_zarr(
        zarr_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_mode="lr",
    )

    assert summary["total_samples"] == 4
    assert summary["channels"] == 2
    assert summary["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert summary["source_count"] == 1


def test_validate_merged_eye_mask_training_zarr_rejects_invalid_frame_indices(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_eye_bad.zarr"
    _write_valid_merged_eye_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    latest = root["crop_runs"].attrs["latest"]
    root[f"crop_runs/{latest}/frame_indices"][:] = np.array([1, 2, 3, 4], dtype=np.int64)

    with pytest.raises(ValueError, match="frame_indices"):
        validate_merged_eye_mask_training_zarr(
            zarr_path,
            expected_input_format="gray",
            expected_total_samples=4,
            expected_label_mode="lr",
        )


def test_export_merged_eye_mask_training_zarr_then_validate(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path)

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        overwrite=True,
    )

    assert summary["total_samples"] == 4
    assert summary["channels"] == 2

    recheck = validate_merged_eye_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_mode="lr",
    )
    assert recheck["total_samples"] == 4
    out_root = zarr.open_group(str(out_path), mode="r")
    eye_latest = str(out_root["eye_masks_runs"].attrs["latest"])
    eye = out_root[f"eye_masks_runs/{eye_latest}"]
    assert "reason_bytes" in eye
    decoded = decode_reason_bytes(np.asarray(eye["reason_bytes"][:], dtype=np.uint8)).tolist()
    assert decoded == ["clean", "clean", "incomplete", "incomplete"]
    assert eye.attrs["reason_encoding"] == "utf8-null-terminated"
    assert eye.attrs["reason_fallback_order"] == ["reason_bytes", "reason", "detection_source"]


def test_export_merged_eye_mask_training_zarr_from_multiple_sources_tracks_source_index(tmp_path: Path) -> None:
    source_a = tmp_path / "source_a.zarr"
    source_b = tmp_path / "source_b.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(
        source_a,
        dataset_id="dataset_a",
        session_uuid="session_a",
        frame_start=100,
    )
    _write_source_eye_zarr(
        source_b,
        dataset_id="dataset_b",
        session_uuid="session_b",
        frame_start=200,
    )

    summary = export_mod.export_merged_eye_mask_training_zarr_from_sources(
        source_specs=[
            export_mod.EyeMergeSourceSpec(
                source_zarr=source_a,
                eye_stage="refined_eye_masks_runs",
                eye_run="refined_eye_masks_001",
            ),
            export_mod.EyeMergeSourceSpec(
                source_zarr=source_b,
                eye_stage="refined_eye_masks_runs",
                eye_run="refined_eye_masks_001",
            ),
        ],
        out_zarr=out_path,
        overwrite=True,
    )
    assert summary["total_samples"] == 8
    assert summary["source_count"] == 2

    recheck = validate_merged_eye_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=8,
        expected_label_mode="lr",
    )
    assert recheck["source_count"] == 2

    root = zarr.open_group(str(out_path), mode="r")
    source_dataset_idx = np.asarray(root["source_index/source_dataset_idx"][:], dtype=np.int64).tolist()
    source_frame_idx = np.asarray(root["source_index/source_frame_idx"][:], dtype=np.int64).tolist()
    source_roi_idx = np.asarray(root["source_index/source_roi_idx"][:], dtype=np.int64).tolist()
    source_dataset_ids = np.asarray(root["source_index/source_dataset_id"][:], dtype=object).tolist()
    source_zarr_paths = np.asarray(root["source_index/source_zarr_path"][:], dtype=object).tolist()

    assert source_dataset_idx == [0, 0, 0, 0, 1, 1, 1, 1]
    assert source_frame_idx == [100, 101, 102, 103, 200, 201, 202, 203]
    assert source_roi_idx == [0, 1, 2, 3, 0, 1, 2, 3]
    assert source_dataset_ids == ["dataset_a", "dataset_b"]
    assert source_zarr_paths == [str(source_a.resolve()), str(source_b.resolve())]


def test_export_union_label_mode_collapses_two_channel_source_masks(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_union_export.zarr"
    _write_source_eye_zarr(source_path)

    source_root = zarr.open_group(str(source_path), mode="a")
    refined = source_root["refined_eye_masks_runs/refined_eye_masks_001"]
    masks_src = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    masks_src[0, 0, 2:4, 2:4] = 1
    masks_src[0, 1, 10:12, 10:12] = 1
    masks_src[1, 0, 3:6, 3:6] = 1
    masks_src[1, 1, 9:12, 9:12] = 1
    refined["masks_roi"][:] = masks_src

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        label_mode="union",
        overwrite=True,
    )
    assert summary["channels"] == 1

    recheck = validate_merged_eye_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_mode="union",
    )
    assert recheck["channels"] == 1

    out_root = zarr.open_group(str(out_path), mode="r")
    eye_latest = str(out_root["eye_masks_runs"].attrs["latest"])
    masks_out = np.asarray(out_root[f"eye_masks_runs/{eye_latest}/masks_roi"][:], dtype=np.uint8)
    expected_union = np.max(masks_src, axis=1, keepdims=True).astype(np.uint8, copy=False)
    np.testing.assert_array_equal(masks_out, expected_union)


def test_export_row_gate_usable_only_keeps_pair_success_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_usable_only.zarr"
    _write_source_eye_zarr(
        source_path,
        ellipse_success=np.array(
            [[True, True], [False, False], [False, False], [True, True]],
            dtype=np.bool_,
        ),
        reason_labels=[
            "clean",
            "fish_present_no_keypoints",
            "fish_present_no_keypoints",
            "clean",
        ],
    )

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        row_gate_policy="usable_only",
        overwrite=True,
    )
    assert summary["total_samples"] == 2
    assert summary["row_gate"]["applied_policy"] == "usable_only"
    assert summary["row_gate"]["selected_rows"] == 2
    assert summary["row_gate"]["total_rows"] == 4

    recheck = validate_merged_eye_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=2,
        expected_label_mode="lr",
    )
    assert recheck["total_samples"] == 2

    root = zarr.open_group(str(out_path), mode="r")
    source_roi_idx = np.asarray(root["source_index/source_roi_idx"][:], dtype=np.int64).tolist()
    assert source_roi_idx == [0, 3]
    eye_latest = str(root["eye_masks_runs"].attrs["latest"])
    eye = root[f"eye_masks_runs/{eye_latest}"]
    assert eye.attrs["row_gate_policy"] == "usable_only"
    assert bool(eye.attrs["row_gate_applied"]) is True


def test_export_row_gate_usable_plus_explicit_negatives_caps_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_usable_plus_neg.zarr"
    _write_source_eye_zarr(
        source_path,
        ellipse_success=np.array(
            [[True, True], [False, False], [False, False], [True, True]],
            dtype=np.bool_,
        ),
        reason_labels=[
            "clean",
            "fish_present_no_keypoints",
            "fish_present_no_keypoints",
            "clean",
        ],
    )

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        row_gate_policy="usable_plus_explicit_negatives",
        explicit_negative_ratio=0.5,
        split_seed=7,
        overwrite=True,
    )
    assert summary["total_samples"] == 3
    assert summary["row_gate"]["applied_policy"] == "usable_plus_explicit_negatives"
    assert summary["row_gate"]["pair_success_rows"] == 2
    assert summary["row_gate"]["explicit_negative_rows"] == 2
    assert summary["row_gate"]["explicit_negative_selected_rows"] == 1

    root = zarr.open_group(str(out_path), mode="r")
    source_roi_idx = np.asarray(root["source_index/source_roi_idx"][:], dtype=np.int64).tolist()
    assert source_roi_idx[0] == 0
    assert source_roi_idx[-1] == 3
    assert len(source_roi_idx) == 3
    assert set(source_roi_idx).issubset({0, 1, 2, 3})


def test_export_records_deterministic_data_card_artifact_paths(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path)

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        run_name="refined_eye_masks_001",
        overwrite=True,
    )

    artifact_paths = summary["artifact_paths"]
    expected_card = out_path.with_suffix(".data_card.json")
    expected_plot_dir = expected_card.parent / f"{expected_card.stem}.plots"
    assert Path(artifact_paths["data_card_json"]) == expected_card
    assert Path(artifact_paths["data_card_plot_dir"]) == expected_plot_dir
    assert artifact_paths["data_card_plot_prefix"] == "refined_eye_masks_001"

    root = zarr.open_group(str(out_path), mode="r")
    training_export = dict(root.attrs["training_export"])
    attr_artifacts = dict(training_export["artifact_paths"])
    assert Path(attr_artifacts["data_card_json"]) == expected_card
    assert Path(attr_artifacts["data_card_plot_dir"]) == expected_plot_dir
    assert attr_artifacts["data_card_plot_prefix"] == "refined_eye_masks_001"


def test_export_source_dataset_id_prefers_dataset_id_attr(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(
        source_path,
        dataset_id="dataset_manifest_entry",
        session_uuid="session_should_not_win",
    )

    export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        overwrite=True,
    )

    root = zarr.open_group(str(out_path), mode="r")
    source_dataset_ids = np.asarray(root["source_index/source_dataset_id"][:], dtype=object).tolist()
    assert source_dataset_ids == ["dataset_manifest_entry"]


def test_export_source_dataset_id_uses_session_uuid_when_dataset_id_missing(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path, session_uuid="session_source_001")

    export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        overwrite=True,
    )

    root = zarr.open_group(str(out_path), mode="r")
    source_dataset_ids = np.asarray(root["source_index/source_dataset_id"][:], dtype=object).tolist()
    assert source_dataset_ids == ["session_source_001"]


def test_export_aggregation_runs_profile_sync_before_card_commands(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path)
    calls: list[list[str]] = []

    class _Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = int(returncode)
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, check=False, capture_output=False, text=False):
        del check, capture_output, text
        calls.append([str(part) for part in cmd])
        return _Completed(0)

    monkeypatch.setattr(export_mod.subprocess, "run", _fake_run)

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        overwrite=True,
        training_set_id="eye_mask_set_v001",
        aggregate_training_data_card=True,
    )

    assert len(calls) == 3
    assert calls[0][:3] == ["scripts/py", "-m", "fisheye.utils.sync_eye_mask_profile_registry"]
    assert "--apply" in calls[0]
    assert calls[1][:3] == ["scripts/py", "-m", "fisheye.utils.aggregate_eye_mask_training_data_card"]
    assert calls[2][:3] == ["scripts/py", "-m", "fisheye.utils.plot_eye_mask_training_data_card"]
    assert summary["data_card_workflow"]["plots_generated"] is True


def test_export_aggregation_sync_failure_has_clear_remediation(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path)

    class _Completed:
        def __init__(self, returncode: int, stderr: str = "") -> None:
            self.returncode = int(returncode)
            self.stdout = ""
            self.stderr = stderr

    call_count = {"value": 0}

    def _fake_run(cmd, check=False, capture_output=False, text=False):
        del cmd, check, capture_output, text
        call_count["value"] += 1
        if call_count["value"] == 1:
            return _Completed(2, stderr="profile sync failed")
        return _Completed(0)

    monkeypatch.setattr(export_mod.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="sync_eye_mask_profile_registry.*Remediation"):
        export_merged_eye_mask_training_zarr(
            source_path,
            out_path,
            eye_stage="refined_eye_masks_runs",
            eye_run="refined_eye_masks_001",
            overwrite=True,
            training_set_id="eye_mask_set_v001",
            aggregate_training_data_card=True,
        )


def test_export_aggregation_requires_training_set_id(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path)

    with pytest.raises(ValueError, match="aggregate_training_data_card requires training_set_id"):
        export_merged_eye_mask_training_zarr(
            source_path,
            out_path,
            eye_stage="refined_eye_masks_runs",
            eye_run="refined_eye_masks_001",
            overwrite=True,
            aggregate_training_data_card=True,
        )


def test_export_registers_merged_dataset_in_registry_when_requested(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path)

    class _Cursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class _Conn:
        def execute(self, sql, params=()):
            del params
            if "FROM training_sets" in sql:
                return _Cursor(None)
            return _Cursor(None)

    class _FakeRegistry:
        instances: list["_FakeRegistry"] = []

        def __init__(self, registry_path):
            self.registry_path = Path(registry_path)
            self.conn = _Conn()
            self.upsert_calls: list[dict[str, object]] = []
            self.lineage_calls: list[dict[str, object]] = []
            _FakeRegistry.instances.append(self)

        def scan_zarr(self, zarr_path):
            name = Path(zarr_path).name
            if name == "merged_eye_export.zarr":
                return "dataset_merged"
            return "dataset_source"

        def upsert_training_set(self, **kwargs):
            self.upsert_calls.append(dict(kwargs))

        def replace_dataset_lineage(self, **kwargs):
            self.lineage_calls.append(dict(kwargs))

        def close(self):
            return None

    monkeypatch.setattr(export_mod, "Registry", _FakeRegistry)

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        overwrite=True,
        registry=tmp_path / "registry.sqlite",
        training_set_id="eye_mask_set_v001",
        training_set_name="Eye Mask Set",
    )

    assert summary["registry_registration"]["merged_dataset_id"] == "dataset_merged"
    assert summary["registry_registration"]["source_dataset_ids"] == ["dataset_source"]
    assert summary["registry_registration"]["training_set_linked"] is True
    fake_registry = _FakeRegistry.instances[-1]
    assert len(fake_registry.upsert_calls) == 1
    assert len(fake_registry.lineage_calls) == 1
    out_root = zarr.open_group(str(out_path), mode="r")
    source_dataset_ids = np.asarray(out_root["source_index/source_dataset_id"][:], dtype=object).tolist()
    assert source_dataset_ids == ["dataset_source"]
