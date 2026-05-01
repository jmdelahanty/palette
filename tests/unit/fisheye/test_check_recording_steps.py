from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.registry.status_ledger import upsert_recording_step_status
from fisheye.utils import check_recording_steps as mod


def _write_minimal_h5(path: Path, *, session_uuid: str, camera_id: str = "cam_1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["session_uuid"] = session_uuid
        h5.attrs["camera_id"] = camera_id


def test_check_zarr_reports_detect_coverage_full_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "full_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((4,), dtype=np.uint8))
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_full"
    run = detect_parent.create_group("detect_full")
    run.create_array("frame_counts", data=np.array([1, 0, 2, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_method"] is None
    assert info["detect_coverage_basis"] == "full"
    assert info["detect_coverage"] == pytest.approx(75.0)


def test_check_zarr_reports_detect_coverage_sampled_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sampled_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("original_frame_indices", data=np.array([0, 2, 4], dtype=np.int32))
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_sampled"
    run = detect_parent.create_group("detect_sampled")
    run.create_array("frame_counts", data=np.array([1, 0, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_coverage_basis"] == "sampled"
    assert info["detect_coverage"] == pytest.approx(66.6667, abs=1e-3)


def test_check_zarr_reports_detect_coverage_inferred_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "inferred_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_inferred"
    run = detect_parent.create_group("detect_inferred")
    run.create_array("frame_indices", data=np.array([0, 0, 2, 5], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_coverage_basis"] == "inferred"
    assert info["detect_coverage"] == pytest.approx(50.0)


def test_check_zarr_reports_detect_missing_when_no_detect_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "empty_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is False
    assert info["detect_method"] is None
    assert info["detect_coverage"] is None
    assert info["detect_coverage_basis"] is None
    assert info["detect_quality_present"] is False
    assert info["detect_quality_run"] is None


def test_check_zarr_reads_detect_method_from_provenance(tmp_path: Path) -> None:
    zarr_path = tmp_path / "method_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_prov"
    run = detect_parent.create_group("detect_prov")
    run.attrs["provenance"] = {"method": "yolo"}
    run.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_method"] == "yolo"


def test_check_zarr_reads_detect_quality_summary(tmp_path: Path) -> None:
    zarr_path = tmp_path / "quality_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_quality_source"
    run = detect_parent.create_group("detect_quality_source")
    run.create_array("frame_counts", data=np.array([1, 1, 0], dtype=np.int32))
    quality_parent = run.create_group("quality_reports")
    quality_parent.attrs["latest"] = "detect_quality_2026-02-09_12-00-00"
    quality = quality_parent.create_group("detect_quality_2026-02-09_12-00-00")
    quality.attrs["quality_score"] = {"grade": "A", "overall_score": 99.2}
    quality.attrs["detection_quality_summary"] = {
        "clean_percentage": 98.7,
        "blip_detections": 3,
        "jump_detections": 2,
        "multi_detections": 1,
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_quality_present"] is True
    assert info["detect_quality_run"] == "detect_quality_2026-02-09_12-00-00"
    assert info["detect_quality_grade"] == "A"
    assert info["detect_quality_score"] == pytest.approx(99.2)
    assert info["detect_quality_clean_percent"] == pytest.approx(98.7)
    assert info["detect_quality_artifacts"] == 6


def test_check_zarr_reads_curated_refined_detect_coverage(tmp_path: Path) -> None:
    zarr_path = tmp_path / "curated_refined_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((4,), dtype=np.uint8))
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["detect_review_status"] = {"state": "approved", "resolved_group": "refined"}
    refined.create_array("refined_row_ids", data=np.array([0, 1, 2, 3], dtype=np.int64))
    refined.create_array("frame_indices", data=np.array([0, 1, 2, 3], dtype=np.int32))
    refined.create_array("entity_ids", data=np.array([0, 0, 0, 0], dtype=np.int32))
    refined.create_array(
        "bbox_img_xyxy",
        data=np.array(
            [
                [1.0, 1.0, 4.0, 4.0],
                [np.nan, np.nan, np.nan, np.nan],
                [1.0, 1.0, 4.0, 4.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
    )
    refined.create_array(
        "bbox_norm_coords",
        data=np.array(
            [
                [0.5, 0.5, 0.2, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
                [0.5, 0.5, 0.2, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
    )
    refined.create_array("status_codes", data=np.array([0, 1, 0, 2], dtype=np.int8))
    refined.create_array("source_kind_codes", data=np.array([1, 0, 1, 0], dtype=np.int8))
    refined.create_array("review_state_codes", data=np.array([1, 1, 1, 1], dtype=np.int8))
    refined.create_array("keypoints_state_codes", data=np.array([0, 0, 0, 0], dtype=np.int8))
    refined.create_array("subject_mask_state_codes", data=np.array([0, 0, 0, 0], dtype=np.int8))
    refined.create_array("eye_mask_state_codes", data=np.array([0, 0, 0, 0], dtype=np.int8))
    refined.create_array("swim_bladder_state_codes", data=np.array([0, 0, 0, 0], dtype=np.int8))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["refined_detect_present"] is True
    assert info["refined_detect_coverage"] == pytest.approx(50.0)
    assert info["refined_detect_method"] == "refined"
    assert info["refined_detect_resolved_group"] == "refined_detect_runs/refined_detect_001"


def test_check_zarr_reads_sparse_refined_instances_path(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sparse_refined_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((4,), dtype=np.uint8))
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["detect_review_status"] = {"state": "approved", "resolved_group": "refined"}
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.array([0, 1], dtype=np.int64))
    instances.create_array("frame_indices", data=np.array([0, 2], dtype=np.int32))
    instances.create_array("frame_offsets", data=np.array([0, 1, 1, 2, 2], dtype=np.int64))
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]], dtype=np.float64),
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
    )
    instances.create_array("source_kind_codes", data=np.array([1, 2], dtype=np.int8))
    instances.create_array("manual_edit_flags", data=np.array([0, 1], dtype=np.int8))
    instances.create_array("source_detect_row_index", data=np.array([0, 2], dtype=np.int32))
    instances.create_array("frame_counts", data=np.array([1, 0, 1, 0], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["refined_detect_present"] is True
    assert info["refined_detect_method"] == "refined"
    assert info["refined_detect_resolved_group"] == "refined_detect_runs/refined_detect_001/instances"


def test_check_zarr_reads_crop_status_from_latest_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "crop_status_latest_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_2026-02-10_12-00-00"
    crop_run = crop_parent.create_group("crop_2026-02-10_12-00-00")
    crop_run.attrs["status"] = "completed"

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_present"] is True
    assert info["crop_status"] == "completed"


def test_check_zarr_reads_crop_status_from_fallback_latest_name(tmp_path: Path) -> None:
    zarr_path = tmp_path / "crop_status_fallback_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.create_group("crop_2026-02-10_11-00-00").attrs["status"] = "failed"
    crop_parent.create_group("crop_2026-02-10_12-00-00").attrs["status"] = "running"

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_present"] is True
    assert info["crop_status"] == "running"


def test_check_zarr_reports_crop_drift_against_current_refined_instances(tmp_path: Path) -> None:
    zarr_path = tmp_path / "crop_drift_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((4,), dtype=np.uint8))

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect = detect_parent.create_group("detect_001")
    detect.create_array("bbox_norm_coords", data=np.zeros((2, 4), dtype=np.float64))

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["detect_review_status"] = {"state": "approved", "resolved_group": "refined"}
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.array([0, 1], dtype=np.int64))
    instances.create_array("frame_indices", data=np.array([100, 100], dtype=np.int32))
    instances.create_array("frame_offsets", data=np.array([0, 0, 0, 0], dtype=np.int64))
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 5.0, 5.0]], dtype=np.float64),
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]], dtype=np.float64),
    )
    instances.create_array("source_kind_codes", data=np.array([1, 2], dtype=np.int8))
    instances.create_array("manual_edit_flags", data=np.array([0, 1], dtype=np.int8))
    instances.create_array("source_detect_row_index", data=np.array([0, 1], dtype=np.int32))
    instances.create_array("frame_counts", data=np.array([0, 0, 0, 0], dtype=np.int32))

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["status"] = "completed"
    crop.attrs["detection_source_path"] = "refined_detect_runs/refined_detect_001/instances"
    crop.create_array("roi_images", data=np.zeros((2, 4, 4), dtype=np.uint8))
    crop.create_array("frame_indices", data=np.array([100, 100], dtype=np.int32))
    crop.create_array(
        "bbox_norm_coords",
        data=np.array([[0.25, 0.25, 0.2, 0.2], [0.70, 0.75, 0.2, 0.2]], dtype=np.float64),
    )

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_present"] is True
    assert info["crop_drift_present"] is True
    assert info["crop_drift_summary"] == "DRIFT (1 issue)"
    details = info["crop_drift_details"]
    assert isinstance(details, list)
    assert any("bbox_norm_coords differ for 1 row(s) across 1 frame(s)." in issue for issue in details)


def test_check_zarr_reports_subject_mask_snapshot_drift_against_current_crop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "subject_mask_drift_analysis.zarr"
    zarr_path.mkdir()
    root = _FakeTuningGroup()
    root["crop_runs"] = _FakeTuningGroup(
        attrs={"latest": "crop_001"},
        crop_001=_FakeTuningGroup(attrs={"status": "completed"}),
    )

    monkeypatch.setattr(mod.zarr, "open_group", lambda *args, **kwargs: root)
    monkeypatch.setattr(
        mod,
        "collect_provenance",
        lambda _root: SimpleNamespace(
            crop_source_drift_issues=[],
            subject_mask_crop_snapshot_issues=[
                "Subject mask run 'subject_masks_001' crop snapshot drifted from crop run 'crop_001': "
                "source_crop_signature='crop_sig_v1' expected 'crop_sig_v2'; "
                "source_crop_revision=1 expected 2."
            ],
            refined_subject_mask_crop_snapshot_issues=[
                "Refined subject mask run 'refined_subject_masks_001' crop snapshot drifted from crop run 'crop_001': "
                "source_crop_signature='crop_sig_v1' expected 'crop_sig_v2'; "
                "source_crop_revision=1 expected 2."
            ],
        ),
    )

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_drift_present"] is False
    assert info["subject_mask_drift_present"] is True
    assert info["subject_mask_drift_summary"] == "DRIFT (1 issue)"
    subject_details = info["subject_mask_drift_details"]
    assert isinstance(subject_details, list)
    assert any("source_crop_signature='crop_sig_v1' expected 'crop_sig_v2'" in issue for issue in subject_details)
    assert any("source_crop_revision=1 expected 2" in issue for issue in subject_details)

    assert info["refined_subject_mask_drift_present"] is True
    assert info["refined_subject_mask_drift_summary"] == "DRIFT (1 issue)"
    refined_subject_details = info["refined_subject_mask_drift_details"]
    assert isinstance(refined_subject_details, list)
    assert any(
        "Refined subject mask run 'refined_subject_masks_001' crop snapshot drifted" in issue
        for issue in refined_subject_details
    )
    assert any("source_crop_signature='crop_sig_v1' expected 'crop_sig_v2'" in issue for issue in refined_subject_details)


def test_crop_status_format_helpers() -> None:
    assert mod._crop_status_text(False, None) == "MISS"  # noqa: SLF001
    assert mod._crop_status_text(True, None) == "OK"  # noqa: SLF001
    assert mod._crop_status_text(True, "FAILED") == "failed"  # noqa: SLF001
    assert mod._crop_status_rich(True, "completed") == "[chartreuse1]completed[/chartreuse1]"  # noqa: SLF001
    assert mod._crop_status_rich(True, "running") == "[yellow]running[/yellow]"  # noqa: SLF001
    assert mod._crop_status_rich(True, "failed") == "[red]failed[/red]"  # noqa: SLF001
    assert mod._crop_drift_text(False, None) == "OK"  # noqa: SLF001
    assert mod._crop_drift_text(True, "DRIFT (1 issue)") == "DRIFT (1 issue)"  # noqa: SLF001
    assert mod._crop_drift_rich(False, None) == "[chartreuse1]OK[/chartreuse1]"  # noqa: SLF001
    assert mod._crop_drift_rich(True, "DRIFT (1 issue)") == "[yellow]DRIFT (1 issue)[/yellow]"  # noqa: SLF001
    assert mod._drift_text(False, None) == "OK"  # noqa: SLF001
    assert mod._drift_rich(True, "DRIFT (2 issues)") == "[yellow]DRIFT (2 issues)[/yellow]"  # noqa: SLF001


def test_track_status_format_helpers() -> None:
    assert mod._track_status_text(False, None, None, None) == "MISS"  # noqa: SLF001
    assert mod._track_status_text(True, None, None, None) == "OK"  # noqa: SLF001
    assert mod._track_status_text(True, None, 0, 0.0) == "OK"  # noqa: SLF001
    assert mod._track_status_text(True, "warn", 1, 0.25) == "WARN (1 unassigned, 0.3%)"  # noqa: SLF001
    assert mod._track_status_text(True, "block", 1, 25.0) == "WARN (1 unassigned, 25.0%)"  # noqa: SLF001
    assert mod._track_status_rich(True, "block", 1, 25.0) == "[yellow]WARN[/yellow] (1 unassigned, 25.0%)"  # noqa: SLF001


def test_eye_angle_status_format_helpers() -> None:
    assert mod._eye_angle_status_text(False, False, None, None, []) == "MISS"  # noqa: SLF001
    assert (  # noqa: SLF001
        mod._eye_angle_status_text(
            True,
            True,
            0.875,
            "subject_shape_eye_geometry",
            [],
        )
        == "OK (valid 87.5%, subject_shape_eye_geometry)"
    )
    assert (  # noqa: SLF001
        mod._eye_angle_status_text(
            True,
            False,
            0.0,
            "subject_shape_eye_geometry",
            ["valid_detection_fraction=0"],
        )
        == "WARN (valid 0.0%, subject_shape_eye_geometry, valid_detection_fraction=0)"
    )


def test_display_field_label_marks_legacy_eye_and_unified_subject_fields() -> None:
    assert mod._display_field_label("eye_masks") == "eye_masks (legacy compat)"  # noqa: SLF001
    assert mod._display_field_label("refined_eye_masks") == "refined_eye_masks (legacy compat)"  # noqa: SLF001
    assert mod._display_field_label("eye_mask_review_status") == "eye_mask_review_status (legacy compat)"  # noqa: SLF001
    assert mod._display_field_label("eye_angles") == "eye_angles (analysis)"  # noqa: SLF001
    assert mod._display_field_label("subject_mask_components") == "subject_mask_components (unified)"  # noqa: SLF001
    assert mod._display_field_label("refined_subject_mask_components") == "refined_subject_mask_components (unified)"  # noqa: SLF001
    assert mod._display_field_label("detect") == "detect"  # noqa: SLF001


def test_subject_mask_component_summary_text_marks_available_and_missing_components() -> None:
    assert (
        mod._subject_mask_component_summary_text(  # noqa: SLF001
            ["eye_left", "eye_right"],
            ["subject_body", "swim_bladder"],
            {"eye_left": "approved", "eye_right": "pending"},
        )
        == "avail: eye_l=appr, eye_r=pend; miss: body, swim"
    )
    assert (
        mod._subject_mask_component_summary_text(  # noqa: SLF001
            ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            [],
            {},
        )
        == "avail: body, eye_l, eye_r, swim"
    )


def test_subject_mask_tuning_component_statuses_detect_body_and_swim_entries() -> None:
    assert mod._subject_mask_tuning_component_statuses(  # noqa: SLF001
        {
            "version": "2.0",
            "components": {
                "subject_body": {"method": "traditional_subject_mask_seed"},
                "swim_bladder": {"subject_method_family": "swim_bladder_polar_boundary_v1"},
            },
        }
    ) == {
        "subject_body": "ok",
        "swim_bladder": "ok",
    }
    assert mod._subject_mask_tuning_component_statuses(None) == {  # noqa: SLF001
        "subject_body": "miss",
        "swim_bladder": "miss",
    }


def test_subject_mask_tuning_component_lines_render_nested_labels() -> None:
    assert mod._subject_mask_tuning_component_lines(  # noqa: SLF001
        {"subject_body": "miss", "swim_bladder": "ok"}
    ) == [
        ("subject_mask_tuning.subject_body", "MISS"),
        ("subject_mask_tuning.swim_bladder", "OK"),
    ]


class _FakeTuningGroup(dict):
    def __init__(self, *args: object, attrs: dict[str, object] | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: object = None) -> object:
        return super().get(key, default)


def test_check_zarr_reads_track_unassigned_warning_from_latest_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "track_warning_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    tracks_parent = root.create_group("tracking_runs")
    tracks_parent.attrs["latest"] = "tracks_001"
    track_run = tracks_parent.create_group("tracks_001")
    track_run.attrs["summary_statistics"] = {
        "n_rows": 4,
        "n_tracks": 3,
        "n_assigned_rows": 3,
        "n_unassigned_rows": 1,
        "unassigned_row_rate_percent": 25.0,
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["track_present"] is True
    assert info["track_qc_state"] == "warn"
    assert info["track_unassigned_rows"] == 1
    assert info["track_unassigned_rate_percent"] == pytest.approx(25.0)


def test_check_zarr_reads_ready_eye_angle_analysis_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_angle_ready_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("eye_angle_runs")
    parent.attrs["latest"] = "eye_angle_001"
    run = parent.create_group("eye_angle_001")
    run.attrs.update(
        {
            "status": "complete",
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 5,
            "method": "ellipse_and_centroid_eye_angles",
            "row_axis": "keypoint_detection_rows",
            "source_geometry_kind": "subject_shape_eye_geometry",
            "source_eye_geometry_stage": "analysis/subject_shape_runs",
            "source_eye_geometry_run": "shape_001",
            "source_keypoints_run": "refined_keypoints_001",
            "valid_detection_fraction": 0.75,
        }
    )

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["eye_angles_present"] is True
    assert info["eye_angles_ready"] is True
    assert info["eye_angle_run"] == "eye_angle_001"
    assert info["eye_angle_valid_detection_fraction"] == pytest.approx(0.75)
    assert info["eye_angle_source_geometry_kind"] == "subject_shape_eye_geometry"
    assert info["eye_angle_readiness_reasons"] == []


def test_check_zarr_reports_eye_angle_analysis_contract_warnings(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_angle_warn_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("eye_angle_runs")
    parent.attrs["latest"] = "eye_angle_001"
    run = parent.create_group("eye_angle_001")
    run.attrs.update(
        {
            "status": "complete",
            "schema_id": "legacy.eye_angle_runs",
            "schema_version": 0,
            "valid_detection_fraction": 0.0,
        }
    )

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["eye_angles_present"] is True
    assert info["eye_angles_ready"] is False
    reasons = info["eye_angle_readiness_reasons"]
    assert "schema_id=legacy.eye_angle_runs" in reasons
    assert "schema_version=0" in reasons
    assert "source_geometry_kind=missing" in reasons
    assert "source_keypoints_run=missing" in reasons
    assert "valid_detection_fraction=0" in reasons


def test_check_zarr_reads_subject_mask_status_and_components(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_mask_status_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_masks_001"
    subject_run = subject_parent.create_group("subject_masks_001")
    subject_run.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    subject_run.attrs["subject_mask_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
    }
    subject_run.attrs["component_review_statuses"] = {
        "eye_left": {"state": "approved"},
        "eye_right": {"state": "approved"},
    }
    subject_run.create_array(
        "available_channels",
        data=np.array([False, True, True, False], dtype=np.bool_),
    )
    subject_metrics = subject_run.create_group("metrics")
    subject_metrics.create_array(
        "mask_present",
        data=np.array(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=np.bool_,
        ),
    )

    refined_parent = root.create_group("refined_subject_masks_runs")
    refined_parent.attrs["latest"] = "refined_subject_masks_001"
    refined_run = refined_parent.create_group("refined_subject_masks_001")
    refined_run.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    refined_run.attrs["refined_subject_mask_review_status"] = {
        "state": "pending",
        "method": "manual",
        "intended_use": "training",
    }
    refined_run.attrs["component_review_statuses"] = {
        "subject_body": {"state": "approved"},
        "eye_left": {"state": "approved"},
        "eye_right": {"state": "approved"},
        "swim_bladder": {"state": "needs_review"},
    }
    refined_run.create_array(
        "available_channels",
        data=np.array([True, True, True, True], dtype=np.bool_),
    )
    refined_metrics = refined_run.create_group("metrics")
    refined_metrics.create_array(
        "mask_present",
        data=np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, False],
                [True, True, True, False],
            ],
            dtype=np.bool_,
        ),
    )

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["subject_masks_present"] is True
    assert info["subject_masks_coverage"] == pytest.approx(100.0)
    assert info["subject_mask_available_components"] == ["eye_left", "eye_right"]
    assert info["subject_mask_unavailable_components"] == ["subject_body", "swim_bladder"]
    assert info["subject_mask_component_review_states"] == {
        "eye_left": "approved",
        "eye_right": "approved",
    }
    assert info["refined_subject_masks_present"] is True
    assert info["refined_subject_masks_coverage"] == pytest.approx(100.0)
    assert info["refined_subject_mask_available_components"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert info["refined_subject_mask_component_review_states"] == {
        "subject_body": "approved",
        "eye_left": "approved",
        "eye_right": "approved",
        "swim_bladder": "needs_review",
    }
    review_status = info["refined_subject_mask_review_status"]
    assert isinstance(review_status, dict)
    assert review_status["state"] == "pending"


def test_check_zarr_reads_subject_mask_tuning_component_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "subject_mask_tuning_training.zarr"
    zarr_path.mkdir()
    root = _FakeTuningGroup(
        attrs={"zarr_use": "training"},
    )
    root["analysis_metadata"] = _FakeTuningGroup(
        attrs={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "subject_body": {"method": "traditional_subject_mask_seed"},
                    "swim_bladder": {"subject_method_family": "swim_bladder_polar_boundary_v1"},
                },
            }
        }
    )
    monkeypatch.setattr(mod.zarr, "open_group", lambda *args, **kwargs: root)

    info = mod._check_zarr(zarr_path, tuning_keys=["subject_mask_tuning"])  # noqa: SLF001

    assert info["tuning_status"]["subject_mask_tuning"] == "ok"
    assert info["subject_mask_tuning_component_status"] == {
        "subject_body": "ok",
        "swim_bladder": "ok",
    }


def test_registry_crop_review_status_for_zarr_returns_latest_review_fields(tmp_path: Path) -> None:
    zarr_path = tmp_path / "a_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=zarr_path,
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose="analysis",
    )
    registry.replace_crop_quality(
        "dataset_a",
        [
            {
                "crop_run": "crop_2026-02-10_10-00-00",
                "recording_id": "recording_a",
                "zarr_use": "analysis",
                "crop_created_utc": "2026-02-10T10:00:00+00:00",
                "source_detect_run": None,
                "source_refined_run": None,
                "detection_source_type": "manual",
                "detection_source_path": None,
                "total_rois": 100,
                "frames_with_crops": 90,
                "total_frames": 100,
                "percent_frames_with_crops": 90.0,
                "includes_interpolated": 0,
                "n_real_detections": 100,
                "n_interpolated_detections": 0,
                "review_state": "approved",
                "review_method": "manual",
                "review_intended_use": "training",
                "review_reviewer": "alice",
                "review_timestamp_utc": "2026-02-10T10:30:00+00:00",
                "review_notes": "looks good",
                "zarr_mtime_ns": int(zarr_path.stat().st_mtime_ns),
                "updated_utc": "2026-02-10T10:30:00+00:00",
            }
        ],
    )

    payload = mod._registry_crop_review_status_for_zarr(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
    )
    registry.close()

    assert payload is not None
    assert payload["crop_run"] == "crop_2026-02-10_10-00-00"
    status = payload["crop_review_status"]
    assert isinstance(status, dict)
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert status["reviewer"] == "alice"
    assert status["timestamp_utc"] == "2026-02-10T10:30:00+00:00"


def test_registry_status_payload_reads_subject_mask_status_and_components(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_mask_registry_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_subject",
        session_uuid="session_subject",
        zarr_path=zarr_path,
        recording_id="recording_subject",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_provenance(
        "dataset_subject",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose="analysis",
    )
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_subject",
        recording_id="recording_subject",
        step_name="subject_masks",
        status="ok",
        run_name="subject_masks_001",
        method="subject_mask_threshold_lr_v1",
        coverage_pct=100.0,
        review_status_json={"state": "approved", "method": "manual"},
        details_json={
            "available_components": ["eye_left", "eye_right"],
            "unavailable_components": ["subject_body", "swim_bladder"],
            "component_review_states": {
                "eye_left": "approved",
                "eye_right": "approved",
            },
        },
        source="unit_test",
    )
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_subject",
        recording_id="recording_subject",
        step_name="refined_subject_masks",
        status="ok",
        run_name="refined_subject_masks_001",
        method="refine_subject_masks",
        coverage_pct=100.0,
        review_status_json={"state": "pending", "method": "manual"},
        details_json={
            "available_components": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "unavailable_components": [],
            "component_review_states": {
                "subject_body": "approved",
                "eye_left": "approved",
                "eye_right": "approved",
                "swim_bladder": "needs_review",
            },
        },
        source="unit_test",
    )

    payload = mod._registry_status_payload(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
        recording_id="recording_subject",
        tuning_keys=[],
    )
    registry.close()

    assert payload["subject_masks_present"] is True
    assert payload["subject_masks_coverage"] == pytest.approx(100.0)
    assert payload["subject_mask_available_components"] == ["eye_left", "eye_right"]
    assert payload["subject_mask_component_review_states"] == {
        "eye_left": "approved",
        "eye_right": "approved",
    }
    assert payload["refined_subject_masks_present"] is True
    assert payload["refined_subject_masks_coverage"] == pytest.approx(100.0)
    assert payload["refined_subject_mask_available_components"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert payload["refined_subject_mask_component_review_states"] == {
        "subject_body": "approved",
        "eye_left": "approved",
        "eye_right": "approved",
        "swim_bladder": "needs_review",
    }


def test_registry_status_payload_overlays_component_registry_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_mask_component_overlay_analysis.zarr"
    zarr_path.mkdir()
    training_zarr_path = tmp_path / "subject_mask_component_overlay_training.zarr"
    training_zarr_path.mkdir()

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_component_overlay",
        session_uuid="session_component_overlay",
        zarr_path=zarr_path,
        recording_id="recording_component_overlay",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_dataset(
        "dataset_component_overlay_training",
        session_uuid="session_component_overlay",
        zarr_path=training_zarr_path,
        recording_id="recording_component_overlay",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_component_overlay",
        recording_id="recording_component_overlay",
        step_name="subject_masks",
        status="ok",
        run_name="subject_masks_001",
        method="subject_mask_threshold_lr_v1",
        coverage_pct=100.0,
        details_json={
            "available_components": ["eye_left"],
            "unavailable_components": ["subject_body", "eye_right", "swim_bladder"],
            "component_review_states": {"eye_left": "approved"},
        },
        source="unit_test",
    )
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_component_overlay",
        recording_id="recording_component_overlay",
        step_name="refined_subject_masks",
        status="ok",
        run_name="refined_subject_masks_partial",
        method="refine_subject_masks",
        coverage_pct=100.0,
        details_json={
            "available_components": ["subject_body"],
            "unavailable_components": ["eye_left", "eye_right", "swim_bladder"],
            "component_review_states": {"subject_body": "approved"},
        },
        source="unit_test",
    )

    def _upsert_component(
        *,
        stage_group: str,
        run_name: str,
        component_name: str,
        available: int,
        review_state: str | None,
        lifecycle_state: str | None = None,
    ) -> None:
        registry.upsert_subject_mask_component_quality(
            dataset_id="dataset_component_overlay",
            stage_group=stage_group,
            run_name=run_name,
            component_name=component_name,
            component_family="eyes" if component_name.startswith("eye_") else component_name,
            run_created_utc="2026-03-03T00:00:00+00:00",
            recording_id="recording_component_overlay",
            zarr_use="analysis",
            subject_mask_method=(
                "refine_subject_masks"
                if stage_group == "refined_subject_masks_runs"
                else "subject_mask_threshold_lr_v1"
            ),
            label_schema_id="subject_v1_lr",
            eye_component_mode="lr",
            source_subject_mask_run=(
                "subject_masks_registry" if stage_group == "refined_subject_masks_runs" else None
            ),
            available=available,
            review_state=review_state,
            review_method="manual" if review_state else None,
            review_intended_use="training" if review_state else None,
            review_reviewer="pytest" if review_state else None,
            review_timestamp_utc="2026-03-03T00:01:00+00:00" if review_state else None,
            total_rois=100,
            rows_with_component_mask=90 if available else 0,
            rows_with_component_mask_rate=0.9 if available else 0.0,
            lifecycle_state=lifecycle_state or review_state,
            lifecycle_reason=lifecycle_state or review_state,
            quality_updated_utc="2026-03-03T00:01:00+00:00",
            zarr_mtime_ns=123,
        )

    _upsert_component(
        stage_group="subject_mask_runs",
        run_name="subject_masks_registry",
        component_name="eyes_union",
        available=1,
        review_state="approved",
    )
    registry.upsert_subject_mask_component_quality(
        dataset_id="dataset_component_overlay_training",
        stage_group="subject_mask_runs",
        run_name="subject_masks_training_legacy_eye_bridge",
        component_name="eye_left",
        component_family="eyes",
        run_created_utc="2026-03-03T00:00:00+00:00",
        recording_id="recording_component_overlay",
        zarr_use="training",
        subject_mask_method="fisheye.utils.backfill_subject_mask_runs",
        label_schema_id="subject_v1_lr",
        eye_component_mode="lr",
        source_subject_mask_run=None,
        available=1,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="pytest",
        review_timestamp_utc="2026-03-03T00:01:00+00:00",
        total_rois=100,
        rows_with_component_mask=90,
        rows_with_component_mask_rate=0.9,
        lifecycle_state="approved",
        lifecycle_reason="approved",
        quality_updated_utc="2026-03-03T00:01:00+00:00",
        zarr_mtime_ns=456,
    )
    for component_name, review_state, lifecycle_state in (
        ("subject_body", "approved", "approved"),
        ("eye_left", "approved", "stale"),
        ("eye_right", "approved", "approved"),
        ("swim_bladder", "needs_review", "needs_review"),
    ):
        _upsert_component(
            stage_group="refined_subject_masks_runs",
            run_name=f"refined_{component_name}",
            component_name=component_name,
            available=1,
            review_state=review_state,
            lifecycle_state=lifecycle_state,
        )

    registry.conn.execute(
        """
        INSERT INTO subject_mask_performance (
            dataset_id,
            stage_group,
            run_name,
            recording_id,
            source_subject_mask_run,
            source_subject_mask_stale_state,
            source_subject_mask_stale_reason,
            source_subject_mask_stale_timestamp_utc,
            lifecycle_state,
            lifecycle_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_component_overlay",
            "refined_subject_masks_runs",
            "refined_eye_left",
            "recording_component_overlay",
            "subject_masks_registry",
            "stale",
            "source_subject_mask_changed",
            "2026-03-03T00:02:00+00:00",
            "stale",
            "source_subject_mask_changed",
        ),
    )
    registry.conn.commit()

    payload = mod._registry_status_payload(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
        recording_id="recording_component_overlay",
        tuning_keys=[],
    )
    registry.close()

    assert payload["subject_mask_available_components"] == ["eyes_union"]
    assert payload["subject_mask_component_review_states"] == {"eyes_union": "approved"}
    assert payload["subject_mask_component_source_subject_mask_stale"] == {}
    assert payload["refined_subject_mask_available_components"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert payload["refined_subject_mask_component_review_states"] == {
        "subject_body": "approved",
        "eye_left": "stale",
        "eye_right": "approved",
        "swim_bladder": "needs_review",
    }
    assert payload["refined_subject_mask_component_source_subject_mask_stale"] == {
        "eye_left": {
            "state": "stale",
            "reason": "source_subject_mask_changed",
            "timestamp_utc": "2026-03-03T00:02:00+00:00",
            "source_subject_mask_run": "subject_masks_registry",
        }
    }


def test_registry_crop_review_status_for_zarr_returns_none_when_row_is_stale(tmp_path: Path) -> None:
    zarr_path = tmp_path / "a_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=zarr_path,
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose="analysis",
    )
    registry.replace_crop_quality(
        "dataset_a",
        [
            {
                "crop_run": "crop_2026-02-10_10-00-00",
                "recording_id": "recording_a",
                "zarr_use": "analysis",
                "crop_created_utc": "2026-02-10T10:00:00+00:00",
                "source_detect_run": None,
                "source_refined_run": None,
                "detection_source_type": "manual",
                "detection_source_path": "manual",
                "total_rois": 100,
                "frames_with_crops": 90,
                "total_frames": 100,
                "percent_frames_with_crops": 90.0,
                "includes_interpolated": 0,
                "n_real_detections": 100,
                "n_interpolated_detections": 0,
                "review_state": "approved",
                "review_method": "manual",
                "review_intended_use": "training",
                "review_reviewer": "alice",
                "review_timestamp_utc": "2026-02-10T10:30:00+00:00",
                "review_notes": "looks good",
                "zarr_mtime_ns": int(zarr_path.stat().st_mtime_ns + 1),
                "updated_utc": "2026-02-10T10:30:00+00:00",
            }
        ],
    )

    payload = mod._registry_crop_review_status_for_zarr(
        registry=registry,
        zarr_path=zarr_path,
    )
    registry.close()

    assert payload is None


def test_open_root_live_prefers_non_consolidated(monkeypatch) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def _fake_open_group(path: str, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"path": path, **kwargs})
        return sentinel

    monkeypatch.setattr(mod.zarr, "open_group", _fake_open_group)

    out = mod._open_root_live(Path("/tmp/example.zarr"))  # noqa: SLF001

    assert out is sentinel
    assert len(calls) == 1
    assert calls[0]["path"] == "/tmp/example.zarr"
    assert calls[0]["mode"] == "r"
    assert calls[0]["use_consolidated"] is False


def test_open_root_live_falls_back_when_use_consolidated_unsupported(monkeypatch) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def _fake_open_group(path: str, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"path": path, **kwargs})
        if "use_consolidated" in kwargs:
            raise TypeError("unsupported kwarg")
        return sentinel

    monkeypatch.setattr(mod.zarr, "open_group", _fake_open_group)

    out = mod._open_root_live(Path("/tmp/example.zarr"))  # noqa: SLF001

    assert out is sentinel
    assert len(calls) == 2
    assert calls[0]["mode"] == "r"
    assert calls[0]["use_consolidated"] is False
    assert calls[1]["mode"] == "r"
    assert "use_consolidated" not in calls[1]


def test_check_zarr_reads_refined_eye_mask_review_status_from_latest_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_review_latest_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"
    run = refined_parent.create_group("refined_eye_masks_001")
    run.attrs["eye_mask_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "reviewer": "alice",
        "timestamp_utc": "2026-02-12T04:50:00+00:00",
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["refined_eye_masks_present"] is True
    status = info["eye_mask_review_status"]
    assert isinstance(status, dict)
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"


def test_check_zarr_reads_refined_eye_mask_review_status_from_parent_latest_pointer(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_review_pointer_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_002"
    refined_parent.attrs["eye_mask_review_status_latest"] = "refined_eye_masks_001"
    refined_parent.create_group("refined_eye_masks_002")
    run = refined_parent.create_group("refined_eye_masks_001")
    run.attrs["eye_mask_review_status"] = {
        "state": "pending",
        "method": "spotcheck",
        "intended_use": "training",
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["refined_eye_masks_present"] is True
    status = info["eye_mask_review_status"]
    assert isinstance(status, dict)
    assert status["state"] == "pending"
    assert status["method"] == "spotcheck"


def test_main_status_source_registry_requires_registry(tmp_path: Path, capsys) -> None:
    root = tmp_path / "recordings"
    root.mkdir(parents=True)

    rc = mod.main([str(root), "--status-source", "registry", "--no-rich"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "--status-source registry requires --registry." in out


def test_main_status_source_registry_uses_registry_views(tmp_path: Path, monkeypatch, capsys) -> None:
    recordings_root = tmp_path / "recordings"
    h5_path = recordings_root / "rec_registry" / "raw" / "capture.h5"
    _write_minimal_h5(h5_path, session_uuid="rec_registry", camera_id="cam_9")

    zarr_path = recordings_root / "rec_registry" / "zarr" / "rec_registry_analysis.zarr"

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_registry_a",
        session_uuid="rec_registry",
        zarr_path=zarr_path,
        recording_id="rec_registry",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, method, coverage_pct, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_registry_a",
            "rec_registry",
            "detect",
            "ok",
            "detect_001",
            "yolo",
            99.0,
            "unit_test",
            "2026-02-22T00:00:00+00:00",
        ),
    )
    registry.conn.commit()
    registry.close()

    def _fail_check_zarr(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("filesystem zarr traversal should not run in registry mode")

    monkeypatch.setattr(mod, "_check_zarr", _fail_check_zarr)

    rc = mod.main(
        [
            str(recordings_root),
            "--status-source",
            "registry",
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "rec_registry" in out
    assert "detect: OK" in out


def test_main_status_source_compare_outputs_mismatches(tmp_path: Path, capsys) -> None:
    recordings_root = tmp_path / "recordings"
    h5_path = recordings_root / "rec_compare" / "raw" / "capture.h5"
    _write_minimal_h5(h5_path, session_uuid="rec_compare", camera_id="cam_2")

    zarr_path = recordings_root / "rec_compare" / "zarr" / "rec_compare_analysis.zarr"

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_compare_a",
        session_uuid="rec_compare",
        zarr_path=zarr_path,
        recording_id="rec_compare",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_compare_a",
            "rec_compare",
            "detect",
            "ok",
            "detect_compare",
            "unit_test",
            "2026-02-22T00:00:00+00:00",
        ),
    )
    registry.conn.commit()
    registry.close()

    rc = mod.main(
        [
            str(recordings_root),
            "--status-source",
            "compare",
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Recording Step Status Mismatches" in out
    assert "field: detect" in out


def test_registry_status_payload_filters_rows_to_matching_recording_id(tmp_path: Path) -> None:
    zarr_path = tmp_path / "shared_training.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_good",
        session_uuid="session_good",
        zarr_path=zarr_path,
        recording_id="rec_target",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_ghost",
        session_uuid="session_ghost",
        zarr_path=zarr_path,
        recording_id=None,
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        [
            ("dataset_good", "rec_target", "eye_masks", "ok", "unit_test", "2026-02-23T01:00:00+00:00"),
            ("dataset_good", "rec_target", "refined_eye_masks", "ok", "unit_test", "2026-02-23T01:00:00+00:00"),
            ("dataset_good", "rec_target", "eye_mask_tuning", "ok", "unit_test", "2026-02-23T01:00:00+00:00"),
            ("dataset_ghost", None, "eye_masks", "missing", "unit_test", "2026-02-23T02:00:00+00:00"),
            ("dataset_ghost", None, "refined_eye_masks", "missing", "unit_test", "2026-02-23T02:00:00+00:00"),
            ("dataset_ghost", None, "eye_mask_tuning", "missing", "unit_test", "2026-02-23T02:00:00+00:00"),
        ],
    )
    registry.conn.commit()

    payload = mod._registry_status_payload(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
        recording_id="rec_target",
        tuning_keys=["eye_mask_tuning"],
    )
    registry.close()

    assert payload["eye_masks_present"] is True
    assert payload["refined_eye_masks_present"] is True
    assert payload["tuning_status"]["eye_mask_tuning"] == "ok"  # type: ignore[index]


def test_registry_status_payload_reads_track_unassigned_warning(tmp_path: Path) -> None:
    zarr_path = tmp_path / "track_warning_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_track_warning",
        session_uuid="session_track_warning",
        zarr_path=zarr_path,
        recording_id="recording_track_warning",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, details_json, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_track_warning",
            "recording_track_warning",
            "tracks",
            "ok",
            "tracks_001",
            json.dumps(
                {
                    "n_assigned_rows": 399,
                    "n_unassigned_rows": 1,
                    "unassigned_row_rate_percent": 0.25,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "unit_test",
            "2026-02-23T03:00:00+00:00",
        ),
    )
    registry.conn.commit()

    payload = mod._registry_status_payload(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
        recording_id="recording_track_warning",
        tuning_keys=[],
    )
    registry.close()

    assert payload["track_present"] is True
    assert payload["track_qc_state"] == "warn"
    assert payload["track_unassigned_rows"] == 1
    assert payload["track_unassigned_rate_percent"] == pytest.approx(0.25)
