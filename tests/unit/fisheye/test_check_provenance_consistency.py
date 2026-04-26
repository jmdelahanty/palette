from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.diagnostics import check_provenance_consistency as mod


class _FakeArray:
    def __init__(self, *, shape: tuple[int, ...] | None = None, data: Any | None = None) -> None:
        if data is None:
            if shape is None:
                raise ValueError("shape or data is required")
            data = np.zeros(shape, dtype=np.float32)
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self.attrs: dict[str, Any] = {}
        self._children: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, shape: tuple[int, ...] | None = None, data: Any | None = None) -> _FakeArray:
        arr = _FakeArray(shape=shape, data=data)
        self._children[name] = arr
        return arr

    def _resolve(self, key: str) -> Any | None:
        node: Any = self
        parts = [p for p in key.split("/") if p]
        for part in parts:
            if not isinstance(node, _FakeGroup):
                return None
            node = node._children.get(part)
            if node is None:
                return None
        return node

    def get(self, key: str) -> Any | None:
        return self._resolve(key)

    def __contains__(self, key: str) -> bool:
        return self._resolve(key) is not None

    def __getitem__(self, key: str) -> Any:
        node = self._resolve(key)
        if node is None:
            raise KeyError(key)
        return node

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()


def _build_root_with_detect() -> _FakeGroup:
    root = _FakeGroup()
    detect_runs = root.create_group("detect_runs")
    detect = detect_runs.create_group("detect_001")
    detect.create_array("bbox_norm_coords", shape=(3, 4))
    detect_runs.attrs["latest"] = "detect_001"
    return root


def test_collect_provenance_handles_missing_refined_interpolated_bbox() -> None:
    root = _build_root_with_detect()

    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.create_group("interpolated")  # Intentionally missing bbox_norm_coords.
    refined_runs.attrs["latest"] = "refined_detect_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert record.detect_rows == 3
    assert record.refined_rows is None
    assert any("missing detection arrays" in issue for issue in record.issues)


def test_collect_provenance_prefers_sparse_refined_instances_for_current_runs() -> None:
    root = _build_root_with_detect()

    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["detect_review_status"] = {"resolved_group": "refined"}
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", shape=(2,))
    instances.create_array("bbox_norm_coords", shape=(2, 4))
    instances.create_array("bbox_img_xyxy", shape=(2, 4))
    instances.create_array("frame_indices", shape=(2,))
    instances.create_array("frame_offsets", shape=(4,))
    instances.create_array("source_kind_codes", shape=(2,))
    instances.create_array("manual_edit_flags", shape=(2,))
    instances.create_array("source_detect_row_index", shape=(2,))
    instances.create_array("frame_counts", shape=(3,))
    refined_runs.attrs["latest"] = "refined_detect_001"

    crop_runs = root.create_group("crop_runs")
    crop = crop_runs.create_group("crop_001")
    crop.attrs["detection_source_path"] = "refined_detect_runs/refined_detect_001/instances"
    crop.create_array("roi_images", shape=(2, 4, 4))
    crop_runs.attrs["latest"] = "crop_001"

    arena_runs = root.create_group("arena_assignment_runs")
    arena = arena_runs.create_group("arena_001")
    arena.attrs["source_detect_run"] = "detect_001"
    arena.attrs["source_refined_run"] = "refined_detect_001"
    arena.create_array("arena_ids", shape=(2,))
    arena_runs.attrs["latest"] = "arena_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert record.refined_rows == 2
    assert record.crop_source_rows == 2
    assert record.arena_assignment_rows == 2
    assert record.issues == []


def test_collect_provenance_handles_missing_optional_stage_arrays() -> None:
    root = _build_root_with_detect()

    crop_runs = root.create_group("crop_runs")
    crop = crop_runs.create_group("crop_001")
    crop.attrs["detection_source_path"] = "detect_runs/detect_001"
    crop_runs.attrs["latest"] = "crop_001"

    keypoints_runs = root.create_group("keypoints_runs")
    keypoints = keypoints_runs.create_group("keypoints_001")
    keypoints.attrs["source_crop_run"] = "crop_001"
    keypoints_runs.attrs["latest"] = "keypoints_001"

    arena_runs = root.create_group("arena_assignment_runs")
    arena_run = arena_runs.create_group("arena_001")
    arena_run.attrs["source_detect_run"] = "detect_001"
    arena_runs.attrs["latest"] = "arena_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert record.crop_rois is None
    assert record.keypoint_rows is None
    assert record.arena_assignment_rows is None


def test_collect_provenance_reports_crop_snapshot_drift_from_upstream_refined_source() -> None:
    root = _build_root_with_detect()

    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["detect_review_status"] = {"resolved_group": "refined"}
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.asarray([10, 11], dtype=np.int64))
    instances.create_array("frame_indices", data=np.asarray([100, 100], dtype=np.int32))
    instances.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]], dtype=np.float64),
    )
    instances.create_array("bbox_img_xyxy", shape=(2, 4))
    instances.create_array("frame_offsets", shape=(102,))
    instances.create_array("source_kind_codes", shape=(2,))
    instances.create_array("manual_edit_flags", shape=(2,))
    instances.create_array("source_detect_row_index", shape=(2,))
    instances.create_array("frame_counts", shape=(101,))
    refined_runs.attrs["latest"] = "refined_detect_001"

    crop_runs = root.create_group("crop_runs")
    crop = crop_runs.create_group("crop_001")
    crop.attrs["detection_source_path"] = "refined_detect_runs/refined_detect_001/instances"
    crop.create_array("roi_images", shape=(2, 4, 4))
    crop.create_array("frame_indices", data=np.asarray([100, 100], dtype=np.int32))
    crop.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.25, 0.25, 0.2, 0.2], [0.70, 0.75, 0.2, 0.2]], dtype=np.float64),
    )
    crop_runs.attrs["latest"] = "crop_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert any("snapshot drifted from upstream source" in issue for issue in record.issues)
    assert any("bbox_norm_coords differ for 1 row(s) across 1 frame(s)." in issue for issue in record.issues)


def test_collect_provenance_ignores_subpixel_bbox_roundoff_when_dimensions_known() -> None:
    root = _build_root_with_detect()

    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["detect_review_status"] = {"resolved_group": "refined"}
    instances = refined.create_group("instances")
    instances.attrs["width"] = 4512
    instances.attrs["height"] = 4512
    instances.create_array("refined_row_ids", data=np.asarray([10, 11], dtype=np.int64))
    instances.create_array("frame_indices", data=np.asarray([100, 101], dtype=np.int32))
    instances.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.6382812261581421, 0.3119140565395355, 0.03281249850988388, 0.03281249850988388],
                [0.637890636920929, 0.3119140565395355, 0.03359375149011612, 0.033203125],
            ],
            dtype=np.float64,
        ),
    )
    instances.create_array("bbox_img_xyxy", shape=(2, 4))
    instances.create_array("frame_offsets", shape=(102,))
    instances.create_array("source_kind_codes", shape=(2,))
    instances.create_array("manual_edit_flags", shape=(2,))
    instances.create_array("source_detect_row_index", shape=(2,))
    instances.create_array("frame_counts", shape=(102,))
    refined_runs.attrs["latest"] = "refined_detect_001"

    crop_runs = root.create_group("crop_runs")
    crop = crop_runs.create_group("crop_001")
    crop.attrs["detection_source_path"] = "refined_detect_runs/refined_detect_001/instances"
    crop.attrs["width"] = 4512
    crop.attrs["height"] = 4512
    crop.create_array("roi_images", shape=(2, 4, 4))
    crop.create_array("frame_indices", data=np.asarray([100, 101], dtype=np.int32))
    crop.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.6382812067126551, 0.3119140516781638, 0.03281245671265514, 0.03281251082183621],
                [0.6378906466436725, 0.3119140516781638, 0.03359379328734486, 0.033203125],
            ],
            dtype=np.float64,
        ),
    )
    crop_runs.attrs["latest"] = "crop_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert record.crop_source_drift_issues == []
    assert not any("bbox_norm_coords differ" in issue for issue in record.issues)


def test_collect_provenance_reports_stale_downstream_crop_snapshots() -> None:
    root = _build_root_with_detect()

    crop_runs = root.create_group("crop_runs")
    crop = crop_runs.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 3}
    crop.attrs["crop_revision"] = 3
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001"
    crop.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
    crop.create_array("bbox_norm_coords", data=np.zeros((2, 4), dtype=np.float32))
    crop.create_array("roi_coordinates_full", data=np.asarray([[0, 0], [1, 1]], dtype=np.int32))
    crop_runs.attrs["latest"] = "crop_001"

    keypoints_runs = root.create_group("keypoints_runs")
    keypoints = keypoints_runs.create_group("kp_001")
    keypoints.attrs["source_crop_run"] = "crop_001"
    keypoints.attrs["source_crop_storage_mode"] = "geometry_only"
    keypoints.attrs["source_crop_signature"] = "stale-sig"
    keypoints.attrs["source_crop_revision"] = 2
    keypoints.attrs["source_detect_review_status_ref"] = "refined_detect_runs/refined_detect_000"
    keypoints.create_array("heading", data=np.asarray([0.0, 1.0], dtype=np.float64))
    keypoints_runs.attrs["latest"] = "kp_001"

    eye_masks_runs = root.create_group("eye_masks_runs")
    eye_masks = eye_masks_runs.create_group("eye_001")
    eye_masks.attrs["source_crop_run"] = "crop_001"
    eye_masks.attrs["source_crop_storage_mode"] = "geometry_only"
    eye_masks.attrs["source_crop_signature"] = "{'signature_version': 2, 'crop_revision': 3}"
    eye_masks.attrs["source_crop_revision"] = 3
    eye_masks_runs.attrs["latest"] = "eye_001"

    subject_mask_runs = root.create_group("subject_mask_runs")
    subject_masks = subject_mask_runs.create_group("subject_001")
    subject_masks.attrs["source_crop_run"] = "crop_001"
    subject_masks.attrs["source_crop_storage_mode"] = "geometry_only"
    subject_masks.attrs["source_crop_signature"] = "stale-subject-sig"
    subject_masks.attrs["source_crop_revision"] = 3
    subject_masks.create_array("masks_roi", shape=(2, 1, 4, 4))
    subject_mask_runs.attrs["latest"] = "subject_001"

    refined_subject_mask_runs = root.create_group("refined_subject_masks_runs")
    refined_subject_masks = refined_subject_mask_runs.create_group("refined_subject_001")
    refined_subject_masks.attrs["source_crop_run"] = "crop_001"
    refined_subject_masks.attrs["source_crop_storage_mode"] = "geometry_only"
    refined_subject_masks.attrs["source_crop_signature"] = "{'signature_version': 2, 'crop_revision': 3}"
    refined_subject_masks.attrs["source_crop_revision"] = 2
    refined_subject_masks.attrs["source_detect_review_status_ref"] = "refined_detect_runs/refined_detect_000"
    refined_subject_masks.create_array("masks_roi", shape=(2, 2, 4, 4))
    refined_subject_mask_runs.attrs["latest"] = "refined_subject_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert any("Keypoint run 'kp_001' crop snapshot drifted" in issue for issue in record.issues)
    assert any("source_crop_signature='stale-sig'" in issue for issue in record.issues)
    assert any("source_crop_revision=2 expected 3" in issue for issue in record.issues)
    assert any("source_detect_review_status_ref='refined_detect_runs/refined_detect_000'" in issue for issue in record.issues)
    assert any("Eye mask run 'eye_001' crop snapshot drifted" in issue for issue in record.issues)
    assert any("missing source_detect_review_status_ref" in issue for issue in record.issues)
    assert any("Subject mask run 'subject_001' crop snapshot drifted" in issue for issue in record.issues)
    assert any("source_crop_signature='stale-subject-sig'" in issue for issue in record.issues)
    assert any("Refined subject mask run 'refined_subject_001' crop snapshot drifted" in issue for issue in record.issues)
    assert any("source_crop_revision=2 expected 3" in issue for issue in record.subject_mask_crop_snapshot_issues + record.refined_subject_mask_crop_snapshot_issues)
    assert record.subject_mask_run == "subject_001"
    assert record.subject_mask_rows == 2
    assert record.refined_subject_mask_run == "refined_subject_001"
    assert record.refined_subject_mask_rows == 2
    assert len(record.downstream_crop_snapshot_issues) == 2
    assert len(record.subject_mask_crop_snapshot_issues) == 1
    assert len(record.refined_subject_mask_crop_snapshot_issues) == 1
