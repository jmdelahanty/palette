from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
)
from fisheye.utils.validate_refined_detect_run import validate_refined_detect_run


def _array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    group.create_array(name, data=np.asarray(data), overwrite=True)


def _write_valid_refined_run(tmp_path: Path, *, source_detect_path: str) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.require_group(source_detect_path)

    refined_family_path = "clips/clip_000000/cameras/2010093/refined_detect_runs"
    refined = root.require_group(f"{refined_family_path}/refined_001")
    refined.attrs.update(
        {
            "source_detect_run": "detect_001",
            "source_detect_path": source_detect_path,
            "source_quality_run": "detect_quality_001",
            "refined_family_path": refined_family_path,
            "curated_primary_surface": "instances",
            "row_identity_policy": "stable_sparse_refined_row_id",
            "refined_storage_semantics": "sparse_instances_v1",
        }
    )

    raw_detect = REFINED_SOURCE_KIND_CODE_MAP["raw_detect"]
    accepted = REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"]
    instances = refined.require_group("instances")
    _array(instances, "refined_row_ids", np.asarray([0], dtype=np.int64))
    _array(instances, "frame_indices", np.asarray([0], dtype=np.int32))
    _array(instances, "frame_offsets", np.asarray([0, 1], dtype=np.int64))
    _array(instances, "bbox_img_xyxy", np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64))
    _array(instances, "bbox_norm_coords", np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64))
    _array(instances, "source_kind_codes", np.asarray([raw_detect], dtype=np.int8))
    _array(instances, "manual_edit_flags", np.asarray([False], dtype=bool))
    _array(instances, "source_detect_row_index", np.asarray([0], dtype=np.int32))
    _array(instances, "frame_counts", np.asarray([1], dtype=np.int32))

    source = refined.require_group("source_detections")
    _array(source, "source_detect_row_index", np.asarray([0], dtype=np.int32))
    _array(source, "frame_indices", np.asarray([0], dtype=np.int32))
    _array(source, "bbox_img_xyxy", np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64))
    _array(source, "bbox_norm_coords", np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64))
    _array(source, "decision_codes", np.asarray([accepted], dtype=np.int8))
    _array(source, "resolved_refined_row_id", np.asarray([0], dtype=np.int64))
    return zarr_path


def test_validate_refined_detect_run_passes_for_clip_local_sparse_surface(tmp_path: Path) -> None:
    source_detect_path = "clips/clip_000000/cameras/2010093/detect_runs/detect_001"
    zarr_path = _write_valid_refined_run(tmp_path, source_detect_path=source_detect_path)

    result = validate_refined_detect_run(
        zarr_path,
        target_group_path="clips/clip_000000/cameras/2010093/refined_detect_runs/refined_001",
    )

    assert result["status"] == "ok"
    assert result["row_counts"] == {"instances": 1, "frame_counts": 1, "source_detections": 1}
    assert result["source_detect_path"] == source_detect_path
    assert result["identity_validation"]["ok"] is True
    assert [item["status"] for item in result["validations"]] == ["pass", "pass", "pass", "pass"]


def test_validate_refined_detect_run_fails_when_source_detect_path_is_missing(tmp_path: Path) -> None:
    zarr_path = _write_valid_refined_run(
        tmp_path,
        source_detect_path="clips/clip_000000/cameras/2010093/detect_runs/detect_001",
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    refined = root["clips/clip_000000/cameras/2010093/refined_detect_runs/refined_001"]
    refined.attrs["source_detect_path"] = "clips/clip_000000/cameras/2010093/detect_runs/missing"

    result = validate_refined_detect_run(
        zarr_path,
        target_group_path="clips/clip_000000/cameras/2010093/refined_detect_runs/refined_001",
    )

    assert result["status"] == "failed"
    assert result["validations"][-1]["name"] == "source_detect_path_exists"
    assert result["validations"][-1]["status"] == "fail"
    assert "source_detect_path does not resolve" in "\n".join(result["errors"])
