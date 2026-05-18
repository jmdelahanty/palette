from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.rechunk_refined_detect_bbox_arrays import rechunk_refined_detect_bbox_arrays


def _write_split_bbox_refined_run(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined = root.require_group("refined_detect_runs/refined_001")
    instances = refined.require_group("instances")
    for name in ("bbox_img_xyxy", "bbox_norm_coords"):
        instances.create_array(
            name,
            data=np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64),
            chunks=(2, 2),
            overwrite=True,
        )
    source = refined.require_group("source_detections")
    source.create_array(
        "bbox_img_xyxy",
        data=np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64),
        chunks=(1, 2),
        overwrite=True,
    )
    return zarr_path


def test_rechunk_refined_detect_bbox_arrays_dry_run_reports_split_chunks(tmp_path: Path) -> None:
    zarr_path = _write_split_bbox_refined_run(tmp_path)

    result = rechunk_refined_detect_bbox_arrays(
        zarr_path,
        target_group_paths=["refined_detect_runs/refined_001"],
    )

    assert result["status"] == "ok"
    assert result["apply"] is False
    assert result["changed_array_count"] == 3
    statuses = [
        array["status"]
        for group in result["target_groups"][0]["groups"]
        for array in group["arrays"]
    ]
    assert statuses == ["would_rechunk", "would_rechunk", "would_rechunk"]


def test_rechunk_refined_detect_bbox_arrays_apply_rewrites_to_full_column_chunks(tmp_path: Path) -> None:
    zarr_path = _write_split_bbox_refined_run(tmp_path)

    result = rechunk_refined_detect_bbox_arrays(
        zarr_path,
        target_group_paths=["refined_detect_runs/refined_001"],
        apply=True,
    )

    assert result["status"] == "ok"
    assert result["changed_array_count"] == 3
    root = zarr.open_group(str(zarr_path), mode="r")
    instances = root["refined_detect_runs/refined_001/instances"]
    source = root["refined_detect_runs/refined_001/source_detections"]
    assert instances["bbox_img_xyxy"].chunks == (2, 4)
    assert instances["bbox_norm_coords"].chunks == (2, 4)
    assert source["bbox_img_xyxy"].chunks == (1, 4)
