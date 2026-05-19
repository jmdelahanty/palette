from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.backfill_refined_detect_bbox_img_xyxy import backfill_refined_detect_bbox_img_xyxy


def _write_inference_space_bbox_fixture(tmp_path: Path) -> tuple[Path, str, np.ndarray]:
    zarr_path = tmp_path / "clipped_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.attrs["total_frames"] = 1

    detect_path = "clips/clip_000000/cameras/2010093/detect_runs/detect_001"
    detect = root.require_group(detect_path)
    detect.attrs["inference_width"] = 640
    detect.attrs["inference_height"] = 640
    detect.attrs["source_video_width"] = 4512
    detect.attrs["source_video_height"] = 4512

    refined_path = "clips/clip_000000/cameras/2010093/refined_detect_runs/refined_001"
    refined = root.require_group(refined_path)
    refined.attrs["source_detect_path"] = detect_path
    refined.attrs["coordinate_space"] = "full_image_xyxy"
    instances = refined.require_group("instances")
    source = refined.require_group("source_detections")
    bbox_norm = np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64)
    old_inference_xyxy = np.asarray([[256.0, 256.0, 384.0, 384.0]], dtype=np.float64)
    for group in (instances, source):
        group.create_array("bbox_norm_coords", data=bbox_norm, overwrite=True)
        group.create_array("bbox_img_xyxy", data=old_inference_xyxy, chunks=(1, 4), overwrite=True)

    collection = root.require_group("experiment_index/finalized_runs/collection_001")
    collection.attrs["selected_runs"] = [{"refined_group_path": refined_path}]
    return zarr_path, refined_path, np.asarray([[1804.8, 1804.8, 2707.2, 2707.2]], dtype=np.float64)


def test_backfill_refined_detect_bbox_img_xyxy_dry_run_reports_inference_space_values(tmp_path: Path) -> None:
    zarr_path, refined_path, _expected = _write_inference_space_bbox_fixture(tmp_path)

    result = backfill_refined_detect_bbox_img_xyxy(
        zarr_path,
        target_group_paths=[refined_path],
    )

    assert result["status"] == "ok"
    assert result["apply"] is False
    assert result["changed_group_count"] == 1
    group_report = result["target_groups"][0]
    assert group_report["status"] == "would_update"
    assert [surface["array_status"] for surface in group_report["surfaces"]] == [
        "would_rewrite",
        "would_rewrite",
    ]
    assert group_report["surfaces"][0]["bbox_img_reference_width"] == 4512
    assert group_report["surfaces"][0]["bbox_norm_reference_width"] == 640


def test_backfill_refined_detect_bbox_img_xyxy_apply_rewrites_source_space_and_attrs(tmp_path: Path) -> None:
    zarr_path, _refined_path, expected = _write_inference_space_bbox_fixture(tmp_path)

    result = backfill_refined_detect_bbox_img_xyxy(
        zarr_path,
        collection_id="collection_001",
        apply=True,
    )

    assert result["status"] == "ok"
    assert result["changed_group_count"] == 1
    root = zarr.open_group(str(zarr_path), mode="r")
    refined = root["clips/clip_000000/cameras/2010093/refined_detect_runs/refined_001"]
    instances = refined["instances"]
    source = refined["source_detections"]
    np.testing.assert_allclose(instances["bbox_img_xyxy"][:], expected)
    np.testing.assert_allclose(source["bbox_img_xyxy"][:], expected)
    assert instances["bbox_img_xyxy"].chunks == (1, 4)
    assert instances.attrs["bbox_img_xyxy_coordinate_space"] == "source_image_xyxy"
    assert instances.attrs["bbox_img_xyxy_reference_width"] == 4512
    assert instances.attrs["bbox_norm_reference_width"] == 640
    assert instances.attrs["bbox_norm_reference_space"] == "inference_image"
    assert refined.attrs["bbox_coordinate_contract_version"] == "refined_detect_bbox_coordinates_v2"
    assert refined.attrs["bbox_img_xyxy_reference_width"] == 4512


def test_backfill_refined_detect_bbox_img_xyxy_discover_empty_store_is_ok(tmp_path: Path) -> None:
    zarr_path = tmp_path / "empty_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    result = backfill_refined_detect_bbox_img_xyxy(zarr_path, discover=True)

    assert result["status"] == "ok"
    assert result["target_group_count"] == 0
    assert result["changed_group_count"] == 0
    assert result["target_groups"] == []
