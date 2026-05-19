from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.refined_detect_curation import write_curated_refined_detect_surfaces


def test_write_curated_refined_detect_surfaces_persists_summary_metadata_on_real_zarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "real_refined_detect.zarr"
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs["width"] = 320
    root.attrs["height"] = 240
    root.attrs["total_frames"] = 5

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect_group = detect_parent.create_group("detect_001")
    detect_group.create_array(
        "frame_indices",
        data=np.asarray([0, 2, 4], dtype=np.int32),
        overwrite=True,
    )
    detect_group.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [[0.2, 0.2, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.15, 0.15]],
            dtype=np.float64,
        ),
        overwrite=True,
    )
    detect_group.create_array(
        "scores",
        data=np.asarray([0.9, 0.85, 0.25], dtype=np.float32),
        overwrite=True,
    )
    detect_group.create_array(
        "class_ids",
        data=np.asarray([0, 0, 0], dtype=np.int32),
        overwrite=True,
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined_run = refined_parent.create_group("refined_detect_001")
    refined_run.attrs["source_detect_run"] = "detect_001"

    write_curated_refined_detect_surfaces(
        root,
        zarr_path=zarr_path,
        refined_run_name="refined_detect_001",
        instance_frame_indices=np.asarray([0, 2], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray(
            [[0.2, 0.2, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2]],
            dtype=np.float64,
        ),
        instance_source_kind_labels=np.asarray(["raw_detect", "raw_detect"], dtype=object),
        instance_reason_labels=np.asarray(["clean", "clean"], dtype=object),
        instance_source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
        instance_manual_edit_flags=np.asarray([False, False], dtype=bool),
        instance_confidence_scores=np.asarray([0.9, 0.85], dtype=np.float32),
        instance_class_ids=np.asarray([0, 0], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray([0, 1, 2], dtype=np.int32),
        source_detection_frame_indices=np.asarray([0, 2, 4], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray(
            [[0.2, 0.2, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.15, 0.15]],
            dtype=np.float64,
        ),
        source_detection_decision_labels=np.asarray(["accepted", "accepted", "filtered"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean", "clean", "filtered_jump"], dtype=object),
        source_detection_confidence_scores=np.asarray([0.9, 0.85, 0.25], dtype=np.float32),
        source_detection_class_ids=np.asarray([0, 0, 0], dtype=np.int32),
        command="test_real_zarr",
        source_context={"source_detect_run": "detect_001", "selection_policy": "unit_test"},
    )

    reopened = zarr.open_group(str(zarr_path), mode="r")
    run = reopened["refined_detect_runs"]["refined_detect_001"]
    summary = dict(run.attrs["summary_statistics"])

    assert run.attrs["curated_row_storage"] == "sparse_instances_v1"
    assert run.attrs["curated_primary_surface"] == "instances"
    assert run.attrs["refined_storage_semantics"] == "sparse_instances_v1"
    assert "dense_projection_storage" not in run.attrs
    assert summary["rows_present"] == 2
    assert summary["rows_missing"] == 3
    assert summary["source_detection_filtered"] == 1
    assert summary["source_detection_candidates"] == 3
    assert "curation_updated_at_utc" in run.attrs
    assert run["instances"]["bbox_img_xyxy"].chunks == (2, 4)
    assert run["instances"]["bbox_norm_coords"].chunks == (2, 4)
    assert run["source_detections"]["bbox_img_xyxy"].chunks == (3, 4)
    assert run["source_detections"]["bbox_norm_coords"].chunks == (3, 4)


def test_write_curated_refined_detect_surfaces_uses_source_pixels_for_bbox_img_xyxy(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_space_refined_detect.zarr"
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.attrs["total_frames"] = 1

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect_group = detect_parent.create_group("detect_001")
    detect_group.attrs["inference_width"] = 640
    detect_group.attrs["inference_height"] = 640
    detect_group.attrs["source_video_width"] = 4512
    detect_group.attrs["source_video_height"] = 4512
    detect_group.create_array("frame_indices", data=np.asarray([0], dtype=np.int32), overwrite=True)
    detect_group.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
        overwrite=True,
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined_run = refined_parent.create_group("refined_detect_001")
    refined_run.attrs["source_detect_run"] = "detect_001"

    write_curated_refined_detect_surfaces(
        root,
        zarr_path=zarr_path,
        refined_run_name="refined_detect_001",
        instance_frame_indices=np.asarray([0], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
        instance_source_kind_labels=np.asarray(["raw_detect"], dtype=object),
        instance_reason_labels=np.asarray(["clean"], dtype=object),
        instance_source_detect_row_index=np.asarray([0], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray([0], dtype=np.int32),
        source_detection_frame_indices=np.asarray([0], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
        source_detection_decision_labels=np.asarray(["accepted"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean"], dtype=object),
        command="test_source_space",
    )

    reopened = zarr.open_group(str(zarr_path), mode="r")
    run = reopened["refined_detect_runs/refined_detect_001"]
    instances = run["instances"]

    expected = np.asarray([[1804.8, 1804.8, 2707.2, 2707.2]], dtype=np.float64)
    np.testing.assert_allclose(instances["bbox_img_xyxy"][:], expected)
    assert instances.attrs["bbox_img_xyxy_coordinate_space"] == "source_image_xyxy"
    assert instances.attrs["bbox_img_xyxy_reference_width"] == 4512
    assert instances.attrs["bbox_norm_reference_width"] == 640
    assert instances.attrs["bbox_norm_reference_space"] == "inference_image"
    assert run.attrs["bbox_coordinate_contract_version"] == "refined_detect_bbox_coordinates_v2"
    assert run.attrs["bbox_img_xyxy_coordinate_space"] == "source_image_xyxy"
