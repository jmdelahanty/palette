from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis import subject_shape_runs as mod


def _patch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "shape-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )


def _disk_mask(height: int, width: int, center_y: float, center_x: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(np.uint8)


def _decode_reason_row(row: np.ndarray) -> str:
    return bytes(np.asarray(row, dtype=np.uint8)).split(b"\0", 1)[0].decode("utf-8")


def _build_refined_root(store_path: Path | None = None) -> zarr.Group:
    root = zarr.open_group(str(store_path), mode="w") if store_path is not None else zarr.group()
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    run.attrs.update(
        {
            "mask_labels": labels,
            "label_schema_id": "subject_v1_lr",
            "component_metrics_schema_id": "refined_subject_component_mask_metrics_v1",
            "source_subject_mask_run": "subject_001",
            "component_review_statuses": {
                component: {"state": "approved", "method": "unit_test"}
                for component in labels
            },
        }
    )
    run.create_array("available_channels", data=np.asarray([True, True, True, True], dtype=bool), overwrite=True)
    run.create_array("detection_source", data=np.asarray([0, 0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11, 12], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1, 2], dtype=np.int32), overwrite=True)
    run.create_array("source_refined_row_ids", data=np.asarray([100, 101, 102], dtype=np.int64), overwrite=True)

    masks = np.zeros((3, 4, 16, 16), dtype=np.uint8)
    for row_idx in range(3):
        masks[row_idx, 0, 2:14, 4:12] = 1
        masks[row_idx, 1] = _disk_mask(16, 16, 5 + row_idx, 5, 2.5)
        masks[row_idx, 2] = _disk_mask(16, 16, 5 + row_idx, 10, 2.5)
        masks[row_idx, 3] = _disk_mask(16, 16, 10, 8, 2.0)
    run.create_array("masks_roi", data=masks, chunks=(2, 1, 16, 16), overwrite=True)
    return root


def test_write_subject_shape_run_group_creates_coherent_components_and_relations(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()

    summary = mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_001",
        chunk_size=2,
    )

    assert summary["status"] == "updated"
    run = root["analysis"]["subject_shape_runs"]["shape_001"]
    assert root["analysis"]["subject_shape_runs"].attrs["latest"] == "shape_001"
    assert run.attrs["schema_id"] == "analysis.subject_shape_runs"
    assert run.attrs["source_refined_subject_masks_run"] == "refined_001"
    assert run.attrs["component_names"] == ["subject_body", "swim_bladder", "eye_left", "eye_right"]
    assert run["row_index"]["frame_indices"][:].tolist() == [10, 11, 12]
    assert run["row_index"]["source_refined_row_ids"][:].tolist() == [100, 101, 102]

    body = run["components"]["subject_body"]
    assert body["mask_present"][:].tolist() == [True, True, True]
    assert np.all(np.asarray(body["principal_axis_valid"][:], dtype=bool))
    assert np.all(np.asarray(body["principal_axis_length_px"][:], dtype=np.float32) > 0)

    eye_pair = run["relations"]["eye_pair"]
    assert eye_pair["separation_valid"][:].tolist() == [True, True, True]
    assert np.all(np.asarray(eye_pair["separation_px"][:], dtype=np.float32) > 0)
    swim_to_body = run["relations"]["swim_bladder_to_body"]
    assert swim_to_body["relation_valid"][:].tolist() == [True, True, True]
    eyes_to_body = run["relations"]["eyes_to_body"]
    assert eyes_to_body["left_eye_relation_valid"][:].tolist() == [True, True, True]

    body_frame = run["body_frame"]
    assert body_frame.attrs["body_frame_estimator"] == "mask_component_axis"
    assert body_frame["valid"][:].tolist() == [True, True, True]
    forward = np.asarray(body_frame["forward_axis_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(forward))
    assert np.all(forward[:, 1] < 0.0)

    swim = run["components"]["swim_bladder"]
    assert swim["caudal_contour_valid"][:].tolist() == [True, True, True]
    caudal = np.asarray(swim["caudal_contour_point_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(caudal))
    assert np.all(caudal[:, 1] > 10.0)

    assert body.attrs["centerline_method"] == "skeleton_longest_endpoint_path_v1"
    assert body["centerline_valid"][:].tolist() == [True, True, True]
    centerline = np.asarray(body["centerline_xy"][:], dtype=np.float32)
    assert centerline.shape == (3, mod.CENTERLINE_SAMPLE_COUNT, 2)
    head = np.asarray(body["head_endpoint_xy"][:], dtype=np.float32)
    tail = np.asarray(body["tail_tip_xy"][:], dtype=np.float32)
    assert np.all(head[:, 1] < tail[:, 1])
    assert body["tail_base_valid"][:].tolist() == [True, True, True]
    assert np.all(np.asarray(body["body_arclength_px"][:], dtype=np.float32) > 0)
    assert np.all(np.asarray(body["tail_segment_arclength_px"][:], dtype=np.float32) >= 0)
    assert run.attrs["provenance"]["stage"] == "analysis.subject_shape_runs"


def test_write_subject_shape_run_uses_dask_worker_chunks(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    _build_refined_root(zarr_path)

    summary = mod.write_subject_shape_run(
        zarr_path,
        refined_run="refined_001",
        run_name="shape_dask",
        chunk_size=1,
        execution_backend="dask_worker_chunks",
        scheduler="single-threaded",
    )

    assert summary["status"] == "updated"
    assert summary["execution_backend"] == "dask_worker_chunks"
    assert summary["chunk_count"] == 3
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["subject_shape_runs"]["shape_dask"]
    assert run.attrs["dask_execution_enabled"] is True
    assert run.attrs["dask_scheduler"] == "single-threaded"
    assert len(run.attrs["subject_shape_chunk_timings"]) == 3
    assert run["components"]["eye_left"]["ellipse_success"][:].tolist() == [True, True, True]
    assert run["relations"]["eye_pair"]["separation_valid"][:].tolist() == [True, True, True]
    assert run["body_frame"]["valid"][:].tolist() == [True, True, True]
    assert run["components"]["subject_body"]["centerline_valid"][:].tolist() == [True, True, True]


def test_write_subject_shape_run_dry_run_does_not_create_analysis_group() -> None:
    root = _build_refined_root()

    summary = mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_dry",
        dry_run=True,
    )

    assert summary["status"] == "planned"
    assert "analysis" not in root


def test_subject_shape_tail_geometry_fails_closed_for_fragmented_body(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()
    refined = root["refined_subject_masks_runs"]["refined_001"]
    masks = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    masks[1, 0] = 0
    masks[1, 0, 2:5, 4:7] = 1
    masks[1, 0, 10:13, 9:12] = 1
    refined["masks_roi"][:] = masks

    mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_fragmented",
        chunk_size=2,
    )

    body = root["analysis"]["subject_shape_runs"]["shape_fragmented"]["components"]["subject_body"]
    assert body["centerline_valid"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["centerline_failure_reason_bytes"][1]) == "fragmented_subject_body_mask"
