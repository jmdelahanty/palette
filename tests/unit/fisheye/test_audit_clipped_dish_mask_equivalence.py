from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.utils.audit_clipped_dish_mask_equivalence import (
    AUDIT_SCHEMA,
    audit_clipped_dish_mask_equivalence,
)
from fisheye.utils.finalize_clipped_detect_refine_workflow import COLLECTION_SCHEMA


def _root() -> zarr.Group:
    root = zarr.group(store=zarr.storage.MemoryStore())
    root.require_group("analysis_metadata").attrs["dish_mask"] = {
        "shape": "circle",
        "detected_circle": {"center": [50, 50], "radius": 40},
        "metrics": {"image_shape": [100, 100]},
    }
    return root


def _add_refined_run(
    root: zarr.Group,
    *,
    clip_index: int,
    bboxes: np.ndarray,
    keys: np.ndarray,
    complete: bool = True,
) -> dict[str, object]:
    clip_id = f"clip_{clip_index:06d}"
    camera_serial = "2010094"
    path = (
        f"clips/{clip_id}/cameras/{camera_serial}/refined_detect_runs/"
        f"refined_{clip_id}"
    )
    run = root.require_group(path)
    if complete:
        run.attrs["palette_run_completion_status"] = "complete"
    instances = run.require_group("instances")
    values = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
    instances.create_array("bbox_norm_coords", data=values)
    instances.create_array("instance_key", data=np.asarray(keys, dtype=np.uint64))
    instances.create_array(
        "frame_indices",
        data=np.arange(values.shape[0], dtype=np.int32),
    )
    return {
        "work_unit_id": f"recording_{clip_id}_cam{camera_serial}",
        "clip_id": clip_id,
        "clip_index": clip_index,
        "camera_serial": camera_serial,
        "refined_group_path": path,
    }


def _publish_collection(root: zarr.Group, selected: list[dict[str, object]]) -> str:
    collection_id = "collection_test"
    path = f"experiment_index/finalized_runs/{collection_id}"
    collection = root.require_group(path)
    collection.attrs.update(
        {
            "schema_version": COLLECTION_SCHEMA,
            "collection_id": collection_id,
            "selected_run_count": len(selected),
            "selected_runs": selected,
        }
    )
    refined_parent = root.require_group("refined_detect_runs")
    refined_parent.attrs["latest_collection"] = collection_id
    refined_parent.attrs["latest_collection_path"] = path
    return path


def _audit(root: zarr.Group, tmp_path: Path) -> dict[str, object]:
    return audit_clipped_dish_mask_equivalence(
        tmp_path / "analysis.zarr",
        output_parquet=tmp_path / "outside.parquet",
        chunk_rows=1,
        dish_mask_boundary_tolerance_mm=0.0,
        root=root,
    )


def test_audit_reports_equivalent_without_materializing_collection(tmp_path: Path) -> None:
    root = _root()
    first = _add_refined_run(
        root,
        clip_index=0,
        bboxes=np.asarray([[0.5, 0.5, 0.1, 0.1], [0.6, 0.5, 0.1, 0.1]]),
        keys=np.asarray([11, 12], dtype=np.uint64),
    )
    second = _add_refined_run(
        root,
        clip_index=1,
        bboxes=np.asarray([[0.5, 0.6, 0.1, 0.1]]),
        keys=np.asarray([13], dtype=np.uint64),
    )
    _publish_collection(root, [second, first])

    report = _audit(root, tmp_path)

    assert report["schema_version"] == AUDIT_SCHEMA
    assert report["equivalence_status"] == "equivalent"
    assert report["selected_row_count"] == 3
    assert report["outside_dish_mask_count"] == 0
    assert report["instance_key_uniqueness"]["status"] == "unique"
    assert report["streaming"]["full_collection_materialized_in_ram"] is False
    assert report["runs"][0]["clip_id"] == "clip_000000"
    evidence = report["affected_rows_parquet"]
    assert evidence["storage_role"] == "sparse_audit_evidence_only"
    assert evidence["canonical_identity_authority"] is False
    assert pq.read_table(tmp_path / "outside.parquet").num_rows == 0


def test_audit_writes_only_outside_rows_with_canonical_instance_keys(tmp_path: Path) -> None:
    root = _root()
    selected = _add_refined_run(
        root,
        clip_index=0,
        bboxes=np.asarray(
            [
                [0.5, 0.5, 0.1, 0.1],
                [0.95, 0.5, 0.1, 0.1],
                [np.nan, 0.5, 0.1, 0.1],
            ]
        ),
        keys=np.asarray([21, 22, 23], dtype=np.uint64),
    )
    _publish_collection(root, [selected])

    report = _audit(root, tmp_path)
    affected = pq.read_table(tmp_path / "outside.parquet").to_pydict()

    assert report["equivalence_status"] == "not_equivalent"
    assert report["outside_dish_mask_count"] == 2
    assert affected["instance_key"] == [22, 23]
    assert affected["member_row_index"] == [1, 2]
    assert affected["clip_local_frame_index"] == [1, 2]
    assert affected["dish_circle_radial_ratio"][0] > 1.0
    assert affected["dish_circle_radial_ratio"][1] is None
    assert report["outside_geometry"]["maximum"] > 1.0


def test_audit_default_half_mm_tolerance_uses_explicit_camera_calibration(tmp_path: Path) -> None:
    root = _root()
    root.require_group("raw_video").attrs.update(
        {"source_video_height": 4512, "source_video_width": 4512}
    )
    selected = _add_refined_run(
        root,
        clip_index=0,
        bboxes=np.asarray([[0.905, 0.5, 0.1, 0.1]]),
        keys=np.asarray([24], dtype=np.uint64),
    )
    _publish_collection(root, [selected])

    report = audit_clipped_dish_mask_equivalence(
        tmp_path / "analysis.zarr",
        output_parquet=tmp_path / "outside.parquet",
        pixels_per_mm_camera=50.0,
        root=root,
    )

    tolerance = report["dish_mask_boundary_tolerance"]
    assert tolerance["requested_mm"] == pytest.approx(0.5)
    assert tolerance["tolerance_source_px"] == pytest.approx(25.0)
    assert tolerance["tolerance_norm_x"] == pytest.approx(25.0 / 4512.0)
    assert tolerance["calibration_source"] == "explicit_argument"
    assert report["equivalence_status"] == "equivalent"


def test_audit_fails_closed_on_collection_wide_duplicate_instance_key(tmp_path: Path) -> None:
    root = _root()
    first = _add_refined_run(
        root,
        clip_index=0,
        bboxes=np.asarray([[0.5, 0.5, 0.1, 0.1]]),
        keys=np.asarray([31], dtype=np.uint64),
    )
    second = _add_refined_run(
        root,
        clip_index=1,
        bboxes=np.asarray([[0.5, 0.5, 0.1, 0.1]]),
        keys=np.asarray([31], dtype=np.uint64),
    )
    _publish_collection(root, [first, second])

    with pytest.raises(ValueError, match="duplicate instance_key"):
        _audit(root, tmp_path)
    assert not (tmp_path / "outside.parquet").exists()


def test_audit_fails_closed_on_incomplete_selected_run(tmp_path: Path) -> None:
    root = _root()
    selected = _add_refined_run(
        root,
        clip_index=0,
        bboxes=np.asarray([[0.5, 0.5, 0.1, 0.1]]),
        keys=np.asarray([41], dtype=np.uint64),
        complete=False,
    )
    _publish_collection(root, [selected])

    with pytest.raises(ValueError, match="not explicitly complete"):
        _audit(root, tmp_path)
    assert not (tmp_path / "outside.parquet").exists()
