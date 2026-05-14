from __future__ import annotations

import numpy as np
import zarr
from pathlib import Path

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.utils import migrate_legacy_detect_labels as mod


def _write_sparse_detection_group(
    group: zarr.Group,
    *,
    frames: np.ndarray,
    boxes: np.ndarray,
    reasons: np.ndarray,
    scores: np.ndarray | None = None,
) -> None:
    group.create_array("frame_indices", data=np.asarray(frames, dtype=np.int32))
    group.create_array("bbox_norm_coords", data=np.asarray(boxes, dtype=np.float64))
    group.create_array(
        "frame_counts",
        data=np.bincount(np.asarray(frames, dtype=np.int32), minlength=5).astype(np.int32),
    )
    group.create_array(
        "class_ids",
        data=np.zeros(int(frames.shape[0]), dtype=np.int32),
    )
    group.create_array(
        "scores",
        data=np.ones(int(frames.shape[0]), dtype=np.float32) if scores is None else scores,
    )
    write_reason_columns(
        group,
        np.asarray(reasons, dtype=object),
        chunk_size=max(1, int(frames.shape[0])),
        include_reason_text=True,
        overwrite=True,
    )


def _make_legacy_zarr(path: Path) -> Path:
    root = zarr.open_group(store=path, mode="w")
    root.attrs["width"] = 100
    root.attrs["height"] = 50
    root.attrs["total_frames"] = 5

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_1"
    detect = detect_parent.create_group("detect_1")
    _write_sparse_detection_group(
        detect,
        frames=np.asarray([0, 2], dtype=np.int32),
        boxes=np.asarray([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.2]], dtype=np.float64),
        reasons=np.asarray(["raw", "raw"], dtype=object),
    )

    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "refined_legacy"
    source = parent.create_group("refined_legacy")
    source.attrs["source_detect_run"] = "detect_1"
    source.attrs["coverage_frames_total"] = 5
    source.attrs["manual_review_latest"] = "manual"
    source.attrs["detect_review_status"] = {"state": "pending", "method": "algorithmic"}
    source.attrs["summary_statistics"] = {
        "rows_present": 2,
        "rows_missing": 3,
        "frame_count_covered": 2,
    }

    instances = source.create_group("instances")
    instances.create_array("refined_row_ids", data=np.asarray([0, 2], dtype=np.int64))
    instances.create_array("frame_indices", data=np.asarray([0, 2], dtype=np.int32))
    instances.create_array("frame_offsets", data=np.asarray([0, 1, 1, 2, 2, 2], dtype=np.int64))
    instances.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.2]], dtype=np.float64),
    )
    instances.create_array("bbox_img_xyxy", data=np.zeros((2, 4), dtype=np.float64))
    instances.create_array("source_kind_codes", data=np.asarray([1, 1], dtype=np.int8))
    instances.create_array("manual_edit_flags", data=np.asarray([False, False]))
    instances.create_array("source_detect_row_index", data=np.asarray([0, 1], dtype=np.int32))
    instances.create_array("frame_counts", data=np.asarray([1, 0, 1, 0, 0], dtype=np.int32))
    write_reason_columns(
        instances,
        np.asarray(["kept", "kept"], dtype=object),
        chunk_size=2,
        include_reason_text=True,
        overwrite=True,
    )

    manual = source.create_group("manual")
    _write_sparse_detection_group(
        manual,
        frames=np.arange(5, dtype=np.int32),
        boxes=np.asarray(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.1, 0.2, 0.1, 0.1],
                [0.3, 0.4, 0.1, 0.2],
                [0.4, 0.5, 0.1, 0.1],
                [0.6, 0.7, 0.1, 0.1],
            ],
            dtype=np.float64,
        ),
        reasons=np.asarray(["kept", "retune", "kept", "retune", "retune"], dtype=object),
    )
    manual.create_array("retune_id", data=np.asarray([-1, 7, -1, 7, 7], dtype=np.int32))
    return path


def test_build_plan_prefers_complete_legacy_manual_group(tmp_path: Path) -> None:
    zarr_path = _make_legacy_zarr(tmp_path / "legacy.zarr")
    root = zarr.open_group(store=zarr_path, mode="r")

    plan = mod.build_legacy_detect_label_plan(root, zarr_path=zarr_path)

    assert plan.complete is True
    assert plan.source_group == "manual"
    assert plan.total_frames == 5
    assert plan.canonical_rows_before == 2
    assert plan.canonical_missing_before == 3
    assert plan.source_rows == 5


def test_migrate_legacy_manual_labels_writes_complete_canonical_run(tmp_path: Path) -> None:
    zarr_path = _make_legacy_zarr(tmp_path / "legacy.zarr")

    result = mod.migrate_legacy_detect_labels_for_zarr(
        zarr_path,
        output_run_name="refined_legacy_manual_canonical",
        reviewer="tester",
        notes="unit test migration",
        apply=True,
        write_profile=False,
        sync_registry=False,
    )

    assert isinstance(result, mod.LegacyLabelApplyResult)
    assert result.rows_written == 5
    assert result.rows_manual == 3
    assert result.rows_raw_detect == 2

    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_detect_runs"]
    assert parent.attrs["latest"] == "refined_legacy_manual_canonical"
    assert parent.attrs["detect_review_status_latest"] == "refined_legacy_manual_canonical"
    run = parent["refined_legacy_manual_canonical"]
    status = dict(run.attrs["detect_review_status"])
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["migration_method"] == "legacy_detect_labels_to_sparse_instances_v1"
    assert status["migration_source_group"] == "manual"
    summary = dict(run.attrs["summary_statistics"])
    assert summary["rows_present"] == 5
    assert summary["rows_missing"] == 0
    assert summary["rows_manual"] == 3
    assert summary["rows_raw_detect"] == 2
    np.testing.assert_array_equal(run["instances/frame_counts"][:], np.ones(5, dtype=np.int32))
