from __future__ import annotations

import numpy as np
import zarr

from fisheye.utils import backfill_keypoint_derived_metrics_schema as mod


def _make_refined_run(*, labels: list[str] | None = None, pose_schema: dict | None = None) -> zarr.Group:
    root = zarr.group()
    run = root.create_group("refined")
    if labels is not None:
        run.attrs["keypoint_labels"] = labels
    if pose_schema is not None:
        run.attrs["pose_schema"] = pose_schema
    run.create_array("keypoints_roi", data=np.zeros((2, 3, 2), dtype=np.float32), chunks=(2, 3, 2))
    run.create_array("triangle_area", data=np.asarray([1.0, 2.0], dtype=np.float32), chunks=(2,))
    run.create_array("triangle_angles", data=np.ones((2, 3), dtype=np.float32), chunks=(2, 3))
    run.create_array("min_angle", data=np.asarray([30.0, 31.0], dtype=np.float32), chunks=(2,))
    run.create_array("geometry_valid", data=np.asarray([True, True], dtype=bool), chunks=(2,))
    return run


def test_backfill_run_group_writes_schema_from_keypoint_labels() -> None:
    run = _make_refined_run(labels=["swim_bladder", "eye_left", "eye_right"])

    result = mod._backfill_run_group(run, overwrite_existing=False, apply=True)

    assert result.status == "ok"
    schema = run.attrs["derived_metrics_schema"]
    assert schema["schema_version"] == 1
    assert schema["entity_kind"] == "keypoint_roi"
    metric = schema["metrics"][0]
    assert metric["name"] == "eye_triangle_geometry"
    assert metric["selectors"]["indices"] == [0, 1, 2]
    assert metric["selectors"]["labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert [item["array"] for item in metric["outputs"]] == [
        "triangle_area",
        "triangle_angles",
        "min_angle",
    ]
    assert schema["quality_gates"][0]["output"]["array"] == "geometry_valid"
    assert "derived_metrics_schema_backfilled_at_utc" in run.attrs


def test_backfill_run_group_dry_run_does_not_write() -> None:
    run = _make_refined_run(labels=["swim_bladder", "eye_left", "eye_right"])

    result = mod._backfill_run_group(run, overwrite_existing=False, apply=False)

    assert result.status == "ok"
    assert "derived_metrics_schema" not in run.attrs
    assert "derived_metrics_schema_backfilled_at_utc" not in run.attrs


def test_backfill_run_group_uses_pose_schema_labels() -> None:
    run = _make_refined_run(
        pose_schema={
            "nodes": [
                {"id": 0, "name": "swim_bladder"},
                {"id": 1, "name": "eye_left"},
                {"id": 2, "name": "eye_right"},
            ]
        }
    )

    result = mod._backfill_run_group(run, overwrite_existing=False, apply=True)

    assert result.status == "ok"
    assert run.attrs["derived_metrics_schema"]["metrics"][0]["selectors"]["labels"] == [
        "swim_bladder",
        "eye_left",
        "eye_right",
    ]


def test_backfill_run_group_skips_existing_without_overwrite() -> None:
    run = _make_refined_run(labels=["swim_bladder", "eye_left", "eye_right"])
    run.attrs["derived_metrics_schema"] = {"existing": True}

    result = mod._backfill_run_group(run, overwrite_existing=False, apply=True)

    assert result.status == "skipped_existing"
    assert run.attrs["derived_metrics_schema"] == {"existing": True}


def test_backfill_run_group_overwrites_existing_when_requested() -> None:
    run = _make_refined_run(labels=["swim_bladder", "eye_left", "eye_right"])
    run.attrs["derived_metrics_schema"] = {"existing": True}

    result = mod._backfill_run_group(run, overwrite_existing=True, apply=True)

    assert result.status == "ok"
    assert run.attrs["derived_metrics_schema"]["schema_version"] == 1


def test_backfill_run_group_requires_existing_metric_arrays() -> None:
    run = _make_refined_run(labels=["swim_bladder", "eye_left", "eye_right"])
    del run["geometry_valid"]

    result = mod._backfill_run_group(run, overwrite_existing=False, apply=True)

    assert result.status == "missing_arrays"
    assert result.reason == "geometry_valid"
    assert "derived_metrics_schema" not in run.attrs


def test_backfill_run_group_reports_missing_labels() -> None:
    run = _make_refined_run()

    result = mod._backfill_run_group(run, overwrite_existing=False, apply=True)

    assert result.status == "missing_labels"
    assert "derived_metrics_schema" not in run.attrs


def test_backfill_run_group_reports_unsupported_labels() -> None:
    run = _make_refined_run(labels=["tail", "midline", "snout"])

    result = mod._backfill_run_group(run, overwrite_existing=False, apply=True)

    assert result.status == "unsupported_labels"
    assert "derived_metrics_schema" not in run.attrs
