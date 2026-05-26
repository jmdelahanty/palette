from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
)
from fisheye.utils.finalize_clipped_detect_refine_workflow import (
    COLLECTION_SCHEMA,
    finalize_clipped_detect_refine_workflow,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import PLAN_SCHEMA


def _array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    group.create_array(name, data=np.asarray(data), overwrite=True)


def _write_sparse_refined_run(
    root: zarr.Group,
    *,
    clip_id: str,
    camera_serial: str,
    detect_run: str,
    refined_run: str,
    frame_count: int,
) -> None:
    source_detect_path = f"clips/{clip_id}/cameras/{camera_serial}/detect_runs/{detect_run}"
    root.require_group(source_detect_path)
    refined_family_path = f"clips/{clip_id}/cameras/{camera_serial}/refined_detect_runs"
    refined = root.require_group(f"{refined_family_path}/{refined_run}")
    refined.attrs.update(
        {
            "source_detect_run": detect_run,
            "source_detect_path": source_detect_path,
            "source_quality_run": f"detect_quality_{clip_id}",
            "refined_family_path": refined_family_path,
            "curated_primary_surface": "instances",
            "row_identity_policy": "stable_sparse_refined_row_id",
        }
    )

    raw_detect = REFINED_SOURCE_KIND_CODE_MAP["raw_detect"]
    accepted = REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"]
    instances = refined.require_group("instances")
    _array(instances, "refined_row_ids", np.asarray([0], dtype=np.int64))
    _array(instances, "frame_indices", np.asarray([0], dtype=np.int32))
    frame_counts = np.zeros((frame_count,), dtype=np.int32)
    frame_counts[0] = 1
    frame_offsets = np.zeros((frame_count + 1,), dtype=np.int64)
    frame_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)
    _array(instances, "frame_offsets", frame_offsets)
    _array(instances, "bbox_img_xyxy", np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64))
    _array(instances, "bbox_norm_coords", np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64))
    _array(instances, "source_kind_codes", np.asarray([raw_detect], dtype=np.int8))
    _array(instances, "manual_edit_flags", np.asarray([False], dtype=bool))
    _array(instances, "source_detect_row_index", np.asarray([0], dtype=np.int32))
    _array(instances, "frame_counts", frame_counts)

    source = refined.require_group("source_detections")
    _array(source, "source_detect_row_index", np.asarray([0], dtype=np.int32))
    _array(source, "frame_indices", np.asarray([0], dtype=np.int32))
    _array(source, "bbox_img_xyxy", np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64))
    _array(source, "bbox_norm_coords", np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64))
    _array(source, "decision_codes", np.asarray([accepted], dtype=np.int8))
    _array(source, "resolved_refined_row_id", np.asarray([0], dtype=np.int64))


def _unit(zarr_path: Path, *, clip_id: str, frame_count: int) -> dict[str, object]:
    camera_serial = "2010093"
    detect_run = f"detect_wf_{clip_id}_cam{camera_serial}"
    quality_run = f"detect_quality_wf_{clip_id}_cam{camera_serial}"
    refined_run = f"refined_detect_wf_{clip_id}_cam{camera_serial}"
    return {
        "work_unit_id": f"recording_{clip_id}_cam{camera_serial}",
        "recording_id": "recording",
        "clip_id": clip_id,
        "clip_index": int(clip_id.rsplit("_", 1)[-1]),
        "camera_serial": camera_serial,
        "frame_count": frame_count,
        "source": {"video_path": f"/recording/clips/{clip_id}/Cam{camera_serial}.mp4"},
        "run_names": {
            "detect": detect_run,
            "detect_quality": quality_run,
            "refined_detect": refined_run,
        },
        "zarr_paths": {
            "detect_target_group_path": f"clips/{clip_id}/cameras/{camera_serial}/detect_runs/{detect_run}",
            "refined_group_path": f"clips/{clip_id}/cameras/{camera_serial}/refined_detect_runs/{refined_run}",
        },
    }


def _write_plan(path: Path, *, zarr_path: Path, frame_count: int = 3, units: int = 1) -> Path:
    payload = {
        "schema_version": PLAN_SCHEMA,
        "workflow_id": "wf",
        "recording_id": "recording",
        "recording_dir": str(path.parent),
        "analysis_zarr": str(zarr_path),
        "work_units": [
            _unit(zarr_path, clip_id=f"clip_{idx:06d}", frame_count=frame_count)
            for idx in range(units)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_recording_frame_index(recording_dir: Path, *, units: list[dict[str, object]]) -> None:
    rows = []
    recording_frame_id = 1
    for unit in units:
        for local_index in range(int(unit["frame_count"])):
            rows.append(
                {
                    "camera_serial": str(unit["camera_serial"]),
                    "clip_id": str(unit["clip_id"]),
                    "clip_local_frame_index": local_index,
                    "recording_frame_id": recording_frame_id,
                }
            )
            recording_frame_id += 1
    frame_index_path = recording_dir / "recording_frame_index.parquet"
    pq.write_table(pa.Table.from_pylist(rows), frame_index_path)
    (recording_dir / "recording_frame_index_manifest.json").write_text(
        json.dumps({"recording_frame_index_path": str(frame_index_path)}),
        encoding="utf-8",
    )


def _load_plan_units(plan_path: Path) -> list[dict[str, object]]:
    return json.loads(plan_path.read_text())["work_units"]


def test_finalize_clipped_detect_refine_workflow_writes_collection_manifest(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    plan_path = _write_plan(tmp_path / "plan.json", zarr_path=zarr_path, frame_count=3)
    units = _load_plan_units(plan_path)
    _write_recording_frame_index(tmp_path, units=units)
    unit = units[0]
    _write_sparse_refined_run(
        root,
        clip_id=unit["clip_id"],
        camera_serial=unit["camera_serial"],
        detect_run=unit["run_names"]["detect"],
        refined_run=unit["run_names"]["refined_detect"],
        frame_count=unit["frame_count"],
    )

    result = finalize_clipped_detect_refine_workflow(plan_path, apply=True)

    assert result["status"] == "ok"
    assert result["applied"] is True
    reopened = zarr.open_group(str(zarr_path), mode="r")
    collection = reopened["experiment_index/finalized_runs/wf"]
    assert collection.attrs["schema_version"] == COLLECTION_SCHEMA
    assert collection.attrs["selected_run_count"] == 1
    assert collection.attrs["selected_runs"][0]["refined_group_path"].endswith(
        "/refined_detect_wf_clip_000000_cam2010093"
    )
    assert reopened["refined_detect_runs"].attrs["latest_collection"] == "wf"


def test_finalize_clipped_detect_refine_workflow_uses_relocated_frame_index(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    plan_path = _write_plan(tmp_path / "plan.json", zarr_path=zarr_path, frame_count=3)
    units = _load_plan_units(plan_path)
    _write_recording_frame_index(tmp_path, units=units)
    old_recording_dir = tmp_path.parent / f"{tmp_path.name}_old_recording"
    old_recording_dir.mkdir()
    (old_recording_dir / "recording_frame_index.parquet").write_text("stale source file", encoding="utf-8")
    (tmp_path / "recording_frame_index_manifest.json").write_text(
        json.dumps({"recording_frame_index_path": str(old_recording_dir / "recording_frame_index.parquet")}),
        encoding="utf-8",
    )
    unit = units[0]
    _write_sparse_refined_run(
        root,
        clip_id=unit["clip_id"],
        camera_serial=unit["camera_serial"],
        detect_run=unit["run_names"]["detect"],
        refined_run=unit["run_names"]["refined_detect"],
        frame_count=unit["frame_count"],
    )

    result = finalize_clipped_detect_refine_workflow(plan_path, apply=True)

    assert result["status"] == "ok"
    assert result["manifest"]["recording_frame_index_validation"]["path"] == str(
        (tmp_path / "recording_frame_index.parquet").resolve()
    )


def test_finalize_clipped_detect_refine_workflow_fails_missing_refined_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")
    plan_path = _write_plan(tmp_path / "plan.json", zarr_path=zarr_path)
    _write_recording_frame_index(tmp_path, units=_load_plan_units(plan_path))

    result = finalize_clipped_detect_refine_workflow(plan_path, apply=True)

    assert result["status"] == "failed"
    assert result["applied"] is False
    assert "refined validation failed" in "\n".join(result["errors"])


def test_finalize_clipped_detect_refine_workflow_fails_frame_count_mismatch(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    plan_path = _write_plan(tmp_path / "plan.json", zarr_path=zarr_path, frame_count=4)
    units = _load_plan_units(plan_path)
    _write_recording_frame_index(tmp_path, units=units)
    unit = units[0]
    _write_sparse_refined_run(
        root,
        clip_id=unit["clip_id"],
        camera_serial=unit["camera_serial"],
        detect_run=unit["run_names"]["detect"],
        refined_run=unit["run_names"]["refined_detect"],
        frame_count=3,
    )

    result = finalize_clipped_detect_refine_workflow(plan_path)

    assert result["status"] == "failed"
    assert "does not match planned frame_count 4" in "\n".join(result["errors"])


def test_finalize_clipped_detect_refine_workflow_requires_submission_stage_statuses(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    plan_path = _write_plan(tmp_path / "plan.json", zarr_path=zarr_path, frame_count=3)
    units = _load_plan_units(plan_path)
    _write_recording_frame_index(tmp_path, units=units)
    unit = units[0]
    _write_sparse_refined_run(
        root,
        clip_id=unit["clip_id"],
        camera_serial=unit["camera_serial"],
        detect_run=unit["run_names"]["detect"],
        refined_run=unit["run_names"]["refined_detect"],
        frame_count=unit["frame_count"],
    )
    missing_status = tmp_path / "missing_status.json"
    submission_manifest = tmp_path / "submission_manifest.json"
    submission_manifest.write_text(
        json.dumps(
            {
                "work_items": [
                    {
                        "work_unit_id": unit["work_unit_id"],
                        "stages": [
                            {
                                "stage": "validate_refined_detect",
                                "status_json": str(missing_status),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = finalize_clipped_detect_refine_workflow(
        plan_path,
        submission_manifest=submission_manifest,
    )

    assert result["status"] == "failed"
    assert "missing stage status JSON" in "\n".join(result["errors"])
