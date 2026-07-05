"""Row-equality proof: reconcile profile extractors vs the retired sync scripts.

Checkpoint 1 of the registry-reconcile brief re-expresses the detection and
keypoint profile syncs as extractors behind ``reconcile_dataset_from_root``.
Before the standalone ``sync_*_profile_registry.py`` scripts are deleted, these
tests prove the new extractor path builds byte-identical ``*_data_profile`` rows
(modulo the volatile ``updated_utc`` write-timestamp) on the same fixture.
"""

from __future__ import annotations

from pathlib import Path
import sys

import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.extractors.detect_performance import (
    _extract_detection_data_profile_rows,
)
from fisheye.registry.extractors.keypoint_performance import (
    _extract_keypoint_profile_rows,
)
from fisheye.utils import sync_detection_profile_registry as det_sync
from fisheye.utils import sync_keypoint_profile_registry as kpt_sync


def _detection_summary() -> dict:
    return {
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {"dataset_id": "d", "recording_id": "rec_summary", "zarr_use": "training"},
        "source": {
            "detection_path": "refined_detect_runs/refined_detect_2026-02-12/manual",
            "detection_type": "manual",
        },
        "coverage": {"frames_total": 100, "frames_with_detections": 95, "coverage_percent": 95.0},
        "counts": {"detections_total": 950, "detections_per_frame": {"p50": 9.0, "p90": 10.0}},
        "geometry_norm": {
            "w": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "h": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "area": {"p10": 0.01, "p50": 0.04, "p90": 0.09},
            "aspect_ratio": {"p10": 0.8, "p50": 1.0, "p90": 1.2},
        },
        "spatial": {"edge_proximity_rate": 0.03},
        "composition": {
            "rig_id": "rig_a",
            "camera_id": "cam_1",
            "arena_id": "arena_x",
            "dish_design": "cedar",
            "canvas_name": "canvas_a",
            "protocol_name": "DefaultScreen",
            "genotype": "Tg(elavl3:gcamp7f)",
            "dpf_at_acquisition": 7,
        },
    }


def _keypoint_summary() -> dict:
    return {
        "schema_name": "keypoint_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {"dataset_id": "d", "recording_id": "rec_summary", "zarr_use": "training"},
        "source": {
            "keypoint_method": "yolo",
            "keypoint_path": "analysis/refined_keypoint_runs/run_a",
            "keypoint_run": "run_a",
            "skeleton_id": "skel_v1",
            "kpt_shape": [7, 3],
            "pose_schema_name": "fish_v1",
            "pose_schema": {"name": "fish_v1", "n": 7},
            "heading_computation_source": "triangle",
            "heading_computation": {"method": "triangle"},
        },
        "quality": {
            "rows_total": 100,
            "rows_usable": 90,
            "usable_keypoints_total": 630,
            "usable_rate": 0.9,
            "confidence_valid_rate": 0.95,
            "geometry_valid_rate": 0.92,
        },
        "geometry": {
            "triangle_area": {"stats": {"p10": 1.0, "p50": 2.0, "p90": 3.0}},
            "min_angle": {"stats": {"p10": 10.0, "p50": 20.0, "p90": 30.0}},
            "heading": {"stats": {"p10": 0.1, "p50": 0.2, "p90": 0.3}},
        },
        "composition": {
            "rig_id": "rig_a",
            "camera_id": "cam_1",
            "arena_id": "arena_x",
            "dish_design": "cedar",
            "canvas_name": "canvas_a",
            "protocol_name": "DefaultScreen",
            "genotype": "Tg(elavl3:gcamp7f)",
            "dpf_at_acquisition": 7,
        },
    }


def _write_profile_run(zarr_path: Path, parent_name: str, run_name: str, summary: dict) -> dict:
    root = zarr.open_group(str(zarr_path), mode="a")
    analysis = root.require_group("analysis")
    parent = analysis.require_group(parent_name)
    run_group = parent.require_group(run_name)
    run_group.attrs["profile_summary"] = summary
    run_group.attrs["created_at_utc"] = summary["created_at_utc"]
    parent.attrs["latest"] = run_name
    return dict(run_group.attrs)


def _drop_volatile(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "updated_utc"}


def test_detection_extractor_matches_retired_sync_builder(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_a_training.zarr"
    run_name = "detection_profile_2026-02-12_03-00-00"
    summary = _detection_summary()
    run_attrs = _write_profile_run(zarr_path, "detection_profile_runs", run_name, summary)

    root = zarr.open_group(str(zarr_path), mode="r")
    extractor_rows = _extract_detection_data_profile_rows(
        root,
        zarr_path=zarr_path,
        dataset_id="d",
        recording_id="rec_fallback",
        zarr_use="training",
        genotype=None,
        dpf_at_acquisition=None,
    )
    assert len(extractor_rows) == 1

    sync_row = det_sync._build_profile_payload(
        dataset_id="d",
        fallback_recording_id="rec_fallback",
        fallback_zarr_use="training",
        fallback_genotype=None,
        fallback_dpf_at_acquisition=None,
        profile_run=run_name,
        summary=summary,
        run_attrs=run_attrs,
        zarr_path=zarr_path,
    )

    assert _drop_volatile(extractor_rows[0]) == sync_row


def test_keypoint_extractor_matches_retired_sync_builder(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_b_training.zarr"
    run_name = "keypoint_profile_2026-02-12_03-00-00"
    summary = _keypoint_summary()
    run_attrs = _write_profile_run(zarr_path, "keypoint_profile_runs", run_name, summary)

    root = zarr.open_group(str(zarr_path), mode="r")
    extractor_rows = _extract_keypoint_profile_rows(
        root,
        zarr_path=zarr_path,
        dataset_id="d",
        recording_id="rec_fallback",
        zarr_use="training",
        genotype=None,
        dpf_at_acquisition=None,
    )
    assert len(extractor_rows) == 1

    sync_row = kpt_sync._build_profile_payload(
        dataset_id="d",
        fallback_recording_id="rec_fallback",
        fallback_zarr_use="training",
        fallback_genotype=None,
        fallback_dpf_at_acquisition=None,
        profile_run=run_name,
        summary=summary,
        run_attrs=run_attrs,
        zarr_path=zarr_path,
    )

    assert _drop_volatile(extractor_rows[0]) == sync_row
