"""Golden-row regression for the reconcile profile extractors.

Checkpoint 1 of the registry-reconcile brief re-expressed the detection and
keypoint profile syncs as extractors behind ``reconcile_dataset_from_root`` and
retired the standalone ``sync_*_profile_registry.py`` scripts. Before those
scripts were deleted (commit adding this file's predecessor), a live equality
test proved the extractors build byte-identical ``*_data_profile`` rows to the
scripts' ``_build_profile_payload`` on this exact fixture. Those golden rows are
frozen here so the row contract is protected now that the scripts are gone.
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

_NON_GOLDEN = ("updated_utc", "zarr_mtime_ns")

_DETECTION_PROFILE_JSON = (
    '{"composition":{"arena_id":"arena_x","camera_id":"cam_1","canvas_name":"canvas_a",'
    '"dish_design":"cedar","dpf_at_acquisition":7,"genotype":"Tg(elavl3:gcamp7f)",'
    '"protocol_name":"DefaultScreen","rig_id":"rig_a"},"counts":{"detections_per_frame":'
    '{"p50":9.0,"p90":10.0},"detections_total":950},"coverage":{"coverage_percent":95.0,'
    '"frames_total":100,"frames_with_detections":95},"created_at_utc":'
    '"2026-02-12T03:00:00+00:00","dataset":{"dataset_id":"d","recording_id":"rec_summary",'
    '"zarr_use":"training"},"geometry_norm":{"area":{"p10":0.01,"p50":0.04,"p90":0.09},'
    '"aspect_ratio":{"p10":0.8,"p50":1.0,"p90":1.2},"h":{"p10":0.1,"p50":0.2,"p90":0.3},'
    '"w":{"p10":0.1,"p50":0.2,"p90":0.3}},"schema_name":"detection_dataset_profile",'
    '"schema_version":"v1","source":{"detection_path":'
    '"refined_detect_runs/refined_detect_2026-02-12/manual","detection_type":"manual"},'
    '"spatial":{"edge_proximity_rate":0.03}}'
)

_KEYPOINT_PROFILE_JSON = (
    '{"composition":{"arena_id":"arena_x","camera_id":"cam_1","canvas_name":"canvas_a",'
    '"dish_design":"cedar","dpf_at_acquisition":7,"genotype":"Tg(elavl3:gcamp7f)",'
    '"protocol_name":"DefaultScreen","rig_id":"rig_a"},"created_at_utc":'
    '"2026-02-12T03:00:00+00:00","dataset":{"dataset_id":"d","recording_id":"rec_summary",'
    '"zarr_use":"training"},"geometry":{"heading":{"stats":{"p10":0.1,"p50":0.2,"p90":0.3}},'
    '"min_angle":{"stats":{"p10":10.0,"p50":20.0,"p90":30.0}},"triangle_area":'
    '{"stats":{"p10":1.0,"p50":2.0,"p90":3.0}}},"quality":{"confidence_valid_rate":0.95,'
    '"geometry_valid_rate":0.92,"rows_total":100,"rows_usable":90,'
    '"usable_keypoints_total":630,"usable_rate":0.9},"schema_name":'
    '"keypoint_dataset_profile","schema_version":"v1","source":{"heading_computation":'
    '{"method":"triangle"},"heading_computation_source":"triangle","keypoint_method":"yolo",'
    '"keypoint_path":"analysis/refined_keypoint_runs/run_a","keypoint_run":"run_a",'
    '"kpt_shape":[7,3],"pose_schema":{"n":7,"name":"fish_v1"},"pose_schema_name":"fish_v1",'
    '"skeleton_id":"skel_v1"}}'
)

_DETECTION_GOLDEN = {
    "dataset_id": "d",
    "profile_run": "detection_profile_2026-02-12_03-00-00",
    "recording_id": "rec_summary",
    "zarr_use": "training",
    "detection_type": "manual",
    "detection_path": "refined_detect_runs/refined_detect_2026-02-12/manual",
    "profile_created_utc": "2026-02-12T03:00:00+00:00",
    "frames_total": 100,
    "frames_with_detections": 95,
    "coverage_percent": 95.0,
    "detections_total": 950,
    "detections_per_frame_p50": 9.0,
    "detections_per_frame_p90": 10.0,
    "w_p10": 0.1,
    "w_p50": 0.2,
    "w_p90": 0.3,
    "h_p10": 0.1,
    "h_p50": 0.2,
    "h_p90": 0.3,
    "area_p10": 0.01,
    "area_p50": 0.04,
    "area_p90": 0.09,
    "aspect_ratio_p10": 0.8,
    "aspect_ratio_p50": 1.0,
    "aspect_ratio_p90": 1.2,
    "edge_proximity_rate": 0.03,
    "rig_id": "rig_a",
    "camera_id": "cam_1",
    "arena_id": "arena_x",
    "dish_design": "cedar",
    "canvas_name": "canvas_a",
    "protocol_name": "DefaultScreen",
    "genotype": "Tg(elavl3:gcamp7f)",
    "dpf_at_acquisition": 7,
    "profile_json": _DETECTION_PROFILE_JSON,
}

_KEYPOINT_GOLDEN = {
    "dataset_id": "d",
    "profile_run": "keypoint_profile_2026-02-12_03-00-00",
    "recording_id": "rec_summary",
    "zarr_use": "training",
    "keypoint_method": "yolo",
    "source_keypoint_path": "analysis/refined_keypoint_runs/run_a",
    "source_keypoint_run": "run_a",
    "skeleton_id": "skel_v1",
    "kpt_shape": "[7,3]",
    "pose_schema_name": "fish_v1",
    "pose_schema_json": '{"n":7,"name":"fish_v1"}',
    "heading_computation_source": "triangle",
    "heading_computation_json": '{"method":"triangle"}',
    "profile_created_utc": "2026-02-12T03:00:00+00:00",
    "rows_total": 100,
    "rows_usable": 90,
    "usable_keypoints_total": 630,
    "usable_rate": 0.9,
    "confidence_valid_rate": 0.95,
    "geometry_valid_rate": 0.92,
    "triangle_area_p10": 1.0,
    "triangle_area_p50": 2.0,
    "triangle_area_p90": 3.0,
    "min_angle_p10": 10.0,
    "min_angle_p50": 20.0,
    "min_angle_p90": 30.0,
    "heading_p10": 0.1,
    "heading_p50": 0.2,
    "heading_p90": 0.3,
    "rig_id": "rig_a",
    "camera_id": "cam_1",
    "arena_id": "arena_x",
    "dish_design": "cedar",
    "canvas_name": "canvas_a",
    "protocol_name": "DefaultScreen",
    "genotype": "Tg(elavl3:gcamp7f)",
    "dpf_at_acquisition": 7,
    "profile_json": _KEYPOINT_PROFILE_JSON,
}


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


def _write_profile_run(zarr_path: Path, parent_name: str, run_name: str, summary: dict) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    analysis = root.require_group("analysis")
    parent = analysis.require_group(parent_name)
    run_group = parent.require_group(run_name)
    run_group.attrs["profile_summary"] = summary
    run_group.attrs["created_at_utc"] = summary["created_at_utc"]
    parent.attrs["latest"] = run_name


def _golden(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in _NON_GOLDEN}


def test_detection_extractor_row_matches_golden(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_a_training.zarr"
    _write_profile_run(
        zarr_path, "detection_profile_runs", "detection_profile_2026-02-12_03-00-00", _detection_summary()
    )
    rows = _extract_detection_data_profile_rows(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        dataset_id="d",
        recording_id="rec_fallback",
        zarr_use="training",
        genotype=None,
        dpf_at_acquisition=None,
    )
    assert len(rows) == 1
    assert _golden(rows[0]) == _DETECTION_GOLDEN


def test_keypoint_extractor_row_matches_golden(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_b_training.zarr"
    _write_profile_run(
        zarr_path, "keypoint_profile_runs", "keypoint_profile_2026-02-12_03-00-00", _keypoint_summary()
    )
    rows = _extract_keypoint_profile_rows(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        dataset_id="d",
        recording_id="rec_fallback",
        zarr_use="training",
        genotype=None,
        dpf_at_acquisition=None,
    )
    assert len(rows) == 1
    assert _golden(rows[0]) == _KEYPOINT_GOLDEN
