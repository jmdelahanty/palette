from __future__ import annotations

from fisheye.shared.crop_signature import build_crop_signature


def test_build_crop_signature_preserves_geometry_lineage_fields() -> None:
    signature = build_crop_signature(
        {
            "detection_source_path": "refined_detect_runs/refined_detect_001/instances",
            "source_coords_path": "refined_detect_runs/refined_detect_001/instances",
            "detection_source_type": "refined",
            "detection_selection_policy": "resolved_group=refined",
            "crop_storage_mode": "materialized",
            "source_detect_run": "detect_001",
            "source_refined_run": "refined_detect_001",
            "source_background_run": "background_001",
            "roi_size": [64, 64],
            "crop_revision": 3,
            "parameter_source": "config",
            "parameters": {"roi_size": [64, 64]},
        }
    )

    assert signature["signature_version"] == 2
    assert signature["detection_source_path"] == "refined_detect_runs/refined_detect_001/instances"
    assert signature["source_coords_path"] == "refined_detect_runs/refined_detect_001/instances"
    assert signature["source_background_run"] == "background_001"
    assert signature["crop_revision"] == 3
