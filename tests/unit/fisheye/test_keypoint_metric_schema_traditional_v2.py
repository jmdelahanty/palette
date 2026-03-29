from __future__ import annotations

import json
from pathlib import Path


def test_traditional_v2_keypoint_metric_schema_matches_contract() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    schema_path = repo_root / "configs" / "fisheye" / "keypoint_metric_schemas" / "traditional_v2.json"

    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    assert payload["schema_name"] == "traditional_v2_derived_metrics"
    assert payload["schema_version"] == "v1"
    assert payload["skeleton_id"] == "pose_skel_traditional_v2"
    assert payload["source_pose_schema"] == "traditional_v2"

    metrics = payload["metrics"]
    assert [metric["name"] for metric in metrics] == [
        "total_length",
        "tail_length",
        "head_length",
        "eye_span",
    ]
    assert all(metric["type"] == "distance" for metric in metrics)
    assert all(metric["units"] == "px" for metric in metrics)
    assert all(metric["normalization"] == "roi_diagonal" for metric in metrics)

    by_name = {metric["name"]: metric for metric in metrics}
    assert by_name["total_length"]["from_label"] == "snout_tip"
    assert by_name["total_length"]["to_label"] == "tail_tip"
    assert by_name["tail_length"]["from_label"] == "swim_bladder"
    assert by_name["tail_length"]["to_label"] == "tail_tip"
    assert by_name["head_length"]["from_label"] == "snout_tip"
    assert by_name["head_length"]["to_label"] == "swim_bladder"
    assert by_name["eye_span"]["from_label"] == "eye_left"
    assert by_name["eye_span"]["to_label"] == "eye_right"
