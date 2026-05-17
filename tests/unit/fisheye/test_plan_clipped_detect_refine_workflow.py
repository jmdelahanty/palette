from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils.plan_clipped_detect_refine_workflow import build_plan


def _write_clip_index(recording_dir: Path) -> None:
    payload = {
        "recording_id": "sleepyfish_2026_05_05_17_45_30_cam2010093",
        "clips": [
            {
                "clip_id": "clip_000000",
                "clip_index": 0,
                "camera_artifacts": [
                    {
                        "camera_serial": "2010093",
                        "video_path": "clips/clip_000000/Cam2010093.mp4",
                        "metadata_path": "clips/clip_000000/Cam2010093_meta.csv",
                        "keyframe_path": "clips/clip_000000/Cam2010093_keyframe.json",
                        "frame_count": 54000,
                    }
                ],
            },
            {
                "clip_id": "clip_000001",
                "clip_index": 1,
                "camera_artifacts": [
                    {
                        "camera_serial": "2010093",
                        "video_path": "clips/clip_000001/Cam2010093.mp4",
                        "metadata_path": "clips/clip_000001/Cam2010093_meta.csv",
                        "keyframe_path": "clips/clip_000001/Cam2010093_keyframe.json",
                        "frame_count": 54000,
                    }
                ],
            },
        ],
    }
    (recording_dir / "recording_clip_index.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_build_plan_emits_deterministic_clip_local_commands(tmp_path: Path) -> None:
    _write_clip_index(tmp_path)
    model = tmp_path / "best.pt"

    plan = build_plan(
        tmp_path,
        model=model,
        workflow_id="smoke_detect_refine_20260517",
        clip_ids=["clip_000000"],
    )

    assert plan["status"] == "ok"
    assert plan["dry_run_only"] is True
    assert plan["work_unit_count"] == 1
    unit = plan["work_units"][0]
    assert unit["clip_id"] == "clip_000000"
    assert unit["camera_serial"] == "2010093"
    assert unit["run_names"] == {
        "detect": "detect_smoke_detect_refine_20260517_clip_000000_cam2010093",
        "detect_quality": "detect_quality_smoke_detect_refine_20260517_clip_000000_cam2010093",
        "refined_detect": "refined_detect_smoke_detect_refine_20260517_clip_000000_cam2010093",
    }
    assert (
        unit["zarr_paths"]["detect_target_group_path"]
        == "clips/clip_000000/cameras/2010093/detect_runs/"
        "detect_smoke_detect_refine_20260517_clip_000000_cam2010093"
    )
    assert "--detect-run-name detect_smoke_detect_refine_20260517_clip_000000_cam2010093" in unit["commands"]["detect_submit"]
    assert "--use-intended-target --apply" in unit["commands"]["import_detect"]
    assert "--target-group-path clips/clip_000000/cameras/2010093/detect_runs/detect_smoke_detect_refine_20260517_clip_000000_cam2010093" in unit["commands"]["validate_detect"]
    assert "--quality-run-name detect_quality_smoke_detect_refine_20260517_clip_000000_cam2010093" in unit["commands"]["detect_quality"]
    assert "--refined-family-path clips/clip_000000/cameras/2010093/refined_detect_runs" in unit["commands"]["refine_detect"]
    assert "--per-frame-top-k 1" in unit["commands"]["refine_detect"]
    assert "fisheye.utils.validate_refined_detect_run" in unit["commands"]["validate_refined_detect"]
    assert "--target-group-path clips/clip_000000/cameras/2010093/refined_detect_runs/refined_detect_smoke_detect_refine_20260517_clip_000000_cam2010093" in unit["commands"]["validate_refined_detect"]
    assert unit["stage_plan"][0]["resource"] == "gpu"
    assert unit["stage_plan"][1]["resource"] == "cpu"
    assert unit["stage_plan"][-1] == {
        "stage": "validate_refined_detect",
        "resource": "cpu",
        "depends_on": "refine_detect",
    }
    assert plan["finalizer"]["status"] == "planned_placeholder"


def test_build_plan_can_disable_top_k_and_limit_rows(tmp_path: Path) -> None:
    _write_clip_index(tmp_path)

    plan = build_plan(
        tmp_path,
        model=tmp_path / "best.pt",
        workflow_id="wf",
        per_frame_top_k=None,
        limit=1,
    )

    assert plan["work_unit_count"] == 1
    assert "--per-frame-top-k" not in plan["work_units"][0]["commands"]["refine_detect"]
