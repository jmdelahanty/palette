"""Shared detect/pose training run naming contract tests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.training_naming_shared import (  # noqa: E402
    build_default_detect_run_name,
    build_default_pose_run_name,
)


def test_build_default_detect_run_name_contract() -> None:
    run_name = build_default_detect_run_name(
        manifest_summary={
            "manifest_rig_name": "omnifin0",
            "manifest_dish_design": "cedar dish",
            "manifest_canvas_name": "DefaultScreen",
            "manifest_set_id": "detect_cedar_shadow_v007",
            "manifest_task": "detect",
            "manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        },
        task_fallback="detect",
        timestamp="20260206-235656",
        pid=123,
    )
    assert run_name == "omnifin0_cedar_dish_defaultscreen_v007_detect_20260206-235656_01234567"


def test_build_default_pose_run_name_contract_matches_detect_style() -> None:
    run_name = build_default_pose_run_name(
        manifest_hints={
            "rig_name": "omnifin0",
            "dish_design": "cedar dish",
            "canvas_name": "Feeding",
            "set_id": "pose_cedar_shadow_v007",
            "task": "pose",
            "manifest_sha256": "89abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
        },
        task_fallback="pose",
        timestamp="20260208-030800",
        pid=456,
    )
    assert run_name == "omnifin0_cedar_dish_feeding_v007_pose_20260208-030800_89abcdef"


def test_build_default_pose_run_name_uses_unknown_rig_when_missing() -> None:
    run_name = build_default_pose_run_name(
        manifest_hints={
            "dish_design": "cedar",
            "canvas_name": "feeding",
            "set_id": "pose_cedar_shadow_v001",
            "task": "pose",
            "manifest_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        },
        task_fallback="pose",
        timestamp="20260208-031500",
        pid=1,
    )
    assert run_name.startswith("unknown_rig_cedar_feeding_v001_pose_20260208-031500_")
