from __future__ import annotations

from fisheye.utils import resolve_detect_model as mod


def _target() -> mod.TargetProfile:
    return mod.TargetProfile(
        recording_id="recA",
        recording_type="behavior",
        recording_subtype="free",
        behavior_mode="free",
        rig_id="omnifin0",
        arena_id="arena_1",
        camera_id="2010093",
        canvas_name="DefaultScreen",
        protocol_name="cedar",
        dish_design="cedar",
        cross_id="17313",
        genotype="Casper_HHMI",
        dpf_at_acquisition=12,
    )


def test_score_rows_prefers_better_metadata_match() -> None:
    target = _target()
    good_rows = [
        {
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "rig_id": "omnifin0",
            "arena_id": "arena_1",
            "camera_id": "2010093",
            "canvas_name": "DefaultScreen",
            "protocol_name": "cedar",
            "dish_design": "cedar",
            "cross_id": "17313",
            "genotype": "Casper_HHMI",
            "dpf_at_acquisition": 12,
        }
    ]
    weak_rows = [
        {
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "rig_id": "omnifin0",
            "arena_id": "arena_2",
            "camera_id": "2010999",
            "canvas_name": "DefaultScreen",
            "protocol_name": "other",
            "dish_design": "other",
            "cross_id": "99999",
            "genotype": "Other",
            "dpf_at_acquisition": 8,
        }
    ]

    good_score, _, _ = mod._score_rows(good_rows, target)  # noqa: SLF001
    weak_score, _, _ = mod._score_rows(weak_rows, target)  # noqa: SLF001

    assert good_score > weak_score
    assert good_score == 1.0


def test_parse_dataset_ids_ignores_invalid_payloads() -> None:
    assert mod._parse_dataset_ids(None) == []  # noqa: SLF001
    assert mod._parse_dataset_ids("{}") == []  # noqa: SLF001
    assert mod._parse_dataset_ids('["a", "b", null, " "]') == ["a", "b"]  # noqa: SLF001


def test_matches_task_filters_pose_from_detect_default() -> None:
    assert mod._matches_task(  # noqa: SLF001
        run_id="omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb",
        set_id="detect_cedar_shadow_v007",
        model_path="/nvme1/models/detect/detect_cedar_shadow_v007/.../best.pt",
        task="detect",
    )
    assert not mod._matches_task(  # noqa: SLF001
        run_id="omnifin0_cedar_shadow_v004_pose_20260208-163716_c9dc72f5",
        set_id="pose_cedar_shadow_manual_gray_latest_traditional_894ad574_v004",
        model_path="/nvme1/models/pose/pose_cedar_shadow_manual_gray_latest_traditional_894ad574_v004/.../best.pt",
        task="detect",
    )
