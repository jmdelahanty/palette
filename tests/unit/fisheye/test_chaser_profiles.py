from __future__ import annotations

import importlib

from fisheye.analysis.chaser_profiles import (
    ANALYSIS_PROFILE_SCHEMA_ID,
    PROTOCOL_PROFILE_SCHEMA_ID,
    default_chaser_analysis_profile_path,
    default_chaser_protocol_profile_path,
    default_goodcopbadcop_source_profile_path,
    load_chaser_analysis_profile,
    load_chaser_protocol_profile,
    resolve_protocol_payload_path,
    resolve_profile_windows,
)


def test_default_profiles_are_versioned_and_separate_protocol_from_analysis() -> None:
    protocol = load_chaser_protocol_profile(default_chaser_protocol_profile_path())
    analysis = load_chaser_analysis_profile(default_chaser_analysis_profile_path())

    assert protocol.to_dict()["schema_id"] == PROTOCOL_PROFILE_SCHEMA_ID
    assert protocol.profile_id == "chaser_event_windows_v1"
    assert protocol.window_policy_id == "event_alias_windows"
    assert analysis.to_dict()["schema_id"] == ANALYSIS_PROFILE_SCHEMA_ID
    assert analysis.profile_id == "chaser_behavior_v1"
    assert "goodcopbadcop" not in str(analysis.to_dict()).lower()
    assert len(protocol.sha256) == 64
    assert len(analysis.sha256) == 64


def test_generic_profile_contains_renamed_modules_and_per_chaser_escape() -> None:
    profile = load_chaser_analysis_profile(default_chaser_analysis_profile_path())
    modules = {module.module_id: module for module in profile.modules}

    assert "chaser_quadrant_occupancy" in modules
    assert "chaser_near_field_occupancy" in modules
    assert "chaser_epoch_behavior_summary" in modules
    assert modules["chaser_escape_freeze_summary"].execution_cardinality == "per_chaser"


def test_profile_schema_contracts_match_their_implementations() -> None:
    profile = load_chaser_analysis_profile(default_chaser_analysis_profile_path())

    for module in profile.modules:
        implementation = importlib.import_module(module.implementation)
        assert implementation.SCHEMA_ID == module.schema_id
        assert implementation.SCHEMA_VERSION == module.schema_version


def test_profile_window_resolution_matches_configured_event_boundaries() -> None:
    profile = load_chaser_protocol_profile(default_chaser_protocol_profile_path())
    windows = resolve_profile_windows(
        profile,
        {
            "CHASER_TRAINING_START": 10,
            "CHASER_POST_PERIOD_START": 30,
            "PROTOCOL_FINISH": 50,
        },
        total_frames=60,
    )

    assert [(row.label, row.start_frame, row.end_frame) for row in windows] == [
        ("pre_event", 0, 9),
        ("training_event", 10, 29),
        ("post_event", 30, 49),
    ]
    assert windows[0].source_start_event_name == "RECORDING_START_FALLBACK"
    assert "missing_pre_start_used_frame_0" in windows[0].source_policy


def test_legacy_goodcopbadcop_protocol_profile_remains_loadable() -> None:
    profile = load_chaser_protocol_profile(default_goodcopbadcop_source_profile_path())

    assert profile.profile_id == "goodcopbadcop_source_v1"
    assert profile.window_policy_id == "event_alias_windows"


def test_protocol_profile_payload_path_expands_sequence_segments() -> None:
    payload = {
        "steps": [
            {"parameters": {"position_transition_duration_s": None}},
            {"parameters": {"position_transition_duration_s": 1.25}},
        ]
    }

    assert (
        resolve_protocol_payload_path(
            payload,
            "protocol_json.steps[].parameters.position_transition_duration_s",
        )
        == 1.25
    )
