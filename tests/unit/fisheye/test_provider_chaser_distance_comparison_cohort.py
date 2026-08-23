from __future__ import annotations

import pytest

from fisheye.analysis.provider_chaser_distance_comparison_cohort import (
    ProviderChaserDistanceCohortError,
    _aggregate_reports,
    select_arena_extremes,
)


def _entry(recording_id: str) -> dict[str, str]:
    return {
        "recording_id": recording_id,
        "arena": recording_id.split("_goodbatbadbat", 1)[0].rsplit("_", 2)[-2]
        + "_"
        + recording_id.split("_goodbatbadbat", 1)[0].rsplit("_", 1)[-1],
    }


def test_arena_extremes_are_deterministic_and_outcome_blind() -> None:
    entries = [
        _entry("2026-08-11T10-00-00Z_arena_2_goodbatbadbat"),
        _entry("2026-08-10T10-00-00Z_arena_1_goodbatbadbat"),
        _entry("2026-08-12T10-00-00Z_arena_1_goodbatbadbat"),
        _entry("2026-08-11T10-00-00Z_arena_1_goodbatbadbat"),
        _entry("2026-08-12T10-00-00Z_arena_2_goodbatbadbat"),
        _entry("2026-08-10T10-00-00Z_arena_2_goodbatbadbat"),
    ]
    selected, record = select_arena_extremes(entries)
    assert [item["recording_id"] for item in selected] == [
        "2026-08-10T10-00-00Z_arena_1_goodbatbadbat",
        "2026-08-10T10-00-00Z_arena_2_goodbatbadbat",
        "2026-08-12T10-00-00Z_arena_1_goodbatbadbat",
        "2026-08-12T10-00-00Z_arena_2_goodbatbadbat",
    ]
    assert record["selected_recording_count"] == 4
    assert len(record["selection_sha256"]) == 64


def test_arena_extremes_require_two_recordings_per_stratum() -> None:
    with pytest.raises(ProviderChaserDistanceCohortError, match="fewer than two"):
        select_arena_extremes([_entry("2026-08-10T10-00-00Z_arena_1_goodbatbadbat")])


def _provider(label: str, *, fraction: float) -> dict[str, object]:
    return {
        "frame_count": 10,
        "valid_source_position_fraction": fraction,
        "manifest_sha256": label * 64,
        "source_position_provider": {
            "coordinate_authority_id": (
                "/analysis/coordinate_frames/source_camera/2010093/"
                "continuous@pixel_frame_authority"
            )
        },
    }


def test_aggregation_uses_recording_level_rows() -> None:
    metrics = []
    for provider_label, valid_fraction, distance in (
        ("detection", 1.0, 20.0),
        ("keypoint", 0.9, 20.1),
    ):
        metrics.append(
            {
                "provider_label": provider_label,
                "epoch_window_id": 0,
                "epoch_label": "pre_event",
                "behavior_role": "aggressive",
                "valid_distance_fraction": valid_fraction,
                "distance_p50_mm": distance,
            }
        )
    report = {
        "recording_id": "2026-08-10T10-00-00Z_arena_1_goodbatbadbat",
        "providers": {
            "detection": _provider("a", fraction=1.0),
            "keypoint": _provider("b", fraction=0.9),
        },
        "overall_provider_comparison": {
            "common_position_frame_count": 9,
            "position_delta_p50_px": 2.0,
            "position_delta_p95_px": 4.0,
            "position_delta_p99_px": 5.0,
            "nearest_chaser_agreement_fraction": 1.0,
        },
        "provider_comparison": [
            {
                "comparison_kind": "epoch_chaser",
                "epoch_window_id": 0,
                "epoch_label": "pre_event",
                "behavior_role": "aggressive",
                "common_distance_frame_count": 9,
                "signed_distance_delta_mean_mm": 0.1,
                "absolute_distance_delta_p50_mm": 0.1,
                "absolute_distance_delta_p95_mm": 0.2,
            }
        ],
        "per_epoch_metrics": metrics,
    }
    recording_rows, epoch_rows, summary = _aggregate_reports(
        [report],
        first_label="detection",
        second_label="keypoint",
    )
    assert len(recording_rows) == 1
    assert recording_rows[0][
        "second_minus_first_coverage_percentage_points"
    ] == pytest.approx(-10.0)
    assert recording_rows[0]["common_position_fraction"] == 0.9
    assert len(epoch_rows) == 1
    assert summary["position_delta_p50_px"]["count"] == 1
