from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fisheye.analysis._exact_tabular_run_schema import MANIFEST_ATTRIBUTE
from fisheye.analysis.swim_bout_frame_axis import (
    FRAME_AXIS_CONTRACT_ATTR,
    FRAME_AXIS_CONTRACT_SHA256_ATTR,
)
from fisheye.analysis.swim_bout_io import (
    SwimBoutCandidate,
    SwimBoutEvents,
    SwimBoutSignalVariant,
)
from fisheye.analysis.swim_bout_schema import (
    SWIM_BOUT_LAYOUT,
    SWIM_BOUT_RUN_SCHEMA_ID,
    SWIM_BOUT_RUN_SCHEMA_VERSION,
)
from fisheye.analytics_exports import activity_spatial_time_bins as mod
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


class _Group:
    def __init__(
        self,
        *,
        attrs: dict[str, Any] | None = None,
        children: dict[str, Any] | None = None,
    ) -> None:
        self.attrs = dict(attrs or {})
        self.children = dict(children or {})

    def __getitem__(self, key: str) -> Any:
        return self.children[key]


def _bout_rows(
    rows: list[tuple[int, int, int]],
    *,
    track_id: int = 7,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("candidate_id", "<i4"),
            ("signal_id", "<i4"),
            ("track_id", "<i4"),
            ("bout_id", "<i4"),
            ("start_frame", "<i8"),
            ("end_frame", "<i8"),
            ("duration_s", "<f8"),
            ("path_length_mm", "<f8"),
        ]
    )
    result = np.zeros(len(rows), dtype=dtype)
    for index, (bout_id, start, end) in enumerate(rows):
        result[index] = (
            0,
            5,
            track_id,
            bout_id,
            start,
            end,
            float(end - start + 1) / 2.0,
            float(end - start + 1),
        )
    return result


def test_binning_contract_rounds_once_and_uses_global_frames() -> None:
    policy = mod.activity_spatial_binning_contract(
        source_sample_rate_hz=700.0,
        requested_bin_size_s=0.001,
    )
    assert policy["bin_size_frames"] == 1
    assert policy["effective_bin_size_s"] == pytest.approx(1.0 / 700.0)
    assert policy["binning_policy"] == (
        "global_acquisition_frame_fixed_width_round_half_up_v1"
    )
    assert policy["payload_sha256"] == canonical_json_sha256(
        {key: value for key, value in policy.items() if key != "payload_sha256"}
    )


@pytest.mark.parametrize("value", (0, -1, float("inf"), float("nan"), True))
def test_binning_contract_rejects_invalid_widths(value: object) -> None:
    with pytest.raises(ValueError, match="requested bin size"):
        mod.activity_spatial_binning_contract(
            source_sample_rate_hz=700.0,
            requested_bin_size_s=value,
        )


def test_time_bins_preserve_gaps_and_split_bout_occupancy() -> None:
    frames = np.asarray([0, 1, 2, 6, 7], dtype=np.int64)
    rows = mod.summarize_activity_spatial_track(
        track_id=7,
        source_acquisition_frame_index=frames,
        source_observed=np.asarray([1, 1, 1, 1, 1], dtype=bool),
        sample_valid=np.asarray([1, 1, 1, 1, 1], dtype=bool),
        position_finite=np.asarray([1, 1, 1, 1, 1], dtype=bool),
        transition_valid=np.asarray([0, 1, 1, 1, 1], dtype=bool),
        positions_mm=np.column_stack((frames, frames * 2)).astype(np.float64),
        filtered_speed_mm_s=np.asarray([np.nan, 1, 2, 3, 4], dtype=np.float32),
        filtered_path_distance_mm=np.asarray(
            [np.nan, 0.5, 1.0, 1.5, 2.0],
            dtype=np.float32,
        ),
        bouts=_bout_rows([(1, 2, 5), (2, 7, 7)]),
        source_sample_rate_hz=2.0,
        requested_bin_size_s=2.0,
    )

    assert [row["time_bin_index"] for row in rows] == [0, 1]
    assert [row["source_sample_count"] for row in rows] == [3, 2]
    assert [row["expected_track_frame_count"] for row in rows] == [4, 4]
    assert [row["bout_count_started"] for row in rows] == [1, 1]
    assert [row["bout_occupied_frame_count"] for row in rows] == [2, 3]
    assert rows[0]["bout_occupancy_fraction"] == 0.5
    assert rows[1]["bout_occupancy_fraction"] == 0.75
    assert rows[0]["path_distance_mm_sum"] == pytest.approx(1.5)
    assert rows[1]["path_distance_mm_sum"] == pytest.approx(3.5)


def test_time_bins_emit_empty_internal_gap_and_union_overlapping_bouts() -> None:
    frames = np.asarray([0, 9], dtype=np.int64)
    rows = mod.summarize_activity_spatial_track(
        track_id=7,
        source_acquisition_frame_index=frames,
        source_observed=np.ones(2, dtype=bool),
        sample_valid=np.ones(2, dtype=bool),
        position_finite=np.ones(2, dtype=bool),
        transition_valid=np.asarray([0, 1], dtype=bool),
        positions_mm=np.asarray([[0.0, 0.0], [9.0, 9.0]], dtype=np.float64),
        filtered_speed_mm_s=np.asarray([np.nan, 1.0], dtype=np.float32),
        filtered_path_distance_mm=np.asarray([np.nan, 1.0], dtype=np.float32),
        bouts=_bout_rows([(1, 1, 3), (2, 2, 4)]),
        source_sample_rate_hz=2.0,
        requested_bin_size_s=2.0,
    )

    assert [row["time_bin_index"] for row in rows] == [0, 1, 2]
    assert rows[1]["source_sample_count"] == 0
    assert rows[1]["bin_valid"] is False
    assert rows[1]["bin_reason_code"] == 1
    assert rows[0]["bout_occupied_frame_count"] == 3
    assert rows[1]["bout_occupied_frame_count"] == 1


def _patch_bound_sources(monkeypatch: pytest.MonkeyPatch):
    frames = np.asarray([0, 1, 2], dtype=np.int64)
    track_record = {
        "track_id": 7,
        "sample_count": 3,
        "selected_surfaces": {
            "source_acquisition_frame_index": {
                "content_sha256": array_values_sha256(frames)
            }
        },
    }
    track_binding = {
        "run_name": "track_run",
        "source_manifest_sha256": "a" * 64,
        "source_sample_rate_hz": 2.0,
        "tracks": [track_record],
        "payload_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        mod.track_export,
        "_source_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            binding=track_binding,
            run_group=object(),
        ),
    )

    frame_contract = {"schema_id": "frame-axis", "schema_version": 2}
    run_attrs = {
        "schema_id": SWIM_BOUT_RUN_SCHEMA_ID,
        "schema_version": SWIM_BOUT_RUN_SCHEMA_VERSION,
        "layout": SWIM_BOUT_LAYOUT,
        MANIFEST_ATTRIBUTE: {"schema_id": "array-manifest", "schema_version": 1},
        "source_track_kinematics_run": "track_run",
        "source_track_motion_manifest_sha256": "a" * 64,
        "track_id": 7,
        "fps": 2.0,
        "palette_run_completion_status": "complete",
        "palette_run_completed_at_utc": "2026-08-04T12:00:00+00:00",
        "stage_selector_eligible": True,
        FRAME_AXIS_CONTRACT_ATTR: frame_contract,
        FRAME_AXIS_CONTRACT_SHA256_ATTR: canonical_json_sha256(frame_contract),
    }
    run = _Group(attrs=run_attrs)
    parent = _Group(
        attrs={
            "latest": "bouts_track_7",
            "latest_complete": "bouts_track_7",
            "palette_completion_epoch": 2,
        },
        children={"bouts_track_7": run},
    )
    root = _Group(children={"analysis": _Group(children={"swim_bout_runs": parent})})
    signal = SwimBoutSignalVariant(
        run_name="bouts_track_7",
        signal_id=5,
        speed_level="speed_filtered",
        signal_name="filtered",
        role="physical_estimator",
        source_level="speed_filtered",
        is_default=True,
        n_bouts=1,
        attrs={},
    )
    candidate = SwimBoutCandidate(
        run_name="bouts_track_7",
        candidate_id=0,
        candidate_name="candidate",
        run_path="analysis/swim_bout_runs/bouts_track_7",
        is_latest=True,
        source_track_kinematics_run="track_run",
        track_id=7,
        detection_method="threshold",
        default_signal_id=5,
        default_speed_level="speed_filtered",
        signals=(signal,),
        attrs={},
    )
    bouts = _bout_rows([(1, 1, 2)])
    events = SwimBoutEvents(
        run_name="bouts_track_7",
        run_path="analysis/swim_bout_runs/bouts_track_7",
        level_path="analysis/swim_bout_runs/bouts_track_7/tables/bouts",
        candidate=candidate,
        signal=signal,
        bouts=bouts,
        run_attrs=run_attrs,
        signal_attrs={},
    )
    monkeypatch.setattr(mod, "is_run_selector_eligible", lambda _run: True)
    monkeypatch.setattr(
        mod,
        "is_run_complete_in_parent",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(mod, "validate_swim_bout_array_manifest", lambda _run: ())
    monkeypatch.setattr(mod, "resolve_swim_bout_candidate", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(mod, "load_swim_bout_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(mod, "resolve_swim_bout_frame_axis", lambda *_args, **_kwargs: frames)
    return root, track_binding, candidate


def test_source_binding_requires_exact_per_track_run_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root, _track_binding, _candidate = _patch_bound_sources(monkeypatch)
    bound = mod.bind_activity_spatial_sources(
        root,
        zarr_path=tmp_path / "recording_analysis.zarr",
        recording_id="recording",
        track_kinematics_run="track_run",
        track_scope="offline",
        swim_bout_runs_by_track={7: "bouts_track_7"},
    )

    assert tuple(bound.bout_sources) == (7,)
    binding = bound.bout_sources[7].binding
    assert binding["candidate_id"] == 0
    assert binding["signal_id"] == 5
    assert binding["bout_count"] == 1
    assert bound.binding["payload_sha256"] == canonical_json_sha256(
        {key: value for key, value in bound.binding.items() if key != "payload_sha256"}
    )

    with pytest.raises(ValueError, match="every and only"):
        mod.bind_activity_spatial_sources(
            root,
            zarr_path=tmp_path / "recording_analysis.zarr",
            recording_id="recording",
            track_kinematics_run="track_run",
            track_scope="offline",
            swim_bout_runs_by_track={},
        )


def test_source_binding_rejects_run_for_another_track(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root, _track_binding, candidate = _patch_bound_sources(monkeypatch)
    root["analysis"]["swim_bout_runs"]["bouts_track_7"].attrs["track_id"] = 8

    with pytest.raises(ValueError, match="does not belong to track"):
        mod.bind_activity_spatial_sources(
            root,
            zarr_path=tmp_path / "recording_analysis.zarr",
            recording_id="recording",
            track_kinematics_run="track_run",
            track_scope="offline",
            swim_bout_runs_by_track={candidate.track_id: "bouts_track_7"},
        )
