from __future__ import annotations

import copy
import math
import json
from pathlib import Path
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
from fisheye.analytics_exports.contracts import ACTIVITY_SPATIAL_TIME_BINS_TABLE
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.runtime_telemetry import (
    validate_export_runtime_telemetry,
)
from fisheye.analytics_exports.validation import (
    ExportValidationError,
    validate_export_run,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_kinematics_samples_export import _eligible_source


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


def test_extraction_policy_batches_whole_bins_with_bounded_source_windows() -> None:
    policy = mod.activity_spatial_extraction_policy(
        source_window_rows=131_072,
        bin_size_frames=150,
    )

    assert policy["requested_source_window_rows"] == 131_072
    assert policy["effective_bins_per_source_window"] == 873
    assert policy["effective_source_frame_span"] == 130_950
    assert policy["read_policy"] == ("consecutive_global_bins_bounded_source_window_v1")
    assert policy["payload_sha256"] == canonical_json_sha256(
        {key: value for key, value in policy.items() if key != "payload_sha256"}
    )


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_extraction_policy_rejects_nonpositive_or_noninteger_windows(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="positive exact integer"):
        mod.activity_spatial_extraction_policy(
            source_window_rows=value,  # type: ignore[arg-type]
            bin_size_frames=150,
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


def test_time_bins_preserve_nan_bout_path_as_explicit_invalid_metric() -> None:
    bouts = _bout_rows([(1, 1, 2)])
    bouts["path_length_mm"][0] = np.nan
    rows = mod.summarize_activity_spatial_track(
        track_id=7,
        source_acquisition_frame_index=np.asarray([0, 1, 2], dtype=np.int64),
        source_observed=np.ones(3, dtype=bool),
        sample_valid=np.ones(3, dtype=bool),
        position_finite=np.ones(3, dtype=bool),
        transition_valid=np.asarray([0, 1, 1], dtype=bool),
        positions_mm=np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.float64),
        filtered_speed_mm_s=np.asarray([np.nan, 1, 1], dtype=np.float32),
        filtered_path_distance_mm=np.asarray([np.nan, 1, 1], dtype=np.float32),
        bouts=bouts,
        source_sample_rate_hz=2.0,
        requested_bin_size_s=2.0,
    )

    assert rows[0]["bout_count_started"] == 1
    assert rows[0]["bout_duration_s_started_sum"] == pytest.approx(1.0)
    assert math.isnan(rows[0]["bout_path_length_mm_started_sum"])
    assert rows[0]["bout_metrics_valid"] is False


@pytest.mark.parametrize("value", (-1.0, float("inf"), float("-inf")))
def test_time_bins_reject_invalid_non_nan_bout_paths(value: float) -> None:
    bouts = _bout_rows([(1, 0, 1)])
    bouts["path_length_mm"][0] = value
    with pytest.raises(ValueError, match="invalid physical values"):
        mod.summarize_activity_spatial_track(
            track_id=7,
            source_acquisition_frame_index=np.asarray([0, 1], dtype=np.int64),
            source_observed=np.ones(2, dtype=bool),
            sample_valid=np.ones(2, dtype=bool),
            position_finite=np.ones(2, dtype=bool),
            transition_valid=np.asarray([0, 1], dtype=bool),
            positions_mm=np.asarray([[0, 0], [1, 1]], dtype=np.float64),
            filtered_speed_mm_s=np.asarray([np.nan, 1], dtype=np.float32),
            filtered_path_distance_mm=np.asarray([np.nan, 1], dtype=np.float32),
            bouts=bouts,
            source_sample_rate_hz=2.0,
            requested_bin_size_s=2.0,
        )


def test_bounded_bin_aggregation_equals_full_track_aggregation() -> None:
    frames = np.asarray([0, 1, 2, 6, 7, 9], dtype=np.int64)
    source_observed = np.asarray([1, 1, 0, 1, 1, 1], dtype=bool)
    sample_valid = np.asarray([1, 1, 0, 1, 1, 1], dtype=bool)
    position_finite = np.asarray([1, 1, 1, 1, 1, 1], dtype=bool)
    transition_valid = np.asarray([0, 1, 0, 1, 1, 1], dtype=bool)
    positions = np.column_stack((frames, frames * 2)).astype(np.float64)
    speeds = np.asarray([np.nan, 1, np.nan, 3, 4, 5], dtype=np.float32)
    paths = np.asarray([np.nan, 0.5, np.nan, 1.5, 2, 2.5], dtype=np.float32)
    bouts = _bout_rows([(1, 2, 5), (2, 7, 9)])
    inputs = {
        "track_id": 7,
        "source_observed": source_observed,
        "sample_valid": sample_valid,
        "position_finite": position_finite,
        "transition_valid": transition_valid,
        "positions_mm": positions,
        "filtered_speed_mm_s": speeds,
        "filtered_path_distance_mm": paths,
        "bouts": bouts,
        "source_sample_rate_hz": 2.0,
        "requested_bin_size_s": 2.0,
    }
    full = mod.summarize_activity_spatial_track(
        source_acquisition_frame_index=frames,
        **inputs,
    )

    bounded: list[dict[str, Any]] = []
    for bin_index in range(3):
        start = bin_index * 4
        stop = start + 4
        selected = (frames >= start) & (frames < stop)
        bounded.extend(
            mod.summarize_activity_spatial_track(
                source_acquisition_frame_index=frames[selected],
                source_observed=source_observed[selected],
                sample_valid=sample_valid[selected],
                position_finite=position_finite[selected],
                transition_valid=transition_valid[selected],
                positions_mm=positions[selected],
                filtered_speed_mm_s=speeds[selected],
                filtered_path_distance_mm=paths[selected],
                track_id=7,
                bouts=bouts,
                source_sample_rate_hz=2.0,
                requested_bin_size_s=2.0,
                track_frame_span=(0, 9),
                time_bin_range=(bin_index, bin_index),
            )
        )

    assert len(bounded) == len(full)
    for observed, expected in zip(bounded, full, strict=True):
        assert observed.keys() == expected.keys()
        for name in observed:
            actual_value = observed[name]
            expected_value = expected[name]
            if isinstance(actual_value, float) and math.isnan(actual_value):
                assert isinstance(expected_value, float)
                assert math.isnan(expected_value), name
            else:
                assert actual_value == expected_value, name


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
    monkeypatch.setattr(
        mod, "resolve_swim_bout_candidate", lambda *_args, **_kwargs: candidate
    )
    monkeypatch.setattr(mod, "load_swim_bout_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(
        mod, "resolve_swim_bout_frame_axis", lambda *_args, **_kwargs: frames
    )
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
    assert binding["bout_path_length_nan_count"] == 0
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


def _publisher_bound_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> mod.BoundActivitySpatialSources:
    root, run, track = _eligible_source(monkeypatch)
    source_path = (tmp_path / "recording_analysis.zarr").resolve()
    track_source = mod.track_export._source_binding(
        root,
        zarr_path=source_path,
        recording_id="recording",
        run_name="motion_physical",
        scope="offline",
    )
    frames = np.asarray(
        track.children["source_acquisition_frame_index"].data,
        dtype=np.int64,
    )
    bouts = _bout_rows([])
    bout_body: dict[str, Any] = {
        "schema_id": mod.ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": mod.ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_VERSION,
        "stage_id": "swim_bouts",
        "track_id": 7,
        "run_name": "bouts_track_7",
        "run_path": "analysis/swim_bout_runs/bouts_track_7",
        "source_schema_id": SWIM_BOUT_RUN_SCHEMA_ID,
        "source_schema_version": SWIM_BOUT_RUN_SCHEMA_VERSION,
        "source_array_manifest_sha256": "c" * 64,
        "source_track_kinematics_run": track_source.binding["run_name"],
        "source_track_motion_manifest_sha256": track_source.binding[
            "source_manifest_sha256"
        ],
        "source_sample_rate_hz": track_source.binding["source_sample_rate_hz"],
        "candidate_id": 0,
        "candidate_name": "candidate",
        "signal_id": 5,
        "signal_name": "filtered",
        "speed_level": "speed_filtered",
        "frame_axis_contract_sha256": "d" * 64,
        "frame_axis_content_sha256": "e" * 64,
        "frame_axis_array_values_sha256": array_values_sha256(frames),
        "frame_axis_first_frame": int(frames[0]),
        "frame_axis_last_frame": int(frames[-1]),
        "bout_count": 0,
        "bout_path_length_nan_count": 0,
        "bout_dtype": bouts.dtype.descr,
        "bout_content_sha256": array_values_sha256(bouts),
        "selection_snapshot": {
            "mode": "explicit_per_track_run",
            "parent_latest": "bouts_track_7",
            "parent_latest_complete": "bouts_track_7",
            "parent_completion_epoch": 1,
        },
        "completion_snapshot": {
            "status": "complete",
            "completed_at_utc": "2026-08-04T12:00:00+00:00",
            "selector_eligible": True,
        },
    }
    bout_binding = {
        **bout_body,
        "payload_sha256": canonical_json_sha256(bout_body),
    }
    source_body: dict[str, Any] = {
        "schema_id": mod.ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": mod.ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_VERSION,
        "recording_id": "recording",
        "zarr_path": str(source_path),
        "track_source_binding": track_source.binding,
        "swim_bout_runs_by_track": {"7": bout_binding},
    }
    source_binding = {
        **source_body,
        "payload_sha256": canonical_json_sha256(source_body),
    }
    return mod.BoundActivitySpatialSources(
        track_source=track_source,
        bout_sources={
            7: mod.BoundSwimBoutSource(
                binding=bout_binding,
                events=SimpleNamespace(bouts=bouts),
                frame_axis=frames,
            )
        },
        binding=source_binding,
    )


def _publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    export_run_id: str,
    output_name: str = "exports",
    source_window_rows: int = mod.DEFAULT_ACTIVITY_SPATIAL_SOURCE_WINDOW_ROWS,
    overwrite: bool = False,
) -> dict[str, Any]:
    bound = _publisher_bound_source(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        mod,
        "bind_activity_spatial_sources",
        lambda *_args, **_kwargs: bound,
    )
    monkeypatch.setattr(mod.track_export, "_recording_id", lambda _path: "recording")
    return mod.export_activity_spatial_time_bins(
        tmp_path / "recording_analysis.zarr",
        track_kinematics_run="motion_physical",
        track_scope="offline",
        swim_bout_runs_by_track={7: "bouts_track_7"},
        requested_bin_size_s=1.0,
        output_root=tmp_path / output_name,
        export_run_id=export_run_id,
        scratch_root=tmp_path / f"scratch_{output_name}",
        source_window_rows=source_window_rows,
        row_group_rows=1,
        overwrite=overwrite,
    )


def test_activity_spatial_publisher_writes_exact_manifest_selected_part(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _publish(monkeypatch, tmp_path, export_run_id="activity")

    assert result["activity_spatial_time_bins_validation"]["valid"] is True
    validate_export_runtime_telemetry(result["runtime_telemetry"])
    assert "runtime_telemetry" not in json.loads(
        Path(result["manifest_path"]).read_text(encoding="utf-8")
    )
    assert result["row_counts_by_table"] == {ACTIVITY_SPATIAL_TIME_BINS_TABLE: 2}
    assert validate_export_run(tmp_path / "exports", "activity")["status"] == "valid"
    assert not list((tmp_path / "scratch_exports").glob("palette_activity_spatial_*"))

    import pyarrow.parquet as pq

    part = next((tmp_path / "exports").rglob("*.parquet"))
    table = pq.read_table(part).to_pydict()
    assert table["track_id"] == [7, 7]
    assert table["time_bin_index"] == [0, 1]
    assert table["source_swim_bout_run"] == ["bouts_track_7"] * 2
    assert table["position_coordinate_space"] == ["physical_mm"] * 2
    extraction = result["activity_spatial_time_bins_export"]["extraction_policy"]
    assert extraction["requested_source_window_rows"] == (
        mod.DEFAULT_ACTIVITY_SPATIAL_SOURCE_WINDOW_ROWS
    )


def test_multi_bin_source_windows_preserve_exact_rows_and_reduce_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = mod.track_export._read_projected_window
    active = "single"
    calls = {"single": 0, "batched": 0}

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls[active] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod.track_export, "_read_projected_window", counted)
    single = _publish(
        monkeypatch,
        tmp_path,
        export_run_id="window_equivalence",
        output_name="single_bin_exports",
        source_window_rows=1,
    )
    active = "batched"
    batched = _publish(
        monkeypatch,
        tmp_path,
        export_run_id="window_equivalence",
        output_name="multi_bin_exports",
        source_window_rows=4,
    )

    assert calls == {"single": 2, "batched": 1}
    assert (
        single["activity_spatial_time_bins_export"]["decoded_payload"]
        == batched["activity_spatial_time_bins_export"]["decoded_payload"]
    )
    single_part = next((tmp_path / "single_bin_exports").rglob("*.parquet"))
    batched_part = next((tmp_path / "multi_bin_exports").rglob("*.parquet"))
    single_payload, single_columns = mod._decoded_part_payload(single_part)
    batched_payload, batched_columns = mod._decoded_part_payload(batched_part)
    assert single_payload == batched_payload
    assert tuple(single_columns) == tuple(batched_columns)
    for name, values in single_columns.items():
        assert len(values) == len(batched_columns[name])
        assert all(
            mod._same_float(left, right)
            for left, right in zip(values, batched_columns[name], strict=True)
        )


def test_activity_spatial_validator_rejects_rehashed_parquet_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _publish(monkeypatch, tmp_path, export_run_id="tamper")
    manifest_path = Path(result["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_path = next((tmp_path / "exports").rglob("*.parquet"))

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(part_path)
    original = parquet_file.read()
    schema = parquet_file.schema_arrow
    column_index = schema.get_field_index("source_speed_level")
    arrays = [original.column(index) for index in range(original.num_columns)]
    arrays[column_index] = pa.chunked_array(
        [pa.array(["raw"] * original.num_rows, type=pa.string())]
    )
    changed = pa.Table.from_arrays(arrays, schema=schema)
    writer = pq.ParquetWriter(
        part_path,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=payload["activity_spatial_time_bins_export"]["parquet_policy"][
            "dictionary_columns"
        ],
    )
    try:
        writer.write_table(changed, row_group_size=1)
    finally:
        writer.close()
    entry = payload["publication"]["parts_by_table"][ACTIVITY_SPATIAL_TIME_BINS_TABLE][
        0
    ]
    entry["sha256"] = sha256_file(part_path)
    entry["size_bytes"] = part_path.stat().st_size
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="decoded payload differs"):
        validate_export_run(tmp_path / "exports", "tamper")


def test_failed_activity_spatial_overwrite_preserves_visible_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _publish(monkeypatch, tmp_path, export_run_id="stable")
    manifest_path = Path(first["manifest_path"])
    baseline = manifest_path.read_bytes()
    original = mod._write_streaming_part

    def fail_after_write(*args: Any, **kwargs: Any) -> dict[str, Any]:
        original(*args, **kwargs)
        raise RuntimeError("injected replacement failure")

    monkeypatch.setattr(mod, "_write_streaming_part", fail_after_write)
    with pytest.raises(RuntimeError, match="injected replacement failure"):
        _publish(
            monkeypatch,
            tmp_path,
            export_run_id="stable",
            overwrite=True,
        )
    assert manifest_path.read_bytes() == baseline
    assert validate_export_run(tmp_path / "exports", "stable")["status"] == "valid"


def test_single_swim_bout_dependency_fails_closed_for_multitrack_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod.track_export, "_recording_id", lambda _path: "recording")
    monkeypatch.setattr(
        mod.track_export,
        "_source_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            binding={"tracks": [{"track_id": 7}, {"track_id": 8}]}
        ),
    )

    with pytest.raises(ValueError, match="exactly one-track source"):
        mod.export_activity_spatial_time_bins(
            tmp_path / "recording_analysis.zarr",
            track_kinematics_run="motion",
            track_scope="offline",
            single_track_swim_bout_run="bouts",
            requested_bin_size_s=5.0,
            output_root=tmp_path / "exports",
            export_run_id="multitrack",
            scratch_root=tmp_path / "scratch",
        )


def test_activity_spatial_source_change_fails_before_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before = _publisher_bound_source(monkeypatch, tmp_path)
    after = copy.deepcopy(before)
    after_bout = after.binding["swim_bout_runs_by_track"]["7"]
    after_bout["completion_snapshot"]["completed_at_utc"] = "2026-08-04T12:05:00+00:00"
    after_bout["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in after_bout.items() if key != "payload_sha256"}
    )
    after.binding["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in after.binding.items() if key != "payload_sha256"}
    )
    calls = 0

    def changing_binding(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return before if calls == 1 else after

    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "bind_activity_spatial_sources", changing_binding)
    monkeypatch.setattr(mod.track_export, "_recording_id", lambda _path: "recording")
    output = tmp_path / "exports"
    with pytest.raises(RuntimeError, match="changed during extraction"):
        mod.export_activity_spatial_time_bins(
            tmp_path / "recording_analysis.zarr",
            track_kinematics_run="motion_physical",
            track_scope="offline",
            swim_bout_runs_by_track={7: "bouts_track_7"},
            requested_bin_size_s=1.0,
            output_root=output,
            export_run_id="changed",
            scratch_root=tmp_path / "scratch",
            row_group_rows=1,
        )
    assert not (output / "v2" / "manifests" / "changed.json").exists()
