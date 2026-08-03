from __future__ import annotations

import io

import numpy as np
import pytest
import zarr
from rich.console import Console

from fisheye.analysis.track_kinematics import _mirror_swim_bouts_to_tracks
from fisheye.shared.zarr.columnar import store_array, write_columnar_dataset
from fisheye.analysis.swim_bout_io import (
    SwimBoutIOError,
    _resolve_run_name,
    discover_swim_bout_candidates,
    load_default_swim_bout_tables,
    load_swim_bout_events,
    load_swim_bout_tables,
    resolve_swim_bout_run_name,
    structured_records_to_dicts,
)
from fisheye.utils.export_cross_recording_analytics import _load_swim_bout_metrics


class _SelectionGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs if attrs is not None else {}

    def group_keys(self) -> list[str]:
        return [
            key
            for key, value in self.items()
            if isinstance(value, _SelectionGroup)
        ]


def _bout_records(offset: int = 0) -> np.ndarray:
    records = np.zeros(
        2,
        dtype=[
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("path_length_mm", "f8"),
        ],
    )
    records["bout_id"] = [0 + offset, 1 + offset]
    records["start_frame"] = [10 + offset, 30 + offset]
    records["end_frame"] = [20 + offset, 42 + offset]
    records["duration_s"] = [0.16, 0.20]
    records["path_length_mm"] = [1.5, 2.25]
    return records


def _peak_records() -> np.ndarray:
    records = np.zeros(
        2,
        dtype=[
            ("bout_id", "i8"),
            ("peak_frame", "i8"),
            ("peak_time_s", "f8"),
            ("peak_signal_value_mm_s", "f8"),
        ],
    )
    records["bout_id"] = [0, 1]
    records["peak_frame"] = [15, 35]
    records["peak_time_s"] = [0.25, 0.58]
    records["peak_signal_value_mm_s"] = [42.0, 51.0]
    return records


def _interval_records() -> np.ndarray:
    records = np.zeros(
        1,
        dtype=[
            ("prev_bout_id", "i8"),
            ("next_bout_id", "i8"),
            ("interval_s", "f8"),
        ],
    )
    records["prev_bout_id"] = [0]
    records["next_bout_id"] = [1]
    records["interval_s"] = [0.18]
    return records


def _build_v1_swim_bout_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_canary"
    parent.attrs["latest_complete"] = "bouts_canary"

    run = parent.create_group("bouts_canary")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 6,
            "source_track_kinematics_run": "tk_hyst4_low2_s005",
            "track_id": 0,
            "detection_method": "peak_event",
            "default_level": "speed_exponential",
            "exponential_tau_s": 0.025,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )

    filtered = run.create_group("speed_filtered")
    filtered.attrs.update(
        {
            "n_bouts": 2,
            "speed_level": "speed_filtered",
            "path_distance_source_level": "filtered",
        }
    )
    write_columnar_dataset(filtered, "bouts", _bout_records(offset=100))
    write_columnar_dataset(filtered, "peak_events", _peak_records())
    write_columnar_dataset(filtered, "inter_bout_intervals", _interval_records())

    exponential = run.create_group("speed_exponential")
    exponential.attrs.update(
        {
            "n_bouts": 2,
            "speed_level": "speed_exponential",
            "path_distance_source_level": "filtered",
            "detection_signal_transform_type": "exponential",
            "detection_signal_source_level": "filtered",
        }
    )
    write_columnar_dataset(exponential, "bouts", _bout_records())
    write_columnar_dataset(exponential, "peak_events", _peak_records())
    write_columnar_dataset(exponential, "inter_bout_intervals", _interval_records())
    store_array(exponential, "detection_signal_mm_s", np.asarray([0.0, 4.0, 8.0], dtype=np.float32))
    store_array(exponential, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))
    return root


def _build_compact_v2_swim_bout_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_compact"
    parent.attrs["latest_complete"] = "bouts_compact"
    run = parent.create_group("bouts_compact")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 7,
            "layout": "compact_tabular_v2",
            "source_track_kinematics_run": "tk_hyst4_low2_s005",
            "track_id": 0,
            "default_candidate_id": 0,
            "default_signal_id": 1,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    indexes = run.create_group("indexes")
    tables = run.create_group("tables")
    signals = run.create_group("signals")

    candidates = np.zeros(
        1,
        dtype=[
            ("candidate_id", "i4"),
            ("candidate_name", "S32"),
            ("is_default", "?"),
            ("detection_method", "S32"),
            ("parameters_json", "S128"),
        ],
    )
    candidates[0] = (0, b"compact_candidate", True, b"peak_event", b'{"method":"peak_event"}')
    write_columnar_dataset(indexes, "candidates", candidates)

    signal_variants = np.zeros(
        2,
        dtype=[
            ("signal_id", "i4"),
            ("speed_level", "S32"),
            ("signal_name", "S32"),
            ("role", "S32"),
            ("source_level", "S32"),
            ("transform_type", "S32"),
            ("transform_source_signal_id", "i4"),
            ("tau_s", "f8"),
            ("units", "S16"),
            ("path_distance_source_level", "S32"),
        ],
    )
    signal_variants[0] = (0, b"speed_filtered", b"filtered", b"physical_estimator", b"speed_filtered", b"identity", -1, np.nan, b"mm/s", b"filtered")
    signal_variants[1] = (1, b"speed_exponential", b"exponential", b"detector_response", b"speed_filtered", b"exponential", 0, 0.025, b"mm/s", b"filtered")
    write_columnar_dataset(indexes, "signal_variants", signal_variants)

    bouts = np.zeros(
        3,
        dtype=[
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("estimator_signal_id", "i4"),
            ("track_id", "i4"),
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("path_length_mm", "f8"),
        ],
    )
    bouts[0] = (0, 0, 0, 0, 10, 0, 5, 0.1, 1.0)
    bouts[1] = (0, 1, 0, 0, 20, 10, 16, 0.12, 2.0)
    bouts[2] = (0, 1, 0, 0, 21, 30, 36, 0.12, 2.5)
    write_columnar_dataset(tables, "bouts", bouts)

    peak_events = np.zeros(
        1,
        dtype=[
            ("peak_event_id", "i8"),
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("bout_id", "i8"),
            ("peak_frame", "i8"),
            ("peak_time_s", "f8"),
            ("peak_signal_value_mm_s", "f8"),
        ],
    )
    peak_events[0] = (0, 0, 1, 20, 13, 0.216, 42.0)
    write_columnar_dataset(tables, "peak_events", peak_events)

    intervals = np.zeros(
        1,
        dtype=[
            ("interval_id", "i8"),
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("prev_bout_id", "i8"),
            ("next_bout_id", "i8"),
            ("interval_s", "f8"),
            ("valid", "?"),
        ],
    )
    intervals[0] = (0, 0, 1, 20, 21, 0.2, True)
    write_columnar_dataset(tables, "inter_bout_intervals", intervals)

    summary = np.zeros(
        2,
        dtype=[
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("metric_name", "S64"),
            ("value", "f8"),
            ("units", "S16"),
            ("source_table", "S32"),
        ],
    )
    summary[0] = (0, 1, b"n_bouts", 2.0, b"count", b"bouts")
    summary[1] = (0, 1, b"total_path_length_mm", 4.5, b"mm", b"bouts")
    write_columnar_dataset(tables, "summary_metrics", summary)

    histograms = np.zeros(
        1,
        dtype=[
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("metric_name", "S64"),
            ("bin_left", "f8"),
            ("bin_right", "f8"),
            ("count", "i8"),
            ("density", "f8"),
            ("units", "S16"),
        ],
    )
    histograms[0] = (0, 1, b"inter_bout_interval_s", 0.1, 0.3, 1, np.nan, b"s")
    write_columnar_dataset(tables, "histograms", histograms)
    write_columnar_dataset(tables, "bout_points", np.zeros(0, dtype=[("candidate_id", "i4"), ("signal_id", "i4"), ("bout_id", "i8")]))
    store_array(signals, "detector_signal_mm_s", np.asarray([[0.0, 5.0, 8.0]], dtype=np.float32))
    store_array(signals, "detector_signal_signal_ids", np.asarray([1], dtype=np.int32))
    store_array(signals, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))
    return root


def test_discover_swim_bout_candidates_maps_v1_levels_to_signals() -> None:
    root = _build_v1_swim_bout_root()

    candidates = discover_swim_bout_candidates(
        root,
        track_run_name="offline/tk_hyst4_low2_s005",
        track_id=0,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.run_name == "bouts_canary"
    assert candidate.is_latest is True
    assert candidate.detection_method == "peak_event"
    assert candidate.default_speed_level == "speed_exponential"
    assert [signal.speed_level for signal in candidate.signals] == [
        "speed_filtered",
        "speed_exponential",
    ]
    assert [signal.role for signal in candidate.signals] == [
        "physical_estimator",
        "detector_response",
    ]
    assert candidate.default_signal_id == 1


def test_load_default_swim_bout_tables_uses_default_level() -> None:
    root = _build_v1_swim_bout_root()

    payload = load_default_swim_bout_tables(root)

    assert payload.run_name == "bouts_canary"
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.signal.role == "detector_response"
    assert payload.level_path == "analysis/swim_bout_runs/bouts_canary/speed_exponential"
    assert payload.bouts["bout_id"].tolist() == [0, 1]
    assert payload.inter_bout_intervals["interval_s"].tolist() == [0.18]
    assert payload.series["detection_signal_mm_s"].tolist() == [0.0, 4.0, 8.0]
    assert structured_records_to_dicts(payload.bouts)[0]["start_frame"] == 10


def test_swim_bout_reader_rejects_explicit_ineligible_run() -> None:
    parent = _SelectionGroup()
    parent["bouts_canary"] = _SelectionGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )

    with pytest.raises(SwimBoutIOError, match="not selector-eligible"):
        _resolve_run_name(parent, "bouts_canary")


def test_swim_bout_reader_fails_closed_during_selector_activation() -> None:
    parent = _SelectionGroup(
        attrs={"latest": "candidate", "latest_complete": "bouts_canary"}
    )
    parent["bouts_canary"] = _SelectionGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    parent["candidate"] = _SelectionGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        },
    )

    with pytest.raises(SwimBoutIOError, match="selector activation may be in progress"):
        _resolve_run_name(parent, None)


def test_swim_bout_legacy_run_requires_explicit_compatibility() -> None:
    parent = _SelectionGroup(attrs={"latest": "bouts_canary"})
    parent["bouts_canary"] = _SelectionGroup()

    with pytest.raises(SwimBoutIOError, match="No stable complete"):
        _resolve_run_name(parent, None)

    assert (
        _resolve_run_name(parent, None, legacy_compatibility=True)
        == "bouts_canary"
    )


def _build_legacy_mirror_target(root: zarr.Group) -> zarr.Group:
    target = root["analysis"].create_group("track_kinematics_target")
    target.create_group("tracks").create_group("id_0")
    return target


def test_track_kinematics_legacy_mirror_requires_explicit_compatibility() -> None:
    root = _build_v1_swim_bout_root()
    parent = root["analysis/swim_bout_runs"]
    run = parent["bouts_canary"]
    del parent.attrs["latest_complete"]
    del run.attrs["palette_run_completion_status"]
    del run.attrs["stage_selector_eligible"]
    target = _build_legacy_mirror_target(root)
    console = Console(file=io.StringIO())

    assert (
        _mirror_swim_bouts_to_tracks(
            root,
            target,
            [0],
            None,
            console,
        )
        is None
    )
    assert "swim_bouts" not in target["tracks/id_0"]

    assert (
        _mirror_swim_bouts_to_tracks(
            root,
            target,
            [0],
            None,
            console,
            legacy_compatibility=True,
        )
        == "bouts_canary"
    )
    assert "swim_bouts" in target["tracks/id_0"]


def test_track_kinematics_legacy_mirror_never_accepts_ineligible_run() -> None:
    root = _build_v1_swim_bout_root()
    run = root["analysis/swim_bout_runs/bouts_canary"]
    run.attrs["stage_selector_eligible"] = False
    target = _build_legacy_mirror_target(root)

    assert (
        _mirror_swim_bouts_to_tracks(
            root,
            target,
            [0],
            None,
            Console(file=io.StringIO()),
            legacy_compatibility=True,
        )
        is None
    )
    assert "swim_bouts" not in target["tracks/id_0"]


@pytest.mark.parametrize(
    "requested_run",
    ("  bouts_canary  ", "/analysis/swim_bout_runs/bouts_canary/"),
)
def test_open_zarr_swim_bout_resolution_normalizes_explicit_run(
    requested_run: str,
) -> None:
    root = _SelectionGroup()
    analysis = _SelectionGroup()
    parent = _SelectionGroup()
    root["analysis"] = analysis
    analysis["swim_bout_runs"] = parent
    parent["bouts_canary"] = _SelectionGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )

    assert (
        resolve_swim_bout_run_name(root, run_name=requested_run)
        == "bouts_canary"
    )


def test_load_swim_bout_tables_can_select_non_default_speed_level() -> None:
    root = _build_v1_swim_bout_root()

    payload = load_swim_bout_tables(root, speed_level="filtered")

    assert payload.signal.speed_level == "speed_filtered"
    assert payload.signal.role == "physical_estimator"
    assert payload.bouts["bout_id"].tolist() == [100, 101]


def test_legacy_latest_fallback_is_explicit_and_consistent() -> None:
    root = _build_v1_swim_bout_root()
    parent = root["analysis"]["swim_bout_runs"]
    del parent.attrs["latest"]
    earlier = parent.create_group("bouts_aaa")
    earlier.attrs.update(
        {
            "source_track_kinematics_run": "tk_hyst4_low2_s005",
            "track_id": 0,
            "detection_method": "threshold",
            "default_level": "speed_filtered",
        }
    )
    filtered = earlier.create_group("speed_filtered")
    write_columnar_dataset(filtered, "bouts", _bout_records(offset=200))

    with pytest.raises(SwimBoutIOError, match="No stable complete"):
        load_default_swim_bout_tables(root)

    candidates = discover_swim_bout_candidates(root, legacy_compatibility=True)
    payload = load_default_swim_bout_tables(
        root,
        legacy_compatibility=True,
    )

    assert candidates[0].run_name == "bouts_canary"
    assert candidates[0].is_latest is True
    assert payload.run_name == "bouts_canary"


def test_load_swim_bout_tables_requires_bouts_table() -> None:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_missing"
    parent.attrs["latest_complete"] = "bouts_missing"
    run = parent.create_group("bouts_missing")
    run.attrs["default_level"] = "speed_filtered"
    run.attrs["palette_run_completion_status"] = "complete"
    run.attrs["stage_selector_eligible"] = True
    run.create_group("speed_filtered")

    with pytest.raises(SwimBoutIOError, match="Missing required swim-bout table"):
        load_default_swim_bout_tables(root)


def test_discovery_counts_structured_array_bouts_without_n_bouts_attr() -> None:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_structured"
    parent.attrs["latest_complete"] = "bouts_structured"
    run = parent.create_group("bouts_structured")
    run.attrs["default_level"] = "speed_filtered"
    run.attrs["palette_run_completion_status"] = "complete"
    run.attrs["stage_selector_eligible"] = True
    filtered = run.create_group("speed_filtered")
    filtered.create_array("bouts", data=_bout_records(), overwrite=True)

    candidates = discover_swim_bout_candidates(root)

    assert candidates[0].signals[0].n_bouts == 2


def test_cross_recording_export_uses_swim_bout_resolver() -> None:
    root = _build_v1_swim_bout_root()

    rows = _load_swim_bout_metrics(
        root,
        export_run_id="export_test",
        zarr_path="/tmp/example_analysis.zarr",
        recording_id="recording_1",
        stimulus_run=None,
        protocol_signature=None,
        steps=[],
        tables={"swim_bout_metrics"},
        diagnostics=[],
    )

    assert len(rows) == 2
    assert rows[0]["swim_bout_run"] == "bouts_canary"
    assert rows[0]["speed_level"] == "speed_exponential"
    assert rows[0]["candidate_id"] == 0
    assert rows[0]["signal_id"] == 1
    assert rows[0]["signal_role"] == "detector_response"
    assert rows[0]["signal_source_level"] == "filtered"
    assert rows[0]["bout_id"] == 0


def test_discover_and_load_compact_v2_swim_bout_tables() -> None:
    root = _build_compact_v2_swim_bout_root()

    candidates = discover_swim_bout_candidates(root, track_run_name="tk_hyst4_low2_s005", track_id=0)
    payload = load_default_swim_bout_tables(root)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == 0
    assert candidates[0].candidate_name == "compact_candidate"
    assert candidates[0].default_signal_id == 1
    assert [signal.speed_level for signal in candidates[0].signals] == [
        "speed_filtered",
        "speed_exponential",
    ]
    assert [signal.n_bouts for signal in candidates[0].signals] == [1, 2]
    assert payload.signal.signal_id == 1
    assert payload.signal.role == "detector_response"
    assert payload.bouts["bout_id"].tolist() == [20, 21]
    assert payload.global_metrics["n_bouts"][0] == 2.0
    assert payload.inter_bout_interval_histogram["count"].tolist() == [1]
    assert payload.series["detection_signal_mm_s"].tolist() == [0.0, 5.0, 8.0]


def test_load_compact_v2_tables_can_select_physical_signal() -> None:
    root = _build_compact_v2_swim_bout_root()

    payload = load_swim_bout_tables(root, run_name="bouts_compact", speed_level="filtered")

    assert payload.signal.signal_id == 0
    assert payload.signal.role == "physical_estimator"
    assert payload.bouts["bout_id"].tolist() == [10]


def test_load_compact_v2_bout_events_skips_companion_tables() -> None:
    root = _build_compact_v2_swim_bout_root()
    candidate = discover_swim_bout_candidates(
        root,
        track_run_name="tk_hyst4_low2_s005",
        track_id=0,
        include_bout_counts=False,
    )[0]
    signal = next(item for item in candidate.signals if item.is_default)

    payload = load_swim_bout_events(root, candidate=candidate, signal=signal)

    assert [item.n_bouts for item in candidate.signals] == [0, 0]
    assert payload.bouts["bout_id"].tolist() == [20, 21]
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.level_path.endswith("candidate_id=0&signal_id=1")
