"""Tests for stimulus_response — Pass 1 base framework + Pass 2 bout integration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pytest
import zarr

import fisheye.analysis.stimulus_response as stimulus_response_module

from fisheye.analysis.stimulus_response import (
    BoutEntry,
    DenseTrack,
    ConcentricStepData,
    GratingStepData,
    LoomStepData,
    LoomTrial,
    ProtocolStep,
    _json_safe_attr_value,
    _resolve_omr_arena_geometry_mm,
    build_frame_annotations,
    compute_concentric_per_fish,
    compute_concentric_per_frame,
    compute_concentric_time_series,
    compute_global_metrics,
    compute_grating_per_fish,
    compute_grating_per_frame,
    compute_grating_time_series,
    compute_global_omr_metrics,
    compute_loom_per_fish,
    compute_loom_per_frame,
    compute_loom_per_trial_per_fish,
    compute_loom_time_series,
    compute_step_omr_metrics,
    compute_step_base_metrics,
    compute_step_bout_metrics,
    load_bout_data,
    load_track_data,
    parse_protocol_steps,
    reconstruct_loom_trials,
    resolve_grating_direction,
    resolve_grating_speed_mm_s,
    resolve_loom_center_mm,
    STIMULUS_RESPONSE_LAYOUT_COMPACT_V2,
    STIMULUS_RESPONSE_LAYOUT_DEFAULT,
    STIMULUS_RESPONSE_LAYOUT_HIERARCHICAL_V1,
    write_stimulus_response_run as _write_stimulus_response_run,
)
from fisheye.analysis.stimulus_response_concentric_omr import (
    compute_step_concentric_radial_omr_metrics,
    resolve_concentric_radial_polarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write_stimulus_response_run(*args, **kwargs):
    """Keep legacy structure tests on hierarchical-v1 unless they opt out."""

    kwargs.setdefault("layout", STIMULUS_RESPONSE_LAYOUT_HIERARCHICAL_V1)
    parameters = dict(kwargs.get("parameters", {}))
    parameters.setdefault("fps", 30.0)
    kwargs["parameters"] = parameters
    kwargs.setdefault(
        "upstream_lineage",
        _synthetic_track_motion_lineage(
            run_name=kwargs["source_kinematics_run"],
            scope=kwargs["source_kinematics_type"],
            track_ids=tuple(
                int(value)
                for value in np.asarray(kwargs["global_metrics"]["fish_id"])
            ),
            fps=float(kwargs.get("parameters", {}).get("fps", 30.0)),
        ),
    )
    return _write_stimulus_response_run(*args, **kwargs)


def _synthetic_track_motion_lineage(
    *,
    run_name: str,
    scope: str,
    track_ids: tuple[int, ...] = (0,),
    fps: float = 30.0,
    source_refs: Dict[str, object] | None = None,
):
    """Mint deterministic test-only evidence below the real strict loader."""

    run_path = f"analysis/track_kinematics_runs/{scope}/{run_name}"
    digest = "a" * 64
    bound = SimpleNamespace(
        run_group=SimpleNamespace(path=run_path),
        manifest_sha256=digest,
        tracks=tuple(SimpleNamespace(track_id=value) for value in track_ids),
        manifest={
            "run_root_attrs": {
                "record": {"immutable_attrs": {"fps": float(fps)}}
            },
            "run_derivation": {
                "record": {
                    "parameters": {"fps": float(fps)},
                    "source_refs": dict(source_refs or {}),
                }
            },
        },
        assert_verified=lambda: None,
    )
    record = stimulus_response_module._expected_track_motion_lineage_record(bound)
    return stimulus_response_module.VerifiedTrackMotionLineage(
        record=record,
        run_path=run_path,
        manifest_sha256=digest,
        track_ids=track_ids,
        fps=fps,
        bound_run=bound,
        _verification_seal=(
            stimulus_response_module._VERIFIED_TRACK_MOTION_LINEAGE_SEAL
        ),
    )


def _make_kinematics_zarr(
    *,
    n_frames: int = 100,
    fish_ids: tuple = (0, 1),
    gap_frames: tuple = (),
    fps: float = 30.0,
) -> zarr.Group:
    """Build a minimal track_kinematics zarr structure for testing."""
    root = zarr.group()

    # analysis/track_kinematics_runs/offline/<run>/
    analysis = root.create_group("analysis")
    kin_parent = analysis.create_group("track_kinematics_runs").create_group("offline")
    run = kin_parent.create_group("test_run")
    kin_parent.attrs["latest"] = "test_run"
    run.attrs["fps"] = fps
    run.attrs["track_motion_publication_manifest_sha256"] = "a" * 64
    run.attrs["source_refs"] = {
        "source_detection_path": "detect_runs/detect_fixture",
        "source_keypoint_path": "keypoints_runs/keypoints_fixture",
        "source_crop_path": "crop_runs/crop_fixture",
        "source_tracking_path": "tracking_runs/tracking_fixture",
    }

    track_ids = np.array(list(fish_ids), dtype=np.int32)
    run.create_array("track_ids", data=track_ids)

    tracks = run.create_group("tracks")
    for fid in fish_ids:
        # All frames except gaps.
        all_frames = np.arange(n_frames, dtype=np.int64)
        mask = np.ones(n_frames, dtype=bool)
        for gf in gap_frames:
            if 0 <= gf < n_frames:
                mask[gf] = False
        frame_indices = all_frames[mask]
        n_samples = len(frame_indices)

        tg = tracks.create_group(f"id_{fid}")
        tg.create_array("frame_indices", data=frame_indices)
        tg.create_array("time_seconds", data=(frame_indices / fps).astype(np.float32))
        # Position: fish moves 1mm per frame in x.
        pos = np.zeros((n_samples, 2), dtype=np.float32)
        pos[:, 0] = frame_indices.astype(np.float32) * 1.0
        frame_path_distance = np.zeros(n_samples, dtype=np.float32)
        if n_samples >= 2:
            consecutive = np.diff(frame_indices) == 1
            step_distance = np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32)
            frame_path_distance[1:][consecutive] = step_distance[consecutive]
        cumulative_path_distance = np.cumsum(frame_path_distance).astype(np.float32)
        tg.create_array("positions_mm", data=pos)
        tg.create_array("heading_degrees", data=np.full(n_samples, 90.0, dtype=np.float32))
        tg.create_array("speed_smoothed_mm", data=np.full(n_samples, 30.0, dtype=np.float32))
        tg.create_array("frame_path_distance_smoothed_mm", data=frame_path_distance)
        tg.create_array("cumulative_path_distance_mm", data=cumulative_path_distance)
        tg.create_array("angular_velocity_deg_s", data=np.zeros(n_samples, dtype=np.float32))
        tg.create_array("detection_source", data=np.zeros(n_samples, dtype=np.int8))

    return root


def _patch_synthetic_verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep sparse-to-dense unit tests below the strict reader boundary."""

    def _load(
        root: zarr.Group,
        *,
        run_name: str,
        scope: str,
        track_id: int,
        required_speed_levels=(),
    ) -> SimpleNamespace:
        del required_speed_levels
        run = root["analysis"]["track_kinematics_runs"][scope][run_name]
        track = run["tracks"][f"id_{int(track_id)}"]
        run_path = f"analysis/track_kinematics_runs/{scope}/{run_name}"
        return SimpleNamespace(
            track_id=int(track_id),
            track_path=f"{run_path}/tracks/id_{int(track_id)}",
            run_path=run_path,
            authority_status="verified_canonical_track_motion_v1",
            motion_manifest_sha256=run.attrs[
                "track_motion_publication_manifest_sha256"
            ],
            frame_indices=np.asarray(track["frame_indices"][:]),
            speed_mm_by_level={
                "smoothed": np.asarray(track["speed_smoothed_mm"][:]),
            },
            heading_degrees=np.asarray(track["heading_degrees"][:]),
            positions_mm=np.asarray(track["positions_mm"][:]),
            angular_velocity_deg_s=np.asarray(
                track["angular_velocity_deg_s"][:]
            ),
            time_seconds=np.asarray(track["time_seconds"][:]),
            detection_source=np.asarray(track["detection_source"][:]),
            frame_path_distance_mm_by_level={
                "smoothed": np.asarray(
                    track["frame_path_distance_smoothed_mm"][:]
                ),
            },
            cumulative_path_distance_mm=np.asarray(
                track["cumulative_path_distance_mm"][:]
            ),
        )

    monkeypatch.setattr(
        stimulus_response_module,
        "load_track_kinematics_track",
        _load,
    )

    def _lineage(root: zarr.Group, kin_group: zarr.Group):
        del root
        track_ids = tuple(
            int(value) for value in np.asarray(kin_group["track_ids"][:])
        )
        evidence = _synthetic_track_motion_lineage(
            run_name=str(kin_group.path).strip("/").rsplit("/", 1)[-1],
            scope=str(kin_group.path).strip("/").split("/")[2],
            track_ids=track_ids,
            fps=float(kin_group.attrs["fps"]),
            source_refs=dict(kin_group.attrs["source_refs"]),
        )
        evidence._bound_run.run_group = kin_group
        return evidence

    monkeypatch.setattr(
        stimulus_response_module,
        "_load_verified_track_motion_lineage",
        _lineage,
    )


def _make_stimulus_zarr(
    root: zarr.Group,
    *,
    steps: list | None = None,
    fps: float = 30.0,
) -> None:
    """Add stimulus_runs with events to an existing root."""
    if steps is None:
        steps = [
            {"start": 0, "end": 50, "mode": 4, "name": "baseline"},     # SOLID_BLACK
            {"start": 50, "end": 100, "mode": 3, "name": "grating_1"},  # MOVING_GRATING
        ]

    analysis = root.require_group("analysis")
    stim_parent = analysis.require_group("stimulus_runs")
    stim_run = stim_parent.create_group("test_stim")
    stim_parent.attrs["latest"] = "test_stim"

    # Build events arrays.
    event_names = []
    step_indices = []
    stimulus_modes = []
    camera_frames = []

    for s in steps:
        # STEP_START
        event_names.append("STEP_START")
        step_indices.append(s.get("index", steps.index(s)))
        stimulus_modes.append(s["mode"])
        camera_frames.append(s["start"])
        # STEP_END
        event_names.append("STEP_END")
        step_indices.append(s.get("index", steps.index(s)))
        stimulus_modes.append(s["mode"])
        camera_frames.append(s["end"])

    events = stim_run.create_group("events")
    # Variable-length UTF-8 strings for zarr v3.
    from zarr.core.dtype import VariableLengthUTF8
    n_events = len(event_names)
    name_data = np.empty(n_events, dtype=object)
    for idx, en in enumerate(event_names):
        name_data[idx] = en
    name_arr = events.create_array(
        "event_name",
        shape=(n_events,),
        chunks=(max(1, n_events),),
        dtype=VariableLengthUTF8(),
        fill_value="",
        overwrite=True,
    )
    name_arr[:] = name_data
    events.create_array("step_index", data=np.array(step_indices, dtype=np.int32))
    events.create_array("stimulus_mode", data=np.array(stimulus_modes, dtype=np.int32))
    events.create_array("camera_frame_id", data=np.array(camera_frames, dtype=np.uint64))

    # Minimal protocol JSON.
    protocol = {
        "steps": [
            {"name": s["name"], "stimulus_mode": s["mode"]}
            for s in steps
        ]
    }
    import json
    stim_run.attrs["protocol_json"] = json.dumps(protocol)


def _fixed_uint8_strings(values: list[str], width: int = 64) -> np.ndarray:
    arr = np.zeros((len(values), width), dtype=np.uint8)
    for idx, value in enumerate(values):
        encoded = value.encode("utf-8")[:width]
        arr[idx, :len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return arr


def _make_modern_stimulus_zarr(root: zarr.Group) -> None:
    """Add modern Citrus-style columnar events and uint8 enum strings."""
    analysis = root.require_group("analysis")

    enums = analysis.require_group("enums")
    event_enum = enums.create_group("events")
    event_enum.create_array("id", data=np.array([11, 12], dtype=np.int32))
    event_enum.create_array(
        "name",
        data=_fixed_uint8_strings(["STEP_START", "STEP_END"], width=32),
    )
    mode_enum = enums.create_group("stimulus_modes")
    mode_enum.create_array("id", data=np.array([3, 4], dtype=np.int32))
    mode_enum.create_array(
        "name",
        data=_fixed_uint8_strings(["MOVING_GRATING", "SOLID_BLACK"], width=32),
    )

    stim_parent = analysis.require_group("stimulus_runs")
    stim_run = stim_parent.create_group("modern_stim")
    stim_parent.attrs["latest"] = "modern_stim"

    events = stim_run.create_group("events")
    events.attrs["storage_layout"] = "columnar"
    events.attrs["field_names"] = [
        "event_type_id",
        "current_step_index",
        "stimulus_mode_id",
        "camera_frame_id",
        "name_or_context",
        "details_json",
    ]
    events.create_array("event_type_id", data=np.array([11, 12, 11, 12], dtype=np.int32))
    events.create_array("current_step_index", data=np.array([0, 0, 1, 1], dtype=np.int32))
    events.create_array("stimulus_mode_id", data=np.array([4, 4, 3, 3], dtype=np.int32))
    events.create_array("camera_frame_id", data=np.array([0, 50, 50, 100], dtype=np.uint64))
    events.create_array(
        "name_or_context",
        data=_fixed_uint8_strings(["baseline", "baseline", "grating", "grating"], width=32),
    )
    events.create_array("details_json", data=_fixed_uint8_strings(["{}", "{}", "{}", "{}"], width=32))

    import json
    stim_run.attrs["protocol_json"] = json.dumps({
        "steps": [
            {"name": "baseline", "stimulus_mode": 4},
            {"name": "grating_1", "stimulus_mode": 3},
        ]
    })


def _make_dense_tracks(
    n_frames: int = 100,
    n_fish: int = 2,
    speed: float = 30.0,
    gap_frames: tuple = (),
) -> List[DenseTrack]:
    """Create synthetic DenseTrack objects for metric computation tests."""
    tracks = []
    for fid in range(n_fish):
        valid = np.ones(n_frames, dtype=bool)
        for gf in gap_frames:
            if 0 <= gf < n_frames:
                valid[gf] = False

        speed_mm = np.full(n_frames, speed, dtype=np.float32)
        speed_mm[~valid] = 0.0

        pos = np.zeros((n_frames, 2), dtype=np.float32)
        pos[:, 0] = np.arange(n_frames, dtype=np.float32) * 1.0
        pos[~valid] = 0.0

        det_src = np.where(valid, np.int8(0), np.int8(-1))

        tracks.append(DenseTrack(
            fish_id=fid,
            speed_mm=speed_mm,
            heading_deg=np.full(n_frames, 90.0, dtype=np.float32),
            positions_mm=pos,
            angular_velocity=np.zeros(n_frames, dtype=np.float32),
            time_seconds=np.arange(n_frames, dtype=np.float32) / 30.0,
            valid=valid,
            detection_source=det_src,
        ))
    return tracks


# ---------------------------------------------------------------------------
# Sparse → dense expansion
# ---------------------------------------------------------------------------


class TestLoadTrackData:

    def test_basic_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        root = _make_kinematics_zarr(n_frames=50, fish_ids=(0,))
        _patch_synthetic_verified_track_reader(monkeypatch)
        tracks, run_name, n_frames, _ = load_track_data(root, kinematics_type="offline")
        assert len(tracks) == 1
        assert run_name == "test_run"
        assert n_frames == 50
        assert tracks[0].fish_id == 0
        assert tracks[0].valid.shape == (50,)
        assert tracks[0].valid.all()

    def test_gaps_produce_zeros_and_false_valid(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _make_kinematics_zarr(n_frames=20, fish_ids=(0,), gap_frames=(5, 10, 15))
        _patch_synthetic_verified_track_reader(monkeypatch)
        tracks, _, n_frames, _ = load_track_data(root, kinematics_type="offline")
        t = tracks[0]
        assert n_frames == 20
        # Gap frames should be invalid.
        assert not t.valid[5]
        assert not t.valid[10]
        assert not t.valid[15]
        # Gap frames should have zero speed.
        assert t.speed_mm[5] == 0.0
        assert t.speed_mm[10] == 0.0
        # Non-gap frames should be valid.
        assert t.valid[0]
        assert t.valid[6]
        # The first valid frame after a missing frame must not inherit the
        # across-gap position jump as distance.
        assert t.frame_path_distance_smoothed_mm is not None
        assert t.frame_path_distance_smoothed_mm[4] == 1.0
        assert t.frame_path_distance_smoothed_mm[5] == 0.0
        assert t.frame_path_distance_smoothed_mm[6] == 0.0
        assert t.frame_path_distance_smoothed_mm[7] == 1.0
        assert t.cumulative_path_distance_mm is not None
        assert t.cumulative_path_distance_mm[5] == t.cumulative_path_distance_mm[4]

    def test_multiple_fish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        root = _make_kinematics_zarr(n_frames=30, fish_ids=(0, 1, 2))
        _patch_synthetic_verified_track_reader(monkeypatch)
        tracks, _, _, _ = load_track_data(root, kinematics_type="offline")
        assert len(tracks) == 3
        assert [t.fish_id for t in tracks] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Protocol step parsing
# ---------------------------------------------------------------------------


class TestParseProtocolSteps:

    def test_basic_parsing(self) -> None:
        root = _make_kinematics_zarr(n_frames=100)
        _make_stimulus_zarr(root)
        steps, run_name, protocol = parse_protocol_steps(root, fps=30.0)
        assert len(steps) == 2
        assert steps[0].stimulus_mode == "SOLID_BLACK"
        assert steps[0].start_frame == 0
        assert steps[0].end_frame == 50
        assert steps[1].stimulus_mode == "MOVING_GRATING"
        assert steps[1].start_frame == 50
        assert steps[1].end_frame == 100

    def test_step_names_from_protocol(self) -> None:
        root = _make_kinematics_zarr(n_frames=100)
        _make_stimulus_zarr(root)
        steps, _, _ = parse_protocol_steps(root, fps=30.0)
        assert steps[0].name == "baseline"
        assert steps[1].name == "grating_1"

    def test_duration_computed(self) -> None:
        root = _make_kinematics_zarr(n_frames=100)
        _make_stimulus_zarr(root, fps=30.0)
        steps, _, _ = parse_protocol_steps(root, fps=30.0)
        assert abs(steps[0].duration_s - 50 / 30.0) < 0.01

    def test_modern_columnar_event_layout(self) -> None:
        root = _make_kinematics_zarr(n_frames=100)
        _make_modern_stimulus_zarr(root)
        steps, run_name, _ = parse_protocol_steps(root, fps=30.0)
        assert run_name == "modern_stim"
        assert len(steps) == 2
        assert steps[0].name == "baseline"
        assert steps[0].stimulus_mode == "SOLID_BLACK"
        assert steps[0].start_frame == 0
        assert steps[0].end_frame == 50
        assert steps[1].name == "grating_1"
        assert steps[1].stimulus_mode == "MOVING_GRATING"
        assert steps[1].start_frame == 50
        assert steps[1].end_frame == 100

    def test_prefers_canonical_imported_step_metadata(self) -> None:
        root = _make_kinematics_zarr(n_frames=100)
        _make_stimulus_zarr(root)
        stim_run = root["analysis"]["stimulus_runs"]["test_stim"]
        steps_group = stim_run.create_group("steps")
        step = steps_group.create_group("step_4")
        step.attrs.update({
            "metadata_schema_version": 1,
            "step_index": 4,
            "step_name": "canonical_grating",
            "stimulus_mode_id": 3,
            "stimulus_mode": "MOVING_GRATING",
            "start_camera_frame": 12,
            "end_camera_frame": 42,
            "duration_s": 1.0,
            "raw_protocol_params_json": json.dumps({
                "parameters": {
                    "orientation_degrees": 270.0,
                    "speed_mm_per_sec": 5.0,
                }
            }),
        })

        steps, _, _ = parse_protocol_steps(root, fps=30.0)

        assert len(steps) == 1
        assert steps[0].index == 4
        assert steps[0].name == "canonical_grating"
        assert steps[0].start_frame == 12
        assert steps[0].end_frame == 42
        assert resolve_grating_direction(steps[0]) == 270.0
        assert resolve_grating_speed_mm_s(steps[0]) == 5.0


# ---------------------------------------------------------------------------
# Base metric computation
# ---------------------------------------------------------------------------


class TestComputeGlobalMetrics:

    def test_basic_global(self) -> None:
        tracks = _make_dense_tracks(n_frames=100, n_fish=2, speed=30.0)
        result = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        assert result["fish_id"].tolist() == [0, 1]
        assert result["mean_speed_mm_s"][0] > 0
        assert result["fraction_moving"][0] > 0

    def test_stationary_fish(self) -> None:
        tracks = _make_dense_tracks(n_frames=50, n_fish=1, speed=0.0)
        result = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        assert result["fraction_moving"][0] == 0.0
        assert result["mean_speed_mm_s"][0] == 0.0

    def test_global_distance_uses_gap_aware_source_path_distance(self) -> None:
        n = 12
        valid = np.zeros(n, dtype=bool)
        valid[[0, 10]] = True
        positions = np.zeros((n, 2), dtype=np.float32)
        positions[10, 0] = 100.0
        tracks = [DenseTrack(
            fish_id=0,
            speed_mm=np.zeros(n, dtype=np.float32),
            heading_deg=np.zeros(n, dtype=np.float32),
            positions_mm=positions,
            angular_velocity=np.zeros(n, dtype=np.float32),
            time_seconds=np.arange(n, dtype=np.float32) / 30.0,
            valid=valid,
            detection_source=np.where(valid, np.int8(0), np.int8(-1)),
            frame_path_distance_smoothed_mm=np.zeros(n, dtype=np.float32),
            cumulative_path_distance_mm=np.zeros(n, dtype=np.float32),
        )]
        result = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        assert result["total_distance_mm"][0] == 0.0


class TestComputeStepBaseMetrics:

    def test_step_slicing(self) -> None:
        tracks = _make_dense_tracks(n_frames=100, n_fish=1, speed=10.0)
        step = ProtocolStep(
            index=0, name="test", stimulus_mode="SOLID_BLACK",
            stimulus_mode_id=4, start_frame=20, end_frame=40,
            duration_s=20 / 30.0,
        )
        result = compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)
        assert result["fish_id"].tolist() == [0]
        assert result["coverage"][0] == 1.0  # No gaps.
        assert result["fraction_moving"][0] == 1.0  # speed=10 > threshold=2.

    def test_coverage_with_gaps(self) -> None:
        # Gaps at frames 25, 30, 35 — step is frames 20-40 (20 frames).
        tracks = _make_dense_tracks(n_frames=100, n_fish=1, speed=10.0, gap_frames=(25, 30, 35))
        step = ProtocolStep(
            index=0, name="test", stimulus_mode="SOLID_BLACK",
            stimulus_mode_id=4, start_frame=20, end_frame=40,
            duration_s=20 / 30.0,
        )
        result = compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)
        # 3 gaps out of 20 step frames.
        assert abs(result["coverage"][0] - 17.0 / 20.0) < 0.01

    def test_step_distance_uses_gap_aware_source_path_distance(self) -> None:
        n = 12
        valid = np.zeros(n, dtype=bool)
        valid[[0, 10]] = True
        positions = np.zeros((n, 2), dtype=np.float32)
        positions[10, 0] = 100.0
        tracks = [DenseTrack(
            fish_id=0,
            speed_mm=np.zeros(n, dtype=np.float32),
            heading_deg=np.zeros(n, dtype=np.float32),
            positions_mm=positions,
            angular_velocity=np.zeros(n, dtype=np.float32),
            time_seconds=np.arange(n, dtype=np.float32) / 30.0,
            valid=valid,
            detection_source=np.where(valid, np.int8(0), np.int8(-1)),
            frame_path_distance_smoothed_mm=np.zeros(n, dtype=np.float32),
            cumulative_path_distance_mm=np.zeros(n, dtype=np.float32),
        )]
        step = ProtocolStep(
            index=0, name="gap", stimulus_mode="SOLID_BLACK",
            stimulus_mode_id=4, start_frame=0, end_frame=n,
            duration_s=n / 30.0,
        )
        result = compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)
        assert result["total_distance_mm"][0] == 0.0

    def test_empty_step(self) -> None:
        tracks = _make_dense_tracks(n_frames=100, n_fish=1, speed=10.0, gap_frames=tuple(range(50, 60)))
        step = ProtocolStep(
            index=0, name="empty", stimulus_mode="SOLID_BLACK",
            stimulus_mode_id=4, start_frame=50, end_frame=60,
            duration_s=10 / 30.0,
        )
        result = compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)
        assert result["coverage"][0] == 0.0
        assert result["mean_speed_mm_s"][0] == 0.0


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------


class TestWriteStimulusResponseRun:

    @staticmethod
    def _minimal_write_kwargs():
        tracks = _make_dense_tracks(n_frames=2, n_fish=1)
        step = ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 2, 2 / 30.0)
        return {
            "global_metrics": compute_global_metrics(
                tracks,
                fps=30.0,
                moving_threshold=2.0,
            ),
            "steps": [step],
            "step_metrics": [
                compute_step_base_metrics(
                    tracks,
                    step,
                    fps=30.0,
                    moving_threshold=2.0,
                )
            ],
            "source_kinematics_run": "test_kin",
            "source_kinematics_type": "offline",
            "source_stimulus_run": "test_stim",
            "parameters": {"fps": 30.0},
            "run_name": "strict_lineage",
        }

    @pytest.mark.parametrize("lineage", (None, {"manifest": "forged"}))
    def test_writer_rejects_missing_or_plain_mapping_lineage(self, lineage) -> None:
        root = zarr.group()
        kwargs = self._minimal_write_kwargs()
        kwargs["upstream_lineage"] = lineage

        with pytest.raises(ValueError, match="requires live verified"):
            _write_stimulus_response_run(root, **kwargs)
        assert "analysis" not in root

    def test_writer_rejects_conflicting_duplicate_source_args(self) -> None:
        root = zarr.group()
        kwargs = self._minimal_write_kwargs()
        kwargs["upstream_lineage"] = _synthetic_track_motion_lineage(
            run_name="different_run",
            scope="offline",
        )

        with pytest.raises(ValueError, match="conflict with verified lineage"):
            _write_stimulus_response_run(root, **kwargs)
        assert "analysis" not in root

    @pytest.mark.parametrize(
        ("track_ids", "fps", "message"),
        (
            ((99,), 30.0, "fish IDs conflict"),
            ((0,), 60.0, "fps conflicts"),
        ),
    )
    def test_writer_rejects_payload_or_fps_lineage_mismatch(
        self,
        track_ids: tuple[int, ...],
        fps: float,
        message: str,
    ) -> None:
        root = zarr.group()
        kwargs = self._minimal_write_kwargs()
        kwargs["upstream_lineage"] = _synthetic_track_motion_lineage(
            run_name="test_kin",
            scope="offline",
            track_ids=track_ids,
            fps=fps,
        )

        with pytest.raises(ValueError, match=message):
            _write_stimulus_response_run(root, **kwargs)
        assert "analysis" not in root

    def test_writer_rechecks_live_lineage_before_completion(self) -> None:
        root = zarr.group()
        kwargs = self._minimal_write_kwargs()
        evidence = _synthetic_track_motion_lineage(
            run_name="test_kin",
            scope="offline",
        )
        calls = 0

        def _assert_stable_then_change() -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ValueError("track source changed during publication")

        evidence._bound_run.assert_verified = _assert_stable_then_change
        kwargs["upstream_lineage"] = evidence

        with pytest.raises(ValueError, match="changed during publication"):
            _write_stimulus_response_run(root, **kwargs)
        run = root["analysis/stimulus_response_runs/strict_lineage"]
        assert run.attrs.get("palette_run_completion_status") != "complete"

    def test_default_layout_is_compact_v2(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        global_metrics = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        steps = [ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)]
        step_metrics = [compute_step_base_metrics(tracks, steps[0], fps=30.0, moving_threshold=2.0)]

        run_name = _write_stimulus_response_run(
            root,
            global_metrics=global_metrics,
            steps=steps,
            step_metrics=step_metrics,
            source_kinematics_run="test_kin",
            source_kinematics_type="offline",
            source_stimulus_run="test_stim",
            upstream_lineage=_synthetic_track_motion_lineage(
                run_name="test_kin",
                scope="offline",
                track_ids=(0,),
            ),
            parameters={"fps": 30.0},
            run_name="default_layout",
        )

        sr = root["analysis"]["stimulus_response_runs"][run_name]
        assert STIMULUS_RESPONSE_LAYOUT_DEFAULT == STIMULUS_RESPONSE_LAYOUT_COMPACT_V2
        assert sr.attrs["layout"] == STIMULUS_RESPONSE_LAYOUT_COMPACT_V2
        assert "step_index" in sr
        assert "steps" not in sr

    def test_roundtrip(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=100, n_fish=2, speed=10.0)
        global_metrics = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)

        steps = [
            ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0),
            ProtocolStep(1, "grating", "MOVING_GRATING", 3, 50, 100, 50 / 30.0),
        ]
        step_metrics = [
            compute_step_base_metrics(tracks, s, fps=30.0, moving_threshold=2.0)
            for s in steps
        ]

        run_name = write_stimulus_response_run(
            root,
            global_metrics=global_metrics,
            steps=steps,
            step_metrics=step_metrics,
            source_kinematics_run="test_kin",
            source_kinematics_type="offline",
            source_stimulus_run="test_stim",
            parameters={"moving_threshold_mm_s": 2.0, "fps": 30.0, "n_frames": 100},
            run_name="test_output",
        )

        assert run_name == "test_output"

        # Verify structure.
        sr = root["analysis"]["stimulus_response_runs"]["test_output"]
        assert sr.attrs["n_steps"] == 2
        assert sr.attrs["n_fish"] == 2

        # Global.
        assert "fish_id" in sr["global"]
        assert sr["global"]["fish_id"][:].tolist() == [0, 1]

        # Steps.
        s0 = sr["steps"]["step_0"]
        assert s0.attrs["stimulus_mode"] == "SOLID_BLACK"
        assert "fish_id" in s0["per_fish"]
        assert "coverage" in s0["per_fish"]

        s1 = sr["steps"]["step_1"]
        assert s1.attrs["stimulus_mode"] == "MOVING_GRATING"

    def test_latest_pointer_set(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        steps = [ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)]
        sm = [compute_step_base_metrics(tracks, steps[0], fps=30.0, moving_threshold=2.0)]

        write_stimulus_response_run(
            root, global_metrics=gm, steps=steps, step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="run_1",
        )

        parent = root["analysis"]["stimulus_response_runs"]
        assert parent.attrs["latest"] == "run_1"

    def test_duplicate_run_name_raises(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        steps = [ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)]
        sm = [compute_step_base_metrics(tracks, steps[0], fps=30.0, moving_threshold=2.0)]

        kwargs = dict(
            global_metrics=gm, steps=steps, step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="dup",
        )
        write_stimulus_response_run(root, **kwargs)
        with pytest.raises(ValueError, match="already exists"):
            write_stimulus_response_run(root, **kwargs)

    def test_overwrite_replaces_duplicate_run_name(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        steps = [ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)]
        sm = [compute_step_base_metrics(tracks, steps[0], fps=30.0, moving_threshold=2.0)]

        kwargs = dict(
            global_metrics=gm, steps=steps, step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={"version": 1}, run_name="dup",
        )
        write_stimulus_response_run(root, **kwargs)
        root["analysis"]["stimulus_response_runs"]["dup"].attrs["sentinel"] = "old"
        kwargs["parameters"] = {"version": 2}
        write_stimulus_response_run(root, **kwargs, overwrite=True)

        parent = root["analysis"]["stimulus_response_runs"]
        assert parent.attrs["latest"] == "dup"
        assert "sentinel" not in parent["dup"].attrs


# ---------------------------------------------------------------------------
# Pass 2: Bout integration
# ---------------------------------------------------------------------------


def _make_bouts(
    fish_id: int = 0,
    bouts: list | None = None,
) -> Dict[int, List[BoutEntry]]:
    """Create synthetic bout data."""
    if bouts is None:
        bouts = [
            {"id": 1, "start": 10, "end": 20, "dur": 0.33, "mean": 15.0, "peak": 25.0},
            {"id": 2, "start": 30, "end": 35, "dur": 0.17, "mean": 10.0, "peak": 18.0},
            {"id": 3, "start": 60, "end": 75, "dur": 0.50, "mean": 20.0, "peak": 35.0},
        ]
    entries = [
        BoutEntry(
            fish_id=fish_id,
            bout_id=b["id"],
            start_frame=b["start"],
            end_frame=b["end"],
            duration_s=b["dur"],
            mean_speed=b["mean"],
            peak_physical_speed=b["peak"],
        )
        for b in bouts
    ]
    return {fish_id: entries}


def test_load_bout_data_reads_current_physical_peak_schema() -> None:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_v6"
    run = parent.create_group("bouts_v6")
    run.attrs["default_level"] = "speed_exponential"
    run.attrs["track_id"] = 0
    level = run.create_group("speed_exponential")
    bouts = level.create_group("bouts")
    bouts.create_array("bout_id", data=np.array([1], dtype=np.int32))
    bouts.create_array("start_frame", data=np.array([10], dtype=np.int64))
    bouts.create_array("end_frame", data=np.array([20], dtype=np.int64))
    bouts.create_array("duration_s", data=np.array([0.25], dtype=np.float32))
    bouts.create_array("mean_speed_mm_s", data=np.array([4.0], dtype=np.float32))
    bouts.create_array("peak_detection_signal_mm_s", data=np.array([3.0], dtype=np.float32))
    bouts.create_array("peak_physical_speed_mm_s", data=np.array([8.0], dtype=np.float32))

    loaded, run_name = load_bout_data(root, bout_run="bouts_v6")

    assert run_name == "bouts_v6"
    assert loaded[0][0].mean_speed == pytest.approx(4.0)
    assert loaded[0][0].peak_physical_speed == pytest.approx(8.0)


class TestComputeStepBoutMetrics:

    def test_bouts_filtered_by_step(self) -> None:
        bouts = _make_bouts(fish_id=0)
        # Step covers frames 0-50 — should include bout 1 (10-20) and 2 (30-35),
        # but not bout 3 (60-75).
        step = ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)
        per_fish, per_bout = compute_step_bout_metrics(bouts, [0], step)
        assert per_fish["num_bouts"][0] == 2
        assert per_bout["fish_id"].tolist() == [0, 0]
        assert per_bout["bout_id"].tolist() == [1, 2]

    def test_no_bouts_in_step(self) -> None:
        bouts = _make_bouts(fish_id=0)
        # Step covers frames 80-100 — no bouts overlap.
        step = ProtocolStep(0, "empty", "SOLID_BLACK", 4, 80, 100, 20 / 30.0)
        per_fish, per_bout = compute_step_bout_metrics(bouts, [0], step)
        assert per_fish["num_bouts"][0] == 0
        assert per_fish["mean_bout_duration_s"][0] == 0.0
        assert per_bout["fish_id"].size == 0

    def test_mean_bout_duration(self) -> None:
        bouts = _make_bouts(fish_id=0)
        # Step covers all bouts.
        step = ProtocolStep(0, "all", "SOLID_BLACK", 4, 0, 100, 100 / 30.0)
        per_fish, per_bout = compute_step_bout_metrics(bouts, [0], step)
        assert per_fish["num_bouts"][0] == 3
        expected_mean_dur = (0.33 + 0.17 + 0.50) / 3
        assert abs(per_fish["mean_bout_duration_s"][0] - expected_mean_dur) < 0.01

    def test_interbout_interval(self) -> None:
        bouts = _make_bouts(fish_id=0)
        step = ProtocolStep(0, "all", "SOLID_BLACK", 4, 0, 100, 100 / 30.0)
        per_fish, per_bout = compute_step_bout_metrics(bouts, [0], step)
        # IBI between bout 1 (end=20) and 2 (start=30) = 10 frames.
        # IBI between bout 2 (end=35) and 3 (start=60) = 25 frames.
        # Mean IBI = 17.5 frames. At 100/30.0 s per 100 frames = 1/30 s per frame.
        # mean_ibi = 17.5 * (100/30.0 / 100) = 17.5 / 30.0 ≈ 0.583
        assert per_fish["mean_interbout_interval_s"][0] > 0

    def test_fish_with_no_bout_data(self) -> None:
        bouts = _make_bouts(fish_id=0)
        # Fish 1 has no bouts.
        step = ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 100, 100 / 30.0)
        per_fish, per_bout = compute_step_bout_metrics(bouts, [0, 1], step)
        assert per_fish["num_bouts"][0] == 3
        assert per_fish["num_bouts"][1] == 0

    def test_per_bout_arrays_correct(self) -> None:
        bouts = _make_bouts(fish_id=0)
        step = ProtocolStep(0, "all", "SOLID_BLACK", 4, 0, 100, 100 / 30.0)
        _, per_bout = compute_step_bout_metrics(bouts, [0], step)
        assert per_bout["duration_s"].tolist() == pytest.approx([0.33, 0.17, 0.50], abs=0.01)
        assert per_bout["mean_speed_mm_s"].tolist() == pytest.approx([15.0, 10.0, 20.0])
        assert per_bout["peak_physical_speed_mm_s"].tolist() == pytest.approx([25.0, 18.0, 35.0])


class TestWriteWithBouts:

    def test_bout_data_written(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=100, n_fish=1, speed=10.0)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)

        steps = [
            ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0),
            ProtocolStep(1, "grating", "MOVING_GRATING", 3, 50, 100, 50 / 30.0),
        ]
        sm = [compute_step_base_metrics(tracks, s, fps=30.0, moving_threshold=2.0) for s in steps]

        bouts = _make_bouts(fish_id=0)
        sbm = [compute_step_bout_metrics(bouts, [0], s) for s in steps]

        write_stimulus_response_run(
            root, global_metrics=gm, steps=steps, step_metrics=sm,
            step_bout_metrics=sbm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", source_bout_run="b",
            parameters={}, run_name="bout_test",
        )

        sr = root["analysis"]["stimulus_response_runs"]["bout_test"]
        assert sr.attrs["source_bout_run"] == "b"

        s0 = sr["steps"]["step_0"]
        assert "num_bouts" in s0["per_fish"]
        assert s0["per_fish"]["num_bouts"][:][0] >= 0

        # Step 0 (frames 0-50) should have bouts, so per_bout should exist.
        if s0["per_fish"]["num_bouts"][:][0] > 0:
            assert "per_bout" in s0
            assert "fish_id" in s0["per_bout"]
            assert "mean_speed_mm_s" in s0["per_bout"]
            assert "peak_physical_speed_mm_s" in s0["per_bout"]

    def test_no_bouts_still_works(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        steps = [ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)]
        sm = [compute_step_base_metrics(tracks, steps[0], fps=30.0, moving_threshold=2.0)]

        # No bout metrics — should still write fine.
        write_stimulus_response_run(
            root, global_metrics=gm, steps=steps, step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="no_bouts",
        )
        sr = root["analysis"]["stimulus_response_runs"]["no_bouts"]
        assert "source_bout_run" not in sr.attrs
        assert "num_bouts" not in sr["steps"]["step_0"]["per_fish"]


# ---------------------------------------------------------------------------
# Pass 3: Grating metrics
# ---------------------------------------------------------------------------


def _grating_step(
    start: int = 0,
    end: int = 100,
    direction_deg: float = 90.0,
    grating_speed: float = 10.0,
) -> ProtocolStep:
    return ProtocolStep(
        index=0,
        name="grating_test",
        stimulus_mode="MOVING_GRATING",
        stimulus_mode_id=3,
        start_frame=start,
        end_frame=end,
        duration_s=(end - start) / 30.0,
        stimulus_params={
            "orientation_degrees": direction_deg,
            "grating_speed_mm_s": grating_speed,
        },
    )


def _make_grating_tracks(
    n_frames: int = 100,
    heading_deg: float = 90.0,
    speed: float = 10.0,
) -> List[DenseTrack]:
    """Create a single fish heading in a fixed direction."""
    valid = np.ones(n_frames, dtype=bool)
    pos = np.zeros((n_frames, 2), dtype=np.float32)
    # Fish moves in heading direction.
    heading_rad = np.deg2rad(heading_deg)
    for f in range(1, n_frames):
        pos[f, 0] = pos[f - 1, 0] + speed / 30.0 * np.cos(heading_rad)
        pos[f, 1] = pos[f - 1, 1] + speed / 30.0 * np.sin(heading_rad)
    return [DenseTrack(
        fish_id=0,
        speed_mm=np.full(n_frames, speed, dtype=np.float32),
        heading_deg=np.full(n_frames, heading_deg, dtype=np.float32),
        positions_mm=pos,
        angular_velocity=np.zeros(n_frames, dtype=np.float32),
        time_seconds=np.arange(n_frames, dtype=np.float32) / 30.0,
        valid=valid,
        detection_source=np.zeros(n_frames, dtype=np.int8),  # all real
    )]


class TestResolveGratingDirection:

    def test_basic(self) -> None:
        step = _grating_step(direction_deg=270.0)
        assert resolve_grating_direction(step) == 270.0

    def test_with_offset(self) -> None:
        step = _grating_step(direction_deg=270.0)
        assert resolve_grating_direction(step, offset_deg=10.0) == 280.0

    def test_offset_wraps_to_camera_space_degrees(self) -> None:
        step = _grating_step(direction_deg=180.0)
        assert resolve_grating_direction(step, offset_deg=180.0) == 0.0

    def test_fallback_keys(self) -> None:
        step = ProtocolStep(0, "t", "MOVING_GRATING", 3, 0, 100, 100 / 30.0,
                            stimulus_params={"angle_degrees": 45.0})
        assert resolve_grating_direction(step) == 45.0

    def test_nested_citrus_protocol_parameters(self) -> None:
        step = ProtocolStep(
            0,
            "t",
            "MOVING_GRATING",
            3,
            0,
            100,
            100 / 30.0,
            stimulus_params={
                "duration_seconds": 100 / 30.0,
                "parameters": {
                    "orientation_degrees": 180.0,
                    "speed_mm_per_sec": 5.0,
                    "type": "ProtocolMovingGratingParams",
                },
                "stimulus_mode_str": "MOVING_GRATING",
            },
        )

        assert resolve_grating_direction(step) == 180.0
        assert resolve_grating_speed_mm_s(step) == 5.0

    def test_configured_canonical_step_direction_is_not_offset_twice(self) -> None:
        step = ProtocolStep(
            0,
            "t",
            "MOVING_GRATING",
            3,
            0,
            100,
            100 / 30.0,
            stimulus_params={
                "moving_grating": {
                    "orientation_degrees_authored": 0.0,
                    "grating_direction_camera_deg": 180.0,
                    "camera_to_projector_offset_deg": 180.0,
                    "direction_mapping_status": "configured_camera_offset",
                    "speed_mm_s": 5.0,
                }
            },
        )

        assert resolve_grating_direction(step, offset_deg=180.0) == 180.0
        assert resolve_grating_speed_mm_s(step) == 5.0

    def test_unvalidated_canonical_step_direction_can_receive_runtime_offset(self) -> None:
        step = ProtocolStep(
            0,
            "t",
            "MOVING_GRATING",
            3,
            0,
            100,
            100 / 30.0,
            stimulus_params={
                "moving_grating": {
                    "orientation_degrees_authored": 0.0,
                    "grating_direction_camera_deg": 0.0,
                    "camera_to_projector_offset_deg": 0.0,
                    "direction_mapping_status": "unvalidated_default_zero_offset",
                }
            },
        )

        assert resolve_grating_direction(step, offset_deg=180.0) == 180.0


class TestComputeGratingPerFrame:

    def test_perfect_following(self) -> None:
        """Fish heading matches grating direction exactly."""
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        # Alignment should be ~0 deg (perfect following).
        assert pf["alignment_angle_deg"].shape == (1, 100)
        assert np.allclose(pf["alignment_angle_deg"][0], 0.0, atol=0.01)
        assert np.allclose(pf["alignment_cos"][0], 1.0, atol=0.01)

    def test_opposing(self) -> None:
        """Fish heading is 180 deg from grating direction."""
        tracks = _make_grating_tracks(heading_deg=270.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        assert np.allclose(pf["alignment_cos"][0], -1.0, atol=0.01)

    def test_perpendicular(self) -> None:
        """Fish heading is 90 deg from grating direction."""
        tracks = _make_grating_tracks(heading_deg=0.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        assert np.allclose(pf["alignment_cos"][0], 0.0, atol=0.01)

    def test_speed_along_grating(self) -> None:
        """Fish swimming in grating direction → speed_along = full speed."""
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        assert np.allclose(pf["speed_along_grating_mm_s"][0], 10.0, atol=0.1)

    def test_valid_mask_all_valid(self) -> None:
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        assert pf["valid"].shape == (1, 100)
        assert pf["valid"].all()

    def test_valid_mask_with_gaps(self) -> None:
        """Gap frames should be marked invalid and have zero alignment."""
        tracks = _make_dense_tracks(n_frames=100, n_fish=1, speed=10.0, gap_frames=(10, 20, 30))
        # Override heading so alignment is nonzero for valid frames.
        tracks[0].heading_deg[:] = 90.0
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        assert not pf["valid"][0, 10]
        assert not pf["valid"][0, 20]
        assert not pf["valid"][0, 30]
        assert pf["valid"][0, 0]
        assert pf["valid"][0, 15]
        # Invalid frames should have zero alignment_cos.
        assert pf["alignment_cos"][0, 10] == 0.0
        assert pf["alignment_cos"][0, 20] == 0.0
        # detection_source should be -1 for gaps, 0 for real detections.
        assert pf["detection_source"][0, 10] == -1
        assert pf["detection_source"][0, 0] == 0

    def test_detection_source_all_real(self) -> None:
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        assert (pf["detection_source"][0] == 0).all()


class TestComputeGratingPerFish:

    def test_perfect_following_summary(self) -> None:
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0, grating_speed=10.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        gpf = compute_grating_per_fish(pf, tracks, step, fps=30.0, grating_speed_mm_s=10.0)
        assert gpf["mean_alignment_cos"][0] > 0.99
        assert gpf["fraction_following"][0] == 1.0
        assert gpf["fraction_opposing"][0] == 0.0
        assert abs(gpf["optomotor_gain"][0] - 1.0) < 0.1

    def test_opposing_summary(self) -> None:
        tracks = _make_grating_tracks(heading_deg=270.0, speed=10.0)
        step = _grating_step(direction_deg=90.0, grating_speed=10.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        gpf = compute_grating_per_fish(pf, tracks, step, fps=30.0, grating_speed_mm_s=10.0)
        assert gpf["mean_alignment_cos"][0] < -0.99
        assert gpf["fraction_opposing"][0] == 1.0
        assert gpf["optomotor_gain"][0] < -0.9

    def test_latency_immediate_follower(self) -> None:
        """Fish follows from frame 0 → latency should be 0."""
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        gpf = compute_grating_per_fish(pf, tracks, step, fps=30.0, follow_threshold=0.5)
        assert gpf["latency_to_follow_s"][0] == 0.0

    def test_latency_never_follows(self) -> None:
        """Fish heading perpendicular → never meets follow threshold."""
        tracks = _make_grating_tracks(heading_deg=0.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        gpf = compute_grating_per_fish(pf, tracks, step, fps=30.0, follow_threshold=0.5)
        assert np.isnan(gpf["latency_to_follow_s"][0])

    def test_drift_along_grating(self) -> None:
        """Fish swimming in grating direction accumulates positive drift."""
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0, n_frames=100)
        step = _grating_step(direction_deg=90.0, start=0, end=100)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        gpf = compute_grating_per_fish(pf, tracks, step, fps=30.0)
        # Drift along grating should be positive (moved in grating direction).
        assert gpf["drift_along_grating_mm"][0] > 0
        # Drift perpendicular should be ~0.
        assert abs(gpf["drift_perp_grating_mm"][0]) < 0.1


class TestComputeGratingTimeSeries:

    def test_bin_count(self) -> None:
        tracks = _make_grating_tracks(n_frames=90, heading_deg=90.0)
        step = _grating_step(start=0, end=90)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        ts = compute_grating_time_series(pf, tracks, step, fps=30.0, bin_size_s=1.0)
        # 90 frames / 30 fps = 3 seconds → 3 bins at 1s each.
        assert ts["bin_center_s"].shape[0] == 3
        assert ts["alignment_cos"].shape == (1, 3)

    def test_following_in_all_bins(self) -> None:
        tracks = _make_grating_tracks(n_frames=60, heading_deg=90.0, speed=10.0)
        step = _grating_step(start=0, end=60, direction_deg=90.0, grating_speed=10.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)
        ts = compute_grating_time_series(pf, tracks, step, fps=30.0, bin_size_s=1.0, grating_speed_mm_s=10.0)
        # All bins should show perfect following.
        assert np.all(ts["alignment_cos"][0] > 0.99)
        assert np.all(ts["fraction_following"][0] == 1.0)


class TestComputeOMRMetrics:

    def test_json_safe_attr_value_converts_nonfinite_metadata_to_null(self) -> None:
        attrs = _json_safe_attr_value({
            "ok": 1.0,
            "missing": np.float64(np.nan),
            "nested": [np.float32(np.inf), -np.inf],
        })

        assert attrs == {"ok": 1.0, "missing": None, "nested": [None, None]}
        assert json.loads(json.dumps(attrs, allow_nan=False)) == attrs

    def test_path_index_following_and_opposing(self) -> None:
        step = _grating_step(direction_deg=90.0)

        following = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        omr_follow = compute_step_omr_metrics(
            following, step, 90.0, fps=30.0, moving_threshold_mm_s=2.0,
        )
        assert omr_follow.per_fish["omr_path_index"][0] > 0.99
        assert omr_follow.per_fish["time_fraction_correct_classified"][0] == 1.0

        opposing = _make_grating_tracks(heading_deg=270.0, speed=10.0)
        omr_oppose = compute_step_omr_metrics(
            opposing, step, 90.0, fps=30.0, moving_threshold_mm_s=2.0,
        )
        assert omr_oppose.per_fish["omr_path_index"][0] < -0.99
        assert omr_oppose.per_fish["time_fraction_correct_classified"][0] == 0.0

    def test_perpendicular_path_index_is_near_zero(self) -> None:
        tracks = _make_grating_tracks(heading_deg=0.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        omr = compute_step_omr_metrics(
            tracks, step, 90.0, fps=30.0, moving_threshold_mm_s=2.0,
        )
        assert abs(float(omr.per_fish["omr_path_index"][0])) < 1e-5

    def test_attrs_capture_moving_grating_direction_and_window_provenance(self) -> None:
        tracks = _make_grating_tracks(heading_deg=180.0, speed=10.0)
        step = ProtocolStep(
            index=0,
            name="grating_provenance",
            stimulus_mode="MOVING_GRATING",
            stimulus_mode_id=3,
            start_frame=0,
            end_frame=100,
            duration_s=100 / 30.0,
            stimulus_params={
                "parameters": {
                    "orientation_degrees": 0.0,
                    "speed_pps": 50.0,
                    "speed_mm_per_sec": 5.0,
                    "spatial_freq_cpp": 0.02,
                    "spatial_freq_cycles_per_mm": 0.15,
                },
                "moving_grating": {
                    "orientation_degrees_authored": 0.0,
                    "grating_direction_camera_deg": 180.0,
                    "camera_to_projector_offset_deg": 180.0,
                    "direction_mapping_source": "protocol_orientation_degrees_plus_configured_offset",
                    "direction_mapping_status": "configured_camera_offset",
                    "direction_mapping_validated": False,
                    "temporal_frequency_hz": 0.75,
                    "actual_rendered_temporal_frequency_hz": 1.8,
                },
            },
        )

        omr = compute_step_omr_metrics(
            tracks,
            step,
            180.0,
            fps=30.0,
            moving_threshold_mm_s=2.0,
            window_lengths_s=(3.0, 7.0),
            early_window_lengths_s=(1.0, 2.0),
        )

        attrs = omr.attrs
        assert attrs["stimulus_direction_deg"] == 180.0
        assert attrs["grating_direction_camera_deg"] == 180.0
        assert attrs["orientation_degrees_authored"] == 0.0
        assert attrs["camera_to_projector_offset_deg"] == 180.0
        assert attrs["direction_mapping_status"] == "configured_camera_offset"
        assert attrs["direction_mapping_validated"] is False
        assert attrs["window_lengths_s"] == [3.0, 7.0]
        assert attrs["early_response_window_lengths_s"] == [1.0, 2.0]
        assert attrs["speed_mm_s"] == 5.0
        assert attrs["spatial_freq_cycles_per_mm"] == 0.15
        assert attrs["temporal_frequency_hz"] == 0.75

    def test_position_occupancy_and_opportunity_are_arena_normalized(self) -> None:
        tracks = _make_grating_tracks(n_frames=31, heading_deg=90.0, speed=30.0)
        step = _grating_step(start=0, end=31, direction_deg=90.0)
        omr = compute_step_omr_metrics(
            tracks,
            step,
            90.0,
            fps=30.0,
            moving_threshold_mm_s=2.0,
            arena_center_mm=(0.0, 10.0),
            arena_axis_extent_mm=20.0,
            arena_geometry_source="test_arena",
        )

        pf = omr.per_fish
        assert pf["start_position_axis_mm"][0] == pytest.approx(-10.0)
        assert pf["end_position_axis_mm"][0] == pytest.approx(20.0)
        assert pf["start_position_axis_norm"][0] == pytest.approx(-0.5)
        assert pf["end_position_axis_norm"][0] == pytest.approx(1.0)
        assert pf["available_forward_space_at_start_mm"][0] == pytest.approx(30.0)
        assert pf["available_backward_space_at_start_mm"][0] == pytest.approx(10.0)
        assert pf["opportunity_normalized_parallel_displacement"][0] == pytest.approx(1.0)
        assert 0.0 < pf["fraction_time_correct_side"][0] < 1.0
        assert omr.attrs["arena_geometry_source"] == "test_arena"

    def test_arena_geometry_uses_projector_scale_for_experimental_area_center(self) -> None:
        root = zarr.group()
        cal = root.create_group("analysis").create_group("calibration")
        cal.attrs.update(
            {
                "pixel_to_mm": 0.0187880455,
                "pixels_per_mm_projector": 4.1693377495,
                "experimental_area_center_x_px": 172.0,
                "experimental_area_center_y_px": 172.0,
                "experimental_area_radius_mm": 39.814476,
            }
        )

        center_mm, extent_mm, source = _resolve_omr_arena_geometry_mm(
            root,
            stimulus_run=None,
            direction_xy=np.array([1.0, 0.0], dtype=np.float64),
        )

        assert center_mm is not None
        assert center_mm[0] == pytest.approx(172.0 / 4.1693377495)
        assert center_mm[1] == pytest.approx(172.0 / 4.1693377495)
        assert extent_mm == pytest.approx(39.814476)
        assert "experimental_area_center_projector_px" in source

    def test_bout_fraction_uses_bout_boundaries_and_physical_positions(self) -> None:
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        bouts = {
            0: [
                BoutEntry(
                    fish_id=0,
                    bout_id=7,
                    start_frame=10,
                    end_frame=40,
                    duration_s=1.0,
                    mean_speed=10.0,
                    peak_physical_speed=10.0,
                )
            ]
        }
        omr = compute_step_omr_metrics(
            tracks,
            step,
            90.0,
            fps=30.0,
            moving_threshold_mm_s=2.0,
            bouts_by_fish=bouts,
        )
        assert omr.per_fish["bout_count_total"][0] == 1
        assert omr.per_fish["bout_count_correct"][0] == 1
        assert omr.per_fish["bout_fraction_correct_classified"][0] == 1.0
        assert omr.per_bout["bout_id"][0] == 7
        assert omr.per_bout["per_bout_omr_score"][0] > 0.99

    def test_weighted_bout_metrics_capture_large_aligned_bout(self) -> None:
        """Weighted bout evidence is not the same as unweighted bout counts."""

        n_frames = 100
        fps = 30.0
        pos = np.zeros((n_frames, 2), dtype=np.float32)
        for frame in range(1, n_frames):
            if frame <= 20:
                pos[frame, 1] = pos[frame - 1, 1] - 0.10
            elif frame <= 30:
                pos[frame, 1] = pos[frame - 1, 1]
            elif frame <= 80:
                pos[frame, 1] = pos[frame - 1, 1] + 0.30
            else:
                pos[frame, 1] = pos[frame - 1, 1]
        tracks = [
            DenseTrack(
                fish_id=0,
                speed_mm=np.full(n_frames, 9.0, dtype=np.float32),
                heading_deg=np.full(n_frames, 90.0, dtype=np.float32),
                positions_mm=pos,
                angular_velocity=np.zeros(n_frames, dtype=np.float32),
                time_seconds=np.arange(n_frames, dtype=np.float32) / fps,
                valid=np.ones(n_frames, dtype=bool),
                detection_source=np.zeros(n_frames, dtype=np.int8),
            )
        ]
        step = _grating_step(direction_deg=90.0, start=0, end=n_frames)
        bouts = {
            0: [
                BoutEntry(0, 1, 0, 20, 20 / fps, 9.0, 9.0),
                BoutEntry(0, 2, 30, 80, 50 / fps, 9.0, 9.0),
            ]
        }

        omr = compute_step_omr_metrics(
            tracks,
            step,
            90.0,
            fps=fps,
            moving_threshold_mm_s=2.0,
            bouts_by_fish=bouts,
        )

        pf = omr.per_fish
        assert pf["bout_count_correct"][0] == 1
        assert pf["bout_count_opposing"][0] == 1
        assert pf["bout_choice_index"][0] == pytest.approx(0.0)
        assert pf["bout_path_index"][0] > 0.7
        assert pf["bout_fraction_correct_weighted_by_path"][0] > 0.85
        assert pf["bout_fraction_correct_weighted_by_displacement"][0] > 0.85

    def test_first_aligned_bout_latency_uses_first_classified_bout(self) -> None:
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        step = _grating_step(direction_deg=90.0)
        bouts = {
            0: [
                BoutEntry(
                    fish_id=0,
                    bout_id=3,
                    start_frame=12,
                    end_frame=30,
                    duration_s=0.6,
                    mean_speed=10.0,
                    peak_physical_speed=10.0,
                ),
                BoutEntry(
                    fish_id=0,
                    bout_id=4,
                    start_frame=40,
                    end_frame=50,
                    duration_s=0.33,
                    mean_speed=10.0,
                    peak_physical_speed=10.0,
                ),
            ]
        }
        omr = compute_step_omr_metrics(
            tracks,
            step,
            90.0,
            fps=30.0,
            moving_threshold_mm_s=2.0,
            bouts_by_fish=bouts,
        )

        pf = omr.per_fish
        assert pf["first_aligned_bout_id"][0] == 3
        assert pf["first_classified_bout_id"][0] == 3
        assert pf["first_aligned_bout_start_frame"][0] == 12
        assert pf["first_aligned_bout_latency_s"][0] == pytest.approx(12 / 30.0)
        assert pf["first_aligned_bout_score"][0] > 0.99
        assert pf["first_opposing_bout_id"][0] == -1
        assert np.isnan(pf["first_opposing_bout_latency_s"][0])

    def test_early_response_windows_start_at_grating_onset(self) -> None:
        n_frames = 120
        fps = 30.0
        pos = np.zeros((n_frames, 2), dtype=np.float32)
        for frame in range(1, n_frames):
            if frame <= 30:
                pos[frame, 1] = pos[frame - 1, 1] + 0.50
            else:
                pos[frame, 1] = pos[frame - 1, 1] - 0.20
        tracks = [
            DenseTrack(
                fish_id=0,
                speed_mm=np.full(n_frames, 15.0, dtype=np.float32),
                heading_deg=np.full(n_frames, 90.0, dtype=np.float32),
                positions_mm=pos,
                angular_velocity=np.zeros(n_frames, dtype=np.float32),
                time_seconds=np.arange(n_frames, dtype=np.float32) / fps,
                valid=np.ones(n_frames, dtype=bool),
                detection_source=np.zeros(n_frames, dtype=np.int8),
            )
        ]
        step = _grating_step(direction_deg=90.0, start=0, end=n_frames)

        omr = compute_step_omr_metrics(
            tracks,
            step,
            90.0,
            fps=fps,
            moving_threshold_mm_s=2.0,
            early_window_lengths_s=(1.0, 2.0),
        )

        ew = omr.early_windows
        assert ew["window_length_s"].tolist() == pytest.approx([1.0, 2.0])
        assert ew["start_frame"].tolist() == [0, 0]
        assert ew["end_frame"].tolist() == [30, 60]
        assert ew["omr_path_index"][0] > 0.99
        assert ew["omr_path_index"][1] > 0.0

    def test_global_omr_aggregates_steps_by_path(self) -> None:
        step = _grating_step(direction_deg=90.0)
        tracks = _make_grating_tracks(heading_deg=90.0, speed=10.0)
        omr = compute_step_omr_metrics(
            tracks, step, 90.0, fps=30.0, moving_threshold_mm_s=2.0,
        )
        global_omr = compute_global_omr_metrics([0], [omr])
        assert global_omr["eligible_step_count"][0] == 1
        assert global_omr["omr_path_index_weighted_by_path"][0] > 0.99


class TestWriteWithGrating:

    def test_grating_subgroup_written(self) -> None:
        root = zarr.group()
        tracks = _make_grating_tracks(n_frames=100, heading_deg=90.0, speed=10.0)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)

        step_base = _grating_step(direction_deg=90.0, grating_speed=10.0)
        sm = [compute_step_base_metrics(tracks, step_base, fps=30.0, moving_threshold=2.0)]

        pf = compute_grating_per_frame(tracks, step_base, 90.0, fps=30.0)
        gpf = compute_grating_per_fish(pf, tracks, step_base, fps=30.0, grating_speed_mm_s=10.0)
        ts = compute_grating_time_series(pf, tracks, step_base, fps=30.0, grating_speed_mm_s=10.0)
        omr = compute_step_omr_metrics(
            tracks, step_base, 90.0, fps=30.0, moving_threshold_mm_s=2.0,
        )
        global_omr = compute_global_omr_metrics([0], [omr])

        gd = {0: GratingStepData(per_frame=pf, per_fish=gpf, time_series=ts, omr=omr)}

        write_stimulus_response_run(
            root, global_metrics=gm, steps=[step_base], step_metrics=sm,
            step_grating_data=gd, global_omr_metrics=global_omr,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="grating_test",
        )

        sr = root["analysis"]["stimulus_response_runs"]["grating_test"]
        s0 = sr["steps"]["step_0"]

        assert "grating" in s0
        assert "per_frame" in s0["grating"]
        assert "per_fish" in s0["grating"]
        assert "time_series" in s0["grating"]

        # Check key arrays exist.
        assert "alignment_cos" in s0["grating"]["per_frame"]
        assert "mean_alignment_cos" in s0["grating"]["per_fish"]
        assert "optomotor_gain" in s0["grating"]["per_fish"]
        assert "bin_center_s" in s0["grating"]["time_series"]
        assert "omr" in s0["grating"]
        assert "omr_path_index" in s0["grating"]["omr"]["per_fish"]
        assert "bout_path_index" in s0["grating"]["omr"]["per_fish"]
        assert "start_position_axis_norm" in s0["grating"]["omr"]["per_fish"]
        assert "first_aligned_bout_latency_s" in s0["grating"]["omr"]["per_fish"]
        assert "windows" in s0["grating"]["omr"]
        assert "fraction_time_correct_side" in s0["grating"]["omr"]["windows"]
        assert "early_windows" in s0["grating"]["omr"]
        assert "omr_path_index" in s0["grating"]["omr"]["early_windows"]
        assert "omr" in sr["global"]
        assert "omr_path_index_weighted_by_path" in sr["global"]["omr"]["per_fish"]
        assert "bout_path_index" in sr["global"]["omr"]["per_fish"]

    def test_non_grating_step_has_no_grating_group(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1, speed=10.0)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        step = ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)
        sm = [compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)]

        write_stimulus_response_run(
            root, global_metrics=gm, steps=[step], step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="no_grating",
        )

        sr = root["analysis"]["stimulus_response_runs"]["no_grating"]
        assert "grating" not in sr["steps"]["step_0"]


# ---------------------------------------------------------------------------
# Frame annotations
# ---------------------------------------------------------------------------


class TestBuildFrameAnnotations:

    def test_basic_annotations(self) -> None:
        steps = [
            ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0),
            ProtocolStep(1, "grating", "MOVING_GRATING", 3, 50, 100, 50 / 30.0),
        ]
        ann = build_frame_annotations(steps, n_frames=100)
        assert ann["step_index"].shape == (100,)
        assert ann["stimulus_mode_id"].shape == (100,)
        # First 50 frames = step 0 (SOLID_BLACK=4).
        assert (ann["step_index"][:50] == 0).all()
        assert (ann["stimulus_mode_id"][:50] == 4).all()
        # Last 50 frames = step 1 (MOVING_GRATING=3).
        assert (ann["step_index"][50:] == 1).all()
        assert (ann["stimulus_mode_id"][50:] == 3).all()

    def test_gap_between_steps(self) -> None:
        steps = [
            ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 10, 40, 30 / 30.0),
            ProtocolStep(1, "grating", "MOVING_GRATING", 3, 60, 90, 30 / 30.0),
        ]
        ann = build_frame_annotations(steps, n_frames=100)
        # Before step 0.
        assert ann["step_index"][5] == -1
        assert ann["stimulus_mode_id"][5] == -1
        # During step 0.
        assert ann["step_index"][20] == 0
        # Between steps.
        assert ann["step_index"][50] == -1
        # During step 1.
        assert ann["step_index"][70] == 1
        # After step 1.
        assert ann["step_index"][95] == -1

    def test_written_to_zarr(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=100, n_fish=1, speed=10.0)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        steps = [
            ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0),
            ProtocolStep(1, "grating", "MOVING_GRATING", 3, 50, 100, 50 / 30.0),
        ]
        sm = [compute_step_base_metrics(tracks, s, fps=30.0, moving_threshold=2.0) for s in steps]
        ann = build_frame_annotations(steps, n_frames=100)

        write_stimulus_response_run(
            root, global_metrics=gm, steps=steps, step_metrics=sm,
            frame_annotations=ann,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="ann_test",
        )

        sr = root["analysis"]["stimulus_response_runs"]["ann_test"]
        assert "frames" in sr
        assert sr["frames"]["step_index"][:].shape == (100,)
        assert sr["frames"]["stimulus_mode_id"][75] == 3  # MOVING_GRATING


# ---------------------------------------------------------------------------
# Integration: multi-stimulus, provenance, detection_source semantics
# ---------------------------------------------------------------------------


class TestMultiDirectionGrating:
    """A recording with two MOVING_GRATING steps at different orientations."""

    def test_different_directions_produce_different_alignment(self) -> None:
        # Fish heading = 90 deg.  Grating at 90 = perfect following.
        # Grating at 270 = perfect opposing.
        tracks = _make_grating_tracks(n_frames=200, heading_deg=90.0, speed=10.0)
        step_a = _grating_step(start=0, end=100, direction_deg=90.0)
        step_b = ProtocolStep(
            index=1, name="grating_270", stimulus_mode="MOVING_GRATING",
            stimulus_mode_id=3, start_frame=100, end_frame=200,
            duration_s=100 / 30.0,
            stimulus_params={"orientation_degrees": 270.0, "grating_speed_mm_s": 10.0},
        )

        pf_a = compute_grating_per_frame(tracks, step_a, 90.0, fps=30.0)
        pf_b = compute_grating_per_frame(tracks, step_b, 270.0, fps=30.0)

        # Step A: fish follows grating (cos ~ +1).
        assert np.mean(pf_a["alignment_cos"][0]) > 0.99
        # Step B: fish opposes grating (cos ~ -1).
        assert np.mean(pf_b["alignment_cos"][0]) < -0.99

    def test_multi_step_written_correctly(self) -> None:
        root = zarr.group()
        tracks = _make_grating_tracks(n_frames=200, heading_deg=90.0, speed=10.0)
        steps = [
            ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0),
            _grating_step(start=50, end=100, direction_deg=90.0),
            ProtocolStep(2, "baseline2", "SOLID_BLACK", 4, 100, 150, 50 / 30.0),
            ProtocolStep(3, "grating_270", "MOVING_GRATING", 3, 150, 200, 50 / 30.0,
                         stimulus_params={"orientation_degrees": 270.0, "grating_speed_mm_s": 10.0}),
        ]
        # Fix step indices.
        steps[1] = ProtocolStep(1, steps[1].name, steps[1].stimulus_mode,
                                steps[1].stimulus_mode_id, steps[1].start_frame,
                                steps[1].end_frame, steps[1].duration_s, steps[1].stimulus_params)

        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        sm = [compute_step_base_metrics(tracks, s, fps=30.0, moving_threshold=2.0) for s in steps]
        ann = build_frame_annotations(steps, n_frames=200)

        # Compute grating data for grating steps only.
        gd = {}
        for s in steps:
            if s.stimulus_mode == "MOVING_GRATING":
                d = resolve_grating_direction(s)
                pf = compute_grating_per_frame(tracks, s, d, fps=30.0)
                gpf = compute_grating_per_fish(pf, tracks, s, fps=30.0, grating_speed_mm_s=10.0)
                ts = compute_grating_time_series(pf, tracks, s, fps=30.0, grating_speed_mm_s=10.0)
                gd[s.index] = GratingStepData(per_frame=pf, per_fish=gpf, time_series=ts)

        write_stimulus_response_run(
            root, global_metrics=gm, steps=steps, step_metrics=sm,
            frame_annotations=ann, step_grating_data=gd,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="multi",
        )

        sr = root["analysis"]["stimulus_response_runs"]["multi"]
        # Baseline steps have no grating subgroup.
        assert "grating" not in sr["steps"]["step_0"]
        assert "grating" not in sr["steps"]["step_2"]
        # Grating steps have grating subgroups.
        assert "grating" in sr["steps"]["step_1"]
        assert "grating" in sr["steps"]["step_3"]
        # Frame annotations span the whole recording.
        assert sr["frames"]["step_index"][:].shape == (200,)
        assert sr["frames"]["stimulus_mode_id"][75] == 3   # MOVING_GRATING
        assert sr["frames"]["stimulus_mode_id"][125] == 4  # SOLID_BLACK


class TestProvenanceLineage:

    def test_upstream_lineage_rejects_unbound_track_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _make_kinematics_zarr(n_frames=2, fish_ids=(0,))
        _patch_synthetic_verified_track_reader(monkeypatch)
        kin = root["analysis"]["track_kinematics_runs"]["offline"]["test_run"]
        stale = SimpleNamespace(
            authority_status="verified_canonical_track_motion_v1",
            motion_manifest_sha256="b" * 64,
            run_path=str(kin.path),
        )

        with pytest.raises(ValueError, match="do not share one freshly verified"):
            stimulus_response_module._snapshot_upstream_lineage(
                root,
                kin,
                [stale],
            )

    def test_upstream_lineage_embedded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _make_kinematics_zarr(n_frames=50, fish_ids=(0,))
        _patch_synthetic_verified_track_reader(monkeypatch)
        kin = root["analysis"]["track_kinematics_runs"]["offline"]["test_run"]
        kin.attrs["source_refs"] = {
            "source_detection_path": "detect_runs/detect_2026-01-01",
            "source_keypoint_path": (
                "refined_keypoints_runs/refined_kp_2026-01-01"
            ),
            "source_crop_path": "crop_runs/crop_2026-01-01",
            "source_tracking_path": "tracking_runs/tracking_2026-01-01",
        }

        tracks, _, _, lineage = load_track_data(root, kinematics_type="offline")
        assert dict(lineage.record) == {
            "schema_id": "palette.stimulus_response.track_motion_lineage",
            "schema_version": 1,
            "source_track_motion_run_ref": (
                "/analysis/track_kinematics_runs/offline/test_run"
            ),
            "source_track_motion_manifest_ref": (
                "/analysis/track_kinematics_runs/offline/test_run"
                "@track_motion_publication_manifest"
            ),
            "source_track_motion_manifest_sha256": "a" * 64,
            "source_track_ids": [0],
            "source_fps": 30.0,
            "source_refs": dict(kin.attrs["source_refs"]),
        }

    def test_archive_identity_from_root_attrs(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        steps = [ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)]
        sm = [compute_step_base_metrics(tracks, steps[0], fps=30.0, moving_threshold=2.0)]

        # Simulate root attrs.
        root.attrs["source_video_path"] = "/data/fish_2026-03-15.avi"
        root.attrs["session_uuid"] = "abc-123"

        write_stimulus_response_run(
            root, global_metrics=gm, steps=steps, step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="prov_test",
        )

        sr = root["analysis"]["stimulus_response_runs"]["prov_test"]
        assert sr.attrs["archive_identity"]["source_video_path"] == "/data/fish_2026-03-15.avi"
        assert sr.attrs["archive_identity"]["session_uuid"] == "abc-123"


class TestMultiFishCoverage:

    def test_different_coverage_per_fish(self) -> None:
        """Two fish with different gap patterns in the same step."""
        tracks = [
            DenseTrack(
                fish_id=0,
                speed_mm=np.full(100, 10.0, dtype=np.float32),
                heading_deg=np.full(100, 90.0, dtype=np.float32),
                positions_mm=np.zeros((100, 2), dtype=np.float32),
                angular_velocity=np.zeros(100, dtype=np.float32),
                time_seconds=np.arange(100, dtype=np.float32) / 30.0,
                valid=np.ones(100, dtype=bool),  # Fish 0: no gaps.
                detection_source=np.zeros(100, dtype=np.int8),
            ),
            DenseTrack(
                fish_id=1,
                speed_mm=np.full(100, 10.0, dtype=np.float32),
                heading_deg=np.full(100, 90.0, dtype=np.float32),
                positions_mm=np.zeros((100, 2), dtype=np.float32),
                angular_velocity=np.zeros(100, dtype=np.float32),
                time_seconds=np.arange(100, dtype=np.float32) / 30.0,
                valid=np.array([i % 2 == 0 for i in range(100)]),  # Fish 1: 50% gaps.
                detection_source=np.where(
                    np.array([i % 2 == 0 for i in range(100)]),
                    np.int8(0), np.int8(-1),
                ),
            ),
        ]
        step = ProtocolStep(0, "test", "SOLID_BLACK", 4, 0, 100, 100 / 30.0)
        result = compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)
        assert result["coverage"][0] == 1.0
        assert abs(result["coverage"][1] - 0.5) < 0.01


class TestDetectionSourceInterpolated:

    def test_interpolated_frames_are_valid_but_marked(self) -> None:
        """detection_source=1 frames should be valid=True but distinguishable."""
        n = 50
        valid = np.ones(n, dtype=bool)
        det_src = np.zeros(n, dtype=np.int8)
        # Mark frames 10-15 as interpolated.
        det_src[10:16] = 1

        tracks = [DenseTrack(
            fish_id=0,
            speed_mm=np.full(n, 10.0, dtype=np.float32),
            heading_deg=np.full(n, 90.0, dtype=np.float32),
            positions_mm=np.zeros((n, 2), dtype=np.float32),
            angular_velocity=np.zeros(n, dtype=np.float32),
            time_seconds=np.arange(n, dtype=np.float32) / 30.0,
            valid=valid,
            detection_source=det_src,
        )]

        step = _grating_step(start=0, end=50, direction_deg=90.0)
        pf = compute_grating_per_frame(tracks, step, 90.0, fps=30.0)

        # All frames valid (interpolated are still detections).
        assert pf["valid"][0].all()
        # But detection_source distinguishes real (0) from interpolated (1).
        assert pf["detection_source"][0, 5] == 0
        assert pf["detection_source"][0, 12] == 1
        # Alignment is computed for both (fish heading=90, grating=90 -> cos=1).
        assert np.allclose(pf["alignment_cos"][0, 5], 1.0, atol=0.01)
        assert np.allclose(pf["alignment_cos"][0, 12], 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Concentric grating
# ---------------------------------------------------------------------------


def _concentric_step(
    start: int = 0,
    end: int = 100,
    center_threshold: float = 2.0,
) -> ProtocolStep:
    return ProtocolStep(
        index=0, name="concentric_test",
        stimulus_mode="CONCENTRIC_GRATING", stimulus_mode_id=6,
        start_frame=start, end_frame=end, duration_s=(end - start) / 30.0,
        stimulus_params={"center_threshold_mm": center_threshold},
    )


def _make_centering_tracks(
    n_frames: int = 100,
    center_mm: Tuple[float, float] = (10.0, 10.0),
    start_pos_mm: Tuple[float, float] = (20.0, 10.0),
    speed: float = 3.0,
) -> List[DenseTrack]:
    """Fish moving linearly toward the center."""
    cx, cy = center_mm
    sx, sy = start_pos_mm
    dx = cx - sx
    dy = cy - sy
    dist = np.sqrt(dx**2 + dy**2)
    heading_deg = float(np.rad2deg(np.arctan2(dy, dx)))

    pos = np.zeros((n_frames, 2), dtype=np.float32)
    for f in range(n_frames):
        frac = min(1.0, (f * speed / 30.0) / dist) if dist > 0 else 0.0
        pos[f, 0] = sx + dx * frac
        pos[f, 1] = sy + dy * frac

    return [DenseTrack(
        fish_id=0,
        speed_mm=np.full(n_frames, speed, dtype=np.float32),
        heading_deg=np.full(n_frames, heading_deg, dtype=np.float32),
        positions_mm=pos,
        angular_velocity=np.zeros(n_frames, dtype=np.float32),
        time_seconds=np.arange(n_frames, dtype=np.float32) / 30.0,
        valid=np.ones(n_frames, dtype=bool),
        detection_source=np.zeros(n_frames, dtype=np.int8),
    )]


def _make_position_track(
    positions_mm: np.ndarray,
    *,
    fish_id: int = 0,
    fps: float = 30.0,
) -> List[DenseTrack]:
    positions = np.asarray(positions_mm, dtype=np.float32)
    n_frames = positions.shape[0]
    deltas = np.vstack([positions[1] - positions[0], np.diff(positions, axis=0)])
    speed = (np.linalg.norm(deltas, axis=1) * fps).astype(np.float32)
    heading = np.rad2deg(np.arctan2(deltas[:, 1], deltas[:, 0])).astype(np.float32)
    heading[~np.isfinite(heading)] = 0.0
    return [DenseTrack(
        fish_id=fish_id,
        speed_mm=speed,
        heading_deg=heading,
        positions_mm=positions,
        angular_velocity=np.zeros(n_frames, dtype=np.float32),
        time_seconds=np.arange(n_frames, dtype=np.float32) / fps,
        valid=np.ones(n_frames, dtype=bool),
        detection_source=np.zeros(n_frames, dtype=np.int8),
    )]


def _radial_omr_step(
    *,
    sign: int = 1,
    start: int = 0,
    end: int = 30,
) -> ProtocolStep:
    return ProtocolStep(
        index=0,
        name="concentric_radial_omr",
        stimulus_mode="CONCENTRIC_GRATING",
        stimulus_mode_id=6,
        start_frame=start,
        end_frame=end,
        duration_s=(end - start) / 30.0,
        stimulus_params={
            "concentric_grating": {
                "radial_sign_authored": sign,
                "radial_polarity_authored": "expanding" if sign > 0 else "contracting",
                "radial_polarity_source": "test",
                "radial_polarity_validated": False,
            }
        },
    )


class TestComputeConcentricPerFrame:

    def test_distance_decreases_toward_center(self) -> None:
        center = (10.0, 10.0)
        tracks = _make_centering_tracks(center_mm=center, start_pos_mm=(20.0, 10.0))
        step = _concentric_step()
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        dist = pf["distance_to_center_mm"][0]
        # Distance should decrease over time (fish approaches center).
        assert dist[0] > dist[-1]
        # First frame should be ~10mm from center.
        assert abs(dist[0] - 10.0) < 0.5

    def test_radial_speed_positive_toward_center(self) -> None:
        center = (10.0, 10.0)
        tracks = _make_centering_tracks(center_mm=center, start_pos_mm=(20.0, 10.0))
        step = _concentric_step()
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        # Fish heading toward center → radial_speed should be positive.
        valid = pf["valid"][0]
        rs = pf["radial_speed_mm_s"][0][valid]
        assert np.mean(rs) > 0

    def test_valid_and_detection_source(self) -> None:
        center = (10.0, 10.0)
        tracks = _make_centering_tracks(center_mm=center)
        step = _concentric_step()
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        assert pf["valid"][0].all()
        assert (pf["detection_source"][0] == 0).all()

    def test_gap_frames_zeroed(self) -> None:
        center = (10.0, 10.0)
        tracks = _make_dense_tracks(n_frames=100, n_fish=1, speed=5.0, gap_frames=(10, 20))
        step = _concentric_step()
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        assert not pf["valid"][0, 10]
        assert pf["distance_to_center_mm"][0, 10] == 0.0


class TestComputeConcentricPerFish:

    def test_centering_fish(self) -> None:
        center = (10.0, 10.0)
        tracks = _make_centering_tracks(center_mm=center, start_pos_mm=(20.0, 10.0), speed=3.0)
        step = _concentric_step()
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        cpf = compute_concentric_per_fish(pf, tracks, step, fps=30.0, center_threshold_mm=2.0)
        # Fish started far, moved toward center.
        assert cpf["initial_distance_to_center_mm"][0] > cpf["final_distance_to_center_mm"][0]
        # Net radial displacement should be negative (moved closer).
        assert cpf["net_radial_displacement_mm"][0] < 0
        # Fraction approaching should be high.
        assert cpf["fraction_approaching"][0] > 0.5
        # Mean radial heading cosine should be positive (heading toward center).
        assert cpf["mean_radial_heading_cos"][0] > 0

    def test_time_to_center(self) -> None:
        center = (10.0, 10.0)
        # Fast fish that reaches center quickly.
        tracks = _make_centering_tracks(center_mm=center, start_pos_mm=(12.0, 10.0), speed=10.0)
        step = _concentric_step(end=100)
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        cpf = compute_concentric_per_fish(pf, tracks, step, fps=30.0, center_threshold_mm=2.0)
        # Fish starts 2mm away at 10mm/s → should reach center quickly.
        assert not np.isnan(cpf["time_to_center_s"][0])
        assert cpf["time_to_center_s"][0] < 1.0

    def test_fish_never_reaches_center(self) -> None:
        center = (10.0, 10.0)
        # Stationary fish far from center.
        tracks = [DenseTrack(
            fish_id=0,
            speed_mm=np.zeros(50, dtype=np.float32),
            heading_deg=np.full(50, 0.0, dtype=np.float32),
            positions_mm=np.full((50, 2), [20.0, 10.0], dtype=np.float32),
            angular_velocity=np.zeros(50, dtype=np.float32),
            time_seconds=np.arange(50, dtype=np.float32) / 30.0,
            valid=np.ones(50, dtype=bool),
            detection_source=np.zeros(50, dtype=np.int8),
        )]
        step = _concentric_step(end=50)
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        cpf = compute_concentric_per_fish(pf, tracks, step, fps=30.0, center_threshold_mm=2.0)
        assert np.isnan(cpf["time_to_center_s"][0])
        assert cpf["fraction_near_center"][0] == 0.0


class TestComputeConcentricTimeSeries:

    def test_bin_count(self) -> None:
        center = (10.0, 10.0)
        tracks = _make_centering_tracks(n_frames=90, center_mm=center)
        step = _concentric_step(end=90)
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        ts = compute_concentric_time_series(pf, tracks, step, fps=30.0, bin_size_s=1.0)
        assert ts["bin_center_s"].shape[0] == 3  # 90 frames / 30fps = 3 bins

    def test_distance_decreases_in_bins(self) -> None:
        center = (10.0, 10.0)
        tracks = _make_centering_tracks(n_frames=90, center_mm=center, start_pos_mm=(20.0, 10.0))
        step = _concentric_step(end=90)
        pf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        ts = compute_concentric_time_series(pf, tracks, step, fps=30.0, bin_size_s=1.0)
        # Distance should decrease across bins.
        assert ts["distance_to_center_mm"][0, 0] > ts["distance_to_center_mm"][0, -1]


class TestConcentricRadialOMR:

    def test_resolves_authored_polarity(self) -> None:
        expanding = _radial_omr_step(sign=1)
        contracting = _radial_omr_step(sign=-1)

        assert resolve_concentric_radial_polarity(expanding)["stimulus_radial_sign"] == 1
        assert resolve_concentric_radial_polarity(contracting)["stimulus_radial_sign"] == -1

    def test_expanding_outward_motion_is_aligned(self) -> None:
        center = (0.0, 0.0)
        x = np.linspace(1.0, 4.0, 30, dtype=np.float32)
        tracks = _make_position_track(np.column_stack([x, np.zeros_like(x)]))
        step = _radial_omr_step(sign=1, end=30)
        bouts = {0: [BoutEntry(0, 1, 0, 29, 29 / 30.0, 3.0, 3.0)]}

        omr = compute_step_concentric_radial_omr_metrics(
            tracks,
            step,
            center,
            fps=30.0,
            moving_threshold_mm_s=0.1,
            bouts_by_fish=bouts,
            arena_radius_mm=5.0,
        )

        assert omr.attrs["stimulus_radial_sign"] == 1
        assert omr.per_fish["omr_path_index"][0] > 0.99
        assert omr.per_fish["bout_count_correct"][0] == 1
        assert omr.per_bout["omr_label"][0] == 1
        assert omr.per_frame["stimulus_aligned_radial_speed_mm_s"][0, 1:].mean() > 0

    def test_contracting_inward_motion_is_aligned(self) -> None:
        center = (0.0, 0.0)
        x = np.linspace(4.0, 1.0, 30, dtype=np.float32)
        tracks = _make_position_track(np.column_stack([x, np.zeros_like(x)]))
        step = _radial_omr_step(sign=-1, end=30)

        omr = compute_step_concentric_radial_omr_metrics(
            tracks,
            step,
            center,
            fps=30.0,
            moving_threshold_mm_s=0.1,
            bouts_by_fish=None,
            arena_radius_mm=5.0,
        )

        assert omr.attrs["stimulus_radial_sign"] == -1
        assert omr.per_fish["omr_path_index"][0] > 0.99
        assert omr.per_fish["radial_path_index"][0] < -0.99

    def test_tangential_motion_has_near_zero_radial_index(self) -> None:
        center = (0.0, 0.0)
        theta = np.linspace(0.0, np.pi / 2.0, 60, dtype=np.float32)
        radius = 3.0
        positions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        tracks = _make_position_track(positions)
        step = _radial_omr_step(sign=1, end=60)

        omr = compute_step_concentric_radial_omr_metrics(
            tracks,
            step,
            center,
            fps=30.0,
            moving_threshold_mm_s=0.1,
            arena_radius_mm=5.0,
        )

        assert abs(float(omr.per_fish["omr_path_index"][0])) < 0.05
        assert omr.per_fish["tangential_bias_index"][0] > 0.9

    def test_center_singularity_marks_quality_flag(self) -> None:
        center = (0.0, 0.0)
        positions = np.zeros((30, 2), dtype=np.float32)
        tracks = _make_position_track(positions)
        step = _radial_omr_step(sign=1, end=30)

        omr = compute_step_concentric_radial_omr_metrics(
            tracks,
            step,
            center,
            fps=30.0,
            moving_threshold_mm_s=0.1,
            radial_singularity_epsilon_mm=0.5,
        )

        assert omr.per_fish["quality_flag"][0] == 1
        assert not omr.per_frame["valid_radial_basis"][0].any()

    def test_attrs_capture_radial_window_and_stimulus_provenance(self) -> None:
        center = (0.0, 0.0)
        x = np.linspace(1.0, 4.0, 30, dtype=np.float32)
        tracks = _make_position_track(np.column_stack([x, np.zeros_like(x)]))
        step = _radial_omr_step(sign=1, end=30)
        step.stimulus_params["concentric_grating"].update(
            {
                "speed_pps": 30.0,
                "speed_mm_s": 3.0,
                "speed_mm_per_sec": 3.0,
                "spatial_freq_rpp": 0.02,
                "spatial_freq_cycles_per_mm": 0.2,
                "temporal_frequency_hz": 0.6,
                "actual_rendered_temporal_frequency_hz": 0.6,
                "stimulus_role": "centering_utility",
                "target_radius_min_mm": 8.0,
                "target_radius_max_mm": 14.0,
            }
        )

        omr = compute_step_concentric_radial_omr_metrics(
            tracks,
            step,
            center,
            fps=30.0,
            moving_threshold_mm_s=0.1,
            arena_radius_mm=5.0,
            window_lengths_s=(3.0, 7.0),
            early_window_lengths_s=(1.0, 2.0),
            radial_singularity_epsilon_mm=0.25,
        )

        attrs = omr.attrs
        assert attrs["stimulus_radial_polarity"] == "expanding"
        assert attrs["stimulus_radial_sign"] == 1
        assert attrs["stimulus_radial_polarity_source"] == "test"
        assert attrs["window_lengths_s"] == [3.0, 7.0]
        assert attrs["early_response_window_lengths_s"] == [1.0, 2.0]
        assert attrs["radial_singularity_epsilon_mm"] == 0.25
        assert attrs["concentric_grating_role"] == "centering_utility"
        assert attrs["speed_pps"] == 30.0
        assert attrs["speed_mm_s"] == 3.0
        assert attrs["spatial_freq_cycles_per_mm"] == 0.2
        assert attrs["target_radius_min_mm"] == 8.0
        assert attrs["target_radius_max_mm"] == 14.0


class TestWriteWithConcentric:

    def test_concentric_subgroup_written(self) -> None:
        root = zarr.group()
        center = (10.0, 10.0)
        tracks = _make_centering_tracks(center_mm=center)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        step = _concentric_step()
        sm = [compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)]

        cpf = compute_concentric_per_frame(tracks, step, center, fps=30.0)
        cpfish = compute_concentric_per_fish(cpf, tracks, step, fps=30.0)
        cts = compute_concentric_time_series(cpf, tracks, step, fps=30.0)
        radial_omr = compute_step_concentric_radial_omr_metrics(
            tracks,
            step,
            center,
            fps=30.0,
            moving_threshold_mm_s=0.1,
            arena_radius_mm=20.0,
        )
        cd = {0: ConcentricStepData(
            per_frame=cpf,
            per_fish=cpfish,
            time_series=cts,
            radial_omr=radial_omr,
        )}

        write_stimulus_response_run(
            root, global_metrics=gm, steps=[step], step_metrics=sm,
            step_concentric_data=cd,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="conc_test",
        )

        sr = root["analysis"]["stimulus_response_runs"]["conc_test"]
        s0 = sr["steps"]["step_0"]
        assert "concentric_grating" in s0
        assert "per_frame" in s0["concentric_grating"]
        assert "per_fish" in s0["concentric_grating"]
        assert "time_series" in s0["concentric_grating"]
        assert "distance_to_center_mm" in s0["concentric_grating"]["per_frame"]
        assert "fraction_approaching" in s0["concentric_grating"]["per_fish"]
        assert "radial_omr" in s0["concentric_grating"]
        assert "omr_path_index" in s0["concentric_grating"]["radial_omr"]["per_fish"]
        assert (
            s0["concentric_grating"]["radial_omr"].attrs["method_version"]
            == "stimulus_response_concentric_radial_omr_v1"
        )

    def test_non_concentric_step_has_no_concentric_group(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        step = ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)
        sm = [compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)]

        write_stimulus_response_run(
            root, global_metrics=gm, steps=[step], step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="no_conc",
        )
        assert "concentric_grating" not in root["analysis"]["stimulus_response_runs"]["no_conc"]["steps"]["step_0"]


# ---------------------------------------------------------------------------
# Loom helpers
# ---------------------------------------------------------------------------


def _make_loom_step(
    start_frame: int = 0,
    end_frame: int = 600,
    step_index: int = 0,
    loom_duration_sec: float = 2.0,
    start_radius_px: float = 2.0,
    end_radius_px: float = 50.0,
    auto_repeat_loom: bool = True,
    inter_loom_interval_sec: float = 5.0,
    target_side: int = 0,
    fps: float = 60.0,
) -> ProtocolStep:
    """Create a LOOMING_DOT ProtocolStep for testing."""
    return ProtocolStep(
        index=step_index,
        name="loom",
        stimulus_mode="LOOMING_DOT",
        stimulus_mode_id=7,
        start_frame=start_frame,
        end_frame=end_frame,
        duration_s=(end_frame - start_frame) / fps,
        stimulus_params={
            "start_radius_px": start_radius_px,
            "end_radius_px": end_radius_px,
            "loom_duration_sec": loom_duration_sec,
            "auto_repeat_loom": auto_repeat_loom,
            "inter_loom_interval_sec": inter_loom_interval_sec,
            "target_side": target_side,
        },
    )


def _make_loom_tracks(
    n_frames: int = 600,
    n_fish: int = 1,
    escape_frame: int = 30,
    escape_speed: float = 50.0,
    base_speed: float = 2.0,
    center_mm: Tuple[float, float] = (10.0, 10.0),
    fps: float = 60.0,
) -> List[DenseTrack]:
    """Tracks with a speed burst at escape_frame (relative to step start)."""
    tracks = []
    for fid in range(n_fish):
        speed_mm = np.full(n_frames, base_speed, dtype=np.float32)
        # Add escape burst.
        burst_start = escape_frame
        burst_end = min(escape_frame + int(0.5 * fps), n_frames)
        speed_mm[burst_start:burst_end] = escape_speed

        pos = np.zeros((n_frames, 2), dtype=np.float32)
        # Start near center, move away at escape.
        pos[:, 0] = center_mm[0] + 1.0
        pos[:, 1] = center_mm[1]

        tracks.append(DenseTrack(
            fish_id=fid,
            speed_mm=speed_mm,
            heading_deg=np.full(n_frames, 180.0, dtype=np.float32),  # heading away
            positions_mm=pos,
            angular_velocity=np.zeros(n_frames, dtype=np.float32),
            time_seconds=np.arange(n_frames, dtype=np.float32) / fps,
            valid=np.ones(n_frames, dtype=bool),
            detection_source=np.zeros(n_frames, dtype=np.int8),
        ))
    return tracks


# ---------------------------------------------------------------------------
# Loom trial reconstruction
# ---------------------------------------------------------------------------


class TestReconstructLoomTrials:
    def test_single_trial(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=600, loom_duration_sec=2.0)
        onset_frames = [0]
        trials = reconstruct_loom_trials(step, onset_frames, fps=60.0)
        assert len(trials) == 1
        assert trials[0].onset_frame == 0
        assert trials[0].offset_frame == 120  # 2.0s * 60fps
        assert trials[0].duration_s == pytest.approx(2.0)

    def test_multiple_trials(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=600, loom_duration_sec=2.0)
        # Three loom onsets: frame 0, 300, 500.
        onset_frames = [0, 300, 500]
        trials = reconstruct_loom_trials(step, onset_frames, fps=60.0)
        assert len(trials) == 3
        assert trials[1].onset_frame == 300
        assert trials[1].offset_frame == 420
        # Last trial should be clamped to step end.
        assert trials[2].offset_frame == 600

    def test_onset_past_step_end_ignored(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=200)
        onset_frames = [0, 100, 300]  # 300 is past step end
        trials = reconstruct_loom_trials(step, onset_frames, fps=60.0)
        assert len(trials) == 2

    def test_zero_duration_returns_empty(self) -> None:
        step = _make_loom_step(loom_duration_sec=0.0)
        trials = reconstruct_loom_trials(step, [0], fps=60.0)
        assert len(trials) == 0


# ---------------------------------------------------------------------------
# Loom center resolution
# ---------------------------------------------------------------------------


class TestResolveLoomCenter:
    def test_precomputed_mm(self) -> None:
        step = _make_loom_step()
        step.stimulus_params["loom_center_x_mm"] = 5.0
        step.stimulus_params["loom_center_y_mm"] = 7.0
        cal = {"homography": None, "pixel_to_mm": None,
               "arena_center_px": None, "pixels_per_mm_projector": None,
               "z_eff_mm": None}
        center = resolve_loom_center_mm(step, cal)
        assert center == pytest.approx((5.0, 7.0))

    def test_arena_center_fallback(self) -> None:
        step = _make_loom_step()
        cal = {"homography": None, "pixel_to_mm": 0.5,
               "arena_center_px": (20.0, 20.0),
               "pixels_per_mm_projector": None, "z_eff_mm": None}
        center = resolve_loom_center_mm(step, cal)
        assert center == pytest.approx((10.0, 10.0))

    def test_no_calibration_returns_none(self) -> None:
        step = _make_loom_step()
        cal = {"homography": None, "pixel_to_mm": None,
               "arena_center_px": None, "pixels_per_mm_projector": None,
               "z_eff_mm": None}
        assert resolve_loom_center_mm(step, cal) is None


# ---------------------------------------------------------------------------
# Loom per-frame
# ---------------------------------------------------------------------------


class TestComputeLoomPerFrame:
    def test_loom_active_mask(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=300, loom_duration_sec=1.0)
        tracks = _make_loom_tracks(n_frames=300)
        trials = [LoomTrial(0, 0, 60, 1.0), LoomTrial(1, 150, 210, 1.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        assert pf["loom_active"][0]
        assert pf["loom_active"][59]
        assert not pf["loom_active"][60]  # offset is exclusive
        assert pf["loom_active"][150]
        assert pf["trial_index"][30] == 0
        assert pf["trial_index"][160] == 1
        assert pf["trial_index"][100] == -1

    def test_radius_reconstruction(self) -> None:
        step = _make_loom_step(
            start_frame=0, end_frame=200,
            start_radius_px=0.0, end_radius_px=60.0, loom_duration_sec=1.0,
        )
        tracks = _make_loom_tracks(n_frames=200)
        trials = [LoomTrial(0, 0, 60, 1.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        # Frame 0: should be near start_radius (0.0).
        assert pf["loom_radius_px"][0] == pytest.approx(0.0, abs=1.0)
        # Frame 59 (last frame before offset): should be near end_radius.
        assert pf["loom_radius_px"][59] > 50.0
        # Outside loom: zero.
        assert pf["loom_radius_px"][70] == 0.0

    def test_visual_angle_nonzero(self) -> None:
        step = _make_loom_step(
            start_frame=0, end_frame=200,
            start_radius_px=5.0, end_radius_px=50.0, loom_duration_sec=1.0,
        )
        tracks = _make_loom_tracks(n_frames=200)
        trials = [LoomTrial(0, 0, 60, 1.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        # During loom, visual angle should be positive.
        assert pf["visual_angle_deg"][30] > 0
        # Outside loom, visual angle should be 0 (radius is 0).
        assert pf["visual_angle_deg"][100] == 0.0

    def test_visual_angle_zero_without_calibration(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=200)
        tracks = _make_loom_tracks(n_frames=200)
        trials = [LoomTrial(0, 0, 60, 1.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=None, z_eff_mm=None,
        )
        assert np.all(pf["visual_angle_deg"] == 0.0)

    def test_distance_to_loom(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=100)
        tracks = _make_loom_tracks(n_frames=100, center_mm=(10.0, 10.0))
        trials = [LoomTrial(0, 0, 60, 1.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        # Fish is at (11, 10), center at (10, 10) → distance = 1.0mm.
        np.testing.assert_allclose(pf["distance_to_loom_mm"][0], 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Loom per-trial per-fish (escape detection)
# ---------------------------------------------------------------------------


class TestComputeLoomPerTrialPerFish:
    def test_escape_detected(self) -> None:
        """Fish with speed burst should be detected as escaped."""
        step = _make_loom_step(start_frame=0, end_frame=300)
        tracks = _make_loom_tracks(n_frames=300, escape_frame=30, escape_speed=50.0)
        trials = [LoomTrial(0, 0, 120, 2.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        ptpf = compute_loom_per_trial_per_fish(
            tracks, step, trials, (10.0, 10.0), pf, fps=60.0,
            escape_speed_threshold_mm_s=30.0, escape_window_s=5.0,
        )
        assert ptpf["escaped"][0, 0]
        assert ptpf["escape_latency_frames"][0, 0] == 30
        assert ptpf["escape_latency_s"][0, 0] == pytest.approx(0.5)

    def test_no_escape_below_threshold(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=300)
        tracks = _make_loom_tracks(n_frames=300, escape_speed=10.0)  # below 30 mm/s
        trials = [LoomTrial(0, 0, 120, 2.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        ptpf = compute_loom_per_trial_per_fish(
            tracks, step, trials, (10.0, 10.0), pf, fps=60.0,
            escape_speed_threshold_mm_s=30.0,
        )
        assert not ptpf["escaped"][0, 0]
        assert np.isnan(ptpf["escape_latency_s"][0, 0])

    def test_configurable_threshold(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=300)
        tracks = _make_loom_tracks(n_frames=300, escape_speed=20.0, escape_frame=30)
        trials = [LoomTrial(0, 0, 120, 2.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        # Should not escape at default threshold.
        ptpf_30 = compute_loom_per_trial_per_fish(
            tracks, step, trials, (10.0, 10.0), pf, fps=60.0,
            escape_speed_threshold_mm_s=30.0,
        )
        assert not ptpf_30["escaped"][0, 0]

        # Should escape at lower threshold.
        ptpf_15 = compute_loom_per_trial_per_fish(
            tracks, step, trials, (10.0, 10.0), pf, fps=60.0,
            escape_speed_threshold_mm_s=15.0,
        )
        assert ptpf_15["escaped"][0, 0]

    def test_peak_escape_speed(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=300)
        tracks = _make_loom_tracks(n_frames=300, escape_frame=30, escape_speed=80.0)
        trials = [LoomTrial(0, 0, 120, 2.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        ptpf = compute_loom_per_trial_per_fish(
            tracks, step, trials, (10.0, 10.0), pf, fps=60.0,
            escape_speed_threshold_mm_s=30.0,
        )
        assert ptpf["peak_escape_speed_mm_s"][0, 0] == pytest.approx(80.0)

    def test_multi_trial(self) -> None:
        """Two trials, fish escapes in first but not second."""
        step = _make_loom_step(start_frame=0, end_frame=600)
        # Create track: fast burst at frame 30, but nothing at frame 330+.
        tracks = _make_loom_tracks(n_frames=600, escape_frame=30, escape_speed=50.0)
        # Zero out the escape speed so second trial has no escape.
        tracks[0].speed_mm[300:] = 2.0

        trials = [LoomTrial(0, 0, 120, 2.0), LoomTrial(1, 300, 420, 2.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        ptpf = compute_loom_per_trial_per_fish(
            tracks, step, trials, (10.0, 10.0), pf, fps=60.0,
            escape_speed_threshold_mm_s=30.0,
        )
        assert ptpf["escaped"][0, 0]
        assert not ptpf["escaped"][0, 1]


# ---------------------------------------------------------------------------
# Loom per-fish summary
# ---------------------------------------------------------------------------


class TestComputeLoomPerFish:
    def test_escape_probability(self) -> None:
        ptpf = {
            "escaped": np.array([[True, True, False]]),
            "escape_latency_s": np.array([[0.5, 1.0, np.nan]], dtype=np.float32),
            "peak_escape_speed_mm_s": np.array([[80.0, 60.0, 0.0]], dtype=np.float32),
            "distance_at_escape_mm": np.array([[5.0, 4.0, np.nan]], dtype=np.float32),
            "visual_angle_at_escape_deg": np.array([[30.0, 40.0, np.nan]], dtype=np.float32),
            "escape_heading_deg": np.array([[180.0, 170.0, np.nan]], dtype=np.float32),
            "escape_latency_frames": np.array([[30, 60, -1]], dtype=np.int32),
        }
        pf = compute_loom_per_fish(ptpf, n_trials=3)
        assert pf["n_escape_responses"][0] == 2
        assert pf["escape_probability"][0] == pytest.approx(2 / 3)
        assert pf["mean_escape_latency_s"][0] == pytest.approx(0.75)
        assert pf["median_escape_latency_s"][0] == pytest.approx(0.75)
        assert pf["mean_peak_escape_speed_mm_s"][0] == pytest.approx(70.0)

    def test_no_escapes(self) -> None:
        ptpf = {
            "escaped": np.array([[False, False]]),
            "escape_latency_s": np.array([[np.nan, np.nan]], dtype=np.float32),
            "peak_escape_speed_mm_s": np.zeros((1, 2), dtype=np.float32),
            "distance_at_escape_mm": np.full((1, 2), np.nan, dtype=np.float32),
            "visual_angle_at_escape_deg": np.full((1, 2), np.nan, dtype=np.float32),
            "escape_heading_deg": np.full((1, 2), np.nan, dtype=np.float32),
            "escape_latency_frames": np.full((1, 2), -1, dtype=np.int32),
        }
        pf = compute_loom_per_fish(ptpf, n_trials=2)
        assert pf["n_escape_responses"][0] == 0
        assert pf["escape_probability"][0] == 0.0
        assert np.isnan(pf["mean_escape_latency_s"][0])

    def test_habituation_positive_slope(self) -> None:
        """Latency increases across trials → positive habituation_index."""
        ptpf = {
            "escaped": np.array([[True, True, True, True]]),
            "escape_latency_s": np.array([[0.1, 0.3, 0.5, 0.7]], dtype=np.float32),
            "peak_escape_speed_mm_s": np.ones((1, 4), dtype=np.float32),
            "distance_at_escape_mm": np.ones((1, 4), dtype=np.float32),
            "visual_angle_at_escape_deg": np.ones((1, 4), dtype=np.float32),
            "escape_heading_deg": np.ones((1, 4), dtype=np.float32),
            "escape_latency_frames": np.array([[6, 18, 30, 42]], dtype=np.int32),
        }
        pf = compute_loom_per_fish(ptpf, n_trials=4)
        assert pf["habituation_index"][0] > 0

    def test_habituation_needs_two_escapes(self) -> None:
        ptpf = {
            "escaped": np.array([[True, False, False]]),
            "escape_latency_s": np.array([[0.5, np.nan, np.nan]], dtype=np.float32),
            "peak_escape_speed_mm_s": np.ones((1, 3), dtype=np.float32),
            "distance_at_escape_mm": np.ones((1, 3), dtype=np.float32),
            "visual_angle_at_escape_deg": np.ones((1, 3), dtype=np.float32),
            "escape_heading_deg": np.ones((1, 3), dtype=np.float32),
            "escape_latency_frames": np.array([[30, -1, -1]], dtype=np.int32),
        }
        pf = compute_loom_per_fish(ptpf, n_trials=3)
        assert np.isnan(pf["habituation_index"][0])


# ---------------------------------------------------------------------------
# Loom time series
# ---------------------------------------------------------------------------


class TestComputeLoomTimeSeries:
    def test_bin_count(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=600)
        tracks = _make_loom_tracks(n_frames=600)
        trials = [LoomTrial(0, 60, 180, 2.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        ts = compute_loom_time_series(
            tracks, step, trials, pf, fps=60.0,
            pre_onset_s=1.0, post_onset_s=5.0, bin_size_s=0.1,
        )
        # (1 + 5) / 0.1 = 60 bins.
        assert ts["trial_time_s"].shape[0] == 60
        assert ts["mean_speed_mm_s"].shape == (1, 60)
        # First bin center should be near -0.95 (= -1.0 + 0.05).
        assert ts["trial_time_s"][0] == pytest.approx(-0.95)

    def test_speed_reflects_escape(self) -> None:
        step = _make_loom_step(start_frame=0, end_frame=600)
        tracks = _make_loom_tracks(
            n_frames=600, escape_frame=70, escape_speed=50.0, base_speed=2.0,
        )
        trials = [LoomTrial(0, 60, 180, 2.0)]
        pf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        ts = compute_loom_time_series(
            tracks, step, trials, pf, fps=60.0,
            pre_onset_s=1.0, post_onset_s=5.0, bin_size_s=0.5,
        )
        # Pre-onset bins should have low speed, post-escape bins should have higher.
        pre_bin = ts["mean_speed_mm_s"][0, 0]  # ~-0.75s
        # Escape at frame 70, onset at frame 60 → 10 frames = ~0.17s after onset.
        # That falls in bin around 0.0-0.5s.
        post_bin_idx = 2  # covers 0.0 to 0.5s (bins: -1 to -0.5, -0.5 to 0, 0 to 0.5)
        post_bin = ts["mean_speed_mm_s"][0, post_bin_idx]
        assert post_bin > pre_bin


# ---------------------------------------------------------------------------
# Loom write integration
# ---------------------------------------------------------------------------


class TestWriteWithLoom:
    def test_loom_subgroup_written(self) -> None:
        root = zarr.group()
        tracks = _make_loom_tracks(n_frames=300, escape_frame=30, escape_speed=50.0)
        gm = compute_global_metrics(tracks, fps=60.0, moving_threshold=2.0)
        step = _make_loom_step(start_frame=0, end_frame=300)
        sm = [compute_step_base_metrics(tracks, step, fps=60.0, moving_threshold=2.0)]

        trials = [LoomTrial(0, 0, 120, 2.0)]
        lpf = compute_loom_per_frame(
            tracks, step, trials, (10.0, 10.0), fps=60.0,
            pixels_per_mm_projector=0.44, z_eff_mm=10.4,
        )
        ltpf = compute_loom_per_trial_per_fish(
            tracks, step, trials, (10.0, 10.0), lpf, fps=60.0,
        )
        lpfish = compute_loom_per_fish(ltpf, 1)
        lts = compute_loom_time_series(tracks, step, trials, lpf, fps=60.0)
        ld = LoomStepData(
            trials=trials, per_frame=lpf,
            per_trial_per_fish=ltpf, per_fish=lpfish, time_series=lts,
        )

        write_stimulus_response_run(
            root, global_metrics=gm, steps=[step], step_metrics=sm,
            step_loom_data={0: ld},
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s",
            parameters={"escape_speed_threshold_mm_s": 30.0, "escape_window_s": 5.0},
            run_name="loom_test",
        )

        sr = root["analysis"]["stimulus_response_runs"]["loom_test"]
        s0 = sr["steps"]["step_0"]
        assert "looming" in s0
        assert "trials" in s0["looming"]
        assert "per_frame" in s0["looming"]
        assert "per_trial_per_fish" in s0["looming"]
        assert "per_fish" in s0["looming"]
        assert "time_series" in s0["looming"]
        # Check trial arrays.
        assert s0["looming"]["trials"]["onset_frame"][:].tolist() == [0]
        # Check escape detected.
        assert s0["looming"]["per_trial_per_fish"]["escaped"][:][0, 0]
        # Check per_fish summary.
        assert s0["looming"]["per_fish"]["n_escape_responses"][:][0] == 1

    def test_non_loom_step_has_no_loom_group(self) -> None:
        root = zarr.group()
        tracks = _make_dense_tracks(n_frames=50, n_fish=1)
        gm = compute_global_metrics(tracks, fps=30.0, moving_threshold=2.0)
        step = ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 50, 50 / 30.0)
        sm = [compute_step_base_metrics(tracks, step, fps=30.0, moving_threshold=2.0)]

        write_stimulus_response_run(
            root, global_metrics=gm, steps=[step], step_metrics=sm,
            source_kinematics_run="k", source_kinematics_type="offline",
            source_stimulus_run="s", parameters={}, run_name="no_loom",
        )
        assert "looming" not in root["analysis"]["stimulus_response_runs"]["no_loom"]["steps"]["step_0"]
