from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis.detect_bouts_multi_level import _load_track_kinematics_track_speeds
from fisheye.analysis.track_kinematics_io import (
    load_legacy_track_kinematics_track_for_inspection,
    load_track_kinematics_track,
    resolve_track_kinematics_run,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256


def _array(group: zarr.Group, name: str, values) -> None:
    group.create_array(name, data=np.asarray(values), overwrite=True)


def _make_track_kinematics_archive(path: Path | None = None) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w") if path is not None else zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("track_kinematics_runs")
    offline = parent.create_group("offline")
    parent.attrs.update(
        {
            "latest": "offline/tk_test",
            "latest_complete": "offline/tk_test",
            "latest_offline": "tk_test",
        }
    )
    offline.attrs["latest"] = "tk_test"
    run = offline.create_group("tk_test")
    run.attrs.update(
        {
            "fps": 60.0,
            "pixel_to_mm": 0.1,
            "created_at_utc": "2026-05-10T00:00:00Z",
            "provenance": {"stage": "track_kinematics", "version": "test", "git": {"commit": "abc"}},
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    _array(run, "track_ids", [0])
    tracks = run.create_group("tracks")
    track = tracks.create_group("id_0")
    track.attrs["track_id"] = 0

    _array(track, "frame_indices", [10, 11, 12])
    _array(track, "source_acquisition_frame_index", [10, 11, 12])
    _array(track, "track_sample_key", [[0, 10], [0, 11], [0, 12]])
    _array(track, "source_frame_interpolation", [0, 0, 0])
    _array(track, "source_instance_key", [100, 101, 102])
    _array(track, "source_row_index", [0, 1, 2])
    _array(track, "positions_mm", [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    _array(track, "positions_px", [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]])
    _array(track, "delta_seconds", [np.nan, 1.0 / 60.0, 1.0 / 60.0])
    _array(track, "transition_valid", [False, True, True])
    _array(track, "sample_valid", [True, True, True])
    _array(track, "time_seconds", [10.0 / 60.0, 11.0 / 60.0, 12.0 / 60.0])
    _array(track, "heading_degrees", [0.0, 1.0, 2.0])
    _array(track, "heading_radians", np.deg2rad([0.0, 1.0, 2.0]))
    _array(track, "smoothed_heading_degrees", [0.0, 1.0, 2.0])
    _array(track, "smoothed_heading_radians", np.deg2rad([0.0, 1.0, 2.0]))
    _array(track, "delta_heading_degrees", [np.nan, 1.0, 1.0])
    _array(track, "delta_heading_smoothed_degrees", [np.nan, 1.0, 1.0])
    _array(track, "angular_velocity_deg_s", [np.nan, 60.0, 60.0])
    _array(track, "angular_velocity_smoothed_deg_s", [np.nan, 60.0, 60.0])
    _array(track, "angular_speed_raw_deg_s", [np.nan, 60.0, 60.0])
    _array(track, "angular_speed_smoothed_deg_s", [np.nan, 60.0, 60.0])
    _array(track, "detection_source", [1, 1, 1])
    _array(track, "sample_reason_code", [0, 0, 0])
    _array(track, "transition_reason_code", [1, 0, 0])
    _array(track, "cumulative_path_distance_px", [0.0, 10.0, 20.0])
    _array(track, "cumulative_path_distance_mm", [0.0, 1.0, 2.0])

    for base, values in {
        "raw": [1.0, 2.0, 3.0],
        "filtered": [4.0, 5.0, 6.0],
        "smoothed": [7.0, 8.0, 9.0],
        "averaged": [10.0, 11.0, 12.0],
    }.items():
        source = f"speed_{base}"
        _array(track, f"{source}_mm", values)
        _array(track, f"{source}_px", np.asarray(values) * 10.0)
    for base, values in {
        "raw": [0.1, 0.2, 0.3],
        "filtered": [0.4, 0.5, 0.6],
        "smoothed": [0.7, 0.8, 0.9],
    }.items():
        _array(track, f"frame_path_distance_{base}_mm", values)
        _array(track, f"frame_path_distance_{base}_px", np.asarray(values) * 10.0)

    movement = track.create_group("movement")
    speed = movement.create_group("speed")
    for level, values in {
        "raw": [10.0, 20.0, 30.0],
        "filtered": [40.0, 50.0, 60.0],
        "smoothed": [70.0, 80.0, 90.0],
        "averaged": [100.0, 110.0, 120.0],
    }.items():
        grouped = speed.create_group(level)
        px = np.asarray(values, dtype=np.float64) * 10.0
        _array(grouped, "mm", values)
        _array(grouped, "px", px)
        _array(grouped, "acceleration_mm", [0.0, 1.0, 1.0])
        _array(grouped, "acceleration_px", [0.0, 10.0, 10.0])
        _array(grouped, "smoothed_acceleration_mm", [0.0, 1.0, 1.0])
        _array(grouped, "smoothed_acceleration_px", [0.0, 10.0, 10.0])
        if level != "averaged":
            _array(grouped, "frame_path_distance_mm", np.asarray(values) / 10.0)
            _array(grouped, "frame_path_distance_px", values)
    return root


def test_track_latest_fails_closed_when_root_selector_pair_disagrees() -> None:
    root = _make_track_kinematics_archive()
    parent = root["analysis/track_kinematics_runs"]
    parent.attrs["latest_complete"] = "offline/concurrent"

    with pytest.raises(ValueError, match="No stable complete selector-eligible"):
        resolve_track_kinematics_run(root, run_name="latest", scope="offline")


@pytest.mark.parametrize(
    ("selector_owner", "selector_name"),
    (("root", "latest_offline"), ("scope", "latest")),
)
def test_track_latest_fails_closed_when_scope_selector_disagrees(
    selector_owner: str,
    selector_name: str,
) -> None:
    root = _make_track_kinematics_archive()
    parent = root["analysis/track_kinematics_runs"]
    scope_group = parent["offline"]
    attrs = parent.attrs if selector_owner == "root" else scope_group.attrs
    attrs[selector_name] = "concurrent"

    with pytest.raises(ValueError, match="selector agreement"):
        resolve_track_kinematics_run(root, run_name="latest", scope="offline")


def test_track_latest_fails_closed_while_selected_child_is_ineligible() -> None:
    root = _make_track_kinematics_archive()
    root["analysis/track_kinematics_runs/offline/tk_test"].attrs[
        "stage_selector_eligible"
    ] = False

    with pytest.raises(ValueError, match="No stable complete selector-eligible"):
        resolve_track_kinematics_run(root, run_name="latest", scope="offline")


@pytest.mark.parametrize(
    "run_spec",
    (
        "garbage/tk_test",
        "analysis/track_kinematics_runs/offline/tk_test/extra",
        "analysis/track_kinematics_runs/online/tk_test",
    ),
)
def test_track_resolver_rejects_nonexact_explicit_path(run_spec: str) -> None:
    root = _make_track_kinematics_archive()

    with pytest.raises(ValueError, match="bare child name or the exact path"):
        resolve_track_kinematics_run(
            root,
            run_name=run_spec,
            scope="offline",
        )


class _FakeDescriptor:
    def __init__(self, digest: str, *, physical: bool = False) -> None:
        self._digest = digest
        self.profile_id = (
            "physical_mm.source_camera_y_down.v1"
            if physical
            else "source_camera_image_px.top_left_y_down.v1"
        )
        self.space_id = "physical_mm" if physical else "source_camera_image_px"
        self.geometry_type = "point_xy"
        self.components = ("x", "y")
        self.component_units = ("mm", "mm") if physical else ("px", "px")
        self.source_camera_overlay = SimpleNamespace(
            status="not_suitable" if physical else "direct",
            transform_refs=(),
        )
        self.reference_extent = SimpleNamespace(
            width=640,
            height=480,
            units="px",
        )

    def digest(self) -> str:
        return self._digest


class _FakeCoordinateBinding:
    def __init__(self, digest: str, *, physical: bool = False) -> None:
        self.descriptor = _FakeDescriptor(digest, physical=physical)


class _FakePositionBinding:
    def __init__(self) -> None:
        self.positions_px = _FakeCoordinateBinding("px-descriptor")
        self.positions_mm = _FakeCoordinateBinding(
            "mm-descriptor",
            physical=True,
        )


class _FakeSurface:
    def __init__(self, node) -> None:
        self.node = node
        self.shape = tuple(int(value) for value in node.shape)
        self.dtype = np.dtype(node.dtype).str
        self.content_sha256 = array_values_sha256(np.asarray(node[:]))


class _FakeBoundTrack:
    def __init__(self, track_group: zarr.Group) -> None:
        self.track_id = 0
        self.track_group = track_group
        self.position_binding = _FakePositionBinding()

    def surface(self, relative_path: str) -> _FakeSurface:
        node = self.track_group
        for part in relative_path.split("/"):
            if part not in node:
                raise KeyError(relative_path)
            node = node[part]
        return _FakeSurface(node)


class _FakeBoundRun:
    def __init__(self, run_group: zarr.Group, *, fail_final_check: bool = False) -> None:
        self.manifest_sha256 = "motion-manifest"
        self.run_group = run_group
        self._track = _FakeBoundTrack(run_group["tracks/id_0"])
        self.tracks = (self._track,)
        self.manifest = {
            "coordinate_binding_status": "bound_canonical_v2",
            "run_root_attrs": {
                "record": {
                    "immutable_attrs": dict(run_group.attrs),
                    "legacy_compatibility": {"attrs": {}},
                }
            },
            "tracks": {
                "id_0": {
                    "groups": {
                        ".": {"attrs": dict(self._track.track_group.attrs)}
                    }
                }
            },
        }
        self._fail_final_check = fail_final_check
        self.final_checks = 0

    def track(self, track_id: int) -> _FakeBoundTrack:
        if int(track_id) != 0:
            raise KeyError(track_id)
        return self._track

    def assert_verified(self) -> None:
        self.final_checks += 1
        if self._fail_final_check:
            raise ValueError("publication changed during read")


def _patch_bound_loader(monkeypatch: pytest.MonkeyPatch, *, fail_final_check: bool = False):
    from fisheye.analysis import track_kinematics as track_mod

    created: list[_FakeBoundRun] = []

    def load(_root, run_group):
        bound = _FakeBoundRun(run_group, fail_final_check=fail_final_check)
        created.append(bound)
        return bound

    monkeypatch.setattr(track_mod, "load_bound_track_motion_run", load)
    return created


def test_track_kinematics_logical_loader_requires_verified_bound_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_track_kinematics_archive()
    created = _patch_bound_loader(monkeypatch)

    track = load_track_kinematics_track(root, run_name="latest", scope="offline", track_id=0)

    assert track.authority_status == "verified_canonical_track_motion_v1"
    assert track.motion_manifest_sha256 == "motion-manifest"
    assert track.positions_px_descriptor_sha256 == "px-descriptor"
    assert track.positions_mm_descriptor_sha256 == "mm-descriptor"
    assert track.run_name == "tk_test"
    assert track.track_path == "analysis/track_kinematics_runs/offline/tk_test/tracks/id_0"
    np.testing.assert_array_equal(track.frame_indices, [10, 11, 12])
    np.testing.assert_allclose(track.speed_mm_by_level["raw"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(track.speed_mm_by_level["filtered"], [40.0, 50.0, 60.0])
    np.testing.assert_allclose(track.frame_path_distance_mm_by_level["filtered"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(track.positions_mm, [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    np.testing.assert_array_equal(track.source_acquisition_frame_index, [10, 11, 12])
    camera_positions, width, height = track.require_direct_source_camera_positions_px()
    np.testing.assert_allclose(camera_positions, [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]])
    assert (width, height) == (640, 480)
    assert created[0].final_checks == 1


def test_normal_track_loader_rejects_selector_handoff_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_track_kinematics_archive()
    root["analysis/track_kinematics_runs"].attrs["latest_complete"] = (
        "offline/concurrent"
    )
    created = _patch_bound_loader(monkeypatch)

    with pytest.raises(ValueError, match="No stable complete selector-eligible"):
        load_track_kinematics_track(
            root,
            run_name="latest",
            scope="offline",
            track_id=0,
        )

    assert created == []


def test_track_reader_rejects_non_camera_positions_for_direct_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_track_kinematics_archive()
    _patch_bound_loader(monkeypatch)
    track = load_track_kinematics_track(root, run_name="latest", scope="offline", track_id=0)
    track.positions_px_descriptor.space_id = "stimulus_texture_px"
    track.positions_px_descriptor.source_camera_overlay.status = "requires_transform"

    with pytest.raises(ValueError, match="not directly suitable"):
        track.require_direct_source_camera_positions_px()


def test_track_kinematics_normal_loader_fails_closed_without_canonical_seal() -> None:
    root = _make_track_kinematics_archive()

    with pytest.raises(ValueError, match="canonical|completion|binding|manifest"):
        load_track_kinematics_track(root, run_name="latest", scope="offline", track_id=0)


def test_explicit_legacy_inspection_loader_retains_historical_layout_fallback() -> None:
    root = _make_track_kinematics_archive()
    root["analysis/track_kinematics_runs"].attrs["latest_complete"] = (
        "offline/concurrent"
    )

    track = load_legacy_track_kinematics_track_for_inspection(
        root,
        run_name="latest",
        scope="offline",
        track_id=0,
    )

    assert track.authority_status == "unverified_legacy_inspection_only"
    assert track.motion_manifest_sha256 is None
    np.testing.assert_allclose(track.speed_mm_by_level["filtered"], [40.0, 50.0, 60.0])
    with pytest.raises(ValueError, match="canonical track-motion read"):
        track.authority_record()


def test_track_kinematics_loader_rechecks_after_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_track_kinematics_archive()
    _patch_bound_loader(monkeypatch, fail_final_check=True)

    with pytest.raises(ValueError, match="changed during read"):
        load_track_kinematics_track(root, run_name="latest", scope="offline", track_id=0)


def test_track_kinematics_loader_rejects_detector_only_speed_level() -> None:
    root = _make_track_kinematics_archive()

    with pytest.raises(ValueError, match="Unsupported physical track speed level"):
        load_track_kinematics_track(
            root,
            run_name="latest",
            scope="offline",
            track_id=0,
            required_speed_levels=("exponential",),
        )


def test_detect_bouts_load_speed_levels_uses_track_kinematics_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "track_io.zarr"
    _make_track_kinematics_archive(zarr_path)
    _patch_bound_loader(monkeypatch)

    speeds, metadata = _load_track_kinematics_track_speeds(zarr_path, track_kinematics_run="latest", track_id=0)

    assert metadata["track_kinematics_run"] == "tk_test"
    assert metadata["track_kinematics_scope"] == "offline"
    assert metadata["track_kinematics_git_commit"] == "abc"
    assert metadata["source_frame_indices_dtype"] == "int64"
    assert metadata["source_frame_indices_shape"] == [3]
    assert metadata["source_array_paths"]["frame_indices"] == (
        "analysis/track_kinematics_runs/offline/tk_test/tracks/id_0/frame_indices"
    )
    assert metadata["source_array_paths"]["track_sample_key"].endswith(
        "/track_sample_key"
    )
    assert metadata["track_motion_authority"]["motion_manifest_sha256"] == (
        "motion-manifest"
    )
    np.testing.assert_array_equal(speeds["frames"], [10, 11, 12])
    np.testing.assert_allclose(speeds["speed_filtered_mm"], [40.0, 50.0, 60.0])
    np.testing.assert_allclose(speeds["frame_path_distance_filtered_mm"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(metadata["positions_px"], [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]])
