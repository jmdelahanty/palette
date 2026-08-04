from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import zarr

import fisheye.analysis.chaser_bout_response as chaser_bout_response_module
from fisheye.analysis.chaser_component_publication import (
    ChaserComponentContract,
    build_chaser_component_handle,
    persist_chaser_component_manifest,
)
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.chaser_egocentric_bearing import (
    METHOD as EGOCENTRIC_METHOD,
    METHOD_VERSION as EGOCENTRIC_METHOD_VERSION,
    SCHEMA_ID as EGOCENTRIC_SCHEMA_ID,
    SCHEMA_VERSION as EGOCENTRIC_SCHEMA_VERSION,
    compute_egocentric_chaser_bearing,
)
from fisheye.analysis.chaser_bout_response import (
    COMPONENT_PARENT_NAME,
    DEFAULT_COMPONENT_NAME,
    CIRCLING_PNG_ARTIFACT_NAME,
    REFERENCE_KIND_OBJECT,
    REFERENCE_KIND_VIRTUAL,
    SCHEMA_ID,
    TURN_BIAS_PNG_ARTIFACT_NAME,
    _resolve_heading,
    _resolve_swim_bout_run,
    _select_bout_row_mask,
    bearing_deg,
    build_chaser_bout_response_result as _build_chaser_bout_response_result,
    write_chaser_bout_response_component,
)
from fisheye.shared.plot_artifacts import PNG_ARTIFACT_SCHEMA_ID
from tests.unit.fisheye.test_chaser_radial_occupancy import (
    ARENA_CENTER_MM,
    _make_archive,
)


FPS = 10.0
CX = CY = ARENA_CENTER_MM  # 50 mm

pytestmark = pytest.mark.usefixtures("logical_chaser_distance_reader")


class _SelectionGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs if attrs is not None else {}


# --------------------------------------------------------------------------------------
# The convention that broke this module once
# --------------------------------------------------------------------------------------


def test_bearing_matches_canonical_egocentric_bearing() -> None:
    """Arena positions are y-DOWN; heading is CCW-from-+x in a y-UP frame. Getting that flip
    wrong yields a plausible bearing that silently inverts the turn bias. This pins our
    formula to chaser_egocentric_bearing's."""

    rng = np.random.default_rng(4)
    n = 500
    fish = rng.uniform(10.0, 90.0, size=(n, 2))
    chaser = rng.uniform(10.0, 90.0, size=(n, 2, 2))
    heading = rng.uniform(-180.0, 180.0, size=n)

    _vec, canonical, _al, _lat, _valid = compute_egocentric_chaser_bearing(
        fish_arena_xy=fish, chaser_arena_xy=chaser, fish_heading_deg=heading
    )
    ours = bearing_deg(chaser - fish[:, None, :], heading)

    delta = (ours - np.asarray(canonical, dtype=np.float64) + 180.0) % 360.0 - 180.0
    assert np.nanmax(np.abs(delta)) < 1e-3  # canonical stores float32


def test_bearing_without_the_y_flip_would_be_wrong() -> None:
    """Guard the guard: the naive y-down bearing genuinely differs, so the test above has teeth."""

    fish = np.asarray([[0.0, 0.0]])
    ref = np.asarray([[[1.0, 1.0]]])          # down-right in y-down arena coords
    heading = np.asarray([0.0])               # pointing along +x
    ours = float(bearing_deg(ref - fish[:, None, :], heading)[0, 0])
    naive = math.degrees(math.atan2(1.0, 1.0))
    assert ours == pytest.approx(-45.0)       # y-down "below" == y-up "to the right" == negative bearing
    assert not math.isclose(ours, naive, abs_tol=1.0)


# --------------------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------------------


def _build_archive(
    tmp_path: Path,
    *,
    fish_mm: np.ndarray,
    chaser_mm: np.ndarray,
    heading_deg: np.ndarray,
    bout_start: np.ndarray,
    bout_end: np.ndarray,
    name: str,
    epoch_label: str = "post_event",
) -> Path:
    n = fish_mm.shape[0]
    windows = (ChaserDistanceWindow(0, epoch_label, 0, n - 1, 0.0, n / FPS, n / FPS),)
    z = _make_archive(
        tmp_path,
        fish_xy_mm=fish_mm,
        chaser_xy_mm=chaser_mm,
        windows=windows,
        fps=FPS,
        name=name,
    )
    root = zarr.open_group(str(z), mode="a", use_consolidated=False)
    root["analysis/stimulus_runs/stimulus_1"].attrs["protocol_json"] = json.dumps(
        {"steps": [{"parameters": {"position_transition_duration_s": 0.0, "chasers": [{"radius_mm": 2.0}]}}]}
    )
    # Minimal egocentric-bearing component: the module only needs heading on the camera-frame axis.
    run = root["analysis/chaser_distance_runs/chaser_distance_1"]
    ego = run.require_group("egocentric_bearing").require_group("track_test")
    frames = ego.require_group("frames")
    frames.create_array("fish_heading_deg", data=np.asarray(heading_deg, dtype=np.float32), overwrite=True)
    frames.create_array("fish_heading_valid", data=np.ones(n, dtype=bool), overwrite=True)
    ego.attrs.update({
        "schema_id": EGOCENTRIC_SCHEMA_ID,
        "schema_version": EGOCENTRIC_SCHEMA_VERSION,
        "method": EGOCENTRIC_METHOD,
        "method_version": EGOCENTRIC_METHOD_VERSION,
    })
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    persist_chaser_component_manifest(
        ego,
        snapshot=snapshot,
        relative_path="egocentric_bearing/track_test",
        contract=ChaserComponentContract(
            component_family="egocentric_bearing",
            component_name="track_test",
            semantic_schema_id=EGOCENTRIC_SCHEMA_ID,
            semantic_schema_version=EGOCENTRIC_SCHEMA_VERSION,
            method_id=EGOCENTRIC_METHOD,
            method_version=EGOCENTRIC_METHOD_VERSION,
            parameters={"fixture": "heading_on_camera_frame_axis"},
            source_authorities={"fixture": "sealed_test_egocentric"},
        ),
    )
    ego.attrs["palette_run_completion_status"] = "complete"
    ego.attrs["stage_selector_eligible"] = False

    bouts = root.require_group("analysis/swim_bout_runs/bouts_1/tables/bouts")
    k = len(bout_start)
    bouts.create_array("bout_id", data=np.arange(k, dtype=np.int32), overwrite=True)
    bouts.create_array("start_frame", data=np.asarray(bout_start, dtype=np.int64), overwrite=True)
    bouts.create_array("end_frame", data=np.asarray(bout_end, dtype=np.int64), overwrite=True)
    bouts.create_array("peak_physical_speed_mm_s", data=np.full(k, 20.0, dtype=np.float64), overwrite=True)
    bouts.create_array("mean_speed_mm_s", data=np.full(k, 10.0, dtype=np.float64), overwrite=True)
    bouts.create_array("duration_s", data=np.full(k, 0.3, dtype=np.float64), overwrite=True)
    bouts.create_array("path_length_mm", data=np.full(k, 3.0, dtype=np.float64), overwrite=True)
    bouts.create_array("net_displacement_mm", data=np.full(k, 2.0, dtype=np.float64), overwrite=True)
    root["analysis/swim_bout_runs"].attrs["latest"] = "bouts_1"
    root["analysis/swim_bout_runs"].attrs["latest_complete"] = "bouts_1"
    root["analysis/swim_bout_runs/bouts_1"].attrs[
        "palette_run_completion_status"
    ] = "complete"
    root["analysis/swim_bout_runs/bouts_1"].attrs[
        "stage_selector_eligible"
    ] = True
    return z


def _egocentric_handle(zarr_path: Path) -> dict[str, object]:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    return build_chaser_component_handle(
        root["analysis/chaser_distance_runs/chaser_distance_1/egocentric_bearing/track_test"],
        snapshot=snapshot,
        relative_path="egocentric_bearing/track_test",
    )


def build_chaser_bout_response_result(zarr_path: Path, **kwargs):
    """Exercise the maintained explicit-candidate dependency path in this suite."""

    return _build_chaser_bout_response_result(
        zarr_path,
        egocentric_dependency_handle=_egocentric_handle(zarr_path),
        swim_bout_legacy_compatibility=True,
        **kwargs,
    )


def _orbit(center: np.ndarray, radius: float, n: int, *, turns: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Fish circling `center` at `radius`. Heading is tangent to the circle (y-up convention)."""

    theta = np.linspace(0.0, turns * 2.0 * math.pi, n)
    pos = np.stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)], axis=1)
    # velocity in y-down coords is d/dtheta; heading is its y-up angle.
    vx, vy = -np.sin(theta), np.cos(theta)
    heading = np.degrees(np.arctan2(-vy, vx))
    return pos, heading


def _bouts_every(n: int, step: int = 10, length: int = 4) -> tuple[np.ndarray, np.ndarray]:
    start = np.arange(0, n - length - 1, step, dtype=np.int64)
    return start, start + length


def test_multi_level_bout_table_is_filtered_to_the_default_level() -> None:
    """detect_bouts_multi_level concatenates bouts from all five speed levels into one table.
    Ingesting all of them counts each physical bout five times and mixes jittery raw peaks with
    smoothed ones. Only the run's default level (default_signal_id) must be read."""

    per_level = 6
    n_levels = 5
    start = np.tile(np.arange(per_level, dtype=np.int64) * 10, n_levels)
    end = start + 4
    signal_id = np.repeat(np.arange(n_levels, dtype=np.int64), per_level)
    peak = np.repeat(
        np.arange(n_levels, dtype=np.float64) * 100.0 + 10.0,
        per_level,
    )

    keep, source_signal_id, selection = _select_bout_row_mask(
        start,
        end,
        total_frames=100,
        signal_id=signal_id,
        default_signal_id=4,
    )

    # exactly one level's worth of bouts, not five
    assert int(keep.sum()) == per_level
    assert source_signal_id == 4
    assert "filtered_to_default_signal_id_4" in selection
    # and it read the DEFAULT level's peak speed (410 mm/s for signal_id=4), not raw's (10)
    assert np.allclose(peak[keep], 410.0)


def test_single_level_bout_table_is_kept_whole() -> None:
    """A plain single-level table has no signal_id column and must be read as-is -- the filter
    must not silently drop it to zero."""

    start = np.asarray([0, 10, 20, 30], dtype=np.int64)
    end = start + 4
    keep, source_signal_id, selection = _select_bout_row_mask(
        start,
        end,
        total_frames=100,
        signal_id=None,
        default_signal_id=None,
    )

    assert np.all(keep)
    assert source_signal_id == -1
    assert "single_level" in selection


def test_swim_bout_selection_routes_through_canonical_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_group = object()
    parent = _SelectionGroup()
    parent["bouts_current"] = run_group
    root = _SelectionGroup()
    root["analysis/swim_bout_runs"] = parent
    calls: list[tuple[object, str, bool]] = []

    def _fake_resolve(candidate_root, *, run_name, legacy_compatibility=False):
        calls.append((candidate_root, run_name, legacy_compatibility))
        return "bouts_current"

    monkeypatch.setattr(
        chaser_bout_response_module,
        "resolve_swim_bout_run_name",
        _fake_resolve,
    )

    resolved_group, resolved_name, resolved_path = _resolve_swim_bout_run(
        root,
        "latest",
    )

    assert resolved_group is run_group
    assert resolved_name == "bouts_current"
    assert resolved_path == "analysis/swim_bout_runs/bouts_current"
    assert calls == [(root, "latest", False)]


def test_heading_selection_rejects_intermediate_selector_handoff() -> None:
    run_group = _SelectionGroup()
    parent = _SelectionGroup(
        attrs={"latest": "candidate", "latest_complete": "previous"}
    )
    parent["previous"] = _SelectionGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    parent["candidate"] = _SelectionGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    run_group["egocentric_bearing"] = parent

    with pytest.raises(ValueError, match="No stable complete selector-eligible"):
        _resolve_heading(
            _SelectionGroup(), snapshot=object(), run_group=run_group,
            total_frames=10, dependency_handle=None,
        )


# --------------------------------------------------------------------------------------
# The virtual controls: does the wall null work?
# --------------------------------------------------------------------------------------


def test_fish_orbiting_the_arena_centre_shows_no_object_specific_circling(tmp_path: Path) -> None:
    """A fish circling the arena centre (the wall-following null) sweeps around a wall-adjacent
    object for free. The raw circling index is high, but the object must show NO excess over
    its rotated virtual twins -- because the virtual twins are just as 'orbited' as it is."""

    n = 1200
    center = np.asarray([CX, CY])
    fish, heading = _orbit(center, radius=30.0, n=n, turns=8.0)
    # 8 mm INSIDE the fish's circular path: the fish sweeps past it, as a wall-follower sweeps
    # past a wall-adjacent object. Putting it ON the path would send the fish straight through.
    obj = np.tile(np.asarray([CX + 22.0, CY]), (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="wallnull.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)

    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    assert float(r.near_circling_index[0, o]) > 0.5          # raw circling is high...
    assert abs(float(r.circling_excess_vs_virtual[0, 0])) < 0.15   # ...but not object-specific
    assert any(ref.kind == REFERENCE_KIND_VIRTUAL for ref in r.references)


def test_fish_orbiting_the_object_shows_a_large_circling_excess(tmp_path: Path) -> None:
    """Now the fish circles the object itself, off-centre in the arena. The virtual twins sit
    elsewhere and are not orbited, so the excess must be large and positive."""

    n = 1200
    obj_pos = np.asarray([CX + 20.0, CY])
    fish, heading = _orbit(obj_pos, radius=6.0, n=n, turns=30.0)  # ~9 mm/s, above the moving threshold
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="objorbit.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)

    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    assert float(r.near_circling_index[0, o]) > 0.95         # motion is almost purely tangential

    # The excess is positive and the object beats every virtual twin. It is not enormous, and
    # that is geometry, not a weak effect: a tight orbit around a point still looks largely
    # tangential from a vantage only ~12 mm away, so the nearby twins score high too. The
    # excess measures what is specific to the object, which is exactly the point.
    twins = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_VIRTUAL]
    twin_values = [float(r.near_circling_index[0, t]) for t in twins]
    finite = [v for v in twin_values if math.isfinite(v)]
    assert finite, "fixture must leave at least one virtual twin with near-band support"
    assert float(r.near_circling_index[0, o]) > max(finite)
    assert float(r.circling_excess_vs_virtual[0, 0]) > 0.1


def test_excess_is_nan_when_no_virtual_twin_has_near_band_support(tmp_path: Path) -> None:
    """Push the object far enough out that the fish never comes near any virtual twin. With no
    control data there is no wall null, so the excess must be NaN rather than a bare number."""

    n = 1200
    obj_pos = np.asarray([CX + 30.0, CY])
    fish, heading = _orbit(obj_pos, radius=6.0, n=n, turns=30.0)
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="notwins.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)

    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    twins = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_VIRTUAL]
    assert math.isfinite(float(r.near_circling_index[0, o]))
    assert all(math.isnan(float(r.near_circling_index[0, t])) for t in twins)
    assert math.isnan(float(r.circling_excess_vs_virtual[0, 0]))


def test_virtual_reference_on_top_of_a_real_object_is_dropped(tmp_path: Path) -> None:
    """Two objects on opposite corners: a 90 deg rotation lands a 'virtual' control straight on
    the other real object. Such a reference is not a null and must be dropped."""

    n = 400
    fish, heading = _orbit(np.asarray([CX, CY]), radius=20.0, n=n)
    a = np.asarray([CX + 20.0, CY])
    b = np.asarray([CX, CY + 20.0])   # exactly 90 deg from a about the centre
    obj = np.stack([np.tile(a, (n, 1)), np.tile(b, (n, 1))], axis=1)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="collide.zarr")
    r = build_chaser_bout_response_result(
        z, chaser_distance_run="chaser_distance_1", virtual_rotations_deg=(90.0, 180.0), min_bin_bouts=5
    )
    virtuals = [ref for ref in r.references if ref.kind == REFERENCE_KIND_VIRTUAL]
    # Rotating `a` by 90 deg lands exactly on `b`, so that twin must be dropped. Rotating `b`
    # by 90 deg lands on empty arena, so that one is a legitimate control and must survive.
    assert not any(ref.parent_chaser_index == 0 and ref.rotation_deg == 90.0 for ref in virtuals)
    assert any(ref.parent_chaser_index == 1 and ref.rotation_deg == 90.0 for ref in virtuals)
    assert any(ref.parent_chaser_index == 0 and ref.rotation_deg == 180.0 for ref in virtuals)
    assert any(w.startswith("virtual_reference_dropped_on_real_object") for w in r.qc_warnings)


# --------------------------------------------------------------------------------------
# Turn bias
# --------------------------------------------------------------------------------------


def test_orbiting_the_object_turns_the_fish_toward_it(tmp_path: Path) -> None:
    """An arc around a point requires centripetal turning: the fish must keep turning toward
    the object. That is what a positive turn bias means -- circling, not approaching."""

    n = 1200
    obj_pos = np.asarray([CX + 20.0, CY])
    fish, heading = _orbit(obj_pos, radius=6.0, n=n, turns=30.0)
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="turntoward.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)

    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    assert float(r.near_turn_toward_fraction[0, o]) > 0.9
    # In a perfect orbit the bearing is pinned abeam, so it has no variance and r(bearing, turn)
    # is not the informative statistic here -- turn_toward_fraction is. See the dedicated
    # turn-bias tests below, which vary the bearing on purpose.


def test_straight_swimming_past_the_object_has_no_turn_bias(tmp_path: Path) -> None:
    n = 600
    fish = np.stack([np.linspace(CX - 30.0, CX + 30.0, n), np.full(n, CY - 8.0)], axis=1)
    heading = np.zeros(n)  # due +x, never turning
    obj = np.tile(np.asarray([CX, CY]), (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="straight.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)

    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    turns = r.bout_turn_deg[np.isfinite(r.bout_turn_deg)]
    assert np.allclose(turns, 0.0, atol=1e-6)
    # No turning at all -> the bias is undefined or zero, never a confident non-zero value.
    value = float(r.near_turn_bias_r[0, o])
    assert math.isnan(value) or abs(value) < 1e-6


# --------------------------------------------------------------------------------------
# Steering: what did the bout do to the fish's aim?
# --------------------------------------------------------------------------------------


def _steering_fixture(tmp_path: Path, *, headings: np.ndarray, name: str) -> Path:
    """Fish held 10 mm from the object with the object dead ahead at heading 0. Only its
    heading changes, and it changes *during* each bout -- so the bout is what re-aims it."""

    n = headings.shape[0]
    obj_pos = np.asarray([CX + 30.0, CY])
    fish = np.tile(obj_pos - np.asarray([10.0, 0.0]), (n, 1))  # object at bearing 0 when heading 0
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n, step=10, length=8)
    return _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=headings,
                          bout_start=bs, bout_end=be, name=name)


def test_bouts_steering_away_widen_the_predicted_miss(tmp_path: Path) -> None:
    n = 600
    z = _steering_fixture(tmp_path, headings=np.linspace(0.0, 80.0, n), name="steer_away.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)
    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]

    assert float(r.near_delta_predicted_miss_mm[0, o]) > 0.1   # each bout aims it wider
    assert float(r.near_fraction_widening_miss[0, o]) > 0.9


def test_bouts_steering_onto_the_object_narrow_the_predicted_miss(tmp_path: Path) -> None:
    """The mirror case. Note the metric uses |sin(bearing)|, so turning either way *off-axis*
    widens the miss -- what narrows it is swinging back toward dead-ahead."""

    n = 600
    z = _steering_fixture(tmp_path, headings=np.linspace(80.0, 0.0, n), name="steer_into.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)
    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]

    assert float(r.near_delta_predicted_miss_mm[0, o]) < -0.1  # each bout aims it tighter
    assert float(r.near_fraction_widening_miss[0, o]) < 0.1


def test_bouts_that_hold_their_aim_show_no_steering(tmp_path: Path) -> None:
    n = 600
    z = _steering_fixture(tmp_path, headings=np.full(n, 30.0), name="steer_none.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)
    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    assert float(r.near_delta_predicted_miss_mm[0, o]) == pytest.approx(0.0, abs=1e-4)


def test_predicted_miss_is_only_measured_with_the_object_ahead(tmp_path: Path) -> None:
    """With the object behind the fish there is no miss distance to steer, so those bouts must
    be excluded rather than contributing a meaningless number."""

    n = 400
    obj_pos = np.asarray([CX, CY])
    fish = np.stack([np.linspace(CX + 8.0, CX + 30.0, n), np.full(n, CY)], axis=1)
    heading = np.zeros(n)  # swimming +x, i.e. directly AWAY from the object behind it
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="behind.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)
    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]

    assert not np.any(r.bout_approaching_at_onset[:, o])
    assert np.all(np.isnan(r.bout_delta_predicted_miss_mm[:, o]))
    assert math.isnan(float(r.near_delta_predicted_miss_mm[0, o]))


# --------------------------------------------------------------------------------------
# Pseudoreplication: bouts inside one visit are not independent
# --------------------------------------------------------------------------------------


def test_visits_are_the_effective_sample_size(tmp_path: Path) -> None:
    """Many bouts inside a single close approach are one approach subsampled. The component
    must report the visit count and flag the pseudoreplication rather than let a bout-count
    n imply significance it does not have."""

    n = 1200
    obj_pos = np.asarray([CX + 20.0, CY])
    fish, heading = _orbit(obj_pos, radius=6.0, n=n, turns=30.0)  # never leaves the object
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="pseudo.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)

    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    assert int(r.near_bout_count[0, o]) > 50        # plenty of bouts...
    assert int(r.near_visit_count[0, o]) == 1       # ...from exactly one visit
    assert any(w.startswith("pseudoreplicated") for w in r.qc_warnings)
    visits = r.bout_visit_id[:, o]
    assert set(np.unique(visits[visits >= 0]).tolist()) == {0}


def test_separate_excursions_are_counted_as_separate_visits(tmp_path: Path) -> None:
    n = 900
    obj_pos = np.asarray([CX + 20.0, CY])
    # Three excursions: in close, far away, in close, far away, in close.
    radius = np.concatenate([np.full(100, 6.0), np.full(100, 30.0)] * 4 + [np.full(100, 6.0)])[:n]
    theta = np.linspace(0.0, 4.0 * math.pi, n)
    fish = np.stack([obj_pos[0] + radius * np.cos(theta), obj_pos[1] + radius * np.sin(theta)], axis=1)
    heading = np.zeros(n)
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n, step=5)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="visits.zarr")
    r = build_chaser_bout_response_result(
        z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5, visit_enter_mm=15.0, visit_exit_mm=20.0
    )
    o = [i for i, ref in enumerate(r.references) if ref.kind == REFERENCE_KIND_OBJECT][0]
    assert int(r.near_visit_count[0, o]) == 5


# --------------------------------------------------------------------------------------
# Guards and persistence
# --------------------------------------------------------------------------------------


def test_missing_egocentric_bearing_component_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="egocentric_bearing"):
        _resolve_heading(
            {}, snapshot=object(), run_group={}, total_frames=300,
            dependency_handle=None,
        )


def test_write_component_round_trips(tmp_path: Path) -> None:
    n = 900
    obj_pos = np.asarray([CX + 20.0, CY])
    fish, heading = _orbit(obj_pos, radius=6.0, n=n, turns=25.0)
    obj = np.tile(obj_pos, (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="persist.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)
    assert r.source_egocentric_manifest_sha256 == _egocentric_handle(z)["component_manifest_sha256"]
    path = write_chaser_bout_response_component(z, r, overwrite=True)

    assert path.endswith(f"{COMPONENT_PARENT_NAME}/{DEFAULT_COMPONENT_NAME}")
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    component = root[path]
    assert component.attrs["schema_id"] == SCHEMA_ID
    assert component.attrs["source_refs"]["source_egocentric_bearing_manifest_sha256"] == (
        r.source_egocentric_manifest_sha256
    )

    n_refs = len(r.references)
    n_bins = int(r.distance_bin_centers_mm.shape[0])
    assert component["binned"]["circling_index"].shape == (1, n_refs, n_bins)
    assert component["binned"]["turn_bias_r"].shape == (1, n_refs, n_bins)
    assert component["bouts_per_reference"]["visit_id"].shape == (len(bs), n_refs)
    assert component["object_vs_virtual"]["circling_excess_vs_virtual"].shape == (1, 1)
    assert component["object_vs_virtual"]["near_visit_count"].shape == (1, n_refs)

    kinds = [bytes(x).split(b"\x00")[0].decode() for x in np.asarray(component["references"]["kind_bytes"][:])]
    assert kinds.count(REFERENCE_KIND_OBJECT) == 1
    assert kinds.count(REFERENCE_KIND_VIRTUAL) >= 1

    for artifact in (CIRCLING_PNG_ARTIFACT_NAME, TURN_BIAS_PNG_ARTIFACT_NAME):
        png = component["visualizations"][artifact]
        assert png.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
        assert np.asarray(png[:], dtype=np.uint8).tobytes().startswith(b"\x89PNG")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        write_chaser_bout_response_component(z, r, overwrite=False, write_png=False)


def test_steering_excess_by_band_is_the_dose_response(tmp_path: Path) -> None:
    """A single near-band scalar averages over bins where nothing happens. On the real cohort
    that halved the signal. The per-band profile must be persisted so a localized, decaying
    response can be told apart from a flat artifact."""

    n = 1200
    center = np.asarray([CX, CY])
    fish, heading = _orbit(center, radius=30.0, n=n, turns=8.0)
    obj = np.tile(np.asarray([CX + 22.0, CY]), (n, 1)).reshape(n, 1, 2)  # twins also get swept
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=obj, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="band.zarr")
    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)

    n_bins = int(r.distance_bin_centers_mm.shape[0])
    assert r.steering_excess_by_band.shape == (1, 1, n_bins)
    assert r.circling_excess_by_band.shape == (1, 1, n_bins)

    # The fish sweeps past the object AND its rotated twins with identical geometry, so the
    # object-minus-twin excess must be ~0 in every supported bin -- a flat null profile.
    band = r.steering_excess_by_band[0, 0, :]
    supported = np.isfinite(band)
    assert supported.sum() >= 2
    assert np.all(np.abs(band[supported]) < 2.0)

    # A bin with no twin support yields NaN rather than a bare number.
    assert np.any(~supported)

    path = write_chaser_bout_response_component(z, r, overwrite=True, write_png=False)
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    ov = root[path]["object_vs_virtual"]
    assert ov["steering_excess_by_band"].shape == (1, 1, n_bins)
    assert ov.attrs["excess_by_band_axis_order"] == ["epoch", "object", "distance_bin"]
