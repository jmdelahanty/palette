"""Tests for chaser_escape_events.

The fixture is a scripted pursuit, built so the module's central failure mode is a test and
not a memory: the fish flees, gains 12 mm, and the chaser closes 10.5 mm of it back, so the
NET distance change from onset to +4 s is ~+1.5 mm -- nearly zero. A component that reported
net change would call this "no escape". gain and recapture must be recovered separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_bout_response import (
    REFERENCE_KIND_OBJECT,
    REFERENCE_KIND_VIRTUAL,
    build_chaser_bout_response_result,
    write_chaser_bout_response_component,
)
from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.chaser_escape_events import (
    COMPONENT_PARENT_NAME,
    DEFAULT_COMPONENT_NAME,
    SCHEMA_ID,
    build_chaser_escape_events_result,
    render_escape_events_png,
    write_chaser_escape_events_component,
)
from tests.unit.fisheye.test_chaser_radial_occupancy import ARENA_CENTER_MM, _make_archive


FPS = 10.0
CX = CY = ARENA_CENTER_MM          # 50 mm
CYCLE = 80                          # frames between escape events
N_CYCLES = 4
EPOCH_FRAMES = CYCLE * N_CYCLES     # 320
ESCAPE_PEAK = 150.0
ORDINARY_PEAK = 20.0
THRESHOLD = 100.0

# Escape onset sits at c+20 in each cycle; the fish flees over c+20..c+25.
ONSET_OFFSET = 20
GAIN_OFFSET = 25                    # +0.5 s at 10 fps
ORDINARY_OFFSETS = (50, 70)


def _piecewise(keys: list[tuple[int, float]], n: int) -> np.ndarray:
    xs = np.asarray([k[0] for k in keys], dtype=np.float64)
    ys = np.asarray([k[1] for k in keys], dtype=np.float64)
    return np.interp(np.arange(n, dtype=np.float64), xs, ys)


def _chase_epoch() -> tuple[np.ndarray, np.ndarray]:
    """Fish flees, chaser closes back in. Distance: 16 -> 8 (closing), 8 -> 20 (escape),
    20 -> 8 (recapture), 8 -> 16 (reset)."""

    fish_keys: list[tuple[int, float]] = []
    dist_keys: list[tuple[int, float]] = []
    for i in range(N_CYCLES):
        c = i * CYCLE
        fish_keys += [(c, 40.0), (c + 20, 40.0), (c + 25, 52.0), (c + 65, 52.0)]
        dist_keys += [(c, 16.0), (c + 20, 8.0), (c + 25, 20.0), (c + 65, 8.0)]
    fish_keys.append((EPOCH_FRAMES, 40.0))
    dist_keys.append((EPOCH_FRAMES, 16.0))
    fish_x = _piecewise(fish_keys, EPOCH_FRAMES)
    d = _piecewise(dist_keys, EPOCH_FRAMES)
    return fish_x, fish_x - d          # object sits to the fish's left


def _static_epoch() -> tuple[np.ndarray, np.ndarray]:
    """Same fish motion, but the object never moves. The escape gains ground and KEEPS it."""

    fish_keys: list[tuple[int, float]] = []
    for i in range(N_CYCLES):
        c = i * CYCLE
        fish_keys += [(c, 40.0), (c + 20, 40.0), (c + 25, 52.0), (c + 65, 52.0)]
    fish_keys.append((EPOCH_FRAMES, 40.0))
    fish_x = _piecewise(fish_keys, EPOCH_FRAMES)
    return fish_x, np.full(EPOCH_FRAMES, 24.0)


TRIGGER_RADIUS_MM = 10.0     # distance falls 16 -> 8 over c..c+20, so the trigger fires at c+15
TRIGGER_OFFSET = 15


def _write_chase_trials(root: zarr.Group, n: int) -> None:
    """One chase trial per cycle of the chase epoch, via the controller's chase_trial_id.

    Only the chase epoch has trials; the post epoch's object never chases.
    """

    dt = np.dtype([
        ("stimulus_frame_num", np.int64),
        ("chaser_index", np.int64),
        ("chase_sequence_active", np.bool_),
        ("is_chasing", np.bool_),
        ("chase_trial_id", np.int64),
        ("initial_distance_mm", np.float64),
    ])
    rec = np.zeros(n, dtype=dt)
    rec["stimulus_frame_num"] = np.arange(n, dtype=np.int64)
    rec["initial_distance_mm"] = TRIGGER_RADIUS_MM
    for i in range(N_CYCLES):
        c = i * CYCLE
        lo, hi = c, c + 65            # active window inside the chase epoch
        rec["chase_sequence_active"][lo:hi] = True
        rec["is_chasing"][lo:hi] = True
        rec["chase_trial_id"][lo:hi] = i + 1
    tracking = root["analysis/stimulus_runs/stimulus_1"].require_group("tracking_data")
    tracking.create_array("chaser_states", data=rec, overwrite=True)


def _build(tmp_path: Path, *, name: str = "escape.zarr",
           dropout: tuple[int, int] | None = None,
           habituating: bool = False,
           with_trials: bool = True) -> Path:
    chase_fish, chase_obj = _chase_epoch()
    post_fish, post_obj = _static_epoch()
    fish_x = np.concatenate([chase_fish, post_fish])
    obj_x = np.concatenate([chase_obj, post_obj])
    n = int(fish_x.size)

    fish = np.stack([fish_x, np.full(n, CY)], axis=1)
    chaser = np.stack([obj_x, np.full(n, CY)], axis=1).reshape(n, 1, 2)

    windows = (
        ChaserDistanceWindow(0, "training_event", 0, EPOCH_FRAMES - 1, 0.0,
                             EPOCH_FRAMES / FPS, EPOCH_FRAMES / FPS),
        ChaserDistanceWindow(1, "post_event", EPOCH_FRAMES, n - 1, EPOCH_FRAMES / FPS,
                             n / FPS, EPOCH_FRAMES / FPS),
    )
    fish_valid = np.ones(n, dtype=bool)
    if dropout is not None:
        fish_valid[dropout[0]:dropout[1]] = False

    z = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser, windows=windows,
                      fps=FPS, fish_valid=fish_valid, name=name)
    root = zarr.open_group(str(z), mode="a", use_consolidated=False)
    root["analysis/stimulus_runs/stimulus_1"].attrs["protocol_json"] = json.dumps(
        {"steps": [{"parameters": {"position_transition_duration_s": 0.0, "chasers": [{"radius_mm": 2.0}]}}]}
    )

    # Heading: the fish moves along +/- x, so heading is 0 or 180. Constant 0 is enough --
    # this module never uses heading; only chaser_bout_response does, to build the table.
    run = root["analysis/chaser_distance_runs/chaser_distance_1"]
    ego = run.require_group("egocentric_bearing").require_group("track_test")
    frames = ego.require_group("frames")
    frames.create_array("fish_heading_deg", data=np.zeros(n, dtype=np.float32), overwrite=True)
    frames.create_array("fish_heading_valid", data=np.ones(n, dtype=bool), overwrite=True)
    run["egocentric_bearing"].attrs["latest"] = "track_test"

    if with_trials:
        _write_chase_trials(root, n)

    starts: list[int] = []
    peaks: list[float] = []
    for base in (0, EPOCH_FRAMES):
        for i in range(N_CYCLES):
            c = base + i * CYCLE
            starts.append(c + ONSET_OFFSET)
            # habituating: only the first two cycles of the CHASE epoch escape; later ones
            # make an ordinary bout in exactly the same place. Everything else is identical,
            # so any declining slope must come from the escapes and not from the geometry.
            fast = ESCAPE_PEAK
            if habituating and base == 0 and i >= 2:
                fast = ORDINARY_PEAK
            starts[-1] = c + ONSET_OFFSET
            peaks.append(fast)
            for off in ORDINARY_OFFSETS:
                starts.append(c + off)
                peaks.append(ORDINARY_PEAK)
    order = np.argsort(np.asarray(starts))
    start = np.asarray(starts, dtype=np.int64)[order]
    peak = np.asarray(peaks, dtype=np.float64)[order]
    k = int(start.size)

    bouts = root.require_group("analysis/swim_bout_runs/bouts_1/tables/bouts")
    bouts.create_array("bout_id", data=np.arange(k, dtype=np.int32), overwrite=True)
    bouts.create_array("start_frame", data=start, overwrite=True)
    bouts.create_array("end_frame", data=start + 4, overwrite=True)
    bouts.create_array("peak_physical_speed_mm_s", data=peak, overwrite=True)
    bouts.create_array("mean_speed_mm_s", data=peak / 3.0, overwrite=True)
    bouts.create_array("duration_s", data=np.full(k, 0.4), overwrite=True)
    bouts.create_array("path_length_mm", data=np.full(k, 4.0), overwrite=True)
    bouts.create_array("net_displacement_mm", data=np.full(k, 3.0), overwrite=True)
    root["analysis/swim_bout_runs"].attrs["latest"] = "bouts_1"

    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=2)
    write_chaser_bout_response_component(z, r, overwrite=True, write_png=False, write_interactive_spec=False)
    return z


def _result(z: Path, **kw):
    return build_chaser_escape_events_result(z, chaser_distance_run="chaser_distance_1",
                                             peak_speed_threshold_mm_s=THRESHOLD, **kw)


def _obj(result) -> int:
    return [i for i, r in enumerate(result.references) if r.kind == REFERENCE_KIND_OBJECT][0]


def _epoch(result, label: str) -> int:
    return [i for i, e in enumerate(result.epochs) if e.label == label][0]


# ======================================================================================
# THE ONE THAT MATTERS: gain and recapture must be separable when net change is ~zero
# ======================================================================================


def test_net_change_is_near_zero_while_gain_and_recapture_are_both_large(tmp_path: Path) -> None:
    """The trap this module exists to avoid.

    The scripted chase gains 12 mm and gives 10.5 mm back, so net change over the window is
    ~+1.5 mm -- indistinguishable from no escape at all. Any component that reports only net
    change reports a null here. gain and recapture must survive separately."""

    r = _result(_build(tmp_path))
    e, o = _epoch(r, "training_event"), _obj(r)

    # The onset baseline is a 4-frame median (8.6 mm here, not the 8.0 mm of the onset frame
    # alone) -- deliberately, so one bad frame at onset cannot set the whole trace's zero.
    assert float(r.pursuit_gain_mm[e, o]) == pytest.approx(11.4, abs=0.5)
    assert float(r.pursuit_recapture_mm[e, o]) == pytest.approx(10.5, abs=0.8)

    # ...and the net is nearly nothing. This is the number that lies.
    assert abs(float(r.pursuit_net_mm[e, o])) < 2.5
    assert float(r.pursuit_gain_mm[e, o]) > 4.0 * abs(float(r.pursuit_net_mm[e, o]))


def test_static_object_gives_the_same_gain_but_no_recapture(tmp_path: Path) -> None:
    """The control that makes the pursuit claim mean anything.

    Identical fish trajectory, identical escape -- but the object does not chase. The gain
    must be the same and the recapture must vanish. If the escape gain differed here, the
    'escape' would be an artifact of how the object moves, not of what the fish does."""

    r = _result(_build(tmp_path))
    chase, post, o = _epoch(r, "training_event"), _epoch(r, "post_event"), _obj(r)

    assert float(r.pursuit_gain_mm[chase, o]) == pytest.approx(float(r.pursuit_gain_mm[post, o]), abs=1.0)
    assert float(r.pursuit_recapture_mm[post, o]) == pytest.approx(0.0, abs=1.0)
    assert float(r.pursuit_recapture_mm[chase, o]) > 5.0
    # Against a static dot the fish keeps every millimetre it won.
    assert float(r.pursuit_net_mm[post, o]) == pytest.approx(12.0, abs=1.0)


# ======================================================================================
# Rate, trigger, controls
# ======================================================================================


def test_escape_events_are_the_fast_bouts_and_nothing_else(tmp_path: Path) -> None:
    r = _result(_build(tmp_path))
    assert int(r.summary["escape_event_count"]) == 2 * N_CYCLES     # one per cycle, both epochs
    assert np.all(np.asarray(r.event_peak_speed_mm_s) > THRESHOLD)
    # the ordinary bouts are counted, not discarded
    assert int(r.ordinary_bout_count.sum()) == 2 * N_CYCLES * len(ORDINARY_OFFSETS)


def test_escapes_fire_closer_to_the_object_than_ordinary_bouts(tmp_path: Path) -> None:
    """The proximity trigger. Escape onsets sit at 8 mm; ordinary bouts at 10-13 mm."""

    r = _result(_build(tmp_path))
    e, o = _epoch(r, "training_event"), _obj(r)
    assert float(r.escape_onset_distance_mm[e, o]) == pytest.approx(8.0, abs=0.5)
    assert float(r.ordinary_onset_distance_mm[e, o]) > float(r.escape_onset_distance_mm[e, o])
    assert float(r.proximity_shift_mm[e, o]) < 0.0     # negative == escapes fire closer


def test_virtual_twins_do_not_reproduce_the_escape_gain(tmp_path: Path) -> None:
    """The wall control. The fish flees the object along +x; the rotated twins sit elsewhere,
    so they must not show the object's gain. If they did, the 'escape' would be geometry."""

    r = _result(_build(tmp_path))
    e, o = _epoch(r, "training_event"), _obj(r)
    ref = r.references[o]
    twins = [i for i, rr in enumerate(r.references)
             if rr.kind == REFERENCE_KIND_VIRTUAL and rr.parent_chaser_index == ref.chaser_index]
    assert twins
    twin_gain = float(np.nanmedian(r.pursuit_gain_mm[e, twins]))
    assert float(r.pursuit_gain_mm[e, o]) > 4.0
    assert float(r.pursuit_gain_mm[e, o]) > twin_gain + 4.0


def test_rate_is_normalized_by_validly_tracked_time_not_wall_clock(tmp_path: Path) -> None:
    """Detector recall collapses on frozen fish, so an epoch that loses frames must not be
    read as an epoch where less happened. The valid-time rate has to be the higher one."""

    z = _build(tmp_path, name="drop.zarr", dropout=(300, EPOCH_FRAMES))   # 20 frames of the chase
    r = _result(z)
    e = _epoch(r, "training_event")
    assert float(r.tracking_dropout_fraction[e]) > 0.0
    assert float(r.valid_duration_s[e]) < float(r.epoch_duration_s[e])
    assert float(r.escape_rate_per_valid_min[e]) > float(r.escape_rate_per_min[e])

    clean = _result(_build(tmp_path, name="clean.zarr"))
    ce = _epoch(clean, "training_event")
    # No frames lost, so the two denominators agree.
    assert float(clean.escape_rate_per_valid_min[ce]) == pytest.approx(
        float(clean.escape_rate_per_min[ce]), rel=1e-3
    )


def test_traces_never_cross_an_epoch_boundary(tmp_path: Path) -> None:
    """Between epochs the objects are repositioned. A 5 s trace that ran past the boundary
    would splice that teleport into the distance curve and read as a giant recapture."""

    r = _result(_build(tmp_path), post_window_s=6.0)   # 60 frames: the last escape's window
    chase = _epoch(r, "training_event")                # (onset 260) would reach frame 320
    ev_epoch = np.asarray(r.event_epoch_index)
    usable = np.asarray(r.event_trace_usable)
    last = int(np.max(np.asarray(r.event_start_frame)[ev_epoch == chase]))
    assert last == 3 * CYCLE + ONSET_OFFSET            # 260
    idx = int(np.flatnonzero((np.asarray(r.event_start_frame) == last) & (ev_epoch == chase))[0])
    assert not bool(usable[idx])                       # 260 + 60 = 320 == the boundary: excluded
    # but it is still counted in the rate
    assert int(r.escape_count[chase]) == N_CYCLES
    assert any(w.startswith("escape_events_without_a_clean_window") for w in r.qc_warnings)


def test_threshold_sweep_is_monotone_and_recorded(tmp_path: Path) -> None:
    r = _result(_build(tmp_path))
    counts = np.asarray(r.sweep_escape_count).sum(axis=1)
    assert np.all(np.diff(counts) <= 0)                # higher threshold, never more events
    lo = int(np.flatnonzero(np.asarray(r.threshold_sweep_mm_s) == 60.0)[0])
    hi = int(np.flatnonzero(np.asarray(r.threshold_sweep_mm_s) == 300.0)[0])
    assert counts[lo] == 2 * N_CYCLES                  # 150 mm/s escapes clear 60
    assert counts[hi] == 0                             # nothing clears 300


# ======================================================================================
# Contracts on the inputs
# ======================================================================================


def test_references_are_inherited_from_the_bout_component_not_recomputed(tmp_path: Path) -> None:
    """The virtual twins must be the SAME ones the bout table was scored against. Recomputing
    them from this module's defaults would let them drift apart silently."""

    z = _build(tmp_path)
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    parent = root["analysis/chaser_distance_runs/chaser_distance_1/chaser_bout_response"]
    src = parent[sorted(parent.keys())[-1]]["references"]
    src_labels = [bytes(v).split(b"\x00")[0].decode() for v in np.asarray(src["label_bytes"][:])]
    src_rot = np.asarray(src["rotation_deg"][:], dtype=np.float64)

    r = _result(z)
    assert [ref.label for ref in r.references] == src_labels
    np.testing.assert_allclose([ref.rotation_deg for ref in r.references], src_rot,
                               rtol=0, atol=0, equal_nan=True)


def test_missing_bout_response_component_is_a_clear_error(tmp_path: Path) -> None:
    n = 200
    fish = np.stack([np.full(n, 40.0), np.full(n, CY)], axis=1)
    chaser = np.stack([np.full(n, 24.0), np.full(n, CY)], axis=1).reshape(n, 1, 2)
    z = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser, fps=FPS, name="bare.zarr")
    with pytest.raises(ValueError, match="chaser_bout_response"):
        build_chaser_escape_events_result(z, chaser_distance_run="chaser_distance_1")


def test_no_escapes_is_a_status_not_a_crash(tmp_path: Path) -> None:
    r = _result(_build(tmp_path, name="none.zarr"), )
    high = build_chaser_escape_events_result(
        _build(tmp_path, name="none2.zarr"), chaser_distance_run="chaser_distance_1",
        peak_speed_threshold_mm_s=500.0,
    )
    assert high.status == "no_escape_events"
    assert int(high.summary["escape_event_count"]) == 0
    assert any("never_reaches_threshold" in w for w in high.qc_warnings)
    assert np.all(np.asarray(high.escape_count) == 0)
    assert not np.any(np.isfinite(np.asarray(high.pursuit_gain_mm)))


# ======================================================================================
# Persistence
# ======================================================================================


def test_component_round_trips(tmp_path: Path) -> None:
    z = _build(tmp_path)
    r = _result(z)
    path = write_chaser_escape_events_component(z, r, overwrite=True, write_png=False)

    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    c = root[path]
    assert c.attrs["schema_id"] == SCHEMA_ID
    assert c.attrs["source_bout_response_component"]
    assert COMPONENT_PARENT_NAME in path and DEFAULT_COMPONENT_NAME in path

    e = [bytes(v).split(b"\x00")[0].decode() for v in np.asarray(c["epochs/label_bytes"][:])]
    chase = e.index("training_event")
    o = 0   # the object is reference 0 by construction in _build_references

    gain = np.asarray(c["pursuit/gain_mm"][:])
    recap = np.asarray(c["pursuit/recapture_mm"][:])
    assert float(gain[chase, o]) == pytest.approx(11.4, abs=0.5)
    assert float(recap[chase, o]) == pytest.approx(10.5, abs=0.8)

    traces = np.asarray(c["traces/delta_distance_mm"][:])
    t = np.asarray(c["traces/time_s"][:])
    assert traces.shape[2] == t.size
    onset = int(np.argmin(np.abs(t)))
    # Baseline-subtracted, so the trace starts at ~0. Not exactly 0: the baseline is a 4-frame
    # median and the chaser is still closing over those frames, so onset sits 0.6 mm below it.
    assert float(traces[chase, o, onset]) == pytest.approx(0.0, abs=1.0)

    # the decomposition guidance has to survive into the archive, not just the docstring
    assert "underpowered" in str(c["pursuit"].attrs["pursuit_decomposition"])
    assert "valid" in str(c["rates"].attrs["rate_denominator"])


def test_png_renders(tmp_path: Path) -> None:
    png = render_escape_events_png(_result(_build(tmp_path)))
    assert png.startswith(b"\x89PNG")
    assert len(png) > 15_000


# ======================================================================================
# Trial-locked habituation
# ======================================================================================


def test_trials_come_from_the_controllers_chase_trial_id(tmp_path: Path) -> None:
    r = _result(_build(tmp_path))
    assert r.trial_segmentation_source == "chase_trial_id"
    assert len(r.trial_ordinal) == N_CYCLES                    # only the chase epoch has trials
    assert list(np.asarray(r.trial_ordinal)) == [1, 2, 3, 4]
    assert list(np.asarray(r.trial_id)) == [1, 2, 3, 4]
    # every trial lies inside the chase epoch, never the post epoch
    assert int(np.max(np.asarray(r.trial_end_frame))) < EPOCH_FRAMES


def test_escapes_are_joined_to_the_trial_they_fall_in(tmp_path: Path) -> None:
    r = _result(_build(tmp_path))
    # one escape per cycle, and the post-epoch escapes belong to no trial at all
    assert list(np.asarray(r.trial_escape_count)) == [1, 1, 1, 1]
    assert np.all(np.asarray(r.trial_any_escape))
    # latency is measured from the proximity trigger, not the trial start
    lat = np.asarray(r.trial_first_escape_latency_s)
    assert np.allclose(lat, (ONSET_OFFSET - TRIGGER_OFFSET) / FPS, atol=0.15)
    assert np.all(lat > 0.0)      # the chaser closes, THEN the fish flees


def test_habituation_slope_is_negative_only_when_escapes_actually_decline(tmp_path: Path) -> None:
    """The whole point of the trial axis. Escape collapse after the first trials is invisible
    when the chase epoch is averaged; it only exists per trial."""

    flat = _result(_build(tmp_path, name="flat.zarr"))
    hab = _result(_build(tmp_path, name="hab.zarr", habituating=True))

    assert list(np.asarray(hab.trial_escape_count)) == [1, 1, 0, 0]
    assert float(flat.habituation_slope_per_trial) == pytest.approx(0.0, abs=1e-6)
    assert float(hab.habituation_slope_per_trial) < 0.0
    assert float(hab.habituation_slope_any_escape) < 0.0

    s = hab.summary
    assert float(s["p_any_escape_trials_1_2"]) == pytest.approx(1.0)
    assert float(s["escape_rate_per_valid_s_trials_1_2"]) > float(s["escape_rate_per_valid_s_trials_5plus"] or 0.0) \
        if s.get("escape_rate_per_valid_s_trials_5plus") is not None else True


def test_trial_rate_uses_valid_time_so_a_lost_fish_is_not_a_calm_fish(tmp_path: Path) -> None:
    """A trial that lost half its frames must not read as a trial where the fish escaped
    half as much. This is the exact artifact that could manufacture fake habituation:
    dropout rises when the fish freezes, and freezing is what we are trying to measure."""

    # blank the second half of trial 3 (frames 160..205 of an 80-frame cycle starting at 160)
    z = _build(tmp_path, name="tdrop.zarr", dropout=(185, 225))
    r = _result(z)
    d = np.asarray(r.trial_dropout_fraction)
    assert float(d[2]) > 0.3 and float(d[0]) == pytest.approx(0.0, abs=1e-6)
    # the trial still has its escape (frame 180 is before the blanked span) ...
    assert int(np.asarray(r.trial_escape_count)[2]) == 1
    # ... and because the denominator shrank, its rate is HIGHER than a clean trial's, not lower
    rate = np.asarray(r.trial_escape_rate_per_valid_s)
    assert float(rate[2]) > float(rate[0])


def test_wall_distance_at_trigger_is_recorded_for_the_confound(tmp_path: Path) -> None:
    """Habituation is confounded with the fish moving to the wall. The component must record
    wall position so the question can be asked -- and must NOT quietly regress it out, since
    wall position is downstream of the chase (a mediator, not a covariate)."""

    r = _result(_build(tmp_path))
    w = np.asarray(r.trial_wall_distance_at_trigger_mm)
    assert np.all(np.isfinite(w))
    # fixture: fish at x=40 in a 40 mm-radius arena centred at 50 -> 10 mm off centre, 30 from wall
    assert np.allclose(w, 30.0, atol=1.0)
    assert "mediator" in str(r.diagnostics["wall_confound"])

    path = write_chaser_escape_events_component(_build(tmp_path, name="w2.zarr"),
                                                _result(_build(tmp_path, name="w2.zarr")),
                                                overwrite=True, write_png=False)
    root = zarr.open_group(str(tmp_path / "w2.zarr"), mode="r", use_consolidated=False)
    trials = root[path]["trials"]
    assert "wall_distance_at_trigger_mm" in list(trials.keys())
    assert "post-treatment" in str(trials.attrs["wall_confound"])


def test_no_stimulus_controller_leaves_trials_empty_rather_than_failing(tmp_path: Path) -> None:
    """A recording with no chase_trial_id still gets rates, traces and pursuit. The trial
    axis simply is not available, and says so."""

    r = _result(_build(tmp_path, name="notrials.zarr", with_trials=False))
    assert len(r.trial_ordinal) == 0
    assert r.trial_segmentation_source == "unavailable"
    assert math.isnan(float(r.habituation_slope_per_trial))
    assert any(w.startswith("trial_segmentation_unavailable") for w in r.qc_warnings)
    # the rest of the component is unaffected
    assert int(r.summary["escape_event_count"]) == 2 * N_CYCLES
    assert float(r.pursuit_gain_mm[_epoch(r, "training_event"), _obj(r)]) > 4.0
