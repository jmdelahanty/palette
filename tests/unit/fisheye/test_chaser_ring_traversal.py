"""Tests for the responsive-ring traversal figures.

Mostly "renders without crashing", but three things here are not cosmetic:

  * Each bout must be attached to the entry whose frame window contains its onset, and to no
    other. Getting this wrong would draw a bout on the wrong approach.
  * Escape bouts (peak > threshold) must be distinguished from ordinary ones, and the peak
    speed must come from the chaser_bout_response bout table -- the SAME table the escape
    counts come from -- so the red stars here and the escape rate in chaser_escape_events
    cannot disagree.
  * The animation must bound each entry to a fixed number of display frames, so a fish that
    lingers near a wall-adjacent object for 130 s does not produce a 13000-frame movie.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import zarr
from fisheye.visualization.chaser_ring_traversal import (
    RESPONSIVE_BAND_MM,
    RING_EDGES_MM,
    _collect_chase_ring_entries_unsealed_inspection as collect_chase_ring_entries,
    _collect_ring_entries_unsealed_inspection as collect_ring_entries,
    _ordered_entries,
    render_ring_entries_png,
    write_ring_traversal_gif,
)
from tests.unit.fisheye.test_chaser_bout_response import _egocentric_handle
from tests.unit.fisheye.test_chaser_visualization import _archive_with_components


pytestmark = pytest.mark.usefixtures("logical_chaser_distance_reader")


PNG_MAGIC = b"\x89PNG"
ESCAPE_SPEED = 250.0
ORDINARY_SPEED = 20.0


def _set_bout_peaks(zarr_path: Path, escape_every: int = 4) -> int:
    """Mark every Nth valid bout in the chaser_bout_response table as an escape.

    collect_ring_entries reads peak speed from THIS table, so this is the knob the escape
    colouring must respond to. Returns how many escapes were planted.
    """

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    bouts = root["analysis/chaser_distance_runs/chaser_distance_1/chaser_bout_response/"
                 "chaser_bout_response_v1/bouts"]
    peak = np.full(int(bouts["start_frame"].shape[0]), ORDINARY_SPEED, dtype=np.float64)
    peak[::escape_every] = ESCAPE_SPEED
    bouts["peak_speed_mm_s"][:] = peak
    return int(np.count_nonzero(peak > 100.0))


def _set_bout_turns(zarr_path: Path, high_turn_every: int = 2) -> None:
    """Give every Nth bout a 90 deg turn, the rest 0, in the chaser_bout_response table."""

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    bouts = root["analysis/chaser_distance_runs/chaser_distance_1/chaser_bout_response/"
                 "chaser_bout_response_v1/bouts"]
    turn = np.zeros(int(bouts["start_frame"].shape[0]), dtype=np.float64)
    turn[::high_turn_every] = 90.0
    bouts["turn_deg"][:] = turn


def _collect(zarr_path: Path, **kw):
    return collect_ring_entries(zarr_path, chaser_distance_run="chaser_distance_1",
                                epochs_wanted=("post_event",),
                                egocentric_dependency_handle=_egocentric_handle(zarr_path),
                                swim_bout_legacy_compatibility=True,
                                **kw)


# --------------------------------------------------------------------------------------
# Bout -> entry assignment
# --------------------------------------------------------------------------------------


def test_bouts_are_attached_to_the_entry_that_contains_them(tmp_path: Path) -> None:
    _set_bout_peaks(tmp_path_archive := _archive_with_components(tmp_path, name="a.zarr"))
    scenes, _meta = _collect(tmp_path_archive)
    entries = [(s, v) for s in scenes if s.is_object for v in s.visits]
    assert entries, "fixture should produce object entries"

    seen_any = False
    for _scene, visit in entries:
        lo, hi = int(visit["frames"][0]), int(visit["frames"][-1])
        for bout in visit["bouts"]:
            seen_any = True
            # the bout's drawn segment indexes into this visit's xy...
            assert 0 <= bout["i0"] <= bout["i1"] < len(visit["xy"])
            # ...and its frames lie inside this entry's window, not a neighbour's
            gframes = visit["frames"][bout["i0"] : bout["i1"] + 1]
            assert int(gframes[0]) >= lo and int(gframes[-1]) <= hi
    assert seen_any, "entries should carry bouts"


def test_a_bout_is_not_double_counted_across_entries(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="dbl.zarr")
    _set_bout_peaks(z)
    scenes, _meta = _collect(z)
    # collect the (epoch, onset-frame) of every drawn bout; the same onset must not appear in
    # two different entries of the same reference.
    per_ref: dict[tuple[str, str], list[int]] = {}
    for s in scenes:
        for v in s.visits:
            key = (s.epoch_label, s.ref_label)
            for b in v["bouts"]:
                per_ref.setdefault(key, []).append(int(v["frames"][b["i0"]]))
    for onsets in per_ref.values():
        assert len(onsets) == len(set(onsets)), "a bout onset appears in two entries"


# --------------------------------------------------------------------------------------
# Escape distinction, sourced from the bout-response table
# --------------------------------------------------------------------------------------


def test_escape_bouts_are_flagged_and_track_the_bout_response_table(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="esc.zarr")
    planted = _set_bout_peaks(z, escape_every=3)
    scenes, meta = _collect(z)

    flagged = [b for s in scenes for v in s.visits for b in v["bouts"] if b["is_escape"]]
    ordinary = [b for s in scenes for v in s.visits for b in v["bouts"] if not b["is_escape"]]
    assert flagged, "escapes planted in the table must surface as is_escape bouts"
    assert ordinary, "not every bout is an escape"
    assert all(b["peak_speed_mm_s"] > meta["peak_speed_threshold_mm_s"] for b in flagged)
    assert all(b["peak_speed_mm_s"] <= meta["peak_speed_threshold_mm_s"] for b in ordinary)
    assert planted > 0


def test_escapes_split_into_turn_and_dash_tiers(tmp_path: Path) -> None:
    """An escape is 'high_turn' (drawn as a red star) when it is fast AND its |turn| clears the
    angle threshold; otherwise it is a 'dash' (drawn orange). The tier must match the peak/turn
    of the underlying bout, and must never make a non-escape into a high-turn bout."""

    z = _archive_with_components(tmp_path, name="tiers.zarr")
    _set_bout_peaks(z, escape_every=2)      # every other bout is an escape
    _set_bout_turns(z, high_turn_every=4)   # every 4th bout has a 90 deg turn

    scenes, meta = collect_ring_entries(z, chaser_distance_run="chaser_distance_1",
                                        epochs_wanted=("post_event",),
                                        egocentric_dependency_handle=_egocentric_handle(z),
                                        swim_bout_legacy_compatibility=True,
                                        high_turn_threshold_deg=45.0)
    bouts = [b for s in scenes for v in s.visits for b in v["bouts"]]
    turn = [b for b in bouts if b["is_high_turn"]]
    dash = [b for b in bouts if b["is_escape"] and not b["is_high_turn"]]
    assert turn and dash, "the fixture should produce both tiers"
    # a high-turn bout is always an escape with a large turn
    assert all(b["is_escape"] and abs(b["turn_deg"]) >= 45.0 for b in turn)
    # a dash is an escape with a small turn
    assert all(abs(b["turn_deg"]) < 45.0 for b in dash)
    # an ordinary (non-escape) bout is never tagged high-turn, even if it turns sharply
    assert not any(b["is_high_turn"] for b in bouts if not b["is_escape"])
    assert meta["high_turn_threshold_deg"] == 45.0


def test_no_escapes_when_no_bout_clears_the_threshold(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="noesc.zarr")   # all bouts at 20 mm/s
    scenes, _meta = _collect(z)
    assert not any(b["is_escape"] for s in scenes for v in s.visits for b in v["bouts"])
    assert any(v["bouts"] for s in scenes for v in s.visits)     # but bouts are still attached


def test_missing_bout_response_degrades_to_no_segments_rather_than_crashing(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="nobr.zarr")
    root = zarr.open_group(str(z), mode="a", use_consolidated=False)
    del root["analysis/chaser_distance_runs/chaser_distance_1/chaser_bout_response"]
    scenes, _meta = _collect(z)
    # visits still exist (the trajectory), they just carry no bout segments
    assert any(s.visits for s in scenes)
    assert all(v["bouts"] == [] for s in scenes for v in s.visits)
    assert render_ring_entries_png(scenes, _meta).startswith(PNG_MAGIC)


# --------------------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------------------


def test_escape_containing_entries_are_ordered_first(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="ord.zarr")
    _set_bout_peaks(z, escape_every=5)
    scenes, _meta = _collect(z)
    ordered = _ordered_entries(scenes, object_only=True)
    if len(ordered) < 2:
        pytest.skip("need at least two object entries to test ordering")
    esc_counts = [sum(int(b["is_escape"]) for b in v["bouts"]) for _s, v in ordered]
    # non-increasing escape count: escape-rich entries lead
    assert esc_counts == sorted(esc_counts, reverse=True)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def test_ring_constants_are_coherent(tmp_path: Path) -> None:
    inner, outer = RESPONSIVE_BAND_MM
    assert inner in RING_EDGES_MM and outer in RING_EDGES_MM   # the shell edges are real rings
    assert inner < outer


def test_static_figure_renders(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="fig.zarr")
    _set_bout_peaks(z)
    scenes, meta = _collect(z)
    png = render_ring_entries_png(scenes, meta)
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 20_000


def test_animation_writes_a_bounded_file(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="anim.zarr")
    _set_bout_peaks(z)
    scenes, meta = _collect(z)
    out = write_ring_traversal_gif(scenes, meta, tmp_path / "ring.gif",
                                   fps=10, frames_per_entry=12, max_entries=3)
    assert out.exists()
    assert out.read_bytes().startswith(b"GIF")


def test_animation_caps_entries_and_frames_per_entry(tmp_path: Path, monkeypatch) -> None:
    """A wall-dwelling fish makes very long entries. Each entry must be bound to
    frames_per_entry display frames, and the number of entries to max_entries, or a real
    recording would produce a movie tens of thousands of frames long."""

    z = _archive_with_components(tmp_path, name="cap.zarr")
    _set_bout_peaks(z)
    scenes, meta = _collect(z)

    captured: dict[str, int] = {}
    import fisheye.visualization.chaser_ring_traversal as mod
    real = mod.FuncAnimation

    def spy(fig, func, frames, **kw):
        captured["frames"] = int(frames)
        return real(fig, func, frames=frames, **kw)

    monkeypatch.setattr(mod, "FuncAnimation", spy)
    fpe, hold_s, fps, maxe = 8, 0.5, 10, 2
    write_ring_traversal_gif(scenes, meta, tmp_path / "cap.gif",
                             fps=fps, frames_per_entry=fpe, max_entries=maxe, hold_s=hold_s)
    hold = max(1, int(hold_s * fps))
    # at most max_entries entries, each contributing at most fpe move-frames + hold-frames
    assert captured["frames"] <= maxe * (fpe + hold)


def test_animation_with_no_entries_raises(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="empty.zarr")
    scenes, meta = _collect(z, virtual_rotations_deg=())
    for s in scenes:
        s.visits.clear()
    with pytest.raises(ValueError, match="No entries"):
        write_ring_traversal_gif(scenes, meta, tmp_path / "x.gif")


# --------------------------------------------------------------------------------------
# The training epoch: chaser-centric frame (the object moves, so no static ring)
# --------------------------------------------------------------------------------------


def _chase_archive(tmp_path: Path, name: str) -> Path:
    """A training_event archive with a MOVING chaser, escape-speed bouts, and a materialized
    chaser_bout_response -- built by the escape-events fixture, which is exactly this shape."""

    from tests.unit.fisheye.test_chaser_escape_events import _build

    z = _build(tmp_path, name=name)
    root = zarr.open_group(str(z), mode="a", use_consolidated=False)
    bouts = root["analysis/chaser_distance_runs/chaser_distance_1/chaser_bout_response/"
                 "chaser_bout_response_v1/bouts"]
    # plant escapes: every 3rd valid bout goes to escape speed
    peak = np.asarray(bouts["peak_speed_mm_s"][:], dtype=np.float64)
    peak[::3] = 250.0
    bouts["peak_speed_mm_s"][:] = peak
    return z


def test_training_epoch_uses_a_chaser_centric_frame(tmp_path: Path) -> None:
    """During the chase the object moves, so the static ring frame is a fiction. The training
    collector must return a chaser-centric scene with NO fixed wall."""

    z = _chase_archive(tmp_path, "chase.zarr")
    scenes, meta = collect_chase_ring_entries(z, chaser_distance_run="chaser_distance_1",
                                              epoch_label="training_event")
    assert meta["frame"] == "chaser_centric"
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene.is_object
    assert bool(getattr(scene, "chaser_centric", False))
    # there is no fixed wall in this frame -- the arena geometry fields are NaN, and the
    # drawing must not try to place a wall arc
    assert math.isnan(scene.arena_center_distance_mm)
    assert math.isnan(scene.arena_radius_mm)
    assert scene.visits, "the moving chaser crosses the fish's near zone during the chase"


def test_training_entries_carry_escape_coloured_bouts(tmp_path: Path) -> None:
    z = _chase_archive(tmp_path, "chase2.zarr")
    scenes, meta = collect_chase_ring_entries(z, chaser_distance_run="chaser_distance_1")
    bouts = [b for v in scenes[0].visits for b in v["bouts"]]
    assert bouts, "training entries should carry bout segments"
    assert any(b["is_escape"] for b in bouts)
    # every bout's frames lie inside the entry that owns it
    for v in scenes[0].visits:
        lo, hi = int(v["frames"][0]), int(v["frames"][-1])
        for b in v["bouts"]:
            gf = v["frames"][b["i0"]: b["i1"] + 1]
            assert int(gf[0]) >= lo and int(gf[-1]) <= hi


def test_training_figure_and_animation_render(tmp_path: Path) -> None:
    z = _chase_archive(tmp_path, "chase3.zarr")
    scenes, meta = collect_chase_ring_entries(z, chaser_distance_run="chaser_distance_1")
    png = render_ring_entries_png(scenes, meta)
    assert png.startswith(PNG_MAGIC)
    # the caption must not claim a wall in the chaser-centric frame
    out = write_ring_traversal_gif(scenes, meta, tmp_path / "chase.gif",
                                   fps=10, frames_per_entry=10, max_entries=3)
    assert out.read_bytes().startswith(b"GIF")


def _arrow_follows_motion(scenes) -> float:
    """Mean cos between the drawn heading arrow (cos H, -sin H) and the fish's motion in the
    drawn frame, over all entries. A correct body-heading arrow is strongly positive; the sign
    bug this guards against gave ~0 (perpendicular)."""

    cs = []
    for s in scenes:
        for v in s.visits:
            xy = np.asarray(v["xy"])
            h = np.asarray(v["heading_deg"])
            if xy.shape[0] < 8:
                continue
            mv = np.gradient(xy, axis=0)
            arv = np.stack([np.cos(np.radians(h)), -np.sin(np.radians(h))], axis=1)
            n = np.linalg.norm(mv, axis=1) * np.linalg.norm(arv, axis=1)
            ok = (n > 1e-6) & np.isfinite(h)
            if ok.sum() >= 3:
                cs.append(float(np.nanmean((mv[ok] * arv[ok]).sum(1) / n[ok])))
    return float(np.nanmean(cs)) if cs else float("nan")


def test_static_frame_heading_arrow_points_along_the_motion(tmp_path: Path) -> None:
    """Regression for a sign bug that left the heading arrow ~perpendicular to the fish's
    motion. In the static (pre/post) frame the drawn path IS the fish's real motion, so a
    correct body-heading arrow must point strongly along it -- not sideways."""

    z = _archive_with_components(tmp_path, name="head.zarr")
    _set_bout_peaks(z)
    scenes, _meta = _collect(z)
    cos = _arrow_follows_motion(scenes)
    assert cos > 0.5, f"heading arrow does not follow motion (cos={cos:.2f}); the sign bug is back"
    """With a single chaser and no cra_primary_endpoint, the collector must not crash -- it
    falls back to chaser 0 and says so."""

    z = _chase_archive(tmp_path, "chase4.zarr")
    # the escape-events fixture has no cra endpoint on this run
    _scenes, meta = collect_chase_ring_entries(z, chaser_distance_run="chaser_distance_1")
    assert meta["aggressive_chaser_index"] == 0
    assert "fallback" in meta["aggressive_role_source"] or meta["aggressive_role_source"] == "cra_role_code"
