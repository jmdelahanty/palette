"""Smoke tests for the two chaser visualization modules.

These are plotting modules: the contract is "renders a valid PNG from a real archive without
crashing, and reads the object roles from the right place". The statistical correctness of what
they draw is covered by the component test suites.

The one thing here that is NOT cosmetic: object roles must come from the CRA endpoint's role
codes, never from chaser index order. Getting that wrong silently mislabels aggressive as inert
on any recording where the ordering differs, and every downstream figure would be wrong while
looking perfectly fine.
"""

from __future__ import annotations

import math
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_bout_response import (
    build_chaser_bout_response_result,
    write_chaser_bout_response_component,
)
from fisheye.analysis.chaser_response_regimes import (
    build_chaser_response_regimes_result,
    write_chaser_response_regimes_component,
)
from fisheye.visualization.chaser_analysis_figures import (
    RecordingData,
    render_cohort_figures,
    render_recording_summary,
)
from fisheye.visualization.chaser_visit_trajectories import (
    collect_visits,
    render_overlay_png,
    render_per_visit_png,
    write_gif,
)
from tests.unit.fisheye.test_chaser_bout_response import CX, CY, _bouts_every, _build_archive, _orbit
from tests.unit.fisheye.test_chaser_response_regimes import (
    _install_verified_track_reader,
)


PNG_MAGIC = b"\x89PNG"


@pytest.fixture(autouse=True)
def _verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_verified_track_reader(monkeypatch)


def _add_object_roles(zarr_path: Path, *, aggressive_chaser_index: int) -> None:
    """A minimal cra_primary_endpoint objects group -- the only place role truth lives."""

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    run = root["analysis/chaser_distance_runs/chaser_distance_1"]
    objects = run.require_group("cra_primary_endpoint").require_group("cra_1").require_group("objects")
    n = int(run["chasers"]["chaser_index"].shape[0])
    index = np.arange(n, dtype=np.int16)
    role = np.where(index == aggressive_chaser_index, 1, 0).astype(np.int8)  # 1 == aggressive
    objects.create_array("object_index", data=index, overwrite=True)
    objects.create_array("object_role_code", data=role, overwrite=True)


def _archive_with_components(tmp_path: Path, *, aggressive_chaser_index: int = 0,
                             name: str = "viz.zarr", recording_id: str | None = None) -> Path:
    """A distance run plus the two components the figures read, on real code paths."""

    n = 1500
    obj_pos = np.asarray([CX + 22.0, CY])
    fish, heading = _orbit(np.asarray([CX, CY]), radius=30.0, n=n, turns=8.0)
    # two objects, so the role mapping has something to get wrong
    chaser = np.stack(
        [np.tile(obj_pos, (n, 1)), np.tile(np.asarray([CX - 22.0, CY]), (n, 1))], axis=1
    )
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=chaser, heading_deg=heading,
                       bout_start=bs, bout_end=be, name=name)
    if recording_id is not None:
        root = zarr.open_group(str(z), mode="a", use_consolidated=False)
        root["analysis/chaser_distance_runs/chaser_distance_1"].attrs["recording_id"] = recording_id
    _add_object_roles(z, aggressive_chaser_index=aggressive_chaser_index)

    r = build_chaser_bout_response_result(z, chaser_distance_run="chaser_distance_1", min_bin_bouts=5)
    write_chaser_bout_response_component(z, r, overwrite=True, write_png=False, write_interactive_spec=False)
    g = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=5)
    write_chaser_response_regimes_component(z, g, overwrite=True, write_png=False, write_interactive_spec=False)
    return z


# --------------------------------------------------------------------------------------
# The one non-cosmetic contract
# --------------------------------------------------------------------------------------


def test_object_roles_come_from_the_cra_endpoint_not_index_order(tmp_path: Path) -> None:
    """Flip which chaser is aggressive and the labelling must follow. Reading roles off index
    order would silently mislabel every figure on a counterbalanced recording."""

    z0 = _archive_with_components(tmp_path, aggressive_chaser_index=0, name="roles0.zarr")
    z1 = _archive_with_components(tmp_path, aggressive_chaser_index=1, name="roles1.zarr")

    assert RecordingData(z0).roles == {0: "aggressive", 1: "inert"}
    assert RecordingData(z1).roles == {0: "inert", 1: "aggressive"}

    # ...and the roles must propagate into what gets plotted.
    _centers, bands = RecordingData(z1).steering_bands()
    assert any(role == "aggressive" for _ep, role in bands)
    assert any(role == "inert" for _ep, role in bands)


def test_missing_cra_endpoint_leaves_roles_unknown_rather_than_guessed(tmp_path: Path) -> None:
    n = 600
    fish, heading = _orbit(np.asarray([CX, CY]), radius=25.0, n=n)
    chaser = np.tile(np.asarray([CX + 20.0, CY]), (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=chaser, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="noroles.zarr")
    # No cra_primary_endpoint written: roles must be empty, not invented.
    assert RecordingData(z).roles == {}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def test_recording_summary_renders(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path)
    png = render_recording_summary(z)
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 20_000


def test_recording_summary_renders_with_only_a_distance_run(tmp_path: Path) -> None:
    """The sheet must degrade gracefully: a fish with no components yet still gets its
    occupancy, wall distance and tracking dropout panels."""

    n = 600
    fish, heading = _orbit(np.asarray([CX, CY]), radius=25.0, n=n)
    chaser = np.tile(np.asarray([CX + 20.0, CY]), (n, 1)).reshape(n, 1, 2)
    bs, be = _bouts_every(n)
    z = _build_archive(tmp_path, fish_mm=fish, chaser_mm=chaser, heading_deg=heading,
                       bout_start=bs, bout_end=be, name="bare.zarr")

    d = RecordingData(z)
    assert d.steering_bands()[1] == {}
    assert d.freeze_curves()[1] == {}
    assert all(math.isnan(v) for v in d.thigmotaxis())

    png = render_recording_summary(z)
    assert png.startswith(PNG_MAGIC)


def test_recording_data_rejects_an_archive_with_no_distance_run(tmp_path: Path) -> None:
    z = tmp_path / "empty.zarr"
    zarr.open_group(str(z), mode="w")
    with pytest.raises(ValueError, match="No chaser_distance_run"):
        RecordingData(z)


def test_wall_distance_and_dropout_are_computed(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path)
    d = RecordingData(z)
    w = d.wall_distance_mm("post_event")
    assert w.size > 0
    # The fixture orbits at radius 30 in a 40 mm arena, so the fish sits ~10 mm off the wall.
    assert 5.0 < float(np.median(w)) < 15.0
    assert d.dropout()["post_event"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Cohort figure
# --------------------------------------------------------------------------------------


def _registry(tmp_path: Path, archives: dict[str, Path]) -> Path:
    db = tmp_path / "registry.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE dataset_context_current
                    (recording_id TEXT, zarr_path TEXT, zarr_use TEXT, dataset_status TEXT)""")
    for rid, p in archives.items():
        conn.execute("INSERT INTO dataset_context_current VALUES (?,?,?,?)",
                     (rid, str(p), "analysis", "active"))
    # a duplicate row for one id: the real registry has one, and it must not double-count
    first = next(iter(archives))
    conn.execute("INSERT INTO dataset_context_current VALUES (?,?,?,?)",
                 (first, str(archives[first]), "analysis", "active"))
    conn.commit()
    conn.close()
    return db


def test_cohort_figure_renders_and_deduplicates(tmp_path: Path) -> None:
    archives = {
        rid: _archive_with_components(tmp_path, name=f"{rid}.zarr", recording_id=rid)
        for rid in ("rec_A_GoodCopBadCop", "rec_B_GoodCopBadCop", "rec_C_GoodCopBadCop")
    }
    db = _registry(tmp_path, archives)

    png, data, skipped = render_cohort_figures(db, "%GoodCopBadCop%")
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 30_000
    # 4 registry rows, 3 unique recordings -- the duplicate must not be counted twice.
    assert len(data) == 3
    assert len({d.recording_id for d in data}) == 3
    assert skipped == []


def test_cohort_figure_skips_unusable_archives(tmp_path: Path) -> None:
    good = _archive_with_components(tmp_path, name="good.zarr",
                                    recording_id="good_GoodCopBadCop")
    bad = tmp_path / "bad.zarr"
    zarr.open_group(str(bad), mode="w")
    missing = tmp_path / "does_not_exist.zarr"
    db = _registry(tmp_path, {"good_GoodCopBadCop": good, "bad_GoodCopBadCop": bad,
                              "gone_GoodCopBadCop": missing})

    png, data, skipped = render_cohort_figures(db, "%GoodCopBadCop%")
    assert png.startswith(PNG_MAGIC)
    assert [d.recording_id for d in data] == ["good_GoodCopBadCop"]
    # A recording that vanishes from a cohort figure without a word is worse than one that
    # errors. The skips must be reported, with a reason.
    assert len(skipped) == 2
    assert all(reason for _name, reason in skipped)


# --------------------------------------------------------------------------------------
# Visit trajectories
# --------------------------------------------------------------------------------------


def test_visit_trajectories_render(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="visits.zarr")
    scenes, meta = collect_visits(z, chaser_distance_run="chaser_distance_1",
                                  epochs_wanted=("post_event",), virtual_rotations_deg=(120.0, 240.0))
    assert scenes
    objects = [s for s in scenes if s.is_object]
    controls = [s for s in scenes if not s.is_object]
    # Every object must be drawn beside virtual controls -- that pairing is the whole point.
    assert objects and controls

    assert render_overlay_png(scenes, meta).startswith(PNG_MAGIC)
    assert render_per_visit_png(scenes, meta).startswith(PNG_MAGIC)


def test_visit_animation_writes_a_file(tmp_path: Path) -> None:
    z = _archive_with_components(tmp_path, name="anim.zarr")
    scenes, meta = collect_visits(z, chaser_distance_run="chaser_distance_1",
                                  epochs_wanted=("post_event",), virtual_rotations_deg=(180.0,))
    out = write_gif(scenes, meta, tmp_path / "visits.gif", fps=10, stride=20)
    assert out.exists()
    assert out.read_bytes().startswith(b"GIF")
