"""Smoke tests for the trial-locked habituation figures.

These are plotting modules, so most of this is "renders a valid PNG without crashing". Two
things here are NOT cosmetic, and both are about not misleading a reader:

  * The cohort figure must not silently drop a recording. A cohort that quietly shrinks is
    worse than one that errors.
  * The per-recording rate panel must MARK dropout-heavy trials. The rate is per validly-
    tracked second, so a trial that lost half its frames divides by a small denominator and
    can spike. Presenting that point identically to a clean one invites exactly the wrong
    conclusion on exactly the metric this analysis turns on.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_escape_events import (
    build_chaser_escape_events_result,
    write_chaser_escape_events_component,
)
from fisheye.visualization.chaser_habituation_figures import (
    CLEAN_DROPOUT,
    HabituationData,
    render_habituation_cohort,
    render_habituation_sheet,
)
from tests.unit.fisheye.test_chaser_escape_events import (
    EPOCH_FRAMES,
    N_CYCLES,
    THRESHOLD,
    _build,
)


PNG_MAGIC = b"\x89PNG"


def _archive(tmp_path: Path, *, name: str, recording_id: str | None = None, **kw) -> Path:
    z = _build(tmp_path, name=name, **kw)
    if recording_id is not None:
        zarr.open_group(str(z), mode="a", use_consolidated=False).attrs["recording_id"] = recording_id
    r = build_chaser_escape_events_result(z, chaser_distance_run="chaser_distance_1",
                                          peak_speed_threshold_mm_s=THRESHOLD)
    write_chaser_escape_events_component(z, r, overwrite=True, write_png=False)
    return z


def _registry(tmp_path: Path, archives: dict[str, Path]) -> Path:
    db = tmp_path / "registry.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE dataset_context_current
                    (recording_id TEXT, zarr_path TEXT, zarr_use TEXT, dataset_status TEXT)""")
    for rid, p in archives.items():
        conn.execute("INSERT INTO dataset_context_current VALUES (?,?,?,?)",
                     (rid, str(p), "analysis", "active"))
    first = next(iter(archives))       # the real registry holds a duplicate row; it must not double-count
    conn.execute("INSERT INTO dataset_context_current VALUES (?,?,?,?)",
                 (first, str(archives[first]), "analysis", "active"))
    conn.commit()
    conn.close()
    return db


# --------------------------------------------------------------------------------------
# Reading the trial record
# --------------------------------------------------------------------------------------


def test_habituation_data_reads_the_trial_record(tmp_path: Path) -> None:
    d = HabituationData(_archive(tmp_path, name="h.zarr"))
    assert list(d.ordinal) == [1, 2, 3, 4]
    assert list(d.escape_count) == [1, 1, 1, 1]
    assert np.all(np.isfinite(d.wall_mm))
    assert np.all(d.clean)                       # the clean fixture loses no frames
    # trials all lie in the chase epoch, so none can carry a post-epoch escape
    assert int(np.max(d.trial_end)) < EPOCH_FRAMES


def test_early_late_split_tracks_a_habituating_fish(tmp_path: Path) -> None:
    flat = HabituationData(_archive(tmp_path, name="flat.zarr"))
    hab = HabituationData(_archive(tmp_path, name="hab.zarr", habituating=True))
    # only 4 trials in the fixture, so "trials 5+" is empty -- the split must return NaN, not 0.
    e_f, l_f = flat.early_late("rate")
    assert np.isfinite(e_f) and not np.isfinite(l_f)
    # ...but the habituating fish's escapes really are gone by trials 3-4
    assert list(hab.escape_count) == [1, 1, 0, 0]
    assert float(np.mean(hab.rate[hab.ordinal <= 2])) > float(np.mean(hab.rate[hab.ordinal >= 3]))


def test_recording_without_trials_is_a_clear_error_not_an_empty_plot(tmp_path: Path) -> None:
    z = _archive(tmp_path, name="notrials.zarr", with_trials=False)
    with pytest.raises(ValueError, match="no trials"):
        HabituationData(z)


def test_recording_without_the_component_is_a_clear_error(tmp_path: Path) -> None:
    z = _build(tmp_path, name="bare.zarr")     # bout response only, no escape events written
    with pytest.raises(ValueError, match="chaser_escape_events"):
        HabituationData(z)


# --------------------------------------------------------------------------------------
# The non-cosmetic contract: dropout-heavy trials must be visibly marked
# --------------------------------------------------------------------------------------


def test_dropout_heavy_trials_are_flagged_so_an_inflated_rate_cannot_pass_as_clean(tmp_path: Path) -> None:
    """The rate divides by validly-tracked time. A trial that lost half its frames therefore
    reports a HIGHER rate on a smaller denominator. That is correct -- and it is exactly the
    point a reader would over-interpret, so the sheet must draw it hollow."""

    d = HabituationData(_archive(tmp_path, name="drop.zarr", dropout=(185, 225)))
    assert float(d.dropout[2]) > CLEAN_DROPOUT
    assert not bool(d.clean[2])
    assert bool(d.clean[0])
    # the inflated point is real: same escape, smaller denominator
    assert int(d.escape_count[2]) == 1
    assert float(d.rate[2]) > float(d.rate[0])

    png = render_habituation_sheet(d.path)
    assert png.startswith(PNG_MAGIC)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def test_recording_sheet_renders(tmp_path: Path) -> None:
    png = render_habituation_sheet(_archive(tmp_path, name="sheet.zarr"))
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 20_000


def test_cohort_figure_renders_and_deduplicates(tmp_path: Path) -> None:
    archives = {
        rid: _archive(tmp_path, name=f"{rid}.zarr", recording_id=rid,
                      habituating=(i % 2 == 0))
        for i, rid in enumerate(("rec_A_GoodCopBadCop", "rec_B_GoodCopBadCop",
                                 "rec_C_GoodCopBadCop", "rec_D_GoodCopBadCop"))
    }
    db = _registry(tmp_path, archives)

    png, data, skipped = render_habituation_cohort(db, "%GoodCopBadCop%")
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 40_000
    # 5 registry rows, 4 unique recordings
    assert len(data) == 4
    assert len({d.recording_id for d in data}) == 4
    assert skipped == []


def test_cohort_figure_reports_what_it_skipped(tmp_path: Path) -> None:
    """A recording that vanishes from a cohort figure without a word is worse than one that
    errors: the reader sees an n and believes it."""

    good = _archive(tmp_path, name="good.zarr", recording_id="good_GoodCopBadCop")
    good2 = _archive(tmp_path, name="good2.zarr", recording_id="good2_GoodCopBadCop")
    good3 = _archive(tmp_path, name="good3.zarr", recording_id="good3_GoodCopBadCop")
    notrials = _archive(tmp_path, name="nt.zarr", recording_id="nt_GoodCopBadCop", with_trials=False)
    gone = tmp_path / "does_not_exist.zarr"
    db = _registry(tmp_path, {
        "good_GoodCopBadCop": good, "good2_GoodCopBadCop": good2, "good3_GoodCopBadCop": good3,
        "nt_GoodCopBadCop": notrials, "gone_GoodCopBadCop": gone,
    })

    png, data, skipped = render_habituation_cohort(db, "%GoodCopBadCop%")
    assert png.startswith(PNG_MAGIC)
    assert sorted(d.recording_id for d in data) == [
        "good2_GoodCopBadCop", "good3_GoodCopBadCop", "good_GoodCopBadCop"
    ]
    assert len(skipped) == 2                       # the trial-less one and the missing file
    assert all(reason for _name, reason in skipped)


def test_cohort_figure_survives_a_cohort_with_no_pre_epoch(tmp_path: Path) -> None:
    """The fixture has no pre_event, so the wall-vs-fast-bout control panel has no data.
    An absent control must leave an empty panel, not kill the figure."""

    archives = {rid: _archive(tmp_path, name=f"{rid}.zarr", recording_id=rid)
                for rid in ("x_GoodCopBadCop", "y_GoodCopBadCop", "z_GoodCopBadCop")}
    d = HabituationData(archives["x_GoodCopBadCop"])
    near, far = d.pre_fast_bout_rate_by_wall()
    assert np.isnan(near) and np.isnan(far)        # no pre epoch -> no control, reported as NaN

    png, data, _skipped = render_habituation_cohort(_registry(tmp_path, archives), "%GoodCopBadCop%")
    assert png.startswith(PNG_MAGIC)
    assert len(data) == 3


def test_cohort_with_no_usable_recording_raises_rather_than_drawing_an_empty_figure(tmp_path: Path) -> None:
    z = _archive(tmp_path, name="nt.zarr", recording_id="nt_GoodCopBadCop", with_trials=False)
    db = _registry(tmp_path, {"nt_GoodCopBadCop": z})
    with pytest.raises(ValueError, match="No recordings"):
        render_habituation_cohort(db, "%GoodCopBadCop%")
