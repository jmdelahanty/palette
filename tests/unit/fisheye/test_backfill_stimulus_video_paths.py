from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.utils import backfill_stimulus_video_paths as mod


def _seed_analysis_zarr(zarr_path: Path, *, source_h5: Path | None) -> zarr.Group:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "analysis"
    analysis = root.create_group("analysis")
    stim_parent = analysis.create_group("stimulus_runs")
    run = stim_parent.create_group("stimulus_001")
    if source_h5 is not None:
        run.attrs["source_h5"] = str(source_h5)
    stim_parent.attrs["latest"] = "stimulus_001"
    return run


def _seed_training_zarr(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "training"


def test_backfill_run_group_updates_when_rendered_video_exists(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    mp4_path = tmp_path / "session.mp4"
    h5_path.touch()
    mp4_path.touch()

    run = _seed_analysis_zarr(tmp_path / "sample_analysis.zarr", source_h5=h5_path)
    dry_result = mod._backfill_run_group(run, overwrite_existing=False, apply=False)  # noqa: SLF001
    assert dry_result.status == "would_update"

    apply_result = mod._backfill_run_group(run, overwrite_existing=False, apply=True)  # noqa: SLF001
    assert apply_result.status == "updated"
    assert run.attrs.get("source_stimulus_video_path") == str(mp4_path.resolve())


def test_backfill_run_group_reports_missing_source_h5(tmp_path: Path) -> None:
    run = _seed_analysis_zarr(tmp_path / "sample_analysis.zarr", source_h5=None)
    result = mod._backfill_run_group(run, overwrite_existing=False, apply=False)  # noqa: SLF001
    assert result.status == "no_source_h5"


def test_main_apply_updates_latest_analysis_stimulus_run(tmp_path: Path, capsys) -> None:
    h5_path = tmp_path / "session.h5"
    mp4_path = tmp_path / "session.mp4"
    zarr_path = tmp_path / "sample_analysis.zarr"
    h5_path.touch()
    mp4_path.touch()

    _seed_analysis_zarr(zarr_path, source_h5=h5_path)

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["stimulus_runs"]["stimulus_001"]
    assert run.attrs.get("source_stimulus_video_path") == str(mp4_path.resolve())

    out = capsys.readouterr().out
    assert "Applied: updated=1" in out


def test_main_any_scope_counts_training_as_not_expected(tmp_path: Path, capsys) -> None:
    analysis_h5 = tmp_path / "analysis_session.h5"
    analysis_mp4 = tmp_path / "analysis_session.mp4"
    analysis_h5.touch()
    analysis_mp4.touch()

    _seed_analysis_zarr(tmp_path / "a_analysis.zarr", source_h5=analysis_h5)
    _seed_training_zarr(tmp_path / "b_training.zarr")

    rc = mod.main([str(tmp_path), "--recursive", "--zarr-use", "any"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "training_not_expected=1" in out
