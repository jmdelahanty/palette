from __future__ import annotations

import argparse
from pathlib import Path

import zarr

from fisheye.analysis import create_analysis_zarr as create_mod


def _base_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        recording_dir=tmp_path,
        video=None,
        h5=None,
        output=None,
        import_video_metadata=True,
        video_metadata_overwrite=False,
        import_stimulus=None,
        stimulus_run_name=None,
        stimulus_overwrite=False,
        stimulus_quiet=True,
        skip_chaser_repair=False,
        register=False,
        registry=None,
        apply=False,
        dry_run=True,
        log_dir=None,
        no_log=True,
    )


def test_build_plan_ok_single_camera_single_h5(tmp_path: Path) -> None:
    rec = tmp_path / "2026-02-09T01-00-00Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True)
    (rec / "raw").mkdir(parents=True)
    (rec / "cams" / "Cam2010093_a.mp4").touch()
    (rec / "raw" / "session.h5").touch()

    args = _base_args(tmp_path)
    args.recording_dir = rec
    plan = create_mod._build_plan(args)  # noqa: SLF001

    assert plan.status == "ok"
    assert plan.video_path == (rec / "cams" / "Cam2010093_a.mp4").resolve()
    assert plan.h5_path == (rec / "raw" / "session.h5").resolve()
    assert plan.import_stimulus is True
    assert plan.output_path == (rec / "zarr" / f"{rec.name}_analysis.zarr").resolve()


def test_build_plan_fails_closed_on_multiple_cameras(tmp_path: Path) -> None:
    rec = tmp_path / "2026-02-09T02-00-00Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True)
    (rec / "raw").mkdir(parents=True)
    (rec / "cams" / "Cam2010093_a.mp4").touch()
    (rec / "cams" / "Cam2010094_b.mp4").touch()
    (rec / "raw" / "session.h5").touch()

    args = _base_args(tmp_path)
    args.recording_dir = rec
    plan = create_mod._build_plan(args)  # noqa: SLF001

    assert plan.status == "missing"
    assert "multiple camera videos found" in (plan.reason or "")


def test_build_plan_no_import_stimulus_allows_missing_h5(tmp_path: Path) -> None:
    rec = tmp_path / "2026-02-09T03-00-00Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True)
    (rec / "cams" / "Cam2010093_a.mp4").touch()

    args = _base_args(tmp_path)
    args.recording_dir = rec
    args.import_stimulus = False
    plan = create_mod._build_plan(args)  # noqa: SLF001

    assert plan.status == "ok"
    assert plan.h5_path is None
    assert plan.import_stimulus is False


def test_ensure_archive_sets_analysis_attrs(tmp_path: Path) -> None:
    output = tmp_path / "rec" / "zarr" / "rec_analysis.zarr"
    video = tmp_path / "rec" / "cams" / "Cam2010093_a.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    plan = create_mod.CreationPlan(
        recording_dir=(tmp_path / "rec"),
        video_path=video,
        h5_path=None,
        output_path=output,
        import_stimulus=False,
        status="ok",
    )
    create_mod._ensure_archive(plan)  # noqa: SLF001

    root = zarr.open_group(str(output), mode="r")
    assert root.attrs.get("zarr_purpose") == "analysis"
    assert root.attrs.get("source_video_path") == str(video)
    assert root.attrs.get("source_video") == video.name


def test_apply_video_metadata_uses_analysis_import_purpose(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "rec" / "zarr" / "rec_analysis.zarr"
    video = tmp_path / "rec" / "cams" / "Cam2010093_a.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    zarr.open_group(str(output), mode="w")
    video.touch()

    plan = create_mod.CreationPlan(
        recording_dir=(tmp_path / "rec"),
        video_path=video,
        h5_path=None,
        output_path=output,
        import_stimulus=False,
        status="ok",
    )

    calls: dict[str, object] = {}

    def _fake_probe(path: Path) -> dict[str, object]:
        calls["probe_path"] = path
        return {"codec": "hevc"}

    def _fake_write(
        root: zarr.Group,
        meta: dict[str, object],
        *,
        overwrite: bool,
        import_purpose: str,
    ) -> dict[str, object]:
        calls["overwrite"] = overwrite
        calls["import_purpose"] = import_purpose
        calls["meta"] = meta
        calls["root"] = root
        return {"root": {"width": 4512}, "raw_video": {"video_codec": "hevc"}}

    monkeypatch.setattr(create_mod, "probe_video_metadata", _fake_probe)
    monkeypatch.setattr(create_mod, "write_video_metadata", _fake_write)

    updates = create_mod._apply_video_metadata(plan, overwrite=True)  # noqa: SLF001

    assert calls["probe_path"] == video
    assert calls["overwrite"] is True
    assert calls["import_purpose"] == "analysis"
    assert updates == {"root_attrs_updated": 1, "raw_video_attrs_updated": 1}
