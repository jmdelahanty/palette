from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from fisheye.utils import copy_recording as mod


def _write_file(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _create_recording(root: Path, name: str = "rec_a") -> Path:
    recording = root / name
    recording.mkdir(parents=True)
    (recording / "recording_manifest.json").write_text(json.dumps({"recording_name": name}), encoding="utf-8")
    _write_file(recording / "cams" / "Cam2010093.mp4", b"video")
    _write_file(recording / "raw" / "session.h5", b"h5")
    _write_file(recording / "zarr" / f"{name}_analysis.zarr" / "zarr.json", b'{"zarr_format":3}')
    _write_file(recording / "zarr" / f"{name}_analysis.zarr" / "raw_video" / "c" / "0", b"A")
    _write_file(recording / "zarr" / f"{name}_training.zarr" / "zarr.json", b'{"zarr_format":3}')
    return recording


def test_build_copy_plan_resolves_recording_name_and_destination_parent(tmp_path: Path) -> None:
    recording_root = tmp_path / "recordings"
    source = _create_recording(recording_root, "rec_a")
    dest_parent = tmp_path / "dest"

    plan = mod.build_copy_plan(Path("rec_a"), dest_parent, recording_root=recording_root)

    assert Path(plan.source_recording) == source.resolve()
    assert Path(plan.destination_recording) == (dest_parent / "rec_a").absolute()
    assert "--exclude=zarr/*.zarr/" in plan.regular_rsync_command
    assert [row.name for row in plan.zarr_stores] == ["rec_a_analysis.zarr", "rec_a_training.zarr"]
    assert all(row.mode == "tar-stream" for row in plan.zarr_stores)
    assert plan.validation == "quick"


def test_build_copy_plan_rejects_nonempty_destination_without_apply_context(tmp_path: Path) -> None:
    recording_root = tmp_path / "recordings"
    _create_recording(recording_root, "rec_a")
    dest_parent = tmp_path / "dest"
    _write_file(dest_parent / "rec_a" / "existing.txt", b"existing")

    plan = mod.build_copy_plan(recording_root / "rec_a", dest_parent)
    result = mod.execute_copy_plan(
        plan,
        resume=False,
        zarr_mode="tar-stream",
        tar_bin="tar",
        rsync_bin="rsync",
        overwrite_tarballs=False,
    )

    assert not result.ok
    assert result.results[0].step == "preflight"
    assert "non-empty" in str(result.results[0].detail)


def test_copy_zarr_tar_stream_copies_complete_store(tmp_path: Path) -> None:
    source = tmp_path / "src" / "zarr" / "rec_analysis.zarr"
    _write_file(source / "zarr.json", b'{"zarr_format":3}')
    _write_file(source / "group" / "array" / "c" / "0", b"chunk")
    destination = tmp_path / "dst" / "zarr" / "rec_analysis.zarr"

    result = mod._copy_zarr_tar_stream(source, destination)  # noqa: SLF001

    assert result.status == "ok"
    assert (destination / "zarr.json").read_bytes() == b'{"zarr_format":3}'
    assert (destination / "group" / "array" / "c" / "0").read_bytes() == b"chunk"


def test_quick_validate_detects_missing_zarr(tmp_path: Path) -> None:
    recording_root = tmp_path / "recordings"
    _create_recording(recording_root, "rec_a")
    dest_parent = tmp_path / "dest"
    plan = mod.build_copy_plan(recording_root / "rec_a", dest_parent)
    _write_file(dest_parent / "rec_a" / "recording_manifest.json", b"{}")

    result = mod._quick_validate(plan)  # noqa: SLF001

    assert result.status == "error"
    assert "missing zarr directory" in str(result.detail)


def test_main_dry_run_prints_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if shutil.which("rsync") is None:
        pytest.skip("rsync is not installed")
    recording_root = tmp_path / "recordings"
    _create_recording(recording_root, "rec_a")

    rc = mod.main(["rec_a", str(tmp_path / "dest"), "--recording-root", str(recording_root)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "source_recording=" in out
    assert "zarr_store=rec_a_analysis.zarr mode=tar-stream" in out
    assert "Dry run: add --apply" in out


def test_tarball_mode_plans_archive_paths(tmp_path: Path) -> None:
    recording_root = tmp_path / "recordings"
    _create_recording(recording_root, "rec_a")
    archive_dir = tmp_path / "archives"

    plan = mod.build_copy_plan(
        recording_root / "rec_a",
        tmp_path / "dest",
        zarr_mode="tarball",
        archive_dir=archive_dir,
    )

    assert [Path(row.tarball_path or "").parent for row in plan.zarr_stores] == [archive_dir, archive_dir]
    assert all((row.tarball_path or "").endswith(".zarr.tar") for row in plan.zarr_stores)
