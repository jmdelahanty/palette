"""Backfill of Orange's empty-dish background into already-imported recordings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.utils import backfill_orange_empty_background as backfill

from tests.unit.fisheye.test_orange_empty_background_loader import SERIALS, _image, _recording

RUN = "background_orange_empty_2026_11_02_15_04_11"


def _published(tmp_path: Path, *, seal_snapshot: bool = True, **kwargs) -> Path:
    """An imported recording: sealed capture on disk, consolidated archive without a run."""

    recording_dir = tmp_path / "rec"
    acquisition_dir = recording_dir / "raw" / "acquisition"
    acquisition_dir.mkdir(parents=True)
    _recording(acquisition_dir, **kwargs)
    data = (acquisition_dir / "recording_snapshot_start.json").read_bytes()
    files = []
    if seal_snapshot:
        files.append(
            {
                "relative_path": "raw/acquisition/recording_snapshot_start.json",
                "source": {
                    "path": "recording_snapshot_start.json",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                },
            }
        )
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps({"camera_id": SERIALS[0], "source_transfer": {"source_to_parent_files": files}})
    )
    zarr_path = recording_dir / "zarr" / "rec_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3, use_consolidated=False)
    root.attrs["marker"] = "published"
    root.create_group("raw_video")
    consolidate_metadata_capture_expected_warnings(zarr_path)
    return recording_dir


def _tree(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def test_dry_run_reports_the_run_it_would_write_and_changes_nothing(tmp_path: Path) -> None:
    recording_dir = _published(tmp_path)
    before = _tree(recording_dir / "zarr")

    plan = backfill.plan_recording(recording_dir)

    assert plan["status"] == "would_import" and plan["run_name"] == RUN
    assert plan["camera_serial"] == SERIALS[0] and plan["existing_background_runs"] == []
    assert _tree(recording_dir / "zarr") == before


def test_apply_writes_the_run_and_the_consolidated_view_selects_it(tmp_path: Path) -> None:
    images = {serial: _image(index) for index, serial in enumerate(SERIALS)}
    recording_dir = _published(tmp_path, images=images)

    result = backfill.apply_recording(recording_dir)

    assert result["status"] == "imported" and result["applied"] is True
    assert result["selected_as_latest"] is True
    # A reader of the published archive (consolidated metadata) sees the run.
    root = zarr.open_group(str(recording_dir / "zarr" / "rec_analysis.zarr"), mode="r", use_consolidated=True)
    assert root.attrs["marker"] == "published" and "raw_video" in root
    assert root["background_runs"].attrs["latest"] == RUN
    assert np.array_equal(root["background_runs"][RUN]["background_full"][:], images[SERIALS[0]])
    assert backfill.plan_recording(recording_dir)["status"] == "already_imported"


def test_applying_twice_keeps_one_run(tmp_path: Path) -> None:
    recording_dir = _published(tmp_path)
    backfill.apply_recording(recording_dir)

    again = backfill.apply_recording(recording_dir)

    assert again["status"] == "already_imported" and again["applied"] is False
    root = zarr.open_group(str(recording_dir / "zarr" / "rec_analysis.zarr"), mode="r", use_consolidated=True)
    assert list(root["background_runs"].group_keys()) == [RUN]


def test_a_capture_that_breaks_its_contract_leaves_the_archive_untouched(tmp_path: Path) -> None:
    recording_dir = _published(tmp_path)
    image_path = (
        recording_dir / "raw" / "acquisition" / "empty_background" / f"Cam{SERIALS[0]}_empty_background.png"
    )
    image_path.write_bytes(image_path.read_bytes()[:-1] + b"\x00")
    before = _tree(recording_dir / "zarr")

    result = backfill.apply_recording(recording_dir)

    assert result["status"] == "refused" and result["applied"] is False
    assert result["code"] == "empty_background_image_sha256_mismatch"
    assert _tree(recording_dir / "zarr") == before


def test_a_recording_sealed_without_a_start_snapshot_is_left_untouched(tmp_path: Path) -> None:
    recording_dir = _published(tmp_path, seal_snapshot=False)
    before = _tree(recording_dir / "zarr")

    result = backfill.apply_recording(recording_dir)

    assert result["status"] == "no_sealed_start_snapshot" and result["applied"] is False
    assert _tree(recording_dir / "zarr") == before


def test_a_run_of_that_name_with_another_image_is_a_conflict(tmp_path: Path) -> None:
    recording_dir = _published(tmp_path)
    root = zarr.open_group(str(recording_dir / "zarr" / "rec_analysis.zarr"), mode="r+", use_consolidated=False)
    other = root.create_group("background_runs").create_group(RUN)
    other.attrs["orange_empty_background"] = {"image_sha256": "sha256:" + "0" * 64}

    assert backfill.plan_recording(recording_dir)["status"] == "conflict"
    with pytest.raises(ValueError, match="holds a different image"):
        backfill.apply_recording(recording_dir)


def test_the_command_is_a_dry_run_unless_apply_is_given(tmp_path: Path, capsys) -> None:
    recording_dir = _published(tmp_path)
    before = _tree(recording_dir / "zarr")

    assert backfill.main([str(recording_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run" and report["recordings"][0]["status"] == "would_import"
    assert _tree(recording_dir / "zarr") == before

    assert backfill.main([str(recording_dir), "--apply"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "apply" and report["recordings"][0]["status"] == "imported"


def test_one_unreadable_recording_is_reported_and_the_rest_still_run(tmp_path: Path, capsys) -> None:
    recording_dir = _published(tmp_path)
    missing = tmp_path / "absent"

    assert backfill.main([str(missing), str(recording_dir)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in report["recordings"]] == ["error", "would_import"]
