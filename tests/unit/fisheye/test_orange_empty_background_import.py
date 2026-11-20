"""Orange's empty-dish background stored as a background run: what is written, what is recorded."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.preprocessing import orange_empty_background_import as ingest
from fisheye.shared.zarr_run_completion import (
    resolve_authoritative_run_name,
    resolve_latest_complete_run_name,
)
from fisheye.utils import import_recording_analysis as importer

from tests.unit.fisheye.test_orange_empty_background_loader import SERIALS, _image, _recording

CAPTURE_ID = "2026_11_02_15_04_11"
RUN = f"background_orange_empty_{CAPTURE_ID}"


def _root(tmp_path: Path) -> zarr.Group:
    return zarr.open_group(
        str(tmp_path / "rec_analysis.zarr"), mode="w", zarr_format=3, use_consolidated=False
    )


def _acquisition(tmp_path: Path, **kwargs) -> Path:
    folder = tmp_path / "rec" / "raw" / "acquisition"
    folder.mkdir(parents=True)
    return _recording(folder, **kwargs)


def test_an_attached_capture_becomes_the_latest_background_run(tmp_path: Path) -> None:
    images = {serial: _image(index) for index, serial in enumerate(SERIALS)}
    acquisition_dir = _acquisition(tmp_path, images=images)
    root = _root(tmp_path)

    record = ingest.import_orange_empty_background(
        root, acquisition_dir=acquisition_dir, camera_serial=SERIALS[1]
    )

    assert record["status"] == "imported" and record["run_name"] == RUN
    parent = root["background_runs"]
    assert resolve_latest_complete_run_name(parent, legacy_default=False) == RUN
    assert resolve_authoritative_run_name(parent) == RUN
    group = parent[RUN]
    # The named camera's pixels, at the native raster, and nothing downsampled.
    assert np.array_equal(group["background_full"][:], images[SERIALS[1]])
    assert group["background_full"].dtype == np.uint8
    assert "background_ds" not in group
    attrs = dict(group.attrs)
    assert attrs["method"] == "orange_empty_background"
    assert attrs["parameters"]["method"] == "orange_empty_background"
    assert attrs["background_source"] == "pre_recording_empty_dish_capture"
    capture = attrs["orange_empty_background"]
    assert capture["capture_id"] == CAPTURE_ID and capture["camera_serial"] == SERIALS[1]
    png = (acquisition_dir / "empty_background" / f"Cam{SERIALS[1]}_empty_background.png").read_bytes()
    assert capture["image_sha256"] == f"sha256:{hashlib.sha256(png).hexdigest()}"
    assert capture["camera_configuration"]["exposure_us"] == 150
    assert dict(root.attrs)[ingest.RECORD_ATTR] == record


def test_importing_the_same_capture_again_writes_nothing_new(tmp_path: Path) -> None:
    acquisition_dir = _acquisition(tmp_path)
    root = _root(tmp_path)
    kwargs = {"acquisition_dir": acquisition_dir, "camera_serial": SERIALS[0]}
    ingest.import_orange_empty_background(root, **kwargs)

    again = ingest.import_orange_empty_background(root, **kwargs)

    assert again["status"] == "already_imported"
    assert list(root["background_runs"].group_keys()) == [RUN]


def test_a_run_of_that_name_with_other_pixels_is_not_overwritten(tmp_path: Path) -> None:
    acquisition_dir = _acquisition(tmp_path)
    root = _root(tmp_path)
    kwargs = {"acquisition_dir": acquisition_dir, "camera_serial": SERIALS[0]}
    ingest.import_orange_empty_background(root, **kwargs)
    root["background_runs"][RUN]["background_full"][0, 0] = 7

    with pytest.raises(ValueError, match="exists and is not this capture"):
        ingest.import_orange_empty_background(root, **kwargs)


def test_a_capture_that_breaks_its_contract_is_recorded_and_writes_no_run(tmp_path: Path) -> None:
    acquisition_dir = _acquisition(tmp_path)
    image_path = acquisition_dir / "empty_background" / f"Cam{SERIALS[0]}_empty_background.png"
    image_path.write_bytes(image_path.read_bytes()[:-1] + b"\x00")
    root = _root(tmp_path)

    record = ingest.import_orange_empty_background(
        root, acquisition_dir=acquisition_dir, camera_serial=SERIALS[0]
    )

    assert record["status"] == "refused"
    assert record["code"] == "empty_background_image_sha256_mismatch"
    assert "background_runs" not in root
    assert dict(root.attrs)[ingest.RECORD_ATTR]["code"] == record["code"]


def test_changed_camera_settings_are_recorded_and_write_no_run(tmp_path: Path) -> None:
    def edit(descriptor, block):
        block["cameras"][SERIALS[0]].update(
            status="settings_changed",
            settings_differences=[{"setting": "iris", "capture_value": 19, "recording_value": 20}],
        )

    root = _root(tmp_path)
    record = ingest.import_orange_empty_background(
        root, acquisition_dir=_acquisition(tmp_path, edit=edit), camera_serial=SERIALS[0]
    )

    assert (record["status"], record["code"]) == ("refused", "empty_background_settings_changed")
    assert "background_runs" not in root


def test_a_recording_without_a_capture_records_that_and_writes_no_run(tmp_path: Path) -> None:
    root = _root(tmp_path)
    record = ingest.import_orange_empty_background(
        root, acquisition_dir=tmp_path, camera_serial=SERIALS[0], start_snapshot={"session": {}}
    )

    assert record["status"] == "undeclared"
    assert "background_runs" not in root


def _plan(tmp_path: Path, *, seal_snapshot: bool = True, tamper: bool = False):
    recording_dir = tmp_path / "rec"
    acquisition_dir = _acquisition(tmp_path)
    snapshot = acquisition_dir / "recording_snapshot_start.json"
    data = snapshot.read_bytes()
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
        json.dumps(
            {"camera_id": SERIALS[0], "source_transfer": {"source_to_parent_files": files}}
        )
    )
    if tamper:
        snapshot.write_bytes(data + b" ")
    _root(tmp_path)
    return importer.RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=None,
        cam_video=None,
        zarr_path=tmp_path / "rec_analysis.zarr",
    )


def test_recording_import_stores_the_manifest_cameras_background(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    record = importer.import_empty_background(plan)

    assert record["status"] == "imported" and record["camera_serial"] == SERIALS[0]
    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
    assert np.array_equal(root["background_runs"][RUN]["background_full"][:], _image(0))


def test_recording_import_skips_a_recording_with_no_sealed_start_snapshot(tmp_path: Path) -> None:
    plan = _plan(tmp_path, seal_snapshot=False)

    assert importer.import_empty_background(plan) is None
    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
    assert "background_runs" not in root and ingest.RECORD_ATTR not in dict(root.attrs)


def test_recording_import_fails_when_the_start_snapshot_differs_from_its_seal(tmp_path: Path) -> None:
    plan = _plan(tmp_path, tamper=True)

    with pytest.raises(ValueError, match="differs from the sealed inventory"):
        importer.import_empty_background(plan)
