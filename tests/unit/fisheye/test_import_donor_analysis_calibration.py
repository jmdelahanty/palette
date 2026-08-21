from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.utils.import_donor_analysis_calibration import (
    import_donor_calibration,
)


def _write_target(path: Path, *, camera: str = "2010093") -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"recording_id": "target", "camera_serials": [camera]})
    root.require_group("analysis")
    recording_dir = path.parent.parent
    geometry = recording_dir / "raw/recording_geometry_bundle/recording_snapshot.json"
    geometry.parent.mkdir(parents=True)
    geometry.write_text(
        json.dumps({"camera_runtime": {camera: {"width": 4512, "height": 4512}}}),
        encoding="utf-8",
    )
    return path


def _write_donor(path: Path, *, camera: str = "2010093") -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "donor"
    calibration = root.require_group("analysis/calibration")
    calibration.attrs.update(
        {
            "active_camera_id": camera,
            "primary_camera_id": camera,
            "pixels_per_mm_camera": 50.0,
            "pixel_to_mm": 0.02,
            "pixels_per_mm_projector": 4.0,
            "native_width_px": 4512,
            "native_height_px": 4512,
        }
    )
    calibration.create_array("homography_matrix", data=np.eye(3))
    physical = calibration.require_group("coordinate_frames/source_camera_physical_mm")
    physical.attrs["physical_frame_calibration"] = {"camera_id": camera}
    return path


def test_import_donor_calibration_is_dry_run_then_copies_complete_group(
    tmp_path: Path,
) -> None:
    target = _write_target(tmp_path / "recording/zarr/target_analysis.zarr")
    donor = _write_donor(tmp_path / "donor_analysis.zarr")

    planned = import_donor_calibration(
        target,
        donor,
        expected_camera="2010093",
        operator_note="same physical rig",
    )
    assert planned["status"] == "planned"
    assert not (target / "analysis/calibration").exists()

    result = import_donor_calibration(
        target,
        donor,
        expected_camera="2010093",
        operator_note="same physical rig",
        apply=True,
    )

    assert result["status"] == "pass"
    root = zarr.open_group(str(target), mode="r", use_consolidated=False)
    calibration = root["analysis/calibration"]
    assert calibration.attrs["active_camera_id"] == "2010093"
    assert calibration.attrs["immediate_donor_zarr"] == str(donor.resolve())
    assert calibration.attrs["operator_configuration_verified"] is True
    np.testing.assert_array_equal(calibration["homography_matrix"][:], np.eye(3))
    assert Path(result["receipt_path"]).is_file()


def test_import_donor_calibration_rejects_camera_mismatch(tmp_path: Path) -> None:
    target = _write_target(tmp_path / "recording/zarr/target_analysis.zarr")
    donor = _write_donor(tmp_path / "donor_analysis.zarr", camera="2010094")

    with pytest.raises(ValueError, match="active_camera_id mismatch"):
        import_donor_calibration(
            target,
            donor,
            expected_camera="2010093",
            operator_note="same physical rig",
        )
