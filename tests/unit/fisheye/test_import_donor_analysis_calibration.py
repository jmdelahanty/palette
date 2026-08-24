from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.selected_calibration import (
    build_selected_camera_source_evidence_from_h5_values,
)
from fisheye.shared.source_camera_physical_authority import (
    load_source_camera_physical_authority,
    publish_source_camera_physical_authority,
)
from fisheye.utils.import_donor_analysis_calibration import (
    MODULE_NAME,
    import_donor_calibration,
)


def _source_video_metadata(camera: str) -> dict[str, object]:
    return {
        "schema_id": "palette.source_video_metadata.v2",
        "layout": "single_video",
        "camera_id": camera,
        "source_path": f"/recording/cams/{camera}.mp4",
        "width": 4512,
        "height": 4512,
        "total_frames": 2,
        "locator": {
            "kind": "recording_relative",
            "relative_path": f"cams/{camera}.mp4",
        },
        "file_fingerprint": {
            "strategy": "size_mtime_sha256_v1",
            "value": "a" * 64,
            "size_bytes": 1234,
            "mtime_ns": 5678,
            "relocation_stable": False,
        },
    }


def _stamp_external_acquisition(root: zarr.Group, *, camera: str) -> None:
    root.attrs["source_video_metadata"] = _source_video_metadata(camera)
    authority = root.require_group(f"analysis/acquisition_camera_frames/{camera}")
    ownership = stamp_acquisition_import_ownership(root, authority)
    stamp_acquisition_camera_frame(
        root,
        authority,
        import_ownership=ownership,
    )
    _, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=camera,
    )
    source = root.require_group(
        f"analysis/coordinate_frames/source_camera/{camera}/continuous"
    )
    stamp_source_camera_pixel_frame_authority(
        source,
        frame_id=f"{camera}_source_camera",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )


def _write_target(path: Path, *, camera: str = "2010093") -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"recording_id": "target", "camera_serials": [camera]})
    root.require_group("analysis")
    _stamp_external_acquisition(root, camera=camera)
    recording_dir = path.parent.parent
    geometry = recording_dir / "raw/recording_geometry_bundle/recording_snapshot.json"
    geometry.parent.mkdir(parents=True)
    geometry.write_text(
        json.dumps({"camera_runtime": {camera: {"width": 4512, "height": 4512}}}),
        encoding="utf-8",
    )
    return path


def _write_source_h5(path: Path, *, camera: str) -> tuple[bytes, dict[str, float]]:
    arena = {
        "active_camera_id": camera,
        "calculated_z_eff_mm": 20.0,
        "camera_calibrations": [
            {
                "camera_id": camera,
                "native_width_px": 4512,
                "native_height_px": 4512,
                "pixels_per_mm_camera": 50.0,
                "pixels_per_mm_projector": 4.0,
                "real_world_ref_mm": 10.0,
            }
        ],
    }
    raw = json.dumps(arena, sort_keys=True).encode("utf-8")
    attrs = {
        "pixels_per_mm_camera": 50.0,
        "pixels_per_mm_projector": 4.0,
        "real_world_ref_mm": 10.0,
    }
    with h5py.File(path, "w") as handle:
        snapshot = handle.create_group("calibration_snapshot")
        snapshot.create_dataset("arena_config_json", data=raw)
        camera_group = snapshot.create_group(camera)
        camera_group.attrs.update(attrs)
    return raw, attrs


def _write_donor(path: Path, *, camera: str = "2010093") -> Path:
    source_h5 = path.with_suffix(".h5")
    arena_raw, camera_attrs = _write_source_h5(source_h5, camera=camera)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"recording_id": "donor", "camera_serials": [camera]})
    root.require_group("analysis")
    _stamp_external_acquisition(root, camera=camera)
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
    evidence = build_selected_camera_source_evidence_from_h5_values(
        source_h5_path=str(source_h5.resolve()),
        arena_config_raw=arena_raw,
        camera_group_path=f"/calibration_snapshot/{camera}",
        camera_group_attrs=camera_attrs,
        expected_camera_id=camera,
    )
    publish_source_camera_physical_authority(
        root,
        source_camera_evidence=evidence,
        source_kind="stimulus_h5_calibration_snapshot",
        provenance={"source_h5": str(source_h5.resolve())},
    )
    return path


def _copy_as_legacy_import(target: Path, donor: Path) -> None:
    target_calibration = target / "analysis/calibration"
    shutil.copytree(donor / "analysis/calibration", target_calibration)
    target_root = zarr.open_group(str(target), mode="r+", use_consolidated=False)
    target_root["analysis/calibration"].attrs.update(
        {
            "immediate_donor_zarr": str(donor.resolve()),
            "operator_configuration_verified": True,
            "imported_by": MODULE_NAME,
        }
    )


def test_import_donor_calibration_rebinds_complete_group_to_target(
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
    assert result["consolidated_metadata_validated"] is True
    root = zarr.open_group(str(target), mode="r", use_consolidated=False)
    donor_root = zarr.open_group(str(donor), mode="r", use_consolidated=False)
    calibration = root["analysis/calibration"]
    target_authority = load_source_camera_physical_authority(root)
    donor_authority = load_source_camera_physical_authority(donor_root)
    assert calibration.attrs["active_camera_id"] == "2010093"
    assert calibration.attrs["immediate_donor_zarr"] == str(donor.resolve())
    assert calibration.attrs["operator_configuration_verified"] is True
    assert target_authority.source_kind == "operator_verified_donor_calibration"
    assert (
        target_authority.physical_frame.source_camera_pixels.record_sha256
        != donor_authority.physical_frame.source_camera_pixels.record_sha256
    )
    np.testing.assert_array_equal(calibration["homography_matrix"][:], np.eye(3))
    assert Path(result["receipt_path"]).is_file()


def test_repair_existing_rebinds_legacy_copied_authority(tmp_path: Path) -> None:
    target = _write_target(tmp_path / "recording/zarr/target_analysis.zarr")
    donor = _write_donor(tmp_path / "donor_analysis.zarr")
    _copy_as_legacy_import(target, donor)

    with pytest.raises(ValueError, match="physical_authority_mismatch"):
        load_source_camera_physical_authority(
            zarr.open_group(str(target), mode="r", use_consolidated=False)
        )

    planned = import_donor_calibration(
        target,
        donor,
        expected_camera="2010093",
        operator_note="same camera-specific configuration",
        repair_existing=True,
    )
    assert planned["status"] == "planned"
    assert planned["target_authority_status"] == "requires_rebind"

    result = import_donor_calibration(
        target,
        donor,
        expected_camera="2010093",
        operator_note="same camera-specific configuration",
        apply=True,
        repair_existing=True,
    )
    root = zarr.open_group(str(target), mode="r", use_consolidated=False)
    authority = load_source_camera_physical_authority(root)
    assert result["status"] == "pass"
    assert authority.source_kind == "operator_verified_donor_calibration"
    assert (
        root["analysis/calibration"].attrs["physical_authority_repaired_by"]
        == MODULE_NAME
    )
    assert Path(result["receipt_path"]).is_file()


def test_repair_existing_rejects_unapproved_target_calibration(tmp_path: Path) -> None:
    target = _write_target(tmp_path / "recording/zarr/target_analysis.zarr")
    donor = _write_donor(tmp_path / "donor_analysis.zarr")
    _copy_as_legacy_import(target, donor)
    root = zarr.open_group(str(target), mode="r+", use_consolidated=False)
    root["analysis/calibration"].attrs["operator_configuration_verified"] = False

    with pytest.raises(ValueError, match="was not operator verified"):
        import_donor_calibration(
            target,
            donor,
            expected_camera="2010093",
            operator_note="same physical rig",
            repair_existing=True,
        )


def test_import_donor_calibration_rejects_camera_mismatch(tmp_path: Path) -> None:
    target = _write_target(tmp_path / "recording/zarr/target_analysis.zarr")
    donor = _write_donor(tmp_path / "donor_analysis.zarr", camera="2010094")

    with pytest.raises(
        ValueError, match="donor physical authority names another camera"
    ):
        import_donor_calibration(
            target,
            donor,
            expected_camera="2010093",
            operator_note="same physical rig",
        )
