from __future__ import annotations

import json

import pytest

from fisheye.shared.selected_calibration import (
    build_selected_camera_source_evidence_from_h5_values,
)
from fisheye.shared.source_camera_physical_authority import (
    BoundSourceCameraPhysicalAuthority,
    SourceCameraPhysicalAuthorityError,
    load_source_camera_physical_authority,
    publish_source_camera_physical_authority,
)
from tests.unit.fisheye.test_directed_transform_chain import FakeGroup, _world


def _recording_world():
    world = _world(convention="continuous", archive_token=object())
    world["root"]["analysis"]["calibration"] = FakeGroup(
        path="analysis/calibration",
        archive_token=world["root"]._coordinate_archive_token,
    )
    return world


def test_recording_calibration_publishes_reloadable_common_authority() -> None:
    world = _recording_world()

    published = publish_source_camera_physical_authority(
        world["root"],
        source_camera_evidence=world["camera_evidence"],
        source_kind="recording_calibration",
        provenance={"operator_verified": True, "donor": "camera-a"},
    )
    loaded = load_source_camera_physical_authority(world["root"])

    assert published.camera_id == "camera-a"
    assert published.source_kind == "recording_calibration"
    assert published.mm_per_pixel == pytest.approx(1.0 / 25.0)
    assert loaded.manifest.record_sha256 == published.manifest.record_sha256
    assert loaded.physical_frame.record_sha256 == published.physical_frame.record_sha256


def test_common_authority_cannot_be_constructed_directly() -> None:
    with pytest.raises(
        SourceCameraPhysicalAuthorityError, match="cannot be constructed"
    ):
        BoundSourceCameraPhysicalAuthority(
            camera_id="camera-a",
            source_kind="recording_calibration",
            archive_identity=None,  # type: ignore[arg-type]
            physical_frame=None,  # type: ignore[arg-type]
            manifest=None,  # type: ignore[arg-type]
            root=None,
        )


def test_common_authority_rejects_conflicting_republication() -> None:
    world = _recording_world()
    publish_source_camera_physical_authority(
        world["root"],
        source_camera_evidence=world["camera_evidence"],
        source_kind="recording_calibration",
        provenance={"operator_verified": True},
    )
    arena = {
        "active_camera_id": "camera-a",
        "calculated_z_eff_mm": 20.0,
        "camera_calibrations": [
            {
                "camera_id": "camera-a",
                "native_width_px": 100,
                "native_height_px": 80,
                "pixels_per_mm_camera": 30.0,
                "pixels_per_mm_projector": 4.0,
                "real_world_ref_mm": 10.0,
            }
        ],
    }
    conflicting = build_selected_camera_source_evidence_from_h5_values(
        source_h5_path="/tmp/conflicting.h5",
        arena_config_raw=json.dumps(arena),
        camera_group_path="/calibration_snapshot/camera-a",
        camera_group_attrs={
            "pixels_per_mm_camera": 30.0,
            "pixels_per_mm_projector": 4.0,
            "real_world_ref_mm": 10.0,
        },
        expected_camera_id="camera-a",
    )

    with pytest.raises(SourceCameraPhysicalAuthorityError, match="conflicts"):
        publish_source_camera_physical_authority(
            world["root"],
            source_camera_evidence=conflicting,
            source_kind="recording_calibration",
            provenance={"operator_verified": True},
        )
