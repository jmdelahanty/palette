from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from fisheye.shared.unified_h5.geometry import NAMESPACE, NAMESPACE_RECORD

from fisheye.shared.recording_geometry import (
    CitrusRegistrationStatus,
    GeometryLoadPolicy,
    MaskGeometryStatus,
    MaterializedAssetStatus,
    RecordingGeometryError,
    bind_registered_dish_mask_to_source_camera_frame,
    load_registered_dish_masks_from_citrus_h5,
    load_registered_dish_masks_from_recording_folder,
    verify_recording_geometry_bundle,
)
from fisheye.shared.recording_geometry_recovery import (
    build_recording_geometry_recovery_receipt,
    load_registered_dish_mask_from_recovery_receipt,
    publish_recording_geometry_recovery,
    validate_recording_geometry_recovery_receipt,
)


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: object, *, newline: bool = False) -> bytes:
    result = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return result + (b"\n" if newline else b"")




def _rim_entry(
    *,
    camera: str = "2010093",
    arena: str = "arena_1",
    source_sha: str = "sha256:" + "a" * 64,
    source_path: str = "/orange/immutable/observation.json",
    gate_radius: float = 205.0,
    rim_version: int = 2,
) -> dict[str, object]:
    return {
        "artifact_schema_id": "orange.calibration.dish_top_rim_observation",
        "artifact_schema_version": rim_version,
        "artifact_id": f"dishrim-{camera}",
        "camera_serial": camera,
        "arena_id": arena,
        "coordinate_space": "camera_native_pixels",
        "available_for_downstream_detection_gating": True,
        "active_in_orange_live_detection_pipeline": False,
        "gating_semantics": "bounding_box_centroid_inside_valid_detection_region",
        "camera": {
            "serial": camera,
            "name": f"Cam{camera}",
            "width": 640,
            "height": 480,
        },
        "accepted_inner_rim_boundary": {
            "coordinate_space": "camera_native_pixels",
            "target_plane": "dish_top_rim",
            "geometry": {
                "type": "circle",
                "center_px": {"x": 320.25, "y": 240.5},
                "radius_px": 200.0,
            },
        },
        "accepted_mask": {
            "image_shape_px": {"width": 640, "height": 480},
        },
        "valid_detection_region": {
            "coordinate_space": "camera_native_pixels",
            "purpose": "bounding_box_centroid_detection_gating",
            "offset_direction": "outward",
            "geometry": {
                "type": "circle",
                "center_px": {"x": 320.25, "y": 240.5},
                "radius_px": gate_radius,
            },
        },
        "operator_review": {"accepted": True},
        "quality": {"quality_flags": []},
        "source": {"path": source_path, "sha256": source_sha},
    }


def _daily_camera(entry: dict[str, object]) -> dict[str, object]:
    return {
        "status": "resolved",
        "arena_id": entry["arena_id"],
        "camera_serial": entry["camera_serial"],
        "recording_snapshot_entry": entry,
    }


def _write_folder_bundle(
    root: Path,
    *,
    include_pointer: bool = True,
    gate_radius: float = 205.0,
    contract_version: int = 1,
    assets_version: int | None = None,
    rim_version: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    # Contract v2 (Orange bb6c4f5) binds assets v2 and rim observation v3.
    assets_version = contract_version if assets_version is None else assets_version
    rim_version = contract_version + 1 if rim_version is None else rim_version
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "recording_geometry_assets"
    observation_rel = (
        "cameras/Cam2010093/daily_registration/rim_observation/observation.json"
    )
    observation_path = assets / observation_rel
    observation_path.parent.mkdir(parents=True)
    observation_bytes = b'{"schema_id":"orange.calibration.dish_top_rim_observation"}\n'
    observation_path.write_bytes(observation_bytes)
    observation_sha = _sha(observation_bytes)

    entry = _rim_entry(source_sha=observation_sha, gate_radius=gate_radius, rim_version=rim_version)
    manifest = {
        "schema_id": "orange.recording.geometry_assets",
        "schema_version": assets_version,
        "status": "complete",
        "materialized_file_count": 1,
        "files": [
            {
                "relative_path": observation_rel,
                "role": "daily_rim_observation",
                "required": True,
                "size_bytes": len(observation_bytes),
                "sha256": observation_sha,
            }
        ],
    }
    manifest_bytes = _json_bytes(manifest)
    (assets / "manifest.json").write_bytes(manifest_bytes)

    registration_id = "dailyreg-1"
    contract = {
        "schema_id": "orange.recording.geometry_contract",
        "schema_version": contract_version,
        "status": "resolved",
        "selection": {
            "rig_id": "omnifin0",
            "selected_canvas_name": "shadow",
        },
        "daily_registration_geometry": {
            "mode": "selected_daily_registration",
            "status": "selected_resolved",
            "registration": {
                "sha256": "sha256:" + "b" * 64,
                "snapshot": {
                    "registration_id": registration_id,
                    "valid_until_utc": "2026-07-23T04:00:00Z",
                },
            },
            "cameras": {"2010093": _daily_camera(entry)},
        },
        "materialized_assets": {
            "schema_id": "orange.recording.geometry_assets",
            "schema_version": assets_version,
            "status": "complete",
            "relative_path": "recording_geometry_assets/manifest.json",
            "sha256": _sha(manifest_bytes),
        },
    }
    contract_bytes = _json_bytes(contract, newline=True)
    (root / "recording_geometry_contract.json").write_bytes(contract_bytes)
    snapshot: dict[str, object] = {
        "schema_version": 2,
        "recording_id": "orange-session-1",
        "camera_runtime": {
            "2010093": {
                "coordinate_frame": {
                    "coordinate_space": "camera_native_pixels",
                    "point_order": "xy",
                    "units": "pixels",
                    "origin": {"name": "top_left_pixel", "x_px": 0, "y_px": 0},
                    "axes": {
                        "x": {"positive_direction": "right"},
                        "y": {"positive_direction": "down"},
                    },
                    "image_shape": {"width": 640, "height": 480},
                }
            }
        },
        "calibrations": {
            "2010093": {
                "dish_top_rim_observation": {
                    "artifact_id": "dishrim-2010093",
                    "sha256": observation_sha,
                }
            }
        },
    }
    if include_pointer:
        snapshot["recording_geometry_contract"] = {
            "relative_path": "recording_geometry_contract.json",
            "sha256": _sha(contract_bytes),
        }
    (root / "recording_snapshot.json").write_bytes(_json_bytes(snapshot))
    return contract, snapshot


def test_folder_loader_returns_normalized_full_precision_mask(tmp_path: Path) -> None:
    _write_folder_bundle(tmp_path)

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID
    assert result.enclosing_selection_status == "selected_resolved"
    assert len(result.masks) == 1
    mask = next(iter(result.masks.values()))
    assert mask.key.camera_serial == "2010093"
    assert mask.key.arena_id == "arena_1"
    assert mask.physical_inner_rim.radius_px == 200.0
    assert mask.valid_detection_gate.radius_px == 205.0
    assert mask.materialized_asset_status is MaterializedAssetStatus.COMPLETE
    assert mask.palette_space_id == "source_camera_image_px"
    assert mask.coordinate_profile_id == "source_camera_image_px.top_left_y_down.v1"
    assert mask.pixel_convention == "continuous"


def test_folder_loader_accepts_embedded_snapshot_rim_source_checksum(
    tmp_path: Path,
) -> None:
    _contract, snapshot = _write_folder_bundle(tmp_path)
    calibrations = snapshot["calibrations"]
    assert isinstance(calibrations, dict)
    camera = calibrations["2010093"]
    assert isinstance(camera, dict)
    rim = camera["dish_top_rim_observation"]
    assert isinstance(rim, dict)
    checksum = rim.pop("sha256")
    rim["source"] = {"sha256": checksum}
    (tmp_path / "recording_snapshot.json").write_bytes(_json_bytes(snapshot))

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID
    assert len(result.masks) == 1


def test_folder_loader_rejects_conflicting_snapshot_rim_checksums(
    tmp_path: Path,
) -> None:
    _contract, snapshot = _write_folder_bundle(tmp_path)
    calibrations = snapshot["calibrations"]
    assert isinstance(calibrations, dict)
    camera = calibrations["2010093"]
    assert isinstance(camera, dict)
    rim = camera["dish_top_rim_observation"]
    assert isinstance(rim, dict)
    rim["source"] = {"sha256": "sha256:" + "0" * 64}
    (tmp_path / "recording_snapshot.json").write_bytes(_json_bytes(snapshot))

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert "checksum fields disagree" in result.issues[0].message


def test_mask_binds_only_to_matching_palette_source_camera_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_folder_bundle(tmp_path)
    mask = next(
        iter(load_registered_dish_masks_from_recording_folder(tmp_path).masks.values())
    )
    verified: list[bool] = []
    frame = SimpleNamespace(
        endpoint=SimpleNamespace(
            space_id="source_camera_image_px",
            pixel_convention="continuous",
            units="px",
            width=640,
            height=480,
        ),
        reference_extent=SimpleNamespace(record=SimpleNamespace(camera_id="2010093")),
        record_ref="/analysis_metadata/coordinate_frames/source_camera@pixel_frame_authority",
        record_sha256="a" * 64,
        assert_verified=lambda: verified.append(True),
    )
    monkeypatch.setattr(
        "fisheye.shared.pixel_frame_authority.require_source_camera_pixel_frame_authority",
        lambda value: value,
    )

    bound = bind_registered_dish_mask_to_source_camera_frame(mask, frame)

    assert verified == [True]
    assert bound.pixel_frame_record_sha256 == "a" * 64
    frame.endpoint.width = 641
    with pytest.raises(RecordingGeometryError, match="dimensions"):
        bind_registered_dish_mask_to_source_camera_frame(mask, frame)


def test_folder_loader_rejects_contract_path_traversal(tmp_path: Path) -> None:
    _contract, snapshot = _write_folder_bundle(tmp_path)
    snapshot["recording_geometry_contract"] = {
        "relative_path": "../recording_geometry_contract.json",
        "sha256": "sha256:" + "0" * 64,
    }
    (tmp_path / "recording_snapshot.json").write_bytes(_json_bytes(snapshot))

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert result.masks == {}
    assert "escapes" in result.issues[0].message


def test_bundle_verifier_preserves_early_bundle_without_snapshot_pointer(tmp_path: Path) -> None:
    _write_folder_bundle(tmp_path, include_pointer=False)

    verification = verify_recording_geometry_bundle(tmp_path)
    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert verification.snapshot_pointer_status == "missing"
    assert verification.manifest_file_count == 1
    assert result.mask_geometry_status is MaskGeometryStatus.LEGACY_MISSING


def _write_recovery_target_h5(
    path: Path,
    *,
    camera: str = "2010093",
    arena: str = "arena_1",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["session_uuid"] = f"2026-07-22T15-44-40Z_{arena}"
        h5.attrs["arena_id"] = arena
        h5.attrs["ipc_source_name"] = f"/shm_cam_{camera}"
        group = h5.create_group("recording_geometry_contract")
        group.attrs["capture_status"] = "not_referenced"
        group.attrs["checksum_verified"] = 0


def test_recovery_receipt_is_explicit_immutable_and_revalidated(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    _write_folder_bundle(source, include_pointer=False)
    recording = tmp_path / "recordings" / "recording-1"
    target_h5 = recording / "raw" / "recording-1.h5"
    _write_recovery_target_h5(target_h5)
    original_snapshot_sha = _sha((source / "recording_snapshot.json").read_bytes())

    planned = build_recording_geometry_recovery_receipt(
        bundle_root=source,
        target_h5_path=target_h5,
        approved_by="test-operator",
        created_at_utc="2026-07-26T12:00:00+00:00",
    )
    assert planned["claims"]["producer_declared_snapshot_contract_link"] is False

    publication = publish_recording_geometry_recovery(
        source_bundle_root=source,
        recording_root=recording,
        target_h5_path=target_h5,
        approved_by="test-operator",
    )
    assert publication.bundle_publication.published is True
    assert publication.receipt_published is True
    assert _sha((source / "recording_snapshot.json").read_bytes()) == original_snapshot_sha
    assert load_registered_dish_masks_from_recording_folder(
        recording / "raw/recording_geometry_bundle"
    ).mask_geometry_status is MaskGeometryStatus.LEGACY_MISSING

    verified = validate_recording_geometry_recovery_receipt(publication.receipt_path)
    recovered = load_registered_dish_mask_from_recovery_receipt(publication.receipt_path)
    mask = next(iter(recovered.masks.values()))
    assert verified.evidence.camera_serial == "2010093"
    assert mask.producer_contract_linkage_status == "operator_approved_recovery_receipt"
    assert mask.recovery_receipt_sha256 == verified.receipt_sha256
    assert mask.independent_fit_required_before_operational_use is True
    assert mask.selected_daily_registration_applied_by_citrus is False

    repeated = publish_recording_geometry_recovery(
        source_bundle_root=source,
        recording_root=recording,
        target_h5_path=target_h5,
        approved_by="test-operator",
    )
    assert repeated.bundle_publication.published is False
    assert repeated.receipt_published is False

    relocated = recording.with_name("relocated-recording")
    recording.rename(relocated)
    relocated_verified = validate_recording_geometry_recovery_receipt(
        relocated / "raw/recording_geometry_recovery.json"
    )
    assert relocated_verified.evidence.session_uuid == verified.evidence.session_uuid


def test_recovery_receipt_fails_after_target_h5_changes(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    _write_folder_bundle(source, include_pointer=False)
    recording = tmp_path / "recordings" / "recording-1"
    target_h5 = recording / "raw" / "recording-1.h5"
    _write_recovery_target_h5(target_h5)
    publication = publish_recording_geometry_recovery(
        source_bundle_root=source,
        recording_root=recording,
        target_h5_path=target_h5,
        approved_by="test-operator",
    )
    with h5py.File(target_h5, "a") as h5:
        h5.attrs["arena_id"] = "arena_2"

    with pytest.raises(RecordingGeometryError, match="arena|checksum"):
        validate_recording_geometry_recovery_receipt(publication.receipt_path)


def test_recovery_refuses_wrong_camera_or_producer_native_link(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    _write_folder_bundle(source, include_pointer=False)
    target_h5 = tmp_path / "recording" / "raw" / "recording.h5"
    _write_recovery_target_h5(target_h5, camera="2010094", arena="arena_1")
    with pytest.raises(RecordingGeometryError, match=r"cameras\[2010094\]"):
        build_recording_geometry_recovery_receipt(
            bundle_root=source,
            target_h5_path=target_h5,
            approved_by="test-operator",
        )

    native = tmp_path / "native"
    _write_folder_bundle(native, include_pointer=True)
    _write_recovery_target_h5(target_h5)
    with pytest.raises(RecordingGeometryError, match="pointer is missing"):
        build_recording_geometry_recovery_receipt(
            bundle_root=native,
            target_h5_path=target_h5,
            approved_by="test-operator",
        )


def test_bundle_verifier_rejects_tampered_asset(tmp_path: Path) -> None:
    _write_folder_bundle(tmp_path)
    observation = (
        tmp_path
        / "recording_geometry_assets/cameras/Cam2010093/daily_registration/"
        "rim_observation/observation.json"
    )
    observation.write_bytes(b"tampered")

    with pytest.raises(RecordingGeometryError, match="size mismatch|checksum mismatch"):
        verify_recording_geometry_bundle(tmp_path)






def test_folder_loader_rejects_inward_gate(tmp_path: Path) -> None:
    _write_folder_bundle(tmp_path, gate_radius=199.0)

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert "smaller" in result.issues[0].message


def test_folder_loader_salvages_resolved_camera_from_selected_partial(tmp_path: Path) -> None:
    contract, snapshot = _write_folder_bundle(tmp_path)
    daily = contract["daily_registration_geometry"]
    assert isinstance(daily, dict)
    daily["status"] = "selected_partial"
    cameras = daily["cameras"]
    assert isinstance(cameras, dict)
    cameras["2010094"] = {
        "status": "missing",
        "camera_serial": "2010094",
        "arena_id": "arena_2",
    }
    contract_bytes = _json_bytes(contract, newline=True)
    (tmp_path / "recording_geometry_contract.json").write_bytes(contract_bytes)
    pointer = snapshot["recording_geometry_contract"]
    assert isinstance(pointer, dict)
    pointer["sha256"] = _sha(contract_bytes)
    (tmp_path / "recording_snapshot.json").write_bytes(_json_bytes(snapshot))

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID
    assert result.enclosing_selection_status == "selected_partial"
    assert len(result.masks) == 1
    assert result.issues[0].code == "camera_geometry_unresolved"
    with pytest.raises(RecordingGeometryError, match="selected_partial"):
        load_registered_dish_masks_from_recording_folder(
            tmp_path,
            policy=GeometryLoadPolicy.REQUIRED,
        )


def test_folder_loader_rejects_native_frame_dimension_mismatch(tmp_path: Path) -> None:
    _contract, snapshot = _write_folder_bundle(tmp_path)
    runtime = snapshot["camera_runtime"]
    assert isinstance(runtime, dict)
    camera = runtime["2010093"]
    assert isinstance(camera, dict)
    frame = camera["coordinate_frame"]
    assert isinstance(frame, dict)
    shape = frame["image_shape"]
    assert isinstance(shape, dict)
    shape["width"] = 641
    (tmp_path / "recording_snapshot.json").write_bytes(_json_bytes(snapshot))

    result = load_registered_dish_masks_from_recording_folder(tmp_path)

    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert "dimensions disagree" in result.issues[0].message


def _write_h5(
    path: Path,
    *,
    runtime_registration_id: str = "dailyreg-1",
    contract_version: int = 1,
    scope_version: int = 1,
    source_version: int | None = None,
    unified: bool = False,
) -> None:
    contract = {
        "schema_id": "orange.recording.geometry_contract",
        "schema_version": contract_version,
        "status": "resolved",
    }
    contract_bytes = _json_bytes(contract, newline=True)
    runtime = {
        "schema_id": "citrus.calibration.daily_registration",
        "schema_version": 1,
        "status": "accepted",
        "registration_id": runtime_registration_id,
        "rig_id": "omnifin0",
        "canvas_name": "shadow",
        "targets": [
            {
                "camera_id": "2010093",
                "arena_id": "arena_1",
                "rim_observation": {
                    "artifact_id": "dishrim-2010093",
                    "path": "/orange/immutable/observation.json",
                    "sha256": "sha256:" + "a" * 64,
                },
            }
        ],
    }
    runtime_bytes = _json_bytes(runtime)
    entry = _rim_entry()
    scope = {
        "schema_id": "citrus.session.orange_recording_geometry_contract_scope",
        "schema_version": scope_version,
        "scope_status": "resolved",
        "source_contract": {
            "sha256": _sha(contract_bytes),
            "schema_version": contract_version if source_version is None else source_version,
        },
        "target": {
            "rig_id": "omnifin0",
            "canvas_name": "shadow",
            "arena_id": "arena_1",
            "associated_camera_ids": ["2010093"],
        },
        "cameras": {
            "2010093": {
                "arena_id": "arena_1",
                "daily_registration_geometry": {
                    "schema_id": "orange.recording.daily_registration_camera_geometry",
                    "schema_version": 1,
                    "status": "resolved",
                    "mode": "selected_daily_registration",
                    "registration_id": "dailyreg-1",
                    "registration": {
                        "sha256": _sha(runtime_bytes),
                        "source_path": "/citrus/immutable/registration.json",
                        "valid_until_utc": "2026-07-23T04:00:00Z",
                    },
                    "recording_snapshot_entry": entry,
                    "selected_daily_registration_applied_by_citrus": True,
                },
            }
        },
    }
    scope_bytes = _json_bytes(scope)

    with h5py.File(path, "w") as h5:
        recording = h5.create_group("recording_geometry_contract")
        recording.attrs["capture_status"] = "embedded_verified"
        recording.attrs["checksum_verified"] = 1
        recording.attrs["schema_id"] = "orange.recording.geometry_contract"
        recording.attrs["schema_version"] = contract_version
        contract_dataset = recording.create_dataset("contract_json", data=contract_bytes)
        contract_dataset.attrs["checksum_sha256"] = _sha(contract_bytes)
        scope_dataset = recording.create_dataset("h5_scope_json", data=scope_bytes)
        scope_dataset.attrs["checksum_sha256"] = _sha(scope_bytes)
        scope_dataset.attrs["source_contract_sha256"] = _sha(contract_bytes)

        if unified:
            # Unified H5: the runtime contract lives at its namespace binding.
            namespace = _json_bytes(NAMESPACE_RECORD)
            dataset = h5.create_dataset(NAMESPACE, data=namespace)
            dataset.attrs["checksum_sha256"] = _sha(namespace)
        runtime_group = h5.create_group(
            "geometry/runtime" if unified else "runtime_geometry_contract"
        )
        runtime_dataset = runtime_group.create_dataset(
            "daily_registration_json",
            data=runtime_bytes,
        )
        runtime_dataset.attrs["checksum_sha256"] = _sha(runtime_bytes)
        runtime_dataset.attrs["load_status"] = "loaded"
        runtime_dataset.attrs["source_path"] = "/citrus/immutable/registration.json"


def test_h5_loader_hashes_exact_payload_and_reconciles_runtime(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_h5(path)

    result = load_registered_dish_masks_from_citrus_h5(path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID
    assert result.issues == ()
    mask = next(iter(result.masks.values()))
    assert mask.citrus_registration_status is CitrusRegistrationStatus.EXACT_MATCH_APPLIED
    assert mask.selected_daily_registration_applied_by_citrus is True
    assert mask.source_contract_sha256 == result.source_contract_sha256
    assert mask.h5_scope_sha256 == result.h5_scope_sha256


def test_h5_loader_reports_registration_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_h5(path, runtime_registration_id="wrong-registration")

    result = load_registered_dish_masks_from_citrus_h5(path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID
    mask = next(iter(result.masks.values()))
    assert mask.citrus_registration_status is CitrusRegistrationStatus.REGISTRATION_ID_MISMATCH
    assert result.issues[0].code == "registration_id_mismatch"

    with pytest.raises(RecordingGeometryError, match="did not exactly match"):
        load_registered_dish_masks_from_citrus_h5(
            path,
            policy=GeometryLoadPolicy.REQUIRED,
        )


def test_h5_loader_rejects_exact_payload_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "experiment.h5"
    _write_h5(path)
    with h5py.File(path, "a") as h5:
        h5["recording_geometry_contract/contract_json"].attrs["checksum_sha256"] = (
            "sha256:" + "0" * 64
        )

    result = load_registered_dish_masks_from_citrus_h5(path)

    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert "checksum mismatch" in result.issues[0].message


def test_h5_loader_classifies_unreferenced_geometry_as_legacy(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("recording_geometry_contract")
        group.attrs["capture_status"] = "not_referenced"
        group.attrs["checksum_verified"] = 0

    result = load_registered_dish_masks_from_citrus_h5(path)

    assert result.mask_geometry_status is MaskGeometryStatus.LEGACY_MISSING
    assert result.issues[0].code == "legacy_missing_recording_bound_mask"


def test_h5_loader_resolves_runtime_contract_through_namespace(tmp_path: Path) -> None:
    legacy, unified = tmp_path / "legacy.h5", tmp_path / "unified.h5"
    _write_h5(legacy)
    _write_h5(unified, unified=True)

    old, new = (load_registered_dish_masks_from_citrus_h5(p) for p in (legacy, unified))

    assert new.mask_geometry_status is MaskGeometryStatus.VALID
    assert new.issues == ()
    (a,), (b,) = old.masks.values(), new.masks.values()
    assert b.citrus_registration_status is CitrusRegistrationStatus.EXACT_MATCH_APPLIED
    assert (a.physical_inner_rim, a.valid_detection_gate) == (b.physical_inner_rim, b.valid_detection_gate)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("other_binding", "not the pinned record"),
        ("checksum", "checksum mismatch"),
        ("literal_too", "both a geometry source-namespace record and a literal"),
        ("unbound", "lacks /geometry/runtime"),
    ],
)
def test_h5_loader_refuses_ambiguous_runtime_namespace(
    tmp_path: Path, damage: str, message: str
) -> None:
    path = tmp_path / "unified.h5"
    _write_h5(path, unified=True)
    with h5py.File(path, "a") as h5:
        if damage == "other_binding":
            record = json.loads(json.dumps(NAMESPACE_RECORD))
            record["bindings"]["/runtime_geometry_contract"] = "/geometry/elsewhere"
            del h5[NAMESPACE]
            payload = _json_bytes(record)
            h5.create_dataset(NAMESPACE, data=payload).attrs["checksum_sha256"] = _sha(payload)
        elif damage == "checksum":
            h5[NAMESPACE].attrs["checksum_sha256"] = "sha256:" + "0" * 64
        elif damage == "literal_too":
            h5.copy("geometry/runtime", "runtime_geometry_contract")
        else:
            h5.move("geometry/runtime", "geometry/runtime_moved")

    result = load_registered_dish_masks_from_citrus_h5(path)

    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert result.masks == {}
    assert message in result.issues[0].message


@pytest.mark.parametrize(
    ("scope_version", "contract_version"),
    [(1, 1), (2, 1), (2, 2)],
)
def test_h5_loader_reads_scope_v2_like_v1(
    tmp_path: Path, scope_version: int, contract_version: int
) -> None:
    # Citrus #35: scope v2 changes only version fields, and may reference a v2
    # geometry contract.
    path = tmp_path / "experiment.h5"
    _write_h5(path, scope_version=scope_version, contract_version=contract_version)

    result = load_registered_dish_masks_from_citrus_h5(path)

    assert result.mask_geometry_status is MaskGeometryStatus.VALID
    assert result.issues == ()
    assert len(result.masks) == 1


@pytest.mark.parametrize(
    ("scope_version", "contract_version", "source_version", "message"),
    [
        (1, 2, 2, "scope version 1 cannot reference geometry contract version 2"),
        (1, 1, None, "cannot reference"),
        (2, 2, 1, "another version than /recording_geometry_contract"),
        (2, 1, 2, "another version than /recording_geometry_contract"),
        (3, 1, 1, "version 1 or 2"),
    ],
)
def test_h5_loader_refuses_mismatched_scope_versions(
    tmp_path: Path,
    scope_version: int,
    contract_version: int,
    source_version: int | None,
    message: str,
) -> None:
    path = tmp_path / "experiment.h5"
    _write_h5(
        path,
        scope_version=scope_version,
        contract_version=contract_version,
        source_version=source_version,
    )
    if source_version is None:
        _drop_scope_source_version(path)

    result = load_registered_dish_masks_from_citrus_h5(path)

    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert result.masks == {}
    assert message in result.issues[0].message


def _drop_scope_source_version(path: Path) -> None:
    with h5py.File(path, "a") as h5:
        group = h5["recording_geometry_contract"]
        scope = json.loads(bytes(group["h5_scope_json"][()]))
        del scope["source_contract"]["schema_version"]
        attrs = dict(group["h5_scope_json"].attrs)
        del group["h5_scope_json"]
        scope_bytes = _json_bytes(scope)
        dataset = group.create_dataset("h5_scope_json", data=scope_bytes)
        for key, value in attrs.items():
            dataset.attrs[key] = value
        dataset.attrs["checksum_sha256"] = _sha(scope_bytes)


def test_folder_loader_reads_a_v2_geometry_contract_like_v1(tmp_path: Path) -> None:
    # frame_identity_v1 (agent-contracts #87): contract v2 / assets v2 / rim v3
    # change only versions and frame-reference names Palette does not read.
    _write_folder_bundle(tmp_path / "v1")
    _write_folder_bundle(tmp_path / "v2", contract_version=2)
    old = load_registered_dish_masks_from_recording_folder(tmp_path / "v1")
    new = load_registered_dish_masks_from_recording_folder(tmp_path / "v2")
    assert new.mask_geometry_status is MaskGeometryStatus.VALID
    (a,), (b,) = old.masks.values(), new.masks.values()
    assert (a.physical_inner_rim, a.valid_detection_gate) == (b.physical_inner_rim, b.valid_detection_gate)


@pytest.mark.parametrize(
    ("contract_version", "assets_version", "rim_version", "message"),
    [
        (2, 2, 2, "must embed dish rim observation version 3"),
        (1, 1, 3, "must embed dish rim observation version 2"),
        (2, 1, 3, "asset manifest must be"),
        (3, 3, 4, "version 1 or 2"),
    ],
)
def test_folder_loader_refuses_mismatched_geometry_versions(
    tmp_path: Path, contract_version: int, assets_version: int, rim_version: int, message: str
) -> None:
    _write_folder_bundle(
        tmp_path, contract_version=contract_version, assets_version=assets_version, rim_version=rim_version
    )
    result = load_registered_dish_masks_from_recording_folder(tmp_path)
    assert result.mask_geometry_status is MaskGeometryStatus.INVALID
    assert result.masks == {}
    assert message in result.issues[0].message
