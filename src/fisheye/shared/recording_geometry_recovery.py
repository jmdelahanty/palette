"""Explicit recovery for acquisition geometry missing its producer pointer.

The recovery receipt is an immutable Palette attestation.  It does not modify
Orange or Citrus artifacts and it does not make claims on their behalf.  It
binds one exact historical H5 recording to one exact camera/arena entry in an
otherwise valid, checksummed recording-geometry bundle.

Normal readers must continue to use :mod:`fisheye.shared.recording_geometry`.
Only callers that explicitly select this recovery surface may consume a mask
whose producer snapshot lacked the contract pointer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.recording_geometry import (
    RECORDING_GEOMETRY_ASSETS_NAME,
    RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH,
    RECORDING_GEOMETRY_CONTRACT_NAME,
    RECORDING_SNAPSHOT_NAME,
    CitrusRegistrationStatus,
    GeometryLoadPolicy,
    GeometryLoadIssue,
    MaskGeometryStatus,
    MaterializedAssetStatus,
    RecordingGeometryBundleVerification,
    RecordingGeometryError,
    RegisteredDishMask,
    RegisteredDishMaskCollection,
    RegisteredDishMaskKey,
    _mask_from_entry,
    _normalized_sha256,
    _registration_identity,
    _required_mapping,
    _required_text,
    _safe_relative,
    _sha256_bytes,
    _sha256_file,
    _snapshot_contract_pointer,
    _strict_json_loads,
    _verify_asset_manifest,
    _verify_snapshot_dimensions,
    verify_recording_geometry_bundle,
)
from fisheye.shared.recording_geometry_bundle import (
    RecordingGeometryBundlePublication,
    publish_recording_geometry_bundle,
)
from fisheye.shared.run_provenance import git_identity


RECOVERY_RECEIPT_SCHEMA_ID = "palette.recording_geometry_recovery_receipt"
RECOVERY_RECEIPT_SCHEMA_VERSION = 1
RECOVERY_RECEIPT_NAME = "recording_geometry_recovery.json"
RECOVERY_REASON = "producer_snapshot_missing_contract_pointer"
RECOVERY_AUTHORITY = "operator_approved_historical_recovery"
RECOVERY_SOURCE_KIND = "palette_recovered_recording_geometry"
RECOVERY_LINKAGE_STATUS = "operator_approved_recovery_receipt"
RECOVERY_ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class RecordingGeometryRecoveryEvidence:
    bundle_root: Path
    bundle_verification: RecordingGeometryBundleVerification
    snapshot_sha256: str
    snapshot_recording_id: str
    target_h5_path: Path
    target_h5_sha256: str
    recording_name: str
    session_uuid: str
    camera_serial: str
    arena_id: str
    h5_geometry_capture_status: str
    h5_geometry_checksum_verified: int
    mask: RegisteredDishMask


@dataclass(frozen=True)
class VerifiedRecordingGeometryRecovery:
    receipt_path: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]
    evidence: RecordingGeometryRecoveryEvidence


@dataclass(frozen=True)
class RecordingGeometryRecoveryPublication:
    bundle_publication: RecordingGeometryBundlePublication
    receipt_path: Path
    receipt_published: bool
    verified: VerifiedRecordingGeometryRecovery


def _text_attr(value: Any, *, label: str) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise RecordingGeometryError(f"{label} must be a scalar string.")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecordingGeometryError(f"{label} must be UTF-8.") from exc
    text = str(value or "").strip()
    if not text:
        raise RecordingGeometryError(f"{label} must be nonempty.")
    return text


def _target_h5_identity(path: Path) -> tuple[str, str, str, str, str, int]:
    target_h5 = path.expanduser().resolve()
    try:
        with h5py.File(target_h5, "r") as h5:
            session_uuid = _text_attr(h5.attrs.get("session_uuid"), label="H5 session_uuid")
            arena_id = _text_attr(h5.attrs.get("arena_id"), label="H5 arena_id")
            ipc_source = _text_attr(
                h5.attrs.get("ipc_source_name"),
                label="H5 ipc_source_name",
            )
            match = re.search(r"(?:^|[/_])cam_(\d+)$", ipc_source, flags=re.IGNORECASE)
            if match is None:
                raise RecordingGeometryError(
                    "H5 ipc_source_name does not identify one exact camera serial."
                )
            camera_serial = match.group(1)
            geometry = h5.get("recording_geometry_contract")
            if not isinstance(geometry, h5py.Group):
                raise RecordingGeometryError(
                    "Target H5 lacks the recording_geometry_contract status group."
                )
            capture_status = str(geometry.attrs.get("capture_status") or "missing")
            checksum_verified = int(geometry.attrs.get("checksum_verified") or 0)
            if capture_status != "not_referenced" or checksum_verified != 0:
                raise RecordingGeometryError(
                    "Historical recovery requires H5 geometry status "
                    "capture_status=not_referenced and checksum_verified=0."
                )
    except OSError as exc:
        raise RecordingGeometryError(f"Unable to read target H5 {target_h5}: {exc}") from exc
    recording_name = target_h5.parent.parent.name
    if not recording_name:
        raise RecordingGeometryError("Target H5 is not inside a named recording/raw directory.")
    return (
        recording_name,
        session_uuid,
        camera_serial,
        arena_id,
        capture_status,
        checksum_verified,
    )


def _verify_manifest_rim(
    *,
    camera_serial: str,
    source_sha256: str,
    asset_status: MaterializedAssetStatus,
    manifest_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    if asset_status is not MaterializedAssetStatus.COMPLETE:
        raise RecordingGeometryError(
            "Historical recovery requires a complete recording-local geometry asset bundle."
        )
    observation_rel = (
        f"cameras/Cam{camera_serial}/daily_registration/"
        "rim_observation/observation.json"
    )
    row = manifest_rows.get(observation_rel)
    if row is None:
        raise RecordingGeometryError(
            f"Asset manifest lacks exact rim observation {observation_rel}."
        )
    row_sha = _normalized_sha256(row.get("sha256"), label="manifest rim sha256")
    if row_sha != source_sha256:
        raise RecordingGeometryError(
            "Materialized rim checksum disagrees with the selected contract entry."
        )


def inspect_recording_geometry_recovery(
    *,
    bundle_root: str | Path,
    target_h5_path: str | Path,
) -> RecordingGeometryRecoveryEvidence:
    """Prove that a recovery receipt may bind this exact bundle and H5."""

    root = Path(bundle_root).expanduser().resolve()
    target_h5 = Path(target_h5_path).expanduser().resolve()
    verification = verify_recording_geometry_bundle(
        root,
        require_snapshot_pointer=False,
        verify_all_assets=True,
    )
    if verification.snapshot_pointer_status != "missing":
        raise RecordingGeometryError(
            "Historical recovery is only valid when the producer snapshot pointer is missing."
        )

    snapshot_path = root / RECORDING_SNAPSHOT_NAME
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = _strict_json_loads(snapshot_bytes, label="recording_snapshot.json")
    pointer_path, pointer_sha, pointer_status = _snapshot_contract_pointer(
        snapshot,
        root=root,
        required=False,
    )
    if pointer_path is not None or pointer_sha is not None or pointer_status != "missing":
        raise RecordingGeometryError("Producer snapshot pointer state is not recoverable-missing.")
    snapshot_recording_id = _required_text(
        snapshot.get("recording_id"),
        label="recording_snapshot.recording_id",
    )

    contract_path = root / RECORDING_GEOMETRY_CONTRACT_NAME
    contract_bytes = contract_path.read_bytes()
    contract_sha = _sha256_bytes(contract_bytes)
    if contract_sha != verification.contract_sha256:
        raise RecordingGeometryError("Bundle verification and contract bytes disagree.")
    contract = _strict_json_loads(contract_bytes, label="recording_geometry_contract.json")
    if contract.get("schema_id") != "orange.recording.geometry_contract" or int(
        contract.get("schema_version", -1)
    ) != 1:
        raise RecordingGeometryError("Recovery supports only Orange geometry contract v1.")

    (
        recording_name,
        session_uuid,
        camera_serial,
        arena_id,
        h5_geometry_capture_status,
        h5_geometry_checksum_verified,
    ) = _target_h5_identity(target_h5)
    daily = _required_mapping(
        contract.get("daily_registration_geometry"),
        label="daily_registration_geometry",
    )
    if daily.get("mode") != "selected_daily_registration" or daily.get("status") != (
        "selected_resolved"
    ):
        raise RecordingGeometryError(
            "Historical recovery requires a fully resolved selected daily registration."
        )
    cameras = _required_mapping(
        daily.get("cameras"),
        label="daily_registration_geometry.cameras",
    )
    camera = _required_mapping(
        cameras.get(camera_serial),
        label=f"daily_registration_geometry.cameras[{camera_serial}]",
    )
    if camera.get("status") != "resolved":
        raise RecordingGeometryError("Target camera geometry is not resolved.")
    if str(camera.get("camera_serial")) != camera_serial:
        raise RecordingGeometryError("Contract camera serial does not match the H5 camera.")
    if str(camera.get("arena_id")) != arena_id:
        raise RecordingGeometryError("Contract arena does not match the H5 arena.")

    selection = _required_mapping(contract.get("selection"), label="geometry selection")
    key = RegisteredDishMaskKey(
        rig_id=_required_text(selection.get("rig_id"), label="selection.rig_id"),
        canvas_name=_required_text(
            selection.get("selected_canvas_name"),
            label="selection.selected_canvas_name",
        ),
        arena_id=arena_id,
        camera_serial=camera_serial,
    )
    registration_id, registration_sha, valid_until = _registration_identity(contract)
    asset_status, _manifest_sha, manifest_rows = _verify_asset_manifest(
        root,
        contract,
        verify_all_files=True,
    )
    entry = _required_mapping(
        camera.get("recording_snapshot_entry"),
        label="recording_snapshot_entry",
    )
    mask = _mask_from_entry(
        entry,
        key=key,
        registration_id=registration_id,
        registration_sha256=registration_sha,
        source_contract_sha256=contract_sha,
        h5_scope_sha256=None,
        asset_status=asset_status,
        citrus_status=CitrusRegistrationStatus.MISSING,
        valid_until_utc=valid_until,
        applied_by_citrus=False,
        source_kind=RECOVERY_SOURCE_KIND,
        source_location=str(root),
    )
    _verify_snapshot_dimensions(
        snapshot,
        camera_serial=camera_serial,
        width=mask.native_width_px,
        height=mask.native_height_px,
    )
    _verify_manifest_rim(
        camera_serial=camera_serial,
        source_sha256=mask.source_observation_sha256,
        asset_status=asset_status,
        manifest_rows=manifest_rows,
    )
    return RecordingGeometryRecoveryEvidence(
        bundle_root=root,
        bundle_verification=verification,
        snapshot_sha256=_sha256_bytes(snapshot_bytes),
        snapshot_recording_id=snapshot_recording_id,
        target_h5_path=target_h5,
        target_h5_sha256=_sha256_file(target_h5),
        recording_name=recording_name,
        session_uuid=session_uuid,
        camera_serial=camera_serial,
        arena_id=arena_id,
        h5_geometry_capture_status=h5_geometry_capture_status,
        h5_geometry_checksum_verified=h5_geometry_checksum_verified,
        mask=mask,
    )


def _receipt_id(evidence: RecordingGeometryRecoveryEvidence) -> str:
    material = "\n".join(
        (
            evidence.bundle_verification.contract_sha256,
            evidence.target_h5_sha256,
            evidence.camera_serial,
            evidence.arena_id,
        )
    ).encode("utf-8")
    return "recording-geometry-recovery-" + hashlib.sha256(material).hexdigest()[:24]


def build_recording_geometry_recovery_receipt(
    *,
    bundle_root: str | Path,
    target_h5_path: str | Path,
    approved_by: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build, but do not write, one exact historical recovery receipt."""

    approver = str(approved_by).strip()
    if not approver:
        raise RecordingGeometryError("approved_by must be nonempty.")
    evidence = inspect_recording_geometry_recovery(
        bundle_root=bundle_root,
        target_h5_path=target_h5_path,
    )
    created = created_at_utc or datetime.now(timezone.utc).isoformat()
    try:
        parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordingGeometryError("created_at_utc must be ISO-8601.") from exc
    if parsed_created.tzinfo is None:
        raise RecordingGeometryError("created_at_utc must include a timezone.")
    git = git_identity(cwd=Path(__file__))
    return {
        "schema_id": RECOVERY_RECEIPT_SCHEMA_ID,
        "schema_version": RECOVERY_RECEIPT_SCHEMA_VERSION,
        "status": "approved",
        "receipt_id": _receipt_id(evidence),
        "recovery_reason": RECOVERY_REASON,
        "authority": RECOVERY_AUTHORITY,
        "created_at_utc": created,
        "approved_by": approver,
        "target": {
            "recording_directory_name_at_recovery": evidence.recording_name,
            "session_uuid": evidence.session_uuid,
            "camera_serial": evidence.camera_serial,
            "arena_id": evidence.arena_id,
            "h5_relative_path": evidence.target_h5_path.name,
            "h5_sha256": evidence.target_h5_sha256,
            "h5_geometry_capture_status": evidence.h5_geometry_capture_status,
            "h5_geometry_checksum_verified": evidence.h5_geometry_checksum_verified,
        },
        "evidence": {
            "geometry_bundle_relative_path": "recording_geometry_bundle",
            "recording_snapshot": {
                "relative_path": f"recording_geometry_bundle/{RECORDING_SNAPSHOT_NAME}",
                "sha256": evidence.snapshot_sha256,
                "recording_id": evidence.snapshot_recording_id,
                "contract_pointer_status": "missing",
            },
            "recording_geometry_contract": {
                "relative_path": (
                    f"recording_geometry_bundle/{RECORDING_GEOMETRY_CONTRACT_NAME}"
                ),
                "sha256": evidence.bundle_verification.contract_sha256,
                "camera_arena_entry_status": "resolved",
            },
            "materialized_assets": {
                "relative_path": (
                    f"recording_geometry_bundle/{RECORDING_GEOMETRY_ASSETS_NAME}/manifest.json"
                ),
                "sha256": evidence.bundle_verification.manifest_sha256,
                "status": evidence.bundle_verification.materialized_asset_status.value,
                "file_count": evidence.bundle_verification.manifest_file_count,
            },
        },
        "claims": {
            "original_producer_artifacts_mutated": False,
            "producer_declared_snapshot_contract_link": False,
            "citrus_runtime_application_claimed": False,
            "operator_approved_exact_h5_camera_arena_binding": True,
            "independent_palette_fit_required_before_operational_use": True,
        },
        "tool": {
            "name": "fisheye.shared.recording_geometry_recovery",
            "algorithm_version": RECOVERY_ALGORITHM_VERSION,
            "palette_git_sha": git.get("git_sha"),
            "palette_git_dirty": git.get("git_dirty"),
        },
    }


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise RecordingGeometryError(
            f"Recovery receipt {label} mismatch: expected {expected!r}, got {actual!r}."
        )


def validate_recording_geometry_recovery_receipt(
    receipt_path: str | Path,
) -> VerifiedRecordingGeometryRecovery:
    """Re-prove every binding declared by an immutable recovery receipt."""

    path = Path(receipt_path).expanduser().resolve()
    payload_bytes = path.read_bytes()
    receipt = _strict_json_loads(payload_bytes, label="recording geometry recovery receipt")
    _require_equal(receipt.get("schema_id"), RECOVERY_RECEIPT_SCHEMA_ID, label="schema_id")
    _require_equal(
        int(receipt.get("schema_version", -1)),
        RECOVERY_RECEIPT_SCHEMA_VERSION,
        label="schema_version",
    )
    _require_equal(receipt.get("status"), "approved", label="status")
    _require_equal(receipt.get("recovery_reason"), RECOVERY_REASON, label="recovery_reason")
    _require_equal(receipt.get("authority"), RECOVERY_AUTHORITY, label="authority")
    _required_text(receipt.get("approved_by"), label="approved_by")
    created = _required_text(receipt.get("created_at_utc"), label="created_at_utc")
    try:
        parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordingGeometryError("Receipt created_at_utc is not ISO-8601.") from exc
    if parsed_created.tzinfo is None:
        raise RecordingGeometryError("Receipt created_at_utc lacks a timezone.")

    target = _required_mapping(receipt.get("target"), label="receipt.target")
    evidence_record = _required_mapping(receipt.get("evidence"), label="receipt.evidence")
    bundle_rel = _required_text(
        evidence_record.get("geometry_bundle_relative_path"),
        label="geometry_bundle_relative_path",
    )
    _require_equal(bundle_rel, "recording_geometry_bundle", label="bundle relative path")
    bundle_root = _safe_relative(path.parent, bundle_rel, label="geometry bundle relative path")
    h5_rel = _required_text(target.get("h5_relative_path"), label="target.h5_relative_path")
    if Path(h5_rel).name != h5_rel:
        raise RecordingGeometryError("Recovery target H5 must be a direct raw/ child.")
    target_h5 = _safe_relative(path.parent, h5_rel, label="target H5 relative path")
    evidence = inspect_recording_geometry_recovery(
        bundle_root=bundle_root,
        target_h5_path=target_h5,
    )

    _require_equal(receipt.get("receipt_id"), _receipt_id(evidence), label="receipt_id")
    _required_text(
        target.get("recording_directory_name_at_recovery"),
        label="recording_directory_name_at_recovery",
    )
    _require_equal(target.get("session_uuid"), evidence.session_uuid, label="session_uuid")
    _require_equal(target.get("camera_serial"), evidence.camera_serial, label="camera_serial")
    _require_equal(target.get("arena_id"), evidence.arena_id, label="arena_id")
    _require_equal(
        target.get("h5_geometry_capture_status"),
        evidence.h5_geometry_capture_status,
        label="H5 geometry capture status",
    )
    _require_equal(
        int(target.get("h5_geometry_checksum_verified", -1)),
        evidence.h5_geometry_checksum_verified,
        label="H5 geometry checksum status",
    )
    _require_equal(
        _normalized_sha256(target.get("h5_sha256"), label="target.h5_sha256"),
        evidence.target_h5_sha256,
        label="H5 checksum",
    )

    snapshot = _required_mapping(
        evidence_record.get("recording_snapshot"),
        label="evidence.recording_snapshot",
    )
    _require_equal(
        snapshot.get("relative_path"),
        f"recording_geometry_bundle/{RECORDING_SNAPSHOT_NAME}",
        label="snapshot path",
    )
    _require_equal(
        _normalized_sha256(snapshot.get("sha256"), label="snapshot.sha256"),
        evidence.snapshot_sha256,
        label="snapshot checksum",
    )
    _require_equal(
        snapshot.get("recording_id"),
        evidence.snapshot_recording_id,
        label="snapshot recording_id",
    )
    _require_equal(snapshot.get("contract_pointer_status"), "missing", label="pointer status")

    contract = _required_mapping(
        evidence_record.get("recording_geometry_contract"),
        label="evidence.recording_geometry_contract",
    )
    _require_equal(
        contract.get("relative_path"),
        f"recording_geometry_bundle/{RECORDING_GEOMETRY_CONTRACT_NAME}",
        label="contract path",
    )
    _require_equal(
        _normalized_sha256(contract.get("sha256"), label="contract.sha256"),
        evidence.bundle_verification.contract_sha256,
        label="contract checksum",
    )
    _require_equal(
        contract.get("camera_arena_entry_status"),
        "resolved",
        label="camera/arena entry status",
    )

    assets = _required_mapping(
        evidence_record.get("materialized_assets"),
        label="evidence.materialized_assets",
    )
    _require_equal(
        assets.get("relative_path"),
        f"recording_geometry_bundle/{RECORDING_GEOMETRY_ASSETS_NAME}/manifest.json",
        label="asset manifest path",
    )
    _require_equal(
        _normalized_sha256(assets.get("sha256"), label="assets.sha256"),
        evidence.bundle_verification.manifest_sha256,
        label="asset manifest checksum",
    )
    _require_equal(assets.get("status"), "complete", label="asset status")
    _require_equal(
        int(assets.get("file_count", -1)),
        evidence.bundle_verification.manifest_file_count,
        label="asset file count",
    )

    claims = _required_mapping(receipt.get("claims"), label="receipt.claims")
    expected_claims = {
        "original_producer_artifacts_mutated": False,
        "producer_declared_snapshot_contract_link": False,
        "citrus_runtime_application_claimed": False,
        "operator_approved_exact_h5_camera_arena_binding": True,
        "independent_palette_fit_required_before_operational_use": True,
    }
    for name, expected in expected_claims.items():
        _require_equal(claims.get(name), expected, label=f"claims.{name}")
    tool = _required_mapping(receipt.get("tool"), label="receipt.tool")
    _require_equal(
        tool.get("name"),
        "fisheye.shared.recording_geometry_recovery",
        label="tool.name",
    )
    _require_equal(
        int(tool.get("algorithm_version", -1)),
        RECOVERY_ALGORITHM_VERSION,
        label="tool.algorithm_version",
    )
    return VerifiedRecordingGeometryRecovery(
        receipt_path=path,
        receipt_sha256=_sha256_bytes(payload_bytes),
        receipt=receipt,
        evidence=evidence,
    )


def load_registered_dish_mask_from_recovery_receipt(
    receipt_path: str | Path,
    *,
    policy: GeometryLoadPolicy | str = GeometryLoadPolicy.REQUIRED,
) -> RegisteredDishMaskCollection:
    """Load one explicitly recovered mask; never entered by normal discovery."""

    policy = GeometryLoadPolicy(policy)
    if policy is GeometryLoadPolicy.OFF:
        return RegisteredDishMaskCollection(
            masks={},
            mask_geometry_status=MaskGeometryStatus.MISSING,
            source_kind=RECOVERY_SOURCE_KIND,
            source_location=str(Path(receipt_path).expanduser().resolve()),
            producer_contract_linkage_status=RECOVERY_LINKAGE_STATUS,
        )
    try:
        verified = validate_recording_geometry_recovery_receipt(receipt_path)
    except (OSError, RecordingGeometryError) as exc:
        if policy is GeometryLoadPolicy.REQUIRED:
            raise RecordingGeometryError(str(exc)) from exc
        return RegisteredDishMaskCollection(
            masks={},
            mask_geometry_status=MaskGeometryStatus.INVALID,
            source_kind=RECOVERY_SOURCE_KIND,
            source_location=str(Path(receipt_path).expanduser().resolve()),
            issues=(
                GeometryLoadIssue(
                    code="invalid_recording_geometry_recovery_receipt",
                    message=str(exc),
                ),
            ),
            producer_contract_linkage_status=RECOVERY_LINKAGE_STATUS,
        )
    evidence = verified.evidence
    mask = registered_dish_mask_from_verified_recovery(verified)
    return RegisteredDishMaskCollection(
        masks={mask.key: mask},
        mask_geometry_status=MaskGeometryStatus.VALID,
        source_kind=RECOVERY_SOURCE_KIND,
        source_location=str(verified.receipt_path),
        source_contract_sha256=evidence.bundle_verification.contract_sha256,
        enclosing_selection_status="selected_resolved",
        producer_contract_linkage_status=RECOVERY_LINKAGE_STATUS,
        recovery_receipt_sha256=verified.receipt_sha256,
    )


def registered_dish_mask_from_verified_recovery(
    verified: VerifiedRecordingGeometryRecovery,
) -> RegisteredDishMask:
    """Return the recovered mask represented by one already-verified receipt."""

    if type(verified) is not VerifiedRecordingGeometryRecovery:
        raise RecordingGeometryError("A verified recording-geometry recovery is required.")
    return replace(
        verified.evidence.mask,
        source_location=str(verified.receipt_path),
        producer_contract_linkage_status=RECOVERY_LINKAGE_STATUS,
        recovery_receipt_sha256=verified.receipt_sha256,
        independent_fit_required_before_operational_use=True,
    )


def publish_recording_geometry_recovery(
    *,
    source_bundle_root: str | Path,
    recording_root: str | Path,
    target_h5_path: str | Path,
    approved_by: str,
) -> RecordingGeometryRecoveryPublication:
    """Publish the verified bundle first and the immutable receipt last."""

    recording = Path(recording_root).expanduser().resolve()
    target_h5 = Path(target_h5_path).expanduser().resolve()
    raw_root = recording / "raw"
    if target_h5.parent != raw_root:
        raise RecordingGeometryError("Target H5 must be a direct child of recording/raw.")
    bundle_publication = publish_recording_geometry_bundle(
        source_root=source_bundle_root,
        recording_root=recording,
    )
    expected_bundle = recording / RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH
    if bundle_publication.verification.root != expected_bundle:
        raise RecordingGeometryError("Published geometry bundle path is inconsistent.")
    receipt_path = raw_root / RECOVERY_RECEIPT_NAME
    if receipt_path.exists():
        verified = validate_recording_geometry_recovery_receipt(receipt_path)
        if str(verified.receipt.get("approved_by")) != str(approved_by).strip():
            raise RecordingGeometryError(
                "Existing recovery receipt has a different approving operator."
            )
        return RecordingGeometryRecoveryPublication(
            bundle_publication=bundle_publication,
            receipt_path=receipt_path,
            receipt_published=False,
            verified=verified,
        )

    receipt = build_recording_geometry_recovery_receipt(
        bundle_root=expected_bundle,
        target_h5_path=target_h5,
        approved_by=approved_by,
    )
    write_json_atomic(receipt_path, receipt, overwrite=False)
    try:
        verified = validate_recording_geometry_recovery_receipt(receipt_path)
    except Exception:
        receipt_path.unlink(missing_ok=True)
        raise
    return RecordingGeometryRecoveryPublication(
        bundle_publication=bundle_publication,
        receipt_path=receipt_path,
        receipt_published=True,
        verified=verified,
    )


__all__ = [
    "RECOVERY_ALGORITHM_VERSION",
    "RECOVERY_AUTHORITY",
    "RECOVERY_LINKAGE_STATUS",
    "RECOVERY_REASON",
    "RECOVERY_RECEIPT_NAME",
    "RECOVERY_RECEIPT_SCHEMA_ID",
    "RECOVERY_RECEIPT_SCHEMA_VERSION",
    "RECOVERY_SOURCE_KIND",
    "RecordingGeometryRecoveryEvidence",
    "RecordingGeometryRecoveryPublication",
    "VerifiedRecordingGeometryRecovery",
    "build_recording_geometry_recovery_receipt",
    "inspect_recording_geometry_recovery",
    "load_registered_dish_mask_from_recovery_receipt",
    "publish_recording_geometry_recovery",
    "registered_dish_mask_from_verified_recovery",
    "validate_recording_geometry_recovery_receipt",
]
