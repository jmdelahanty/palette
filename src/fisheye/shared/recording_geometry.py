"""Recording-bound acquisition geometry loading and preservation checks.

This module implements the shared, read-only normalization boundary for the
Orange recording-folder and Citrus H5 representations.  It intentionally does
not select an operational mask, fit recording images, gate detections, publish
Zarr data, or consult a current calibration pointer.

The folder authority is discovered only through the checksummed pointer in
``recording_snapshot.json``.  A separate bundle verifier exists for the
organizer: it can prove that a fixed version-1 subtree was copied byte-for-byte
even for early producer fixtures whose snapshot predates that pointer.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID = "orange.recording.geometry_contract"
RECORDING_GEOMETRY_CONTRACT_SCHEMA_VERSION = 1
RECORDING_GEOMETRY_ASSETS_SCHEMA_ID = "orange.recording.geometry_assets"
RECORDING_GEOMETRY_ASSETS_SCHEMA_VERSION = 1
RIM_OBSERVATION_SCHEMA_ID = "orange.calibration.dish_top_rim_observation"
RIM_OBSERVATION_SCHEMA_VERSION = 2
H5_SCOPE_SCHEMA_ID = "citrus.session.orange_recording_geometry_contract_scope"
H5_SCOPE_SCHEMA_VERSION = 1
DAILY_REGISTRATION_SCHEMA_ID = "citrus.calibration.daily_registration"
DAILY_REGISTRATION_SCHEMA_VERSION = 1

RECORDING_SNAPSHOT_NAME = "recording_snapshot.json"
RECORDING_GEOMETRY_CONTRACT_NAME = "recording_geometry_contract.json"
RECORDING_GEOMETRY_ASSETS_NAME = "recording_geometry_assets"
RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH = Path("raw/recording_geometry_bundle")


class GeometryLoadPolicy(str, Enum):
    """How callers handle unavailable recording-bound geometry."""

    OFF = "off"
    IF_AVAILABLE = "if_available"
    REQUIRED = "required"


class MaskGeometryStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"
    LEGACY_MISSING = "legacy_missing"


class MaterializedAssetStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    INVALID = "invalid"


class CitrusRegistrationStatus(str, Enum):
    EXACT_MATCH_APPLIED = "exact_match_applied"
    MISSING = "missing"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    REGISTRATION_ID_MISMATCH = "registration_id_mismatch"
    CAMERA_ARENA_TARGET_MISSING = "camera_arena_target_missing"
    RIM_OBSERVATION_MISMATCH = "rim_observation_mismatch"
    INVALID = "invalid"


class RecordingGeometryError(ValueError):
    """Raised when required recording-bound geometry cannot be proven."""


@dataclass(frozen=True, order=True)
class RegisteredDishMaskKey:
    rig_id: str
    canvas_name: str
    arena_id: str
    camera_serial: str


@dataclass(frozen=True)
class CircleGeometry:
    center_x_native_px: float
    center_y_native_px: float
    radius_px: float


@dataclass(frozen=True)
class RegisteredDishMask:
    """One immutable acquisition mask normalized across producer containers."""

    key: RegisteredDishMaskKey
    artifact_id: str
    source_observation_sha256: str
    registration_id: str
    registration_sha256: str | None
    source_contract_sha256: str
    h5_scope_sha256: str | None
    physical_inner_rim: CircleGeometry
    valid_detection_gate: CircleGeometry
    native_width_px: int
    native_height_px: int
    coordinate_space: str
    palette_space_id: str
    coordinate_profile_id: str
    pixel_convention: str
    origin: str
    positive_x: str
    positive_y: str
    target_plane: str
    gating_semantics: str
    materialized_asset_status: MaterializedAssetStatus
    citrus_registration_status: CitrusRegistrationStatus
    source_valid_until_utc: str | None
    producer_operator_accepted: bool
    producer_quality_flags: tuple[str, ...]
    selected_daily_registration_applied_by_citrus: bool | None
    source_kind: str
    source_location: str
    producer_contract_linkage_status: str = "producer_native"
    recovery_receipt_sha256: str | None = None
    independent_fit_required_before_operational_use: bool = False


@dataclass(frozen=True)
class BoundRegisteredDishMask:
    """A normalized mask bound to one persisted Palette camera-frame proof."""

    mask: RegisteredDishMask
    pixel_frame_record_ref: str
    pixel_frame_record_sha256: str


@dataclass(frozen=True)
class GeometryLoadIssue:
    code: str
    message: str
    camera_serial: str | None = None
    arena_id: str | None = None


@dataclass(frozen=True)
class RegisteredDishMaskCollection:
    """Result of one adapter load, including structured missing/invalid state."""

    masks: Mapping[RegisteredDishMaskKey, RegisteredDishMask]
    mask_geometry_status: MaskGeometryStatus
    source_kind: str
    source_location: str
    source_contract_sha256: str | None = None
    h5_scope_sha256: str | None = None
    enclosing_selection_status: str | None = None
    issues: tuple[GeometryLoadIssue, ...] = field(default_factory=tuple)
    producer_contract_linkage_status: str = "producer_native"
    recovery_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "masks", MappingProxyType(dict(self.masks)))


@dataclass(frozen=True)
class RecordingGeometryBundleVerification:
    root: Path
    contract_sha256: str
    manifest_sha256: str | None
    manifest_file_count: int
    materialized_asset_status: MaterializedAssetStatus
    snapshot_pointer_status: str


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _normalized_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise RecordingGeometryError(f"{label} must be a SHA-256 string.")
    normalized = value.strip().lower()
    if not normalized.startswith("sha256:"):
        normalized = f"sha256:{normalized}"
    digest = normalized.removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RecordingGeometryError(f"{label} is not a valid SHA-256 digest.")
    return normalized


def _strict_json_loads(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecordingGeometryError(f"{label} must contain UTF-8 JSON.") from exc

    def reject_constant(value: str) -> None:
        raise RecordingGeometryError(f"{label} contains non-finite JSON value {value}.")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecordingGeometryError) as exc:
        if isinstance(exc, RecordingGeometryError):
            raise
        raise RecordingGeometryError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RecordingGeometryError(f"{label} must contain a JSON object.")
    return value


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordingGeometryError(f"{label} must be an object.")
    return value


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RecordingGeometryError(f"{label} must be a non-empty trimmed string.")
    return value


def _required_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RecordingGeometryError(f"{label} must be an integer.")
    result = int(value)
    if result <= 0:
        raise RecordingGeometryError(f"{label} must be positive.")
    return result


def _required_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RecordingGeometryError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise RecordingGeometryError(f"{label} must be finite.")
    return result


def _safe_relative(root: Path, value: Any, *, label: str) -> Path:
    text = _required_text(value, label=label)
    relative = Path(text)
    if relative.is_absolute():
        raise RecordingGeometryError(f"{label} must be relative.")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RecordingGeometryError(f"{label} escapes the recording root.")
    return resolved


def _require_schema(
    payload: Mapping[str, Any],
    *,
    schema_id: str,
    schema_version: int,
    id_field: str = "schema_id",
    version_field: str = "schema_version",
    label: str,
) -> None:
    if payload.get(id_field) != schema_id or payload.get(version_field) != schema_version:
        raise RecordingGeometryError(
            f"{label} must be {schema_id!r} version {schema_version}."
        )


def _circle_from_boundary(
    boundary: Mapping[str, Any],
    *,
    label: str,
    require_target_plane: bool,
) -> CircleGeometry:
    if boundary.get("coordinate_space") != "camera_native_pixels":
        raise RecordingGeometryError(f"{label} must use camera_native_pixels.")
    if require_target_plane and boundary.get("target_plane") != "dish_top_rim":
        raise RecordingGeometryError(f"{label} must target dish_top_rim.")
    geometry = _required_mapping(boundary.get("geometry"), label=f"{label}.geometry")
    if geometry.get("type") != "circle":
        raise RecordingGeometryError(f"{label}.geometry must be a circle.")
    center = _required_mapping(geometry.get("center_px"), label=f"{label}.geometry.center_px")
    radius = _required_finite(geometry.get("radius_px"), label=f"{label}.geometry.radius_px")
    if radius <= 0:
        raise RecordingGeometryError(f"{label}.geometry.radius_px must be positive.")
    return CircleGeometry(
        center_x_native_px=_required_finite(center.get("x"), label=f"{label}.center.x"),
        center_y_native_px=_required_finite(center.get("y"), label=f"{label}.center.y"),
        radius_px=radius,
    )


def _validate_mask_entry(
    entry: Mapping[str, Any],
    *,
    camera_serial: str,
    arena_id: str,
) -> tuple[CircleGeometry, CircleGeometry, int, int, bool, tuple[str, ...]]:
    _require_schema(
        entry,
        schema_id=RIM_OBSERVATION_SCHEMA_ID,
        schema_version=RIM_OBSERVATION_SCHEMA_VERSION,
        id_field="artifact_schema_id",
        version_field="artifact_schema_version",
        label="recording_snapshot_entry",
    )
    if str(entry.get("camera_serial")) != camera_serial:
        raise RecordingGeometryError("Rim observation camera_serial does not match its camera key.")
    if entry.get("arena_id") != arena_id:
        raise RecordingGeometryError("Rim observation arena_id does not match its camera scope.")
    if entry.get("coordinate_space") != "camera_native_pixels":
        raise RecordingGeometryError("Rim observation must use camera_native_pixels.")
    if entry.get("available_for_downstream_detection_gating") is not True:
        raise RecordingGeometryError("Rim observation is unavailable for downstream gating.")

    physical = _circle_from_boundary(
        _required_mapping(
            entry.get("accepted_inner_rim_boundary"),
            label="accepted_inner_rim_boundary",
        ),
        label="accepted_inner_rim_boundary",
        require_target_plane=True,
    )
    gate_payload = _required_mapping(entry.get("valid_detection_region"), label="valid_detection_region")
    if gate_payload.get("purpose") != "bounding_box_centroid_detection_gating":
        raise RecordingGeometryError("valid_detection_region has the wrong purpose.")
    if gate_payload.get("offset_direction") != "outward":
        raise RecordingGeometryError("valid_detection_region must use an outward offset.")
    gate = _circle_from_boundary(
        gate_payload,
        label="valid_detection_region",
        require_target_plane=False,
    )
    if not math.isclose(
        physical.center_x_native_px,
        gate.center_x_native_px,
        rel_tol=0.0,
        abs_tol=1e-6,
    ) or not math.isclose(
        physical.center_y_native_px,
        gate.center_y_native_px,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RecordingGeometryError("valid_detection_region is not concentric with the rim.")
    if gate.radius_px < physical.radius_px:
        raise RecordingGeometryError("valid_detection_region radius is smaller than the rim.")

    camera = _required_mapping(entry.get("camera"), label="recording_snapshot_entry.camera")
    width = _required_int(camera.get("width"), label="camera.width")
    height = _required_int(camera.get("height"), label="camera.height")
    if str(camera.get("serial")) != camera_serial:
        raise RecordingGeometryError("recording_snapshot_entry.camera.serial is inconsistent.")

    accepted_mask = _required_mapping(entry.get("accepted_mask"), label="accepted_mask")
    mask_shape = _required_mapping(accepted_mask.get("image_shape_px"), label="accepted_mask.image_shape_px")
    if (
        _required_int(mask_shape.get("width"), label="accepted_mask width") != width
        or _required_int(mask_shape.get("height"), label="accepted_mask height") != height
    ):
        raise RecordingGeometryError("accepted_mask dimensions disagree with the camera.")

    review = entry.get("operator_review")
    accepted = bool(isinstance(review, Mapping) and review.get("accepted") is True)
    if not accepted:
        raise RecordingGeometryError("Rim observation lacks operator acceptance.")
    quality = entry.get("quality")
    flags_value = quality.get("quality_flags", []) if isinstance(quality, Mapping) else []
    flags = tuple(str(value) for value in flags_value) if isinstance(flags_value, Sequence) and not isinstance(flags_value, (str, bytes)) else ()
    return physical, gate, width, height, accepted, flags


def _verify_snapshot_dimensions(
    snapshot: Mapping[str, Any],
    *,
    camera_serial: str,
    width: int,
    height: int,
) -> None:
    runtime = _required_mapping(snapshot.get("camera_runtime"), label="recording_snapshot.camera_runtime")
    camera = _required_mapping(runtime.get(camera_serial), label=f"camera_runtime[{camera_serial}]")
    frame = _required_mapping(camera.get("coordinate_frame"), label="camera_runtime.coordinate_frame")
    if frame.get("coordinate_space") != "camera_native_pixels":
        raise RecordingGeometryError("recording_snapshot camera frame is not native pixels.")
    if frame.get("point_order") != "xy" or frame.get("units") != "pixels":
        raise RecordingGeometryError("recording_snapshot camera axes are not canonical xy pixels.")
    origin = _required_mapping(frame.get("origin"), label="camera_runtime.coordinate_frame.origin")
    if origin.get("name") != "top_left_pixel" or origin.get("x_px") != 0 or origin.get("y_px") != 0:
        raise RecordingGeometryError("recording_snapshot camera origin is not the top-left pixel.")
    axes = _required_mapping(frame.get("axes"), label="camera_runtime.coordinate_frame.axes")
    x_axis = _required_mapping(axes.get("x"), label="camera x axis")
    y_axis = _required_mapping(axes.get("y"), label="camera y axis")
    if x_axis.get("positive_direction") != "right" or y_axis.get("positive_direction") != "down":
        raise RecordingGeometryError("recording_snapshot camera axis directions are unsupported.")
    shape = _required_mapping(frame.get("image_shape"), label="camera_runtime.coordinate_frame.image_shape")
    if int(shape.get("width", -1)) != width or int(shape.get("height", -1)) != height:
        raise RecordingGeometryError("recording_snapshot camera dimensions disagree with the rim.")


def _verify_snapshot_rim_reference(
    snapshot: Mapping[str, Any],
    *,
    camera_serial: str,
    artifact_id: str,
    source_sha256: str,
) -> None:
    calibrations = _required_mapping(snapshot.get("calibrations"), label="recording_snapshot.calibrations")
    camera = _required_mapping(calibrations.get(camera_serial), label=f"calibrations[{camera_serial}]")
    rim = _required_mapping(camera.get("dish_top_rim_observation"), label="dish_top_rim_observation")
    if rim.get("artifact_id") != artifact_id:
        raise RecordingGeometryError("recording_snapshot rim artifact_id disagrees with the contract.")
    if _normalized_sha256(rim.get("sha256"), label="recording_snapshot rim sha256") != source_sha256:
        raise RecordingGeometryError("recording_snapshot rim checksum disagrees with the contract.")


def _manifest_status(value: Any) -> MaterializedAssetStatus:
    text = str(value or "missing")
    if text == "complete":
        return MaterializedAssetStatus.COMPLETE
    if text in {"partial", "selected_partial"}:
        return MaterializedAssetStatus.PARTIAL
    if text in {"missing", "not_requested", "unavailable"}:
        return MaterializedAssetStatus.MISSING
    return MaterializedAssetStatus.INVALID


def _verify_asset_manifest(
    root: Path,
    contract: Mapping[str, Any],
    *,
    verify_all_files: bool,
) -> tuple[MaterializedAssetStatus, str | None, Mapping[str, Mapping[str, Any]]]:
    materialized = contract.get("materialized_assets")
    if not isinstance(materialized, Mapping):
        return MaterializedAssetStatus.MISSING, None, {}
    status = _manifest_status(materialized.get("status"))
    relative = materialized.get("relative_path")
    checksum = materialized.get("sha256")
    if relative is None or checksum is None:
        if status is MaterializedAssetStatus.COMPLETE:
            raise RecordingGeometryError("Complete materialized assets lack a manifest pointer.")
        return status, None, {}
    manifest_path = _safe_relative(root, relative, label="materialized_assets.relative_path")
    payload = manifest_path.read_bytes()
    expected = _normalized_sha256(checksum, label="materialized_assets.sha256")
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise RecordingGeometryError("Geometry asset manifest checksum mismatch.")
    manifest = _strict_json_loads(payload, label="recording geometry asset manifest")
    _require_schema(
        manifest,
        schema_id=RECORDING_GEOMETRY_ASSETS_SCHEMA_ID,
        schema_version=RECORDING_GEOMETRY_ASSETS_SCHEMA_VERSION,
        label="recording geometry asset manifest",
    )
    rows = manifest.get("files")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RecordingGeometryError("Geometry asset manifest files must be an array.")
    by_path: dict[str, Mapping[str, Any]] = {}
    assets_root = root / RECORDING_GEOMETRY_ASSETS_NAME
    for raw_row in rows:
        row = _required_mapping(raw_row, label="geometry asset manifest row")
        rel_text = _required_text(row.get("relative_path"), label="geometry asset relative_path")
        if rel_text in by_path:
            raise RecordingGeometryError(f"Duplicate geometry asset manifest path: {rel_text}")
        asset_path = _safe_relative(assets_root, rel_text, label="geometry asset relative_path")
        by_path[rel_text] = row
        if not verify_all_files and row.get("role") != "daily_rim_observation":
            continue
        if not asset_path.is_file():
            raise RecordingGeometryError(f"Missing materialized geometry asset: {rel_text}")
        if "size_bytes" in row and asset_path.stat().st_size != int(row["size_bytes"]):
            raise RecordingGeometryError(f"Geometry asset size mismatch: {rel_text}")
        expected_file_sha = _normalized_sha256(row.get("sha256"), label=f"{rel_text} sha256")
        if _sha256_file(asset_path) != expected_file_sha:
            raise RecordingGeometryError(f"Geometry asset checksum mismatch: {rel_text}")
    declared_count = manifest.get("materialized_file_count")
    if declared_count is not None and int(declared_count) != len(rows):
        raise RecordingGeometryError("Geometry asset manifest file count is inconsistent.")
    return status, actual, by_path


def _snapshot_contract_pointer(
    snapshot: Mapping[str, Any],
    *,
    root: Path,
    required: bool,
) -> tuple[Path | None, str | None, str]:
    pointer = snapshot.get("recording_geometry_contract")
    if not isinstance(pointer, Mapping):
        if required:
            raise RecordingGeometryError(
                "recording_snapshot lacks the recording_geometry_contract pointer."
            )
        return None, None, "missing"
    path = _safe_relative(root, pointer.get("relative_path"), label="recording geometry contract path")
    checksum = _normalized_sha256(pointer.get("sha256"), label="recording geometry contract sha256")
    return path, checksum, "verified"


def verify_recording_geometry_bundle(
    root: str | Path,
    *,
    require_snapshot_pointer: bool = False,
    verify_all_assets: bool = True,
) -> RecordingGeometryBundleVerification:
    """Verify the fixed v1 subtree without turning it into scientific authority."""

    bundle_root = Path(root).expanduser().resolve()
    snapshot_path = bundle_root / RECORDING_SNAPSHOT_NAME
    contract_path = bundle_root / RECORDING_GEOMETRY_CONTRACT_NAME
    assets_root = bundle_root / RECORDING_GEOMETRY_ASSETS_NAME
    if not snapshot_path.is_file() or not contract_path.is_file() or not assets_root.is_dir():
        raise RecordingGeometryError(
            "A geometry bundle requires recording_snapshot.json, "
            "recording_geometry_contract.json, and recording_geometry_assets/."
        )
    snapshot = _strict_json_loads(snapshot_path.read_bytes(), label="recording_snapshot.json")
    pointer_path, pointer_sha, pointer_status = _snapshot_contract_pointer(
        snapshot,
        root=bundle_root,
        required=require_snapshot_pointer,
    )
    if pointer_path is not None and pointer_path != contract_path:
        raise RecordingGeometryError("Snapshot contract pointer does not name the v1 contract file.")
    contract_bytes = contract_path.read_bytes()
    contract_sha = _sha256_bytes(contract_bytes)
    if pointer_sha is not None and pointer_sha != contract_sha:
        raise RecordingGeometryError("Snapshot contract checksum mismatch.")
    contract = _strict_json_loads(contract_bytes, label="recording_geometry_contract.json")
    _require_schema(
        contract,
        schema_id=RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID,
        schema_version=RECORDING_GEOMETRY_CONTRACT_SCHEMA_VERSION,
        label="recording geometry contract",
    )
    asset_status, manifest_sha, rows = _verify_asset_manifest(
        bundle_root,
        contract,
        verify_all_files=verify_all_assets,
    )
    return RecordingGeometryBundleVerification(
        root=bundle_root,
        contract_sha256=contract_sha,
        manifest_sha256=manifest_sha,
        manifest_file_count=len(rows),
        materialized_asset_status=asset_status,
        snapshot_pointer_status=pointer_status,
    )


def _collection_failure(
    *,
    policy: GeometryLoadPolicy,
    source_kind: str,
    source_location: str,
    status: MaskGeometryStatus,
    code: str,
    message: str,
) -> RegisteredDishMaskCollection:
    if policy is GeometryLoadPolicy.REQUIRED:
        raise RecordingGeometryError(message)
    return RegisteredDishMaskCollection(
        masks={},
        mask_geometry_status=status,
        source_kind=source_kind,
        source_location=source_location,
        issues=(GeometryLoadIssue(code=code, message=message),),
    )


def _registration_identity(contract: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    daily = _required_mapping(contract.get("daily_registration_geometry"), label="daily_registration_geometry")
    registration = _required_mapping(daily.get("registration"), label="daily_registration_geometry.registration")
    snapshot = _required_mapping(registration.get("snapshot"), label="daily registration snapshot")
    registration_id = _required_text(snapshot.get("registration_id"), label="registration_id")
    checksum = registration.get("sha256")
    normalized = _normalized_sha256(checksum, label="registration sha256") if checksum else None
    valid_until = snapshot.get("valid_until_utc")
    return registration_id, normalized, str(valid_until) if valid_until else None


def _mask_from_entry(
    entry: Mapping[str, Any],
    *,
    key: RegisteredDishMaskKey,
    registration_id: str,
    registration_sha256: str | None,
    source_contract_sha256: str,
    h5_scope_sha256: str | None,
    asset_status: MaterializedAssetStatus,
    citrus_status: CitrusRegistrationStatus,
    valid_until_utc: str | None,
    applied_by_citrus: bool | None,
    source_kind: str,
    source_location: str,
) -> RegisteredDishMask:
    physical, gate, width, height, operator_accepted, flags = _validate_mask_entry(
        entry,
        camera_serial=key.camera_serial,
        arena_id=key.arena_id,
    )
    artifact_id = _required_text(entry.get("artifact_id"), label="rim artifact_id")
    source = _required_mapping(entry.get("source"), label="rim observation source")
    source_sha = _normalized_sha256(source.get("sha256"), label="rim observation source sha256")
    return RegisteredDishMask(
        key=key,
        artifact_id=artifact_id,
        source_observation_sha256=source_sha,
        registration_id=registration_id,
        registration_sha256=registration_sha256,
        source_contract_sha256=source_contract_sha256,
        h5_scope_sha256=h5_scope_sha256,
        physical_inner_rim=physical,
        valid_detection_gate=gate,
        native_width_px=width,
        native_height_px=height,
        coordinate_space="camera_native_pixels",
        palette_space_id="source_camera_image_px",
        coordinate_profile_id="source_camera_image_px.top_left_y_down.v1",
        # Circle centers and radii are continuous point geometry in the native
        # camera image plane. They are not discrete array-index samples.
        pixel_convention="continuous",
        origin="top_left",
        positive_x="right",
        positive_y="down",
        target_plane="dish_top_rim",
        gating_semantics=str(
            entry.get("gating_semantics")
            or "bounding_box_centroid_inside_valid_detection_region"
        ),
        materialized_asset_status=asset_status,
        citrus_registration_status=citrus_status,
        source_valid_until_utc=valid_until_utc,
        producer_operator_accepted=operator_accepted,
        producer_quality_flags=flags,
        selected_daily_registration_applied_by_citrus=applied_by_citrus,
        source_kind=source_kind,
        source_location=source_location,
    )


def bind_registered_dish_mask_to_source_camera_frame(
    mask: RegisteredDishMask,
    source_camera_frame: Any,
) -> BoundRegisteredDishMask:
    """Bind acquisition geometry to Palette's exact persisted frame authority.

    Loaders cannot mint a pixel-frame authority: that authority belongs to the
    imported acquisition arrays in the analysis Zarr.  This boundary verifies
    that the loader's native-camera declaration is the same camera, extent,
    pixel convention, and semantic space before a publisher or consumer may
    combine the two.
    """

    from fisheye.shared.pixel_frame_authority import (
        require_source_camera_pixel_frame_authority,
    )

    frame = require_source_camera_pixel_frame_authority(source_camera_frame)
    endpoint = frame.endpoint
    camera_id = str(frame.reference_extent.record.camera_id)
    if camera_id != mask.key.camera_serial:
        raise RecordingGeometryError(
            "Registered dish mask camera does not match the source-camera authority."
        )
    if endpoint.space_id != mask.palette_space_id:
        raise RecordingGeometryError(
            "Registered dish mask space does not match the source-camera authority."
        )
    if endpoint.pixel_convention != mask.pixel_convention:
        raise RecordingGeometryError(
            "Registered dish mask pixel convention does not match the source-camera authority."
        )
    if endpoint.units != "px":
        raise RecordingGeometryError("Source-camera authority must use pixel units.")
    if endpoint.width != mask.native_width_px or endpoint.height != mask.native_height_px:
        raise RecordingGeometryError(
            "Registered dish mask dimensions do not match the source-camera authority."
        )
    frame.assert_verified()
    return BoundRegisteredDishMask(
        mask=mask,
        pixel_frame_record_ref=frame.record_ref,
        pixel_frame_record_sha256=frame.record_sha256,
    )


def load_registered_dish_masks_from_recording_folder(
    root: str | Path,
    *,
    policy: GeometryLoadPolicy | str = GeometryLoadPolicy.IF_AVAILABLE,
) -> RegisteredDishMaskCollection:
    """Load exact recording-bound masks through the snapshot pointer."""

    policy = GeometryLoadPolicy(policy)
    source_root = Path(root).expanduser().resolve()
    location = str(source_root)
    if policy is GeometryLoadPolicy.OFF:
        return RegisteredDishMaskCollection(
            masks={},
            mask_geometry_status=MaskGeometryStatus.MISSING,
            source_kind="orange_recording_folder",
            source_location=location,
        )
    snapshot_path = source_root / RECORDING_SNAPSHOT_NAME
    if not snapshot_path.is_file():
        return _collection_failure(
            policy=policy,
            source_kind="orange_recording_folder",
            source_location=location,
            status=MaskGeometryStatus.LEGACY_MISSING,
            code="legacy_missing_recording_snapshot",
            message="Recording folder lacks recording_snapshot.json.",
        )
    try:
        snapshot = _strict_json_loads(snapshot_path.read_bytes(), label="recording_snapshot.json")
        contract_path, expected_sha, _ = _snapshot_contract_pointer(
            snapshot,
            root=source_root,
            required=True,
        )
        assert contract_path is not None and expected_sha is not None
        contract_bytes = contract_path.read_bytes()
        contract_sha = _sha256_bytes(contract_bytes)
        if contract_sha != expected_sha:
            raise RecordingGeometryError("Recording geometry contract checksum mismatch.")
        contract = _strict_json_loads(contract_bytes, label="recording geometry contract")
        _require_schema(
            contract,
            schema_id=RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID,
            schema_version=RECORDING_GEOMETRY_CONTRACT_SCHEMA_VERSION,
            label="recording geometry contract",
        )
        daily = _required_mapping(contract.get("daily_registration_geometry"), label="daily_registration_geometry")
        selection_status = str(daily.get("status") or "missing")
        if daily.get("mode") != "selected_daily_registration" or selection_status not in {
            "selected_resolved",
            "selected_partial",
        }:
            return _collection_failure(
                policy=policy,
                source_kind="orange_recording_folder",
                source_location=location,
                status=MaskGeometryStatus.MISSING,
                code="recording_bound_mask_unavailable",
                message="Recording contract has no selected daily registration geometry.",
            )
        if selection_status == "selected_partial" and policy is GeometryLoadPolicy.REQUIRED:
            raise RecordingGeometryError("Required geometry does not accept selected_partial.")
        registration_id, registration_sha, valid_until = _registration_identity(contract)
        asset_status, _manifest_sha, manifest_rows = _verify_asset_manifest(
            source_root,
            contract,
            verify_all_files=selection_status == "selected_resolved",
        )
        selection = _required_mapping(contract.get("selection"), label="recording geometry selection")
        rig_id = _required_text(selection.get("rig_id"), label="selection.rig_id")
        canvas_name = _required_text(
            selection.get("selected_canvas_name"),
            label="selection.selected_canvas_name",
        )
        cameras = _required_mapping(daily.get("cameras"), label="daily_registration_geometry.cameras")
        masks: dict[RegisteredDishMaskKey, RegisteredDishMask] = {}
        issues: list[GeometryLoadIssue] = []
        for camera_key, raw_camera in cameras.items():
            camera_serial = str(camera_key)
            camera = _required_mapping(raw_camera, label=f"daily camera {camera_serial}")
            if camera.get("status") != "resolved":
                if selection_status == "selected_partial":
                    issues.append(
                        GeometryLoadIssue(
                            code="camera_geometry_unresolved",
                            message=f"Camera {camera_serial} is unresolved in selected_partial.",
                            camera_serial=camera_serial,
                        )
                    )
                    continue
                raise RecordingGeometryError(f"Camera {camera_serial} geometry is unresolved.")
            arena_id = _required_text(camera.get("arena_id"), label="camera arena_id")
            if str(camera.get("camera_serial")) != camera_serial:
                raise RecordingGeometryError("Daily camera_serial does not match its map key.")
            entry = _required_mapping(camera.get("recording_snapshot_entry"), label="recording_snapshot_entry")
            key = RegisteredDishMaskKey(rig_id, canvas_name, arena_id, camera_serial)
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
                applied_by_citrus=None,
                source_kind="orange_recording_folder",
                source_location=location,
            )
            _verify_snapshot_dimensions(
                snapshot,
                camera_serial=camera_serial,
                width=mask.native_width_px,
                height=mask.native_height_px,
            )
            _verify_snapshot_rim_reference(
                snapshot,
                camera_serial=camera_serial,
                artifact_id=mask.artifact_id,
                source_sha256=mask.source_observation_sha256,
            )
            observation_rel = (
                f"cameras/Cam{camera_serial}/daily_registration/"
                "rim_observation/observation.json"
            )
            if asset_status is MaterializedAssetStatus.COMPLETE:
                row = manifest_rows.get(observation_rel)
                if row is None:
                    raise RecordingGeometryError(
                        f"Asset manifest lacks exact rim observation {observation_rel}."
                    )
                if _normalized_sha256(row.get("sha256"), label="manifest rim sha256") != mask.source_observation_sha256:
                    raise RecordingGeometryError("Materialized rim checksum disagrees with the contract.")
            if key in masks:
                raise RecordingGeometryError(f"Duplicate geometry key: {key}")
            masks[key] = mask
        if not masks:
            raise RecordingGeometryError("No resolved recording-bound dish masks were found.")
        return RegisteredDishMaskCollection(
            masks=masks,
            mask_geometry_status=MaskGeometryStatus.VALID,
            source_kind="orange_recording_folder",
            source_location=location,
            source_contract_sha256=contract_sha,
            enclosing_selection_status=selection_status,
            issues=tuple(issues),
        )
    except (OSError, RecordingGeometryError) as exc:
        status = (
            MaskGeometryStatus.LEGACY_MISSING
            if "lacks the recording_geometry_contract pointer" in str(exc)
            else MaskGeometryStatus.INVALID
        )
        return _collection_failure(
            policy=policy,
            source_kind="orange_recording_folder",
            source_location=location,
            status=status,
            code=(
                "legacy_missing_recording_bound_mask"
                if status is MaskGeometryStatus.LEGACY_MISSING
                else "invalid_recording_bound_geometry"
            ),
            message=str(exc),
        )


def _exact_h5_scalar_bytes(dataset: h5py.Dataset, *, label: str) -> bytes:
    value: Any = dataset[()]
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise RecordingGeometryError(f"{label} must be a scalar UTF-8 dataset.")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecordingGeometryError(f"{label} is not UTF-8.") from exc
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise RecordingGeometryError(f"{label} must be a scalar UTF-8 dataset.")


def _verified_h5_json_dataset(
    group: h5py.Group,
    name: str,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes, str]:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise RecordingGeometryError(f"Missing H5 dataset {group.name}/{name}.")
    dataset = group[name]
    payload = _exact_h5_scalar_bytes(dataset, label=label)
    expected = _normalized_sha256(dataset.attrs.get("checksum_sha256"), label=f"{label} checksum")
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise RecordingGeometryError(f"{label} checksum mismatch.")
    return _strict_json_loads(payload, label=label), payload, actual


def _h5_runtime_status(
    *,
    runtime: Mapping[str, Any],
    dataset: h5py.Dataset,
    registration_id: str,
    registration_sha256: str,
    rig_id: str,
    canvas_name: str,
    camera_serial: str,
    arena_id: str,
    rim_artifact_id: str,
    rim_source_path: str,
    rim_source_sha256: str,
) -> CitrusRegistrationStatus:
    if dataset.attrs.get("load_status") != "loaded":
        return CitrusRegistrationStatus.MISSING
    if _normalized_sha256(dataset.attrs.get("checksum_sha256"), label="runtime checksum") != registration_sha256:
        return CitrusRegistrationStatus.CHECKSUM_MISMATCH
    if runtime.get("schema_id") != DAILY_REGISTRATION_SCHEMA_ID or runtime.get("schema_version") != DAILY_REGISTRATION_SCHEMA_VERSION or runtime.get("status") != "accepted":
        return CitrusRegistrationStatus.INVALID
    if runtime.get("registration_id") != registration_id:
        return CitrusRegistrationStatus.REGISTRATION_ID_MISMATCH
    if runtime.get("rig_id") != rig_id or runtime.get("canvas_name") != canvas_name:
        return CitrusRegistrationStatus.INVALID
    targets = runtime.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return CitrusRegistrationStatus.INVALID
    matching = [
        item
        for item in targets
        if isinstance(item, Mapping)
        and str(item.get("camera_id")) == camera_serial
        and item.get("arena_id") == arena_id
    ]
    if len(matching) != 1:
        return CitrusRegistrationStatus.CAMERA_ARENA_TARGET_MISSING
    rim = matching[0].get("rim_observation")
    if not isinstance(rim, Mapping):
        return CitrusRegistrationStatus.RIM_OBSERVATION_MISMATCH
    try:
        rim_sha = _normalized_sha256(rim.get("sha256"), label="runtime rim sha256")
    except RecordingGeometryError:
        return CitrusRegistrationStatus.RIM_OBSERVATION_MISMATCH
    if (
        rim.get("artifact_id") != rim_artifact_id
        or rim.get("path") != rim_source_path
        or rim_sha != rim_source_sha256
    ):
        return CitrusRegistrationStatus.RIM_OBSERVATION_MISMATCH
    return CitrusRegistrationStatus.EXACT_MATCH_APPLIED


def load_registered_dish_masks_from_citrus_h5(
    source_h5: str | Path,
    *,
    policy: GeometryLoadPolicy | str = GeometryLoadPolicy.IF_AVAILABLE,
) -> RegisteredDishMaskCollection:
    """Load arena-scoped masks from exact embedded Citrus H5 payloads."""

    policy = GeometryLoadPolicy(policy)
    path = Path(source_h5).expanduser().resolve()
    location = str(path)
    if policy is GeometryLoadPolicy.OFF:
        return RegisteredDishMaskCollection(
            masks={},
            mask_geometry_status=MaskGeometryStatus.MISSING,
            source_kind="citrus_h5",
            source_location=location,
        )
    try:
        with h5py.File(path, "r") as h5:
            if "recording_geometry_contract" not in h5:
                return _collection_failure(
                    policy=policy,
                    source_kind="citrus_h5",
                    source_location=location,
                    status=MaskGeometryStatus.LEGACY_MISSING,
                    code="legacy_missing_recording_bound_mask",
                    message="H5 lacks /recording_geometry_contract.",
                )
            recording_group = h5["recording_geometry_contract"]
            if (
                recording_group.attrs.get("capture_status") != "embedded_verified"
                or int(recording_group.attrs.get("checksum_verified", 0)) != 1
            ):
                return _collection_failure(
                    policy=policy,
                    source_kind="citrus_h5",
                    source_location=location,
                    status=MaskGeometryStatus.LEGACY_MISSING,
                    code="legacy_missing_recording_bound_mask",
                    message="H5 recording geometry was not embedded and verified.",
                )
            if (
                recording_group.attrs.get("schema_id")
                != RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID
                or int(recording_group.attrs.get("schema_version", -1))
                != RECORDING_GEOMETRY_CONTRACT_SCHEMA_VERSION
            ):
                raise RecordingGeometryError(
                    "H5 /recording_geometry_contract schema declaration is unsupported."
                )
            contract, _contract_bytes, contract_sha = _verified_h5_json_dataset(
                recording_group,
                "contract_json",
                label="recording geometry contract_json",
            )
            _require_schema(
                contract,
                schema_id=RECORDING_GEOMETRY_CONTRACT_SCHEMA_ID,
                schema_version=RECORDING_GEOMETRY_CONTRACT_SCHEMA_VERSION,
                label="embedded recording geometry contract",
            )
            scope, _scope_bytes, scope_sha = _verified_h5_json_dataset(
                recording_group,
                "h5_scope_json",
                label="recording geometry h5_scope_json",
            )
            _require_schema(
                scope,
                schema_id=H5_SCOPE_SCHEMA_ID,
                schema_version=H5_SCOPE_SCHEMA_VERSION,
                label="recording geometry H5 scope",
            )
            if scope.get("scope_status") != "resolved":
                raise RecordingGeometryError("H5 recording geometry scope is unresolved.")
            source_contract = _required_mapping(scope.get("source_contract"), label="scope.source_contract")
            if _normalized_sha256(source_contract.get("sha256"), label="scope source contract sha256") != contract_sha:
                raise RecordingGeometryError("H5 scope source-contract checksum mismatch.")
            attr_source_sha = recording_group["h5_scope_json"].attrs.get("source_contract_sha256")
            if attr_source_sha is None:
                raise RecordingGeometryError(
                    "H5 scope dataset lacks source_contract_sha256."
                )
            if _normalized_sha256(attr_source_sha, label="scope dataset source checksum") != contract_sha:
                raise RecordingGeometryError("H5 scope dataset source-contract checksum mismatch.")

            target = _required_mapping(scope.get("target"), label="scope.target")
            rig_id = _required_text(target.get("rig_id"), label="scope target rig_id")
            canvas_name = _required_text(target.get("canvas_name"), label="scope target canvas_name")
            arena_id = _required_text(target.get("arena_id"), label="scope target arena_id")
            associated = target.get("associated_camera_ids")
            if not isinstance(associated, Sequence) or isinstance(associated, (str, bytes)) or not associated:
                raise RecordingGeometryError("H5 scope has no associated camera IDs.")

            if "runtime_geometry_contract" not in h5:
                raise RecordingGeometryError("H5 lacks /runtime_geometry_contract.")
            runtime_group = h5["runtime_geometry_contract"]
            runtime, _runtime_bytes, runtime_sha = _verified_h5_json_dataset(
                runtime_group,
                "daily_registration_json",
                label="runtime daily_registration_json",
            )
            runtime_dataset = runtime_group["daily_registration_json"]
            cameras = _required_mapping(scope.get("cameras"), label="scope.cameras")
            masks: dict[RegisteredDishMaskKey, RegisteredDishMask] = {}
            issues: list[GeometryLoadIssue] = []
            for raw_serial in associated:
                camera_serial = _required_text(str(raw_serial), label="associated camera serial")
                camera = _required_mapping(cameras.get(camera_serial), label=f"scope camera {camera_serial}")
                if camera.get("arena_id") != arena_id:
                    raise RecordingGeometryError("H5 scope camera arena differs from target arena.")
                daily = _required_mapping(camera.get("daily_registration_geometry"), label="camera daily_registration_geometry")
                if daily.get("schema_id") != "orange.recording.daily_registration_camera_geometry" or daily.get("schema_version") != 1 or daily.get("status") != "resolved" or daily.get("mode") != "selected_daily_registration":
                    raise RecordingGeometryError("H5 camera daily registration geometry is invalid.")
                registration_id = _required_text(daily.get("registration_id"), label="camera registration_id")
                registration = _required_mapping(daily.get("registration"), label="camera registration")
                registration_sha = _normalized_sha256(registration.get("sha256"), label="camera registration sha256")
                expected_runtime_path = _required_text(
                    registration.get("source_path"),
                    label="camera registration source_path",
                )
                if registration_sha != runtime_sha:
                    citrus_status = CitrusRegistrationStatus.CHECKSUM_MISMATCH
                elif runtime_dataset.attrs.get("source_path") != expected_runtime_path:
                    citrus_status = CitrusRegistrationStatus.INVALID
                else:
                    entry_for_status = _required_mapping(daily.get("recording_snapshot_entry"), label="recording_snapshot_entry")
                    source_for_status = _required_mapping(entry_for_status.get("source"), label="rim source")
                    citrus_status = _h5_runtime_status(
                        runtime=runtime,
                        dataset=runtime_dataset,
                        registration_id=registration_id,
                        registration_sha256=registration_sha,
                        rig_id=rig_id,
                        canvas_name=canvas_name,
                        camera_serial=camera_serial,
                        arena_id=arena_id,
                        rim_artifact_id=_required_text(entry_for_status.get("artifact_id"), label="rim artifact_id"),
                        rim_source_path=_required_text(source_for_status.get("path"), label="rim source path"),
                        rim_source_sha256=_normalized_sha256(source_for_status.get("sha256"), label="rim source sha256"),
                    )
                entry = _required_mapping(daily.get("recording_snapshot_entry"), label="recording_snapshot_entry")
                applied_value = daily.get("selected_daily_registration_applied_by_citrus")
                applied = applied_value if isinstance(applied_value, bool) else None
                if applied is not True:
                    citrus_status = (
                        CitrusRegistrationStatus.INVALID
                        if applied is False
                        else CitrusRegistrationStatus.MISSING
                    )
                key = RegisteredDishMaskKey(rig_id, canvas_name, arena_id, camera_serial)
                mask = _mask_from_entry(
                    entry,
                    key=key,
                    registration_id=registration_id,
                    registration_sha256=registration_sha,
                    source_contract_sha256=contract_sha,
                    h5_scope_sha256=scope_sha,
                    asset_status=MaterializedAssetStatus.MISSING,
                    citrus_status=citrus_status,
                    valid_until_utc=str(registration.get("valid_until_utc")) if registration.get("valid_until_utc") else None,
                    applied_by_citrus=applied,
                    source_kind="citrus_h5",
                    source_location=location,
                )
                if citrus_status is not CitrusRegistrationStatus.EXACT_MATCH_APPLIED:
                    issues.append(
                        GeometryLoadIssue(
                            code=citrus_status.value,
                            message=f"Citrus runtime identity did not exactly match for {camera_serial}/{arena_id}.",
                            camera_serial=camera_serial,
                            arena_id=arena_id,
                        )
                    )
                    if policy is GeometryLoadPolicy.REQUIRED:
                        raise RecordingGeometryError(issues[-1].message)
                if key in masks:
                    raise RecordingGeometryError(f"Duplicate H5 geometry key: {key}")
                masks[key] = mask
            if not masks:
                raise RecordingGeometryError("No arena-scoped H5 masks were resolved.")
            return RegisteredDishMaskCollection(
                masks=masks,
                mask_geometry_status=MaskGeometryStatus.VALID,
                source_kind="citrus_h5",
                source_location=location,
                source_contract_sha256=contract_sha,
                h5_scope_sha256=scope_sha,
                enclosing_selection_status="selected_resolved",
                issues=tuple(issues),
            )
    except (OSError, RecordingGeometryError) as exc:
        return _collection_failure(
            policy=policy,
            source_kind="citrus_h5",
            source_location=location,
            status=MaskGeometryStatus.INVALID,
            code="invalid_recording_bound_geometry",
            message=str(exc),
        )


__all__ = [
    "BoundRegisteredDishMask",
    "CircleGeometry",
    "CitrusRegistrationStatus",
    "GeometryLoadIssue",
    "GeometryLoadPolicy",
    "MaskGeometryStatus",
    "MaterializedAssetStatus",
    "RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH",
    "RecordingGeometryBundleVerification",
    "RecordingGeometryError",
    "RegisteredDishMask",
    "RegisteredDishMaskCollection",
    "RegisteredDishMaskKey",
    "bind_registered_dish_mask_to_source_camera_frame",
    "load_registered_dish_masks_from_citrus_h5",
    "load_registered_dish_masks_from_recording_folder",
    "verify_recording_geometry_bundle",
]
