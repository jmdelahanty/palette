"""Common physical authority for source-camera pixel coordinates.

This authority is independent of stimulus selection. Stimulus H5 import and an
operator-verified donor repair may both publish the same
camera/dimension/scale-bound physical frame and identify their evidence source
in the manifest. A future non-H5 acquisition-calibration schema should feed the
same authority without pretending that manifest evidence originated in H5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from fisheye.shared.archive_identity import ArchiveIdentity, archive_identity
from fisheye.shared.coordinate_frame_record import (
    PHYSICAL_FRAME_CALIBRATION_ATTR,
    REFERENCE_EXTENT_FINITE,
    BoundPhysicalFrameCalibration,
    build_physical_frame_calibration_record,
    load_bound_physical_frame_calibration,
    load_bound_selected_camera_frame_evidence,
    stamp_physical_frame_calibration_record,
    stamp_selected_camera_frame_evidence,
    verify_bound_coordinate_frame,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
    verify_bound_coordinate_record,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.selected_calibration import (
    VerifiedSelectedCameraSourceEvidence,
    require_verified_selected_camera_source_evidence,
)

SOURCE_CAMERA_PHYSICAL_MANIFEST_ATTR = "source_camera_physical_authority"
SOURCE_CAMERA_PHYSICAL_MANIFEST_DIGEST_ATTR = (
    f"{SOURCE_CAMERA_PHYSICAL_MANIFEST_ATTR}_sha256"
)
SOURCE_CAMERA_PHYSICAL_SCHEMA_ID = "palette.source_camera_physical_authority"
SOURCE_CAMERA_PHYSICAL_SCHEMA_VERSION = 1
_SEAL = object()


class SourceCameraPhysicalAuthorityError(ValueError):
    """Raised when recording/stimulus-neutral physical evidence is invalid."""


@dataclass(frozen=True, init=False)
class BoundSourceCameraPhysicalAuthority:
    camera_id: str
    source_kind: str
    archive_identity: ArchiveIdentity
    physical_frame: BoundPhysicalFrameCalibration = field(repr=False, compare=False)
    manifest: BoundCoordinateRecord = field(repr=False, compare=False)
    _root: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        camera_id: str,
        source_kind: str,
        archive_identity: ArchiveIdentity,
        physical_frame: BoundPhysicalFrameCalibration,
        manifest: BoundCoordinateRecord,
        root: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _SEAL:
            raise SourceCameraPhysicalAuthorityError(
                "Bound source-camera physical authority cannot be constructed directly."
            )
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "archive_identity", archive_identity)
        object.__setattr__(self, "physical_frame", physical_frame)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def mm_per_pixel(self) -> float:
        return float(self.physical_frame.record.mm_per_pixel)

    def assert_verified(self) -> None:
        current = load_source_camera_physical_authority(self._root)
        if (
            current.manifest.record_ref != self.manifest.record_ref
            or current.manifest.record_sha256 != self.manifest.record_sha256
            or current.physical_frame.record_ref != self.physical_frame.record_ref
            or current.physical_frame.record_sha256 != self.physical_frame.record_sha256
        ):
            raise SourceCameraPhysicalAuthorityError(
                "Source-camera physical authority changed after binding."
            )


def require_bound_source_camera_physical_authority(
    value: Any,
) -> BoundSourceCameraPhysicalAuthority:
    if (
        type(value) is not BoundSourceCameraPhysicalAuthority
        or value._seal is not _SEAL
    ):
        raise SourceCameraPhysicalAuthorityError(
            "A freshly loader-minted source-camera physical authority is required."
        )
    value.assert_verified()
    return value


def _group(parent: Any, name: str) -> Any:
    try:
        return parent[name]
    except KeyError:
        create = getattr(parent, "create_group", None)
        if callable(create):
            return create(name)
        token = getattr(parent, "_coordinate_archive_token", None)
        child = type(parent)(
            path=f"{parent.path}/{name}" if parent.path else name,
            archive_token=token,
        )
        parent[name] = child
        return child


def _container(root: Any, *, create: bool = False) -> Any:
    analysis = root["analysis"]
    calibration = _group(analysis, "calibration") if create else analysis["calibration"]
    return (
        _group(calibration, "coordinate_frames")
        if create
        else calibration["coordinate_frames"]
    )


def _child(root: Any, path: str) -> Any:
    node = root
    for name in path.split("/"):
        node = node[name]
    return node


def _manifest_record(
    *,
    camera_id: str,
    source_kind: str,
    evidence_ref: str,
    evidence_sha256: str,
    physical: BoundPhysicalFrameCalibration,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": SOURCE_CAMERA_PHYSICAL_SCHEMA_ID,
        "schema_version": SOURCE_CAMERA_PHYSICAL_SCHEMA_VERSION,
        "camera_id": camera_id,
        "source_kind": source_kind,
        "selected_camera_evidence": {
            "record_ref": evidence_ref,
            "record_sha256": evidence_sha256,
        },
        "physical_frame": {
            "record_ref": physical.record_ref,
            "record_sha256": physical.record_sha256,
        },
        "source_camera_frame": {
            "record_ref": physical.source_camera_pixels.record_ref,
            "record_sha256": physical.source_camera_pixels.record_sha256,
        },
        "mm_per_pixel": float(physical.record.mm_per_pixel),
        "provenance": dict(provenance),
    }


def publish_source_camera_physical_authority(
    root: Any,
    *,
    source_camera_evidence: VerifiedSelectedCameraSourceEvidence,
    source_kind: str,
    provenance: Mapping[str, Any],
) -> BoundSourceCameraPhysicalAuthority:
    if (
        not isinstance(source_kind, str)
        or not source_kind
        or source_kind != source_kind.strip()
    ):
        raise SourceCameraPhysicalAuthorityError(
            "source_kind must be nonempty without surrounding whitespace."
        )
    if not isinstance(provenance, Mapping):
        raise SourceCameraPhysicalAuthorityError("provenance must be an object.")
    evidence = require_verified_selected_camera_source_evidence(source_camera_evidence)
    camera_id = evidence.active_camera_id
    _, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=camera_id,
    )
    analysis = root["analysis"]
    try:
        calibration = analysis["calibration"]
    except KeyError:
        calibration = None
    if calibration is None:
        existing_container = None
    else:
        try:
            existing_container = calibration["coordinate_frames"]
        except KeyError:
            existing_container = None
    if (
        existing_container is not None
        and SOURCE_CAMERA_PHYSICAL_MANIFEST_ATTR in existing_container.attrs
    ):
        existing = load_source_camera_physical_authority(root)
        existing_record = existing.physical_frame.record
        existing_endpoint = existing.physical_frame.source_camera_pixels.endpoint
        if (
            existing.camera_id != camera_id
            or existing_endpoint.width != evidence.native_width_px
            or existing_endpoint.height != evidence.native_height_px
            or evidence.pixels_per_mm_camera is None
            or existing_record.pixels_per_mm_camera != evidence.pixels_per_mm_camera
        ):
            raise SourceCameraPhysicalAuthorityError(
                "Existing source-camera physical authority conflicts with new "
                "camera, dimensions, or scale evidence."
            )
        return existing
    coordinate_frames = _group(analysis, "coordinate_frames")
    source_camera = _group(coordinate_frames, "source_camera")
    camera = _group(source_camera, camera_id)
    source_node = _group(camera, "continuous")
    if "pixel_frame_authority" in source_node.attrs:
        source_frame = load_source_camera_pixel_frame_authority(
            source_node,
            acquisition_frame=acquisition,
        )
    else:
        source_frame = stamp_source_camera_pixel_frame_authority(
            source_node,
            frame_id=f"{camera_id}_source_camera",
            pixel_convention="continuous",
            acquisition_frame=acquisition,
        )
    container = _container(root, create=True)
    selected_node = _group(container, "selected_camera_evidence")
    physical_node = _group(container, "source_camera_physical_mm")
    selected = stamp_selected_camera_frame_evidence(
        selected_node,
        source_camera=evidence,
    )
    record = build_physical_frame_calibration_record(
        frame_id=f"recording_{camera_id}_source_camera_physical_mm",
        source_camera_pixels=source_frame,
        selected_camera_evidence=selected,
        physical_extent_mode=REFERENCE_EXTENT_FINITE,
    )
    physical = stamp_physical_frame_calibration_record(
        physical_node,
        record,
        expected_record_ref=(
            f"/{physical_node.path}@{PHYSICAL_FRAME_CALIBRATION_ATTR}"
        ),
        source_camera_pixels=source_frame,
        selected_camera_evidence=selected,
    )
    stamp_and_bind_persisted_coordinate_record(
        container,
        _manifest_record(
            camera_id=camera_id,
            source_kind=source_kind,
            evidence_ref=selected.record_ref,
            evidence_sha256=selected.record_sha256,
            physical=physical,
            provenance=provenance,
        ),
        attr_name=SOURCE_CAMERA_PHYSICAL_MANIFEST_ATTR,
        digest_attr_name=SOURCE_CAMERA_PHYSICAL_MANIFEST_DIGEST_ATTR,
    )
    return load_source_camera_physical_authority(root)


def load_source_camera_physical_authority(
    root: Any,
) -> BoundSourceCameraPhysicalAuthority:
    container = _container(root)
    manifest = bind_persisted_coordinate_record(
        container,
        attr_name=SOURCE_CAMERA_PHYSICAL_MANIFEST_ATTR,
        digest_attr_name=SOURCE_CAMERA_PHYSICAL_MANIFEST_DIGEST_ATTR,
    )
    payload = manifest.record
    if (
        payload.get("schema_id") != SOURCE_CAMERA_PHYSICAL_SCHEMA_ID
        or payload.get("schema_version") != SOURCE_CAMERA_PHYSICAL_SCHEMA_VERSION
        or set(payload)
        != {
            "schema_id",
            "schema_version",
            "camera_id",
            "source_kind",
            "selected_camera_evidence",
            "physical_frame",
            "source_camera_frame",
            "mm_per_pixel",
            "provenance",
        }
    ):
        raise SourceCameraPhysicalAuthorityError(
            "Unsupported source-camera physical authority manifest."
        )
    camera_id = str(payload.get("camera_id") or "")
    source_kind = payload.get("source_kind")
    provenance = payload.get("provenance")
    if (
        not camera_id
        or not isinstance(source_kind, str)
        or not source_kind.strip()
        or not isinstance(provenance, Mapping)
    ):
        raise SourceCameraPhysicalAuthorityError(
            "Source-camera physical authority identity/provenance is invalid."
        )
    _, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=camera_id,
    )
    source_node = _child(
        root,
        f"analysis/coordinate_frames/source_camera/{camera_id}/continuous",
    )
    source_frame = load_source_camera_pixel_frame_authority(
        source_node,
        acquisition_frame=acquisition,
    )
    selected_node = container["selected_camera_evidence"]
    selected_pointer = payload["selected_camera_evidence"]
    if not isinstance(selected_pointer, Mapping) or set(selected_pointer) != {
        "record_ref",
        "record_sha256",
    }:
        raise SourceCameraPhysicalAuthorityError(
            "Selected-camera evidence pointer is not closed."
        )
    selected = load_bound_selected_camera_frame_evidence(
        selected_node,
        expected_record_ref=selected_pointer["record_ref"],
        expected_record_sha256=selected_pointer["record_sha256"],
        expected_camera_id=camera_id,
    )
    physical_node = container["source_camera_physical_mm"]
    physical_pointer = payload["physical_frame"]
    if not isinstance(physical_pointer, Mapping) or set(physical_pointer) != {
        "record_ref",
        "record_sha256",
    }:
        raise SourceCameraPhysicalAuthorityError(
            "Physical-frame pointer is not closed."
        )
    physical = load_bound_physical_frame_calibration(
        physical_node,
        expected_record_ref=physical_pointer["record_ref"],
        expected_record_sha256=physical_pointer["record_sha256"],
        expected_camera_id=camera_id,
        source_camera_pixels=source_frame,
        selected_camera_evidence=selected,
    )
    verify_bound_coordinate_record(manifest)
    verify_bound_coordinate_frame(physical, expected_kind="physical_frame_calibration")
    source_pointer = payload["source_camera_frame"]
    if (
        not isinstance(source_pointer, Mapping)
        or set(source_pointer) != {"record_ref", "record_sha256"}
        or source_pointer["record_ref"] != source_frame.record_ref
        or source_pointer["record_sha256"] != source_frame.record_sha256
    ):
        raise SourceCameraPhysicalAuthorityError(
            "Manifest source-camera frame pointer differs from the bound frame."
        )
    if float(payload.get("mm_per_pixel")) != float(physical.record.mm_per_pixel):
        raise SourceCameraPhysicalAuthorityError(
            "Manifest scale differs from physical frame."
        )
    return BoundSourceCameraPhysicalAuthority(
        camera_id=camera_id,
        source_kind=source_kind,
        archive_identity=archive_identity(container),
        physical_frame=physical,
        manifest=manifest,
        root=root,
        _verification_seal=_SEAL,
    )


__all__ = [
    "BoundSourceCameraPhysicalAuthority",
    "SourceCameraPhysicalAuthorityError",
    "load_source_camera_physical_authority",
    "publish_source_camera_physical_authority",
    "require_bound_source_camera_physical_authority",
]
