"""Position-surface resolver branch for sealed geometry-only crop profile v2.

The crop manifest is already the immutable coordinate authority. This module
validates that profile at full strength and projects its existing proof into
the common source-camera position interface without adding attrs, arrays, or a
successor rowset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    require_bound_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    BoundSourceRowTemporalAuthority,
    _bind_manifest_row_identity_contract,
    _bind_manifest_source_row_temporal_authority,
    require_bound_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    _bind_persisted_manifest_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_POINT_XY
from fisheye.shared.observation_coordinate_publication import (
    ObservationCoordinatePublicationError,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
    require_trusted_coordinate_attrs,
)
from fisheye.shared.proof_verification import proof_verification_operation
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_ATTRIBUTE,
    CROP_RUN_MANIFEST_SCHEMA_ID,
    crop_pixel_authority_from_manifest,
)
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)

_DIRECT_PATH_SELECTION_CONTRACT = "none_shadow_direct_path_only"
_LEGACY_POSITION_LINEAGE_ATTRS = (
    "detection_acquisition_frame_mapping",
    "crop_geometry_selection",
    "collection_proxy_coordinate_successor_mapping",
)
_MANIFEST_CROP_POSITION_PROOF_SEAL = object()


@dataclass(frozen=True, init=False)
class _BoundManifestCropPositionProof:
    """Internal profile proof accepted only by the shared position resolver."""

    coordinates: BoundCanonicalCoordinateDescriptor
    temporal_authority: BoundSourceRowTemporalAuthority
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        coordinates: BoundCanonicalCoordinateDescriptor,
        temporal_authority: BoundSourceRowTemporalAuthority,
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _MANIFEST_CROP_POSITION_PROOF_SEAL:
            raise TypeError(
                "Manifest crop position proof must come from the validated profile loader."
            )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "temporal_authority", temporal_authority)
        object.__setattr__(self, "_seal", _seal)


def _fail(message: str) -> None:
    raise ObservationCoordinatePublicationError(message)


def _is_manifest_bound_crop_position_rowset(rowset: Any) -> bool:
    """Return whether dispatch metadata declares the sealed geometry profile."""

    attrs = getattr(rowset, "attrs", None)
    return bool(
        isinstance(attrs, Mapping)
        and attrs.get("artifact_class") == "geometry_only_analysis"
        and isinstance(attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE), Mapping)
    )


def _local_archive_path(root_node: Any) -> Path:
    identity = archive_identity(root_node)
    if identity.kind != "local_store_root" or not identity.key:
        _fail(
            "Sealed geometry-only crop validation currently requires one exact "
            "local archive store so direct and consolidated metadata can be "
            "reopened and compared."
        )
    return Path(str(identity.key[0]))


def _position_array_values(
    rowset: Any,
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    payload = manifest["payload"]
    logical_content = payload["logical_content"]["document"]
    declarations = logical_content["arrays"]
    values: dict[str, np.ndarray] = {}
    for name in (
        "instance_key",
        "source_acquisition_frame_index",
        "bbox_img_xyxy",
        "centers_img_xy",
    ):
        node = rowset[name]
        array = np.array(node[:], copy=True, order="C")
        declaration = declarations.get(name)
        observed = {
            "shape": [int(item) for item in array.shape],
            "dtype": str(array.dtype),
            "digest_algorithm": "sha256_c_contiguous_bytes_v1",
            "sha256": sha256_array(array),
        }
        if not isinstance(declaration, Mapping) or dict(declaration) != observed:
            _fail(
                f"Sealed geometry-only crop array {name!r} differs from its "
                "validated manifest declaration."
            )
        values[name] = array
    bbox = values["bbox_img_xyxy"]
    centers = values["centers_img_xy"]
    expected_centers = np.column_stack(
        (
            (bbox[:, 0] + bbox[:, 2]) * np.float32(0.5),
            (bbox[:, 1] + bbox[:, 3]) * np.float32(0.5),
        )
    ).astype(np.float32, copy=False)
    if not np.array_equal(centers, expected_centers):
        _fail(
            "Sealed geometry-only crop centers are not the exact persisted "
            "bbox midpoints."
        )
    return values


def _validated_profile(
    root_node: Any,
    rowset_path: str,
) -> tuple[Any, Mapping[str, Any], Any, Any]:
    normalized = str(rowset_path).strip().strip("/")
    parts = normalized.split("/")
    if len(parts) != 2 or parts[0] != "crop_runs" or not parts[1]:
        _fail(
            "Sealed geometry-only crop positions require one exact "
            "crop_runs/<run> rowset."
        )
    try:
        rowset = root_node[normalized]
    except Exception as exc:
        _fail(f"Unable to open sealed geometry-only crop rowset: {exc}.")
    if canonical_node_path(rowset) != normalized:
        _fail("Sealed geometry-only crop rowset resolved to an unexpected path.")
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Sealed geometry-only crop position rowset",
    )
    if any(name in attrs for name in _LEGACY_POSITION_LINEAGE_ATTRS):
        _fail(
            "Sealed geometry-only crop rowset cannot also declare a legacy "
            "position-lineage profile."
        )
    publication = open_persisted_crop_geometry_publication(
        _local_archive_path(root_node),
        run_id=parts[1],
    )
    manifest = publication.manifest
    payload = manifest.get("payload")
    publication_record = (
        payload.get("publication") if isinstance(payload, Mapping) else None
    )
    if (
        manifest.get("schema_id") != CROP_RUN_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        or attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE) != manifest
        or not isinstance(publication_record, Mapping)
        or publication_record.get("artifact_class") != "geometry_only_analysis"
        or publication_record.get("completion_status") != "complete"
        or publication_record.get("stage_selector_eligible") is not False
        or attrs.get("status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("artifact_class") != "geometry_only_analysis"
        or attrs.get("production_candidate") is not True
        or attrs.get("immutable_snapshot") is not True
    ):
        _fail(
            "Sealed geometry-only crop profile identity, completion, "
            "immutability, or selector polarity is invalid."
        )
    parent_selection = root_node["crop_runs"].attrs.get("selection_contract")
    if parent_selection not in (None, _DIRECT_PATH_SELECTION_CONTRACT):
        _fail("Geometry-only crop parent has an unsupported selection contract.")
    if set(rowset.keys()) != set(CROP_GEOMETRY_SCHEMA_V1.binding_paths):
        _fail(
            "Sealed geometry-only crop rowset does not have the exact profile "
            "array topology."
        )

    dimensions = publication.dimensions
    arrays = _position_array_values(rowset, manifest)
    row_count = int(arrays["instance_key"].shape[0])
    if row_count != int(dimensions.n_instances):
        _fail("Geometry-only crop row count differs from its sealed dimensions.")
    # A manifest with n_instances==0 is the explicit empty-row proof for this
    # profile: the full 13-array topology and every zero-length digest remain
    # required. No legacy empty-observation attr is inferred or accepted.

    pixel_manifest = payload.get("source_pixel_authority")
    if not isinstance(pixel_manifest, Mapping):
        _fail("Geometry-only crop manifest lacks source-pixel authority.")
    try:
        pixel_authority = crop_pixel_authority_from_manifest(pixel_manifest)
        _, acquisition = load_persisted_acquisition_camera_authority(
            root_node,
            expected_camera_id=pixel_authority.camera_identity,
        )
        camera = load_source_camera_pixel_frame_authority(
            root_node[
                "analysis/coordinate_frames/source_camera/"
                f"{pixel_authority.camera_identity}/continuous"
            ],
            acquisition_frame=acquisition,
        )
    except Exception as exc:
        _fail(
            "Geometry-only crop source-pixel and live acquisition authority "
            f"cannot be jointly validated: {exc}."
        )
    if (
        pixel_authority.recording_identity != acquisition.record.recording_id
        or pixel_authority.camera_identity != acquisition.record.camera_id
        or pixel_authority.n_frames != acquisition.record.source_total_frames
        or pixel_authority.source_width != camera.endpoint.width
        or pixel_authority.source_height != camera.endpoint.height
        or dimensions.n_frames != pixel_authority.n_frames
        or dimensions.source_width != pixel_authority.source_width
        or dimensions.source_height != pixel_authority.source_height
    ):
        _fail(
            "Geometry-only crop source-pixel manifest disagrees with the live "
            "acquisition camera authority."
        )
    frame_indices = arrays["source_acquisition_frame_index"]
    if frame_indices.size and (
        np.any(frame_indices < 0)
        or np.any(frame_indices >= acquisition.record.source_total_frames)
    ):
        _fail("Geometry-only crop acquisition frames are outside the live domain.")
    return rowset, manifest, acquisition, camera


@proof_verification_operation
def _load_manifest_crop_position_proof(
    root_node: Any,
    rowset_path: str,
) -> _BoundManifestCropPositionProof:
    """Validate one geometry-only profile for the shared position resolver."""

    rowset, _manifest, acquisition, camera = _validated_profile(
        root_node,
        rowset_path,
    )
    manifest_authority = _bind_persisted_manifest_coordinate_record(rowset)
    identity = _bind_manifest_row_identity_contract(
        rowset,
        rowset["instance_key"],
        manifest_authority=manifest_authority,
    )
    temporal = _bind_manifest_source_row_temporal_authority(
        rowset,
        rowset["source_acquisition_frame_index"],
        source_row_identity=identity,
        acquisition_frame=acquisition,
        manifest_authority=manifest_authority,
    )
    coordinates = build_bound_canonical_coordinate_descriptor(
        rowset["centers_img_xy"],
        **SOURCE_CAMERA_POINT_XY.descriptor_kwargs(),
        row_identity=identity,
        reference_frame_authority=camera,
        lineage_records=(manifest_authority,),
    )
    return _BoundManifestCropPositionProof(
        coordinates,
        temporal,
        _seal=_MANIFEST_CROP_POSITION_PROOF_SEAL,
    )


def _require_manifest_crop_position_proof(
    value: Any,
) -> _BoundManifestCropPositionProof:
    """Revalidate the internal proof before the resolver builds its surface."""

    if (
        type(value) is not _BoundManifestCropPositionProof
        or value._seal is not _MANIFEST_CROP_POSITION_PROOF_SEAL
    ):
        _fail("A sealed manifest-crop position profile proof is required.")
    require_bound_canonical_coordinate_descriptor(value.coordinates)
    require_bound_source_row_temporal_authority(value.temporal_authority)
    return value


# This module is an implementation branch of the shared position resolver, not
# a consumer API.  Explicit imports by that resolver are intentionally private.
__all__: list[str] = []
