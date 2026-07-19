"""Canonical coordinate publication for exact track-position subsets.

Track kinematics changes row identity: an immediate source rowset is selected
and reordered into per-track rows identified by ``track_sample_key``.  This
module is the narrow publication boundary that proves the numerical subset,
persists that derivation, and reuses (rather than reconstructs) the exact frame
and direction-labelled transform evidence of the selected source surface.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    require_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_NOT_SUITABLE,
)
from fisheye.shared.coordinate_frame_record import (
    PHYSICAL_FRAME_CALIBRATION_KIND,
    BoundPhysicalFrameCalibration,
    array_payload_sha256,
    verify_bound_coordinate_frame,
)
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_DOMAIN,
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    identity_array_content_sha256,
    require_bound_row_identity_contract,
    require_bound_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path


TRACK_POSITION_DERIVATION_ATTR = "track_position_derivation"
TRACK_POSITION_DERIVATION_SCHEMA_ID = "palette.track_position_derivation"
TRACK_POSITION_DERIVATION_SCHEMA_VERSION = 1
TRACK_POSITION_DERIVATION_OPERATION = "exact_subset_reorder_v1"
SOURCE_CAMERA_PROFILE_ID = "source_camera_image_px.top_left_y_down.v1"
PHYSICAL_SOURCE_CAMERA_PROFILE_ID = "physical_mm.source_camera_y_down.v1"


class TrackCoordinatePublicationError(ValueError):
    """Raised when a track surface lacks exact coordinate authority."""


@dataclass(frozen=True)
class TrackPositionPublicationResult:
    """Sealed coordinate bindings and derivation for one track subgroup."""

    positions_px: BoundCanonicalCoordinateDescriptor
    positions_mm: BoundCanonicalCoordinateDescriptor | None
    derivation: BoundCoordinateRecord


def _fail(message: str) -> None:
    raise TrackCoordinatePublicationError(message)


def _same_row_identity(
    left: BoundRowIdentityContract,
    right: BoundRowIdentityContract,
) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.rowset_path == right.rowset_path
        and left.key_array_path == right.key_array_path
        and left.contract == right.contract
    )


def _same_pixel_frame(left: Any, right: Any) -> bool:
    return (
        type(left) is type(right)
        and left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.endpoint == right.endpoint
    )


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        values = np.array(node[:], copy=True, order="C")
    except Exception as exc:
        _fail(f"Unable to read exact {label} array: {exc}.")
    declared_shape = getattr(node, "shape", None)
    try:
        declared_dtype = np.dtype(getattr(node, "dtype"))
    except (AttributeError, TypeError) as exc:
        _fail(f"{label} has no exact NumPy dtype: {exc}.")
    if values.shape != declared_shape or values.dtype != declared_dtype:
        _fail(f"{label} values disagree with declared dtype/shape.")
    if values.dtype.hasobject:
        _fail(f"{label} cannot use object dtype.")
    return values


def _payload_record(node: Any, values: np.ndarray) -> dict[str, Any]:
    return {
        "array_ref": f"/{canonical_node_path(node)}",
        "dtype": values.dtype.str,
        "shape": [int(item) for item in values.shape],
        "content_sha256": array_payload_sha256(node),
    }


def _trusted_attrs(node: Any, *, label: str) -> Any:
    attrs = getattr(node, "attrs", None)
    trusted = type(attrs) is dict
    if not trusted:
        try:
            from zarr.core.attributes import Attributes as ZarrAttributes
        except ImportError:  # pragma: no cover - Palette depends on zarr
            ZarrAttributes = None  # type: ignore[assignment,misc]
        trusted = ZarrAttributes is not None and type(attrs) is ZarrAttributes
    if not trusted or not all(
        callable(getattr(attrs, name, None))
        for name in ("update", "__setitem__", "__delitem__")
    ):
        _fail(
            f"{label} attrs must be an exact built-in dict or exact Zarr "
            "Attributes transaction boundary; no publication was attempted."
        )
    return attrs


def _raw_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _raw_equal(left[name], right[name]) for name in left
        )
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(
            _raw_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result


def _restore_attrs(attrs: Any, snapshot: dict[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    attrs.update(copy.deepcopy(snapshot))
    if not _raw_equal(dict(attrs), snapshot):
        raise RuntimeError("restored attrs differ from the exact snapshot")


def _exact_values_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left, right, equal_nan=True)
    )


def _track_lineage_source_matches(
    output_identity: BoundRowIdentityContract,
    source_temporal: BoundSourceRowTemporalAuthority,
    source_row_index_node: Any,
) -> None:
    if output_identity.contract.domain != TRACK_SAMPLE_DOMAIN:
        _fail("Track positions require track_sample row identity.")
    lineage = output_identity.track_time_lineage
    if lineage is None:
        _fail("Track positions require sealed track-sample time lineage.")
    lineage.assert_verified()
    expected = lineage.record.source_row_temporal_authority
    if (
        expected.record_ref != source_temporal.record_ref
        or expected.record_sha256 != source_temporal.record_sha256
    ):
        _fail(
            "Track row identity does not bind the selected immediate-source "
            "temporal authority."
        )
    expected_source_rows = lineage.record.source_row_index
    if expected_source_rows.ref != f"/{canonical_node_path(source_row_index_node)}":
        _fail("source_row_index node differs from track time-lineage authority.")
    source_rows = _array(source_row_index_node, label="source_row_index")
    if (
        source_rows.dtype != np.dtype("<i8")
        or source_rows.ndim != 1
        or expected_source_rows.content_sha256
        != identity_array_content_sha256(source_rows)
    ):
        _fail("source_row_index payload differs from track time-lineage authority.")


def _physical_frame(
    value: BoundPhysicalFrameCalibration | None,
    source: BoundCanonicalCoordinateDescriptor,
) -> BoundPhysicalFrameCalibration | None:
    if value is None:
        return None
    verified = verify_bound_coordinate_frame(
        value,
        expected_kind=PHYSICAL_FRAME_CALIBRATION_KIND,
    )
    if type(verified) is not BoundPhysicalFrameCalibration:
        _fail("Physical position authority is not a physical calibration frame.")
    if source.descriptor.profile_id != SOURCE_CAMERA_PROFILE_ID:
        _fail(
            "Canonical source-camera physical positions can be derived only from "
            "the exact source-camera pixel profile."
        )
    if source.reference_frame_authority is None or not _same_pixel_frame(
        verified.source_camera_pixels,
        source.reference_frame_authority,
    ):
        _fail(
            "Physical calibration source-camera frame differs from positions_px "
            "frame authority."
        )
    if verified.archive_identity != archive_identity(source.coordinate_node):
        _fail("Physical calibration and source coordinates use different archives.")
    return verified


def _derivation_record(
    *,
    source: BoundCanonicalCoordinateDescriptor,
    source_temporal: BoundSourceRowTemporalAuthority,
    source_values: np.ndarray,
    source_row_index_node: Any,
    source_rows: np.ndarray,
    output_identity: BoundRowIdentityContract,
    positions_px_node: Any,
    output_px: np.ndarray,
    positions_mm_node: Any | None,
    output_mm: np.ndarray | None,
    physical: BoundPhysicalFrameCalibration | None,
) -> dict[str, Any]:
    descriptor_ref = (
        f"/{canonical_node_path(source.coordinate_node)}@coordinate_descriptor"
    )
    return {
        "schema_id": TRACK_POSITION_DERIVATION_SCHEMA_ID,
        "schema_version": TRACK_POSITION_DERIVATION_SCHEMA_VERSION,
        "operation": TRACK_POSITION_DERIVATION_OPERATION,
        "source_coordinate": {
            **_payload_record(source.coordinate_node, source_values),
            "descriptor_ref": descriptor_ref,
            "descriptor_sha256": source.descriptor.digest(),
            "row_identity_ref": source.row_identity.record_ref,
            "row_identity_sha256": source.row_identity.record_sha256,
        },
        "source_temporal_authority": {
            "record_ref": source_temporal.record_ref,
            "record_sha256": source_temporal.record_sha256,
        },
        "selection": _payload_record(source_row_index_node, source_rows),
        "output_row_identity": {
            "record_ref": output_identity.record_ref,
            "record_sha256": output_identity.record_sha256,
        },
        "positions_px": _payload_record(positions_px_node, output_px),
        "physical_derivation": (
            {
                "profile_id": PHYSICAL_SOURCE_CAMERA_PROFILE_ID,
                "physical_frame_ref": physical.record_ref,
                "physical_frame_sha256": physical.record_sha256,
                "scale_quantity": "mm_per_pixel",
                "scale_value": float(physical.record.mm_per_pixel),
                "formula": "positions_px_times_mm_per_pixel_v1",
                "positions_mm": _payload_record(positions_mm_node, output_mm),
            }
            if physical is not None
            and positions_mm_node is not None
            and output_mm is not None
            else None
        ),
    }


def _stamp_publication_transaction(
    *,
    track_group: Any,
    positions_px_node: Any,
    source_row_index_node: Any,
    output_identity: BoundRowIdentityContract,
    source: BoundCanonicalCoordinateDescriptor,
    source_temporal: BoundSourceRowTemporalAuthority,
    positions_mm_node: Any | None,
    physical: BoundPhysicalFrameCalibration | None,
    derivation_record: dict[str, Any],
) -> TrackPositionPublicationResult:
    derivation = stamp_and_bind_persisted_coordinate_record(
        track_group,
        derivation_record,
        attr_name=TRACK_POSITION_DERIVATION_ATTR,
    )
    if (
        derivation.record["source_coordinate"]["content_sha256"]
        != array_payload_sha256(source.coordinate_node)
        or derivation.record["selection"]["content_sha256"]
        != array_payload_sha256(source_row_index_node)
        or derivation.record["positions_px"]["content_sha256"]
        != array_payload_sha256(positions_px_node)
    ):
        _fail("Track position arrays changed while derivation authority was published.")
    if positions_mm_node is not None and physical is not None:
        if derivation.record["physical_derivation"]["positions_mm"][
            "content_sha256"
        ] != array_payload_sha256(positions_mm_node):
            _fail("positions_mm changed while derivation authority was published.")

    source_descriptor = source.descriptor
    lineage = (*source.lineage_records, derivation)
    px_binding = build_bound_canonical_coordinate_descriptor(
        positions_px_node,
        profile_id=source_descriptor.profile_id,
        geometry_type=source_descriptor.geometry_type,
        components=source_descriptor.components,
        component_units=source_descriptor.component_units,
        pixel_convention=source_descriptor.pixel_convention,
        row_identity=output_identity,
        reference_extent=source.reference_extent,
        reference_frame_authority=source.reference_frame_authority,
        source_camera_overlay_status=source_descriptor.source_camera_overlay.status,
        transform_chain=source.transform_chain,
        lineage_records=lineage,
        frame_record=source.frame_record,
    )
    bindings = [px_binding]
    if positions_mm_node is not None and physical is not None:
        bindings.append(
            build_bound_canonical_coordinate_descriptor(
                positions_mm_node,
                profile_id=PHYSICAL_SOURCE_CAMERA_PROFILE_ID,
                geometry_type=source_descriptor.geometry_type,
                components=source_descriptor.components,
                component_units=("mm", "mm"),
                pixel_convention="not_applicable",
                row_identity=output_identity,
                source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
                lineage_records=(derivation,),
                frame_record=physical,
            )
        )
    stamp_bound_canonical_coordinate_descriptors(bindings)
    return load_track_position_coordinates(
        track_group,
        positions_px_node,
        source_row_index_node,
        track_row_identity=output_identity,
        source_positions=source,
        source_temporal_authority=source_temporal,
        positions_mm_node=positions_mm_node,
        physical_frame=physical,
    )


def publish_track_position_coordinates(
    track_group: Any,
    positions_px_node: Any,
    source_row_index_node: Any,
    *,
    track_row_identity: BoundRowIdentityContract,
    source_positions: BoundCanonicalCoordinateDescriptor,
    source_temporal_authority: BoundSourceRowTemporalAuthority,
    positions_mm_node: Any | None = None,
    physical_frame: BoundPhysicalFrameCalibration | None = None,
) -> TrackPositionPublicationResult:
    """Publish canonical per-track position descriptors from exact evidence.

    Version 1 is deliberately limited to a numerically exact subset/reorder.
    It cannot authorize track-stage interpolation, inferred dimensions, parsed
    descriptor dictionaries, scalar texture/camera ratios, or an unframed
    ``positions_mm`` array.
    """

    source = require_bound_canonical_coordinate_descriptor(source_positions)
    source_temporal = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    output_identity = require_bound_row_identity_contract(track_row_identity)
    if not _same_row_identity(
        source.row_identity,
        source_temporal.source_row_identity,
    ):
        _fail(
            "Selected positions and source temporal authority do not bind the "
            "same exact source row identity."
        )
    common_archive = archive_identity(source.coordinate_node)
    for label, node in (
        ("track_group", track_group),
        ("positions_px", positions_px_node),
        ("source_row_index", source_row_index_node),
    ):
        if archive_identity(node) != common_archive:
            _fail(f"{label} and selected source coordinates use different archives.")
    if output_identity.archive_identity != common_archive:
        _fail("Track row identity and source coordinates use different archives.")
    _track_lineage_source_matches(
        output_identity,
        source_temporal,
        source_row_index_node,
    )

    source_values = _array(source.coordinate_node, label="source positions")
    output_px = _array(positions_px_node, label="track positions_px")
    source_rows = _array(source_row_index_node, label="source_row_index")
    if (
        source_values.ndim != 2
        or source_values.shape[1] != 2
        or output_px.ndim != 2
        or output_px.shape[1] != 2
        or source_values.shape[0] != source.row_identity.leading_dimension
        or output_px.shape[0] != output_identity.leading_dimension
        or source_rows.shape != (output_identity.leading_dimension,)
    ):
        _fail("Track/source position arrays do not have exact row-aligned (N, 2) shape.")
    if np.any(source_rows < 0) or np.any(source_rows >= source_values.shape[0]):
        _fail("source_row_index resolves outside the exact selected source surface.")
    selected = np.asarray(source_values[source_rows])
    if not _exact_values_equal(output_px, selected):
        _fail(
            "positions_px is not an exact dtype-preserving subset/reorder of the "
            "selected source coordinate surface."
        )

    physical = _physical_frame(physical_frame, source)
    if (positions_mm_node is None) != (physical is None):
        _fail(
            "positions_mm must be present exactly when a sealed compatible physical "
            "frame is supplied."
        )
    output_mm: np.ndarray | None = None
    if positions_mm_node is not None:
        if archive_identity(positions_mm_node) != common_archive:
            _fail("positions_mm and source coordinates use different archives.")
        output_mm = _array(positions_mm_node, label="track positions_mm")
        assert physical is not None
        if output_mm.shape != output_px.shape or output_mm.dtype != output_px.dtype:
            _fail("positions_mm must match positions_px dtype and shape.")
        scale = np.asarray(physical.record.mm_per_pixel, dtype=output_px.dtype)
        expected_mm = np.asarray(output_px * scale, dtype=output_px.dtype)
        if not _exact_values_equal(output_mm, expected_mm):
            _fail(
                "positions_mm does not equal positions_px multiplied by the exact "
                "typed source-camera mm_per_pixel calibration."
            )

    derivation_record = _derivation_record(
        source=source,
        source_temporal=source_temporal,
        source_values=source_values,
        source_row_index_node=source_row_index_node,
        source_rows=source_rows,
        output_identity=output_identity,
        positions_px_node=positions_px_node,
        output_px=output_px,
        positions_mm_node=positions_mm_node,
        output_mm=output_mm,
        physical=physical,
    )
    attrs_targets = [
        _trusted_attrs(track_group, label="track_group"),
        _trusted_attrs(positions_px_node, label="positions_px"),
    ]
    if positions_mm_node is not None:
        attrs_targets.append(_trusted_attrs(positions_mm_node, label="positions_mm"))
    snapshots = [copy.deepcopy(dict(attrs)) for attrs in attrs_targets]
    try:
        return _stamp_publication_transaction(
            track_group=track_group,
            positions_px_node=positions_px_node,
            source_row_index_node=source_row_index_node,
            output_identity=output_identity,
            source=source,
            source_temporal=source_temporal,
            positions_mm_node=positions_mm_node,
            physical=physical,
            derivation_record=derivation_record,
        )
    except Exception as exc:
        failures: list[str] = []
        for attrs, snapshot in zip(attrs_targets, snapshots, strict=True):
            try:
                _restore_attrs(attrs, snapshot)
            except Exception as rollback_exc:  # pragma: no cover - hostile store
                failures.append(str(rollback_exc))
        if failures:
            raise TrackCoordinatePublicationError(
                "Track coordinate publication failed and exact attrs rollback "
                f"was incomplete: {failures!r}."
            ) from exc
        raise


def load_track_position_coordinates(
    track_group: Any,
    positions_px_node: Any,
    source_row_index_node: Any,
    *,
    track_row_identity: BoundRowIdentityContract,
    source_positions: BoundCanonicalCoordinateDescriptor,
    source_temporal_authority: BoundSourceRowTemporalAuthority,
    positions_mm_node: Any | None = None,
    physical_frame: BoundPhysicalFrameCalibration | None = None,
) -> TrackPositionPublicationResult:
    """Freshly revalidate one persisted exact track-position publication."""

    source = require_bound_canonical_coordinate_descriptor(source_positions)
    source_temporal = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    output_identity = require_bound_row_identity_contract(track_row_identity)
    if not _same_row_identity(
        source.row_identity,
        source_temporal.source_row_identity,
    ):
        _fail(
            "Selected positions and source temporal authority do not bind the "
            "same exact source row identity."
        )
    common_archive = archive_identity(source.coordinate_node)
    for label, node in (
        ("track_group", track_group),
        ("positions_px", positions_px_node),
        ("source_row_index", source_row_index_node),
    ):
        if archive_identity(node) != common_archive:
            _fail(f"{label} and selected source coordinates use different archives.")
    if output_identity.archive_identity != common_archive:
        _fail("Track row identity and source coordinates use different archives.")
    _track_lineage_source_matches(
        output_identity,
        source_temporal,
        source_row_index_node,
    )

    source_values = _array(source.coordinate_node, label="source positions")
    output_px = _array(positions_px_node, label="track positions_px")
    source_rows = _array(source_row_index_node, label="source_row_index")
    if (
        source_values.ndim != 2
        or source_values.shape[1] != 2
        or output_px.shape != (output_identity.leading_dimension, 2)
        or source_rows.shape != (output_identity.leading_dimension,)
        or np.any(source_rows < 0)
        or np.any(source_rows >= source_values.shape[0])
        or not _exact_values_equal(output_px, source_values[source_rows])
    ):
        _fail(
            "Persisted positions_px is not the exact declared source subset/reorder."
        )

    physical = _physical_frame(physical_frame, source)
    if (positions_mm_node is None) != (physical is None):
        _fail(
            "positions_mm must be present exactly when a sealed compatible physical "
            "frame is supplied."
        )
    output_mm = None
    if positions_mm_node is not None and physical is not None:
        if archive_identity(positions_mm_node) != common_archive:
            _fail("positions_mm and source coordinates use different archives.")
        output_mm = _array(positions_mm_node, label="track positions_mm")
        scale = np.asarray(physical.record.mm_per_pixel, dtype=output_px.dtype)
        expected_mm = np.asarray(output_px * scale, dtype=output_px.dtype)
        if not _exact_values_equal(output_mm, expected_mm):
            _fail(
                "Persisted positions_mm does not equal the exact typed physical "
                "calibration derivation."
            )

    derivation = bind_persisted_coordinate_record(
        track_group,
        attr_name=TRACK_POSITION_DERIVATION_ATTR,
    )
    expected_record = _derivation_record(
        source=source,
        source_temporal=source_temporal,
        source_values=source_values,
        source_row_index_node=source_row_index_node,
        source_rows=source_rows,
        output_identity=output_identity,
        positions_px_node=positions_px_node,
        output_px=output_px,
        positions_mm_node=positions_mm_node,
        output_mm=output_mm,
        physical=physical,
    )
    if derivation.record != expected_record:
        _fail(
            "Persisted track position derivation differs from exact live source, "
            "selection, output, identity, or calibration evidence."
        )

    source_descriptor = source.descriptor
    lineage = (*source.lineage_records, derivation)
    loaded_px = load_bound_canonical_coordinate_descriptor(
        positions_px_node,
        row_identity=output_identity,
        reference_extent=source.reference_extent,
        reference_frame_authority=source.reference_frame_authority,
        transform_chain=source.transform_chain,
        lineage_records=lineage,
        frame_record=source.frame_record,
    )
    copied_semantics = (
        loaded_px.descriptor.profile_id,
        loaded_px.descriptor.space_id,
        loaded_px.descriptor.geometry_type,
        loaded_px.descriptor.components,
        loaded_px.descriptor.component_units,
        loaded_px.descriptor.origin,
        loaded_px.descriptor.positive_directions,
        loaded_px.descriptor.reference_extent,
        loaded_px.descriptor.pixel_convention,
        loaded_px.descriptor.source_camera_overlay,
        loaded_px.descriptor.frame_record,
    )
    source_semantics = (
        source_descriptor.profile_id,
        source_descriptor.space_id,
        source_descriptor.geometry_type,
        source_descriptor.components,
        source_descriptor.component_units,
        source_descriptor.origin,
        source_descriptor.positive_directions,
        source_descriptor.reference_extent,
        source_descriptor.pixel_convention,
        source_descriptor.source_camera_overlay,
        source_descriptor.frame_record,
    )
    if copied_semantics != source_semantics:
        _fail("Track positions_px semantics differ from the exact selected source.")

    loaded_mm = None
    if positions_mm_node is not None and physical is not None:
        loaded_mm = load_bound_canonical_coordinate_descriptor(
            positions_mm_node,
            row_identity=output_identity,
            lineage_records=(derivation,),
            frame_record=physical,
        )
        if loaded_mm.descriptor.profile_id != PHYSICAL_SOURCE_CAMERA_PROFILE_ID:
            _fail("Track positions_mm uses an unsupported physical frame profile.")
    return TrackPositionPublicationResult(
        positions_px=loaded_px,
        positions_mm=loaded_mm,
        derivation=derivation,
    )


__all__ = [
    "TRACK_POSITION_DERIVATION_ATTR",
    "TRACK_POSITION_DERIVATION_OPERATION",
    "TRACK_POSITION_DERIVATION_SCHEMA_ID",
    "TRACK_POSITION_DERIVATION_SCHEMA_VERSION",
    "TrackCoordinatePublicationError",
    "TrackPositionPublicationResult",
    "load_track_position_coordinates",
    "publish_track_position_coordinates",
]
