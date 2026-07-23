"""Shared fail-closed descriptors for persisted measurement and semantic arrays.

Coordinate descriptors describe geometry in a frame.  Scalar distances, areas,
speeds, angles, counts, fractions, validity flags, and other non-coordinate
values instead use this record.  The record owns the exact output payload and
binds its coordinate, measurement, row-identity, collection, calibration, and
derivation authorities without inventing an origin or X/Y axes for a scalar.

The wire schema predates this shared module in the chaser-distance publisher.
Its schema id and existing chaser records are intentionally preserved.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    require_bound_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_DIGEST_SUFFIX,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_identity import (
    BoundRowIdentityContract,
    require_bound_row_identity_contract,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
    verify_bound_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path


ARRAY_MEASUREMENT_DESCRIPTOR_ATTR = "measurement_descriptor"
ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_ID = "palette.array_measurement_descriptor"
ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_VERSION = 1
SCALAR_MEASUREMENT_OVERLAY_STATUS = "not_suitable_scalar_measurement"

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.:+-]*$")
_UNITS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/^*-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArrayMeasurementDescriptorError(ValueError):
    """Raised when one persisted measurement descriptor is unsafe."""


def _fail(message: str) -> None:
    raise ArrayMeasurementDescriptorError(message)


def _token(value: Any, *, label: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        _fail(f"{label} must be one exact controlled token.")
    return value


def _units(value: Any) -> str:
    if type(value) is not str or _UNITS_RE.fullmatch(value) is None:
        _fail("Measurement units must be one exact controlled unit symbol.")
    return value


def _record_pointer_shape(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {"record_ref", "record_sha256"}:
        _fail(f"{label} must be one exact digest-bound record pointer.")
    record_ref = value.get("record_ref")
    digest = value.get("record_sha256")
    if (
        type(record_ref) is not str
        or not record_ref.startswith("/")
        or "@" not in record_ref
        or type(digest) is not str
        or _SHA256_RE.fullmatch(digest) is None
    ):
        _fail(f"{label} has an invalid record reference or SHA-256 digest.")
    return {"record_ref": record_ref, "record_sha256": digest}


def _payload_shape(value: Any, *, label: str) -> dict[str, Any]:
    required = {"array_ref", "dtype", "shape", "content_sha256"}
    if type(value) is not dict or set(value) != required:
        _fail(f"{label} must be one exact array payload identity.")
    array_ref = value.get("array_ref")
    dtype_value = value.get("dtype")
    shape = value.get("shape")
    digest = value.get("content_sha256")
    if type(array_ref) is not str or not array_ref.startswith("/"):
        _fail(f"{label} array_ref must be one absolute archive path.")
    try:
        dtype = np.dtype(dtype_value)
    except (TypeError, ValueError) as exc:
        _fail(f"{label} dtype is invalid: {exc}.")
    if dtype.hasobject:
        _fail(f"{label} cannot describe an object array.")
    if (
        type(shape) is not list
        or any(type(item) is not int or item < 0 for item in shape)
        or type(digest) is not str
        or _SHA256_RE.fullmatch(digest) is None
    ):
        _fail(f"{label} has an invalid shape or SHA-256 digest.")
    return copy.deepcopy(dict(value))


def _axis_order(node: Any, axes: Sequence[str]) -> tuple[str, ...]:
    try:
        shape = tuple(int(value) for value in node.shape)
    except Exception as exc:
        _fail(f"Measurement array lacks one exact shape: {exc}.")
    axis_order = tuple(_token(value, label="Measurement axis") for value in axes)
    if len(axis_order) != len(shape) or len(set(axis_order)) != len(axis_order):
        _fail("Measurement axis order must be unique and match the exact array rank.")
    return axis_order


def array_measurement_payload(node: Any) -> dict[str, Any]:
    """Return the canonical payload identity for one persisted array."""

    try:
        dtype = np.dtype(node.dtype)
        shape = tuple(int(value) for value in node.shape)
        path = canonical_node_path(node)
    except Exception as exc:
        _fail(f"Measurement array metadata is unavailable: {exc}.")
    if dtype.hasobject or any(value < 0 for value in shape):
        _fail("Measurement arrays require a non-object dtype and nonnegative shape.")
    return {
        "array_ref": f"/{path}",
        "dtype": dtype.str,
        "shape": [int(value) for value in shape],
        "content_sha256": array_payload_sha256(node),
    }


def measurement_record_pointer(value: BoundCoordinateRecord) -> dict[str, str]:
    """Return a pointer only after freshly verifying its persisted record."""

    try:
        verified = verify_bound_coordinate_record(value)
    except Exception as exc:
        _fail(f"Measurement lineage record is stale or unverified: {exc}.")
    return {
        "record_ref": verified.record_ref,
        "record_sha256": verified.record_sha256,
    }


def coordinate_descriptor_pointer(
    value: BoundCanonicalCoordinateDescriptor,
) -> dict[str, str]:
    """Return a pointer only after freshly verifying its coordinate descriptor."""

    try:
        verified = require_bound_canonical_coordinate_descriptor(value)
        node = verified.coordinate_node
        digest = node.attrs.get(
            f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
        )
    except Exception as exc:
        _fail(f"Source coordinate descriptor is stale or unverified: {exc}.")
    if digest != verified.descriptor.digest():
        _fail("Source coordinate descriptor digest changed before measurement binding.")
    return {
        "record_ref": f"/{canonical_node_path(node)}@{COORDINATE_DESCRIPTOR_ATTR}",
        "record_sha256": str(digest),
    }


def _validate_collection_authority(
    collection: BoundCoordinateRecord,
    *,
    axis: int,
    role: str,
    cardinality: int,
) -> None:
    record = collection.record
    if (
        record.get("axis") != axis
        or record.get("role") != role
        or record.get("cardinality") != cardinality
    ):
        _fail(
            "Collection authority axis, role, or cardinality differs from the "
            "measurement collection axis."
        )


def build_array_measurement_descriptor(
    node: Any,
    *,
    quantity: str,
    units: str,
    operation: str,
    axes: Sequence[str],
    coordinate_inputs: Sequence[BoundCanonicalCoordinateDescriptor] = (),
    measurement_inputs: Sequence[BoundCoordinateRecord] = (),
    row_identity: BoundRowIdentityContract,
    collection: BoundCoordinateRecord,
    measurement_authority: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    row_axis_name: str | None = None,
    collection_axis_name: str | None = None,
    collection_axis_role: str | None = None,
    epoch_windows: BoundCoordinateRecord | None = None,
    validity_node: Any | None = None,
    validity_policy: str | None = None,
    selected_collection_members: Sequence[str] = (),
    semantic_kind: str | None = None,
) -> dict[str, Any]:
    """Build one canonical descriptor from freshly verified persisted evidence."""

    quantity = _token(quantity, label="Measurement quantity")
    units = _units(units)
    operation = _token(operation, label="Measurement operation")
    axis_order = _axis_order(node, axes)
    try:
        identity = require_bound_row_identity_contract(row_identity)
    except Exception as exc:
        _fail(f"Measurement row identity is stale or unverified: {exc}.")
    collection_pointer = measurement_record_pointer(collection)
    record: dict[str, Any] = {
        "schema_id": ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_ID,
        "schema_version": ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_VERSION,
        "quantity": quantity,
        "units": units,
        "operation": operation,
        "axis_order": list(axis_order),
        "output": array_measurement_payload(node),
        "source_coordinate_descriptors": [
            coordinate_descriptor_pointer(value) for value in coordinate_inputs
        ],
        "source_measurement_descriptors": [
            measurement_record_pointer(value) for value in measurement_inputs
        ],
        "source_row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "source_collection_authority": collection_pointer,
        "measurement_authority": measurement_record_pointer(measurement_authority),
        "derivation": measurement_record_pointer(derivation),
        "source_camera_overlay_status": SCALAR_MEASUREMENT_OVERLAY_STATUS,
    }
    if row_axis_name is not None:
        row_axis = _token(row_axis_name, label="Measurement row axis")
        if row_axis not in axis_order or axis_order.index(row_axis) != 0:
            _fail("Row-aligned measurements require their declared row axis at axis zero.")
        if int(node.shape[0]) != identity.leading_dimension:
            _fail("Measurement row axis differs from the exact row identity cardinality.")
        record["output_row_identity"] = copy.deepcopy(record["source_row_identity"])
    if collection_axis_name is not None:
        collection_axis = _token(
            collection_axis_name,
            label="Measurement collection axis",
        )
        role = _token(collection_axis_role, label="Measurement collection role")
        if collection_axis not in axis_order:
            _fail("Declared measurement collection axis is absent from axis_order.")
        axis = axis_order.index(collection_axis)
        cardinality = int(node.shape[axis])
        _validate_collection_authority(
            collection,
            axis=axis,
            role=role,
            cardinality=cardinality,
        )
        record["collection_axis"] = {
            "axis": axis,
            "role": role,
            "cardinality": cardinality,
            "label_authority": collection_pointer,
        }
    elif collection_axis_role is not None:
        _fail("Collection-axis role cannot be declared without an axis name.")
    if "stimulus_epoch_window" in axis_order:
        if epoch_windows is None or axis_order.index("stimulus_epoch_window") != 0:
            _fail("Epoch-window measurements require exact axis-zero identity.")
        record["output_epoch_window_identity"] = measurement_record_pointer(epoch_windows)
    elif epoch_windows is not None:
        record["aggregation_epoch_window_identity"] = measurement_record_pointer(
            epoch_windows
        )
    if validity_node is not None:
        if validity_policy is None:
            _fail("Measurement validity payload requires one exact validity policy.")
        record["validity"] = {
            "payload": array_measurement_payload(validity_node),
            "policy": _token(validity_policy, label="Measurement validity policy"),
        }
    elif validity_policy is not None:
        _fail("Measurement validity policy cannot exist without a validity array.")
    members = tuple(
        _token(value, label="Selected collection member")
        for value in selected_collection_members
    )
    if members:
        if len(set(members)) != len(members):
            _fail("Selected collection members must be unique and ordered.")
        record["selected_collection_members"] = list(members)
    if semantic_kind is not None:
        record["semantic_kind"] = _token(
            semantic_kind,
            label="Measurement semantic kind",
        )
    return record


def validate_array_measurement_descriptor(
    node: Any,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one stored record against its live output array."""

    if type(record) is not dict:
        _fail("Array measurement descriptor must be one exact JSON object.")
    required = {
        "schema_id",
        "schema_version",
        "quantity",
        "units",
        "operation",
        "axis_order",
        "output",
        "source_coordinate_descriptors",
        "source_measurement_descriptors",
        "source_row_identity",
        "source_collection_authority",
        "measurement_authority",
        "derivation",
        "source_camera_overlay_status",
    }
    optional = {
        "output_row_identity",
        "collection_axis",
        "output_epoch_window_identity",
        "aggregation_epoch_window_identity",
        "validity",
        "selected_collection_members",
        "semantic_kind",
    }
    if set(record) - required - optional or not required.issubset(record):
        _fail("Array measurement descriptor has missing or unsupported fields.")
    if (
        record.get("schema_id") != ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_ID
        or record.get("schema_version") != ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_VERSION
        or record.get("source_camera_overlay_status")
        != SCALAR_MEASUREMENT_OVERLAY_STATUS
    ):
        _fail("Array measurement descriptor schema or scalar overlay policy is invalid.")
    for name in ("quantity", "operation"):
        _token(record.get(name), label=f"Measurement {name}")
    _units(record.get("units"))
    axes = record.get("axis_order")
    if type(axes) is not list:
        _fail("Measurement axis_order must be one exact list.")
    _axis_order(node, axes)
    _payload_shape(record.get("output"), label="Measurement output")
    if record.get("output") != array_measurement_payload(node):
        _fail("Measurement descriptor output payload differs from the live array.")
    for name in ("source_coordinate_descriptors", "source_measurement_descriptors"):
        values = record.get(name)
        if type(values) is not list:
            _fail(f"Measurement descriptor {name} must be one exact list.")
        for index, value in enumerate(values):
            _record_pointer_shape(value, label=f"{name}[{index}]")
    for name in (
        "source_row_identity",
        "source_collection_authority",
        "measurement_authority",
        "derivation",
    ):
        _record_pointer_shape(record.get(name), label=name)
    if "output_row_identity" in record:
        output_identity = _record_pointer_shape(
            record["output_row_identity"],
            label="output_row_identity",
        )
        if output_identity != record["source_row_identity"]:
            _fail("Measurement output row identity differs from its source identity.")
        if not axes or axes[0] not in {"observation", "camera_frame"}:
            _fail("Measurement output row identity requires a controlled row axis at zero.")
    for name in (
        "output_epoch_window_identity",
        "aggregation_epoch_window_identity",
    ):
        if name in record:
            _record_pointer_shape(record[name], label=name)
    if "collection_axis" in record:
        value = record["collection_axis"]
        if type(value) is not dict or set(value) != {
            "axis",
            "role",
            "cardinality",
            "label_authority",
        }:
            _fail("Measurement collection_axis must be one exact object.")
        axis = value.get("axis")
        cardinality = value.get("cardinality")
        if (
            type(axis) is not int
            or axis < 0
            or axis >= len(axes)
            or type(cardinality) is not int
            or cardinality != int(node.shape[axis])
        ):
            _fail("Measurement collection axis or cardinality differs from the output.")
        _token(value.get("role"), label="Measurement collection role")
        label_authority = _record_pointer_shape(
            value.get("label_authority"),
            label="Measurement collection label authority",
        )
        if label_authority != record["source_collection_authority"]:
            _fail("Measurement collection label authority differs from its source authority.")
    if "validity" in record:
        validity = record["validity"]
        if type(validity) is not dict or set(validity) != {"payload", "policy"}:
            _fail("Measurement validity must be one exact payload/policy object.")
        _payload_shape(validity.get("payload"), label="Measurement validity payload")
        _token(validity.get("policy"), label="Measurement validity policy")
    if "selected_collection_members" in record:
        members = record["selected_collection_members"]
        if type(members) is not list:
            _fail("Measurement selected_collection_members must be one exact list.")
        controlled = tuple(
            _token(value, label="Selected collection member") for value in members
        )
        if len(set(controlled)) != len(controlled):
            _fail("Selected collection members must be unique and ordered.")
    if "semantic_kind" in record:
        _token(record["semantic_kind"], label="Measurement semantic kind")
    return copy.deepcopy(dict(record))


def stamp_and_bind_array_measurement_descriptor(
    node: Any,
    record: Mapping[str, Any],
    *,
    attr_name: str = ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
) -> BoundCoordinateRecord:
    """Validate, stamp, and bind one measurement descriptor transactionally."""

    validated = validate_array_measurement_descriptor(node, record)
    return stamp_and_bind_persisted_coordinate_record(
        node,
        validated,
        attr_name=attr_name,
    )


def load_bound_array_measurement_descriptor(
    node: Any,
    *,
    expected_record: Mapping[str, Any] | None = None,
    attr_name: str = ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
) -> BoundCoordinateRecord:
    """Load one descriptor, prove its output, and optionally compare live evidence."""

    bound = bind_persisted_coordinate_record(node, attr_name=attr_name)
    validated = validate_array_measurement_descriptor(node, bound.record)
    if expected_record is not None:
        expected = validate_array_measurement_descriptor(node, expected_record)
        if validated != expected:
            _fail("Persisted measurement descriptor is stale or incomplete.")
    return bound


__all__ = [
    "ARRAY_MEASUREMENT_DESCRIPTOR_ATTR",
    "ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_ID",
    "ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_VERSION",
    "ArrayMeasurementDescriptorError",
    "SCALAR_MEASUREMENT_OVERLAY_STATUS",
    "array_measurement_payload",
    "build_array_measurement_descriptor",
    "coordinate_descriptor_pointer",
    "load_bound_array_measurement_descriptor",
    "measurement_record_pointer",
    "stamp_and_bind_array_measurement_descriptor",
    "validate_array_measurement_descriptor",
]
