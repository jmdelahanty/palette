"""Strict coordinate publication for future tail-derived analysis runs.

Tail algorithms consume an already canonical subject-shape publication.  This
module makes the downstream boundary equally explicit: exact observation
identity and acquisition-frame identity are copied to direct rowset children,
source-camera point arrays receive canonical descriptors, and one closed
manifest binds every scientific array, group attribute, array attribute, and
the exact authoritative subject-shape record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.array_measurement_descriptor import (
    ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
    build_array_measurement_descriptor,
    coordinate_descriptor_pointer,
    load_bound_array_measurement_descriptor,
    stamp_and_bind_array_measurement_descriptor,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    COORDINATE_DESCRIPTOR_ATTR,
    CanonicalCollectionAxis,
    DigestBoundCoordinateRecordRef,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    BoundRowIdentityContract,
    build_row_identity_contract,
    load_bound_row_identity_contract,
    stamp_and_bind_row_identity_contract,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.selector_activation import (
    DeferredSelectorActivation,
    SelectorActivationError,
    activate_selector_eligible_run,
    commit_deferred_selector_activation,
    rollback_deferred_selector_activation,
    write_activation_attr,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    BoundSubjectShapeCoordinatePublication,
    SubjectShapeCoordinatePublicationError,
    load_persisted_subject_shape_coordinate_publication,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


TAIL_COORDINATE_CONTRACT = "canonical_v2"
TAIL_PUBLICATION_MANIFEST_ATTR = "tail_coordinate_publication_manifest"
TAIL_PUBLICATION_MANIFEST_DIGEST_ATTR = f"{TAIL_PUBLICATION_MANIFEST_ATTR}_sha256"
TAIL_PUBLICATION_MANIFEST_ALIAS_ATTR = "tail_coordinate_publication_manifest_sha256"
TAIL_SOURCE_AUTHORITY_ATTR = "tail_source_subject_shape_authority"
TAIL_DERIVATION_ATTR = "tail_coordinate_derivation"
TAIL_COLLECTION_AXIS_ATTR = "tail_point_collection_axis"
TAIL_MEASUREMENT_COLLECTION_AXIS_ATTR = "tail_measurement_collection_axis"
TAIL_MEASUREMENT_AUTHORITY_ATTR = "tail_measurement_authority"
TAIL_PUBLICATION_OWNER_ATTR = "tail_publication_owner_uuid"
TAIL_PARENT_PUBLICATION_LEASE_ATTR = "tail_publication_lease"
TAIL_PUBLICATION_GENERATION_ATTR = "publication_generation"
TAIL_PUBLICATION_POLICY_ATTR = "publication_policy"
TAIL_PUBLICATION_POLICY = "owner_generation_guarded_selectors_then_eligibility_v1"

_SCHEMA_VERSION = 1
_KINEMATICS_KIND = "tail_kinematics"
_POSTURE_KIND = "tail_posture_view"
_KINEMATICS_PREFIX = "analysis/tail_kinematics_runs/"
_POSTURE_PREFIX = "analysis/tail_posture_view_runs/"
_KINEMATICS_GEOMETRY = ("tail_angle_sample_xy",)
_POSTURE_GEOMETRY = ("head_xy", "tail_keypoints_xy")

_KINEMATICS_ARRAYS = frozenset(
    {
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "valid",
        "failure_reason_bytes",
        "tail_angle_sample_s",
        "tail_angle_sample_xy",
        "tail_angle_rad",
        "tail_angle_deg",
        "tail_tip_angle_rad",
        "tail_tip_angle_deg",
        "tail_lateral_deflection_px",
        "tail_tip_lateral_deflection_px",
        "max_abs_tail_angle_rad",
        "max_abs_tail_angle_deg",
        "tail_angle_rms_rad",
        "tail_angle_rms_deg",
        "integrated_abs_tail_angle_rad",
        "tail_curvature_px_inv",
        "max_abs_tail_curvature_px_inv",
        "integrated_abs_tail_curvature",
    }
)
_POSTURE_ARRAYS = frozenset(
    {
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "valid",
        "failure_reason_bytes",
        "head_xy",
        "head_yaw_rad",
        "tail_keypoints_xy",
        "tail_angle_rad",
        "tail_angle_deg",
    }
)
_OPTIONAL_REVISION_ARRAYS = frozenset(
    {
        "source_refined_subject_masks/row_revision",
        "source_refined_subject_masks/row_revision_available",
    }
)
_RECORD_GROUPS = frozenset(
    {
        "coordinate_records",
        "coordinate_records/source_subject_shape",
        "coordinate_records/point_collection_axis",
        "coordinate_records/measurement_collection_axis",
        "coordinate_records/derivation",
        "coordinate_records/measurement_authority",
    }
)

# These attrs are lifecycle/publisher state rather than scientific authority.
# They may be written after the scientific manifest is sealed.  Every other
# run attr is part of the exact manifest snapshot; a new unsealed attr fails.
_OPERATIONAL_RUN_ATTRS = frozenset(
    {
        RUN_COMPLETION_STATUS_ATTR,
        "palette_run_completion_contract",
        "palette_run_started_at_utc",
        "palette_run_completed_at_utc",
        "palette_run_failed_at_utc",
        "palette_run_error",
        "palette_run_name",
        "palette_run_stage",
        "run_provenance",
        "cli_run_provenance",
        "run_provenance_enforcement_bypass",
        "stage_selector_eligible",
        "cluster_output_staging",
        TAIL_PUBLICATION_MANIFEST_ATTR,
        TAIL_PUBLICATION_MANIFEST_DIGEST_ATTR,
        TAIL_PUBLICATION_MANIFEST_ALIAS_ATTR,
    }
)


class TailCoordinatePublicationError(ValueError):
    """Raised when a tail coordinate publication is incomplete or stale."""


def _fail(message: str) -> None:
    raise TailCoordinatePublicationError(message)


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            json_attr_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _publication_owner_uuid(run: Any) -> str:
    value = run.attrs.get(TAIL_PUBLICATION_OWNER_ATTR)
    try:
        normalized = str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        _fail(f"Tail publication owner UUID is invalid: {exc}.")
    if value != normalized:
        _fail("Tail publication owner UUID must use canonical string form.")
    return normalized


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_copy(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pointer(record: BoundCoordinateRecord) -> dict[str, str]:
    return {
        "record_ref": record.record_ref,
        "record_sha256": record.record_sha256,
    }


def _array_record(relative_ref: str, node: Any) -> dict[str, Any]:
    return {
        "relative_ref": str(relative_ref),
        "dtype": np.dtype(node.dtype).str,
        "shape": [int(value) for value in node.shape],
        "content_sha256": array_payload_sha256(node),
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
    }


def _array_paths(group: Any, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for name in sorted(str(value) for value in group.array_keys()):
        paths.append(f"{prefix}/{name}".strip("/"))
    for name in sorted(str(value) for value in group.group_keys()):
        child_prefix = f"{prefix}/{name}".strip("/")
        paths.extend(_array_paths(group[name], child_prefix))
    return tuple(paths)


def _group_paths(group: Any, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for name in sorted(str(value) for value in group.group_keys()):
        child_prefix = f"{prefix}/{name}".strip("/")
        paths.append(child_prefix)
        paths.extend(_group_paths(group[name], child_prefix))
    return tuple(paths)


def _node_attrs_record(node: Any, *, excluded: Sequence[str] = ()) -> dict[str, Any]:
    excluded_names = frozenset(str(value) for value in excluded)
    payload = {
        str(name): _json_copy(value)
        for name, value in node.attrs.items()
        if str(name) not in excluded_names
    }
    return {
        "names": sorted(payload),
        "values": payload,
        "content_sha256": _sha256(payload),
        "canonicalization": "finite_canonical_json_sorted_attribute_mapping_v1",
    }


def _expected_prefix(kind: str) -> str:
    if kind == _KINEMATICS_KIND:
        return _KINEMATICS_PREFIX
    if kind == _POSTURE_KIND:
        return _POSTURE_PREFIX
    _fail(f"Unsupported tail publication kind {kind!r}.")
    raise AssertionError


def _expected_arrays(run: Any, kind: str) -> tuple[str, ...]:
    expected = set(_KINEMATICS_ARRAYS if kind == _KINEMATICS_KIND else _POSTURE_ARRAYS)
    revision = run.get("source_refined_subject_masks")
    if revision is not None:
        present = {
            path for path in _OPTIONAL_REVISION_ARRAYS if run.get(path) is not None
        }
        if present != set(_OPTIONAL_REVISION_ARRAYS):
            _fail(
                "source_refined_subject_masks must contain the complete controlled "
                "revision pair or be absent."
            )
        expected.update(_OPTIONAL_REVISION_ARRAYS)
    return tuple(sorted(expected))


def _expected_groups(run: Any) -> tuple[str, ...]:
    expected = set(_RECORD_GROUPS)
    if run.get("source_refined_subject_masks") is not None:
        expected.add("source_refined_subject_masks")
    return tuple(sorted(expected))


def _validate_closed_schema(run: Any, kind: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_arrays = _expected_arrays(run, kind)
    observed_arrays = _array_paths(run)
    if observed_arrays != expected_arrays:
        _fail(
            "Tail array inventory differs from the controlled schema: "
            f"expected={expected_arrays!r}, observed={observed_arrays!r}."
        )
    expected_groups = _expected_groups(run)
    observed_groups = _group_paths(run)
    if observed_groups != expected_groups:
        _fail(
            "Tail group inventory differs from the controlled schema: "
            f"expected={expected_groups!r}, observed={observed_groups!r}."
        )
    return expected_arrays, expected_groups


def _source_array(source_run: Any, name: str, *, row_count: int) -> Any:
    node = source_run.get(name)
    if node is None:
        _fail(f"Canonical subject-shape source lacks required direct array {name!r}.")
    shape = tuple(int(value) for value in node.shape)
    if shape != (int(row_count),):
        _fail(f"Canonical subject-shape {name!r} is not aligned with the exact rowset.")
    return node


def _validate_exact_identity_copies(
    run: Any,
    source: BoundSubjectShapeCoordinatePublication,
) -> dict[str, Any]:
    source_run = source._run
    row_count = int(source.row_identity.leading_dimension)
    result: dict[str, Any] = {}
    for name in (
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
    ):
        source_node = _source_array(source_run, name, row_count=row_count)
        target_node = run.get(name)
        if target_node is None or tuple(int(value) for value in target_node.shape) != (
            row_count,
        ):
            _fail(f"Tail publication lacks direct row-aligned {name!r}.")
        source_values = np.asarray(source_node[:])
        target_values = np.asarray(target_node[:])
        if source_values.dtype != target_values.dtype or not np.array_equal(
            source_values,
            target_values,
        ):
            _fail(f"Tail {name!r} is not an exact dtype-preserving source copy.")
        result[name] = _array_record(name, target_node)
    if np.dtype(run["instance_key"].dtype) != np.dtype("uint64"):
        _fail("Canonical tail instance_key must use exact uint64 dtype.")
    return result


def _source_authority_record(
    run: Any,
    source: BoundSubjectShapeCoordinatePublication,
) -> dict[str, Any]:
    expected_path = str(run.attrs.get("source_subject_shape_path") or "").strip("/")
    if expected_path != source.run_path:
        _fail(
            "Tail source_subject_shape_path differs from the exact loaded publication."
        )
    manifest_attr = run.attrs.get("source_subject_shape_publication_manifest_sha256")
    if manifest_attr != source.manifest.record_sha256:
        _fail(
            "Tail source subject-shape manifest attr differs from the exact publication."
        )
    identity_arrays = _validate_exact_identity_copies(run, source)
    return {
        "schema_id": "palette.tail_source_subject_shape_authority",
        "schema_version": _SCHEMA_VERSION,
        "tail_run_ref": f"/{canonical_node_path(run)}",
        "source_subject_shape_run_ref": f"/{source.run_path}",
        "source_publication_manifest": _pointer(source.manifest),
        "source_row_identity": {
            "record_ref": source.row_identity.record_ref,
            "record_sha256": source.row_identity.record_sha256,
        },
        "source_temporal_authority": {
            "record_ref": source.temporal_authority.record_ref,
            "record_sha256": source.temporal_authority.record_sha256,
        },
        "source_scientific_configuration": _pointer(source.scientific_configuration),
        "source_tail_sample_axis": _pointer(source.tail_sample_axis),
        "source_derivation": _pointer(source.derivation),
        "source_body_frame": {
            "record_ref": source.body_frame.record_ref,
            "record_sha256": source.body_frame.record_sha256,
        },
        "identity_arrays": identity_arrays,
        "exact_source_required": True,
    }


def _stamp_or_load_source_authority(
    run: Any,
    source: BoundSubjectShapeCoordinatePublication,
    *,
    load: bool,
) -> BoundCoordinateRecord:
    node = run["coordinate_records/source_subject_shape"]
    expected = _source_authority_record(run, source)
    record = (
        bind_persisted_coordinate_record(node, attr_name=TAIL_SOURCE_AUTHORITY_ATTR)
        if load
        else stamp_and_bind_persisted_coordinate_record(
            node,
            expected,
            attr_name=TAIL_SOURCE_AUTHORITY_ATTR,
        )
    )
    if record.record != expected:
        _fail("Tail source authority differs from the exact subject-shape publication.")
    return record


def _collection_axis_record(run: Any, kind: str) -> dict[str, Any]:
    if kind == _KINEMATICS_KIND:
        values = np.asarray(run["tail_angle_sample_s"][:])
        if (
            values.dtype.kind != "f"
            or values.ndim != 1
            or values.size < 2
            or not np.isfinite(values).all()
            or float(values[0]) != 0.0
            or float(values[-1]) != 1.0
            or not np.all(np.diff(values.astype(np.float64)) > 0.0)
        ):
            _fail("tail_angle_sample_s must be a finite increasing closed [0,1] axis.")
        coordinate = _array_record("tail_angle_sample_s", run["tail_angle_sample_s"])
        count = int(values.size)
        if type(run.attrs.get("tail_angle_sample_count")) is not int or int(
            run.attrs["tail_angle_sample_count"]
        ) != count:
            _fail(
                "tail_angle_sample_count must equal the exact persisted collection "
                "axis cardinality."
            )
        labels = [f"tail_angle_sample_{index:03d}" for index in range(count)]
        bound_surfaces = ["tail_angle_sample_xy"]
    else:
        node = run["tail_keypoints_xy"]
        shape = tuple(int(value) for value in node.shape)
        if len(shape) != 3 or shape[2] != 2 or shape[1] < 2:
            _fail("tail_keypoints_xy must have shape (rows, keypoints>=2, 2).")
        count = shape[1]
        if type(run.attrs.get("keypoint_count")) is not int or int(
            run.attrs["keypoint_count"]
        ) != count:
            _fail(
                "keypoint_count must equal the exact persisted collection-axis "
                "cardinality."
            )
        labels = [f"tail_keypoint_{index:03d}" for index in range(count)]
        coordinate = {
            "kind": "implicit_evenly_spaced_normalized_arclength",
            "domain": "closed_0_to_1",
            "cardinality": count,
        }
        bound_surfaces = ["tail_keypoints_xy"]
    return {
        "schema_id": "palette.tail_point_collection_axis",
        "schema_version": _SCHEMA_VERSION,
        "axis": 1,
        "role": "keypoint",
        "cardinality": count,
        "labels": labels,
        "sample_direction": "tail_base_to_tail_tip",
        "coordinate": coordinate,
        "bound_surfaces": bound_surfaces,
    }


def _stamp_or_load_collection_axis(run: Any, kind: str, *, load: bool) -> BoundCoordinateRecord:
    node = run["coordinate_records/point_collection_axis"]
    expected = _collection_axis_record(run, kind)
    record = (
        bind_persisted_coordinate_record(node, attr_name=TAIL_COLLECTION_AXIS_ATTR)
        if load
        else stamp_and_bind_persisted_coordinate_record(
            node,
            expected,
            attr_name=TAIL_COLLECTION_AXIS_ATTR,
        )
    )
    if record.record != expected:
        _fail("Tail point collection axis differs from its exact live surface.")
    return record


def _measurement_collection_axis_record(
    run: Any,
    kind: str,
    point_collection_axis: BoundCoordinateRecord,
) -> dict[str, Any]:
    point_record = point_collection_axis.record
    if kind == _KINEMATICS_KIND:
        count = int(point_record["cardinality"])
        role = "keypoint"
        labels = list(point_record["labels"])
        membership = "one_measurement_sample_per_tail_point_sample_v1"
        bound_surfaces = [
            "tail_angle_rad",
            "tail_angle_deg",
            "tail_lateral_deflection_px",
            "tail_curvature_px_inv",
        ]
    else:
        point_count = int(point_record["cardinality"])
        count = point_count - 1
        if count < 1:
            _fail("Tail posture requires at least one keypoint-to-keypoint segment.")
        role = "tail_segment"
        labels = [f"tail_segment_{index:03d}" for index in range(count)]
        membership = "segment_i_connects_tail_keypoint_i_to_i_plus_1_v1"
        bound_surfaces = ["tail_angle_rad", "tail_angle_deg"]
    return {
        "schema_id": "palette.tail_measurement_collection_axis",
        "schema_version": _SCHEMA_VERSION,
        "axis": 1,
        "role": role,
        "cardinality": count,
        "labels": labels,
        "sample_direction": "tail_base_to_tail_tip",
        "membership": membership,
        "point_collection_axis": _pointer(point_collection_axis),
        "bound_surfaces": bound_surfaces,
    }


def _stamp_or_load_measurement_collection_axis(
    run: Any,
    kind: str,
    point_collection_axis: BoundCoordinateRecord,
    *,
    load: bool,
) -> BoundCoordinateRecord:
    node = run["coordinate_records/measurement_collection_axis"]
    expected = _measurement_collection_axis_record(
        run,
        kind,
        point_collection_axis,
    )
    record = (
        bind_persisted_coordinate_record(
            node,
            attr_name=TAIL_MEASUREMENT_COLLECTION_AXIS_ATTR,
        )
        if load
        else stamp_and_bind_persisted_coordinate_record(
            node,
            expected,
            attr_name=TAIL_MEASUREMENT_COLLECTION_AXIS_ATTR,
        )
    )
    if record.record != expected:
        _fail("Tail measurement collection axis differs from its exact live surfaces.")
    return record


def _derivation_record(
    run: Any,
    kind: str,
    source: BoundSubjectShapeCoordinatePublication,
    source_authority: BoundCoordinateRecord,
    collection_axis: BoundCoordinateRecord,
    measurement_collection_axis: BoundCoordinateRecord,
) -> dict[str, Any]:
    expected_parameters = (
        {
            "method": "tail_metrics_from_subject_shape",
            "method_version": 1,
            "tail_angle_reference_axis": "caudal_axis=-forward_axis",
            "tail_angle_positive_direction": "anatomical_left",
        }
        if kind == _KINEMATICS_KIND
        else {
            "method": "tail_posture_view_from_subject_shape",
            "method_version": 1,
            "angle_convention": "megabouts_cumulative_segment_angle",
        }
    )
    contradictions = {
        name: run.attrs.get(name)
        for name, expected in expected_parameters.items()
        if run.attrs.get(name) != expected
    }
    if contradictions:
        _fail(
            "Tail derivation parameters contradict the controlled operation: "
            f"{contradictions!r}."
        )
    parameter_names = (
        ("method", "method_version", "tail_angle_sample_count", "tail_angle_reference_axis", "tail_angle_positive_direction")
        if kind == _KINEMATICS_KIND
        else ("method", "method_version", "view_family", "head_source", "keypoint_count", "angle_convention")
    )
    output_names = sorted(
        name
        for name in (_KINEMATICS_ARRAYS if kind == _KINEMATICS_KIND else _POSTURE_ARRAYS)
        if name not in {"instance_key", "source_crop_row_ids", "source_acquisition_frame_index"}
    )
    source_tail = source.descriptors[
        "components/subject_body/tail_sample_xy"
    ]
    if kind == _KINEMATICS_KIND:
        coordinate_outputs = {
            "tail_angle_sample_xy": {
                "operation": (
                    "linear_interpolation_over_normalized_tail_arclength_v1"
                ),
                "source_coordinate_descriptor": coordinate_descriptor_pointer(
                    source_tail
                ),
                "source_collection_axis": _pointer(source.tail_sample_axis),
                "target_collection_axis": _pointer(collection_axis),
                "output": _array_record(
                    "tail_angle_sample_xy",
                    run["tail_angle_sample_xy"],
                ),
            }
        }
    else:
        head_source = str(run.attrs.get("head_source") or "")
        head_path = f"components/subject_body/{head_source}"
        try:
            source_head = source.descriptors[head_path]
        except KeyError as exc:
            _fail(f"Tail posture head source descriptor is missing: {exc}.")
        coordinate_outputs = {
            "head_xy": {
                "operation": "exact_rowwise_coordinate_copy_v1",
                "source_coordinate_descriptor": coordinate_descriptor_pointer(
                    source_head
                ),
                "output": _array_record("head_xy", run["head_xy"]),
            },
            "tail_keypoints_xy": {
                "operation": (
                    "linear_interpolation_over_normalized_tail_arclength_v1"
                ),
                "source_coordinate_descriptor": coordinate_descriptor_pointer(
                    source_tail
                ),
                "source_collection_axis": _pointer(source.tail_sample_axis),
                "target_collection_axis": _pointer(collection_axis),
                "output": _array_record(
                    "tail_keypoints_xy",
                    run["tail_keypoints_xy"],
                ),
            },
        }
    return {
        "schema_id": "palette.tail_coordinate_derivation",
        "schema_version": _SCHEMA_VERSION,
        "kind": kind,
        "run_ref": f"/{canonical_node_path(run)}",
        "parameters": {
            name: _json_copy(run.attrs.get(name)) for name in parameter_names
        },
        "source_authority": _pointer(source_authority),
        "point_collection_axis": _pointer(collection_axis),
        "measurement_collection_axis": _pointer(measurement_collection_axis),
        "coordinate_outputs": coordinate_outputs,
        "outputs": {
            name: _array_record(name, run[name]) for name in output_names
        },
        "validity": _array_record("valid", run["valid"]),
        "row_order_preserved_from_source": True,
    }


def _stamp_or_load_derivation(
    run: Any,
    kind: str,
    source: BoundSubjectShapeCoordinatePublication,
    source_authority: BoundCoordinateRecord,
    collection_axis: BoundCoordinateRecord,
    measurement_collection_axis: BoundCoordinateRecord,
    *,
    load: bool,
) -> BoundCoordinateRecord:
    node = run["coordinate_records/derivation"]
    expected = _derivation_record(
        run,
        kind,
        source,
        source_authority,
        collection_axis,
        measurement_collection_axis,
    )
    record = (
        bind_persisted_coordinate_record(node, attr_name=TAIL_DERIVATION_ATTR)
        if load
        else stamp_and_bind_persisted_coordinate_record(
            node,
            expected,
            attr_name=TAIL_DERIVATION_ATTR,
        )
    )
    if record.record != expected:
        _fail("Tail coordinate derivation differs from live arrays or parameters.")
    return record


def _identity(run: Any, *, load: bool) -> BoundRowIdentityContract:
    key = run.get("instance_key")
    if key is None:
        _fail("Canonical tail publication requires direct instance_key.")
    if load:
        return load_bound_row_identity_contract(run, key)
    values = np.asarray(key[:])
    if values.dtype != np.dtype("uint64") or values.ndim != 1:
        _fail("Canonical tail instance_key must be one-dimensional uint64.")
    return stamp_and_bind_row_identity_contract(
        run,
        key,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=values,
        ),
    )


def _descriptor_bindings(
    run: Any,
    kind: str,
    source: BoundSubjectShapeCoordinatePublication,
    identity: BoundRowIdentityContract,
    source_authority: BoundCoordinateRecord,
    collection_axis_record: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    *,
    load: bool,
) -> dict[str, BoundCanonicalCoordinateDescriptor]:
    if kind == _KINEMATICS_KIND:
        source_descriptor = source.descriptors[
            "components/subject_body/tail_sample_xy"
        ]
        specs = {
            "tail_angle_sample_xy": {
                # The canonical descriptor vocabulary represents a labelled
                # point sequence as collected point_xy geometry.  Ordering and
                # tail-base-to-tip semantics are digest-bound by the collection
                # axis record below.
                "geometry_type": "point_xy",
                "collection": True,
            }
        }
    else:
        head_source = str(run.attrs.get("head_source") or "")
        source_descriptor = source.descriptors[
            f"components/subject_body/{head_source}"
        ]
        tail_descriptor = source.descriptors[
            "components/subject_body/tail_sample_xy"
        ]
        head_frame = source_descriptor.reference_frame_authority
        tail_frame = tail_descriptor.reference_frame_authority
        if (
            head_frame is None
            or tail_frame is None
            or head_frame.record_ref != tail_frame.record_ref
            or head_frame.record_sha256 != tail_frame.record_sha256
        ):
            _fail("Tail-posture head and tail sources use different camera frames.")
        specs = {
            "head_xy": {"geometry_type": "point_xy", "collection": False},
            "tail_keypoints_xy": {
                "geometry_type": "point_xy",
                "collection": True,
            },
        }
    frame = source_descriptor.reference_frame_authority
    if frame is None:
        _fail("Source subject-shape geometry lacks exact camera-frame authority.")
    # The canonical descriptor builder inserts collection label authority
    # immediately after reference-frame authority.  Preserve that canonical
    # order in the exact evidence tuple so fresh verification is byte-for-byte
    # stable rather than merely set-equivalent.
    lineage = (
        collection_axis_record,
        source.manifest,
        source_authority,
        derivation,
    )
    collection = CanonicalCollectionAxis(
        axis=1,
        role="keypoint",
        cardinality=int(collection_axis_record.record["cardinality"]),
        label_authority=DigestBoundCoordinateRecordRef(
            record_ref=collection_axis_record.record_ref,
            record_sha256=collection_axis_record.record_sha256,
        ),
    )
    result: dict[str, BoundCanonicalCoordinateDescriptor] = {}
    for name, spec in specs.items():
        node = run[name]
        kwargs = {
            "row_identity": identity,
            "reference_frame_authority": frame,
            "lineage_records": lineage,
        }
        result[name] = (
            load_bound_canonical_coordinate_descriptor(node, **kwargs)
            if load
            else build_bound_canonical_coordinate_descriptor(
                node,
                profile_id="source_camera_image_px.top_left_y_down.v1",
                geometry_type=str(spec["geometry_type"]),
                components=("x", "y"),
                component_units=("px", "px"),
                pixel_convention="continuous",
                row_identity=identity,
                reference_frame_authority=frame,
                source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
                lineage_records=lineage,
                collection_axis=collection if bool(spec["collection"]) else None,
            )
        )
    return result


def _measurement_specs(kind: str) -> dict[str, dict[str, Any]]:
    if kind == _KINEMATICS_KIND:
        return {
            "tail_angle_rad": {
                "quantity": "tail_signed_angle",
                "units": "rad",
                "operation": "signed_tail_tangent_angle_in_body_frame_v1",
                "axes": ("observation", "tail_sample"),
                "coordinate_inputs": (
                    "source_tail_tangent_xy",
                    "source_body_forward_axis_xy",
                    "source_body_left_axis_xy",
                ),
                "measurement_inputs": (),
                "collection_axis": True,
                "semantic_kind": "row_profile",
            },
            "tail_angle_deg": {
                "quantity": "tail_signed_angle",
                "units": "deg",
                "operation": "radians_to_degrees_v1",
                "axes": ("observation", "tail_sample"),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_angle_rad",),
                "collection_axis": True,
                "semantic_kind": "row_profile",
            },
            "tail_tip_angle_rad": {
                "quantity": "tail_tip_signed_angle",
                "units": "rad",
                "operation": "select_last_tail_sample_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_angle_rad",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "tail_tip_angle_deg": {
                "quantity": "tail_tip_signed_angle",
                "units": "deg",
                "operation": "radians_to_degrees_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_tip_angle_rad",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "tail_lateral_deflection_px": {
                "quantity": "tail_lateral_deflection",
                "units": "px",
                "operation": "project_tail_offset_onto_body_left_axis_v1",
                "axes": ("observation", "tail_sample"),
                "coordinate_inputs": (
                    "tail_angle_sample_xy",
                    "source_tail_base_xy",
                    "source_body_left_axis_xy",
                ),
                "measurement_inputs": (),
                "collection_axis": True,
                "semantic_kind": "row_profile",
            },
            "tail_tip_lateral_deflection_px": {
                "quantity": "tail_tip_lateral_deflection",
                "units": "px",
                "operation": "select_last_tail_sample_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_lateral_deflection_px",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "max_abs_tail_angle_rad": {
                "quantity": "maximum_absolute_tail_angle",
                "units": "rad",
                "operation": "row_maximum_absolute_value_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_angle_rad",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "max_abs_tail_angle_deg": {
                "quantity": "maximum_absolute_tail_angle",
                "units": "deg",
                "operation": "radians_to_degrees_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("max_abs_tail_angle_rad",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "tail_angle_rms_rad": {
                "quantity": "tail_angle_root_mean_square",
                "units": "rad",
                "operation": "row_root_mean_square_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_angle_rad",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "tail_angle_rms_deg": {
                "quantity": "tail_angle_root_mean_square",
                "units": "deg",
                "operation": "radians_to_degrees_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_angle_rms_rad",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "integrated_abs_tail_angle_rad": {
                "quantity": "integrated_absolute_tail_angle",
                "units": "rad",
                "operation": "trapezoidal_integral_over_normalized_tail_arclength_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_angle_rad",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "tail_curvature_px_inv": {
                "quantity": "tail_curvature",
                "units": "px^-1",
                "operation": "interpolate_source_curvature_on_tail_sample_axis_v1",
                "axes": ("observation", "tail_sample"),
                "coordinate_inputs": ("tail_angle_sample_xy",),
                "measurement_inputs": ("source_tail_curvature_px_inv",),
                "collection_axis": True,
                "semantic_kind": "row_profile",
            },
            "max_abs_tail_curvature_px_inv": {
                "quantity": "maximum_absolute_tail_curvature",
                "units": "px^-1",
                "operation": "row_maximum_absolute_value_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_curvature_px_inv",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "integrated_abs_tail_curvature": {
                "quantity": "integrated_absolute_tail_curvature",
                "units": "px^-1",
                "operation": "trapezoidal_integral_over_normalized_tail_arclength_v1",
                "axes": ("observation",),
                "coordinate_inputs": (),
                "measurement_inputs": ("tail_curvature_px_inv",),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
        }
    if kind == _POSTURE_KIND:
        return {
            "head_yaw_rad": {
                "quantity": "head_yaw",
                "units": "rad",
                "operation": "signed_head_axis_angle_in_source_camera_frame_v1",
                "axes": ("observation",),
                "coordinate_inputs": ("head_xy", "tail_keypoints_xy"),
                "measurement_inputs": (),
                "collection_axis": False,
                "semantic_kind": "row_scalar",
            },
            "tail_angle_rad": {
                "quantity": "tail_segment_signed_angle",
                "units": "rad",
                "operation": "cumulative_signed_keypoint_segment_angle_v1",
                "axes": ("observation", "tail_segment"),
                "coordinate_inputs": ("head_xy", "tail_keypoints_xy"),
                "measurement_inputs": (),
                "collection_axis": True,
                "semantic_kind": "row_profile",
            },
            "tail_angle_deg": {
                "quantity": "tail_segment_signed_angle",
                "units": "deg",
                "operation": "radians_to_degrees_v1",
                "axes": ("observation", "tail_segment"),
                "coordinate_inputs": ("head_xy", "tail_keypoints_xy"),
                "measurement_inputs": ("tail_angle_rad",),
                "collection_axis": True,
                "semantic_kind": "row_profile",
            },
        }
    _fail(f"Unsupported tail publication kind {kind!r}.")
    raise AssertionError


def _source_measurement_inputs(
    kind: str,
    source: BoundSubjectShapeCoordinatePublication,
) -> tuple[
    dict[str, BoundCanonicalCoordinateDescriptor],
    dict[str, BoundCoordinateRecord],
]:
    """Bind the exact subject-shape inputs consumed by tail measurements."""

    if kind != _KINEMATICS_KIND:
        return {}, {}
    coordinate_paths = {
        "source_tail_tangent_xy": "components/subject_body/tail_tangent_xy",
        "source_tail_base_xy": "components/subject_body/tail_base_xy",
        "source_body_forward_axis_xy": "body_frame/forward_axis_xy",
        "source_body_left_axis_xy": "body_frame/left_axis_xy",
    }
    try:
        coordinates = {
            name: source.descriptors[path]
            for name, path in coordinate_paths.items()
        }
        curvature = source.require_scalar_surface(
            "components/subject_body/tail_curvature_px_inv",
            units="px^-1",
            surface_kind="row_profile",
        )
    except (KeyError, SubjectShapeCoordinatePublicationError) as exc:
        _fail(f"Tail kinematics lacks one exact source measurement input: {exc}.")
    return coordinates, {
        "source_tail_curvature_px_inv": curvature.semantics,
    }


def _measurement_authority_record(
    run: Any,
    kind: str,
    identity: BoundRowIdentityContract,
    source_authority: BoundCoordinateRecord,
    point_collection_axis: BoundCoordinateRecord,
    measurement_collection_axis: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    descriptors: Mapping[str, BoundCanonicalCoordinateDescriptor],
    source_coordinate_inputs: Mapping[str, BoundCanonicalCoordinateDescriptor],
    source_measurement_inputs: Mapping[str, BoundCoordinateRecord],
) -> dict[str, Any]:
    specs = _measurement_specs(kind)
    all_coordinates = {**descriptors, **source_coordinate_inputs}
    return {
        "schema_id": "palette.tail_measurement_authority",
        "schema_version": _SCHEMA_VERSION,
        "policy": "closed_world_array_specific_measurement_semantics_v1",
        "kind": kind,
        "run_ref": f"/{canonical_node_path(run)}",
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "source_authority": _pointer(source_authority),
        "point_collection_axis": _pointer(point_collection_axis),
        "measurement_collection_axis": _pointer(measurement_collection_axis),
        "derivation": _pointer(derivation),
        "coordinate_inputs": {
            name: {
                "record_ref": (
                    f"/{canonical_node_path(binding.coordinate_node)}"
                    f"@{COORDINATE_DESCRIPTOR_ATTR}"
                ),
                "record_sha256": binding.descriptor.digest(),
            }
            for name, binding in sorted(all_coordinates.items())
        },
        "measurement_inputs": {
            name: _pointer(binding)
            for name, binding in sorted(source_measurement_inputs.items())
        },
        "surfaces": {
            name: {
                "quantity": str(spec["quantity"]),
                "units": str(spec["units"]),
                "operation": str(spec["operation"]),
                "axes": list(spec["axes"]),
                "coordinate_inputs": list(spec["coordinate_inputs"]),
                "measurement_inputs": list(spec["measurement_inputs"]),
                "collection_axis": bool(spec["collection_axis"]),
                "semantic_kind": str(spec["semantic_kind"]),
                "output": _array_record(name, run[name]),
            }
            for name, spec in specs.items()
        },
        "validity": {
            "payload": _array_record("valid", run["valid"]),
            "policy": "run_valid_true_and_value_finite_v1",
        },
    }


def _stamp_or_load_measurement_authority(
    run: Any,
    kind: str,
    identity: BoundRowIdentityContract,
    source_authority: BoundCoordinateRecord,
    point_collection_axis: BoundCoordinateRecord,
    measurement_collection_axis: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    descriptors: Mapping[str, BoundCanonicalCoordinateDescriptor],
    source_coordinate_inputs: Mapping[str, BoundCanonicalCoordinateDescriptor],
    source_measurement_inputs: Mapping[str, BoundCoordinateRecord],
    *,
    load: bool,
) -> BoundCoordinateRecord:
    node = run["coordinate_records/measurement_authority"]
    expected = _measurement_authority_record(
        run,
        kind,
        identity,
        source_authority,
        point_collection_axis,
        measurement_collection_axis,
        derivation,
        descriptors,
        source_coordinate_inputs,
        source_measurement_inputs,
    )
    record = (
        bind_persisted_coordinate_record(
            node,
            attr_name=TAIL_MEASUREMENT_AUTHORITY_ATTR,
        )
        if load
        else stamp_and_bind_persisted_coordinate_record(
            node,
            expected,
            attr_name=TAIL_MEASUREMENT_AUTHORITY_ATTR,
        )
    )
    if record.record != expected:
        _fail("Tail measurement authority differs from live arrays or semantics.")
    return record


def _measurement_bindings(
    run: Any,
    kind: str,
    identity: BoundRowIdentityContract,
    coordinate_bindings: Mapping[str, BoundCanonicalCoordinateDescriptor],
    source_coordinate_bindings: Mapping[str, BoundCanonicalCoordinateDescriptor],
    source_measurement_bindings: Mapping[str, BoundCoordinateRecord],
    measurement_collection_axis: BoundCoordinateRecord,
    measurement_authority: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    *,
    load: bool,
) -> dict[str, BoundCoordinateRecord]:
    specs = _measurement_specs(kind)
    all_coordinate_bindings = {
        **coordinate_bindings,
        **source_coordinate_bindings,
    }
    result: dict[str, BoundCoordinateRecord] = {}
    remaining = dict(specs)
    while remaining:
        progressed = False
        for name in tuple(remaining):
            spec = remaining[name]
            input_names = tuple(spec["measurement_inputs"])
            if any(
                input_name not in result
                and input_name not in source_measurement_bindings
                for input_name in input_names
            ):
                continue
            coordinate_names = tuple(spec["coordinate_inputs"])
            try:
                expected = build_array_measurement_descriptor(
                    run[name],
                    quantity=str(spec["quantity"]),
                    units=str(spec["units"]),
                    operation=str(spec["operation"]),
                    axes=tuple(spec["axes"]),
                    coordinate_inputs=tuple(
                        all_coordinate_bindings[input_name]
                        for input_name in coordinate_names
                    ),
                    measurement_inputs=tuple(
                        (
                            result[input_name]
                            if input_name in result
                            else source_measurement_bindings[input_name]
                        )
                        for input_name in input_names
                    ),
                    row_identity=identity,
                    collection=measurement_collection_axis,
                    measurement_authority=measurement_authority,
                    derivation=derivation,
                    row_axis_name="observation",
                    collection_axis_name=(
                        str(spec["axes"][1]) if bool(spec["collection_axis"]) else None
                    ),
                    collection_axis_role=(
                        str(measurement_collection_axis.record["role"])
                        if bool(spec["collection_axis"])
                        else None
                    ),
                    validity_node=run["valid"],
                    validity_policy="run_valid_true_and_value_finite_v1",
                    semantic_kind=str(spec["semantic_kind"]),
                )
                result[name] = (
                    load_bound_array_measurement_descriptor(
                        run[name],
                        expected_record=expected,
                        attr_name=ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
                    )
                    if load
                    else stamp_and_bind_array_measurement_descriptor(
                        run[name],
                        expected,
                        attr_name=ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
                    )
                )
            except Exception as exc:
                _fail(f"Measurement descriptor for {name!r} is invalid: {exc}.")
            del remaining[name]
            progressed = True
        if not progressed:
            _fail(
                "Tail measurement descriptor dependencies contain a cycle or "
                f"missing input: {tuple(sorted(remaining))!r}."
            )
    return result


def _manifest_record(
    run: Any,
    kind: str,
    source: BoundSubjectShapeCoordinatePublication,
    identity: BoundRowIdentityContract,
    source_authority: BoundCoordinateRecord,
    collection_axis: BoundCoordinateRecord,
    measurement_collection_axis: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    descriptors: Mapping[str, BoundCanonicalCoordinateDescriptor],
    measurement_authority: BoundCoordinateRecord,
    measurements: Mapping[str, BoundCoordinateRecord],
) -> dict[str, Any]:
    arrays, groups = _validate_closed_schema(run, kind)
    array_attrs = {path: _node_attrs_record(run[path]) for path in arrays}
    group_attrs = {path: _node_attrs_record(run[path]) for path in groups}
    return {
        "schema_id": "palette.tail_coordinate_publication_manifest",
        "schema_version": _SCHEMA_VERSION,
        "kind": kind,
        "run_ref": f"/{canonical_node_path(run)}",
        "publication_owner_uuid": _publication_owner_uuid(run),
        "source_subject_shape": {
            "run_ref": f"/{source.run_path}",
            "publication_manifest": _pointer(source.manifest),
            "authority": _pointer(source_authority),
        },
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "point_collection_axis": _pointer(collection_axis),
        "measurement_collection_axis": _pointer(measurement_collection_axis),
        "derivation": _pointer(derivation),
        "coordinate_descriptors": {
            name: {
                "record_ref": f"/{canonical_node_path(binding.coordinate_node)}@{COORDINATE_DESCRIPTOR_ATTR}",
                "descriptor_sha256": binding.descriptor.digest(),
            }
            for name, binding in sorted(descriptors.items())
        },
        "measurement_authority": _pointer(measurement_authority),
        "measurement_descriptors": {
            name: _pointer(binding)
            for name, binding in sorted(measurements.items())
        },
        "arrays": {path: _array_record(path, run[path]) for path in arrays},
        "schema_inventory": {
            "groups": list(groups),
            "arrays": list(arrays),
            "run_attrs": _node_attrs_record(run, excluded=_OPERATIONAL_RUN_ATTRS),
            "group_attrs": group_attrs,
            "array_attrs": array_attrs,
            "excluded_operational_run_attrs": sorted(_OPERATIONAL_RUN_ATTRS),
            "closed_group_inventory": True,
            "closed_array_inventory": True,
            "closed_attr_inventory": True,
        },
    }


@dataclass(frozen=True, init=False)
class BoundTailCoordinatePublication:
    run_path: str
    kind: str
    source: BoundSubjectShapeCoordinatePublication = field(repr=False)
    row_identity: BoundRowIdentityContract = field(repr=False)
    source_authority: BoundCoordinateRecord = field(repr=False)
    collection_axis: BoundCoordinateRecord = field(repr=False)
    measurement_collection_axis: BoundCoordinateRecord = field(repr=False)
    derivation: BoundCoordinateRecord = field(repr=False)
    descriptors: Mapping[str, BoundCanonicalCoordinateDescriptor]
    measurement_authority: BoundCoordinateRecord = field(repr=False)
    measurements: Mapping[str, BoundCoordinateRecord]
    manifest: BoundCoordinateRecord = field(repr=False)
    _run: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _BOUND_TAIL_PUBLICATION_SEAL:
            _fail("Tail coordinate publications cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


_BOUND_TAIL_PUBLICATION_SEAL = object()


def _source_publication(root: Any, run: Any) -> BoundSubjectShapeCoordinatePublication:
    source_path = str(run.attrs.get("source_subject_shape_path") or "").strip("/")
    if not source_path:
        _fail("Tail run lacks source_subject_shape_path.")
    try:
        return load_persisted_subject_shape_coordinate_publication(root, source_path)
    except SubjectShapeCoordinatePublicationError as exc:
        raise TailCoordinatePublicationError(
            f"Tail source is not one exact canonical subject-shape publication: {exc}"
        ) from exc


def _prepare_record_groups(run: Any) -> None:
    records = run.require_group("coordinate_records")
    records.require_group("source_subject_shape")
    records.require_group("point_collection_axis")
    records.require_group("measurement_collection_axis")
    records.require_group("derivation")
    records.require_group("measurement_authority")


def publish_tail_coordinate_surfaces(
    root: Any,
    run: Any,
    *,
    kind: str,
) -> BoundTailCoordinatePublication:
    """Seal one complete numerical tail run before selector activation."""

    prefix = _expected_prefix(kind)
    run_path = canonical_node_path(run)
    if not run_path.startswith(prefix) or "/" in run_path[len(prefix) :]:
        _fail(f"Tail run path {run_path!r} is outside the controlled {kind!r} parent.")
    if run.attrs.get("stage_selector_eligible") is not False:
        _fail("Tail publication requires a literal selector-ineligible staged run.")
    source = _source_publication(root, run)
    run.attrs["coordinate_contract"] = TAIL_COORDINATE_CONTRACT
    run.attrs["tail_coordinate_publication_kind"] = kind
    _prepare_record_groups(run)
    identity = _identity(run, load=False)
    source_authority = _stamp_or_load_source_authority(run, source, load=False)
    collection_axis = _stamp_or_load_collection_axis(run, kind, load=False)
    measurement_collection_axis = _stamp_or_load_measurement_collection_axis(
        run,
        kind,
        collection_axis,
        load=False,
    )
    derivation = _stamp_or_load_derivation(
        run,
        kind,
        source,
        source_authority,
        collection_axis,
        measurement_collection_axis,
        load=False,
    )
    descriptors = _descriptor_bindings(
        run,
        kind,
        source,
        identity,
        source_authority,
        collection_axis,
        derivation,
        load=False,
    )
    stamp_bound_canonical_coordinate_descriptors(descriptors.values())
    source_coordinate_inputs, source_measurement_inputs = (
        _source_measurement_inputs(kind, source)
    )
    measurement_authority = _stamp_or_load_measurement_authority(
        run,
        kind,
        identity,
        source_authority,
        collection_axis,
        measurement_collection_axis,
        derivation,
        descriptors,
        source_coordinate_inputs,
        source_measurement_inputs,
        load=False,
    )
    measurements = _measurement_bindings(
        run,
        kind,
        identity,
        descriptors,
        source_coordinate_inputs,
        source_measurement_inputs,
        measurement_collection_axis,
        measurement_authority,
        derivation,
        load=False,
    )
    expected_manifest = _manifest_record(
        run,
        kind,
        source,
        identity,
        source_authority,
        collection_axis,
        measurement_collection_axis,
        derivation,
        descriptors,
        measurement_authority,
        measurements,
    )
    manifest = stamp_and_bind_persisted_coordinate_record(
        run,
        expected_manifest,
        attr_name=TAIL_PUBLICATION_MANIFEST_ATTR,
    )
    run.attrs[TAIL_PUBLICATION_MANIFEST_ALIAS_ATTR] = manifest.record_sha256
    return _load_tail_coordinate_publication(
        root,
        run_path,
        expected_kind=kind,
        require_complete=False,
        expected_selector_eligible=False,
    )


def _load_tail_coordinate_publication(
    root: Any,
    run_path: str,
    *,
    expected_selector_eligible: bool,
    expected_kind: str | None = None,
    require_complete: bool = True,
) -> BoundTailCoordinatePublication:
    """Freshly validate one closed future tail publication."""

    path = str(run_path).strip("/")
    run = root.get(path)
    if run is None:
        _fail(f"Tail run {path!r} is missing.")
    kind = str(run.attrs.get("tail_coordinate_publication_kind") or "")
    if expected_kind is not None and kind != expected_kind:
        _fail(f"Tail publication kind {kind!r} differs from {expected_kind!r}.")
    prefix = _expected_prefix(kind)
    if not path.startswith(prefix) or "/" in path[len(prefix) :]:
        _fail("Tail publication path does not match its declared kind.")
    if run.attrs.get("coordinate_contract") != TAIL_COORDINATE_CONTRACT:
        _fail("Tail publication lacks canonical_v2 coordinate_contract.")
    if require_complete and run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        _fail("Tail coordinate publication is not complete.")
    if run.attrs.get("stage_selector_eligible") is not expected_selector_eligible:
        state = "eligible" if expected_selector_eligible else "ineligible"
        _fail(f"Tail coordinate publication is not literally selector-{state}.")
    source = _source_publication(root, run)
    identity = _identity(run, load=True)
    source_authority = _stamp_or_load_source_authority(run, source, load=True)
    collection_axis = _stamp_or_load_collection_axis(run, kind, load=True)
    measurement_collection_axis = _stamp_or_load_measurement_collection_axis(
        run,
        kind,
        collection_axis,
        load=True,
    )
    derivation = _stamp_or_load_derivation(
        run,
        kind,
        source,
        source_authority,
        collection_axis,
        measurement_collection_axis,
        load=True,
    )
    descriptors = _descriptor_bindings(
        run,
        kind,
        source,
        identity,
        source_authority,
        collection_axis,
        derivation,
        load=True,
    )
    source_coordinate_inputs, source_measurement_inputs = (
        _source_measurement_inputs(kind, source)
    )
    measurement_authority = _stamp_or_load_measurement_authority(
        run,
        kind,
        identity,
        source_authority,
        collection_axis,
        measurement_collection_axis,
        derivation,
        descriptors,
        source_coordinate_inputs,
        source_measurement_inputs,
        load=True,
    )
    measurements = _measurement_bindings(
        run,
        kind,
        identity,
        descriptors,
        source_coordinate_inputs,
        source_measurement_inputs,
        measurement_collection_axis,
        measurement_authority,
        derivation,
        load=True,
    )
    manifest = bind_persisted_coordinate_record(
        run,
        attr_name=TAIL_PUBLICATION_MANIFEST_ATTR,
    )
    expected_manifest = _manifest_record(
        run,
        kind,
        source,
        identity,
        source_authority,
        collection_axis,
        measurement_collection_axis,
        derivation,
        descriptors,
        measurement_authority,
        measurements,
    )
    if manifest.record != expected_manifest:
        _fail("Tail publication manifest differs from live arrays, attrs, or lineage.")
    if run.attrs.get(TAIL_PUBLICATION_MANIFEST_ALIAS_ATTR) != manifest.record_sha256:
        _fail("Tail publication manifest digest alias is stale.")
    return BoundTailCoordinatePublication(
        run_path=path,
        kind=kind,
        source=source,
        row_identity=identity,
        source_authority=source_authority,
        collection_axis=collection_axis,
        measurement_collection_axis=measurement_collection_axis,
        derivation=derivation,
        descriptors=descriptors,
        measurement_authority=measurement_authority,
        measurements=measurements,
        manifest=manifest,
        _run=run,
        _verification_seal=_BOUND_TAIL_PUBLICATION_SEAL,
    )


def load_tail_coordinate_publication(
    root: Any,
    run_path: str,
    *,
    expected_kind: str | None = None,
) -> BoundTailCoordinatePublication:
    try:
        return _load_tail_coordinate_publication(
            root,
            run_path,
            expected_kind=expected_kind,
            require_complete=True,
            expected_selector_eligible=True,
        )
    except TailCoordinatePublicationError:
        raise
    except Exception as exc:
        raise TailCoordinatePublicationError(
            f"Tail coordinate publication is invalid: {exc}"
        ) from exc


def publish_tail_kinematics_coordinate_surfaces(root: Any, run: Any) -> BoundTailCoordinatePublication:
    return publish_tail_coordinate_surfaces(root, run, kind=_KINEMATICS_KIND)


def publish_tail_posture_coordinate_surfaces(root: Any, run: Any) -> BoundTailCoordinatePublication:
    return publish_tail_coordinate_surfaces(root, run, kind=_POSTURE_KIND)


def _tail_activation_inputs(
    root: Any,
    parent: Any,
    run: Any,
    *,
    run_name: str,
    additional_selector_attrs: Sequence[str],
) -> tuple[str, str, str, tuple[str, ...], Any]:
    """Resolve one exact candidate plus its fresh digest-bound proof loader."""

    name = str(run_name).strip()
    kind = str(run.attrs.get("tail_coordinate_publication_kind") or "")
    prefix = _expected_prefix(kind)
    expected_path = f"{prefix}{name}"
    expected_parent = prefix.rstrip("/")
    if (
        not name
        or "/" in name
        or canonical_node_path(run) != expected_path
        or canonical_node_path(parent) != expected_parent
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        _fail("Tail activation requires one exact selector-ineligible staged run.")
    selector_names = tuple(
        dict.fromkeys(("latest_complete", "latest", *additional_selector_attrs))
    )
    if any(
        type(selector) is not str
        or not selector
        or "/" in selector
        or selector == "stage_selector_eligible"
        for selector in selector_names
    ):
        _fail("Tail activation received an invalid parent selector attribute.")

    def proof() -> tuple[Any, ...]:
        bound = _load_tail_coordinate_publication(
            root,
            expected_path,
            expected_selector_eligible=False,
            expected_kind=kind,
            require_complete=True,
        )
        return (
            bound.manifest.record_sha256,
            bound.source.manifest.record_sha256,
            bound.row_identity.record_ref,
            bound.row_identity.record_sha256,
            bound.source_authority.record_sha256,
            bound.collection_axis.record_sha256,
            bound.measurement_collection_axis.record_sha256,
            bound.derivation.record_sha256,
            bound.measurement_authority.record_sha256,
            tuple(
                (key, value.descriptor.digest())
                for key, value in sorted(bound.descriptors.items())
            ),
            tuple(
                (key, value.record_sha256)
                for key, value in sorted(bound.measurements.items())
            ),
        )

    return name, kind, expected_path, selector_names, proof


def _write_tail_activation_attr(attrs: Any, name: str, value: Any) -> None:
    """Perform one injectable tail activation write."""

    write_activation_attr(attrs, name, value)


def activate_tail_coordinate_publication(
    root: Any,
    parent: Any,
    run: Any,
    *,
    run_name: str,
    expected_publication_owner_uuid: str,
    additional_selector_attrs: Sequence[str] = (),
) -> None:
    """Lease one generation and expose eligibility as the literal commit."""

    name, _kind, expected_path, selectors, proof = _tail_activation_inputs(
        root,
        parent,
        run,
        run_name=run_name,
        additional_selector_attrs=additional_selector_attrs,
    )
    try:
        activate_selector_eligible_run(
            root,
            parent,
            run,
            parent_path=canonical_node_path(parent),
            run_path=expected_path,
            run_name=name,
            owner_attr=TAIL_PUBLICATION_OWNER_ATTR,
            expected_owner_uuid=expected_publication_owner_uuid,
            policy_attr=TAIL_PUBLICATION_POLICY_ATTR,
            generation_attr=TAIL_PUBLICATION_GENERATION_ATTR,
            lease_attr=TAIL_PARENT_PUBLICATION_LEASE_ATTR,
            policy=TAIL_PUBLICATION_POLICY,
            lease_schema_id="palette.tail_publication_lease",
            proof_loader=proof,
            selector_attrs=selectors,
            attr_writer=_write_tail_activation_attr,
        )
    except SelectorActivationError as exc:
        raise TailCoordinatePublicationError(
            f"Tail activation lost exact ownership: {exc}."
        ) from exc


def defer_tail_coordinate_publication_activation(
    root: Any,
    parent: Any,
    run: Any,
    *,
    run_name: str,
    expected_publication_owner_uuid: str,
    additional_selector_attrs: Sequence[str] = (),
) -> DeferredSelectorActivation:
    """Lease/select one run while deferring its final eligibility write."""

    name, _kind, expected_path, selectors, proof = _tail_activation_inputs(
        root,
        parent,
        run,
        run_name=run_name,
        additional_selector_attrs=additional_selector_attrs,
    )
    try:
        activation = activate_selector_eligible_run(
            root,
            parent,
            run,
            parent_path=canonical_node_path(parent),
            run_path=expected_path,
            run_name=name,
            owner_attr=TAIL_PUBLICATION_OWNER_ATTR,
            expected_owner_uuid=expected_publication_owner_uuid,
            policy_attr=TAIL_PUBLICATION_POLICY_ATTR,
            generation_attr=TAIL_PUBLICATION_GENERATION_ATTR,
            lease_attr=TAIL_PARENT_PUBLICATION_LEASE_ATTR,
            policy=TAIL_PUBLICATION_POLICY,
            lease_schema_id="palette.tail_publication_lease",
            proof_loader=proof,
            selector_attrs=selectors,
            attr_writer=_write_tail_activation_attr,
            defer_eligibility=True,
        )
    except SelectorActivationError as exc:
        raise TailCoordinatePublicationError(
            f"Tail deferred activation lost exact ownership: {exc}."
        ) from exc
    if not isinstance(activation, DeferredSelectorActivation):
        _fail("Tail deferred activation did not return its owned lease receipt.")
    return activation


def commit_deferred_tail_coordinate_publication_activation(
    activation: DeferredSelectorActivation,
    *,
    root: Any,
    parent: Any,
    run: Any,
) -> None:
    """Rebind a tail lease to fresh atomic-publisher handles and commit it."""

    try:
        additional_selectors = tuple(
            selector
            for selector in activation.selectors
            if selector not in {"latest_complete", "latest"}
        )
        name, _kind, expected_path, selectors, proof = _tail_activation_inputs(
            root,
            parent,
            run,
            run_name=activation.run_name,
            additional_selector_attrs=additional_selectors,
        )
        if (
            name != activation.run_name
            or expected_path != activation.run_path
            or selectors != activation.selectors
        ):
            _fail("Fresh tail activation handles do not match the deferred receipt.")
        commit_deferred_selector_activation(
            activation,
            root=root,
            parent_group=parent,
            run_group=run,
            proof_loader=proof,
        )
    except SelectorActivationError as exc:
        raise TailCoordinatePublicationError(
            f"Tail deferred activation lost exact ownership: {exc}."
        ) from exc


def rollback_deferred_tail_coordinate_publication_activation(
    activation: DeferredSelectorActivation,
) -> None:
    """Conditionally cancel a staged tail lease without clobbering a successor."""

    try:
        rollback_deferred_selector_activation(activation)
    except SelectorActivationError as exc:
        raise TailCoordinatePublicationError(
            f"Tail deferred activation rollback was incomplete: {exc}."
        ) from exc


def load_tail_kinematics_coordinate_publication(root: Any, run_path: str) -> BoundTailCoordinatePublication:
    return load_tail_coordinate_publication(root, run_path, expected_kind=_KINEMATICS_KIND)


def load_tail_posture_coordinate_publication(root: Any, run_path: str) -> BoundTailCoordinatePublication:
    return load_tail_coordinate_publication(root, run_path, expected_kind=_POSTURE_KIND)


__all__ = [
    "ARRAY_MEASUREMENT_DESCRIPTOR_ATTR",
    "BoundTailCoordinatePublication",
    "TAIL_COORDINATE_CONTRACT",
    "TAIL_PARENT_PUBLICATION_LEASE_ATTR",
    "TAIL_PUBLICATION_GENERATION_ATTR",
    "TAIL_PUBLICATION_MANIFEST_ATTR",
    "TAIL_PUBLICATION_MANIFEST_ALIAS_ATTR",
    "TAIL_PUBLICATION_OWNER_ATTR",
    "TAIL_PUBLICATION_POLICY",
    "TAIL_PUBLICATION_POLICY_ATTR",
    "TailCoordinatePublicationError",
    "activate_tail_coordinate_publication",
    "commit_deferred_tail_coordinate_publication_activation",
    "defer_tail_coordinate_publication_activation",
    "load_tail_coordinate_publication",
    "load_tail_kinematics_coordinate_publication",
    "load_tail_posture_coordinate_publication",
    "publish_tail_coordinate_surfaces",
    "publish_tail_kinematics_coordinate_surfaces",
    "publish_tail_posture_coordinate_surfaces",
    "rollback_deferred_tail_coordinate_publication_activation",
]
