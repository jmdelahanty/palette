"""Fail-closed coordinate publication for canonical base keypoint runs.

The base YOLO pose writer consumes a persisted canonical crop.  This module is
the only boundary allowed to turn that crop into keypoint coordinate evidence.
It deliberately has two phases:

* :func:`prepare_keypoint_coordinate_context` runs before inference.  It loads
  the selected crop from the archive, copies its exact observation identity,
  temporal mapping, placement, and ROI extent into the new rowset, and persists
  the exact constant model-input preprocessing transform that inference must
  use.
* :func:`publish_keypoint_coordinate_surfaces` runs after numeric output is
  durable.  It verifies dtype-preserving ROI/image/normalized relationships,
  stamps array-owned canonical-v2 descriptors transactionally, and performs a
  fresh root-based load before returning.

Neither API accepts a caller-constructed crop frame, extent, placement, or
transform. Each supported crop profile must pass the shared resolver first.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.pose_model_schema_binding import (
    validate_pose_model_schema_binding,
)

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CanonicalCollectionAxis,
    DigestBoundCoordinateRecordRef,
)
from fisheye.shared.coordinate_identity import (
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
    load_bound_row_identity_contract,
    load_bound_source_row_temporal_authority,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_surface_contract import (
    ROI_BBOX_XYXY,
    ROI_POINT_XY,
    SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    SOURCE_CAMERA_BBOX_XYXY,
    SOURCE_CAMERA_CROP_XYWH,
    SOURCE_CAMERA_NORMALIZED_BBOX_XYXY,
    SOURCE_CAMERA_NORMALIZED_POINT_XY,
    SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
    SOURCE_CAMERA_POINT_XY,
)
from fisheye.shared.coordinate_reference import (
    bind_array_reference_extent,
    bind_persisted_record_reference_extent,
    canonical_node_path,
)
from fisheye.shared.directed_transform_chain import (
    BoundDirectedTransformChain,
    apply_bound_directed_transform_chain,
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_ATTR,
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    BoundDirectedTransformV2,
    load_bound_directed_transform_v2,
    stamp_directed_transform_v2,
)
from fisheye.shared.model_input_transform import ModelInputTransform
from fisheye.shared.observation_coordinate_publication import (
    BoundCropObservationGeometry,
    CROP_ROI_BBOX_EDGE_FRAME_RELATIVE_PATH,
    CropRoiGeometryPublicationResult,
    load_crop_roi_bbox_edge_reference_extent,
    load_crop_roi_geometry,
    load_persisted_crop_observation_geometry,
    require_bound_crop_observation_geometry,
)
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    BoundCropPlacementOwnership,
    BoundPixelFrameAuthority,
    array_values_sha256,
    load_crop_placement_ownership,
    load_model_input_pixel_frame_authority,
    load_normalized_pixel_frame_authority,
    load_roi_pixel_frame_authority,
    model_input_to_roi_matrix,
    normalized_to_pixel_matrix,
    stamp_crop_placement_ownership,
    stamp_model_input_pixel_frame_authority,
    stamp_normalized_pixel_frame_authority,
    stamp_roi_pixel_frame_authority,
)
from fisheye.shared.proof_verification import (
    finish_proof_verification,
    load_verified_value,
    verify_persisted_proof,
)
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    load_bound_transform_authority,
    stamp_crop_placement_transform_authority,
    stamp_model_input_transform_authority,
    stamp_normalized_to_pixel_transform_authority,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
)


KEYPOINT_COORDINATE_CONTEXT_ATTR = "keypoint_coordinate_context"
KEYPOINT_COORDINATE_DERIVATION_ATTR = "keypoint_coordinate_derivation"
KEYPOINT_LABEL_AUTHORITY_ATTR = "keypoint_label_authority"
KEYPOINT_ROI_REFERENCE_EXTENT_ATTR = "keypoint_roi_reference_extent"
KEYPOINT_MODEL_REFERENCE_EXTENT_ATTR = "keypoint_model_reference_extent"
KEYPOINT_PUBLICATION_OWNER_ATTR = "keypoint_publication_owner"
KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR = "keypoint_publication_lease"
KEYPOINT_PUBLICATION_GENERATION_ATTR = "publication_generation"
KEYPOINT_PUBLICATION_POLICY_ATTR = "publication_policy"

KEYPOINT_COORDINATE_CONTEXT_SCHEMA_ID = "palette.keypoint_coordinate_context"
KEYPOINT_COORDINATE_CONTEXT_SCHEMA_VERSION = 2
KEYPOINT_COORDINATE_DERIVATION_SCHEMA_ID = "palette.keypoint_coordinate_derivation"
KEYPOINT_COORDINATE_DERIVATION_SCHEMA_VERSION = 2
KEYPOINT_LABEL_AUTHORITY_SCHEMA_ID = "palette.keypoint_label_authority"
KEYPOINT_LABEL_AUTHORITY_SCHEMA_VERSION = 1
KEYPOINT_REFERENCE_EXTENT_SCHEMA_ID = "palette.keypoint_reference_extent"
KEYPOINT_REFERENCE_EXTENT_SCHEMA_VERSION = 1

KEYPOINT_ARRAY_NAMES = (
    "keypoints_roi",
    "keypoints_img",
    "keypoints_norm",
    "pose_bbox_xyxy_roi",
    "pose_bbox_xyxy_img",
    "pose_bbox_xyxy_norm",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOUND_SOURCE_SEAL = object()
_BOUND_CONTEXT_SEAL = object()
_BOUND_SURFACES_SEAL = object()
_PUBLICATION_CHECKPOINT_SEAL = object()
_PUBLICATION_OWNER_RE = re.compile(r"^[0-9a-f]{32}$")
_KEYPOINT_PUBLICATION_POLICY = (
    "owner_generation_guarded_selectors_then_eligibility_v1"
)
_KEYPOINT_GUARDED_PARENT_SELECTORS = ("latest", "latest_complete")
_KEYPOINT_GUARDED_ROOT_SELECTORS = ("current_keypoint_group_path",)


class KeypointCoordinatePublicationError(ValueError):
    """Raised when a base keypoint run cannot prove canonical coordinates."""


def _fail(message: str) -> None:
    raise KeypointCoordinatePublicationError(message)


def _publication_owner(run: Any, *, expected: str | None = None) -> str:
    value = getattr(run, "attrs", {}).get(KEYPOINT_PUBLICATION_OWNER_ATTR)
    if not isinstance(value, str) or _PUBLICATION_OWNER_RE.fullmatch(value) is None:
        _fail("Canonical keypoint run lacks one unguessable publication owner.")
    if expected is not None and value != expected:
        _fail("Canonical keypoint run was replaced by another publication owner.")
    return value


def _snapshot_value(
    snapshot: Mapping[str, tuple[bool, Any]],
    name: str,
) -> tuple[bool, Any]:
    value = snapshot.get(name)
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not bool
    ):
        _fail(f"Keypoint selector snapshot lacks exact {name!r} state.")
    return value


def _require_snapshot_unchanged(
    node: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    attrs = getattr(node, "attrs", {})
    for name in names:
        present, value = _snapshot_value(snapshot, name)
        if (name in attrs) is not present or (
            present and attrs.get(name) != value
        ):
            _fail(
                "Canonical keypoint activation observed concurrent mutation "
                f"of {label} {name!r}."
            )


def _publication_generation_from_snapshot(
    snapshot: Mapping[str, tuple[bool, Any]],
) -> int:
    present, value = _snapshot_value(
        snapshot,
        KEYPOINT_PUBLICATION_GENERATION_ATTR,
    )
    if not present:
        return 0
    if type(value) is not int or value < 0:
        _fail("Keypoint publication generation must be one nonnegative integer.")
    return value


def _keypoint_publication_lease_record(
    *,
    run_path: str,
    publication_owner: str,
    base_generation: int,
) -> dict[str, Any]:
    return {
        "schema_id": "palette.keypoint_publication_lease",
        "schema_version": 1,
        "policy": _KEYPOINT_PUBLICATION_POLICY,
        "run_path": run_path,
        "publication_owner": publication_owner,
        "base_generation": base_generation,
        "next_generation": base_generation + 1,
    }


def _acquire_keypoint_parent_publication_lease(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    *,
    run_path: str,
    publication_owner: str,
) -> dict[str, Any]:
    publication_attrs = (
        KEYPOINT_PUBLICATION_GENERATION_ATTR,
        KEYPOINT_PUBLICATION_POLICY_ATTR,
        KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR,
    )
    _require_snapshot_unchanged(
        parent,
        snapshot,
        publication_attrs,
        label="parent publication state",
    )
    base_generation = _publication_generation_from_snapshot(snapshot)
    policy_present, policy = _snapshot_value(
        snapshot,
        KEYPOINT_PUBLICATION_POLICY_ATTR,
    )
    lease_present, previous_lease = _snapshot_value(
        snapshot,
        KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR,
    )
    if policy_present and policy != _KEYPOINT_PUBLICATION_POLICY:
        _fail("Keypoint parent uses an unsupported publication policy.")
    if base_generation > 0 and not policy_present:
        _fail("Keypoint parent generation lacks its publication policy.")
    if lease_present:
        if (
            type(previous_lease) is not dict
            or previous_lease.get("schema_id")
            != "palette.keypoint_publication_lease"
            or previous_lease.get("schema_version") != 1
            or previous_lease.get("policy") != _KEYPOINT_PUBLICATION_POLICY
            or previous_lease.get("next_generation") != base_generation
            or previous_lease.get("base_generation") != base_generation - 1
        ):
            _fail("Keypoint parent has an active or invalid publication lease.")
    elif base_generation > 0:
        _fail("Keypoint parent generation lacks its committed publication lease.")
    lease = _keypoint_publication_lease_record(
        run_path=run_path,
        publication_owner=publication_owner,
        base_generation=base_generation,
    )
    parent.attrs[KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR] = copy.deepcopy(lease)
    if parent.attrs.get(KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR) != lease:
        _fail("Keypoint parent publication lease did not persist exactly.")
    return lease


def _require_keypoint_parent_publication_lease(
    parent: Any,
    lease: Mapping[str, Any],
    *,
    expected_generation: int,
) -> None:
    attrs = getattr(parent, "attrs", {})
    if attrs.get(KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR) != dict(lease):
        _fail("Keypoint parent publication lease was replaced.")
    generation = attrs.get(KEYPOINT_PUBLICATION_GENERATION_ATTR, 0)
    if type(generation) is not int or generation != expected_generation:
        _fail("Keypoint parent publication generation changed concurrently.")
    policy = attrs.get(KEYPOINT_PUBLICATION_POLICY_ATTR)
    if expected_generation == 0:
        if policy not in (None, _KEYPOINT_PUBLICATION_POLICY):
            _fail("Keypoint parent publication policy changed concurrently.")
    elif policy != _KEYPOINT_PUBLICATION_POLICY:
        _fail("Keypoint parent publication policy changed concurrently.")


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        value = np.asarray(node[:])
    except Exception as exc:
        _fail(f"Unable to read exact {label}: {exc}.")
    if value.dtype.hasobject:
        _fail(f"{label} cannot use object dtype.")
    return np.ascontiguousarray(value)


def _read_selected_rows(node: Any, rows: np.ndarray, *, label: str) -> np.ndarray:
    selected = np.asarray(rows)
    if selected.dtype != np.dtype("<i8") or selected.ndim != 1 or selected.size == 0:
        _fail(f"{label} requires a nonempty little-endian int64 row selection.")
    first = int(selected[0])
    contiguous = np.array_equal(
        selected,
        np.arange(first, first + selected.size, dtype="<i8"),
    )
    try:
        if contiguous:
            values = np.asarray(node[first : first + selected.size])
        elif hasattr(node, "oindex"):
            values = np.asarray(node.oindex[selected])
        else:
            values = np.stack([np.asarray(node[int(row)]) for row in selected], axis=0)
    except Exception as exc:
        _fail(f"Unable to reload exact {label}: {exc}.")
    if values.dtype.hasobject or values.shape[0] != selected.size:
        _fail(f"Reloaded {label} has invalid dtype or row cardinality.")
    return np.ascontiguousarray(values)


def _payload(node: Any, values: np.ndarray | None = None) -> dict[str, Any]:
    value = _array(node, label=canonical_node_path(node)) if values is None else values
    return {
        "array_ref": f"/{canonical_node_path(node)}",
        "array_values_sha256": array_values_sha256(node),
        "shape": [int(item) for item in value.shape],
        "dtype": value.dtype.str,
    }


def _same_identity(left: BoundRowIdentityContract, right: BoundRowIdentityContract) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.rowset_path == right.rowset_path
        and left.key_array_path == right.key_array_path
        and left.leading_dimension == right.leading_dimension
    )


def _canonical_path(value: str, *, prefix: str, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be one canonical archive-relative path.")
    path = value.strip().strip("/")
    if path != value or not path.startswith(prefix) or any(
        item in {"", ".", ".."} for item in path.split("/")
    ):
        _fail(f"{label} path {value!r} is not canonical or uses the wrong parent.")
    return path


def _node(root: Any, path: str, *, label: str) -> Any:
    try:
        result = root[path]
    except Exception as exc:
        _fail(f"Persisted {label} is unavailable at {path!r}: {exc}.")
    if canonical_node_path(result) != path:
        _fail(f"Persisted {label} resolved to an unexpected path.")
    return result


def _child(group: Any, name: str, *, label: str) -> Any:
    base = canonical_node_path(group)
    try:
        result = group[name]
    except Exception as exc:
        _fail(f"{label} is unavailable: {exc}.")
    if canonical_node_path(result) != f"{base}/{name}":
        _fail(f"{label} resolved outside its canonical rowset.")
    return result


def _exact_label_list(value: Any, *, label: str) -> list[str]:
    if type(value) is not list or not value:
        _fail(f"{label} must be one nonempty exact JSON string list.")
    result: list[str] = []
    for item in value:
        if type(item) is not str or not item or item.strip() != item:
            _fail(f"{label} contains a missing or noncanonical label.")
        result.append(item)
    if len(set(result)) != len(result):
        _fail(f"{label} must contain unique ordered labels.")
    return result


def _keypoint_label_authority_record(
    run_group: Any,
    identity: BoundRowIdentityContract,
) -> dict[str, Any]:
    """Build the exact axis-1 authority from mutually consistent live metadata."""

    labels = _exact_label_list(
        getattr(run_group, "attrs", {}).get("keypoint_labels"),
        label="keypoint_labels",
    )
    confidence_labels = _exact_label_list(
        run_group.attrs.get("keypoint_confidence_labels"),
        label="keypoint_confidence_labels",
    )
    if confidence_labels != labels:
        _fail(
            "keypoint_confidence_labels must exactly equal the ordered "
            "keypoint_labels authority."
        )
    cardinality = len(labels)
    skeleton_id = run_group.attrs.get("skeleton_id")
    if (
        type(skeleton_id) is not str
        or not skeleton_id
        or skeleton_id.strip() != skeleton_id
    ):
        _fail("skeleton_id must be one nonempty canonical string.")
    kpt_shape = run_group.attrs.get("kpt_shape")
    if (
        type(kpt_shape) is not list
        or len(kpt_shape) != 2
        or any(type(item) is not int for item in kpt_shape)
        or kpt_shape != [cardinality, 2]
    ):
        _fail("kpt_shape must exactly equal [keypoint cardinality, 2].")
    model_kpt_shape = run_group.attrs.get("model_kpt_shape")
    if model_kpt_shape is not None and (
        type(model_kpt_shape) is not list
        or len(model_kpt_shape) != 2
        or any(type(item) is not int for item in model_kpt_shape)
        or model_kpt_shape[0] != cardinality
        or model_kpt_shape[1] <= 0
    ):
        _fail("model_kpt_shape conflicts with canonical keypoint cardinality.")

    pose_schema = run_group.attrs.get("pose_schema")
    expected_pose_fields = {
        "name",
        "skeleton_id",
        "kpt_shape",
        "keypoint_labels",
        "nodes",
        "edges",
        "metadata",
        "source",
    }
    if type(pose_schema) is not dict or set(pose_schema) != expected_pose_fields:
        _fail("pose_schema must use the complete controlled future schema payload.")
    if (
        type(pose_schema.get("name")) is not str
        or not pose_schema["name"]
        or pose_schema.get("skeleton_id") != skeleton_id
        or pose_schema.get("kpt_shape") != kpt_shape
        or pose_schema.get("keypoint_labels") != labels
        or type(pose_schema.get("metadata")) is not dict
        or type(pose_schema.get("source")) is not str
        or not pose_schema["source"]
    ):
        _fail(
            "pose_schema name, skeleton_id, kpt_shape, labels, metadata, or "
            "source conflicts with the canonical keypoint axis."
        )
    metadata_skeleton_id = pose_schema["metadata"].get("skeleton_id")
    if metadata_skeleton_id is not None and metadata_skeleton_id != skeleton_id:
        _fail(
            "pose_schema metadata skeleton_id conflicts with the canonical "
            "keypoint axis."
        )
    nodes = pose_schema.get("nodes")
    if type(nodes) is not list or len(nodes) != cardinality:
        _fail("pose_schema nodes do not match keypoint cardinality.")
    for index, node in enumerate(nodes):
        if (
            type(node) is not dict
            or set(node) != {"id", "name"}
            or type(node.get("id")) is not int
            or node.get("id") != index
            or node.get("name") != labels[index]
        ):
            _fail("pose_schema nodes must exactly enumerate the ordered labels.")
    edges = pose_schema.get("edges")
    if type(edges) is not list:
        _fail("pose_schema edges must be one exact JSON list.")
    for edge in edges:
        if (
            type(edge) is not list
            or len(edge) != 2
            or any(type(item) is not int for item in edge)
            or any(item < 0 or item >= cardinality for item in edge)
        ):
            _fail("pose_schema edges must reference exact in-range keypoint IDs.")

    row_count = identity.leading_dimension
    arrays: dict[str, dict[str, Any]] = {}
    for name in (
        "keypoints_roi",
        "keypoints_img",
        "keypoints_norm",
        "keypoint_confidences",
    ):
        node = _child(run_group, name, label=name)
        shape = tuple(int(item) for item in getattr(node, "shape", ()))
        expected_shape = (
            (row_count, cardinality, 2)
            if name != "keypoint_confidences"
            else (row_count, cardinality)
        )
        if shape != expected_shape:
            _fail(
                f"{name} physical shape does not match row identity and the "
                "canonical keypoint axis."
            )
        dtype = np.dtype(getattr(node, "dtype", None))
        if dtype.kind != "f":
            _fail(f"{name} must use a floating dtype.")
        arrays[name] = {
            "array_ref": f"/{canonical_node_path(node)}",
            "shape": [int(item) for item in shape],
            "dtype": dtype.str,
            "keypoint_axis": 1,
        }

    return {
        "schema_id": KEYPOINT_LABEL_AUTHORITY_SCHEMA_ID,
        "schema_version": KEYPOINT_LABEL_AUTHORITY_SCHEMA_VERSION,
        "axis0": {
            "role": "observation_instance",
            "row_identity_ref": identity.record_ref,
            "row_identity_sha256": identity.record_sha256,
        },
        "axis1": {
            "role": "keypoint",
            "cardinality": cardinality,
            "labels": list(labels),
        },
        "coordinate_component_axis": {
            "axis": 2,
            "components": ["x", "y"],
        },
        "confidence_labels": list(confidence_labels),
        "skeleton_id": skeleton_id,
        "kpt_shape": list(kpt_shape),
        "model_kpt_shape": (
            list(model_kpt_shape) if model_kpt_shape is not None else None
        ),
        "pose_schema": copy.deepcopy(pose_schema),
        "arrays": arrays,
    }


def _require_explicit_run_status(
    group: Any,
    *,
    status: str,
    label: str,
    expected_selector_eligible: bool = True,
) -> None:
    attrs = getattr(group, "attrs", {})
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != status
        or attrs.get("stage_selector_eligible")
        is not expected_selector_eligible
    ):
        _fail(
            f"{label} must carry the exact Palette completion contract with "
            f"status={status!r} and selector eligibility "
            f"{expected_selector_eligible!r}."
        )


def _require_group(parent: Any, name: str) -> tuple[Any, bool]:
    if name in parent:
        node = parent[name]
        if hasattr(node, "shape"):
            _fail(f"Expected group {name!r}, found an array.")
        return node, False
    create = getattr(parent, "create_group", None)
    if not callable(create):
        _fail(f"Cannot create required coordinate group {name!r}.")
    return create(name), True


def _create_matrix(parent: Any, name: str, values: np.ndarray) -> tuple[Any, bool]:
    if name in parent:
        node = parent[name]
        actual = _array(node, label=name)
        if actual.dtype.str != "<f8" or not np.array_equal(actual, values):
            _fail(f"Existing preprocessing matrix {name!r} conflicts with exact policy.")
        return node, False
    create = getattr(parent, "create_array", None)
    if not callable(create):
        _fail(f"Cannot create exact preprocessing matrix {name!r}.")
    return (
        create(
            name,
            data=np.asarray(values, dtype="<f8"),
            chunks=(3, 3),
            overwrite=False,
        ),
        True,
    )


def _attrs_snapshot(*nodes: Any) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    unique: list[Any] = []
    seen: set[int] = set()
    for node in nodes:
        if id(node) in seen:
            continue
        seen.add(id(node))
        attrs = getattr(node, "attrs", None)
        if attrs is None or not hasattr(attrs, "keys"):
            _fail("Coordinate publication target lacks mutable attrs.")
        unique.append(node)
    return tuple(unique), tuple(copy.deepcopy(dict(node.attrs)) for node in unique)


def _restore_attrs(nodes: tuple[Any, ...], snapshots: tuple[dict[str, Any], ...]) -> None:
    failures: list[str] = []
    for node, snapshot in zip(nodes, snapshots, strict=True):
        try:
            attrs = node.attrs
            for name in tuple(attrs.keys()):
                del attrs[name]
            attrs.update(copy.deepcopy(snapshot))
            if dict(attrs) != snapshot:
                raise RuntimeError("restored attrs differ from the exact snapshot")
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{canonical_node_path(node)}: {exc}")
    if failures:
        raise RuntimeError(f"Coordinate attrs rollback was incomplete: {failures!r}.")


def _delete_created(root: Any, paths: list[str]) -> None:
    for path in reversed(paths):
        try:
            del root[path]
        except BaseException:
            pass
    remaining: list[str] = []
    for path in paths:
        try:
            root[path]
        except BaseException:
            continue
        remaining.append(path)
    if remaining:
        raise RuntimeError(
            f"Created coordinate evidence nodes survived rollback: {remaining!r}."
        )


def _strict_model_artifact(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Canonical keypoint inference requires one exact model-artifact mapping.")
    artifact = copy.deepcopy(value)
    if (
        artifact.get("role") != "keypoint_model"
        or artifact.get("fingerprint_scheme") != "content_v1"
        or not isinstance(artifact.get("sha256"), str)
        or _SHA256_RE.fullmatch(artifact["sha256"]) is None
        or artifact.get("mismatch") is True
        or "error" in artifact
    ):
        _fail("Canonical keypoint inference requires a successful exact content fingerprint.")
    for name in ("path", "source"):
        if not isinstance(artifact.get(name), str) or not artifact[name]:
            _fail(f"Canonical model artifact lacks exact {name!r}.")
    for name in ("size_bytes", "mtime_ns"):
        if type(artifact.get(name)) is not int or artifact[name] < 0:
            _fail(f"Canonical model artifact field {name!r} must be an exact nonnegative int.")
    try:
        artifact["pose_schema_binding"] = validate_pose_model_schema_binding(
            artifact.get("pose_schema_binding"),
            expected_model_sha256=artifact["sha256"],
        )
    except ValueError as exc:
        _fail(f"Canonical model artifact lacks exact ordered pose identity: {exc}")
    return artifact


def _model_transform_payload(value: ModelInputTransform) -> dict[str, Any]:
    if type(value) is not ModelInputTransform:
        _fail("Canonical inference requires one exact ModelInputTransform value.")
    attrs = value.to_attrs()
    expected = {
        "name",
        "native_shape_hw",
        "model_shape_hw",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "coordinate_mapping",
    }
    if set(attrs) != expected:
        _fail("Model-input transform attrs are not the exact supported schema.")
    # The typed pixel-frame/transform gates perform the remaining semantic checks.
    return copy.deepcopy(attrs)


def _model_transform_from_payload(value: Any) -> ModelInputTransform:
    if type(value) is not dict:
        _fail("Persisted model-input transform is not an exact mapping.")
    expected = {
        "name",
        "native_shape_hw",
        "model_shape_hw",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "coordinate_mapping",
    }
    if set(value) != expected or value.get("coordinate_mapping") != (
        "native_xy = model_xy - [pad_left, pad_top]"
    ):
        _fail("Persisted model-input transform fields or direction are invalid.")
    native = value["native_shape_hw"]
    model = value["model_shape_hw"]
    if (
        type(native) is not list
        or type(model) is not list
        or len(native) != 2
        or len(model) != 2
        or any(type(item) is not int for item in (*native, *model))
    ):
        _fail("Persisted model/native shapes must be exact two-integer lists.")
    pads = []
    for name in ("pad_top", "pad_bottom", "pad_left", "pad_right"):
        item = value[name]
        if type(item) is not int or item < 0:
            _fail(f"Persisted {name} must be an exact nonnegative integer.")
        pads.append(item)
    result = ModelInputTransform(
        name=value["name"],
        native_height=native[0],
        native_width=native[1],
        model_height=model[0],
        model_width=model[1],
        pad_top=pads[0],
        pad_bottom=pads[1],
        pad_left=pads[2],
        pad_right=pads[3],
    )
    if result.to_attrs() != value:
        _fail("Persisted model-input transform is not its exact supported canonical form.")
    # This validates centered padding, dimensions, and the exact matrix formula.
    model_input_to_roi_matrix(result)
    return result


@dataclass(frozen=True, init=False)
class BoundKeypointCropSource:
    crop_geometry: BoundCropObservationGeometry
    roi_geometry: CropRoiGeometryPublicationResult
    crop_placement_ownership: BoundCropPlacementOwnership = field(repr=False)
    roi_frame: BoundPixelFrameAuthority = field(repr=False)
    roi_to_source_camera: BoundDirectedTransformChain = field(repr=False)
    bbox_crop_placement_ownership: BoundCropPlacementOwnership = field(repr=False)
    bbox_roi_frame: BoundPixelFrameAuthority = field(repr=False)
    bbox_roi_to_source_camera: BoundDirectedTransformChain = field(repr=False)
    crop_path: str
    crop_profile: str
    placement_ownership_attr: str
    placement_pixel_center_ownership_attr: str
    placement_pixel_edge_ownership_attr: str
    _root: Any = field(repr=False, compare=False)
    _rowset_node: Any = field(repr=False, compare=False)
    _roi_images_node: Any = field(repr=False, compare=False)
    _placement_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        crop_geometry: BoundCropObservationGeometry,
        roi_geometry: CropRoiGeometryPublicationResult,
        crop_placement_ownership: BoundCropPlacementOwnership,
        roi_frame: BoundPixelFrameAuthority,
        roi_to_source_camera: BoundDirectedTransformChain,
        bbox_crop_placement_ownership: BoundCropPlacementOwnership,
        bbox_roi_frame: BoundPixelFrameAuthority,
        bbox_roi_to_source_camera: BoundDirectedTransformChain,
        crop_path: str,
        root: Any,
        rowset_node: Any,
        roi_images_node: Any,
        placement_node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_SOURCE_SEAL:
            _fail("Canonical keypoint crop sources cannot be constructed directly.")
        object.__setattr__(self, "crop_geometry", crop_geometry)
        object.__setattr__(self, "roi_geometry", roi_geometry)
        object.__setattr__(self, "crop_placement_ownership", crop_placement_ownership)
        object.__setattr__(self, "roi_frame", roi_frame)
        object.__setattr__(self, "roi_to_source_camera", roi_to_source_camera)
        object.__setattr__(
            self,
            "bbox_crop_placement_ownership",
            bbox_crop_placement_ownership,
        )
        object.__setattr__(self, "bbox_roi_frame", bbox_roi_frame)
        object.__setattr__(
            self,
            "bbox_roi_to_source_camera",
            bbox_roi_to_source_camera,
        )
        object.__setattr__(self, "crop_path", crop_path)
        object.__setattr__(self, "crop_profile", "materialized_canonical_v2")
        object.__setattr__(
            self,
            "placement_ownership_attr",
            CROP_PLACEMENT_OWNERSHIP_ATTR,
        )
        object.__setattr__(
            self,
            "placement_pixel_center_ownership_attr",
            CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
        )
        object.__setattr__(
            self,
            "placement_pixel_edge_ownership_attr",
            CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
        )
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_rowset_node", rowset_node)
        object.__setattr__(self, "_roi_images_node", roi_images_node)
        object.__setattr__(self, "_placement_node", placement_node)
        object.__setattr__(self, "_seal", _verification_seal)


def _load_persisted_materialized_keypoint_crop_source_fresh(
    root_node: Any,
    crop_path: str,
) -> BoundKeypointCropSource:
    """Load one exact canonical materialized crop for base keypoint inference."""

    path = _canonical_path(crop_path, prefix="crop_runs/", label="crop rowset")
    crop = require_bound_crop_observation_geometry(
        load_persisted_crop_observation_geometry(root_node, path)
    )
    rowset = _node(root_node, path, label="crop rowset")
    if rowset is not crop._rowset_node and canonical_node_path(crop._rowset_node) != path:
        _fail("Canonical crop loader returned a different rowset.")
    attrs = getattr(rowset, "attrs", {})
    _require_explicit_run_status(
        rowset,
        status=RUN_STATUS_COMPLETE,
        label="Selected canonical crop",
    )
    if attrs.get("coordinate_contract") != "canonical_v2":
        _fail("Selected crop is not explicitly canonical_v2.")
    if attrs.get("crop_storage_mode") != "materialized":
        _fail(
            "Canonical base keypoints require an explicitly published materialized "
            "crop; geometry-only and composite sources are unsupported."
        )
    roi_images = _child(rowset, "roi_images", label="authoritative crop ROI pixels")
    root_roi_images = _node(
        root_node,
        f"{path}/roi_images",
        label="root-owned authoritative crop ROI pixels",
    )
    if (
        canonical_node_path(root_roi_images) != canonical_node_path(roi_images)
        or archive_identity(root_roi_images) != archive_identity(roi_images)
    ):
        _fail("Canonical crop ROI pixels are not owned by the exact archive root.")
    placement = _child(rowset, "source_crop_xywh", label="crop placement")
    bbox_roi = _child(rowset, "bbox_roi_xyxy", label="crop ROI bbox")
    roi_shape = tuple(int(item) for item in getattr(roi_images, "shape", ()))
    try:
        roi_dtype = np.dtype(getattr(roi_images, "dtype"))
    except (AttributeError, TypeError) as exc:
        _fail(f"Canonical crop ROI pixels lack one exact NumPy dtype: {exc}.")
    if (
        len(roi_shape) != 3
        or roi_dtype != np.dtype("uint8")
        or roi_shape[0] != crop.row_identity.leading_dimension
    ):
        _fail("Canonical base keypoints require row-aligned materialized uint8 ROI pixels.")
    point_camera = crop.source_geometry.frame_evidence.source_camera_frame
    point_ownership = load_crop_placement_ownership(
        placement,
        row_identity=crop.row_identity,
        source_camera_frame=point_camera,
        attr_name=CROP_PLACEMENT_OWNERSHIP_ATTR,
    )
    point_extent = bind_array_reference_extent(roi_images, units="px")
    point_roi_frame = load_roi_pixel_frame_authority(
        roi_images,
        reference_extent=point_extent,
        crop_placement_ownership=point_ownership,
    )
    point_authority = load_bound_transform_authority(
        placement,
        payload_node=placement,
        source_frame=point_roi_frame,
        target_frame=point_camera,
        row_identity=crop.row_identity,
        attr_name=TRANSFORM_AUTHORITY_ATTR,
    )
    point_transform = load_bound_directed_transform_v2(
        placement,
        authority=point_authority,
        source_frame=point_roi_frame,
        target_frame=point_camera,
        row_identity=crop.row_identity,
        attr_name=DIRECTED_TRANSFORM_V2_ATTR,
    )
    point_chain = resolve_bound_directed_transform_chain((point_transform,))

    bbox_ownership = load_crop_placement_ownership(
        placement,
        row_identity=crop.row_identity,
        source_camera_frame=(
            crop.source_geometry.frame_evidence.bbox_source_camera_frame
        ),
        attr_name=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    )
    bbox_frame_node = _node(
        root_node,
        f"{path}/{CROP_ROI_BBOX_EDGE_FRAME_RELATIVE_PATH}",
        label="canonical crop ROI bbox-edge frame",
    )
    bbox_extent = load_crop_roi_bbox_edge_reference_extent(
        bbox_frame_node,
        roi_images,
    )
    bbox_roi_frame = load_roi_pixel_frame_authority(
        bbox_frame_node,
        reference_extent=bbox_extent,
        crop_placement_ownership=bbox_ownership,
    )
    bbox_camera = crop.source_geometry.frame_evidence.bbox_source_camera_frame
    bbox_authority = load_bound_transform_authority(
        placement,
        payload_node=placement,
        source_frame=bbox_roi_frame,
        target_frame=bbox_camera,
        row_identity=crop.row_identity,
        attr_name=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    )
    bbox_transform = load_bound_directed_transform_v2(
        placement,
        authority=bbox_authority,
        source_frame=bbox_roi_frame,
        target_frame=bbox_camera,
        row_identity=crop.row_identity,
        attr_name=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    bbox_chain = resolve_bound_directed_transform_chain((bbox_transform,))
    roi_geometry = load_crop_roi_geometry(
        placement,
        bbox_roi,
        crop_geometry=crop,
        crop_placement_ownership=bbox_ownership,
        roi_frame=bbox_roi_frame,
        roi_to_source_camera=bbox_chain,
    )
    placements = _array(placement, label="crop source placement")
    point_wh = np.asarray(
        [point_roi_frame.endpoint.width, point_roi_frame.endpoint.height]
    )
    bbox_wh = np.asarray(
        [bbox_roi_frame.endpoint.width, bbox_roi_frame.endpoint.height]
    )
    if (
        not np.array_equal(point_wh, bbox_wh)
        or placements.shape != (crop.row_identity.leading_dimension, 4)
        or not np.all(placements[:, 2:] == point_wh)
    ):
        _fail(
            "Canonical base keypoint inference currently requires one exact constant "
            "native ROI extent for every crop row."
        )
    return BoundKeypointCropSource(
        crop_geometry=crop,
        roi_geometry=roi_geometry,
        crop_placement_ownership=point_ownership,
        roi_frame=point_roi_frame,
        roi_to_source_camera=point_chain,
        bbox_crop_placement_ownership=bbox_ownership,
        bbox_roi_frame=bbox_roi_frame,
        bbox_roi_to_source_camera=bbox_chain,
        crop_path=path,
        root=root_node,
        rowset_node=rowset,
        roi_images_node=roi_images,
        placement_node=placement,
        _verification_seal=_BOUND_SOURCE_SEAL,
    )


def _load_persisted_keypoint_crop_source_fresh(
    root_node: Any,
    crop_path: str,
) -> Any:
    """Dispatch one crop rowset to exactly one full-strength profile branch."""

    path = _canonical_path(crop_path, prefix="crop_runs/", label="crop rowset")
    rowset = _node(root_node, path, label="crop rowset")
    attrs = getattr(rowset, "attrs", {})
    manifest_profile = (
        isinstance(attrs.get("run_manifest"), Mapping)
        or attrs.get("artifact_class") == "geometry_only_analysis"
    )
    attr_stamped_profile = any(
        name in attrs
        for name in (
            "crop_geometry_selection",
            "detection_acquisition_frame_mapping",
            "collection_proxy_coordinate_successor_mapping",
        )
    ) or attrs.get("crop_storage_mode") == "materialized"
    if manifest_profile and attr_stamped_profile:
        _fail(
            "Selected crop declares both sealed geometry-only and attr-stamped "
            "crop authority profiles."
        )
    if manifest_profile:
        from fisheye.shared.zarr.sealed_geometry_crop_profile import (
            load_sealed_geometry_crop_source,
        )

        return load_sealed_geometry_crop_source(root_node, path)
    return _load_persisted_materialized_keypoint_crop_source_fresh(root_node, path)


def _keypoint_crop_source_proof_key(root_node: Any, crop_path: str) -> tuple[Any, ...]:
    identity = archive_identity(root_node)
    return (
        "palette.keypoint_crop_source.v1",
        identity.kind,
        identity.key,
        _canonical_path(crop_path, prefix="crop_runs/", label="crop rowset"),
    )


def _assert_keypoint_crop_source_unchanged(
    value: Any,
) -> None:
    current = _load_persisted_keypoint_crop_source_fresh(
        value._root,
        value.crop_path,
    )
    if current.crop_profile != value.crop_profile:
        _fail("Selected crop authority profile changed after binding.")
    if current.crop_profile == "sealed_geometry_only_v2":
        if (
            current.crop_geometry.selection_derivation.record_sha256
            != value.crop_geometry.selection_derivation.record_sha256
            or current.roi_geometry.derivation.record_sha256
            != value.roi_geometry.derivation.record_sha256
            or current.roi_frame.record_sha256 != value.roi_frame.record_sha256
            or current.bbox_roi_frame.record_sha256
            != value.bbox_roi_frame.record_sha256
            or current.placement_ownership_attr != value.placement_ownership_attr
            or current.placement_pixel_center_ownership_attr
            != value.placement_pixel_center_ownership_attr
            or current.placement_pixel_edge_ownership_attr
            != value.placement_pixel_edge_ownership_attr
        ):
            _fail("Selected sealed geometry crop changed after binding.")
        return
    if (
        current.crop_geometry.selection_derivation.record_sha256
        != value.crop_geometry.selection_derivation.record_sha256
        or current.roi_geometry.derivation.record_sha256
        != value.roi_geometry.derivation.record_sha256
        or current.roi_frame.record_sha256 != value.roi_frame.record_sha256
        or current.bbox_roi_frame.record_sha256
        != value.bbox_roi_frame.record_sha256
        or tuple(
            item.record_sha256
            for item in current.roi_to_source_camera.transform_records
        )
        != tuple(
            item.record_sha256
            for item in value.roi_to_source_camera.transform_records
        )
        or tuple(
            item.record_sha256
            for item in current.bbox_roi_to_source_camera.transform_records
        )
        != tuple(
            item.record_sha256
            for item in value.bbox_roi_to_source_camera.transform_records
        )
    ):
        _fail("Selected canonical crop changed after binding.")


def load_persisted_keypoint_crop_source(
    root_node: Any,
    crop_path: str,
) -> Any:
    """Resolve one supported crop profile through the shared consumer gate.

    Materialized crops retain their pixel-bearing validation branch. Sealed
    geometry-only crops retain full manifest validation and expose coordinate
    evidence only. A rowset declaring both grammars is rejected.
    """

    path = _canonical_path(crop_path, prefix="crop_runs/", label="crop rowset")
    key = _keypoint_crop_source_proof_key(root_node, path)
    return load_verified_value(
        key,
        lambda: _load_persisted_keypoint_crop_source_fresh(root_node, path),
        _assert_keypoint_crop_source_unchanged,
    )


def require_bound_keypoint_crop_source(value: Any) -> BoundKeypointCropSource:
    if type(value) is not BoundKeypointCropSource or value._seal is not _BOUND_SOURCE_SEAL:
        _fail("A sealed persisted canonical keypoint crop source is required.")
    verify_persisted_proof(
        _keypoint_crop_source_proof_key(value._root, value.crop_path),
        lambda: _assert_keypoint_crop_source_unchanged(value),
    )
    return value


def _resolve_keypoint_crop_source(
    root_node: Any,
    crop_path: str,
    *,
    resolved_source: Any | None = None,
) -> Any:
    """Return one revalidated profile-bound crop source."""

    path = _canonical_path(crop_path, prefix="crop_runs/", label="crop rowset")
    if resolved_source is None:
        return load_persisted_keypoint_crop_source(root_node, path)
    if type(resolved_source) is BoundKeypointCropSource:
        if resolved_source.crop_path != path:
            _fail("Resolved materialized crop source names another rowset.")
        return require_bound_keypoint_crop_source(resolved_source)
    from fisheye.shared.zarr.sealed_geometry_crop_profile import (
        require_bound_sealed_geometry_crop_source,
    )

    try:
        return require_bound_sealed_geometry_crop_source(
            resolved_source,
            root=root_node,
            crop_path=path,
        )
    except Exception as exc:
        _fail(f"Resolved sealed geometry crop source is invalid: {exc}.")


def require_direct_keypoint_crop_pixel_source(
    value: Any,
    active_roi_images_node: Any,
) -> BoundKeypointCropSource:
    """Require inference to read the exact root-owned persisted ``roi_images``."""

    bound = require_bound_keypoint_crop_source(value)
    try:
        active_path = canonical_node_path(active_roi_images_node)
        active_archive = archive_identity(active_roi_images_node)
    except Exception as exc:
        _fail(f"Active keypoint ROI pixel source is not a persisted archive array: {exc}.")
    if (
        active_path != canonical_node_path(bound._roi_images_node)
        or active_archive != archive_identity(bound._roi_images_node)
    ):
        _fail(
            "Canonical keypoint inference must read the exact root-owned crop "
            "roi_images array; caches, work packages, and live/composite pixels "
            "are unsupported."
        )
    return bound


def _selected_source_values(
    source: BoundKeypointCropSource,
    source_rows_node: Any,
) -> dict[str, np.ndarray]:
    rows = _array(source_rows_node, label="source_crop_row_ids")
    if rows.dtype != np.dtype("<i8") or rows.ndim != 1 or rows.size == 0:
        _fail("Canonical source_crop_row_ids must be a nonempty little-endian int64 array.")
    if (
        int(rows.min()) < 0
        or int(rows.max()) >= source.crop_geometry.row_identity.leading_dimension
        or np.unique(rows).size != rows.size
    ):
        _fail("source_crop_row_ids is not one exact unique in-range crop selection.")
    source_group = source._rowset_node
    values = {
        "source_crop_row_ids": rows,
        "instance_key": _array(_child(source_group, "instance_key", label="crop instance_key"), label="crop instance_key")[rows],
        "source_acquisition_frame_index": _array(
            _child(source_group, "source_acquisition_frame_index", label="crop acquisition frame"),
            label="crop acquisition frame",
        )[rows],
        "source_crop_xywh": _array(source._placement_node, label="crop placement")[rows],
    }
    return {name: np.ascontiguousarray(item) for name, item in values.items()}


def _validate_output_selection(
    source: BoundKeypointCropSource,
    run_group: Any,
) -> dict[str, np.ndarray]:
    source_rows_node = _child(run_group, "source_crop_row_ids", label="keypoint crop rows")
    selected = _selected_source_values(source, source_rows_node)
    expected_dtypes = {
        "instance_key": np.dtype("<u8"),
        "source_acquisition_frame_index": np.dtype("<i8"),
    }
    for name in ("instance_key", "source_acquisition_frame_index", "source_crop_xywh"):
        output = _array(_child(run_group, name, label=f"keypoint {name}"), label=name)
        if name in expected_dtypes and output.dtype != expected_dtypes[name]:
            _fail(f"Canonical keypoint {name} must use exact dtype {expected_dtypes[name]}.")
        if output.dtype != selected[name].dtype or not np.array_equal(output, selected[name]):
            _fail(f"Keypoint {name} is not an exact dtype-preserving crop subset/reorder.")
    return selected


def _extent_record(
    *,
    role: str,
    width: int,
    height: int,
    source_frame: BoundPixelFrameAuthority,
    source_rows_node: Any,
) -> dict[str, Any]:
    return {
        "schema_id": KEYPOINT_REFERENCE_EXTENT_SCHEMA_ID,
        "schema_version": KEYPOINT_REFERENCE_EXTENT_SCHEMA_VERSION,
        "role": role,
        "operation": "exact_bound_frame_extent_copy_v1",
        "width": int(width),
        "height": int(height),
        "units": "px",
        "source_frame": {
            "record_ref": source_frame.record_ref,
            "record_sha256": source_frame.record_sha256,
        },
        "source_crop_row_ids": _payload(source_rows_node),
    }


def _stamp_extent(
    node: Any,
    *,
    attr_name: str,
    record: dict[str, Any],
) -> Any:
    attrs = node.attrs
    width = record["width"]
    height = record["height"]
    for name, expected in (("width", width), ("height", height)):
        if name in attrs and (type(attrs[name]) is not int or attrs[name] != expected):
            _fail(f"Existing {name} conflicts with the exact coordinate extent.")
        attrs[name] = expected
    stamp_and_bind_persisted_coordinate_record(node, record, attr_name=attr_name)
    return bind_persisted_record_reference_extent(
        node,
        record_attr=attr_name,
        digest_attr=f"{attr_name}_sha256",
        width_field="width",
        height_field="height",
        units_field="units",
    )


def _context_record(
    *,
    source: BoundKeypointCropSource,
    run_group: Any,
    identity: BoundRowIdentityContract,
    temporal: BoundSourceRowTemporalAuthority,
    roi_frame: BoundPixelFrameAuthority,
    roi_chain: BoundDirectedTransformChain,
    bbox_roi_frame: BoundPixelFrameAuthority,
    bbox_roi_chain: BoundDirectedTransformChain,
    point_normalized_frame: BoundPixelFrameAuthority,
    point_normalized_chain: BoundDirectedTransformChain,
    model_frame: BoundPixelFrameAuthority,
    model_link: BoundDirectedTransformV2,
    model_transform: ModelInputTransform,
    preprocessing_input_mode: str,
    model_artifact: Mapping[str, Any],
    label_authority: BoundCoordinateRecord,
) -> dict[str, Any]:
    binding = model_artifact.get("pose_schema_binding")
    bound_schema = binding.get("pose_schema") if type(binding) is dict else None
    live_schema = label_authority.record.get("pose_schema")
    if bound_schema != live_schema:
        _fail(
            "Persisted keypoint axis metadata differs from the exact "
            "model pose-schema binding."
        )
    bound_model_shape = (
        bound_schema.get("metadata", {}).get("model_kpt_shape")
        if type(bound_schema) is dict
        else None
    )
    if label_authority.record.get("model_kpt_shape") != bound_model_shape:
        _fail(
            "Persisted model_kpt_shape differs from the exact model pose-schema "
            "binding."
        )
    source_rows = _child(run_group, "source_crop_row_ids", label="keypoint crop rows")
    return {
        "schema_id": KEYPOINT_COORDINATE_CONTEXT_SCHEMA_ID,
        "schema_version": KEYPOINT_COORDINATE_CONTEXT_SCHEMA_VERSION,
        "publication_scope": "one_constant_preprocessing_transform_for_all_rows_v1",
        "source_crop_path": source.crop_path,
        "source_crop_selection": {
            "selection_record_ref": source.crop_geometry.selection_derivation.record_ref,
            "selection_record_sha256": source.crop_geometry.selection_derivation.record_sha256,
            "roi_derivation_ref": source.roi_geometry.derivation.record_ref,
            "roi_derivation_sha256": source.roi_geometry.derivation.record_sha256,
            "source_crop_row_ids": _payload(source_rows),
        },
        "output_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
            "instance_key": _payload(_child(run_group, "instance_key", label="keypoint identity")),
        },
        "output_temporal_authority": {
            "record_ref": temporal.record_ref,
            "record_sha256": temporal.record_sha256,
            "source_acquisition_frame_index": _payload(
                _child(run_group, "source_acquisition_frame_index", label="keypoint time")
            ),
        },
        "keypoint_collection_axis": {
            "axis": 1,
            "role": "keypoint",
            "label_authority_ref": label_authority.record_ref,
            "label_authority_sha256": label_authority.record_sha256,
        },
        "roi_placement": {
            "coordinate_role": "continuous_points",
            "source_crop_xywh": _payload(
                _child(run_group, "source_crop_xywh", label="keypoint placement")
            ),
            "roi_frame_ref": roi_frame.record_ref,
            "roi_frame_sha256": roi_frame.record_sha256,
            "direction": "roi_local_px_to_source_camera_image_px",
            "transform_chain": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in roi_chain.transform_records
            ],
        },
        "bbox_roi_placement": {
            "coordinate_role": "pixel_edge_half_open_bboxes",
            "source_crop_xywh": _payload(
                _child(run_group, "source_crop_xywh", label="keypoint placement")
            ),
            "roi_frame_ref": bbox_roi_frame.record_ref,
            "roi_frame_sha256": bbox_roi_frame.record_sha256,
            "direction": "roi_local_bbox_px_to_source_camera_bbox_px",
            "transform_chain": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in bbox_roi_chain.transform_records
            ],
        },
        "point_normalization": {
            "coordinate_role": "continuous_points",
            "normalized_frame_ref": point_normalized_frame.record_ref,
            "normalized_frame_sha256": point_normalized_frame.record_sha256,
            "direction": "source_camera_normalized_point_xy_to_source_camera_point_px",
            "transform_chain": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in point_normalized_chain.transform_records
            ],
        },
        "model_preprocessing": {
            "implementation": "palette.keypoint_model_input_preparation.v1",
            "submitted_input_mode": preprocessing_input_mode,
            "submitted_dtype": (
                "uint8" if preprocessing_input_mode == "numpy-list" else "float32"
            ),
            "submitted_value_range": (
                "closed_0_255"
                if preprocessing_input_mode == "numpy-list"
                else "closed_0_1"
            ),
            "submitted_normalization": (
                "ultralytics_internal_uint8_to_float_v1"
                if preprocessing_input_mode == "numpy-list"
                else "palette_uint8_divide_255_to_float32_v1"
            ),
            "submitted_input_shape_hw": [
                int(model_transform.model_height),
                int(model_transform.model_width),
            ],
            "submitted_channel_semantics": "luma_repeated_to_three_rgb_channels_v1",
            "result_coordinate_contract": (
                "ultralytics_result_xy_postprocessed_to_submitted_input_px_v1"
            ),
            "direction_used_for_input": "roi_local_px_to_detector_model_input_px",
            "persisted_transform_direction": "detector_model_input_px_to_roi_local_px",
            "policy": _model_transform_payload(model_transform),
            "model_frame_ref": model_frame.record_ref,
            "model_frame_sha256": model_frame.record_sha256,
            "inverse_transform_ref": model_link.record_ref,
            "inverse_transform_sha256": model_link.transform_sha256,
            "matrix": _payload(model_link._node),
        },
        "model_artifact": copy.deepcopy(dict(model_artifact)),
    }


@dataclass(frozen=True, init=False)
class BoundKeypointCoordinateContext:
    source: BoundKeypointCropSource
    row_identity: BoundRowIdentityContract = field(repr=False)
    temporal_authority: BoundSourceRowTemporalAuthority = field(repr=False)
    roi_frame: BoundPixelFrameAuthority = field(repr=False)
    roi_to_source_camera: BoundDirectedTransformChain = field(repr=False)
    bbox_roi_frame: BoundPixelFrameAuthority = field(repr=False)
    bbox_roi_to_source_camera: BoundDirectedTransformChain = field(repr=False)
    point_normalized_frame: BoundPixelFrameAuthority = field(repr=False)
    point_normalized_to_source_camera: BoundDirectedTransformChain = field(repr=False)
    model_input_frame: BoundPixelFrameAuthority = field(repr=False)
    model_input_to_roi: BoundDirectedTransformV2 = field(repr=False)
    model_input_transform: ModelInputTransform
    preprocessing_input_mode: str
    context_record: BoundCoordinateRecord = field(repr=False)
    keypoint_label_authority: BoundCoordinateRecord = field(repr=False)
    keypoint_labels: tuple[str, ...]
    model_artifact: Mapping[str, Any] = field(repr=False)
    run_path: str
    completion_status: str
    selector_eligible: bool
    source_crop_row_ids: np.ndarray = field(repr=False, compare=False)
    instance_key: np.ndarray = field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = field(repr=False, compare=False)
    source_crop_xywh: np.ndarray = field(repr=False, compare=False)
    _batch_nodes: tuple[Any, ...] = field(repr=False, compare=False)
    _batch_attrs: tuple[dict[str, Any], ...] = field(repr=False, compare=False)
    _model_matrix_values: np.ndarray = field(repr=False, compare=False)
    _root: Any = field(repr=False, compare=False)
    _run_group: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _BOUND_CONTEXT_SEAL:
            _fail("Keypoint coordinate contexts cannot be constructed directly.")
        for name, value in values.items():
            if name in {
                "source_crop_row_ids",
                "instance_key",
                "source_acquisition_frame_index",
                "source_crop_xywh",
                "_model_matrix_values",
            }:
                value = np.array(value, copy=True, order="C")
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


def _ensure_evidence_nodes(
    run_group: Any,
    transform: ModelInputTransform,
    point_camera: BoundPixelFrameAuthority,
    *,
    created: list[str],
) -> dict[str, Any]:
    frames, made = _require_group(run_group, "coordinate_frames")
    if made:
        created.append(canonical_node_path(frames))
    roi_node, made = _require_group(frames, "roi_local")
    if made:
        created.append(canonical_node_path(roi_node))
    bbox_roi_node, made = _require_group(frames, "roi_bbox_pixel_edge_half_open")
    if made:
        created.append(canonical_node_path(bbox_roi_node))
    point_normalized_node, made = _require_group(
        frames,
        "source_camera_normalized_points",
    )
    if made:
        created.append(canonical_node_path(point_normalized_node))
    model_node, made = _require_group(frames, "model_input")
    if made:
        created.append(canonical_node_path(model_node))
    transforms, made = _require_group(run_group, "coordinate_transforms")
    if made:
        created.append(canonical_node_path(transforms))
    matrix, made = _create_matrix(
        transforms,
        "model_input_to_roi",
        model_input_to_roi_matrix(transform),
    )
    if made:
        created.append(canonical_node_path(matrix))
    authority, made = _require_group(transforms, "model_input_to_roi_authority")
    if made:
        created.append(canonical_node_path(authority))
    point_normalized_matrix, made = _create_matrix(
        transforms,
        "source_camera_normalized_points_to_image",
        normalized_to_pixel_matrix(point_camera),
    )
    if made:
        created.append(canonical_node_path(point_normalized_matrix))
    point_normalized_authority, made = _require_group(
        transforms,
        "source_camera_normalized_points_to_image_authority",
    )
    if made:
        created.append(canonical_node_path(point_normalized_authority))
    return {
        "roi_frame_node": roi_node,
        "bbox_roi_frame_node": bbox_roi_node,
        "point_normalized_frame_node": point_normalized_node,
        "model_frame_node": model_node,
        "model_matrix_node": matrix,
        "model_authority_node": authority,
        "point_normalized_matrix_node": point_normalized_matrix,
        "point_normalized_authority_node": point_normalized_authority,
    }


def prepare_keypoint_coordinate_context(
    root_node: Any,
    run_path: str,
    *,
    crop_path: str,
    model_input_transform: ModelInputTransform,
    preprocessing_input_mode: str,
    model_artifact: Mapping[str, Any],
    _resolved_crop_source: Any | None = None,
) -> BoundKeypointCoordinateContext:
    """Persist and reload the exact transform context used by every model batch."""

    path = _canonical_path(run_path, prefix="keypoints_runs/", label="keypoint rowset")
    run_group = _node(root_node, path, label="keypoint rowset")
    _require_explicit_run_status(
        run_group,
        status=RUN_STATUS_RUNNING,
        label="Keypoint coordinate preflight target",
        expected_selector_eligible=False,
    )
    source = _resolve_keypoint_crop_source(
        root_node,
        crop_path,
        resolved_source=_resolved_crop_source,
    )
    transform = _model_transform_from_payload(_model_transform_payload(model_input_transform))
    if preprocessing_input_mode not in {"numpy-list", "tensor"}:
        _fail(
            "Canonical inference requires the exact effective submitted input mode "
            "('numpy-list' or 'tensor') before inference begins."
        )
    artifact = _strict_model_artifact(model_artifact)
    selected = _validate_output_selection(source, run_group)
    point_camera = (
        source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
    )
    bbox_camera = (
        source.crop_geometry.source_geometry.frame_evidence.bbox_source_camera_frame
    )
    if (
        point_camera.pixel_convention != SOURCE_CAMERA_POINT_PIXEL_CONVENTION
        or bbox_camera.pixel_convention != SOURCE_CAMERA_BBOX_PIXEL_CONVENTION
        or source.roi_frame.pixel_convention != SOURCE_CAMERA_POINT_PIXEL_CONVENTION
        or source.bbox_roi_frame.pixel_convention
        != SOURCE_CAMERA_BBOX_PIXEL_CONVENTION
        or source.roi_frame.endpoint.width != source.bbox_roi_frame.endpoint.width
        or source.roi_frame.endpoint.height != source.bbox_roi_frame.endpoint.height
    ):
        _fail(
            "Canonical keypoint publication requires distinct continuous point "
            "and half-open bbox crop authorities."
        )
    if transform.native_shape != (
        int(source.roi_frame.endpoint.height),
        int(source.roi_frame.endpoint.width),
    ):
        _fail("Persisted model native extent differs from the exact selected crop ROI frame.")
    created: list[str] = []
    targets: tuple[Any, ...] = ()
    snapshots: tuple[dict[str, Any], ...] = ()
    try:
        evidence = _ensure_evidence_nodes(
            run_group,
            transform,
            point_camera,
            created=created,
        )
        placement_node = _child(run_group, "source_crop_xywh", label="keypoint placement")
        key_node = _child(run_group, "instance_key", label="keypoint identity")
        time_node = _child(
            run_group,
            "source_acquisition_frame_index",
            label="keypoint acquisition time",
        )
        targets, snapshots = _attrs_snapshot(
            run_group,
            key_node,
            time_node,
            placement_node,
            evidence["roi_frame_node"],
            evidence["bbox_roi_frame_node"],
            evidence["point_normalized_frame_node"],
            evidence["model_frame_node"],
            evidence["model_matrix_node"],
            evidence["model_authority_node"],
            evidence["point_normalized_matrix_node"],
            evidence["point_normalized_authority_node"],
        )
        identity = stamp_and_bind_row_identity_contract(
            run_group,
            key_node,
            contract=build_row_identity_contract(
                domain=OBSERVATION_INSTANCE_DOMAIN,
                values=selected["instance_key"],
            ),
        )
        label_authority = stamp_and_bind_persisted_coordinate_record(
            run_group,
            _keypoint_label_authority_record(run_group, identity),
            attr_name=KEYPOINT_LABEL_AUTHORITY_ATTR,
        )
        temporal = stamp_source_row_temporal_authority(
            run_group,
            time_node,
            source_row_identity=identity,
            acquisition_frame=source.crop_geometry.source_geometry.frame_evidence.acquisition_frame,
        )
        point_placement_ownership = stamp_crop_placement_ownership(
            placement_node,
            row_identity=identity,
            source_camera_frame=point_camera,
            attr_name=source.placement_ownership_attr,
        )
        bbox_placement_ownership = stamp_crop_placement_ownership(
            placement_node,
            row_identity=identity,
            source_camera_frame=bbox_camera,
            attr_name=source.placement_pixel_edge_ownership_attr,
        )
        roi_extent_record = _extent_record(
            role="keypoint_native_roi",
            width=int(source.roi_frame.endpoint.width),
            height=int(source.roi_frame.endpoint.height),
            source_frame=source.roi_frame,
            source_rows_node=_child(run_group, "source_crop_row_ids", label="crop rows"),
        )
        roi_extent = _stamp_extent(
            evidence["roi_frame_node"],
            attr_name=KEYPOINT_ROI_REFERENCE_EXTENT_ATTR,
            record=roi_extent_record,
        )
        token = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
        roi_frame = stamp_roi_pixel_frame_authority(
            roi_extent,
            frame_id=f"keypoint_roi_{token}",
            pixel_convention=SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
            crop_placement_ownership=point_placement_ownership,
        )
        roi_authority = stamp_crop_placement_transform_authority(
            placement_node,
            authority_id=f"keypoint_roi_to_source_camera_{token}",
            source_frame=roi_frame,
            target_frame=point_camera,
            attr_name=TRANSFORM_AUTHORITY_ATTR,
        )
        roi_link = stamp_directed_transform_v2(
            placement_node,
            transform_id=f"keypoint_roi_to_source_camera_{token}",
            authority=roi_authority,
            source_frame=roi_frame,
            target_frame=point_camera,
            row_identity=identity,
            attr_name=DIRECTED_TRANSFORM_V2_ATTR,
        )
        roi_chain = resolve_bound_directed_transform_chain((roi_link,))

        bbox_roi_extent_record = _extent_record(
            role="keypoint_native_roi_bbox",
            width=int(source.bbox_roi_frame.endpoint.width),
            height=int(source.bbox_roi_frame.endpoint.height),
            source_frame=source.bbox_roi_frame,
            source_rows_node=_child(run_group, "source_crop_row_ids", label="crop rows"),
        )
        bbox_roi_extent = _stamp_extent(
            evidence["bbox_roi_frame_node"],
            attr_name=KEYPOINT_ROI_REFERENCE_EXTENT_ATTR,
            record=bbox_roi_extent_record,
        )
        bbox_roi_frame = stamp_roi_pixel_frame_authority(
            bbox_roi_extent,
            frame_id=f"keypoint_roi_bbox_{token}",
            pixel_convention=SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
            crop_placement_ownership=bbox_placement_ownership,
        )
        bbox_roi_authority = stamp_crop_placement_transform_authority(
            placement_node,
            authority_id=f"keypoint_roi_bbox_to_source_camera_{token}",
            source_frame=bbox_roi_frame,
            target_frame=bbox_camera,
            attr_name=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
        )
        bbox_roi_link = stamp_directed_transform_v2(
            placement_node,
            transform_id=f"keypoint_roi_bbox_to_source_camera_{token}",
            authority=bbox_roi_authority,
            source_frame=bbox_roi_frame,
            target_frame=bbox_camera,
            row_identity=identity,
            attr_name=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
        )
        bbox_roi_chain = resolve_bound_directed_transform_chain((bbox_roi_link,))

        point_normalized_frame = stamp_normalized_pixel_frame_authority(
            evidence["point_normalized_frame_node"],
            frame_id=f"keypoint_source_camera_normalized_points_{token}",
            pixel_frame=point_camera,
        )
        point_normalized_authority = stamp_normalized_to_pixel_transform_authority(
            evidence["point_normalized_authority_node"],
            authority_id=f"keypoint_source_camera_normalized_points_to_image_{token}",
            matrix_node=evidence["point_normalized_matrix_node"],
            source_frame=point_normalized_frame,
            target_frame=point_camera,
        )
        point_normalized_link = stamp_directed_transform_v2(
            evidence["point_normalized_matrix_node"],
            transform_id=f"keypoint_source_camera_normalized_points_to_image_{token}",
            authority=point_normalized_authority,
            source_frame=point_normalized_frame,
            target_frame=point_camera,
        )
        point_normalized_chain = resolve_bound_directed_transform_chain(
            (point_normalized_link,)
        )

        model_extent_record = _extent_record(
            role="keypoint_detector_model_input",
            width=int(transform.model_width),
            height=int(transform.model_height),
            source_frame=roi_frame,
            source_rows_node=_child(run_group, "source_crop_row_ids", label="crop rows"),
        )
        model_extent_record["model_input_transform"] = _model_transform_payload(transform)
        model_extent = _stamp_extent(
            evidence["model_frame_node"],
            attr_name=KEYPOINT_MODEL_REFERENCE_EXTENT_ATTR,
            record=model_extent_record,
        )
        model_frame = stamp_model_input_pixel_frame_authority(
            model_extent,
            frame_id=f"keypoint_model_input_{token}",
            pixel_convention="continuous",
            preprocessing_node=evidence["model_matrix_node"],
            transform=transform,
            roi_frame=roi_frame,
        )
        model_authority = stamp_model_input_transform_authority(
            evidence["model_authority_node"],
            authority_id=f"keypoint_model_input_to_roi_{token}",
            matrix_node=evidence["model_matrix_node"],
            source_frame=model_frame,
            target_frame=roi_frame,
        )
        model_link = stamp_directed_transform_v2(
            evidence["model_matrix_node"],
            transform_id=f"keypoint_model_input_to_roi_{token}",
            authority=model_authority,
            source_frame=model_frame,
            target_frame=roi_frame,
        )
        context_record = _context_record(
            source=source,
            run_group=run_group,
            identity=identity,
            temporal=temporal,
            roi_frame=roi_frame,
            roi_chain=roi_chain,
            bbox_roi_frame=bbox_roi_frame,
            bbox_roi_chain=bbox_roi_chain,
            point_normalized_frame=point_normalized_frame,
            point_normalized_chain=point_normalized_chain,
            model_frame=model_frame,
            model_link=model_link,
            model_transform=transform,
            preprocessing_input_mode=preprocessing_input_mode,
            model_artifact=artifact,
            label_authority=label_authority,
        )
        stamp_and_bind_persisted_coordinate_record(
            run_group,
            context_record,
            attr_name=KEYPOINT_COORDINATE_CONTEXT_ATTR,
        )
        return _load_persisted_keypoint_coordinate_context(
            root_node,
            path,
            require_complete=False,
            expected_selector_eligible=False,
            resolved_crop_source=_resolved_crop_source,
        )
    except BaseException as exc:
        rollback_failures: list[str] = []
        try:
            _restore_attrs(targets, snapshots)
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            rollback_failures.append(f"attrs: {rollback_exc}")
        try:
            _delete_created(root_node, created)
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            rollback_failures.append(f"nodes: {rollback_exc}")
        if rollback_failures:
            raise KeypointCoordinatePublicationError(
                "Keypoint context preparation failed and rollback was incomplete: "
                f"{rollback_failures!r}."
            ) from exc
        raise


def _load_persisted_keypoint_coordinate_context(
    root_node: Any,
    run_path: str,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
    resolved_crop_source: Any | None = None,
) -> BoundKeypointCoordinateContext:
    """Freshly reconstruct a persisted context with an explicit status policy."""

    path = _canonical_path(run_path, prefix="keypoints_runs/", label="keypoint rowset")
    run_group = _node(root_node, path, label="keypoint rowset")
    _require_explicit_run_status(
        run_group,
        status=RUN_STATUS_COMPLETE if require_complete else RUN_STATUS_RUNNING,
        label="Canonical keypoint rowset",
        expected_selector_eligible=expected_selector_eligible,
    )
    if require_complete and getattr(run_group, "attrs", {}).get(
        "coordinate_contract"
    ) != "canonical_v2":
        _fail("Complete keypoint coordinate contexts require canonical_v2 publication.")
    context = bind_persisted_coordinate_record(
        run_group,
        attr_name=KEYPOINT_COORDINATE_CONTEXT_ATTR,
    )
    raw = context.record
    source_path = raw.get("source_crop_path")
    source = _resolve_keypoint_crop_source(
        root_node,
        source_path,
        resolved_source=resolved_crop_source,
    )
    selected = _validate_output_selection(source, run_group)
    identity = load_bound_row_identity_contract(
        run_group,
        _child(run_group, "instance_key", label="keypoint identity"),
    )
    label_authority = bind_persisted_coordinate_record(
        run_group,
        attr_name=KEYPOINT_LABEL_AUTHORITY_ATTR,
    )
    expected_label_authority = _keypoint_label_authority_record(
        run_group,
        identity,
    )
    if label_authority.record != expected_label_authority:
        _fail(
            "Persisted keypoint label authority differs from live pose metadata "
            "or collection-axis shapes."
        )
    temporal = load_bound_source_row_temporal_authority(
        run_group,
        _child(run_group, "source_acquisition_frame_index", label="keypoint time"),
        source_row_identity=identity,
        acquisition_frame=source.crop_geometry.source_geometry.frame_evidence.acquisition_frame,
    )
    placement = _child(run_group, "source_crop_xywh", label="keypoint placement")
    point_camera = (
        source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
    )
    bbox_camera = (
        source.crop_geometry.source_geometry.frame_evidence.bbox_source_camera_frame
    )
    point_ownership = load_crop_placement_ownership(
        placement,
        row_identity=identity,
        source_camera_frame=point_camera,
        attr_name=source.placement_ownership_attr,
    )
    bbox_ownership = load_crop_placement_ownership(
        placement,
        row_identity=identity,
        source_camera_frame=bbox_camera,
        attr_name=source.placement_pixel_edge_ownership_attr,
    )
    roi_frame_node = _node(root_node, f"{path}/coordinate_frames/roi_local", label="keypoint ROI frame")
    roi_extent = bind_persisted_record_reference_extent(
        roi_frame_node,
        record_attr=KEYPOINT_ROI_REFERENCE_EXTENT_ATTR,
        digest_attr=f"{KEYPOINT_ROI_REFERENCE_EXTENT_ATTR}_sha256",
        width_field="width",
        height_field="height",
        units_field="units",
    )
    expected_roi_extent = _extent_record(
        role="keypoint_native_roi",
        width=int(source.roi_frame.endpoint.width),
        height=int(source.roi_frame.endpoint.height),
        source_frame=source.roi_frame,
        source_rows_node=_child(run_group, "source_crop_row_ids", label="crop rows"),
    )
    if roi_extent.authority_record != {
        **expected_roi_extent,
        "bound_record_attr": KEYPOINT_ROI_REFERENCE_EXTENT_ATTR,
        "bound_digest_attr": f"{KEYPOINT_ROI_REFERENCE_EXTENT_ATTR}_sha256",
        "bound_width_field": "width",
        "bound_height_field": "height",
        "bound_units_field": "units",
    }:
        _fail("Persisted keypoint ROI extent differs from the exact selected crop frame.")
    roi_frame = load_roi_pixel_frame_authority(
        roi_frame_node,
        reference_extent=roi_extent,
        crop_placement_ownership=point_ownership,
    )
    roi_authority = load_bound_transform_authority(
        placement,
        payload_node=placement,
        source_frame=roi_frame,
        target_frame=point_camera,
        row_identity=identity,
        attr_name=TRANSFORM_AUTHORITY_ATTR,
    )
    roi_link = load_bound_directed_transform_v2(
        placement,
        authority=roi_authority,
        source_frame=roi_frame,
        target_frame=point_camera,
        row_identity=identity,
        attr_name=DIRECTED_TRANSFORM_V2_ATTR,
    )
    roi_chain = resolve_bound_directed_transform_chain((roi_link,))

    bbox_roi_frame_node = _node(
        root_node,
        f"{path}/coordinate_frames/roi_bbox_pixel_edge_half_open",
        label="keypoint bbox ROI frame",
    )
    bbox_roi_extent = bind_persisted_record_reference_extent(
        bbox_roi_frame_node,
        record_attr=KEYPOINT_ROI_REFERENCE_EXTENT_ATTR,
        digest_attr=f"{KEYPOINT_ROI_REFERENCE_EXTENT_ATTR}_sha256",
        width_field="width",
        height_field="height",
        units_field="units",
    )
    expected_bbox_roi_extent = _extent_record(
        role="keypoint_native_roi_bbox",
        width=int(source.bbox_roi_frame.endpoint.width),
        height=int(source.bbox_roi_frame.endpoint.height),
        source_frame=source.bbox_roi_frame,
        source_rows_node=_child(run_group, "source_crop_row_ids", label="crop rows"),
    )
    if bbox_roi_extent.authority_record != {
        **expected_bbox_roi_extent,
        "bound_record_attr": KEYPOINT_ROI_REFERENCE_EXTENT_ATTR,
        "bound_digest_attr": f"{KEYPOINT_ROI_REFERENCE_EXTENT_ATTR}_sha256",
        "bound_width_field": "width",
        "bound_height_field": "height",
        "bound_units_field": "units",
    }:
        _fail("Persisted keypoint bbox ROI extent differs from the selected crop frame.")
    bbox_roi_frame = load_roi_pixel_frame_authority(
        bbox_roi_frame_node,
        reference_extent=bbox_roi_extent,
        crop_placement_ownership=bbox_ownership,
    )
    bbox_roi_authority = load_bound_transform_authority(
        placement,
        payload_node=placement,
        source_frame=bbox_roi_frame,
        target_frame=bbox_camera,
        row_identity=identity,
        attr_name=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    )
    bbox_roi_link = load_bound_directed_transform_v2(
        placement,
        authority=bbox_roi_authority,
        source_frame=bbox_roi_frame,
        target_frame=bbox_camera,
        row_identity=identity,
        attr_name=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    bbox_roi_chain = resolve_bound_directed_transform_chain((bbox_roi_link,))

    point_normalized_frame_node = _node(
        root_node,
        f"{path}/coordinate_frames/source_camera_normalized_points",
        label="keypoint normalized point frame",
    )
    point_normalized_frame = load_normalized_pixel_frame_authority(
        point_normalized_frame_node,
        pixel_frame=point_camera,
    )
    point_normalized_matrix = _node(
        root_node,
        f"{path}/coordinate_transforms/source_camera_normalized_points_to_image",
        label="normalized point-to-camera matrix",
    )
    point_normalized_authority_node = _node(
        root_node,
        f"{path}/coordinate_transforms/source_camera_normalized_points_to_image_authority",
        label="normalized point transform authority",
    )
    point_normalized_authority = load_bound_transform_authority(
        point_normalized_authority_node,
        payload_node=point_normalized_matrix,
        source_frame=point_normalized_frame,
        target_frame=point_camera,
    )
    point_normalized_link = load_bound_directed_transform_v2(
        point_normalized_matrix,
        authority=point_normalized_authority,
        source_frame=point_normalized_frame,
        target_frame=point_camera,
    )
    point_normalized_chain = resolve_bound_directed_transform_chain(
        (point_normalized_link,)
    )

    model_payload = raw.get("model_preprocessing")
    if type(model_payload) is not dict:
        _fail("Persisted keypoint context lacks exact model preprocessing.")
    transform = _model_transform_from_payload(model_payload.get("policy"))
    preprocessing_input_mode = model_payload.get("submitted_input_mode")
    if preprocessing_input_mode not in {"numpy-list", "tensor"}:
        _fail("Persisted keypoint context has an unsupported submitted input mode.")
    if transform.native_shape != (roi_frame.endpoint.height, roi_frame.endpoint.width):
        _fail("Persisted model preprocessing native extent differs from the ROI frame.")
    model_frame_node = _node(root_node, f"{path}/coordinate_frames/model_input", label="model frame")
    model_extent = bind_persisted_record_reference_extent(
        model_frame_node,
        record_attr=KEYPOINT_MODEL_REFERENCE_EXTENT_ATTR,
        digest_attr=f"{KEYPOINT_MODEL_REFERENCE_EXTENT_ATTR}_sha256",
        width_field="width",
        height_field="height",
        units_field="units",
    )
    expected_model_extent = _extent_record(
        role="keypoint_detector_model_input",
        width=int(transform.model_width),
        height=int(transform.model_height),
        source_frame=roi_frame,
        source_rows_node=_child(run_group, "source_crop_row_ids", label="crop rows"),
    )
    expected_model_extent["model_input_transform"] = _model_transform_payload(transform)
    expected_bound_model_extent = {
        **expected_model_extent,
        "bound_record_attr": KEYPOINT_MODEL_REFERENCE_EXTENT_ATTR,
        "bound_digest_attr": f"{KEYPOINT_MODEL_REFERENCE_EXTENT_ATTR}_sha256",
        "bound_width_field": "width",
        "bound_height_field": "height",
        "bound_units_field": "units",
    }
    if model_extent.authority_record != expected_bound_model_extent:
        _fail("Persisted model-input extent differs from its exact preprocessing policy.")
    matrix = _node(
        root_node,
        f"{path}/coordinate_transforms/model_input_to_roi",
        label="model-input-to-ROI matrix",
    )
    model_frame = load_model_input_pixel_frame_authority(
        model_frame_node,
        reference_extent=model_extent,
        preprocessing_node=matrix,
        transform=transform,
        roi_frame=roi_frame,
    )
    authority_node = _node(
        root_node,
        f"{path}/coordinate_transforms/model_input_to_roi_authority",
        label="model transform authority",
    )
    model_authority = load_bound_transform_authority(
        authority_node,
        payload_node=matrix,
        source_frame=model_frame,
        target_frame=roi_frame,
    )
    model_link = load_bound_directed_transform_v2(
        matrix,
        authority=model_authority,
        source_frame=model_frame,
        target_frame=roi_frame,
    )
    artifact = _strict_model_artifact(raw.get("model_artifact"))
    expected_context = _context_record(
        source=source,
        run_group=run_group,
        identity=identity,
        temporal=temporal,
        roi_frame=roi_frame,
        roi_chain=roi_chain,
        bbox_roi_frame=bbox_roi_frame,
        bbox_roi_chain=bbox_roi_chain,
        point_normalized_frame=point_normalized_frame,
        point_normalized_chain=point_normalized_chain,
        model_frame=model_frame,
        model_link=model_link,
        model_transform=transform,
        preprocessing_input_mode=preprocessing_input_mode,
        model_artifact=artifact,
        label_authority=label_authority,
    )
    if context.record != expected_context:
        _fail("Persisted keypoint context differs from exact live source/transform evidence.")
    batch_nodes = tuple(
        node
        for node in (
            run_group,
            _child(run_group, "source_crop_row_ids", label="crop rows"),
            _child(run_group, "instance_key", label="keypoint identity"),
            _child(run_group, "source_acquisition_frame_index", label="keypoint time"),
            placement,
            roi_frame_node,
            bbox_roi_frame_node,
            point_normalized_frame_node,
            model_frame_node,
            matrix,
            authority_node,
            point_normalized_matrix,
            point_normalized_authority_node,
            source._rowset_node,
            _child(source._rowset_node, "instance_key", label="crop identity"),
            _child(
                source._rowset_node,
                "source_acquisition_frame_index",
                label="crop time",
            ),
            source._placement_node,
            source._roi_images_node,
        )
        if node is not None
    )
    return BoundKeypointCoordinateContext(
        source=source,
        row_identity=identity,
        temporal_authority=temporal,
        roi_frame=roi_frame,
        roi_to_source_camera=roi_chain,
        bbox_roi_frame=bbox_roi_frame,
        bbox_roi_to_source_camera=bbox_roi_chain,
        point_normalized_frame=point_normalized_frame,
        point_normalized_to_source_camera=point_normalized_chain,
        model_input_frame=model_frame,
        model_input_to_roi=model_link,
        model_input_transform=transform,
        preprocessing_input_mode=preprocessing_input_mode,
        context_record=context,
        keypoint_label_authority=label_authority,
        keypoint_labels=tuple(
            str(item)
            for item in label_authority.record["axis1"]["labels"]
        ),
        model_artifact=artifact,
        run_path=path,
        completion_status=(
            RUN_STATUS_COMPLETE if require_complete else RUN_STATUS_RUNNING
        ),
        selector_eligible=expected_selector_eligible,
        source_crop_row_ids=selected["source_crop_row_ids"],
        instance_key=selected["instance_key"],
        source_acquisition_frame_index=selected[
            "source_acquisition_frame_index"
        ],
        source_crop_xywh=selected["source_crop_xywh"],
        _batch_nodes=batch_nodes,
        _batch_attrs=tuple(
            copy.deepcopy(dict(node.attrs))
            for node in batch_nodes
        ),
        _model_matrix_values=_array(matrix, label="model-input-to-ROI matrix"),
        _root=root_node,
        _run_group=run_group,
        _verification_seal=_BOUND_CONTEXT_SEAL,
    )


def load_persisted_keypoint_coordinate_context(
    root_node: Any,
    run_path: str,
) -> BoundKeypointCoordinateContext:
    """Load a context only from an explicitly complete canonical keypoint run."""

    return _load_persisted_keypoint_coordinate_context(
        root_node,
        run_path,
        require_complete=True,
        expected_selector_eligible=True,
    )


def require_bound_keypoint_coordinate_context(value: Any) -> BoundKeypointCoordinateContext:
    """Require a context minted by the full persisted root loader."""

    if type(value) is not BoundKeypointCoordinateContext or value._seal is not _BOUND_CONTEXT_SEAL:
        _fail("A sealed persisted keypoint coordinate context is required.")
    return value


def _revalidate_live_context_metadata(value: BoundKeypointCoordinateContext) -> None:
    if len(value._batch_nodes) != len(value._batch_attrs):
        _fail("Persisted keypoint context metadata snapshot is incomplete.")
    for node, expected in zip(value._batch_nodes, value._batch_attrs, strict=True):
        if dict(getattr(node, "attrs", {})) != expected:
            _fail(
                "Persisted keypoint coordinate metadata changed after the exact "
                "context was root-loaded."
            )
    matrix = _array(value.model_input_to_roi._node, label="model-input-to-ROI matrix")
    if matrix.dtype != value._model_matrix_values.dtype or not np.array_equal(
        matrix,
        value._model_matrix_values,
    ):
        _fail("Persisted model-input-to-ROI matrix changed after context binding.")


def revalidate_keypoint_coordinate_batch_context(
    context: BoundKeypointCoordinateContext,
    *,
    row_start: int,
    row_stop: int,
) -> BoundKeypointCoordinateContext:
    """Recheck exact identity, time, crop selection, and placement for one batch."""

    bound = require_bound_keypoint_coordinate_context(context)
    _revalidate_live_context_metadata(bound)
    if type(row_start) is not int or type(row_stop) is not int or not (
        0 <= row_start < row_stop <= bound.row_identity.leading_dimension
    ):
        _fail("Keypoint batch row slice is invalid for persisted coordinate context.")
    row_slice = slice(row_start, row_stop)
    expected = {
        "source_crop_row_ids": bound.source_crop_row_ids[row_slice],
        "instance_key": bound.instance_key[row_slice],
        "source_acquisition_frame_index": bound.source_acquisition_frame_index[
            row_slice
        ],
        "source_crop_xywh": bound.source_crop_xywh[row_slice],
    }
    for name, values in expected.items():
        node = _child(bound._run_group, name, label=f"keypoint batch {name}")
        try:
            current = np.asarray(node[row_slice])
        except Exception as exc:
            _fail(f"Unable to reload exact keypoint batch {name}: {exc}.")
        if current.dtype != values.dtype or not np.array_equal(current, values):
            _fail(
                f"Keypoint batch {name} changed after the coordinate context was bound."
            )
    source_rows = bound.source_crop_row_ids[row_slice]
    source_expected = {
        "instance_key": bound.instance_key[row_slice],
        "source_acquisition_frame_index": bound.source_acquisition_frame_index[
            row_slice
        ],
        "source_crop_xywh": bound.source_crop_xywh[row_slice],
    }
    for name, values in source_expected.items():
        node = _child(
            bound.source._rowset_node,
            name,
            label=f"selected crop batch {name}",
        )
        current = _read_selected_rows(
            node,
            source_rows,
            label=f"selected crop batch {name}",
        )
        if current.dtype != values.dtype or not np.array_equal(current, values):
            _fail(
                f"Selected persisted crop batch {name} changed after context binding."
            )
    return bound


def model_input_batch_to_roi(
    points_xy: Any,
    *,
    context: BoundKeypointCoordinateContext,
    output_dtype: Any,
) -> np.ndarray:
    """Invert model points with the exact persisted transform used by inference."""

    bound = require_bound_keypoint_coordinate_context(context)
    raw = np.asarray(points_xy)
    dtype = np.dtype(output_dtype)
    if dtype.kind != "f" or raw.ndim < 1 or raw.shape[-1] != 2:
        _fail("Model point inversion requires floating XY output semantics.")
    if not np.isfinite(raw).all():
        _fail("Model point outputs must be finite before failed rows are represented as NaN.")
    values = np.asarray(raw, dtype=np.float64)
    flat = values.reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(flat.shape[0], dtype=np.float64)))
    projected = homogeneous @ bound._model_matrix_values.T
    if np.any(projected[:, 2] == 0.0) or not np.isfinite(projected).all():
        _fail("Persisted model-input-to-ROI transform produced invalid homogeneous points.")
    result = (projected[:, :2] / projected[:, 2, None]).reshape(values.shape)
    return np.asarray(result, dtype=dtype)


def model_input_bbox_batch_to_roi(
    bbox_xyxy: Any,
    *,
    context: BoundKeypointCoordinateContext,
    output_dtype: Any,
) -> np.ndarray:
    raw = np.asarray(bbox_xyxy)
    if raw.shape[-1:] != (4,) or not np.isfinite(raw).all():
        _fail("Model bbox inversion requires finite (...,4) xyxy values.")
    points = raw.reshape(*raw.shape[:-1], 2, 2)
    result = model_input_batch_to_roi(
        points,
        context=context,
        output_dtype=output_dtype,
    )
    return result.reshape(raw.shape)


def _roi_to_image(
    values: np.ndarray,
    *,
    context: BoundKeypointCoordinateContext,
    row_slice: slice | None = None,
    bbox: bool = False,
) -> np.ndarray:
    bound = require_bound_keypoint_coordinate_context(context)
    dtype = values.dtype
    if dtype.kind != "f":
        _fail("Keypoint geometry must use a real floating dtype.")
    points = values.reshape(*values.shape[:-1], 2, 2) if bbox else values
    finite = np.isfinite(points)
    if np.isinf(points).any():
        _fail("Keypoint coordinate arrays cannot contain infinity.")
    filled = np.where(finite, points, np.zeros((), dtype=dtype))
    if row_slice is None:
        transformed = apply_bound_directed_transform_chain(
            filled,
            (
                bound.bbox_roi_to_source_camera
                if bbox
                else bound.roi_to_source_camera
            ),
            row_identity=bound.row_identity,
        )
    else:
        placements = bound.source_crop_xywh[row_slice].astype(np.float64)
        if filled.shape[0] != placements.shape[0]:
            _fail("Keypoint batch rows do not match the exact placement slice.")
        scales = placements[:, 2:] / np.asarray(
            [
                (
                    bound.bbox_roi_frame.endpoint.width
                    if bbox
                    else bound.roi_frame.endpoint.width
                ),
                (
                    bound.bbox_roi_frame.endpoint.height
                    if bbox
                    else bound.roi_frame.endpoint.height
                ),
            ],
            dtype=np.float64,
        )
        shape = (placements.shape[0],) + (1,) * (filled.ndim - 2) + (2,)
        transformed = (
            filled.astype(np.float64) * scales.reshape(shape)
            + placements[:, :2].reshape(shape)
        )
    output = np.asarray(transformed, dtype=dtype)
    output[~finite] = np.nan
    return output.reshape(values.shape) if bbox else output


def _image_to_normalized(values: np.ndarray, *, context: BoundKeypointCoordinateContext) -> np.ndarray:
    bound = require_bound_keypoint_coordinate_context(context)
    if values.ndim == 0 or values.shape[-1] not in {2, 4}:
        _fail("Image-to-normalized geometry must end in point XY or bbox XYXY.")
    frame_evidence = bound.source.crop_geometry.source_geometry.frame_evidence
    bbox = values.shape[-1] == 4
    camera = (
        frame_evidence.bbox_source_camera_frame
        if bbox
        else frame_evidence.source_camera_frame
    )
    expected_convention = "pixel_edge_half_open" if bbox else "continuous"
    if camera.pixel_convention != expected_convention:
        _fail(
            "Image-to-normalized geometry is cross-wired to a source-camera "
            f"endpoint with convention {camera.pixel_convention!r}; expected "
            f"{expected_convention!r}."
        )
    dtype = values.dtype
    factor = np.asarray([camera.endpoint.width, camera.endpoint.height], dtype=np.float64)
    if bbox:
        factor = np.tile(factor, 2)
    output = np.asarray(values.astype(np.float64) / factor, dtype=dtype)
    output[np.isnan(values)] = np.nan
    return output


def derive_keypoint_coordinate_batch(
    *,
    context: BoundKeypointCoordinateContext,
    row_start: int,
    row_stop: int,
    keypoints_roi: np.ndarray,
    pose_bbox_xyxy_roi: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive image/normalized surfaces from one exact output-row slice."""

    bound = require_bound_keypoint_coordinate_context(context)
    if type(row_start) is not int or type(row_stop) is not int or not (
        0 <= row_start < row_stop <= bound.row_identity.leading_dimension
    ):
        _fail("Keypoint batch row slice is invalid for the bound row identity.")
    expected_rows = row_stop - row_start
    keypoints = np.asarray(keypoints_roi)
    bbox = np.asarray(pose_bbox_xyxy_roi)
    if (
        keypoints.ndim != 3
        or keypoints.shape[0] != expected_rows
        or keypoints.shape[1] != len(bound.keypoint_labels)
        or keypoints.shape[-1] != 2
        or bbox.shape != (expected_rows, 4)
    ):
        _fail("Keypoint batch geometry shapes do not match the exact row slice.")
    row_slice = slice(row_start, row_stop)
    keypoints_img = _roi_to_image(keypoints, context=bound, row_slice=row_slice)
    bbox_img = _roi_to_image(bbox, context=bound, row_slice=row_slice, bbox=True)
    return {
        "keypoints_img": keypoints_img,
        "keypoints_norm": _image_to_normalized(keypoints_img, context=bound),
        "pose_bbox_xyxy_img": bbox_img,
        "pose_bbox_xyxy_norm": _image_to_normalized(bbox_img, context=bound),
    }


def _validate_geometry(context: BoundKeypointCoordinateContext) -> dict[str, np.ndarray]:
    run = context._run_group
    arrays = {name: _array(_child(run, name, label=name), label=name) for name in KEYPOINT_ARRAY_NAMES}
    n = context.row_identity.leading_dimension
    key_shape = arrays["keypoints_roi"].shape
    if (
        len(key_shape) != 3
        or key_shape[0] != n
        or key_shape[1] != len(context.keypoint_labels)
        or key_shape[-1] != 2
    ):
        _fail("Canonical keypoints must have shape (N,K,2).")
    if any(arrays[name].shape != key_shape for name in ("keypoints_img", "keypoints_norm")):
        _fail("Keypoint ROI/image/normalized surfaces have different shapes.")
    bbox_shape = (n, 4)
    if any(arrays[name].shape != bbox_shape for name in KEYPOINT_ARRAY_NAMES[3:]):
        _fail("Pose bbox ROI/image/normalized surfaces must all have shape (N,4).")
    if any(arrays[name].dtype != arrays["keypoints_roi"].dtype for name in KEYPOINT_ARRAY_NAMES[:3]):
        _fail("Keypoint coordinate derivations did not preserve dtype exactly.")
    if any(arrays[name].dtype != arrays["pose_bbox_xyxy_roi"].dtype for name in KEYPOINT_ARRAY_NAMES[3:]):
        _fail("Pose bbox coordinate derivations did not preserve dtype exactly.")
    if arrays["keypoints_roi"].dtype.kind != "f" or arrays["pose_bbox_xyxy_roi"].dtype.kind != "f":
        _fail("Canonical keypoint and bbox surfaces require floating dtypes.")
    keypoint_nan = np.isnan(arrays["keypoints_roi"])
    if np.any(keypoint_nan[..., 0] != keypoint_nan[..., 1]):
        _fail("Each canonical keypoint must be wholly finite or wholly NaN.")
    finite_keypoints = ~keypoint_nan[..., 0]
    if np.any(finite_keypoints):
        points = arrays["keypoints_roi"][finite_keypoints]
        width = float(context.roi_frame.endpoint.width)
        height = float(context.roi_frame.endpoint.height)
        if (
            np.any(points[:, 0] < 0.0)
            or np.any(points[:, 1] < 0.0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] >= height)
        ):
            _fail(
                "Finite canonical keypoints must lie inside the continuous ROI "
                "point domain [0,roi_width) x [0,roi_height)."
            )
    bbox_nan = np.isnan(arrays["pose_bbox_xyxy_roi"])
    if np.any(np.any(bbox_nan, axis=1) != np.all(bbox_nan, axis=1)):
        _fail("Each canonical pose bbox must be wholly finite or wholly NaN.")
    finite_bbox = ~np.all(bbox_nan, axis=1)
    if np.any(finite_bbox):
        boxes = arrays["pose_bbox_xyxy_roi"][finite_bbox]
        width = float(context.bbox_roi_frame.endpoint.width)
        height = float(context.bbox_roi_frame.endpoint.height)
        if (
            np.any(boxes[:, 0] < 0.0)
            or np.any(boxes[:, 1] < 0.0)
            or np.any(boxes[:, 2] > width)
            or np.any(boxes[:, 3] > height)
            or np.any(boxes[:, 2] <= boxes[:, 0])
            or np.any(boxes[:, 3] <= boxes[:, 1])
        ):
            _fail(
                "Finite canonical pose bboxes must be positive half-open edge "
                "boxes inside [0,roi_width] x [0,roi_height]."
            )
    expected_key_img = _roi_to_image(arrays["keypoints_roi"], context=context)
    expected_bbox_img = _roi_to_image(
        arrays["pose_bbox_xyxy_roi"], context=context, bbox=True
    )
    expected = {
        "keypoints_img": expected_key_img,
        "keypoints_norm": _image_to_normalized(expected_key_img, context=context),
        "pose_bbox_xyxy_img": expected_bbox_img,
        "pose_bbox_xyxy_norm": _image_to_normalized(expected_bbox_img, context=context),
    }
    for name, value in expected.items():
        if not np.array_equal(arrays[name], value, equal_nan=True):
            _fail(
                f"{name} is not the exact dtype-preserving declared ROI/image/normalization derivation."
            )
    return arrays


def _derivation_record(
    context: BoundKeypointCoordinateContext,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    run = context._run_group
    point_camera = (
        context.source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
    )
    bbox_camera = (
        context.source.crop_geometry.source_geometry.frame_evidence.bbox_source_camera_frame
    )
    bbox_normalized = (
        context.source.crop_geometry.source_geometry.frame_evidence.normalized_frame
    )
    return {
        "schema_id": KEYPOINT_COORDINATE_DERIVATION_SCHEMA_ID,
        "schema_version": KEYPOINT_COORDINATE_DERIVATION_SCHEMA_VERSION,
        "coordinate_context": {
            "record_ref": context.context_record.record_ref,
            "record_sha256": context.context_record.record_sha256,
        },
        "keypoint_label_authority": {
            "record_ref": context.keypoint_label_authority.record_ref,
            "record_sha256": context.keypoint_label_authority.record_sha256,
            "axis": 1,
            "role": "keypoint",
            "cardinality": len(context.keypoint_labels),
        },
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "temporal_authority": {
            "record_ref": context.temporal_authority.record_ref,
            "record_sha256": context.temporal_authority.record_sha256,
        },
        "operations": [
            {
                "operation": "model_input_to_roi_via_exact_persisted_transform_v1",
                "direction": "detector_model_input_px_to_roi_local_px",
                "transform_ref": context.model_input_to_roi.record_ref,
                "transform_sha256": context.model_input_to_roi.transform_sha256,
            },
            {
                "operation": "roi_point_xy_to_source_camera_via_exact_row_placement_v1",
                "direction": "roi_local_point_px_to_source_camera_point_px",
                "transform_refs": [
                    {
                        "record_ref": item.record_ref,
                        "record_sha256": item.record_sha256,
                    }
                    for item in context.roi_to_source_camera.transform_records
                ],
            },
            {
                "operation": "roi_bbox_xyxy_to_source_camera_via_exact_row_placement_v1",
                "direction": "roi_local_bbox_px_to_source_camera_bbox_px",
                "transform_refs": [
                    {
                        "record_ref": item.record_ref,
                        "record_sha256": item.record_sha256,
                    }
                    for item in context.bbox_roi_to_source_camera.transform_records
                ],
            },
            {
                "operation": "source_camera_point_to_normalized_extent_v1",
                "direction": "source_camera_point_px_to_source_camera_normalized_point_xy",
                "formula": "normalized_xy=image_xy/[reference_width_px,reference_height_px]",
                "reference_width_px": int(point_camera.endpoint.width),
                "reference_height_px": int(point_camera.endpoint.height),
                "source_camera_frame": {
                    "record_ref": point_camera.record_ref,
                    "record_sha256": point_camera.record_sha256,
                },
                "normalized_frame": {
                    "record_ref": context.point_normalized_frame.record_ref,
                    "record_sha256": context.point_normalized_frame.record_sha256,
                },
            },
            {
                "operation": "source_camera_bbox_to_normalized_extent_v1",
                "direction": "source_camera_bbox_px_to_source_camera_normalized_bbox_xy",
                "formula": "normalized_xy=image_xy/[reference_width_px,reference_height_px]",
                "reference_width_px": int(bbox_camera.endpoint.width),
                "reference_height_px": int(bbox_camera.endpoint.height),
                "source_camera_frame": {
                    "record_ref": bbox_camera.record_ref,
                    "record_sha256": bbox_camera.record_sha256,
                },
                "normalized_frame": {
                    "record_ref": bbox_normalized.record_ref,
                    "record_sha256": bbox_normalized.record_sha256,
                },
            },
        ],
        "arrays": {
            name: _payload(_child(run, name, label=name), arrays[name])
            for name in KEYPOINT_ARRAY_NAMES
        },
        "source_crop_xywh": _payload(
            _child(run, "source_crop_xywh", label="keypoint placement")
        ),
        "nan_policy": (
            "point_xy_and_bbox_tuple_all_finite_or_all_nan_with_identical_sibling_mask_v1"
        ),
        "dtype_policy": "roi_image_normalized_siblings_preserve_exact_numpy_dtype_v1",
    }


@dataclass(frozen=True, init=False)
class BoundKeypointCoordinateSurfaces:
    keypoints_roi: BoundCanonicalCoordinateDescriptor
    keypoints_img: BoundCanonicalCoordinateDescriptor
    keypoints_norm: BoundCanonicalCoordinateDescriptor
    pose_bbox_xyxy_roi: BoundCanonicalCoordinateDescriptor
    pose_bbox_xyxy_img: BoundCanonicalCoordinateDescriptor
    pose_bbox_xyxy_norm: BoundCanonicalCoordinateDescriptor
    source_crop_xywh: BoundCanonicalCoordinateDescriptor
    context: BoundKeypointCoordinateContext = field(repr=False)
    derivation: BoundCoordinateRecord = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _BOUND_SURFACES_SEAL:
            _fail("Canonical keypoint surfaces cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


@dataclass(frozen=True, init=False)
class KeypointCoordinatePublicationCheckpoint:
    run_path: str
    publication_owner: str
    _root: Any = field(repr=False, compare=False)
    _paths: tuple[str, ...] = field(repr=False, compare=False)
    _attrs: tuple[dict[str, Any], ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        run_path: str,
        publication_owner: str,
        root: Any,
        paths: tuple[str, ...],
        attrs: tuple[dict[str, Any], ...],
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _PUBLICATION_CHECKPOINT_SEAL:
            _fail("Keypoint publication checkpoints cannot be constructed directly.")
        object.__setattr__(self, "run_path", run_path)
        object.__setattr__(self, "publication_owner", publication_owner)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_paths", paths)
        object.__setattr__(self, "_attrs", attrs)
        object.__setattr__(self, "_seal", _verification_seal)


def capture_keypoint_coordinate_publication_checkpoint(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str | None = None,
) -> KeypointCoordinatePublicationCheckpoint:
    """Capture every attrs surface that canonical publication may mutate."""

    path = _canonical_path(run_path, prefix="keypoints_runs/", label="keypoint rowset")
    run = _node(root_node, path, label="keypoint rowset")
    publication_owner = _publication_owner(
        run,
        expected=expected_publication_owner,
    )
    _require_explicit_run_status(
        run,
        status=RUN_STATUS_RUNNING,
        label="Keypoint publication checkpoint target",
        expected_selector_eligible=False,
    )
    placement = _child(run, "source_crop_xywh", label="keypoint placement")
    geometry = tuple(_child(run, name, label=name) for name in KEYPOINT_ARRAY_NAMES)
    nodes, attrs = _attrs_snapshot(run, placement, *geometry)
    return KeypointCoordinatePublicationCheckpoint(
        run_path=path,
        publication_owner=publication_owner,
        root=root_node,
        paths=tuple(canonical_node_path(node) for node in nodes),
        attrs=attrs,
        _verification_seal=_PUBLICATION_CHECKPOINT_SEAL,
    )


def rollback_keypoint_coordinate_publication(
    checkpoint: KeypointCoordinatePublicationCheckpoint,
) -> None:
    """Restore a pre-publication attrs graph completely or fail loudly."""

    if (
        type(checkpoint) is not KeypointCoordinatePublicationCheckpoint
        or checkpoint._seal is not _PUBLICATION_CHECKPOINT_SEAL
    ):
        _fail("A sealed keypoint publication checkpoint is required for rollback.")
    try:
        run = _node(
            checkpoint._root,
            checkpoint.run_path,
            label="keypoint rollback rowset",
        )
    except KeypointCoordinatePublicationError:
        return
    if run.attrs.get(KEYPOINT_PUBLICATION_OWNER_ATTR) != checkpoint.publication_owner:
        return
    nodes = tuple(
        _node(checkpoint._root, path, label="keypoint rollback target")
        for path in checkpoint._paths
    )
    if any(
        not canonical_node_path(node).startswith(f"{checkpoint.run_path}/")
        and canonical_node_path(node) != checkpoint.run_path
        for node in nodes
    ):
        _fail("Keypoint rollback target escaped its attempt-owned child.")
    _restore_attrs(nodes, checkpoint._attrs)


def _bindings(
    context: BoundKeypointCoordinateContext,
    derivation: BoundCoordinateRecord,
    *,
    load: bool,
) -> dict[str, BoundCanonicalCoordinateDescriptor]:
    run = context._run_group
    base_lineage = (context.context_record, derivation)
    keypoint_lineage = (
        context.keypoint_label_authority,
        context.context_record,
        derivation,
    )
    collection = CanonicalCollectionAxis(
        axis=1,
        role="keypoint",
        cardinality=len(context.keypoint_labels),
        label_authority=DigestBoundCoordinateRecordRef(
            record_ref=context.keypoint_label_authority.record_ref,
            record_sha256=context.keypoint_label_authority.record_sha256,
        ),
    )
    point_camera = (
        context.source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
    )
    bbox_camera = (
        context.source.crop_geometry.source_geometry.frame_evidence.bbox_source_camera_frame
    )
    bbox_normalized = (
        context.source.crop_geometry.source_geometry.frame_evidence.normalized_frame
    )
    bbox_normalized_chain = (
        context.source.crop_geometry.source_geometry.frame_evidence.normalized_to_source_camera
    )
    specs: dict[str, dict[str, Any]] = {
        "source_crop_xywh": {
            **SOURCE_CAMERA_CROP_XYWH.descriptor_kwargs(),
            "reference_frame_authority": bbox_camera,
        },
        "keypoints_roi": {
            **ROI_POINT_XY.descriptor_kwargs(),
            "reference_frame_authority": context.roi_frame,
            "transform_chain": context.roi_to_source_camera,
            "collection_axis": True,
        },
        "keypoints_img": {
            **SOURCE_CAMERA_POINT_XY.descriptor_kwargs(),
            "reference_frame_authority": point_camera,
            "collection_axis": True,
        },
        "keypoints_norm": {
            **SOURCE_CAMERA_NORMALIZED_POINT_XY.descriptor_kwargs(),
            "reference_frame_authority": context.point_normalized_frame,
            "transform_chain": context.point_normalized_to_source_camera,
            "collection_axis": True,
        },
        "pose_bbox_xyxy_roi": {
            **ROI_BBOX_XYXY.descriptor_kwargs(),
            "reference_frame_authority": context.bbox_roi_frame,
            "transform_chain": context.bbox_roi_to_source_camera,
        },
        "pose_bbox_xyxy_img": {
            **SOURCE_CAMERA_BBOX_XYXY.descriptor_kwargs(),
            "reference_frame_authority": bbox_camera,
        },
        "pose_bbox_xyxy_norm": {
            **SOURCE_CAMERA_NORMALIZED_BBOX_XYXY.descriptor_kwargs(),
            "reference_frame_authority": bbox_normalized,
            "transform_chain": bbox_normalized_chain,
        },
    }
    result: dict[str, BoundCanonicalCoordinateDescriptor] = {}
    for name, spec in specs.items():
        node = _child(run, name, label=name)
        lineage = keypoint_lineage if spec.get("collection_axis") else base_lineage
        evidence_kwargs = {
            "row_identity": context.row_identity,
            "lineage_records": lineage,
        }
        if "reference_frame_authority" in spec:
            evidence_kwargs["reference_frame_authority"] = spec[
                "reference_frame_authority"
            ]
        if "transform_chain" in spec:
            evidence_kwargs["transform_chain"] = spec["transform_chain"]
        result[name] = (
            load_bound_canonical_coordinate_descriptor(node, **evidence_kwargs)
            if load
            else build_bound_canonical_coordinate_descriptor(
                node,
                **{
                    key: value
                    for key, value in spec.items()
                    if key not in {"collection_axis"}
                },
                row_identity=context.row_identity,
                lineage_records=lineage,
                collection_axis=(
                    collection if spec.get("collection_axis") else None
                ),
            )
        )
        descriptor = result[name].descriptor
        if (
            descriptor.profile_id != spec["profile_id"]
            or descriptor.geometry_type != spec["geometry_type"]
            or descriptor.components != tuple(spec["components"])
            or descriptor.component_units != tuple(spec["component_units"])
            or descriptor.pixel_convention != spec["pixel_convention"]
            or descriptor.source_camera_overlay.status
            != spec["source_camera_overlay_status"]
            or descriptor.collection_axis
            != (collection if spec.get("collection_axis") else None)
        ):
            _fail(
                f"Persisted {name} descriptor differs from the controlled "
                "keypoint coordinate and collection-axis contract."
            )
    return result


def publish_keypoint_coordinate_surfaces(
    root_node: Any,
    run_path: str,
    *,
    _resolved_crop_source: Any | None = None,
) -> BoundKeypointCoordinateSurfaces:
    """Transactionally publish and freshly load all canonical keypoint surfaces."""

    context = _load_persisted_keypoint_coordinate_context(
        root_node,
        run_path,
        require_complete=False,
        expected_selector_eligible=False,
        resolved_crop_source=_resolved_crop_source,
    )
    arrays = _validate_geometry(context)
    run = context._run_group
    geometry_nodes = [_child(run, name, label=name) for name in KEYPOINT_ARRAY_NAMES]
    placement = _child(run, "source_crop_xywh", label="keypoint placement")
    targets, snapshots = _attrs_snapshot(run, placement, *geometry_nodes)
    try:
        record = _derivation_record(context, arrays)
        derivation = stamp_and_bind_persisted_coordinate_record(
            run,
            record,
            attr_name=KEYPOINT_COORDINATE_DERIVATION_ATTR,
        )
        bindings = _bindings(context, derivation, load=False)
        stamp_bound_canonical_coordinate_descriptors(bindings.values())
        run.attrs["coordinate_contract"] = "canonical_v2"
        # The loader starts again from the root and revalidates every record and array.
        return _load_persisted_keypoint_coordinate_surfaces(
            root_node,
            context.run_path,
            require_complete=False,
            expected_selector_eligible=False,
            resolved_crop_source=_resolved_crop_source,
        )
    except BaseException as exc:
        try:
            _restore_attrs(targets, snapshots)
        except BaseException as rollback_exc:  # pragma: no cover
            raise KeypointCoordinatePublicationError(
                f"Keypoint surface publication failed and rollback was incomplete: {rollback_exc}."
            ) from exc
        raise


def _load_persisted_sealed_crop_successor_binding(
    root_node: Any,
    run: Any,
    *,
    run_path: str,
) -> Any:
    """Rebind one sealed crop source from exact persisted successor evidence."""

    from pathlib import Path

    from fisheye.shared.model_input_transform import (
        model_input_transform_from_attrs,
    )
    from fisheye.shared.zarr.coordinate_successor_authority import (
        KEYPOINT_COORDINATE_SUCCESSOR_KIND,
        load_coordinate_successor_authority,
    )
    from fisheye.shared.zarr.sealed_geometry_crop_profile import (
        SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR,
        bind_sealed_geometry_crop_successor_source,
        load_sealed_geometry_bbox_normalization_from_successor,
    )
    from fisheye.shared.zarr.keypoint_manifest import (
        KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
        keypoint_preprocessing_from_manifest,
        validate_keypoint_run_manifest,
    )
    from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

    try:
        authority = load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run_path,
        )
        padded = bind_persisted_coordinate_record(
            run,
            attr_name="coordinate_successor_padded_crop_lineage",
        )
        crop_evidence = bind_persisted_coordinate_record(
            run,
            attr_name="coordinate_successor_historical_crop_adapter",
        )
        bbox_normalization = bind_persisted_coordinate_record(
            run,
            attr_name=SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR,
        )
    except Exception as exc:
        _fail(f"Sealed keypoint crop successor authority is invalid: {exc}.")

    authority_payload = authority["payload"]
    expected_padded = {
        "record_ref": padded.record_ref,
        "record_sha256": padded.record_sha256,
    }
    if authority_payload["coordinate_records"].get(
        "padded_crop_lineage"
    ) != expected_padded:
        _fail("Sealed keypoint crop successor padded lineage is stale.")
    expected_bbox_normalization = {
        "record_ref": bbox_normalization.record_ref,
        "record_sha256": bbox_normalization.record_sha256,
    }
    if authority_payload["coordinate_records"].get(
        "historical_bbox_normalization"
    ) != expected_bbox_normalization:
        _fail("Sealed keypoint bbox-normalization authority is stale.")
    crop_evidence_record = padded.record.get("source_crop_adapter")
    if (
        not isinstance(crop_evidence_record, Mapping)
        or crop_evidence.record != crop_evidence_record
    ):
        _fail(
            "Sealed keypoint crop successor evidence differs from its "
            "authority-bound padded lineage."
        )

    source_authority = authority_payload["source"]
    source_run_path = source_authority.get("run_path")
    if (
        source_authority.get("family") != "keypoints_runs"
        or type(source_run_path) is not str
        or source_run_path != crop_evidence_record.get("source_run_path")
    ):
        _fail("Sealed keypoint crop successor source path is inconsistent.")
    source_run = _node(
        root_node,
        source_run_path,
        label="sealed keypoint source core",
    )
    source_manifest = source_run.attrs.get(KEYPOINT_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(source_manifest, Mapping):
        _fail("Sealed keypoint source core lacks its run manifest.")
    source_errors = validate_keypoint_run_manifest(source_manifest)
    if source_errors:
        _fail(
            "Sealed keypoint source core manifest is invalid: "
            + "; ".join(source_errors)
        )
    source_payload = source_manifest["payload"]
    source_logical = source_payload.get("logical_content")
    if (
        source_authority.get("manifest_payload_digest")
        != source_manifest.get("payload_digest")
        or source_authority.get("manifest_document_digest")
        != canonical_json_sha256(source_manifest)
        or not isinstance(source_logical, Mapping)
        or source_authority.get("logical_content_digest")
        != source_logical.get("digest")
    ):
        _fail("Sealed keypoint source manifest changed after publication.")

    try:
        preprocessing = keypoint_preprocessing_from_manifest(
            source_payload["preprocessing"]
        )
        transform = model_input_transform_from_attrs(
            dict(preprocessing.document["model_input_transform"])
        )
        identity = archive_identity(root_node)
        if identity.kind != "local_store_root":
            _fail(
                "Persisted sealed keypoint crop successors require a stable "
                "local archive identity."
            )
        binding = bind_sealed_geometry_crop_successor_source(
            analysis_zarr=Path(identity.key[0]),
            root=root_node,
            crop_reference=source_payload["source_crop_snapshot"],
            source_manifest=source_manifest,
            source_arrays={
                name: _child(
                    source_run,
                    name,
                    label=f"sealed keypoint source {name}",
                )
                for name in (
                    "source_crop_row_ids",
                    "instance_key",
                    "source_acquisition_frame_index",
                    "source_crop_row_signature",
                )
            },
            source_run_path=source_run_path,
            model_input_transform=transform,
        )
        binding = load_sealed_geometry_bbox_normalization_from_successor(
            binding,
            root=root_node,
            successor_run=run,
            successor_run_path=run_path,
        )
    except KeypointCoordinatePublicationError:
        raise
    except Exception as exc:
        _fail(f"Persisted sealed keypoint crop cannot be rebound: {exc}.")
    if binding.as_record() != crop_evidence_record:
        _fail("Persisted sealed keypoint crop successor evidence changed.")
    return binding


def _load_persisted_keypoint_coordinate_surfaces_impl(
    root_node: Any,
    run_path: str,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
    resolved_crop_source: Any | None = None,
) -> BoundKeypointCoordinateSurfaces:
    """Freshly verify a graph under an explicit completion-state policy."""

    context = _load_persisted_keypoint_coordinate_context(
        root_node,
        run_path,
        require_complete=require_complete,
        expected_selector_eligible=expected_selector_eligible,
        resolved_crop_source=resolved_crop_source,
    )
    run = context._run_group
    if getattr(run, "attrs", {}).get("coordinate_contract") != "canonical_v2":
        _fail("Selected keypoint rowset is not explicitly canonical_v2.")
    arrays = _validate_geometry(context)
    derivation = bind_persisted_coordinate_record(
        run,
        attr_name=KEYPOINT_COORDINATE_DERIVATION_ATTR,
    )
    expected = _derivation_record(context, arrays)
    if derivation.record != expected:
        _fail("Persisted keypoint derivation differs from exact live arrays or lineage.")
    bindings = _bindings(context, derivation, load=True)
    return BoundKeypointCoordinateSurfaces(
        keypoints_roi=bindings["keypoints_roi"],
        keypoints_img=bindings["keypoints_img"],
        keypoints_norm=bindings["keypoints_norm"],
        pose_bbox_xyxy_roi=bindings["pose_bbox_xyxy_roi"],
        pose_bbox_xyxy_img=bindings["pose_bbox_xyxy_img"],
        pose_bbox_xyxy_norm=bindings["pose_bbox_xyxy_norm"],
        source_crop_xywh=bindings["source_crop_xywh"],
        context=context,
        derivation=derivation,
        _verification_seal=_BOUND_SURFACES_SEAL,
    )


def _load_persisted_keypoint_coordinate_surfaces(
    root_node: Any,
    run_path: str,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
    resolved_crop_source: Any | None = None,
) -> BoundKeypointCoordinateSurfaces:
    """Verify one graph through the shared profile-aware crop resolver."""

    path = _canonical_path(
        run_path,
        prefix="keypoints_runs/",
        label="keypoint rowset",
    )
    run = _node(root_node, path, label="keypoint rowset")
    crop_evidence_attr = "coordinate_successor_historical_crop_adapter"
    if require_complete and crop_evidence_attr in getattr(run, "attrs", {}):
        # The attr name is frozen persisted v1 evidence. Validate it, but do
        # not install or invoke a loader replacement; the ordinary resolver
        # below opens the sealed crop profile directly.
        binding = _load_persisted_sealed_crop_successor_binding(
            root_node,
            run,
            run_path=path,
        )
        if resolved_crop_source is not None:
            _fail("Complete successor loading does not accept a crop-source override.")
        resolved_crop_source = binding.source
    return _load_persisted_keypoint_coordinate_surfaces_impl(
        root_node,
        path,
        require_complete=require_complete,
        expected_selector_eligible=expected_selector_eligible,
        resolved_crop_source=resolved_crop_source,
    )


def load_persisted_keypoint_coordinate_surfaces(
    root_node: Any,
    run_path: str,
) -> BoundKeypointCoordinateSurfaces:
    """Load canonical surfaces only from an explicitly complete keypoint run."""

    return _load_persisted_keypoint_coordinate_surfaces(
        root_node,
        run_path,
        require_complete=True,
        expected_selector_eligible=True,
    )


def load_persisted_ineligible_keypoint_coordinate_surfaces(
    root_node: Any,
    run_path: str,
) -> BoundKeypointCoordinateSurfaces:
    """Load complete keypoint surfaces that remain selector-ineligible.

    This verifies the coordinate publication only.  Consumers must separately
    prove an exact immutable bundle authority before treating the named member
    as an analysis input.
    """

    return _load_persisted_keypoint_coordinate_surfaces(
        root_node,
        run_path,
        require_complete=True,
        expected_selector_eligible=False,
    )


def _load_completed_ineligible_keypoint_coordinate_surfaces(
    root_node: Any,
    run_path: str,
) -> BoundKeypointCoordinateSurfaces:
    """Producer-only fresh validation immediately before selector activation."""

    return _load_persisted_keypoint_coordinate_surfaces(
        root_node,
        run_path,
        require_complete=True,
        expected_selector_eligible=False,
    )


def _activate_validated_keypoint_coordinate_surfaces(
    root_node: Any,
    run_parent: Any,
    value: BoundKeypointCoordinateSurfaces,
    *,
    run_name: str,
    publication_owner_token: str,
    parent_selector_snapshot: Mapping[str, tuple[bool, Any]],
    root_pointer_snapshot: Mapping[str, tuple[bool, Any]],
) -> None:
    """Advance owned selectors under one generation lease, eligibility last."""

    if type(value) is not BoundKeypointCoordinateSurfaces or value._seal is not _BOUND_SURFACES_SEAL:
        _fail("Keypoint activation requires sealed coordinate surfaces.")
    context = value.context
    expected_path = f"keypoints_runs/{run_name}"
    _publication_owner(
        context._run_group,
        expected=publication_owner_token,
    )
    if (
        context.completion_status != RUN_STATUS_COMPLETE
        or context.selector_eligible is not False
        or context.run_path != expected_path
        or canonical_node_path(run_parent) != "keypoints_runs"
        or canonical_node_path(context._run_group) != expected_path
        or archive_identity(root_node) != archive_identity(run_parent)
        or archive_identity(root_node) != archive_identity(context._run_group)
    ):
        _fail(
            "Keypoint activation requires the exact complete, ineligible, "
            "freshly validated canonical child."
        )

    def fresh_parent() -> Any:
        parent = _node(root_node, "keypoints_runs", label="keypoint parent")
        if archive_identity(parent) != archive_identity(run_parent):
            _fail("Keypoint parent changed archives during activation.")
        return parent

    active_parent = fresh_parent()
    _require_snapshot_unchanged(
        active_parent,
        parent_selector_snapshot,
        _KEYPOINT_GUARDED_PARENT_SELECTORS,
        label="parent selector",
    )
    _require_snapshot_unchanged(
        root_node,
        root_pointer_snapshot,
        _KEYPOINT_GUARDED_ROOT_SELECTORS,
        label="root selector",
    )
    if active_parent.attrs.get("latest_pending") != str(run_name):
        _fail(
            "Canonical keypoint activation requires the attempt-owned "
            "latest_pending selector."
        )
    current = _load_completed_ineligible_keypoint_coordinate_surfaces(
        root_node,
        expected_path,
    )
    _publication_owner(
        current.context._run_group,
        expected=publication_owner_token,
    )
    if current.derivation.record_sha256 != value.derivation.record_sha256:
        _fail("Keypoint coordinate publication changed before activation.")

    active_parent = fresh_parent()
    _require_snapshot_unchanged(
        active_parent,
        parent_selector_snapshot,
        _KEYPOINT_GUARDED_PARENT_SELECTORS,
        label="parent selector",
    )
    _require_snapshot_unchanged(
        root_node,
        root_pointer_snapshot,
        _KEYPOINT_GUARDED_ROOT_SELECTORS,
        label="root selector",
    )
    if active_parent.attrs.get("latest_pending") != str(run_name):
        _fail("Keypoint pending selector changed before activation.")

    # Close the completed-child proof phase before the publication lease or
    # any selector is mutated. This makes reuse a validation optimization,
    # never an authorization to publish stale evidence.
    finish_proof_verification()
    lease = _acquire_keypoint_parent_publication_lease(
        active_parent,
        parent_selector_snapshot,
        run_path=expected_path,
        publication_owner=publication_owner_token,
    )
    base_generation = int(lease["base_generation"])
    next_generation = int(lease["next_generation"])

    active_parent = fresh_parent()
    _require_keypoint_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    _require_snapshot_unchanged(
        active_parent,
        parent_selector_snapshot,
        _KEYPOINT_GUARDED_PARENT_SELECTORS,
        label="parent selector",
    )
    _require_snapshot_unchanged(
        root_node,
        root_pointer_snapshot,
        _KEYPOINT_GUARDED_ROOT_SELECTORS,
        label="root selector",
    )
    if active_parent.attrs.get("latest_pending") != str(run_name):
        _fail("Keypoint pending selector changed after lease acquisition.")

    active_parent.attrs["latest_complete"] = str(run_name)
    active_parent = fresh_parent()
    _require_keypoint_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    _require_snapshot_unchanged(
        active_parent,
        parent_selector_snapshot,
        ("latest",),
        label="parent selector",
    )
    _require_snapshot_unchanged(
        root_node,
        root_pointer_snapshot,
        _KEYPOINT_GUARDED_ROOT_SELECTORS,
        label="root selector",
    )
    if (
        active_parent.attrs.get("latest_complete") != str(run_name)
        or active_parent.attrs.get("latest_pending") != str(run_name)
    ):
        _fail("Keypoint latest_complete or pending selector changed in activation.")

    active_parent.attrs["latest"] = str(run_name)
    active_parent = fresh_parent()
    _require_keypoint_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    _require_snapshot_unchanged(
        root_node,
        root_pointer_snapshot,
        _KEYPOINT_GUARDED_ROOT_SELECTORS,
        label="root selector",
    )
    if (
        active_parent.attrs.get("latest_complete") != str(run_name)
        or active_parent.attrs.get("latest") != str(run_name)
        or active_parent.attrs.get("latest_pending") != str(run_name)
    ):
        _fail("Keypoint parent selectors changed during activation.")

    root_node.attrs["current_keypoint_group_path"] = expected_path
    active_parent = fresh_parent()
    _require_keypoint_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    if (
        root_node.attrs.get("current_keypoint_group_path") != expected_path
        or active_parent.attrs.get("latest_complete") != str(run_name)
        or active_parent.attrs.get("latest") != str(run_name)
        or active_parent.attrs.get("latest_pending") != str(run_name)
    ):
        _fail("Keypoint root or parent selectors changed during activation.")

    del active_parent.attrs["latest_pending"]
    active_parent = fresh_parent()
    _require_keypoint_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    activation_run = _node(root_node, expected_path, label="keypoint child")
    _publication_owner(activation_run, expected=publication_owner_token)
    if (
        root_node.attrs.get("current_keypoint_group_path") != expected_path
        or active_parent.attrs.get("latest_complete") != str(run_name)
        or active_parent.attrs.get("latest") != str(run_name)
        or "latest_pending" in active_parent.attrs
        or activation_run.attrs.get("stage_selector_eligible") is not False
    ):
        _fail("Canonical keypoint selectors did not persist before activation.")

    active_parent.attrs[KEYPOINT_PUBLICATION_POLICY_ATTR] = (
        _KEYPOINT_PUBLICATION_POLICY
    )
    active_parent = fresh_parent()
    _require_keypoint_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    if (
        active_parent.attrs.get(KEYPOINT_PUBLICATION_POLICY_ATTR)
        != _KEYPOINT_PUBLICATION_POLICY
    ):
        _fail("Keypoint publication policy did not persist exactly.")
    active_parent.attrs[KEYPOINT_PUBLICATION_GENERATION_ATTR] = next_generation
    active_parent = fresh_parent()
    _require_keypoint_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=next_generation,
    )
    activation_run = _node(root_node, expected_path, label="keypoint child")
    _publication_owner(activation_run, expected=publication_owner_token)
    if (
        root_node.attrs.get("current_keypoint_group_path") != expected_path
        or active_parent.attrs.get("latest_complete") != str(run_name)
        or active_parent.attrs.get("latest") != str(run_name)
        or "latest_pending" in active_parent.attrs
        or activation_run.attrs.get("stage_selector_eligible") is not False
    ):
        _fail("Keypoint publication epoch did not persist before eligibility.")
    activation_run.attrs["stage_selector_eligible"] = True


def require_bound_keypoint_coordinate_surfaces(value: Any) -> BoundKeypointCoordinateSurfaces:
    if type(value) is not BoundKeypointCoordinateSurfaces or value._seal is not _BOUND_SURFACES_SEAL:
        _fail("A sealed persisted canonical keypoint surface graph is required.")
    current = load_persisted_keypoint_coordinate_surfaces(value.context._root, value.context.run_path)
    if current.derivation.record_sha256 != value.derivation.record_sha256:
        _fail("Canonical keypoint coordinate surfaces changed after binding.")
    return value


def require_bound_ineligible_keypoint_coordinate_surfaces(
    value: Any,
) -> BoundKeypointCoordinateSurfaces:
    """Revalidate one sealed coordinate graph that remains selector-ineligible."""

    if (
        type(value) is not BoundKeypointCoordinateSurfaces
        or value._seal is not _BOUND_SURFACES_SEAL
    ):
        _fail("A sealed ineligible keypoint coordinate graph is required.")
    current = load_persisted_ineligible_keypoint_coordinate_surfaces(
        value.context._root,
        value.context.run_path,
    )
    if current.derivation.record_sha256 != value.derivation.record_sha256:
        _fail("Ineligible keypoint coordinate surfaces changed after binding.")
    return value


__all__ = [
    "KEYPOINT_ARRAY_NAMES",
    "KEYPOINT_COORDINATE_CONTEXT_ATTR",
    "KEYPOINT_COORDINATE_DERIVATION_ATTR",
    "KEYPOINT_LABEL_AUTHORITY_ATTR",
    "KEYPOINT_PARENT_PUBLICATION_LEASE_ATTR",
    "KEYPOINT_PUBLICATION_GENERATION_ATTR",
    "KEYPOINT_PUBLICATION_OWNER_ATTR",
    "KEYPOINT_PUBLICATION_POLICY_ATTR",
    "BoundKeypointCoordinateContext",
    "BoundKeypointCoordinateSurfaces",
    "BoundKeypointCropSource",
    "KeypointCoordinatePublicationCheckpoint",
    "KeypointCoordinatePublicationError",
    "capture_keypoint_coordinate_publication_checkpoint",
    "derive_keypoint_coordinate_batch",
    "load_persisted_keypoint_coordinate_context",
    "load_persisted_keypoint_coordinate_surfaces",
    "load_persisted_ineligible_keypoint_coordinate_surfaces",
    "load_persisted_keypoint_crop_source",
    "model_input_batch_to_roi",
    "model_input_bbox_batch_to_roi",
    "prepare_keypoint_coordinate_context",
    "publish_keypoint_coordinate_surfaces",
    "revalidate_keypoint_coordinate_batch_context",
    "require_bound_keypoint_coordinate_context",
    "require_bound_keypoint_coordinate_surfaces",
    "require_bound_ineligible_keypoint_coordinate_surfaces",
    "require_bound_keypoint_crop_source",
    "require_direct_keypoint_crop_pixel_source",
    "rollback_keypoint_coordinate_publication",
]
