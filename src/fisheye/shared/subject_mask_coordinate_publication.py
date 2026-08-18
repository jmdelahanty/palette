"""Canonical coordinate publication for future-normal raw subject masks.

The raw U-Net writer consumes one exact canonical materialized crop and keeps
its numerical inference kernels in ROI-local pixels.  This module binds that
existing output to the selected crop's observation identity, acquisition time,
ROI extent, component labels, and direction-labelled ROI-to-camera placement.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
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
from fisheye.shared.coordinate_reference import (
    bind_persisted_record_reference_extent,
    canonical_node_path,
)
from fisheye.shared.directed_transform_chain import (
    BoundDirectedTransformChain,
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_ATTR,
    DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    load_bound_directed_transform_v2,
    stamp_directed_transform_v2,
)
from fisheye.shared.keypoint_coordinate_publication import (
    BoundKeypointCropSource,
    load_persisted_keypoint_crop_source,
    require_direct_keypoint_crop_pixel_source,
)
from fisheye.shared.model_input_transform import (
    ModelInputTransform,
    model_input_transform_from_attrs,
    resolve_model_input_transform,
)
from fisheye.shared.pixel_frame_authority import (
    ARRAY_VALUES_CANONICALIZATION,
    CROP_PLACEMENT_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    BoundPixelFrameAuthority,
    load_crop_placement_ownership,
    load_roi_pixel_frame_authority,
    load_source_camera_pixel_frame_authority,
    stamp_crop_placement_ownership,
    stamp_roi_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.proof_verification import (
    finish_proof_verification,
    proof_verification_operation,
)
from fisheye.shared.row_lineage import ROW_LINEAGE_ARRAYS
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    load_bound_transform_authority,
    stamp_crop_placement_transform_authority,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
)
from fisheye.shared.zarr.coordinate_successor_authority import (
    SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
    CoordinateSuccessorAuthorityError,
    load_coordinate_successor_authority,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    validate_subject_mask_core_run_manifest,
)
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID,
    validate_subject_mask_bundle_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_coordinate_validation_receipt import (
    RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
    SubjectMaskCoordinateValidationReceiptError,
    load_subject_mask_coordinate_validation_receipt,
)

SUBJECT_MASK_COMPONENT_LABELS_ATTR = "subject_mask_component_labels"
SUBJECT_MASK_COORDINATE_CONTEXT_ATTR = "subject_mask_coordinate_context"
SUBJECT_MASK_COORDINATE_DERIVATION_ATTR = "subject_mask_coordinate_derivation"
SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR = "subject_mask_inference_authority"
SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR = "subject_mask_array_interpretation"
SUBJECT_MASK_SURFACE_INVENTORY_ATTR = "subject_mask_surface_inventory"
SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR = "subject_mask_roi_reference_extent"
SUBJECT_MASK_PUBLICATION_OWNER_ATTR = "subject_mask_publication_owner"
SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR = "subject_mask_publication_lease"
SUBJECT_MASK_PUBLICATION_GENERATION_ATTR = "publication_generation"
SUBJECT_MASK_PUBLICATION_POLICY_ATTR = "publication_policy"

SUBJECT_MASK_COMPONENT_LABELS_SCHEMA_ID = "palette.subject_mask_component_labels"
SUBJECT_MASK_COORDINATE_CONTEXT_SCHEMA_ID = "palette.subject_mask_coordinate_context"
SUBJECT_MASK_COORDINATE_DERIVATION_SCHEMA_ID = (
    "palette.subject_mask_coordinate_derivation"
)
SUBJECT_MASK_INFERENCE_AUTHORITY_SCHEMA_ID = "palette.subject_mask_inference_authority"
SUBJECT_MASK_ARRAY_INTERPRETATION_SCHEMA_ID = (
    "palette.subject_mask_array_interpretation"
)
SUBJECT_MASK_SURFACE_INVENTORY_SCHEMA_ID = "palette.subject_mask_surface_inventory"
SUBJECT_MASK_REFERENCE_EXTENT_SCHEMA_ID = "palette.subject_mask_reference_extent"
SUBJECT_MASK_SCHEMA_VERSION = 1

_PAYLOAD_SCAN_TARGET_BYTES = 8 * 1024 * 1024
_CENTROID_FLOAT32_ABSOLUTE_TOLERANCE_ULPS = 16

_SUBJECT_MASK_COMPANION_SPECS: dict[str, dict[str, Any]] = {
    "available_channels": {
        "path": "available_channels",
        "shape": "component",
        "dtype": "bool",
        "units": "boolean",
        "operation": "model_output_component_availability_v1",
    },
    "prob_max": {
        "path": "metrics/prob_max",
        "shape": "row_component",
        "dtype": "float32",
        "units": "unit_probability",
        "operation": "maximum_native_roi_probability_v1",
    },
    "mask_present": {
        "path": "metrics/mask_present",
        "shape": "row_component",
        "dtype": "bool",
        "units": "boolean",
        "operation": "area_px_greater_than_zero_v1",
    },
    "area_px": {
        "path": "metrics/area_px",
        "shape": "row_component",
        "dtype": "float32",
        "units": "px^2",
        "operation": "thresholded_native_roi_foreground_pixel_count_v1",
    },
    "centroid_valid": {
        "path": "metrics/centroid_valid",
        "shape": "row_component",
        "dtype": "bool",
        "units": "boolean",
        "operation": "centroid_defined_if_mask_present_v1",
    },
    "bbox_valid": {
        "path": "metrics/bbox_valid",
        "shape": "row_component",
        "dtype": "bool",
        "units": "boolean",
        "operation": "bbox_defined_if_mask_present_v1",
    },
}

_BOUND_CONTEXT_SEAL = object()
_BOUND_SURFACES_SEAL = object()
_CHECKPOINT_SEAL = object()
_PUBLICATION_OWNER_RE = re.compile(r"^[0-9a-f]{32}$")

_ACTIVATION_GUARDED_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "authoritative_run",
    "authoritative_run_provenance",
)
_SUBJECT_MASK_PUBLICATION_POLICY = (
    "owner_generation_guarded_selectors_then_eligibility_v1"
)
_RAW_COORDINATE_VALIDATION_RECORD_NAMES = (
    "context",
    "derivation",
    "padded_crop_lineage",
    "row_identity",
    "surface_inventory",
    "temporal_authority",
)
_RAW_FAMILY = "subject_mask_runs"
_COORDINATE_RECORD_POINTER_FIELDS = {"record_ref", "record_sha256"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SubjectMaskCoordinatePublicationError(ValueError):
    """Raised when a raw subject-mask run cannot prove canonical coordinates."""


def _fail(message: str) -> None:
    raise SubjectMaskCoordinatePublicationError(message)


def _publication_owner(
    run: Any,
    *,
    expected: str | None = None,
) -> str:
    value = getattr(run, "attrs", {}).get(SUBJECT_MASK_PUBLICATION_OWNER_ATTR)
    if not isinstance(value, str) or _PUBLICATION_OWNER_RE.fullmatch(value) is None:
        _fail("Canonical subject-mask run lacks one unguessable publication owner.")
    if expected is not None and value != expected:
        _fail("Canonical subject-mask run was replaced by another publication owner.")
    return value


def _selector_snapshot_value(
    snapshot: Mapping[str, tuple[bool, Any]],
    name: str,
) -> tuple[bool, Any]:
    value = snapshot.get(name)
    if not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not bool:
        _fail(f"Subject-mask selector snapshot lacks exact {name!r} state.")
    return value


def _require_selector_snapshot_unchanged(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    *,
    names: Sequence[str] = _ACTIVATION_GUARDED_SELECTOR_ATTRS,
) -> None:
    attrs = getattr(parent, "attrs", {})
    for name in names:
        present, value = _selector_snapshot_value(snapshot, name)
        if (name in attrs) is not present or (present and attrs.get(name) != value):
            _fail(
                "Canonical subject-mask activation observed concurrent mutation "
                f"of parent selector {name!r}."
            )


def _require_parent_attr_snapshot_unchanged(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    names: Sequence[str],
) -> None:
    attrs = getattr(parent, "attrs", {})
    for name in names:
        present, value = _selector_snapshot_value(snapshot, name)
        if (name in attrs) is not present or (present and attrs.get(name) != value):
            _fail(
                "Canonical subject-mask activation observed concurrent mutation "
                f"of parent publication state {name!r}."
            )


def _publication_generation_from_snapshot(
    snapshot: Mapping[str, tuple[bool, Any]],
) -> int:
    present, value = _selector_snapshot_value(
        snapshot,
        SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    )
    if not present:
        return 0
    if type(value) is not int or value < 0:
        _fail("Subject-mask publication generation must be one nonnegative integer.")
    return value


def _publication_lease_record(
    *,
    run_path: str,
    publication_owner: str,
    base_generation: int,
) -> dict[str, Any]:
    return {
        "schema_id": "palette.subject_mask_publication_lease",
        "schema_version": 1,
        "policy": _SUBJECT_MASK_PUBLICATION_POLICY,
        "run_path": run_path,
        "publication_owner": publication_owner,
        "base_generation": base_generation,
        "next_generation": base_generation + 1,
    }


def _acquire_parent_publication_lease(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    *,
    run_path: str,
    publication_owner: str,
) -> dict[str, Any]:
    _require_parent_attr_snapshot_unchanged(
        parent,
        snapshot,
        (
            SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        ),
    )
    base_generation = _publication_generation_from_snapshot(snapshot)
    policy_present, policy = _selector_snapshot_value(
        snapshot,
        SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    )
    if policy_present and policy != _SUBJECT_MASK_PUBLICATION_POLICY:
        _fail("Subject-mask parent uses an unsupported publication policy.")
    if base_generation > 0 and not policy_present:
        _fail("Subject-mask parent generation lacks its publication policy.")
    lease = _publication_lease_record(
        run_path=run_path,
        publication_owner=publication_owner,
        base_generation=base_generation,
    )
    parent.attrs[SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR] = copy.deepcopy(lease)
    if parent.attrs.get(SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR) != lease:
        _fail("Subject-mask parent publication lease did not persist exactly.")
    return lease


def _require_parent_publication_lease(
    parent: Any,
    lease: Mapping[str, Any],
    *,
    expected_generation: int,
) -> None:
    attrs = getattr(parent, "attrs", {})
    if attrs.get(SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR) != dict(lease):
        _fail("Subject-mask parent publication lease was replaced.")
    generation = attrs.get(SUBJECT_MASK_PUBLICATION_GENERATION_ATTR, 0)
    if type(generation) is not int or generation != expected_generation:
        _fail("Subject-mask parent publication generation changed concurrently.")
    policy = attrs.get(SUBJECT_MASK_PUBLICATION_POLICY_ATTR)
    if expected_generation == 0:
        if policy not in (None, _SUBJECT_MASK_PUBLICATION_POLICY):
            _fail("Subject-mask parent publication policy changed concurrently.")
    elif policy != _SUBJECT_MASK_PUBLICATION_POLICY:
        _fail("Subject-mask parent publication policy changed concurrently.")


def _canonical_path(value: str, *, prefix: str, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be one canonical archive-relative path.")
    path = value.strip().strip("/")
    if (
        path != value
        or not path.startswith(prefix)
        or any(item in {"", ".", ".."} for item in path.split("/"))
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


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        value = np.asarray(node[:])
    except Exception as exc:
        _fail(f"Unable to read exact {label}: {exc}.")
    if value.dtype.hasobject:
        _fail(f"{label} cannot use object dtype.")
    return np.ascontiguousarray(value)


def _array_metadata(node: Any) -> dict[str, Any]:
    try:
        dtype = np.dtype(node.dtype)
        shape = tuple(int(item) for item in node.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(f"Coordinate surface lacks exact array metadata: {exc}.")
    return {
        "array_ref": f"/{canonical_node_path(node)}",
        "shape": list(shape),
        "dtype": dtype.str,
    }


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Subject-mask payload metadata is not canonical JSON: {exc}.")


def _payload_digest_state(node: Any) -> tuple[Any, dict[str, Any]]:
    metadata = _array_metadata(node)
    dtype = np.dtype(node.dtype)
    header = {
        "canonicalization": ARRAY_VALUES_CANONICALIZATION,
        "dtype": np.lib.format.dtype_to_descr(dtype),
        "shape": list(metadata["shape"]),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\x00")
    return digest, metadata


def _update_payload_digest(
    digest: Any,
    node: Any,
    values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype.hasobject:
        _fail(f"{label} cannot use object dtype.")
    if result.dtype != np.dtype(node.dtype):
        _fail(f"{label} changed dtype while its payload was scanned.")
    contiguous = np.ascontiguousarray(result)
    digest.update(contiguous.tobytes(order="C"))
    return contiguous


def _finish_payload(
    node: Any,
    digest: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **copy.deepcopy(dict(metadata)),
        "array_values_sha256": digest.hexdigest(),
    }


def _payload_row_chunk(node: Any) -> int:
    shape = tuple(int(item) for item in node.shape)
    if not shape or shape[0] <= 0:
        return 1
    row_elements = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    row_bytes = max(1, row_elements * int(np.dtype(node.dtype).itemsize))
    return max(1, min(shape[0], _PAYLOAD_SCAN_TARGET_BYTES // row_bytes))


def _payload(node: Any) -> dict[str, Any]:
    digest, metadata = _payload_digest_state(node)
    shape = tuple(int(item) for item in node.shape)
    label = canonical_node_path(node)
    if not shape:
        try:
            values = np.asarray(node[()])
        except Exception as exc:
            _fail(f"Unable to read exact {label}: {exc}.")
        _update_payload_digest(digest, node, values, label=label)
    else:
        chunk_rows = _payload_row_chunk(node)
        for start in range(0, shape[0], chunk_rows):
            stop = min(shape[0], start + chunk_rows)
            try:
                values = np.asarray(node[start:stop])
            except Exception as exc:
                _fail(f"Unable to read exact {label} rows {start}:{stop}: {exc}.")
            expected = (stop - start, *shape[1:])
            if tuple(values.shape) != expected:
                _fail(
                    f"{label} changed shape while its payload was scanned; "
                    f"expected {expected!r}, got {tuple(values.shape)!r}."
                )
            _update_payload_digest(digest, node, values, label=label)
    return _finish_payload(node, digest, metadata)


def _require_explicit_run_status(
    group: Any,
    *,
    status: str,
    expected_selector_eligible: bool,
    label: str,
) -> None:
    attrs = getattr(group, "attrs", {})
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != status
        or attrs.get("stage_selector_eligible") is not expected_selector_eligible
    ):
        _fail(
            f"{label} must carry the exact Palette completion contract with "
            f"status={status!r} and selector eligibility "
            f"{expected_selector_eligible!r}."
        )


def _fresh_owned_ineligible_run(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
    allowed_statuses: Sequence[str],
    label: str,
) -> Any:
    run = _node(root_node, run_path, label=label)
    _publication_owner(run, expected=expected_publication_owner)
    attrs = getattr(run, "attrs", {})
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) not in set(allowed_statuses)
        or attrs.get("stage_selector_eligible") is not False
    ):
        _fail(
            f"{label} must remain the exact owned selector-ineligible child with "
            f"status in {tuple(allowed_statuses)!r}."
        )
    return run


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


def _require_group_path(root: Any, path: str) -> tuple[Any, tuple[str, ...]]:
    """Create one canonical nested group path and report newly made nodes."""

    current = root
    current_path = ""
    created: list[str] = []
    for name in path.split("/"):
        current_path = f"{current_path}/{name}".strip("/")
        try:
            existing = root[current_path]
        except Exception:
            existing = None
        if existing is not None:
            if hasattr(existing, "shape"):
                _fail(f"Expected coordinate group {current_path!r}, found an array.")
            current = existing
            continue
        create = getattr(current, "create_group", None)
        require = getattr(current, "require_group", None)
        if callable(create):
            current = create(name)
        elif callable(require):
            current = require(name)
        else:
            _fail(f"Cannot create required coordinate group {current_path!r}.")
        if canonical_node_path(current) != current_path:
            _fail(f"Coordinate group {current_path!r} resolved to an unexpected path.")
        created.append(current_path)
    return current, tuple(created)


def _attrs_snapshot(*nodes: Any) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
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
    return (
        tuple(canonical_node_path(node) for node in unique),
        tuple(copy.deepcopy(dict(node.attrs)) for node in unique),
    )


def _restore_attrs(
    root_node: Any,
    paths: Sequence[str],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    run_path: str,
    expected_publication_owner: str,
) -> None:
    _fresh_owned_ineligible_run(
        root_node,
        run_path,
        expected_publication_owner=expected_publication_owner,
        allowed_statuses=(RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE),
        label="Subject-mask coordinate rollback target",
    )
    failures: list[str] = []
    for path, snapshot in zip(paths, snapshots, strict=True):
        try:
            _fresh_owned_ineligible_run(
                root_node,
                run_path,
                expected_publication_owner=expected_publication_owner,
                allowed_statuses=(RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE),
                label="Subject-mask coordinate rollback target",
            )
            node = _node(root_node, path, label="subject-mask rollback node")
            attrs = node.attrs
            for name in tuple(attrs.keys()):
                del attrs[name]
            attrs.update(copy.deepcopy(dict(snapshot)))
            if dict(attrs) != dict(snapshot):
                raise RuntimeError("restored attrs differ from the exact snapshot")
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{path}: {exc}")
    if failures:
        raise RuntimeError(f"Coordinate attrs rollback was incomplete: {failures!r}.")


def _delete_created(
    root: Any,
    paths: Sequence[str],
    *,
    run_path: str,
    expected_publication_owner: str,
) -> None:
    for path in reversed(tuple(paths)):
        try:
            _fresh_owned_ineligible_run(
                root,
                run_path,
                expected_publication_owner=expected_publication_owner,
                allowed_statuses=(RUN_STATUS_RUNNING,),
                label="Subject-mask coordinate rollback target",
            )
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
            f"Created coordinate nodes survived rollback: {remaining!r}."
        )


def _labels(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        _fail("Subject-component labels must be a non-empty sequence of names.")
    labels = tuple(str(item) for item in value)
    if (
        not labels
        or len(set(labels)) != len(labels)
        or any(not item or item != item.strip() for item in labels)
    ):
        _fail("Subject-component labels must be unique, non-empty canonical strings.")
    return labels


def _model_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Subject-mask model artifact must be one exact fingerprint mapping.")
    required = {
        "role",
        "path",
        "fingerprint_scheme",
        "sha256",
        "size_bytes",
        "mtime_ns",
        "source",
    }
    if set(value) != required:
        _fail(
            "Subject-mask model artifact must contain only the strict content-v1 "
            f"fingerprint fields {sorted(required)!r}."
        )
    result = copy.deepcopy(dict(value))
    if result["role"] != "subject_mask_unet_checkpoint":
        _fail("Subject-mask model artifact has the wrong role.")
    if (
        not isinstance(result["path"], str)
        or not result["path"].startswith("/")
        or result["path"] != result["path"].strip()
    ):
        _fail("Subject-mask model artifact path must be canonical and absolute.")
    if result["fingerprint_scheme"] != "content_v1":
        _fail("Subject-mask model artifact must use content_v1 fingerprinting.")
    if (
        not isinstance(result["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", result["sha256"]) is None
    ):
        _fail("Subject-mask model artifact requires one lowercase SHA-256 digest.")
    if (
        type(result["size_bytes"]) is not int
        or result["size_bytes"] <= 0
        or type(result["mtime_ns"]) is not int
        or result["mtime_ns"] <= 0
    ):
        _fail("Subject-mask model artifact requires positive exact stat evidence.")
    if result["source"] not in {
        "computed",
        "sidecar",
        "registry",
        "direct_scientific_commit_rehash",
    }:
        _fail("Subject-mask model artifact has an unsupported fingerprint source.")
    return result


def _model_input_transform_attrs(
    transform: ModelInputTransform,
    *,
    native_shape: tuple[int, int],
) -> dict[str, Any]:
    if type(transform) is not ModelInputTransform:
        _fail("Subject-mask inference requires one exact ModelInputTransform.")
    if transform.native_shape != native_shape:
        _fail(
            "Subject-mask model-input native dimensions differ from the exact "
            "selected crop ROI extent."
        )
    if transform.name not in {"identity", "pad_to_size"}:
        _fail("Subject-mask model-input transform mode is unsupported.")
    try:
        expected = resolve_model_input_transform(
            native_shape,
            mode=transform.name,
            model_hw=transform.model_shape,
        )
    except ValueError as exc:
        raise SubjectMaskCoordinatePublicationError(
            f"Subject-mask model-input transform is invalid: {exc}"
        ) from exc
    if transform != expected or transform.to_attrs() != expected.to_attrs():
        _fail(
            "Subject-mask model-input padding or inverse coordinate mapping "
            "differs from the exact supported transform."
        )
    return copy.deepcopy(transform.to_attrs())


def _threshold(value: Any) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        _fail("Subject-mask probability threshold must be one finite float in (0, 1).")
    return value


def _inference_authority_record(
    *,
    model_input_transform: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
    mask_probability_threshold: float,
) -> dict[str, Any]:
    return {
        "schema_id": SUBJECT_MASK_INFERENCE_AUTHORITY_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "operation": "native_roi_model_inference_then_exact_inverse_crop_v1",
        "transform_direction": "native_roi_px_to_model_input_px",
        "output_direction": "model_output_px_to_native_roi_px",
        "model_output_activation": "sigmoid",
        "probability_normalization": "unit_interval_closed",
        "channel_semantics": "independent_sigmoid_multilabel",
        "model_input_transform": copy.deepcopy(dict(model_input_transform)),
        "mask_probability_threshold": mask_probability_threshold,
        "model_artifact": copy.deepcopy(dict(model_artifact)),
    }


def _validate_inference_authority_record(
    record: Mapping[str, Any],
    *,
    native_shape: tuple[int, int],
) -> dict[str, Any]:
    if type(record) is not dict:
        _fail("Subject-mask inference authority must be an exact mapping.")
    transform_attrs = record.get("model_input_transform")
    if type(transform_attrs) is not dict:
        _fail("Subject-mask inference authority lacks exact transform attrs.")
    expected_transform_keys = {
        "name",
        "native_shape_hw",
        "model_shape_hw",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "coordinate_mapping",
    }
    if set(transform_attrs) != expected_transform_keys:
        _fail("Subject-mask inference transform attrs use unsupported fields.")
    try:
        native = transform_attrs["native_shape_hw"]
        model = transform_attrs["model_shape_hw"]
        if (
            type(native) is not list
            or len(native) != 2
            or any(type(item) is not int for item in native)
            or type(model) is not list
            or len(model) != 2
            or any(type(item) is not int for item in model)
            or any(
                type(transform_attrs[name]) is not int
                for name in ("pad_top", "pad_bottom", "pad_left", "pad_right")
            )
        ):
            raise ValueError("non-exact transform dimensions")
        transform = ModelInputTransform(
            name=transform_attrs["name"],
            native_height=native[0],
            native_width=native[1],
            model_height=model[0],
            model_width=model[1],
            pad_top=transform_attrs["pad_top"],
            pad_bottom=transform_attrs["pad_bottom"],
            pad_left=transform_attrs["pad_left"],
            pad_right=transform_attrs["pad_right"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SubjectMaskCoordinatePublicationError(
            f"Subject-mask inference transform is malformed: {exc}."
        ) from exc
    canonical_transform = _model_input_transform_attrs(
        transform,
        native_shape=native_shape,
    )
    artifact = _model_artifact(record.get("model_artifact"))
    threshold = _threshold(record.get("mask_probability_threshold"))
    expected = _inference_authority_record(
        model_input_transform=canonical_transform,
        model_artifact=artifact,
        mask_probability_threshold=threshold,
    )
    if record != expected:
        _fail("Subject-mask inference authority differs from its canonical form.")
    return expected


def _require_run_inference_attrs(
    run: Any,
    authority: Mapping[str, Any],
) -> None:
    artifact = authority["model_artifact"]
    expected = {
        "model_input_transform": authority["model_input_transform"],
        "mask_probability_threshold": authority["mask_probability_threshold"],
        "source_checkpoint": artifact["path"],
        "subject_mask_model_artifact": artifact,
    }
    for name, value in expected.items():
        if run.attrs.get(name) != value:
            _fail(
                f"Subject-mask run attr {name!r} differs from exact inference authority."
            )


def load_persisted_subject_mask_crop_source(
    root_node: Any,
    crop_path: str,
) -> BoundKeypointCropSource:
    """Load the shared exact canonical materialized-crop publication."""

    try:
        return load_persisted_keypoint_crop_source(root_node, crop_path)
    except Exception as exc:
        raise SubjectMaskCoordinatePublicationError(
            f"Selected canonical subject-mask crop is invalid: {exc}"
        ) from exc


def require_direct_subject_mask_crop_pixel_source(
    source: BoundKeypointCropSource,
    active_roi_images_node: Any,
) -> BoundKeypointCropSource:
    try:
        return require_direct_keypoint_crop_pixel_source(
            source,
            active_roi_images_node,
        )
    except Exception as exc:
        raise SubjectMaskCoordinatePublicationError(
            f"Active subject-mask pixels are not the exact selected crop: {exc}"
        ) from exc


def selected_subject_mask_crop_values(
    source: BoundKeypointCropSource,
    source_crop_row_ids: Any,
) -> dict[str, np.ndarray]:
    """Return one exact unique crop subset/reorder from persisted source bytes."""

    rows = np.asarray(source_crop_row_ids)
    if rows.dtype != np.dtype("<i8") or rows.ndim != 1 or rows.size == 0:
        _fail("source_crop_row_ids must be a nonempty little-endian int64 array.")
    if (
        int(rows.min()) < 0
        or int(rows.max()) >= source.crop_geometry.row_identity.leading_dimension
        or np.unique(rows).size != rows.size
    ):
        _fail("source_crop_row_ids is not one exact unique in-range crop selection.")
    crop = source._rowset_node
    values = {
        "source_crop_row_ids": rows,
        "instance_key": _array(
            _child(crop, "instance_key", label="crop instance_key"),
            label="crop instance_key",
        )[rows],
        "source_acquisition_frame_index": _array(
            _child(
                crop,
                "source_acquisition_frame_index",
                label="crop acquisition frame",
            ),
            label="crop acquisition frame",
        )[rows],
        "source_crop_xywh": _array(
            source._placement_node,
            label="crop placement",
        )[rows],
    }
    return {name: np.ascontiguousarray(item) for name, item in values.items()}


def _validate_output_selection(
    source: BoundKeypointCropSource,
    run_group: Any,
) -> dict[str, np.ndarray]:
    if "detection_source" in run_group or "detection_source" in getattr(
        run_group, "attrs", {}
    ):
        _fail(
            "Canonical subject-mask rows must explicitly omit detection_source; "
            "the exact selected crop rowset is the sole observation-identity and "
            "coordinate-lineage authority."
        )
    forbidden = tuple(
        name
        for name in ROW_LINEAGE_ARRAYS
        if name
        not in {
            "instance_key",
            "source_crop_row_ids",
            "source_acquisition_frame_index",
        }
        and name in run_group
    )
    if forbidden:
        _fail(
            "Canonical subject-mask rows must use only instance_key plus exact "
            "source_crop_row_ids/source_acquisition_frame_index lineage; legacy "
            f"row aliases are forbidden: {forbidden!r}."
        )
    rows_node = _child(run_group, "source_crop_row_ids", label="subject-mask crop rows")
    selected = selected_subject_mask_crop_values(
        source, _array(rows_node, label="crop rows")
    )
    expected_dtypes = {
        "instance_key": np.dtype("<u8"),
        "source_acquisition_frame_index": np.dtype("<i8"),
    }
    for name in ("instance_key", "source_acquisition_frame_index"):
        output = _array(_child(run_group, name, label=name), label=name)
        if output.dtype != expected_dtypes[name] or not np.array_equal(
            output,
            selected[name],
        ):
            _fail(f"Subject-mask {name} is not an exact canonical crop selection.")
    output = _array(
        _child(run_group, "source_crop_xywh", label="source_crop_xywh"),
        label="source_crop_xywh",
    )
    if output.dtype != selected["source_crop_xywh"].dtype or not np.array_equal(
        output,
        selected["source_crop_xywh"],
    ):
        _fail(
            "Subject-mask source_crop_xywh is not an exact dtype-preserving "
            "crop placement."
        )
    return selected


def _label_record(labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_id": SUBJECT_MASK_COMPONENT_LABELS_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "axis": 1,
        "role": "subject_component",
        "cardinality": len(labels),
        "labels": list(labels),
    }


def _extent_record(
    *,
    width: int,
    height: int,
    convention: str,
    source_frame: BoundPixelFrameAuthority,
    source_rows_node: Any,
) -> dict[str, Any]:
    return {
        "schema_id": SUBJECT_MASK_REFERENCE_EXTENT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "role": "subject_mask_native_roi",
        "operation": "exact_bound_crop_roi_extent_copy_v1",
        "pixel_convention": convention,
        "width": int(width),
        "height": int(height),
        "units": "px",
        "source_crop_frame": {
            "record_ref": source_frame.record_ref,
            "record_sha256": source_frame.record_sha256,
        },
        "source_crop_row_ids": _payload(source_rows_node),
    }


def _stamp_extent(node: Any, *, record: dict[str, Any]) -> Any:
    attrs = node.attrs
    for name in ("width", "height"):
        expected = record[name]
        if name in attrs and (type(attrs[name]) is not int or attrs[name] != expected):
            _fail(f"Existing {name} conflicts with the exact subject-mask ROI extent.")
        attrs[name] = expected
    stamp_and_bind_persisted_coordinate_record(
        node,
        record,
        attr_name=SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
    )
    return bind_persisted_record_reference_extent(
        node,
        record_attr=SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
        digest_attr=f"{SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR}_sha256",
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
    component_labels: BoundCoordinateRecord,
    inference_authority: BoundCoordinateRecord,
    continuous_frame: BoundPixelFrameAuthority,
    continuous_chain: BoundDirectedTransformChain,
    pixel_center_frame: BoundPixelFrameAuthority,
    pixel_center_chain: BoundDirectedTransformChain,
    pixel_center_camera: BoundPixelFrameAuthority,
    pixel_edge_frame: BoundPixelFrameAuthority,
    pixel_edge_chain: BoundDirectedTransformChain,
    pixel_edge_camera: BoundPixelFrameAuthority,
    publication_owner: str,
) -> dict[str, Any]:
    camera = source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
    return {
        "schema_id": SUBJECT_MASK_COORDINATE_CONTEXT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "publication_owner": publication_owner,
        "source_crop_path": source.crop_path,
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "temporal_authority": {
            "record_ref": temporal.record_ref,
            "record_sha256": temporal.record_sha256,
        },
        "component_labels": {
            "record_ref": component_labels.record_ref,
            "record_sha256": component_labels.record_sha256,
        },
        "inference_authority": {
            "record_ref": inference_authority.record_ref,
            "record_sha256": inference_authority.record_sha256,
        },
        "selection": {
            name: _payload(_child(run_group, name, label=name))
            for name in (
                "source_crop_row_ids",
                "instance_key",
                "source_acquisition_frame_index",
                "source_crop_xywh",
            )
        },
        "source_camera_frames": {
            "continuous": {
                "record_ref": camera.record_ref,
                "record_sha256": camera.record_sha256,
            },
            "pixel_center": {
                "record_ref": pixel_center_camera.record_ref,
                "record_sha256": pixel_center_camera.record_sha256,
            },
            "pixel_edge_half_open": {
                "record_ref": pixel_edge_camera.record_ref,
                "record_sha256": pixel_edge_camera.record_sha256,
            },
        },
        "roi_frames": {
            "continuous": {
                "record_ref": continuous_frame.record_ref,
                "record_sha256": continuous_frame.record_sha256,
            },
            "pixel_center": {
                "record_ref": pixel_center_frame.record_ref,
                "record_sha256": pixel_center_frame.record_sha256,
            },
            "pixel_edge_half_open": {
                "record_ref": pixel_edge_frame.record_ref,
                "record_sha256": pixel_edge_frame.record_sha256,
            },
        },
        "roi_to_source_camera": {
            "direction": "roi_local_px_to_source_camera_image_px",
            "continuous": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in continuous_chain.transform_records
            ],
            "pixel_center": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in pixel_center_chain.transform_records
            ],
            "pixel_edge_half_open": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in pixel_edge_chain.transform_records
            ],
        },
        "detection_source": {
            "status": "omitted",
            "reason": "not_coordinate_or_observation_identity_authority",
            "lineage_authority": "exact_source_crop_row_selection",
        },
    }


@dataclass(frozen=True, init=False)
class BoundSubjectMaskCoordinateContext:
    source: BoundKeypointCropSource
    row_identity: BoundRowIdentityContract = field(repr=False)
    temporal_authority: BoundSourceRowTemporalAuthority = field(repr=False)
    component_labels: BoundCoordinateRecord = field(repr=False)
    inference_authority: BoundCoordinateRecord = field(repr=False)
    labels: tuple[str, ...]
    continuous_frame: BoundPixelFrameAuthority = field(repr=False)
    continuous_chain: BoundDirectedTransformChain = field(repr=False)
    pixel_center_frame: BoundPixelFrameAuthority = field(repr=False)
    pixel_center_chain: BoundDirectedTransformChain = field(repr=False)
    pixel_edge_frame: BoundPixelFrameAuthority = field(repr=False)
    pixel_edge_chain: BoundDirectedTransformChain = field(repr=False)
    context_record: BoundCoordinateRecord = field(repr=False)
    run_path: str
    completion_status: str
    selector_eligible: bool
    publication_owner: str
    _root: Any = field(repr=False, compare=False)
    _run_group: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _BOUND_CONTEXT_SEAL:
            _fail("Subject-mask coordinate contexts cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


@proof_verification_operation
def prepare_subject_mask_coordinate_context(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
    crop_path: str,
    mask_labels: Sequence[str],
    model_input_transform: ModelInputTransform,
    model_artifact: Mapping[str, Any],
    mask_probability_threshold: float,
) -> BoundSubjectMaskCoordinateContext:
    """Bind exact crop selection, identity, frames, and transforms pre-inference."""

    path = _canonical_path(
        run_path, prefix="subject_mask_runs/", label="subject-mask rowset"
    )
    run = _fresh_owned_ineligible_run(
        root_node,
        path,
        expected_publication_owner=expected_publication_owner,
        allowed_statuses=(RUN_STATUS_RUNNING,),
        label="Subject-mask coordinate preflight target",
    )
    publication_owner = _publication_owner(
        run,
        expected=expected_publication_owner,
    )
    source = load_persisted_subject_mask_crop_source(root_node, crop_path)
    labels = _labels(mask_labels)
    _validate_output_selection(source, run)
    native_shape = (
        int(source.roi_frame.endpoint.height),
        int(source.roi_frame.endpoint.width),
    )
    transform_attrs = _model_input_transform_attrs(
        model_input_transform,
        native_shape=native_shape,
    )
    inference_record = _inference_authority_record(
        model_input_transform=transform_attrs,
        model_artifact=_model_artifact(model_artifact),
        mask_probability_threshold=_threshold(mask_probability_threshold),
    )
    _require_run_inference_attrs(run, inference_record)
    created: list[str] = []
    targets: tuple[Any, ...] = ()
    snapshots: tuple[dict[str, Any], ...] = ()
    try:
        acquisition = (
            source.crop_geometry.source_geometry.frame_evidence.acquisition_frame
        )
        pixel_center_camera_path = (
            "analysis/coordinate_frames/source_camera/"
            f"{acquisition.record.camera_id}/pixel_center"
        )
        pixel_edge_camera_path = (
            "analysis/coordinate_frames/source_camera/"
            f"{acquisition.record.camera_id}/pixel_edge_half_open"
        )
        pixel_center_camera_node, _shared_center_paths = _require_group_path(
            root_node,
            pixel_center_camera_path,
        )
        pixel_edge_camera_node, _shared_edge_paths = _require_group_path(
            root_node,
            pixel_edge_camera_path,
        )
        frames, made = _require_group(run, "coordinate_frames")
        if made:
            created.append(canonical_node_path(frames))
        continuous_node, made = _require_group(frames, "roi_local_continuous")
        if made:
            created.append(canonical_node_path(continuous_node))
        pixel_center_node, made = _require_group(frames, "roi_local_pixel_center")
        if made:
            created.append(canonical_node_path(pixel_center_node))
        pixel_edge_node, made = _require_group(
            frames,
            "roi_local_pixel_edge_half_open",
        )
        if made:
            created.append(canonical_node_path(pixel_edge_node))
        key_node = _child(run, "instance_key", label="subject-mask identity")
        time_node = _child(
            run,
            "source_acquisition_frame_index",
            label="subject-mask acquisition time",
        )
        placement = _child(
            run,
            "source_crop_xywh",
            label="canonical crop placement",
        )
        targets, snapshots = _attrs_snapshot(
            run,
            key_node,
            time_node,
            placement,
            continuous_node,
            pixel_center_node,
            pixel_edge_node,
        )

        def authorize_context_mutation() -> None:
            _fresh_owned_ineligible_run(
                root_node,
                path,
                expected_publication_owner=publication_owner,
                allowed_statuses=(RUN_STATUS_RUNNING,),
                label="Subject-mask coordinate preflight mutation target",
            )

        authorize_context_mutation()
        identity = stamp_and_bind_row_identity_contract(
            run,
            key_node,
            contract=build_row_identity_contract(
                domain=OBSERVATION_INSTANCE_DOMAIN,
                values=_array(key_node, label="subject-mask instance_key"),
            ),
        )
        temporal = stamp_source_row_temporal_authority(
            run,
            time_node,
            source_row_identity=identity,
            acquisition_frame=(
                source.crop_geometry.source_geometry.frame_evidence.acquisition_frame
            ),
        )
        label_authority = stamp_and_bind_persisted_coordinate_record(
            run,
            _label_record(labels),
            attr_name=SUBJECT_MASK_COMPONENT_LABELS_ATTR,
        )
        inference_authority = stamp_and_bind_persisted_coordinate_record(
            run,
            inference_record,
            attr_name=SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR,
        )
        authorize_context_mutation()
        camera = source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
        pixel_center_camera = stamp_source_camera_pixel_frame_authority(
            pixel_center_camera_node,
            frame_id=f"{acquisition.record.camera_id}_source_camera_pixel_center",
            pixel_convention="pixel_center",
            acquisition_frame=acquisition,
        )
        pixel_center_camera = load_source_camera_pixel_frame_authority(
            _node(
                root_node,
                pixel_center_camera_path,
                label="shared pixel-center source-camera frame",
            ),
            acquisition_frame=acquisition,
        )
        pixel_edge_camera = stamp_source_camera_pixel_frame_authority(
            pixel_edge_camera_node,
            frame_id=(
                f"{acquisition.record.camera_id}_source_camera_pixel_edge_half_open"
            ),
            pixel_convention="pixel_edge_half_open",
            acquisition_frame=acquisition,
        )
        pixel_edge_camera = load_source_camera_pixel_frame_authority(
            _node(
                root_node,
                pixel_edge_camera_path,
                label="shared half-open source-camera frame",
            ),
            acquisition_frame=acquisition,
        )
        width = int(source.roi_frame.endpoint.width)
        height = int(source.roi_frame.endpoint.height)
        rows_node = _child(run, "source_crop_row_ids", label="subject-mask crop rows")
        token = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]

        def frame_and_chain(
            frame_node: Any,
            *,
            convention: str,
            target_camera: BoundPixelFrameAuthority,
            ownership_attr: str,
            authority_attr: str,
            transform_attr: str,
        ) -> tuple[BoundPixelFrameAuthority, BoundDirectedTransformChain]:
            authorize_context_mutation()
            ownership = stamp_crop_placement_ownership(
                placement,
                row_identity=identity,
                source_camera_frame=target_camera,
                attr_name=ownership_attr,
            )
            extent = _stamp_extent(
                frame_node,
                record=_extent_record(
                    width=width,
                    height=height,
                    convention=convention,
                    source_frame=source.roi_frame,
                    source_rows_node=rows_node,
                ),
            )
            frame = stamp_roi_pixel_frame_authority(
                extent,
                frame_id=f"subject_mask_roi_{convention}_{token}",
                pixel_convention=convention,
                crop_placement_ownership=ownership,
            )
            authority = stamp_crop_placement_transform_authority(
                placement,
                authority_id=f"subject_mask_roi_{convention}_to_source_camera_{token}",
                source_frame=frame,
                target_frame=target_camera,
                attr_name=authority_attr,
            )
            link = stamp_directed_transform_v2(
                placement,
                transform_id=f"subject_mask_roi_{convention}_to_source_camera_{token}",
                authority=authority,
                source_frame=frame,
                target_frame=target_camera,
                row_identity=identity,
                attr_name=transform_attr,
            )
            return frame, resolve_bound_directed_transform_chain((link,))

        continuous_frame, continuous_chain = frame_and_chain(
            continuous_node,
            convention="continuous",
            target_camera=camera,
            ownership_attr=CROP_PLACEMENT_OWNERSHIP_ATTR,
            authority_attr=TRANSFORM_AUTHORITY_ATTR,
            transform_attr=DIRECTED_TRANSFORM_V2_ATTR,
        )
        pixel_center_frame, pixel_center_chain = frame_and_chain(
            pixel_center_node,
            convention="pixel_center",
            target_camera=pixel_center_camera,
            ownership_attr=CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
            authority_attr=TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
            transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
        )
        pixel_edge_frame, pixel_edge_chain = frame_and_chain(
            pixel_edge_node,
            convention="pixel_edge_half_open",
            target_camera=pixel_edge_camera,
            ownership_attr=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
            authority_attr=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
            transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
        )
        context = _context_record(
            source=source,
            run_group=run,
            identity=identity,
            temporal=temporal,
            component_labels=label_authority,
            inference_authority=inference_authority,
            continuous_frame=continuous_frame,
            continuous_chain=continuous_chain,
            pixel_center_frame=pixel_center_frame,
            pixel_center_chain=pixel_center_chain,
            pixel_center_camera=pixel_center_camera,
            pixel_edge_frame=pixel_edge_frame,
            pixel_edge_chain=pixel_edge_chain,
            pixel_edge_camera=pixel_edge_camera,
            publication_owner=publication_owner,
        )
        authorize_context_mutation()
        stamp_and_bind_persisted_coordinate_record(
            run,
            context,
            attr_name=SUBJECT_MASK_COORDINATE_CONTEXT_ATTR,
        )
        return _load_subject_mask_coordinate_context(
            root_node,
            path,
            require_complete=False,
            expected_selector_eligible=False,
            expected_publication_owner=publication_owner,
        )
    except BaseException as exc:
        failures: list[str] = []
        rollback_authorized = False
        try:
            _fresh_owned_ineligible_run(
                root_node,
                path,
                expected_publication_owner=publication_owner,
                allowed_statuses=(RUN_STATUS_RUNNING,),
                label="Subject-mask context rollback target",
            )
            rollback_authorized = True
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            failures.append(f"ownership: {rollback_exc}")
        if rollback_authorized:
            try:
                _restore_attrs(
                    root_node,
                    targets,
                    snapshots,
                    run_path=path,
                    expected_publication_owner=publication_owner,
                )
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                failures.append(f"attrs: {rollback_exc}")
            try:
                _delete_created(
                    root_node,
                    created,
                    run_path=path,
                    expected_publication_owner=publication_owner,
                )
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                failures.append(f"nodes: {rollback_exc}")
        if failures:
            raise SubjectMaskCoordinatePublicationError(
                "Subject-mask context preparation failed and rollback was incomplete: "
                f"{failures!r}."
            ) from exc
        raise


def _load_subject_mask_coordinate_context(
    root_node: Any,
    run_path: str,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
    expected_publication_owner: str | None = None,
) -> BoundSubjectMaskCoordinateContext:
    """Load one context, including its sealed successor-only crop adapter."""

    path = _canonical_path(
        run_path, prefix="subject_mask_runs/", label="subject-mask rowset"
    )
    run = _node(root_node, path, label="subject-mask rowset")
    adapter_attr = "coordinate_successor_historical_crop_adapter"
    if require_complete and adapter_attr in getattr(run, "attrs", {}):
        binding = _load_persisted_historical_crop_successor_binding(
            root_node,
            run,
            run_path=path,
        )
        from fisheye.shared.zarr.historical_geometry_only_crop_adapter import (
            historical_geometry_only_crop_loader,
        )

        with historical_geometry_only_crop_loader(binding):
            return _load_subject_mask_coordinate_context_impl(
                root_node,
                path,
                require_complete=require_complete,
                expected_selector_eligible=expected_selector_eligible,
                expected_publication_owner=expected_publication_owner,
            )
    return _load_subject_mask_coordinate_context_impl(
        root_node,
        path,
        require_complete=require_complete,
        expected_selector_eligible=expected_selector_eligible,
        expected_publication_owner=expected_publication_owner,
    )


def _load_persisted_historical_crop_successor_binding(
    root_node: Any,
    run: Any,
    *,
    run_path: str,
) -> Any:
    """Rebuild the exact historical crop adapter from successor-bound evidence."""

    from fisheye.shared.zarr.historical_geometry_only_crop_adapter import (
        bind_historical_geometry_only_crop_source,
    )

    try:
        authority = load_coordinate_successor_authority(
            run,
            expected_kind=SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run_path,
        )
        load_subject_mask_coordinate_validation_receipt(
            run,
            expected_kind=RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
            expected_successor_run_path=run_path,
            expected_coordinate_record_names=(
                *_RAW_COORDINATE_VALIDATION_RECORD_NAMES,
            ),
        )
        padded = bind_persisted_coordinate_record(
            run,
            attr_name="coordinate_successor_padded_crop_lineage",
        )
        adapter = bind_persisted_coordinate_record(
            run,
            attr_name="coordinate_successor_historical_crop_adapter",
        )
    except Exception as exc:
        _fail(f"Historical crop successor authority is invalid: {exc}.")

    authority_records = authority["payload"]["coordinate_records"]
    if authority_records.get("padded_crop_lineage") != _record_pointer(padded):
        _fail("Historical crop successor padded-lineage authority is stale.")
    adapter_record = padded.record.get("source_crop_adapter")
    if not isinstance(adapter_record, Mapping) or adapter.record != adapter_record:
        _fail(
            "Historical crop successor adapter differs from its authority-bound "
            "padded lineage."
        )

    source_run_path = authority["payload"]["source"].get("run_path")
    if (
        type(source_run_path) is not str
        or source_run_path != adapter_record.get("source_run_path")
    ):
        _fail("Historical crop successor source path is inconsistent.")
    source_run = _node(
        root_node,
        source_run_path,
        label="historical subject-mask source core",
    )
    source_manifest = source_run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(source_manifest, Mapping):
        _fail("Historical subject-mask source core lacks its run manifest.")
    source_errors = validate_subject_mask_core_run_manifest(source_manifest)
    if source_errors:
        _fail(
            "Historical subject-mask source core manifest is invalid: "
            + "; ".join(source_errors)
        )
    source_arrays = {
        name: _child(source_run, name, label=f"historical source {name}")
        for name in (
            "source_crop_row_ids",
            "instance_key",
            "source_acquisition_frame_index",
            "source_crop_xywh",
        )
    }
    if "source_crop_row_signature" in source_run:
        source_arrays["source_crop_row_signature"] = source_run[
            "source_crop_row_signature"
        ]
    try:
        crop_reference = source_manifest["payload"]["coordinate_dependencies"][
            "document"
        ]["crop"]
        transform = model_input_transform_from_attrs(
            dict(adapter_record["model_input_transform"])
        )
        identity = archive_identity(root_node)
        if identity.kind != "local_store_root":
            _fail(
                "Persisted historical crop successors require a stable local archive "
                "identity."
            )
        binding = bind_historical_geometry_only_crop_source(
            analysis_zarr=Path(identity.key[0]),
            root=root_node,
            crop_reference=crop_reference,
            source_manifest=source_manifest,
            source_arrays=source_arrays,
            source_run_path=source_run_path,
            model_input_transform=transform,
        )
    except SubjectMaskCoordinatePublicationError:
        raise
    except Exception as exc:
        _fail(f"Persisted historical crop successor cannot be rebound: {exc}.")
    if binding.as_record() != adapter_record:
        _fail("Persisted historical crop successor adapter evidence changed.")
    return binding


def _load_subject_mask_coordinate_context_impl(
    root_node: Any,
    run_path: str,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
    expected_publication_owner: str | None = None,
) -> BoundSubjectMaskCoordinateContext:
    path = _canonical_path(
        run_path, prefix="subject_mask_runs/", label="subject-mask rowset"
    )
    run = _node(root_node, path, label="subject-mask rowset")
    publication_owner = _publication_owner(
        run,
        expected=expected_publication_owner,
    )
    status = RUN_STATUS_COMPLETE if require_complete else RUN_STATUS_RUNNING
    _require_explicit_run_status(
        run,
        status=status,
        expected_selector_eligible=expected_selector_eligible,
        label="Canonical subject-mask rowset",
    )
    if require_complete and run.attrs.get("coordinate_contract") != "canonical_v2":
        _fail("Complete subject-mask contexts require canonical_v2 publication.")
    context = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_MASK_COORDINATE_CONTEXT_ATTR,
    )
    source_path = context.record.get("source_crop_path")
    source = load_persisted_subject_mask_crop_source(root_node, source_path)
    _validate_output_selection(source, run)
    labels_record = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_MASK_COMPONENT_LABELS_ATTR,
    )
    raw_labels = labels_record.record
    labels = _labels(raw_labels.get("labels", ()))
    if raw_labels != _label_record(labels) or list(
        run.attrs.get("mask_labels", ())
    ) != list(labels):
        _fail("Persisted subject-component labels differ from their exact authority.")
    inference_authority = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR,
    )
    identity = load_bound_row_identity_contract(
        run,
        _child(run, "instance_key", label="subject-mask identity"),
    )
    temporal = load_bound_source_row_temporal_authority(
        run,
        _child(run, "source_acquisition_frame_index", label="subject-mask time"),
        source_row_identity=identity,
        acquisition_frame=(
            source.crop_geometry.source_geometry.frame_evidence.acquisition_frame
        ),
    )
    camera = source.crop_geometry.source_geometry.frame_evidence.source_camera_frame
    acquisition = source.crop_geometry.source_geometry.frame_evidence.acquisition_frame
    pixel_camera_path = (
        "analysis/coordinate_frames/source_camera/"
        f"{acquisition.record.camera_id}/pixel_center"
    )
    pixel_center_camera = load_source_camera_pixel_frame_authority(
        _node(root_node, pixel_camera_path, label="pixel-center source-camera frame"),
        acquisition_frame=acquisition,
    )
    pixel_edge_camera_path = (
        "analysis/coordinate_frames/source_camera/"
        f"{acquisition.record.camera_id}/pixel_edge_half_open"
    )
    pixel_edge_camera = load_source_camera_pixel_frame_authority(
        _node(
            root_node,
            pixel_edge_camera_path,
            label="half-open source-camera frame",
        ),
        acquisition_frame=acquisition,
    )
    width = int(source.roi_frame.endpoint.width)
    height = int(source.roi_frame.endpoint.height)
    inference_record = _validate_inference_authority_record(
        inference_authority.record,
        native_shape=(height, width),
    )
    _require_run_inference_attrs(run, inference_record)
    rows_node = _child(run, "source_crop_row_ids", label="subject-mask crop rows")

    def load_frame_and_chain(
        frame_name: str,
        *,
        convention: str,
        target_camera: BoundPixelFrameAuthority,
        ownership_attr: str,
        authority_attr: str,
        transform_attr: str,
    ) -> tuple[BoundPixelFrameAuthority, BoundDirectedTransformChain]:
        placement = _child(run, "source_crop_xywh", label="canonical crop placement")
        ownership = load_crop_placement_ownership(
            placement,
            row_identity=identity,
            source_camera_frame=target_camera,
            attr_name=ownership_attr,
        )
        frame_node = _node(
            root_node,
            f"{path}/coordinate_frames/{frame_name}",
            label=f"subject-mask {convention} ROI frame",
        )
        extent = bind_persisted_record_reference_extent(
            frame_node,
            record_attr=SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
            digest_attr=f"{SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR}_sha256",
            width_field="width",
            height_field="height",
            units_field="units",
        )
        expected_extent = _extent_record(
            width=width,
            height=height,
            convention=convention,
            source_frame=source.roi_frame,
            source_rows_node=rows_node,
        )
        expected_bound_extent = {
            **expected_extent,
            "bound_record_attr": SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
            "bound_digest_attr": f"{SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR}_sha256",
            "bound_width_field": "width",
            "bound_height_field": "height",
            "bound_units_field": "units",
        }
        if extent.authority_record != expected_bound_extent:
            _fail(
                "Persisted subject-mask ROI extent differs from the exact crop extent."
            )
        frame = load_roi_pixel_frame_authority(
            frame_node,
            reference_extent=extent,
            crop_placement_ownership=ownership,
        )
        authority = load_bound_transform_authority(
            placement,
            payload_node=placement,
            source_frame=frame,
            target_frame=target_camera,
            row_identity=identity,
            attr_name=authority_attr,
        )
        link = load_bound_directed_transform_v2(
            placement,
            authority=authority,
            source_frame=frame,
            target_frame=target_camera,
            row_identity=identity,
            attr_name=transform_attr,
        )
        return frame, resolve_bound_directed_transform_chain((link,))

    continuous_frame, continuous_chain = load_frame_and_chain(
        "roi_local_continuous",
        convention="continuous",
        target_camera=camera,
        ownership_attr=CROP_PLACEMENT_OWNERSHIP_ATTR,
        authority_attr=TRANSFORM_AUTHORITY_ATTR,
        transform_attr=DIRECTED_TRANSFORM_V2_ATTR,
    )
    pixel_center_frame, pixel_center_chain = load_frame_and_chain(
        "roi_local_pixel_center",
        convention="pixel_center",
        target_camera=pixel_center_camera,
        ownership_attr=CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
        authority_attr=TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
        transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
    )
    pixel_edge_frame, pixel_edge_chain = load_frame_and_chain(
        "roi_local_pixel_edge_half_open",
        convention="pixel_edge_half_open",
        target_camera=pixel_edge_camera,
        ownership_attr=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
        authority_attr=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
        transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    expected_context = _context_record(
        source=source,
        run_group=run,
        identity=identity,
        temporal=temporal,
        component_labels=labels_record,
        inference_authority=inference_authority,
        continuous_frame=continuous_frame,
        continuous_chain=continuous_chain,
        pixel_center_frame=pixel_center_frame,
        pixel_center_chain=pixel_center_chain,
        pixel_center_camera=pixel_center_camera,
        pixel_edge_frame=pixel_edge_frame,
        pixel_edge_chain=pixel_edge_chain,
        pixel_edge_camera=pixel_edge_camera,
        publication_owner=publication_owner,
    )
    if context.record != expected_context:
        _fail("Persisted subject-mask context differs from exact live crop evidence.")
    return BoundSubjectMaskCoordinateContext(
        source=source,
        row_identity=identity,
        temporal_authority=temporal,
        component_labels=labels_record,
        inference_authority=inference_authority,
        labels=labels,
        continuous_frame=continuous_frame,
        continuous_chain=continuous_chain,
        pixel_center_frame=pixel_center_frame,
        pixel_center_chain=pixel_center_chain,
        pixel_edge_frame=pixel_edge_frame,
        pixel_edge_chain=pixel_edge_chain,
        context_record=context,
        run_path=path,
        completion_status=status,
        selector_eligible=expected_selector_eligible,
        publication_owner=publication_owner,
        _root=root_node,
        _run_group=run,
        _verification_seal=_BOUND_CONTEXT_SEAL,
    )


def _surface_nodes(run: Any) -> dict[str, Any]:
    metrics = _child(run, "metrics", label="subject-mask metrics")
    result = {
        "mask_probs_roi": _child(run, "mask_probs_roi", label="mask probabilities"),
        "centroid_xy": _child(metrics, "centroid_xy", label="mask centroids"),
        "bbox_xyxy": _child(metrics, "bbox_xyxy", label="mask bounding boxes"),
    }
    if "masks_roi" in run:
        result["masks_roi"] = _child(run, "masks_roi", label="binary masks")
    return result


def _validate_surface_metadata(
    context: BoundSubjectMaskCoordinateContext,
) -> dict[str, Any]:
    nodes = _surface_nodes(context._run_group)
    n = context.row_identity.leading_dimension
    c = len(context.labels)
    if n <= 0:
        _fail(
            "Canonical subject-mask publication does not support zero-row runs; "
            "omit the run instead of publishing an identity-free coordinate surface."
        )
    h = int(context.pixel_center_frame.endpoint.height)
    w = int(context.pixel_center_frame.endpoint.width)
    expected = {
        "mask_probs_roi": (n, c, h, w),
        "centroid_xy": (n, c, 2),
        "bbox_xyxy": (n, c, 4),
    }
    if "masks_roi" in nodes:
        expected["masks_roi"] = (n, c, h, w)
    for name, shape in expected.items():
        node = nodes[name]
        actual_shape = tuple(int(item) for item in node.shape)
        try:
            dtype = np.dtype(node.dtype)
        except (AttributeError, TypeError) as exc:
            _fail(f"Subject-mask {name} lacks one exact dtype: {exc}.")
        if actual_shape != shape:
            _fail(
                f"Subject-mask {name} shape {actual_shape!r} disagrees with exact "
                f"row/component/ROI reference dimensions {shape!r}."
            )
        if name == "masks_roi" and dtype != np.dtype("uint8"):
            _fail("Canonical masks_roi must use uint8 storage.")
        if name == "mask_probs_roi" and dtype not in {
            np.dtype("uint8"),
            np.dtype("float16"),
        }:
            _fail("Canonical mask_probs_roi must use uint8 or float16 storage.")
        if name in {"centroid_xy", "bbox_xyxy"} and dtype != np.dtype("float32"):
            _fail(f"Canonical {name} must use exact float32 storage.")
    probability_dtype = np.dtype(nodes["mask_probs_roi"].dtype)
    expected_probability_attrs = {
        "output_semantics": "multilabel",
        "overlap_policy": "independent_sigmoid",
        "probability_semantics": "sigmoid_multilabel_logits",
        "probabilities_dtype": (
            "uint8" if probability_dtype == np.dtype("uint8") else "float16"
        ),
        "probabilities_encoding": (
            "linear_uint8_0_255"
            if probability_dtype == np.dtype("uint8")
            else "unit_float"
        ),
    }
    for name, expected_value in expected_probability_attrs.items():
        if context._run_group.attrs.get(name) != expected_value:
            _fail(
                f"Subject-mask run attr {name!r} differs from its exact persisted "
                "probability encoding or channel semantics."
            )
    masks_materialized = "masks_roi" in nodes
    for name in ("masks_roi_materialized", "binary_masks_materialized"):
        if context._run_group.attrs.get(name) is not masks_materialized:
            _fail(
                f"Subject-mask run attr {name!r} must exactly declare whether "
                "the thresholded binary cache is physically present."
            )
    threshold = _threshold(
        context.inference_authority.record.get("mask_probability_threshold")
    )
    expected_binary_source = (
        f"threshold(mask_probs_roi, threshold={threshold})"
        if masks_materialized
        else "not_materialized"
    )
    if context._run_group.attrs.get("binary_masks_source") != expected_binary_source:
        _fail(
            "Subject-mask binary_masks_source differs from the exact optional "
            "threshold-cache derivation."
        )
    expected_bbox_attrs = {
        "bbox_xyxy_convention": "pixel_edge_half_open",
        "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
    }
    for name, expected_value in expected_bbox_attrs.items():
        if context._run_group.attrs.get(name) != expected_value:
            _fail(
                f"Subject-mask run attr {name!r} differs from the canonical "
                "half-open bbox contract."
            )
    return nodes


def _companion_nodes(run: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in _SUBJECT_MASK_COMPANION_SPECS.items():
        node = run
        for part in str(spec["path"]).split("/"):
            node = _child(node, part, label=f"subject-mask {name}")
        result[name] = node
    return result


def _read_row_block(
    node: Any,
    *,
    start: int,
    stop: int,
    label: str,
) -> np.ndarray:
    shape = tuple(int(item) for item in node.shape)
    try:
        values = np.asarray(node[start:stop])
    except Exception as exc:
        _fail(f"Unable to read exact {label} rows {start}:{stop}: {exc}.")
    expected = (stop - start, *shape[1:])
    if tuple(values.shape) != expected or values.dtype != np.dtype(node.dtype):
        _fail(
            f"Subject-mask {label} changed shape or dtype during bounded "
            "derivation validation."
        )
    if values.dtype.hasobject:
        _fail(f"Subject-mask {label} cannot use object dtype.")
    return np.ascontiguousarray(values)


def _centroid_absolute_tolerance_px(*, height: int, width: int) -> float:
    return float(
        np.finfo(np.float32).eps
        * max(int(height), int(width))
        * _CENTROID_FLOAT32_ABSOLUTE_TOLERANCE_ULPS
    )


def _derive_thresholded_probability_metrics(
    stored_probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, np.ndarray]:
    stored = np.asarray(stored_probabilities)
    if stored.ndim != 4:
        _fail("Subject-mask probabilities must have shape (N,C,H,W).")
    if stored.dtype == np.dtype("uint8"):
        probabilities = stored.astype(np.float32) / np.float32(255.0)
    elif stored.dtype == np.dtype("float16"):
        if (
            not np.all(np.isfinite(stored))
            or np.any(stored < np.float16(0.0))
            or np.any(stored > np.float16(1.0))
        ):
            _fail("Float16 subject-mask probabilities must be finite values in [0,1].")
        probabilities = stored.astype(np.float32)
    else:  # guarded by _validate_surface_metadata
        _fail("Subject-mask probabilities use an unsupported storage dtype.")

    binary = probabilities >= np.float32(threshold)
    area_int = binary.sum(axis=(2, 3), dtype=np.int64)
    area_px = area_int.astype(np.float32)
    valid = area_int > 0
    prob_max = probabilities.max(axis=(2, 3)).astype(np.float32, copy=False)

    row_count, component_count, height, width = binary.shape
    centroid_xy = np.zeros((row_count, component_count, 2), dtype=np.float64)
    if np.any(valid):
        y_counts = binary.sum(axis=3, dtype=np.int64)
        x_counts = binary.sum(axis=2, dtype=np.int64)
        y_coords = np.arange(height, dtype=np.float64).reshape(1, 1, height)
        x_coords = np.arange(width, dtype=np.float64).reshape(1, 1, width)
        denominator = np.maximum(area_int, 1).astype(np.float64)
        centroid_xy[:, :, 0] = (x_counts.astype(np.float64) * x_coords).sum(
            axis=2, dtype=np.float64
        ) / denominator
        centroid_xy[:, :, 1] = (y_counts.astype(np.float64) * y_coords).sum(
            axis=2, dtype=np.float64
        ) / denominator
        centroid_xy[~valid] = 0.0

    row_has_mask = binary.any(axis=3)
    col_has_mask = binary.any(axis=2)
    y_indices = np.arange(height, dtype=np.int64).reshape(1, 1, height)
    x_indices = np.arange(width, dtype=np.int64).reshape(1, 1, width)
    y_min = np.where(row_has_mask, y_indices, height).min(axis=2)
    y_max_exclusive = np.where(row_has_mask, y_indices + 1, 0).max(axis=2)
    x_min = np.where(col_has_mask, x_indices, width).min(axis=2)
    x_max_exclusive = np.where(col_has_mask, x_indices + 1, 0).max(axis=2)
    bbox_xyxy = np.stack(
        (x_min, y_min, x_max_exclusive, y_max_exclusive),
        axis=2,
    ).astype(
        np.float32,
        copy=False,
    )
    bbox_xyxy[~valid] = 0.0
    return {
        "decoded_probabilities": probabilities,
        "binary": binary.astype(np.uint8, copy=False),
        "prob_max": prob_max,
        "mask_present": valid.astype(bool, copy=False),
        "area_px": area_px,
        "centroid_xy": centroid_xy,
        "centroid_valid": valid.astype(bool, copy=False),
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": valid.astype(bool, copy=False),
    }


def _validate_companion_metadata_and_values(
    context: BoundSubjectMaskCoordinateContext,
    geometry_nodes: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    nodes = _companion_nodes(context._run_group)
    n = context.row_identity.leading_dimension
    c = len(context.labels)
    for name, spec in _SUBJECT_MASK_COMPANION_SPECS.items():
        node = nodes[name]
        expected_shape = (c,) if spec["shape"] == "component" else (n, c)
        actual_shape = tuple(int(item) for item in node.shape)
        dtype = np.dtype(node.dtype)
        expected_dtype = np.dtype(bool if spec["dtype"] == "bool" else np.float32)
        if actual_shape != expected_shape or dtype != expected_dtype:
            _fail(
                f"Subject-mask companion {name} must have exact shape/dtype "
                f"{expected_shape!r}/{expected_dtype.str!r}; got "
                f"{actual_shape!r}/{dtype.str!r}."
            )

    all_nodes = {**dict(geometry_nodes), **nodes}
    digest_states = {
        name: (*_payload_digest_state(node), node) for name, node in all_nodes.items()
    }
    available = _array(nodes["available_channels"], label="available_channels")
    available_digest, _available_metadata, available_node = digest_states[
        "available_channels"
    ]
    _update_payload_digest(
        available_digest,
        available_node,
        available,
        label="available_channels",
    )
    if not np.any(available):
        _fail(
            "Canonical raw subject-mask output must retain at least one available "
            "model component channel."
        )

    h = int(context.continuous_frame.endpoint.height)
    w = int(context.continuous_frame.endpoint.width)
    centroid_atol = _centroid_absolute_tolerance_px(height=h, width=w)
    threshold = _threshold(
        context.inference_authority.record.get("mask_probability_threshold")
    )
    row_aligned_names = tuple(
        name for name in all_nodes if name != "available_channels"
    )
    chunk_rows = _payload_row_chunk(geometry_nodes["mask_probs_roi"])
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        blocks: dict[str, np.ndarray] = {}
        for name in row_aligned_names:
            node = all_nodes[name]
            block = _read_row_block(
                node,
                start=start,
                stop=stop,
                label=name,
            )
            digest, _metadata, digest_node = digest_states[name]
            _update_payload_digest(
                digest,
                digest_node,
                block,
                label=name,
            )
            blocks[name] = block

        unavailable = ~available
        if np.any(unavailable):
            for name, block in blocks.items():
                if np.any(block[:, unavailable, ...] != 0):
                    _fail(
                        "Unavailable subject-mask component channels must contain "
                        f"exact zero placeholders in {name}."
                    )

        expected = _derive_thresholded_probability_metrics(
            blocks["mask_probs_roi"],
            threshold=threshold,
        )
        if "masks_roi" in blocks:
            masks = blocks["masks_roi"]
            if np.any((masks != 0) & (masks != 1)):
                _fail("Canonical masks_roi must contain only exact uint8 0/1 values.")
            if not np.array_equal(masks, expected["binary"]):
                _fail(
                    "Canonical masks_roi must exactly equal mask_probs_roi "
                    "thresholded at the persisted inclusive threshold."
                )

        probability = blocks["prob_max"]
        if (
            not np.all(np.isfinite(probability))
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
            or not np.array_equal(probability, expected["prob_max"])
        ):
            _fail(
                "Subject-mask prob_max must exactly equal the finite decoded "
                "native-ROI probability maximum."
            )
        area = blocks["area_px"]
        if (
            not np.all(np.isfinite(area))
            or np.any(area < 0.0)
            or not np.array_equal(area, expected["area_px"])
        ):
            _fail(
                "Subject-mask area_px must exactly equal the thresholded native-ROI "
                "foreground pixel count."
            )
        for name in ("mask_present", "centroid_valid", "bbox_valid"):
            if not np.array_equal(blocks[name], expected[name]):
                _fail(
                    f"Subject-mask {name} must exactly equal the declared area_px>0 "
                    "validity relationship and thresholded probability derivation."
                )

        centroid = blocks["centroid_xy"]
        centroid_valid = blocks["centroid_valid"]
        if not np.all(np.isfinite(centroid)):
            _fail("Subject-mask centroid coordinates must be finite; NaN is forbidden.")
        if np.any(centroid[~centroid_valid] != 0.0):
            _fail(
                "Invalid subject-mask centroid entries must be exact zero sentinels "
                "paired with centroid_valid=false."
            )
        if np.any(centroid_valid):
            delta = np.abs(
                centroid[centroid_valid].astype(np.float64)
                - expected["centroid_xy"][centroid_valid]
            )
            if np.any(delta > centroid_atol):
                _fail(
                    "Valid subject-mask centroids differ from the independently "
                    "derived foreground pixel-center mean."
                )

        bbox = blocks["bbox_xyxy"]
        bbox_valid = blocks["bbox_valid"]
        if not np.all(np.isfinite(bbox)):
            _fail("Subject-mask bbox coordinates must be finite; NaN is forbidden.")
        if np.any(bbox[~bbox_valid] != 0.0):
            _fail(
                "Invalid subject-mask bbox entries must be exact zero sentinels paired "
                "with bbox_valid=false."
            )
        if not np.array_equal(bbox, expected["bbox_xyxy"]):
            _fail(
                "Subject-mask bbox_xyxy must exactly equal foreground half-open "
                "pixel edges [x_min,y_min,x_max+1,y_max+1]."
            )

    payloads = {
        name: _finish_payload(node, digest, metadata)
        for name, (digest, metadata, node) in digest_states.items()
    }
    return nodes, payloads


def _record_pointer(value: BoundCoordinateRecord) -> dict[str, str]:
    return {
        "record_ref": value.record_ref,
        "record_sha256": value.record_sha256,
    }


def _row_component_axes(
    context: BoundSubjectMaskCoordinateContext,
    *,
    component_axis: int,
    include_row_axis: bool,
) -> dict[str, Any]:
    return {
        "component_axis": {
            "axis": component_axis,
            "role": "subject_component",
            "cardinality": len(context.labels),
            "label_authority": _record_pointer(context.component_labels),
        },
        "row_axis": (
            {
                "axis": 0,
                "role": "observation_instance",
                "cardinality": context.row_identity.leading_dimension,
                "row_identity": {
                    "record_ref": context.row_identity.record_ref,
                    "record_sha256": context.row_identity.record_sha256,
                },
            }
            if include_row_axis
            else None
        ),
    }


def _companion_interpretation_record(
    context: BoundSubjectMaskCoordinateContext,
    *,
    name: str,
    node: Any,
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    spec = _SUBJECT_MASK_COMPANION_SPECS[name]
    component_axis = 0 if spec["shape"] == "component" else 1
    record: dict[str, Any] = {
        "schema_id": SUBJECT_MASK_ARRAY_INTERPRETATION_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "array_role": name,
        "operation": spec["operation"],
        "units": spec["units"],
        "payload": copy.deepcopy(dict(payloads[name])),
        **_row_component_axes(
            context,
            component_axis=component_axis,
            include_row_axis=spec["shape"] == "row_component",
        ),
        "inference_authority": _record_pointer(context.inference_authority),
    }
    if name in {"mask_present", "centroid_valid", "bbox_valid"}:
        record["validity_relationship"] = {
            "defined_by": "area_px > 0",
            "area_px_payload": copy.deepcopy(dict(payloads["area_px"])),
        }
    if name == "centroid_valid":
        record["geometry_relationship"] = {
            "geometry_payload": copy.deepcopy(dict(payloads["centroid_xy"])),
            "false_value_policy": "zero_xy_is_invalid_sentinel_not_coordinate",
        }
    if name == "bbox_valid":
        record["geometry_relationship"] = {
            "geometry_payload": copy.deepcopy(dict(payloads["bbox_xyxy"])),
            "false_value_policy": "zero_xyxy_is_invalid_sentinel_not_geometry",
        }
    if name == "area_px":
        record["measurement"] = {
            "dimension": "area",
            "pixel_measure": "native_roi_pixel_count",
            "unit_symbol": "px^2",
        }
    if name == "available_channels":
        record["availability_semantics"] = (
            "true_means_model_output_component_is_materialized_for_every_row;"
            "false_means_authenticated_exact_zero_placeholder"
        )
    return record


def _geometry_interpretation_record(
    context: BoundSubjectMaskCoordinateContext,
    *,
    name: str,
    node: Any,
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    threshold = _threshold(
        context.inference_authority.record.get("mask_probability_threshold")
    )
    record: dict[str, Any] = {
        "schema_id": SUBJECT_MASK_ARRAY_INTERPRETATION_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "array_role": name,
        "payload": copy.deepcopy(dict(payloads[name])),
        **_row_component_axes(
            context,
            component_axis=1,
            include_row_axis=True,
        ),
        "inference_authority": _record_pointer(context.inference_authority),
        "coordinate_context": _record_pointer(context.context_record),
    }
    if name == "mask_probs_roi":
        dtype = np.dtype(node.dtype)
        record.update(
            {
                "surface_role": "authoritative_model_probability_raster",
                "operation": "model_output_inverse_preprocessing_to_native_roi_raster_v1",
                "probability_encoding": (
                    "linear_uint8_0_255_to_unit_probability_v1"
                    if dtype == np.dtype("uint8")
                    else "unit_float16_v1"
                ),
                "model_output_activation": "sigmoid",
                "probability_normalization": "unit_interval_closed",
                "decoded_probability_range": [0.0, 1.0],
                "nonfinite_policy": "forbidden",
                "channel_semantics": "independent_sigmoid_multilabel",
            }
        )
    elif name == "masks_roi":
        record.update(
            {
                "surface_role": "derived_exact_threshold_cache",
                "operation": "threshold_mask_probs_roi_at_declared_run_threshold_v1",
                "source_probability_payload": copy.deepcopy(
                    dict(payloads["mask_probs_roi"])
                ),
                "threshold": threshold,
                "threshold_comparison": "greater_than_or_equal",
                "value_domain": [0, 1],
            }
        )
    elif name == "centroid_xy":
        record.update(
            {
                "surface_role": "derived_thresholded_mask_geometry",
                "operation": "mean_foreground_pixel_center_xy_v1",
                "source_probability_payload": copy.deepcopy(
                    dict(payloads["mask_probs_roi"])
                ),
                "validity_payload": copy.deepcopy(dict(payloads["centroid_valid"])),
                "valid_value_policy": "finite_within_native_roi",
                "invalid_value_policy": "exact_zero_xy_sentinel",
                "nan_policy": "forbidden",
                "validation_tolerance": {
                    "rtol": 0.0,
                    "atol_px": _centroid_absolute_tolerance_px(
                        height=int(context.continuous_frame.endpoint.height),
                        width=int(context.continuous_frame.endpoint.width),
                    ),
                    "basis": "float32_epsilon_times_max_extent_times_16",
                },
            }
        )
    elif name == "bbox_xyxy":
        record.update(
            {
                "surface_role": "derived_thresholded_mask_geometry",
                "operation": "foreground_half_open_pixel_edges_xyxy_v1",
                "source_probability_payload": copy.deepcopy(
                    dict(payloads["mask_probs_roi"])
                ),
                "validity_payload": copy.deepcopy(dict(payloads["bbox_valid"])),
                "bound_convention": "pixel_edge_half_open",
                "valid_value_policy": "finite_exact_integer_pixel_edges_with_positive_extent",
                "invalid_value_policy": "exact_zero_xyxy_sentinel",
                "nan_policy": "forbidden",
            }
        )
    else:  # pragma: no cover - caller owns the closed geometry set
        _fail(f"Unsupported subject-mask geometry role {name!r}.")
    return record


def _bind_interpretations(
    context: BoundSubjectMaskCoordinateContext,
    geometry_nodes: Mapping[str, Any],
    companion_nodes: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    stamp: bool,
) -> dict[str, BoundCoordinateRecord]:
    result: dict[str, BoundCoordinateRecord] = {}
    for name, node in {**dict(geometry_nodes), **dict(companion_nodes)}.items():
        expected = (
            _geometry_interpretation_record(
                context,
                name=name,
                node=node,
                payloads=payloads,
            )
            if name in geometry_nodes
            else _companion_interpretation_record(
                context,
                name=name,
                node=node,
                payloads=payloads,
            )
        )
        bound = (
            stamp_and_bind_persisted_coordinate_record(
                node,
                expected,
                attr_name=SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR,
            )
            if stamp
            else bind_persisted_coordinate_record(
                node,
                attr_name=SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR,
            )
        )
        if bound.record != expected:
            _fail(
                f"Subject-mask {name} interpretation differs from live array, "
                "row, component, inference, or validity evidence."
            )
        result[name] = bound
    return result


def _inventory_record(
    context: BoundSubjectMaskCoordinateContext,
    geometry_nodes: Mapping[str, Any],
    companion_nodes: Mapping[str, Any],
    interpretations: Mapping[str, BoundCoordinateRecord],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    geometry = {
        name: {
            "array_role": name,
            "payload": copy.deepcopy(dict(payloads[name])),
            "publication": "canonical_coordinate_descriptor_v2",
            "interpretation": _record_pointer(interpretations[name]),
        }
        for name in sorted(geometry_nodes)
    }
    companions = {
        name: {
            "array_role": name,
            "payload": copy.deepcopy(dict(payloads[name])),
            "interpretation": _record_pointer(interpretations[name]),
        }
        for name in sorted(companion_nodes)
    }
    return {
        "schema_id": SUBJECT_MASK_SURFACE_INVENTORY_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "component_labels": _record_pointer(context.component_labels),
        "inference_authority": _record_pointer(context.inference_authority),
        "required_geometry": ["bbox_xyxy", "centroid_xy", "mask_probs_roi"],
        "optional_geometry": {
            "masks_roi": "masks_roi" in geometry_nodes,
        },
        "required_companions": sorted(_SUBJECT_MASK_COMPANION_SPECS),
        "geometry": geometry,
        "companions": companions,
    }


def _derivation_record(
    context: BoundSubjectMaskCoordinateContext,
    nodes: Mapping[str, Any],
    inventory: BoundCoordinateRecord,
    interpretations: Mapping[str, BoundCoordinateRecord],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    probability_dtype = np.dtype(nodes["mask_probs_roi"].dtype)
    return {
        "schema_id": SUBJECT_MASK_COORDINATE_DERIVATION_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCHEMA_VERSION,
        "coordinate_context": {
            "record_ref": context.context_record.record_ref,
            "record_sha256": context.context_record.record_sha256,
        },
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "component_labels": {
            "record_ref": context.component_labels.record_ref,
            "record_sha256": context.component_labels.record_sha256,
        },
        "inference_authority": _record_pointer(context.inference_authority),
        "surface_inventory": _record_pointer(inventory),
        "array_interpretations": {
            name: _record_pointer(value)
            for name, value in sorted(interpretations.items())
        },
        "validation_rules": {
            "content_binding": ARRAY_VALUES_CANONICALIZATION,
            "zero_row_policy": "unsupported_omit_run_instead_of_publishing",
            "probability_encoding": (
                "linear_uint8_0_255_to_unit_probability_v1"
                if probability_dtype == np.dtype("uint8")
                else "unit_float16_v1"
            ),
            "threshold": _threshold(
                context.inference_authority.record.get("mask_probability_threshold")
            ),
            "threshold_comparison": "greater_than_or_equal",
            "binary_value_domain": [0, 1],
            "invalid_centroid_sentinel": [0.0, 0.0],
            "invalid_bbox_sentinel": [0.0, 0.0, 0.0, 0.0],
            "nan_policy": "forbidden_all_persisted_probability_and_geometry_surfaces",
            "centroid_atol_px": _centroid_absolute_tolerance_px(
                height=int(context.continuous_frame.endpoint.height),
                width=int(context.continuous_frame.endpoint.width),
            ),
            "centroid_rtol": 0.0,
            "bbox_convention": "pixel_edge_half_open",
        },
        "operations": {
            name: operation
            for name, operation in {
                "mask_probs_roi": "model_output_inverse_preprocessing_to_native_roi_raster_v1",
                "masks_roi": "threshold_mask_probs_roi_at_declared_run_threshold_v1",
                "centroid_xy": "mean_foreground_pixel_center_xy_v1",
                "bbox_xyxy": "foreground_half_open_pixel_edges_xyxy_v1",
            }.items()
            if name in nodes
        },
        "arrays": {
            name: {
                "array_role": name,
                "payload": copy.deepcopy(dict(payloads[name])),
                "interpretation": _record_pointer(interpretations[name]),
            }
            for name in sorted(nodes)
        },
    }


def _collection_axis(
    context: BoundSubjectMaskCoordinateContext,
) -> CanonicalCollectionAxis:
    return CanonicalCollectionAxis(
        axis=1,
        role="subject_component",
        cardinality=len(context.labels),
        label_authority=DigestBoundCoordinateRecordRef(
            record_ref=context.component_labels.record_ref,
            record_sha256=context.component_labels.record_sha256,
        ),
    )


def _bindings(
    context: BoundSubjectMaskCoordinateContext,
    derivation: BoundCoordinateRecord,
    inventory: BoundCoordinateRecord,
    interpretations: Mapping[str, BoundCoordinateRecord],
    *,
    load: bool,
) -> dict[str, BoundCanonicalCoordinateDescriptor]:
    nodes = _validate_surface_metadata(context)
    collection = _collection_axis(context)
    specs: dict[str, dict[str, Any]] = {
        "mask_probs_roi": {
            "geometry_type": "raster_yx",
            "components": ("y", "x"),
            "component_units": ("px", "px"),
            "pixel_convention": "pixel_center",
            "reference_frame_authority": context.pixel_center_frame,
            "transform_chain": context.pixel_center_chain,
        },
        "centroid_xy": {
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "component_units": ("px", "px"),
            "pixel_convention": "continuous",
            "reference_frame_authority": context.continuous_frame,
            "transform_chain": context.continuous_chain,
        },
        "bbox_xyxy": {
            "geometry_type": "bbox_xyxy",
            "components": ("x_min", "y_min", "x_max", "y_max"),
            "component_units": ("px", "px", "px", "px"),
            "pixel_convention": "pixel_edge_half_open",
            "reference_frame_authority": context.pixel_edge_frame,
            "transform_chain": context.pixel_edge_chain,
        },
    }
    if "masks_roi" in nodes:
        specs["masks_roi"] = dict(specs["mask_probs_roi"])
    result: dict[str, BoundCanonicalCoordinateDescriptor] = {}
    for name, spec in specs.items():
        lineage = (
            context.component_labels,
            context.inference_authority,
            context.context_record,
            inventory,
            derivation,
            interpretations[name],
        )
        evidence = {
            "row_identity": context.row_identity,
            "reference_frame_authority": spec["reference_frame_authority"],
            "transform_chain": spec["transform_chain"],
            "lineage_records": lineage,
        }
        result[name] = (
            load_bound_canonical_coordinate_descriptor(nodes[name], **evidence)
            if load
            else build_bound_canonical_coordinate_descriptor(
                nodes[name],
                profile_id="roi_local_px.top_left_y_down.v1",
                geometry_type=spec["geometry_type"],
                components=spec["components"],
                component_units=spec["component_units"],
                pixel_convention=spec["pixel_convention"],
                row_identity=context.row_identity,
                reference_frame_authority=spec["reference_frame_authority"],
                source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
                transform_chain=spec["transform_chain"],
                lineage_records=lineage,
                collection_axis=collection,
            )
        )
        if result[name].descriptor.collection_axis != collection:
            _fail(
                f"Subject-mask {name} does not bind the exact ordered "
                "subject-component label authority."
            )
    return result


def _receipt_pointer(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _COORDINATE_RECORD_POINTER_FIELDS:
        _fail(f"{label} is not one exact coordinate-record pointer.")
    record_ref = value.get("record_ref")
    record_sha256 = value.get("record_sha256")
    if (
        type(record_ref) is not str
        or not record_ref.startswith("/")
        or record_ref.count("@") != 1
        or type(record_sha256) is not str
        or _SHA256_RE.fullmatch(record_sha256) is None
    ):
        _fail(f"{label} is not one canonical coordinate-record pointer.")
    return {"record_ref": record_ref, "record_sha256": record_sha256}


def _require_receipt_pointer(
    receipt_records: Mapping[str, Any],
    name: str,
    value: Any,
) -> None:
    expected = _receipt_pointer(receipt_records.get(name), label=f"receipt {name}")
    actual = _receipt_pointer(
        value if isinstance(value, Mapping) else _record_pointer(value),
        label=f"live {name}",
    )
    if actual != expected:
        _fail(f"Coordinate validation receipt {name!r} pointer is stale.")


def _validate_raw_coordinate_successor_receipt_authority(
    context: BoundSubjectMaskCoordinateContext,
    receipt: Mapping[str, Any],
    receipt_binding: BoundCoordinateRecord,
) -> None:
    try:
        authority = load_coordinate_successor_authority(
            context._run_group,
            expected_kind=SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=context.run_path,
        )
    except (CoordinateSuccessorAuthorityError, ValueError) as exc:
        _fail(f"Raw subject-mask coordinate successor authority is invalid: {exc}.")

    payload = authority["payload"]
    receipt_payload = receipt["payload"]
    manifest = context._run_group.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        _fail("Raw coordinate successor target lacks its core run manifest.")
    manifest_errors = validate_subject_mask_core_run_manifest(manifest)
    if manifest_errors:
        _fail(
            "Raw coordinate successor target core run manifest is invalid: "
            + "; ".join(manifest_errors)
        )
    manifest_payload = manifest["payload"]
    target_run_id = context.run_path.rsplit("/", 1)[-1]
    if (
        manifest_payload.get("run_id") != target_run_id
        or manifest_payload.get("stage_family") != _RAW_FAMILY
        or manifest_payload.get("kind") != "raw_probability_uint8"
    ):
        _fail("Raw coordinate successor target manifest path or kind is stale.")
    if context._run_group.attrs.get("coordinate_contract") != "canonical_v2":
        _fail("Raw coordinate successor target is not marked canonical_v2.")
    source = payload["source"]
    receipt_source = receipt_payload["source"]
    source_run_path = source.get("run_path")
    if (
        source.get("family") != _RAW_FAMILY
        or type(source_run_path) is not str
        or source_run_path.split("/")[0] != _RAW_FAMILY
        or len(source_run_path.split("/")) != 2
        or not source_run_path.split("/", 1)[1]
    ):
        _fail("Raw coordinate successor authority names a non-raw source family.")
    try:
        source_run = _node(
            context._root,
            source_run_path,
            label="raw coordinate successor source core",
        )
    except SubjectMaskCoordinatePublicationError as exc:
        _fail(f"Raw coordinate successor source core is unavailable: {exc}.")
    source_manifest = source_run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(source_manifest, Mapping):
        _fail("Raw coordinate successor source core lacks its run manifest.")
    source_manifest_errors = validate_subject_mask_core_run_manifest(source_manifest)
    if source_manifest_errors:
        _fail(
            "Raw coordinate successor source core manifest is invalid: "
            + "; ".join(source_manifest_errors)
        )
    source_manifest_payload = source_manifest["payload"]
    source_manifest_logical = source_manifest_payload.get("logical_content")
    if not isinstance(source_manifest_logical, Mapping):
        _fail("Raw coordinate successor source core lacks logical-content evidence.")
    source_run_id = source_run_path.rsplit("/", 1)[-1]
    if (
        source_manifest_payload.get("run_id") != source_run_id
        or source_manifest_payload.get("stage_family") != _RAW_FAMILY
        or source_manifest_payload.get("kind") != "raw_probability_uint8"
        or source_manifest.get("payload_digest")
        != source["manifest_payload_digest"]
        or canonical_json_sha256(source_manifest) != source["manifest_document_digest"]
        or source_manifest_logical.get("digest") != source["logical_content_digest"]
    ):
        _fail(
            "Raw coordinate successor source core manifest differs from the "
            "successor authority or receipt."
        )
    expected_source = {
        "run_path": receipt_source["run_path"],
        "manifest_payload_digest": receipt_source["core_manifest_payload_digest"],
        "manifest_document_digest": receipt_source["core_manifest_document_digest"],
        "logical_content_digest": receipt_source["logical_content_digest"],
    }
    if source.get("family") != _RAW_FAMILY or source.get("run_path") != expected_source[
        "run_path"
    ]:
        _fail("Raw coordinate receipt source run differs from successor authority.")
    for authority_name, receipt_name in (
        ("manifest_payload_digest", "manifest_payload_digest"),
        ("manifest_document_digest", "manifest_document_digest"),
        ("logical_content_digest", "logical_content_digest"),
    ):
        if source.get(authority_name) != expected_source[receipt_name]:
            _fail(
                "Raw coordinate receipt source manifest or logical-content digest "
                "differs from successor authority."
            )
    target_logical = manifest_payload.get("logical_content")
    if not isinstance(target_logical, Mapping):
        _fail("Raw coordinate successor target manifest lacks logical-content bindings.")
    if target_logical.get("digest") != receipt_source["logical_content_digest"]:
        _fail(
            "Raw coordinate successor target manifest logical identity differs "
            "from the receipt and successor authority."
        )

    source_authority = payload["source_authority"]
    bundle = receipt_payload["bundle_authority"]
    bundle_manifest = source_authority.get("record")
    if not isinstance(bundle_manifest, Mapping):
        _fail("Raw coordinate successor bundle authority is absent.")
    bundle_errors = validate_subject_mask_bundle_manifest(bundle_manifest)
    if bundle_errors:
        _fail(
            "Raw coordinate successor bundle authority is invalid: "
            + "; ".join(bundle_errors)
        )
    bundle_members = bundle_manifest.get("payload", {}).get("members")
    raw_member = bundle_members.get("raw") if isinstance(bundle_members, Mapping) else None
    expected_source = {
        name: raw_member.get(name) if isinstance(raw_member, Mapping) else None
        for name in (
            "family",
            "run_path",
            "manifest_schema_id",
            "manifest_schema_version",
            "manifest_payload_digest",
            "manifest_document_digest",
            "logical_content_digest",
        )
    }
    if not isinstance(raw_member, Mapping) or source != expected_source:
        _fail(
            "Raw coordinate successor source authority is not bound to the "
            "bundle raw member."
        )
    if (
        source_authority.get("kind") != bundle["kind"]
        or bundle["kind"] != "inactive_subject_mask_bundle_v3"
        or bundle_manifest.get("schema_id") != SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID
        or bundle_manifest.get("schema_version") != 3
        or source_authority.get("record_sha256") != bundle["document_digest"]
    ):
        _fail("Raw coordinate receipt bundle authority differs from successor authority.")

    authority_equivalence = payload["payload_equivalence"]
    receipt_equivalence = receipt_payload["payload_equivalence"]
    if authority_equivalence.get("source_logical_content_digest") != receipt_source[
        "logical_content_digest"
    ]:
        _fail("Raw coordinate payload equivalence is bound to another source digest.")
    file_equivalence = authority_equivalence.get("payload_file_equivalence")
    if not isinstance(file_equivalence, Mapping):
        _fail("Raw coordinate successor lacks payload-file equivalence evidence.")
    for name in (
        "schema_id",
        "schema_version",
        "receipt_digest",
        "inventory_digest",
        "payload_file_count",
    ):
        if file_equivalence.get(name) != receipt_equivalence.get(name):
            _fail(
                "Raw coordinate receipt payload-file equivalence differs from "
                "successor authority."
            )

    authority_records = payload["coordinate_records"]
    receipt_records = receipt_payload["coordinate_records"]
    for name in _RAW_COORDINATE_VALIDATION_RECORD_NAMES:
        if name not in authority_records:
            _fail(f"Successor authority omits raw coordinate record {name!r}.")
        _require_receipt_pointer(receipt_records, name, authority_records[name])
    expected_receipt_pointer = _record_pointer(receipt_binding)
    if authority_records.get("coordinate_validation_receipt") != expected_receipt_pointer:
        _fail("Successor authority does not bind the exact coordinate validation receipt.")


def _validate_coordinate_payload_inventory_entry(
    name: str,
    entry: Any,
    node: Any,
    *,
    publication: str | None,
) -> None:
    if not isinstance(entry, Mapping):
        _fail(f"Raw coordinate inventory entry {name!r} is not an object.")
    expected_fields = {"array_role", "payload", "interpretation"}
    if publication is not None:
        expected_fields.add("publication")
    if set(entry) != expected_fields or entry.get("array_role") != name:
        _fail(f"Raw coordinate inventory entry {name!r} has an unexpected shape.")
    if publication is not None and entry.get("publication") != publication:
        _fail(f"Raw coordinate inventory entry {name!r} has an unexpected publication.")
    payload = entry.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {
        "array_ref",
        "shape",
        "dtype",
        "array_values_sha256",
    }:
        _fail(f"Raw coordinate inventory payload {name!r} is incomplete.")
    live = _array_metadata(node)
    if payload.get("array_ref") != live["array_ref"]:
        _fail(f"Raw coordinate inventory array_ref for {name!r} is stale.")
    if payload.get("shape") != live["shape"] or payload.get("dtype") != live["dtype"]:
        _fail(f"Raw coordinate inventory shape or dtype for {name!r} is stale.")
    if (
        type(payload.get("array_values_sha256")) is not str
        or _SHA256_RE.fullmatch(payload["array_values_sha256"]) is None
    ):
        _fail(f"Raw coordinate inventory payload digest for {name!r} is malformed.")


def _validate_raw_coordinate_inventory_metadata(
    context: BoundSubjectMaskCoordinateContext,
    geometry_nodes: Mapping[str, Any],
    companion_nodes: Mapping[str, Any],
    inventory: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
) -> dict[str, BoundCoordinateRecord]:
    record = inventory.record
    expected_fields = {
        "schema_id",
        "schema_version",
        "row_identity",
        "component_labels",
        "inference_authority",
        "required_geometry",
        "optional_geometry",
        "required_companions",
        "geometry",
        "companions",
    }
    if set(record) != expected_fields:
        _fail("Raw coordinate surface inventory record fields are not exact.")
    if record["row_identity"] != _record_pointer(context.row_identity):
        _fail("Raw coordinate surface inventory has a stale row-identity pointer.")
    if record["component_labels"] != _record_pointer(context.component_labels):
        _fail("Raw coordinate surface inventory has a stale component-label pointer.")
    if record["inference_authority"] != _record_pointer(context.inference_authority):
        _fail("Raw coordinate surface inventory has a stale inference pointer.")
    if record["required_geometry"] != ["bbox_xyxy", "centroid_xy", "mask_probs_roi"]:
        _fail("Raw coordinate surface inventory required geometry set is not exact.")
    if record["optional_geometry"] != {"masks_roi": "masks_roi" in geometry_nodes}:
        _fail("Raw coordinate surface inventory optional geometry set is stale.")
    if record["required_companions"] != sorted(_SUBJECT_MASK_COMPANION_SPECS):
        _fail("Raw coordinate surface inventory companion set is not exact.")
    geometry_inventory = record["geometry"]
    companion_inventory = record["companions"]
    if (
        not isinstance(geometry_inventory, Mapping)
        or not isinstance(companion_inventory, Mapping)
        or set(geometry_inventory) != set(geometry_nodes)
        or set(companion_inventory) != set(companion_nodes)
    ):
        _fail("Raw coordinate surface inventory does not cover the live array set.")

    all_nodes = {**dict(geometry_nodes), **dict(companion_nodes)}
    all_inventory = {**dict(geometry_inventory), **dict(companion_inventory)}
    interpretations: dict[str, BoundCoordinateRecord] = {}
    for name, node in all_nodes.items():
        _validate_coordinate_payload_inventory_entry(
            name,
            all_inventory[name],
            node,
            publication=(
                "canonical_coordinate_descriptor_v2" if name in geometry_nodes else None
            ),
        )
        try:
            interpretation = bind_persisted_coordinate_record(
                node,
                attr_name=SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR,
            )
        except Exception as exc:
            _fail(f"Raw coordinate interpretation {name!r} is unavailable: {exc}.")
        entry = all_inventory[name]
        if entry["interpretation"] != _record_pointer(interpretation):
            _fail(f"Raw coordinate inventory interpretation {name!r} is stale.")
        interpretation_record = interpretation.record
        if (
            interpretation_record.get("array_role") != name
            or interpretation_record.get("payload") != entry["payload"]
        ):
            _fail(f"Raw coordinate interpretation {name!r} disagrees with inventory.")
        interpretations[name] = interpretation

    derivation_record = derivation.record
    expected_derivation_fields = {
        "schema_id",
        "schema_version",
        "coordinate_context",
        "row_identity",
        "component_labels",
        "inference_authority",
        "surface_inventory",
        "array_interpretations",
        "validation_rules",
        "operations",
        "arrays",
    }
    if set(derivation_record) != expected_derivation_fields:
        _fail("Raw coordinate derivation record fields are not exact.")
    for name, expected in (
        ("coordinate_context", context.context_record),
        ("row_identity", context.row_identity),
        ("component_labels", context.component_labels),
        ("inference_authority", context.inference_authority),
        ("surface_inventory", inventory),
    ):
        if derivation_record[name] != _record_pointer(expected):
            _fail(f"Raw coordinate derivation has a stale {name} pointer.")
    if set(derivation_record["array_interpretations"]) != set(interpretations):
        _fail("Raw coordinate derivation interpretation set is stale.")
    for name, interpretation in interpretations.items():
        if derivation_record["array_interpretations"][name] != _record_pointer(
            interpretation
        ):
            _fail(f"Raw coordinate derivation interpretation {name!r} is stale.")
    all_nodes = {**dict(geometry_nodes), **dict(companion_nodes)}
    if set(derivation_record["operations"]) != set(geometry_nodes):
        _fail("Raw coordinate derivation operation set is stale.")
    if set(derivation_record["arrays"]) != set(all_nodes):
        _fail("Raw coordinate derivation array set is stale.")
    for name in all_nodes:
        array_record = derivation_record["arrays"].get(name)
        inventory_entry = all_inventory[name]
        if (
            not isinstance(array_record, Mapping)
            or set(array_record) != {"array_role", "payload", "interpretation"}
            or array_record["array_role"] != name
            or array_record["payload"] != inventory_entry["payload"]
            or array_record["interpretation"]
            != _record_pointer(interpretations[name])
        ):
            _fail(f"Raw coordinate derivation array {name!r} is stale.")
    validation_rules = derivation_record["validation_rules"]
    if not isinstance(validation_rules, Mapping):
        _fail("Raw coordinate derivation validation rules are absent.")
    if validation_rules.get("content_binding") != ARRAY_VALUES_CANONICALIZATION:
        _fail("Raw coordinate derivation content binding is unsupported.")
    if validation_rules.get("threshold_comparison") != "greater_than_or_equal":
        _fail("Raw coordinate derivation threshold semantics are unsupported.")
    return interpretations


def _load_raw_subject_mask_coordinate_surfaces_from_receipt(
    context: BoundSubjectMaskCoordinateContext,
) -> BoundSubjectMaskCoordinateSurfaces:
    run = context._run_group
    try:
        receipt = load_subject_mask_coordinate_validation_receipt(
            run,
            expected_kind=RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
            expected_successor_run_path=context.run_path,
            expected_coordinate_record_names=_RAW_COORDINATE_VALIDATION_RECORD_NAMES,
        )
    except SubjectMaskCoordinateValidationReceiptError as exc:
        _fail(f"Raw subject-mask coordinate validation receipt is invalid: {exc}.")

    try:
        receipt_binding = bind_persisted_coordinate_record(
            run,
            attr_name=SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
        )
    except Exception as exc:
        _fail(f"Raw coordinate validation receipt pointer is unavailable: {exc}.")
    _validate_raw_coordinate_successor_receipt_authority(
        context,
        receipt,
        receipt_binding,
    )

    try:
        inventory = bind_persisted_coordinate_record(
            run,
            attr_name=SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
        )
        derivation = bind_persisted_coordinate_record(
            run,
            attr_name=SUBJECT_MASK_COORDINATE_DERIVATION_ATTR,
        )
        padded = bind_persisted_coordinate_record(
            run,
            attr_name="coordinate_successor_padded_crop_lineage",
        )
    except Exception as exc:
        _fail(f"Raw coordinate receipt lineage record is unavailable: {exc}.")

    live_records = {
        "context": context.context_record,
        "derivation": derivation,
        "padded_crop_lineage": padded,
        "row_identity": context.row_identity,
        "surface_inventory": inventory,
        "temporal_authority": context.temporal_authority,
    }
    receipt_records = receipt["payload"]["coordinate_records"]
    for name in _RAW_COORDINATE_VALIDATION_RECORD_NAMES:
        _require_receipt_pointer(receipt_records, name, live_records[name])

    geometry_nodes = _validate_surface_metadata(context)
    companion_nodes = _companion_nodes(run)
    interpretations = _validate_raw_coordinate_inventory_metadata(
        context,
        geometry_nodes,
        companion_nodes,
        inventory,
        derivation,
    )
    bindings = _bindings(
        context,
        derivation,
        inventory,
        interpretations,
        load=True,
    )
    return BoundSubjectMaskCoordinateSurfaces(
        mask_probs_roi=bindings["mask_probs_roi"],
        masks_roi=bindings.get("masks_roi"),
        centroid_xy=bindings["centroid_xy"],
        bbox_xyxy=bindings["bbox_xyxy"],
        context=context,
        derivation=derivation,
        inventory=inventory,
        interpretations=interpretations,
        _verification_seal=_BOUND_SURFACES_SEAL,
    )


@dataclass(frozen=True, init=False)
class BoundSubjectMaskCoordinateSurfaces:
    mask_probs_roi: BoundCanonicalCoordinateDescriptor
    centroid_xy: BoundCanonicalCoordinateDescriptor
    bbox_xyxy: BoundCanonicalCoordinateDescriptor
    masks_roi: BoundCanonicalCoordinateDescriptor | None
    context: BoundSubjectMaskCoordinateContext = field(repr=False)
    derivation: BoundCoordinateRecord = field(repr=False)
    inventory: BoundCoordinateRecord = field(repr=False)
    interpretations: Mapping[str, BoundCoordinateRecord] = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _BOUND_SURFACES_SEAL:
            _fail("Subject-mask coordinate surfaces cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


@dataclass(frozen=True, init=False)
class SubjectMaskCoordinatePublicationCheckpoint:
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
        if _verification_seal is not _CHECKPOINT_SEAL:
            _fail(
                "Subject-mask publication checkpoints cannot be constructed directly."
            )
        object.__setattr__(self, "run_path", run_path)
        object.__setattr__(self, "publication_owner", publication_owner)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_paths", paths)
        object.__setattr__(self, "_attrs", attrs)
        object.__setattr__(self, "_seal", _verification_seal)


def capture_subject_mask_coordinate_publication_checkpoint(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
) -> SubjectMaskCoordinatePublicationCheckpoint:
    path = _canonical_path(
        run_path, prefix="subject_mask_runs/", label="subject-mask rowset"
    )
    run = _fresh_owned_ineligible_run(
        root_node,
        path,
        expected_publication_owner=expected_publication_owner,
        allowed_statuses=(RUN_STATUS_RUNNING,),
        label="Subject-mask publication checkpoint target",
    )
    nodes = _surface_nodes(run)
    companions = _companion_nodes(run)
    targets, attrs = _attrs_snapshot(run, *nodes.values(), *companions.values())
    return SubjectMaskCoordinatePublicationCheckpoint(
        run_path=path,
        publication_owner=expected_publication_owner,
        root=root_node,
        paths=targets,
        attrs=attrs,
        _verification_seal=_CHECKPOINT_SEAL,
    )


def rollback_subject_mask_coordinate_publication(
    checkpoint: SubjectMaskCoordinatePublicationCheckpoint,
) -> None:
    if (
        type(checkpoint) is not SubjectMaskCoordinatePublicationCheckpoint
        or checkpoint._seal is not _CHECKPOINT_SEAL
    ):
        _fail("A sealed subject-mask publication checkpoint is required.")
    _restore_attrs(
        checkpoint._root,
        checkpoint._paths,
        checkpoint._attrs,
        run_path=checkpoint.run_path,
        expected_publication_owner=checkpoint.publication_owner,
    )


@proof_verification_operation
def publish_subject_mask_coordinate_surfaces(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
) -> BoundSubjectMaskCoordinateSurfaces:
    path = _canonical_path(
        run_path, prefix="subject_mask_runs/", label="subject-mask rowset"
    )
    checkpoint = capture_subject_mask_coordinate_publication_checkpoint(
        root_node,
        path,
        expected_publication_owner=expected_publication_owner,
    )
    context = _load_subject_mask_coordinate_context(
        root_node,
        path,
        require_complete=False,
        expected_selector_eligible=False,
        expected_publication_owner=expected_publication_owner,
    )
    nodes = _validate_surface_metadata(context)
    companions, payloads = _validate_companion_metadata_and_values(context, nodes)

    def authorize_publication_mutation() -> Any:
        return _fresh_owned_ineligible_run(
            root_node,
            path,
            expected_publication_owner=expected_publication_owner,
            allowed_statuses=(RUN_STATUS_RUNNING,),
            label="Subject-mask coordinate publication target",
        )

    try:
        authorize_publication_mutation()
        interpretations = _bind_interpretations(
            context,
            nodes,
            companions,
            payloads,
            stamp=True,
        )
        active_run = authorize_publication_mutation()
        inventory = stamp_and_bind_persisted_coordinate_record(
            active_run,
            _inventory_record(
                context,
                nodes,
                companions,
                interpretations,
                payloads,
            ),
            attr_name=SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
        )
        active_run = authorize_publication_mutation()
        derivation = stamp_and_bind_persisted_coordinate_record(
            active_run,
            _derivation_record(
                context,
                {**nodes, **companions},
                inventory,
                interpretations,
                payloads,
            ),
            attr_name=SUBJECT_MASK_COORDINATE_DERIVATION_ATTR,
        )
        authorize_publication_mutation()
        bindings = _bindings(
            context,
            derivation,
            inventory,
            interpretations,
            load=False,
        )
        stamp_bound_canonical_coordinate_descriptors(bindings.values())
        active_run = authorize_publication_mutation()
        active_run.attrs["coordinate_contract"] = "canonical_v2"
        active_run = authorize_publication_mutation()
        if active_run.attrs.get("coordinate_contract") != "canonical_v2":
            _fail("Subject-mask coordinate contract did not persist exactly.")
        return BoundSubjectMaskCoordinateSurfaces(
            mask_probs_roi=bindings["mask_probs_roi"],
            masks_roi=bindings.get("masks_roi"),
            centroid_xy=bindings["centroid_xy"],
            bbox_xyxy=bindings["bbox_xyxy"],
            context=context,
            derivation=derivation,
            inventory=inventory,
            interpretations=interpretations,
            _verification_seal=_BOUND_SURFACES_SEAL,
        )
    except BaseException as exc:
        try:
            rollback_subject_mask_coordinate_publication(checkpoint)
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            raise SubjectMaskCoordinatePublicationError(
                "Subject-mask coordinate publication failed and rollback was incomplete: "
                f"{rollback_exc}."
            ) from exc
        raise


def _load_subject_mask_coordinate_surfaces(
    root_node: Any,
    run_path: str,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
    expected_publication_owner: str | None = None,
) -> BoundSubjectMaskCoordinateSurfaces:
    context = _load_subject_mask_coordinate_context(
        root_node,
        run_path,
        require_complete=require_complete,
        expected_selector_eligible=expected_selector_eligible,
        expected_publication_owner=expected_publication_owner,
    )
    if SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE in context._run_group.attrs:
        return _load_raw_subject_mask_coordinate_surfaces_from_receipt(context)
    nodes = _validate_surface_metadata(context)
    companions, payloads = _validate_companion_metadata_and_values(context, nodes)
    interpretations = _bind_interpretations(
        context,
        nodes,
        companions,
        payloads,
        stamp=False,
    )
    inventory = bind_persisted_coordinate_record(
        context._run_group,
        attr_name=SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
    )
    if inventory.record != _inventory_record(
        context,
        nodes,
        companions,
        interpretations,
        payloads,
    ):
        _fail("Persisted subject-mask surface inventory differs from live arrays.")
    derivation = bind_persisted_coordinate_record(
        context._run_group,
        attr_name=SUBJECT_MASK_COORDINATE_DERIVATION_ATTR,
    )
    if derivation.record != _derivation_record(
        context,
        {**nodes, **companions},
        inventory,
        interpretations,
        payloads,
    ):
        _fail("Persisted subject-mask derivation differs from live surface metadata.")
    bindings = _bindings(
        context,
        derivation,
        inventory,
        interpretations,
        load=True,
    )
    return BoundSubjectMaskCoordinateSurfaces(
        mask_probs_roi=bindings["mask_probs_roi"],
        masks_roi=bindings.get("masks_roi"),
        centroid_xy=bindings["centroid_xy"],
        bbox_xyxy=bindings["bbox_xyxy"],
        context=context,
        derivation=derivation,
        inventory=inventory,
        interpretations=interpretations,
        _verification_seal=_BOUND_SURFACES_SEAL,
    )


@proof_verification_operation
def load_persisted_subject_mask_coordinate_surfaces(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str | None = None,
) -> BoundSubjectMaskCoordinateSurfaces:
    return _load_subject_mask_coordinate_surfaces(
        root_node,
        run_path,
        require_complete=True,
        expected_selector_eligible=True,
        expected_publication_owner=expected_publication_owner,
    )


@proof_verification_operation
def load_persisted_ineligible_subject_mask_coordinate_surfaces(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str | None = None,
) -> BoundSubjectMaskCoordinateSurfaces:
    """Load a complete raw coordinate successor that remains ineligible."""

    return _load_subject_mask_coordinate_surfaces(
        root_node,
        run_path,
        require_complete=True,
        expected_selector_eligible=False,
        expected_publication_owner=expected_publication_owner,
    )


@proof_verification_operation
def _load_completed_ineligible_subject_mask_coordinate_surfaces(
    root_node: Any,
    run_path: str,
    *,
    expected_publication_owner: str | None = None,
) -> BoundSubjectMaskCoordinateSurfaces:
    return _load_subject_mask_coordinate_surfaces(
        root_node,
        run_path,
        require_complete=True,
        expected_selector_eligible=False,
        expected_publication_owner=expected_publication_owner,
    )


@proof_verification_operation
def _activate_validated_subject_mask_coordinate_surfaces(
    root_node: Any,
    run_parent: Any,
    value: BoundSubjectMaskCoordinateSurfaces,
    *,
    run_name: str,
    publication_owner_token: str,
    selector_snapshot: Mapping[str, tuple[bool, Any]],
) -> None:
    """Write selectors first and eligibility as the literal final mutation."""

    if (
        type(value) is not BoundSubjectMaskCoordinateSurfaces
        or value._seal is not _BOUND_SURFACES_SEAL
    ):
        _fail("Subject-mask activation requires sealed coordinate surfaces.")
    expected_path = f"subject_mask_runs/{run_name}"
    context = value.context
    _publication_owner(
        context._run_group,
        expected=publication_owner_token,
    )
    if (
        context.completion_status not in (RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE)
        or context.selector_eligible is not False
        or context.run_path != expected_path
        or canonical_node_path(run_parent) != "subject_mask_runs"
        or canonical_node_path(context._run_group) != expected_path
        or archive_identity(root_node) != archive_identity(run_parent)
        or archive_identity(root_node) != archive_identity(context._run_group)
    ):
        _fail(
            "Subject-mask activation requires an exact ineligible publication "
            "proof for the canonical child."
        )

    def fresh_parent() -> Any:
        parent = _node(root_node, "subject_mask_runs", label="subject-mask parent")
        if archive_identity(parent) != archive_identity(run_parent):
            _fail("Subject-mask parent changed archives during activation.")
        return parent

    active_parent = fresh_parent()
    _require_selector_snapshot_unchanged(active_parent, selector_snapshot)
    if active_parent.attrs.get("latest_pending") != str(run_name):
        _fail(
            "Canonical subject-mask activation requires the attempt-owned "
            "latest_pending selector."
        )
    current = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root_node,
        expected_path,
        expected_publication_owner=publication_owner_token,
    )
    if current.derivation.record_sha256 != value.derivation.record_sha256:
        _fail("Subject-mask coordinate publication changed before activation.")
    active_parent = fresh_parent()
    _require_selector_snapshot_unchanged(active_parent, selector_snapshot)
    if active_parent.attrs.get("latest_pending") != str(run_name):
        _fail("Subject-mask pending selector changed before activation.")
    # Close and freshly recheck the completed-child proof before the first
    # parent mutation. Proof reuse is a validation optimization and must never
    # authorize selector publication from stale evidence.
    finish_proof_verification()
    lease = _acquire_parent_publication_lease(
        active_parent,
        selector_snapshot,
        run_path=expected_path,
        publication_owner=publication_owner_token,
    )
    base_generation = int(lease["base_generation"])
    next_generation = int(lease["next_generation"])
    active_parent = fresh_parent()
    _require_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    _require_selector_snapshot_unchanged(active_parent, selector_snapshot)
    if active_parent.attrs.get("latest_pending") != str(run_name):
        _fail("Subject-mask pending selector changed after lease acquisition.")
    active_parent.attrs["latest_complete"] = str(run_name)
    active_parent = fresh_parent()
    _require_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    _require_selector_snapshot_unchanged(
        active_parent,
        selector_snapshot,
        names=("latest", "authoritative_run", "authoritative_run_provenance"),
    )
    if active_parent.attrs.get("latest_complete") != str(
        run_name
    ) or active_parent.attrs.get("latest_pending") != str(run_name):
        _fail("Subject-mask latest_complete or pending selector changed in activation.")
    active_parent.attrs["latest"] = str(run_name)
    active_parent = fresh_parent()
    _require_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    _require_selector_snapshot_unchanged(
        active_parent,
        selector_snapshot,
        names=("authoritative_run", "authoritative_run_provenance"),
    )
    if active_parent.attrs.get("latest_pending") != str(run_name):
        _fail("Subject-mask pending selector changed during activation.")
    if active_parent.attrs.get("latest_complete") != str(
        run_name
    ) or active_parent.attrs.get("latest") != str(run_name):
        _fail("Subject-mask latest selectors changed during activation.")
    del active_parent.attrs["latest_pending"]
    active_parent = fresh_parent()
    _require_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    _require_selector_snapshot_unchanged(
        active_parent,
        selector_snapshot,
        names=("authoritative_run", "authoritative_run_provenance"),
    )
    activation_run = _node(root_node, expected_path, label="subject-mask child")
    if (
        active_parent.attrs.get("latest_complete") != str(run_name)
        or active_parent.attrs.get("latest") != str(run_name)
        or "latest_pending" in active_parent.attrs
        or activation_run.attrs.get("stage_selector_eligible") is not False
        or context.publication_owner != publication_owner_token
        or _publication_owner(activation_run) != publication_owner_token
    ):
        _fail("Canonical subject-mask selectors did not persist before activation.")
    active_parent.attrs[SUBJECT_MASK_PUBLICATION_POLICY_ATTR] = (
        _SUBJECT_MASK_PUBLICATION_POLICY
    )
    active_parent = fresh_parent()
    _require_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=base_generation,
    )
    if (
        active_parent.attrs.get(SUBJECT_MASK_PUBLICATION_POLICY_ATTR)
        != _SUBJECT_MASK_PUBLICATION_POLICY
    ):
        _fail("Subject-mask publication policy did not persist exactly.")
    active_parent.attrs[SUBJECT_MASK_PUBLICATION_GENERATION_ATTR] = next_generation
    active_parent = fresh_parent()
    _require_parent_publication_lease(
        active_parent,
        lease,
        expected_generation=next_generation,
    )
    activation_run = _fresh_owned_ineligible_run(
        root_node,
        expected_path,
        expected_publication_owner=publication_owner_token,
        allowed_statuses=(RUN_STATUS_COMPLETE,),
        label="Subject-mask activation target",
    )
    if (
        active_parent.attrs.get("latest_complete") != str(run_name)
        or active_parent.attrs.get("latest") != str(run_name)
        or "latest_pending" in active_parent.attrs
    ):
        _fail("Subject-mask parent publication epoch did not persist exactly.")
    activation_run.attrs["stage_selector_eligible"] = True


__all__ = [
    "SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR",
    "SUBJECT_MASK_COMPONENT_LABELS_ATTR",
    "SUBJECT_MASK_COORDINATE_CONTEXT_ATTR",
    "SUBJECT_MASK_COORDINATE_DERIVATION_ATTR",
    "SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR",
    "SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR",
    "SUBJECT_MASK_PUBLICATION_GENERATION_ATTR",
    "SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR",
    "SUBJECT_MASK_PUBLICATION_OWNER_ATTR",
    "SUBJECT_MASK_PUBLICATION_POLICY_ATTR",
    "SUBJECT_MASK_SURFACE_INVENTORY_ATTR",
    "BoundSubjectMaskCoordinateContext",
    "BoundSubjectMaskCoordinateSurfaces",
    "SubjectMaskCoordinatePublicationCheckpoint",
    "SubjectMaskCoordinatePublicationError",
    "_activate_validated_subject_mask_coordinate_surfaces",
    "_load_completed_ineligible_subject_mask_coordinate_surfaces",
    "capture_subject_mask_coordinate_publication_checkpoint",
    "load_persisted_subject_mask_coordinate_surfaces",
    "load_persisted_ineligible_subject_mask_coordinate_surfaces",
    "load_persisted_subject_mask_crop_source",
    "prepare_subject_mask_coordinate_context",
    "publish_subject_mask_coordinate_surfaces",
    "require_direct_subject_mask_crop_pixel_source",
    "rollback_subject_mask_coordinate_publication",
    "selected_subject_mask_crop_values",
]
