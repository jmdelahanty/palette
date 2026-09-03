"""Build frame-level tail kinematics from subject-shape tail geometry."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..shared.archive_identity import archive_identity
from ..shared.coordinate_reference import canonical_node_path
from ..shared.detect_reason_codec import decode_reason_bytes
from ..shared.coordinate_frame_record import array_payload_sha256, array_values_sha256
from ..shared.json_safety import json_attr_safe
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from ..shared.tail_coordinate_publication import (
    TAIL_PUBLICATION_OWNER_ATTR,
    activate_tail_coordinate_publication,
    publish_tail_kinematics_coordinate_surfaces,
)
from ..shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_FAILED,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
    utc_now_iso,
)
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.zarr_io import open_zarr_root
from ..shared.zarr.storage_profiles import StorageProfile, get_storage_profile
from .tail_kinematics_schema import (
    TailKinematicsDimensions,
    infer_tail_kinematics_dimensions,
    stamp_tail_kinematics_array_schema,
    validate_tail_kinematics_array_schema,
)
from .tail_kinematics_storage import (
    build_tail_kinematics_storage_receipt,
    consolidate_and_validate_tail_kinematics_metadata,
    create_tail_kinematics_arrays_from_receipt,
    persist_tail_kinematics_storage_receipt,
    validate_tail_kinematics_storage_receipt,
)
from .subject_shape_io import (
    SubjectShapeIOError,
    resolve_canonical_subject_shape_run,
)
from .subject_shape_spline import tail_sample_positions

TAIL_KINEMATICS_SCHEMA_ID = "analysis.tail_kinematics_runs"
TAIL_KINEMATICS_SCHEMA_VERSION = 2
TAIL_KINEMATICS_METHOD = "tail_metrics_from_subject_shape"
TAIL_KINEMATICS_METHOD_VERSION = 1
TAIL_KINEMATICS_COMPUTE_KERNEL = "vectorized_shared_grid_v1"
TAIL_KINEMATICS_STAGE_NAME = "analysis.tail_kinematics_runs"
SOURCE_TAIL_GEOMETRY_KIND = "subject_shape_bspline_tail_resample"
DEFAULT_TAIL_ANGLE_SAMPLE_COUNT = 10
DEFAULT_BLOCK_ROWS = 16_384
DEFAULT_OUTPUT_SHARD_ROWS = 262_144
TAIL_KINEMATICS_EXECUTION_BACKENDS = frozenset({"serial", "process_shards"})
REASON_BYTES_WIDTH = 64
TAIL_PUBLICATION_TOMBSTONE_ATTR = "tail_publication_tombstone"
ROW_LINEAGE_NAMES = (
    "source_crop_row_ids",
    "instance_key",
    "source_acquisition_frame_index",
)
SUBJECT_SHAPE_BODY_ARRAY_NAMES = (
    "tail_sample_s",
    "tail_sample_xy",
    "tail_tangent_xy",
    "tail_curvature_px_inv",
    "tail_sample_valid",
    "bspline_valid",
    "tail_base_xy",
    "tail_sample_failure_reason_bytes",
    "bspline_failure_reason_bytes",
)
SUBJECT_SHAPE_BODY_FRAME_ARRAY_NAMES = (
    "forward_axis_xy",
    "left_axis_xy",
    "axis_valid",
    "failure_reason_bytes",
)
SOURCE_REVISION_ARRAY_NAMES = ("row_revision", "row_revision_available")
TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_ID = (
    "palette.tail_kinematics_staged_subject_shape_authority"
)
TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_VERSION = 1
TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCOPE = (
    "tail_kinematics_exact_digest_bound_staged_subset_only"
)
TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCHEMA_ID = (
    "palette.tail_kinematics_staged_input_integrity_receipt"
)
TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCHEMA_VERSION = 1
TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCOPE = (
    "materializer_private_tail_worker_inputs_not_coordinate_authority"
)
TAIL_KINEMATICS_WORKER_INPUT_ATTESTATION_SCHEMA_ID = (
    "palette.tail_kinematics_worker_input_attestation"
)
TAIL_KINEMATICS_WORKER_INPUT_ATTESTATION_SCHEMA_VERSION = 1
TAIL_KINEMATICS_CANDIDATE_PROFILE_ID = "published_http_v1"

_REQUIRED_SOURCE_ARRAY_PATHS = (
    "components/subject_body/tail_sample_s",
    "components/subject_body/tail_sample_xy",
    "components/subject_body/tail_tangent_xy",
    "components/subject_body/tail_curvature_px_inv",
    "components/subject_body/tail_sample_valid",
    "components/subject_body/bspline_valid",
    "components/subject_body/tail_base_xy",
    "body_frame/forward_axis_xy",
    "body_frame/left_axis_xy",
    "body_frame/axis_valid",
    "source_crop_row_ids",
    "instance_key",
    "source_acquisition_frame_index",
)
_OPTIONAL_SOURCE_ARRAY_PATHS = (
    "components/subject_body/tail_sample_failure_reason_bytes",
    "components/subject_body/bspline_failure_reason_bytes",
    "body_frame/failure_reason_bytes",
    *(f"source_refined_subject_masks/{name}" for name in SOURCE_REVISION_ARRAY_NAMES),
)
_SOURCE_CONTRACT_ATTR_NAMES = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "palette_run_completion_status",
    "source_refined_subject_masks_run",
    "body_frame_schema_id",
    "tail_geometry_schema_id",
)

_TAIL_WORKER_ROW_SOURCE_PATHS = {
    "tail_sample_xy": "components/subject_body/tail_sample_xy",
    "tail_tangent_xy": "components/subject_body/tail_tangent_xy",
    "tail_curvature_px_inv": "components/subject_body/tail_curvature_px_inv",
    "tail_sample_valid": "components/subject_body/tail_sample_valid",
    "bspline_valid": "components/subject_body/bspline_valid",
    "tail_base_xy": "components/subject_body/tail_base_xy",
    "body_forward_axis_xy": "body_frame/forward_axis_xy",
    "body_left_axis_xy": "body_frame/left_axis_xy",
    "body_frame_valid": "body_frame/axis_valid",
    "tail_sample_failure_reason": (
        "components/subject_body/tail_sample_failure_reason_bytes"
    ),
    "bspline_failure_reason": ("components/subject_body/bspline_failure_reason_bytes"),
    "body_frame_failure_reason": "body_frame/failure_reason_bytes",
}
_TAIL_WORKER_STATIC_SOURCE_PATHS = {
    "source_tail_sample_s": "components/subject_body/tail_sample_s",
}


@dataclass(frozen=True)
class TailKinematicsBatch:
    """Computed tail-kinematics arrays for one row-aligned batch."""

    tail_angle_sample_s: np.ndarray
    tail_angle_sample_xy: np.ndarray
    tail_angle_rad: np.ndarray
    tail_angle_deg: np.ndarray
    tail_tip_angle_rad: np.ndarray
    tail_tip_angle_deg: np.ndarray
    tail_lateral_deflection_px: np.ndarray
    tail_tip_lateral_deflection_px: np.ndarray
    max_abs_tail_angle_rad: np.ndarray
    max_abs_tail_angle_deg: np.ndarray
    tail_angle_rms_rad: np.ndarray
    tail_angle_rms_deg: np.ndarray
    integrated_abs_tail_angle_rad: np.ndarray
    tail_curvature_px_inv: np.ndarray
    max_abs_tail_curvature_px_inv: np.ndarray
    integrated_abs_tail_curvature: np.ndarray
    valid: np.ndarray
    failure_reason: np.ndarray
    failure_reason_bytes: np.ndarray


@dataclass(frozen=True)
class TailKinematicsSources:
    """Lazy source-array handles for one subject-shape run."""

    source_tail_sample_s: np.ndarray
    arrays: Mapping[str, Any]
    reason_arrays: Mapping[str, Any | None]
    row_count: int
    source_sample_count: int
    source_publication_manifest_sha256: str
    source_authority_mode: str
    source_authority: Mapping[str, Any] = field(repr=False)
    staged_input_integrity_receipt: Mapping[str, Any] | None = field(
        repr=False,
    )
    shape_group: Any = field(repr=False, compare=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"tail_kinematics_{stamp}"


_json_safe = json_attr_safe


def _canonical_json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            json_attr_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        json_attr_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_contract_attrs(shape_group: Any) -> dict[str, Any]:
    return _canonical_json_copy(
        {
            name: shape_group.attrs.get(name)
            for name in _SOURCE_CONTRACT_ATTR_NAMES
            if name in shape_group.attrs
        }
    )


def _build_staged_source_authority(
    shape_group: Any,
    *,
    run_name: str,
    row_count: int,
    source_sample_count: int,
    publication: Any,
) -> dict[str, Any]:
    """Build a closed detached receipt from an already verified publication."""

    manifest_arrays = publication.manifest.record.get("arrays")
    if not isinstance(manifest_arrays, Mapping):
        raise SubjectShapeIOError(
            "Canonical subject-shape publication lacks a closed array manifest."
        )
    curvature = publication.require_scalar_surface(
        "components/subject_body/tail_curvature_px_inv",
        units="px^-1",
        surface_kind="row_profile",
    )
    allowed_arrays: dict[str, Any] = {}
    for relative_ref in (*_REQUIRED_SOURCE_ARRAY_PATHS, *_OPTIONAL_SOURCE_ARRAY_PATHS):
        node = shape_group.get(relative_ref)
        if node is None:
            if relative_ref in _REQUIRED_SOURCE_ARRAY_PATHS:
                raise SubjectShapeIOError(
                    f"Canonical subject-shape source is missing required array {relative_ref!r}."
                )
            continue
        manifest_entry = manifest_arrays.get(relative_ref)
        if not isinstance(manifest_entry, Mapping):
            raise SubjectShapeIOError(
                f"Canonical subject-shape manifest does not bind staged array {relative_ref!r}."
            )
        if str(manifest_entry.get("relative_ref")) != relative_ref:
            raise SubjectShapeIOError(
                f"Canonical array-manifest reference for {relative_ref!r} is inconsistent."
            )
        allowed_arrays[relative_ref] = _canonical_json_copy(manifest_entry)

    record = {
        "schema_id": TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_ID,
        "schema_version": TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCOPE,
        "source_subject_shape_run": str(run_name),
        "source_subject_shape_run_ref": f"/analysis/subject_shape_runs/{run_name}",
        "row_count": int(row_count),
        "source_sample_count": int(source_sample_count),
        "canonical_publication": {
            "manifest_ref": publication.manifest.record_ref,
            "manifest_sha256": publication.manifest.record_sha256,
            "row_identity_ref": publication.row_identity.record_ref,
            "row_identity_sha256": publication.row_identity.record_sha256,
            "tail_sample_axis_ref": publication.tail_sample_axis.record_ref,
            "tail_sample_axis_sha256": publication.tail_sample_axis.record_sha256,
            "tail_curvature_semantics_ref": curvature.semantics.record_ref,
            "tail_curvature_semantics_sha256": curvature.semantics.record_sha256,
            "body_frame_ref": publication.body_frame.record_ref,
            "body_frame_sha256": publication.body_frame.record_sha256,
        },
        "source_contract_attrs": _source_contract_attrs(shape_group),
        "allowed_arrays": allowed_arrays,
        "closed_array_inventory": True,
        "normal_reader_authority": False,
    }
    return {
        **record,
        "record_sha256": _canonical_sha256(record),
    }


def _validated_staged_source_authority(
    shape_group: Any,
    *,
    run_name: str,
    authority: Mapping[str, Any],
    verify_payload: bool = True,
) -> dict[str, Any]:
    """Validate one explicit materializer-only detached source receipt."""

    if not isinstance(authority, Mapping):
        raise SubjectShapeIOError("Staged subject-shape authority must be a mapping.")
    canonical = _canonical_json_copy(authority)
    digest = canonical.pop("record_sha256", None)
    if not _is_sha256(digest) or digest != _canonical_sha256(canonical):
        raise SubjectShapeIOError(
            "Staged subject-shape authority digest is missing or stale."
        )
    if canonical.get("schema_id") != TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_ID:
        raise SubjectShapeIOError("Unsupported staged subject-shape authority schema.")
    if (
        canonical.get("schema_version")
        != TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_VERSION
    ):
        raise SubjectShapeIOError(
            "Unsupported staged subject-shape authority schema version."
        )
    if (
        canonical.get("authority_scope")
        != TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCOPE
    ):
        raise SubjectShapeIOError("Staged subject-shape authority has the wrong scope.")
    if canonical.get("normal_reader_authority") is not False:
        raise SubjectShapeIOError(
            "Detached staging receipts cannot grant normal reader authority."
        )
    if canonical.get("closed_array_inventory") is not True:
        raise SubjectShapeIOError("Staged subject-shape array inventory is not closed.")
    expected_ref = f"/analysis/subject_shape_runs/{run_name}"
    if (
        canonical.get("source_subject_shape_run") != run_name
        or canonical.get("source_subject_shape_run_ref") != expected_ref
    ):
        raise SubjectShapeIOError(
            "Staged subject-shape authority names a different source run."
        )
    publication = canonical.get("canonical_publication")
    required_publication_digests = (
        "manifest_sha256",
        "row_identity_sha256",
        "tail_sample_axis_sha256",
        "tail_curvature_semantics_sha256",
        "body_frame_sha256",
    )
    if not isinstance(publication, Mapping) or any(
        not _is_sha256(publication.get(name)) for name in required_publication_digests
    ):
        raise SubjectShapeIOError(
            "Staged authority lacks exact canonical-publication proof digests."
        )
    if canonical.get("source_contract_attrs") != _source_contract_attrs(shape_group):
        raise SubjectShapeIOError(
            "Staged source contract attrs differ from the canonical receipt."
        )

    allowed = canonical.get("allowed_arrays")
    if not isinstance(allowed, Mapping):
        raise SubjectShapeIOError(
            "Staged subject-shape authority lacks its array inventory."
        )
    allowed_names = set(str(name) for name in allowed)
    supported_names = set(
        (*_REQUIRED_SOURCE_ARRAY_PATHS, *_OPTIONAL_SOURCE_ARRAY_PATHS)
    )
    missing = set(_REQUIRED_SOURCE_ARRAY_PATHS) - allowed_names
    unsupported = allowed_names - supported_names
    if missing:
        raise SubjectShapeIOError(
            f"Staged subject-shape authority omits required arrays: {sorted(missing)!r}."
        )
    if unsupported:
        raise SubjectShapeIOError(
            f"Staged subject-shape authority grants unsupported arrays: {sorted(unsupported)!r}."
        )

    for relative_ref in sorted(allowed_names):
        declared = allowed.get(relative_ref)
        node = shape_group.get(relative_ref)
        if not isinstance(declared, Mapping) or node is None:
            raise SubjectShapeIOError(
                f"Staged source array {relative_ref!r} is missing from receipt or subset."
            )
        expected_shape = declared.get("shape")
        if (
            declared.get("relative_ref") != relative_ref
            or declared.get("canonicalization") != "numpy_dtype_shape_c_order_bytes_v1"
            or not isinstance(expected_shape, list)
            or any(type(value) is not int or value < 0 for value in expected_shape)
            or declared.get("dtype") != np.dtype(node.dtype).str
            or expected_shape != [int(value) for value in node.shape]
            or not _is_sha256(declared.get("content_sha256"))
        ):
            raise SubjectShapeIOError(
                f"Staged source metadata for {relative_ref!r} differs from its receipt."
            )
        if verify_payload:
            try:
                observed_digest = array_payload_sha256(node)
            except Exception as exc:
                raise SubjectShapeIOError(
                    f"Staged source array {relative_ref!r} could not be verified: {exc}"
                ) from exc
            if observed_digest != declared.get("content_sha256"):
                raise SubjectShapeIOError(
                    f"Staged source array {relative_ref!r} differs from its canonical payload."
                )

    return {**canonical, "record_sha256": str(digest)}


def _staged_array_digest_header(dtype: np.dtype[Any], shape: tuple[int, ...]) -> Any:
    header = {
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": np.lib.format.dtype_to_descr(dtype),
        "shape": [int(value) for value in shape],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\x00")
    return digest


def _scan_staged_source_array_once(
    relative_ref: str,
    node: Any,
    *,
    row_count: int,
    chunk_rows: int,
    retain_worker_chunks: bool,
) -> dict[str, Any]:
    """Read one staged source array once into whole-array and chunk evidence."""

    dtype = np.dtype(node.dtype)
    shape = tuple(int(value) for value in node.shape)
    if dtype.hasobject:
        raise SubjectShapeIOError(
            f"Staged source array {relative_ref!r} has no deterministic payload grammar."
        )
    full_digest = _staged_array_digest_header(dtype, shape)
    chunks: list[dict[str, Any]] = []
    row_aligned = bool(shape and shape[0] == int(row_count))
    if row_aligned:
        trailing = (slice(None),) * (len(shape) - 1)
        for chunk_index, start in enumerate(range(0, shape[0], int(chunk_rows))):
            stop = min(shape[0], start + int(chunk_rows))
            values = np.ascontiguousarray(node[(slice(start, stop), *trailing)])
            if values.dtype != dtype or values.shape != (stop - start, *shape[1:]):
                raise SubjectShapeIOError(
                    f"Staged source array {relative_ref!r} changed during receipt scan."
                )
            full_digest.update(values.tobytes(order="C"))
            if retain_worker_chunks:
                chunks.append(
                    {
                        "chunk_index": int(chunk_index),
                        "start_row": int(start),
                        "stop_row": int(stop),
                        "dtype": dtype.str,
                        "shape": [int(value) for value in values.shape],
                        "canonicalization": ("numpy_dtype_shape_c_order_bytes_v1"),
                        "content_sha256": array_values_sha256(values),
                    }
                )
    else:
        values = np.ascontiguousarray(node[...])
        if values.dtype != dtype or values.shape != shape:
            raise SubjectShapeIOError(
                f"Staged source array {relative_ref!r} changed during receipt scan."
            )
        full_digest.update(values.tobytes(order="C"))
    if (
        np.dtype(node.dtype) != dtype
        or tuple(int(value) for value in node.shape) != shape
    ):
        raise SubjectShapeIOError(
            f"Staged source array {relative_ref!r} metadata changed during receipt scan."
        )
    return {
        "relative_ref": relative_ref,
        "dtype": dtype.str,
        "shape": [int(value) for value in shape],
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "content_sha256": full_digest.hexdigest(),
        "row_aligned": row_aligned,
        "worker_chunks": chunks,
    }


def _canonical_staged_input_chunk(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SubjectShapeIOError("Staged tail input chunk must be a mapping.")
    canonical = _canonical_json_copy(value)
    digest = canonical.pop("record_sha256", None)
    if (
        set(canonical) != {"chunk_index", "start_row", "stop_row", "source_arrays"}
        or type(canonical.get("chunk_index")) is not int
        or type(canonical.get("start_row")) is not int
        or type(canonical.get("stop_row")) is not int
        or not (0 <= canonical["start_row"] < canonical["stop_row"])
        or not isinstance(canonical.get("source_arrays"), Mapping)
        or not _is_sha256(digest)
        or digest != _canonical_sha256(canonical)
    ):
        raise SubjectShapeIOError(
            "Staged tail input chunk receipt is malformed or stale."
        )
    return {**canonical, "record_sha256": str(digest)}


def _canonical_staged_input_integrity_receipt(
    shape_group: Any,
    *,
    run_name: str,
    authority: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one staged-input receipt using metadata and sealed digests only."""

    staged_authority = _validated_staged_source_authority(
        shape_group,
        run_name=run_name,
        authority=authority,
        verify_payload=False,
    )
    if not isinstance(receipt, Mapping):
        raise SubjectShapeIOError(
            "Staged tail input integrity receipt must be a mapping."
        )
    canonical = _canonical_json_copy(receipt)
    digest = canonical.pop("record_sha256", None)
    expected_fields = {
        "schema_id",
        "schema_version",
        "integrity_scope",
        "receipt_role",
        "source_subject_shape_run",
        "staged_source_authority_sha256",
        "row_count",
        "source_sample_count",
        "requested_chunk_rows",
        "source_arrays",
        "worker_row_input_paths",
        "worker_static_input_paths",
        "worker_static_inputs",
        "chunks",
        "closed_source_array_inventory",
        "normal_reader_authority",
        "coordinate_authority",
    }
    if (
        set(canonical) != expected_fields
        or canonical.get("schema_id")
        != TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCHEMA_ID
        or canonical.get("schema_version")
        != TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCHEMA_VERSION
        or canonical.get("integrity_scope")
        != TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCOPE
        or canonical.get("receipt_role")
        != "materializer_private_integrity_not_coordinate_authority"
        or canonical.get("source_subject_shape_run") != run_name
        or canonical.get("staged_source_authority_sha256")
        != staged_authority["record_sha256"]
        or canonical.get("closed_source_array_inventory") is not True
        or canonical.get("normal_reader_authority") is not False
        or canonical.get("coordinate_authority") is not False
        or not _is_sha256(digest)
        or digest != _canonical_sha256(canonical)
    ):
        raise SubjectShapeIOError(
            "Staged tail input integrity receipt is unsupported, stale, or unbound."
        )
    row_count = canonical.get("row_count")
    source_sample_count = canonical.get("source_sample_count")
    chunk_rows = canonical.get("requested_chunk_rows")
    if (
        type(row_count) is not int
        or row_count < 0
        or type(source_sample_count) is not int
        or source_sample_count < 2
        or type(chunk_rows) is not int
        or chunk_rows <= 0
        or row_count != staged_authority.get("row_count")
        or source_sample_count != staged_authority.get("source_sample_count")
    ):
        raise SubjectShapeIOError(
            "Staged tail input receipt cardinality differs from its authority."
        )
    allowed = staged_authority["allowed_arrays"]
    source_arrays = canonical.get("source_arrays")
    if not isinstance(source_arrays, Mapping) or set(source_arrays) != set(allowed):
        raise SubjectShapeIOError(
            "Staged tail input receipt source-array inventory is not closed."
        )
    for relative_ref in sorted(source_arrays):
        record = source_arrays[relative_ref]
        declared = allowed[relative_ref]
        node = shape_group.get(relative_ref)
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {"relative_ref", "dtype", "shape", "canonicalization", "content_sha256"}
            or node is None
            or record.get("relative_ref") != relative_ref
            or record.get("dtype") != np.dtype(node.dtype).str
            or record.get("shape") != [int(value) for value in node.shape]
            or record.get("canonicalization") != "numpy_dtype_shape_c_order_bytes_v1"
            or record.get("content_sha256") != declared.get("content_sha256")
        ):
            raise SubjectShapeIOError(
                f"Staged tail input receipt differs for source array {relative_ref!r}."
            )
    expected_static_paths = list(sorted(_TAIL_WORKER_STATIC_SOURCE_PATHS.values()))
    expected_row_paths = list(
        sorted(
            path for path in _TAIL_WORKER_ROW_SOURCE_PATHS.values() if path in allowed
        )
    )
    if (
        canonical.get("worker_row_input_paths") != expected_row_paths
        or canonical.get("worker_static_input_paths") != expected_static_paths
        or canonical.get("worker_static_inputs")
        != {path: source_arrays[path] for path in expected_static_paths}
    ):
        raise SubjectShapeIOError(
            "Staged tail worker input inventory differs from the maintained kernel."
        )
    chunks = canonical.get("chunks")
    if not isinstance(chunks, list):
        raise SubjectShapeIOError("Staged tail input chunk inventory must be a list.")
    canonical_chunks: list[dict[str, Any]] = []
    cursor = 0
    for expected_index, raw_chunk in enumerate(chunks):
        chunk = _canonical_staged_input_chunk(raw_chunk)
        start = chunk["start_row"]
        stop = chunk["stop_row"]
        if (
            chunk["chunk_index"] != expected_index
            or start != cursor
            or stop > row_count
            or stop - start > chunk_rows
            or (stop < row_count and stop - start != chunk_rows)
            or set(chunk["source_arrays"]) != set(expected_row_paths)
        ):
            raise SubjectShapeIOError(
                "Staged tail input chunks have a gap, overlap, or wrong inventory."
            )
        for relative_ref in expected_row_paths:
            record = chunk["source_arrays"][relative_ref]
            full = source_arrays[relative_ref]
            if (
                not isinstance(record, Mapping)
                or set(record)
                != {"dtype", "shape", "canonicalization", "content_sha256"}
                or record.get("dtype") != full["dtype"]
                or record.get("shape") != [stop - start, *full["shape"][1:]]
                or record.get("canonicalization")
                != "numpy_dtype_shape_c_order_bytes_v1"
                or not _is_sha256(record.get("content_sha256"))
            ):
                raise SubjectShapeIOError(
                    f"Staged tail chunk declaration differs for {relative_ref!r}."
                )
        cursor = stop
        canonical_chunks.append(chunk)
    if cursor != row_count or (row_count == 0 and canonical_chunks):
        raise SubjectShapeIOError(
            "Staged tail input chunks do not cover every row exactly once."
        )
    canonical["chunks"] = canonical_chunks
    return {**canonical, "record_sha256": str(digest)}


def build_tail_kinematics_staged_input_integrity_receipt(
    shape_group: Any,
    *,
    run_name: str,
    authority: Mapping[str, Any],
    chunk_rows: int,
    read_workers: int = 4,
) -> dict[str, Any]:
    """Scan staged inputs once and seal the chunks workers will consume."""

    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("Tail staged-input receipt chunk_rows must be positive.")
    if type(read_workers) is not int or read_workers <= 0:
        raise ValueError("Tail staged-input receipt read_workers must be positive.")
    staged_authority = _validated_staged_source_authority(
        shape_group,
        run_name=run_name,
        authority=authority,
        verify_payload=False,
    )
    row_count = int(staged_authority["row_count"])
    worker_paths = set(_TAIL_WORKER_ROW_SOURCE_PATHS.values())
    allowed = staged_authority["allowed_arrays"]
    effective_workers = min(int(read_workers), max(1, len(allowed)))
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(
                _scan_staged_source_array_once,
                relative_ref,
                shape_group[relative_ref],
                row_count=row_count,
                chunk_rows=int(chunk_rows),
                retain_worker_chunks=relative_ref in worker_paths,
            )
            for relative_ref in sorted(allowed)
        ]
        results = sorted(
            (future.result() for future in futures),
            key=lambda item: str(item["relative_ref"]),
        )
    source_arrays: dict[str, Any] = {}
    result_by_path: dict[str, Mapping[str, Any]] = {}
    for result in results:
        relative_ref = str(result["relative_ref"])
        if result["content_sha256"] != allowed[relative_ref]["content_sha256"]:
            raise SubjectShapeIOError(
                f"Staged source array {relative_ref!r} differs from its canonical payload."
            )
        source_arrays[relative_ref] = {
            key: result[key]
            for key in (
                "relative_ref",
                "dtype",
                "shape",
                "canonicalization",
                "content_sha256",
            )
        }
        result_by_path[relative_ref] = result
    worker_row_paths = list(
        sorted(path for path in worker_paths if path in source_arrays)
    )
    worker_static_paths = list(sorted(_TAIL_WORKER_STATIC_SOURCE_PATHS.values()))
    chunk_records: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, row_count, int(chunk_rows))):
        stop = min(row_count, start + int(chunk_rows))
        chunk_arrays: dict[str, Any] = {}
        for relative_ref in worker_row_paths:
            leaves = result_by_path[relative_ref]["worker_chunks"]
            leaf = leaves[chunk_index]
            if leaf["start_row"] != start or leaf["stop_row"] != stop:
                raise RuntimeError(
                    "Tail staged-input receipt scanner produced inconsistent chunks."
                )
            chunk_arrays[relative_ref] = {
                key: leaf[key]
                for key in ("dtype", "shape", "canonicalization", "content_sha256")
            }
        chunk_body = {
            "chunk_index": int(chunk_index),
            "start_row": int(start),
            "stop_row": int(stop),
            "source_arrays": chunk_arrays,
        }
        chunk_records.append(
            {**chunk_body, "record_sha256": _canonical_sha256(chunk_body)}
        )
    body = {
        "schema_id": TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCHEMA_ID,
        "schema_version": TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCHEMA_VERSION,
        "integrity_scope": TAIL_KINEMATICS_STAGED_INPUT_INTEGRITY_SCOPE,
        "receipt_role": "materializer_private_integrity_not_coordinate_authority",
        "source_subject_shape_run": str(run_name),
        "staged_source_authority_sha256": staged_authority["record_sha256"],
        "row_count": row_count,
        "source_sample_count": int(staged_authority["source_sample_count"]),
        "requested_chunk_rows": int(chunk_rows),
        "source_arrays": source_arrays,
        "worker_row_input_paths": worker_row_paths,
        "worker_static_input_paths": worker_static_paths,
        "worker_static_inputs": {
            path: source_arrays[path] for path in worker_static_paths
        },
        "chunks": chunk_records,
        "closed_source_array_inventory": True,
        "normal_reader_authority": False,
        "coordinate_authority": False,
    }
    receipt = {**body, "record_sha256": _canonical_sha256(body)}
    return _canonical_staged_input_integrity_receipt(
        shape_group,
        run_name=run_name,
        authority=staged_authority,
        receipt=receipt,
    )


def _encode_reasons(
    reasons: Sequence[object], *, width: int = REASON_BYTES_WIDTH
) -> np.ndarray:
    out = np.zeros((len(reasons), int(width)), dtype=np.uint8)
    for idx, reason in enumerate(reasons):
        payload = str(reason or "").encode("utf-8", errors="replace")[
            : max(0, int(width) - 1)
        ]
        if payload:
            out[int(idx), : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return out


def _set_reason_bytes_attrs(
    group: zarr.Group, *, width: int = REASON_BYTES_WIDTH
) -> None:
    group.attrs["reason_encoding"] = "utf8-null-terminated"
    group.attrs["reason_bytes_width"] = int(width)
    group.attrs["reason_bytes_null_terminated"] = True


def _metric_chunks(total_rows: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows),)


def _metric_chunks_lastdim(total_rows: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(width))


def _metric_chunks_3d(total_rows: int, middle: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(middle), int(width))


def _write_array(
    group: zarr.Group,
    name: str,
    data: np.ndarray,
    *,
    chunks: Optional[Sequence[int]] = None,
) -> None:
    if name in group:
        del group[name]
    kwargs: dict[str, object] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = tuple(int(value) for value in chunks)
    group.create_array(name, **kwargs)


def _create_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    shards: Sequence[int] | None = None,
    fill_value: object | None = None,
) -> Any:
    if name in group:
        del group[name]
    kwargs: dict[str, object] = {
        "shape": tuple(int(value) for value in shape),
        "dtype": dtype,
        "chunks": tuple(int(value) for value in chunks),
        "overwrite": True,
    }
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    if shards is not None:
        kwargs["shards"] = tuple(int(value) for value in shards)
    return group.create_array(name, **kwargs)


def _source_chunks(source: Any, shape: Sequence[int]) -> tuple[int, ...]:
    raw_chunks = getattr(source, "chunks", None)
    dims = tuple(int(value) for value in shape)
    if raw_chunks is None:
        if not dims:
            return ()
        return tuple(max(1, min(int(dim), 256)) for dim in dims)
    if isinstance(raw_chunks, int):
        chunks = (int(raw_chunks),)
    else:
        chunks = tuple(int(value) for value in raw_chunks)
    if len(chunks) != len(dims):
        raise ValueError(f"Source chunks {chunks!r} do not match array shape {dims!r}.")
    return tuple(
        max(1, min(int(chunk), int(dim))) if int(dim) > 0 else 1
        for chunk, dim in zip(chunks, dims)
    )


def _effective_block_rows(*, row_count: int, requested_block_rows: int) -> int:
    requested = int(requested_block_rows)
    if requested <= 0:
        raise ValueError("block_rows must be positive.")
    output_row_chunk = int(_metric_chunks(int(row_count))[0])
    aligned = max(
        output_row_chunk,
        ((requested + output_row_chunk - 1) // output_row_chunk) * output_row_chunk,
    )
    return min(aligned, int(row_count)) if int(row_count) > 0 else aligned


def _effective_output_shard_rows(
    *,
    row_count: int,
    requested_output_shard_rows: int,
) -> int:
    requested = int(requested_output_shard_rows)
    if requested <= 0:
        raise ValueError("output_shard_rows must be positive.")
    output_row_chunk = int(_metric_chunks(int(row_count))[0])
    aligned_requested = max(
        output_row_chunk,
        ((requested + output_row_chunk - 1) // output_row_chunk) * output_row_chunk,
    )
    if int(row_count) <= 0:
        return aligned_requested
    aligned_extent = (
        (int(row_count) + output_row_chunk - 1) // output_row_chunk
    ) * output_row_chunk
    return min(aligned_requested, aligned_extent)


def _output_shards(
    chunks: Sequence[int],
    *,
    shard_rows: int | None,
    dtype: object,
) -> tuple[int, ...] | None:
    if shard_rows is None:
        return None
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return None
    if kind in {"O", "S", "U"}:
        return None
    chunk_shape = tuple(int(value) for value in chunks)
    if not chunk_shape:
        return None
    outer_rows = max(
        int(chunk_shape[0]),
        ((int(shard_rows) + int(chunk_shape[0]) - 1) // int(chunk_shape[0]))
        * int(chunk_shape[0]),
    )
    return (outer_rows, *chunk_shape[1:])


def _iter_row_slices(row_count: int, block_rows: int) -> Sequence[slice]:
    return tuple(
        slice(start, min(int(row_count), start + int(block_rows)))
        for start in range(0, int(row_count), int(block_rows))
    )


def _validate_process_shard_slices(
    row_slices: Sequence[slice],
    *,
    row_count: int,
    output_row_chunk: int,
    output_shard_rows: int,
) -> None:
    """Prove that each process owns complete, non-overlapping output storage."""

    previous_stop = 0
    for row_slice in row_slices:
        start = int(row_slice.start or 0)
        stop = int(row_slice.stop or 0)
        if start != previous_stop or stop <= start:
            raise ValueError(
                "Worker row slices must be contiguous and non-overlapping."
            )
        if start % int(output_row_chunk) != 0:
            raise ValueError("Worker row slices must start on the output chunk grid.")
        if start % int(output_shard_rows) != 0:
            raise ValueError("Worker row slices must start on the output shard grid.")
        if stop != int(row_count):
            if stop - start != int(output_shard_rows):
                raise ValueError(
                    "Non-final workers must own one complete output shard."
                )
            if stop % int(output_row_chunk) != 0:
                raise ValueError("Non-final workers must end on the output chunk grid.")
        previous_stop = stop
    if previous_stop != int(row_count):
        raise ValueError("Worker row slices do not cover the complete row axis.")


def _copy_array_bounded(
    target_group: zarr.Group,
    name: str,
    source: Any,
    *,
    block_rows: int,
    row_aligned_count: int | None = None,
    shard_rows: int | None = None,
    precreated: bool = False,
) -> Any:
    shape = tuple(int(value) for value in source.shape)
    if row_aligned_count is not None:
        if not shape or int(shape[0]) != int(row_aligned_count):
            raise ValueError(
                f"Source array {name!r} has shape {shape!r}; expected first axis {int(row_aligned_count)}."
            )
    if precreated:
        target = target_group.get(name)
        if target is None:
            raise ValueError(f"Precreated candidate array {name!r} is missing.")
        if tuple(int(value) for value in target.shape) != shape:
            raise ValueError(
                f"Precreated candidate array {name!r} has the wrong shape."
            )
        if np.dtype(target.dtype) != np.dtype(source.dtype):
            raise ValueError(
                f"Precreated candidate array {name!r} has the wrong dtype."
            )
    else:
        chunks = _source_chunks(source, shape)
        target = _create_array(
            target_group,
            name,
            shape=shape,
            dtype=source.dtype,
            chunks=chunks,
            shards=_output_shards(chunks, shard_rows=shard_rows, dtype=source.dtype),
        )
    if not shape:
        target[...] = np.asarray(source[...])
    elif row_aligned_count is not None:
        for row_slice in _iter_row_slices(int(row_aligned_count), int(block_rows)):
            target[row_slice] = np.asarray(source[row_slice])
    else:
        target[:] = np.asarray(source[:])
    return target


def _copy_optional_source_revision_snapshot(
    target_run: zarr.Group,
    shape_group: zarr.Group,
    *,
    shape_run_name: str,
    row_count: int,
    block_rows: int,
    output_shard_rows: int,
    precreated: bool = False,
) -> bool:
    source = shape_group.get("source_refined_subject_masks")
    if not isinstance(source, zarr.Group):
        target_run.attrs["source_refined_subject_masks_revision_snapshot"] = False
        return False

    present = {
        name for name in SOURCE_REVISION_ARRAY_NAMES if source.get(name) is not None
    }
    if not present:
        target_run.attrs["source_refined_subject_masks_revision_snapshot"] = False
        return False
    if precreated and present != set(SOURCE_REVISION_ARRAY_NAMES):
        raise ValueError(
            "Candidate source-revision snapshot must contain its complete optional bundle."
        )
    target = target_run.require_group("source_refined_subject_masks")
    for key, value in source.attrs.items():
        target.attrs[str(key)] = _json_safe(value)
    target.attrs["copied_from_subject_shape_run"] = str(shape_run_name)
    target.attrs["snapshot_semantics"] = (
        "refined mask row revisions copied from the source subject-shape run used by this tail-kinematics run"
    )
    copied = []
    for name in ("row_revision", "row_revision_available"):
        arr = source.get(name)
        if arr is None:
            continue
        _copy_array_bounded(
            target,
            name,
            arr,
            block_rows=int(block_rows),
            row_aligned_count=int(row_count) if name == "row_revision" else None,
            shard_rows=int(output_shard_rows) if name == "row_revision" else None,
            precreated=precreated,
        )
        copied.append(name)
    target.attrs["copied_arrays"] = copied
    target_run.attrs["source_refined_subject_masks_revision_snapshot"] = bool(copied)
    return bool(copied)


def _read_optional_reason_labels(arr: Any | None, row_slice: slice) -> np.ndarray:
    row_count = int((row_slice.stop or 0) - (row_slice.start or 0))
    if arr is None:
        return np.full((int(row_count),), "", dtype=object)
    data = np.asarray(arr[row_slice])
    if data.ndim == 2 and np.issubdtype(data.dtype, np.integer):
        decoded = decode_reason_bytes(data)
    else:
        decoded = np.asarray(data, dtype=object).reshape(-1)
    if int(decoded.shape[0]) != int(row_count):
        raise ValueError(
            f"Failure-reason slice has {decoded.shape[0]} rows; expected {row_count}."
        )
    return decoded


def _normalize_vectors(vectors_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors_xy, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1)
    normalized = np.full(vectors.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(norms) & (norms > 1e-12)
    normalized[valid] = vectors[valid] / norms[valid, None]
    return normalized, valid


def _interpolation_plan(
    source_s: np.ndarray, target_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared-grid interpolation indices and weights matching ``np.interp`` endpoints."""

    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    insertion = np.searchsorted(source, target, side="left")
    right = np.clip(insertion, 0, int(source.shape[0]) - 1).astype(np.intp)
    left = np.clip(right - 1, 0, int(source.shape[0]) - 1).astype(np.intp)

    exact = (insertion < int(source.shape[0])) & (source[right] == target)
    left[exact] = right[exact]
    below = target <= source[0]
    above = target >= source[-1]
    left[below] = 0
    right[below] = 0
    left[above] = int(source.shape[0]) - 1
    right[above] = int(source.shape[0]) - 1

    weights = np.zeros(target.shape, dtype=np.float64)
    interpolated = left != right
    weights[interpolated] = (target[interpolated] - source[left[interpolated]]) / (
        source[right[interpolated]] - source[left[interpolated]]
    )
    return left, right, weights


def _interp_rows_2d(
    source_s: np.ndarray,
    values: np.ndarray,
    target_s: np.ndarray,
    row_valid: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("2D row interpolation values must have shape (N, S, D).")
    valid_rows = np.asarray(row_valid, dtype=bool).reshape(-1)
    if int(valid_rows.shape[0]) != int(data.shape[0]):
        raise ValueError(
            "row_valid must have the same row count as interpolation values."
        )
    left, right, weights = _interpolation_plan(source, target)
    lower = data[:, left, :]
    upper = data[:, right, :]
    values_interp = lower + (upper - lower) * weights[None, :, None]
    eligible = valid_rows & np.all(np.isfinite(data), axis=(1, 2))
    out = np.full(
        (int(data.shape[0]), int(target.shape[0]), int(data.shape[2])),
        np.nan,
        dtype=np.float32,
    )
    out[eligible] = values_interp[eligible].astype(np.float32)
    return out


def _interp_rows_1d(
    source_s: np.ndarray,
    values: np.ndarray,
    target_s: np.ndarray,
    row_valid: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("1D row interpolation values must have shape (N, S).")
    valid_rows = np.asarray(row_valid, dtype=bool).reshape(-1)
    if int(valid_rows.shape[0]) != int(data.shape[0]):
        raise ValueError(
            "row_valid must have the same row count as interpolation values."
        )
    left, right, weights = _interpolation_plan(source, target)
    lower = data[:, left]
    upper = data[:, right]
    values_interp = lower + (upper - lower) * weights[None, :]
    eligible = valid_rows & np.all(np.isfinite(data), axis=1)
    out = np.full((int(data.shape[0]), int(target.shape[0])), np.nan, dtype=np.float32)
    out[eligible] = values_interp[eligible].astype(np.float32)
    return out


def _reason_values(
    values: Optional[Sequence[object]],
    *,
    row_count: int,
    fallback: str,
) -> np.ndarray:
    if values is None:
        return np.full((int(row_count),), str(fallback), dtype=object)
    reasons = np.asarray(values, dtype=object).reshape(-1)
    if int(reasons.shape[0]) != int(row_count):
        raise ValueError(
            f"Failure-reason array has {reasons.shape[0]} rows; expected {row_count}."
        )
    return reasons


def _assign_failure_reasons(
    target: np.ndarray,
    mask: np.ndarray,
    source: np.ndarray,
    *,
    fallback: str,
) -> None:
    for row_idx in np.flatnonzero(mask):
        reason = str(source[row_idx] or fallback)
        target[row_idx] = reason if reason != "ok" else fallback


def compute_tail_kinematics_from_subject_shape_arrays(
    *,
    source_tail_sample_s: np.ndarray,
    tail_sample_xy: np.ndarray,
    tail_tangent_xy: np.ndarray,
    tail_curvature_px_inv: np.ndarray,
    tail_sample_valid: np.ndarray,
    bspline_valid: np.ndarray,
    tail_base_xy: np.ndarray,
    body_forward_axis_xy: np.ndarray,
    body_left_axis_xy: np.ndarray,
    body_frame_valid: np.ndarray,
    tail_sample_failure_reason: Optional[Sequence[object]] = None,
    bspline_failure_reason: Optional[Sequence[object]] = None,
    body_frame_failure_reason: Optional[Sequence[object]] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
) -> TailKinematicsBatch:
    """Compute signed body-frame tail angles from subject-shape tail samples.

    The source tangents are expected to point from tail base toward tail tip.
    Angles are measured relative to the caudal body axis (-forward), positive
    toward anatomical left. All dense numerical operations are vectorized over
    the bounded input batch; failure-label normalization remains sparse.
    """

    source_s = np.asarray(source_tail_sample_s, dtype=np.float64).reshape(-1)
    if source_s.ndim != 1 or int(source_s.shape[0]) < 2:
        raise ValueError("source_tail_sample_s must contain at least two positions.")
    if np.any(~np.isfinite(source_s)) or np.any(np.diff(source_s) <= 0.0):
        raise ValueError("source_tail_sample_s must be finite and strictly increasing.")

    xy = np.asarray(tail_sample_xy, dtype=np.float64)
    tangent = np.asarray(tail_tangent_xy, dtype=np.float64)
    curvature = np.asarray(tail_curvature_px_inv, dtype=np.float64)
    if xy.ndim != 3 or int(xy.shape[2]) != 2:
        raise ValueError("tail_sample_xy must have shape (N, S, 2).")
    if tangent.shape != xy.shape:
        raise ValueError("tail_tangent_xy must have the same shape as tail_sample_xy.")
    if curvature.shape != xy.shape[:2]:
        raise ValueError("tail_curvature_px_inv must have shape (N, S).")
    if int(xy.shape[1]) != int(source_s.shape[0]):
        raise ValueError("source_tail_sample_s length must match tail sample arrays.")

    row_count = int(xy.shape[0])
    tail_valid = np.asarray(tail_sample_valid, dtype=bool).reshape(-1)
    spline_valid = np.asarray(bspline_valid, dtype=bool).reshape(-1)
    body_valid = np.asarray(body_frame_valid, dtype=bool).reshape(-1)
    if any(
        int(values.shape[0]) != row_count
        for values in (tail_valid, spline_valid, body_valid)
    ):
        raise ValueError("validity arrays must have the same row count as tail arrays.")
    source_valid = tail_valid & spline_valid & body_valid

    target_s = tail_sample_positions(int(tail_angle_sample_count)).astype(np.float32)
    sampled_xy = _interp_rows_2d(source_s, xy, target_s, source_valid)
    sampled_tangent = _interp_rows_2d(source_s, tangent, target_s, source_valid)
    sampled_curvature = _interp_rows_1d(source_s, curvature, target_s, source_valid)
    sampled_tangent64, tangent_norm_valid = _normalize_vectors(sampled_tangent)

    forward, forward_valid = _normalize_vectors(
        np.asarray(body_forward_axis_xy, dtype=np.float64)
    )
    left, left_valid = _normalize_vectors(
        np.asarray(body_left_axis_xy, dtype=np.float64)
    )
    tail_base = np.asarray(tail_base_xy, dtype=np.float64)
    if (
        forward.shape != (row_count, 2)
        or left.shape != (row_count, 2)
        or tail_base.shape != (row_count, 2)
    ):
        raise ValueError("body-frame and tail-base arrays must have shape (N, 2).")

    angle_rad = np.full((row_count, int(target_s.shape[0])), np.nan, dtype=np.float32)
    lateral_px = np.full_like(angle_rad, np.nan)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = np.full((row_count,), "ok", dtype=object)

    tail_reasons = _reason_values(
        tail_sample_failure_reason,
        row_count=row_count,
        fallback="tail_sample_invalid",
    )
    bspline_reasons = _reason_values(
        bspline_failure_reason,
        row_count=row_count,
        fallback="bspline_invalid",
    )
    body_reasons = _reason_values(
        body_frame_failure_reason,
        row_count=row_count,
        fallback="body_frame_invalid",
    )

    caudal = -forward
    remaining = np.ones((row_count,), dtype=bool)
    body_invalid = remaining & ~(body_valid & forward_valid & left_valid)
    _assign_failure_reasons(
        reasons,
        body_invalid,
        body_reasons,
        fallback="body_frame_invalid",
    )
    remaining &= ~body_invalid

    spline_invalid = remaining & ~spline_valid
    _assign_failure_reasons(
        reasons,
        spline_invalid,
        bspline_reasons,
        fallback="bspline_invalid",
    )
    remaining &= ~spline_invalid

    tail_invalid = remaining & ~tail_valid
    _assign_failure_reasons(
        reasons,
        tail_invalid,
        tail_reasons,
        fallback="tail_sample_invalid",
    )
    remaining &= ~tail_invalid

    geometry_finite = (
        np.all(tangent_norm_valid, axis=1)
        & np.all(np.isfinite(sampled_xy), axis=(1, 2))
        & np.all(np.isfinite(sampled_curvature), axis=1)
        & np.all(np.isfinite(tail_base), axis=1)
    )
    geometry_invalid = remaining & ~geometry_finite
    reasons[geometry_invalid] = "tail_geometry_nonfinite"
    remaining &= ~geometry_invalid

    dot_left = np.sum(sampled_tangent64 * left[:, None, :], axis=-1)
    dot_caudal = np.sum(sampled_tangent64 * caudal[:, None, :], axis=-1)
    angles = np.arctan2(dot_left, dot_caudal)
    offsets = np.asarray(sampled_xy, dtype=np.float64) - tail_base[:, None, :]
    lateral = np.sum(offsets * left[:, None, :], axis=-1)
    calculation_finite = np.all(np.isfinite(angles), axis=1) & np.all(
        np.isfinite(lateral), axis=1
    )
    calculation_invalid = remaining & ~calculation_finite
    reasons[calculation_invalid] = "tail_geometry_nonfinite"
    valid = remaining & calculation_finite
    angle_rad[valid] = angles[valid].astype(np.float32)
    lateral_px[valid] = lateral[valid].astype(np.float32)

    angle_deg = np.rad2deg(angle_rad).astype(np.float32)
    tail_tip_angle_rad = angle_rad[:, -1].astype(np.float32)
    tail_tip_angle_deg = angle_deg[:, -1].astype(np.float32)
    tail_tip_lateral_px = lateral_px[:, -1].astype(np.float32)

    max_abs_angle_rad = np.full((row_count,), np.nan, dtype=np.float32)
    angle_rms_rad = np.full((row_count,), np.nan, dtype=np.float32)
    max_abs_curvature = np.full((row_count,), np.nan, dtype=np.float32)
    integrated_abs_angle = np.full((row_count,), np.nan, dtype=np.float32)
    integrated_abs_curvature = np.full((row_count,), np.nan, dtype=np.float32)
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size:
        valid_angles = angle_rad[valid_rows]
        valid_curvature = sampled_curvature[valid_rows]
        max_abs_angle_rad[valid_rows] = np.max(np.abs(valid_angles), axis=1).astype(
            np.float32
        )
        angle_rms_rad[valid_rows] = np.sqrt(
            np.mean(np.square(valid_angles), axis=1)
        ).astype(np.float32)
        max_abs_curvature[valid_rows] = np.max(np.abs(valid_curvature), axis=1).astype(
            np.float32
        )
        integrated_abs_angle[valid_rows] = np.trapezoid(
            np.abs(valid_angles).astype(np.float64),
            target_s.astype(np.float64),
            axis=1,
        ).astype(np.float32)
        integrated_abs_curvature[valid_rows] = np.trapezoid(
            np.abs(valid_curvature).astype(np.float64),
            target_s.astype(np.float64),
            axis=1,
        ).astype(np.float32)

    return TailKinematicsBatch(
        tail_angle_sample_s=target_s,
        tail_angle_sample_xy=sampled_xy.astype(np.float32),
        tail_angle_rad=angle_rad,
        tail_angle_deg=angle_deg,
        tail_tip_angle_rad=tail_tip_angle_rad,
        tail_tip_angle_deg=tail_tip_angle_deg,
        tail_lateral_deflection_px=lateral_px.astype(np.float32),
        tail_tip_lateral_deflection_px=tail_tip_lateral_px,
        max_abs_tail_angle_rad=max_abs_angle_rad,
        max_abs_tail_angle_deg=np.rad2deg(max_abs_angle_rad).astype(np.float32),
        tail_angle_rms_rad=angle_rms_rad,
        tail_angle_rms_deg=np.rad2deg(angle_rms_rad).astype(np.float32),
        integrated_abs_tail_angle_rad=integrated_abs_angle,
        tail_curvature_px_inv=sampled_curvature.astype(np.float32),
        max_abs_tail_curvature_px_inv=max_abs_curvature,
        integrated_abs_tail_curvature=integrated_abs_curvature,
        valid=valid,
        failure_reason=reasons,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _require_group(parent: zarr.Group, name: str, *, path: str) -> zarr.Group:
    value = parent.get(name)
    if not isinstance(value, zarr.Group):
        raise SubjectShapeIOError(f"{path}/{name} is missing or is not a Zarr group.")
    return value


def _require_array_handle(parent: zarr.Group, name: str, *, path: str) -> Any:
    value = parent.get(name)
    if (
        value is None
        or not hasattr(value, "shape")
        or not hasattr(value, "__getitem__")
    ):
        raise SubjectShapeIOError(
            f"{path}/{name} is missing or is not a readable Zarr array."
        )
    return value


def _validate_source_shape(name: str, source: Any, expected: Sequence[int]) -> None:
    actual = tuple(int(value) for value in source.shape)
    wanted = tuple(int(value) for value in expected)
    if actual != wanted:
        raise SubjectShapeIOError(f"{name} has shape {actual!r}; expected {wanted!r}.")


def _resolve_tail_kinematics_sources(
    root: zarr.Group,
    shape_run: Optional[str],
    *,
    _staged_source_authority: Mapping[str, Any] | None = None,
    _staged_input_integrity_receipt: Mapping[str, Any] | None = None,
    _verify_staged_payload: bool = True,
) -> tuple[str, zarr.Group, TailKinematicsSources]:
    """Resolve lazy source handles through canonical or explicit staged proof.

    The private staged path is reserved for the node-local materializer.  It
    validates a closed detached receipt made from a fully verified canonical
    publication.  It is not exposed by the normal subject-shape reader API.
    """

    publication: Any | None = None
    staged_authority: dict[str, Any] | None = None
    if _staged_source_authority is None:
        (
            shape_group,
            run_name,
            run_path,
            publication,
        ) = resolve_canonical_subject_shape_run(root, shape_run)
    else:
        requested = str(shape_run or "").strip()
        if not requested or requested.lower() == "latest":
            raise SubjectShapeIOError(
                "Digest-bound staged source resolution requires an exact subject-shape run name."
            )
        normalized = requested.strip("/")
        prefix = "analysis/subject_shape_runs/"
        run_name = (
            normalized[len(prefix) :] if normalized.startswith(prefix) else normalized
        )
        if not run_name or "/" in run_name:
            raise SubjectShapeIOError(
                f"Invalid staged subject-shape run name {shape_run!r}."
            )
        run_path = f"analysis/subject_shape_runs/{run_name}"
        shape_group = root.get(run_path)
        if not isinstance(shape_group, zarr.Group):
            raise SubjectShapeIOError(
                f"Staged subject-shape run {run_path!r} is missing."
            )
        staged_authority = _validated_staged_source_authority(
            shape_group,
            run_name=run_name,
            authority=_staged_source_authority,
            verify_payload=(
                bool(_verify_staged_payload) and _staged_input_integrity_receipt is None
            ),
        )
    if _staged_source_authority is None and _staged_input_integrity_receipt is not None:
        raise SubjectShapeIOError(
            "Staged tail input integrity cannot replace canonical source authority."
        )
    components = _require_group(shape_group, "components", path=run_path)
    body_path = f"{run_path}/components"
    body = _require_group(components, "subject_body", path=body_path)
    body_frame = _require_group(shape_group, "body_frame", path=run_path)

    source_s_array = _require_array_handle(
        body, "tail_sample_s", path=f"{body_path}/subject_body"
    )
    source_s_raw = np.asarray(source_s_array[:])
    source_s = np.asarray(source_s_raw, dtype=np.float32)
    if source_s.ndim != 1 or int(source_s.shape[0]) < 2:
        raise SubjectShapeIOError(
            "tail_sample_s must be one-dimensional with at least two positions."
        )
    if np.any(~np.isfinite(source_s)) or np.any(
        np.diff(source_s.astype(np.float64)) <= 0.0
    ):
        raise SubjectShapeIOError(
            "tail_sample_s must be finite and strictly increasing."
        )

    arrays = {
        "tail_sample_xy": _require_array_handle(
            body, "tail_sample_xy", path=f"{body_path}/subject_body"
        ),
        "tail_tangent_xy": _require_array_handle(
            body, "tail_tangent_xy", path=f"{body_path}/subject_body"
        ),
        "tail_curvature_px_inv": _require_array_handle(
            body, "tail_curvature_px_inv", path=f"{body_path}/subject_body"
        ),
        "tail_sample_valid": _require_array_handle(
            body, "tail_sample_valid", path=f"{body_path}/subject_body"
        ),
        "bspline_valid": _require_array_handle(
            body, "bspline_valid", path=f"{body_path}/subject_body"
        ),
        "tail_base_xy": _require_array_handle(
            body, "tail_base_xy", path=f"{body_path}/subject_body"
        ),
        "body_forward_axis_xy": _require_array_handle(
            body_frame, "forward_axis_xy", path=f"{run_path}/body_frame"
        ),
        "body_left_axis_xy": _require_array_handle(
            body_frame, "left_axis_xy", path=f"{run_path}/body_frame"
        ),
        "body_frame_valid": _require_array_handle(
            body_frame, "axis_valid", path=f"{run_path}/body_frame"
        ),
    }
    tail_xy_shape = tuple(int(value) for value in arrays["tail_sample_xy"].shape)
    if len(tail_xy_shape) != 3 or int(tail_xy_shape[2]) != 2:
        raise SubjectShapeIOError(
            f"tail_sample_xy has shape {tail_xy_shape!r}; expected (N, S, 2)."
        )
    row_count, source_sample_count, _xy = tail_xy_shape
    if int(source_sample_count) != int(source_s.shape[0]):
        raise SubjectShapeIOError(
            "tail_sample_s length does not match the second axis of tail_sample_xy "
            f"({source_s.shape[0]} != {source_sample_count})."
        )
    _validate_source_shape("tail_tangent_xy", arrays["tail_tangent_xy"], tail_xy_shape)
    _validate_source_shape(
        "tail_curvature_px_inv",
        arrays["tail_curvature_px_inv"],
        (row_count, source_sample_count),
    )
    for name in ("tail_sample_valid", "bspline_valid", "body_frame_valid"):
        _validate_source_shape(name, arrays[name], (row_count,))
    for name in ("tail_base_xy", "body_forward_axis_xy", "body_left_axis_xy"):
        _validate_source_shape(name, arrays[name], (row_count, 2))

    reason_arrays = {
        "tail_sample_failure_reason": body.get("tail_sample_failure_reason_bytes"),
        "bspline_failure_reason": body.get("bspline_failure_reason_bytes"),
        "body_frame_failure_reason": body_frame.get("failure_reason_bytes"),
    }
    for name, source in reason_arrays.items():
        if source is not None and (
            not source.shape or int(source.shape[0]) != int(row_count)
        ):
            raise SubjectShapeIOError(
                f"{name} has shape {tuple(int(value) for value in source.shape)!r}; expected first axis {row_count}."
            )

    if staged_authority is not None:
        if staged_authority.get("row_count") != int(row_count) or staged_authority.get(
            "source_sample_count"
        ) != int(source_sample_count):
            raise SubjectShapeIOError(
                "Staged subject-shape cardinality differs from its canonical receipt."
            )
        source_authority = staged_authority
        source_authority_mode = "digest_bound_staged_subset"
        staged_input_receipt = (
            _canonical_staged_input_integrity_receipt(
                shape_group,
                run_name=run_name,
                authority=staged_authority,
                receipt=_staged_input_integrity_receipt,
            )
            if _staged_input_integrity_receipt is not None
            else None
        )
        if staged_input_receipt is not None:
            static_path = _TAIL_WORKER_STATIC_SOURCE_PATHS["source_tail_sample_s"]
            expected_static = staged_input_receipt["worker_static_inputs"][static_path]
            if (
                np.dtype(source_s_raw.dtype).str != expected_static["dtype"]
                or [int(value) for value in source_s_raw.shape]
                != expected_static["shape"]
                or array_values_sha256(source_s_raw)
                != expected_static["content_sha256"]
            ):
                raise SubjectShapeIOError(
                    "Staged tail_sample_s differs from its input integrity receipt."
                )
    else:
        assert publication is not None
        source_authority = _build_staged_source_authority(
            shape_group,
            run_name=run_name,
            row_count=int(row_count),
            source_sample_count=int(source_sample_count),
            publication=publication,
        )
        source_authority_mode = "canonical_publication"
        staged_input_receipt = None
    manifest_sha256 = source_authority["canonical_publication"]["manifest_sha256"]

    return (
        run_name,
        shape_group,
        TailKinematicsSources(
            source_tail_sample_s=source_s,
            arrays=arrays,
            reason_arrays=reason_arrays,
            row_count=int(row_count),
            source_sample_count=int(source_sample_count),
            source_publication_manifest_sha256=str(manifest_sha256),
            source_authority_mode=source_authority_mode,
            source_authority=source_authority,
            staged_input_integrity_receipt=staged_input_receipt,
            shape_group=shape_group,
        ),
    )


def _revalidate_tail_kinematics_sources(sources: TailKinematicsSources) -> None:
    """Rebind staged metadata after reads without another whole-input scan."""

    if sources.source_authority_mode != "digest_bound_staged_subset":
        return
    run_name = str(sources.source_authority.get("source_subject_shape_run") or "")
    if sources.staged_input_integrity_receipt is not None:
        _canonical_staged_input_integrity_receipt(
            sources.shape_group,
            run_name=run_name,
            authority=sources.source_authority,
            receipt=sources.staged_input_integrity_receipt,
        )
    else:
        _validated_staged_source_authority(
            sources.shape_group,
            run_name=run_name,
            authority=sources.source_authority,
        )


def _staged_input_chunk_for_slice(
    sources: TailKinematicsSources,
    row_slice: slice,
) -> Mapping[str, Any] | None:
    receipt = sources.staged_input_integrity_receipt
    if receipt is None:
        return None
    start = int(row_slice.start or 0)
    stop = int(row_slice.stop or 0)
    chunk_rows = int(receipt["requested_chunk_rows"])
    chunk_index = start // chunk_rows
    chunks = receipt["chunks"]
    if not 0 <= chunk_index < len(chunks):
        raise SubjectShapeIOError(
            "Tail worker row slice is absent from its staged-input receipt."
        )
    chunk = chunks[chunk_index]
    if chunk["start_row"] != start or chunk["stop_row"] != stop:
        raise SubjectShapeIOError(
            "Tail worker row slice differs from its staged-input receipt."
        )
    return chunk


def _validate_staged_block_value(
    chunk: Mapping[str, Any] | None,
    *,
    relative_ref: str,
    values: np.ndarray,
) -> None:
    if chunk is None:
        return
    record = chunk["source_arrays"].get(relative_ref)
    if (
        not isinstance(record, Mapping)
        or np.dtype(values.dtype).str != record.get("dtype")
        or [int(value) for value in values.shape] != record.get("shape")
        or array_values_sha256(values) != record.get("content_sha256")
    ):
        raise SubjectShapeIOError(
            f"Tail worker input {relative_ref!r} differs from its staged receipt."
        )


def _complete_staged_input_worker_attestation(
    receipt: Mapping[str, Any],
    observed_chunk_receipts: Sequence[str],
) -> dict[str, Any]:
    expected = [str(chunk["record_sha256"]) for chunk in receipt["chunks"]]
    observed = [str(value) for value in observed_chunk_receipts]
    if (
        any(not _is_sha256(value) for value in observed)
        or len(observed) != len(set(observed))
        or sorted(observed) != sorted(expected)
    ):
        raise RuntimeError(
            "Tail workers did not attest the exact complete staged-input chunk set."
        )
    body = {
        "schema_id": TAIL_KINEMATICS_WORKER_INPUT_ATTESTATION_SCHEMA_ID,
        "schema_version": TAIL_KINEMATICS_WORKER_INPUT_ATTESTATION_SCHEMA_VERSION,
        "staged_input_integrity_receipt_sha256": receipt["record_sha256"],
        "chunk_count": len(expected),
        "ordered_chunk_receipt_set_sha256": _canonical_sha256(expected),
        "complete_worker_chunk_set": True,
        "verification_location": (
            "worker_owned_decoded_block_before_scientific_compute"
        ),
    }
    return {**body, "record_sha256": _canonical_sha256(body)}


def _read_tail_kinematics_source_block(
    sources: TailKinematicsSources,
    row_slice: slice,
) -> dict[str, np.ndarray]:
    """Read one bounded row block from the lazy subject-shape source handles."""

    arrays = sources.arrays
    chunk = _staged_input_chunk_for_slice(sources, row_slice)
    raw_values = {
        "tail_sample_xy": np.asarray(arrays["tail_sample_xy"][row_slice]),
        "tail_tangent_xy": np.asarray(arrays["tail_tangent_xy"][row_slice]),
        "tail_curvature_px_inv": np.asarray(arrays["tail_curvature_px_inv"][row_slice]),
        "tail_sample_valid": np.asarray(arrays["tail_sample_valid"][row_slice]),
        "bspline_valid": np.asarray(arrays["bspline_valid"][row_slice]),
        "tail_base_xy": np.asarray(arrays["tail_base_xy"][row_slice]),
        "body_forward_axis_xy": np.asarray(arrays["body_forward_axis_xy"][row_slice]),
        "body_left_axis_xy": np.asarray(arrays["body_left_axis_xy"][row_slice]),
        "body_frame_valid": np.asarray(arrays["body_frame_valid"][row_slice]),
    }
    for logical_name, values in raw_values.items():
        _validate_staged_block_value(
            chunk,
            relative_ref=_TAIL_WORKER_ROW_SOURCE_PATHS[logical_name],
            values=values,
        )
    reason_values: dict[str, np.ndarray] = {}
    for logical_name in (
        "tail_sample_failure_reason",
        "bspline_failure_reason",
        "body_frame_failure_reason",
    ):
        relative_ref = _TAIL_WORKER_ROW_SOURCE_PATHS[logical_name]
        source = sources.reason_arrays.get(logical_name)
        if source is None:
            reason_values[logical_name] = _read_optional_reason_labels(None, row_slice)
            continue
        raw = np.asarray(source[row_slice])
        _validate_staged_block_value(
            chunk,
            relative_ref=relative_ref,
            values=raw,
        )
        if raw.ndim == 2 and np.issubdtype(raw.dtype, np.integer):
            reason_values[logical_name] = decode_reason_bytes(raw)
        else:
            reason_values[logical_name] = np.asarray(raw, dtype=object).reshape(-1)
    return {
        "source_tail_sample_s": sources.source_tail_sample_s,
        "tail_sample_xy": np.asarray(raw_values["tail_sample_xy"], dtype=np.float32),
        "tail_tangent_xy": np.asarray(raw_values["tail_tangent_xy"], dtype=np.float32),
        "tail_curvature_px_inv": np.asarray(
            raw_values["tail_curvature_px_inv"], dtype=np.float32
        ),
        "tail_sample_valid": np.asarray(raw_values["tail_sample_valid"], dtype=bool),
        "bspline_valid": np.asarray(raw_values["bspline_valid"], dtype=bool),
        "tail_base_xy": np.asarray(raw_values["tail_base_xy"], dtype=np.float32),
        "body_forward_axis_xy": np.asarray(
            raw_values["body_forward_axis_xy"], dtype=np.float32
        ),
        "body_left_axis_xy": np.asarray(
            raw_values["body_left_axis_xy"], dtype=np.float32
        ),
        "body_frame_valid": np.asarray(raw_values["body_frame_valid"], dtype=bool),
        **reason_values,
    }


def _copy_row_lineage_bounded(
    run_group: zarr.Group,
    shape_group: zarr.Group,
    *,
    row_count: int,
    block_rows: int,
    output_shard_rows: int,
    precreated: bool = False,
) -> tuple[list[str], list[str], str]:
    copied: list[str] = []
    for name in ROW_LINEAGE_NAMES:
        source_array = shape_group.get(name)
        if source_array is None:
            raise SubjectShapeIOError(
                f"Canonical subject-shape source lacks required direct {name!r}."
            )
        values = np.asarray(source_array[:])
        if values.ndim != 1 or int(values.shape[0]) != int(row_count):
            raise SubjectShapeIOError(
                f"Canonical subject-shape {name!r} must be one-dimensional and row aligned."
            )
        if name == "instance_key":
            if (
                values.dtype != np.dtype("uint64")
                or np.unique(values).size != values.size
            ):
                raise SubjectShapeIOError(
                    "Canonical subject-shape instance_key must be unique uint64."
                )
        elif values.dtype.kind not in {"i", "u"} or np.any(values < 0):
            raise SubjectShapeIOError(
                f"Canonical subject-shape {name!r} must be non-negative integer identity."
            )
        _copy_array_bounded(
            run_group,
            name,
            source_array,
            block_rows=int(block_rows),
            row_aligned_count=int(row_count),
            shard_rows=int(output_shard_rows),
            precreated=precreated,
        )
        copied.append(name)
    return copied, [], "source_acquisition_frame_index"


def _candidate_tail_kinematics_dimensions(
    shape_group: zarr.Group,
    *,
    row_count: int,
    tail_angle_sample_count: int,
) -> TailKinematicsDimensions:
    """Close exact source dtypes and the optional revision bundle before writes."""

    expected_lineage_dtypes = {
        "instance_key": np.dtype(np.uint64),
        "source_crop_row_ids": np.dtype(np.int64),
        "source_acquisition_frame_index": np.dtype(np.int64),
    }
    for name, expected_dtype in expected_lineage_dtypes.items():
        node = shape_group.get(name)
        if node is None:
            raise SubjectShapeIOError(
                f"Candidate source lacks required direct lineage array {name!r}."
            )
        if (
            tuple(int(value) for value in node.shape) != (int(row_count),)
            or np.dtype(node.dtype) != expected_dtype
        ):
            raise SubjectShapeIOError(
                f"Candidate source {name!r} must have shape {(int(row_count),)!r} "
                f"and exact dtype {expected_dtype}."
            )

    revision_group = shape_group.get("source_refined_subject_masks")
    if revision_group is None:
        return TailKinematicsDimensions(
            n_rows=int(row_count),
            n_tail_samples=int(tail_angle_sample_count),
        )
    if not isinstance(revision_group, zarr.Group):
        raise SubjectShapeIOError(
            "Candidate source_refined_subject_masks node is not a group."
        )
    revision = revision_group.get("row_revision")
    available = revision_group.get("row_revision_available")
    if (revision is None) != (available is None):
        raise SubjectShapeIOError(
            "Candidate source-revision snapshot is a partial optional bundle."
        )
    if revision is None:
        return TailKinematicsDimensions(
            n_rows=int(row_count),
            n_tail_samples=int(tail_angle_sample_count),
        )
    revision_shape = tuple(int(value) for value in revision.shape)
    available_shape = tuple(int(value) for value in available.shape)
    if (
        len(revision_shape) != 2
        or revision_shape[0] != int(row_count)
        or revision_shape[1] <= 0
        or available_shape != (revision_shape[1],)
        or np.dtype(revision.dtype) != np.dtype(np.int64)
        or np.dtype(available.dtype) != np.dtype(bool)
    ):
        raise SubjectShapeIOError(
            "Candidate source-revision snapshot has the wrong exact shape/dtype contract."
        )
    return TailKinematicsDimensions(
        n_rows=int(row_count),
        n_tail_samples=int(tail_angle_sample_count),
        n_components=int(revision_shape[1]),
    )


def _create_tail_kinematics_public_candidate(
    parent: zarr.Group,
    *,
    run_name: str,
    publication_owner_uuid: str,
) -> zarr.Group:
    """Create one owner-bound public child; injectable for ambiguity tests."""

    return parent.create_group(
        run_name,
        attributes={
            TAIL_PUBLICATION_OWNER_ATTR: publication_owner_uuid,
            "stage_selector_eligible": False,
        },
    )


def _write_tail_kinematics_failure_attr(attrs: Any, name: str, value: Any) -> None:
    attrs[name] = value


def _delete_tail_kinematics_failure_attr(attrs: Any, name: str) -> None:
    del attrs[name]


def _fresh_owned_tail_kinematics_candidate(
    root: zarr.Group,
    *,
    run_path: str,
    publication_owner_uuid: str,
) -> Optional[zarr.Group]:
    candidate = root.get(run_path)
    if not isinstance(candidate, zarr.Group):
        return None
    try:
        exact_binding = canonical_node_path(candidate) == run_path and archive_identity(
            candidate
        ) == archive_identity(root)
    except Exception:
        return None
    if (
        not exact_binding
        or candidate.attrs.get(TAIL_PUBLICATION_OWNER_ATTR) != publication_owner_uuid
    ):
        return None
    return candidate


def _persist_owned_tail_kinematics_attr(
    root: zarr.Group,
    *,
    run_path: str,
    publication_owner_uuid: str,
    name: str,
    value: Any,
) -> bool:
    candidate = _fresh_owned_tail_kinematics_candidate(
        root,
        run_path=run_path,
        publication_owner_uuid=publication_owner_uuid,
    )
    if candidate is None:
        return False
    try:
        _write_tail_kinematics_failure_attr(candidate.attrs, name, value)
    except BaseException:
        fresh = _fresh_owned_tail_kinematics_candidate(
            root,
            run_path=run_path,
            publication_owner_uuid=publication_owner_uuid,
        )
        if fresh is not None and fresh.attrs.get(name) == value:
            return True
        raise
    fresh = _fresh_owned_tail_kinematics_candidate(
        root,
        run_path=run_path,
        publication_owner_uuid=publication_owner_uuid,
    )
    if fresh is None:
        return False
    if fresh.attrs.get(name) != value:
        raise RuntimeError(f"Tail-kinematics cleanup attr {name!r} did not persist.")
    return True


def _delete_owned_tail_kinematics_attr(
    root: zarr.Group,
    *,
    run_path: str,
    publication_owner_uuid: str,
    name: str,
) -> bool:
    candidate = _fresh_owned_tail_kinematics_candidate(
        root,
        run_path=run_path,
        publication_owner_uuid=publication_owner_uuid,
    )
    if candidate is None:
        return False
    if name not in candidate.attrs:
        return True
    try:
        _delete_tail_kinematics_failure_attr(candidate.attrs, name)
    except BaseException:
        fresh = _fresh_owned_tail_kinematics_candidate(
            root,
            run_path=run_path,
            publication_owner_uuid=publication_owner_uuid,
        )
        if fresh is not None and name not in fresh.attrs:
            return True
        raise
    fresh = _fresh_owned_tail_kinematics_candidate(
        root,
        run_path=run_path,
        publication_owner_uuid=publication_owner_uuid,
    )
    if fresh is None:
        return False
    if name in fresh.attrs:
        raise RuntimeError(f"Tail-kinematics cleanup attr {name!r} was not deleted.")
    return True


def _cleanup_failed_tail_kinematics_candidate(
    root: zarr.Group,
    *,
    run_name: str,
    publication_owner_uuid: str,
    error: BaseException,
    completed_block_count: int,
    completed_worker_task_count: int,
) -> bool:
    run_path = f"analysis/tail_kinematics_runs/{run_name}"

    def persist(name: str, value: Any) -> bool:
        return _persist_owned_tail_kinematics_attr(
            root,
            run_path=run_path,
            publication_owner_uuid=publication_owner_uuid,
            name=name,
            value=value,
        )

    for name, value in (
        ("stage_selector_eligible", False),
        (RUN_COMPLETION_CONTRACT_ATTR, RUN_COMPLETION_CONTRACT),
        (RUN_COMPLETION_STATUS_ATTR, RUN_STATUS_FAILED),
        ("palette_run_failed_at_utc", utc_now_iso()),
        (RUN_NAME_ATTR, run_name),
        ("palette_run_error", f"{type(error).__name__}: {error}"),
        ("completed_block_count", int(completed_block_count)),
        ("completed_worker_task_count", int(completed_worker_task_count)),
    ):
        if not persist(name, value):
            return False
    if not _delete_owned_tail_kinematics_attr(
        root,
        run_path=run_path,
        publication_owner_uuid=publication_owner_uuid,
        name=RUN_COMPLETED_AT_ATTR,
    ):
        return False
    tombstone = json_attr_safe(
        {
            "schema_id": "palette.tail_publication_tombstone",
            "schema_version": 1,
            "run_family": "tail_kinematics",
            "publication_owner_uuid": publication_owner_uuid,
            "run_name": run_name,
            "public_path_retained": True,
            "selector_eligible": False,
            "retry_policy": "new_immutable_run_name_required",
        }
    )
    if not persist(TAIL_PUBLICATION_TOMBSTONE_ATTR, tombstone):
        return False
    fresh = _fresh_owned_tail_kinematics_candidate(
        root,
        run_path=run_path,
        publication_owner_uuid=publication_owner_uuid,
    )
    if fresh is None:
        return False
    if (
        fresh.attrs.get("stage_selector_eligible") is not False
        or fresh.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED
        or RUN_COMPLETED_AT_ATTR in fresh.attrs
        or fresh.attrs.get(TAIL_PUBLICATION_TOMBSTONE_ATTR) != tombstone
    ):
        raise RuntimeError("Owned tail-kinematics tombstone did not verify exactly.")
    return True


def _prepare_tail_kinematics_run(
    root: zarr.Group,
    *,
    target_run: str,
    shape_run_name: str,
    shape_group: zarr.Group,
    row_count: int,
    tail_angle_sample_count: int,
    source_geometry_tail_sample_count: int,
    requested_block_rows: int,
    effective_block_rows: int,
    requested_output_shard_rows: int,
    effective_output_shard_rows: int,
    execution_backend: str,
    worker_count_requested: int,
    worker_count_effective: int,
    source_publication_manifest_sha256: str,
    source_authority_mode: str,
    source_authority: Mapping[str, Any],
    stage_command: str,
    publication_owner_uuid: str,
    overwrite: bool,
    storage_receipt: Any | None = None,
    storage_dimensions: TailKinematicsDimensions | None = None,
) -> zarr.Group:
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "tail_kinematics_runs")
    if target_run in parent:
        raise ValueError(
            f"analysis/tail_kinematics_runs/{target_run} already exists. "
            "Canonical tail runs are immutable; use a new run name or explicit "
            "failed-run maintenance tooling."
        )
    run_group = _create_tail_kinematics_public_candidate(
        parent,
        run_name=target_run,
        publication_owner_uuid=publication_owner_uuid,
    )
    fresh = _fresh_owned_tail_kinematics_candidate(
        root,
        run_path=f"analysis/tail_kinematics_runs/{target_run}",
        publication_owner_uuid=publication_owner_uuid,
    )
    if fresh is None:
        raise RuntimeError(
            "Tail-kinematics child did not freshly bind to its exact owner, path, "
            "and archive."
        )
    run_group = fresh
    mark_run_started(run_group, run_name=target_run, stage="tail_kinematics")
    if (
        run_group.attrs.get(TAIL_PUBLICATION_OWNER_ATTR) != publication_owner_uuid
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        raise RuntimeError(
            "Tail-kinematics child did not persist its exact owner-bound, "
            "selector-ineligible creation state."
        )
    _set_reason_bytes_attrs(run_group)

    if (storage_receipt is None) != (storage_dimensions is None):
        raise ValueError(
            "Tail-kinematics candidate receipt and dimensions must be provided together."
        )
    byte_planner_adopted = storage_receipt is not None
    if byte_planner_adopted:
        assert storage_dimensions is not None
        create_tail_kinematics_arrays_from_receipt(
            run_group,
            receipt=storage_receipt,
            dimensions=storage_dimensions,
        )
        stamp_tail_kinematics_array_schema(
            run_group,
            storage_dimensions,
            byte_planner_adopted=True,
        )
        persist_tail_kinematics_storage_receipt(run_group, storage_receipt)
        run_group.attrs.update(
            {
                "physical_layout": "analysis_storage_plan_receipt_v1",
                "byte_planner_adopted": True,
                "storage_candidate_status": "unpromoted_selector_ineligible",
                "storage_candidate_execution_policy": "serial_single_writer_only",
            }
        )

    source_refined_run = shape_group.attrs.get("source_refined_subject_masks_run")
    created = _utc_now()

    copied, missing, acquisition_frame_index_source = _copy_row_lineage_bounded(
        run_group,
        shape_group,
        row_count=int(row_count),
        block_rows=int(effective_block_rows),
        output_shard_rows=int(effective_output_shard_rows),
        precreated=byte_planner_adopted,
    )
    output_row_chunk = int(_metric_chunks(int(row_count))[0])
    block_count = len(_iter_row_slices(int(row_count), int(effective_block_rows)))
    output_shard_count = len(
        _iter_row_slices(int(row_count), int(effective_output_shard_rows))
    )
    materialization_mode = (
        "bounded_streaming_single_writer"
        if execution_backend == "serial"
        else "bounded_process_shards"
    )

    run_group.attrs.update(
        {
            "schema_id": TAIL_KINEMATICS_SCHEMA_ID,
            "schema_version": TAIL_KINEMATICS_SCHEMA_VERSION,
            "method": TAIL_KINEMATICS_METHOD,
            "method_version": TAIL_KINEMATICS_METHOD_VERSION,
            "created_at_utc": created,
            "created_utc": created,
            "row_axis": "observation_instance",
            "source_subject_shape_run": str(shape_run_name),
            "source_subject_shape_path": f"analysis/subject_shape_runs/{shape_run_name}",
            "source_subject_shape_publication_manifest_sha256": str(
                source_publication_manifest_sha256
            ),
            "source_subject_shape_authority_mode": str(source_authority_mode),
            "source_subject_shape_authority_sha256": str(
                source_authority.get("record_sha256")
            ),
            "source_subject_shape_authority": json_attr_safe(source_authority),
            "source_refined_subject_masks_run": (
                str(source_refined_run) if source_refined_run is not None else None
            ),
            "source_tail_geometry_kind": SOURCE_TAIL_GEOMETRY_KIND,
            "body_frame_convention": shape_group.attrs.get(
                "body_frame_schema_id", "fish_anatomical_body_frame"
            ),
            "body_frame_source": f"analysis/subject_shape_runs/{shape_run_name}/body_frame",
            "tail_angle_reference_axis": "caudal_axis=-forward_axis",
            "tail_angle_positive_direction": "anatomical_left",
            "tail_angle_units_primary": "rad",
            "tail_sample_domain": "tail_segment_normalized_arclength",
            "tail_angle_sample_count": int(tail_angle_sample_count),
            "source_geometry_tail_sample_count": int(source_geometry_tail_sample_count),
            "curvature_source": "subject_shape.tail_curvature_px_inv",
            "acquisition_frame_index_source": acquisition_frame_index_source,
            "row_lineage_copied": copied,
            "row_lineage_missing": missing,
            "materialization_mode": materialization_mode,
            "compute_kernel": TAIL_KINEMATICS_COMPUTE_KERNEL,
            "execution_backend": str(execution_backend),
            "worker_count_requested": int(worker_count_requested),
            "worker_count_effective": int(worker_count_effective),
            "worker_chunk_size_requested": int(requested_block_rows),
            "worker_chunk_size_effective": int(effective_block_rows),
            "worker_chunk_alignment": "compute_blocks_align_to_output_row_chunks",
            "worker_write_ownership": (
                "single_driver"
                if execution_backend == "serial"
                else "one_complete_nonoverlapping_output_shard_per_task"
            ),
            "worker_compute_blocking": (
                "single_driver_bounded_blocks"
                if execution_backend == "serial"
                else "bounded_subblocks_within_owned_output_shard"
            ),
            "requested_block_rows": int(requested_block_rows),
            "effective_block_rows": int(effective_block_rows),
            "compute_block_rows_requested": int(requested_block_rows),
            "compute_block_rows_effective": int(effective_block_rows),
            "output_row_chunk": output_row_chunk,
            "requested_output_shard_rows": int(requested_output_shard_rows),
            "effective_output_shard_rows": int(effective_output_shard_rows),
            "output_shard_rows": int(effective_output_shard_rows),
            "output_shard_count": int(output_shard_count),
            "output_shard_scope": (
                "canonical_tail_arrays_and_identity; copied_lineage_preserves_"
                "source_chunks_with_output_shard_as_floor"
            ),
            "worker_task_count": (
                0 if execution_backend == "serial" else int(output_shard_count)
            ),
            "block_count": int(block_count),
            "source_refs": {
                "subject_shape_run": f"analysis/subject_shape_runs/{shape_run_name}",
                "subject_shape_body_component": f"analysis/subject_shape_runs/{shape_run_name}/components/subject_body",
                "subject_shape_body_frame": f"analysis/subject_shape_runs/{shape_run_name}/body_frame",
            },
        }
    )
    _copy_optional_source_revision_snapshot(
        run_group,
        shape_group,
        shape_run_name=shape_run_name,
        row_count=int(row_count),
        block_rows=int(effective_block_rows),
        output_shard_rows=int(effective_output_shard_rows),
        precreated=byte_planner_adopted,
    )

    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    provenance = build_stage_provenance(
        stage=TAIL_KINEMATICS_STAGE_NAME,
        command=stage_command,
        created_at_utc=created,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        parameters={
            "method": TAIL_KINEMATICS_METHOD,
            "method_version": TAIL_KINEMATICS_METHOD_VERSION,
            "tail_angle_sample_count": int(tail_angle_sample_count),
            "tail_angle_reference_axis": "caudal_axis=-forward_axis",
            "tail_angle_positive_direction": "anatomical_left",
            "source_tail_geometry_kind": SOURCE_TAIL_GEOMETRY_KIND,
            "materialization_mode": materialization_mode,
            "compute_kernel": TAIL_KINEMATICS_COMPUTE_KERNEL,
            "execution_backend": str(execution_backend),
            "worker_count_requested": int(worker_count_requested),
            "worker_count_effective": int(worker_count_effective),
            "worker_chunk_size_requested": int(requested_block_rows),
            "worker_chunk_size_effective": int(effective_block_rows),
            "worker_chunk_alignment": "compute_blocks_align_to_output_row_chunks",
            "worker_write_ownership": (
                "single_driver"
                if execution_backend == "serial"
                else "one_complete_nonoverlapping_output_shard_per_task"
            ),
            "worker_compute_blocking": (
                "single_driver_bounded_blocks"
                if execution_backend == "serial"
                else "bounded_subblocks_within_owned_output_shard"
            ),
            "requested_block_rows": int(requested_block_rows),
            "effective_block_rows": int(effective_block_rows),
            "compute_block_rows_requested": int(requested_block_rows),
            "compute_block_rows_effective": int(effective_block_rows),
            "output_row_chunk": output_row_chunk,
            "requested_output_shard_rows": int(requested_output_shard_rows),
            "effective_output_shard_rows": int(effective_output_shard_rows),
            "output_shard_rows": int(effective_output_shard_rows),
            "output_shard_count": int(output_shard_count),
            "output_shard_scope": (
                "canonical_tail_arrays_and_identity; copied_lineage_preserves_"
                "source_chunks_with_output_shard_as_floor"
            ),
            "worker_task_count": (
                0 if execution_backend == "serial" else int(output_shard_count)
            ),
            "block_count": int(block_count),
        },
        inputs={
            "source_subject_shape_run": shape_run_name,
            "source_refined_subject_masks_run": source_refined_run,
            "source_subject_shape_publication_manifest_sha256": str(
                source_publication_manifest_sha256
            ),
            "source_subject_shape_authority_mode": str(source_authority_mode),
            "source_subject_shape_authority_sha256": str(
                source_authority.get("record_sha256")
            ),
        },
    )
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="tail_kinematics_run")
    return run_group


def _prepare_tail_kinematics_output_arrays(
    run_group: zarr.Group,
    *,
    row_count: int,
    tail_angle_sample_s: np.ndarray,
    shard_rows: int | None = None,
    precreated: bool = False,
) -> None:
    sample_s = np.asarray(tail_angle_sample_s, dtype=np.float32).reshape(-1)
    sample_count = int(sample_s.shape[0])
    if precreated:
        expected_names = {
            "valid",
            "failure_reason_bytes",
            "tail_angle_sample_s",
            "tail_angle_sample_xy",
            "tail_angle_rad",
            "tail_angle_deg",
            "tail_lateral_deflection_px",
            "tail_curvature_px_inv",
            "tail_tip_angle_rad",
            "tail_tip_angle_deg",
            "tail_tip_lateral_deflection_px",
            "max_abs_tail_angle_rad",
            "max_abs_tail_angle_deg",
            "tail_angle_rms_rad",
            "tail_angle_rms_deg",
            "integrated_abs_tail_angle_rad",
            "max_abs_tail_curvature_px_inv",
            "integrated_abs_tail_curvature",
        }
        missing = sorted(name for name in expected_names if run_group.get(name) is None)
        if missing:
            raise ValueError(
                f"Precreated tail-kinematics candidate arrays are missing: {missing!r}."
            )
        run_group["tail_angle_sample_s"][:] = sample_s
        return
    chunks_1d = _metric_chunks(row_count)
    shards_1d = _output_shards(chunks_1d, shard_rows=shard_rows, dtype=bool)
    _create_array(
        run_group,
        "valid",
        shape=(row_count,),
        dtype=bool,
        chunks=chunks_1d,
        shards=shards_1d,
    )
    reason_chunks = _metric_chunks_lastdim(row_count, REASON_BYTES_WIDTH)
    _create_array(
        run_group,
        "failure_reason_bytes",
        shape=(row_count, REASON_BYTES_WIDTH),
        dtype=np.uint8,
        chunks=reason_chunks,
        shards=_output_shards(reason_chunks, shard_rows=shard_rows, dtype=np.uint8),
    )
    _write_array(run_group, "tail_angle_sample_s", sample_s, chunks=(sample_count,))
    sample_xy_chunks = _metric_chunks_3d(row_count, sample_count, 2)
    _create_array(
        run_group,
        "tail_angle_sample_xy",
        shape=(row_count, sample_count, 2),
        dtype=np.float32,
        chunks=sample_xy_chunks,
        shards=_output_shards(
            sample_xy_chunks, shard_rows=shard_rows, dtype=np.float32
        ),
        fill_value=np.nan,
    )
    for name in (
        "tail_angle_rad",
        "tail_angle_deg",
        "tail_lateral_deflection_px",
        "tail_curvature_px_inv",
    ):
        chunks_2d = _metric_chunks_lastdim(row_count, sample_count)
        _create_array(
            run_group,
            name,
            shape=(row_count, sample_count),
            dtype=np.float32,
            chunks=chunks_2d,
            shards=_output_shards(chunks_2d, shard_rows=shard_rows, dtype=np.float32),
            fill_value=np.nan,
        )
    for name in (
        "tail_tip_angle_rad",
        "tail_tip_angle_deg",
        "tail_tip_lateral_deflection_px",
        "max_abs_tail_angle_rad",
        "max_abs_tail_angle_deg",
        "tail_angle_rms_rad",
        "tail_angle_rms_deg",
        "integrated_abs_tail_angle_rad",
        "max_abs_tail_curvature_px_inv",
        "integrated_abs_tail_curvature",
    ):
        _create_array(
            run_group,
            name,
            shape=(row_count,),
            dtype=np.float32,
            chunks=chunks_1d,
            shards=_output_shards(chunks_1d, shard_rows=shard_rows, dtype=np.float32),
            fill_value=np.nan,
        )


def _write_tail_kinematics_batch_slice(
    run_group: zarr.Group,
    row_slice: slice,
    batch: TailKinematicsBatch,
) -> None:
    run_group["valid"][row_slice] = batch.valid.astype(bool)
    run_group["failure_reason_bytes"][row_slice] = batch.failure_reason_bytes
    run_group["tail_angle_sample_xy"][row_slice] = batch.tail_angle_sample_xy
    run_group["tail_angle_rad"][row_slice] = batch.tail_angle_rad
    run_group["tail_angle_deg"][row_slice] = batch.tail_angle_deg
    run_group["tail_tip_angle_rad"][row_slice] = batch.tail_tip_angle_rad
    run_group["tail_tip_angle_deg"][row_slice] = batch.tail_tip_angle_deg
    run_group["tail_lateral_deflection_px"][
        row_slice
    ] = batch.tail_lateral_deflection_px
    run_group["tail_tip_lateral_deflection_px"][
        row_slice
    ] = batch.tail_tip_lateral_deflection_px
    run_group["max_abs_tail_angle_rad"][row_slice] = batch.max_abs_tail_angle_rad
    run_group["max_abs_tail_angle_deg"][row_slice] = batch.max_abs_tail_angle_deg
    run_group["tail_angle_rms_rad"][row_slice] = batch.tail_angle_rms_rad
    run_group["tail_angle_rms_deg"][row_slice] = batch.tail_angle_rms_deg
    run_group["integrated_abs_tail_angle_rad"][
        row_slice
    ] = batch.integrated_abs_tail_angle_rad
    run_group["tail_curvature_px_inv"][row_slice] = batch.tail_curvature_px_inv
    run_group["max_abs_tail_curvature_px_inv"][
        row_slice
    ] = batch.max_abs_tail_curvature_px_inv
    run_group["integrated_abs_tail_curvature"][
        row_slice
    ] = batch.integrated_abs_tail_curvature


def _tail_kinematics_batch_counts(
    batch: TailKinematicsBatch,
) -> tuple[int, dict[str, int]]:
    valid_count = int(np.count_nonzero(batch.valid))
    labels, counts = np.unique(
        np.asarray(batch.failure_reason, dtype=str),
        return_counts=True,
    )
    return valid_count, {
        str(label): int(count)
        for label, count in zip(labels.tolist(), counts.tolist(), strict=True)
    }


def _process_tail_kinematics_shard(
    zarr_path: str,
    *,
    shape_run: str,
    run_name: str,
    row_start: int,
    row_stop: int,
    tail_angle_sample_count: int,
    block_rows: int,
    staged_source_authority: Mapping[str, Any] | None = None,
    staged_input_integrity_receipt: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Compute one worker-owned output shard in bounded row sub-blocks."""

    root = open_zarr_root(zarr_path, mode="a")
    resolved_shape_run, _shape_group, sources = _resolve_tail_kinematics_sources(
        root,
        shape_run,
        _staged_source_authority=staged_source_authority,
        _staged_input_integrity_receipt=staged_input_integrity_receipt,
    )
    if resolved_shape_run != shape_run:
        raise RuntimeError(
            f"Worker resolved subject-shape run {resolved_shape_run!r}, expected {shape_run!r}."
        )
    run_group = root["analysis"]["tail_kinematics_runs"][run_name]
    valid_count = 0
    reason_counts: dict[str, int] = {}
    read_duration = 0.0
    compute_duration = 0.0
    write_duration = 0.0
    completed_block_count = 0
    attested_input_chunks: list[str] = []
    for start in range(int(row_start), int(row_stop), int(block_rows)):
        row_slice = slice(start, min(start + int(block_rows), int(row_stop)))
        read_started = time.perf_counter()
        block_sources = _read_tail_kinematics_source_block(sources, row_slice)
        input_chunk = _staged_input_chunk_for_slice(sources, row_slice)
        if input_chunk is not None:
            attested_input_chunks.append(str(input_chunk["record_sha256"]))
        read_duration += float(time.perf_counter() - read_started)
        compute_started = time.perf_counter()
        batch = compute_tail_kinematics_from_subject_shape_arrays(
            **block_sources,
            tail_angle_sample_count=int(tail_angle_sample_count),
        )
        compute_duration += float(time.perf_counter() - compute_started)
        write_started = time.perf_counter()
        _write_tail_kinematics_batch_slice(run_group, row_slice, batch)
        write_duration += float(time.perf_counter() - write_started)
        block_valid_count, block_reason_counts = _tail_kinematics_batch_counts(batch)
        valid_count += int(block_valid_count)
        for reason, count in block_reason_counts.items():
            reason_counts[str(reason)] = int(
                reason_counts.get(str(reason), 0) + int(count)
            )
        completed_block_count += 1
    _revalidate_tail_kinematics_sources(sources)
    return {
        "row_start": int(row_start),
        "row_stop": int(row_stop),
        "valid_row_count": int(valid_count),
        "failure_reason_counts": reason_counts,
        "source_read_duration_seconds": read_duration,
        "compute_duration_seconds": compute_duration,
        "write_duration_seconds": write_duration,
        "completed_block_count": int(completed_block_count),
        "completed_worker_task_count": 1,
        "staged_input_chunk_receipt_sha256": attested_input_chunks,
    }


def _write_tail_kinematics_batch(
    run_group: zarr.Group, batch: TailKinematicsBatch
) -> None:
    """Compatibility helper for writing an already-materialized batch."""

    row_count = int(batch.valid.shape[0])
    _prepare_tail_kinematics_output_arrays(
        run_group,
        row_count=row_count,
        tail_angle_sample_s=batch.tail_angle_sample_s,
        shard_rows=None,
    )
    _write_tail_kinematics_batch_slice(run_group, slice(0, row_count), batch)


def write_tail_kinematics_run_group(
    root: zarr.Group,
    *,
    shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = "serial",
    num_workers: int = 1,
    worker_zarr_path: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    stage_command: Optional[str] = None,
    storage_profile: StorageProfile | None = None,
    _staged_source_authority: Mapping[str, Any] | None = None,
    _staged_input_integrity_receipt: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Write one tail-kinematics run from an existing subject-shape run."""

    if int(tail_angle_sample_count) < 2:
        raise ValueError("tail_angle_sample_count must be >= 2.")
    if int(block_rows) <= 0:
        raise ValueError("block_rows must be positive.")
    if int(output_shard_rows) <= 0:
        raise ValueError("output_shard_rows must be positive.")
    backend = str(execution_backend).strip().lower()
    if backend not in TAIL_KINEMATICS_EXECUTION_BACKENDS:
        raise ValueError(
            f"Unsupported execution_backend {execution_backend!r}; expected one of "
            f"{sorted(TAIL_KINEMATICS_EXECUTION_BACKENDS)!r}."
        )
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive.")
    if storage_profile is not None:
        if not isinstance(storage_profile, StorageProfile):
            raise TypeError("storage_profile must be an explicit StorageProfile.")
        if storage_profile.profile_id != TAIL_KINEMATICS_CANDIDATE_PROFILE_ID:
            raise ValueError(
                "Tail-kinematics candidate mode supports only the explicit "
                f"{TAIL_KINEMATICS_CANDIDATE_PROFILE_ID!r} profile."
            )
        if backend != "serial" or int(num_workers) != 1:
            raise ValueError(
                "Tail-kinematics byte-planner candidate mode requires one serial "
                "writer; process/Dask writes are rejected until whole-shard "
                "ownership is proven for every planned array."
            )
    if backend == "process_shards" and not dry_run and worker_zarr_path is None:
        raise ValueError(
            "process_shards requires worker_zarr_path for independent worker opens."
        )
    if (
        not dry_run
        and _staged_source_authority is not None
        and _staged_input_integrity_receipt is None
    ):
        raise ValueError(
            "Staged tail execution requires one sealed input-integrity receipt."
        )
    shape_run_name, shape_group, sources = _resolve_tail_kinematics_sources(
        root,
        shape_run,
        _staged_source_authority=_staged_source_authority,
        _staged_input_integrity_receipt=_staged_input_integrity_receipt,
    )
    row_count = int(sources.row_count)
    storage_dimensions: TailKinematicsDimensions | None = None
    storage_receipt: Any | None = None
    if storage_profile is not None:
        storage_dimensions = _candidate_tail_kinematics_dimensions(
            shape_group,
            row_count=row_count,
            tail_angle_sample_count=int(tail_angle_sample_count),
        )
        storage_receipt = build_tail_kinematics_storage_receipt(
            storage_dimensions,
            profile=storage_profile,
        )
    effective_block_rows = _effective_block_rows(
        row_count=row_count,
        requested_block_rows=int(block_rows),
    )
    effective_output_shard_rows = _effective_output_shard_rows(
        row_count=row_count,
        requested_output_shard_rows=int(output_shard_rows),
    )
    compute_block_slices = _iter_row_slices(row_count, effective_block_rows)
    worker_shard_slices = _iter_row_slices(row_count, effective_output_shard_rows)
    worker_count_effective = (
        1
        if backend == "serial"
        else min(int(num_workers), max(1, len(worker_shard_slices)))
    )
    if sources.staged_input_integrity_receipt is not None:
        receipt_slices = [
            (int(chunk["start_row"]), int(chunk["stop_row"]))
            for chunk in sources.staged_input_integrity_receipt["chunks"]
        ]
        expected_slices = [
            (int(item.start or 0), int(item.stop or 0)) for item in compute_block_slices
        ]
        if receipt_slices != expected_slices:
            raise ValueError(
                "Staged tail input receipt chunking differs from compute blocking."
            )
    output_row_chunk = int(_metric_chunks(row_count)[0])
    if backend == "process_shards":
        if len(worker_shard_slices) > 1 and (
            int(effective_output_shard_rows) < int(effective_block_rows)
            or int(effective_output_shard_rows) % int(effective_block_rows) != 0
        ):
            raise ValueError(
                "For process_shards, non-final output shards must contain a whole "
                "number of effective compute blocks. "
                f"effective_output_shard_rows={effective_output_shard_rows}, "
                f"effective_block_rows={effective_block_rows}."
            )
        _validate_process_shard_slices(
            worker_shard_slices,
            row_count=row_count,
            output_row_chunk=output_row_chunk,
            output_shard_rows=int(effective_output_shard_rows),
        )
    materialization_mode = (
        "bounded_streaming_single_writer"
        if backend == "serial"
        else "bounded_process_shards"
    )
    target_run = str(run_name or _default_run_name())
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "tail_kinematics_run": target_run,
        "source_subject_shape_run": shape_run_name,
        "source_refined_subject_masks_run": shape_group.attrs.get(
            "source_refined_subject_masks_run"
        ),
        "roi_count": int(row_count),
        "tail_angle_sample_count": int(tail_angle_sample_count),
        "materialization_mode": materialization_mode,
        "compute_kernel": TAIL_KINEMATICS_COMPUTE_KERNEL,
        "execution_backend": backend,
        "worker_count_requested": int(num_workers),
        "worker_count_effective": int(worker_count_effective),
        "worker_chunk_size_requested": int(block_rows),
        "worker_chunk_size_effective": int(effective_block_rows),
        "worker_chunk_alignment": "compute_blocks_align_to_output_row_chunks",
        "worker_shard_rows_requested": int(output_shard_rows),
        "worker_shard_rows_effective": int(effective_output_shard_rows),
        "worker_write_ownership": (
            "single_driver"
            if backend == "serial"
            else "one_complete_nonoverlapping_output_shard_per_task"
        ),
        "worker_compute_blocking": (
            "single_driver_bounded_blocks"
            if backend == "serial"
            else "bounded_subblocks_within_owned_output_shard"
        ),
        "requested_block_rows": int(block_rows),
        "effective_block_rows": int(effective_block_rows),
        "compute_block_rows_requested": int(block_rows),
        "compute_block_rows_effective": int(effective_block_rows),
        "output_row_chunk": output_row_chunk,
        "requested_output_shard_rows": int(output_shard_rows),
        "effective_output_shard_rows": int(effective_output_shard_rows),
        "output_shard_rows": int(effective_output_shard_rows),
        "output_shard_count": int(len(worker_shard_slices)),
        "output_shard_scope": (
            "canonical_tail_arrays_and_identity; copied_lineage_preserves_"
            "source_chunks_with_output_shard_as_floor"
        ),
        "worker_task_count": (
            0 if backend == "serial" else int(len(worker_shard_slices))
        ),
        "block_count": int(len(compute_block_slices)),
        "mutates_archive": not bool(dry_run),
        "storage_profile_id": (
            storage_profile.profile_id if storage_profile is not None else None
        ),
        "byte_planner_candidate": storage_profile is not None,
        "selector_eligible": False if storage_profile is not None else None,
    }
    if dry_run:
        return dict(_json_safe(summary))

    started = time.perf_counter()
    command = stage_command or (" ".join(sys.argv) if sys.argv else "unknown")
    run_group: zarr.Group | None = None
    completed_block_count = 0
    completed_worker_task_count = 0
    valid_count = 0
    reason_counts: dict[str, int] = {}
    source_read_duration_seconds_sum = 0.0
    compute_duration_seconds_sum = 0.0
    write_duration_seconds_sum = 0.0
    attested_input_chunk_receipts: list[str] = []
    publication_owner_uuid = str(uuid.uuid4())

    def record_block_result(block_result: Mapping[str, object]) -> None:
        nonlocal completed_block_count
        nonlocal completed_worker_task_count
        nonlocal valid_count
        nonlocal source_read_duration_seconds_sum
        nonlocal compute_duration_seconds_sum
        nonlocal write_duration_seconds_sum
        valid_count += int(block_result["valid_row_count"])
        for reason, count in dict(block_result["failure_reason_counts"]).items():
            key = str(reason or "")
            reason_counts[key] = int(reason_counts.get(key, 0) + int(count))
        source_read_duration_seconds_sum += float(
            block_result["source_read_duration_seconds"]
        )
        compute_duration_seconds_sum += float(block_result["compute_duration_seconds"])
        write_duration_seconds_sum += float(block_result["write_duration_seconds"])
        completed_block_count += int(block_result.get("completed_block_count", 1))
        completed_worker_task_count += int(
            block_result.get("completed_worker_task_count", 0)
        )
        raw_attestations = block_result.get("staged_input_chunk_receipt_sha256", [])
        if not isinstance(raw_attestations, Sequence) or isinstance(
            raw_attestations, (str, bytes)
        ):
            raise RuntimeError("Tail worker returned malformed input attestation.")
        attested_input_chunk_receipts.extend(str(value) for value in raw_attestations)

    try:
        run_group = _prepare_tail_kinematics_run(
            root,
            target_run=target_run,
            shape_run_name=shape_run_name,
            shape_group=shape_group,
            row_count=row_count,
            tail_angle_sample_count=int(tail_angle_sample_count),
            source_geometry_tail_sample_count=int(sources.source_sample_count),
            requested_block_rows=int(block_rows),
            effective_block_rows=int(effective_block_rows),
            requested_output_shard_rows=int(output_shard_rows),
            effective_output_shard_rows=int(effective_output_shard_rows),
            execution_backend=backend,
            worker_count_requested=int(num_workers),
            worker_count_effective=int(worker_count_effective),
            source_publication_manifest_sha256=(
                sources.source_publication_manifest_sha256
            ),
            source_authority_mode=sources.source_authority_mode,
            source_authority=sources.source_authority,
            stage_command=command,
            publication_owner_uuid=publication_owner_uuid,
            overwrite=overwrite,
            storage_receipt=storage_receipt,
            storage_dimensions=storage_dimensions,
        )
        target_s = tail_sample_positions(int(tail_angle_sample_count)).astype(
            np.float32
        )
        _prepare_tail_kinematics_output_arrays(
            run_group,
            row_count=row_count,
            tail_angle_sample_s=target_s,
            shard_rows=int(effective_output_shard_rows),
            precreated=storage_receipt is not None,
        )
        if storage_receipt is None:
            # The logical array schema is scientific publication evidence, not
            # a byte-planner-only optimization.  Stamp the ordinary maintained
            # layout as well so every selector-eligible tail publication can be
            # admitted through the same strict reader contract.
            stamp_tail_kinematics_array_schema(
                run_group,
                infer_tail_kinematics_dimensions(run_group),
                byte_planner_adopted=False,
            )
        if backend == "serial":
            for row_slice in compute_block_slices:
                read_started = time.perf_counter()
                block_sources = _read_tail_kinematics_source_block(sources, row_slice)
                input_chunk = _staged_input_chunk_for_slice(sources, row_slice)
                read_duration = float(time.perf_counter() - read_started)
                compute_started = time.perf_counter()
                batch = compute_tail_kinematics_from_subject_shape_arrays(
                    **block_sources,
                    tail_angle_sample_count=int(tail_angle_sample_count),
                )
                compute_duration = float(time.perf_counter() - compute_started)
                write_started = time.perf_counter()
                _write_tail_kinematics_batch_slice(run_group, row_slice, batch)
                write_duration = float(time.perf_counter() - write_started)
                block_valid_count, block_reason_counts = _tail_kinematics_batch_counts(
                    batch
                )
                record_block_result(
                    {
                        "valid_row_count": block_valid_count,
                        "failure_reason_counts": block_reason_counts,
                        "source_read_duration_seconds": read_duration,
                        "compute_duration_seconds": compute_duration,
                        "write_duration_seconds": write_duration,
                        "completed_block_count": 1,
                        "completed_worker_task_count": 0,
                        "staged_input_chunk_receipt_sha256": (
                            [str(input_chunk["record_sha256"])]
                            if input_chunk is not None
                            else []
                        ),
                    }
                )
        else:
            worker_path = str(Path(worker_zarr_path).expanduser().resolve())
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=int(worker_count_effective),
                mp_context=context,
            ) as pool:
                futures = [
                    pool.submit(
                        _process_tail_kinematics_shard,
                        worker_path,
                        shape_run=shape_run_name,
                        run_name=target_run,
                        row_start=int(row_slice.start or 0),
                        row_stop=int(row_slice.stop or 0),
                        tail_angle_sample_count=int(tail_angle_sample_count),
                        block_rows=int(effective_block_rows),
                        staged_source_authority=(
                            _canonical_json_copy(_staged_source_authority)
                            if _staged_source_authority is not None
                            else None
                        ),
                        staged_input_integrity_receipt=(
                            _canonical_json_copy(_staged_input_integrity_receipt)
                            if _staged_input_integrity_receipt is not None
                            else None
                        ),
                    )
                    for row_slice in worker_shard_slices
                ]
                for future in as_completed(futures):
                    record_block_result(future.result())

        _revalidate_tail_kinematics_sources(sources)

        array_schema_errors = validate_tail_kinematics_array_schema(
            run_group,
            byte_planner_adopted=storage_receipt is not None,
        )
        if array_schema_errors:
            raise RuntimeError(
                "Tail-kinematics exact array-schema validation failed: "
                + "; ".join(array_schema_errors)
            )

        staged_input_attestation = (
            _complete_staged_input_worker_attestation(
                sources.staged_input_integrity_receipt,
                attested_input_chunk_receipts,
            )
            if sources.staged_input_integrity_receipt is not None
            else None
        )

        if storage_profile is not None:
            storage_errors = validate_tail_kinematics_storage_receipt(run_group)
            if storage_errors:
                raise RuntimeError(
                    "Tail-kinematics storage candidate validation failed: "
                    + "; ".join(storage_errors)
                )

        if completed_block_count != len(compute_block_slices):
            raise RuntimeError(
                "Tail-kinematics compute block accounting mismatch: "
                f"completed={completed_block_count}, expected={len(compute_block_slices)}."
            )
        expected_worker_tasks = 0 if backend == "serial" else len(worker_shard_slices)
        if completed_worker_task_count != expected_worker_tasks:
            raise RuntimeError(
                "Tail-kinematics worker task accounting mismatch: "
                f"completed={completed_worker_task_count}, expected={expected_worker_tasks}."
            )
        duration_seconds = float(time.perf_counter() - started)
        invalid_count = int(row_count - valid_count)
        rows_per_second = (
            float(row_count / duration_seconds)
            if duration_seconds > 0.0
            else float("inf")
        )
        run_group.attrs["duration_seconds"] = duration_seconds
        run_group.attrs["rows_per_second"] = rows_per_second
        run_group.attrs["valid_row_count"] = int(valid_count)
        run_group.attrs["invalid_row_count"] = invalid_count
        run_group.attrs["completed_block_count"] = int(completed_block_count)
        run_group.attrs["completed_worker_task_count"] = int(
            completed_worker_task_count
        )
        run_group.attrs["failure_reason_counts"] = reason_counts
        run_group.attrs["source_read_duration_seconds_sum"] = (
            source_read_duration_seconds_sum
        )
        run_group.attrs["compute_duration_seconds_sum"] = compute_duration_seconds_sum
        run_group.attrs["write_duration_seconds_sum"] = write_duration_seconds_sum
        if sources.staged_input_integrity_receipt is not None:
            run_group.attrs["staged_input_integrity_receipt_sha256"] = (
                sources.staged_input_integrity_receipt["record_sha256"]
            )
            run_group.attrs["staged_input_worker_attestation"] = (
                staged_input_attestation
            )
        if sources.source_authority_mode == "canonical_publication":
            publish_tail_kinematics_coordinate_surfaces(root, run_group)
        else:
            # A node-local staged subset proves its numeric inputs but cannot
            # grant live source-camera frame authority.  The atomic publisher
            # seals the copied child against the authoritative source archive.
            run_group.attrs["tail_coordinate_publication_deferred"] = (
                "authoritative_archive_post_copy_pre_activation"
            )
        parent = root["analysis"]["tail_kinematics_runs"]
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=target_run,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command=command,
            ),
        )
        if storage_profile is not None:
            if run_group.attrs.get("stage_selector_eligible") is not False:
                raise RuntimeError(
                    "Tail-kinematics storage candidate became selector eligible."
                )
            summary.update(
                {
                    "status": "updated",
                    "valid_row_count": valid_count,
                    "invalid_row_count": invalid_count,
                    "failure_reason_counts": reason_counts,
                    "duration_seconds": duration_seconds,
                    "rows_per_second": rows_per_second,
                    "completed_block_count": int(completed_block_count),
                    "completed_worker_task_count": int(completed_worker_task_count),
                    "source_read_duration_seconds_sum": (
                        source_read_duration_seconds_sum
                    ),
                    "compute_duration_seconds_sum": compute_duration_seconds_sum,
                    "write_duration_seconds_sum": write_duration_seconds_sum,
                    "selector_eligible": False,
                    "metadata_equivalence_state": (
                        "deferred_until_authoritative_atomic_copy"
                        if sources.source_authority_mode != "canonical_publication"
                        else "direct_and_consolidated_array_declarations_equal"
                    ),
                }
            )
            if sources.source_authority_mode == "canonical_publication":
                compared = consolidate_and_validate_tail_kinematics_metadata(
                    root,
                    run_path=f"analysis/tail_kinematics_runs/{target_run}",
                )
                summary["direct_consolidated_array_declaration_count"] = int(compared)
            return dict(_json_safe(summary))
        if sources.source_authority_mode == "canonical_publication":
            summary.update(
                {
                    "status": "updated",
                    "valid_row_count": valid_count,
                    "invalid_row_count": invalid_count,
                    "failure_reason_counts": reason_counts,
                    "duration_seconds": duration_seconds,
                    "rows_per_second": rows_per_second,
                    "completed_block_count": int(completed_block_count),
                    "completed_worker_task_count": int(completed_worker_task_count),
                    "source_read_duration_seconds_sum": source_read_duration_seconds_sum,
                    "compute_duration_seconds_sum": compute_duration_seconds_sum,
                    "write_duration_seconds_sum": write_duration_seconds_sum,
                }
            )
            result = dict(_json_safe(summary))
            activate_tail_coordinate_publication(
                root,
                parent,
                run_group,
                run_name=target_run,
                expected_publication_owner_uuid=publication_owner_uuid,
            )
            return result
    except BaseException as exc:
        try:
            _cleanup_failed_tail_kinematics_candidate(
                root,
                run_name=target_run,
                publication_owner_uuid=publication_owner_uuid,
                error=exc,
                completed_block_count=completed_block_count,
                completed_worker_task_count=completed_worker_task_count,
            )
        except BaseException as cleanup_exc:
            raise RuntimeError(
                "Tail-kinematics failure cleanup could not persist the exact "
                "owned tombstone."
            ) from cleanup_exc
        raise

    summary.update(
        {
            "status": "updated",
            "valid_row_count": valid_count,
            "invalid_row_count": invalid_count,
            "failure_reason_counts": reason_counts,
            "duration_seconds": duration_seconds,
            "rows_per_second": rows_per_second,
            "completed_block_count": int(completed_block_count),
            "completed_worker_task_count": int(completed_worker_task_count),
            "source_read_duration_seconds_sum": source_read_duration_seconds_sum,
            "compute_duration_seconds_sum": compute_duration_seconds_sum,
            "write_duration_seconds_sum": write_duration_seconds_sum,
        }
    )
    return dict(_json_safe(summary))


def write_tail_kinematics_run(
    zarr_path: str | Path,
    *,
    shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = "serial",
    num_workers: int = 1,
    overwrite: bool = False,
    dry_run: bool = False,
    storage_profile: StorageProfile | None = None,
) -> dict[str, object]:
    if not dry_run:
        raise RuntimeError(
            "Direct archive tail-kinematics publication is retired because it "
            "cannot issue the maintained atomic payload receipt. Use "
            "fisheye.analysis_workflows.materializers.tail_kinematics instead."
        )
    root = open_zarr_root(zarr_path, mode="a")
    return write_tail_kinematics_run_group(
        root,
        shape_run=shape_run,
        run_name=run_name,
        tail_angle_sample_count=tail_angle_sample_count,
        block_rows=block_rows,
        output_shard_rows=output_shard_rows,
        execution_backend=execution_backend,
        num_workers=num_workers,
        worker_zarr_path=zarr_path,
        overwrite=overwrite,
        dry_run=dry_run,
        storage_profile=storage_profile,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write analysis/tail_kinematics_runs from subject-shape tail geometry."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument(
        "--shape-run",
        help="analysis/subject_shape_runs/<run> to consume; defaults to latest.",
    )
    parser.add_argument(
        "--run-name",
        help="Target analysis/tail_kinematics_runs/<run>; defaults to timestamped.",
    )
    parser.add_argument(
        "--tail-angle-sample-count",
        type=int,
        default=DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
        help="Low-dimensional behavior-facing tail samples from base to tip.",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=DEFAULT_BLOCK_ROWS,
        help=(
            "Requested rows per bounded compute block. The effective value is rounded up "
            "to the output row-chunk grid."
        ),
    )
    parser.add_argument(
        "--output-shard-rows",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_ROWS,
        help=(
            "Requested physical outer row-shard span. This is aligned to the logical "
            "row-chunk grid independently of bounded compute blocks."
        ),
    )
    parser.add_argument(
        "--execution-backend",
        choices=tuple(sorted(TAIL_KINEMATICS_EXECUTION_BACKENDS)),
        default="serial",
        help="Serial driver writes or chunk/shard-aligned node-local process workers.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--storage-profile",
        choices=(TAIL_KINEMATICS_CANDIDATE_PROFILE_ID,),
        help=(
            "Opt into the unpromoted shared byte-planner candidate. This forces "
            "one serial writer and leaves the completed run selector-ineligible."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Legacy compatibility flag; canonical public run names are immutable "
            "and an existing child cannot be reused."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs without mutating the archive.",
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = write_tail_kinematics_run(
        args.zarr_path,
        shape_run=args.shape_run,
        run_name=args.run_name,
        tail_angle_sample_count=int(args.tail_angle_sample_count),
        block_rows=int(args.block_rows),
        output_shard_rows=int(args.output_shard_rows),
        execution_backend=str(args.execution_backend),
        num_workers=int(args.num_workers),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        storage_profile=(
            get_storage_profile(args.storage_profile)
            if args.storage_profile is not None
            else None
        ),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
