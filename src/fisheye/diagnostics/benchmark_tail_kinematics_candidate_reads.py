"""Read-only source/candidate matrix for exact tail-kinematics snapshots.

The selected source is opened through Palette's maintained public reader.  The
byte-planned candidate is intentionally opened only by this diagnostic.  The
matrix never changes selectors, never treats benchmark success as promotion,
and records physical-I/O evidence as unavailable unless a future external
tracer supplies it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.analysis.tail_kinematics_io import (
    TAIL_SCALAR_SERIES,
    load_tail_kinematics_window,
)
from fisheye.analysis.tail_kinematics_schema import (
    TAIL_KINEMATICS_SOURCE_REVISION_BUNDLE,
    build_tail_kinematics_array_declarations,
    infer_tail_kinematics_dimensions,
    validate_tail_kinematics_array_schema,
)
from fisheye.analysis.tail_kinematics_runs import (
    ROW_LINEAGE_NAMES,
    SOURCE_REVISION_ARRAY_NAMES,
    SOURCE_TAIL_GEOMETRY_KIND,
    SUBJECT_SHAPE_BODY_ARRAY_NAMES,
    SUBJECT_SHAPE_BODY_FRAME_ARRAY_NAMES,
    TAIL_KINEMATICS_COMPUTE_KERNEL,
    TAIL_KINEMATICS_METHOD,
    TAIL_KINEMATICS_METHOD_VERSION,
    TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_ID,
    TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_VERSION,
    TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCOPE,
)
from fisheye.analysis.tail_kinematics_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    ANALYSIS_STORAGE_PROFILE_ID_ATTR,
    ANALYSIS_STORAGE_PROFILE_ROLE,
    ANALYSIS_STORAGE_PROFILE_ROLE_ATTR,
    validate_tail_kinematics_storage_receipt,
)
from fisheye.shared.atomic_run_publisher import (
    ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
    ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    SERIALIZATION_POLICY,
)
from fisheye.analysis_workflows.materializers.tail_kinematics import (
    PUBLISH_SCHEMA_ID,
)
from fisheye.shared import tail_coordinate_publication as tail_publication
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import peak_rss_bytes, storage_stats
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1

PARENT_PATH = "analysis/tail_kinematics_runs"
FAMILY_ID = "tail_kinematics_v2"
BENCHMARK_ID = "tail_kinematics_v2_source_candidate_read_matrix_v1"
PAIR_SCHEMA_ID = "palette.tail_kinematics.v2_pair_validation"
WORKLOAD_SCHEMA_ID = "palette.tail_kinematics.v2_read_workload"
PUBLIC_WORKLOAD_SCHEMA_ID = "palette.tail_kinematics.v2_public_consumer_workload"
TRIAL_SCHEMA_ID = "palette.tail_kinematics.v2_read_trial"
MATRIX_SCHEMA_ID = "palette.tail_kinematics.v2_read_matrix"
SCIENTIFIC_IDENTITY_SCHEMA_ID = "palette.tail_kinematics.v2_scientific_identity"
SCHEMA_VERSION = 1
DEFAULT_REPETITIONS = 5
DEFAULT_SEED = 29
DEFAULT_WINDOW_ROWS = 4096
DEFAULT_HASH_BLOCK_ROWS = 65536
MODULE_NAME = "fisheye.diagnostics.benchmark_tail_kinematics_candidate_reads"

_COPY_BACKEND_VERIFICATION = {
    "python": "sha256_all_physical_files",
    "rsync": "rsync_checksum_dry_run",
}
_SELECTOR_KEYS = (
    "latest",
    "latest_complete",
    "publication_generation",
    "publication_policy",
    "tail_publication_lease",
)
_PUBLISH_POLICY = "node_local_source_and_output_atomic_run_group_publish"
_ROLLBACK_POLICY = (
    "retain_owner_bound_failed_public_tombstone_and_"
    "stage_specific_receipt_rollback_only"
)
_PUBLISH_RECEIPT_FIELDS = {
    "staged_zarr",
    "source_run_path",
    "copy_backend",
    "source_staging",
    "schema_id",
    "publisher_contract",
    "policy",
    "serialization_policy",
    "rollback_policy",
    "published_at_utc",
    "host",
    "lsb_jobid",
    "source_zarr",
    "publication_source_run_path",
    "target_run_path",
    "publication_owner_attr",
    "publication_owner_uuid",
    "failed_public_child_policy",
    "hidden_temporary_policy",
    "copy_duration_seconds",
    "physical_copy",
    "parent_attrs_before",
    "local_validation",
    "temporary_validation",
    "pre_pointer_validation",
    "final_validation",
    "parent_attrs_after",
}
_PHYSICAL_COPY_FIELDS = {
    "backend",
    "content_sha256",
    "file_count",
    "inventory_sha256",
    "physical_bytes",
    "verification",
}
_VALIDATION_FIELDS = {
    "valid",
    "errors",
    "row_count",
    "sample_count",
    "valid_row_count",
    "invalid_row_count",
    "output_row_chunk",
    "requested_output_shard_rows",
    "effective_output_shard_rows",
    "output_shard_rows",
    "output_shard_count",
    "completed_worker_task_count",
}
_SOURCE_CONTRACT_ATTR_FIELDS = {
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "palette_run_completion_status",
    "source_refined_subject_masks_run",
    "body_frame_schema_id",
    "tail_geometry_schema_id",
}
_CANONICAL_PUBLICATION_FIELDS = {
    "manifest_ref",
    "manifest_sha256",
    "row_identity_ref",
    "row_identity_sha256",
    "tail_sample_axis_ref",
    "tail_sample_axis_sha256",
    "tail_curvature_semantics_ref",
    "tail_curvature_semantics_sha256",
    "body_frame_ref",
    "body_frame_sha256",
}
_STAGED_ARRAY_FIELDS = {
    "array_ref",
    "relative_ref",
    "dtype",
    "shape",
    "content_sha256",
    "canonicalization",
}
_REQUIRED_STAGED_SOURCE_REFS = {
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
    *ROW_LINEAGE_NAMES,
}
_ORDERED_STAGED_SOURCE_REFS = (
    *(f"components/subject_body/{name}" for name in SUBJECT_SHAPE_BODY_ARRAY_NAMES),
    *(f"body_frame/{name}" for name in SUBJECT_SHAPE_BODY_FRAME_ARRAY_NAMES),
    *ROW_LINEAGE_NAMES,
    *(f"source_refined_subject_masks/{name}" for name in SOURCE_REVISION_ARRAY_NAMES),
)
_SUPPORTED_STAGED_SOURCE_REFS = frozenset(_ORDERED_STAGED_SOURCE_REFS)
_STORAGE_STAT_FIELDS = {
    "file_count",
    "metadata_file_count",
    "payload_file_count",
    "apparent_bytes",
    "allocated_bytes",
}
_PAIR_PAYLOAD_FIELDS = {
    "archive",
    "source_run_path",
    "candidate_run_path",
    "source_consumer",
    "candidate_consumer",
    "dimensions",
    "array_count",
    "optional_revision_bundle",
    "source_schema",
    "stable_scientific_identity",
    "stable_scientific_identity_sha256",
    "candidate_storage_receipt",
    "logical_hashes",
    "logical_hashes_sha256",
    "source_coordinate_publication_sha256",
    "candidate_coordinate_publication_sha256",
    "source_subject_shape_publication_sha256",
    "publication_receipt",
    "publication_receipt_sha256",
    "publication_physical_copy_receipt_role",
    "metadata_equivalence",
    "selectors",
    "archive_guard",
    "selector_eligible_candidate",
    "profile_promoted",
    "promotion_authorized",
    "physical_io",
    "physical_io_measured",
}
_TRIAL_PAYLOAD_FIELDS = {
    "benchmark_id",
    "family_id",
    "archive",
    "source_run_name",
    "candidate_run_name",
    "role",
    "repetition_index",
    "seed",
    "window_rows",
    "driver_pid",
    "pid",
    "parent_pid",
    "created_at_utc",
    "pair_validation",
    "workload",
    "workload_result",
    "public_consumer_result",
    "timing",
    "storage",
    "peak_rss_bytes",
    "physical_io",
    "physical_io_measured",
    "profile_promoted",
    "promotion_authorized",
    "archive_guard",
    "selectors",
}
_MATRIX_PAYLOAD_FIELDS = {
    "benchmark_id",
    "family_id",
    "created_at_utc",
    "archive",
    "source_run_name",
    "candidate_run_name",
    "repetitions",
    "seed",
    "window_rows",
    "driver_pid",
    "trials",
    "process_receipts",
    "aggregate",
    "environment",
    "archive_guard",
    "physical_io",
    "physical_io_measured",
    "profile_promoted",
    "promotion_authorized",
    "promotion_decision",
}
_SCIENTIFIC_IDENTITY_FIELDS = {
    "schema_id",
    "schema_version",
    "logical_contract",
    "source_identity",
    "geometry_semantics",
    "lineage_semantics",
    "source_refs",
    "timing",
}
_SCIENTIFIC_LOGICAL_FIELDS = {
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "row_axis",
    "compute_kernel",
    "row_count",
}
_SCIENTIFIC_SOURCE_FIELDS = {
    "source_subject_shape_run",
    "source_subject_shape_path",
    "source_subject_shape_publication_manifest_sha256",
    "source_subject_shape_authority_sha256",
    "source_refined_subject_masks_run",
    "source_refined_subject_masks_revision_snapshot",
}
_SCIENTIFIC_GEOMETRY_FIELDS = {
    "source_tail_geometry_kind",
    "body_frame_convention",
    "body_frame_source",
    "tail_angle_reference_axis",
    "tail_angle_positive_direction",
    "tail_angle_units_primary",
    "tail_sample_domain",
    "tail_angle_sample_count",
    "source_geometry_tail_sample_count",
    "curvature_source",
}
_SCIENTIFIC_LINEAGE_FIELDS = {
    "acquisition_frame_index_source",
    "row_lineage_copied",
    "row_lineage_missing",
}
_SCIENTIFIC_SOURCE_REF_FIELDS = {
    "subject_shape_run",
    "subject_shape_body_component",
    "subject_shape_body_frame",
}


def _strict_envelope(schema_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    json.dumps(normalized, allow_nan=False)
    return {
        "schema_id": schema_id,
        "schema_version": SCHEMA_VERSION,
        "payload": normalized,
        "payload_digest": canonical_json_sha256(normalized),
    }


def _require_envelope(value: Mapping[str, Any], *, schema_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("Tail benchmark envelope field set differs.")
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Tail benchmark envelope schema identity differs.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Tail benchmark envelope payload is not an object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Tail benchmark envelope digest differs.")
    json.dumps(value, allow_nan=False)
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonnegative(value: object, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be finite and nonnegative.")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string.")
    if (
        not value
        or value != value.strip()
        or value in {".", "..", "latest", "latest_complete"}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return value


def _safe_archive(value: str | Path) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Archive must be one existing non-symlink directory.")
    archive = path.resolve(strict=True)
    if archive != path or not (archive / "zarr.json").is_file():
        raise ValueError("Archive path must be canonical Zarr v3 root without aliases.")
    return archive


def _guard_relative_path(archive: Path, relative: str) -> Path:
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError(f"Unsafe archive-relative path {relative!r}.")
    current = archive
    for component in relative.split("/"):
        current = current / component
        if not current.exists():
            raise FileNotFoundError(
                f"Required archive component is missing: {current}."
            )
        if current.is_symlink():
            raise ValueError(f"Archive component must not be a symlink: {current}.")
        current.resolve(strict=True).relative_to(archive)
    return current


def _guard_archive_tree(archive: Path, relative: str, *, label: str) -> Path:
    root = _guard_relative_path(archive, relative)
    if not root.is_dir():
        raise ValueError(f"{label} must be one directory hierarchy.")
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directory_names, *file_names):
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"{label} contains a forbidden symlink: {child}.")
            child.resolve(strict=True).relative_to(archive)
    return root


def _safe_output(archive: Path, value: str | Path) -> Path:
    output = Path(value).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    requested_parent = output.parent
    if not requested_parent.is_dir():
        raise ValueError("Benchmark output parent must be one existing directory.")
    if requested_parent.is_symlink():
        raise ValueError("Benchmark output parent must be canonical without aliases.")
    parent = requested_parent.resolve(strict=True)
    if parent != requested_parent:
        raise ValueError("Benchmark output parent must be canonical without aliases.")
    if parent == archive or parent.is_relative_to(archive):
        raise ValueError("Benchmark output must remain outside the source archive.")
    if archive == parent or archive.is_relative_to(parent):
        raise ValueError("Benchmark output must not contain the source archive.")
    return output


def _run_path(name: str) -> str:
    return f"{PARENT_PATH}/{name}"


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _iter_array_paths(group: Any, prefix: str = "") -> tuple[str, ...]:
    result: list[str] = []
    for name, _array in group.arrays():
        result.append(f"{prefix}/{name}" if prefix else str(name))
    for name, child in group.groups():
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        result.extend(_iter_array_paths(child, child_prefix))
    return tuple(sorted(result))


def _selector_snapshot(parent: Any) -> dict[str, Any]:
    return {key: parent.attrs.get(key) for key in _SELECTOR_KEYS}


def _metadata_guard(archive: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(archive.rglob("zarr.json")):
        if path.is_symlink():
            raise ValueError(f"Metadata file must not be a symlink: {path}.")
        payload = path.read_bytes()
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(archive).as_posix(),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not records:
        raise ValueError("Archive has no Zarr metadata files.")
    return {
        "metadata_file_count": len(records),
        "metadata_root_sha256": canonical_json_sha256(records),
        "records": records,
    }


def _archive_guard(archive: Path) -> dict[str, Any]:
    return {"metadata": _metadata_guard(archive), "storage": storage_stats(archive)}


def _require_storage_stats(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _STORAGE_STAT_FIELDS:
        raise ValueError(f"{label} storage-stat field set differs.")
    if any(type(value[field]) is not int or value[field] < 0 for field in value):
        raise ValueError(f"{label} storage-stat values differ.")
    if (
        value["file_count"]
        != value["metadata_file_count"] + value["payload_file_count"]
    ):
        raise ValueError(f"{label} storage-stat accounting differs.")


def _require_archive_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"metadata", "storage"}:
        raise ValueError("Tail archive-guard field set differs.")
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "metadata_file_count",
        "metadata_root_sha256",
        "records",
    }:
        raise ValueError("Tail metadata-guard field set differs.")
    records = metadata["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("Tail metadata-guard inventory is empty.")
    paths: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
        }:
            raise ValueError("Tail metadata-guard record field set differs.")
        path = record["path"]
        if (
            type(path) is not str
            or not path.endswith("zarr.json")
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or type(record["mtime_ns"]) is not int
            or record["mtime_ns"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise ValueError("Tail metadata-guard record is invalid.")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise ValueError("Tail metadata-guard paths are not sorted and unique.")
    if metadata["metadata_file_count"] != len(records) or metadata[
        "metadata_root_sha256"
    ] != canonical_json_sha256(records):
        raise ValueError("Tail metadata-guard digest differs.")
    _require_storage_stats(value["storage"], label="archive guard")
    if value["storage"]["metadata_file_count"] != len(records):
        raise ValueError("Tail archive-guard metadata accounting differs.")


def _hash_values(digest: Any, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(contiguous.tobytes(order="C"))


def _array_digest(array: Any, *, block_rows: int) -> str:
    shape = tuple(int(value) for value in array.shape)
    dtype = np.dtype(array.dtype)
    digest = hashlib.sha256()
    header = {
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": dtype.str,
        "shape": list(shape),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\x00")
    if not shape:
        _hash_values(digest, np.asarray(array[...]))
    elif shape[0] == 0:
        _hash_values(digest, np.empty(shape, dtype=dtype))
    else:
        for start in range(0, shape[0], int(block_rows)):
            stop = min(shape[0], start + int(block_rows))
            _hash_values(digest, np.asarray(array[start:stop]))
    return digest.hexdigest()


def _logical_hashes(group: Any, *, paths: Sequence[str]) -> dict[str, str]:
    return {
        path: _array_digest(
            _array_at_path(group, path), block_rows=DEFAULT_HASH_BLOCK_ROWS
        )
        for path in paths
    }


def _source_schema(group: Any) -> dict[str, Any]:
    dimensions = infer_tail_kinematics_dimensions(group)
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=False,
    )
    expected = {item.path for item in declarations}
    observed = set(_iter_array_paths(group))
    if observed != expected:
        raise ValueError(
            "Selected tail source does not match the exact maintained array inventory: "
            f"missing={sorted(expected - observed)!r}, unexpected={sorted(observed - expected)!r}."
        )
    errors: list[str] = []
    for declaration in declarations:
        errors.extend(
            f"{declaration.path}: {message}"
            for message in declaration.contract.validate_observation(
                _array_at_path(group, declaration.path),
                dimensions=dimensions.contract_dimensions,
            )
        )
    if errors:
        raise ValueError("Selected tail source schema is invalid: " + "; ".join(errors))
    payload = {
        "run_schema_id": "analysis.tail_kinematics_runs",
        "run_schema_version": 2,
        "dimensions": dimensions.contract_dimensions,
        "enabled_optional_bundles": (
            [TAIL_KINEMATICS_SOURCE_REVISION_BUNDLE]
            if dimensions.include_source_revision_bundle
            else []
        ),
        "declarations": [item.as_manifest() for item in declarations],
        "closed_array_inventory": True,
        "persisted_array_schema_required": False,
        "consumer": "maintained_tail_kinematics_io",
    }
    return {
        "schema_id": "palette.tail_kinematics.v2_executable_source_schema",
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _candidate_coordinate_publication(root: Any, run_path: str) -> Any:
    """Validate an ineligible candidate through the publication implementation.

    There is intentionally no public candidate reader.  Calling the internal
    loader here is confined to this diagnostic and cannot mint selector
    authority.
    """

    return tail_publication._load_tail_coordinate_publication(  # noqa: SLF001
        root,
        run_path,
        expected_selector_eligible=False,
        expected_kind="tail_kinematics",
        require_complete=True,
    )


def _require_stable_scientific_identity(
    value: object,
    *,
    dimensions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SCIENTIFIC_IDENTITY_FIELDS:
        raise ValueError("Tail stable scientific identity field set differs.")
    logical = value["logical_contract"]
    source = value["source_identity"]
    geometry = value["geometry_semantics"]
    lineage = value["lineage_semantics"]
    refs = value["source_refs"]
    timing = value["timing"]
    if (
        value["schema_id"] != SCIENTIFIC_IDENTITY_SCHEMA_ID
        or value["schema_version"] != SCHEMA_VERSION
        or not isinstance(logical, Mapping)
        or set(logical) != _SCIENTIFIC_LOGICAL_FIELDS
        or not isinstance(source, Mapping)
        or set(source) != _SCIENTIFIC_SOURCE_FIELDS
        or not isinstance(geometry, Mapping)
        or set(geometry) != _SCIENTIFIC_GEOMETRY_FIELDS
        or not isinstance(lineage, Mapping)
        or set(lineage) != _SCIENTIFIC_LINEAGE_FIELDS
        or not isinstance(refs, Mapping)
        or set(refs) != _SCIENTIFIC_SOURCE_REF_FIELDS
        or not isinstance(timing, Mapping)
        or set(timing) != {"fps"}
    ):
        raise ValueError("Tail stable scientific identity envelope differs.")
    source_run = source["source_subject_shape_run"]
    if type(source_run) is not str:
        raise ValueError("Tail stable source-subject-shape identity differs.")
    _safe_name(source_run, label="source subject-shape run")
    refined_run = source["source_refined_subject_masks_run"]
    if refined_run is not None:
        _safe_name(refined_run, label="source refined-subject-mask run")
    expected_source_path = f"analysis/subject_shape_runs/{source_run}"
    expected_refs = {
        "subject_shape_run": expected_source_path,
        "subject_shape_body_component": (
            f"{expected_source_path}/components/subject_body"
        ),
        "subject_shape_body_frame": f"{expected_source_path}/body_frame",
    }
    if (
        logical["schema_id"] != "analysis.tail_kinematics_runs"
        or logical["schema_version"] != 2
        or logical["method"] != TAIL_KINEMATICS_METHOD
        or logical["method_version"] != TAIL_KINEMATICS_METHOD_VERSION
        or logical["row_axis"] != "observation_instance"
        or logical["compute_kernel"] != TAIL_KINEMATICS_COMPUTE_KERNEL
        or type(logical["row_count"]) is not int
        or logical["row_count"] < 0
        or source["source_subject_shape_path"] != expected_source_path
        or not _is_sha256(source["source_subject_shape_publication_manifest_sha256"])
        or not _is_sha256(source["source_subject_shape_authority_sha256"])
        or type(source["source_refined_subject_masks_revision_snapshot"]) is not bool
        or geometry["source_tail_geometry_kind"] != SOURCE_TAIL_GEOMETRY_KIND
        or type(geometry["body_frame_convention"]) is not str
        or not geometry["body_frame_convention"]
        or geometry["body_frame_source"] != f"{expected_source_path}/body_frame"
        or geometry["tail_angle_reference_axis"] != "caudal_axis=-forward_axis"
        or geometry["tail_angle_positive_direction"] != "anatomical_left"
        or geometry["tail_angle_units_primary"] != "rad"
        or geometry["tail_sample_domain"] != "tail_segment_normalized_arclength"
        or type(geometry["tail_angle_sample_count"]) is not int
        or geometry["tail_angle_sample_count"] < 2
        or type(geometry["source_geometry_tail_sample_count"]) is not int
        or geometry["source_geometry_tail_sample_count"] < 2
        or geometry["curvature_source"] != "subject_shape.tail_curvature_px_inv"
        or lineage["acquisition_frame_index_source"] != "source_acquisition_frame_index"
        or lineage["row_lineage_copied"] != list(ROW_LINEAGE_NAMES)
        or lineage["row_lineage_missing"] != []
        or dict(refs) != expected_refs
        or isinstance(timing["fps"], bool)
        or not isinstance(timing["fps"], (int, float))
        or not math.isfinite(float(timing["fps"]))
        or float(timing["fps"]) <= 0.0
    ):
        raise ValueError("Tail stable scientific identity semantics differ.")
    if dimensions is not None and (
        logical["row_count"] != dimensions.get("n_rows")
        or geometry["tail_angle_sample_count"] != dimensions.get("n_tail_samples")
        or source["source_refined_subject_masks_revision_snapshot"]
        is not bool(dimensions.get("n_components") is not None)
    ):
        raise ValueError("Tail stable scientific identity dimensions differ.")
    return dict(value)


def _stable_scientific_identity(
    run: Any,
    *,
    dimensions: Any,
    expected_subject_shape_publication_sha256: str,
) -> dict[str, Any]:
    attrs = run.attrs
    authority = attrs.get("source_subject_shape_authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "schema_id",
        "schema_version",
        "authority_scope",
        "source_subject_shape_run",
        "source_subject_shape_run_ref",
        "row_count",
        "source_sample_count",
        "canonical_publication",
        "source_contract_attrs",
        "allowed_arrays",
        "closed_array_inventory",
        "normal_reader_authority",
        "record_sha256",
    }:
        raise ValueError("Tail stable subject-shape authority differs.")
    authority_record = dict(authority)
    authority_digest = authority_record.pop("record_sha256")
    canonical_publication = authority.get("canonical_publication")
    source_contract = authority.get("source_contract_attrs")
    source_run = attrs.get("source_subject_shape_run")
    source_path = f"analysis/subject_shape_runs/{source_run}"
    refined_run = attrs.get("source_refined_subject_masks_run")
    body_frame_convention = attrs.get("body_frame_convention")
    if (
        not _is_sha256(authority_digest)
        or authority_digest != canonical_json_sha256(authority_record)
        or attrs.get("source_subject_shape_authority_sha256") != authority_digest
        or not isinstance(canonical_publication, Mapping)
        or set(canonical_publication) != _CANONICAL_PUBLICATION_FIELDS
        or canonical_publication.get("manifest_sha256")
        != expected_subject_shape_publication_sha256
        or attrs.get("source_subject_shape_publication_manifest_sha256")
        != expected_subject_shape_publication_sha256
        or authority.get("source_subject_shape_run") != source_run
        or authority.get("source_subject_shape_run_ref") != f"/{source_path}"
        or authority.get("row_count") != dimensions.n_rows
        or authority.get("source_sample_count")
        != attrs.get("source_geometry_tail_sample_count")
        or not isinstance(source_contract, Mapping)
        or source_contract.get("source_refined_subject_masks_run") != refined_run
        or source_contract.get("body_frame_schema_id", "fish_anatomical_body_frame")
        != body_frame_convention
    ):
        raise ValueError("Tail stable source authority binding differs.")
    payload = {
        "schema_id": SCIENTIFIC_IDENTITY_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "logical_contract": {
            "schema_id": attrs.get("schema_id"),
            "schema_version": attrs.get("schema_version"),
            "method": attrs.get("method"),
            "method_version": attrs.get("method_version"),
            "row_axis": attrs.get("row_axis"),
            "compute_kernel": attrs.get("compute_kernel"),
            "row_count": int(dimensions.n_rows),
        },
        "source_identity": {
            "source_subject_shape_run": source_run,
            "source_subject_shape_path": attrs.get("source_subject_shape_path"),
            "source_subject_shape_publication_manifest_sha256": attrs.get(
                "source_subject_shape_publication_manifest_sha256"
            ),
            "source_subject_shape_authority_sha256": attrs.get(
                "source_subject_shape_authority_sha256"
            ),
            "source_refined_subject_masks_run": refined_run,
            "source_refined_subject_masks_revision_snapshot": attrs.get(
                "source_refined_subject_masks_revision_snapshot"
            ),
        },
        "geometry_semantics": {
            name: attrs.get(name) for name in _SCIENTIFIC_GEOMETRY_FIELDS
        },
        "lineage_semantics": {
            name: attrs.get(name) for name in _SCIENTIFIC_LINEAGE_FIELDS
        },
        "source_refs": attrs.get("source_refs"),
        "timing": {"fps": attrs.get("fps")},
    }
    return _require_stable_scientific_identity(
        payload,
        dimensions=dimensions.contract_dimensions,
    )


def _require_validation_receipt(
    value: object,
    *,
    dimensions: Mapping[str, int],
    candidate: Any,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _VALIDATION_FIELDS:
        raise ValueError(f"{label} validation field set differs.")
    n_rows = int(dimensions["n_rows"])
    n_samples = int(dimensions["n_tail_samples"])
    if (
        value["valid"] is not True
        or value["errors"] != []
        or value["row_count"] != n_rows
        or value["sample_count"] != n_samples
        or type(value["valid_row_count"]) is not int
        or type(value["invalid_row_count"]) is not int
        or value["valid_row_count"] + value["invalid_row_count"] != n_rows
    ):
        raise ValueError(f"{label} validation does not bind live dimensions.")
    for field in _VALIDATION_FIELDS - {
        "valid",
        "errors",
        "row_count",
        "sample_count",
        "valid_row_count",
        "invalid_row_count",
    }:
        if value[field] != candidate.attrs.get(field):
            raise ValueError(
                f"{label} validation field {field!r} differs from live run."
            )


def _require_source_staging(
    value: object,
    *,
    archive: Path,
    expected_staged_zarr: Path,
    expected_row_count: int,
) -> None:
    fields = {
        "schema_id",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "duration_seconds",
        "mib_per_second",
        "copy_backend",
        "host",
        "lsb_jobid",
        "source_zarr",
        "staged_zarr",
        "shape_run",
        "row_count",
        "selected_paths",
        "source_metadata_sha256",
        "source_contract",
        "staged_source_authority_sha256",
        "staged_source_authority",
        "inventory",
        "capacity",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Candidate source-staging receipt field set differs.")
    authority = value["staged_source_authority"]
    if not isinstance(authority, Mapping):
        raise ValueError("Candidate staged-source authority is not an object.")
    authority_fields = {
        "schema_id",
        "schema_version",
        "authority_scope",
        "source_subject_shape_run",
        "source_subject_shape_run_ref",
        "row_count",
        "source_sample_count",
        "canonical_publication",
        "source_contract_attrs",
        "allowed_arrays",
        "closed_array_inventory",
        "normal_reader_authority",
        "record_sha256",
    }
    if (
        set(authority) != authority_fields
        or not isinstance(authority["source_contract_attrs"], Mapping)
        or not isinstance(authority["allowed_arrays"], Mapping)
    ):
        raise ValueError("Candidate staged-source authority field set differs.")
    staged_path = Path(str(value["staged_zarr"]))
    if (
        value["schema_id"] != "palette.tail_kinematics_source_staging.v1"
        or value["status"] != "complete"
        or value["source_zarr"] != str(archive)
        or not staged_path.is_absolute()
        or staged_path != expected_staged_zarr
        or value["copy_backend"] not in _COPY_BACKEND_VERIFICATION
        or value["row_count"] != expected_row_count
        or not _is_sha256(value["source_metadata_sha256"])
        or value["staged_source_authority_sha256"] != authority.get("record_sha256")
        or not _is_sha256(value["staged_source_authority_sha256"])
        or not isinstance(value["selected_paths"], list)
        or type(value["shape_run"]) is not str
        or not value["shape_run"]
        or any(
            type(value[name]) is not str or not value[name]
            for name in ("started_at_utc", "completed_at_utc", "host")
        )
        or (value["lsb_jobid"] is not None and type(value["lsb_jobid"]) is not str)
    ):
        raise ValueError("Candidate source-staging identity differs.")
    source_contract = value["source_contract"]
    allowed_source_contract_fields = {
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "palette_run_completion_status",
        "source_refined_subject_masks_run",
        "body_frame_schema_id",
        "tail_geometry_schema_id",
        "canonical_publication_manifest_sha256",
        "staged_source_authority_sha256",
    }
    if (
        not isinstance(source_contract, Mapping)
        or not {
            "schema_id",
            "schema_version",
            "canonical_publication_manifest_sha256",
            "staged_source_authority_sha256",
        }.issubset(source_contract)
        or set(source_contract)
        - {
            "canonical_publication_manifest_sha256",
            "staged_source_authority_sha256",
        }
        != set(authority.get("source_contract_attrs", {}))
        or not set(source_contract).issubset(allowed_source_contract_fields)
        or not _is_sha256(source_contract["canonical_publication_manifest_sha256"])
        or source_contract["staged_source_authority_sha256"]
        != value["staged_source_authority_sha256"]
    ):
        raise ValueError("Candidate source-staging source contract differs.")
    authority_record = dict(authority)
    authority_digest = authority_record.pop("record_sha256")
    if (
        authority_digest != canonical_json_sha256(authority_record)
        or authority["schema_id"] != TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_ID
        or authority["schema_version"]
        != TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_VERSION
        or authority["authority_scope"] != TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCOPE
        or authority["source_subject_shape_run"] != value["shape_run"]
        or authority["source_subject_shape_run_ref"]
        != f"/analysis/subject_shape_runs/{value['shape_run']}"
        or authority["closed_array_inventory"] is not True
        or authority["normal_reader_authority"] is not False
        or authority["row_count"] != value["row_count"]
        or set(authority["source_contract_attrs"]) - _SOURCE_CONTRACT_ATTR_FIELDS
        or dict(authority["source_contract_attrs"])
        != {
            key: source_contract[key]
            for key in source_contract
            if key in _SOURCE_CONTRACT_ATTR_FIELDS
        }
        or type(authority["source_sample_count"]) is not int
        or authority["source_sample_count"] < 2
    ):
        raise ValueError("Candidate staged-source authority digest differs.")
    publication = authority["canonical_publication"]
    if (
        not isinstance(publication, Mapping)
        or set(publication) != _CANONICAL_PUBLICATION_FIELDS
        or any(
            not _is_sha256(publication[name])
            for name in _CANONICAL_PUBLICATION_FIELDS
            if name.endswith("_sha256")
        )
        or any(
            type(publication[name]) is not str or not publication[name].startswith("/")
            for name in _CANONICAL_PUBLICATION_FIELDS
            if name.endswith("_ref")
        )
        or source_contract["canonical_publication_manifest_sha256"]
        != publication["manifest_sha256"]
    ):
        raise ValueError("Candidate staged-source canonical publication differs.")
    allowed_arrays = authority["allowed_arrays"]
    allowed_names = set(allowed_arrays)
    if not _REQUIRED_STAGED_SOURCE_REFS.issubset(
        allowed_names
    ) or not allowed_names.issubset(_SUPPORTED_STAGED_SOURCE_REFS):
        raise ValueError("Candidate staged-source array inventory differs.")
    run_prefix = f"analysis/subject_shape_runs/{value['shape_run']}"
    expected_selected_paths = [
        f"{run_prefix}/{relative_ref}"
        for relative_ref in _ORDERED_STAGED_SOURCE_REFS
        if relative_ref in allowed_names
    ]
    if value["selected_paths"] != expected_selected_paths:
        raise ValueError("Candidate staged-source selected-path order differs.")
    for relative_ref, declaration in allowed_arrays.items():
        if (
            not isinstance(declaration, Mapping)
            or set(declaration) != _STAGED_ARRAY_FIELDS
            or declaration["relative_ref"] != relative_ref
            or declaration["array_ref"] != f"/{run_prefix}/{relative_ref}"
            or declaration["canonicalization"] != "numpy_dtype_shape_c_order_bytes_v1"
            or not _is_sha256(declaration["content_sha256"])
            or type(declaration["dtype"]) is not str
            or not isinstance(declaration["shape"], list)
            or not declaration["shape"]
            or any(
                type(dimension) is not int or dimension < 0
                for dimension in declaration["shape"]
            )
        ):
            raise ValueError("Candidate staged-source array declaration differs.")
    _require_nonnegative(value["duration_seconds"], label="staging duration")
    if value["mib_per_second"] is not None:
        _require_nonnegative(value["mib_per_second"], label="staging throughput")
    inventory = value["inventory"]
    if not isinstance(inventory, Mapping) or set(inventory) != {
        "valid",
        "expected_file_count",
        "observed_file_count",
        "expected_bytes",
        "observed_bytes",
        "expected_inventory_sha256",
        "observed_inventory_sha256",
        "missing",
        "size_mismatches",
    }:
        raise ValueError("Candidate source-staging inventory field set differs.")
    if (
        inventory["valid"] is not True
        or inventory["missing"] != []
        or inventory["size_mismatches"] != []
        or inventory["expected_file_count"] != inventory["observed_file_count"]
        or inventory["expected_bytes"] != inventory["observed_bytes"]
        or inventory["expected_inventory_sha256"]
        != inventory["observed_inventory_sha256"]
        or not _is_sha256(inventory["expected_inventory_sha256"])
    ):
        raise ValueError("Candidate source-staging inventory differs.")
    capacity = value["capacity"]
    if not isinstance(capacity, Mapping) or set(capacity) != {
        "check_enabled",
        "free_bytes_before_copy",
        "required_bytes_estimate",
        "estimated_output_bytes",
        "margin_bytes",
    }:
        raise ValueError("Candidate source-staging capacity field set differs.")
    if type(capacity["check_enabled"]) is not bool or any(
        type(capacity[field]) is not int or capacity[field] < 0
        for field in set(capacity) - {"check_enabled"}
    ):
        raise ValueError("Candidate source-staging capacity values differ.")
    if capacity["required_bytes_estimate"] != (
        inventory["expected_bytes"]
        + capacity["estimated_output_bytes"]
        + capacity["margin_bytes"]
    ):
        raise ValueError("Candidate source-staging capacity arithmetic differs.")


def _publication_receipt(
    candidate: Any,
    *,
    archive: Path,
    candidate_name: str,
    dimensions: Mapping[str, int],
    expected_parent_attrs: Mapping[str, Any],
) -> dict[str, Any]:
    raw = candidate.attrs.get("cluster_output_staging")
    if not isinstance(raw, Mapping) or set(raw) != _PUBLISH_RECEIPT_FIELDS:
        raise ValueError("Candidate atomic publication receipt field set differs.")
    receipt = dict(raw)
    contract = receipt.get("publisher_contract")
    if contract != {
        "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
        "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    }:
        raise ValueError("Candidate publisher contract differs.")
    target = archive.joinpath(*_run_path(candidate_name).split("/"))
    source_path = Path(str(receipt.get("source_zarr", ""))).expanduser().absolute()
    target_path = Path(str(receipt.get("target_run_path", ""))).expanduser().absolute()
    publication_source = Path(str(receipt.get("publication_source_run_path", "")))
    physical = receipt.get("physical_copy")
    owner = candidate.attrs.get(tail_publication.TAIL_PUBLICATION_OWNER_ATTR)
    if not isinstance(physical, Mapping) or set(physical) != _PHYSICAL_COPY_FIELDS:
        raise ValueError("Candidate physical-copy field set differs.")
    if (
        receipt.get("schema_id") != PUBLISH_SCHEMA_ID
        or receipt.get("policy") != _PUBLISH_POLICY
        or receipt.get("serialization_policy") != SERIALIZATION_POLICY
        or receipt.get("rollback_policy") != _ROLLBACK_POLICY
        or source_path != archive
        or target_path != target
        or receipt.get("publication_owner_attr")
        != tail_publication.TAIL_PUBLICATION_OWNER_ATTR
        or receipt.get("publication_owner_uuid") != owner
        or not publication_source.is_absolute()
        or publication_source.is_relative_to(archive)
        or tuple(publication_source.parts[-3:])
        != tuple(_run_path(candidate_name).split("/"))
        or receipt.get("failed_public_child_policy")
        != "retain_owner_bound_selector_ineligible_tombstone"
        or receipt.get("hidden_temporary_policy")
        != "same_parent_hidden_sibling_then_os_replace"
        or receipt.get("copy_backend") != physical.get("backend")
        or receipt.get("source_run_path") != receipt.get("publication_source_run_path")
        or receipt.get("staged_zarr")
        != str(Path(receipt["publication_source_run_path"]).parents[2])
    ):
        raise ValueError(
            "Candidate atomic publication receipt differs from live state."
        )
    _require_nonnegative(receipt["copy_duration_seconds"], label="publication copy")
    if type(receipt["published_at_utc"]) is not str or not receipt["published_at_utc"]:
        raise ValueError("Candidate publication timestamp differs.")
    if type(receipt["host"]) is not str or not receipt["host"]:
        raise ValueError("Candidate publication host differs.")
    if receipt["lsb_jobid"] is not None and type(receipt["lsb_jobid"]) is not str:
        raise ValueError("Candidate publication LSF identity differs.")
    if (
        _COPY_BACKEND_VERIFICATION.get(physical["backend"]) != physical["verification"]
        or not _is_sha256(physical["content_sha256"])
        or not _is_sha256(physical["inventory_sha256"])
        or type(physical["file_count"]) is not int
        or physical["file_count"] < 1
        or type(physical["physical_bytes"]) is not int
        or physical["physical_bytes"] < 1
    ):
        raise ValueError("Candidate physical-copy evidence differs.")
    for field in (
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
    ):
        _require_validation_receipt(
            receipt[field],
            dimensions=dimensions,
            candidate=candidate,
            label=f"Candidate publication {field}",
        )
    before = receipt["parent_attrs_before"]
    after = receipt["parent_attrs_after"]
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or set(before) != {PARENT_PATH}
        or dict(before) != dict(after)
        or dict(before[PARENT_PATH]) != dict(expected_parent_attrs)
    ):
        raise ValueError("Candidate publication parent selector snapshot differs.")
    _require_source_staging(
        receipt["source_staging"],
        archive=archive,
        expected_staged_zarr=Path(str(receipt["staged_zarr"])),
        expected_row_count=int(dimensions["n_rows"]),
    )
    return receipt


def validate_pair(
    archive: str | Path,
    *,
    source_run_name: str,
    candidate_run_name: str,
) -> dict[str, Any]:
    archive_path = _safe_archive(archive)
    source_name = _safe_name(source_run_name, label="source run")
    candidate_name = _safe_name(candidate_run_name, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")
    _guard_archive_tree(
        archive_path, _run_path(source_name), label="Selected tail source"
    )
    _guard_archive_tree(
        archive_path, _run_path(candidate_name), label="Selected tail candidate"
    )
    archive_before = _archive_guard(archive_path)
    root = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    parent = root[PARENT_PATH]
    source = parent[source_name]
    candidate = parent[candidate_name]
    selectors = _selector_snapshot(parent)
    if (
        source.attrs.get("schema_id") != "analysis.tail_kinematics_runs"
        or source.attrs.get("schema_version") != 2
        or source.attrs.get("palette_run_completion_status") != "complete"
        or source.attrs.get("stage_selector_eligible") is not True
        or selectors.get("latest") != source_name
        or selectors.get("latest_complete") != source_name
    ):
        raise ValueError(
            "Selected tail source is not the exact active complete v2 run."
        )
    if (
        candidate.attrs.get("schema_id") != "analysis.tail_kinematics_runs"
        or candidate.attrs.get("schema_version") != 2
        or candidate.attrs.get("palette_run_completion_status") != "complete"
        or candidate.attrs.get("stage_selector_eligible") is not False
        or candidate.attrs.get("byte_planner_adopted") is not True
        or candidate.attrs.get("storage_candidate_status")
        != "unpromoted_selector_ineligible"
        or candidate.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
        != ANALYSIS_STORAGE_PROFILE_ROLE
        or candidate.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR) != "published_http_v1"
        or candidate.attrs.get("storage_candidate_profile_promoted") is True
    ):
        raise ValueError("Tail candidate is not one explicit unpromoted v2 candidate.")

    source_schema = _source_schema(source)
    candidate_schema_errors = validate_tail_kinematics_array_schema(
        candidate, byte_planner_adopted=True
    )
    storage_errors = validate_tail_kinematics_storage_receipt(candidate)
    if candidate_schema_errors or storage_errors:
        raise ValueError(
            "Tail candidate schema/storage validation failed: "
            + "; ".join((*candidate_schema_errors, *storage_errors))
        )
    source_dimensions = infer_tail_kinematics_dimensions(source)
    candidate_dimensions = infer_tail_kinematics_dimensions(candidate)
    if source_dimensions != candidate_dimensions:
        raise ValueError("Source and candidate tail dimensions differ.")
    paths = tuple(
        item.path
        for item in build_tail_kinematics_array_declarations(
            include_source_revision_bundle=(
                source_dimensions.include_source_revision_bundle
            ),
            byte_planner_adopted=True,
        )
    )
    source_hashes = _logical_hashes(source, paths=paths)
    candidate_hashes = _logical_hashes(candidate, paths=paths)
    if candidate_hashes != source_hashes:
        raise ValueError("Complete tail decoded equality differs.")

    source_publication = tail_publication.load_tail_kinematics_coordinate_publication(
        root, _run_path(source_name)
    )
    candidate_publication = _candidate_coordinate_publication(
        root, _run_path(candidate_name)
    )
    if (
        source_publication.source.manifest.record_sha256
        != candidate_publication.source.manifest.record_sha256
    ):
        raise ValueError(
            "Source and candidate bind different subject-shape authorities."
        )
    persisted_receipt = candidate.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(persisted_receipt, Mapping):
        raise ValueError("Candidate storage receipt is missing.")
    parsed = analysis_storage_plan_receipt_from_manifest(persisted_receipt)
    if parsed.as_manifest() != dict(persisted_receipt):
        raise ValueError("Candidate storage receipt does not round-trip exactly.")
    if (
        parsed.profile != PUBLISHED_HTTP_V1
        or parsed.profile.as_manifest() != PUBLISHED_HTTP_V1.as_manifest()
    ):
        raise ValueError(
            "Candidate storage receipt does not pin the exact registered "
            "published_http_v1 profile."
        )
    source_identity = _stable_scientific_identity(
        source,
        dimensions=source_dimensions,
        expected_subject_shape_publication_sha256=(
            source_publication.source.manifest.record_sha256
        ),
    )
    candidate_identity = _stable_scientific_identity(
        candidate,
        dimensions=candidate_dimensions,
        expected_subject_shape_publication_sha256=(
            candidate_publication.source.manifest.record_sha256
        ),
    )
    if candidate_identity != source_identity:
        raise ValueError("Source and candidate stable scientific identities differ.")
    publication = _publication_receipt(
        candidate,
        archive=archive_path,
        candidate_name=candidate_name,
        dimensions=source_dimensions.contract_dimensions,
        expected_parent_attrs=dict(parent.attrs),
    )
    source_metadata = validate_direct_consolidated_subtree(
        archive_path, subtree_path=_run_path(source_name)
    )
    candidate_metadata = validate_direct_consolidated_subtree(
        archive_path, subtree_path=_run_path(candidate_name)
    )
    if int(source_metadata.array_count) != len(paths) or int(
        candidate_metadata.array_count
    ) != len(paths):
        raise ValueError("Direct/consolidated metadata array count differs.")
    archive_after = _archive_guard(archive_path)
    if archive_after != archive_before:
        raise RuntimeError("Tail pair validation changed archive metadata or storage.")
    payload = {
        "archive": str(archive_path),
        "source_run_path": _run_path(source_name),
        "candidate_run_path": _run_path(candidate_name),
        "source_consumer": {
            "public_consumer_implemented": True,
            "consumer": "load_tail_kinematics_window",
            "authority": "selected_complete_tail_kinematics_v2",
        },
        "candidate_consumer": {
            "public_consumer_implemented": False,
            "diagnostic_consumer_implemented": True,
            "consumer": MODULE_NAME,
            "authority": "diagnostic_only_explicit_candidate",
        },
        "dimensions": source_dimensions.contract_dimensions,
        "array_count": len(paths),
        "optional_revision_bundle": (source_dimensions.include_source_revision_bundle),
        "source_schema": source_schema,
        "stable_scientific_identity": source_identity,
        "stable_scientific_identity_sha256": canonical_json_sha256(source_identity),
        "candidate_storage_receipt": dict(persisted_receipt),
        "logical_hashes": source_hashes,
        "logical_hashes_sha256": canonical_json_sha256(source_hashes),
        "source_coordinate_publication_sha256": (
            source_publication.manifest.record_sha256
        ),
        "candidate_coordinate_publication_sha256": (
            candidate_publication.manifest.record_sha256
        ),
        "source_subject_shape_publication_sha256": (
            source_publication.source.manifest.record_sha256
        ),
        "publication_receipt": publication,
        "publication_receipt_sha256": canonical_json_sha256(publication),
        "publication_physical_copy_receipt_role": (
            "opaque_publisher_provenance_not_replayed_not_benchmark_authority"
        ),
        "metadata_equivalence": {
            "source": source_metadata.to_json(),
            "candidate": candidate_metadata.to_json(),
        },
        "selectors": selectors,
        "archive_guard": archive_after,
        "selector_eligible_candidate": False,
        "profile_promoted": False,
        "promotion_authorized": False,
        "physical_io": None,
        "physical_io_measured": False,
    }
    return _strict_envelope(PAIR_SCHEMA_ID, payload)


def _require_pair_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _require_envelope(value, schema_id=PAIR_SCHEMA_ID)
    if set(payload) != _PAIR_PAYLOAD_FIELDS:
        raise ValueError("Tail pair payload field set differs.")
    if (
        payload["source_consumer"]
        != {
            "public_consumer_implemented": True,
            "consumer": "load_tail_kinematics_window",
            "authority": "selected_complete_tail_kinematics_v2",
        }
        or payload["candidate_consumer"]
        != {
            "public_consumer_implemented": False,
            "diagnostic_consumer_implemented": True,
            "consumer": MODULE_NAME,
            "authority": "diagnostic_only_explicit_candidate",
        }
        or payload["selector_eligible_candidate"] is not False
        or payload["profile_promoted"] is not False
        or payload["promotion_authorized"] is not False
        or payload["physical_io"] is not None
        or payload["physical_io_measured"] is not False
        or payload["publication_physical_copy_receipt_role"]
        != "opaque_publisher_provenance_not_replayed_not_benchmark_authority"
    ):
        raise ValueError("Tail pair consumer or nonpromotion boundary differs.")
    optional = payload["optional_revision_bundle"]
    expected_count = 23 if optional is True else 21 if optional is False else -1
    hashes = payload["logical_hashes"]
    if (
        payload["array_count"] != expected_count
        or not isinstance(hashes, Mapping)
        or len(hashes) != expected_count
        or not all(
            type(path) is str and _is_sha256(digest) for path, digest in hashes.items()
        )
        or payload["logical_hashes_sha256"] != canonical_json_sha256(hashes)
    ):
        raise ValueError("Tail pair logical-hash binding differs.")
    for field in (
        "source_coordinate_publication_sha256",
        "candidate_coordinate_publication_sha256",
        "source_subject_shape_publication_sha256",
        "publication_receipt_sha256",
    ):
        if not _is_sha256(payload[field]):
            raise ValueError(f"Tail pair digest {field!r} differs.")
    if payload["publication_receipt_sha256"] != canonical_json_sha256(
        payload["publication_receipt"]
    ):
        raise ValueError("Tail pair publication receipt digest differs.")
    identity = _require_stable_scientific_identity(
        payload["stable_scientific_identity"],
        dimensions=payload["dimensions"],
    )
    if (
        payload["stable_scientific_identity_sha256"] != canonical_json_sha256(identity)
        or identity["source_identity"][
            "source_subject_shape_publication_manifest_sha256"
        ]
        != payload["source_subject_shape_publication_sha256"]
    ):
        raise ValueError("Tail pair stable scientific identity digest differs.")
    receipt = payload["candidate_storage_receipt"]
    if not isinstance(receipt, Mapping):
        raise ValueError("Tail pair storage receipt does not replay.")
    parsed_receipt = analysis_storage_plan_receipt_from_manifest(receipt)
    if (
        parsed_receipt.as_manifest() != dict(receipt)
        or parsed_receipt.profile != PUBLISHED_HTTP_V1
        or parsed_receipt.profile.as_manifest() != PUBLISHED_HTTP_V1.as_manifest()
    ):
        raise ValueError("Tail pair storage receipt does not pin the exact profile.")
    metadata = payload["metadata_equivalence"]
    if not isinstance(metadata, Mapping) or set(metadata) != {"source", "candidate"}:
        raise ValueError("Tail pair metadata-equivalence field set differs.")
    for role, value_part in metadata.items():
        if not isinstance(value_part, Mapping) or set(value_part) != {
            "schema_id",
            "schema_version",
            "subtree_path",
            "node_count",
            "group_count",
            "array_count",
            "declarations_sha256",
        }:
            raise ValueError(f"Tail pair {role} metadata receipt differs.")
        if value_part["array_count"] != expected_count or not _is_sha256(
            value_part["declarations_sha256"]
        ):
            raise ValueError(f"Tail pair {role} metadata accounting differs.")
    _require_archive_guard(payload["archive_guard"])
    return payload


def _deterministic_starts(*, n_rows: int, window_rows: int, seed: int) -> list[int]:
    if n_rows <= window_rows:
        return [0]
    limit = n_rows - window_rows
    anchors = {0, limit // 2, limit}
    rng = np.random.default_rng(int(seed))
    for value in rng.integers(0, limit + 1, size=min(8, limit + 1)):
        anchors.add(int(value))
    return sorted(anchors)


def build_workload(
    group: Any,
    *,
    role: str,
    window_rows: int,
    seed: int,
) -> dict[str, Any]:
    if role not in {"source", "candidate"}:
        raise ValueError("Workload role must be source or candidate.")
    dimensions = infer_tail_kinematics_dimensions(group)
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=(role == "candidate"),
    )
    rows = min(max(1, int(window_rows)), max(1, dimensions.n_rows))
    starts = _deterministic_starts(
        n_rows=dimensions.n_rows,
        window_rows=rows,
        seed=seed,
    )
    accesses = []
    for declaration in declarations:
        node = _array_at_path(group, declaration.path)
        row_aligned = bool(node.shape) and int(node.shape[0]) == dimensions.n_rows
        accesses.append(
            {
                "path": declaration.path,
                "mode": "windowed" if row_aligned else "eager",
                "starts": starts if row_aligned else [],
                "rows": rows if row_aligned else None,
            }
        )
    payload = {
        "role": role,
        "seed": int(seed),
        "n_rows": dimensions.n_rows,
        "window_rows": rows,
        "accesses": accesses,
        "physical_io": None,
    }
    return _strict_envelope(WORKLOAD_SCHEMA_ID, payload)


def _require_workload(
    group: Any,
    workload: Mapping[str, Any],
    *,
    role: str,
    seed: int,
    window_rows: int,
) -> Mapping[str, Any]:
    expected = build_workload(
        group, role=role, seed=int(seed), window_rows=int(window_rows)
    )
    if dict(workload) != expected:
        raise ValueError("Tail workload differs from live declaration reconstruction.")
    payload = _require_envelope(workload, schema_id=WORKLOAD_SCHEMA_ID)
    if set(payload) != {
        "role",
        "seed",
        "n_rows",
        "window_rows",
        "accesses",
        "physical_io",
    }:
        raise ValueError("Tail workload payload field set differs.")
    accesses = payload["accesses"]
    if not isinstance(accesses, list) or any(
        not isinstance(access, Mapping)
        or set(access) != {"path", "mode", "starts", "rows"}
        for access in accesses
    ):
        raise ValueError("Tail workload access field set differs.")
    return payload


def execute_workload(
    group: Any,
    workload: Mapping[str, Any],
    *,
    role: str,
    seed: int,
    window_rows: int,
) -> dict[str, Any]:
    payload = _require_workload(
        group,
        workload,
        role=role,
        seed=seed,
        window_rows=window_rows,
    )
    digest = hashlib.sha256()
    bytes_decoded = 0
    read_count = 0
    for access in payload.get("accesses", []):
        if not isinstance(access, Mapping):
            raise ValueError("Tail workload access is not an object.")
        path = str(access.get("path"))
        array = _array_at_path(group, path)
        if access.get("mode") == "eager":
            values = np.asarray(array[:])
            _hash_values(digest, values)
            bytes_decoded += int(values.nbytes)
            read_count += 1
        elif access.get("mode") == "windowed":
            rows = int(access.get("rows", 0))
            for start_value in access.get("starts", []):
                start = int(start_value)
                values = np.asarray(array[start : start + rows])
                _hash_values(digest, values)
                bytes_decoded += int(values.nbytes)
                read_count += 1
        else:
            raise ValueError(f"Unsupported tail workload mode for {path!r}.")
    return {
        "workload_payload_digest": workload["payload_digest"],
        "decoded_digest": digest.hexdigest(),
        "decoded_bytes": bytes_decoded,
        "read_count": read_count,
        "physical_io": None,
        "physical_io_measured": False,
    }


def _public_consumer_workload(source: Any, *, window_rows: int) -> dict[str, Any]:
    frames = np.asarray(source["source_acquisition_frame_index"][:], dtype=np.int64)
    if frames.size == 0:
        raise ValueError("Public source workload requires at least one tail row.")
    fps = float(source.attrs.get("fps", 700.0))
    stop_index = min(frames.size, max(1, int(window_rows))) - 1
    return {
        "run_name": str(source.name).split("/")[-1],
        "start_s": float(frames[0]) / fps,
        "stop_s": float(frames[stop_index]) / fps,
        "scalar_series": list(TAIL_SCALAR_SERIES),
        "include_native_angles": True,
        "include_dense_curvature": False,
        "max_rows": max(int(window_rows) * 4, 1024),
    }


def _execute_public_consumer(
    root: Any, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    window = load_tail_kinematics_window(root, **dict(parameters))
    digest = hashlib.sha256()
    arrays = [window.frame_indices, window.time_seconds, window.valid, window.angle_deg]
    arrays.extend(window.scalar_series[name] for name in sorted(window.scalar_series))
    for values in arrays:
        _hash_values(digest, np.asarray(values))
    payload = {
        "parameters": dict(parameters),
        "decoded_digest": digest.hexdigest(),
        "row_count": int(np.asarray(window.frame_indices).size),
        "consumer": "load_tail_kinematics_window",
    }
    return _strict_envelope(PUBLIC_WORKLOAD_SCHEMA_ID, payload)


def _reconstruct_public_consumer(
    root: Any,
    *,
    source_name: str,
    window_rows: int,
) -> dict[str, Any]:
    source = root[_run_path(source_name)]
    parameters = _public_consumer_workload(source, window_rows=window_rows)
    return _execute_public_consumer(root, parameters)


def run_trial(
    archive: str | Path,
    *,
    source_run_name: str,
    candidate_run_name: str,
    role: str,
    repetition_index: int,
    seed: int,
    window_rows: int,
    driver_pid: int,
) -> dict[str, Any]:
    archive_path = _safe_archive(archive)
    source_name = _safe_name(source_run_name, label="source run")
    candidate_name = _safe_name(candidate_run_name, label="candidate run")
    if role not in {"source", "candidate"}:
        raise ValueError("Trial role must be source or candidate.")
    if type(driver_pid) is not int or driver_pid <= 0 or os.getppid() != driver_pid:
        raise ValueError("Trial is not a direct child of its declared matrix driver.")
    _guard_archive_tree(
        archive_path, _run_path(source_name), label="Selected tail source"
    )
    _guard_archive_tree(
        archive_path, _run_path(candidate_name), label="Selected tail candidate"
    )
    archive_before = _archive_guard(archive_path)
    root = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    parent = root[PARENT_PATH]
    selectors_before = _selector_snapshot(parent)
    validate_started = time.perf_counter()
    pair = validate_pair(
        archive_path,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
    )
    validation_seconds = time.perf_counter() - validate_started
    selected_name = source_name if role == "source" else candidate_name
    selected = root[_run_path(selected_name)]
    workload = build_workload(selected, role=role, window_rows=window_rows, seed=seed)
    read_started = time.perf_counter()
    result = execute_workload(
        selected,
        workload,
        role=role,
        seed=seed,
        window_rows=window_rows,
    )
    read_seconds = time.perf_counter() - read_started
    public_result: dict[str, Any] | None = None
    if role == "source":
        public_result = _reconstruct_public_consumer(
            root, source_name=source_name, window_rows=window_rows
        )
    selectors_after = _selector_snapshot(parent)
    archive_after = _archive_guard(archive_path)
    pair_payload = _require_pair_receipt(pair)
    if (
        selectors_after != selectors_before
        or selectors_after != pair_payload["selectors"]
    ):
        raise RuntimeError("Read-only tail trial changed parent selectors.")
    if (
        archive_after != archive_before
        or archive_after != pair_payload["archive_guard"]
    ):
        raise RuntimeError("Read-only tail trial changed archive metadata or storage.")
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive": str(archive_path),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "role": role,
        "repetition_index": int(repetition_index),
        "seed": int(seed),
        "window_rows": int(window_rows),
        "driver_pid": int(driver_pid),
        "pid": int(os.getpid()),
        "parent_pid": int(os.getppid()),
        "created_at_utc": _now_utc(),
        "pair_validation": pair,
        "workload": workload,
        "workload_result": result,
        "public_consumer_result": public_result,
        "timing": {
            "pair_validation_seconds": float(validation_seconds),
            "workload_seconds": float(read_seconds),
        },
        "storage": storage_stats(archive_path),
        "peak_rss_bytes": peak_rss_bytes(),
        "physical_io": None,
        "physical_io_measured": False,
        "profile_promoted": False,
        "promotion_authorized": False,
        "archive_guard": archive_after,
        "selectors": selectors_after,
    }
    return _strict_envelope(TRIAL_SCHEMA_ID, payload)


def _trial_order(repetition_index: int) -> tuple[str, str]:
    return (
        ("source", "candidate")
        if int(repetition_index) % 2 == 0
        else ("candidate", "source")
    )


def _run_fresh_trial(
    *,
    archive: Path,
    source_name: str,
    candidate_name: str,
    role: str,
    repetition_index: int,
    seed: int,
    window_rows: int,
    output: Path,
    driver_pid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "trial",
        "--archive",
        str(archive),
        "--source-run",
        source_name,
        "--candidate-run",
        candidate_name,
        "--role",
        role,
        "--repetition-index",
        str(repetition_index),
        "--seed",
        str(seed),
        "--window-rows",
        str(window_rows),
        "--driver-pid",
        str(driver_pid),
        "--output",
        str(output),
    ]
    environment = dict(os.environ)
    environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    process = subprocess.Popen(command, env=environment)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    result = json.loads(output.read_text(encoding="utf-8"))
    output.unlink()
    payload = _require_envelope(result, schema_id=TRIAL_SCHEMA_ID)
    if (
        process.pid != payload.get("pid")
        or payload.get("parent_pid") != driver_pid
        or payload.get("driver_pid") != driver_pid
    ):
        raise ValueError("Fresh tail subprocess identity receipt differs.")
    return result, {
        "repetition_index": int(repetition_index),
        "role": role,
        "spawned_pid": int(process.pid),
        "driver_pid": int(driver_pid),
        "return_code": int(return_code),
    }


def _require_trial_receipt(
    trial: Mapping[str, Any],
    *,
    archive: Path,
    source_name: str,
    candidate_name: str,
    role: str,
    repetition_index: int,
    seed: int,
    window_rows: int,
    driver_pid: int,
) -> Mapping[str, Any]:
    payload = _require_envelope(trial, schema_id=TRIAL_SCHEMA_ID)
    if set(payload) != _TRIAL_PAYLOAD_FIELDS:
        raise ValueError("Tail trial payload field set differs.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or payload["archive"] != str(archive)
        or payload["source_run_name"] != source_name
        or payload["candidate_run_name"] != candidate_name
        or payload["role"] != role
        or payload["repetition_index"] != repetition_index
        or payload["seed"] != seed
        or payload["window_rows"] != window_rows
        or payload["driver_pid"] != driver_pid
        or payload["parent_pid"] != driver_pid
        or type(payload["pid"]) is not int
        or payload["pid"] <= 0
        or payload["pid"] == driver_pid
        or payload["profile_promoted"] is not False
        or payload["promotion_authorized"] is not False
        or payload["physical_io"] is not None
        or payload["physical_io_measured"] is not False
    ):
        raise ValueError("Tail trial identity or hard nonpromotion state differs.")
    pair_payload = _require_pair_receipt(payload["pair_validation"])
    if (
        pair_payload["archive"] != str(archive)
        or pair_payload["source_run_path"] != _run_path(source_name)
        or pair_payload["candidate_run_path"] != _run_path(candidate_name)
    ):
        raise ValueError("Tail trial pair identity differs.")
    timing = payload["timing"]
    if not isinstance(timing, Mapping) or set(timing) != {
        "pair_validation_seconds",
        "workload_seconds",
    }:
        raise ValueError("Tail trial timing field set differs.")
    for name, value in timing.items():
        _require_nonnegative(value, label=f"trial timing {name}")
    _require_storage_stats(payload["storage"], label="trial")
    if type(payload["peak_rss_bytes"]) is not int or payload["peak_rss_bytes"] < 0:
        raise ValueError("Tail trial RSS differs.")
    _require_archive_guard(payload["archive_guard"])
    if payload["selectors"] != pair_payload["selectors"]:
        raise ValueError("Tail trial selectors differ from pair validation.")
    return payload


def _replay_trial(
    archive: Path,
    trial_payload: Mapping[str, Any],
    *,
    live_pair: Mapping[str, Any],
) -> None:
    role = str(trial_payload["role"])
    name = (
        str(trial_payload["source_run_name"])
        if role == "source"
        else str(trial_payload["candidate_run_name"])
    )
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    if trial_payload["pair_validation"] != live_pair:
        raise ValueError("Tail trial pair receipt differs from live pair validation.")
    observed = execute_workload(
        root[_run_path(name)],
        trial_payload["workload"],
        role=role,
        seed=int(trial_payload["seed"]),
        window_rows=int(trial_payload["window_rows"]),
    )
    if observed != trial_payload["workload_result"]:
        raise ValueError(
            "Persisted tail trial workload does not replay from live arrays."
        )
    public = trial_payload["public_consumer_result"]
    if role == "source":
        if not isinstance(public, Mapping):
            raise ValueError("Source trial lacks public-consumer evidence.")
        replayed = _reconstruct_public_consumer(
            root,
            source_name=str(trial_payload["source_run_name"]),
            window_rows=int(trial_payload["window_rows"]),
        )
        if replayed != dict(public):
            raise ValueError("Source public-consumer workload does not replay.")
    elif public is not None:
        raise ValueError("Candidate trial improperly claims a public consumer.")
    if trial_payload["archive_guard"] != _archive_guard(archive):
        raise ValueError("Tail trial archive guard differs from live storage.")


def validate_matrix(matrix: Mapping[str, Any], *, replay_live: bool = True) -> None:
    payload = _require_envelope(matrix, schema_id=MATRIX_SCHEMA_ID)
    if set(payload) != _MATRIX_PAYLOAD_FIELDS:
        raise ValueError("Tail matrix payload field set differs.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or payload["profile_promoted"] is not False
        or payload["promotion_authorized"] is not False
        or payload["physical_io"] is not None
        or payload["physical_io_measured"] is not False
        or payload["promotion_decision"] != "not_evaluated_diagnostic_only"
    ):
        raise ValueError("Tail matrix identity or hard nonpromotion state differs.")
    archive = _safe_archive(str(payload["archive"]))
    source_name = _safe_name(payload["source_run_name"], label="source run")
    candidate_name = _safe_name(payload["candidate_run_name"], label="candidate run")
    repetitions = payload["repetitions"]
    seed = payload["seed"]
    window_rows = payload["window_rows"]
    driver_pid = payload["driver_pid"]
    if any(
        type(value) is not int or value <= 0
        for value in (repetitions, window_rows, driver_pid)
    ):
        raise ValueError("Tail matrix cardinality/window/driver identity differs.")
    if type(seed) is not int:
        raise ValueError("Tail matrix seed differs.")
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != repetitions * 2:
        raise ValueError("Tail matrix trial cardinality differs.")
    receipts = payload["process_receipts"]
    if not isinstance(receipts, list) or len(receipts) != len(trials):
        raise ValueError("Tail matrix process-receipt cardinality differs.")
    live_pair = validate_pair(
        archive,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
    )
    pids: list[int] = []
    timing_by_role: dict[str, list[float]] = {"source": [], "candidate": []}
    for index, trial in enumerate(trials):
        repetition = index // 2
        role_index = index % 2
        role = _trial_order(repetition)[role_index]
        trial_payload = _require_trial_receipt(
            trial,
            archive=archive,
            source_name=source_name,
            candidate_name=candidate_name,
            role=role,
            repetition_index=repetition,
            seed=seed + repetition,
            window_rows=window_rows,
            driver_pid=driver_pid,
        )
        process_receipt = receipts[index]
        if (
            not isinstance(process_receipt, Mapping)
            or set(process_receipt)
            != {"repetition_index", "role", "spawned_pid", "driver_pid", "return_code"}
            or process_receipt["repetition_index"] != repetition
            or process_receipt["role"] != role
            or process_receipt["spawned_pid"] != trial_payload["pid"]
            or process_receipt["driver_pid"] != driver_pid
            or process_receipt["return_code"] != 0
        ):
            raise ValueError("Tail matrix fresh-process receipt differs.")
        pids.append(trial_payload["pid"])
        timing_by_role[role].append(trial_payload["timing"]["workload_seconds"])
        if replay_live:
            _replay_trial(archive, trial_payload, live_pair=live_pair)
    if len(set(pids)) != len(pids):
        raise ValueError("Tail matrix trials were not distinct fresh processes.")
    expected_aggregate = {
        role: {
            "median_workload_seconds": statistics.median(values),
            "trial_count": len(values),
        }
        for role, values in timing_by_role.items()
    }
    if payload["aggregate"] != expected_aggregate:
        raise ValueError("Tail matrix aggregate differs from exact trials.")
    if payload["archive_guard"] != _archive_guard(archive):
        raise ValueError("Tail matrix archive guard differs from live storage.")
    _require_archive_guard(payload["archive_guard"])
    environment = payload["environment"]
    if not isinstance(environment, Mapping) or set(environment) != {
        "python",
        "platform",
        "thread_environment",
        "palette_git",
    }:
        raise ValueError("Tail matrix environment field set differs.")


def run_matrix(
    archive: str | Path,
    *,
    source_run_name: str,
    candidate_run_name: str,
    repetitions: int,
    seed: int,
    window_rows: int,
    output: str | Path,
) -> dict[str, Any]:
    archive_path = _safe_archive(archive)
    output_path = _safe_output(archive_path, output)
    if int(repetitions) <= 0:
        raise ValueError("repetitions must be positive.")
    source_name = _safe_name(source_run_name, label="source run")
    candidate_name = _safe_name(candidate_run_name, label="candidate run")
    trials: list[dict[str, Any]] = []
    process_receipts: list[dict[str, Any]] = []
    driver_pid = int(os.getpid())
    for repetition in range(int(repetitions)):
        for role in _trial_order(repetition):
            temporary = output_path.with_name(
                f".{output_path.name}.{repetition}.{role}.trial.json"
            )
            if temporary.exists():
                raise FileExistsError(f"Temporary trial output exists: {temporary}")
            trial, process_receipt = _run_fresh_trial(
                archive=archive_path,
                source_name=source_name,
                candidate_name=candidate_name,
                role=role,
                repetition_index=repetition,
                seed=int(seed) + repetition,
                window_rows=int(window_rows),
                output=temporary,
                driver_pid=driver_pid,
            )
            trials.append(trial)
            process_receipts.append(process_receipt)
    timing_by_role = {
        role: [
            float(
                _require_envelope(trial, schema_id=TRIAL_SCHEMA_ID)["timing"][
                    "workload_seconds"
                ]
            )
            for trial in trials
            if _require_envelope(trial, schema_id=TRIAL_SCHEMA_ID)["role"] == role
        ]
        for role in ("source", "candidate")
    }
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "created_at_utc": _now_utc(),
        "archive": str(archive_path),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "window_rows": int(window_rows),
        "driver_pid": driver_pid,
        "trials": trials,
        "process_receipts": process_receipts,
        "aggregate": {
            role: {
                "median_workload_seconds": statistics.median(values),
                "trial_count": len(values),
            }
            for role, values in timing_by_role.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": dict(STORAGE_BENCHMARK_THREAD_ENVIRONMENT),
            "palette_git": get_git_info(repo_path=Path(__file__).resolve().parents[3]),
        },
        "archive_guard": _archive_guard(archive_path),
        "physical_io": None,
        "physical_io_measured": False,
        "profile_promoted": False,
        "promotion_authorized": False,
        "promotion_decision": "not_evaluated_diagnostic_only",
    }
    matrix = _strict_envelope(MATRIX_SCHEMA_ID, payload)
    validate_matrix(matrix, replay_live=True)
    output_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return matrix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("trial", "matrix"):
        child = sub.add_parser(name)
        child.add_argument("--archive", required=True)
        child.add_argument("--source-run", required=True)
        child.add_argument("--candidate-run", required=True)
        child.add_argument("--seed", type=int, default=DEFAULT_SEED)
        child.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS)
        child.add_argument("--output", required=True)
    trial = sub.choices["trial"]
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--driver-pid", type=int, required=True)
    matrix = sub.choices["matrix"]
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archive = _safe_archive(args.archive)
    output = _safe_output(archive, args.output)
    if args.command == "trial":
        result = run_trial(
            archive,
            source_run_name=args.source_run,
            candidate_run_name=args.candidate_run,
            role=args.role,
            repetition_index=args.repetition_index,
            seed=args.seed,
            window_rows=args.window_rows,
            driver_pid=args.driver_pid,
        )
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return 0
    run_matrix(
        archive,
        source_run_name=args.source_run,
        candidate_run_name=args.candidate_run,
        repetitions=args.repetitions,
        seed=args.seed,
        window_rows=args.window_rows,
        output=output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
