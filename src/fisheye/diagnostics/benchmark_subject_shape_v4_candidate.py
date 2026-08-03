"""Benchmark one exact subject-shape v4 source/candidate pair read-only.

The controller launches fresh processes in rotated order.  It never opens the
archive for mutation and writes strict benchmark evidence only to a new,
disjoint benchmark directory.  Runnable evidence is not profile promotion.
"""

from __future__ import annotations

import argparse
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
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import zarr

from fisheye.shared import subject_shape_coordinate_publication as subject_shape_contract
from fisheye.analysis.subject_shape_storage import (
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
    build_subject_shape_storage_receipt,
    subject_shape_fill_value,
    validate_subject_shape_candidate_storage,
)
from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
    ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    SERIALIZATION_POLICY,
)
from fisheye.analysis_workflows.materializers.subject_shape import (
    MATERIALIZATION_SCHEMA_ID,
    PUBLISH_SCHEMA_ID,
)
from fisheye.shared.coordinate_frame_record import ARRAY_PAYLOAD_CANONICALIZATION
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
    SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR,
    SUBJECT_SHAPE_MANIFEST_ATTR,
    SUBJECT_SHAPE_MANIFEST_SCHEMA_ID,
    SUBJECT_SHAPE_SCHEMA_VERSION,
    SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
    SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS,
    SUBJECT_SHAPE_STORAGE_PLAN_ATTR,
    SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR,
    SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
    load_completed_ineligible_subject_shape_coordinate_publication,
    load_persisted_subject_shape_coordinate_publication,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.array_factory import array_metadata_declaration_from_plan
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)


PARENT_PATH = "analysis/subject_shape_runs"
FAMILY_ID = "subject_shape_v4"
BENCHMARK_ID = "subject_shape_v4_source_candidate_read_matrix_v1"
TRIAL_SCHEMA_ID = "palette.subject_shape.v4_read_trial"
MATRIX_SCHEMA_ID = "palette.subject_shape.v4_read_matrix"
SCHEMA_VERSION = 1
DEFAULT_SEED = 17
DEFAULT_REPETITIONS = 5
FULL_SCAN_TARGET_BLOCK_BYTES = 8 * 1024 * 1024
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_SHA256_LENGTH = 64
_TRIAL_PHYSICAL_AVAILABILITY = (
    "unavailable_without_os_or_filesystem_tracing; logical decoded bytes "
    "and file counts are not physical transfer telemetry"
)
_MATRIX_PHYSICAL_AVAILABILITY = "unavailable_without_os_or_filesystem_tracing"
_ROLLBACK_POLICY = (
    "retain_owner_bound_failed_public_tombstone_and_"
    "stage_specific_receipt_rollback_only"
)
_PUBLISH_RECEIPT_FIELDS = {
    "local_publication_run",
    "storage_profile_id",
    "promotion_policy",
    "materialization",
    "canonical_binding",
    "storage_plan",
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
_FINAL_MANIFEST_FIELDS = {
    "schema_id",
    "schema_version",
    "run_ref",
    "row_identity",
    "temporal_authority",
    "source_refined_subject_masks",
    "component_schema",
    "scientific_configuration",
    "consumed_unbound_stage",
    "tail_sample_axis",
    "scalar_surface_inventory",
    "scalar_surfaces",
    "derivation",
    "body_frame",
    "heading_semantics",
    "coordinate_descriptors",
    "arrays",
    "schema_inventory",
    "closed_array_inventory",
    "closed_group_inventory",
    "closed_attr_inventory",
}
_FINAL_ARRAY_FIELDS = {
    "array_ref",
    "relative_ref",
    "dtype",
    "shape",
    "content_sha256",
    "canonicalization",
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
        raise ValueError("Subject-shape evidence envelope has an unexpected field set.")
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Subject-shape evidence schema identity is unsupported.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Subject-shape evidence payload must be one object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Subject-shape evidence payload digest mismatch.")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Subject-shape evidence is not strict JSON: {exc}") from exc
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonnegative_number(value: object, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be one finite nonnegative number.")


def _require_timing(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "wall_seconds",
        "cpu_seconds",
    }:
        raise ValueError(f"{label} timing has an unexpected field set.")
    _require_nonnegative_number(value["wall_seconds"], label=f"{label} wall")
    _require_nonnegative_number(value["cpu_seconds"], label=f"{label} CPU")


def _measure(call: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    wall = time.perf_counter()
    cpu = time.process_time()
    result = call()
    return result, {
        "wall_seconds": float(time.perf_counter() - wall),
        "cpu_seconds": float(time.process_time() - cpu),
    }


def _safe_name(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string.")
    name = value.strip()
    if (
        not name
        or name != value
        or name in _ALIASES
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return name


def _require_parent(value: str) -> str:
    if value != PARENT_PATH:
        raise ValueError(f"parent must be exact {PARENT_PATH!r}.")
    return value


def _safe_archive(path: Path | str) -> Path:
    archive = Path(path).expanduser().resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive not found: {archive}.")
    return archive


def _require_nonsymlink_subtree(
    archive: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    components = relative_path.split("/")
    if not components or any(part in {"", ".", ".."} for part in components):
        raise ValueError(f"{label} path is not canonical.")
    node = archive
    for component in components:
        node = node / component
        if node.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {node}.")
        if not node.is_dir():
            raise FileNotFoundError(f"{label} directory is absent: {node}.")
    try:
        node.resolve().relative_to(archive)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the archive.") from exc
    return node


def _safe_output(path: Path | str, *, archive: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}.")
    if output == archive or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError("Benchmark output must be disjoint from the source archive.")
    if not any("benchmark" in component.lower() for component in output.parts):
        raise ValueError("Benchmark output must have a benchmark-only path component.")
    if output in {Path("/"), Path.home().resolve()}:
        raise ValueError("Benchmark output path is too broad.")
    return output


def _safe_trial_output(path: Path | str, *, root: Path) -> Path:
    output = Path(path).expanduser().resolve()
    benchmark_root = root.expanduser().resolve()
    if (
        not benchmark_root.is_dir()
        or output.exists()
        or not output.is_relative_to(benchmark_root)
        or output.suffix != ".json"
    ):
        raise ValueError("Trial output must be new JSON inside the benchmark root.")
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace benchmark evidence: {path}.")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError("Benchmark JSON must be one object.")
    return value


def _group(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _iter_arrays(group: Any, prefix: str = ""):
    for name in sorted(str(value) for value in group.array_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path, group[name]
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield from _iter_arrays(group[name], path)


def _array(group: Any, path: str) -> Any:
    return _group(group, path)


def _array_digest(array: Any, *, block_rows: int = 65_536) -> str:
    dtype = np.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
                "dtype": np.lib.format.dtype_to_descr(dtype),
                "shape": list(shape),
            }
        )
    )
    digest.update(b"\x00")
    if not shape:
        digest.update(np.asarray(array[...]).tobytes(order="C"))
        return digest.hexdigest()
    for start in range(0, shape[0], max(1, block_rows)):
        stop = min(shape[0], start + max(1, block_rows))
        digest.update(np.ascontiguousarray(array[start:stop]).tobytes(order="C"))
    return digest.hexdigest()


def _logical_inventory(group: Any, paths: Sequence[str]) -> dict[str, Any]:
    return {
        path: {
            "dtype": np.dtype(_array(group, path).dtype).str,
            "shape": [int(value) for value in _array(group, path).shape],
            "logical_digest": _array_digest(_array(group, path)),
        }
        for path in paths
    }


def _publication_validation(
    value: object,
    *,
    role: str,
    phase: str,
    row_count: int,
    run_name: str,
) -> None:
    base_fields = {
        "valid",
        "errors",
        "row_count",
        "require_sharded",
        "binding_status",
        "physical_storage_layout",
        "storage_profile_id",
        "storage_candidate",
    }
    expected_fields = base_fields | ({"canonical_validation"} if phase == "final" else set())
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"Subject-shape publication {phase} validation is malformed.")
    candidate = role == "candidate"
    expected_profile = (
        SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
        if candidate
        else SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE
    )
    expected_binding = (
        SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS
        if phase in {"local", "temporary"}
        else SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
    )
    if (
        value["valid"] is not True
        or value["errors"] != []
        or value["row_count"] != row_count
        or value["require_sharded"] is not (not candidate)
        or value["binding_status"] != expected_binding
        or value["storage_profile_id"] != expected_profile
        or value["storage_candidate"] is not candidate
    ):
        raise ValueError(f"Subject-shape publication {phase} validation did not pass.")
    if phase == "final":
        canonical = value["canonical_validation"]
        if (
            not isinstance(canonical, Mapping)
            or set(canonical) != {"valid", "run_name", "row_count", "manifest_sha256"}
            or canonical["valid"] is not True
            or canonical["run_name"] != run_name
            or canonical["row_count"] != row_count
            or not _is_sha256(canonical["manifest_sha256"])
        ):
            raise ValueError("Subject-shape final canonical validation is malformed.")


def _publication_receipt(
    group: Any,
    *,
    role: str,
    archive: Path,
    run_path: str,
    run_name: str,
    row_count: int,
    manifest_sha256: str,
    storage_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    receipt = group.attrs.get("cluster_output_staging")
    if not isinstance(receipt, Mapping):
        if role == "candidate":
            raise ValueError("Subject-shape candidate lacks its atomic publication receipt.")
        return {
            "availability": "not_recorded_on_source",
            "receipt_sha256": None,
            "copy_duration_seconds": None,
            "published_at_utc": None,
        }
    if set(receipt) != _PUBLISH_RECEIPT_FIELDS:
        raise ValueError("Subject-shape atomic publication receipt field set differs.")
    candidate = role == "candidate"
    expected_profile = (
        SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
        if candidate
        else SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE
    )
    expected_policy = (
        "read_only_compute_unbound_stage_byte_plan_final_path_bind_then_publish"
        if candidate
        else "read_only_compute_unbound_stage_shard_final_path_bind_then_publish"
    )
    expected_promotion = (
        "immutable_named_candidate_no_pointer_or_registry_activation"
        if candidate
        else "complete_ineligible_then_pointers_then_eligibility_final"
    )
    owner = receipt.get("publication_owner_uuid")
    if (
        receipt.get("schema_id") != PUBLISH_SCHEMA_ID
        or receipt.get("publisher_contract")
        != {
            "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
            "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
        }
        or receipt.get("policy") != expected_policy
        or receipt.get("serialization_policy") != SERIALIZATION_POLICY
        or receipt.get("rollback_policy") != _ROLLBACK_POLICY
        or receipt.get("storage_profile_id") != expected_profile
        or receipt.get("promotion_policy") != expected_promotion
        or receipt.get("source_zarr") != str(archive)
        or Path(str(receipt.get("target_run_path"))).resolve()
        != archive.joinpath(*run_path.split("/"))
        or receipt.get("publication_owner_attr")
        != SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR
        or type(owner) is not str
        or not owner
        or group.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR) != owner
        or receipt.get("failed_public_child_policy")
        != "retain_owner_bound_selector_ineligible_tombstone"
        or receipt.get("hidden_temporary_policy")
        != "same_parent_hidden_sibling_then_os_replace"
    ):
        raise ValueError("Subject-shape atomic publication receipt contract differs.")
    local_path = receipt.get("local_publication_run")
    if (
        type(local_path) is not str
        or not Path(local_path).is_absolute()
        or receipt.get("publication_source_run_path") != local_path
    ):
        raise ValueError("Subject-shape atomic publication source path differs.")
    for field in ("host", "published_at_utc"):
        if type(receipt.get(field)) is not str or not receipt[field].strip():
            raise ValueError(f"Subject-shape publication {field} is invalid.")
    if receipt.get("lsb_jobid") is not None and (
        type(receipt["lsb_jobid"]) is not str or not receipt["lsb_jobid"].strip()
    ):
        raise ValueError("Subject-shape publication LSF identity is invalid.")
    _require_nonnegative_number(
        receipt.get("copy_duration_seconds"),
        label="Subject-shape copy duration",
    )
    materialization = receipt.get("materialization")
    if (
        not isinstance(materialization, Mapping)
        or set(materialization)
        != {
            "schema_id",
            "status",
            "completed_at_utc",
            "source_access_policy",
            "node_local_compute",
            "node_local_compute_validation",
            "node_local_sharding",
            "publishing_validation",
            "native_thread_environment",
            "capacity",
        }
        or materialization.get("schema_id") != MATERIALIZATION_SCHEMA_ID
        or materialization.get("status") != "complete"
        or materialization.get("source_access_policy")
        != "authoritative_shared_read_only"
    ):
        raise ValueError("Subject-shape materialization receipt is not exact.")
    canonical_binding = receipt.get("canonical_binding")
    if (
        not isinstance(canonical_binding, Mapping)
        or canonical_binding.get("valid") is not True
        or canonical_binding.get("status") != SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
        or canonical_binding.get("run_name") != run_name
        or canonical_binding.get("row_count") != row_count
        or canonical_binding.get("publication_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Subject-shape canonical-binding receipt differs.")
    if (receipt.get("storage_plan") is not None) is not candidate:
        raise ValueError("Subject-shape publication storage-plan role differs.")
    if candidate and receipt.get("storage_plan") != storage_receipt:
        raise ValueError("Subject-shape publication storage plan differs from the run.")
    physical = receipt.get("physical_copy")
    if not isinstance(physical, Mapping) or set(physical) != {
        "backend",
        "verification",
        "file_count",
        "physical_bytes",
        "inventory_sha256",
        "content_sha256",
    }:
        raise ValueError("Subject-shape physical-copy receipt is malformed.")
    if (
        physical["backend"] not in {"python", "rsync"}
        or physical["verification"]
        not in {"sha256_all_physical_files", "rsync_checksum_dry_run"}
        or type(physical["file_count"]) is not int
        or physical["file_count"] < 1
        or type(physical["physical_bytes"]) is not int
        or physical["physical_bytes"] < 1
        or not _is_sha256(physical["inventory_sha256"])
        or not _is_sha256(physical["content_sha256"])
    ):
        raise ValueError("Subject-shape physical-copy receipt is invalid.")
    if not isinstance(receipt.get("parent_attrs_before"), Mapping) or not isinstance(
        receipt.get("parent_attrs_after"), Mapping
    ):
        raise ValueError("Subject-shape publication parent snapshots are malformed.")
    for field, phase in (
        ("local_validation", "local"),
        ("temporary_validation", "temporary"),
        ("pre_pointer_validation", "pre_pointer"),
        ("final_validation", "final"),
    ):
        _publication_validation(
            receipt[field],
            role=role,
            phase=phase,
            row_count=row_count,
            run_name=run_name,
        )
    receipt_sha256 = canonical_json_sha256(receipt)
    return {
        "availability": "recorded_exact_atomic_receipt",
        "receipt_sha256": receipt_sha256,
        "copy_duration_seconds": float(receipt["copy_duration_seconds"]),
        "published_at_utc": receipt["published_at_utc"],
    }


def _validate_role(root: Any, *, run_path: str, role: str) -> dict[str, Any]:
    group = _group(root, run_path)
    owner = group.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
    if type(owner) is not str or not owner:
        raise ValueError("Subject-shape run lacks its publication owner.")
    if role == "source":
        publication = load_persisted_subject_shape_coordinate_publication(
            root,
            run_path,
            expected_publication_owner=owner,
        )
    elif role == "candidate":
        publication = load_completed_ineligible_subject_shape_coordinate_publication(
            root,
            run_path,
            expected_publication_owner=owner,
        )
        errors = validate_subject_shape_candidate_storage(group, phase="bound")
        if errors:
            raise ValueError(f"Invalid subject-shape storage candidate: {errors!r}.")
    else:
        raise ValueError("Subject-shape role is unsupported.")
    manifest = group.attrs.get(SUBJECT_SHAPE_MANIFEST_ATTR)
    if (
        not isinstance(manifest, Mapping)
        or coordinate_record_sha256(manifest) != publication.manifest.record_sha256
    ):
        raise ValueError("Subject-shape final manifest binding differs.")
    resolved_receipt = build_subject_shape_storage_receipt(group, phase="bound")
    receipt_manifest = resolved_receipt.as_manifest()
    if role == "candidate":
        persisted = group.attrs.get(SUBJECT_SHAPE_STORAGE_PLAN_ATTR)
        if (
            not isinstance(persisted, Mapping)
            or analysis_storage_plan_receipt_from_manifest(persisted).as_manifest()
            != persisted
            or persisted != receipt_manifest
        ):
            raise ValueError("Subject-shape candidate storage receipt is not executable.")
    source_link: Mapping[str, Any] | None = None
    source_link_sha256: str | None = None
    if role == "candidate":
        source_link = group.attrs.get(SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR)
        source_link_sha256 = group.attrs.get(
            f"{SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR}_sha256"
        )
        if (
            not isinstance(source_link, Mapping)
            or not _is_sha256(source_link_sha256)
            or coordinate_record_sha256(source_link) != source_link_sha256
        ):
            raise ValueError("Subject-shape candidate producer-manifest link differs.")
    declarations = [entry.declaration.as_manifest() for entry in resolved_receipt.entries]
    paths = [entry.declaration.path for entry in resolved_receipt.entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Subject-shape declaration paths are not unique and sorted.")
    row_count = dict(resolved_receipt.dimensions).get("n_rows")
    if type(row_count) is not int or row_count < 0:
        raise ValueError("Subject-shape storage receipt row count is invalid.")
    source_refined_run = _safe_name(
        group.attrs.get("source_refined_subject_masks_run"),
        label="source_refined_subject_masks_run",
    )
    consumed_unbound_manifest = _group(
        group, "coordinate_records/consumed_unbound_stage"
    ).attrs.get(SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR)
    if not isinstance(consumed_unbound_manifest, Mapping):
        raise ValueError("Subject-shape run lacks its consumed unbound manifest.")
    return {
        "group": group,
        "manifest": dict(manifest),
        "manifest_sha256": publication.manifest.record_sha256,
        "source_manifest_link": (
            dict(source_link) if source_link is not None else None
        ),
        "source_manifest_link_sha256": source_link_sha256,
        "receipt": resolved_receipt,
        "receipt_manifest": receipt_manifest,
        "declarations": declarations,
        "declarations_sha256": canonical_json_sha256(declarations),
        "paths": paths,
        "row_count": row_count,
        "source_refined_run": source_refined_run,
        "consumed_unbound_manifest": dict(consumed_unbound_manifest),
    }


def _direct_source_refined_run(archive: Path, run_path: str) -> str:
    run_root = _require_nonsymlink_subtree(
        archive, run_path, label="selected subject-shape run"
    )
    document = _read_json(run_root / "zarr.json")
    attributes = document.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("Selected subject-shape run attributes are absent.")
    return _safe_name(
        attributes.get("source_refined_subject_masks_run"),
        label="source_refined_subject_masks_run",
    )


def _refined_dependency_path(archive: Path, source_refined_run: str) -> str:
    run_name = _safe_name(
        source_refined_run, label="source_refined_subject_masks_run"
    )
    candidates = (
        f"refined_subject_masks_runs/{run_name}",
        f"analysis/refined_subject_masks_runs/{run_name}",
    )
    present = [
        relative
        for relative in candidates
        if (
            archive.joinpath(*relative.split("/")).exists()
            or archive.joinpath(*relative.split("/")).is_symlink()
        )
    ]
    if len(present) != 1:
        raise ValueError("Refined-mask dependency must have one canonical location.")
    _require_nonsymlink_subtree(
        archive, present[0], label="refined-mask source run"
    )
    return present[0]


def _preflight(
    archive: Path,
    *,
    source_path: str,
    candidate_path: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    _require_nonsymlink_subtree(archive, source_path, label="source run")
    _require_nonsymlink_subtree(archive, candidate_path, label="candidate run")
    direct_source_refined_run = _direct_source_refined_run(archive, source_path)
    if (
        _direct_source_refined_run(archive, candidate_path)
        != direct_source_refined_run
    ):
        raise ValueError("Subject-shape source/candidate refined authorities differ.")
    _refined_dependency_path(archive, direct_source_refined_run)
    root = zarr.open_group(str(archive), mode="r", zarr_format=3, use_consolidated=True)
    source = _validate_role(root, run_path=source_path, role="source")
    candidate = _validate_role(root, run_path=candidate_path, role="candidate")
    if (
        source["declarations"] != candidate["declarations"]
        or source["receipt_manifest"] != candidate["receipt_manifest"]
        or source["paths"] != candidate["paths"]
        or source["row_count"] != candidate["row_count"]
        or source["source_refined_run"] != candidate["source_refined_run"]
    ):
        raise ValueError("Subject-shape source/candidate logical declarations differ.")
    source_logical = _logical_inventory(source["group"], source["paths"])
    candidate_logical = _logical_inventory(candidate["group"], candidate["paths"])
    if source_logical != candidate_logical:
        raise ValueError("Subject-shape source/candidate decoded arrays differ.")
    if source["source_refined_run"] != direct_source_refined_run:
        raise ValueError("Direct/consolidated refined authority differs.")
    for path in (source_path, candidate_path):
        validate_direct_consolidated_subtree(archive, subtree_path=path)
    scale = AnalysisBenchmarkScale(
        scale_id="observed_candidate",
        dimensions=tuple(sorted(candidate["receipt"].dimensions)),
        description="Observed exact subject-shape v4 source/candidate dimensions.",
    )
    suite = build_analysis_benchmark_suite(
        family_id=FAMILY_ID,
        scale=scale,
        storage_receipt=candidate["receipt"],
        seed=seed,
        repetitions=repetitions,
    )
    require_analysis_benchmark_suite_manifest(suite)
    source_publication = _publication_receipt(
        source["group"],
        role="source",
        archive=archive,
        run_path=source_path,
        run_name=source_path.rsplit("/", 1)[-1],
        row_count=source["row_count"],
        manifest_sha256=source["manifest_sha256"],
        storage_receipt=None,
    )
    candidate_publication = _publication_receipt(
        candidate["group"],
        role="candidate",
        archive=archive,
        run_path=candidate_path,
        run_name=candidate_path.rsplit("/", 1)[-1],
        row_count=candidate["row_count"],
        manifest_sha256=candidate["manifest_sha256"],
        storage_receipt=candidate["receipt_manifest"],
    )
    artifacts = {
        "source_final_manifest": source["manifest"],
        "candidate_final_manifest": candidate["manifest"],
        "candidate_retained_producer_manifest_link": candidate[
            "source_manifest_link"
        ],
        "candidate_storage_receipt": candidate["receipt_manifest"],
        "source_atomic_publication_receipt": source["group"].attrs.get(
            "cluster_output_staging"
        ),
        "candidate_atomic_publication_receipt": candidate["group"].attrs.get(
            "cluster_output_staging"
        ),
        "source_consumed_unbound_manifest": source["consumed_unbound_manifest"],
        "candidate_consumed_unbound_manifest": candidate[
            "consumed_unbound_manifest"
        ],
        "metadata_documents": {
            "source": _metadata_declarations(archive, source_path),
            "candidate": _metadata_declarations(archive, candidate_path),
        },
    }
    return {
        "suite": suite,
        "array_paths": candidate["paths"],
        "declarations_sha256": candidate["declarations_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "candidate_source_manifest_link_sha256": candidate[
            "source_manifest_link_sha256"
        ],
        "source_publication_receipt_sha256": source_publication["receipt_sha256"],
        "candidate_publication_receipt_sha256": candidate_publication[
            "receipt_sha256"
        ],
        "candidate_storage_receipt_payload_digest": candidate["receipt_manifest"][
            "payload_digest"
        ],
        "contract_artifacts": artifacts,
        "logical_arrays": source_logical,
        "source_refined_run": source["source_refined_run"],
    }


def _growth_axes(suite: Mapping[str, Any]) -> dict[str, int | None]:
    return {
        str(item["path"]): (
            None
            if item["observed_facts"]["growth_axis"] is None
            else int(item["observed_facts"]["growth_axis"])
        )
        for item in suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]
    }


def _read_cases(suite: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases: list[Mapping[str, Any]] = []
    for item in suite["payload"]["array_cases"]:
        workload = str(item["case"]["workload"]["workload_id"])
        if workload.endswith(".write_materialization.v1") or workload.endswith(
            ".full_scan_read.v1"
        ):
            continue
        if item["selection"].get("mode") not in {
            "whole_array",
            "bounded_row_windows",
            "random_complete_rows",
        }:
            raise ValueError(
                "Subject-shape receipt produced an unsupported primary workload."
            )
        cases.append(item)
    paths = [str(item["array_path"]) for item in cases]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Subject-shape suite lacks one sorted primary read per array.")
    return cases


def _slice(array: Any, axis: int, start: int, stop: int) -> tuple[slice, ...]:
    selection = [slice(None)] * int(array.ndim)
    selection[axis] = slice(start, stop)
    return tuple(selection)


def _primary_read(
    array: Any,
    selection: Mapping[str, Any],
    *,
    growth_axis: int | None,
) -> dict[str, Any]:
    mode = selection["mode"]
    axis = 0 if growth_axis is None else int(growth_axis)
    digest = hashlib.sha256()
    decoded_bytes = 0
    operations = 0

    def consume(value: Any) -> None:
        nonlocal decoded_bytes, operations
        block = np.ascontiguousarray(value)
        digest.update(block.tobytes(order="C"))
        decoded_bytes += int(block.nbytes)
        operations += 1

    if mode == "whole_array":
        consume(array[...])
    elif mode == "bounded_row_windows":
        for start, stop in selection["ranges"]:
            consume(array[_slice(array, axis, int(start), int(stop))])
    elif mode == "random_complete_rows":
        for row in selection["row_indices"]:
            consume(array[_slice(array, axis, int(row), int(row) + 1)])
    else:  # protected by _read_cases
        raise ValueError(f"Unsupported subject-shape access mode {mode!r}.")
    return {
        "mode": mode,
        "execution_axis": axis,
        "operation_count": operations,
        "decoded_bytes": decoded_bytes,
        "selection_digest": digest.hexdigest(),
    }


def _full_scan(array: Any, *, growth_axis: int | None) -> dict[str, Any]:
    dtype = np.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
                "dtype": np.lib.format.dtype_to_descr(dtype),
                "shape": list(shape),
            }
        )
    )
    digest.update(b"\x00")
    axis = 0 if growth_axis is None else int(growth_axis)
    if not shape:
        block = np.ascontiguousarray(array[...])
        digest.update(block.tobytes(order="C"))
        return {
            "dtype": dtype.str,
            "shape": [],
            "decoded_bytes": int(block.nbytes),
            "block_count": 1,
            "logical_digest": digest.hexdigest(),
        }
    trailing_values = max(
        1,
        int(np.prod([extent for index, extent in enumerate(shape) if index != axis])),
    )
    block_extent = max(
        1,
        FULL_SCAN_TARGET_BLOCK_BYTES // max(1, dtype.itemsize * trailing_values),
    )
    decoded_bytes = 0
    block_count = 0
    for start in range(0, shape[axis], block_extent):
        stop = min(shape[axis], start + block_extent)
        block = np.ascontiguousarray(array[_slice(array, axis, start, stop)])
        digest.update(block.tobytes(order="C"))
        decoded_bytes += int(block.nbytes)
        block_count += 1
    return {
        "dtype": dtype.str,
        "shape": list(shape),
        "decoded_bytes": decoded_bytes,
        "block_count": block_count,
        "logical_digest": digest.hexdigest(),
    }


_STORAGE_STAT_FIELDS = {
    "file_count",
    "metadata_file_count",
    "payload_file_count",
    "apparent_bytes",
    "allocated_bytes",
}


def _selected_storage_stats(
    archive: Path,
    run_path: str,
    paths: Sequence[str],
) -> dict[str, Any]:
    totals = {field: 0 for field in _STORAGE_STAT_FIELDS}
    run_root = archive.joinpath(*run_path.split("/"))
    by_array: dict[str, Any] = {}
    for path in paths:
        stats = storage_stats(run_root.joinpath(*path.split("/")))
        by_array[path] = stats
        for field in totals:
            totals[field] += int(stats[field])
    return {
        "scope": "exact_declared_array_directories; group_metadata_excluded",
        "totals": totals,
        "payload_object_count": totals["payload_file_count"],
        "by_array": by_array,
        "whole_run_tree": storage_stats(run_root),
    }


def _environment(archive: Path, cache_state: str) -> dict[str, Any]:
    git = get_git_info()
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "palette_commit": git.get("commit_hash"),
        "palette_dirty": bool(git.get("is_dirty")),
        "cache_state": cache_state,
        "archive_device": int(archive.stat().st_dev),
        "thread_environment": {
            name: os.environ.get(name) for name in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def _metadata_guard(
    archive: Path,
    *,
    run_paths: Sequence[str],
    source_refined_run: str,
) -> dict[str, Any]:
    paths = [
        archive / "zarr.json",
        archive.joinpath(*PARENT_PATH.split("/"), "zarr.json"),
    ]
    for run_path in run_paths:
        run_root = _require_nonsymlink_subtree(
            archive, run_path, label="metadata-guard selected run"
        )
        paths.extend(sorted(run_root.rglob("zarr.json")))
    dependency = _refined_dependency_path(archive, source_refined_run)
    dependency_root = _require_nonsymlink_subtree(
        archive, dependency, label="metadata-guard refined dependency"
    )
    if dependency.startswith("analysis/"):
        paths.append(archive / "analysis" / "refined_subject_masks_runs" / "zarr.json")
    else:
        paths.append(archive / "refined_subject_masks_runs" / "zarr.json")
    paths.extend(sorted(dependency_root.rglob("zarr.json")))
    records = []
    for path in sorted(set(paths), key=str):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required nonsymlink metadata file is absent: {path}.")
        data = path.read_bytes()
        stat = path.stat()
        records.append(
            {
                "path": str(path.relative_to(archive)),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "guard_scope": (
            "root_subject_shape_parent_complete_selected_runs_and_present_"
            "refined_source_zarr_json"
        ),
        "metadata_file_count": len(records),
        "metadata_tree_sha256": canonical_json_sha256(records),
        "files": records,
    }


def _metadata_declarations(archive: Path, run_path: str) -> dict[str, Any]:
    _require_nonsymlink_subtree(archive, run_path, label="metadata subtree")
    validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    root_document = _read_json(archive / "zarr.json")
    envelope = root_document.get("consolidated_metadata")
    inline = envelope.get("metadata") if isinstance(envelope, Mapping) else None
    if not isinstance(inline, Mapping):
        raise ValueError("Archive lacks its inline consolidated metadata document.")
    run_root = archive.joinpath(*run_path.split("/"))
    declarations: dict[str, Any] = {}
    for path in sorted(run_root.rglob("zarr.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Subject-shape metadata declaration path is unsafe.")
        node_path = path.parent.relative_to(archive).as_posix()
        direct = metadata_without_empty_group_consolidation(
            _read_json(path), path=node_path
        )
        consolidated = inline.get(node_path)
        if not isinstance(consolidated, Mapping):
            raise ValueError("Consolidated metadata omits a direct declaration.")
        normalized_inline = metadata_without_empty_group_consolidation(
            consolidated, path=node_path
        )
        if direct != normalized_inline:
            raise ValueError("Direct/consolidated normalized metadata differs.")
        declarations[node_path] = direct
    if run_path not in declarations:
        raise ValueError("Subject-shape metadata document omits its run group.")
    return declarations


class _MetadataGroupProxy:
    def __init__(
        self,
        declarations: Mapping[str, Mapping[str, Any]],
        path: str,
    ) -> None:
        self._declarations = declarations
        self.path = path
        declaration = declarations.get(path)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Metadata proxy path is absent: {path}.")
        attributes = declaration.get("attributes")
        self.attrs = dict(attributes) if isinstance(attributes, Mapping) else {}

    def _absolute(self, relative: str) -> str:
        return f"{self.path}/{relative.strip('/')}"

    def get(self, relative: str) -> _MetadataGroupProxy | None:
        absolute = self._absolute(relative)
        if absolute in self._declarations or any(
            path.startswith(f"{absolute}/") for path in self._declarations
        ):
            return _MetadataGroupProxy(self._declarations, absolute)
        return None

    def __getitem__(self, relative: str) -> _MetadataGroupProxy:
        result = self.get(relative)
        if result is None:
            raise KeyError(relative)
        return result

    def __contains__(self, relative: object) -> bool:
        return isinstance(relative, str) and self.get(relative) is not None


def _installed_transformable_paths(
    declarations: Mapping[str, Mapping[str, Any]],
    *,
    run_path: str,
) -> set[str]:
    run = _MetadataGroupProxy(declarations, run_path)
    component_names = tuple(str(value) for value in run.attrs.get("component_names", ()))
    geometry = subject_shape_contract._geometry_specs(run, component_names)
    scalar = subject_shape_contract._scalar_surface_specs(run)
    return set(geometry) | set(scalar)


def _require_metadata_documents(
    value: object,
    *,
    source_run: str,
    candidate_run: str,
    logical: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    producer_link: Mapping[str, Any],
    storage_receipt: Mapping[str, Any],
    source_atomic_receipt: Mapping[str, Any] | None,
    candidate_atomic_receipt: Mapping[str, Any],
    source_consumed: Mapping[str, Any],
    candidate_consumed: Mapping[str, Any],
) -> set[str]:
    if not isinstance(value, Mapping) or set(value) != {"source", "candidate"}:
        raise ValueError("Subject-shape metadata documents are not role-exact.")
    parsed_receipt = analysis_storage_plan_receipt_from_manifest(storage_receipt)
    entry_by_path = {entry.declaration.path: entry for entry in parsed_receipt.entries}
    transform_sets: dict[str, set[str]] = {}
    for role, run_name, manifest, atomic, consumed in (
        ("source", source_run, source_manifest, source_atomic_receipt, source_consumed),
        (
            "candidate",
            candidate_run,
            candidate_manifest,
            candidate_atomic_receipt,
            candidate_consumed,
        ),
    ):
        declarations = value[role]
        run_path = f"{PARENT_PATH}/{run_name}"
        if not isinstance(declarations, Mapping) or run_path not in declarations:
            raise ValueError(f"Subject-shape {role} metadata document is absent.")
        group = declarations[run_path]
        if not isinstance(group, Mapping) or group.get("node_type") != "group":
            raise ValueError(f"Subject-shape {role} run metadata is not a group.")
        attrs = group.get("attributes")
        if (
            not isinstance(attrs, Mapping)
            or attrs.get(SUBJECT_SHAPE_MANIFEST_ATTR) != manifest
            or attrs.get(f"{SUBJECT_SHAPE_MANIFEST_ATTR}_sha256")
            != coordinate_record_sha256(manifest)
            or attrs.get("cluster_output_staging") != atomic
        ):
            raise ValueError(f"Subject-shape {role} group metadata artifacts differ.")
        if role == "candidate" and (
            attrs.get(SUBJECT_SHAPE_STORAGE_PLAN_ATTR) != storage_receipt
            or attrs.get(SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR) != producer_link
        ):
            raise ValueError("Subject-shape candidate metadata contract differs.")
        consumed_path = f"{run_path}/coordinate_records/consumed_unbound_stage"
        consumed_node = declarations.get(consumed_path)
        if (
            not isinstance(consumed_node, Mapping)
            or not isinstance(consumed_node.get("attributes"), Mapping)
            or consumed_node["attributes"].get(SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR)
            != consumed
        ):
            raise ValueError(f"Subject-shape {role} consumed metadata differs.")
        array_paths = {
            path[len(run_path) + 1 :]
            for path, declaration in declarations.items()
            if path.startswith(f"{run_path}/")
            and isinstance(declaration, Mapping)
            and declaration.get("node_type") == "array"
        }
        if array_paths != set(logical):
            raise ValueError(f"Subject-shape {role} metadata array inventory differs.")
        for path in sorted(logical):
            declaration = declarations[f"{run_path}/{path}"]
            expected = logical[path]
            if (
                declaration.get("node_type") != "array"
                or declaration.get("shape") != expected["shape"]
                or np.dtype(declaration.get("data_type")).str != expected["dtype"]
            ):
                raise ValueError(f"Subject-shape {role} array metadata {path!r} differs.")
            if role == "candidate":
                entry = entry_by_path[path]
                planned = array_metadata_declaration_from_plan(
                    contract=entry.declaration.contract,
                    plan=entry.plan,
                    fill_value=subject_shape_fill_value(
                        path, entry.plan.logical_dtype
                    ),
                )
                for field in (
                    "shape",
                    "data_type",
                    "chunk_grid",
                    "chunk_key_encoding",
                    "codecs",
                    "storage_transformers",
                ):
                    if declaration.get(field) != planned.get(field):
                        raise ValueError(
                            f"Subject-shape candidate physical metadata {path!r} differs."
                        )
                planned_fill = planned.get("fill_value")
                observed_fill = declaration.get("fill_value")
                fill_matches = (
                    observed_fill in {"NaN", "nan"}
                    if isinstance(planned_fill, float) and math.isnan(planned_fill)
                    else observed_fill == planned_fill
                )
                if not fill_matches:
                    raise ValueError(
                        f"Subject-shape candidate fill metadata {path!r} differs."
                    )
                planned_attrs = planned["attributes"]
                observed_attrs = declaration.get("attributes")
                if not isinstance(observed_attrs, Mapping) or any(
                    observed_attrs.get(name) != expected_value
                    for name, expected_value in planned_attrs.items()
                ):
                    raise ValueError(
                        f"Subject-shape candidate storage attrs {path!r} differ."
                    )
        transform_sets[role] = _installed_transformable_paths(
            declarations, run_path=run_path
        )
    if transform_sets["source"] != transform_sets["candidate"]:
        raise ValueError("Subject-shape installed transform vocabularies differ.")
    if (
        set(source_manifest["coordinate_descriptors"])
        | set(source_manifest["scalar_surfaces"])
        != transform_sets["source"]
        or set(candidate_manifest["coordinate_descriptors"])
        | set(candidate_manifest["scalar_surfaces"])
        != transform_sets["candidate"]
    ):
        raise ValueError("Subject-shape manifest transform vocabulary is not executable.")
    return transform_sets["candidate"]


def _require_metadata_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "guard_scope",
        "metadata_file_count",
        "metadata_tree_sha256",
        "files",
    }:
        raise ValueError("Subject-shape metadata guard has an unexpected field set.")
    if value["guard_scope"] != (
        "root_subject_shape_parent_complete_selected_runs_and_present_"
        "refined_source_zarr_json"
    ):
        raise ValueError("Subject-shape metadata guard scope is unsupported.")
    files = value["files"]
    if not isinstance(files, list):
        raise ValueError("Subject-shape metadata guard inventory must be one list.")
    paths: list[str] = []
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
        }:
            raise ValueError("Subject-shape metadata guard file record is malformed.")
        path = record["path"]
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or not path.endswith("zarr.json")
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or type(record["mtime_ns"]) is not int
            or record["mtime_ns"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise ValueError("Subject-shape metadata guard file facts are invalid.")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise ValueError("Subject-shape metadata guard paths are not unique and sorted.")
    if (
        type(value["metadata_file_count"]) is not int
        or value["metadata_file_count"] != len(files)
        or not _is_sha256(value["metadata_tree_sha256"])
        or value["metadata_tree_sha256"] != canonical_json_sha256(files)
    ):
        raise ValueError("Subject-shape metadata guard inventory digest mismatch.")


def _require_storage(value: object, *, paths: Sequence[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "scope",
        "totals",
        "payload_object_count",
        "by_array",
        "whole_run_tree",
    }:
        raise ValueError("Subject-shape storage evidence is malformed.")
    if (
        value["scope"] != "exact_declared_array_directories; group_metadata_excluded"
        or not isinstance(value["by_array"], Mapping)
        or set(value["by_array"]) != set(paths)
    ):
        raise ValueError("Subject-shape storage evidence scope is invalid.")
    records = [("totals", value["totals"]), ("whole_run_tree", value["whole_run_tree"])]
    records.extend((f"by_array/{path}", item) for path, item in value["by_array"].items())
    for label, record in records:
        if (
            not isinstance(record, Mapping)
            or set(record) != _STORAGE_STAT_FIELDS
            or any(type(record[field]) is not int or record[field] < 0 for field in record)
        ):
            raise ValueError(f"Subject-shape storage {label} is invalid.")
    reconstructed = {
        field: sum(int(record[field]) for record in value["by_array"].values())
        for field in _STORAGE_STAT_FIELDS
    }
    if (
        dict(value["totals"]) != reconstructed
        or value["payload_object_count"] != reconstructed["payload_file_count"]
    ):
        raise ValueError("Subject-shape selected storage totals are not reconstructed.")


def _require_publication_summary(value: object, *, role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "availability",
        "receipt_sha256",
        "copy_duration_seconds",
        "published_at_utc",
    }:
        raise ValueError("Subject-shape publication timing is malformed.")
    if value["availability"] == "not_recorded_on_source":
        if role != "source" or value != {
            "availability": "not_recorded_on_source",
            "receipt_sha256": None,
            "copy_duration_seconds": None,
            "published_at_utc": None,
        }:
            raise ValueError("Subject-shape missing publication receipt role differs.")
        return
    if (
        value["availability"] != "recorded_exact_atomic_receipt"
        or not _is_sha256(value["receipt_sha256"])
        or type(value["published_at_utc"]) is not str
        or not value["published_at_utc"].strip()
    ):
        raise ValueError("Subject-shape publication receipt summary is invalid.")
    _require_nonnegative_number(
        value["copy_duration_seconds"], label="Subject-shape publication copy duration"
    )


def _require_final_manifest(
    value: object,
    *,
    run_name: str,
    manifest_sha256: str,
    logical: Mapping[str, Any],
    source_refined_run: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _FINAL_MANIFEST_FIELDS
        or value.get("schema_id") != SUBJECT_SHAPE_MANIFEST_SCHEMA_ID
        or value.get("schema_version") != SUBJECT_SHAPE_SCHEMA_VERSION
        or value.get("run_ref") != f"/{PARENT_PATH}/{run_name}"
        or value.get("closed_array_inventory") is not True
        or value.get("closed_group_inventory") is not True
        or value.get("closed_attr_inventory") is not True
        or coordinate_record_sha256(value) != manifest_sha256
    ):
        raise ValueError("Subject-shape embedded final manifest is not canonical.")
    source = value["source_refined_subject_masks"]
    expected_source_path = f"refined_subject_masks_runs/{source_refined_run}"
    if (
        not isinstance(source, Mapping)
        or source.get("run_path") != expected_source_path
    ):
        raise ValueError("Subject-shape final manifest source authority differs.")
    arrays = value["arrays"]
    inventory = value["schema_inventory"]
    if (
        not isinstance(arrays, Mapping)
        or set(arrays) != set(logical)
        or not isinstance(inventory, Mapping)
        or not isinstance(inventory.get("arrays"), Mapping)
        or set(inventory["arrays"]) != set(arrays)
    ):
        raise ValueError("Subject-shape final manifest array inventory differs.")
    for path, record in arrays.items():
        logical_record = logical[path]
        if (
            not isinstance(record, Mapping)
            or set(record) != _FINAL_ARRAY_FIELDS
            or record["array_ref"] != f"/{PARENT_PATH}/{run_name}/{path}"
            or record["relative_ref"] != path
            or record["dtype"] != logical_record["dtype"]
            or record["shape"] != logical_record["shape"]
            or record["content_sha256"] != logical_record["logical_digest"]
            or record["canonicalization"] != ARRAY_PAYLOAD_CANONICALIZATION
        ):
            raise ValueError(f"Subject-shape final manifest array {path!r} differs.")


def _require_contract_artifacts(
    value: object,
    *,
    archive_path: str,
    source_run: str,
    candidate_run: str,
    source_manifest_sha256: str,
    candidate_manifest_sha256: str,
    producer_link_sha256: str,
    source_publication_sha256: str | None,
    candidate_publication_sha256: str,
    storage_receipt_digest: str,
    source_refined_run: str,
    logical: Mapping[str, Any],
) -> None:
    fields = {
        "source_final_manifest",
        "candidate_final_manifest",
        "candidate_retained_producer_manifest_link",
        "candidate_storage_receipt",
        "source_atomic_publication_receipt",
        "candidate_atomic_publication_receipt",
        "source_consumed_unbound_manifest",
        "candidate_consumed_unbound_manifest",
        "metadata_documents",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Subject-shape embedded contract artifacts are not exact.")
    source_manifest = value["source_final_manifest"]
    candidate_manifest = value["candidate_final_manifest"]
    _require_final_manifest(
        source_manifest,
        run_name=source_run,
        manifest_sha256=source_manifest_sha256,
        logical=logical,
        source_refined_run=source_refined_run,
    )
    _require_final_manifest(
        candidate_manifest,
        run_name=candidate_run,
        manifest_sha256=candidate_manifest_sha256,
        logical=logical,
        source_refined_run=source_refined_run,
    )
    receipt_manifest = value["candidate_storage_receipt"]
    if (
        not isinstance(receipt_manifest, Mapping)
        or receipt_manifest.get("payload_digest") != storage_receipt_digest
        or analysis_storage_plan_receipt_from_manifest(receipt_manifest).as_manifest()
        != receipt_manifest
    ):
        raise ValueError("Subject-shape embedded storage receipt is not executable.")
    receipt_paths = [
        str(item["declaration"]["path"])
        for item in receipt_manifest["payload"]["arrays"]
    ]
    if receipt_paths != sorted(logical):
        raise ValueError("Subject-shape embedded storage receipt paths differ.")
    link = value["candidate_retained_producer_manifest_link"]
    expected_link_fields = {
        "schema_id",
        "schema_version",
        "source_manifest_attr",
        "source_manifest_record_ref",
        "source_manifest_sha256",
        "source_manifest",
        "array_comparison_canonicalization",
        "destination_decoded_equality_required",
    }
    if (
        not isinstance(link, Mapping)
        or set(link) != expected_link_fields
        or coordinate_record_sha256(link) != producer_link_sha256
        or link.get("array_comparison_canonicalization")
        != ARRAY_PAYLOAD_CANONICALIZATION
        or link.get("destination_decoded_equality_required") is not True
    ):
        raise ValueError("Subject-shape embedded producer-manifest link differs.")
    producer = link["source_manifest"]
    if (
        not isinstance(producer, Mapping)
        or coordinate_record_sha256(producer) != link.get("source_manifest_sha256")
        or link.get("source_manifest_record_ref")
        != f"/{PARENT_PATH}/{candidate_run}@{SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR}"
        or not isinstance(producer.get("arrays"), Mapping)
        or not set(producer["arrays"]).issubset(logical)
    ):
        raise ValueError("Subject-shape embedded producer seal is not cross-bound.")
    candidate_transformed = _require_metadata_documents(
        value["metadata_documents"],
        source_run=source_run,
        candidate_run=candidate_run,
        logical=logical,
        source_manifest=source_manifest,
        candidate_manifest=candidate_manifest,
        producer_link=link,
        storage_receipt=receipt_manifest,
        source_atomic_receipt=value["source_atomic_publication_receipt"],
        candidate_atomic_receipt=value["candidate_atomic_publication_receipt"],
        source_consumed=value["source_consumed_unbound_manifest"],
        candidate_consumed=value["candidate_consumed_unbound_manifest"],
    )
    for path, record in producer["arrays"].items():
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "relative_ref",
                "dtype",
                "shape",
                "content_sha256",
                "canonicalization",
            }
            or record.get("relative_ref") != path
            or record.get("dtype") != logical[path]["dtype"]
            or record.get("shape") != logical[path]["shape"]
            or not _is_sha256(record.get("content_sha256"))
            or record.get("canonicalization") != ARRAY_PAYLOAD_CANONICALIZATION
            or (
                path not in candidate_transformed
                and record.get("content_sha256") != logical[path]["logical_digest"]
            )
        ):
            raise ValueError(f"Subject-shape producer-sealed array {path!r} differs.")
    consumed_records = {
        "source": (
            value["source_consumed_unbound_manifest"],
            source_run,
            source_manifest["consumed_unbound_stage"].get("record_sha256"),
        ),
        "candidate": (
            value["candidate_consumed_unbound_manifest"],
            candidate_run,
            candidate_manifest["consumed_unbound_stage"].get("record_sha256"),
        ),
    }
    consumed_array_facts: dict[str, dict[str, Any]] = {}
    expected_unbound_fields = {
        "schema_id",
        "schema_version",
        "run_name",
        "binding_status",
        "source_refined_subject_masks_run",
        "method",
        "method_version",
        "component_names",
        "scientific_configuration",
        "schema_inventory",
        "arrays",
        "closed_array_inventory",
        "closed_group_inventory",
        "closed_attr_inventory",
        "coordinate_descriptors_present",
    }
    for role, (record, run_name, digest) in consumed_records.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_unbound_fields
            or record.get("run_name") != run_name
            or record.get("source_refined_subject_masks_run") != source_refined_run
            or record.get("closed_array_inventory") is not True
            or record.get("closed_group_inventory") is not True
            or record.get("closed_attr_inventory") is not True
            or not _is_sha256(digest)
            or coordinate_record_sha256(record) != digest
            or not isinstance(record.get("arrays"), Mapping)
            or not isinstance(record.get("schema_inventory"), Mapping)
            or not isinstance(record["schema_inventory"].get("arrays"), Mapping)
            or set(record["arrays"]) != set(record["schema_inventory"]["arrays"])
            or not set(record["arrays"]).issubset(logical)
        ):
            raise ValueError(
                f"Subject-shape embedded {role} consumed-stage manifest differs."
            )
        facts: dict[str, Any] = {}
        for path, item in record["arrays"].items():
            expected = logical[path]
            if (
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "relative_ref",
                    "dtype",
                    "shape",
                    "content_sha256",
                    "canonicalization",
                }
                or item.get("relative_ref") != path
                or item.get("dtype") != expected["dtype"]
                or item.get("shape") != expected["shape"]
                or not _is_sha256(item.get("content_sha256"))
                or item.get("canonicalization") != ARRAY_PAYLOAD_CANONICALIZATION
                or (
                    path not in candidate_transformed
                    and item.get("content_sha256") != expected["logical_digest"]
                )
            ):
                raise ValueError(
                    f"Subject-shape {role} consumed-stage array {path!r} differs."
                )
            facts[path] = {
                "dtype": item["dtype"],
                "shape": item["shape"],
                "content_sha256": item["content_sha256"],
            }
        consumed_array_facts[role] = facts
    producer_facts = {
        path: {
            "dtype": item["dtype"],
            "shape": item["shape"],
            "content_sha256": item["content_sha256"],
        }
        for path, item in producer["arrays"].items()
    }
    if (
        consumed_array_facts["source"] != consumed_array_facts["candidate"]
        or producer_facts != consumed_array_facts["candidate"]
    ):
        raise ValueError("Subject-shape producer and consumed-stage payloads differ.")
    row_count = dict(
        analysis_storage_plan_receipt_from_manifest(receipt_manifest).dimensions
    )["n_rows"]
    for role, receipt, digest, manifest_digest, run_name in (
        (
            "source",
            value["source_atomic_publication_receipt"],
            source_publication_sha256,
            source_manifest_sha256,
            source_run,
        ),
        (
            "candidate",
            value["candidate_atomic_publication_receipt"],
            candidate_publication_sha256,
            candidate_manifest_sha256,
            candidate_run,
        ),
    ):
        if receipt is None:
            if role != "source" or digest is not None:
                raise ValueError("Subject-shape embedded atomic receipt is absent.")
            continue
        if not isinstance(receipt, Mapping):
            raise ValueError("Subject-shape embedded atomic receipt is malformed.")
        fake = SimpleNamespace(
            attrs={
                "cluster_output_staging": receipt,
                SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: receipt.get(
                    "publication_owner_uuid"
                ),
            }
        )
        summary = _publication_receipt(
            fake,
            role=role,
            archive=Path(archive_path),
            run_path=f"{PARENT_PATH}/{run_name}",
            run_name=run_name,
            row_count=int(row_count),
            manifest_sha256=manifest_digest,
            storage_receipt=receipt_manifest if role == "candidate" else None,
        )
        if summary["receipt_sha256"] != digest:
            raise ValueError("Subject-shape embedded atomic receipt digest differs.")


def require_trial_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    expected = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "parent_path",
        "source_run_name",
        "candidate_run_name",
        "role",
        "run_name",
        "run_path",
        "repetition_index",
        "order_position",
        "process_id",
        "driver_process_id",
        "seed",
        "cache_state",
        "suite_payload_digest",
        "candidate_storage_receipt_payload_digest",
        "array_paths",
        "declarations_sha256",
        "source_manifest_sha256",
        "candidate_manifest_sha256",
        "candidate_source_manifest_link_sha256",
        "source_publication_receipt_sha256",
        "candidate_publication_receipt_sha256",
        "source_refined_run",
        "contract_artifacts",
        "started_at_utc",
        "finished_at_utc",
        "environment",
        "validation",
        "metadata",
        "primary_access",
        "full_scan",
        "logical_arrays",
        "storage",
        "publication_timing",
        "runtime",
        "physical_io",
    }
    if set(payload) != expected:
        raise ValueError("Subject-shape trial payload has an unexpected field set.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or payload["parent_path"] != PARENT_PATH
    ):
        raise ValueError("Subject-shape trial identity or parent differs.")
    source_name = _safe_name(payload["source_run_name"], label="source_run_name")
    candidate_name = _safe_name(payload["candidate_run_name"], label="candidate_run_name")
    if source_name == candidate_name:
        raise ValueError("Subject-shape source/candidate names must differ.")
    role = payload["role"]
    if role not in {"source", "candidate"}:
        raise ValueError("Subject-shape trial role is unsupported.")
    expected_name = source_name if role == "source" else candidate_name
    if (
        payload["run_name"] != expected_name
        or payload["run_path"] != f"{PARENT_PATH}/{expected_name}"
    ):
        raise ValueError("Subject-shape trial role/run/path binding differs.")
    if type(payload["repetition_index"]) is not int or payload["repetition_index"] < 0:
        raise ValueError("Subject-shape trial repetition is invalid.")
    if payload["order_position"] not in {0, 1}:
        raise ValueError("Subject-shape trial order position is invalid.")
    if type(payload["seed"]) is not int or payload["seed"] < 0:
        raise ValueError("Subject-shape trial seed is invalid.")
    if (
        type(payload["process_id"]) is not int
        or payload["process_id"] < 1
        or type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] < 1
        or payload["process_id"] == payload["driver_process_id"]
    ):
        raise ValueError("Subject-shape trial process identities are invalid.")
    if (
        _trial_order(
            seed=payload["seed"], repetition_index=payload["repetition_index"]
        )[payload["order_position"]]
        != role
    ):
        raise ValueError("Subject-shape trial role/position binding differs.")
    for field in (
        "suite_payload_digest",
        "candidate_storage_receipt_payload_digest",
        "declarations_sha256",
        "source_manifest_sha256",
        "candidate_manifest_sha256",
        "candidate_source_manifest_link_sha256",
        "candidate_publication_receipt_sha256",
    ):
        if not _is_sha256(payload[field]):
            raise ValueError(f"Subject-shape trial {field} is invalid.")
    source_publication_digest = payload["source_publication_receipt_sha256"]
    if source_publication_digest is not None and not _is_sha256(source_publication_digest):
        raise ValueError("Subject-shape source publication receipt digest is invalid.")
    paths = payload["array_paths"]
    if (
        not isinstance(paths, list)
        or not paths
        or paths != sorted(set(paths))
        or any(type(path) is not str or not path for path in paths)
    ):
        raise ValueError("Subject-shape trial array inventory is invalid.")
    if type(payload["source_refined_run"]) is not str or not payload[
        "source_refined_run"
    ].strip():
        raise ValueError("Subject-shape trial refined-source identity is invalid.")
    validation = payload["validation"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "valid",
        "role",
        "array_count",
        "declarations_sha256",
        "run_manifest_sha256",
        "candidate_source_manifest_link_sha256",
        "publication_receipt_sha256",
        "timing",
    }:
        raise ValueError("Subject-shape trial validation receipt is malformed.")
    expected_manifest = (
        payload["source_manifest_sha256"]
        if role == "source"
        else payload["candidate_manifest_sha256"]
    )
    expected_link = (
        payload["candidate_source_manifest_link_sha256"]
        if role == "candidate"
        else None
    )
    expected_publication = (
        payload["source_publication_receipt_sha256"]
        if role == "source"
        else payload["candidate_publication_receipt_sha256"]
    )
    if (
        validation["valid"] is not True
        or validation["role"] != role
        or validation["array_count"] != len(paths)
        or validation["declarations_sha256"] != payload["declarations_sha256"]
        or validation["run_manifest_sha256"] != expected_manifest
        or validation["candidate_source_manifest_link_sha256"] != expected_link
        or validation["publication_receipt_sha256"] != expected_publication
    ):
        raise ValueError("Subject-shape trial validation identity binding differs.")
    _require_timing(validation["timing"], label="validation")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "equivalent",
        "array_count",
        "group_count",
        "node_count",
        "subtree_declarations_digest",
        "declarations",
        "direct_open",
        "consolidated_open",
        "comparison",
    }:
        raise ValueError("Subject-shape trial metadata receipt is malformed.")
    if (
        metadata["equivalent"] is not True
        or metadata["array_count"] != len(paths)
        or type(metadata["group_count"]) is not int
        or metadata["group_count"] < 1
        or metadata["node_count"]
        != metadata["array_count"] + metadata["group_count"]
        or not _is_sha256(metadata["subtree_declarations_digest"])
        or metadata["subtree_declarations_digest"]
        != canonical_json_sha256(metadata["declarations"])
        or metadata["declarations"]
        != payload["contract_artifacts"]["metadata_documents"][role]
    ):
        raise ValueError("Subject-shape metadata equivalence receipt is invalid.")
    for field in ("direct_open", "consolidated_open", "comparison"):
        _require_timing(metadata[field], label=f"metadata {field}")
    logical = payload["logical_arrays"]
    if not isinstance(logical, Mapping) or set(logical) != set(paths):
        raise ValueError("Subject-shape trial logical inventory is not exact.")
    for path, record in logical.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != {"dtype", "shape", "logical_digest"}
            or type(record["dtype"]) is not str
            or not isinstance(record["shape"], list)
            or not _is_sha256(record["logical_digest"])
        ):
            raise ValueError(f"Subject-shape logical record {path!r} is malformed.")
    _require_contract_artifacts(
        payload["contract_artifacts"],
        archive_path=payload["archive_path"],
        source_run=source_name,
        candidate_run=candidate_name,
        source_manifest_sha256=payload["source_manifest_sha256"],
        candidate_manifest_sha256=payload["candidate_manifest_sha256"],
        producer_link_sha256=payload["candidate_source_manifest_link_sha256"],
        source_publication_sha256=payload["source_publication_receipt_sha256"],
        candidate_publication_sha256=payload[
            "candidate_publication_receipt_sha256"
        ],
        storage_receipt_digest=payload[
            "candidate_storage_receipt_payload_digest"
        ],
        source_refined_run=payload["source_refined_run"],
        logical=logical,
    )
    primary = payload["primary_access"]
    full = payload["full_scan"]
    if (
        not isinstance(primary, Mapping)
        or set(primary) != {"arrays", "total_wall_seconds", "total_cpu_seconds"}
        or not isinstance(primary["arrays"], Mapping)
        or set(primary["arrays"]) != set(paths)
    ):
        raise ValueError("Subject-shape primary-access evidence is not exact.")
    if (
        not isinstance(full, Mapping)
        or set(full)
        != {"arrays", "total_wall_seconds", "total_cpu_seconds", "total_decoded_bytes"}
        or not isinstance(full["arrays"], Mapping)
        or set(full["arrays"]) != set(paths)
    ):
        raise ValueError("Subject-shape full-scan evidence is not exact.")
    for field in ("total_wall_seconds", "total_cpu_seconds"):
        _require_nonnegative_number(primary[field], label=f"primary {field}")
        _require_nonnegative_number(full[field], label=f"full scan {field}")
    if type(full["total_decoded_bytes"]) is not int or full["total_decoded_bytes"] < 0:
        raise ValueError("Subject-shape full-scan byte total is invalid.")
    for path, record in primary["arrays"].items():
        if not isinstance(record, Mapping) or set(record) != {
            "mode",
            "execution_axis",
            "operation_count",
            "decoded_bytes",
            "selection_digest",
            "workload_id",
            "selection",
            "timing",
        }:
            raise ValueError(f"Subject-shape primary record {path!r} is malformed.")
        if (
            record["mode"]
            not in {"whole_array", "bounded_row_windows", "random_complete_rows"}
            or type(record["execution_axis"]) is not int
            or type(record["operation_count"]) is not int
            or record["operation_count"] < 0
            or type(record["decoded_bytes"]) is not int
            or record["decoded_bytes"] < 0
            or not _is_sha256(record["selection_digest"])
            or not isinstance(record["selection"], Mapping)
            or record["selection"].get("mode") != record["mode"]
            or type(record["workload_id"]) is not str
        ):
            raise ValueError(f"Subject-shape primary record {path!r} is invalid.")
        _require_timing(record["timing"], label=f"primary {path}")
    for path, record in full["arrays"].items():
        if not isinstance(record, Mapping) or set(record) != {
            "dtype",
            "shape",
            "decoded_bytes",
            "block_count",
            "logical_digest",
            "timing",
        }:
            raise ValueError(f"Subject-shape full-scan record {path!r} is malformed.")
        if (
            type(record["decoded_bytes"]) is not int
            or record["decoded_bytes"] < 0
            or type(record["block_count"]) is not int
            or record["block_count"] < 0
            or logical[path]
            != {
                "dtype": record["dtype"],
                "shape": record["shape"],
                "logical_digest": record["logical_digest"],
            }
        ):
            raise ValueError(f"Subject-shape full-scan record {path!r} is invalid.")
        _require_timing(record["timing"], label=f"full scan {path}")
    if (
        primary["total_wall_seconds"]
        != sum(float(record["timing"]["wall_seconds"]) for record in primary["arrays"].values())
        or primary["total_cpu_seconds"]
        != sum(float(record["timing"]["cpu_seconds"]) for record in primary["arrays"].values())
        or full["total_wall_seconds"]
        != sum(float(record["timing"]["wall_seconds"]) for record in full["arrays"].values())
        or full["total_cpu_seconds"]
        != sum(float(record["timing"]["cpu_seconds"]) for record in full["arrays"].values())
        or full["total_decoded_bytes"]
        != sum(int(record["decoded_bytes"]) for record in full["arrays"].values())
    ):
        raise ValueError("Subject-shape read aggregate totals are not reconstructed.")
    _require_storage(payload["storage"], paths=paths)
    _require_publication_summary(payload["publication_timing"], role=role)
    if payload["publication_timing"]["receipt_sha256"] != expected_publication:
        raise ValueError("Subject-shape publication summary digest binding differs.")
    runtime = payload["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "initial_peak_rss_bytes",
        "final_peak_rss_bytes",
        "peak_rss_is_process_high_water_mark",
        "total_wall_seconds",
        "total_cpu_seconds",
    }:
        raise ValueError("Subject-shape runtime evidence is malformed.")
    if (
        runtime["peak_rss_is_process_high_water_mark"] is not True
        or type(runtime["initial_peak_rss_bytes"]) is not int
        or runtime["initial_peak_rss_bytes"] < 0
        or type(runtime["final_peak_rss_bytes"]) is not int
        or runtime["final_peak_rss_bytes"] < runtime["initial_peak_rss_bytes"]
    ):
        raise ValueError("Subject-shape RSS evidence is invalid.")
    _require_nonnegative_number(runtime["total_wall_seconds"], label="runtime wall")
    _require_nonnegative_number(runtime["total_cpu_seconds"], label="runtime CPU")
    physical = payload["physical_io"]
    if (
        not isinstance(physical, Mapping)
        or set(physical) != {"request_count", "transferred_bytes", "availability"}
        or physical["request_count"] is not None
        or physical["transferred_bytes"] is not None
        or physical["availability"] != _TRIAL_PHYSICAL_AVAILABILITY
    ):
        raise ValueError("Subject-shape trial fabricates physical-I/O telemetry.")


def run_single_trial(
    archive_path: Path | str,
    *,
    parent: str,
    source_run: str,
    candidate_run: str,
    role: str,
    repetition_index: int,
    order_position: int,
    driver_process_id: int,
    seed: int,
    cache_state: str,
    suite_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    archive = _safe_archive(archive_path)
    parent = _require_parent(parent)
    source_name = _safe_name(source_run, label="source_run")
    candidate_name = _safe_name(candidate_run, label="candidate_run")
    if source_name == candidate_name:
        raise ValueError("source_run and candidate_run must differ.")
    if role not in {"source", "candidate"}:
        raise ValueError("role must be source or candidate.")
    if type(repetition_index) is not int or repetition_index < 0:
        raise ValueError("repetition_index must be nonnegative.")
    if order_position not in {0, 1}:
        raise ValueError("order_position must be 0 or 1.")
    if (
        type(driver_process_id) is not int
        or driver_process_id < 1
        or driver_process_id == os.getpid()
    ):
        raise ValueError("driver_process_id must differ from this exact process.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative exact integer.")
    if type(cache_state) is not str or not cache_state.strip():
        raise ValueError("cache_state must be explicitly declared.")
    require_analysis_benchmark_suite_manifest(suite_manifest)
    suite_payload = suite_manifest["payload"]
    if (
        suite_payload["family_id"] != FAMILY_ID
        or suite_payload["seed"] != seed
        or repetition_index >= suite_payload["repetitions"]
        or _trial_order(seed=seed, repetition_index=repetition_index)[order_position]
        != role
    ):
        raise ValueError("Trial differs from its deterministic suite/order binding.")
    source_path = f"{parent}/{source_name}"
    candidate_path = f"{parent}/{candidate_name}"
    run_path = source_path if role == "source" else candidate_path
    receipt_manifest = suite_payload["storage_plan_receipt"]
    receipt_digest = receipt_manifest["payload_digest"]
    growth_axes = _growth_axes(suite_manifest)
    paths = sorted(growth_axes)
    if [str(item["array_path"]) for item in _read_cases(suite_manifest)] != paths:
        raise ValueError("Subject-shape suite primary inventory differs from its plan.")
    trial_wall = time.perf_counter()
    trial_cpu = time.process_time()
    started = utc_now()
    initial_rss = peak_rss_bytes()

    (direct_root, direct_group), direct_open = _measure(
        lambda: (
            (root := zarr.open_group(str(archive), mode="r", use_consolidated=False)),
            _group(root, run_path),
        )
    )
    (consolidated_root, consolidated_group), consolidated_open = _measure(
        lambda: (
            (root := zarr.open_group(str(archive), mode="r", use_consolidated=True)),
            _group(root, run_path),
        )
    )

    def validate() -> dict[str, Any]:
        source = _validate_role(direct_root, run_path=source_path, role="source")
        candidate = _validate_role(
            direct_root, run_path=candidate_path, role="candidate"
        )
        if (
            source["declarations"] != candidate["declarations"]
            or source["receipt_manifest"] != candidate["receipt_manifest"]
            or source["paths"] != candidate["paths"]
            or source["row_count"] != candidate["row_count"]
            or source["source_refined_run"] != candidate["source_refined_run"]
            or source["paths"] != paths
            or candidate["receipt_manifest"] != receipt_manifest
        ):
            raise ValueError("Subject-shape trial pair differs from the frozen suite.")
        source_publication = _publication_receipt(
            source["group"],
            role="source",
            archive=archive,
            run_path=source_path,
            run_name=source_name,
            row_count=source["row_count"],
            manifest_sha256=source["manifest_sha256"],
            storage_receipt=None,
        )
        candidate_publication = _publication_receipt(
            candidate["group"],
            role="candidate",
            archive=archive,
            run_path=candidate_path,
            run_name=candidate_name,
            row_count=candidate["row_count"],
            manifest_sha256=candidate["manifest_sha256"],
            storage_receipt=candidate["receipt_manifest"],
        )
        current = source if role == "source" else candidate
        publication = source_publication if role == "source" else candidate_publication
        return {
            "valid": True,
            "role": role,
            "array_count": len(paths),
            "declarations_sha256": candidate["declarations_sha256"],
            "run_manifest_sha256": current["manifest_sha256"],
            "validation_candidate_source_manifest_link_sha256": (
                candidate["source_manifest_link_sha256"]
                if role == "candidate"
                else None
            ),
            "publication_receipt_sha256": publication["receipt_sha256"],
            "source_manifest_sha256": source["manifest_sha256"],
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "candidate_source_manifest_link_sha256_pair": candidate[
                "source_manifest_link_sha256"
            ],
            "source_publication": source_publication,
            "candidate_publication": candidate_publication,
            "source_refined_run": candidate["source_refined_run"],
            "contract_artifacts": {
                "source_final_manifest": source["manifest"],
                "candidate_final_manifest": candidate["manifest"],
                "candidate_retained_producer_manifest_link": candidate[
                    "source_manifest_link"
                ],
                "candidate_storage_receipt": candidate["receipt_manifest"],
                "source_atomic_publication_receipt": source["group"].attrs.get(
                    "cluster_output_staging"
                ),
                "candidate_atomic_publication_receipt": candidate[
                    "group"
                ].attrs.get("cluster_output_staging"),
                "source_consumed_unbound_manifest": source[
                    "consumed_unbound_manifest"
                ],
                "candidate_consumed_unbound_manifest": candidate[
                    "consumed_unbound_manifest"
                ],
                "metadata_documents": {
                    "source": _metadata_declarations(archive, source_path),
                    "candidate": _metadata_declarations(archive, candidate_path),
                },
            },
        }

    pair, validation_timing = _measure(validate)
    validation = {
        key: pair[key]
        for key in (
            "valid",
            "role",
            "array_count",
            "declarations_sha256",
            "run_manifest_sha256",
            "publication_receipt_sha256",
        )
    }
    validation["candidate_source_manifest_link_sha256"] = pair[
        "validation_candidate_source_manifest_link_sha256"
    ]

    def compare_metadata() -> dict[str, Any]:
        receipt = validate_direct_consolidated_subtree(archive, subtree_path=run_path)
        if receipt.array_count != len(paths):
            raise ValueError("Persisted metadata does not declare the exact array set.")
        declarations = _metadata_declarations(archive, run_path)
        declarations_digest = canonical_json_sha256(declarations)
        if receipt.declarations_sha256 != declarations_digest:
            raise ValueError("Metadata receipt differs from embedded declarations.")
        return {
            "equivalent": True,
            "array_count": receipt.array_count,
            "group_count": receipt.group_count,
            "node_count": receipt.node_count,
            "subtree_declarations_digest": declarations_digest,
            "declarations": declarations,
        }

    metadata, comparison_timing = _measure(compare_metadata)
    primary: dict[str, Any] = {}
    primary_wall = 0.0
    primary_cpu = 0.0
    for case in _read_cases(suite_manifest):
        path = str(case["array_path"])
        result, timing = _measure(
            lambda path=path, case=case: _primary_read(
                _array(consolidated_group, path),
                case["selection"],
                growth_axis=growth_axes[path],
            )
        )
        primary[path] = {
            **result,
            "workload_id": case["case"]["workload"]["workload_id"],
            "selection": case["selection"],
            "timing": timing,
        }
        primary_wall += timing["wall_seconds"]
        primary_cpu += timing["cpu_seconds"]
    scans: dict[str, Any] = {}
    scan_wall = 0.0
    scan_cpu = 0.0
    for path in paths:
        result, timing = _measure(
            lambda path=path: _full_scan(
                _array(consolidated_group, path), growth_axis=growth_axes[path]
            )
        )
        scans[path] = {**result, "timing": timing}
        scan_wall += timing["wall_seconds"]
        scan_cpu += timing["cpu_seconds"]
    logical = {
        path: {
            "dtype": scans[path]["dtype"],
            "shape": scans[path]["shape"],
            "logical_digest": scans[path]["logical_digest"],
        }
        for path in paths
    }
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive),
        "parent_path": parent,
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "role": role,
        "run_name": source_name if role == "source" else candidate_name,
        "run_path": run_path,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "process_id": os.getpid(),
        "driver_process_id": driver_process_id,
        "seed": seed,
        "cache_state": cache_state,
        "suite_payload_digest": suite_manifest["payload_digest"],
        "candidate_storage_receipt_payload_digest": receipt_digest,
        "array_paths": paths,
        "declarations_sha256": pair["declarations_sha256"],
        "source_manifest_sha256": pair["source_manifest_sha256"],
        "candidate_manifest_sha256": pair["candidate_manifest_sha256"],
        "candidate_source_manifest_link_sha256": pair[
            "candidate_source_manifest_link_sha256_pair"
        ],
        "source_publication_receipt_sha256": pair["source_publication"][
            "receipt_sha256"
        ],
        "candidate_publication_receipt_sha256": pair["candidate_publication"][
            "receipt_sha256"
        ],
        "source_refined_run": pair["source_refined_run"],
        "contract_artifacts": pair["contract_artifacts"],
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "environment": _environment(archive, cache_state),
        "validation": {**validation, "timing": validation_timing},
        "metadata": {
            **metadata,
            "direct_open": direct_open,
            "consolidated_open": consolidated_open,
            "comparison": comparison_timing,
        },
        "primary_access": {
            "arrays": primary,
            "total_wall_seconds": primary_wall,
            "total_cpu_seconds": primary_cpu,
        },
        "full_scan": {
            "arrays": scans,
            "total_wall_seconds": scan_wall,
            "total_cpu_seconds": scan_cpu,
            "total_decoded_bytes": sum(
                int(item["decoded_bytes"]) for item in scans.values()
            ),
        },
        "logical_arrays": logical,
        "storage": _selected_storage_stats(archive, run_path, paths),
        "publication_timing": (
            pair["source_publication"]
            if role == "source"
            else pair["candidate_publication"]
        ),
        "runtime": {
            "initial_peak_rss_bytes": initial_rss,
            "final_peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_is_process_high_water_mark": True,
            "total_wall_seconds": float(time.perf_counter() - trial_wall),
            "total_cpu_seconds": float(time.process_time() - trial_cpu),
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": _TRIAL_PHYSICAL_AVAILABILITY,
        },
    }
    result = _strict_envelope(TRIAL_SCHEMA_ID, payload)
    require_trial_result(result)
    return result


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    return (
        ("source", "candidate")
        if (seed + repetition_index) % 2 == 0
        else ("candidate", "source")
    )


def _median(
    trials: Sequence[Mapping[str, Any]], role: str, path: Sequence[str]
) -> float:
    values: list[float] = []
    for trial in trials:
        value: Any = trial["payload"]
        if value["role"] != role:
            continue
        for component in path:
            value = value[component]
        values.append(float(value))
    return float(statistics.median(values))


def _summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("source", "candidate"):
        example = next(
            trial["payload"] for trial in trials if trial["payload"]["role"] == role
        )
        result[role] = {
            "median_direct_open_wall_seconds": _median(
                trials, role, ("metadata", "direct_open", "wall_seconds")
            ),
            "median_consolidated_open_wall_seconds": _median(
                trials, role, ("metadata", "consolidated_open", "wall_seconds")
            ),
            "median_validation_wall_seconds": _median(
                trials, role, ("validation", "timing", "wall_seconds")
            ),
            "median_primary_access_wall_seconds": _median(
                trials, role, ("primary_access", "total_wall_seconds")
            ),
            "median_full_scan_wall_seconds": _median(
                trials, role, ("full_scan", "total_wall_seconds")
            ),
            "median_total_wall_seconds": _median(
                trials, role, ("runtime", "total_wall_seconds")
            ),
            "median_total_cpu_seconds": _median(
                trials, role, ("runtime", "total_cpu_seconds")
            ),
            "median_peak_rss_bytes": _median(
                trials, role, ("runtime", "final_peak_rss_bytes")
            ),
            "selected_array_storage": example["storage"]["totals"],
            "payload_object_count": example["storage"]["payload_object_count"],
            "publication_timing": example["publication_timing"],
        }
    return result


def require_matrix_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    expected = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "parent_path",
        "source_run_name",
        "candidate_run_name",
        "seed",
        "repetitions",
        "cache_state",
        "driver_process_id",
        "started_at_utc",
        "finished_at_utc",
        "suite",
        "array_paths",
        "declarations_sha256",
        "source_manifest_sha256",
        "candidate_manifest_sha256",
        "candidate_source_manifest_link_sha256",
        "source_publication_receipt_sha256",
        "candidate_publication_receipt_sha256",
        "candidate_storage_receipt_payload_digest",
        "source_refined_run",
        "contract_artifacts",
        "trial_order",
        "trial_files",
        "trials",
        "correctness",
        "performance_summary",
        "archive_read_only_metadata_guard",
        "physical_io",
        "promotion_decision",
    }
    if set(payload) != expected:
        raise ValueError("Subject-shape matrix payload has an unexpected field set.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or payload["parent_path"] != PARENT_PATH
    ):
        raise ValueError("Subject-shape matrix identity or parent differs.")
    source_name = _safe_name(payload["source_run_name"], label="source_run_name")
    candidate_name = _safe_name(payload["candidate_run_name"], label="candidate_run_name")
    if source_name == candidate_name:
        raise ValueError("Subject-shape matrix source/candidate names must differ.")
    if type(payload["seed"]) is not int or payload["seed"] < 0:
        raise ValueError("Subject-shape matrix seed is invalid.")
    if type(payload["repetitions"]) is not int or payload["repetitions"] < 1:
        raise ValueError("Subject-shape matrix repetitions are invalid.")
    if type(payload["driver_process_id"]) is not int or payload[
        "driver_process_id"
    ] < 1:
        raise ValueError("Subject-shape matrix driver process is invalid.")
    require_analysis_benchmark_suite_manifest(payload["suite"])
    suite = payload["suite"]
    paths = payload["array_paths"]
    if (
        not isinstance(paths, list)
        or paths != sorted(set(paths))
        or not paths
        or suite["payload"]["family_id"] != FAMILY_ID
        or suite["payload"]["seed"] != payload["seed"]
        or suite["payload"]["repetitions"] != payload["repetitions"]
        or sorted(_growth_axes(suite)) != paths
        or [str(case["array_path"]) for case in _read_cases(suite)] != paths
        or suite["payload"]["storage_plan_receipt"]["payload_digest"]
        != payload["candidate_storage_receipt_payload_digest"]
    ):
        raise ValueError("Subject-shape matrix suite binding differs.")
    for field in (
        "declarations_sha256",
        "source_manifest_sha256",
        "candidate_manifest_sha256",
        "candidate_source_manifest_link_sha256",
        "candidate_publication_receipt_sha256",
        "candidate_storage_receipt_payload_digest",
    ):
        if not _is_sha256(payload[field]):
            raise ValueError(f"Subject-shape matrix {field} is invalid.")
    if payload["source_publication_receipt_sha256"] is not None and not _is_sha256(
        payload["source_publication_receipt_sha256"]
    ):
        raise ValueError("Subject-shape source publication receipt digest is invalid.")
    if type(payload["source_refined_run"]) is not str or not payload[
        "source_refined_run"
    ].strip():
        raise ValueError("Subject-shape matrix source identity is invalid.")
    expected_order = [
        {
            "repetition_index": index,
            "roles": list(_trial_order(seed=payload["seed"], repetition_index=index)),
        }
        for index in range(payload["repetitions"])
    ]
    if payload["trial_order"] != expected_order:
        raise ValueError("Subject-shape matrix order is not deterministic v1.")
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != 2 * payload["repetitions"]:
        raise ValueError("Subject-shape matrix trial count is invalid.")
    identity_fields = (
        "array_paths",
        "declarations_sha256",
        "source_manifest_sha256",
        "candidate_manifest_sha256",
        "candidate_source_manifest_link_sha256",
        "source_publication_receipt_sha256",
        "candidate_publication_receipt_sha256",
        "candidate_storage_receipt_payload_digest",
        "source_refined_run",
        "contract_artifacts",
    )
    for trial in trials:
        require_trial_result(trial)
        observed = trial["payload"]
        if (
            observed["archive_path"] != payload["archive_path"]
            or observed["parent_path"] != PARENT_PATH
            or observed["source_run_name"] != source_name
            or observed["candidate_run_name"] != candidate_name
            or observed["seed"] != payload["seed"]
            or observed["cache_state"] != payload["cache_state"]
            or observed["driver_process_id"] != payload["driver_process_id"]
            or observed["suite_payload_digest"] != suite["payload_digest"]
            or any(observed[field] != payload[field] for field in identity_fields)
        ):
            raise ValueError("Subject-shape matrix/trial identity binding differs.")
    observed_order = [
        {
            "repetition_index": index,
            "roles": [
                trial["payload"]["role"]
                for trial in sorted(
                    (
                        item
                        for item in trials
                        if item["payload"]["repetition_index"] == index
                    ),
                    key=lambda item: item["payload"]["order_position"],
                )
            ],
        }
        for index in range(payload["repetitions"])
    ]
    if observed_order != expected_order:
        raise ValueError("Subject-shape trials differ from declared order.")
    for index in range(payload["repetitions"]):
        repeated = [
            trial["payload"]
            for trial in trials
            if trial["payload"]["repetition_index"] == index
        ]
        if (
            len(repeated) != 2
            or {item["order_position"] for item in repeated} != {0, 1}
            or {
                (item["order_position"], item["role"])
                for item in repeated
            }
            != set(enumerate(_trial_order(seed=payload["seed"], repetition_index=index)))
        ):
            raise ValueError("Subject-shape repetition role/position evidence differs.")
    process_ids = [trial["payload"]["process_id"] for trial in trials]
    if (
        len(process_ids) != len(set(process_ids))
        or payload["driver_process_id"] in process_ids
    ):
        raise ValueError("Subject-shape fresh-process identities are not distinct.")
    expected_files = [
        f"trials/rep_{index:02d}_pos_{position}_{role}.json"
        for index in range(payload["repetitions"])
        for position, role in enumerate(
            _trial_order(seed=payload["seed"], repetition_index=index)
        )
    ]
    if payload["trial_files"] != expected_files:
        raise ValueError("Subject-shape matrix trial-file inventory differs.")
    logical = trials[0]["payload"]["logical_arrays"]
    _require_contract_artifacts(
        payload["contract_artifacts"],
        archive_path=payload["archive_path"],
        source_run=source_name,
        candidate_run=candidate_name,
        source_manifest_sha256=payload["source_manifest_sha256"],
        candidate_manifest_sha256=payload["candidate_manifest_sha256"],
        producer_link_sha256=payload["candidate_source_manifest_link_sha256"],
        source_publication_sha256=payload["source_publication_receipt_sha256"],
        candidate_publication_sha256=payload[
            "candidate_publication_receipt_sha256"
        ],
        storage_receipt_digest=payload[
            "candidate_storage_receipt_payload_digest"
        ],
        source_refined_run=payload["source_refined_run"],
        logical=logical,
    )
    primary = {
        path: record["selection_digest"]
        for path, record in trials[0]["payload"]["primary_access"]["arrays"].items()
    }
    if not all(trial["payload"]["logical_arrays"] == logical for trial in trials):
        raise ValueError("Subject-shape matrix full decoded values differ.")
    if not all(
        {
            path: record["selection_digest"]
            for path, record in trial["payload"]["primary_access"]["arrays"].items()
        }
        == primary
        for trial in trials
    ):
        raise ValueError("Subject-shape matrix primary decoded values differ.")
    suite_cases = {str(case["array_path"]): case for case in _read_cases(suite)}
    for trial in trials:
        for path, case in suite_cases.items():
            record = trial["payload"]["primary_access"]["arrays"][path]
            if (
                record["workload_id"] != case["case"]["workload"]["workload_id"]
                or record["selection"] != case["selection"]
            ):
                raise ValueError("Subject-shape trial workload differs from its suite.")
    correctness = payload["correctness"]
    if correctness != {
        "full_decoded_logical_equality": True,
        "primary_access_decoded_equality": True,
        "direct_consolidated_metadata_equivalence": True,
        "source_and_candidate_validation": True,
        "archive_metadata_unchanged": True,
        "all_passed": True,
    }:
        raise ValueError("Subject-shape correctness gates did not all pass.")
    guard = payload["archive_read_only_metadata_guard"]
    if (
        not isinstance(guard, Mapping)
        or set(guard) != {"before", "after", "unchanged"}
        or guard["unchanged"] is not True
        or guard["before"] != guard["after"]
    ):
        raise ValueError("Subject-shape metadata immutability gate did not pass.")
    _require_metadata_guard(guard["before"])
    _require_metadata_guard(guard["after"])
    physical = payload["physical_io"]
    if (
        not isinstance(physical, Mapping)
        or set(physical) != {"request_count", "transferred_bytes", "availability"}
        or physical["request_count"] is not None
        or physical["transferred_bytes"] is not None
        or physical["availability"] != _MATRIX_PHYSICAL_AVAILABILITY
    ):
        raise ValueError("Subject-shape matrix fabricates physical-I/O telemetry.")
    if payload["promotion_decision"] != {
        "authorized": False,
        "reason": "benchmark_only; profile promotion requires a separate reviewed decision",
    }:
        raise ValueError("Subject-shape benchmark cannot authorize promotion.")
    if payload["performance_summary"] != _summary(trials):
        raise ValueError("Subject-shape performance summary differs from its trials.")


def run_benchmark_matrix(
    archive_path: Path | str,
    *,
    parent: str,
    source_run: str,
    candidate_run: str,
    output_dir: Path | str,
    cache_state: str,
    seed: int = DEFAULT_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    archive = _safe_archive(archive_path)
    parent = _require_parent(parent)
    source_name = _safe_name(source_run, label="source_run")
    candidate_name = _safe_name(candidate_run, label="candidate_run")
    if source_name == candidate_name:
        raise ValueError("source_run and candidate_run must differ.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative exact integer.")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive exact integer.")
    if type(cache_state) is not str or not cache_state.strip():
        raise ValueError("cache_state must be explicitly declared.")
    output = _safe_output(output_dir, archive=archive)
    source_path = f"{parent}/{source_name}"
    candidate_path = f"{parent}/{candidate_name}"
    preflight = _preflight(
        archive,
        source_path=source_path,
        candidate_path=candidate_path,
        seed=seed,
        repetitions=repetitions,
    )
    guard_before = _metadata_guard(
        archive,
        run_paths=(source_path, candidate_path),
        source_refined_run=preflight["source_refined_run"],
    )
    output.mkdir(parents=True, exist_ok=False)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    suite_path = output / "analysis_benchmark_suite.json"
    _write_json(suite_path, preflight["suite"])
    trials: list[Mapping[str, Any]] = []
    trial_order: list[dict[str, Any]] = []
    trial_files: list[str] = []
    started = utc_now()
    environment = os.environ.copy()
    environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for repetition_index in range(repetitions):
        order = _trial_order(seed=seed, repetition_index=repetition_index)
        trial_order.append({"repetition_index": repetition_index, "roles": list(order)})
        for position, role in enumerate(order):
            filename = f"rep_{repetition_index:02d}_pos_{position}_{role}.json"
            trial_path = trials_dir / filename
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_subject_shape_v4_candidate",
                "trial",
                str(archive),
                "--parent",
                parent,
                "--source-run",
                source_name,
                "--candidate-run",
                candidate_name,
                "--role",
                role,
                "--repetition-index",
                str(repetition_index),
                "--order-position",
                str(position),
                "--driver-process-id",
                str(os.getpid()),
                "--seed",
                str(seed),
                "--cache-state",
                cache_state,
                "--suite-file",
                str(suite_path),
                "--benchmark-root",
                str(output),
                "--output-file",
                str(trial_path),
            ]
            completed = subprocess.run(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Fresh-process subject-shape trial failed: "
                    f"command={command!r}, stdout={completed.stdout!r}, "
                    f"stderr={completed.stderr!r}."
                )
            trial = _read_json(trial_path)
            require_trial_result(trial)
            trials.append(trial)
            trial_files.append(str(trial_path.relative_to(output)))
    logical_equality = all(
        trial["payload"]["logical_arrays"] == trials[0]["payload"]["logical_arrays"]
        for trial in trials
    )
    primary_equality = all(
        {
            path: item["selection_digest"]
            for path, item in trial["payload"]["primary_access"]["arrays"].items()
        }
        == {
            path: item["selection_digest"]
            for path, item in trials[0]["payload"]["primary_access"]["arrays"].items()
        }
        for trial in trials
    )
    if not logical_equality or not primary_equality:
        raise ValueError("Subject-shape source/candidate decoded benchmark values differ.")
    guard_after = _metadata_guard(
        archive,
        run_paths=(source_path, candidate_path),
        source_refined_run=preflight["source_refined_run"],
    )
    if guard_before != guard_after:
        raise RuntimeError("Archive metadata changed during read-only benchmark execution.")
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive),
        "parent_path": parent,
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "seed": seed,
        "repetitions": repetitions,
        "cache_state": cache_state,
        "driver_process_id": os.getpid(),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "suite": preflight["suite"],
        "array_paths": preflight["array_paths"],
        "declarations_sha256": preflight["declarations_sha256"],
        "source_manifest_sha256": preflight["source_manifest_sha256"],
        "candidate_manifest_sha256": preflight["candidate_manifest_sha256"],
        "candidate_source_manifest_link_sha256": preflight[
            "candidate_source_manifest_link_sha256"
        ],
        "source_publication_receipt_sha256": preflight[
            "source_publication_receipt_sha256"
        ],
        "candidate_publication_receipt_sha256": preflight[
            "candidate_publication_receipt_sha256"
        ],
        "candidate_storage_receipt_payload_digest": preflight[
            "candidate_storage_receipt_payload_digest"
        ],
        "source_refined_run": preflight["source_refined_run"],
        "contract_artifacts": preflight["contract_artifacts"],
        "trial_order": trial_order,
        "trial_files": trial_files,
        "trials": trials,
        "correctness": {
            "full_decoded_logical_equality": True,
            "primary_access_decoded_equality": True,
            "direct_consolidated_metadata_equivalence": True,
            "source_and_candidate_validation": True,
            "archive_metadata_unchanged": True,
            "all_passed": True,
        },
        "performance_summary": _summary(trials),
        "archive_read_only_metadata_guard": {
            "before": guard_before,
            "after": guard_after,
            "unchanged": True,
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": _MATRIX_PHYSICAL_AVAILABILITY,
        },
        "promotion_decision": {
            "authorized": False,
            "reason": "benchmark_only; profile promotion requires a separate reviewed decision",
        },
    }
    result = _strict_envelope(MATRIX_SCHEMA_ID, payload)
    require_matrix_result(result)
    _write_json(output / "matrix_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    matrix = commands.add_parser("matrix")
    trial = commands.add_parser("trial")
    for child in (matrix, trial):
        child.add_argument("zarr_path", type=Path)
        child.add_argument("--parent", required=True)
        child.add_argument("--source-run", required=True)
        child.add_argument("--candidate-run", required=True)
        child.add_argument("--seed", type=int, default=DEFAULT_SEED)
        child.add_argument("--cache-state", required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--order-position", type=int, choices=(0, 1), required=True)
    trial.add_argument("--driver-process-id", type=int, required=True)
    trial.add_argument("--suite-file", type=Path, required=True)
    trial.add_argument("--benchmark-root", type=Path, required=True)
    trial.add_argument("--output-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        result = run_benchmark_matrix(
            args.zarr_path,
            parent=args.parent,
            source_run=args.source_run,
            candidate_run=args.candidate_run,
            output_dir=args.output_dir,
            cache_state=args.cache_state,
            seed=args.seed,
            repetitions=args.repetitions,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "matrix_result": str(
                        Path(args.output_dir).expanduser().resolve()
                        / "matrix_result.json"
                    ),
                    "payload_digest": result["payload_digest"],
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    suite = _read_json(args.suite_file.expanduser().resolve())
    output = _safe_trial_output(
        args.output_file,
        root=args.benchmark_root.expanduser().resolve(),
    )
    result = run_single_trial(
        args.zarr_path,
        parent=args.parent,
        source_run=args.source_run,
        candidate_run=args.candidate_run,
        role=args.role,
        repetition_index=args.repetition_index,
        order_position=args.order_position,
        driver_process_id=args.driver_process_id,
        seed=args.seed,
        cache_state=args.cache_state,
        suite_manifest=suite,
    )
    _write_json(output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "DEFAULT_REPETITIONS",
    "DEFAULT_SEED",
    "FAMILY_ID",
    "MATRIX_SCHEMA_ID",
    "PARENT_PATH",
    "TRIAL_SCHEMA_ID",
    "require_matrix_result",
    "require_trial_result",
    "run_benchmark_matrix",
    "run_single_trial",
]
