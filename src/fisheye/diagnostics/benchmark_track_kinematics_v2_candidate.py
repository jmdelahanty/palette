"""Read-only benchmark for one sealed track-motion v1/v2 candidate pair.

The source is opened through Palette's maintained public track reader.  The
flat-lineage v2 candidate has no public consumer and is therefore opened only
through the explicit diagnostic validator in this module.  The controller
rotates source/candidate order across fresh child processes and writes evidence
outside the archive.  Nothing in this module promotes a profile or changes a
selector.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
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
from typing import Any

import numpy as np
import zarr

from fisheye.analysis import track_kinematics as track_contract
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.analysis.track_kinematics_schema import (
    TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS,
    TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS,
    TRACK_KINEMATICS_RUN_DECLARATIONS,
)
from fisheye.analysis.track_kinematics_storage import (
    TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_ATTR,
    TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID,
    build_flat_candidate_declarations,
    build_flat_candidate_storage_receipt,
    flat_candidate_logical_hashes,
    source_flat_projection_hashes,
    validate_flat_candidate,
)
from fisheye.shared.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
    ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    SERIALIZATION_POLICY,
)
from fisheye.analysis_workflows.materializers.track_kinematics_candidate import (
    PUBLISH_SCHEMA_ID,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import peak_rss_bytes, storage_stats, utc_now
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import get_storage_profile

PARENT_PATH = "analysis/track_kinematics_runs"
RUN_TYPE = "offline"
FAMILY_ID = "track_kinematics_v2_flat_lineage"
BENCHMARK_ID = "track_kinematics_v2_source_candidate_read_matrix_v1"
PAIR_SCHEMA_ID = "palette.track_kinematics.v2_pair_validation"
TRIAL_SCHEMA_ID = "palette.track_kinematics.v2_read_trial"
MATRIX_SCHEMA_ID = "palette.track_kinematics.v2_read_matrix"
SCHEMA_VERSION = 1
DEFAULT_REPETITIONS = 5
DEFAULT_SEED = 17
DEFAULT_WINDOW_ROWS = 4096
DEFAULT_WINDOWS_PER_ARRAY = 4
PHYSICAL_BUNDLE_BENCHMARKED = False
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_SHA256_LENGTH = 64
_PUBLISH_POLICY = "track_flat_lineage_byte_planned_atomic_nonpromoting_publish"
_ROLLBACK_POLICY = "retain_failed_public_tombstone_leave_track_selectors_untouched"
_PHYSICAL_IO_REASON = (
    "unavailable_without_os_or_filesystem_tracing; logical decoded bytes and "
    "filesystem object sizes are not physical transfer telemetry"
)
_CACHE_STATE = "fresh_child_post_pair_validation_os_cache_uncontrolled"
_RUN_TREE_STORAGE_SCOPE = "selected_run_tree_current_post_read"
_PUBLISH_PHYSICAL_RECEIPT_ROLE = (
    "opaque_publisher_provenance_not_replayed_not_benchmark_authority"
)
_COPY_BACKEND_VERIFICATION = {
    "python": "sha256_all_physical_files",
    "rsync": "rsync_checksum_dry_run",
}
_SOURCE_STRUCTURED_FIELDS = {
    "source_frame_interpolation/left_source_frame_index": (
        "source_frame_interpolation",
        "left_source_frame_index",
    ),
    "source_frame_interpolation/right_source_frame_index": (
        "source_frame_interpolation",
        "right_source_frame_index",
    ),
    "source_frame_interpolation/right_weight": (
        "source_frame_interpolation",
        "right_weight",
    ),
    "source_instance_key/valid": ("source_instance_key", "valid"),
    "source_instance_key/value": ("source_instance_key", "instance_key"),
}
_PUBLISH_RECEIPT_FIELDS = {
    "profile_id",
    "source_run",
    "source_projection_hashes",
    "local_direct_consolidated_array_count",
    "materialization_seconds",
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
        raise ValueError("Track benchmark envelope has an unexpected field set.")
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Track benchmark envelope schema identity is unsupported.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Track benchmark envelope payload must be one object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Track benchmark envelope payload digest mismatch.")
    json.dumps(value, allow_nan=False)
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonnegative(value: object, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be one finite nonnegative number.")


def _require_utc_timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one UTC timestamp string.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must carry an explicit UTC offset.")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must use UTC.")
    return parsed


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
    if (
        not value
        or value != value.strip()
        or value in _ALIASES
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return value


def _safe_archive_path(value: str | Path) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Archive must be one existing non-symlink directory.")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("Archive path must be canonical and contain no symlink alias.")
    return resolved


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
        resolved = current.resolve(strict=True)
        try:
            resolved.relative_to(archive)
        except ValueError as exc:
            raise ValueError(
                f"Archive component escapes containment: {current}."
            ) from exc
    return current


def _guard_archive_tree(archive: Path, relative: str, *, label: str) -> Path:
    """Reject every symlink below a selected hierarchy before Zarr opens it."""

    root = _guard_relative_path(archive, relative)
    if not root.is_dir():
        raise ValueError(f"{label} must resolve to one directory hierarchy.")
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directory_names, *file_names):
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"{label} contains a forbidden symlink: {child}.")
            try:
                child.resolve(strict=True).relative_to(archive)
            except ValueError as exc:
                raise ValueError(
                    f"{label} descendant escapes the archive: {child}."
                ) from exc
    return root


def _guard_output_path(archive: Path, output: str | Path) -> Path:
    path = Path(output).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing existing benchmark output: {path}.")
    parent = path.parent.resolve(strict=True)
    if parent.is_symlink():
        raise ValueError("Benchmark output parent must not be a symlink.")
    try:
        parent.relative_to(archive)
    except ValueError:
        pass
    else:
        raise ValueError("Benchmark output must be outside the analysis archive.")
    try:
        archive.relative_to(parent)
    except ValueError:
        pass
    else:
        raise ValueError("Benchmark output must not contain the analysis archive.")
    return path


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _iter_array_paths(group: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    for name, _array in sorted(group.arrays(), key=lambda item: str(item[0])):
        result.append(f"{prefix}/{name}" if prefix else str(name))
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        result.extend(_iter_array_paths(child, child_prefix))
    return result


def _iter_group_paths(group: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        result.append(child_prefix)
        result.extend(_iter_group_paths(child, child_prefix))
    return result


def _metadata_guard(archive: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(archive.rglob("zarr.json")):
        if path.is_symlink():
            raise ValueError(f"Metadata file must not be a symlink: {path}.")
        relative = path.relative_to(archive).as_posix()
        payload = path.read_bytes()
        stat = path.stat()
        record = {
            "path": relative,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        records.append(record)
    if not records:
        raise ValueError("Archive has no Zarr metadata files.")
    return {
        "metadata_file_count": len(records),
        "metadata_root_sha256": canonical_json_sha256(records),
        "records": records,
    }


def _archive_guard(archive: Path) -> dict[str, Any]:
    stats = storage_stats(archive)
    return {
        "metadata": _metadata_guard(archive),
        "storage": stats,
    }


_STORAGE_STAT_FIELDS = {
    "file_count",
    "metadata_file_count",
    "payload_file_count",
    "apparent_bytes",
    "allocated_bytes",
}


def _require_storage_stats(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _STORAGE_STAT_FIELDS:
        raise ValueError(f"{label} storage-stat field set differs.")
    if any(type(value[field]) is not int or value[field] < 0 for field in value):
        raise ValueError(f"{label} storage-stat values are invalid.")
    if value["file_count"] != (
        value["metadata_file_count"] + value["payload_file_count"]
    ):
        raise ValueError(f"{label} storage-stat file accounting differs.")


def _require_metadata_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "metadata_file_count",
        "metadata_root_sha256",
        "records",
    }:
        raise ValueError("Track metadata-guard field set differs.")
    records = value["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("Track metadata-guard inventory is invalid.")
    paths: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
        }:
            raise ValueError("Track metadata-guard record field set differs.")
        path = record["path"]
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(component in {"", ".", ".."} for component in path.split("/"))
            or not path.endswith("zarr.json")
        ):
            raise ValueError("Track metadata-guard path is unsafe.")
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or type(record["mtime_ns"]) is not int
            or record["mtime_ns"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise ValueError("Track metadata-guard record facts are invalid.")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise ValueError("Track metadata-guard paths are not sorted and unique.")
    if (
        type(value["metadata_file_count"]) is not int
        or value["metadata_file_count"] != len(records)
        or value["metadata_root_sha256"] != canonical_json_sha256(records)
    ):
        raise ValueError("Track metadata-guard inventory digest differs.")


def _require_archive_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"metadata", "storage"}:
        raise ValueError("Track archive-guard field set differs.")
    _require_metadata_guard(value["metadata"])
    _require_storage_stats(value["storage"], label="archive guard")
    if (
        value["storage"]["metadata_file_count"]
        != value["metadata"]["metadata_file_count"]
    ):
        raise ValueError("Track archive-guard metadata accounting differs.")


def _selected_run_storage(
    archive: Path,
    *,
    role: str,
    run_path: str,
) -> dict[str, Any]:
    run_tree = archive.joinpath(*run_path.split("/"))
    return {
        "scope": _RUN_TREE_STORAGE_SCOPE,
        "role": role,
        "run_path": run_path,
        "filesystem_path": str(run_tree),
        **storage_stats(run_tree),
    }


def _require_selected_run_storage(
    value: object,
    *,
    archive: Path,
    role: str,
    run_path: str,
) -> None:
    fields = {
        "scope",
        "role",
        "run_path",
        "filesystem_path",
        *_STORAGE_STAT_FIELDS,
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Track selected-run storage field set differs.")
    if (
        value["scope"] != _RUN_TREE_STORAGE_SCOPE
        or value["role"] != role
        or value["run_path"] != run_path
        or value["filesystem_path"] != str(archive.joinpath(*run_path.split("/")))
    ):
        raise ValueError("Track selected-run storage identity binding differs.")
    _require_storage_stats(
        {field: value[field] for field in _STORAGE_STAT_FIELDS},
        label="selected run",
    )
    if value["file_count"] < 1 or value["metadata_file_count"] < 1:
        raise ValueError("Track selected-run storage inventory is empty.")


def _collect_dependency_refs(value: Any, *, key: str | None = None) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            result.update(_collect_dependency_refs(child, key=str(child_key)))
    elif isinstance(value, list):
        for child in value:
            result.update(_collect_dependency_refs(child, key=key))
    elif (
        type(value) is str
        and value.startswith("/")
        and key is not None
        and (key == "ref" or key.endswith("_ref") or key == "array_ref")
    ):
        relative = value[1:].split("#", 1)[0].split("@", 1)[0]
        if relative:
            result.add(relative)
    return result


def _source_dependency_paths(source: Any) -> tuple[str, ...]:
    manifest = source.attrs.get(track_contract.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        raise ValueError("Source lacks its canonical full-motion manifest.")
    source_prefix = f"{source.path}/"
    return tuple(
        sorted(
            path
            for path in _collect_dependency_refs(manifest)
            if path != source.path and not path.startswith(source_prefix)
        )
    )


def _source_run_path(name: str) -> str:
    return f"{PARENT_PATH}/{RUN_TYPE}/{name}"


def _source_dimensions(source: Any) -> dict[str, Any]:
    track_ids = np.asarray(source["track_ids"][:])
    if track_ids.dtype != np.dtype("int32") or track_ids.ndim != 1:
        raise ValueError("Source track_ids must be exact int32[N].")
    normalized = [int(value) for value in track_ids]
    if not normalized or normalized != sorted(set(normalized)):
        raise ValueError("Source track IDs must be nonempty and strictly increasing.")
    arena_present = source.get("track_arena_ids") is not None
    physical_presence: list[bool] = []
    row_counts: dict[str, int] = {}
    second_counts: dict[str, int] = {}
    for track_id in normalized:
        track = source[f"tracks/id_{track_id}"]
        row_counts[str(track_id)] = int(track["track_sample_key"].shape[0])
        second_counts[str(track_id)] = int(track["second_indices"].shape[0])
        physical_presence.append(track.get("positions_mm") is not None)
    if len(set(physical_presence)) != 1:
        raise ValueError("Physical track surfaces must be all present or all absent.")
    return {
        "track_ids": normalized,
        "track_count": len(normalized),
        "track_row_counts": row_counts,
        "track_second_counts": second_counts,
        "arena_inventory_present": arena_present,
        "physical_surfaces_present": physical_presence[0],
        "physical_bundle_array_count_per_track": len(
            TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
        ),
        "physical_bundle_benchmarked": PHYSICAL_BUNDLE_BENCHMARKED,
        "source_core_array_count_per_track": len(
            TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
        ),
        "candidate_core_array_count_per_track": 72,
        "source_inventory_formula": (
            "run_arrays=(1+arena_inventory_present); "
            "arrays_per_track=69+(35 if physical_surfaces_present else 0)"
        ),
        "candidate_inventory_formula": (
            "run_arrays=(1+arena_inventory_present); "
            "arrays_per_track=72+(35 if physical_surfaces_present else 0)"
        ),
    }


def build_source_logical_schema(source: Any) -> dict[str, Any]:
    """Replay the exact v1 source declaration inventory against live arrays."""

    dimensions = _source_dimensions(source)
    include_physical = bool(dimensions["physical_surfaces_present"])
    include_arena = bool(dimensions["arena_inventory_present"])
    if include_physical and not PHYSICAL_BUNDLE_BENCHMARKED:
        raise ValueError(
            "This benchmark version intentionally does not claim physical-bundle "
            "coverage; use a no-physical source."
        )
    declarations: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    run_declarations = tuple(
        declaration
        for declaration in TRACK_KINEMATICS_RUN_DECLARATIONS
        if declaration.relative_path != "track_arena_ids" or include_arena
    )
    for declaration in run_declarations:
        node = _array_at_path(source, declaration.relative_path)
        expected_shape = (int(dimensions["track_count"]),)
        if (
            np.dtype(node.dtype) != declaration.dtype
            or tuple(node.shape) != expected_shape
        ):
            raise ValueError(
                f"Source {declaration.relative_path} differs from its exact dtype/shape."
            )
        expected_paths.add(declaration.relative_path)
        declarations.append(
            {
                **declaration.as_manifest(),
                "resolved_shape": list(expected_shape),
                "resolved_dtype": str(np.dtype(node.dtype)),
            }
        )
    per_track = TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS + (
        TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS if include_physical else ()
    )
    for track_id in dimensions["track_ids"]:
        row_count = int(dimensions["track_row_counts"][str(track_id)])
        second_count = int(dimensions["track_second_counts"][str(track_id)])
        for declaration in per_track:
            path = f"tracks/id_{track_id}/{declaration.relative_path}"
            node = _array_at_path(source, path)
            expected_shape = tuple(
                (
                    row_count
                    if value == "n_track_samples"
                    else second_count if value == "n_track_seconds" else int(value)
                )
                for value in declaration.shape_template
            )
            if (
                np.dtype(node.dtype) != declaration.dtype
                or tuple(node.shape) != expected_shape
            ):
                raise ValueError(f"Source {path} differs from its exact dtype/shape.")
            expected_paths.add(path)
            manifest = declaration.bind_track(int(track_id)).as_manifest()
            declarations.append(
                {
                    **manifest,
                    "resolved_shape": list(expected_shape),
                    "resolved_dtype": str(np.dtype(node.dtype)),
                }
            )
    observed = set(_iter_array_paths(source))
    if observed != expected_paths:
        raise ValueError(
            "Source logical array inventory is not closed "
            f"(missing={sorted(expected_paths - observed)!r}, "
            f"unexpected={sorted(observed - expected_paths)!r})."
        )
    expected_groups = {
        "/".join(path.split("/")[:end])
        for path in expected_paths
        for end in range(1, len(path.split("/")))
    }
    observed_groups = set(_iter_group_paths(source))
    if observed_groups != expected_groups:
        raise ValueError("Source logical group inventory is not closed.")
    payload = {
        "schema_id": "palette.track_kinematics.v1_executable_source_schema",
        "schema_version": 1,
        "source_schema_id": "analysis.track_kinematics_runs",
        "source_schema_version": 1,
        "dimensions": dimensions,
        "array_count": len(declarations),
        "arrays": declarations,
    }
    return _strict_envelope(
        "palette.track_kinematics.v1_executable_source_schema_envelope",
        payload,
    )


def _projected_source_array(source: Any, candidate_path: str) -> tuple[Any, str | None]:
    source_path, field = _source_projection_identity(candidate_path)
    return _array_at_path(source, source_path), field


def _source_projection_identity(candidate_path: str) -> tuple[str, str | None]:
    components = candidate_path.split("/")
    if len(components) >= 4 and components[0] == "tracks":
        relative = "/".join(components[2:])
        mapped = _SOURCE_STRUCTURED_FIELDS.get(relative)
        if mapped is not None:
            return "/".join(components[:2] + [mapped[0]]), mapped[1]
    return candidate_path, None


def _selector_snapshot(root: Any) -> dict[str, Any]:
    parent = root[PARENT_PATH]
    offline = parent[RUN_TYPE]
    return {
        PARENT_PATH: {
            name: parent.attrs.get(name)
            for name in ("latest", "latest_complete", "latest_offline")
        },
        f"{PARENT_PATH}/{RUN_TYPE}": {
            name: offline.attrs.get(name) for name in ("latest", "latest_complete")
        },
    }


def _parent_attrs_snapshot(root: Any) -> dict[str, Any]:
    parent = root[PARENT_PATH]
    offline = parent[RUN_TYPE]
    return {
        PARENT_PATH: dict(parent.attrs),
        f"{PARENT_PATH}/{RUN_TYPE}": dict(offline.attrs),
    }


def _require_validation_receipt(
    value: object,
    *,
    expected_array_count: int,
    expected_hashes: Mapping[str, str],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "valid",
        "errors",
        "array_count",
        "logical_hashes",
    }:
        raise ValueError(f"{label} has an unexpected field set.")
    if (
        value["valid"] is not True
        or value["errors"] != []
        or value["array_count"] != expected_array_count
        or value["logical_hashes"] != expected_hashes
    ):
        raise ValueError(f"{label} does not bind the exact candidate payload.")


def _require_atomic_publication_receipt(
    candidate: Any,
    *,
    archive: Path,
    source_name: str,
    candidate_name: str,
    expected_hashes: Mapping[str, str],
    expected_parent_attrs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate publisher envelope semantics, not historical copy attestation.

    The nested physical-copy hashes describe a now-removed node-local source and
    cannot be replayed from the surviving candidate alone.  Pair/matrix evidence
    therefore hard-codes that receipt as opaque and never uses its counts, bytes,
    or hashes as benchmark authority.
    """

    raw = candidate.attrs.get("cluster_output_staging")
    if not isinstance(raw, Mapping) or set(raw) != _PUBLISH_RECEIPT_FIELDS:
        raise ValueError("Candidate atomic publication receipt field set differs.")
    receipt = dict(raw)
    candidate_path = _source_run_path(candidate_name)
    publisher = receipt.get("publisher_contract")
    publication_source = Path(str(receipt.get("publication_source_run_path")))
    target = Path(str(receipt.get("target_run_path")))
    candidate_components = tuple(candidate_path.split("/"))
    if publisher != {
        "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
        "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    }:
        raise ValueError("Candidate atomic publisher identity differs.")
    if (
        receipt.get("schema_id") != PUBLISH_SCHEMA_ID
        or receipt.get("policy") != _PUBLISH_POLICY
        or receipt.get("serialization_policy") != SERIALIZATION_POLICY
        or receipt.get("rollback_policy") != _ROLLBACK_POLICY
        or receipt.get("profile_id") != "published_http_v1"
        or receipt.get("source_run") != source_name
        or receipt.get("source_projection_hashes") != expected_hashes
        or not publication_source.is_absolute()
        or tuple(publication_source.parts[-len(candidate_components) :])
        != candidate_components
        or publication_source.is_relative_to(archive)
        or target != archive.joinpath(*candidate_path.split("/"))
        or receipt.get("publication_owner_attr") != ATOMIC_PUBLICATION_OWNER_ATTR
        or receipt.get("publication_owner_uuid")
        != candidate.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
        or receipt.get("source_zarr") != str(archive)
    ):
        raise ValueError("Candidate atomic publication identity/policy differs.")
    if (
        receipt.get("failed_public_child_policy")
        != "retain_owner_bound_selector_ineligible_tombstone"
        or receipt.get("hidden_temporary_policy")
        != "same_parent_hidden_sibling_then_os_replace"
    ):
        raise ValueError("Candidate atomic failure/temporary policy differs.")
    for field in ("copy_duration_seconds", "materialization_seconds"):
        _require_nonnegative(receipt.get(field), label=f"publication {field}")
    if (
        type(receipt.get("published_at_utc")) is not str
        or not receipt["published_at_utc"]
    ):
        raise ValueError("Candidate publication timestamp is invalid.")
    if type(receipt.get("host")) is not str or not receipt["host"]:
        raise ValueError("Candidate publication host is invalid.")
    if (
        receipt.get("lsb_jobid") is not None
        and type(receipt.get("lsb_jobid")) is not str
    ):
        raise ValueError("Candidate publication LSF identity is invalid.")
    physical = receipt.get("physical_copy")
    if not isinstance(physical, Mapping) or set(physical) != _PHYSICAL_COPY_FIELDS:
        raise ValueError("Candidate atomic physical-copy receipt differs.")
    backend = physical.get("backend")
    if (
        type(backend) is not str
        or _COPY_BACKEND_VERIFICATION.get(backend) != physical.get("verification")
        or not _is_sha256(physical.get("content_sha256"))
        or not _is_sha256(physical.get("inventory_sha256"))
        or type(physical.get("file_count")) is not int
        or physical["file_count"] < 1
        or type(physical.get("physical_bytes")) is not int
        or physical["physical_bytes"] < 1
    ):
        raise ValueError("Candidate atomic physical-copy evidence is invalid.")
    expected_array_count = len(expected_hashes)
    if receipt.get("local_direct_consolidated_array_count") != expected_array_count:
        raise ValueError("Candidate local metadata receipt array count differs.")
    for field in (
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
    ):
        _require_validation_receipt(
            receipt.get(field),
            expected_array_count=expected_array_count,
            expected_hashes=expected_hashes,
            label=f"Candidate publication {field}",
        )
    before = receipt.get("parent_attrs_before")
    after = receipt.get("parent_attrs_after")
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or dict(before) != dict(after)
        or dict(before) != dict(expected_parent_attrs)
    ):
        raise ValueError(
            "Candidate publication parent snapshot differs from the live selected "
            "source authority."
        )
    expected_parent_fields = {PARENT_PATH, f"{PARENT_PATH}/{RUN_TYPE}"}
    if set(before) != expected_parent_fields:
        raise ValueError("Candidate publication parent snapshot scope differs.")
    return receipt


def validate_pair(
    archive: str | Path,
    *,
    source_run_name: str,
    candidate_run_name: str,
) -> dict[str, Any]:
    """Validate authority, layout, equality, metadata, and nonpromotion."""

    archive_path = _safe_archive_path(archive)
    source_name = _safe_name(source_run_name, label="source run")
    candidate_name = _safe_name(candidate_run_name, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate names must differ.")
    source_path = _source_run_path(source_name)
    candidate_path = _source_run_path(candidate_name)
    _guard_archive_tree(archive_path, source_path, label="Selected source run")
    _guard_archive_tree(archive_path, candidate_path, label="Selected candidate run")
    _guard_relative_path(archive_path, "zarr.json")

    root = zarr.open_group(
        str(archive_path), mode="r", zarr_format=3, use_consolidated=True
    )
    source = root[source_path]
    candidate = root[candidate_path]
    dependencies = _source_dependency_paths(source)
    for dependency in dependencies:
        _guard_archive_tree(archive_path, dependency, label="Source dependency")

    if (
        source.attrs.get("schema_id") != "analysis.track_kinematics_runs"
        or source.attrs.get("schema_version") != 1
        or source.attrs.get("palette_run_completion_status") != "complete"
        or source.attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError("Source is not the maintained complete public v1 authority.")
    if (
        candidate.attrs.get("schema_id") != TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID
        or candidate.attrs.get("schema_version") != 2
        or candidate.attrs.get("palette_run_completion_status") != "complete"
        or candidate.attrs.get("stage_selector_eligible") is not False
        or candidate.attrs.get("storage_candidate_profile_promoted") is not False
    ):
        raise ValueError(
            "Candidate is not complete, explicit, ineligible, and unpromoted."
        )
    source_schema = build_source_logical_schema(source)
    source_dimensions = _require_envelope(
        source_schema,
        schema_id="palette.track_kinematics.v1_executable_source_schema_envelope",
    )["dimensions"]
    if (
        source_dimensions["physical_surfaces_present"] is not False
        or source_dimensions["physical_bundle_benchmarked"] is not False
    ):
        raise ValueError("This benchmark must remain explicit no-physical evidence.")
    track_ids = tuple(int(value) for value in source_dimensions["track_ids"])
    public_reads: list[dict[str, Any]] = []
    for track_id in track_ids:
        loaded = load_track_kinematics_track(
            root,
            run_name=source_name,
            scope=RUN_TYPE,
            track_id=track_id,
        )
        if loaded.authority_status != "verified_canonical_track_motion_v1":
            raise ValueError("Maintained source reader did not mint public authority.")
        public_reads.append(
            {
                "track_id": track_id,
                "row_count": int(loaded.frame_indices.shape[0]),
                "motion_manifest_sha256": loaded.motion_manifest_sha256,
                "authority_status": loaded.authority_status,
            }
        )

    declarations = build_flat_candidate_declarations(source)
    candidate_validation = validate_flat_candidate(candidate, source_group=source)
    if not candidate_validation["valid"]:
        raise ValueError(
            f"Candidate validation failed: {candidate_validation['errors']!r}."
        )
    expected_hashes = source_flat_projection_hashes(source, declarations)
    if candidate_validation["logical_hashes"] != expected_hashes:
        raise ValueError("Complete candidate decoded equality differs from source.")
    replayed_receipt = build_flat_candidate_storage_receipt(
        source,
        profile=get_storage_profile("published_http_v1"),
    ).as_manifest()
    if candidate.attrs.get("analysis_storage_plan_receipt") != replayed_receipt:
        raise ValueError(
            "Candidate persisted byte plan differs from executable replay."
        )
    manifest = candidate.attrs.get(TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        raise ValueError("Candidate logical manifest is absent.")
    selectors = _selector_snapshot(root)
    expected_selected = {
        "latest": f"offline/{source_name}",
        "latest_complete": f"offline/{source_name}",
        "latest_offline": source_name,
    }
    if selectors[PARENT_PATH] != expected_selected or selectors[
        f"{PARENT_PATH}/{RUN_TYPE}"
    ] != {"latest": source_name, "latest_complete": source_name}:
        raise ValueError(
            "Candidate altered or displaced the selected source authority."
        )
    parent_attrs = _parent_attrs_snapshot(root)
    publication = _require_atomic_publication_receipt(
        candidate,
        archive=archive_path,
        source_name=source_name,
        candidate_name=candidate_name,
        expected_hashes=expected_hashes,
        expected_parent_attrs=parent_attrs,
    )
    source_metadata = validate_direct_consolidated_subtree(
        archive_path, subtree_path=source_path
    ).to_json()
    candidate_metadata = validate_direct_consolidated_subtree(
        archive_path, subtree_path=candidate_path
    ).to_json()
    if (
        source_metadata["schema_id"] != "palette.zarr.metadata_equivalence"
        or source_metadata["schema_version"] != 1
        or candidate_metadata["schema_id"] != "palette.zarr.metadata_equivalence"
        or candidate_metadata["schema_version"] != 1
        or source_metadata["array_count"]
        != _require_envelope(
            source_schema,
            schema_id=("palette.track_kinematics.v1_executable_source_schema_envelope"),
        )["array_count"]
        or candidate_metadata["array_count"] != len(declarations)
    ):
        raise ValueError("Direct/consolidated metadata receipt identity differs.")
    return {
        "valid": True,
        "source_run_path": source_path,
        "candidate_run_path": candidate_path,
        "source_consumer": {
            "public_consumer_implemented": True,
            "consumer": "load_track_kinematics_track",
            "authority": "verified_canonical_track_motion_v1",
            "tracks": public_reads,
        },
        "candidate_consumer": {
            "public_consumer_implemented": False,
            "diagnostic_consumer_implemented": True,
            "consumer": "benchmark_track_kinematics_v2_candidate.validate_pair",
            "authority": "diagnostic_only_explicit_candidate",
        },
        "source_schema": source_schema,
        "candidate_manifest": dict(manifest),
        "candidate_storage_receipt": replayed_receipt,
        "logical_hashes": expected_hashes,
        "complete_decoded_equality": True,
        "dependencies": list(dependencies),
        "metadata_equivalence": {
            "source": source_metadata,
            "candidate": candidate_metadata,
        },
        "publication_receipt_sha256": canonical_json_sha256(publication),
        "publication_physical_copy_receipt_role": _PUBLISH_PHYSICAL_RECEIPT_ROLE,
        "publication_physical_copy_replayed": False,
        "selectors": selectors,
        "physical_surfaces_present": False,
        "physical_bundle_benchmarked": False,
        "profile_promoted": False,
        "selector_eligible_candidate": False,
    }


def _read_slice(array: Any, *, field: str | None, start: int, stop: int) -> np.ndarray:
    trailing = (slice(None),) * (int(array.ndim) - 1)
    values = np.asarray(array[(slice(start, stop), *trailing)])
    if field is not None:
        values = values[field]
    return np.ascontiguousarray(values)


def _hash_values(digest: Any, values: np.ndarray) -> None:
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(values.tobytes(order="C"))


def _primary_logical_projection(
    arrays: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "path",
        "declared_access_pattern",
        "read_mode",
        "dtype",
        "shape",
        "spans",
        "read_operations",
        "logical_decoded_bytes",
        "payload_sha256",
    )
    return [{field: item[field] for field in fields} for item in arrays]


def _primary_projection_digest(arrays: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(_primary_logical_projection(arrays))


def _deterministic_starts(
    *, path: str, rows: int, window_rows: int, count: int, seed: int
) -> tuple[int, ...]:
    if rows <= window_rows:
        return (0,)
    upper = rows - window_rows
    path_seed = int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng((int(seed) ^ path_seed) & ((1 << 63) - 1))
    anchors = {0, upper, upper // 2}
    while len(anchors) < min(count, upper + 1):
        anchors.add(int(rng.integers(0, upper + 1)))
    return tuple(sorted(anchors)[:count])


def run_primary_access(
    group: Any,
    *,
    role: str,
    declarations: Sequence[Any],
    window_rows: int,
    windows_per_array: int,
    seed: int,
) -> dict[str, Any]:
    if role not in {"source", "candidate"}:
        raise ValueError("Primary workload role is unsupported.")
    if window_rows < 1 or windows_per_array < 1:
        raise ValueError("Primary workload windows must be positive.")
    records: list[dict[str, Any]] = []
    decoded_bytes = 0
    read_operations = 0
    for declaration in sorted(declarations, key=lambda item: item.path):
        path = declaration.path
        if role == "source":
            array, field = _projected_source_array(group, path)
            read_path, expected_field = _source_projection_identity(path)
            if field != expected_field:
                raise RuntimeError(
                    "Source projection field resolution is inconsistent."
                )
        else:
            array, field = _array_at_path(group, path), None
            read_path = path
        dtype = np.dtype(array.dtype if field is None else array.dtype.fields[field][0])
        shape = tuple(int(value) for value in array.shape)
        array_digest = hashlib.sha256()
        spans: list[list[int]] = []
        array_decoded_bytes = 0
        array_read_operations = 0
        if not shape:
            values = np.asarray(array[...])
            if field is not None:
                values = values[field]
            values = np.ascontiguousarray(values)
            read_mode = "scalar"
            _hash_values(array_digest, values)
            array_decoded_bytes += int(values.nbytes)
            array_read_operations += 1
        elif declaration.access_pattern.value == "eager":
            stop = int(shape[0])
            values = _read_slice(
                array,
                field=field,
                start=0,
                stop=stop,
            )
            spans.append([0, stop])
            read_mode = "eager_full"
            _hash_values(array_digest, values)
            array_decoded_bytes += int(values.nbytes)
            array_read_operations += 1
        else:
            effective = min(window_rows, max(1, int(shape[0])))
            starts = _deterministic_starts(
                path=path,
                rows=int(shape[0]),
                window_rows=effective,
                count=windows_per_array,
                seed=seed,
            )
            read_mode = "deterministic_first_axis_windows"
            for start in starts:
                stop = min(start + effective, int(shape[0]))
                values = _read_slice(
                    array,
                    field=field,
                    start=start,
                    stop=stop,
                )
                spans.append([int(start), int(stop)])
                _hash_values(array_digest, values)
                array_decoded_bytes += int(values.nbytes)
                array_read_operations += 1
        decoded_bytes += array_decoded_bytes
        read_operations += array_read_operations
        records.append(
            {
                "path": path,
                "read_path": read_path,
                "projected_field": field,
                "declared_access_pattern": declaration.access_pattern.value,
                "read_mode": read_mode,
                "dtype": str(dtype),
                "shape": list(shape),
                "spans": spans,
                "read_operations": array_read_operations,
                "logical_decoded_bytes": array_decoded_bytes,
                "payload_sha256": array_digest.hexdigest(),
            }
        )
    return {
        "workload_id": "all_declared_arrays_eager_or_deterministic_windows_v1",
        "role": role,
        "window_rows": int(window_rows),
        "windows_per_array": int(windows_per_array),
        "seed": int(seed),
        "array_count": len(records),
        "read_operations": read_operations,
        "logical_decoded_bytes": decoded_bytes,
        "payload_sha256": _primary_projection_digest(records),
        "arrays": records,
    }


def _trial_environment(*, archive: Path) -> dict[str, Any]:
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
        "archive_device": int(archive.stat().st_dev),
        "cache_state": _CACHE_STATE,
        "thread_environment": {
            name: os.environ.get(name) for name in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def _require_trial_environment(value: object, *, archive: Path) -> None:
    fields = {
        "hostname",
        "system",
        "release",
        "machine",
        "python",
        "numpy",
        "zarr",
        "palette_commit",
        "palette_dirty",
        "archive_device",
        "cache_state",
        "thread_environment",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Track trial environment field set differs.")
    for field in (
        "hostname",
        "system",
        "release",
        "machine",
        "python",
        "numpy",
        "zarr",
        "palette_commit",
    ):
        if type(value[field]) is not str or not value[field]:
            raise ValueError(f"Track trial environment {field} is invalid.")
    if (
        type(value["palette_dirty"]) is not bool
        or type(value["archive_device"]) is not int
        or value["archive_device"] != int(archive.stat().st_dev)
        or value["cache_state"] != _CACHE_STATE
    ):
        raise ValueError("Track trial environment identity differs.")
    thread_environment = value["thread_environment"]
    if (
        not isinstance(thread_environment, Mapping)
        or set(thread_environment) != set(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
        or any(
            observed is not None and type(observed) is not str
            for observed in thread_environment.values()
        )
    ):
        raise ValueError("Track trial thread environment is malformed.")


def run_trial(
    archive: str | Path,
    *,
    source_run_name: str,
    candidate_run_name: str,
    role: str,
    repetition_index: int,
    order_position: int,
    seed: int,
    driver_process_id: int,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    windows_per_array: int = DEFAULT_WINDOWS_PER_ARRAY,
) -> dict[str, Any]:
    archive_path = _safe_archive_path(archive)
    source_name = _safe_name(source_run_name, label="source run")
    candidate_name = _safe_name(candidate_run_name, label="candidate run")
    if role not in {"source", "candidate"}:
        raise ValueError("Trial role is unsupported.")
    if (
        type(driver_process_id) is not int
        or driver_process_id <= 0
        or driver_process_id != os.getppid()
    ):
        raise ValueError("Trial driver PID must equal the live parent process ID.")
    if type(repetition_index) is not int or repetition_index < 0:
        raise ValueError("Trial repetition index is invalid.")
    if order_position not in {0, 1}:
        raise ValueError("Trial order position is invalid.")
    if (
        _trial_order(seed=seed, repetition_index=repetition_index)[order_position]
        != role
    ):
        raise ValueError("Trial role/order rotation binding differs.")

    source_path = _source_run_path(source_name)
    candidate_path = _source_run_path(candidate_name)
    selected_path = source_path if role == "source" else candidate_path
    _guard_archive_tree(archive_path, source_path, label="Selected source run")
    _guard_archive_tree(archive_path, candidate_path, label="Selected candidate run")
    _guard_relative_path(archive_path, "zarr.json")
    before = _archive_guard(archive_path)
    started = utc_now()
    validation, validation_timing = _measure(
        lambda: validate_pair(
            archive_path,
            source_run_name=source_name,
            candidate_run_name=candidate_name,
        )
    )
    root = zarr.open_group(
        str(archive_path), mode="r", zarr_format=3, use_consolidated=True
    )
    source = root[source_path]
    selected = source if role == "source" else root[candidate_path]
    declarations = build_flat_candidate_declarations(source)
    primary, primary_timing = _measure(
        lambda: run_primary_access(
            selected,
            role=role,
            declarations=declarations,
            window_rows=window_rows,
            windows_per_array=windows_per_array,
            seed=seed,
        )
    )
    full_hashes, full_scan_timing = _measure(
        lambda: (
            source_flat_projection_hashes(source, declarations)
            if role == "source"
            else flat_candidate_logical_hashes(selected, declarations)
        )
    )
    after = _archive_guard(archive_path)
    selectors_after = _selector_snapshot(root)
    if before != after:
        raise RuntimeError("Read-only trial changed archive metadata or storage facts.")
    if selectors_after != validation["selectors"]:
        raise RuntimeError("Read-only trial changed track selectors.")
    if full_hashes != validation["logical_hashes"]:
        raise RuntimeError("Trial full scan differs from validated source projection.")
    result_payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive_path),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "role": role,
        "run_name": source_name if role == "source" else candidate_name,
        "run_path": selected_path,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "seed": int(seed),
        "window_rows": int(window_rows),
        "windows_per_array": int(windows_per_array),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "environment": _trial_environment(archive=archive_path),
        "validation": {
            "receipt": validation,
            "timing": validation_timing,
        },
        "primary_access": {"receipt": primary, "timing": primary_timing},
        "full_scan": {
            "logical_hashes": full_hashes,
            "timing": full_scan_timing,
        },
        "storage": {
            "run_tree": _selected_run_storage(
                archive_path,
                role=role,
                run_path=selected_path,
            ),
            "archive_guard": before,
        },
        "runtime": {
            "process_id": os.getpid(),
            "parent_process_id": os.getppid(),
            "driver_process_id": driver_process_id,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "physical_io": {
            "available": False,
            "file_reads": None,
            "range_reads": None,
            "transferred_bytes": None,
            "reason": _PHYSICAL_IO_REASON,
        },
        "physical_surfaces_present": False,
        "physical_bundle_benchmarked": False,
        "public_consumer_implemented": role == "source",
        "diagnostic_consumer_implemented": role == "candidate",
        "profile_promoted": False,
        "selector_eligible_candidate": False,
    }
    result = _strict_envelope(TRIAL_SCHEMA_ID, result_payload)
    require_trial_result(result)
    return result


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    if (
        type(seed) is not int
        or type(repetition_index) is not int
        or repetition_index < 0
    ):
        raise ValueError("Track benchmark rotation inputs must be exact integers.")
    return (
        ("candidate", "source")
        if (seed + repetition_index) % 2
        else ("source", "candidate")
    )


def _require_timing(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "wall_seconds",
        "cpu_seconds",
    }:
        raise ValueError(f"{label} timing field set differs.")
    _require_nonnegative(value["wall_seconds"], label=f"{label} wall time")
    _require_nonnegative(value["cpu_seconds"], label=f"{label} CPU time")


def _require_pair_validation(value: object) -> Mapping[str, Any]:
    fields = {
        "valid",
        "source_run_path",
        "candidate_run_path",
        "source_consumer",
        "candidate_consumer",
        "source_schema",
        "candidate_manifest",
        "candidate_storage_receipt",
        "logical_hashes",
        "complete_decoded_equality",
        "dependencies",
        "metadata_equivalence",
        "publication_receipt_sha256",
        "publication_physical_copy_receipt_role",
        "publication_physical_copy_replayed",
        "selectors",
        "physical_surfaces_present",
        "physical_bundle_benchmarked",
        "profile_promoted",
        "selector_eligible_candidate",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Track pair-validation field set differs.")
    if (
        value["valid"] is not True
        or value["complete_decoded_equality"] is not True
        or value["physical_surfaces_present"] is not False
        or value["physical_bundle_benchmarked"] is not False
        or value["profile_promoted"] is not False
        or value["selector_eligible_candidate"] is not False
        or value["publication_physical_copy_receipt_role"]
        != _PUBLISH_PHYSICAL_RECEIPT_ROLE
        or value["publication_physical_copy_replayed"] is not False
    ):
        raise ValueError(
            "Track pair-validation correctness/nonpromotion state differs."
        )
    source_consumer = value["source_consumer"]
    candidate_consumer = value["candidate_consumer"]
    if (
        not isinstance(source_consumer, Mapping)
        or set(source_consumer)
        != {
            "public_consumer_implemented",
            "consumer",
            "authority",
            "tracks",
        }
        or source_consumer["public_consumer_implemented"] is not True
        or source_consumer["consumer"] != "load_track_kinematics_track"
        or source_consumer["authority"] != "verified_canonical_track_motion_v1"
        or not isinstance(source_consumer["tracks"], list)
        or not source_consumer["tracks"]
    ):
        raise ValueError("Track source public-consumer receipt differs.")
    if candidate_consumer != {
        "public_consumer_implemented": False,
        "diagnostic_consumer_implemented": True,
        "consumer": "benchmark_track_kinematics_v2_candidate.validate_pair",
        "authority": "diagnostic_only_explicit_candidate",
    }:
        raise ValueError("Track candidate diagnostic-consumer boundary differs.")
    source_schema = value["source_schema"]
    schema_payload = _require_envelope(
        source_schema,
        schema_id="palette.track_kinematics.v1_executable_source_schema_envelope",
    )
    if set(schema_payload) != {
        "schema_id",
        "schema_version",
        "source_schema_id",
        "source_schema_version",
        "dimensions",
        "array_count",
        "arrays",
    }:
        raise ValueError("Track executable source-schema payload differs.")
    dimensions = schema_payload["dimensions"]
    if (
        not isinstance(dimensions, Mapping)
        or dimensions.get("physical_surfaces_present") is not False
        or dimensions.get("physical_bundle_benchmarked") is not False
        or dimensions.get("source_inventory_formula")
        != (
            "run_arrays=(1+arena_inventory_present); "
            "arrays_per_track=69+(35 if physical_surfaces_present else 0)"
        )
        or dimensions.get("candidate_inventory_formula")
        != (
            "run_arrays=(1+arena_inventory_present); "
            "arrays_per_track=72+(35 if physical_surfaces_present else 0)"
        )
        or schema_payload["array_count"]
        != (1 + int(bool(dimensions.get("arena_inventory_present"))))
        + 69 * int(dimensions.get("track_count", -1))
        or not isinstance(schema_payload["arrays"], list)
        or len(schema_payload["arrays"]) != schema_payload["array_count"]
    ):
        raise ValueError("Track executable no-physical inventory formula differs.")
    logical_hashes = value["logical_hashes"]
    manifest = value["candidate_manifest"]
    manifest_payload = (
        manifest.get("payload") if isinstance(manifest, Mapping) else None
    )
    candidate_arrays = (
        manifest_payload.get("arrays")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    if (
        not isinstance(logical_hashes, Mapping)
        or not isinstance(candidate_arrays, list)
        or set(logical_hashes)
        != {
            str(item.get("path"))
            for item in candidate_arrays
            if isinstance(item, Mapping)
        }
        or any(not _is_sha256(digest) for digest in logical_hashes.values())
    ):
        raise ValueError("Track pair logical-hash inventory differs.")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"payload", "payload_digest"}
        or not isinstance(manifest.get("payload"), Mapping)
        or manifest["payload_digest"] != canonical_json_sha256(manifest["payload"])
        or manifest["payload"].get("candidate_logical_hashes") != logical_hashes
        or manifest["payload"].get("source_projection_hashes") != logical_hashes
        or manifest["payload"].get("status") != "unpromoted_selector_ineligible"
    ):
        raise ValueError("Track candidate logical manifest differs.")
    storage_receipt = value["candidate_storage_receipt"]
    parsed = analysis_storage_plan_receipt_from_manifest(storage_receipt)
    if (
        parsed.as_manifest() != storage_receipt
        or parsed.profile.profile_id != "published_http_v1"
        or parsed.profile.as_manifest()
        != get_storage_profile("published_http_v1").as_manifest()
        or len(parsed.entries) != len(logical_hashes)
        or {entry.declaration.path for entry in parsed.entries} != set(logical_hashes)
    ):
        raise ValueError("Track candidate storage receipt is not exact/executable.")
    metadata = value["metadata_equivalence"]
    if not isinstance(metadata, Mapping) or set(metadata) != {"source", "candidate"}:
        raise ValueError("Track metadata-equivalence receipts differ.")
    for role in ("source", "candidate"):
        receipt = metadata[role]
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema_id") != "palette.zarr.metadata_equivalence"
            or receipt.get("schema_version") != 1
            or receipt.get("array_count")
            != (
                schema_payload["array_count"]
                if role == "source"
                else len(logical_hashes)
            )
            or not _is_sha256(receipt.get("declarations_sha256"))
        ):
            raise ValueError("Track metadata-equivalence receipt identity differs.")
    if not _is_sha256(value["publication_receipt_sha256"]):
        raise ValueError("Track publication receipt digest is invalid.")
    if not isinstance(value["dependencies"], list) or any(
        type(path) is not str or not path for path in value["dependencies"]
    ):
        raise ValueError("Track dependency inventory is invalid.")
    return value


def _require_primary_access(
    value: object,
    *,
    entries: Sequence[Any],
    role: str,
    expected_seed: int,
    expected_window_rows: int,
    expected_windows_per_array: int,
) -> list[dict[str, Any]]:
    fields = {
        "workload_id",
        "role",
        "window_rows",
        "windows_per_array",
        "seed",
        "array_count",
        "read_operations",
        "logical_decoded_bytes",
        "payload_sha256",
        "arrays",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Track primary-access receipt field set differs.")
    arrays = value["arrays"]
    expected_paths = [entry.declaration.path for entry in entries]
    if (
        value["workload_id"] != "all_declared_arrays_eager_or_deterministic_windows_v1"
        or value["role"] != role
        or value["window_rows"] != expected_window_rows
        or value["windows_per_array"] != expected_windows_per_array
        or value["seed"] != expected_seed
        or type(value["array_count"]) is not int
        or value["array_count"] != len(entries)
        or not isinstance(arrays, list)
        or len(arrays) != value["array_count"]
        or [item.get("path") for item in arrays if isinstance(item, Mapping)]
        != expected_paths
        or type(value["read_operations"]) is not int
        or type(value["logical_decoded_bytes"]) is not int
        or not _is_sha256(value["payload_sha256"])
    ):
        raise ValueError("Track primary-access receipt is invalid.")
    record_fields = {
        "path",
        "read_path",
        "projected_field",
        "declared_access_pattern",
        "read_mode",
        "dtype",
        "shape",
        "spans",
        "read_operations",
        "logical_decoded_bytes",
        "payload_sha256",
    }
    expected_operations = 0
    expected_decoded_bytes = 0
    for entry, record in zip(entries, arrays, strict=True):
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("Track primary-access array record field set differs.")
        path = entry.declaration.path
        shape = tuple(int(value) for value in entry.facts.shape)
        dtype = np.dtype(entry.facts.dtype)
        if role == "source":
            read_path, projected_field = _source_projection_identity(path)
        else:
            read_path, projected_field = path, None
        if not shape:
            read_mode = "scalar"
            spans: list[list[int]] = []
        elif entry.declaration.access_pattern.value == "eager":
            read_mode = "eager_full"
            spans = [[0, int(shape[0])]]
        else:
            read_mode = "deterministic_first_axis_windows"
            effective = min(expected_window_rows, max(1, int(shape[0])))
            starts = _deterministic_starts(
                path=path,
                rows=int(shape[0]),
                window_rows=effective,
                count=expected_windows_per_array,
                seed=expected_seed,
            )
            spans = [
                [int(start), min(int(start) + effective, int(shape[0]))]
                for start in starts
            ]
        operations = 1 if not shape else len(spans)
        if not shape:
            decoded_bytes = int(dtype.itemsize)
        else:
            record_items = math.prod(shape[1:])
            decoded_bytes = int(
                sum(stop - start for start, stop in spans)
                * record_items
                * dtype.itemsize
            )
        if record != {
            "path": path,
            "read_path": read_path,
            "projected_field": projected_field,
            "declared_access_pattern": entry.declaration.access_pattern.value,
            "read_mode": read_mode,
            "dtype": str(dtype),
            "shape": list(shape),
            "spans": spans,
            "read_operations": operations,
            "logical_decoded_bytes": decoded_bytes,
            "payload_sha256": record.get("payload_sha256"),
        } or not _is_sha256(record.get("payload_sha256")):
            raise ValueError(f"Track primary-access array record differs for {path!r}.")
        expected_operations += operations
        expected_decoded_bytes += decoded_bytes
    logical_projection = _primary_logical_projection(arrays)
    if (
        value["read_operations"] != expected_operations
        or value["logical_decoded_bytes"] != expected_decoded_bytes
        or value["payload_sha256"] != canonical_json_sha256(logical_projection)
    ):
        raise ValueError("Track primary-access aggregate differs from exact records.")
    return logical_projection


def require_trial_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    fields = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "source_run_name",
        "candidate_run_name",
        "role",
        "run_name",
        "run_path",
        "repetition_index",
        "order_position",
        "seed",
        "window_rows",
        "windows_per_array",
        "started_at_utc",
        "finished_at_utc",
        "environment",
        "validation",
        "primary_access",
        "full_scan",
        "storage",
        "runtime",
        "physical_io",
        "physical_surfaces_present",
        "physical_bundle_benchmarked",
        "public_consumer_implemented",
        "diagnostic_consumer_implemented",
        "profile_promoted",
        "selector_eligible_candidate",
    }
    if set(payload) != fields:
        raise ValueError("Track trial field set differs.")
    if type(payload["archive_path"]) is not str:
        raise ValueError("Track trial archive path must be one exact string.")
    archive = _safe_archive_path(payload["archive_path"])
    started = _require_utc_timestamp(
        payload["started_at_utc"], label="Track trial start"
    )
    finished = _require_utc_timestamp(
        payload["finished_at_utc"], label="Track trial finish"
    )
    if finished < started:
        raise ValueError("Track trial finish precedes its start.")
    role = payload["role"]
    source_name = _safe_name(payload["source_run_name"], label="source run")
    candidate_name = _safe_name(payload["candidate_run_name"], label="candidate run")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or role not in {"source", "candidate"}
        or type(payload["repetition_index"]) is not int
        or payload["repetition_index"] < 0
        or payload["order_position"] not in {0, 1}
        or type(payload["seed"]) is not int
        or type(payload["window_rows"]) is not int
        or payload["window_rows"] < 1
        or type(payload["windows_per_array"]) is not int
        or payload["windows_per_array"] < 1
        or _trial_order(
            seed=payload["seed"], repetition_index=payload["repetition_index"]
        )[payload["order_position"]]
        != role
        or payload["run_name"] != (source_name if role == "source" else candidate_name)
        or payload["run_path"] != _source_run_path(payload["run_name"])
    ):
        raise ValueError("Track trial role/order/identity binding differs.")
    validation = payload["validation"]
    if not isinstance(validation, Mapping) or set(validation) != {"receipt", "timing"}:
        raise ValueError("Track trial pair-validation wrapper differs.")
    pair = _require_pair_validation(validation["receipt"])
    _require_timing(validation["timing"], label="pair validation")
    if pair["source_run_path"] != _source_run_path(source_name) or pair[
        "candidate_run_path"
    ] != _source_run_path(candidate_name):
        raise ValueError("Track trial pair/run binding differs.")
    primary = payload["primary_access"]
    if not isinstance(primary, Mapping) or set(primary) != {"receipt", "timing"}:
        raise ValueError("Track trial primary-access wrapper differs.")
    storage_receipt = analysis_storage_plan_receipt_from_manifest(
        pair["candidate_storage_receipt"]
    )
    _require_primary_access(
        primary["receipt"],
        entries=storage_receipt.entries,
        role=role,
        expected_seed=payload["seed"],
        expected_window_rows=payload["window_rows"],
        expected_windows_per_array=payload["windows_per_array"],
    )
    _require_timing(primary["timing"], label="primary access")
    full_scan = payload["full_scan"]
    if (
        not isinstance(full_scan, Mapping)
        or set(full_scan) != {"logical_hashes", "timing"}
        or full_scan["logical_hashes"] != pair["logical_hashes"]
    ):
        raise ValueError("Track trial complete scan differs from pair validation.")
    _require_timing(full_scan["timing"], label="full scan")
    _require_trial_environment(payload["environment"], archive=archive)
    storage = payload["storage"]
    if not isinstance(storage, Mapping) or set(storage) != {
        "run_tree",
        "archive_guard",
    }:
        raise ValueError("Track trial storage wrapper differs.")
    _require_selected_run_storage(
        storage["run_tree"],
        archive=archive,
        role=role,
        run_path=payload["run_path"],
    )
    _require_archive_guard(storage["archive_guard"])
    runtime = payload["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "process_id",
            "parent_process_id",
            "driver_process_id",
            "peak_rss_bytes",
        }
        or any(type(runtime[name]) is not int or runtime[name] <= 0 for name in runtime)
        or runtime["parent_process_id"] != runtime["driver_process_id"]
        or runtime["process_id"] == runtime["driver_process_id"]
    ):
        raise ValueError("Track trial fresh-child runtime binding differs.")
    physical_io = payload["physical_io"]
    if physical_io != {
        "available": False,
        "file_reads": None,
        "range_reads": None,
        "transferred_bytes": None,
        "reason": _PHYSICAL_IO_REASON,
    }:
        raise ValueError("Track trial fabricates physical I/O evidence.")
    if (
        payload["physical_surfaces_present"] is not False
        or payload["physical_bundle_benchmarked"] is not False
        or payload["public_consumer_implemented"] is not (role == "source")
        or payload["diagnostic_consumer_implemented"] is not (role == "candidate")
        or payload["profile_promoted"] is not False
        or payload["selector_eligible_candidate"] is not False
    ):
        raise ValueError("Track trial scope/nonpromotion flags differ.")


def _matrix_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("source", "candidate"):
        selected = [
            trial["payload"] for trial in trials if trial["payload"]["role"] == role
        ]
        result[role] = {
            "trial_count": len(selected),
            "median_primary_wall_seconds": statistics.median(
                item["primary_access"]["timing"]["wall_seconds"] for item in selected
            ),
            "median_full_scan_wall_seconds": statistics.median(
                item["full_scan"]["timing"]["wall_seconds"] for item in selected
            ),
            "median_validation_wall_seconds": statistics.median(
                item["validation"]["timing"]["wall_seconds"] for item in selected
            ),
            "median_peak_rss_bytes": statistics.median(
                item["runtime"]["peak_rss_bytes"] for item in selected
            ),
            "median_payload_file_count": statistics.median(
                item["storage"]["run_tree"]["payload_file_count"] for item in selected
            ),
            "median_apparent_bytes": statistics.median(
                item["storage"]["run_tree"]["apparent_bytes"] for item in selected
            ),
        }
    return result


def require_matrix_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    fields = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "source_run_name",
        "candidate_run_name",
        "repetitions",
        "seed",
        "window_rows",
        "windows_per_array",
        "driver_process_id",
        "started_at_utc",
        "finished_at_utc",
        "pair_validation",
        "trials",
        "summary",
        "archive_guard_before",
        "archive_guard_after",
        "archive_read_only",
        "selectors_unchanged",
        "balanced_fresh_process_matrix_complete",
        "physical_io_availability",
        "physical_surfaces_present",
        "physical_bundle_benchmarked",
        "public_source_consumer_implemented",
        "public_candidate_consumer_implemented",
        "diagnostic_candidate_consumer_implemented",
        "profile_promoted",
        "selector_eligible_candidate",
    }
    if set(payload) != fields:
        raise ValueError("Track matrix field set differs.")
    if type(payload["archive_path"]) is not str:
        raise ValueError("Track matrix archive path must be one exact string.")
    archive = _safe_archive_path(payload["archive_path"])
    started = _require_utc_timestamp(
        payload["started_at_utc"], label="Track matrix start"
    )
    finished = _require_utc_timestamp(
        payload["finished_at_utc"], label="Track matrix finish"
    )
    if finished < started:
        raise ValueError("Track matrix finish precedes its start.")
    _require_archive_guard(payload["archive_guard_before"])
    _require_archive_guard(payload["archive_guard_after"])
    pair = _require_pair_validation(payload["pair_validation"])
    source_name = _safe_name(payload["source_run_name"], label="source run")
    candidate_name = _safe_name(payload["candidate_run_name"], label="candidate run")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or type(payload["repetitions"]) is not int
        or payload["repetitions"] < 1
        or type(payload["seed"]) is not int
        or type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] <= 0
        or type(payload["window_rows"]) is not int
        or payload["window_rows"] < 1
        or type(payload["windows_per_array"]) is not int
        or payload["windows_per_array"] < 1
        or pair["source_run_path"] != _source_run_path(source_name)
        or pair["candidate_run_path"] != _source_run_path(candidate_name)
    ):
        raise ValueError("Track matrix invocation binding differs.")
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != 2 * payload["repetitions"]:
        raise ValueError("Track matrix trial count differs.")
    process_ids: list[int] = []
    observed_environment: Mapping[str, Any] | None = None
    logical_by_repetition: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for index, trial in enumerate(trials):
        require_trial_result(trial)
        child = trial["payload"]
        repetition = index // 2
        position = index % 2
        if (
            child["repetition_index"] != repetition
            or child["order_position"] != position
            or child["role"]
            != _trial_order(seed=payload["seed"], repetition_index=repetition)[position]
            or child["runtime"]["driver_process_id"] != payload["driver_process_id"]
            or child["seed"] != payload["seed"]
            or child["window_rows"] != payload["window_rows"]
            or child["windows_per_array"] != payload["windows_per_array"]
            or child["validation"]["receipt"] != pair
            or child["archive_path"] != payload["archive_path"]
            or child["source_run_name"] != payload["source_run_name"]
            or child["candidate_run_name"] != payload["candidate_run_name"]
            or child["storage"]["archive_guard"] != payload["archive_guard_before"]
            or child["environment"]["thread_environment"]
            != dict(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
        ):
            raise ValueError("Track matrix trial ordering/pair binding differs.")
        logical_by_repetition.setdefault(repetition, {})[child["role"]] = (
            _primary_logical_projection(child["primary_access"]["receipt"]["arrays"])
        )
        if observed_environment is None:
            observed_environment = child["environment"]
        elif child["environment"] != observed_environment:
            raise ValueError("Track matrix child environments differ.")
        process_ids.append(child["runtime"]["process_id"])
    if any(
        set(by_role) != {"source", "candidate"}
        or by_role["source"] != by_role["candidate"]
        for by_role in logical_by_repetition.values()
    ):
        raise ValueError(
            "Track matrix matched source/candidate primary logical workloads differ."
        )
    if (
        len(set(process_ids)) != len(process_ids)
        or payload["driver_process_id"] in process_ids
    ):
        raise ValueError("Track matrix did not use distinct fresh child processes.")
    if payload["summary"] != _matrix_summary(trials):
        raise ValueError("Track matrix summary differs from executable aggregation.")
    if (
        payload["archive_guard_before"] != payload["archive_guard_after"]
        or payload["archive_read_only"] is not True
        or payload["selectors_unchanged"] is not True
        or payload["balanced_fresh_process_matrix_complete"] is not True
        or payload["physical_io_availability"] != _PHYSICAL_IO_REASON
        or payload["physical_surfaces_present"] is not False
        or payload["physical_bundle_benchmarked"] is not False
        or payload["public_source_consumer_implemented"] is not True
        or payload["public_candidate_consumer_implemented"] is not False
        or payload["diagnostic_candidate_consumer_implemented"] is not True
        or payload["profile_promoted"] is not False
        or payload["selector_eligible_candidate"] is not False
    ):
        raise ValueError("Track matrix safety/scope/nonpromotion state differs.")
    live_guard = _archive_guard(archive)
    if (
        payload["archive_guard_before"] != live_guard
        or payload["archive_guard_after"] != live_guard
    ):
        raise ValueError(
            "Track matrix archive/storage observations diverge from the live "
            "immutable archive."
        )
    live_run_storage = {
        "source": _selected_run_storage(
            archive,
            role="source",
            run_path=_source_run_path(source_name),
        ),
        "candidate": _selected_run_storage(
            archive,
            role="candidate",
            run_path=_source_run_path(candidate_name),
        ),
    }
    if any(
        trial["payload"]["storage"]["run_tree"]
        != live_run_storage[trial["payload"]["role"]]
        for trial in trials
    ):
        raise ValueError(
            "Track matrix selected-run storage observations diverge from the live "
            "immutable archive."
        )
    live_pair = validate_pair(
        archive,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
    )
    if live_pair != pair:
        raise ValueError(
            "Track matrix pair/metadata/publication receipts diverge from the live "
            "immutable archive."
        )
    root = zarr.open_group(str(archive), mode="r", zarr_format=3, use_consolidated=True)
    source = root[_source_run_path(source_name)]
    candidate = root[_source_run_path(candidate_name)]
    declarations = build_flat_candidate_declarations(source)
    live_primary = {
        "source": run_primary_access(
            source,
            role="source",
            declarations=declarations,
            window_rows=payload["window_rows"],
            windows_per_array=payload["windows_per_array"],
            seed=payload["seed"],
        ),
        "candidate": run_primary_access(
            candidate,
            role="candidate",
            declarations=declarations,
            window_rows=payload["window_rows"],
            windows_per_array=payload["windows_per_array"],
            seed=payload["seed"],
        ),
    }
    if any(
        trial["payload"]["primary_access"]["receipt"]
        != live_primary[trial["payload"]["role"]]
        for trial in trials
    ):
        raise ValueError(
            "Track matrix primary workload observations diverge from live replay."
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing existing benchmark evidence path: {path}.")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Benchmark evidence file is unsafe or absent: {path}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Benchmark evidence document must be one object.")
    return value


def run_benchmark_matrix(
    archive: str | Path,
    *,
    source_run_name: str,
    candidate_run_name: str,
    output: str | Path,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    windows_per_array: int = DEFAULT_WINDOWS_PER_ARRAY,
) -> dict[str, Any]:
    archive_path = _safe_archive_path(archive)
    source_name = _safe_name(source_run_name, label="source run")
    candidate_name = _safe_name(candidate_run_name, label="candidate run")
    output_path = _guard_output_path(archive_path, output)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Matrix repetitions must be positive.")
    if type(seed) is not int or window_rows < 1 or windows_per_array < 1:
        raise ValueError("Matrix seed/window parameters are invalid.")
    pair = validate_pair(
        archive_path,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
    )
    _require_pair_validation(pair)
    before = _archive_guard(archive_path)
    output_path.mkdir(parents=False, exist_ok=False)
    pair_path = output_path / "pair_validation.json"
    _write_json(pair_path, _strict_envelope(PAIR_SCHEMA_ID, pair))
    started = utc_now()
    driver_process_id = os.getpid()
    child_environment = os.environ.copy()
    child_environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    trials: list[Mapping[str, Any]] = []
    for repetition in range(repetitions):
        for position, role in enumerate(
            _trial_order(seed=seed, repetition_index=repetition)
        ):
            trial_path = output_path / f"trial_{repetition:03d}_{position}_{role}.json"
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_track_kinematics_v2_candidate",
                "trial",
                "--archive",
                str(archive_path),
                "--source-run",
                source_name,
                "--candidate-run",
                candidate_name,
                "--role",
                role,
                "--repetition-index",
                str(repetition),
                "--order-position",
                str(position),
                "--seed",
                str(seed),
                "--driver-process-id",
                str(driver_process_id),
                "--window-rows",
                str(window_rows),
                "--windows-per-array",
                str(windows_per_array),
                "--output",
                str(trial_path),
            ]
            subprocess.run(command, check=True, env=child_environment)
            trial = _load_json(trial_path)
            require_trial_result(trial)
            trials.append(trial)
    after = _archive_guard(archive_path)
    pair_after = validate_pair(
        archive_path,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
    )
    if pair_after != pair:
        raise RuntimeError("Track pair validation changed during the read-only matrix.")
    result = _strict_envelope(
        MATRIX_SCHEMA_ID,
        {
            "benchmark_id": BENCHMARK_ID,
            "family_id": FAMILY_ID,
            "archive_path": str(archive_path),
            "source_run_name": source_name,
            "candidate_run_name": candidate_name,
            "repetitions": repetitions,
            "seed": seed,
            "window_rows": window_rows,
            "windows_per_array": windows_per_array,
            "driver_process_id": driver_process_id,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "pair_validation": pair,
            "trials": trials,
            "summary": _matrix_summary(trials),
            "archive_guard_before": before,
            "archive_guard_after": after,
            "archive_read_only": before == after,
            "selectors_unchanged": pair_after["selectors"] == pair["selectors"],
            "balanced_fresh_process_matrix_complete": True,
            "physical_io_availability": _PHYSICAL_IO_REASON,
            "physical_surfaces_present": False,
            "physical_bundle_benchmarked": False,
            "public_source_consumer_implemented": True,
            "public_candidate_consumer_implemented": False,
            "diagnostic_candidate_consumer_implemented": True,
            "profile_promoted": False,
            "selector_eligible_candidate": False,
        },
    )
    require_matrix_result(result)
    _write_json(output_path / "matrix.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    trial = subparsers.add_parser("trial")
    matrix = subparsers.add_parser("matrix")
    for child in (validate, trial, matrix):
        child.add_argument("--archive", type=Path, required=True)
        child.add_argument("--source-run", required=True)
        child.add_argument("--candidate-run", required=True)
    validate.add_argument("--output", type=Path)
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--order-position", type=int, required=True)
    trial.add_argument("--seed", type=int, required=True)
    trial.add_argument("--driver-process-id", type=int, required=True)
    trial.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS)
    trial.add_argument(
        "--windows-per-array", type=int, default=DEFAULT_WINDOWS_PER_ARRAY
    )
    trial.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    matrix.add_argument("--seed", type=int, default=DEFAULT_SEED)
    matrix.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS)
    matrix.add_argument(
        "--windows-per-array", type=int, default=DEFAULT_WINDOWS_PER_ARRAY
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result = _strict_envelope(
            PAIR_SCHEMA_ID,
            validate_pair(
                args.archive,
                source_run_name=args.source_run,
                candidate_run_name=args.candidate_run,
            ),
        )
        if args.output is not None:
            _write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "trial":
        result = run_trial(
            args.archive,
            source_run_name=args.source_run,
            candidate_run_name=args.candidate_run,
            role=args.role,
            repetition_index=args.repetition_index,
            order_position=args.order_position,
            seed=args.seed,
            driver_process_id=args.driver_process_id,
            window_rows=args.window_rows,
            windows_per_array=args.windows_per_array,
        )
        _write_json(args.output, result)
        return 0
    run_benchmark_matrix(
        args.archive,
        source_run_name=args.source_run,
        candidate_run_name=args.candidate_run,
        output=args.output,
        repetitions=args.repetitions,
        seed=args.seed,
        window_rows=args.window_rows,
        windows_per_array=args.windows_per_array,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
