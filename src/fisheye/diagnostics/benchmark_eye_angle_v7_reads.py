"""Read-only source/candidate benchmark for maintained compact eye-angle v7.

The established run is selected through the maintained public reader.  The
byte-planned run is an explicitly named, immutable, selector-ineligible
candidate, so this diagnostic validates its exact payload contract before
calling the same compact-v7 table adapter.  No benchmark result authorizes
selection or profile promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zarr

from fisheye.analysis import eye_angle_io
from fisheye.analysis.eye_angle_analysis import (
    validate_eye_angle_persisted_contract_manifests,
)
from fisheye.analysis.eye_angle_schema import (
    EYE_ANGLE_ARRAY_SCHEMA_ATTR,
    EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
    EYE_ANGLE_RUN_PARENT,
    EyeAngleDimensions,
    build_eye_angle_array_declarations,
    collect_eye_angle_arrays,
    eye_angle_array_schema_manifest,
    eye_angle_dimensions_from_run_attrs,
    validate_eye_angle_compact_run,
)
from fisheye.analysis.eye_angle_storage import (
    EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
    EYE_ANGLE_STORAGE_CANDIDATE_ATTR,
    EYE_ANGLE_STORAGE_PLAN_ATTR,
    build_eye_angle_candidate_storage_plan,
    eye_angle_planned_fill_value,
    eye_angle_storage_entries_by_path,
    validate_eye_angle_candidate_storage,
    validate_eye_angle_direct_consolidated_storage,
)
from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
    ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    SERIALIZATION_POLICY,
)
from fisheye.analysis_workflows.materializers.eye_angles import (
    MATERIALIZATION_SCHEMA_ID,
    PUBLISH_SCHEMA_ID,
    SOURCE_REVISION_AUDIT_SCHEMA_ID,
    STAGING_SCHEMA_ID,
)
from fisheye.shared.zarr.array_factory import (
    array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.metadata_equivalence import (
    METADATA_EQUIVALENCE_SCHEMA_ID,
    METADATA_EQUIVALENCE_SCHEMA_VERSION,
    validate_direct_consolidated_subtree,
)

FAMILY_ID = "eye_angle_compact_v7"
BENCHMARK_ID = "eye_angle_v7_source_candidate_reads_v1"
WORKLOAD_ID = "eye_angle_v7_complete_arrays_and_tables_v1"
WORKLOAD_SCHEMA_ID = "palette.eye_angle.v7_read_workload"
TRIAL_SCHEMA_ID = "palette.eye_angle.v7_read_trial"
MATRIX_SCHEMA_ID = "palette.eye_angle.v7_read_matrix"
SCHEMA_VERSION = 1
DEFAULT_SEED = 37
DEFAULT_REPETITIONS = 5
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_PHYSICAL_IO_AVAILABILITY = "not_collected_requires_external_trace"
_CANDIDATE_PROMOTION_POLICY = (
    "immutable_named_candidate_no_pointer_or_registry_activation"
)
_ATOMIC_ROLLBACK_POLICY = (
    "retain_failed_public_tombstone_leave_unleased_parent_state_untouched"
)
_CANDIDATE_PUBLICATION_POLICY = (
    "exact_source_subset_staged_local_compute_then_shard_then_atomic_run_group_publish"
)
_RESERVED_ARRAY_ATTRIBUTES = frozenset(
    {
        "logical_schema_id",
        "logical_schema_version",
        "storage_policy_version",
        "storage_profile_id",
        "codec_profile_id",
        "access_pattern",
        "write_mode",
    }
)
_SOURCE_IDENTITY_ATTRS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "palette_run_completion_status",
    "source_fingerprint",
    "source_lineage_hash",
    "lineage_hash",
    "fingerprint_status",
    "git_commit",
    "git_dirty",
    "created_at_utc",
    "completed_at_utc",
)


def _strict_json_copy(value: object) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Eye-angle evidence is not strict JSON: {exc}") from exc
    return json.loads(
        encoded,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON token {token}")
        ),
    )


def _envelope(schema_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _strict_json_copy(dict(payload))
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
        raise ValueError("Eye-angle evidence envelope field set is not exact.")
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Eye-angle evidence schema identity is unsupported.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Eye-angle evidence payload must be one object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Eye-angle evidence payload digest mismatch.")
    _strict_json_copy(value)
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_run_name(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string.")
    if (
        not value
        or value != value.strip()
        or value.lower() in _ALIASES
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return value


def _safe_archive(value: Path | str) -> Path:
    raw = Path(value).expanduser()
    if any(path.exists() and path.is_symlink() for path in (raw, *raw.parents)):
        raise ValueError("Eye-angle benchmark archive must not be a symlink.")
    archive = raw.resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}.")
    return archive


def _run_path(name: str) -> str:
    return f"{EYE_ANGLE_RUN_PARENT}/{name}"


def _require_archive_directory(
    archive: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        type(relative_path) is not str
        or not relative_path
        or relative_path.startswith("/")
        or ".." in pure.parts
        or any(part in {"", "."} for part in pure.parts)
    ):
        raise ValueError(f"{label} path is not one canonical archive-relative path.")
    node = archive
    for component in pure.parts:
        node = node / component
        try:
            node.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} path is absent: {node}.") from exc
        if node.is_symlink():
            raise ValueError(f"{label} path component is a forbidden symlink: {node}.")
        if not node.is_dir():
            raise ValueError(f"{label} path component is not a directory: {node}.")
    try:
        node.resolve(strict=True).relative_to(archive)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the archive.") from exc
    return node


def _require_nonsymlink_tree(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be one nonsymlink directory.")
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"{label} contains a forbidden symlink: {child}.")


def _safe_run_directory(archive: Path, *, name: str) -> Path:
    path = _require_archive_directory(
        archive,
        _run_path(name),
        label="Selected eye-angle run",
    )
    _require_nonsymlink_tree(path, label="Selected eye-angle run")
    return path


def _safe_output(value: Path | str, *, archive: Path) -> Path:
    raw = Path(value).expanduser()
    if any(path.exists() and path.is_symlink() for path in (raw, *raw.parents)):
        raise ValueError("Benchmark output ancestry must not contain a symlink.")
    output = raw.resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}.")
    if (
        output == archive
        or output.is_relative_to(archive)
        or archive.is_relative_to(output)
        or output in {Path("/"), Path.home().resolve()}
        or not any("benchmark" in part.lower() for part in output.parts)
    ):
        raise ValueError("Benchmark output must be a new disjoint benchmark path.")
    return output


def _safe_trial_output(value: Path | str, *, root: Path) -> Path:
    raw = Path(value).expanduser()
    if any(path.exists() and path.is_symlink() for path in (raw, *raw.parents)):
        raise ValueError("Trial output ancestry must not contain a symlink.")
    output = raw.resolve()
    benchmark_root = root.resolve()
    if (
        not benchmark_root.is_dir()
        or benchmark_root.is_symlink()
        or output.exists()
        or not output.is_relative_to(benchmark_root)
        or output.suffix != ".json"
    ):
        raise ValueError(
            "Trial output must be a new JSON file below its benchmark root."
        )
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return value


def _measure(callable_: Any) -> tuple[Any, dict[str, float]]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = callable_()
    return result, {
        "wall_seconds": time.perf_counter() - wall_started,
        "cpu_seconds": time.process_time() - cpu_started,
    }


def _require_timing(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "wall_seconds",
        "cpu_seconds",
    }:
        raise ValueError(f"{label} timing field set is not exact.")
    for field in ("wall_seconds", "cpu_seconds"):
        observed = value[field]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0
        ):
            raise ValueError(f"{label} {field} is invalid.")


def _open_root(archive: Path, *, consolidated: bool) -> zarr.Group:
    return zarr.open_group(str(archive), mode="r", use_consolidated=consolidated)


def _group(root: zarr.Group, path: str) -> zarr.Group:
    value = root.get(path)
    if not isinstance(value, zarr.Group):
        raise ValueError(f"Missing eye-angle group {path!r}.")
    return value


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_document(array: Any) -> dict[str, Any]:
    values = np.asarray(array[:])
    return {
        "dtype": np.dtype(values.dtype).str,
        "shape": [int(value) for value in values.shape],
        "decoded_nbytes": int(values.nbytes),
        "decoded_sha256": _sha256_array(values),
    }


def _full_scan(run: Any, *, paths: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    arrays = collect_eye_angle_arrays(run)
    if set(arrays) != set(paths):
        raise ValueError("Eye-angle array inventory differs from exact workload.")
    documents = {path: _array_document(arrays[path]) for path in paths}
    return {
        "array_count": len(documents),
        "decoded_nbytes": sum(item["decoded_nbytes"] for item in documents.values()),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "arrays": documents,
        "arrays_digest": canonical_json_sha256(documents),
    }


def _logical_value(value: Any, *, include_attrs: bool = True) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": np.dtype(value.dtype).str,
            "shape": [int(extent) for extent in value.shape],
            "decoded_sha256": _sha256_array(value),
        }
    if isinstance(value, np.generic):
        return _logical_value(value.item(), include_attrs=include_attrs)
    if is_dataclass(value):
        return {
            item.name: _logical_value(
                getattr(value, item.name), include_attrs=include_attrs
            )
            for item in fields(value)
            if include_attrs or item.name != "attrs"
        }
    if isinstance(value, Mapping):
        return {
            str(key): _logical_value(child, include_attrs=include_attrs)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_logical_value(child, include_attrs=include_attrs) for child in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            return {"palette_exact_float": "nan" if math.isnan(value) else str(value)}
        return value
    raise TypeError(f"Unsupported logical eye-angle value {type(value).__name__}.")


def _logical_tables_document(tables: Any) -> dict[str, Any]:
    value = _logical_value(
        {
            "roi": tables.roi,
            "frame": tables.frame,
            "qa_roi": tables.qa_roi,
            "qa_frame": tables.qa_frame,
            "support": tables.support,
        },
        include_attrs=False,
    )
    if not isinstance(value, dict):
        raise TypeError("Eye-angle logical table result must be one object.")
    return value


def _candidate_tables(run: zarr.Group, *, run_name: str) -> Any:
    """Use the strict v7 adapter without weakening public selector checks."""

    eye_angle_io._require_eye_angle_payload_contract(  # noqa: SLF001
        run, legacy_compatibility=False
    )
    attrs = dict(run.attrs)
    if attrs.get("layout") != EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        raise ValueError("Candidate is not maintained compact-dense v7.")
    return eye_angle_io._compact_dense_tables(  # noqa: SLF001
        run,
        resolved_run=run_name,
        run_path=_run_path(run_name),
        attrs=attrs,
    )


def _metadata_receipt(archive: Path, *, run_path: str) -> dict[str, Any]:
    return validate_direct_consolidated_subtree(
        archive, subtree_path=run_path
    ).to_json()


def _metadata_declarations(
    archive: Path, *, run_path: str
) -> dict[str, dict[str, Any]]:
    root = archive.joinpath(*run_path.split("/"))
    declarations: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("zarr.json")):
        if path.is_symlink():
            raise ValueError("Eye-angle metadata declaration must not be a symlink.")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("Eye-angle Zarr declaration must be one object.")
        node_path = path.parent.relative_to(archive).as_posix()
        declarations[node_path] = metadata_without_empty_group_consolidation(
            dict(raw), path=node_path
        )
    return declarations


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_metadata(child)
            for key, child in value.items()
            if key != "consolidated_metadata"
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_metadata(child) for child in value]
    if value == "NaN" or (
        isinstance(value, (float, np.floating)) and math.isnan(float(value))
    ):
        return {"palette_exact_float": "nan"}
    return value


def _require_metadata_receipt(
    value: object, *, run_path: str, array_count: int
) -> None:
    expected_fields = {
        "schema_id",
        "schema_version",
        "subtree_path",
        "node_count",
        "group_count",
        "array_count",
        "declarations_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("Eye-angle metadata-equivalence receipt is not exact.")
    if (
        value["schema_id"] != METADATA_EQUIVALENCE_SCHEMA_ID
        or value["schema_version"] != METADATA_EQUIVALENCE_SCHEMA_VERSION
        or value["subtree_path"] != run_path
        or value["array_count"] != array_count
        or not _is_sha256(value["declarations_sha256"])
        or any(
            type(value[name]) is not int or value[name] < 0
            for name in ("node_count", "group_count", "array_count")
        )
        or value["node_count"] != value["group_count"] + value["array_count"]
    ):
        raise ValueError("Eye-angle metadata-equivalence receipt is invalid.")


def _resolved_source_shape(
    shape_template: Sequence[str | int], dimensions: Mapping[str, int]
) -> list[int]:
    return [
        int(dimensions[item]) if isinstance(item, str) else int(item)
        for item in shape_template
    ]


def _require_source_array_declarations(
    value: Mapping[str, Any],
    *,
    run_path: str,
    dimensions: EyeAngleDimensions,
) -> None:
    expected = {
        declaration.path: declaration
        for declaration in build_eye_angle_array_declarations(
            byte_planner_adopted=False
        )
    }
    prefix = f"{run_path}/"
    observed = {
        path.removeprefix(prefix): declaration
        for path, declaration in value.items()
        if path.startswith(prefix)
        and isinstance(declaration, Mapping)
        and declaration.get("node_type") == "array"
    }
    if set(observed) != set(expected):
        raise ValueError("Source array declaration inventory differs from executable v7.")
    resolved_dimensions = dimensions.contract_dimensions
    for path, contract in expected.items():
        declaration = observed[path]
        try:
            observed_dtype = np.dtype(declaration.get("data_type")).str
        except TypeError as exc:
            raise ValueError(
                f"Source array dtype is invalid at {path!r}."
            ) from exc
        expected_dtype = np.dtype(contract.contract.dtype.numpy_dtype).str
        expected_shape = _resolved_source_shape(
            contract.contract.shape_template,
            resolved_dimensions,
        )
        if (
            declaration.get("shape") != expected_shape
            or observed_dtype != expected_dtype
        ):
            raise ValueError(
                f"Source array logical declaration differs from executable v7 at {path!r}."
            )
        chunk_grid = declaration.get("chunk_grid")
        codecs = declaration.get("codecs")
        if (
            not isinstance(chunk_grid, Mapping)
            or chunk_grid.get("name") != "regular"
            or not isinstance(chunk_grid.get("configuration"), Mapping)
            or not isinstance(
                chunk_grid["configuration"].get("chunk_shape"), list
            )
            or len(chunk_grid["configuration"]["chunk_shape"])
            != len(expected_shape)
            or not isinstance(codecs, list)
            or not codecs
        ):
            raise ValueError(
                f"Source array physical declaration is malformed at {path!r}."
            )


def _require_metadata_declarations(
    value: object,
    *,
    receipt: Mapping[str, Any],
    run_path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Eye-angle metadata declarations must be nonempty.")
    prefix = f"{run_path}/"
    groups = 0
    arrays = 0
    for path, declaration in value.items():
        if (
            type(path) is not str
            or (path != run_path and not path.startswith(prefix))
            or not isinstance(declaration, Mapping)
        ):
            raise ValueError("Eye-angle metadata declaration path is invalid.")
        if declaration.get("node_type") == "group":
            groups += 1
        elif declaration.get("node_type") == "array":
            arrays += 1
        else:
            raise ValueError("Eye-angle metadata declaration node type is invalid.")
    if (
        len(value) != receipt["node_count"]
        or groups != receipt["group_count"]
        or arrays != receipt["array_count"]
        or canonical_json_sha256(value) != receipt["declarations_sha256"]
    ):
        raise ValueError("Eye-angle metadata declarations differ from receipt.")
    return value


def _require_candidate_physical_declarations(
    value: Mapping[str, Any],
    *,
    run_path: str,
    dimensions: EyeAngleDimensions,
    receipt: Mapping[str, Any],
) -> None:
    executable = build_eye_angle_candidate_storage_plan(dimensions)
    if dict(receipt) != executable.as_manifest():
        raise ValueError("Candidate storage receipt differs from executable plan.")
    entries = eye_angle_storage_entries_by_path(executable)
    for relative_path, entry in entries.items():
        path = f"{run_path}/{relative_path}"
        declaration = value.get(path)
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("node_type") != "array"
        ):
            raise ValueError(f"Candidate array declaration is missing {path!r}.")
        attributes = declaration.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError(f"Candidate attributes are invalid at {path!r}.")
        nonreserved = {
            str(key): child
            for key, child in attributes.items()
            if key not in _RESERVED_ARRAY_ATTRIBUTES
        }
        expected = array_metadata_declaration_from_plan(
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=eye_angle_planned_fill_value(entry),
            attributes=nonreserved,
        )
        observed = {
            key: child
            for key, child in declaration.items()
            if key not in {"zarr_format", "node_type", "consolidated_metadata"}
        }
        if _normalize_metadata(observed) != _normalize_metadata(expected):
            raise ValueError(
                f"Candidate metadata differs from executable plan at {path!r}."
            )


def _physical_array_declarations(
    declarations: Mapping[str, Any], *, run_path: str
) -> dict[str, Any]:
    prefix = f"{run_path}/"
    result: dict[str, Any] = {}
    for path, declaration in declarations.items():
        if not path.startswith(prefix) or not isinstance(declaration, Mapping):
            continue
        if declaration.get("node_type") != "array":
            continue
        result[path.removeprefix(prefix)] = {
            name: _normalize_metadata(declaration.get(name))
            for name in ("chunk_grid", "chunk_key_encoding", "codecs", "fill_value")
        }
    return result


def _exact_candidate_envelope() -> dict[str, Any]:
    return {
        "profile_id": EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        "status": "selector_ineligible_candidate",
        "activation_allowed": False,
        "whole_shard_write_ownership": "single_serial_writer",
    }


def _require_completion(
    run: zarr.Group, *, run_name: str, selector_eligible: bool
) -> None:
    attrs = dict(run.attrs)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not selector_eligible
        or attrs.get("run_name") not in {None, run_name}
        or "atomic_publication_tombstone" in attrs
    ):
        raise ValueError("Eye-angle lifecycle/completion binding is invalid.")


def _validate_static_contracts(attrs: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_eye_angle_persisted_contract_manifests(attrs)
    if errors:
        raise ValueError(
            "Eye-angle persisted contracts are invalid: " + "; ".join(errors)
        )
    required = {
        "eye_angle_output_schema",
        "eye_angle_variant_schema",
        "eye_angle_algorithm_contract",
        "eye_angle_source_contracts",
    }
    if not required.issubset(attrs):
        raise ValueError("Eye-angle persisted contract manifest set is incomplete.")
    return {name: canonical_json_sha256(attrs[name]) for name in sorted(required)}


def _validate_source(
    root: zarr.Group, run: zarr.Group, *, run_name: str
) -> dict[str, Any]:
    issues = validate_eye_angle_compact_run(run)
    if issues:
        raise ValueError(
            "Invalid compact-v7 source: "
            + "; ".join(f"{item.code}:{item.path}:{item.message}" for item in issues)
        )
    _require_completion(run, run_name=run_name, selector_eligible=True)
    attrs = dict(run.attrs)
    dimensions = eye_angle_dimensions_from_run_attrs(attrs)
    if (
        attrs.get(EYE_ANGLE_STORAGE_CANDIDATE_ATTR) is not None
        or attrs.get(EYE_ANGLE_STORAGE_PLAN_ATTR) is not None
    ):
        raise ValueError("Established source unexpectedly carries candidate markers.")
    if attrs.get(EYE_ANGLE_ARRAY_SCHEMA_ATTR) != eye_angle_array_schema_manifest(
        dimensions, byte_planner_adopted=False
    ):
        raise ValueError("Source array manifest differs from executable v7 schema.")
    tables = eye_angle_io.load_eye_angle_run_tables(root, run_name=run_name)
    return {
        "consumer_path": "maintained_selector_eligible_compact_v7_reader",
        "dimensions": dimensions.contract_dimensions,
        "logical_table_digest": canonical_json_sha256(_logical_tables_document(tables)),
        "contract_digests": _validate_static_contracts(attrs),
    }


def _validate_candidate(run: zarr.Group, *, run_name: str) -> dict[str, Any]:
    issues = validate_eye_angle_compact_run(run)
    attrs = dict(run.attrs)
    dimensions = eye_angle_dimensions_from_run_attrs(attrs)
    issues = (
        *issues,
        *validate_eye_angle_candidate_storage(run, dimensions=dimensions),
    )
    if issues:
        raise ValueError(
            "Invalid compact-v7 candidate: "
            + "; ".join(f"{item.code}:{item.path}:{item.message}" for item in issues)
        )
    _require_completion(run, run_name=run_name, selector_eligible=False)
    if attrs.get(EYE_ANGLE_STORAGE_CANDIDATE_ATTR) != _exact_candidate_envelope():
        raise ValueError("Candidate envelope is not exact.")
    if attrs.get("publication_scope") != "storage_benchmark_candidate_only":
        raise ValueError("Candidate publication scope is not benchmark-only.")
    if attrs.get(EYE_ANGLE_ARRAY_SCHEMA_ATTR) != eye_angle_array_schema_manifest(
        dimensions, byte_planner_adopted=True
    ):
        raise ValueError("Candidate array manifest differs from executable v7 schema.")
    tables = _candidate_tables(run, run_name=run_name)
    return {
        "consumer_path": "explicit_ineligible_strict_compact_v7_candidate_adapter",
        "dimensions": dimensions.contract_dimensions,
        "logical_table_digest": canonical_json_sha256(_logical_tables_document(tables)),
        "contract_digests": _validate_static_contracts(attrs),
        "storage_receipt_digest": canonical_json_sha256(
            attrs[EYE_ANGLE_STORAGE_PLAN_ATTR]
        ),
    }


def _relative_dependency_paths(attrs: Mapping[str, Any]) -> tuple[str, ...]:
    contracts = attrs.get("eye_angle_source_contracts")
    paths: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(
                    child,
                    (
                        "resolved_array_path"
                        if key == "resolved_arrays"
                        else str(child_key)
                    ),
                )
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if type(value) is not str or "path" not in key:
            return
        candidate = PurePosixPath(value.strip("/"))
        if value.startswith("/") or ".." in candidate.parts or not candidate.parts:
            return
        paths.add(candidate.as_posix())

    visit(contracts)
    return tuple(sorted(paths))


def _dependency_declarations(
    archive: Path, *, paths: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in paths:
        dependency = _require_archive_directory(
            archive,
            relative,
            label="Eye-angle dependency",
        )
        _require_nonsymlink_tree(dependency, label="Eye-angle dependency")
        metadata = dependency / "zarr.json"
        if not metadata.is_file():
            continue
        if metadata.is_symlink():
            raise ValueError("Eye-angle dependency metadata must not be a symlink.")
        raw = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("Eye-angle dependency metadata must be one object.")
        result[relative] = metadata_without_empty_group_consolidation(
            dict(raw), path=relative
        )
    return result


def _require_source_contract_dependency_binding(
    contracts: object, declarations: object
) -> str:
    if not isinstance(contracts, Mapping) or not isinstance(declarations, Mapping):
        raise ValueError("Eye-angle source contract/dependency evidence is invalid.")

    def declaration(path: object) -> Mapping[str, Any]:
        if type(path) is not str or path not in declarations:
            raise ValueError(f"Eye-angle bound dependency is missing {path!r}.")
        value = declarations[path]
        if not isinstance(value, Mapping):
            raise ValueError("Eye-angle dependency declaration is invalid.")
        return value

    for section_name in (
        "eye_geometry",
        "keypoints",
        "diagnostic_base_keypoints",
    ):
        section = contracts.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(f"Eye-angle source section {section_name!r} is missing.")
        if section.get("available") is True:
            attrs = declaration(section.get("path")).get("attributes")
            if not isinstance(attrs, Mapping):
                raise ValueError("Eye-angle dependency group attributes are invalid.")
            for name in _SOURCE_IDENTITY_ATTRS:
                if name in section and section[name] != attrs.get(name):
                    raise ValueError(
                        f"Eye-angle dependency identity differs for {section_name}.{name}."
                    )
    geometry = contracts.get("eye_geometry")
    components = geometry.get("components") if isinstance(geometry, Mapping) else None
    if not isinstance(components, list) or len(components) != 2:
        raise ValueError("Eye-angle geometry component contract is incomplete.")
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("Eye-angle geometry component contract is invalid.")
        attrs = declaration(component.get("group_path")).get("attributes")
        frozen = component.get("ellipse_source_contract")
        if not isinstance(attrs, Mapping) or not isinstance(frozen, Mapping):
            raise ValueError("Eye-angle ellipse source binding is invalid.")
        if any(attrs.get(name) != child for name, child in frozen.items()):
            raise ValueError("Eye-angle ellipse source attrs differ from dependency.")
        for name in ("ellipse_params_path", "ellipse_success_path"):
            if declaration(component.get(name)).get("node_type") != "array":
                raise ValueError("Eye-angle ellipse source array binding is invalid.")
    resolved = contracts.get("resolved_arrays")
    if not isinstance(resolved, Mapping):
        raise ValueError("Eye-angle resolved source arrays are missing.")
    for path in resolved.values():
        if declaration(path).get("node_type") != "array":
            raise ValueError("Eye-angle resolved source array binding is invalid.")
    return canonical_json_sha256(declarations)


def _metadata_guard(
    archive: Path, *, run_names: Sequence[str], dependency_paths: Sequence[str]
) -> dict[str, Any]:
    paths = {archive / "zarr.json"}
    for relative in ("analysis", EYE_ANGLE_RUN_PARENT):
        candidate = archive.joinpath(*relative.split("/"), "zarr.json")
        if candidate.is_file():
            paths.add(candidate)
    for run_name in run_names:
        run_root = _safe_run_directory(archive, name=run_name)
        paths.update(run_root.rglob("zarr.json"))
    for relative in dependency_paths:
        node = archive.joinpath(*PurePosixPath(relative).parts)
        if not node.resolve().is_relative_to(archive):
            raise ValueError("Eye-angle dependency path escapes the archive.")
        current = node if node.is_dir() else node.parent
        if current.exists():
            _require_nonsymlink_tree(current, label="Eye-angle source dependency")
            paths.update(current.rglob("zarr.json"))
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError("Eye-angle metadata guard rejects symlinks.")
        stat = path.stat()
        files[path.relative_to(archive).as_posix()] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"files": files, "digest": canonical_json_sha256(files)}


def _require_metadata_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"files", "digest"}:
        raise ValueError("Eye-angle metadata guard field set is not exact.")
    files = value["files"]
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Eye-angle metadata guard is empty.")
    for path, facts in files.items():
        if (
            type(path) is not str
            or not isinstance(facts, Mapping)
            or set(facts) != {"size", "mtime_ns", "sha256"}
            or type(facts["size"]) is not int
            or facts["size"] < 0
            or type(facts["mtime_ns"]) is not int
            or facts["mtime_ns"] < 0
            or not _is_sha256(facts["sha256"])
        ):
            raise ValueError("Eye-angle metadata guard entry is invalid.")
    if value["digest"] != canonical_json_sha256(files):
        raise ValueError("Eye-angle metadata guard digest mismatch.")


def _require_publication_receipt(
    value: object,
    *,
    archive: Path,
    run_path: str,
    owner_uuid: object,
    parent_attrs: Mapping[str, Any],
    candidate_attrs: Mapping[str, Any],
    dimensions: EyeAngleDimensions,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Candidate lacks an atomic publication receipt.")
    required = {
        "authoritative_source_zarr",
        "node_local_staged_zarr",
        "node_local_regular_run",
        "node_local_sharded_run",
        "node_local_publication_run",
        "materialization",
        "source_revision_audit",
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
        "parent_attrs_after",
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
        "promotion_policy",
        "metadata_visibility_policy",
        "storage_profile_id",
    }
    if set(value) != required:
        raise ValueError("Candidate atomic publication receipt field set is not exact.")
    if (
        value.get("schema_id") != PUBLISH_SCHEMA_ID
        or value.get("publisher_contract")
        != {
            "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
            "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
        }
        or value.get("serialization_policy") != SERIALIZATION_POLICY
        or value.get("policy") != _CANDIDATE_PUBLICATION_POLICY
        or value.get("rollback_policy") != _ATOMIC_ROLLBACK_POLICY
        or value.get("publication_owner_attr") != ATOMIC_PUBLICATION_OWNER_ATTR
        or value.get("publication_owner_uuid") != owner_uuid
        or value.get("failed_public_child_policy")
        != "retain_owner_bound_selector_ineligible_tombstone"
        or value.get("hidden_temporary_policy")
        != "same_parent_hidden_sibling_then_os_replace"
        or value.get("promotion_policy") != _CANDIDATE_PROMOTION_POLICY
        or value.get("storage_profile_id")
        != EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
        or Path(str(value.get("authoritative_source_zarr"))).resolve() != archive
        or Path(str(value.get("source_zarr"))).resolve() != archive
        or Path(str(value.get("target_run_path"))).resolve()
        != archive.joinpath(*run_path.split("/"))
        or value.get("parent_attrs_before") != value.get("parent_attrs_after")
        or value.get("parent_attrs_after") != {EYE_ANGLE_RUN_PARENT: dict(parent_attrs)}
    ):
        raise ValueError("Candidate atomic publication receipt binding is invalid.")
    local_publication = Path(str(value.get("node_local_publication_run")))
    local_regular = Path(str(value.get("node_local_regular_run")))
    local_staged = Path(str(value.get("node_local_staged_zarr")))
    if (
        type(value.get("publication_source_run_path")) is not str
        or not Path(value["publication_source_run_path"]).is_absolute()
        or not local_publication.is_absolute()
        or not local_regular.is_absolute()
        or not local_staged.is_absolute()
        or Path(value["publication_source_run_path"]) != local_publication
        or local_regular != local_publication
        or value.get("node_local_sharded_run") is not None
        or not local_publication.is_relative_to(local_staged)
    ):
        raise ValueError("Candidate publication source path is invalid.")
    if (
        type(value.get("published_at_utc")) is not str
        or not value["published_at_utc"]
        or type(value.get("host")) is not str
        or not value["host"]
        or value.get("lsb_jobid") is not None
        and type(value.get("lsb_jobid")) is not str
    ):
        raise ValueError("Candidate publication runtime identity is invalid.")
    for field in ("copy_duration_seconds",):
        observed = value.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0
        ):
            raise ValueError(f"Candidate publication {field} is invalid.")
    physical = value.get("physical_copy")
    if (
        not isinstance(physical, Mapping)
        or set(physical)
        != {
            "backend",
            "verification",
            "file_count",
            "physical_bytes",
            "inventory_sha256",
            "content_sha256",
        }
        or physical.get("backend") not in {"python", "rsync"}
        or physical.get("verification")
        not in {"sha256_all_physical_files", "rsync_checksum_dry_run"}
        or (
            physical.get("backend") == "python"
            and physical.get("verification") != "sha256_all_physical_files"
        )
        or (
            physical.get("backend") == "rsync"
            and physical.get("verification") != "rsync_checksum_dry_run"
        )
        or type(physical.get("file_count")) is not int
        or physical["file_count"] < 1
        or type(physical.get("physical_bytes")) is not int
        or physical["physical_bytes"] < 1
        or not _is_sha256(physical.get("inventory_sha256"))
        or not _is_sha256(physical.get("content_sha256"))
    ):
        raise ValueError("Candidate physical-copy receipt is invalid.")
    expected_validation_fields = {
        "valid",
        "errors",
        "exact_compact_v7_valid",
        "row_count",
        "frame_count",
        "angle_channel_count",
        "qa_channel_count",
        "array_count",
        "sharded_array_count",
        "require_sharded",
        "source_contract_sha256",
        "instance_key_sha256",
        "source_acquisition_frame_index_sha256",
        "algorithm_contract_sha256",
        "output_schema_sha256",
        "physical_storage_layout",
        "storage_profile_id",
        "candidate_storage_valid",
    }
    materialization = value.get("materialization")
    persisted_materialization = candidate_attrs.get("node_local_materialization")
    if (
        not isinstance(materialization, Mapping)
        or materialization != persisted_materialization
        or set(materialization)
        != {
            "schema_id",
            "status",
            "completed_at_utc",
            "authoritative_source_zarr",
            "node_local_staged_zarr",
            "source_access_policy",
            "source_staging",
            "compute",
            "regular_validation",
            "source_contract_sha256",
            "source_metadata_sha256",
            "staged_input_integrity_receipt_sha256",
            "staged_input_integrity_receipt",
            "algorithm_contract",
            "output_contract",
            "local_direct_consolidated_array_count",
            "final_physical_validation",
        }
        or materialization.get("schema_id") != MATERIALIZATION_SCHEMA_ID
        or materialization.get("status") != "complete"
        or materialization.get("authoritative_source_zarr") != str(archive)
        or Path(str(materialization.get("node_local_staged_zarr"))) != local_staged
        or materialization.get("source_access_policy")
        != "authoritative_shared_read_only_then_exact_local_subset"
        or materialization.get("local_direct_consolidated_array_count") != 41
    ):
        raise ValueError("Candidate materialization receipt is not exact.")
    final_physical_validation = materialization.get("final_physical_validation")
    regular_validation = materialization.get("regular_validation")
    for field in (
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
    ):
        validation = value.get(field)
        if (
            not isinstance(validation, Mapping)
            or set(validation) != expected_validation_fields
            or validation.get("valid") is not True
            or validation.get("errors") != []
            or validation.get("array_count") != 41
            or validation.get("exact_compact_v7_valid") is not True
            or validation.get("row_count") != dimensions.n_roi_rows
            or validation.get("frame_count") != dimensions.n_frames
            or validation.get("angle_channel_count")
            != dimensions.contract_dimensions["n_angle_channels"]
            or validation.get("qa_channel_count")
            != dimensions.contract_dimensions["n_qa_channels"]
            or validation.get("sharded_array_count") != 0
            or validation.get("require_sharded") is not False
            or validation.get("physical_storage_layout") is not None
            or validation.get("storage_profile_id")
            != EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
            or validation.get("candidate_storage_valid") is not True
            or any(
                not _is_sha256(validation.get(name))
                for name in (
                    "source_contract_sha256",
                    "instance_key_sha256",
                    "source_acquisition_frame_index_sha256",
                    "algorithm_contract_sha256",
                    "output_schema_sha256",
                )
            )
            or validation != final_physical_validation
        ):
            raise ValueError(f"Candidate publication {field} did not pass exactly.")
    if (
        not isinstance(regular_validation, Mapping)
        or set(regular_validation) != expected_validation_fields
        or regular_validation != final_physical_validation
    ):
        raise ValueError("Candidate materialization validation phases differ.")
    staging = materialization.get("source_staging")
    if (
        not isinstance(staging, Mapping)
        or staging.get("schema_id") != STAGING_SCHEMA_ID
        or staging.get("status") != "complete"
        or staging.get("authoritative_source_zarr") != str(archive)
        or Path(str(staging.get("node_local_staged_zarr"))) != local_staged
    ):
        raise ValueError("Candidate source-staging receipt is invalid.")
    revision = value.get("source_revision_audit")
    if (
        not isinstance(revision, Mapping)
        or set(revision)
        != {
            "schema_id",
            "status",
            "checked_at_utc",
            "authoritative_source_zarr",
            "subject_shape_run",
            "keypoint_run",
            "inventory",
            "expected_source_metadata_sha256",
            "observed_source_metadata_sha256",
            "expected_source_contract_sha256",
            "observed_source_contract_sha256",
            "full_selected_scientific_input_content_hash",
            "errors",
        }
        or revision.get("schema_id") != SOURCE_REVISION_AUDIT_SCHEMA_ID
        or revision.get("status") != "current"
        or revision.get("authoritative_source_zarr") != str(archive)
        or revision.get("errors") != []
        or revision.get("full_selected_scientific_input_content_hash") is not True
        or revision.get("expected_source_metadata_sha256")
        != revision.get("observed_source_metadata_sha256")
        or revision.get("expected_source_contract_sha256")
        != revision.get("observed_source_contract_sha256")
        or not _is_sha256(revision.get("expected_source_metadata_sha256"))
        or not _is_sha256(revision.get("expected_source_contract_sha256"))
    ):
        raise ValueError("Candidate source-revision receipt is invalid.")
    visibility = value.get("metadata_visibility_policy")
    if visibility != {
        "authoritative_root_consolidation": "after_final_publisher_metadata_write",
        "direct_consolidated_group_attrs_required": True,
        "direct_consolidated_array_declarations_required": 41,
        "consolidated_parent_selectors_must_match_publication_snapshot": True,
    }:
        raise ValueError("Candidate metadata visibility policy is invalid.")
    return value


def _environment(*, archive: Path, cache_state: str) -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "archive_device": int(archive.stat().st_dev),
        "cache_state": cache_state,
        "thread_environment": {
            name: os.environ.get(name) for name in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def _require_environment(value: object, *, cache_state: str) -> None:
    fields_ = {
        "hostname",
        "system",
        "release",
        "machine",
        "python",
        "numpy",
        "zarr",
        "archive_device",
        "cache_state",
        "thread_environment",
    }
    if not isinstance(value, Mapping) or set(value) != fields_:
        raise ValueError("Eye-angle environment field set is not exact.")
    if value["cache_state"] != cache_state:
        raise ValueError("Eye-angle cache-state binding differs.")
    threads = value["thread_environment"]
    if not isinstance(threads, Mapping) or set(threads) != set(
        STORAGE_BENCHMARK_THREAD_ENVIRONMENT
    ):
        raise ValueError("Eye-angle thread environment is incomplete.")


def _physical_io() -> dict[str, Any]:
    return {
        "availability": _PHYSICAL_IO_AVAILABILITY,
        "read_operations": None,
        "transferred_bytes": None,
        "range_reads": None,
        "trace_artifact": None,
    }


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    if (
        type(seed) is not int
        or type(repetition_index) is not int
        or repetition_index < 0
    ):
        raise ValueError("Eye-angle rotation inputs must be exact integers.")
    return (
        ("candidate", "source")
        if (seed + repetition_index) % 2
        else ("source", "candidate")
    )


def _preflight(
    archive_value: Path | str,
    *,
    source_run_name: str,
    candidate_run_name: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    archive = _safe_archive(archive_value)
    source_name = _safe_run_name(source_run_name, label="source run")
    candidate_name = _safe_run_name(candidate_run_name, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate eye-angle names must differ.")
    if type(seed) is not int or type(repetitions) is not int or repetitions < 1:
        raise ValueError("Eye-angle seed/repetitions are invalid.")
    _safe_run_directory(archive, name=source_name)
    _safe_run_directory(archive, name=candidate_name)

    direct = _open_root(archive, consolidated=False)
    consolidated = _open_root(archive, consolidated=True)
    source_direct = _group(direct, _run_path(source_name))
    source_consolidated = _group(consolidated, _run_path(source_name))
    candidate_direct = _group(direct, _run_path(candidate_name))
    candidate_consolidated = _group(consolidated, _run_path(candidate_name))

    source_validation = _validate_source(direct, source_direct, run_name=source_name)
    source_consolidated_validation = _validate_source(
        consolidated, source_consolidated, run_name=source_name
    )
    candidate_validation = _validate_candidate(
        candidate_direct, run_name=candidate_name
    )
    candidate_consolidated_validation = _validate_candidate(
        candidate_consolidated, run_name=candidate_name
    )
    if source_validation != source_consolidated_validation:
        raise ValueError("Source direct/consolidated strict-reader evidence differs.")
    if candidate_validation != candidate_consolidated_validation:
        raise ValueError(
            "Candidate direct/consolidated strict-reader evidence differs."
        )
    if source_validation["dimensions"] != candidate_validation["dimensions"]:
        raise ValueError("Source and candidate eye-angle dimensions differ.")
    if (
        source_validation["logical_table_digest"]
        != candidate_validation["logical_table_digest"]
    ):
        raise ValueError("Source public-reader and candidate diagnostic tables differ.")
    if (
        source_validation["contract_digests"]
        != candidate_validation["contract_digests"]
    ):
        raise ValueError("Source and candidate scientific/lineage contracts differ.")

    paths = tuple(sorted(collect_eye_angle_arrays(source_direct)))
    if len(paths) != 41:
        raise ValueError("Maintained eye-angle v7 must contain exactly 41 arrays.")
    source_scan = _full_scan(source_direct, paths=paths)
    candidate_scan = _full_scan(candidate_direct, paths=paths)
    if source_scan["arrays"] != candidate_scan["arrays"]:
        raise ValueError("Source and candidate decoded eye-angle arrays differ.")

    dimensions = eye_angle_dimensions_from_run_attrs(dict(candidate_direct.attrs))
    direct_consolidated_issues = validate_eye_angle_direct_consolidated_storage(
        candidate_direct,
        candidate_consolidated,
        dimensions=dimensions,
    )
    if direct_consolidated_issues:
        raise ValueError(
            "Candidate direct/consolidated metadata differs: "
            + "; ".join(item.message for item in direct_consolidated_issues)
        )
    source_metadata = _metadata_receipt(archive, run_path=_run_path(source_name))
    candidate_metadata = _metadata_receipt(archive, run_path=_run_path(candidate_name))
    source_declarations = _metadata_declarations(
        archive, run_path=_run_path(source_name)
    )
    candidate_declarations = _metadata_declarations(
        archive, run_path=_run_path(candidate_name)
    )
    if _physical_array_declarations(
        source_declarations, run_path=_run_path(source_name)
    ) == _physical_array_declarations(
        candidate_declarations, run_path=_run_path(candidate_name)
    ):
        raise ValueError("Source and candidate physical array layouts do not differ.")
    candidate_attrs = dict(candidate_direct.attrs)
    source_attrs = dict(source_direct.attrs)
    direct_parent_attrs = dict(direct[EYE_ANGLE_RUN_PARENT].attrs)
    consolidated_parent_attrs = dict(consolidated[EYE_ANGLE_RUN_PARENT].attrs)
    if direct_parent_attrs != consolidated_parent_attrs:
        raise ValueError("Eye-angle parent direct/consolidated attributes differ.")
    publication = _require_publication_receipt(
        candidate_attrs.get("cluster_output_staging"),
        archive=archive,
        run_path=_run_path(candidate_name),
        owner_uuid=candidate_attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR),
        parent_attrs=direct_parent_attrs,
        candidate_attrs=candidate_attrs,
        dimensions=dimensions,
    )
    dependencies = tuple(
        sorted(
            set(_relative_dependency_paths(source_attrs))
            | set(_relative_dependency_paths(candidate_attrs))
        )
    )
    dependency_declarations = _dependency_declarations(archive, paths=dependencies)
    dependency_digest = _require_source_contract_dependency_binding(
        source_attrs["eye_angle_source_contracts"], dependency_declarations
    )
    payload = {
        "family_id": FAMILY_ID,
        "benchmark_id": BENCHMARK_ID,
        "workload_id": WORKLOAD_ID,
        "archive": str(archive),
        "parent_path": EYE_ANGLE_RUN_PARENT,
        "source_run": source_name,
        "candidate_run": candidate_name,
        "source_run_path": _run_path(source_name),
        "candidate_run_path": _run_path(candidate_name),
        "seed": seed,
        "repetitions": repetitions,
        "array_order": list(paths),
        "dimensions": dimensions.contract_dimensions,
        "source_array_schema_manifest": source_attrs[EYE_ANGLE_ARRAY_SCHEMA_ATTR],
        "candidate_array_schema_manifest": candidate_attrs[EYE_ANGLE_ARRAY_SCHEMA_ATTR],
        "candidate_storage_receipt": candidate_attrs[EYE_ANGLE_STORAGE_PLAN_ATTR],
        "candidate_envelope": candidate_attrs[EYE_ANGLE_STORAGE_CANDIDATE_ATTR],
        "candidate_publication_receipt": publication,
        "candidate_publication_receipt_digest": canonical_json_sha256(publication),
        "parent_selector_snapshot": direct_parent_attrs,
        "source_metadata_equivalence": source_metadata,
        "candidate_metadata_equivalence": candidate_metadata,
        "source_metadata_declarations": source_declarations,
        "candidate_metadata_declarations": candidate_declarations,
        "source_contract_digests": source_validation["contract_digests"],
        "candidate_contract_digests": candidate_validation["contract_digests"],
        "physical_array_layouts_differ": True,
        "logical_table_digest": source_validation["logical_table_digest"],
        "expected_arrays": source_scan["arrays"],
        "expected_arrays_digest": canonical_json_sha256(source_scan["arrays"]),
        "expected_decoded_nbytes": source_scan["decoded_nbytes"],
        "dependency_paths": list(dependencies),
        "dependency_metadata_declarations": dependency_declarations,
        "dependency_metadata_declarations_digest": dependency_digest,
        "candidate_profile_id": EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        "source_profile_id": EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
        "candidate_selector_eligible": False,
        "promotion_authorized": False,
        "palette_consumer_implemented": False,
        "candidate_adapter_scope": "diagnostic_only_private_strict_payload_adapter",
        "physical_io_availability": _PHYSICAL_IO_AVAILABILITY,
    }
    result = _envelope(WORKLOAD_SCHEMA_ID, payload)
    require_workload(result)
    return result


def _dimensions_from_manifest(value: object) -> EyeAngleDimensions:
    if not isinstance(value, Mapping) or set(value) != {
        "n_roi_rows",
        "n_frames",
        "n_angle_channels",
        "n_vector_channels",
        "n_qa_channels",
        "angle_block_width",
    }:
        raise ValueError("Eye-angle dimension field set is not exact.")
    return EyeAngleDimensions(
        n_roi_rows=value["n_roi_rows"],
        n_frames=value["n_frames"],
        angle_block_width=value["angle_block_width"],
    )


def _require_run_group_bindings(
    source_declarations: Mapping[str, Any],
    candidate_declarations: Mapping[str, Any],
    *,
    source_path: str,
    candidate_path: str,
    source_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    candidate_envelope: Mapping[str, Any],
    candidate_publication: Mapping[str, Any],
    source_contract_digests: Mapping[str, Any],
    candidate_contract_digests: Mapping[str, Any],
) -> None:
    source = source_declarations.get(source_path)
    candidate = candidate_declarations.get(candidate_path)
    if not isinstance(source, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("Eye-angle run-group declarations are missing.")
    source_attrs = source.get("attributes")
    candidate_attrs = candidate.get("attributes")
    if not isinstance(source_attrs, Mapping) or not isinstance(
        candidate_attrs, Mapping
    ):
        raise ValueError("Eye-angle run-group attributes are invalid.")
    if (
        source_attrs.get(EYE_ANGLE_ARRAY_SCHEMA_ATTR) != source_manifest
        or source_attrs.get("stage_selector_eligible") is not True
        or source_attrs.get("palette_run_completion_status") != "complete"
        or EYE_ANGLE_STORAGE_CANDIDATE_ATTR in source_attrs
        or EYE_ANGLE_STORAGE_PLAN_ATTR in source_attrs
    ):
        raise ValueError("Eye-angle source run-group binding is invalid.")
    if (
        candidate_attrs.get(EYE_ANGLE_ARRAY_SCHEMA_ATTR) != candidate_manifest
        or candidate_attrs.get(EYE_ANGLE_STORAGE_PLAN_ATTR) != candidate_plan
        or candidate_attrs.get(EYE_ANGLE_STORAGE_CANDIDATE_ATTR) != candidate_envelope
        or candidate_attrs.get("cluster_output_staging") != candidate_publication
        or candidate_attrs.get("stage_selector_eligible") is not False
        or candidate_attrs.get("palette_run_completion_status") != "complete"
        or candidate_attrs.get("publication_scope")
        != "storage_benchmark_candidate_only"
    ):
        raise ValueError("Eye-angle candidate run-group binding is invalid.")
    required_contracts = {
        "eye_angle_output_schema",
        "eye_angle_variant_schema",
        "eye_angle_algorithm_contract",
        "eye_angle_source_contracts",
    }
    for attrs, expected in (
        (source_attrs, source_contract_digests),
        (candidate_attrs, candidate_contract_digests),
    ):
        if set(expected) != required_contracts:
            raise ValueError("Eye-angle contract-digest field set is incomplete.")
        if validate_eye_angle_persisted_contract_manifests(attrs):
            raise ValueError("Eye-angle embedded executable contracts are invalid.")
        observed = {
            name: canonical_json_sha256(attrs.get(name))
            for name in sorted(required_contracts)
        }
        if observed != expected:
            raise ValueError("Eye-angle embedded contract digest binding differs.")
    if any(
        source_attrs.get(name) != candidate_attrs.get(name)
        for name in required_contracts
    ):
        raise ValueError("Source/candidate scientific and lineage contracts differ.")


def require_workload(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=WORKLOAD_SCHEMA_ID)
    fields_ = {
        "family_id",
        "benchmark_id",
        "workload_id",
        "archive",
        "parent_path",
        "source_run",
        "candidate_run",
        "source_run_path",
        "candidate_run_path",
        "seed",
        "repetitions",
        "array_order",
        "dimensions",
        "source_array_schema_manifest",
        "candidate_array_schema_manifest",
        "candidate_storage_receipt",
        "candidate_envelope",
        "candidate_publication_receipt",
        "candidate_publication_receipt_digest",
        "parent_selector_snapshot",
        "source_metadata_equivalence",
        "candidate_metadata_equivalence",
        "source_metadata_declarations",
        "candidate_metadata_declarations",
        "source_contract_digests",
        "candidate_contract_digests",
        "physical_array_layouts_differ",
        "logical_table_digest",
        "expected_arrays",
        "expected_arrays_digest",
        "expected_decoded_nbytes",
        "dependency_paths",
        "dependency_metadata_declarations",
        "dependency_metadata_declarations_digest",
        "candidate_profile_id",
        "source_profile_id",
        "candidate_selector_eligible",
        "promotion_authorized",
        "palette_consumer_implemented",
        "candidate_adapter_scope",
        "physical_io_availability",
    }
    if set(payload) != fields_:
        raise ValueError("Eye-angle workload field set is not exact.")
    if (
        payload["family_id"] != FAMILY_ID
        or payload["benchmark_id"] != BENCHMARK_ID
        or payload["workload_id"] != WORKLOAD_ID
        or payload["parent_path"] != EYE_ANGLE_RUN_PARENT
    ):
        raise ValueError("Eye-angle workload identity is invalid.")
    archive = _safe_archive(payload["archive"])
    source = _safe_run_name(payload["source_run"], label="source run")
    candidate = _safe_run_name(payload["candidate_run"], label="candidate run")
    if (
        source == candidate
        or payload["source_run_path"] != _run_path(source)
        or payload["candidate_run_path"] != _run_path(candidate)
    ):
        raise ValueError("Eye-angle workload run binding is invalid.")
    _safe_run_directory(archive, name=source)
    _safe_run_directory(archive, name=candidate)
    if (
        type(payload["seed"]) is not int
        or type(payload["repetitions"]) is not int
        or payload["repetitions"] < 1
    ):
        raise ValueError("Eye-angle workload matrix dimensions are invalid.")
    dimensions = _dimensions_from_manifest(payload["dimensions"])
    if dimensions.contract_dimensions != payload["dimensions"]:
        raise ValueError("Eye-angle executable dimensions differ.")
    source_manifest = eye_angle_array_schema_manifest(
        dimensions, byte_planner_adopted=False
    )
    candidate_manifest = eye_angle_array_schema_manifest(
        dimensions, byte_planner_adopted=True
    )
    if (
        payload["source_array_schema_manifest"] != source_manifest
        or payload["candidate_array_schema_manifest"] != candidate_manifest
        or payload["candidate_envelope"] != _exact_candidate_envelope()
        or payload["candidate_profile_id"]
        != EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
        or payload["source_profile_id"] != EYE_ANGLE_LEGACY_EXPLICIT_STORAGE
        or payload["physical_array_layouts_differ"] is not True
    ):
        raise ValueError(
            "Eye-angle schema/profile evidence differs from executable policy."
        )
    candidate_plan = build_eye_angle_candidate_storage_plan(dimensions).as_manifest()
    if payload["candidate_storage_receipt"] != candidate_plan:
        raise ValueError("Eye-angle storage receipt differs from executable byte plan.")
    paths = [item.path for item in build_eye_angle_array_declarations()]
    if payload["array_order"] != sorted(paths) or len(payload["array_order"]) != 41:
        raise ValueError("Eye-angle workload array inventory is not exact.")
    expected_arrays = payload["expected_arrays"]
    if not isinstance(expected_arrays, Mapping) or set(expected_arrays) != set(paths):
        raise ValueError("Eye-angle expected decoded array inventory differs.")
    total_bytes = 0
    for path, item in expected_arrays.items():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"dtype", "shape", "decoded_nbytes", "decoded_sha256"}
            or not _is_sha256(item["decoded_sha256"])
            or type(item["decoded_nbytes"]) is not int
            or item["decoded_nbytes"] < 0
            or type(item["shape"]) is not list
            or any(type(extent) is not int or extent < 0 for extent in item["shape"])
        ):
            raise ValueError(f"Eye-angle expected array entry is invalid at {path!r}.")
        total_bytes += item["decoded_nbytes"]
    if (
        payload["expected_arrays_digest"] != canonical_json_sha256(expected_arrays)
        or payload["expected_decoded_nbytes"] != total_bytes
        or not _is_sha256(payload["logical_table_digest"])
    ):
        raise ValueError("Eye-angle decoded/logical digest evidence is invalid.")
    for label in ("source", "candidate"):
        receipt = payload[f"{label}_metadata_equivalence"]
        _require_metadata_receipt(
            receipt, run_path=payload[f"{label}_run_path"], array_count=41
        )
    source_declarations = _require_metadata_declarations(
        payload["source_metadata_declarations"],
        receipt=payload["source_metadata_equivalence"],
        run_path=payload["source_run_path"],
    )
    candidate_declarations = _require_metadata_declarations(
        payload["candidate_metadata_declarations"],
        receipt=payload["candidate_metadata_equivalence"],
        run_path=payload["candidate_run_path"],
    )
    _require_source_array_declarations(
        source_declarations,
        run_path=payload["source_run_path"],
        dimensions=dimensions,
    )
    if _physical_array_declarations(
        source_declarations, run_path=payload["source_run_path"]
    ) == _physical_array_declarations(
        candidate_declarations, run_path=payload["candidate_run_path"]
    ):
        raise ValueError(
            "Eye-angle evidence does not contain distinct physical layouts."
        )
    _require_candidate_physical_declarations(
        candidate_declarations,
        run_path=payload["candidate_run_path"],
        dimensions=dimensions,
        receipt=payload["candidate_storage_receipt"],
    )
    _require_run_group_bindings(
        source_declarations,
        candidate_declarations,
        source_path=payload["source_run_path"],
        candidate_path=payload["candidate_run_path"],
        source_manifest=payload["source_array_schema_manifest"],
        candidate_manifest=payload["candidate_array_schema_manifest"],
        candidate_plan=payload["candidate_storage_receipt"],
        candidate_envelope=payload["candidate_envelope"],
        candidate_publication=payload["candidate_publication_receipt"],
        source_contract_digests=payload["source_contract_digests"],
        candidate_contract_digests=payload["candidate_contract_digests"],
    )
    candidate_group = candidate_declarations[payload["candidate_run_path"]]
    candidate_attrs = candidate_group["attributes"]
    parent_snapshot = payload["parent_selector_snapshot"]
    if (
        not isinstance(parent_snapshot, Mapping)
        or parent_snapshot.get("latest") != source
        or parent_snapshot.get("latest_complete") != source
        or candidate in parent_snapshot.values()
    ):
        raise ValueError("Eye-angle candidate did not preserve source selectors.")
    publication = _require_publication_receipt(
        payload["candidate_publication_receipt"],
        archive=archive,
        run_path=payload["candidate_run_path"],
        owner_uuid=candidate_attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR),
        parent_attrs=parent_snapshot,
        candidate_attrs=candidate_attrs,
        dimensions=dimensions,
    )
    if payload["candidate_publication_receipt_digest"] != canonical_json_sha256(
        publication
    ):
        raise ValueError("Eye-angle publication receipt digest differs.")
    dependencies = payload["dependency_paths"]
    if (
        type(dependencies) is not list
        or any(type(path) is not str for path in dependencies)
        or dependencies != sorted(set(dependencies))
    ):
        raise ValueError("Eye-angle dependency path evidence is invalid.")
    for path in dependencies:
        pure = PurePosixPath(path)
        if path.startswith("/") or ".." in pure.parts or not pure.parts:
            raise ValueError("Eye-angle dependency path is unsafe.")
    dependency_digest = _require_source_contract_dependency_binding(
        source_declarations[payload["source_run_path"]]["attributes"][
            "eye_angle_source_contracts"
        ],
        payload["dependency_metadata_declarations"],
    )
    if (
        payload["dependency_metadata_declarations_digest"] != dependency_digest
        or set(payload["dependency_metadata_declarations"]) != set(dependencies)
        or _dependency_declarations(archive, paths=dependencies)
        != payload["dependency_metadata_declarations"]
    ):
        raise ValueError("Eye-angle dependency metadata binding differs.")
    if _metadata_declarations(
        archive,
        run_path=payload["source_run_path"],
    ) != source_declarations:
        raise ValueError("Source physical metadata declarations changed or were rehashed.")
    if (
        payload["candidate_selector_eligible"] is not False
        or payload["promotion_authorized"] is not False
        or payload["palette_consumer_implemented"] is not False
        or payload["candidate_adapter_scope"]
        != "diagnostic_only_private_strict_payload_adapter"
        or payload["physical_io_availability"] != _PHYSICAL_IO_AVAILABILITY
    ):
        raise ValueError("Eye-angle workload violates nonpromotion/telemetry policy.")


def _require_scan(value: object, *, expected: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "array_count",
        "decoded_nbytes",
        "wall_seconds",
        "cpu_seconds",
        "arrays",
        "arrays_digest",
    }:
        raise ValueError("Eye-angle full-scan field set is not exact.")
    if (
        value["array_count"] != len(expected)
        or value["arrays"] != expected
        or value["decoded_nbytes"]
        != sum(item["decoded_nbytes"] for item in expected.values())
        or value["arrays_digest"] != canonical_json_sha256(expected)
        or isinstance(value["wall_seconds"], bool)
        or not isinstance(value["wall_seconds"], (int, float))
        or not math.isfinite(float(value["wall_seconds"]))
        or value["wall_seconds"] < 0
        or isinstance(value["cpu_seconds"], bool)
        or not isinstance(value["cpu_seconds"], (int, float))
        or not math.isfinite(float(value["cpu_seconds"]))
        or value["cpu_seconds"] < 0
    ):
        raise ValueError("Eye-angle full-scan evidence differs from workload.")


def run_single_trial(
    archive_value: Path | str,
    *,
    source_run: str,
    candidate_run: str,
    role: str,
    repetition_index: int,
    order_position: int,
    driver_process_id: int,
    seed: int,
    cache_state: str,
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    require_workload(workload)
    expected = workload["payload"]
    archive = _safe_archive(archive_value)
    source = _safe_run_name(source_run, label="source run")
    candidate = _safe_run_name(candidate_run, label="candidate run")
    if (
        str(archive) != expected["archive"]
        or source != expected["source_run"]
        or candidate != expected["candidate_run"]
        or seed != expected["seed"]
    ):
        raise ValueError("Eye-angle trial invocation differs from workload.")
    if role not in {"source", "candidate"}:
        raise ValueError("Eye-angle trial role is invalid.")
    order = _trial_order(seed=seed, repetition_index=repetition_index)
    if order_position not in {0, 1} or order[order_position] != role:
        raise ValueError("Eye-angle trial rotation binding is invalid.")
    if (
        type(driver_process_id) is not int
        or driver_process_id <= 0
        or driver_process_id == os.getpid()
        or driver_process_id != os.getppid()
    ):
        raise ValueError(
            "Eye-angle trial driver PID must equal the live parent and differ from "
            "the child."
        )
    if (
        type(cache_state) is not str
        or not cache_state
        or cache_state != cache_state.strip()
    ):
        raise ValueError("Eye-angle cache-state label is invalid.")
    run_name = source if role == "source" else candidate
    run_path = _run_path(run_name)
    if (
        _dependency_declarations(archive, paths=expected["dependency_paths"])
        != expected["dependency_metadata_declarations"]
    ):
        raise ValueError(
            "Eye-angle source dependency metadata changed after preflight."
        )
    started = utc_now()
    root, open_timing = _measure(lambda: _open_root(archive, consolidated=True))
    run = _group(root, run_path)
    validation, validation_timing = _measure(
        lambda: (
            _validate_source(root, run, run_name=run_name)
            if role == "source"
            else _validate_candidate(run, run_name=run_name)
        )
    )
    metadata, metadata_timing = _measure(
        lambda: _metadata_receipt(archive, run_path=run_path)
    )
    if metadata != expected[f"{role}_metadata_equivalence"]:
        raise ValueError("Eye-angle trial metadata receipt differs from workload.")
    scan = _full_scan(run, paths=expected["array_order"])
    if scan["arrays"] != expected["expected_arrays"]:
        raise ValueError("Eye-angle trial decoded values differ from workload.")
    if validation["logical_table_digest"] != expected["logical_table_digest"]:
        raise ValueError("Eye-angle trial logical-table result differs from workload.")
    payload = {
        "family_id": FAMILY_ID,
        "benchmark_id": BENCHMARK_ID,
        "workload_id": WORKLOAD_ID,
        "workload_payload_digest": workload["payload_digest"],
        "archive": str(archive),
        "source_run": source,
        "candidate_run": candidate,
        "role": role,
        "run_name": run_name,
        "run_path": run_path,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "seed": seed,
        "cache_state": cache_state,
        "process_id": os.getpid(),
        "driver_process_id": driver_process_id,
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "environment": _environment(archive=archive, cache_state=cache_state),
        "timings": {
            "consolidated_open": open_timing,
            "strict_validation_and_reader": validation_timing,
            "metadata_equivalence": metadata_timing,
            "complete_array_scan": {
                "wall_seconds": scan["wall_seconds"],
                "cpu_seconds": scan["cpu_seconds"],
            },
        },
        "validation": validation,
        "metadata_equivalence": metadata,
        "full_scan": scan,
        "storage": storage_stats(_safe_run_directory(archive, name=run_name)),
        "peak_rss_bytes": peak_rss_bytes(),
        "physical_io": _physical_io(),
        "candidate_selector_eligible": False,
        "promotion_authorized": False,
        "palette_consumer_implemented": False,
        "candidate_adapter_scope": "diagnostic_only_private_strict_payload_adapter",
    }
    result = _envelope(TRIAL_SCHEMA_ID, payload)
    require_trial_result(result, workload=workload)
    return result


def require_trial_result(
    value: Mapping[str, Any], *, workload: Mapping[str, Any]
) -> None:
    require_workload(workload)
    expected = workload["payload"]
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    fields_ = {
        "family_id",
        "benchmark_id",
        "workload_id",
        "workload_payload_digest",
        "archive",
        "source_run",
        "candidate_run",
        "role",
        "run_name",
        "run_path",
        "repetition_index",
        "order_position",
        "seed",
        "cache_state",
        "process_id",
        "driver_process_id",
        "started_at_utc",
        "completed_at_utc",
        "environment",
        "timings",
        "validation",
        "metadata_equivalence",
        "full_scan",
        "storage",
        "peak_rss_bytes",
        "physical_io",
        "candidate_selector_eligible",
        "promotion_authorized",
        "palette_consumer_implemented",
        "candidate_adapter_scope",
    }
    if set(payload) != fields_:
        raise ValueError("Eye-angle trial field set is not exact.")
    for name in (
        "family_id",
        "benchmark_id",
        "workload_id",
        "archive",
        "source_run",
        "candidate_run",
        "seed",
    ):
        if payload[name] != expected[name]:
            raise ValueError(f"Eye-angle trial {name} binding differs.")
    if payload["workload_payload_digest"] != workload["payload_digest"]:
        raise ValueError("Eye-angle trial workload digest differs.")
    role = payload["role"]
    if role not in {"source", "candidate"}:
        raise ValueError("Eye-angle trial role is invalid.")
    if (
        payload["run_name"] != expected[f"{role}_run"]
        or payload["run_path"] != expected[f"{role}_run_path"]
        or type(payload["repetition_index"]) is not int
        or not 0 <= payload["repetition_index"] < expected["repetitions"]
        or type(payload["order_position"]) is not int
        or payload["order_position"] not in {0, 1}
        or _trial_order(
            seed=payload["seed"], repetition_index=payload["repetition_index"]
        )[payload["order_position"]]
        != role
        or type(payload["process_id"]) is not int
        or payload["process_id"] <= 0
        or type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] <= 0
        or payload["driver_process_id"] == payload["process_id"]
    ):
        raise ValueError("Eye-angle trial role/order/process binding is invalid.")
    if any(
        type(payload[name]) is not str or not payload[name]
        for name in ("started_at_utc", "completed_at_utc", "cache_state")
    ):
        raise ValueError("Eye-angle trial timestamp/cache state is invalid.")
    _require_environment(payload["environment"], cache_state=payload["cache_state"])
    timings = payload["timings"]
    if not isinstance(timings, Mapping) or set(timings) != {
        "consolidated_open",
        "strict_validation_and_reader",
        "metadata_equivalence",
        "complete_array_scan",
    }:
        raise ValueError("Eye-angle trial timing field set is not exact.")
    for label, timing in timings.items():
        _require_timing(timing, label=label)
    validation = payload["validation"]
    required_validation = {
        "consumer_path",
        "dimensions",
        "logical_table_digest",
        "contract_digests",
    }
    if role == "candidate":
        required_validation.add("storage_receipt_digest")
    if not isinstance(validation, Mapping) or set(validation) != required_validation:
        raise ValueError("Eye-angle trial validation field set is not exact.")
    if (
        validation["dimensions"] != expected["dimensions"]
        or validation["logical_table_digest"] != expected["logical_table_digest"]
        or validation["contract_digests"] != expected[f"{role}_contract_digests"]
    ):
        raise ValueError("Eye-angle trial strict-reader binding differs.")
    expected_consumer = (
        "maintained_selector_eligible_compact_v7_reader"
        if role == "source"
        else "explicit_ineligible_strict_compact_v7_candidate_adapter"
    )
    if validation["consumer_path"] != expected_consumer:
        raise ValueError("Eye-angle trial consumer path is invalid.")
    if role == "candidate" and validation[
        "storage_receipt_digest"
    ] != canonical_json_sha256(expected["candidate_storage_receipt"]):
        raise ValueError("Eye-angle trial storage receipt binding differs.")
    if payload["metadata_equivalence"] != expected[f"{role}_metadata_equivalence"]:
        raise ValueError("Eye-angle trial metadata-equivalence result differs.")
    _require_scan(payload["full_scan"], expected=expected["expected_arrays"])
    storage = payload["storage"]
    if (
        not isinstance(storage, Mapping)
        or set(storage)
        != {
            "file_count",
            "metadata_file_count",
            "payload_file_count",
            "apparent_bytes",
            "allocated_bytes",
        }
        or any(type(item) is not int or item < 0 for item in storage.values())
    ):
        raise ValueError("Eye-angle trial storage facts are invalid.")
    if type(payload["peak_rss_bytes"]) is not int or payload["peak_rss_bytes"] <= 0:
        raise ValueError("Eye-angle trial peak RSS is invalid.")
    if payload["physical_io"] != _physical_io():
        raise ValueError("Eye-angle trial fabricated physical I/O telemetry.")
    if (
        payload["candidate_selector_eligible"] is not False
        or payload["promotion_authorized"] is not False
        or payload["palette_consumer_implemented"] is not False
        or payload["candidate_adapter_scope"]
        != "diagnostic_only_private_strict_payload_adapter"
    ):
        raise ValueError("Eye-angle trial violates hard nonpromotion policy.")


def _matrix_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("source", "candidate"):
        rows = [item["payload"] for item in trials if item["payload"]["role"] == role]
        result[role] = {
            "trial_count": len(rows),
            "median_open_wall_seconds": statistics.median(
                item["timings"]["consolidated_open"]["wall_seconds"] for item in rows
            ),
            "median_reader_wall_seconds": statistics.median(
                item["timings"]["strict_validation_and_reader"]["wall_seconds"]
                for item in rows
            ),
            "median_scan_wall_seconds": statistics.median(
                item["full_scan"]["wall_seconds"] for item in rows
            ),
            "median_peak_rss_bytes": statistics.median(
                item["peak_rss_bytes"] for item in rows
            ),
        }
    return result


def require_matrix_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    fields_ = {
        "family_id",
        "benchmark_id",
        "workload",
        "workload_payload_digest",
        "archive",
        "source_run",
        "candidate_run",
        "seed",
        "repetitions",
        "cache_state",
        "driver_process_id",
        "trials",
        "summary",
        "correctness",
        "archive_read_only_metadata_guard",
        "balanced_fresh_process_matrix_complete",
        "candidate_selector_eligible",
        "promotion_authorized",
        "physical_io_availability",
        "external_evidence_binding_required",
        "palette_consumer_implemented",
        "candidate_adapter_scope",
        "started_at_utc",
        "completed_at_utc",
    }
    if set(payload) != fields_:
        raise ValueError("Eye-angle matrix field set is not exact.")
    workload = payload["workload"]
    if not isinstance(workload, Mapping):
        raise ValueError("Eye-angle matrix workload is missing.")
    require_workload(workload)
    expected = workload["payload"]
    for name in (
        "family_id",
        "benchmark_id",
        "archive",
        "source_run",
        "candidate_run",
        "seed",
        "repetitions",
    ):
        if payload[name] != expected[name]:
            raise ValueError(f"Eye-angle matrix {name} binding differs.")
    if payload["workload_payload_digest"] != workload["payload_digest"]:
        raise ValueError("Eye-angle matrix workload digest differs.")
    if (
        type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] <= 0
        or type(payload["cache_state"]) is not str
        or not payload["cache_state"]
        or payload["cache_state"] != payload["cache_state"].strip()
    ):
        raise ValueError("Eye-angle matrix driver/cache binding is invalid.")
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != 2 * payload["repetitions"]:
        raise ValueError("Eye-angle matrix trial count is invalid.")
    process_ids: list[int] = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            raise ValueError("Eye-angle matrix trial is not an envelope.")
        require_trial_result(trial, workload=workload)
        trial_payload = trial["payload"]
        repetition = index // 2
        position = index % 2
        if (
            trial_payload["repetition_index"] != repetition
            or trial_payload["order_position"] != position
            or trial_payload["role"]
            != _trial_order(seed=payload["seed"], repetition_index=repetition)[position]
            or trial_payload["cache_state"] != payload["cache_state"]
            or trial_payload["driver_process_id"] != payload["driver_process_id"]
        ):
            raise ValueError("Eye-angle matrix trial order binding differs.")
        process_ids.append(trial_payload["process_id"])
    if (
        len(set(process_ids)) != len(process_ids)
        or payload["driver_process_id"] in process_ids
    ):
        raise ValueError(
            "Eye-angle matrix trials did not use distinct fresh processes."
        )
    if payload["summary"] != _matrix_summary(trials):
        raise ValueError(
            "Eye-angle matrix summary differs from executable aggregation."
        )
    if payload["correctness"] != {
        "all_passed": True,
        "decoded_arrays_equal": True,
        "source_public_reader_candidate_diagnostic_adapter_equal": True,
        "direct_consolidated_equal": True,
        "complete_contract_binding": True,
        "zero_archive_metadata_mutations": True,
    }:
        raise ValueError("Eye-angle matrix correctness receipt is not exact.")
    guard = payload["archive_read_only_metadata_guard"]
    if not isinstance(guard, Mapping) or set(guard) != {"before", "after", "unchanged"}:
        raise ValueError("Eye-angle matrix metadata guard is not exact.")
    _require_metadata_guard(guard["before"])
    _require_metadata_guard(guard["after"])
    if guard["unchanged"] is not True or guard["before"] != guard["after"]:
        raise ValueError("Eye-angle benchmark modified archive metadata.")
    if payload["balanced_fresh_process_matrix_complete"] is not (
        payload["repetitions"] == DEFAULT_REPETITIONS
    ):
        raise ValueError("Eye-angle balanced-matrix classification is invalid.")
    if (
        payload["candidate_selector_eligible"] is not False
        or payload["promotion_authorized"] is not False
        or payload["physical_io_availability"] != _PHYSICAL_IO_AVAILABILITY
        or payload["external_evidence_binding_required"]
        != "outer_sha256_or_signature_required_for_portable_authority"
        or payload["palette_consumer_implemented"] is not False
        or payload["candidate_adapter_scope"]
        != "diagnostic_only_private_strict_payload_adapter"
    ):
        raise ValueError("Eye-angle matrix violates evidence/nonpromotion policy.")


def run_benchmark_matrix(
    archive_value: Path | str,
    *,
    source_run: str,
    candidate_run: str,
    output_value: Path | str,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    cache_state: str = "fresh_process_os_cache_uncontrolled",
) -> dict[str, Any]:
    archive = _safe_archive(archive_value)
    output = _safe_output(output_value, archive=archive)
    workload = _preflight(
        archive,
        source_run_name=source_run,
        candidate_run_name=candidate_run,
        seed=seed,
        repetitions=repetitions,
    )
    output.mkdir(parents=True, exist_ok=False)
    workload_path = output / "workload.json"
    _write_json(workload_path, workload)
    expected = workload["payload"]
    before = _metadata_guard(
        archive,
        run_names=(expected["source_run"], expected["candidate_run"]),
        dependency_paths=expected["dependency_paths"],
    )
    started = utc_now()
    driver_process_id = os.getpid()
    trials: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for position, role in enumerate(
            _trial_order(seed=seed, repetition_index=repetition)
        ):
            trial_path = output / f"trial_{repetition:03d}_{position}_{role}.json"
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_eye_angle_v7_reads",
                "trial",
                "--archive",
                str(archive),
                "--source-run",
                expected["source_run"],
                "--candidate-run",
                expected["candidate_run"],
                "--role",
                role,
                "--repetition-index",
                str(repetition),
                "--order-position",
                str(position),
                "--driver-process-id",
                str(driver_process_id),
                "--seed",
                str(seed),
                "--cache-state",
                cache_state,
                "--workload",
                str(workload_path),
                "--output",
                str(trial_path),
                "--benchmark-root",
                str(output),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=dict(os.environ),
            )
            if completed.returncode:
                raise RuntimeError(
                    "Eye-angle fresh-process trial failed: "
                    f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
            trial = _read_json(trial_path)
            require_trial_result(trial, workload=workload)
            trials.append(trial)
    after = _metadata_guard(
        archive,
        run_names=(expected["source_run"], expected["candidate_run"]),
        dependency_paths=expected["dependency_paths"],
    )
    payload = {
        "family_id": FAMILY_ID,
        "benchmark_id": BENCHMARK_ID,
        "workload": workload,
        "workload_payload_digest": workload["payload_digest"],
        "archive": str(archive),
        "source_run": expected["source_run"],
        "candidate_run": expected["candidate_run"],
        "seed": seed,
        "repetitions": repetitions,
        "cache_state": cache_state,
        "driver_process_id": driver_process_id,
        "trials": trials,
        "summary": _matrix_summary(trials),
        "correctness": {
            "all_passed": True,
            "decoded_arrays_equal": True,
            "source_public_reader_candidate_diagnostic_adapter_equal": True,
            "direct_consolidated_equal": True,
            "complete_contract_binding": True,
            "zero_archive_metadata_mutations": before == after,
        },
        "archive_read_only_metadata_guard": {
            "before": before,
            "after": after,
            "unchanged": before == after,
        },
        "balanced_fresh_process_matrix_complete": repetitions == DEFAULT_REPETITIONS,
        "candidate_selector_eligible": False,
        "promotion_authorized": False,
        "physical_io_availability": _PHYSICAL_IO_AVAILABILITY,
        "external_evidence_binding_required": (
            "outer_sha256_or_signature_required_for_portable_authority"
        ),
        "palette_consumer_implemented": False,
        "candidate_adapter_scope": "diagnostic_only_private_strict_payload_adapter",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
    }
    result = _envelope(MATRIX_SCHEMA_ID, payload)
    require_matrix_result(result)
    _write_json(output / "matrix.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--archive", type=Path, required=True)
    matrix.add_argument("--source-run", required=True)
    matrix.add_argument("--candidate-run", required=True)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    matrix.add_argument("--seed", type=int, default=DEFAULT_SEED)
    matrix.add_argument("--cache-state", default="fresh_process_os_cache_uncontrolled")
    trial = subparsers.add_parser("trial")
    trial.add_argument("--archive", type=Path, required=True)
    trial.add_argument("--source-run", required=True)
    trial.add_argument("--candidate-run", required=True)
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--order-position", type=int, required=True)
    trial.add_argument("--driver-process-id", type=int, required=True)
    trial.add_argument("--seed", type=int, required=True)
    trial.add_argument("--cache-state", required=True)
    trial.add_argument("--workload", type=Path, required=True)
    trial.add_argument("--output", type=Path, required=True)
    trial.add_argument("--benchmark-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        run_benchmark_matrix(
            args.archive,
            source_run=args.source_run,
            candidate_run=args.candidate_run,
            output_value=args.output,
            repetitions=args.repetitions,
            seed=args.seed,
            cache_state=args.cache_state,
        )
        return 0
    workload = _read_json(args.workload)
    benchmark_root = args.benchmark_root.expanduser().resolve()
    output = _safe_trial_output(args.output, root=benchmark_root)
    result = run_single_trial(
        args.archive,
        source_run=args.source_run,
        candidate_run=args.candidate_run,
        role=args.role,
        repetition_index=args.repetition_index,
        order_position=args.order_position,
        driver_process_id=args.driver_process_id,
        seed=args.seed,
        cache_state=args.cache_state,
        workload=workload,
    )
    _write_json(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
