"""Build deterministic run-family and producer aggregates for a coordinate audit.

This utility is intentionally metadata-only.  It consumes an immutable audit
JSONL and reads only the metadata file for each exact run path already recorded
by that audit.  It never opens a Zarr store through the Zarr API and never
writes to a registry or archive.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from fisheye.utils.audit_coordinate_contracts import (
    AUDIT_SCHEMA_ID,
    AUDIT_SCHEMA_VERSION,
    NORMALIZED_ARTIFACT_FILENAMES,
    STATUSES,
    _atomic_write_text,
    _strict_json_loads,
    verify_normalized_artifact_generation,
)


AGGREGATE_SCHEMA_ID = "palette.coordinate_contract_audit.aggregate"
AGGREGATE_SCHEMA_VERSION = 1
_PORTABLE_RECORDING_BUCKET = "__portable_or_unbound__"
_NO_RUN_FAMILY = "__no_run_family__"
_SHA256_HEX_LENGTH = 64

_PRODUCER_METHOD_PATHS = (
    ("method",),
    ("lineage_payload_json", "method"),
)
_PRODUCER_METHOD_VERSION_PATHS = (
    ("method_version",),
    ("lineage_payload_json", "method_version"),
)
_PRODUCER_COMMIT_PATHS = (
    ("git_commit",),
    ("git_commit_hash",),
    ("provenance", "git", "commit"),
    ("run_provenance", "git_sha"),
    ("lineage_payload_json", "code", "git_commit"),
)
_SOFTWARE_VERSION_PATHS = (("run_provenance", "fisheye_version"),)
_RUN_SCHEMA_KEYS = (
    "schema_id",
    "schema_version",
    "analysis_schema_id",
    "analysis_schema_version",
    "run_schema_id",
    "run_schema_version",
    "output_schema_id",
    "output_schema_version",
    "palette_run_schema_id",
    "palette_run_schema_version",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _relative_run_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value.startswith("/"):
        return None
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    normalized = path.as_posix()
    if normalized != value:
        return None
    return normalized


def _absolute_zarr_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        return None
    return value


def _metadata_path(
    zarr_path: Path,
    run_path: str,
) -> tuple[Path | None, str, str | None]:
    root = zarr_path.resolve(strict=False)
    run_dir = root.joinpath(*PurePosixPath(run_path).parts)
    zarr_v3 = run_dir / "zarr.json"
    zarr_v2 = run_dir / ".zattrs"
    try:
        v3_present = zarr_v3.is_file()
        v2_present = zarr_v2.is_file()
    except (OSError, RuntimeError) as exc:
        return None, "unreadable", f"{type(exc).__name__}: {exc}"
    if v3_present and v2_present:
        return (
            None,
            "conflicting_formats",
            "both zarr.json and .zattrs are present at the exact run path",
        )
    if not v3_present and not v2_present:
        return None, "missing", None
    candidate = zarr_v3 if v3_present else zarr_v2
    metadata_format = "zarr_v3" if v3_present else "zarr_v2"
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return None, "unreadable", f"{type(exc).__name__}: {exc}"
    if not _is_relative_to(resolved, root):
        return (
            None,
            "unsafe_path",
            "run metadata resolves outside the declared Zarr root",
        )
    return resolved, metadata_format, None


def _parse_run_attributes(content: bytes, metadata_format: str) -> dict[str, Any]:
    raw = _strict_json_loads(content.decode("utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("run metadata root is not an object")
    if metadata_format == "zarr_v3":
        attrs = raw.get("attributes")
        if not isinstance(attrs, Mapping):
            raise ValueError("zarr.json attributes is not an object")
        return dict(attrs)
    return dict(raw)


def _candidate_value(
    attrs: Mapping[str, Any],
    path: Sequence[str],
) -> tuple[str, Any, str | None]:
    current: Any = attrs
    for index, part in enumerate(path):
        if part == "lineage_payload_json" and index == 0:
            if part not in attrs:
                return "missing", None, None
            raw = attrs.get(part)
            if not isinstance(raw, str) or not raw:
                return "invalid", None, "lineage_payload_json is not a non-empty string"
            try:
                current = _strict_json_loads(raw)
            except (TypeError, ValueError) as exc:
                return (
                    "invalid",
                    None,
                    f"lineage_payload_json is invalid: {type(exc).__name__}: {exc}",
                )
            if not isinstance(current, Mapping):
                return "invalid", None, "lineage_payload_json is not an object"
            continue
        if not isinstance(current, Mapping):
            return (
                "invalid",
                None,
                f"{'.'.join(path[:index])} is not an object",
            )
        if part not in current:
            return "missing", None, None
        current = current[part]
    if current in (None, ""):
        return "invalid", None, "declared value is null or empty"
    return "declared", current, None


def _declared_candidates(
    attrs: Mapping[str, Any],
    paths: Iterable[Sequence[str]],
    *,
    allow_integer: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        state, value, error = _candidate_value(attrs, path)
        if state == "missing":
            continue
        valid_scalar = (
            isinstance(value, str)
            and bool(value.strip())
            or allow_integer
            and isinstance(value, int)
            and not isinstance(value, bool)
        )
        candidate: dict[str, Any] = {
            "attribute_path": ".".join(path),
            "valid_scalar": state == "declared" and valid_scalar,
        }
        if state == "declared":
            candidate["value"] = value
        if error is not None:
            candidate["invalid_reason"] = error
        candidates.append(candidate)
    return candidates


def _resolve_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[Any, str]:
    valid_values = {
        _canonical_json(candidate.get("value")): candidate.get("value")
        for candidate in candidates
        if candidate.get("valid_scalar") is True
    }
    if any(candidate.get("valid_scalar") is not True for candidate in candidates):
        return None, "invalid"
    if not valid_values:
        return None, "missing"
    if len(valid_values) > 1:
        return None, "conflicting"
    return next(iter(valid_values.values())), "resolved"


def _producer_evidence(
    *,
    zarr_path: str,
    run_path: str | None,
) -> dict[str, Any]:
    base = {
        "zarr_path": zarr_path,
        "run_path": run_path,
        "metadata_path": None,
        "metadata_format": None,
        "metadata_sha256": None,
        "metadata_status": "run_context_missing",
        "method": None,
        "method_version": None,
        "git_commit": None,
        "software_version": None,
        "producer_status": "unavailable",
        "producer_key": "unavailable:run_context_missing",
        "method_candidates": [],
        "method_version_candidates": [],
        "git_commit_candidates": [],
        "software_version_candidates": [],
        "run_schema_evidence": {},
    }
    if run_path is None:
        return base

    try:
        root = Path(zarr_path).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        return {
            **base,
            "metadata_format": "unsafe_path",
            "metadata_status": "invalid_location",
            "metadata_error": f"{type(exc).__name__}: {exc}",
            "producer_key": "unavailable:run_metadata_unsafe_path",
        }
    metadata_path, metadata_format, path_error = _metadata_path(root, run_path)
    if metadata_path is None:
        metadata_status = (
            "missing" if metadata_format == "missing" else "invalid_location"
        )
        unavailable_reason = {
            "missing": "run_metadata_missing",
            "conflicting_formats": "run_metadata_conflicting_formats",
            "unsafe_path": "run_metadata_unsafe_path",
            "unreadable": "run_metadata_unreadable",
        }.get(metadata_format, "run_metadata_invalid_location")
        return {
            **base,
            "metadata_format": metadata_format,
            "metadata_status": metadata_status,
            "metadata_error": path_error,
            "producer_key": f"unavailable:{unavailable_reason}",
        }

    relative_metadata_path = metadata_path.relative_to(root).as_posix()
    metadata_content: bytes | None = None
    metadata_sha256: str | None = None
    try:
        metadata_content = metadata_path.read_bytes()
        metadata_sha256 = hashlib.sha256(metadata_content).hexdigest()
        attrs = _parse_run_attributes(metadata_content, metadata_format)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        return {
            **base,
            "metadata_path": relative_metadata_path,
            "metadata_format": metadata_format,
            "metadata_sha256": metadata_sha256,
            "metadata_status": "invalid",
            "metadata_error": f"{type(exc).__name__}: {exc}",
            "producer_key": "unavailable:run_metadata_invalid",
        }

    candidate_groups = {
        "method": _declared_candidates(attrs, _PRODUCER_METHOD_PATHS),
        "method_version": _declared_candidates(
            attrs,
            _PRODUCER_METHOD_VERSION_PATHS,
            allow_integer=True,
        ),
        "git_commit": _declared_candidates(attrs, _PRODUCER_COMMIT_PATHS),
        "software_version": _declared_candidates(attrs, _SOFTWARE_VERSION_PATHS),
    }
    resolved: dict[str, Any] = {}
    states: dict[str, str] = {}
    for field, candidates in candidate_groups.items():
        resolved[field], states[field] = _resolve_candidates(candidates)

    if any(state in {"invalid", "conflicting"} for state in states.values()):
        producer_status = "conflicting_or_invalid"
        producer_key = f"conflicting:{_fingerprint(candidate_groups)[:16]}"
    elif resolved["method"] is None and resolved["git_commit"] is None:
        producer_status = "unavailable"
        producer_key = "unavailable:no_method_or_commit"
    else:
        producer_status = "resolved"
        producer_key = _canonical_json(
            {
                "method": resolved["method"],
                "method_version": resolved["method_version"],
                "git_commit": resolved["git_commit"],
                "software_version": resolved["software_version"],
            }
        )

    return {
        **base,
        "metadata_path": relative_metadata_path,
        "metadata_format": metadata_format,
        "metadata_sha256": metadata_sha256,
        "metadata_status": "readable",
        **resolved,
        "producer_status": producer_status,
        "producer_key": producer_key,
        "method_candidates": candidate_groups["method"],
        "method_version_candidates": candidate_groups["method_version"],
        "git_commit_candidates": candidate_groups["git_commit"],
        "software_version_candidates": candidate_groups["software_version"],
        "candidate_resolution": states,
        "run_schema_evidence": {
            key: attrs[key] for key in _RUN_SCHEMA_KEYS if key in attrs
        },
    }


def _iter_surface_records(
    path: Path,
    *,
    dataset_zarr_paths: set[str] | None = None,
) -> Iterable[dict[str, Any]]:
    dataset_record_keys: set[str] = set()
    dataset_zarr_by_key: dict[str, str] = {}
    surface_zarr_by_dataset: dict[str, set[str]] = defaultdict(set)
    surface_dataset_keys: set[str] = set()
    surface_keys: set[tuple[str, str, str]] = set()
    bundle_by_dataset: dict[str, str] = {}
    bundle_hashers: dict[str, Any] = {}
    bundle_record_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = _strict_json_loads(line)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid audit JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"audit line {line_number} is not an object")
            if record.get("audit_schema_id") != AUDIT_SCHEMA_ID:
                raise ValueError(
                    f"audit line {line_number} has an unsupported schema_id"
                )
            if record.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
                raise ValueError(
                    f"audit line {line_number} has an unsupported schema_version"
                )
            record_type = record.get("record_type")
            if record_type not in {"coordinate_dataset", "coordinate_surface"}:
                raise ValueError(
                    f"audit line {line_number} has an unsupported record_type"
                )
            dataset_key = record.get("dataset_key")
            if not isinstance(dataset_key, str) or not dataset_key.strip():
                raise ValueError(
                    f"audit line {line_number} has an invalid dataset_key"
                )
            bundle_sha256 = record.get("record_bundle_sha256")
            if not _is_sha256(bundle_sha256):
                raise ValueError(
                    f"audit line {line_number} has an invalid record_bundle_sha256"
                )
            prior_bundle = bundle_by_dataset.setdefault(dataset_key, bundle_sha256)
            if prior_bundle != bundle_sha256:
                raise ValueError(
                    f"audit line {line_number} conflicts with its dataset record bundle"
                )
            bundle_hasher = bundle_hashers.get(dataset_key)
            if bundle_hasher is None:
                bundle_hasher = hashlib.sha256()
                bundle_hasher.update(b"[")
                bundle_hashers[dataset_key] = bundle_hasher
            elif bundle_record_counts[dataset_key]:
                bundle_hasher.update(b",")
            unsigned_record = dict(record)
            unsigned_record.pop("record_bundle_sha256", None)
            bundle_hasher.update(_canonical_json(unsigned_record).encode("utf-8"))
            bundle_record_counts[dataset_key] += 1
            if record_type == "coordinate_dataset":
                if dataset_key in dataset_record_keys:
                    raise ValueError(
                        f"audit line {line_number} duplicates a coordinate_dataset"
                    )
                zarr_path = _absolute_zarr_path(record.get("zarr_path"))
                if zarr_path is None:
                    raise ValueError(
                        f"audit line {line_number} has an unsafe or invalid "
                        "coordinate_dataset zarr_path"
                    )
                dataset_record_keys.add(dataset_key)
                dataset_zarr_by_key[dataset_key] = zarr_path
                if dataset_zarr_paths is not None:
                    dataset_zarr_paths.add(zarr_path)
                continue

            zarr_path = _absolute_zarr_path(record.get("zarr_path"))
            if zarr_path is None:
                raise ValueError(
                    f"audit line {line_number} has an unsafe or non-canonical zarr_path"
                )
            surface_path = _relative_run_path(record.get("surface_path"))
            if surface_path is None:
                raise ValueError(
                    f"audit line {line_number} has an unsafe or invalid surface_path"
                )
            surface_type = record.get("surface_type")
            if not isinstance(surface_type, str) or not surface_type.strip():
                raise ValueError(
                    f"audit line {line_number} has an invalid surface_type"
                )
            recording_id = record.get("recording_id")
            if recording_id is not None and (
                not isinstance(recording_id, str) or not recording_id.strip()
            ):
                raise ValueError(
                    f"audit line {line_number} has an invalid recording_id"
                )
            if recording_id == _PORTABLE_RECORDING_BUCKET:
                raise ValueError(
                    f"audit line {line_number} uses a reserved recording_id"
                )
            status = record.get("status")
            if status not in STATUSES:
                raise ValueError(f"audit line {line_number} has an invalid status")
            issue_codes = record.get("issue_codes")
            if (
                not isinstance(issue_codes, list)
                or any(
                    not isinstance(code, str) or not code.strip()
                    for code in issue_codes
                )
                or len(issue_codes) != len(set(issue_codes))
            ):
                raise ValueError(
                    f"audit line {line_number} has invalid or duplicate issue_codes"
                )
            run_context = record.get("run_context")
            if not isinstance(run_context, Mapping):
                raise ValueError(
                    f"audit line {line_number} has an invalid run_context"
                )
            family = run_context.get("family")
            raw_run_path = run_context.get("run_path")
            if family is None and raw_run_path is None:
                pass
            elif (
                not isinstance(family, str)
                or not family.strip()
                or _relative_run_path(raw_run_path) is None
            ):
                raise ValueError(
                    f"audit line {line_number} has an invalid run family/path context"
                )
            if family == _NO_RUN_FAMILY:
                raise ValueError(f"audit line {line_number} uses a reserved run family")
            surface_key = (dataset_key, zarr_path, surface_path)
            if surface_key in surface_keys:
                raise ValueError(
                    f"audit line {line_number} duplicates a coordinate surface"
                )
            surface_keys.add(surface_key)
            surface_dataset_keys.add(dataset_key)
            surface_zarr_by_dataset[dataset_key].add(zarr_path)
            yield dict(record)

    if not dataset_record_keys:
        raise ValueError("audit inventory contains no coordinate_dataset records")
    missing_dataset_records = sorted(surface_dataset_keys - dataset_record_keys)
    if missing_dataset_records:
        raise ValueError(
            "coordinate surfaces lack coordinate_dataset records: "
            + ", ".join(missing_dataset_records[:5])
        )
    mismatched_dataset_roots = sorted(
        dataset_key
        for dataset_key, zarr_paths in surface_zarr_by_dataset.items()
        if zarr_paths != {dataset_zarr_by_key[dataset_key]}
    )
    if mismatched_dataset_roots:
        raise ValueError(
            "coordinate surfaces disagree with their coordinate_dataset zarr_path: "
            + ", ".join(mismatched_dataset_roots[:5])
        )
    for dataset_key, bundle_hasher in bundle_hashers.items():
        bundle_hasher.update(b"]")
        if bundle_hasher.hexdigest() != bundle_by_dataset[dataset_key]:
            raise ValueError(
                f"coordinate dataset record bundle digest mismatch: {dataset_key}"
            )


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _verify_source_artifact_manifest(
    artifact_manifest: Path,
    inventory_jsonl: Path,
) -> tuple[dict[str, Any], list[str]]:
    manifest_path = artifact_manifest.resolve(strict=True)
    if manifest_path.name != "artifact_manifest.json":
        raise ValueError("source artifact manifest must be named artifact_manifest.json")
    raw_payload = _strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, Mapping):
        raise ValueError("source artifact manifest root is not an object")
    file_records = raw_payload.get("files")
    if not isinstance(file_records, Mapping):
        raise ValueError("source artifact manifest has no file records")

    bound_paths = {str(manifest_path)}
    for name, raw_record in file_records.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"source artifact file record is invalid: {name}")
        raw_path = raw_record.get("path")
        path_kind = raw_record.get("path_kind")
        if not isinstance(raw_path, str):
            raise ValueError(f"source artifact file path is invalid: {name}")
        if path_kind == "relative_to_manifest":
            relative_path = _relative_run_path(raw_path)
            if relative_path is None:
                raise ValueError(f"source artifact relative path is unsafe: {name}")
            resolved_path = manifest_path.parent.joinpath(
                *PurePosixPath(relative_path).parts
            ).resolve(strict=True)
            if not _is_relative_to(resolved_path, manifest_path.parent):
                raise ValueError(f"source artifact relative path escapes: {name}")
        elif path_kind == "absolute":
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                raise ValueError(f"source artifact absolute path is invalid: {name}")
            resolved_path = candidate.resolve(strict=True)
        else:
            raise ValueError(f"source artifact path kind is invalid: {name}")
        bound_paths.add(str(resolved_path))

    payload = verify_normalized_artifact_generation(manifest_path.parent)
    if payload.get("complete") is not True:
        raise ValueError("source artifact generation is not marked complete")
    inventory_record = file_records.get("external:inventory_jsonl")
    if not isinstance(inventory_record, Mapping):
        raise ValueError("source artifact manifest does not bind inventory_jsonl")
    if inventory_record.get("path_kind") != "absolute":
        raise ValueError("source inventory binding is not an absolute path")
    declared_inventory = inventory_record.get("path")
    if not isinstance(declared_inventory, str):
        raise ValueError("source inventory binding path is invalid")
    if Path(declared_inventory).resolve(strict=True) != inventory_jsonl:
        raise ValueError("source artifact manifest binds a different inventory_jsonl")
    inventory_sha256 = _file_sha256(inventory_jsonl)
    if inventory_record.get("sha256") != inventory_sha256:
        raise ValueError("source artifact manifest inventory digest mismatch")
    if inventory_record.get("size_bytes") != inventory_jsonl.stat().st_size:
        raise ValueError("source artifact manifest inventory size mismatch")

    expected_artifacts = {
        f"artifact:{name}"
        for name in NORMALIZED_ARTIFACT_FILENAMES
        if name != "artifact_manifest.json"
    }
    if not expected_artifacts <= set(file_records):
        raise ValueError("source artifact normalized file set is incomplete")
    return dict(payload), sorted(bound_paths)


def _summary_record(
    *,
    surface_count: int,
    run_ids: set[str],
    dataset_keys: set[str],
    recording_ids: set[str],
    portable_or_unbound_surface_count: int,
    statuses: Counter[str],
    issues: Counter[str],
) -> dict[str, Any]:
    bound_recording_ids = sorted(
        recording_id
        for recording_id in recording_ids
        if recording_id != _PORTABLE_RECORDING_BUCKET
    )
    return {
        "surface_count": surface_count,
        "run_count": len(run_ids),
        "dataset_count": len(dataset_keys),
        "affected_recording_count": len(bound_recording_ids),
        "affected_recording_ids": bound_recording_ids,
        "portable_or_unbound_surface_count": portable_or_unbound_surface_count,
        "status_counts": _counter_dict(statuses),
        "issue_counts": _counter_dict(issues),
    }


def build_coordinate_audit_aggregate(
    inventory_jsonl: Path,
    *,
    artifact_manifest: Path | None = None,
) -> dict[str, Any]:
    """Return an integrity-bound aggregate for one completed audit JSONL."""

    inventory_jsonl = inventory_jsonl.resolve(strict=True)
    if artifact_manifest is None:
        raise ValueError("artifact_manifest is required for a complete aggregate")
    generator_source = Path(__file__).resolve(strict=True)
    generator_source_sha256_before = _file_sha256(generator_source)
    inventory_sha256_before = _file_sha256(inventory_jsonl)
    source_manifest_payload: dict[str, Any] | None = None
    source_artifact_bound_files: list[str] = []
    artifact_manifest_path: Path | None = None
    artifact_manifest_sha256_before: str | None = None
    artifact_manifest_path = artifact_manifest.resolve(strict=True)
    source_manifest_payload, source_artifact_bound_files = (
        _verify_source_artifact_manifest(
            artifact_manifest_path,
            inventory_jsonl,
        )
    )
    artifact_manifest_sha256_before = _file_sha256(artifact_manifest_path)
    source_zarr_roots: set[str] = set()
    surfaces = list(
        _iter_surface_records(
            inventory_jsonl,
            dataset_zarr_paths=source_zarr_roots,
        )
    )

    run_keys: dict[str, tuple[str, str | None]] = {}
    surface_run_ids: list[str] = []
    for surface in surfaces:
        zarr_path = str(surface["zarr_path"])
        run_context = surface.get("run_context")
        raw_run_path = (
            run_context.get("run_path") if isinstance(run_context, Mapping) else None
        )
        run_path = _relative_run_path(raw_run_path)
        run_id = _fingerprint({"zarr_path": zarr_path, "run_path": run_path})
        run_keys[run_id] = (zarr_path, run_path)
        surface_run_ids.append(run_id)

    run_records: dict[str, dict[str, Any]] = {}
    for run_id, (zarr_path, run_path) in sorted(run_keys.items()):
        run_records[run_id] = {
            "run_id": run_id,
            **_producer_evidence(zarr_path=zarr_path, run_path=run_path),
        }

    changed_run_ids: list[str] = []
    for run_id, record in run_records.items():
        current_evidence = _producer_evidence(
            zarr_path=str(record["zarr_path"]),
            run_path=(
                str(record["run_path"])
                if isinstance(record.get("run_path"), str)
                else None
            ),
        )
        expected_evidence = dict(record)
        expected_evidence.pop("run_id", None)
        if current_evidence != expected_evidence:
            changed_run_ids.append(run_id)

    family_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "surface_count": 0,
            "run_ids": set(),
            "dataset_keys": set(),
            "recording_ids": set(),
            "portable_or_unbound_surface_count": 0,
            "statuses": Counter(),
            "issues": Counter(),
            "producers": Counter(),
        }
    )
    producer_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "surface_count": 0,
            "run_ids": set(),
            "dataset_keys": set(),
            "recording_ids": set(),
            "portable_or_unbound_surface_count": 0,
            "statuses": Counter(),
            "issues": Counter(),
            "families": Counter(),
        }
    )
    recording_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "surface_count": 0,
            "run_ids": set(),
            "dataset_keys": set(),
            "statuses": Counter(),
            "issues": Counter(),
            "families": Counter(),
            "producers": Counter(),
        }
    )
    issue_family_state: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "occurrence_count": 0,
            "recording_ids": set(),
            "portable_or_unbound_occurrence_count": 0,
        }
    )

    for surface, run_id in zip(surfaces, surface_run_ids, strict=True):
        run_context = surface.get("run_context")
        raw_family = (
            run_context.get("family") if isinstance(run_context, Mapping) else None
        )
        family = str(raw_family or _NO_RUN_FAMILY)
        producer = str(run_records[run_id]["producer_key"])
        dataset_key = str(surface["dataset_key"])
        recording_id = str(surface.get("recording_id") or _PORTABLE_RECORDING_BUCKET)
        status = str(surface["status"])
        issue_codes = list(surface["issue_codes"])

        family_entry = family_state[family]
        family_entry["surface_count"] += 1
        family_entry["run_ids"].add(run_id)
        family_entry["dataset_keys"].add(dataset_key)
        family_entry["recording_ids"].add(recording_id)
        if recording_id == _PORTABLE_RECORDING_BUCKET:
            family_entry["portable_or_unbound_surface_count"] += 1
        family_entry["statuses"][status] += 1
        family_entry["producers"][producer] += 1

        producer_entry = producer_state[producer]
        producer_entry["surface_count"] += 1
        producer_entry["run_ids"].add(run_id)
        producer_entry["dataset_keys"].add(dataset_key)
        producer_entry["recording_ids"].add(recording_id)
        if recording_id == _PORTABLE_RECORDING_BUCKET:
            producer_entry["portable_or_unbound_surface_count"] += 1
        producer_entry["statuses"][status] += 1
        producer_entry["families"][family] += 1

        recording_entry = recording_state[recording_id]
        recording_entry["surface_count"] += 1
        recording_entry["run_ids"].add(run_id)
        recording_entry["dataset_keys"].add(dataset_key)
        recording_entry["statuses"][status] += 1
        recording_entry["families"][family] += 1
        recording_entry["producers"][producer] += 1

        for issue_code in issue_codes:
            family_entry["issues"][issue_code] += 1
            producer_entry["issues"][issue_code] += 1
            recording_entry["issues"][issue_code] += 1
            issue_family_entry = issue_family_state[(issue_code, family)]
            issue_family_entry["occurrence_count"] += 1
            if recording_id == _PORTABLE_RECORDING_BUCKET:
                issue_family_entry["portable_or_unbound_occurrence_count"] += 1
            else:
                issue_family_entry["recording_ids"].add(recording_id)

    by_run_family: dict[str, Any] = {}
    for family, state in sorted(family_state.items()):
        by_run_family[family] = {
            **_summary_record(
                surface_count=state["surface_count"],
                run_ids=state["run_ids"],
                dataset_keys=state["dataset_keys"],
                recording_ids=state["recording_ids"],
                portable_or_unbound_surface_count=state[
                    "portable_or_unbound_surface_count"
                ],
                statuses=state["statuses"],
                issues=state["issues"],
            ),
            "producer_surface_counts": _counter_dict(state["producers"]),
        }

    by_producer: dict[str, Any] = {}
    for producer, state in sorted(producer_state.items()):
        by_producer[producer] = {
            **_summary_record(
                surface_count=state["surface_count"],
                run_ids=state["run_ids"],
                dataset_keys=state["dataset_keys"],
                recording_ids=state["recording_ids"],
                portable_or_unbound_surface_count=state[
                    "portable_or_unbound_surface_count"
                ],
                statuses=state["statuses"],
                issues=state["issues"],
            ),
            "run_family_surface_counts": _counter_dict(state["families"]),
        }

    by_recording: dict[str, Any] = {}
    for recording_id, state in sorted(recording_state.items()):
        by_recording[recording_id] = {
            "surface_count": state["surface_count"],
            "run_count": len(state["run_ids"]),
            "dataset_count": len(state["dataset_keys"]),
            "status_counts": _counter_dict(state["statuses"]),
            "issue_counts": _counter_dict(state["issues"]),
            "run_family_surface_counts": _counter_dict(state["families"]),
            "producer_surface_counts": _counter_dict(state["producers"]),
        }

    bound_recording_ids = sorted(
        recording_id
        for recording_id in by_recording
        if recording_id != _PORTABLE_RECORDING_BUCKET
    )
    portable_or_unbound_surface_count = by_recording.get(
        _PORTABLE_RECORDING_BUCKET, {}
    ).get("surface_count", 0)

    issue_by_run_family = [
        {
            "issue_code": issue_code,
            "run_family": family,
            "occurrence_count": state["occurrence_count"],
            "affected_recording_count": len(state["recording_ids"]),
            "affected_recording_ids": sorted(state["recording_ids"]),
            "portable_or_unbound_occurrence_count": state[
                "portable_or_unbound_occurrence_count"
            ],
        }
        for (issue_code, family), state in sorted(issue_family_state.items())
    ]

    inventory_sha256_after = _file_sha256(inventory_jsonl)
    artifact_manifest_sha256_after = (
        _file_sha256(artifact_manifest_path)
        if artifact_manifest_path is not None
        else None
    )
    artifact_manifest_stable = (
        artifact_manifest_sha256_before is not None
        and artifact_manifest_sha256_before == artifact_manifest_sha256_after
    )
    persisted_run_context_count = sum(
        record.get("run_path") is not None for record in run_records.values()
    )
    no_run_context_count = len(run_records) - persisted_run_context_count
    generator_source_sha256_after = _file_sha256(generator_source)
    payload: dict[str, Any] = {
        "schema_id": AGGREGATE_SCHEMA_ID,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "generation_complete": inventory_sha256_before == inventory_sha256_after
        and not changed_run_ids
        and source_manifest_payload is not None
        and artifact_manifest_stable
        and generator_source_sha256_before == generator_source_sha256_after,
        "generator_source_path": "src/fisheye/utils/summarize_coordinate_audit.py",
        "generator_source_sha256": generator_source_sha256_before,
        "generator_source_stable_during_generation": (
            generator_source_sha256_before == generator_source_sha256_after
        ),
        "inventory_jsonl": str(inventory_jsonl),
        "inventory_sha256": inventory_sha256_before,
        "inventory_stable_during_generation": (
            inventory_sha256_before == inventory_sha256_after
        ),
        "source_artifact_manifest": (
            str(artifact_manifest_path) if artifact_manifest_path is not None else None
        ),
        "source_artifact_manifest_sha256": artifact_manifest_sha256_before,
        "source_artifact_manifest_stable_during_generation": (
            artifact_manifest_stable
        ),
        "source_artifact_generation_sha256": (
            source_manifest_payload.get("generation_sha256")
            if source_manifest_payload is not None
            else None
        ),
        "source_artifact_integrity_verified": source_manifest_payload is not None,
        "source_artifact_bound_files": source_artifact_bound_files,
        "source_zarr_roots": sorted(source_zarr_roots),
        "surface_count": len(surfaces),
        "run_count": len(run_records),
        "persisted_run_context_count": persisted_run_context_count,
        "no_run_context_count": no_run_context_count,
        "run_family_count": len(by_run_family),
        "producer_key_count": len(by_producer),
        "recording_count": len(bound_recording_ids),
        "recording_bucket_count": len(by_recording),
        "bound_recording_ids": bound_recording_ids,
        "portable_or_unbound_surface_count": portable_or_unbound_surface_count,
        "changed_run_ids": sorted(changed_run_ids),
        "run_records": [run_records[key] for key in sorted(run_records)],
        "by_run_family": by_run_family,
        "by_producer": by_producer,
        "by_recording": by_recording,
        "issue_by_run_family": issue_by_run_family,
    }
    payload["aggregate_payload_sha256"] = _fingerprint(payload)
    return payload


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _sorted_unique_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and value == sorted(set(value))
    )


def _valid_count_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and key and _nonnegative_int(count)
        for key, count in value.items()
    )


def _valid_summary_record(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    integer_fields = (
        "surface_count",
        "run_count",
        "dataset_count",
        "affected_recording_count",
        "portable_or_unbound_surface_count",
    )
    if any(not _nonnegative_int(value.get(field)) for field in integer_fields):
        return False
    recording_ids = value.get("affected_recording_ids")
    if not _sorted_unique_strings(recording_ids):
        return False
    if _PORTABLE_RECORDING_BUCKET in recording_ids:
        return False
    if value.get("affected_recording_count") != len(recording_ids):
        return False
    statuses = value.get("status_counts")
    issues = value.get("issue_counts")
    if not _valid_count_mapping(statuses) or not _valid_count_mapping(issues):
        return False
    return sum(statuses.values()) == value.get("surface_count")


def _load_verified_aggregate_sources(
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]] | None:
    try:
        raw_inventory_path = payload["inventory_jsonl"]
        raw_manifest_path = payload["source_artifact_manifest"]
        if not isinstance(raw_inventory_path, str) or not raw_inventory_path:
            return None
        if not isinstance(raw_manifest_path, str) or not raw_manifest_path:
            return None
        inventory_path = Path(raw_inventory_path).resolve(strict=True)
        manifest_path = Path(raw_manifest_path).resolve(strict=True)
        manifest_payload, bound_paths = _verify_source_artifact_manifest(
            manifest_path,
            inventory_path,
        )
        source_zarr_roots: set[str] = set()
        surfaces = list(
            _iter_surface_records(
                inventory_path,
                dataset_zarr_paths=source_zarr_roots,
            )
        )
        sorted_source_zarr_roots = sorted(source_zarr_roots)
        if not (
            _file_sha256(inventory_path) == payload.get("inventory_sha256")
            and _file_sha256(manifest_path)
            == payload.get("source_artifact_manifest_sha256")
            and manifest_payload.get("generation_sha256")
            == payload.get("source_artifact_generation_sha256")
            and bound_paths == payload.get("source_artifact_bound_files")
            and sorted_source_zarr_roots == payload.get("source_zarr_roots")
        ):
            return None
        return surfaces, sorted_source_zarr_roots
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def verify_coordinate_audit_aggregate(payload: Mapping[str, Any]) -> bool:
    """Return whether *payload* is complete and structurally self-consistent."""

    if payload.get("schema_id") != AGGREGATE_SCHEMA_ID:
        return False
    if payload.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        return False
    if payload.get("generation_complete") is not True:
        return False
    if payload.get("inventory_stable_during_generation") is not True:
        return False
    if payload.get("source_artifact_integrity_verified") is not True:
        return False
    if payload.get("source_artifact_manifest_stable_during_generation") is not True:
        return False
    if payload.get("generator_source_stable_during_generation") is not True:
        return False
    declared = payload.get("aggregate_payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("aggregate_payload_sha256", None)
    try:
        digest_matches = isinstance(declared, str) and declared == _fingerprint(
            unsigned
        )
    except (TypeError, UnicodeError, ValueError):
        return False
    if not digest_matches:
        return False

    top_count_fields = (
        "surface_count",
        "run_count",
        "persisted_run_context_count",
        "no_run_context_count",
        "run_family_count",
        "producer_key_count",
        "recording_count",
        "recording_bucket_count",
        "portable_or_unbound_surface_count",
    )
    if any(not _nonnegative_int(payload.get(field)) for field in top_count_fields):
        return False
    run_records = payload.get("run_records")
    by_run_family = payload.get("by_run_family")
    by_producer = payload.get("by_producer")
    by_recording = payload.get("by_recording")
    issue_by_run_family = payload.get("issue_by_run_family")
    if (
        not isinstance(run_records, list)
        or not isinstance(by_run_family, Mapping)
        or not isinstance(by_producer, Mapping)
        or not isinstance(by_recording, Mapping)
        or not isinstance(issue_by_run_family, list)
    ):
        return False
    if len(run_records) != payload.get("run_count"):
        return False
    run_ids = [
        record.get("run_id")
        for record in run_records
        if isinstance(record, Mapping)
    ]
    if len(run_ids) != len(run_records) or not _sorted_unique_strings(run_ids):
        return False
    run_records_by_id: dict[str, Mapping[str, Any]] = {}
    run_producers: dict[str, str] = {}
    for record in run_records:
        if not isinstance(record, Mapping):
            return False
        run_id = record.get("run_id")
        zarr_path = _absolute_zarr_path(record.get("zarr_path"))
        run_path = record.get("run_path")
        producer_key = record.get("producer_key")
        if not isinstance(run_id, str) or not _is_sha256(run_id):
            return False
        if zarr_path is None:
            return False
        if run_path is not None and _relative_run_path(run_path) != run_path:
            return False
        if not isinstance(producer_key, str) or not producer_key:
            return False
        if run_id != _fingerprint({"zarr_path": zarr_path, "run_path": run_path}):
            return False
        run_records_by_id[run_id] = record
        run_producers[run_id] = producer_key
    persisted_run_count = sum(
        isinstance(record.get("run_path"), str)
        for record in run_records
        if isinstance(record, Mapping)
    )
    if persisted_run_count != payload.get("persisted_run_context_count"):
        return False
    if len(run_records) - persisted_run_count != payload.get("no_run_context_count"):
        return False
    changed_run_ids = payload.get("changed_run_ids")
    if not _sorted_unique_strings(changed_run_ids):
        return False
    if not set(changed_run_ids) <= set(run_ids):
        return False
    if changed_run_ids:
        return False

    source_evidence = _load_verified_aggregate_sources(payload)
    if source_evidence is None:
        return False
    source_surfaces, source_zarr_roots = source_evidence
    if len(source_surfaces) != payload.get("surface_count"):
        return False
    if not set(record["zarr_path"] for record in run_records) <= set(
        source_zarr_roots
    ):
        return False

    family_run_ids: dict[str, set[str]] = defaultdict(set)
    family_dataset_keys: dict[str, set[str]] = defaultdict(set)
    family_recording_ids: dict[str, set[str]] = defaultdict(set)
    producer_run_ids: dict[str, set[str]] = defaultdict(set)
    producer_dataset_keys: dict[str, set[str]] = defaultdict(set)
    producer_recording_ids: dict[str, set[str]] = defaultdict(set)
    recording_run_ids: dict[str, set[str]] = defaultdict(set)
    recording_dataset_keys: dict[str, set[str]] = defaultdict(set)
    issue_family_occurrences: Counter[tuple[str, str]] = Counter()
    issue_family_recording_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    issue_family_portable_occurrences: Counter[tuple[str, str]] = Counter()
    source_run_ids: set[str] = set()
    source_portable_surface_count = 0
    for surface in source_surfaces:
        zarr_path = str(surface["zarr_path"])
        run_context = surface["run_context"]
        raw_run_path = run_context.get("run_path")
        run_path = _relative_run_path(raw_run_path)
        run_id = _fingerprint({"zarr_path": zarr_path, "run_path": run_path})
        run_record = run_records_by_id.get(run_id)
        if (
            run_record is None
            or run_record.get("zarr_path") != zarr_path
            or run_record.get("run_path") != run_path
        ):
            return False
        source_run_ids.add(run_id)
        family = str(run_context.get("family") or _NO_RUN_FAMILY)
        dataset_key = str(surface["dataset_key"])
        recording_id = str(
            surface.get("recording_id") or _PORTABLE_RECORDING_BUCKET
        )
        producer = run_producers[run_id]

        family_run_ids[family].add(run_id)
        family_dataset_keys[family].add(dataset_key)
        family_recording_ids[family].add(recording_id)
        producer_run_ids[producer].add(run_id)
        producer_dataset_keys[producer].add(dataset_key)
        producer_recording_ids[producer].add(recording_id)
        recording_run_ids[recording_id].add(run_id)
        recording_dataset_keys[recording_id].add(dataset_key)
        if recording_id == _PORTABLE_RECORDING_BUCKET:
            source_portable_surface_count += 1
        for issue_code in surface["issue_codes"]:
            issue_key = (str(issue_code), family)
            issue_family_occurrences[issue_key] += 1
            if recording_id == _PORTABLE_RECORDING_BUCKET:
                issue_family_portable_occurrences[issue_key] += 1
            else:
                issue_family_recording_ids[issue_key].add(recording_id)
    if source_run_ids != set(run_ids):
        return False

    if len(by_run_family) != payload.get("run_family_count"):
        return False
    if len(by_producer) != payload.get("producer_key_count"):
        return False
    if len(by_recording) != payload.get("recording_bucket_count"):
        return False
    if any(not _valid_summary_record(record) for record in by_run_family.values()):
        return False
    if any(not _valid_summary_record(record) for record in by_producer.values()):
        return False
    if set(by_run_family) != set(family_run_ids):
        return False
    if set(by_producer) != set(producer_run_ids):
        return False
    for family, family_record in by_run_family.items():
        expected_recording_ids = sorted(
            recording_id
            for recording_id in family_recording_ids[family]
            if recording_id != _PORTABLE_RECORDING_BUCKET
        )
        if (
            family_record.get("run_count") != len(family_run_ids[family])
            or family_record.get("dataset_count")
            != len(family_dataset_keys[family])
            or family_record.get("affected_recording_ids")
            != expected_recording_ids
        ):
            return False
    for producer, producer_record in by_producer.items():
        expected_recording_ids = sorted(
            recording_id
            for recording_id in producer_recording_ids[producer]
            if recording_id != _PORTABLE_RECORDING_BUCKET
        )
        if (
            producer_record.get("run_count") != len(producer_run_ids[producer])
            or producer_record.get("dataset_count")
            != len(producer_dataset_keys[producer])
            or producer_record.get("affected_recording_ids")
            != expected_recording_ids
        ):
            return False
    surface_count = payload.get("surface_count")
    if sum(record["surface_count"] for record in by_run_family.values()) != surface_count:
        return False
    if sum(record["surface_count"] for record in by_producer.values()) != surface_count:
        return False
    if set(by_producer) != set(run_producers.values()):
        return False
    family_producer_totals: Counter[str] = Counter()
    family_status_totals: Counter[str] = Counter()
    family_issue_totals: Counter[str] = Counter()
    for family_record in by_run_family.values():
        producer_counts = family_record.get("producer_surface_counts")
        if not _valid_count_mapping(producer_counts):
            return False
        if not set(producer_counts) <= set(by_producer):
            return False
        if sum(producer_counts.values()) != family_record.get("surface_count"):
            return False
        family_producer_totals.update(producer_counts)
        family_status_totals.update(family_record["status_counts"])
        family_issue_totals.update(family_record["issue_counts"])
    if family_producer_totals != Counter(
        {
            producer: producer_record["surface_count"]
            for producer, producer_record in by_producer.items()
        }
    ):
        return False
    producer_family_totals: Counter[str] = Counter()
    producer_status_totals: Counter[str] = Counter()
    producer_issue_totals: Counter[str] = Counter()
    for producer_record in by_producer.values():
        family_counts = producer_record.get("run_family_surface_counts")
        if not _valid_count_mapping(family_counts):
            return False
        if not set(family_counts) <= set(by_run_family):
            return False
        if sum(family_counts.values()) != producer_record.get("surface_count"):
            return False
        producer_family_totals.update(family_counts)
        producer_status_totals.update(producer_record["status_counts"])
        producer_issue_totals.update(producer_record["issue_counts"])
    if producer_family_totals != Counter(
        {
            family: family_record["surface_count"]
            for family, family_record in by_run_family.items()
        }
    ):
        return False
    if producer_status_totals != family_status_totals:
        return False
    if producer_issue_totals != family_issue_totals:
        return False
    if (
        sum(
            record["portable_or_unbound_surface_count"]
            for record in by_run_family.values()
        )
        != payload.get("portable_or_unbound_surface_count")
    ):
        return False
    if (
        sum(
            record["portable_or_unbound_surface_count"]
            for record in by_producer.values()
        )
        != payload.get("portable_or_unbound_surface_count")
    ):
        return False

    bound_recording_ids = payload.get("bound_recording_ids")
    if not _sorted_unique_strings(bound_recording_ids):
        return False
    if _PORTABLE_RECORDING_BUCKET in bound_recording_ids:
        return False
    if len(bound_recording_ids) != payload.get("recording_count"):
        return False
    source_bound_recording_ids = sorted(
        recording_id
        for recording_id in recording_run_ids
        if recording_id != _PORTABLE_RECORDING_BUCKET
    )
    if bound_recording_ids != source_bound_recording_ids:
        return False
    if source_portable_surface_count != payload.get(
        "portable_or_unbound_surface_count"
    ):
        return False
    expected_recording_buckets = set(recording_run_ids)
    if set(by_recording) != expected_recording_buckets:
        return False
    if any(
        not isinstance(record, Mapping)
        or not _nonnegative_int(record.get("surface_count"))
        or not _nonnegative_int(record.get("run_count"))
        or not _nonnegative_int(record.get("dataset_count"))
        or not _valid_count_mapping(record.get("status_counts"))
        or not _valid_count_mapping(record.get("issue_counts"))
        or not _valid_count_mapping(record.get("run_family_surface_counts"))
        or not _valid_count_mapping(record.get("producer_surface_counts"))
        or sum(record["status_counts"].values()) != record.get("surface_count")
        for record in by_recording.values()
    ):
        return False
    for recording_id, recording_record in by_recording.items():
        if (
            recording_record.get("run_count")
            != len(recording_run_ids[recording_id])
            or recording_record.get("dataset_count")
            != len(recording_dataset_keys[recording_id])
        ):
            return False
    recording_family_totals: Counter[str] = Counter()
    recording_producer_totals: Counter[str] = Counter()
    recording_status_totals: Counter[str] = Counter()
    recording_issue_totals: Counter[str] = Counter()
    for recording_record in by_recording.values():
        family_counts = recording_record["run_family_surface_counts"]
        producer_counts = recording_record["producer_surface_counts"]
        if not set(family_counts) <= set(by_run_family):
            return False
        if not set(producer_counts) <= set(by_producer):
            return False
        if sum(family_counts.values()) != recording_record.get("surface_count"):
            return False
        if sum(producer_counts.values()) != recording_record.get("surface_count"):
            return False
        recording_family_totals.update(family_counts)
        recording_producer_totals.update(producer_counts)
        recording_status_totals.update(recording_record["status_counts"])
        recording_issue_totals.update(recording_record["issue_counts"])
    if recording_family_totals != producer_family_totals:
        return False
    if recording_producer_totals != family_producer_totals:
        return False
    if recording_status_totals != family_status_totals:
        return False
    if recording_issue_totals != family_issue_totals:
        return False
    if sum(record["surface_count"] for record in by_recording.values()) != surface_count:
        return False
    portable_record = by_recording.get(_PORTABLE_RECORDING_BUCKET, {})
    if portable_record.get("surface_count", 0) != payload.get(
        "portable_or_unbound_surface_count"
    ):
        return False

    issue_keys: set[tuple[str, str]] = set()
    issue_family_counts: Counter[tuple[str, str]] = Counter()
    for record in issue_by_run_family:
        if not isinstance(record, Mapping):
            return False
        issue_code = record.get("issue_code")
        family = record.get("run_family")
        if not isinstance(issue_code, str) or not issue_code:
            return False
        if not isinstance(family, str) or family not in by_run_family:
            return False
        key = (issue_code, family)
        if key in issue_keys:
            return False
        issue_keys.add(key)
        affected_ids = record.get("affected_recording_ids")
        if not _sorted_unique_strings(affected_ids):
            return False
        if _PORTABLE_RECORDING_BUCKET in affected_ids:
            return False
        if record.get("affected_recording_count") != len(affected_ids):
            return False
        if not _nonnegative_int(record.get("occurrence_count")):
            return False
        if not _nonnegative_int(record.get("portable_or_unbound_occurrence_count")):
            return False
        if record["portable_or_unbound_occurrence_count"] > record["occurrence_count"]:
            return False
        if not set(affected_ids) <= set(bound_recording_ids):
            return False
        if (
            record.get("occurrence_count") != issue_family_occurrences[key]
            or affected_ids != sorted(issue_family_recording_ids[key])
            or record.get("portable_or_unbound_occurrence_count")
            != issue_family_portable_occurrences[key]
        ):
            return False
        issue_family_counts[key] = record["occurrence_count"]
    expected_issue_family_counts = Counter(
        {
            (issue_code, family): count
            for family, family_record in by_run_family.items()
            for issue_code, count in family_record["issue_counts"].items()
        }
    )
    if issue_family_counts != expected_issue_family_counts:
        return False
    if issue_keys != set(issue_family_occurrences):
        return False
    if payload.get("generator_source_path") != (
        "src/fisheye/utils/summarize_coordinate_audit.py"
    ):
        return False
    if not _is_sha256(payload.get("generator_source_sha256")):
        return False
    if not _is_sha256(payload.get("inventory_sha256")):
        return False
    if not _is_sha256(payload.get("source_artifact_manifest_sha256")):
        return False
    if not _is_sha256(payload.get("source_artifact_generation_sha256")):
        return False
    return True


def _validate_aggregate_output_path(path: Path, payload: Mapping[str, Any]) -> Path:
    resolved = path.resolve(strict=False)
    if resolved.suffix != ".json" or resolved.name in {"zarr.json", ".zattrs"}:
        raise ValueError("aggregate output must be a non-Zarr .json file")
    forbidden_paths: set[Path] = set()
    for field in ("inventory_jsonl", "source_artifact_manifest"):
        raw_path = payload.get(field)
        if isinstance(raw_path, str):
            forbidden_paths.add(Path(raw_path).resolve(strict=False))
    raw_bound_paths = payload.get("source_artifact_bound_files")
    if isinstance(raw_bound_paths, list):
        forbidden_paths.update(
            Path(raw_path).resolve(strict=False)
            for raw_path in raw_bound_paths
            if isinstance(raw_path, str)
    )
    if resolved in forbidden_paths:
        raise ValueError("aggregate output aliases a source audit artifact")
    raw_source_zarr_roots = payload.get("source_zarr_roots")
    if (
        not _sorted_unique_strings(raw_source_zarr_roots)
        or not raw_source_zarr_roots
    ):
        raise ValueError("aggregate payload has no valid source Zarr root set")
    for raw_zarr_path in raw_source_zarr_roots:
        if _absolute_zarr_path(raw_zarr_path) is None:
            raise ValueError("aggregate payload has an invalid source Zarr root")
        zarr_root = Path(raw_zarr_path).resolve(strict=False)
        if resolved == zarr_root or _is_relative_to(resolved, zarr_root):
            raise ValueError("aggregate output must be outside source Zarr roots")
    return resolved


def write_coordinate_audit_aggregate(path: Path, payload: Mapping[str, Any]) -> None:
    safe_path = _validate_aggregate_output_path(path, payload)
    _atomic_write_text(
        safe_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-jsonl", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = build_coordinate_audit_aggregate(
        args.inventory_jsonl,
        artifact_manifest=args.artifact_manifest,
    )
    write_coordinate_audit_aggregate(args.output_json, payload)
    print(
        json.dumps(
            {
                "generation_complete": payload["generation_complete"],
                "surface_count": payload["surface_count"],
                "run_count": payload["run_count"],
                "run_family_count": payload["run_family_count"],
                "producer_key_count": payload["producer_key_count"],
                "recording_count": payload["recording_count"],
                "aggregate_payload_sha256": payload["aggregate_payload_sha256"],
                "output_json": str(args.output_json.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["generation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
