"""Utilities for Palette virtual collection manifests.

This module intentionally avoids registry or Zarr access. It validates and
hashes manifest payloads that have already been assembled by higher-level
code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_ID = "palette.virtual_collection_manifest"
SCHEMA_VERSION = 1
CANONICALIZATION = "json_sorted_keys_no_hash_fields_v1"

REGISTRY_SNAPSHOT_STATUSES = {
    "recorded",
    "not_recorded",
    "not_registry_derived",
}
RUN_SELECTIONS = {
    "explicit",
    "resolved_latest",
    "query_default",
}
FINGERPRINT_STATUSES = {
    "complete",
    "best_effort",
    "missing",
    "not_applicable",
}
STIMULUS_RESPONSE_VALIDATION_STATUSES = {
    "pass",
    "warn",
    "fail",
    "not_applicable",
}


class VirtualCollectionManifestError(ValueError):
    """Raised when a virtual collection manifest is invalid."""


def _path_join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _normalize_json_value(value: Any, *, path: str = "$") -> Any:
    """Return a JSON-safe, Unicode-normalized copy of ``value``.

    The manifest hash contract rejects non-JSON numerics instead of silently
    converting them. Mapping keys must be strings because JSON object keys are
    strings.
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VirtualCollectionManifestError(f"{path}: non-finite float is not JSON-safe")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str):
                raise VirtualCollectionManifestError(
                    f"{path}: mapping key {raw_key!r} is not a string"
                )
            key = unicodedata.normalize("NFC", raw_key)
            if key in out:
                raise VirtualCollectionManifestError(
                    f"{path}: duplicate key after Unicode NFC normalization: {key!r}"
                )
            out[key] = _normalize_json_value(raw_item, path=_path_join(path, key))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json_value(item, path=_path_join(path, index))
            for index, item in enumerate(value)
        ]
    raise VirtualCollectionManifestError(
        f"{path}: unsupported value type {type(value).__name__}; expected JSON value"
    )


def canonical_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical hash payload for a collection manifest.

    Per the v1 contract, only ``manifest_sha256`` itself is excluded. The
    returned payload is normalized but not serialized.
    """

    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_sha256", None)
    return _normalize_json_value(payload)


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Serialize ``manifest`` using the v1 canonical JSON rules."""

    payload = canonical_manifest_payload(manifest)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def compute_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Compute the v1 ``manifest_sha256`` for a collection manifest."""

    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def with_manifest_sha256(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``manifest`` with ``manifest_sha256`` populated."""

    out = copy.deepcopy(dict(manifest))
    out["manifest_sha256"] = compute_manifest_sha256(out)
    return out


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return structural validation errors for a v1 virtual collection manifest."""

    errors: list[str] = []

    try:
        _normalize_json_value(manifest)
    except VirtualCollectionManifestError as exc:
        errors.append(str(exc))
        return errors

    _require_equal(errors, manifest, "schema_id", SCHEMA_ID)
    _require_equal(errors, manifest, "schema_version", SCHEMA_VERSION)
    _require_string(errors, manifest, "collection_id")
    _require_string(errors, manifest, "collection_name")
    _require_string(errors, manifest, "created_utc")
    _require_string(errors, manifest, "purpose")
    _require_equal(errors, manifest, "manifest_canonicalization", CANONICALIZATION)

    selection_policy = _require_mapping(errors, manifest, "selection_policy")
    if selection_policy is not None:
        _require_bool(errors, selection_policy, "latest_allowed_during_selection", "selection_policy")
        _require_bool(errors, selection_policy, "latest_resolved_before_export", "selection_policy")
        _require_bool(errors, selection_policy, "production_requires_explicit_runs", "selection_policy")

    query = _require_mapping(errors, manifest, "query")
    if query is not None:
        _validate_query(errors, query)

    _validate_export_profiles(errors, manifest.get("export_profiles"))
    _validate_records(errors, manifest.get("records"))

    return errors


def assert_valid_manifest(manifest: Mapping[str, Any]) -> None:
    """Raise ``VirtualCollectionManifestError`` if ``manifest`` is invalid."""

    errors = validate_manifest(manifest)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise VirtualCollectionManifestError(f"invalid virtual collection manifest:\n{detail}")


def verify_manifest_sha256(manifest: Mapping[str, Any]) -> bool:
    """Return whether an existing ``manifest_sha256`` matches the payload."""

    expected = manifest.get("manifest_sha256")
    return isinstance(expected, str) and expected == compute_manifest_sha256(manifest)


def _require_mapping(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str = "$",
) -> Mapping[str, Any] | None:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{prefix}.{key}: required mapping")
        return None
    return value


def _require_list(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str = "$",
) -> list[Any] | None:
    value = mapping.get(key)
    if not isinstance(value, list):
        errors.append(f"{prefix}.{key}: required list")
        return None
    return value


def _require_string(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str = "$",
    allow_empty: bool = False,
) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        errors.append(f"{prefix}.{key}: required non-empty string")
        return None
    return value


def _require_bool(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    prefix: str,
) -> bool | None:
    value = mapping.get(key)
    if not isinstance(value, bool):
        errors.append(f"{prefix}.{key}: required boolean")
        return None
    return value


def _require_equal(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    prefix: str = "$",
) -> None:
    if mapping.get(key) != expected:
        errors.append(f"{prefix}.{key}: expected {expected!r}, got {mapping.get(key)!r}")


def _validate_query(errors: list[str], query: Mapping[str, Any]) -> None:
    status = query.get("registry_snapshot_status")
    if status not in REGISTRY_SNAPSHOT_STATUSES:
        errors.append(
            "query.registry_snapshot_status: expected one of "
            f"{sorted(REGISTRY_SNAPSHOT_STATUSES)}, got {status!r}"
        )
    if status == "recorded" and not isinstance(query.get("registry_snapshot_sha256"), str):
        errors.append("query.registry_snapshot_sha256: required string when status is 'recorded'")
    if status in {"recorded", "not_recorded"}:
        _require_string(errors, query, "registry_path", prefix="query")
        _require_mapping(errors, query, "filters", prefix="query")
        _require_list(errors, query, "ordering", prefix="query")


def _validate_export_profiles(errors: list[str], value: Any) -> None:
    profiles = value
    if not isinstance(profiles, list):
        errors.append("$.export_profiles: required list")
        return
    if not profiles:
        errors.append("$.export_profiles: at least one export profile is required")
        return
    seen: set[str] = set()
    for index, profile in enumerate(profiles):
        prefix = f"export_profiles[{index}]"
        if not isinstance(profile, Mapping):
            errors.append(f"{prefix}: required mapping")
            continue
        profile_id = _require_string(errors, profile, "profile_id", prefix=prefix)
        if profile_id:
            if profile_id in seen:
                errors.append(f"{prefix}.profile_id: duplicate profile id {profile_id!r}")
            seen.add(profile_id)
        _require_string_list(errors, profile, "required_run_families", prefix)
        _require_string_list(errors, profile, "optional_run_families", prefix)


def _require_string_list(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    prefix: str,
) -> None:
    values = mapping.get(key)
    if not isinstance(values, list):
        errors.append(f"{prefix}.{key}: required list")
        return
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            errors.append(f"{prefix}.{key}[{index}]: required non-empty string")


def _validate_records(errors: list[str], value: Any) -> None:
    if not isinstance(value, list):
        errors.append("$.records: required list")
        return
    for index, record in enumerate(value):
        prefix = f"records[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{prefix}: required mapping")
            continue
        _require_string(errors, record, "recording_id", prefix=prefix)
        _require_string(errors, record, "dataset_id", prefix=prefix)
        _require_string(errors, record, "artifact_kind", prefix=prefix)
        locator = _require_mapping(errors, record, "locator_at_selection", prefix=prefix)
        if locator is not None:
            _require_string(errors, locator, "uri", prefix=f"{prefix}.locator_at_selection")
            _require_string(errors, locator, "storage_tier", prefix=f"{prefix}.locator_at_selection")
            _require_string(errors, locator, "last_verified_utc", prefix=f"{prefix}.locator_at_selection")
        _require_mapping(errors, record, "recording_attrs", prefix=prefix)
        _require_mapping(errors, record, "protocol", prefix=prefix)
        _validate_stimulus_response_validation(
            errors,
            record.get("stimulus_response_validation"),
            prefix=prefix,
        )
        source_runs = _require_mapping(errors, record, "source_runs", prefix=prefix)
        if source_runs is not None:
            for run_family, run in source_runs.items():
                _validate_source_run(errors, run_family, run, prefix=f"{prefix}.source_runs")
        status = _require_mapping(errors, record, "status", prefix=prefix)
        if status is not None:
            included = status.get("included")
            if not isinstance(included, bool):
                errors.append(f"{prefix}.status.included: required boolean")
            _require_list(errors, status, "warnings", prefix=f"{prefix}.status")
            _require_list(errors, status, "exclusions", prefix=f"{prefix}.status")


def _validate_stimulus_response_validation(
    errors: list[str],
    value: Any,
    *,
    prefix: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        errors.append(f"{prefix}.stimulus_response_validation: required mapping")
        return
    required = value.get("required")
    if not isinstance(required, bool):
        errors.append(f"{prefix}.stimulus_response_validation.required: required boolean")
    status = value.get("status")
    if status not in STIMULUS_RESPONSE_VALIDATION_STATUSES:
        errors.append(
            f"{prefix}.stimulus_response_validation.status: expected one of "
            f"{sorted(STIMULUS_RESPONSE_VALIDATION_STATUSES)}, got {status!r}"
        )
    warnings = value.get("warnings")
    if not isinstance(warnings, list):
        errors.append(f"{prefix}.stimulus_response_validation.warnings: required list")


def _validate_source_run(
    errors: list[str],
    run_family: str,
    run: Any,
    *,
    prefix: str,
) -> None:
    path = f"{prefix}.{run_family}"
    if not isinstance(run, Mapping):
        errors.append(f"{path}: required mapping")
        return
    present = run.get("present")
    if not isinstance(present, bool):
        errors.append(f"{path}.present: required boolean")
        return
    required = run.get("required")
    if not isinstance(required, bool):
        errors.append(f"{path}.required: required boolean")
    fingerprint_status = run.get("fingerprint_status")
    if fingerprint_status not in FINGERPRINT_STATUSES:
        errors.append(
            f"{path}.fingerprint_status: expected one of "
            f"{sorted(FINGERPRINT_STATUSES)}, got {fingerprint_status!r}"
        )
    if present:
        _require_string(errors, run, "run_id", prefix=path)
        _require_string(errors, run, "path", prefix=path)
        selection = run.get("selection")
        if selection not in RUN_SELECTIONS:
            errors.append(
                f"{path}.selection: expected one of {sorted(RUN_SELECTIONS)}, got {selection!r}"
            )
    else:
        if run.get("run_id") is not None:
            errors.append(f"{path}.run_id: absent runs must use null run_id")
        if run.get("path") is not None:
            errors.append(f"{path}.path: absent runs must use null path")
        _require_string(errors, run, "reason", prefix=path)
        if fingerprint_status != "not_applicable":
            errors.append(
                f"{path}.fingerprint_status: absent runs should use 'not_applicable'"
            )


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a virtual collection manifest from JSON."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise VirtualCollectionManifestError(f"{path}: manifest root must be a JSON object")
    return payload


def write_manifest(path: Path, manifest: Mapping[str, Any], *, overwrite: bool = False) -> None:
    """Write a manifest JSON file.

    Existing files are not overwritten unless ``overwrite`` is true. This keeps
    the default behavior aligned with immutable collection manifests.
    """

    if path.exists() and not overwrite:
        raise VirtualCollectionManifestError(
            f"{path}: already exists; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _normalize_json_value(manifest),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and hash Palette virtual collection manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a manifest")
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument(
        "--check-hash",
        action="store_true",
        help="also require manifest_sha256 to match the canonical payload",
    )

    hash_parser = subparsers.add_parser("hash", help="print manifest_sha256")
    hash_parser.add_argument("manifest", type=Path)

    stamp_parser = subparsers.add_parser(
        "stamp",
        help="write a copy with manifest_sha256 populated",
    )
    stamp_parser.add_argument("manifest", type=Path)
    stamp_parser.add_argument("--output", "-o", type=Path, required=True)
    stamp_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output manifest",
    )
    stamp_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip structural validation before writing",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for virtual collection manifest utilities."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            errors = validate_manifest(manifest)
            if args.check_hash and not verify_manifest_sha256(manifest):
                errors.append("manifest_sha256: does not match canonical payload")
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 1
            print("valid")
            return 0

        if args.command == "hash":
            print(compute_manifest_sha256(manifest))
            return 0

        if args.command == "stamp":
            if not args.no_validate:
                assert_valid_manifest(manifest)
            stamped = with_manifest_sha256(manifest)
            write_manifest(args.output, stamped, overwrite=args.overwrite)
            print(stamped["manifest_sha256"])
            return 0
    except (OSError, json.JSONDecodeError, VirtualCollectionManifestError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
