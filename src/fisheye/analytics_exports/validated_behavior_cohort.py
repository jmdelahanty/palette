"""Receipt-backed sharding and immutable publication for behavior cohorts.

This module is intentionally protocol-neutral.  Its built-in row producers
materialize only the closed membership, bundle, and capability relations.  A
scientific family extends the same engine by supplying exact table specs and
recording-scoped row extractors; the engine never discovers Zarr runs or
reconstructs unavailable evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping, Sequence
import uuid

from fisheye.analysis_workflows.validated_behavior_cohort import (
    read_validated_behavior_bundle_set,
    read_validated_behavior_cohort_membership,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .arrow_contract_core import (
    canonical_bytes,
    exact_schema,
    validate_contract_envelope,
    validate_exact_schema,
)
from .publication import (
    commit_validated_immutable_generation,
    manifest_identity,
    safe_component,
    sha256_file,
)
from .validated_behavior_contracts import (
    ARROW_ENVELOPE_SCHEMA_ID,
    ARROW_ENVELOPE_SCHEMA_VERSION,
    CORE_METADATA_PROFILE_ID,
    CORE_TABLE_SPECS,
    ValidatedBehaviorTableSpec,
    table_contract_envelope,
    validate_table_specs,
)

EXPORT_PLAN_SCHEMA_ID = "palette.analytics.validated_behavior_export_plan"
LEGACY_EXPORT_PLAN_SCHEMA_VERSION = 1
LEGACY_EXPORT_PLAN_METHOD_ID = "closed_membership_recording_shard_plan_v1"
EXPORT_PLAN_SCHEMA_VERSION = 2
EXPORT_PLAN_METHOD_ID = "closed_membership_recording_shard_plan_v2"
EXPORT_PLAN_STATUS = "planned_selector_ineligible"

EVIDENCE_PROFILE_SCHEMA_ID = (
    "palette.analytics.validated_behavior.finalization_evidence_profile"
)
EVIDENCE_PROFILE_SCHEMA_VERSION = 1
EVIDENCE_PROFILE_ID = "receipt_composed_parquet_finalization_v2"

SHARD_SCHEMA_ID = "palette.analytics.validated_behavior_export_shard"
LEGACY_SHARD_SCHEMA_VERSION = 1
LEGACY_SHARD_METHOD_ID = "recording_owned_exact_parquet_parts_v1"
LEGACY_SHARD_VALIDATION_POLICY = "exact_inputs_parts_arrow_and_primary_keys_v1"
SHARD_SCHEMA_VERSION = 2
SHARD_METHOD_ID = "recording_owned_exact_parquet_parts_v2"
SHARD_STATUS = "complete_validated"
SHARD_VALIDATION_POLICY = "exact_inputs_parts_and_semantic_proofs_v2"

SHARD_SEMANTIC_SCHEMA_ID = "palette.analytics.validated_behavior_export_shard_semantics"
SHARD_SEMANTIC_SCHEMA_VERSION = 1
SHARD_SEMANTIC_METHOD_ID = "trusted_writer_exact_part_semantics_v1"
SHARD_SEMANTIC_STATUS = "complete"

TRANSFER_RECEIPT_SCHEMA_ID = (
    "palette.analytics.validated_behavior_cohort_transfer_receipt"
)
TRANSFER_RECEIPT_SCHEMA_VERSION = 1
TRANSFER_RECEIPT_METHOD_ID = "copy_then_destination_sha256_v1"
TRANSFER_RECEIPT_STATUS = "complete"
TRANSFER_VERIFICATION_POLICY = "one_destination_sha256_per_copied_part_v1"

EXPORT_SCHEMA_ID = "palette.analytics.validated_behavior_cohort_export"
LEGACY_EXPORT_SCHEMA_VERSION = 1
LEGACY_EXPORT_METHOD_ID = "receipt_barrier_manifest_selected_parquet_v1"
EXPORT_SCHEMA_VERSION = 2
EXPORT_METHOD_ID = "receipt_composed_manifest_selected_parquet_v2"
EXPORT_STATUS = "complete_selector_ineligible"
PUBLICATION_SCHEMA_ID = "palette.analytics.validated_behavior.publication"
PUBLICATION_SCHEMA_VERSION = 1

VALIDATION_RECEIPT_SCHEMA_ID = "palette.analytics.validated_behavior_cohort_validation"
LEGACY_VALIDATION_RECEIPT_SCHEMA_VERSION = 1
LEGACY_VALIDATION_POLICY = "manifest_selected_schema_key_foreign_key_inventory_v1"
VALIDATION_RECEIPT_SCHEMA_VERSION = 2
VALIDATION_POLICY = "receipt_composed_schema_key_foreign_key_inventory_v2"

GENERATION_COMPOSITION_POLICY = (
    "recording_partitioned_owner_primary_key_foreign_key_composition_v1"
)
FULL_AUDIT_MODE = "full_part_hashes_and_decoded_relations_v1"

MUTATION_EXCLUSION_POLICY = {
    "policy_id": "cooperative_read_only_regular_files_v1",
    "trusted_storage_model": "trusted_palette_group_storage_v1",
    "part_file_mode": "0444",
    "receipt_file_mode": "0444",
    "directory_mutation_detection": "closed_inventory_before_visibility_v1",
}

SAFETY = {
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
    "source_mutation": False,
    "zarr_mutation": False,
}

DEFAULT_EXPORT_PARAMETERS = {
    "parquet_compression": "zstd",
    "requested_row_group_rows": 65536,
    "effective_row_group_rows": 65536,
    "part_policy": "one_independently_validated_part_per_recording_and_table_v1",
}

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PLAN_FIELDS_V1 = {
    "schema_id",
    "schema_version",
    "method_id",
    "status",
    "export_run_id",
    "export_profile",
    "membership",
    "bundle_set",
    "member_count",
    "members",
    "table_names",
    "table_specs",
    "table_coverage",
    "arrow_schema_contracts",
    "parameters",
    "shard_root",
    "publication_root",
    "software_authority",
    "created_at_utc",
    "safety",
    "plan_sha256",
}
_PLAN_FIELDS_V2 = _PLAN_FIELDS_V1 | {"evidence_profile"}
_SHARD_PART_FIELDS = {
    "path",
    "size_bytes",
    "row_count",
    "file_sha256",
    "arrow_schema_id",
    "arrow_schema_version",
    "arrow_schema_sha256",
    "primary_key",
    "primary_key_bounds",
}
_SHARD_FIELDS_V1 = {
    "schema_id",
    "schema_version",
    "method_id",
    "status",
    "export_run_id",
    "export_plan",
    "member",
    "membership",
    "bundle_set",
    "requested_tables",
    "table_coverage",
    "parts_by_table",
    "zero_row_reasons_by_table",
    "parameters",
    "validation_policy",
    "software_authority",
    "created_at_utc",
    "safety",
    "record_sha256",
}
_SHARD_FIELDS_V2 = _SHARD_FIELDS_V1 | {
    "semantic_validation",
    "mutation_exclusion",
}
_PUBLICATION_PART_FIELDS = _SHARD_PART_FIELDS | {
    "member_ordinal",
    "recording_id",
    "generation_path",
    "source_shard_record_sha256",
}

_MANIFEST_FIELDS_V1 = {
    "schema_id",
    "schema_version",
    "method_id",
    "status",
    "export_run_id",
    "export_plan",
    "export_profile",
    "membership",
    "bundle_set",
    "member_count",
    "membership_state_counts",
    "bundle_state_counts",
    "capability_matrix_sha256",
    "table_names",
    "table_specs",
    "table_coverage",
    "arrow_schema_contracts",
    "shard_receipts",
    "shard_receipts_sha256",
    "row_counts_by_table",
    "parameters",
    "analysis_unit_policy",
    "acquisition_batch_policy",
    "temporal_alignment_policy",
    "publication",
    "validation_receipt",
    "software_authority",
    "created_at_utc",
    "safety",
    "record_sha256",
}
_MANIFEST_FIELDS_V2 = _MANIFEST_FIELDS_V1 | {"transfer_receipt"}


@dataclass(frozen=True)
class ValidatedBehaviorBatchSource:
    """One-pass bounded column batches for a recording-owned dense table.

    Every non-empty batch carries the exact contract column roster. Dense
    adapters emit rows in strictly increasing primary-key order, allowing the
    writer and receipt validator to prove uniqueness with constant key memory.
    """

    batches: Iterable[Mapping[str, Any]]
    zero_row_reason: str | None = None


RowExtractor = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    tuple[Sequence[Mapping[str, Any]], str | None] | ValidatedBehaviorBatchSource,
]


class ValidatedBehaviorExportError(ValueError):
    """A plan, shard, publication, or selected part is not exact."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorExportError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field} must be one array.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _DIGEST_RE.fullmatch(result) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _timestamp(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidatedBehaviorExportError(
            f"{field} must be one ISO-8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{field} must carry a UTC offset.")
    if parsed.utcoffset().total_seconds() != 0:
        _fail(f"{field} must be expressed in UTC.")
    return result


def _absolute_path(value: object, *, field: str) -> Path:
    raw = str(value) if isinstance(value, Path) else _text(value, field=field)
    path = Path(raw)
    resolved = path.expanduser().resolve(strict=False)
    if not path.is_absolute() or str(resolved) != raw or "\\" in raw:
        _fail(f"{field} must be one canonical absolute POSIX path.")
    if any(part.casefold() in {"latest", "current", "selected"} for part in path.parts):
        _fail(f"{field} contains a selector-named path component.")
    return resolved


def _strict_object(path: str | Path, *, field: str) -> tuple[Path, dict[str, Any]]:
    requested = Path(path).expanduser()
    unresolved = requested if requested.is_absolute() else Path.cwd() / requested
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise FileNotFoundError(
                f"{field} contains a symbolic-link alias: {current}"
            )
    source = unresolved.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{field} is absent or aliased: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorExportError(
            f"Cannot read {field} as strict JSON: {source}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{field} must contain one object.")
    return source, value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _freeze_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"Cannot freeze absent or aliased file: {path}")
    path.chmod(0o444)


def _require_frozen_regular_file(path: Path, *, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{field} is absent or aliased.")
    if path.stat().st_mode & 0o777 != 0o444:
        _fail(f"{field} is not cooperatively frozen read-only.")


def _sealed(body: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    normalized = _plain(body)
    canonical_bytes(normalized)
    return {**normalized, digest_field: canonical_json_sha256(normalized)}


def _validate_self_digest(
    value: Mapping[str, Any], *, digest_field: str, field: str
) -> dict[str, Any]:
    normalized = _plain(value)
    persisted = _digest(normalized.pop(digest_field, None), field=digest_field)
    if canonical_json_sha256(normalized) != persisted:
        _fail(f"{field} self digest is stale.")
    return {**normalized, digest_field: persisted}


def _file_binding(path: Path, record_sha256: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "record_sha256": record_sha256,
    }


def _validate_file_binding(value: object, *, field: str) -> dict[str, Any]:
    binding = _mapping(value, field=field)
    if set(binding) != {"path", "file_sha256", "record_sha256"}:
        _fail(f"{field} field set is inexact.")
    path = _absolute_path(binding.get("path"), field=f"{field}.path")
    file_sha = _digest(binding.get("file_sha256"), field=f"{field}.file_sha256")
    record_sha = _digest(binding.get("record_sha256"), field=f"{field}.record_sha256")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != file_sha:
        _fail(f"{field} file is absent, aliased, or changed.")
    return {
        "path": str(path),
        "file_sha256": file_sha,
        "record_sha256": record_sha,
    }


def _validate_plan_file_binding(
    value: object,
    *,
    expected_plan: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    binding = _mapping(value, field="export_plan")
    if set(binding) != {"path", "file_sha256", "plan_sha256"}:
        _fail("export_plan file binding is inexact.")
    path = _absolute_path(binding.get("path"), field="export_plan.path")
    file_sha = _digest(binding.get("file_sha256"), field="export_plan.file_sha256")
    plan_sha = _digest(binding.get("plan_sha256"), field="export_plan.plan_sha256")
    if path.is_symlink() or not path.is_file() or sha256_file(path) != file_sha:
        _fail("Bound export-plan file is absent, aliased, or changed.")
    _source, raw = _strict_object(path, field="export plan")
    sealed = _validate_self_digest(raw, digest_field="plan_sha256", field="Export plan")
    if sealed["plan_sha256"] != plan_sha:
        _fail("Bound export-plan record digest is stale.")
    if expected_plan is not None and _plain(sealed) != _plain(expected_plan):
        _fail("Shard or manifest binds a different export-plan document.")
    return path, {
        "path": str(path),
        "file_sha256": file_sha,
        "plan_sha256": plan_sha,
    }


def _spec_records(
    specs: Mapping[str, ValidatedBehaviorTableSpec],
) -> dict[str, dict[str, object]]:
    return {name: specs[name].to_dict() for name in validate_table_specs(specs)}


def _validate_installed_specs(
    value: object,
    *,
    specs: Mapping[str, ValidatedBehaviorTableSpec],
) -> dict[str, dict[str, object]]:
    records = _mapping(value, field="table_specs")
    expected = _spec_records(specs)
    if _plain(records) != expected:
        _fail("Export table specs differ from the installed exact contracts.")
    return expected


def _software_authority(
    *, palette_commit: str, palette_repo: str | Path
) -> dict[str, str]:
    commit = _text(palette_commit, field="palette_commit")
    if _COMMIT_RE.fullmatch(commit) is None:
        _fail("palette_commit must be one full lowercase Git object ID.")
    return {
        "repository": "palette",
        "commit": commit,
        "deployment_path": str(_absolute_path(palette_repo, field="palette_repo")),
    }


def _validate_software(value: object) -> dict[str, str]:
    software = _mapping(value, field="software_authority")
    if set(software) != {"repository", "commit", "deployment_path"}:
        _fail("software_authority field set is inexact.")
    if software.get("repository") != "palette":
        _fail("software_authority.repository is invalid.")
    commit = _text(software.get("commit"), field="software_authority.commit")
    if _COMMIT_RE.fullmatch(commit) is None:
        _fail("software_authority.commit is invalid.")
    deployment = _absolute_path(
        software.get("deployment_path"), field="software_authority.deployment_path"
    )
    return {
        "repository": "palette",
        "commit": commit,
        "deployment_path": str(deployment),
    }


def _default_created_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_evidence_profile() -> dict[str, Any]:
    body = {
        "schema_id": EVIDENCE_PROFILE_SCHEMA_ID,
        "schema_version": EVIDENCE_PROFILE_SCHEMA_VERSION,
        "profile_id": EVIDENCE_PROFILE_ID,
        "required_shard_receipt": {
            "schema_id": SHARD_SCHEMA_ID,
            "schema_version": SHARD_SCHEMA_VERSION,
            "validation_policy": SHARD_VALIDATION_POLICY,
            "semantic_schema_id": SHARD_SEMANTIC_SCHEMA_ID,
            "semantic_schema_version": SHARD_SEMANTIC_SCHEMA_VERSION,
            "semantic_method_id": SHARD_SEMANTIC_METHOD_ID,
        },
        "transfer_verification_policy": TRANSFER_VERIFICATION_POLICY,
        "generation_composition_policy": GENERATION_COMPOSITION_POLICY,
        "normal_finalization_payload_decoding": False,
        "full_audit_mode": FULL_AUDIT_MODE,
    }
    return _sealed(body, digest_field="record_sha256")


def _validate_evidence_profile(value: object) -> Mapping[str, Any]:
    profile = _validate_self_digest(
        _mapping(value, field="evidence_profile"),
        digest_field="record_sha256",
        field="Evidence profile",
    )
    expected = _current_evidence_profile()
    if profile != expected:
        _fail("Export-plan evidence profile is unsupported or incomplete.")
    return profile


def _require_current_plan_evidence(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("schema_version") != EXPORT_PLAN_SCHEMA_VERSION
        or plan.get("method_id") != EXPORT_PLAN_METHOD_ID
    ):
        _fail(
            "Current execution requires a receipt-composed v2 export plan; "
            "legacy plans remain read-only."
        )
    _validate_evidence_profile(plan.get("evidence_profile"))


def _member_plan_record(
    membership_member: Mapping[str, Any], bundle_member: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "ordinal": int(membership_member["ordinal"]),
        "recording_id": str(membership_member["recording_id"]),
        "analysis_zarr": str(membership_member["analysis_zarr"]),
        "membership_state": str(membership_member["membership_state"]),
        "membership_member_sha256": str(membership_member["member_sha256"]),
        "bundle_state": str(bundle_member["bundle_state"]),
        "bundle_set_member_sha256": str(bundle_member["member_sha256"]),
        "capabilities_sha256": str(bundle_member["capabilities_sha256"]),
    }


def _table_coverage_records(
    bundle_set: Mapping[str, Any],
    specs: Mapping[str, ValidatedBehaviorTableSpec],
) -> dict[str, dict[str, Any]]:
    members = list(bundle_set["members"])
    capability_keys = set(bundle_set["capability_contract"]["keys"])
    records: dict[str, dict[str, Any]] = {}
    for table_name in validate_table_specs(specs):
        spec = specs[table_name]
        if spec.capability_policy == "all_parent_metadata":
            body = {
                "capability_policy": spec.capability_policy,
                "required_capability": None,
                "contributing_member_ordinals": [
                    int(member["ordinal"]) for member in members
                ],
                "capability_state_counts": None,
                "member_capability_states": None,
            }
        else:
            capability = str(spec.required_capability)
            if capability not in capability_keys:
                _fail(
                    f"{table_name}: required capability is absent from the bundle profile."
                )
            states = [member["capabilities"][capability]["state"] for member in members]
            contributors = [
                int(member["ordinal"])
                for member, state in zip(members, states, strict=True)
                if state == "complete"
            ]
            if spec.capability_policy == "required_all_admitted":
                missing = [
                    str(member["recording_id"])
                    for member, state in zip(members, states, strict=True)
                    if member["bundle_state"] == "complete" and state != "complete"
                ]
                if missing:
                    _fail(
                        f"{table_name}: required capability is incomplete for admitted "
                        f"recordings: {missing!r}."
                    )
            body = {
                "capability_policy": spec.capability_policy,
                "required_capability": capability,
                "contributing_member_ordinals": contributors,
                "capability_state_counts": {
                    state: states.count(state)
                    for state in bundle_set["capability_contract"]["states"]
                },
                "member_capability_states": [
                    {
                        "member_ordinal": int(member["ordinal"]),
                        "state": member["capabilities"][capability]["state"],
                        "reason_code": member["capabilities"][capability][
                            "reason_code"
                        ],
                    }
                    for member in members
                ],
            }
        records[table_name] = _sealed(body, digest_field="record_sha256")
    return records


def _require_declared_bundle_export_profile(
    bundle_set: Mapping[str, Any], export_profile_id: object
) -> str:
    """Bind profile-aware bundle sets without invalidating legacy bundles.

    Recording-bundle v1 artifacts predate an explicit export-profile field and
    remain readable. New bundle profiles that declare the field are fail-closed:
    a planner cannot reinterpret them through another installed table profile.
    """

    requested = safe_component(export_profile_id, label="export profile ID")
    bundle_profile = _mapping(bundle_set.get("bundle_profile"), field="bundle_profile")
    declared = bundle_profile.get("export_profile_id")
    if declared is None:
        return requested
    normalized = safe_component(declared, label="bundle export profile ID")
    if normalized != requested:
        _fail(
            "Bundle set declares export profile "
            f"{normalized!r}, not requested profile {requested!r}."
        )
    return requested


def build_validated_behavior_export_plan(
    *,
    membership_path: str | Path,
    bundle_set_path: str | Path,
    export_run_id: str,
    shard_root: str | Path,
    publication_root: str | Path,
    palette_commit: str,
    palette_repo: str | Path,
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
    export_profile_id: str = CORE_METADATA_PROFILE_ID,
    parameters: Mapping[str, Any] = DEFAULT_EXPORT_PARAMETERS,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, closed fanout plan from exact manifests."""

    membership_file = Path(membership_path).expanduser().resolve()
    bundle_file = Path(bundle_set_path).expanduser().resolve()
    membership = read_validated_behavior_cohort_membership(membership_file)
    bundle_set = read_validated_behavior_bundle_set(bundle_file, membership=membership)
    if bundle_set["membership"]["file_sha256"] != sha256_file(membership_file):
        _fail("Bundle set does not bind the selected membership file bytes.")
    table_names = validate_table_specs(table_specs)
    table_records = _spec_records(table_specs)
    coverage = _table_coverage_records(bundle_set, table_specs)
    requested_profile_id = _require_declared_bundle_export_profile(
        bundle_set, export_profile_id
    )
    profile_body = {
        "profile_id": requested_profile_id,
        "table_names": list(table_names),
        "table_specs_sha256": canonical_json_sha256(table_records),
    }
    profile = _sealed(profile_body, digest_field="record_sha256")
    members = [
        _member_plan_record(member, bundle_member)
        for member, bundle_member in zip(
            membership["members"], bundle_set["members"], strict=True
        )
    ]
    body = {
        "schema_id": EXPORT_PLAN_SCHEMA_ID,
        "schema_version": EXPORT_PLAN_SCHEMA_VERSION,
        "method_id": EXPORT_PLAN_METHOD_ID,
        "status": EXPORT_PLAN_STATUS,
        "export_run_id": safe_component(export_run_id, label="export run ID"),
        "export_profile": profile,
        "evidence_profile": _current_evidence_profile(),
        "membership": _file_binding(membership_file, str(membership["record_sha256"])),
        "bundle_set": _file_binding(bundle_file, str(bundle_set["record_sha256"])),
        "member_count": len(members),
        "members": members,
        "table_names": list(table_names),
        "table_specs": table_records,
        "table_coverage": coverage,
        "arrow_schema_contracts": table_contract_envelope(table_specs),
        "parameters": _plain(_mapping(parameters, field="parameters")),
        "shard_root": str(_absolute_path(shard_root, field="shard_root")),
        "publication_root": str(
            _absolute_path(publication_root, field="publication_root")
        ),
        "software_authority": _software_authority(
            palette_commit=palette_commit, palette_repo=palette_repo
        ),
        "created_at_utc": _timestamp(
            created_at_utc or _default_created_at(), field="created_at_utc"
        ),
        "safety": SAFETY,
    }
    plan = _sealed(body, digest_field="plan_sha256")
    validate_validated_behavior_export_plan(
        plan,
        membership=membership,
        bundle_set=bundle_set,
        table_specs=table_specs,
    )
    return plan


def validate_validated_behavior_export_plan(
    value: object,
    *,
    membership: Mapping[str, Any],
    bundle_set: Mapping[str, Any],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
) -> Mapping[str, Any]:
    plan = _validate_self_digest(
        _mapping(value, field="export plan"),
        digest_field="plan_sha256",
        field="Export plan",
    )
    schema_version = plan.get("schema_version")
    if schema_version == LEGACY_EXPORT_PLAN_SCHEMA_VERSION:
        required_fields = _PLAN_FIELDS_V1
        expected_method = LEGACY_EXPORT_PLAN_METHOD_ID
    elif schema_version == EXPORT_PLAN_SCHEMA_VERSION:
        required_fields = _PLAN_FIELDS_V2
        expected_method = EXPORT_PLAN_METHOD_ID
    else:
        _fail("Export-plan schema version is unsupported.")
    if set(plan) != required_fields:
        _fail("Export-plan field set is inexact.")
    if (
        plan.get("schema_id") != EXPORT_PLAN_SCHEMA_ID
        or plan.get("method_id") != expected_method
        or plan.get("status") != EXPORT_PLAN_STATUS
        or plan.get("safety") != SAFETY
    ):
        _fail("Export-plan identity, method, status, or safety is invalid.")
    if schema_version == EXPORT_PLAN_SCHEMA_VERSION:
        _validate_evidence_profile(plan.get("evidence_profile"))
    safe_component(plan.get("export_run_id"), label="export run ID")
    table_names = validate_table_specs(table_specs)
    if plan.get("table_names") != list(table_names):
        _fail("Export-plan table roster differs from installed specs.")
    table_records = _validate_installed_specs(
        plan.get("table_specs"), specs=table_specs
    )
    expected_coverage = _table_coverage_records(bundle_set, table_specs)
    if plan.get("table_coverage") != expected_coverage:
        _fail("Export-plan table capability coverage is stale or incomplete.")
    profile = _validate_self_digest(
        _mapping(plan.get("export_profile"), field="export_profile"),
        digest_field="record_sha256",
        field="Export profile",
    )
    if set(profile) != {
        "profile_id",
        "table_names",
        "table_specs_sha256",
        "record_sha256",
    }:
        _fail("Export-profile field set is inexact.")
    profile_id = safe_component(profile.get("profile_id"), label="export profile ID")
    _require_declared_bundle_export_profile(bundle_set, profile_id)
    if profile.get("table_names") != list(table_names) or profile.get(
        "table_specs_sha256"
    ) != canonical_json_sha256(table_records):
        _fail("Export profile does not close the installed table suite.")
    contracts = {name: table_specs[name].contract for name in table_names}
    validate_contract_envelope(
        plan.get("arrow_schema_contracts"),
        table_names,
        known_table_names=table_names,
        contracts=contracts,
        schema_id=ARROW_ENVELOPE_SCHEMA_ID,
        schema_version=ARROW_ENVELOPE_SCHEMA_VERSION,
    )
    membership_binding = _mapping(plan.get("membership"), field="membership")
    bundle_binding = _mapping(plan.get("bundle_set"), field="bundle_set")
    if (
        membership_binding.get("record_sha256") != membership["record_sha256"]
        or bundle_binding.get("record_sha256") != bundle_set["record_sha256"]
    ):
        _fail("Export plan binds another membership or bundle-set generation.")
    members = _list(plan.get("members"), field="members")
    if (
        plan.get("member_count") != len(members)
        or len(members) != membership["member_count"]
    ):
        _fail("Export-plan member count is invalid.")
    expected = [
        _member_plan_record(member, bundle_member)
        for member, bundle_member in zip(
            membership["members"], bundle_set["members"], strict=True
        )
    ]
    if members != expected:
        _fail("Export-plan member roster differs from its exact inputs.")
    parameters = _mapping(plan.get("parameters"), field="parameters")
    row_group = parameters.get("effective_row_group_rows")
    if type(row_group) is not int or row_group <= 0:
        _fail("Export-plan effective row-group size must be positive.")
    _absolute_path(plan.get("shard_root"), field="shard_root")
    _absolute_path(plan.get("publication_root"), field="publication_root")
    _validate_software(plan.get("software_authority"))
    _timestamp(plan.get("created_at_utc"), field="created_at_utc")
    return plan


def read_validated_behavior_export_plan(
    path: str | Path,
    *,
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
    require_current_evidence: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Read a plan and revalidate its small exact input manifests."""

    _plan_path, raw = _strict_object(path, field="export plan")
    membership_binding = _validate_file_binding(
        raw.get("membership"), field="membership"
    )
    membership = read_validated_behavior_cohort_membership(membership_binding["path"])
    bundle_binding = _validate_file_binding(raw.get("bundle_set"), field="bundle_set")
    bundle_set = read_validated_behavior_bundle_set(
        bundle_binding["path"], membership=membership
    )
    plan = validate_validated_behavior_export_plan(
        raw,
        membership=membership,
        bundle_set=bundle_set,
        table_specs=table_specs,
    )
    if require_current_evidence:
        _require_current_plan_evidence(plan)
    return plan, membership, bundle_set


def write_validated_behavior_export_plan(
    path: str | Path, plan: Mapping[str, Any]
) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    _write_json(target, plan)
    return target


def shard_relative_path(plan_sha256: str, ordinal: int, recording_id: str) -> Path:
    digest = _digest(plan_sha256, field="plan_sha256")
    if type(ordinal) is not int or ordinal <= 0:
        _fail("member ordinal must be positive.")
    recording = safe_component(recording_id, label="recording ID")
    return Path(f"plan={digest}") / f"member={ordinal:06d}-{recording}"


def planned_shard_root(plan: Mapping[str, Any], member: Mapping[str, Any]) -> Path:
    root = _absolute_path(plan.get("shard_root"), field="shard_root")
    return root / shard_relative_path(
        str(plan["plan_sha256"]),
        int(member["ordinal"]),
        str(member["recording_id"]),
    )


def planned_shard_receipt_path(
    plan: Mapping[str, Any], member: Mapping[str, Any]
) -> Path:
    return planned_shard_root(plan, member) / "receipt.json"


def _canonical_json_text(value: object) -> str:
    return canonical_bytes(_plain(value)).decode("ascii")


def _core_rows(
    table_name: str,
    plan: Mapping[str, Any],
    membership_member: Mapping[str, Any],
    bundle_member: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], None]:
    run_id = str(plan["export_run_id"])
    common = {
        "export_run_id": run_id,
        "member_ordinal": int(membership_member["ordinal"]),
        "membership_member_sha256": str(membership_member["member_sha256"]),
        "recording_id": str(membership_member["recording_id"]),
    }
    if table_name == "cohort_recordings":
        return [
            {
                **common,
                "source_ordinal": int(membership_member["source_ordinal"]),
                "source_member_sha256": str(membership_member["source_member_sha256"]),
                "dataset_id": str(membership_member["dataset_id"]),
                "analysis_zarr": str(membership_member["analysis_zarr"]),
                "protocol_names": list(membership_member["protocol_names"]),
                "protocol_hashes": list(membership_member["protocol_hashes"]),
                "source_subject_ids": list(membership_member["source_subject_ids"]),
                "source_subject_identity_status": str(
                    membership_member["source_subject_identity_status"]
                ),
                "acquisition_batch_id": membership_member["acquisition_batch_id"],
                "acquisition_batch_identity_status": str(
                    membership_member["acquisition_batch_identity_status"]
                ),
                "analysis_unit_kind": str(membership_member["analysis_unit_kind"]),
                "analysis_unit_id": str(membership_member["analysis_unit_id"]),
                "membership_state": str(membership_member["membership_state"]),
                "reason_code": membership_member["reason_code"],
            }
        ], None
    bundle = bundle_member["bundle"]
    if table_name == "recording_bundles":
        return [
            {
                **common,
                "bundle_set_member_sha256": str(bundle_member["member_sha256"]),
                "analysis_zarr": str(bundle_member["analysis_zarr"]),
                "bundle_state": str(bundle_member["bundle_state"]),
                "reason_code": bundle_member["reason_code"],
                "bundle_adapter_id": None if bundle is None else bundle["adapter_id"],
                "bundle_path": None if bundle is None else bundle["path"],
                "bundle_file_sha256": (
                    None if bundle is None else bundle["file_sha256"]
                ),
                "bundle_record_sha256": (
                    None if bundle is None else bundle["record_sha256"]
                ),
                "bundle_schema_id": None if bundle is None else bundle["schema_id"],
                "bundle_schema_version": (
                    None if bundle is None else int(bundle["schema_version"])
                ),
                "bundle_method_id": None if bundle is None else bundle["method_id"],
                "bundle_status": None if bundle is None else bundle["status"],
                "bundle_binding_inventory_sha256": (
                    None if bundle is None else bundle["binding_inventory_sha256"]
                ),
                "capabilities_sha256": str(bundle_member["capabilities_sha256"]),
            }
        ], None
    if table_name == "recording_capabilities":
        rows = []
        for capability_id in sorted(bundle_member["capabilities"]):
            capability = bundle_member["capabilities"][capability_id]
            binding = capability["binding"]
            rows.append(
                {
                    **common,
                    "bundle_set_member_sha256": str(bundle_member["member_sha256"]),
                    "capability_id": capability_id,
                    "state": capability["state"],
                    "reason_code": capability["reason_code"],
                    "detail": capability["detail"],
                    "binding_json": (
                        None if binding is None else _canonical_json_text(binding)
                    ),
                    "binding_sha256": (
                        None
                        if binding is None
                        else canonical_json_sha256(_plain(binding))
                    ),
                    "capabilities_sha256": str(bundle_member["capabilities_sha256"]),
                }
            )
        return rows, None
    _fail(f"No installed core row producer exists for {table_name!r}.")


def _resident_row_semantics(
    rows: Sequence[Mapping[str, Any]],
    spec: ValidatedBehaviorTableSpec,
    *,
    export_run_id: str,
    recording_id: str,
) -> dict[str, Any]:
    """Validate owner and primary-key semantics while extracted rows are resident."""

    keys: list[tuple[Any, ...]] = []
    previous: tuple[Any, ...] | None = None
    for index, row in enumerate(rows):
        if row["export_run_id"] != export_run_id or row["recording_id"] != recording_id:
            _fail(f"{spec.table_name}: extracted row {index} differs from its owner.")
        key = tuple(row[name] for name in spec.contract.primary_key)
        if (
            spec.primary_key_validation == "strictly_increasing_v1"
            and previous is not None
            and key <= previous
        ):
            _fail(f"{spec.table_name}: primary keys are not strictly increasing.")
        previous = key
        keys.append(key)
    if spec.primary_key_validation == "unordered_unique_v1" and len(set(keys)) != len(
        keys
    ):
        _fail(f"{spec.table_name}: shard contains a duplicate primary key.")
    bounds = (
        None if not keys else {"minimum": list(min(keys)), "maximum": list(max(keys))}
    )
    return {
        "row_count": len(rows),
        "primary_key_bounds": bounds,
        "primary_key_distinct_count": len(keys),
    }


def _semantic_table_result(
    *,
    spec: ValidatedBehaviorTableSpec,
    part: Mapping[str, Any],
    export_run_id: str,
    recording_id: str,
    primary_key_distinct_count: int,
) -> dict[str, Any]:
    row_count = int(part["row_count"])
    required_fields = [item.name for item in spec.contract.fields if not item.nullable]
    return {
        "table_name": spec.table_name,
        "part_path": part["path"],
        "part_file_sha256": part["file_sha256"],
        "row_count": row_count,
        "arrow_footer_validation": {
            "schema_id": spec.contract.schema_id,
            "schema_version": spec.contract.schema_version,
            "schema_sha256": spec.contract.payload_sha256,
            "status": "complete",
        },
        "required_field_validation": {
            "fields": required_fields,
            "observed_row_count": row_count,
            "null_count": 0,
            "method_id": "trusted_writer_resident_values_v1",
            "status": "complete",
        },
        "row_owner_validation": {
            "fields": ["export_run_id", "recording_id"],
            "values": [export_run_id, recording_id],
            "observed_row_count": row_count,
            "mismatched_row_count": 0,
            "method_id": "trusted_writer_resident_values_v1",
            "status": "complete",
        },
        "primary_key_validation": {
            "fields": list(spec.contract.primary_key),
            "validation_mode": spec.primary_key_validation,
            "observed_row_count": row_count,
            "distinct_key_count": primary_key_distinct_count,
            "duplicate_count": 0,
            "bounds": _plain(part["primary_key_bounds"]),
            "method_id": "trusted_writer_resident_values_v1",
            "status": "complete",
        },
    }


def _advance_ordered_key_summary(
    table: Any,
    spec: ValidatedBehaviorTableSpec,
    *,
    previous: tuple[Any, ...] | None,
    minimum: tuple[Any, ...] | None,
    expected_owner: tuple[str, str] | None = None,
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
    """Prove strict key order for one Arrow batch without retaining all keys."""

    if table.num_rows == 0:
        return previous, minimum
    columns = [table.column(name).to_pylist() for name in spec.contract.primary_key]
    for key in zip(*columns, strict=True):
        if expected_owner is not None and key[:2] != expected_owner:
            _fail(f"{spec.table_name}: extracted batch differs from its owner.")
        if previous is not None and key <= previous:
            _fail(
                f"{spec.table_name}: dense extractor primary keys are not strictly "
                "increasing."
            )
        if minimum is None:
            minimum = key
        previous = key
    return previous, minimum


def _validate_extracted_columns(
    columns: Mapping[str, Any], spec: ValidatedBehaviorTableSpec, *, batch_index: int
) -> None:
    expected = {item.name for item in spec.contract.fields}
    actual = set(columns)
    if actual != expected:
        _fail(
            f"{spec.table_name}: extracted column batch {batch_index} has an inexact "
            f"field set; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}."
        )


def _part_footer(
    *, plan: Mapping[str, Any], member: Mapping[str, Any], table_name: str
) -> dict[bytes, bytes]:
    return {
        b"palette.dataset_family": b"validated_behavior_cohort",
        b"palette.export_run_id": str(plan["export_run_id"]).encode("utf-8"),
        b"palette.export_plan_sha256": str(plan["plan_sha256"]).encode("ascii"),
        b"palette.membership_record_sha256": str(
            plan["membership"]["record_sha256"]
        ).encode("ascii"),
        b"palette.bundle_set_record_sha256": str(
            plan["bundle_set"]["record_sha256"]
        ).encode("ascii"),
        b"palette.member_ordinal": str(member["ordinal"]).encode("ascii"),
        b"palette.recording_id": str(member["recording_id"]).encode("utf-8"),
        b"palette.table_name": table_name.encode("utf-8"),
    }


def _validate_extracted_rows(
    rows: Sequence[Mapping[str, Any]], spec: ValidatedBehaviorTableSpec
) -> None:
    """Reject adapter shape/null drift before Arrow can normalize it silently."""

    fields = tuple(spec.contract.fields)
    expected = {item.name for item in fields}
    required = {item.name for item in fields if not item.nullable}
    for index, row in enumerate(rows):
        actual = set(row)
        if actual != expected:
            _fail(
                f"{spec.table_name}: extracted row {index} has an inexact field set; "
                f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}."
            )
        null_required = sorted(name for name in required if row[name] is None)
        if null_required:
            _fail(
                f"{spec.table_name}: extracted row {index} has null required fields: "
                f"{null_required!r}."
            )


def _part_receipt(
    *,
    part: Path,
    relative_path: str,
    row_count: int,
    spec: ValidatedBehaviorTableSpec,
    key_bounds: Mapping[str, object] | None,
) -> dict[str, Any]:
    return {
        "path": relative_path,
        "size_bytes": part.stat().st_size,
        "row_count": row_count,
        "file_sha256": sha256_file(part),
        "arrow_schema_id": spec.contract.schema_id,
        "arrow_schema_version": spec.contract.schema_version,
        "arrow_schema_sha256": spec.contract.payload_sha256,
        "primary_key": list(spec.contract.primary_key),
        "primary_key_bounds": _plain(key_bounds),
    }


def _planned_zero_row_reason(
    plan: Mapping[str, Any], *, table_name: str, member_ordinal: int
) -> str | None:
    coverage = _mapping(plan["table_coverage"][table_name], field="table coverage")
    contributors = coverage["contributing_member_ordinals"]
    if member_ordinal in contributors:
        return None
    states = coverage.get("member_capability_states")
    if not isinstance(states, list):
        _fail(f"{table_name}: metadata member cannot be a non-contributor.")
    matching = [
        item
        for item in states
        if isinstance(item, Mapping) and item.get("member_ordinal") == member_ordinal
    ]
    if len(matching) != 1:
        _fail(f"{table_name}: member capability coverage is incomplete.")
    state = _text(matching[0].get("state"), field=f"{table_name} capability state")
    reason = matching[0].get("reason_code")
    reason_text = (
        "no-reason"
        if reason is None
        else _text(reason, field=f"{table_name} capability reason")
    )
    return safe_component(f"capability-{state}-{reason_text}", label="zero-row reason")


def _validate_part(
    part: Path,
    receipt: Mapping[str, Any],
    *,
    spec: ValidatedBehaviorTableSpec,
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    hash_bytes: bool,
) -> None:
    import pyarrow.parquet as pq

    if part.is_symlink() or not part.is_file():
        _fail(f"Selected Parquet part is absent or aliased: {part}")
    if part.stat().st_size != receipt.get("size_bytes"):
        _fail(f"{spec.table_name}: part size differs from its receipt.")
    if hash_bytes and sha256_file(part) != receipt.get("file_sha256"):
        _fail(f"{spec.table_name}: part bytes differ from its receipt.")
    parquet = pq.ParquetFile(part)
    if int(parquet.metadata.num_rows) != receipt.get("row_count"):
        _fail(f"{spec.table_name}: Parquet row count differs from its receipt.")
    validate_exact_schema(spec.contract, parquet.schema_arrow)
    metadata = parquet.schema_arrow.metadata or {}
    expected = _part_footer(plan=plan, member=member, table_name=spec.table_name)
    for key, value in expected.items():
        if metadata.get(key) != value:
            _fail(f"{spec.table_name}: Parquet provenance footer is invalid.")


def _observed_primary_key_summary(
    part: Path,
    spec: ValidatedBehaviorTableSpec,
    *,
    plan: Mapping[str, Any] | None = None,
    member: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, object] | None]:
    import pyarrow.parquet as pq

    seen: set[tuple[Any, ...]] | None = (
        set() if spec.primary_key_validation == "unordered_unique_v1" else None
    )
    count = 0
    previous: tuple[Any, ...] | None = None
    minimum: tuple[Any, ...] | None = None
    maximum: tuple[Any, ...] | None = None
    parquet = pq.ParquetFile(part)
    for batch in parquet.iter_batches(columns=list(spec.contract.primary_key)):
        columns = [column.to_pylist() for column in batch.columns]
        for key in zip(*columns, strict=True):
            if (
                plan is not None
                and member is not None
                and key[:2]
                != (
                    plan["export_run_id"],
                    member["recording_id"],
                )
            ):
                _fail(
                    f"{spec.table_name}: in-row shard identity differs from its owner."
                )
            if seen is not None:
                if key in seen:
                    _fail(f"{spec.table_name}: shard contains a duplicate primary key.")
                seen.add(key)
            elif previous is not None and key <= previous:
                _fail(
                    f"{spec.table_name}: shard primary keys are not strictly increasing."
                )
            previous = key
            count += 1
            minimum = key if minimum is None or key < minimum else minimum
            maximum = key if maximum is None or key > maximum else maximum
    bounds = (
        None
        if minimum is None
        else {"minimum": list(minimum), "maximum": list(maximum)}
    )
    return count, bounds


def _validate_shard_receipt(
    value: object,
    *,
    shard_root: Path | None,
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec],
    hash_parts: bool,
) -> Mapping[str, Any]:
    receipt = _validate_self_digest(
        _mapping(value, field="shard receipt"),
        digest_field="record_sha256",
        field="Shard receipt",
    )
    schema_version = receipt.get("schema_version")
    if schema_version == LEGACY_SHARD_SCHEMA_VERSION:
        required = _SHARD_FIELDS_V1
        expected_method = LEGACY_SHARD_METHOD_ID
        expected_policy = LEGACY_SHARD_VALIDATION_POLICY
    elif schema_version == SHARD_SCHEMA_VERSION:
        required = _SHARD_FIELDS_V2
        expected_method = SHARD_METHOD_ID
        expected_policy = SHARD_VALIDATION_POLICY
    else:
        _fail("Shard receipt schema version is unsupported.")
    if set(receipt) != required:
        _fail("Shard receipt field set is inexact.")
    if (
        receipt.get("schema_id") != SHARD_SCHEMA_ID
        or receipt.get("method_id") != expected_method
        or receipt.get("status") != SHARD_STATUS
        or receipt.get("validation_policy") != expected_policy
        or receipt.get("safety") != SAFETY
    ):
        _fail(
            "Shard receipt identity, method, status, validation, or safety is invalid."
        )
    if receipt.get("export_run_id") != plan["export_run_id"]:
        _fail("Shard belongs to another export run.")
    _plan_path, plan_binding = _validate_plan_file_binding(
        receipt.get("export_plan"), expected_plan=plan
    )
    if plan_binding["plan_sha256"] != plan["plan_sha256"]:
        _fail("Shard binds another export plan.")
    expected_member = {
        "ordinal": member["ordinal"],
        "recording_id": member["recording_id"],
        "analysis_zarr": member["analysis_zarr"],
        "membership_member_sha256": member["membership_member_sha256"],
        "bundle_set_member_sha256": member["bundle_set_member_sha256"],
        "bundle_state": member["bundle_state"],
        "capabilities_sha256": member["capabilities_sha256"],
    }
    if receipt.get("member") != expected_member:
        _fail("Shard member identity differs from its plan.")
    if (
        receipt.get("membership") != plan["membership"]
        or receipt.get("bundle_set") != plan["bundle_set"]
        or receipt.get("parameters") != plan["parameters"]
        or receipt.get("table_coverage") != plan["table_coverage"]
        or receipt.get("software_authority") != plan["software_authority"]
    ):
        _fail("Shard inputs, parameters, or software differ from its plan.")
    table_names = validate_table_specs(table_specs)
    if receipt.get("requested_tables") != list(table_names):
        _fail("Shard requested-table roster is inexact.")
    parts = _mapping(receipt.get("parts_by_table"), field="parts_by_table")
    zeros = _mapping(
        receipt.get("zero_row_reasons_by_table"), field="zero_row_reasons_by_table"
    )
    if set(parts) != set(table_names) or set(zeros) != set(table_names):
        _fail("Shard part or zero-row inventory is incomplete.")
    expected_files = {"receipt.json"}
    for table_name in table_names:
        spec = table_specs[table_name]
        raw = _mapping(parts[table_name], field=f"parts_by_table.{table_name}")
        if set(raw) != _SHARD_PART_FIELDS:
            _fail(f"{table_name}: shard part receipt field set is inexact.")
        expected_path = f"tables/{table_name}/part-00000.parquet"
        if raw.get("path") != expected_path:
            _fail(f"{table_name}: shard part path is invalid.")
        expected_files.add(expected_path)
        row_count = raw.get("row_count")
        zero_reason = zeros[table_name]
        planned_noncontributor_reason = _planned_zero_row_reason(
            plan,
            table_name=table_name,
            member_ordinal=int(member["ordinal"]),
        )
        if type(row_count) is not int or row_count < 0:
            _fail(f"{table_name}: row count is invalid.")
        if type(raw.get("size_bytes")) is not int or raw["size_bytes"] <= 0:
            _fail(f"{table_name}: part size is invalid.")
        if row_count == 0:
            if not spec.zero_rows_allowed or type(zero_reason) is not str:
                _fail(f"{table_name}: zero rows lack one permitted typed reason.")
            safe_component(zero_reason, label="zero-row reason")
            if (
                planned_noncontributor_reason is not None
                and zero_reason != planned_noncontributor_reason
            ):
                _fail(f"{table_name}: non-contributor zero-row reason is invalid.")
        elif zero_reason is not None:
            _fail(f"{table_name}: non-empty part cannot carry a zero-row reason.")
        elif planned_noncontributor_reason is not None:
            _fail(f"{table_name}: non-contributor emitted scientific rows.")
        if (
            raw.get("arrow_schema_id") != spec.contract.schema_id
            or raw.get("arrow_schema_version") != spec.contract.schema_version
            or raw.get("arrow_schema_sha256") != spec.contract.payload_sha256
            or raw.get("primary_key") != list(spec.contract.primary_key)
        ):
            _fail(f"{table_name}: part contract identity is invalid.")
        _digest(raw.get("file_sha256"), field=f"{table_name}.file_sha256")
        if shard_root is not None:
            _validate_part(
                shard_root / expected_path,
                raw,
                spec=spec,
                plan=plan,
                member=member,
                hash_bytes=hash_parts,
            )
        if schema_version == LEGACY_SHARD_SCHEMA_VERSION:
            if shard_root is None:
                _fail("Legacy shard validation requires its physical shard root.")
            key_count, key_bounds = _observed_primary_key_summary(
                shard_root / expected_path,
                spec,
                plan=plan,
                member=member,
            )
            if key_count != row_count or raw.get("primary_key_bounds") != key_bounds:
                _fail(f"{table_name}: primary-key count or bounds are stale.")
        else:
            if shard_root is not None:
                _require_frozen_regular_file(
                    shard_root / expected_path, field=f"{table_name} shard part"
                )
    if schema_version == SHARD_SCHEMA_VERSION:
        if receipt.get("mutation_exclusion") != MUTATION_EXCLUSION_POLICY:
            _fail("Shard mutation-exclusion policy is invalid.")
        _validate_shard_semantic_validation(
            receipt.get("semantic_validation"),
            plan=plan,
            member=member,
            table_specs=table_specs,
            parts=parts,
        )
        if shard_root is not None:
            _require_frozen_regular_file(
                shard_root / "receipt.json", field="shard receipt"
            )
    if shard_root is not None:
        actual_files = {
            path.relative_to(shard_root).as_posix()
            for path in shard_root.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            _fail("Shard directory contains files outside its closed inventory.")
    _validate_software(receipt.get("software_authority"))
    _timestamp(receipt.get("created_at_utc"), field="created_at_utc")
    return receipt


def read_validated_behavior_shard_receipt(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
    hash_parts: bool = True,
) -> Mapping[str, Any]:
    source, value = _strict_object(path, field="shard receipt")
    expected = planned_shard_receipt_path(plan, member)
    if source != expected:
        _fail("Shard receipt path differs from the deterministic plan path.")
    return _validate_shard_receipt(
        value,
        shard_root=source.parent,
        plan=plan,
        member=member,
        table_specs=table_specs,
        hash_parts=hash_parts,
    )


def write_validated_behavior_recording_shard(
    *,
    plan_path: str | Path,
    member_ordinal: int,
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
    row_extractors: Mapping[str, RowExtractor] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write or exactly reuse one recording-owned shard."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    plan_file = Path(plan_path).expanduser().resolve()
    plan, membership, bundle_set = read_validated_behavior_export_plan(
        plan_file, table_specs=table_specs
    )
    _require_current_plan_evidence(plan)
    if (
        type(member_ordinal) is not int
        or not 1 <= member_ordinal <= plan["member_count"]
    ):
        _fail("member_ordinal is outside the closed plan axis.")
    member = plan["members"][member_ordinal - 1]
    membership_member = membership["members"][member_ordinal - 1]
    bundle_member = bundle_set["members"][member_ordinal - 1]
    final_root = planned_shard_root(plan, member)
    final_receipt = final_root / "receipt.json"
    if final_root.exists() or final_root.is_symlink():
        receipt = read_validated_behavior_shard_receipt(
            final_receipt,
            plan=plan,
            member=member,
            table_specs=table_specs,
            hash_parts=False,
        )
        return {**_plain(receipt), "receipt_path": str(final_receipt), "reused": True}
    stage = final_root.parent / f".{final_root.name}.{uuid.uuid4().hex}.tmp"
    stage.mkdir(parents=True, exist_ok=False)
    extractors = dict(row_extractors or {})
    table_names = validate_table_specs(table_specs)
    try:
        parts: dict[str, dict[str, Any]] = {}
        primary_key_distinct_counts: dict[str, int] = {}
        zero_reasons: dict[str, str | None] = {}
        for table_name in table_names:
            spec = table_specs[table_name]
            planned_zero_reason = _planned_zero_row_reason(
                plan, table_name=table_name, member_ordinal=member_ordinal
            )
            if planned_zero_reason is not None:
                rows, zero_reason = [], planned_zero_reason
            elif (
                table_name in CORE_TABLE_SPECS and spec == CORE_TABLE_SPECS[table_name]
            ):
                rows, zero_reason = _core_rows(
                    table_name, plan, membership_member, bundle_member
                )
            elif table_name in extractors:
                extracted = extractors[table_name](
                    plan, membership_member, bundle_member
                )
                if isinstance(extracted, ValidatedBehaviorBatchSource):
                    rows = None
                    zero_reason = extracted.zero_row_reason
                else:
                    raw_rows, zero_reason = extracted
                    rows = [dict(row) for row in raw_rows]
            else:
                _fail(f"No recording-scoped extractor is installed for {table_name!r}.")
            if rows is not None:
                if not rows and (not spec.zero_rows_allowed or zero_reason is None):
                    _fail(
                        f"{table_name}: extractor returned an uncontracted empty result."
                    )
                if rows and zero_reason is not None:
                    _fail(
                        f"{table_name}: non-empty rows cannot carry a zero-row reason."
                    )
                _validate_extracted_rows(rows, spec)
                resident_semantics = _resident_row_semantics(
                    rows,
                    spec,
                    export_run_id=str(plan["export_run_id"]),
                    recording_id=str(member["recording_id"]),
                )
            table_dir = stage / "tables" / table_name
            table_dir.mkdir(parents=True, exist_ok=False)
            part = table_dir / "part-00000.parquet"
            schema = exact_schema(
                spec.contract,
                metadata=_part_footer(plan=plan, member=member, table_name=table_name),
            )
            temporary_part = table_dir / ".part-00000.parquet.tmp"
            if rows is not None:
                table = pa.Table.from_pylist(rows, schema=schema)
                pq.write_table(
                    table,
                    temporary_part,
                    compression=str(plan["parameters"]["parquet_compression"]),
                    row_group_size=int(plan["parameters"]["effective_row_group_rows"]),
                )
                row_count = table.num_rows
                key_bounds = resident_semantics["primary_key_bounds"]
                primary_key_distinct_count = int(
                    resident_semantics["primary_key_distinct_count"]
                )
            else:
                if spec.primary_key_validation != "strictly_increasing_v1":
                    _fail(
                        f"{table_name}: batch sources require strictly increasing "
                        "primary-key validation."
                    )
                writer = pq.ParquetWriter(
                    temporary_part,
                    schema,
                    compression=str(plan["parameters"]["parquet_compression"]),
                )
                row_count = 0
                minimum_key: tuple[Any, ...] | None = None
                previous_key: tuple[Any, ...] | None = None
                try:
                    for batch_index, columns in enumerate(extracted.batches):
                        if not isinstance(columns, Mapping):
                            _fail(
                                f"{table_name}: extracted column batch {batch_index} "
                                "is not one mapping."
                            )
                        _validate_extracted_columns(
                            columns, spec, batch_index=batch_index
                        )
                        table = pa.Table.from_pydict(dict(columns), schema=schema)
                        for field_contract in spec.contract.fields:
                            if (
                                not field_contract.nullable
                                and table.column(field_contract.name).null_count
                            ):
                                _fail(
                                    f"{table_name}: extracted column batch "
                                    f"{batch_index} has null required field "
                                    f"{field_contract.name!r}."
                                )
                        previous_key, minimum_key = _advance_ordered_key_summary(
                            table,
                            spec,
                            previous=previous_key,
                            minimum=minimum_key,
                            expected_owner=(
                                str(plan["export_run_id"]),
                                str(member["recording_id"]),
                            ),
                        )
                        if table.num_rows:
                            writer.write_table(
                                table,
                                row_group_size=int(
                                    plan["parameters"]["effective_row_group_rows"]
                                ),
                            )
                            row_count += table.num_rows
                finally:
                    writer.close()
                if row_count == 0 and (
                    not spec.zero_rows_allowed or zero_reason is None
                ):
                    _fail(
                        f"{table_name}: extractor returned an uncontracted empty result."
                    )
                if row_count and zero_reason is not None:
                    _fail(
                        f"{table_name}: non-empty rows cannot carry a zero-row reason."
                    )
                key_bounds = (
                    None
                    if minimum_key is None
                    else {
                        "minimum": list(minimum_key),
                        "maximum": list(previous_key),
                    }
                )
                primary_key_distinct_count = row_count
            os.replace(temporary_part, part)
            parts[table_name] = _part_receipt(
                part=part,
                relative_path=f"tables/{table_name}/part-00000.parquet",
                row_count=row_count,
                spec=spec,
                key_bounds=key_bounds,
            )
            primary_key_distinct_counts[table_name] = primary_key_distinct_count
            zero_reasons[table_name] = zero_reason
        semantic_validation = _build_shard_semantic_validation(
            shard_root=stage,
            plan=plan,
            member=member,
            table_specs=table_specs,
            parts=parts,
            primary_key_distinct_counts=primary_key_distinct_counts,
        )
        body = {
            "schema_id": SHARD_SCHEMA_ID,
            "schema_version": SHARD_SCHEMA_VERSION,
            "method_id": SHARD_METHOD_ID,
            "status": SHARD_STATUS,
            "export_run_id": plan["export_run_id"],
            "export_plan": {
                "path": str(plan_file),
                "file_sha256": sha256_file(plan_file),
                "plan_sha256": plan["plan_sha256"],
            },
            "member": {
                "ordinal": member["ordinal"],
                "recording_id": member["recording_id"],
                "analysis_zarr": member["analysis_zarr"],
                "membership_member_sha256": member["membership_member_sha256"],
                "bundle_set_member_sha256": member["bundle_set_member_sha256"],
                "bundle_state": member["bundle_state"],
                "capabilities_sha256": member["capabilities_sha256"],
            },
            "membership": plan["membership"],
            "bundle_set": plan["bundle_set"],
            "requested_tables": list(table_names),
            "table_coverage": plan["table_coverage"],
            "parts_by_table": parts,
            "zero_row_reasons_by_table": zero_reasons,
            "parameters": plan["parameters"],
            "validation_policy": SHARD_VALIDATION_POLICY,
            "semantic_validation": semantic_validation,
            "mutation_exclusion": MUTATION_EXCLUSION_POLICY,
            "software_authority": plan["software_authority"],
            "created_at_utc": _timestamp(
                created_at_utc or _default_created_at(), field="created_at_utc"
            ),
            "safety": SAFETY,
        }
        receipt = _sealed(body, digest_field="record_sha256")
        _write_json(stage / "receipt.json", receipt)
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                _freeze_regular_file(path)
        _validate_shard_receipt(
            receipt,
            shard_root=stage,
            plan=plan,
            member=member,
            table_specs=table_specs,
            hash_parts=False,
        )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final_root)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {**receipt, "receipt_path": str(final_receipt), "reused": False}


def validated_behavior_manifest_path(root: str | Path, export_run_id: str) -> Path:
    publication = _absolute_path(root, field="publication_root")
    run_id = safe_component(export_run_id, label="export run ID")
    return (
        publication
        / "validated_behavior"
        / "v1"
        / "manifests"
        / f"export_run_id={run_id}.json"
    )


def _generation_relative_path(export_run_id: str, generation_id: str) -> Path:
    run_id = safe_component(export_run_id, label="export run ID")
    generation = safe_component(generation_id, label="generation ID")
    return (
        Path("validated_behavior")
        / "v1"
        / ".generations"
        / f"export_run_id={run_id}"
        / f"generation={generation}"
    )


def _safe_selected_path(root: Path, relative_text: object, *, field: str) -> Path:
    text = _text(relative_text, field=field)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != text:
        _fail(f"{field} is not a safe manifest-relative path.")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            _fail(f"{field} contains a symbolic-link alias.")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidatedBehaviorExportError(
            f"{field} escapes publication root."
        ) from exc
    return resolved


def _validate_published_v2_shard_evidence(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec],
) -> None:
    validated = _validate_shard_receipt(
        receipt,
        shard_root=None,
        plan=plan,
        member=member,
        table_specs=table_specs,
        hash_parts=False,
    )
    if validated.get("schema_version") != SHARD_SCHEMA_VERSION:
        _fail("Published receipt is not the required v2 shard evidence.")


def _validate_published_shard_roster(
    generation_root: Path,
    *,
    generation_relative_path: str,
    plan: Mapping[str, Any],
    roster: object,
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] | None = None,
) -> tuple[set[str], str | None]:
    entries = _list(roster, field="shard_receipts")
    if len(entries) != plan["member_count"]:
        _fail("Published shard-receipt roster does not close the member axis.")
    expected_files: set[str] = set()
    normalized_for_digest: list[dict[str, Any]] = []
    semantic_digests: list[str] = []
    for member, raw in zip(plan["members"], entries, strict=True):
        entry = _mapping(raw, field="shard receipt roster entry")
        if set(entry) != {
            "member_ordinal",
            "recording_id",
            "source_path",
            "path",
            "size_bytes",
            "file_sha256",
            "record_sha256",
        }:
            _fail("Published shard-receipt roster field set is inexact.")
        ordinal = int(member["ordinal"])
        recording_id = str(member["recording_id"])
        relative_inside_generation = (
            Path("provenance")
            / "shard_receipts"
            / f"member={ordinal:06d}-{safe_component(recording_id, label='recording ID')}.json"
        )
        expected_path = (
            Path(generation_relative_path) / relative_inside_generation
        ).as_posix()
        if (
            entry.get("member_ordinal") != ordinal
            or entry.get("recording_id") != recording_id
            or entry.get("path") != expected_path
        ):
            _fail("Published shard-receipt roster identity or path is invalid.")
        _absolute_path(entry.get("source_path"), field="shard_receipt.source_path")
        if type(entry.get("size_bytes")) is not int or entry["size_bytes"] <= 0:
            _fail("Published shard-receipt size is invalid.")
        _digest(entry.get("file_sha256"), field="shard_receipt.file_sha256")
        record_sha = _digest(
            entry.get("record_sha256"), field="shard_receipt.record_sha256"
        )
        receipt_path = generation_root / relative_inside_generation
        expected_files.add(relative_inside_generation.as_posix())
        if (
            receipt_path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.stat().st_size != entry["size_bytes"]
            or sha256_file(receipt_path) != entry["file_sha256"]
        ):
            _fail("Published shard-receipt copy is absent, aliased, or changed.")
        _source, receipt_raw = _strict_object(
            receipt_path, field="published shard receipt"
        )
        receipt = _validate_self_digest(
            receipt_raw,
            digest_field="record_sha256",
            field="Published shard receipt",
        )
        receipt_plan = _mapping(
            receipt.get("export_plan"), field="published shard export_plan"
        )
        receipt_member = _mapping(receipt.get("member"), field="published shard member")
        if (
            receipt["record_sha256"] != record_sha
            or receipt.get("export_run_id") != plan["export_run_id"]
            or receipt_plan.get("plan_sha256") != plan["plan_sha256"]
            or receipt_member.get("ordinal") != ordinal
            or receipt_member.get("recording_id") != recording_id
        ):
            _fail("Published shard receipt binds another plan or member.")
        if receipt.get("schema_version") == SHARD_SCHEMA_VERSION:
            if table_specs is None:
                _fail("Published v2 shard evidence requires installed table specs.")
            _validate_published_v2_shard_evidence(
                receipt,
                plan=plan,
                member=member,
                table_specs=table_specs,
            )
            _require_frozen_regular_file(receipt_path, field="published shard receipt")
            semantic_digests.append(
                _digest(
                    _mapping(
                        receipt.get("semantic_validation"),
                        field="semantic_validation",
                    ).get("record_sha256"),
                    field="semantic_validation.record_sha256",
                )
            )
        elif receipt.get("schema_version") != LEGACY_SHARD_SCHEMA_VERSION:
            _fail("Published shard receipt schema version is unsupported.")
        normalized_for_digest.append(_plain(entry))
    if normalized_for_digest != _plain(entries):
        _fail("Published shard-receipt roster is not deterministically ordered.")
    if semantic_digests and len(semantic_digests) != len(entries):
        _fail("Published shard roster mixes legacy and receipt-composed evidence.")
    return (
        expected_files,
        canonical_json_sha256(semantic_digests) if semantic_digests else None,
    )


def _part_relation_values(part: Path, fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
    """Collect one recording-scoped target relation, never a cohort relation."""

    import pyarrow.parquet as pq

    values: set[tuple[Any, ...]] = set()
    parquet = pq.ParquetFile(part)
    for batch in parquet.iter_batches(columns=list(fields)):
        columns = [column.to_pylist() for column in batch.columns]
        values.update(zip(*columns, strict=True))
    return values


def _part_foreign_key_is_closed(
    local_part: Path,
    local_fields: tuple[str, ...],
    target_values: set[tuple[Any, ...]],
) -> bool:
    """Stream one local relation against a recording-bounded target set."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(local_part)
    for batch in parquet.iter_batches(columns=list(local_fields)):
        columns = [column.to_pylist() for column in batch.columns]
        if any(key not in target_values for key in zip(*columns, strict=True)):
            return False
    return True


def _foreign_key_observation(
    local_part: Path,
    local_fields: tuple[str, ...],
    target_part: Path,
    target_fields: tuple[str, ...],
) -> tuple[int, int, int]:
    """Return local rows, target distinct keys, and unmatched local rows."""

    import pyarrow.parquet as pq

    target_values = _part_relation_values(target_part, target_fields)
    local_rows = 0
    unmatched = 0
    parquet = pq.ParquetFile(local_part)
    for batch in parquet.iter_batches(columns=list(local_fields)):
        columns = [column.to_pylist() for column in batch.columns]
        for key in zip(*columns, strict=True):
            local_rows += 1
            if key not in target_values:
                unmatched += 1
    return local_rows, len(target_values), unmatched


def _build_shard_semantic_validation(
    *,
    shard_root: Path,
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec],
    parts: Mapping[str, Mapping[str, Any]],
    primary_key_distinct_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Seal recording-local semantic proofs against exact part hashes."""

    table_names = validate_table_specs(table_specs)
    table_results = {
        table_name: _semantic_table_result(
            spec=table_specs[table_name],
            part=parts[table_name],
            export_run_id=str(plan["export_run_id"]),
            recording_id=str(member["recording_id"]),
            primary_key_distinct_count=int(primary_key_distinct_counts[table_name]),
        )
        for table_name in table_names
    }
    foreign_key_results: list[dict[str, Any]] = []
    owner_fields = ("export_run_id", "recording_id")
    for table_name in table_names:
        spec = table_specs[table_name]
        for local_fields, target_table, target_fields in spec.foreign_keys:
            local_entry = parts[table_name]
            target_entry = parts[target_table]
            local_count = int(local_entry["row_count"])
            target_count = int(target_entry["row_count"])
            if (
                tuple(local_fields) == owner_fields
                and tuple(target_fields) == owner_fields
            ):
                target_distinct_count = 1 if target_count else 0
                unmatched_count = 0 if local_count == 0 or target_count else local_count
                method_id = "recording_owner_partition_implication_v1"
            else:
                (
                    observed_local_count,
                    target_distinct_count,
                    unmatched_count,
                ) = _foreign_key_observation(
                    shard_root / str(local_entry["path"]),
                    tuple(local_fields),
                    shard_root / str(target_entry["path"]),
                    tuple(target_fields),
                )
                if observed_local_count != local_count:
                    _fail(f"{table_name}: foreign-key local row count is stale.")
                method_id = "exact_recording_local_relation_scan_v1"
            if unmatched_count:
                _fail(f"{table_name}: foreign key to {target_table} is not closed.")
            foreign_key_results.append(
                {
                    "local_table": table_name,
                    "local_fields": list(local_fields),
                    "target_table": target_table,
                    "target_fields": list(target_fields),
                    "local_part_file_sha256": local_entry["file_sha256"],
                    "target_part_file_sha256": target_entry["file_sha256"],
                    "local_row_count": local_count,
                    "target_row_count": target_count,
                    "target_distinct_key_count": target_distinct_count,
                    "unmatched_count": unmatched_count,
                    "method_id": method_id,
                    "status": "complete",
                }
            )
    body = {
        "schema_id": SHARD_SEMANTIC_SCHEMA_ID,
        "schema_version": SHARD_SEMANTIC_SCHEMA_VERSION,
        "method_id": SHARD_SEMANTIC_METHOD_ID,
        "status": SHARD_SEMANTIC_STATUS,
        "partition_contract": {
            "method_id": "one_part_per_member_table_recording_owner_v1",
            "owner_fields": list(owner_fields),
            "owner_values": [plan["export_run_id"], member["recording_id"]],
            "table_count": len(table_names),
            "part_count": len(table_names),
            "status": "complete",
        },
        "table_results": table_results,
        "foreign_key_results": foreign_key_results,
        "foreign_key_result_count": len(foreign_key_results),
    }
    return _sealed(body, digest_field="record_sha256")


def _validate_shard_semantic_validation(
    value: object,
    *,
    plan: Mapping[str, Any],
    member: Mapping[str, Any],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec],
    parts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    semantic = _validate_self_digest(
        _mapping(value, field="semantic_validation"),
        digest_field="record_sha256",
        field="Shard semantic validation",
    )
    if set(semantic) != {
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "partition_contract",
        "table_results",
        "foreign_key_results",
        "foreign_key_result_count",
        "record_sha256",
    }:
        _fail("Shard semantic-validation field set is inexact.")
    if (
        semantic.get("schema_id") != SHARD_SEMANTIC_SCHEMA_ID
        or semantic.get("schema_version") != SHARD_SEMANTIC_SCHEMA_VERSION
        or semantic.get("method_id") != SHARD_SEMANTIC_METHOD_ID
        or semantic.get("status") != SHARD_SEMANTIC_STATUS
    ):
        _fail("Shard semantic-validation identity or status is invalid.")
    table_names = validate_table_specs(table_specs)
    expected_partition = {
        "method_id": "one_part_per_member_table_recording_owner_v1",
        "owner_fields": ["export_run_id", "recording_id"],
        "owner_values": [plan["export_run_id"], member["recording_id"]],
        "table_count": len(table_names),
        "part_count": len(table_names),
        "status": "complete",
    }
    if semantic.get("partition_contract") != expected_partition:
        _fail("Shard semantic partition contract is invalid.")
    table_results = _mapping(
        semantic.get("table_results"), field="semantic table_results"
    )
    if set(table_results) != set(table_names):
        _fail("Shard semantic table-result roster is incomplete.")
    for table_name in table_names:
        expected = _semantic_table_result(
            spec=table_specs[table_name],
            part=parts[table_name],
            export_run_id=str(plan["export_run_id"]),
            recording_id=str(member["recording_id"]),
            primary_key_distinct_count=int(parts[table_name]["row_count"]),
        )
        if table_results[table_name] != expected:
            _fail(f"{table_name}: semantic table proof is invalid or incomplete.")
    raw_foreign_keys = _list(
        semantic.get("foreign_key_results"), field="foreign_key_results"
    )
    expected_declarations = [
        (table_name, tuple(local), target, tuple(target_fields))
        for table_name in table_names
        for local, target, target_fields in table_specs[table_name].foreign_keys
    ]
    if semantic.get("foreign_key_result_count") != len(raw_foreign_keys) or len(
        raw_foreign_keys
    ) != len(expected_declarations):
        _fail("Shard foreign-key proof roster is incomplete.")
    for index, (raw, declaration) in enumerate(
        zip(raw_foreign_keys, expected_declarations, strict=True)
    ):
        proof = _mapping(raw, field=f"foreign_key_results[{index}]")
        if set(proof) != {
            "local_table",
            "local_fields",
            "target_table",
            "target_fields",
            "local_part_file_sha256",
            "target_part_file_sha256",
            "local_row_count",
            "target_row_count",
            "target_distinct_key_count",
            "unmatched_count",
            "method_id",
            "status",
        }:
            _fail("Shard foreign-key proof field set is inexact.")
        table_name, local_fields, target_table, target_fields = declaration
        local_count = int(parts[table_name]["row_count"])
        target_count = int(parts[target_table]["row_count"])
        prefix_only = (
            local_fields
            == target_fields
            == (
                "export_run_id",
                "recording_id",
            )
        )
        expected_method = (
            "recording_owner_partition_implication_v1"
            if prefix_only
            else "exact_recording_local_relation_scan_v1"
        )
        if (
            proof.get("local_table") != table_name
            or proof.get("local_fields") != list(local_fields)
            or proof.get("target_table") != target_table
            or proof.get("target_fields") != list(target_fields)
            or proof.get("local_part_file_sha256") != parts[table_name]["file_sha256"]
            or proof.get("target_part_file_sha256")
            != parts[target_table]["file_sha256"]
            or proof.get("local_row_count") != local_count
            or proof.get("target_row_count") != target_count
            or proof.get("unmatched_count") != 0
            or proof.get("method_id") != expected_method
            or proof.get("status") != "complete"
        ):
            _fail(f"{table_name}: foreign-key proof is invalid or incomplete.")
        distinct = proof.get("target_distinct_key_count")
        if type(distinct) is not int or not 0 <= distinct <= target_count:
            _fail(f"{table_name}: foreign-key target cardinality is invalid.")
        if prefix_only and distinct != (1 if target_count else 0):
            _fail(f"{table_name}: recording-owner foreign-key proof is invalid.")
        if local_count and target_count == 0:
            _fail(f"{table_name}: foreign key to {target_table} is not closed.")
    return semantic


def _global_validate_generation(
    generation_root: Path,
    *,
    generation_relative_path: str,
    plan: Mapping[str, Any],
    inventory: Mapping[str, Sequence[Mapping[str, Any]]],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec],
    hash_parts: bool,
    validate_keys: bool = True,
    additional_expected_files: set[str] | None = None,
    require_frozen_files: bool = False,
) -> dict[str, Any]:
    table_names = validate_table_specs(table_specs)
    if set(inventory) != set(table_names):
        _fail("Publication part inventory is incomplete.")
    row_counts: dict[str, int] = {}
    primary_key_counts: dict[str, int | None] = {}
    expected_files: set[str] = set()
    member_by_ordinal = {int(item["ordinal"]): item for item in plan["members"]}
    expected_ordinals = tuple(member_by_ordinal)
    if (
        len(member_by_ordinal) != plan["member_count"]
        or len({str(item["recording_id"]) for item in plan["members"]})
        != plan["member_count"]
    ):
        _fail("Export plan member identities are not unique.")
    parts_by_table_ordinal: dict[str, dict[int, Path]] = {}
    row_counts_by_table_ordinal: dict[str, dict[int, int]] = {}
    for table_name in table_names:
        spec = table_specs[table_name]
        count = 0
        entries = list(inventory[table_name])
        if len(entries) != plan["member_count"]:
            _fail(f"{table_name}: publication lacks one part per parent member.")
        observed_ordinals = tuple(entry.get("member_ordinal") for entry in entries)
        if observed_ordinals != expected_ordinals:
            _fail(f"{table_name}: publication part roster is not exact and ordered.")
        parts_by_ordinal: dict[int, Path] = {}
        rows_by_ordinal: dict[int, int] = {}
        for entry in entries:
            if set(entry) != _PUBLICATION_PART_FIELDS:
                _fail(f"{table_name}: publication part field set is inexact.")
            ordinal = entry.get("member_ordinal")
            if type(ordinal) is not int or ordinal not in member_by_ordinal:
                _fail(f"{table_name}: inventory member ordinal is invalid.")
            member = member_by_ordinal[ordinal]
            if entry.get("recording_id") != member["recording_id"]:
                _fail(f"{table_name}: inventory recording identity is invalid.")
            expected_inside_generation = (
                Path("tables")
                / table_name
                / f"member={ordinal:06d}-{safe_component(member['recording_id'], label='recording ID')}"
                / "part-00000.parquet"
            )
            if entry.get("generation_path") != generation_relative_path:
                _fail(f"{table_name}: inventory generation path is invalid.")
            expected_part_path = (
                Path(str(entry["generation_path"])) / expected_inside_generation
            ).as_posix()
            if entry.get("path") != expected_part_path:
                _fail(f"{table_name}: inventory part path is invalid.")
            relative_inside_generation = expected_inside_generation
            part = generation_root / relative_inside_generation
            parts_by_ordinal[ordinal] = part
            rows_by_ordinal[ordinal] = int(entry["row_count"])
            expected_files.add(relative_inside_generation.as_posix())
            _validate_part(
                part,
                entry,
                spec=spec,
                plan=plan,
                member=member,
                hash_bytes=hash_parts,
            )
            if require_frozen_files:
                _require_frozen_regular_file(part, field=f"{table_name} published part")
            if validate_keys:
                part_key_count, part_bounds = _observed_primary_key_summary(
                    part,
                    spec,
                    plan=plan,
                    member=member,
                )
                if (
                    part_key_count != entry["row_count"]
                    or entry.get("primary_key_bounds") != part_bounds
                ):
                    _fail(f"{table_name}: part primary-key bounds are stale.")
            count += int(entry["row_count"])
        parts_by_table_ordinal[table_name] = parts_by_ordinal
        row_counts_by_table_ordinal[table_name] = rows_by_ordinal
        row_counts[table_name] = count
        # Parts are recording-owned, every row's recording identity is checked
        # against that owner, and plan recording IDs are unique. Once each part
        # is unique, cross-part primary-key collisions are therefore impossible.
        primary_key_counts[table_name] = count if validate_keys else None
    if validate_keys:
        for table_name in table_names:
            spec = table_specs[table_name]
            for local_fields, target, target_fields in spec.foreign_keys:
                for ordinal in expected_ordinals:
                    if (
                        tuple(local_fields)
                        == tuple(target_fields)
                        == (
                            "export_run_id",
                            "recording_id",
                        )
                    ):
                        if (
                            row_counts_by_table_ordinal[table_name][ordinal]
                            and not row_counts_by_table_ordinal[target][ordinal]
                        ):
                            _fail(
                                f"{table_name}: foreign key to {target} is not closed."
                            )
                        continue
                    target_values = _part_relation_values(
                        parts_by_table_ordinal[target][ordinal],
                        tuple(target_fields),
                    )
                    if not _part_foreign_key_is_closed(
                        parts_by_table_ordinal[table_name][ordinal],
                        tuple(local_fields),
                        target_values,
                    ):
                        _fail(f"{table_name}: foreign key to {target} is not closed.")
    if (
        "cohort_recordings" in row_counts
        and row_counts["cohort_recordings"] != plan["member_count"]
    ):
        _fail("cohort_recordings does not close the parent roster.")
    if (
        "recording_bundles" in row_counts
        and row_counts["recording_bundles"] != plan["member_count"]
    ):
        _fail("recording_bundles does not close the parent roster.")
    part_file_count = len(expected_files)
    expected_files |= additional_expected_files or set()
    expected_generation_files = expected_files | {"validation/receipt.json"}
    actual_files = {
        path.relative_to(generation_root).as_posix()
        for path in generation_root.rglob("*")
        if path.is_file()
    }
    # During the first validation pass the validation receipt has not been
    # serialized yet.  After it is written, it must be the sole non-table file.
    if actual_files not in (expected_files, expected_generation_files):
        _fail("Publication generation contains files outside its closed inventory.")
    if require_frozen_files:
        for relative_path in sorted(actual_files):
            _require_frozen_regular_file(
                generation_root / relative_path,
                field=f"published generation file {relative_path}",
            )
    return {
        "row_counts_by_table": row_counts,
        "primary_key_counts_by_table": primary_key_counts,
        "foreign_key_validation": "complete" if validate_keys else "receipt_bound",
        "inventory_file_count": part_file_count,
    }


def _compose_generation_validation(
    *,
    plan: Mapping[str, Any],
    shard_receipts: Sequence[Mapping[str, Any]],
    table_specs: Mapping[str, ValidatedBehaviorTableSpec],
) -> dict[str, Any]:
    """Compose cohort closure from exact recording-partitioned shard proofs."""

    table_names = validate_table_specs(table_specs)
    if len(shard_receipts) != plan["member_count"]:
        _fail("Receipt composition does not close the member axis.")
    row_counts = {table_name: 0 for table_name in table_names}
    foreign_key_proof_count = 0
    for member, receipt in zip(plan["members"], shard_receipts, strict=True):
        parts = _mapping(receipt.get("parts_by_table"), field="parts_by_table")
        _validate_shard_semantic_validation(
            receipt.get("semantic_validation"),
            plan=plan,
            member=member,
            table_specs=table_specs,
            parts=parts,
        )
        semantic = _mapping(
            receipt.get("semantic_validation"), field="semantic_validation"
        )
        foreign_key_proof_count += int(semantic["foreign_key_result_count"])
        for table_name in table_names:
            row_counts[table_name] += int(parts[table_name]["row_count"])
    if (
        "cohort_recordings" in row_counts
        and row_counts["cohort_recordings"] != plan["member_count"]
    ):
        _fail("cohort_recordings does not close the parent roster.")
    if (
        "recording_bundles" in row_counts
        and row_counts["recording_bundles"] != plan["member_count"]
    ):
        _fail("recording_bundles does not close the parent roster.")
    return {
        "row_counts_by_table": row_counts,
        "primary_key_counts_by_table": dict(row_counts),
        "owner_validation": "complete_receipt_composed",
        "foreign_key_validation": "complete_receipt_composed",
        "inventory_file_count": len(table_names) * int(plan["member_count"]),
        "shard_receipt_count": len(shard_receipts),
        "semantic_table_proof_count": len(table_names) * len(shard_receipts),
        "foreign_key_proof_count": foreign_key_proof_count,
        "composition_policy": GENERATION_COMPOSITION_POLICY,
    }


def _validate_transfer_receipt(
    value: object,
    *,
    plan: Mapping[str, Any],
    generation_id: str,
    generation_path: str,
    inventory: Mapping[str, Sequence[Mapping[str, Any]]],
    shard_receipts_sha256: str,
    receipt_path: Path,
    binding: object | None = None,
) -> Mapping[str, Any]:
    receipt = _validate_self_digest(
        _mapping(value, field="transfer receipt"),
        digest_field="record_sha256",
        field="Transfer receipt",
    )
    if set(receipt) != {
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "export_run_id",
        "export_plan_sha256",
        "generation_id",
        "generation_path",
        "staging_attempt_id",
        "shard_receipts_sha256",
        "part_inventory_sha256",
        "transfer_verification_policy",
        "part_count",
        "transfers",
        "software_authority",
        "verified_at_utc",
        "safety",
        "record_sha256",
    }:
        _fail("Transfer-receipt field set is inexact.")
    if (
        receipt.get("schema_id") != TRANSFER_RECEIPT_SCHEMA_ID
        or receipt.get("schema_version") != TRANSFER_RECEIPT_SCHEMA_VERSION
        or receipt.get("method_id") != TRANSFER_RECEIPT_METHOD_ID
        or receipt.get("status") != TRANSFER_RECEIPT_STATUS
        or receipt.get("transfer_verification_policy") != TRANSFER_VERIFICATION_POLICY
        or receipt.get("safety") != SAFETY
    ):
        _fail("Transfer-receipt identity, status, policy, or safety is invalid.")
    safe_component(receipt.get("staging_attempt_id"), label="staging attempt ID")
    inventory_sha = canonical_json_sha256(_plain(inventory))
    if (
        receipt.get("export_run_id") != plan["export_run_id"]
        or receipt.get("export_plan_sha256") != plan["plan_sha256"]
        or receipt.get("generation_id") != generation_id
        or receipt.get("generation_path") != generation_path
        or receipt.get("shard_receipts_sha256") != shard_receipts_sha256
        or receipt.get("part_inventory_sha256") != inventory_sha
        or receipt.get("software_authority") != plan["software_authority"]
    ):
        _fail("Transfer receipt binds another plan, roster, or generation.")
    transfers = _list(receipt.get("transfers"), field="transfers")
    expected_entries = [
        (table_name, entry)
        for table_name in sorted(inventory)
        for entry in inventory[table_name]
    ]
    if receipt.get("part_count") != len(transfers) or len(transfers) != len(
        expected_entries
    ):
        _fail("Transfer receipt does not close the part inventory.")
    transfer_fields = {
        "table_name",
        "member_ordinal",
        "recording_id",
        "source_shard_record_sha256",
        "source_part_path",
        "source_part_file_sha256",
        "source_size_bytes",
        "destination_path",
        "observed_size_bytes",
        "observed_file_sha256",
        "method_id",
        "status",
    }
    for index, (raw, (table_name, entry)) in enumerate(
        zip(transfers, expected_entries, strict=True)
    ):
        transfer = _mapping(raw, field=f"transfers[{index}]")
        if set(transfer) != transfer_fields:
            _fail("Transfer entry field set is inexact.")
        if (
            transfer.get("table_name") != table_name
            or transfer.get("member_ordinal") != entry["member_ordinal"]
            or transfer.get("recording_id") != entry["recording_id"]
            or transfer.get("source_shard_record_sha256")
            != entry["source_shard_record_sha256"]
            or transfer.get("source_part_path")
            != f"tables/{table_name}/part-00000.parquet"
            or transfer.get("source_part_file_sha256") != entry["file_sha256"]
            or transfer.get("source_size_bytes") != entry["size_bytes"]
            or transfer.get("destination_path") != entry["path"]
            or transfer.get("observed_size_bytes") != entry["size_bytes"]
            or transfer.get("observed_file_sha256") != entry["file_sha256"]
            or transfer.get("method_id") != TRANSFER_RECEIPT_METHOD_ID
            or transfer.get("status") != "complete"
        ):
            _fail("Transfer entry is stale, incomplete, or out of order.")
    if binding is not None:
        bound = _mapping(binding, field="transfer_receipt")
        if set(bound) != {"path", "size_bytes", "file_sha256", "record_sha256"}:
            _fail("Transfer-receipt file binding is inexact.")
        expected_binding_path = (
            Path(generation_path) / "validation" / "transfer_receipt.json"
        ).as_posix()
        if (
            bound.get("path") != expected_binding_path
            or bound.get("record_sha256") != receipt["record_sha256"]
            or type(bound.get("size_bytes")) is not int
            or bound.get("size_bytes") != receipt_path.stat().st_size
            or bound.get("file_sha256") != sha256_file(receipt_path)
        ):
            _fail("Transfer-receipt file binding is stale.")
    _validate_software(receipt.get("software_authority"))
    _timestamp(receipt.get("verified_at_utc"), field="verified_at_utc")
    return receipt


def publish_validated_behavior_cohort(
    *,
    plan_path: str | Path,
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
    generation_id: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Fan in every exact shard and commit one immutable manifest last."""

    operation_started = time.perf_counter()
    plan_file = Path(plan_path).expanduser().resolve()
    plan, membership, bundle_set = read_validated_behavior_export_plan(
        plan_file, table_specs=table_specs
    )
    _require_current_plan_evidence(plan)
    table_names = validate_table_specs(table_specs)
    shard_receipts: list[tuple[Path, Mapping[str, Any]]] = []
    for member in plan["members"]:
        path = planned_shard_receipt_path(plan, member)
        receipt = read_validated_behavior_shard_receipt(
            path,
            plan=plan,
            member=member,
            table_specs=table_specs,
            hash_parts=False,
        )
        shard_receipts.append((path, receipt))
    shard_receipts_validated_at = time.perf_counter()
    publication_root = _absolute_path(
        plan.get("publication_root"), field="publication_root"
    )
    run_id = str(plan["export_run_id"])
    generation = safe_component(
        generation_id or uuid.uuid4().hex, label="generation ID"
    )
    staging_attempt_id = uuid.uuid4().hex
    validated_at = _timestamp(
        created_at_utc or _default_created_at(), field="validated_at_utc"
    )
    generation_relative = _generation_relative_path(run_id, generation)
    final_generation = publication_root / generation_relative
    manifest_path = validated_behavior_manifest_path(publication_root, run_id)
    stage = (
        publication_root
        / "validated_behavior"
        / "v1"
        / ".staging"
        / f"export_run_id={run_id}-generation={generation}"
    )
    if (
        stage.exists()
        or stage.is_symlink()
        or final_generation.exists()
        or manifest_path.exists()
    ):
        raise FileExistsError("Export staging, generation, or manifest already exists")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        inventory: dict[str, list[dict[str, Any]]] = {name: [] for name in table_names}
        shard_roster: list[dict[str, Any]] = []
        transfers: list[dict[str, Any]] = []
        for member, (receipt_path, receipt) in zip(
            plan["members"], shard_receipts, strict=True
        ):
            receipt_inside_generation = (
                Path("provenance")
                / "shard_receipts"
                / f"member={int(member['ordinal']):06d}-{safe_component(member['recording_id'], label='recording ID')}.json"
            )
            published_receipt = stage / receipt_inside_generation
            published_receipt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(receipt_path, published_receipt)
            _freeze_regular_file(published_receipt)
            shard_roster.append(
                {
                    "member_ordinal": member["ordinal"],
                    "recording_id": member["recording_id"],
                    "source_path": str(receipt_path),
                    "path": (
                        generation_relative / receipt_inside_generation
                    ).as_posix(),
                    "size_bytes": published_receipt.stat().st_size,
                    "file_sha256": sha256_file(published_receipt),
                    "record_sha256": receipt["record_sha256"],
                }
            )
            for table_name in table_names:
                source_entry = receipt["parts_by_table"][table_name]
                source = receipt_path.parent / source_entry["path"]
                relative_inside_generation = (
                    Path("tables")
                    / table_name
                    / f"member={int(member['ordinal']):06d}-{safe_component(member['recording_id'], label='recording ID')}"
                    / "part-00000.parquet"
                )
                target = stage / relative_inside_generation
                target.parent.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(source, target)
                observed_size = target.stat().st_size
                observed_sha256 = sha256_file(target)
                if (
                    observed_size != source_entry["size_bytes"]
                    or observed_sha256 != source_entry["file_sha256"]
                ):
                    _fail(
                        f"{table_name}: destination bytes differ from the sealed "
                        "source-part receipt."
                    )
                final_relative = (
                    generation_relative / relative_inside_generation
                ).as_posix()
                inventory[table_name].append(
                    {
                        **_plain(source_entry),
                        "member_ordinal": member["ordinal"],
                        "recording_id": member["recording_id"],
                        "path": final_relative,
                        "generation_path": generation_relative.as_posix(),
                        "source_shard_record_sha256": receipt["record_sha256"],
                    }
                )
                transfers.append(
                    {
                        "table_name": table_name,
                        "member_ordinal": member["ordinal"],
                        "recording_id": member["recording_id"],
                        "source_shard_record_sha256": receipt["record_sha256"],
                        "source_part_path": source_entry["path"],
                        "source_part_file_sha256": source_entry["file_sha256"],
                        "source_size_bytes": source_entry["size_bytes"],
                        "destination_path": final_relative,
                        "observed_size_bytes": observed_size,
                        "observed_file_sha256": observed_sha256,
                        "method_id": TRANSFER_RECEIPT_METHOD_ID,
                        "status": "complete",
                    }
                )
                _freeze_regular_file(target)
        transfer_completed_at = time.perf_counter()
        shard_roster_files, published_semantic_roster_sha = (
            _validate_published_shard_roster(
                stage,
                generation_relative_path=generation_relative.as_posix(),
                plan=plan,
                roster=shard_roster,
                table_specs=table_specs,
            )
        )
        inventory_sha = canonical_json_sha256(inventory)
        roster_sha = canonical_json_sha256(shard_roster)
        transfers.sort(
            key=lambda item: (
                table_names.index(str(item["table_name"])),
                int(item["member_ordinal"]),
            )
        )
        transfer_body = {
            "schema_id": TRANSFER_RECEIPT_SCHEMA_ID,
            "schema_version": TRANSFER_RECEIPT_SCHEMA_VERSION,
            "method_id": TRANSFER_RECEIPT_METHOD_ID,
            "status": TRANSFER_RECEIPT_STATUS,
            "export_run_id": run_id,
            "export_plan_sha256": plan["plan_sha256"],
            "generation_id": generation,
            "generation_path": generation_relative.as_posix(),
            "staging_attempt_id": staging_attempt_id,
            "shard_receipts_sha256": roster_sha,
            "part_inventory_sha256": inventory_sha,
            "transfer_verification_policy": TRANSFER_VERIFICATION_POLICY,
            "part_count": len(transfers),
            "transfers": transfers,
            "software_authority": plan["software_authority"],
            "verified_at_utc": validated_at,
            "safety": SAFETY,
        }
        transfer_receipt = _sealed(transfer_body, digest_field="record_sha256")
        transfer_stage_path = stage / "validation" / "transfer_receipt.json"
        _write_json(transfer_stage_path, transfer_receipt)
        transfer_relative = (
            generation_relative / "validation" / "transfer_receipt.json"
        ).as_posix()
        transfer_binding = {
            "path": transfer_relative,
            "size_bytes": transfer_stage_path.stat().st_size,
            "file_sha256": sha256_file(transfer_stage_path),
            "record_sha256": transfer_receipt["record_sha256"],
        }
        _freeze_regular_file(transfer_stage_path)
        _validate_transfer_receipt(
            transfer_receipt,
            plan=plan,
            generation_id=generation,
            generation_path=generation_relative.as_posix(),
            inventory=inventory,
            shard_receipts_sha256=roster_sha,
            receipt_path=transfer_stage_path,
            binding=transfer_binding,
        )
        validation = _compose_generation_validation(
            plan=plan,
            shard_receipts=[receipt for _path, receipt in shard_receipts],
            table_specs=table_specs,
        )
        semantic_roster_sha = canonical_json_sha256(
            [
                receipt["semantic_validation"]["record_sha256"]
                for _path, receipt in shard_receipts
            ]
        )
        if published_semantic_roster_sha != semantic_roster_sha:
            _fail("Published shard semantic-proof roster digest is stale.")
        validation_body = {
            "schema_id": VALIDATION_RECEIPT_SCHEMA_ID,
            "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
            "status": "complete_receipt_composed_v2",
            "export_run_id": run_id,
            "export_plan_sha256": plan["plan_sha256"],
            "generation_id": generation,
            "generation_path": generation_relative.as_posix(),
            "part_inventory_sha256": inventory_sha,
            "shard_receipts_sha256": roster_sha,
            "shard_semantic_proofs_sha256": semantic_roster_sha,
            "transfer_receipt_sha256": transfer_receipt["record_sha256"],
            "staging_attempt_id": staging_attempt_id,
            "generation_composition_policy": GENERATION_COMPOSITION_POLICY,
            "mutation_exclusion": MUTATION_EXCLUSION_POLICY,
            "validation_policy": VALIDATION_POLICY,
            "validation_result": validation,
            "software_authority": plan["software_authority"],
            "validated_at_utc": validated_at,
            "safety": SAFETY,
        }
        validation_receipt = _sealed(validation_body, digest_field="record_sha256")
        validation_stage_path = stage / "validation" / "receipt.json"
        _write_json(validation_stage_path, validation_receipt)
        _freeze_regular_file(validation_stage_path)
        validation_relative = (
            generation_relative / "validation" / "receipt.json"
        ).as_posix()
        publication = {
            "schema_id": PUBLICATION_SCHEMA_ID,
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "state": "complete",
            "generation_id": generation,
            "generation_path": generation_relative.as_posix(),
            "parts_by_table": inventory,
            "part_inventory_sha256": inventory_sha,
        }
        body = {
            "schema_id": EXPORT_SCHEMA_ID,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "method_id": EXPORT_METHOD_ID,
            "status": EXPORT_STATUS,
            "export_run_id": run_id,
            "export_plan": {
                "path": str(plan_file),
                "file_sha256": sha256_file(plan_file),
                "plan_sha256": plan["plan_sha256"],
            },
            "export_profile": plan["export_profile"],
            "membership": plan["membership"],
            "bundle_set": plan["bundle_set"],
            "member_count": plan["member_count"],
            "membership_state_counts": _plain(membership["state_counts"]),
            "bundle_state_counts": _plain(bundle_set["state_counts"]),
            "capability_matrix_sha256": bundle_set["capability_matrix_sha256"],
            "table_names": list(table_names),
            "table_specs": plan["table_specs"],
            "table_coverage": plan["table_coverage"],
            "arrow_schema_contracts": plan["arrow_schema_contracts"],
            "shard_receipts": shard_roster,
            "shard_receipts_sha256": roster_sha,
            "row_counts_by_table": validation["row_counts_by_table"],
            "parameters": plan["parameters"],
            "analysis_unit_policy": _plain(membership["analysis_unit_policy"]),
            "acquisition_batch_policy": _plain(membership["acquisition_batch_policy"]),
            "temporal_alignment_policy": _plain(
                membership["temporal_alignment_policy"]
            ),
            "publication": publication,
            "transfer_receipt": transfer_binding,
            "validation_receipt": {
                "path": validation_relative,
                "size_bytes": validation_stage_path.stat().st_size,
                "file_sha256": sha256_file(validation_stage_path),
                "record_sha256": validation_receipt["record_sha256"],
            },
            "software_authority": plan["software_authority"],
            "created_at_utc": validation_receipt["validated_at_utc"],
            "safety": SAFETY,
        }
        manifest = _sealed(body, digest_field="record_sha256")
        receipts_composed_at = time.perf_counter()
        receipt_inventory_files = shard_roster_files | {
            "validation/transfer_receipt.json"
        }

        def validate_staging() -> None:
            _validate_plan_file_binding(manifest["export_plan"], expected_plan=plan)
            if (
                _validate_file_binding(plan["membership"], field="membership")
                != plan["membership"]
                or _validate_file_binding(plan["bundle_set"], field="bundle_set")
                != plan["bundle_set"]
            ):
                _fail("Plan inputs changed before publication commit.")
            observed = _global_validate_generation(
                stage,
                generation_relative_path=generation_relative.as_posix(),
                plan=plan,
                inventory=inventory,
                table_specs=table_specs,
                hash_parts=False,
                validate_keys=False,
                additional_expected_files=receipt_inventory_files,
                require_frozen_files=True,
            )
            if observed["row_counts_by_table"] != validation["row_counts_by_table"]:
                _fail("Staged row-count inventory changed before commit.")
            _validate_transfer_receipt(
                transfer_receipt,
                plan=plan,
                generation_id=generation,
                generation_path=generation_relative.as_posix(),
                inventory=inventory,
                shard_receipts_sha256=roster_sha,
                receipt_path=transfer_stage_path,
                binding=manifest["transfer_receipt"],
            )
            _validate_validation_receipt(
                validation_receipt,
                manifest=manifest,
                receipt_path=validation_stage_path,
                transfer_receipt=transfer_receipt,
            )

        commit_validated_immutable_generation(
            publication_root,
            stage,
            final_generation,
            manifest_path,
            manifest,
            baseline_manifest_identity=manifest_identity(manifest_path),
            lock_directory=publication_root / "validated_behavior" / "v1" / ".locks",
            validate_staging=validate_staging,
        )
        committed_at = time.perf_counter()
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "process_telemetry": {
            "policy_id": "validated_behavior_finalize_process_telemetry_v1",
            "source_shard_receipt_validation_seconds": round(
                shard_receipts_validated_at - operation_started, 6
            ),
            "destination_copy_and_hash_seconds": round(
                transfer_completed_at - shard_receipts_validated_at, 6
            ),
            "receipt_composition_seconds": round(
                receipts_composed_at - transfer_completed_at, 6
            ),
            "receipt_only_precommit_and_atomic_commit_seconds": round(
                committed_at - receipts_composed_at, 6
            ),
            "total_seconds": round(committed_at - operation_started, 6),
            "copied_part_count": len(transfers),
            "copied_size_bytes": sum(
                int(entry["observed_size_bytes"]) for entry in transfers
            ),
        },
    }


def _validate_validation_receipt(
    value: object,
    *,
    manifest: Mapping[str, Any],
    receipt_path: Path,
    transfer_receipt: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    receipt = _validate_self_digest(
        _mapping(value, field="validation receipt"),
        digest_field="record_sha256",
        field="Validation receipt",
    )
    binding = _mapping(manifest.get("validation_receipt"), field="validation_receipt")
    fields_v1 = {
        "schema_id",
        "schema_version",
        "status",
        "export_run_id",
        "export_plan_sha256",
        "generation_id",
        "generation_path",
        "part_inventory_sha256",
        "shard_receipts_sha256",
        "validation_policy",
        "validation_result",
        "software_authority",
        "validated_at_utc",
        "safety",
        "record_sha256",
    }
    fields_v2 = fields_v1 | {
        "shard_semantic_proofs_sha256",
        "transfer_receipt_sha256",
        "staging_attempt_id",
        "generation_composition_policy",
        "mutation_exclusion",
    }
    schema_version = receipt.get("schema_version")
    if schema_version == LEGACY_VALIDATION_RECEIPT_SCHEMA_VERSION:
        required_fields = fields_v1
        expected_status = "complete"
        expected_policy = LEGACY_VALIDATION_POLICY
    elif schema_version == VALIDATION_RECEIPT_SCHEMA_VERSION:
        required_fields = fields_v2
        expected_status = "complete_receipt_composed_v2"
        expected_policy = VALIDATION_POLICY
    else:
        _fail("Validation-receipt schema version is unsupported.")
    expected_manifest_version = (
        LEGACY_EXPORT_SCHEMA_VERSION
        if schema_version == LEGACY_VALIDATION_RECEIPT_SCHEMA_VERSION
        else EXPORT_SCHEMA_VERSION
    )
    if manifest.get("schema_version") != expected_manifest_version:
        _fail("Validation receipt and export manifest versions are incompatible.")
    if set(receipt) != required_fields:
        _fail("Validation-receipt field set is inexact.")
    if set(binding) != {"path", "size_bytes", "file_sha256", "record_sha256"}:
        _fail("Validation-receipt file binding is inexact.")
    if (
        receipt.get("schema_id") != VALIDATION_RECEIPT_SCHEMA_ID
        or receipt.get("status") != expected_status
        or receipt.get("validation_policy") != expected_policy
        or receipt.get("safety") != SAFETY
    ):
        _fail("Validation-receipt identity, status, policy, or safety is invalid.")
    if (
        receipt.get("export_run_id") != manifest["export_run_id"]
        or receipt.get("export_plan_sha256") != manifest["export_plan"]["plan_sha256"]
        or receipt.get("generation_id") != manifest["publication"]["generation_id"]
        or receipt.get("generation_path") != manifest["publication"]["generation_path"]
        or receipt.get("part_inventory_sha256")
        != manifest["publication"]["part_inventory_sha256"]
        or receipt.get("shard_receipts_sha256") != manifest["shard_receipts_sha256"]
    ):
        _fail("Validation receipt binds another plan, shard roster, or generation.")
    result = _mapping(receipt.get("validation_result"), field="validation_result")
    result_fields_v1 = {
        "row_counts_by_table",
        "primary_key_counts_by_table",
        "foreign_key_validation",
        "inventory_file_count",
    }
    result_fields_v2 = result_fields_v1 | {
        "owner_validation",
        "shard_receipt_count",
        "semantic_table_proof_count",
        "foreign_key_proof_count",
        "composition_policy",
    }
    expected_result_fields = (
        result_fields_v1
        if schema_version == LEGACY_VALIDATION_RECEIPT_SCHEMA_VERSION
        else result_fields_v2
    )
    if set(result) != expected_result_fields:
        _fail("Validation result field set is inexact.")
    row_counts = _mapping(
        result.get("row_counts_by_table"), field="validation row counts"
    )
    key_counts = _mapping(
        result.get("primary_key_counts_by_table"),
        field="validation primary-key counts",
    )
    inventory = _mapping(
        manifest["publication"].get("parts_by_table"), field="parts_by_table"
    )
    expected_inventory_files = sum(len(entries) for entries in inventory.values())
    common_result_invalid = (
        row_counts != manifest["row_counts_by_table"]
        or key_counts != row_counts
        or result.get("inventory_file_count") != expected_inventory_files
        or receipt.get("software_authority") != manifest["software_authority"]
        or receipt.get("validated_at_utc") != manifest["created_at_utc"]
    )
    if schema_version == LEGACY_VALIDATION_RECEIPT_SCHEMA_VERSION:
        result_invalid = (
            common_result_invalid or result.get("foreign_key_validation") != "complete"
        )
    else:
        transfer_binding = _mapping(
            manifest.get("transfer_receipt"), field="transfer_receipt"
        )
        if transfer_receipt is None:
            _fail("Receipt-composed validation requires its transfer receipt.")
        member_count = int(manifest["member_count"])
        table_count = len(inventory)
        foreign_key_count = sum(
            len(_mapping(spec, field="table spec").get("foreign_keys", []))
            for spec in _mapping(manifest["table_specs"], field="table_specs").values()
        )
        _digest(
            receipt.get("shard_semantic_proofs_sha256"),
            field="shard_semantic_proofs_sha256",
        )
        result_invalid = (
            common_result_invalid
            or result.get("owner_validation") != "complete_receipt_composed"
            or result.get("foreign_key_validation") != "complete_receipt_composed"
            or result.get("shard_receipt_count") != member_count
            or result.get("semantic_table_proof_count") != member_count * table_count
            or result.get("foreign_key_proof_count") != member_count * foreign_key_count
            or result.get("composition_policy") != GENERATION_COMPOSITION_POLICY
            or receipt.get("transfer_receipt_sha256")
            != transfer_binding.get("record_sha256")
            or receipt.get("staging_attempt_id") is None
            or receipt.get("staging_attempt_id")
            != transfer_receipt.get("staging_attempt_id")
            or receipt.get("transfer_receipt_sha256")
            != transfer_receipt.get("record_sha256")
            or receipt.get("generation_composition_policy")
            != GENERATION_COMPOSITION_POLICY
            or receipt.get("mutation_exclusion") != MUTATION_EXCLUSION_POLICY
        )
        safe_component(receipt.get("staging_attempt_id"), label="staging attempt ID")
    if result_invalid:
        _fail("Validation result, software, or timestamp differs from the manifest.")
    if (
        binding.get("record_sha256") != receipt["record_sha256"]
        or type(binding.get("size_bytes")) is not int
        or binding.get("size_bytes") != receipt_path.stat().st_size
        or binding.get("file_sha256") != sha256_file(receipt_path)
    ):
        _fail("Validation-receipt file binding is stale.")
    if schema_version == VALIDATION_RECEIPT_SCHEMA_VERSION:
        _require_frozen_regular_file(
            receipt_path, field="generation validation receipt"
        )
    _validate_software(receipt.get("software_authority"))
    _timestamp(receipt.get("validated_at_utc"), field="validated_at_utc")
    return receipt


def read_validated_behavior_export_manifest(
    root: str | Path,
    export_run_id: str,
    *,
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
    validate_parts: str = "receipt",
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Read one exact selected manifest without globbing or source-Zarr reads.

    ``validate_parts='receipt'`` checks the sealed validation receipt, exact
    paths, sizes, and Arrow contracts.  ``'full'`` additionally rehashes every
    Parquet part.  Both modes revalidate the small membership and bundle-set
    manifests; neither mode opens a source Zarr.
    """

    if validate_parts not in {"receipt", "full"}:
        _fail("validate_parts must be 'receipt' or 'full'.")
    publication_root = _absolute_path(root, field="publication_root")
    manifest_path = validated_behavior_manifest_path(publication_root, export_run_id)
    selected_path, raw = _strict_object(manifest_path, field="export manifest")
    manifest = _validate_self_digest(
        raw, digest_field="record_sha256", field="Export manifest"
    )
    schema_version = manifest.get("schema_version")
    if schema_version == LEGACY_EXPORT_SCHEMA_VERSION:
        required = _MANIFEST_FIELDS_V1
        expected_method = LEGACY_EXPORT_METHOD_ID
    elif schema_version == EXPORT_SCHEMA_VERSION:
        required = _MANIFEST_FIELDS_V2
        expected_method = EXPORT_METHOD_ID
    else:
        _fail("Export manifest schema version is unsupported.")
    if set(manifest) != required:
        _fail("Export manifest field set is inexact.")
    if (
        manifest.get("schema_id") != EXPORT_SCHEMA_ID
        or manifest.get("method_id") != expected_method
        or manifest.get("status") != EXPORT_STATUS
        or manifest.get("export_run_id") != export_run_id
        or manifest.get("safety") != SAFETY
    ):
        _fail("Export manifest identity, method, status, run, or safety is invalid.")
    membership_binding = _validate_file_binding(
        manifest.get("membership"), field="membership"
    )
    membership = read_validated_behavior_cohort_membership(membership_binding["path"])
    bundle_binding = _validate_file_binding(
        manifest.get("bundle_set"), field="bundle_set"
    )
    bundle_set = read_validated_behavior_bundle_set(
        bundle_binding["path"], membership=membership
    )
    plan_path, plan_binding = _validate_plan_file_binding(manifest.get("export_plan"))
    plan, plan_membership, plan_bundle_set = read_validated_behavior_export_plan(
        plan_path, table_specs=table_specs
    )
    expected_plan_version = (
        LEGACY_EXPORT_PLAN_SCHEMA_VERSION
        if schema_version == LEGACY_EXPORT_SCHEMA_VERSION
        else EXPORT_PLAN_SCHEMA_VERSION
    )
    if (
        plan["plan_sha256"] != plan_binding["plan_sha256"]
        or plan.get("schema_version") != expected_plan_version
        or plan_membership["record_sha256"] != membership["record_sha256"]
        or plan_bundle_set["record_sha256"] != bundle_set["record_sha256"]
        or manifest.get("membership") != plan["membership"]
        or manifest.get("bundle_set") != plan["bundle_set"]
        or manifest.get("export_profile") != plan["export_profile"]
    ):
        _fail("Export manifest binds another plan or input generation.")
    if (
        manifest.get("member_count") != membership["member_count"]
        or manifest.get("membership_state_counts") != _plain(membership["state_counts"])
        or manifest.get("bundle_state_counts") != _plain(bundle_set["state_counts"])
        or manifest.get("capability_matrix_sha256")
        != bundle_set["capability_matrix_sha256"]
        or manifest.get("analysis_unit_policy")
        != _plain(membership["analysis_unit_policy"])
        or manifest.get("acquisition_batch_policy")
        != _plain(membership["acquisition_batch_policy"])
        or manifest.get("temporal_alignment_policy")
        != _plain(membership["temporal_alignment_policy"])
        or manifest.get("parameters") != plan["parameters"]
        or manifest.get("table_coverage") != plan["table_coverage"]
        or manifest.get("software_authority") != plan["software_authority"]
    ):
        _fail(
            "Export manifest cohort counts, policies, parameters, or software drifted."
        )
    table_names = validate_table_specs(table_specs)
    if manifest.get("table_names") != list(table_names):
        _fail("Export manifest table roster differs from installed specs.")
    _validate_installed_specs(manifest.get("table_specs"), specs=table_specs)
    validate_contract_envelope(
        manifest.get("arrow_schema_contracts"),
        table_names,
        known_table_names=table_names,
        contracts={name: table_specs[name].contract for name in table_names},
        schema_id=ARROW_ENVELOPE_SCHEMA_ID,
        schema_version=ARROW_ENVELOPE_SCHEMA_VERSION,
    )
    publication = _mapping(manifest.get("publication"), field="publication")
    if set(publication) != {
        "schema_id",
        "schema_version",
        "state",
        "generation_id",
        "generation_path",
        "parts_by_table",
        "part_inventory_sha256",
    }:
        _fail("Validated-behavior publication field set is inexact.")
    generation_id = safe_component(
        publication.get("generation_id"), label="generation ID"
    )
    expected_generation_relative = _generation_relative_path(
        export_run_id, generation_id
    ).as_posix()
    if (
        publication.get("schema_id") != PUBLICATION_SCHEMA_ID
        or publication.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        or publication.get("state") != "complete"
        or publication.get("generation_path") != expected_generation_relative
        or publication.get("part_inventory_sha256")
        != canonical_json_sha256(_plain(publication.get("parts_by_table")))
    ):
        _fail("Validated-behavior publication envelope is invalid.")
    generation_root = _safe_selected_path(
        publication_root,
        publication.get("generation_path"),
        field="publication.generation_path",
    )
    if not generation_root.is_dir() or generation_root.is_symlink():
        _fail("Selected immutable generation is absent or aliased.")
    inventory = _mapping(publication.get("parts_by_table"), field="parts_by_table")
    transfer_expected_files: set[str] = set()
    validated_transfer_receipt: Mapping[str, Any] | None = None
    if schema_version == EXPORT_SCHEMA_VERSION:
        transfer_binding = _mapping(
            manifest.get("transfer_receipt"), field="transfer_receipt"
        )
        transfer_path = _safe_selected_path(
            publication_root,
            transfer_binding.get("path"),
            field="transfer_receipt.path",
        )
        _transfer_path, transfer_raw = _strict_object(
            transfer_path, field="transfer receipt"
        )
        validated_transfer_receipt = _validate_transfer_receipt(
            transfer_raw,
            plan=plan,
            generation_id=generation_id,
            generation_path=expected_generation_relative,
            inventory=inventory,
            shard_receipts_sha256=str(manifest["shard_receipts_sha256"]),
            receipt_path=transfer_path,
            binding=transfer_binding,
        )
        _require_frozen_regular_file(transfer_path, field="generation transfer receipt")
        transfer_expected_files.add("validation/transfer_receipt.json")
    validation_binding = _mapping(
        manifest.get("validation_receipt"), field="validation_receipt"
    )
    validation_path = _safe_selected_path(
        publication_root,
        validation_binding.get("path"),
        field="validation_receipt.path",
    )
    _receipt_path, validation_raw = _strict_object(
        validation_path, field="validation receipt"
    )
    validated_receipt = _validate_validation_receipt(
        validation_raw,
        manifest=manifest,
        receipt_path=validation_path,
        transfer_receipt=(validated_transfer_receipt),
    )
    shard_roster = _list(manifest.get("shard_receipts"), field="shard_receipts")
    if canonical_json_sha256(shard_roster) != manifest.get("shard_receipts_sha256"):
        _fail("Export manifest shard-receipt roster digest is stale.")
    shard_roster_files, semantic_roster_sha = _validate_published_shard_roster(
        generation_root,
        generation_relative_path=expected_generation_relative,
        plan=plan,
        roster=shard_roster,
        table_specs=(table_specs if schema_version == EXPORT_SCHEMA_VERSION else None),
    )
    if schema_version == EXPORT_SCHEMA_VERSION and (
        semantic_roster_sha != validated_receipt.get("shard_semantic_proofs_sha256")
    ):
        _fail("Validation receipt binds another shard semantic-proof roster.")
    observed = _global_validate_generation(
        generation_root,
        generation_relative_path=expected_generation_relative,
        plan=plan,
        inventory=inventory,
        table_specs=table_specs,
        hash_parts=validate_parts == "full",
        validate_keys=validate_parts == "full",
        additional_expected_files=shard_roster_files | transfer_expected_files,
        require_frozen_files=(schema_version == EXPORT_SCHEMA_VERSION),
    )
    if observed["row_counts_by_table"] != manifest["row_counts_by_table"]:
        _fail("Selected generation row counts differ from its manifest.")
    _validate_software(manifest.get("software_authority"))
    _timestamp(manifest.get("created_at_utc"), field="created_at_utc")
    if selected_path != manifest_path:
        _fail("Selected export manifest path changed during validation.")
    return manifest, membership, bundle_set


def selected_table_parts(
    root: str | Path,
    manifest: Mapping[str, Any],
    table_name: str,
) -> tuple[Path, ...]:
    """Resolve only the exact parts declared by a validated manifest."""

    publication_root = _absolute_path(root, field="publication_root")
    table = safe_component(table_name, label="table name")
    inventory = _mapping(
        _mapping(manifest.get("publication"), field="publication").get(
            "parts_by_table"
        ),
        field="parts_by_table",
    )
    if table not in inventory:
        raise KeyError(f"Unknown validated-behavior table: {table}")
    return tuple(
        _safe_selected_path(publication_root, entry["path"], field=f"{table}.part.path")
        for entry in inventory[table]
    )


__all__ = [
    "DEFAULT_EXPORT_PARAMETERS",
    "EXPORT_PLAN_SCHEMA_ID",
    "EXPORT_SCHEMA_ID",
    "SHARD_SCHEMA_ID",
    "ValidatedBehaviorExportError",
    "build_validated_behavior_export_plan",
    "planned_shard_receipt_path",
    "publish_validated_behavior_cohort",
    "read_validated_behavior_export_manifest",
    "read_validated_behavior_export_plan",
    "read_validated_behavior_shard_receipt",
    "selected_table_parts",
    "shard_relative_path",
    "validated_behavior_manifest_path",
    "validate_validated_behavior_export_plan",
    "write_validated_behavior_export_plan",
    "write_validated_behavior_recording_shard",
]
