"""Receipt-backed sharding and immutable publication for behavior cohorts.

This module is intentionally protocol-neutral.  Its built-in row producers
materialize only the closed membership, bundle, and capability relations.  A
scientific family extends the same engine by supplying exact table specs and
recording-scoped row extractors; the engine never discovers Zarr runs or
reconstructs unavailable evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
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
EXPORT_PLAN_SCHEMA_VERSION = 1
EXPORT_PLAN_METHOD_ID = "closed_membership_recording_shard_plan_v1"
EXPORT_PLAN_STATUS = "planned_selector_ineligible"

SHARD_SCHEMA_ID = "palette.analytics.validated_behavior_export_shard"
SHARD_SCHEMA_VERSION = 1
SHARD_METHOD_ID = "recording_owned_exact_parquet_parts_v1"
SHARD_STATUS = "complete_validated"
SHARD_VALIDATION_POLICY = "exact_inputs_parts_arrow_and_primary_keys_v1"

EXPORT_SCHEMA_ID = "palette.analytics.validated_behavior_cohort_export"
EXPORT_SCHEMA_VERSION = 1
EXPORT_METHOD_ID = "receipt_barrier_manifest_selected_parquet_v1"
EXPORT_STATUS = "complete_selector_ineligible"
PUBLICATION_SCHEMA_ID = "palette.analytics.validated_behavior.publication"
PUBLICATION_SCHEMA_VERSION = 1

VALIDATION_RECEIPT_SCHEMA_ID = "palette.analytics.validated_behavior_cohort_validation"
VALIDATION_RECEIPT_SCHEMA_VERSION = 1
VALIDATION_POLICY = "manifest_selected_schema_key_foreign_key_inventory_v1"

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
_PLAN_FIELDS = {
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
_PUBLICATION_PART_FIELDS = _SHARD_PART_FIELDS | {
    "member_ordinal",
    "recording_id",
    "generation_path",
    "source_shard_record_sha256",
}

RowExtractor = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    tuple[Sequence[Mapping[str, Any]], str | None],
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
    profile_body = {
        "profile_id": safe_component(export_profile_id, label="export profile ID"),
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
    if set(plan) != _PLAN_FIELDS:
        _fail("Export-plan field set is inexact.")
    if (
        plan.get("schema_id") != EXPORT_PLAN_SCHEMA_ID
        or plan.get("schema_version") != EXPORT_PLAN_SCHEMA_VERSION
        or plan.get("method_id") != EXPORT_PLAN_METHOD_ID
        or plan.get("status") != EXPORT_PLAN_STATUS
        or plan.get("safety") != SAFETY
    ):
        _fail("Export-plan identity, method, status, or safety is invalid.")
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
    safe_component(profile.get("profile_id"), label="export profile ID")
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


def _primary_key_bounds(
    rows: Sequence[Mapping[str, Any]], spec: ValidatedBehaviorTableSpec
) -> dict[str, object] | None:
    if not rows:
        return None
    keys = [tuple(row[name] for name in spec.contract.primary_key) for row in rows]
    return {"minimum": list(min(keys)), "maximum": list(max(keys))}


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
    if hash_bytes and parquet.metadata.num_rows:
        identities = parquet.read(columns=["export_run_id", "recording_id"])
        observed_runs = set(identities.column("export_run_id").to_pylist())
        observed_recordings = set(identities.column("recording_id").to_pylist())
        if observed_runs != {plan["export_run_id"]} or observed_recordings != {
            member["recording_id"]
        }:
            _fail(f"{spec.table_name}: in-row shard identity differs from its owner.")


def _observed_primary_key_summary(
    part: Path, spec: ValidatedBehaviorTableSpec
) -> tuple[int, dict[str, object] | None]:
    import pyarrow.parquet as pq

    seen: set[tuple[Any, ...]] = set()
    minimum: tuple[Any, ...] | None = None
    maximum: tuple[Any, ...] | None = None
    parquet = pq.ParquetFile(part)
    for batch in parquet.iter_batches(columns=list(spec.contract.primary_key)):
        columns = [column.to_pylist() for column in batch.columns]
        for key in zip(*columns, strict=True):
            if key in seen:
                _fail(f"{spec.table_name}: shard contains a duplicate primary key.")
            seen.add(key)
            minimum = key if minimum is None or key < minimum else minimum
            maximum = key if maximum is None or key > maximum else maximum
    bounds = (
        None
        if minimum is None
        else {"minimum": list(minimum), "maximum": list(maximum)}
    )
    return len(seen), bounds


def _validate_shard_receipt(
    value: object,
    *,
    shard_root: Path,
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
    required = {
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
    if set(receipt) != required:
        _fail("Shard receipt field set is inexact.")
    if (
        receipt.get("schema_id") != SHARD_SCHEMA_ID
        or receipt.get("schema_version") != SHARD_SCHEMA_VERSION
        or receipt.get("method_id") != SHARD_METHOD_ID
        or receipt.get("status") != SHARD_STATUS
        or receipt.get("validation_policy") != SHARD_VALIDATION_POLICY
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
        _validate_part(
            shard_root / expected_path,
            raw,
            spec=spec,
            plan=plan,
            member=member,
            hash_bytes=hash_parts,
        )
        key_count, key_bounds = _observed_primary_key_summary(
            shard_root / expected_path, spec
        )
        if key_count != row_count or raw.get("primary_key_bounds") != key_bounds:
            _fail(f"{table_name}: primary-key count or bounds are stale.")
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
            hash_parts=True,
        )
        return {**_plain(receipt), "receipt_path": str(final_receipt), "reused": True}
    stage = final_root.parent / f".{final_root.name}.{uuid.uuid4().hex}.tmp"
    stage.mkdir(parents=True, exist_ok=False)
    extractors = dict(row_extractors or {})
    table_names = validate_table_specs(table_specs)
    try:
        parts: dict[str, dict[str, Any]] = {}
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
                raw_rows, zero_reason = extractors[table_name](
                    plan, membership_member, bundle_member
                )
                rows = [dict(row) for row in raw_rows]
            else:
                _fail(f"No recording-scoped extractor is installed for {table_name!r}.")
            if not rows and (not spec.zero_rows_allowed or zero_reason is None):
                _fail(f"{table_name}: extractor returned an uncontracted empty result.")
            if rows and zero_reason is not None:
                _fail(f"{table_name}: non-empty rows cannot carry a zero-row reason.")
            _validate_extracted_rows(rows, spec)
            table_dir = stage / "tables" / table_name
            table_dir.mkdir(parents=True, exist_ok=False)
            part = table_dir / "part-00000.parquet"
            schema = exact_schema(
                spec.contract,
                metadata=_part_footer(plan=plan, member=member, table_name=table_name),
            )
            table = pa.Table.from_pylist(rows, schema=schema)
            temporary_part = table_dir / ".part-00000.parquet.tmp"
            pq.write_table(
                table,
                temporary_part,
                compression=str(plan["parameters"]["parquet_compression"]),
                row_group_size=int(plan["parameters"]["effective_row_group_rows"]),
            )
            os.replace(temporary_part, part)
            parts[table_name] = _part_receipt(
                part=part,
                relative_path=f"tables/{table_name}/part-00000.parquet",
                row_count=table.num_rows,
                spec=spec,
                key_bounds=_primary_key_bounds(rows, spec),
            )
            zero_reasons[table_name] = zero_reason
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
            "software_authority": plan["software_authority"],
            "created_at_utc": _timestamp(
                created_at_utc or _default_created_at(), field="created_at_utc"
            ),
            "safety": SAFETY,
        }
        receipt = _sealed(body, digest_field="record_sha256")
        _write_json(stage / "receipt.json", receipt)
        _validate_shard_receipt(
            receipt,
            shard_root=stage,
            plan=plan,
            member=member,
            table_specs=table_specs,
            hash_parts=True,
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


def _validate_published_shard_roster(
    generation_root: Path,
    *,
    generation_relative_path: str,
    plan: Mapping[str, Any],
    roster: object,
) -> set[str]:
    entries = _list(roster, field="shard_receipts")
    if len(entries) != plan["member_count"]:
        _fail("Published shard-receipt roster does not close the member axis.")
    expected_files: set[str] = set()
    normalized_for_digest: list[dict[str, Any]] = []
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
        normalized_for_digest.append(_plain(entry))
    if normalized_for_digest != _plain(entries):
        _fail("Published shard-receipt roster is not deterministically ordered.")
    return expected_files


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
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table_names = validate_table_specs(table_specs)
    if set(inventory) != set(table_names):
        _fail("Publication part inventory is incomplete.")
    keys_by_table: dict[str, set[tuple[Any, ...]]] = {}
    row_counts: dict[str, int] = {}
    expected_files: set[str] = set()
    member_by_ordinal = {int(item["ordinal"]): item for item in plan["members"]}
    for table_name in table_names:
        spec = table_specs[table_name]
        seen: set[tuple[Any, ...]] = set()
        count = 0
        entries = list(inventory[table_name])
        if len(entries) != plan["member_count"]:
            _fail(f"{table_name}: publication lacks one part per parent member.")
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
            expected_files.add(relative_inside_generation.as_posix())
            _validate_part(
                part,
                entry,
                spec=spec,
                plan=plan,
                member=member,
                hash_bytes=hash_parts,
            )
            if validate_keys:
                parquet = pq.ParquetFile(part)
                part_seen: set[tuple[Any, ...]] = set()
                part_minimum: tuple[Any, ...] | None = None
                part_maximum: tuple[Any, ...] | None = None
                for batch in parquet.iter_batches(
                    columns=list(spec.contract.primary_key)
                ):
                    columns = [column.to_pylist() for column in batch.columns]
                    for key in zip(*columns, strict=True):
                        if key in part_seen:
                            _fail(f"{table_name}: duplicate primary key within part.")
                        if key in seen:
                            _fail(f"{table_name}: duplicate primary key across parts.")
                        part_seen.add(key)
                        seen.add(key)
                        part_minimum = (
                            key
                            if part_minimum is None or key < part_minimum
                            else part_minimum
                        )
                        part_maximum = (
                            key
                            if part_maximum is None or key > part_maximum
                            else part_maximum
                        )
                part_bounds = (
                    None
                    if part_minimum is None
                    else {
                        "minimum": list(part_minimum),
                        "maximum": list(part_maximum),
                    }
                )
                if (
                    len(part_seen) != entry["row_count"]
                    or entry.get("primary_key_bounds") != part_bounds
                ):
                    _fail(f"{table_name}: part primary-key bounds are stale.")
            count += int(entry["row_count"])
        keys_by_table[table_name] = seen
        row_counts[table_name] = count
    if validate_keys:
        for table_name in table_names:
            spec = table_specs[table_name]
            for local_fields, target, target_fields in spec.foreign_keys:
                local_indices = [
                    spec.contract.primary_key.index(name) for name in local_fields
                ]
                target_spec = table_specs[target]
                target_indices = [
                    target_spec.contract.primary_key.index(name)
                    for name in target_fields
                ]
                local_values = {
                    tuple(key[index] for index in local_indices)
                    for key in keys_by_table[table_name]
                }
                target_values = {
                    tuple(key[index] for index in target_indices)
                    for key in keys_by_table[target]
                }
                if not local_values.issubset(target_values):
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
    return {
        "row_counts_by_table": row_counts,
        "primary_key_counts_by_table": {
            name: len(keys_by_table[name]) if validate_keys else None
            for name in table_names
        },
        "foreign_key_validation": "complete" if validate_keys else "receipt_bound",
        "inventory_file_count": part_file_count,
    }


def publish_validated_behavior_cohort(
    *,
    plan_path: str | Path,
    table_specs: Mapping[str, ValidatedBehaviorTableSpec] = CORE_TABLE_SPECS,
    generation_id: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Fan in every exact shard and commit one immutable manifest last."""

    plan_file = Path(plan_path).expanduser().resolve()
    plan, membership, bundle_set = read_validated_behavior_export_plan(
        plan_file, table_specs=table_specs
    )
    table_names = validate_table_specs(table_specs)
    shard_receipts: list[tuple[Path, Mapping[str, Any]]] = []
    for member in plan["members"]:
        path = planned_shard_receipt_path(plan, member)
        receipt = read_validated_behavior_shard_receipt(
            path,
            plan=plan,
            member=member,
            table_specs=table_specs,
            hash_parts=True,
        )
        shard_receipts.append((path, receipt))
    publication_root = _absolute_path(
        plan.get("publication_root"), field="publication_root"
    )
    run_id = str(plan["export_run_id"])
    generation = safe_component(
        generation_id or uuid.uuid4().hex, label="generation ID"
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
        shard_roster_files = _validate_published_shard_roster(
            stage,
            generation_relative_path=generation_relative.as_posix(),
            plan=plan,
            roster=shard_roster,
        )
        validation = _global_validate_generation(
            stage,
            generation_relative_path=generation_relative.as_posix(),
            plan=plan,
            inventory=inventory,
            table_specs=table_specs,
            hash_parts=True,
            additional_expected_files=shard_roster_files,
        )
        inventory_sha = canonical_json_sha256(inventory)
        roster_sha = canonical_json_sha256(shard_roster)
        validation_body = {
            "schema_id": VALIDATION_RECEIPT_SCHEMA_ID,
            "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "export_run_id": run_id,
            "export_plan_sha256": plan["plan_sha256"],
            "generation_id": generation,
            "generation_path": generation_relative.as_posix(),
            "part_inventory_sha256": inventory_sha,
            "shard_receipts_sha256": roster_sha,
            "validation_policy": VALIDATION_POLICY,
            "validation_result": validation,
            "software_authority": plan["software_authority"],
            "validated_at_utc": _timestamp(
                created_at_utc or _default_created_at(), field="validated_at_utc"
            ),
            "safety": SAFETY,
        }
        validation_receipt = _sealed(validation_body, digest_field="record_sha256")
        validation_stage_path = stage / "validation" / "receipt.json"
        _write_json(validation_stage_path, validation_receipt)
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
                hash_parts=True,
                additional_expected_files=shard_roster_files,
            )
            if observed != validation:
                _fail("Staged global validation result changed before commit.")
            _validate_validation_receipt(
                validation_receipt,
                manifest=manifest,
                receipt_path=validation_stage_path,
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
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {**manifest, "manifest_path": str(manifest_path)}


def _validate_validation_receipt(
    value: object,
    *,
    manifest: Mapping[str, Any],
    receipt_path: Path,
) -> Mapping[str, Any]:
    receipt = _validate_self_digest(
        _mapping(value, field="validation receipt"),
        digest_field="record_sha256",
        field="Validation receipt",
    )
    binding = _mapping(manifest.get("validation_receipt"), field="validation_receipt")
    if set(receipt) != {
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
    }:
        _fail("Validation-receipt field set is inexact.")
    if set(binding) != {"path", "size_bytes", "file_sha256", "record_sha256"}:
        _fail("Validation-receipt file binding is inexact.")
    if (
        receipt.get("schema_id") != VALIDATION_RECEIPT_SCHEMA_ID
        or receipt.get("schema_version") != VALIDATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("validation_policy") != VALIDATION_POLICY
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
    if set(result) != {
        "row_counts_by_table",
        "primary_key_counts_by_table",
        "foreign_key_validation",
        "inventory_file_count",
    }:
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
    if (
        row_counts != manifest["row_counts_by_table"]
        or key_counts != row_counts
        or result.get("foreign_key_validation") != "complete"
        or result.get("inventory_file_count") != expected_inventory_files
        or receipt.get("software_authority") != manifest["software_authority"]
        or receipt.get("validated_at_utc") != manifest["created_at_utc"]
    ):
        _fail("Validation result, software, or timestamp differs from the manifest.")
    if (
        binding.get("record_sha256") != receipt["record_sha256"]
        or type(binding.get("size_bytes")) is not int
        or binding.get("size_bytes") != receipt_path.stat().st_size
        or binding.get("file_sha256") != sha256_file(receipt_path)
    ):
        _fail("Validation-receipt file binding is stale.")
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
    required = {
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
    if set(manifest) != required:
        _fail("Export manifest field set is inexact.")
    if (
        manifest.get("schema_id") != EXPORT_SCHEMA_ID
        or manifest.get("schema_version") != EXPORT_SCHEMA_VERSION
        or manifest.get("method_id") != EXPORT_METHOD_ID
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
    if (
        plan["plan_sha256"] != plan_binding["plan_sha256"]
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
    _validate_validation_receipt(
        validation_raw, manifest=manifest, receipt_path=validation_path
    )
    shard_roster = _list(manifest.get("shard_receipts"), field="shard_receipts")
    if canonical_json_sha256(shard_roster) != manifest.get("shard_receipts_sha256"):
        _fail("Export manifest shard-receipt roster digest is stale.")
    shard_roster_files = _validate_published_shard_roster(
        generation_root,
        generation_relative_path=expected_generation_relative,
        plan=plan,
        roster=shard_roster,
    )
    observed = _global_validate_generation(
        generation_root,
        generation_relative_path=expected_generation_relative,
        plan=plan,
        inventory=_mapping(publication.get("parts_by_table"), field="parts_by_table"),
        table_specs=table_specs,
        hash_parts=validate_parts == "full",
        validate_keys=validate_parts == "full",
        additional_expected_files=shard_roster_files,
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
