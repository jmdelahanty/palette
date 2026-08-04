"""Materialize and atomically publish one exact stimulus-epoch v2 candidate.

The established v1 writer remains a compatibility producer.  This boundary
reads one explicit complete v1 run, validates its complete logical table,
rematerializes all twelve arrays on node-local scratch through the shared byte
planner and Zarr v3 factory, and publishes a selector-ineligible immutable v2
candidate without changing parent pointers or production policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.exact_tabular_storage import (
    build_exact_tabular_storage_receipt,
    persist_exact_tabular_storage_receipt,
    rematerialize_exact_tabular_candidate,
    validate_exact_tabular_storage_receipt,
)
from fisheye.analysis.stimulus_epoch_schema import (
    LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
    LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
    STIMULUS_EPOCH_LAYOUT,
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
    STIMULUS_SOURCE_FINGERPRINT_ALGORITHM,
    build_stimulus_epoch_candidate_lineage_payload,
    build_stimulus_epoch_array_declarations,
    stimulus_epoch_logical_content_sha256,
    stimulus_group_logical_fingerprint,
    validate_legacy_stimulus_epoch_source,
    validate_stimulus_epoch_array_manifest,
    validate_stimulus_epoch_candidate_lineage,
    validate_stimulus_epoch_run_manifest,
    write_stimulus_epoch_array_manifest,
    write_stimulus_epoch_run_manifest,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_lineage_fingerprint import (
    canonical_lineage_json,
    compute_run_lineage_hash,
    write_run_lineage_attrs,
)
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.benchmark_runtime import storage_stats
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group
from .runtime_telemetry import PhaseTelemetry

PARENT_PATH = "analysis/stimulus_epoch_runs"
SUPPORTED_PROFILE_ID = "published_http_v1"
MATERIALIZATION_SCHEMA_ID = "palette.stimulus_epoch_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.stimulus_epoch_candidate_publish.v1"
STIMULUS_EPOCH_EXECUTION_PHASE_ORDER = (
    "plan",
    "source_staging",
    "logical_rematerialization",
    "local_validation",
    "local_consolidation",
    "local_direct_consolidated_comparison",
    "atomic_publication",
    "published_validation",
    "published_direct_consolidated_comparison",
    "decoded_equality",
    "physical_inventory",
    "publication_acceptance_validation",
)
EXECUTION_BINDING_ATTR = "analysis_candidate_execution_binding"
EXECUTION_FAILURE_TOMBSTONE_ATTR = "analysis_candidate_execution_tombstone"
PublicationAcceptanceValidator = Callable[
    [zarr.Group, zarr.Group, zarr.Group], Mapping[str, Any]
]


def _safe_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name or name in {".", "..", "latest", "latest_complete"}:
        raise ValueError(f"{label} must be one explicit immutable run name.")
    if "/" in name or "\\" in name or any(character.isspace() for character in name):
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return name


def _require_contained(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the authoritative archive.") from exc


def _ordered_runtime_telemetry(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {
        name: index for index, name in enumerate(STIMULUS_EPOCH_EXECUTION_PHASE_ORDER)
    }
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("Stimulus-epoch telemetry contains an unknown phase.")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    return result


def _copy_group_tree(source: Path, target: Path, *, backend: str) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing staged source group: {target}.")
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Authoritative stimulus-epoch source must be symlink-free.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if backend == "python":
        shutil.copytree(source, target, symlinks=False)
    elif backend == "rsync":
        target.mkdir()
        subprocess.run(
            ["rsync", "-aL", "--", f"{source}/", f"{target}/"],
            check=True,
        )
    else:
        raise ValueError(f"Unsupported source-staging backend: {backend!r}.")
    if target.is_symlink() or any(path.is_symlink() for path in target.rglob("*")):
        raise ValueError("Staged stimulus-epoch source must be symlink-free.")


@dataclass(frozen=True)
class StimulusEpochCandidatePlan:
    source_zarr: Path
    source_run_name: str
    run_name: str
    scratch_root: Path
    local_zarr: Path
    profile_id: str
    source_stimulus_run: str
    source_stimulus_path: str
    source_stimulus_fingerprint: str
    source_epoch_lineage_hash: str
    source_epoch_lineage_payload_sha256: str
    source_epoch_logical_content_sha256: str
    latest_before: Any
    latest_complete_before: Any

    @property
    def source_run_path(self) -> str:
        return f"{PARENT_PATH}/{self.source_run_name}"

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"

    @property
    def source_run_physical_path(self) -> Path:
        return self.source_zarr.joinpath(*self.source_run_path.split("/"))

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "source_run_name": self.source_run_name,
            "source_run_path": self.source_run_path,
            "run_name": self.run_name,
            "run_path": self.run_path,
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "profile_id": self.profile_id,
            "source_stimulus_run": self.source_stimulus_run,
            "source_stimulus_path": self.source_stimulus_path,
            "source_stimulus_fingerprint_algorithm": (
                STIMULUS_SOURCE_FINGERPRINT_ALGORITHM
            ),
            "source_stimulus_fingerprint": self.source_stimulus_fingerprint,
            "source_epoch_lineage_hash": self.source_epoch_lineage_hash,
            "source_epoch_lineage_payload_sha256": (
                self.source_epoch_lineage_payload_sha256
            ),
            "source_epoch_logical_content_sha256": (
                self.source_epoch_logical_content_sha256
            ),
            "latest_before": self.latest_before,
            "latest_complete_before": self.latest_complete_before,
            "source_schema": {
                "schema_id": LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
                "schema_version": LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
                "role": "explicit_legacy_input_only",
            },
            "candidate_schema": {
                "schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
                "schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
                "layout": STIMULUS_EPOCH_LAYOUT,
            },
            "publication_policy": (
                "atomic_named_candidate_selector_ineligible_no_pointer_update"
            ),
        }


def _validated_lineage_identity(group: Any, *, label: str) -> tuple[str, str]:
    lineage_json = group.attrs.get("lineage_payload_json")
    if type(lineage_json) is not str:
        raise ValueError(f"{label} lacks canonical lineage_payload_json.")
    try:
        payload = json.loads(lineage_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} lineage_payload_json is invalid JSON.") from exc
    if type(payload) is not dict or lineage_json != canonical_lineage_json(payload):
        raise ValueError(f"{label} lineage payload is not one canonical object.")
    lineage_hash = compute_run_lineage_hash(payload)
    for attr_name in ("source_fingerprint", "source_lineage_hash", "lineage_hash"):
        if group.attrs.get(attr_name) != lineage_hash:
            raise ValueError(f"{label} {attr_name} differs from its lineage payload.")
    if group.attrs.get("fingerprint_status") not in {"best_effort", "complete"}:
        raise ValueError(f"{label} fingerprint_status is not usable.")
    if (
        group.attrs.get("lineage_fingerprint_schema_id")
        != ("palette.run_lineage_fingerprint_attrs")
        or group.attrs.get("lineage_fingerprint_schema_version") != 1
    ):
        raise ValueError(f"{label} lineage attribute schema identity mismatch.")
    if group.attrs.get("lineage_fingerprint_canonicalization") != (
        "json_sorted_keys_run_lineage_v1"
    ):
        raise ValueError(f"{label} lineage canonicalization mismatch.")
    return lineage_hash, hashlib.sha256(lineage_json.encode("utf-8")).hexdigest()


def _iter_group_tree(group: Any, prefix: str = ""):
    yield prefix, group
    for name, child in sorted(group.groups(), key=lambda item: item[0]):
        child_path = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_group_tree(child, child_path)


def _stimulus_group_fingerprint(group: Any) -> str:
    return stimulus_group_logical_fingerprint(group)


def build_stimulus_epoch_candidate_plan(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
) -> StimulusEpochCandidatePlan:
    if profile_id != SUPPORTED_PROFILE_ID:
        raise ValueError(
            f"Stimulus-epoch candidates require profile {SUPPORTED_PROFILE_ID!r}."
        )
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}.")
    if (
        scratch == source
        or scratch.is_relative_to(source)
        or source.is_relative_to(scratch)
    ):
        raise ValueError(
            "Scratch and authoritative archive trees must be disjoint in both directions."
        )
    source_name = _safe_name(source_run, label="source run")
    candidate_name = _safe_name(run_name, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")

    source_run_path = source.joinpath(*PARENT_PATH.split("/"), source_name)
    if source_run_path.is_symlink():
        raise ValueError("Source stimulus-epoch run cannot be a symbolic link.")
    _require_contained(source_run_path, source, label="source run path")
    if any(path.is_symlink() for path in source_run_path.rglob("*")):
        raise ValueError("Source stimulus-epoch run cannot contain symbolic links.")
    target_run_path = source.joinpath(*PARENT_PATH.split("/"), candidate_name)
    _require_contained(target_run_path.parent, source, label="candidate parent path")

    root = open_zarr_root(source, mode="r")
    parent = root.get(PARENT_PATH)
    if not isinstance(parent, zarr.Group):
        raise KeyError(f"Missing stimulus-epoch parent {PARENT_PATH!r}.")
    source_group = parent.get(source_name)
    if not isinstance(source_group, zarr.Group):
        raise KeyError(f"Source stimulus-epoch run {source_name!r} does not exist.")
    if source_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError("Source stimulus-epoch run is not explicitly complete.")
    if source_group.attrs.get("stage_selector_eligible") is False:
        raise ValueError("Source stimulus-epoch run is explicitly selector-ineligible.")
    errors = validate_legacy_stimulus_epoch_source(source_group)
    if errors:
        raise ValueError(
            "Legacy stimulus-epoch source is invalid: " + "; ".join(errors)
        )
    source_epoch_lineage_hash, source_epoch_lineage_payload_sha256 = (
        _validated_lineage_identity(source_group, label="Source stimulus-epoch run")
    )
    source_epoch_logical_content_sha256 = stimulus_epoch_logical_content_sha256(
        source_group
    )
    source_stimulus_run = source_group.attrs.get("source_stimulus_run")
    source_stimulus_path = source_group.attrs.get("source_stimulus_path")
    if type(source_stimulus_run) is not str or not source_stimulus_run.strip():
        raise ValueError("Source epoch lacks an exact source_stimulus_run.")
    if (
        type(source_stimulus_path) is not str
        or source_stimulus_path != f"analysis/stimulus_runs/{source_stimulus_run}"
    ):
        raise ValueError("Source epoch source stimulus run/path binding is invalid.")
    stimulus_group = root.get(source_stimulus_path)
    if not isinstance(stimulus_group, zarr.Group):
        raise ValueError("Source epoch's exact source stimulus group is missing.")
    stimulus_physical_path = source.joinpath(*source_stimulus_path.split("/"))
    if stimulus_physical_path.is_symlink() or any(
        path.is_symlink() for path in stimulus_physical_path.rglob("*")
    ):
        raise ValueError("Source stimulus group cannot contain symbolic links.")
    _require_contained(
        stimulus_physical_path,
        source,
        label="source stimulus path",
    )
    source_stimulus_fingerprint = _stimulus_group_fingerprint(stimulus_group)
    if (
        candidate_name in parent
        or target_run_path.exists()
        or target_run_path.is_symlink()
    ):
        raise FileExistsError(f"Candidate run {candidate_name!r} already exists.")

    return StimulusEpochCandidatePlan(
        source_zarr=source,
        source_run_name=source_name,
        run_name=candidate_name,
        scratch_root=scratch,
        local_zarr=scratch / "stimulus-epoch-candidate.zarr",
        profile_id=profile_id,
        source_stimulus_run=source_stimulus_run,
        source_stimulus_path=source_stimulus_path,
        source_stimulus_fingerprint=source_stimulus_fingerprint,
        source_epoch_lineage_hash=source_epoch_lineage_hash,
        source_epoch_lineage_payload_sha256=source_epoch_lineage_payload_sha256,
        source_epoch_logical_content_sha256=(source_epoch_logical_content_sha256),
        latest_before=parent.attrs.get("latest"),
        latest_complete_before=parent.attrs.get("latest_complete"),
    )


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _logical_hashes(group: Any, declarations: Sequence[Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for declaration in declarations:
        array = _array_at_path(group, declaration.path)
        values = np.ascontiguousarray(array[...])
        digest = hashlib.sha256()
        digest.update(str(np.dtype(array.dtype)).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        hashes[declaration.path] = digest.hexdigest()
    return hashes


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {str(key): _normalize_metadata(child) for key, child in value.items()}
        if result.get("node_type") == "group":
            envelope = result.get("consolidated_metadata")
            if envelope is None or envelope == {
                "kind": "inline",
                "must_understand": False,
                "metadata": {},
            }:
                result.pop("consolidated_metadata", None)
        return result
    if isinstance(value, (tuple, list)):
        return [_normalize_metadata(child) for child in value]
    if value == "NaN" or (
        isinstance(value, (float, np.floating)) and np.isnan(float(value))
    ):
        return {"palette_exact_float": "nan"}
    return value


def _normalize_node_declaration(value: Mapping[str, Any], *, path: str) -> Any:
    declaration = dict(value)
    if declaration.get("zarr_format") != 3:
        raise ValueError(f"Zarr declaration {path!r} is not format 3.")
    node_type = declaration.get("node_type")
    if node_type == "group":
        required = {"zarr_format", "node_type", "attributes"}
        optional = {"consolidated_metadata"}
        if not required.issubset(declaration) or not set(declaration).issubset(
            required | optional
        ):
            raise ValueError(f"Group declaration {path!r} has unexpected fields.")
        envelope = declaration.pop("consolidated_metadata", None)
        if envelope is not None and not isinstance(envelope, Mapping):
            raise ValueError(
                f"Group declaration {path!r} has malformed consolidated metadata."
            )
    elif node_type == "array":
        if "consolidated_metadata" in declaration:
            raise ValueError(
                f"Array declaration {path!r} cannot contain consolidated metadata."
            )
    else:
        raise ValueError(f"Zarr declaration {path!r} has invalid node_type.")
    canonical_json_bytes(declaration)
    return _normalize_metadata(declaration)


def _metadata_declaration_tree(
    group: Any,
    declarations: Sequence[Any],
) -> dict[str, Any]:
    observed_groups = {path for path, _node in _iter_group_tree(group)}
    if observed_groups != {"", "windows"}:
        raise ValueError(
            "Stimulus-epoch declaration tree has unexpected groups: "
            f"{sorted(observed_groups)!r}."
        )
    groups = {
        "": _normalize_node_declaration(group.metadata.to_dict(), path=""),
        "windows": _normalize_node_declaration(
            group["windows"].metadata.to_dict(), path="windows"
        ),
    }
    arrays = {
        declaration.path: _normalize_node_declaration(
            _array_at_path(group, declaration.path).metadata.to_dict(),
            path=declaration.path,
        )
        for declaration in declarations
    }
    return {
        "schema_id": "palette.stimulus_epoch.normalized_zarr_declaration_tree",
        "schema_version": 1,
        "path_basis": "relative_to_stimulus_epoch_run_empty_string_is_root",
        "groups": groups,
        "arrays": arrays,
    }


def _direct_consolidated_check(
    zarr_path: Path,
    *,
    run_path: str,
    declarations: Sequence[Any],
    consolidate: bool = True,
) -> int:
    if consolidate:
        consolidate_metadata_capture_expected_warnings(zarr_path)
    direct_root = zarr.open_group(
        str(zarr_path), mode="r", zarr_format=3, use_consolidated=False
    )
    consolidated_root = zarr.open_group(
        str(zarr_path), mode="r", zarr_format=3, use_consolidated=True
    )
    direct = direct_root[run_path]
    consolidated = consolidated_root[run_path]
    direct_tree = _metadata_declaration_tree(direct, declarations)
    consolidated_tree = _metadata_declaration_tree(consolidated, declarations)
    if canonical_json_bytes(direct_tree) != canonical_json_bytes(consolidated_tree):
        raise ValueError(
            "Direct and consolidated stimulus-epoch declaration trees differ."
        )
    return len(direct_tree["arrays"])


def _validate_candidate_group(
    group: Any,
    *,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    errors = list(
        validate_stimulus_epoch_array_manifest(
            group,
            byte_planner_adopted=True,
        )
    )
    errors.extend(validate_stimulus_epoch_candidate_lineage(group))
    errors.extend(validate_stimulus_epoch_run_manifest(group))
    try:
        declarations = build_stimulus_epoch_array_declarations(
            group,
            byte_planner_adopted=True,
        )
        errors.extend(
            validate_exact_tabular_storage_receipt(
                group,
                declarations=declarations,
            )
        )
        hashes = _logical_hashes(group, declarations)
        if dict(hashes) != dict(expected_hashes):
            errors.append("candidate decoded logical hashes differ from source")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        declarations = ()
        hashes = {}
    if group.attrs.get("stage_selector_eligible") is not False:
        errors.append("candidate is not selector-ineligible")
    if group.attrs.get("storage_candidate_profile_promoted") is not False:
        errors.append("storage_candidate_profile_promoted is not exact false")
    if group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("candidate is not complete")
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": len(declarations),
        "logical_hashes": hashes,
    }


def tombstone_stimulus_epoch_execution_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail one published benchmark candidate owned by the named execution."""

    archive = Path(source_zarr).expanduser().resolve()
    name = _safe_name(run_name, label="candidate run")
    expected_binding = json_attr_safe(dict(expected_execution_binding))
    if not expected_binding:
        raise ValueError("expected_execution_binding must be nonempty.")
    tombstone_payload = {
        "schema_id": "palette.analysis_candidate_execution_tombstone",
        "schema_version": 1,
        "execution_binding": expected_binding,
        "failure_phase": str(failure_phase),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }
    tombstone = {
        **tombstone_payload,
        "payload_sha256": canonical_json_sha256(tombstone_payload),
    }
    run_path = f"{PARENT_PATH}/{name}"
    with archive_metadata_publication_lock(archive):
        root = open_zarr_root(archive, mode="a")
        parent = root[PARENT_PATH]
        pointers_before = {
            key: parent.attrs.get(key) for key in ("latest", "latest_complete")
        }
        run = parent.get(name)
        if not isinstance(run, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        attrs = run.attrs
        if attrs.get(EXECUTION_BINDING_ATTR) != expected_binding:
            raise RuntimeError(
                "Refusing to tombstone a stimulus-epoch candidate owned by another "
                "execution."
            )
        if (
            attrs.get("stage_selector_eligible") is not False
            or attrs.get("storage_candidate_profile_promoted") is not False
        ):
            raise RuntimeError(
                "Refusing to tombstone a selector-eligible or promoted "
                "stimulus-epoch candidate."
            )
        existing = attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR)
        if attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_FAILED:
            if existing != tombstone:
                raise RuntimeError(
                    "Existing stimulus-epoch execution tombstone differs."
                )
        else:
            if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                raise RuntimeError(
                    "Stimulus-epoch execution candidate is neither complete nor failed."
                )
            mark_run_failed(
                run,
                parent_group=None,
                run_name=name,
                error=f"{error_type}: {error_message}",
            )
            run.attrs["stage_selector_eligible"] = False
            run.attrs["storage_candidate_profile_promoted"] = False
            run.attrs[EXECUTION_FAILURE_TOMBSTONE_ATTR] = tombstone
        if {
            key: parent.attrs.get(key) for key in ("latest", "latest_complete")
        } != pointers_before:
            raise RuntimeError("Stimulus-epoch tombstone changed selector state.")
        consolidate_metadata_capture_expected_warnings(archive)
        receipt = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
        for fresh in (
            zarr.open_group(
                str(archive), mode="r", zarr_format=3, use_consolidated=False
            )[run_path],
            zarr.open_group(
                str(archive), mode="r", zarr_format=3, use_consolidated=True
            )[run_path],
        ):
            if (
                fresh.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED
                or fresh.attrs.get("stage_selector_eligible") is not False
                or fresh.attrs.get("storage_candidate_profile_promoted") is not False
                or fresh.attrs.get(EXECUTION_BINDING_ATTR) != expected_binding
                or fresh.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR) != tombstone
            ):
                raise RuntimeError(
                    "Stimulus-epoch execution tombstone did not persist exactly."
                )
    return {
        "target_present": True,
        "tombstoned": True,
        "metadata_declarations_sha256": receipt.declarations_sha256,
    }


def materialize_stimulus_epoch_candidate(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    stage_source_to_scratch: bool = False,
    execution_binding: Mapping[str, Any] | None = None,
    expected_source_logical_hashes: Mapping[str, str] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    telemetry = PhaseTelemetry(
        materializer="stimulus_epoch_candidate",
        context={
            "source_run": source_run,
            "run_name": run_name,
            "stage_source_to_scratch": bool(stage_source_to_scratch),
        },
    )
    with telemetry.phase("plan"):
        plan = build_stimulus_epoch_candidate_plan(
            source_zarr,
            source_run=source_run,
            run_name=run_name,
            scratch_root=scratch_root,
            profile_id=profile_id,
        )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.as_dict(),
    }
    if not apply:
        result["runtime_telemetry"] = _ordered_runtime_telemetry(telemetry)
        return result
    succeeded = False
    try:
        if plan.scratch_root.exists() or plan.scratch_root.is_symlink():
            raise FileExistsError(
                f"Refusing existing scratch root: {plan.scratch_root}."
            )
        plan.scratch_root.mkdir(parents=True)
        with telemetry.phase("source_staging"):
            authoritative_root = open_zarr_root(plan.source_zarr, mode="r")
            authoritative_source = authoritative_root[plan.source_run_path]
            authoritative_stimulus = authoritative_root[plan.source_stimulus_path]
            observed_source_lineage_hash, observed_source_lineage_payload_sha256 = (
                _validated_lineage_identity(
                    authoritative_source,
                    label="Source stimulus-epoch run",
                )
            )
            observed_source_content_sha256 = stimulus_epoch_logical_content_sha256(
                authoritative_source
            )
            observed_stimulus_fingerprint = _stimulus_group_fingerprint(
                authoritative_stimulus
            )
            if (
                observed_source_lineage_hash != plan.source_epoch_lineage_hash
                or observed_source_lineage_payload_sha256
                != plan.source_epoch_lineage_payload_sha256
                or observed_source_content_sha256
                != plan.source_epoch_logical_content_sha256
                or observed_stimulus_fingerprint != plan.source_stimulus_fingerprint
            ):
                raise RuntimeError(
                    "Stimulus-epoch or source-stimulus identity changed after "
                    "planning."
                )
            authoritative_declarations = build_stimulus_epoch_array_declarations(
                authoritative_source,
                byte_planner_adopted=False,
            )
            source_hashes = _logical_hashes(
                authoritative_source,
                authoritative_declarations,
            )
            if expected_source_logical_hashes is not None and source_hashes != dict(
                expected_source_logical_hashes
            ):
                raise ValueError(
                    "Stimulus-epoch source logical hashes differ from the execution "
                    "request."
                )
            if stage_source_to_scratch:
                staged_source_path = plan.scratch_root / "staged-epoch-source"
                staged_stimulus_path = plan.scratch_root / "staged-stimulus-source"
                _copy_group_tree(
                    plan.source_run_physical_path,
                    staged_source_path,
                    backend=copy_backend,
                )
                _copy_group_tree(
                    plan.source_zarr.joinpath(*plan.source_stimulus_path.split("/")),
                    staged_stimulus_path,
                    backend=copy_backend,
                )
                source_group = zarr.open_group(
                    str(staged_source_path),
                    mode="r",
                    zarr_format=3,
                    use_consolidated=False,
                )
                staged_stimulus = zarr.open_group(
                    str(staged_stimulus_path),
                    mode="r",
                    zarr_format=3,
                    use_consolidated=False,
                )
                staged_errors = validate_legacy_stimulus_epoch_source(source_group)
                staged_lineage, staged_lineage_payload = _validated_lineage_identity(
                    source_group,
                    label="Staged stimulus-epoch run",
                )
                staged_declarations = build_stimulus_epoch_array_declarations(
                    source_group,
                    byte_planner_adopted=False,
                )
                if (
                    staged_errors
                    or staged_lineage != plan.source_epoch_lineage_hash
                    or staged_lineage_payload
                    != plan.source_epoch_lineage_payload_sha256
                    or stimulus_epoch_logical_content_sha256(source_group)
                    != plan.source_epoch_logical_content_sha256
                    or _stimulus_group_fingerprint(staged_stimulus)
                    != plan.source_stimulus_fingerprint
                    or _logical_hashes(source_group, staged_declarations)
                    != source_hashes
                ):
                    raise ValueError(
                        "Node-local stimulus-epoch source staging differs from the "
                        "authoritative source."
                    )
            else:
                source_group = authoritative_source
            candidate_declarations = build_stimulus_epoch_array_declarations(
                source_group,
                byte_planner_adopted=True,
            )
            receipt = build_exact_tabular_storage_receipt(
                source_group,
                declarations=candidate_declarations,
                profile=get_storage_profile(plan.profile_id),
            )

        with telemetry.phase("logical_rematerialization"):
            local_root = zarr.open_group(
                str(plan.local_zarr),
                mode="w-",
                zarr_format=3,
                use_consolidated=False,
            )
            local_parent = local_root
            for component in PARENT_PATH.split("/"):
                local_parent = local_parent.require_group(component)
            local_group = local_parent.create_group(plan.run_name)
            rematerialize_exact_tabular_candidate(
                source_group,
                local_group,
                receipt=receipt,
            )
            git = get_git_info(Path(__file__).resolve().parents[4])
            local_group.attrs.update(
                {
                    "schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
                    "schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
                    "layout": STIMULUS_EPOCH_LAYOUT,
                    "row_axis": "epoch_windows",
                    "legacy_source_schema_id": LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
                    "legacy_source_schema_version": (
                        LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION
                    ),
                    "storage_candidate_source_run": plan.source_run_name,
                    "storage_candidate_source_run_path": plan.source_run_path,
                    "storage_candidate_profile_promoted": False,
                    "source_stimulus_epoch_run": plan.source_run_name,
                    "source_stimulus_epoch_path": plan.source_run_path,
                    "source_stimulus_fingerprint_algorithm": (
                        STIMULUS_SOURCE_FINGERPRINT_ALGORITHM
                    ),
                    "source_stimulus_fingerprint": plan.source_stimulus_fingerprint,
                    "source_stimulus_epoch_lineage_hash": (
                        plan.source_epoch_lineage_hash
                    ),
                    "source_stimulus_epoch_lineage_payload_sha256": (
                        plan.source_epoch_lineage_payload_sha256
                    ),
                    "source_stimulus_epoch_logical_content_sha256": (
                        plan.source_epoch_logical_content_sha256
                    ),
                    "candidate_materializer_git_commit": git.get("commit_hash"),
                    "candidate_materializer_git_dirty": bool(git.get("is_dirty")),
                    "source_staging_mode": (
                        "epoch_and_stimulus_logical_copy_v1"
                        if stage_source_to_scratch
                        else "authoritative_direct_compatibility"
                    ),
                    "stage_selector_eligible": False,
                    RUN_NAME_ATTR: plan.run_name,
                    RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_RUNNING,
                }
            )
            if execution_binding is not None:
                binding = json_attr_safe(dict(execution_binding))
                if not binding:
                    raise ValueError("execution_binding must be one nonempty mapping.")
                local_group.attrs[EXECUTION_BINDING_ATTR] = binding
            local_group.attrs.pop(RUN_COMPLETED_AT_ATTR, None)
            write_stimulus_epoch_array_manifest(
                local_group,
                byte_planner_adopted=True,
            )
            persist_exact_tabular_storage_receipt(local_group, receipt)
            write_run_lineage_attrs(
                local_group,
                build_stimulus_epoch_candidate_lineage_payload(local_group),
                fingerprint_status="complete",
                overwrite=True,
            )
            mark_run_complete(
                local_group,
                parent_group=None,
                run_name=plan.run_name,
                run_provenance=build_run_provenance_from_stage_record(
                    source_group.attrs.get("provenance", {}),
                    fallback_command="stimulus_epoch_candidate_materializer",
                ),
            )
            write_stimulus_epoch_run_manifest(local_group)
        with telemetry.phase("local_validation"):
            local_validation = _validate_candidate_group(
                local_group,
                expected_hashes=source_hashes,
            )
            if not local_validation["valid"]:
                raise RuntimeError(
                    f"Local stimulus-epoch candidate is invalid: {local_validation}."
                )
        with telemetry.phase("local_consolidation"):
            consolidate_metadata_capture_expected_warnings(plan.local_zarr)
        with telemetry.phase("local_direct_consolidated_comparison"):
            local_compared = _direct_consolidated_check(
                plan.local_zarr,
                run_path=plan.run_path,
                declarations=candidate_declarations,
                consolidate=False,
            )
        materialization_seconds = float(
            sum(
                telemetry.duration_seconds(name) or 0.0
                for name in (
                    "logical_rematerialization",
                    "local_validation",
                    "local_consolidation",
                    "local_direct_consolidated_comparison",
                )
            )
        )
        stable_root = open_zarr_root(plan.source_zarr, mode="r")
        stable_source = stable_root[plan.source_run_path]
        stable_lineage_hash, stable_lineage_payload_sha256 = (
            _validated_lineage_identity(
                stable_source,
                label="Source stimulus-epoch run",
            )
        )
        if (
            stable_lineage_hash != plan.source_epoch_lineage_hash
            or stable_lineage_payload_sha256 != plan.source_epoch_lineage_payload_sha256
            or stimulus_epoch_logical_content_sha256(stable_source)
            != plan.source_epoch_logical_content_sha256
            or _stimulus_group_fingerprint(stable_root[plan.source_stimulus_path])
            != plan.source_stimulus_fingerprint
        ):
            raise RuntimeError(
                "Stimulus-epoch or source-stimulus identity changed during "
                "candidate materialization."
            )

        def validate(path: Path) -> dict[str, Any]:
            return _validate_candidate_group(
                open_zarr_root(path, mode="r"),
                expected_hashes=source_hashes,
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, "stimulus_epoch_runs"),)

        def complete(
            _root: zarr.Group,
            _parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            mark_run_complete(
                run_group,
                parent_group=None,
                run_name=plan.run_name,
                run_provenance=run_group.attrs.get("run_provenance"),
            )
            run_group.attrs["stage_selector_eligible"] = False
            run_group.attrs["storage_candidate_profile_promoted"] = False

        def verify(root: zarr.Group) -> None:
            parent = root[PARENT_PATH]
            if (
                parent.attrs.get("latest") != plan.latest_before
                or parent.attrs.get("latest_complete") != plan.latest_complete_before
            ):
                raise RuntimeError("Stimulus-epoch candidate changed parent selectors.")
            candidate = parent.get(plan.run_name)
            if not isinstance(candidate, zarr.Group):
                raise RuntimeError("Published stimulus-epoch candidate is missing.")
            if (
                candidate.attrs.get("stage_selector_eligible") is not False
                or candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError(
                    "Published stimulus-epoch candidate is not complete and ineligible."
                )

        publication_acceptance: dict[str, Any] = {}

        def consolidate_archive(
            root: zarr.Group,
            parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            if (
                run_group.attrs.get("stage_selector_eligible") is not False
                or run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError(
                    "Stimulus-epoch candidate lost its fail-closed state before "
                    "consolidation."
                )
            stable_source = root[plan.source_run_path]
            stable_lineage_hash, stable_lineage_payload_sha256 = (
                _validated_lineage_identity(
                    stable_source,
                    label="Published source stimulus-epoch run",
                )
            )
            if (
                stable_lineage_hash != plan.source_epoch_lineage_hash
                or stable_lineage_payload_sha256
                != plan.source_epoch_lineage_payload_sha256
                or stimulus_epoch_logical_content_sha256(stable_source)
                != plan.source_epoch_logical_content_sha256
                or _stimulus_group_fingerprint(root[plan.source_stimulus_path])
                != plan.source_stimulus_fingerprint
            ):
                raise RuntimeError(
                    "Stimulus-epoch or source-stimulus identity changed during "
                    "publication."
                )
            with telemetry.phase("published_validation"):
                published_validation = _validate_candidate_group(
                    run_group,
                    expected_hashes=source_hashes,
                )
                if not published_validation["valid"]:
                    raise RuntimeError(
                        "Published stimulus-epoch candidate is invalid: "
                        f"{published_validation}."
                    )
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
            with telemetry.phase("published_direct_consolidated_comparison"):
                published_compared = _direct_consolidated_check(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    declarations=candidate_declarations,
                    consolidate=False,
                )
            with telemetry.phase("decoded_equality"):
                published_hashes = _logical_hashes(
                    run_group,
                    candidate_declarations,
                )
                if published_hashes != source_hashes:
                    raise RuntimeError(
                        "Published stimulus-epoch decoded values differ from source."
                    )
            with telemetry.phase("physical_inventory"):
                published_storage = storage_stats(plan.target_run_path)
                if (
                    published_storage["file_count"] < 1
                    or published_storage["apparent_bytes"] < 1
                    or published_storage["allocated_bytes"] < 1
                ):
                    raise RuntimeError(
                        "Published stimulus-epoch candidate has no physical payload."
                    )
            publication_acceptance.update(
                archive_direct_consolidated_array_count=published_compared,
                published_validation=published_validation,
                published_direct_consolidated_array_count=published_compared,
                published_hashes=published_hashes,
                output_storage=published_storage,
            )
            if publication_acceptance_validator is not None:
                with telemetry.phase("publication_acceptance_validation"):
                    publication_acceptance["caller_acceptance"] = json_attr_safe(
                        dict(publication_acceptance_validator(root, parent, run_group))
                    )

        def repair_failed_visibility(_target_path: Path) -> None:
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)

        with telemetry.phase("atomic_publication"):
            publication = atomic_publish_run_group(
                AtomicRunPublishSpec(
                    source_zarr=plan.source_zarr,
                    local_run_path=plan.local_run_path,
                    target_run_path=plan.target_run_path,
                    run_name=plan.run_name,
                    lock_suffix="stimulus-epoch-storage-candidate",
                    publish_schema_id=PUBLISH_SCHEMA_ID,
                    policy=(
                        "stimulus_epoch_v2_byte_planned_atomic_nonpromoting_publish"
                    ),
                    rollback_policy=(
                        "retain_failed_public_tombstone_leave_parent_selectors_untouched"
                    ),
                    content_checksum=True,
                ),
                copy_backend=copy_backend,
                validate_run=validate,
                prepare_parents=prepare,
                complete_run=complete,
                verify_pointers=verify,
                activate_run=consolidate_archive,
                repair_failed_publication_visibility=repair_failed_visibility,
                payload_metadata={
                    "profile_id": plan.profile_id,
                    "source_run": plan.source_run_name,
                    "source_logical_hashes": source_hashes,
                    "local_direct_consolidated_array_count": local_compared,
                    "materialization_seconds": materialization_seconds,
                },
            )
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=local_compared,
            archive_direct_consolidated_array_count=publication_acceptance[
                "archive_direct_consolidated_array_count"
            ],
            published_validation=publication_acceptance["published_validation"],
            published_direct_consolidated_array_count=publication_acceptance[
                "published_direct_consolidated_array_count"
            ],
            source_logical_manifest_sha256=canonical_json_sha256(source_hashes),
            published_logical_manifest_sha256=canonical_json_sha256(
                publication_acceptance["published_hashes"]
            ),
            output_storage=publication_acceptance["output_storage"],
            caller_acceptance=publication_acceptance.get("caller_acceptance"),
            materialization_seconds=materialization_seconds,
            publication=publication,
            runtime_telemetry=_ordered_runtime_telemetry(telemetry),
        )
        succeeded = True
        return json_attr_safe(result)
    except BaseException as exc:
        try:
            setattr(
                exc,
                "palette_runtime_telemetry",
                _ordered_runtime_telemetry(telemetry),
            )
        except BaseException:
            pass
        raise
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_stimulus_epoch_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_stimulus_epoch_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument(
        "--profile", choices=(SUPPORTED_PROFILE_ID,), default=SUPPORTED_PROFILE_ID
    )
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_stimulus_epoch_candidate(
        args.zarr_path,
        source_run=args.source_run,
        run_name=args.run_name,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        profile_id=args.profile,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MATERIALIZATION_SCHEMA_ID",
    "PUBLISH_SCHEMA_ID",
    "SUPPORTED_PROFILE_ID",
    "StimulusEpochCandidatePlan",
    "build_stimulus_epoch_candidate_plan",
    "materialize_stimulus_epoch_candidate",
]
