"""Recording-level subject-mask bundle publication and atomic activation.

Raw probabilities, refined dense authority, and quality diagnostics are
materialized and validated independently.  This module imports those complete
selector-ineligible children into one recording archive, proves their exact
cross-run identity, seals one bundle candidate, and optionally activates only
that bundle through a single root authority envelope.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    validate_persisted_subject_mask_core_publication,
    validate_subject_mask_core_run_manifest,
)
from fisheye.shared.zarr.subject_mask_quality_manifest import (
    SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE,
    validate_subject_mask_quality_publication,
    validate_subject_mask_quality_run_manifest,
)
from fisheye.shared.zarr.subject_mask_quality_publication import (
    subject_mask_quality_metadata_declaration_maps,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)

SUBJECT_MASK_BUNDLE_FAMILY = "subject_mask_bundle_runs"
SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE = "run_manifest"
SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID = "palette.subject_mask.bundle_manifest"
SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_VERSION = 1
SUBJECT_MASK_BUNDLE_PUBLICATION_SCHEMA_ID = "palette.subject_mask.bundle_publication"
SUBJECT_MASK_BUNDLE_PUBLICATION_SCHEMA_VERSION = 1
SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR = "subject_mask_authority"
SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR = "subject_mask_authority_generation"
SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR = "subject_mask_authority_lease"
SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR = "subject_mask_bundle_selector_eligible"
SUBJECT_MASK_BUNDLE_METADATA_DIGEST_SCOPE = (
    "exact_bundle_group_declaration_redacting_manifest_lifecycle_and_bundle_"
    "eligibility"
)
SUBJECT_MASK_BUNDLE_PUBLICATION_POLICY = (
    "independent_node_local_materialization_then_atomic_run_import_v1"
)
SUBJECT_MASK_BUNDLE_ROLLBACK_POLICY = (
    "retain_imported_complete_selector_ineligible_candidates_v1"
)
SUBJECT_MASK_BUNDLE_ACTIVATION_POLICY = (
    "bundle_members_ready_then_single_root_authority_commit_v1"
)

_MEMBER_SPECS = {
    "raw": ("subject_mask_runs", "raw_probability_uint8"),
    "refined": ("refined_subject_masks_runs", "refined_dense_core"),
    "quality": ("subject_mask_quality_runs", "subject_mask_quality"),
}
_IDENTITY_PATHS = (
    "source_crop_row_ids",
    "instance_key",
    "source_acquisition_frame_index",
    "frame_row_offsets",
    "source_crop_xywh",
    "available_channels",
)
_FAMILY_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _require_run_id(value: str, *, name: str) -> str:
    resolved = str(value).strip()
    if (
        not resolved
        or "/" in resolved
        or resolved in {".", ".."}
        or any(character.isspace() for character in resolved)
    ):
        raise ValueError(f"{name} must be one safe nonempty run id.")
    return resolved


def _manifest_array_document(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("payload")
    logical = payload.get("logical_content") if isinstance(payload, Mapping) else None
    document = logical.get("document") if isinstance(logical, Mapping) else None
    arrays = document.get("arrays") if isinstance(document, Mapping) else None
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("Run manifest lacks its exact logical array inventory.")
    return arrays


def _manifest_dimensions(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("payload")
    logical = payload.get("logical_schema") if isinstance(payload, Mapping) else None
    dimensions = logical.get("dimensions") if isinstance(logical, Mapping) else None
    if not isinstance(dimensions, Mapping):
        raise ValueError("Run manifest lacks exact dimensions.")
    return dimensions


def _manifest_components(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("payload")
    logical = payload.get("logical_schema") if isinstance(payload, Mapping) else None
    components = logical.get("components") if isinstance(logical, Mapping) else None
    if not isinstance(components, Mapping):
        raise ValueError("Run manifest lacks an exact component registry.")
    return components


def _array_metadata_errors(
    run: Any,
    declarations: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for path, declaration in declarations.items():
        if not isinstance(declaration, Mapping):
            errors.append(f"array declaration is invalid at {path}")
            continue
        try:
            array = run[str(path)]
        except (KeyError, FileNotFoundError):
            errors.append(f"array is absent at {path}")
            continue
        if list(array.shape) != declaration.get("shape"):
            errors.append(f"array shape differs at {path}")
        if str(np.dtype(array.dtype)) != declaration.get("dtype"):
            errors.append(f"array dtype differs at {path}")
    return errors


def _validate_local_core_tree(
    run_path: Path,
    *,
    family: str,
    run_id: str,
    kind: str,
) -> dict[str, object]:
    errors: list[str] = []
    try:
        run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        manifest = run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
        if not isinstance(manifest, Mapping):
            return {"valid": False, "errors": ["core run_manifest is absent"]}
        errors.extend(validate_subject_mask_core_run_manifest(manifest))
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping):
            errors.append("core manifest payload is absent")
        else:
            if (
                payload.get("run_id") != run_id
                or payload.get("stage_family") != family
                or payload.get("kind") != kind
            ):
                errors.append("core manifest path/kind binding differs")
            errors.extend(
                _array_metadata_errors(run, _manifest_array_document(manifest))
            )
        if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            errors.append("core run is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("core run is not selector-ineligible")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"valid": not errors, "errors": errors}


def _validate_local_quality_tree(
    run_path: Path,
    *,
    run_id: str,
) -> dict[str, object]:
    errors: list[str] = []
    try:
        run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        manifest = run.attrs.get(SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE)
        if not isinstance(manifest, Mapping):
            return {"valid": False, "errors": ["quality run_manifest is absent"]}
        errors.extend(validate_subject_mask_quality_run_manifest(manifest))
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping) or payload.get("run_id") != run_id:
            errors.append("quality manifest path binding differs")
        errors.extend(_array_metadata_errors(run, _manifest_array_document(manifest)))
        if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            errors.append("quality run is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("quality run is not selector-ineligible")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"valid": not errors, "errors": errors}


def _member_reference(
    *, role: str, family: str, run_id: str, manifest: Mapping[str, Any]
) -> dict[str, object]:
    payload = manifest["payload"]
    logical = payload["logical_content"]
    return {
        "role": role,
        "family": family,
        "run_id": run_id,
        "run_path": f"{family}/{run_id}",
        "manifest_schema_id": manifest["schema_id"],
        "manifest_schema_version": manifest["schema_version"],
        "manifest_payload_digest": manifest["payload_digest"],
        "manifest_document_digest": canonical_json_sha256(manifest),
        "logical_content_digest": logical["digest"],
    }


def _bundle_cross_binding(
    *,
    raw_manifest: Mapping[str, Any],
    refined_manifest: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    refined_run_id: str,
) -> dict[str, object]:
    raw_dimensions = dict(_manifest_dimensions(raw_manifest))
    refined_dimensions = dict(_manifest_dimensions(refined_manifest))
    if raw_dimensions != refined_dimensions:
        raise ValueError("Raw and refined subject-mask dimensions differ.")
    raw_components = dict(_manifest_components(raw_manifest))
    refined_components = dict(_manifest_components(refined_manifest))
    if raw_components != refined_components:
        raise ValueError("Raw and refined subject-mask component registries differ.")
    raw_arrays = _manifest_array_document(raw_manifest)
    refined_arrays = _manifest_array_document(refined_manifest)
    identity: dict[str, str] = {}
    for path in _IDENTITY_PATHS:
        raw_declaration = raw_arrays.get(path)
        refined_declaration = refined_arrays.get(path)
        if not isinstance(raw_declaration, Mapping) or not isinstance(
            refined_declaration, Mapping
        ):
            raise ValueError(f"Bundle identity array {path!r} is absent.")
        raw_hash = raw_declaration.get("sha256")
        refined_hash = refined_declaration.get("sha256")
        if raw_hash != refined_hash:
            raise ValueError(f"Raw/refined identity differs at {path!r}.")
        identity[path] = str(raw_hash)

    quality_payload = quality_manifest.get("payload")
    quality_source = (
        quality_payload.get("source_refined_subject_mask_snapshot")
        if isinstance(quality_payload, Mapping)
        else None
    )
    if not isinstance(quality_source, Mapping):
        raise ValueError("Quality manifest lacks its refined-source binding.")
    expected_source_hashes = {
        path: str(refined_arrays[path]["sha256"])
        for path in (
            "masks_roi",
            "instance_key",
            "source_crop_row_ids",
            "source_acquisition_frame_index",
            "frame_row_offsets",
            "available_channels",
        )
    }
    expected_quality_source = {
        "run_name": refined_run_id,
        "run_path": f"refined_subject_masks_runs/{refined_run_id}",
        "manifest_digest": canonical_json_sha256(refined_manifest),
        "dense_array_values_sha256": expected_source_hashes["masks_roi"],
        "component_registry_digest": canonical_json_sha256(refined_components),
        "source_array_values_sha256": expected_source_hashes,
    }
    for name, expected in expected_quality_source.items():
        if quality_source.get(name) != expected:
            raise ValueError(f"Quality/refined source binding differs for {name!r}.")
    quality_dimensions = _manifest_dimensions(quality_manifest)
    for name in ("n_frames", "n_rois", "n_channels", "roi_height", "roi_width"):
        if quality_dimensions.get(name) != refined_dimensions.get(name):
            raise ValueError(f"Quality/refined dimensions differ for {name!r}.")
    return {
        "dimensions": refined_dimensions,
        "components": refined_components,
        "component_registry_digest": canonical_json_sha256(refined_components),
        "raw_refined_identity_array_values_sha256": identity,
        "quality_source_array_values_sha256": expected_source_hashes,
        "quality_source_manifest_digest": canonical_json_sha256(refined_manifest),
        "identity_policy": "exact_logical_array_hash_equality_v1",
    }


def _normalized_bundle_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = metadata_without_empty_group_consolidation(value, path="")
    attributes = normalized.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("Bundle group metadata attributes are absent.")
    redacted = dict(attributes)
    for name in (
        SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE,
        "status",
        RUN_COMPLETION_STATUS_ATTR,
        RUN_COMPLETED_AT_ATTR,
        SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR,
    ):
        redacted.pop(name, None)
    normalized["attributes"] = redacted
    return normalized


def _bundle_metadata_digest(archive: Path, *, bundle_id: str) -> str:
    run_path = archive / SUBJECT_MASK_BUNDLE_FAMILY / bundle_id
    direct = _strict_json(run_path / "zarr.json")
    root = _strict_json(archive / "zarr.json")
    envelope = root.get("consolidated_metadata")
    flattened = envelope.get("metadata") if isinstance(envelope, Mapping) else None
    full_path = f"{SUBJECT_MASK_BUNDLE_FAMILY}/{bundle_id}"
    consolidated = flattened.get(full_path) if isinstance(flattened, Mapping) else None
    if not isinstance(consolidated, Mapping):
        raise ValueError("Consolidated metadata lacks the subject-mask bundle.")
    direct_normalized = _normalized_bundle_metadata(direct)
    consolidated_normalized = _normalized_bundle_metadata(consolidated)
    if direct_normalized != consolidated_normalized:
        raise ValueError("Direct and consolidated bundle metadata differ.")
    return canonical_json_sha256(direct_normalized)


def build_subject_mask_bundle_manifest(
    *,
    bundle_id: str,
    recording_identity: str,
    members: Mapping[str, Mapping[str, Any]],
    cross_binding: Mapping[str, Any],
    import_receipt_digests: Mapping[str, str],
    metadata_digest: str,
) -> dict[str, object]:
    resolved_id = _require_run_id(bundle_id, name="bundle_id")
    identity = str(recording_identity).strip()
    if not identity:
        raise ValueError("recording_identity cannot be empty.")
    if set(members) != set(_MEMBER_SPECS):
        raise ValueError("Subject-mask bundle member roles are not exact.")
    if set(import_receipt_digests) != set(_MEMBER_SPECS):
        raise ValueError("Subject-mask bundle import receipt roles are not exact.")
    payload = {
        "bundle_id": resolved_id,
        "recording_identity": identity,
        "publication": {
            "completion_contract": RUN_COMPLETION_CONTRACT,
            "completion_status": RUN_STATUS_COMPLETE,
            "stage_selector_eligible": False,
            "bundle_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_digest_scope": SUBJECT_MASK_BUNDLE_METADATA_DIGEST_SCOPE,
            "metadata_digest": str(metadata_digest),
            "activation_policy": SUBJECT_MASK_BUNDLE_ACTIVATION_POLICY,
            "activation_state": "deferred",
        },
        "members": {role: dict(members[role]) for role in sorted(members)},
        "cross_binding": dict(cross_binding),
        "import_receipt_digests": {
            role: str(import_receipt_digests[role])
            for role in sorted(import_receipt_digests)
        },
    }
    envelope = {
        "schema_id": SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def validate_subject_mask_bundle_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(manifest) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("subject-mask bundle manifest fields are not exact")
    payload = manifest.get("payload")
    if (
        manifest.get("schema_id") != SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_VERSION
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, Mapping)
    ):
        errors.append("subject-mask bundle manifest envelope mismatch")
        return tuple(errors)
    try:
        if manifest.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("subject-mask bundle payload digest mismatch")
    except (TypeError, ValueError) as exc:
        errors.append(f"subject-mask bundle is not strict JSON: {exc}")
    if set(payload) != {
        "bundle_id",
        "recording_identity",
        "publication",
        "members",
        "cross_binding",
        "import_receipt_digests",
    }:
        errors.append("subject-mask bundle payload fields are not exact")
    try:
        _require_run_id(str(payload.get("bundle_id") or ""), name="bundle_id")
    except ValueError as exc:
        errors.append(str(exc))
    if not str(payload.get("recording_identity") or "").strip():
        errors.append("subject-mask bundle recording identity is absent")
    publication = payload.get("publication")
    expected_publication = {
        "completion_contract": RUN_COMPLETION_CONTRACT,
        "completion_status": RUN_STATUS_COMPLETE,
        "stage_selector_eligible": False,
        "bundle_selector_eligible": False,
        "metadata_state": "direct_and_consolidated_validated",
        "metadata_digest_scope": SUBJECT_MASK_BUNDLE_METADATA_DIGEST_SCOPE,
        "metadata_digest": (
            publication.get("metadata_digest")
            if isinstance(publication, Mapping)
            else None
        ),
        "activation_policy": SUBJECT_MASK_BUNDLE_ACTIVATION_POLICY,
        "activation_state": "deferred",
    }
    if (
        not isinstance(publication, Mapping)
        or dict(publication) != expected_publication
    ):
        errors.append("subject-mask bundle publication declaration differs")
    members = payload.get("members")
    if not isinstance(members, Mapping) or set(members) != set(_MEMBER_SPECS):
        errors.append("subject-mask bundle members are not exact")
    else:
        for role, (family, _kind) in _MEMBER_SPECS.items():
            member = members.get(role)
            if (
                not isinstance(member, Mapping)
                or member.get("role") != role
                or member.get("family") != family
                or member.get("run_path") != f"{family}/{member.get('run_id')}"
            ):
                errors.append(f"subject-mask bundle member differs for {role}")
    receipts = payload.get("import_receipt_digests")
    if not isinstance(receipts, Mapping) or set(receipts) != set(_MEMBER_SPECS):
        errors.append("subject-mask bundle import receipt digests are not exact")
    return tuple(errors)


def _require_unselected(root: Any, *, family: str, run_id: str) -> None:
    parent = root[family]
    selected = [
        name for name in _FAMILY_SELECTOR_ATTRS if parent.attrs.get(name) == run_id
    ]
    if selected:
        raise RuntimeError(
            f"Selector-ineligible subject-mask run {run_id!r} is selected by {selected!r}."
        )
    if parent[run_id].attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(f"Subject-mask run {run_id!r} became selector-eligible.")


def _atomic_import_member(
    *,
    archive: Path,
    local_run_path: Path,
    family: str,
    run_id: str,
    role: str,
    kind: str,
    copy_backend: str,
) -> dict[str, Any]:
    target = archive / family / run_id
    if target.exists():
        raise FileExistsError(f"Immutable subject-mask member exists: {target}")

    def validate(path: Path) -> Mapping[str, Any]:
        if role == "quality":
            return _validate_local_quality_tree(path, run_id=run_id)
        return _validate_local_core_tree(
            path,
            family=family,
            run_id=run_id,
            kind=kind,
        )

    def prepare(root: Any) -> Sequence[Any]:
        return (root.require_group(family),)

    def complete(_root: Any, _parent: Any, run: Any) -> None:
        if (
            run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Imported subject-mask member is not complete/ineligible."
            )

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=archive,
            local_run_path=local_run_path,
            target_run_path=target,
            run_name=run_id,
            lock_suffix="subject_mask_bundle_publication",
            publish_schema_id=SUBJECT_MASK_BUNDLE_PUBLICATION_SCHEMA_ID,
            policy=SUBJECT_MASK_BUNDLE_PUBLICATION_POLICY,
            rollback_policy=SUBJECT_MASK_BUNDLE_ROLLBACK_POLICY,
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=lambda root: _require_unselected(
            root, family=family, run_id=run_id
        ),
        payload_metadata={
            "bundle_member_role": role,
            "selector_activation": "deferred_bundle_only",
        },
    )


def _preflight_immutable_targets(
    root: Any,
    *,
    ids: Mapping[str, str],
    bundle_id: str,
) -> None:
    existing: list[str] = []
    for role, (family, _kind) in _MEMBER_SPECS.items():
        if family in root and ids[role] in root[family]:
            existing.append(f"{family}/{ids[role]}")
    if (
        SUBJECT_MASK_BUNDLE_FAMILY in root
        and bundle_id in root[SUBJECT_MASK_BUNDLE_FAMILY]
    ):
        existing.append(f"{SUBJECT_MASK_BUNDLE_FAMILY}/{bundle_id}")
    if existing:
        raise FileExistsError(
            "Immutable subject-mask publication targets already exist: "
            + ", ".join(sorted(existing))
        )


def _persisted_manifests(
    root: Any, *, raw_run_id: str, refined_run_id: str, quality_run_id: str
) -> dict[str, Mapping[str, Any]]:
    locations = {
        "raw": (
            "subject_mask_runs",
            raw_run_id,
            SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
        ),
        "refined": (
            "refined_subject_masks_runs",
            refined_run_id,
            SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
        ),
        "quality": (
            "subject_mask_quality_runs",
            quality_run_id,
            SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE,
        ),
    }
    result: dict[str, Mapping[str, Any]] = {}
    for role, (family, run_id, attribute) in locations.items():
        manifest = root[f"{family}/{run_id}"].attrs.get(attribute)
        if not isinstance(manifest, Mapping):
            raise RuntimeError(f"Published {role} member lacks its run_manifest.")
        result[role] = manifest
    return result


def _validate_persisted_members(
    archive: Path,
    *,
    raw_run_id: str,
    refined_run_id: str,
    quality_run_id: str,
    refined_manifest: Mapping[str, Any],
) -> None:
    raw_errors = validate_persisted_subject_mask_core_publication(
        archive,
        family="subject_mask_runs",
        run_id=raw_run_id,
    )
    refined_errors = validate_persisted_subject_mask_core_publication(
        archive,
        family="refined_subject_masks_runs",
        run_id=refined_run_id,
    )
    quality_run = zarr.open_group(
        str(archive / "subject_mask_quality_runs" / quality_run_id),
        mode="r",
        use_consolidated=False,
    )
    quality_manifest = quality_run.attrs.get(
        SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE
    )
    quality_errors: tuple[str, ...]
    if not isinstance(quality_manifest, Mapping):
        quality_errors = ("quality run_manifest is absent",)
    else:
        payload = quality_manifest.get("payload")
        storage_plan = (
            payload.get("storage_plan") if isinstance(payload, Mapping) else None
        )
        plans = (
            storage_plan.get("arrays") if isinstance(storage_plan, Mapping) else None
        )
        if not isinstance(plans, list):
            quality_errors = ("quality storage plan is absent",)
        else:

            class _Plans:
                entries = tuple(
                    type(
                        "_Entry",
                        (),
                        {"rule": type("_Rule", (), {"path": item["path"]})()},
                    )()
                    for item in plans
                )

            direct, consolidated = subject_mask_quality_metadata_declaration_maps(
                archive,
                run_id=quality_run_id,
                plans=_Plans(),
            )
            arrays = {
                path: quality_run[path]
                for path in SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths
            }
            quality_errors = validate_subject_mask_quality_publication(
                quality_manifest,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
                arrays=arrays,
                source_manifest=refined_manifest,
            )
            if quality_run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                quality_errors = (*quality_errors, "quality run is not complete")
            if quality_run.attrs.get("stage_selector_eligible") is not False:
                quality_errors = (
                    *quality_errors,
                    "quality run is not selector-ineligible",
                )
    if raw_errors or refined_errors or quality_errors:
        raise RuntimeError(
            "Persisted subject-mask bundle member validation failed: "
            f"raw={list(raw_errors)}, refined={list(refined_errors)}, "
            f"quality={list(quality_errors)}"
        )


def publish_subject_mask_bundle_candidate(
    *,
    analysis_zarr: Path,
    recording_identity: str,
    raw_snapshot_root: Path,
    raw_run_id: str,
    refined_snapshot_root: Path,
    refined_run_id: str,
    quality_snapshot_root: Path,
    quality_run_id: str,
    bundle_id: str,
    copy_backend: str = "python",
) -> dict[str, object]:
    """Import and seal one complete selector-ineligible three-run bundle."""

    started = time.perf_counter()
    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    identity = str(recording_identity).strip()
    if not identity:
        raise ValueError("recording_identity cannot be empty.")
    root = open_zarr_root(archive, mode="r")
    if str(root.attrs.get("recording_id") or "").strip() != identity:
        raise ValueError("Requested recording identity differs from the archive root.")
    ids = {
        "raw": _require_run_id(raw_run_id, name="raw_run_id"),
        "refined": _require_run_id(refined_run_id, name="refined_run_id"),
        "quality": _require_run_id(quality_run_id, name="quality_run_id"),
    }
    resolved_bundle = _require_run_id(bundle_id, name="bundle_id")
    local_paths = {
        "raw": raw_snapshot_root.expanduser().resolve()
        / "subject_mask_runs"
        / ids["raw"],
        "refined": refined_snapshot_root.expanduser().resolve()
        / "refined_subject_masks_runs"
        / ids["refined"],
        "quality": quality_snapshot_root.expanduser().resolve()
        / "subject_mask_quality_runs"
        / ids["quality"],
    }
    local_validations = {
        "raw": _validate_local_core_tree(
            local_paths["raw"],
            family="subject_mask_runs",
            run_id=ids["raw"],
            kind="raw_probability_uint8",
        ),
        "refined": _validate_local_core_tree(
            local_paths["refined"],
            family="refined_subject_masks_runs",
            run_id=ids["refined"],
            kind="refined_dense_core",
        ),
        "quality": _validate_local_quality_tree(
            local_paths["quality"], run_id=ids["quality"]
        ),
    }
    invalid = {
        role: value for role, value in local_validations.items() if not value["valid"]
    }
    if invalid:
        raise RuntimeError(f"Local subject-mask bundle members are invalid: {invalid}")

    # Detect every predictable immutable-name collision before importing the
    # first member.  A failed retry must not leave a new orphan merely because
    # a later member or the bundle id was already occupied.
    _preflight_immutable_targets(root, ids=ids, bundle_id=resolved_bundle)

    import_receipts: dict[str, dict[str, Any]] = {}
    for role in ("raw", "refined", "quality"):
        family, kind = _MEMBER_SPECS[role]
        import_receipts[role] = _atomic_import_member(
            archive=archive,
            local_run_path=local_paths[role],
            family=family,
            run_id=ids[role],
            role=role,
            kind=kind,
            copy_backend=copy_backend,
        )
    consolidate_metadata_capture_expected_warnings(archive)
    root = open_zarr_root(archive, mode="a")
    manifests = _persisted_manifests(
        root,
        raw_run_id=ids["raw"],
        refined_run_id=ids["refined"],
        quality_run_id=ids["quality"],
    )
    cross_binding = _bundle_cross_binding(
        raw_manifest=manifests["raw"],
        refined_manifest=manifests["refined"],
        quality_manifest=manifests["quality"],
        refined_run_id=ids["refined"],
    )
    _validate_persisted_members(
        archive,
        raw_run_id=ids["raw"],
        refined_run_id=ids["refined"],
        quality_run_id=ids["quality"],
        refined_manifest=manifests["refined"],
    )

    family = root.require_group(SUBJECT_MASK_BUNDLE_FAMILY)
    if resolved_bundle in family:
        raise FileExistsError(
            f"Immutable subject-mask bundle exists: {resolved_bundle}"
        )
    bundle = family.create_group(resolved_bundle)
    mark_run_started(
        bundle,
        run_name=resolved_bundle,
        stage="subject_mask_bundle",
    )
    bundle.attrs.update(
        {
            "schema_id": SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "status": "running",
            "stage_selector_eligible": False,
            SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR: False,
            "recording_identity": identity,
            "member_run_paths": {
                role: f"{_MEMBER_SPECS[role][0]}/{ids[role]}" for role in sorted(ids)
            },
        }
    )
    try:
        consolidate_metadata_capture_expected_warnings(archive)
        metadata_digest = _bundle_metadata_digest(archive, bundle_id=resolved_bundle)
        members = {
            role: _member_reference(
                role=role,
                family=_MEMBER_SPECS[role][0],
                run_id=ids[role],
                manifest=manifests[role],
            )
            for role in sorted(ids)
        }
        import_digests = {
            role: canonical_json_sha256(import_receipts[role])
            for role in sorted(import_receipts)
        }
        manifest = build_subject_mask_bundle_manifest(
            bundle_id=resolved_bundle,
            recording_identity=identity,
            members=members,
            cross_binding=cross_binding,
            import_receipt_digests=import_digests,
            metadata_digest=metadata_digest,
        )
        manifest_errors = validate_subject_mask_bundle_manifest(manifest)
        if manifest_errors:
            raise RuntimeError("; ".join(manifest_errors))
        bundle.attrs[SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE] = manifest
        consolidate_metadata_capture_expected_warnings(archive)
        if (
            _bundle_metadata_digest(archive, bundle_id=resolved_bundle)
            != metadata_digest
        ):
            raise RuntimeError(
                "Bundle metadata digest changed after manifest insertion."
            )
        _validate_persisted_members(
            archive,
            raw_run_id=ids["raw"],
            refined_run_id=ids["refined"],
            quality_run_id=ids["quality"],
            refined_manifest=manifests["refined"],
        )
        writable = zarr.open_group(
            str(archive / SUBJECT_MASK_BUNDLE_FAMILY / resolved_bundle),
            mode="a",
            use_consolidated=False,
        )
        writable.attrs["status"] = "complete"
        mark_run_complete(writable, run_name=resolved_bundle)
        consolidate_metadata_capture_expected_warnings(archive)
        if (
            _bundle_metadata_digest(archive, bundle_id=resolved_bundle)
            != metadata_digest
        ):
            raise RuntimeError("Bundle metadata digest changed during completion.")
        reopened = zarr.open_group(
            str(archive / SUBJECT_MASK_BUNDLE_FAMILY / resolved_bundle),
            mode="r",
            use_consolidated=False,
        )
        if (
            reopened.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or reopened.attrs.get("stage_selector_eligible") is not False
            or reopened.attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR)
            is not False
            or reopened.attrs.get(SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE) != manifest
        ):
            raise RuntimeError("Completed subject-mask bundle did not reopen exactly.")
    except BaseException as exc:
        failed = zarr.open_group(
            str(archive / SUBJECT_MASK_BUNDLE_FAMILY / resolved_bundle),
            mode="a",
            use_consolidated=False,
        )
        failed.attrs["stage_selector_eligible"] = False
        failed.attrs[SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR] = False
        failed.attrs["status"] = "failed"
        mark_run_failed(failed, run_name=resolved_bundle, error=str(exc))
        consolidate_metadata_capture_expected_warnings(archive)
        raise
    return {
        "schema_id": SUBJECT_MASK_BUNDLE_PUBLICATION_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_BUNDLE_PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "analysis_zarr": str(archive),
        "recording_identity": identity,
        "bundle_id": resolved_bundle,
        "bundle_path": f"{SUBJECT_MASK_BUNDLE_FAMILY}/{resolved_bundle}",
        "bundle_manifest_digest": manifest["payload_digest"],
        "members": members,
        "import_receipt_digests": import_digests,
        "import_receipts": import_receipts,
        "selector_eligible": False,
        "activation_state": "deferred",
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _validate_live_bundle(
    archive: Path, *, bundle_id: str
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    root = open_zarr_root(archive, mode="r")
    bundle = root[f"{SUBJECT_MASK_BUNDLE_FAMILY}/{bundle_id}"]
    manifest = bundle.attrs.get(SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Subject-mask bundle manifest is absent.")
    errors = validate_subject_mask_bundle_manifest(manifest)
    if errors:
        raise RuntimeError("Invalid subject-mask bundle: " + "; ".join(errors))
    if bundle.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise RuntimeError("Subject-mask bundle is not complete.")
    if bundle.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError("Subject-mask bundle is individually selector-eligible.")
    payload = manifest["payload"]
    if payload["recording_identity"] != root.attrs.get("recording_id"):
        raise RuntimeError("Subject-mask bundle recording identity changed.")
    members = payload["members"]
    member_ids = {role: str(members[role]["run_id"]) for role in _MEMBER_SPECS}
    manifests = _persisted_manifests(
        root,
        raw_run_id=member_ids["raw"],
        refined_run_id=member_ids["refined"],
        quality_run_id=member_ids["quality"],
    )
    observed_cross = _bundle_cross_binding(
        raw_manifest=manifests["raw"],
        refined_manifest=manifests["refined"],
        quality_manifest=manifests["quality"],
        refined_run_id=member_ids["refined"],
    )
    if observed_cross != payload["cross_binding"]:
        raise RuntimeError("Subject-mask bundle cross-binding changed.")
    for role in _MEMBER_SPECS:
        expected = _member_reference(
            role=role,
            family=_MEMBER_SPECS[role][0],
            run_id=member_ids[role],
            manifest=manifests[role],
        )
        if expected != members[role]:
            raise RuntimeError(f"Subject-mask bundle member changed for {role}.")
    _validate_persisted_members(
        archive,
        raw_run_id=member_ids["raw"],
        refined_run_id=member_ids["refined"],
        quality_run_id=member_ids["quality"],
        refined_manifest=manifests["refined"],
    )
    if (
        _bundle_metadata_digest(archive, bundle_id=bundle_id)
        != payload["publication"]["metadata_digest"]
    ):
        raise RuntimeError("Subject-mask bundle metadata digest changed.")
    return dict(manifest), manifests


def validate_subject_mask_bundle_candidate(
    *, analysis_zarr: Path, bundle_id: str
) -> dict[str, object]:
    """Reopen and deeply validate one complete inactive bundle candidate."""

    archive = analysis_zarr.expanduser().resolve()
    resolved_bundle = _require_run_id(bundle_id, name="bundle_id")
    manifest, manifests = _validate_live_bundle(archive, bundle_id=resolved_bundle)
    return {
        "status": "valid",
        "analysis_zarr": str(archive),
        "bundle_id": resolved_bundle,
        "bundle_manifest_digest": str(manifest["payload_digest"]),
        "member_manifest_digests": {
            role: str(value["payload_digest"])
            for role, value in sorted(manifests.items())
        },
        "selector_eligible": False,
    }


def _recover_stale_activation_lease(archive: Path) -> None:
    """Repair an interrupted pre-commit activation while holding its file lock."""

    root = open_zarr_root(archive, mode="r")
    lease = root.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR)
    if lease is None:
        return
    required = {
        "owner_uuid",
        "bundle_id",
        "bundle_manifest_digest",
        "next_generation",
        "policy",
    }
    if not isinstance(lease, Mapping) or set(lease) != required:
        raise RuntimeError("Subject-mask activation has a malformed stale lease.")
    if lease.get("policy") != SUBJECT_MASK_BUNDLE_ACTIVATION_POLICY:
        raise RuntimeError("Subject-mask activation has an unknown stale lease policy.")
    bundle_id = _require_run_id(
        str(lease.get("bundle_id") or ""), name="lease bundle_id"
    )
    bundle = root[f"{SUBJECT_MASK_BUNDLE_FAMILY}/{bundle_id}"]
    manifest = bundle.attrs.get(SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise RuntimeError(
            "Subject-mask activation stale lease has no bundle manifest."
        )
    manifest_errors = validate_subject_mask_bundle_manifest(manifest)
    if manifest_errors:
        raise RuntimeError(
            "Subject-mask activation stale lease has an invalid bundle: "
            + "; ".join(manifest_errors)
        )
    if bundle.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise RuntimeError("Subject-mask activation stale lease bundle is incomplete.")
    if manifest.get("payload_digest") != lease.get("bundle_manifest_digest"):
        raise RuntimeError("Subject-mask activation stale lease binding changed.")

    authority = root.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR)
    active_paths: set[str] = set()
    if isinstance(authority, Mapping):
        bundle_path = authority.get("bundle_path")
        members = authority.get("members")
        if not isinstance(bundle_path, str) or not isinstance(members, Mapping):
            raise RuntimeError("Existing subject-mask authority is malformed.")
        active_paths.add(bundle_path)
        for role in _MEMBER_SPECS:
            member = members.get(role)
            if not isinstance(member, Mapping) or not isinstance(
                member.get("run_path"), str
            ):
                raise RuntimeError(
                    "Existing subject-mask authority members are malformed."
                )
            active_paths.add(str(member["run_path"]))

    target_paths = {
        f"{SUBJECT_MASK_BUNDLE_FAMILY}/{bundle_id}",
        *(
            str(manifest["payload"]["members"][role]["run_path"])
            for role in _MEMBER_SPECS
        ),
    }
    writable = open_zarr_root(archive, mode="a")
    for path in target_paths:
        writable[path].attrs[SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR] = (
            path in active_paths
        )
    attrs = copy.deepcopy(dict(writable.attrs))
    attrs.pop(SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR, None)
    writable.attrs.put(attrs)
    consolidate_metadata_capture_expected_warnings(archive)

    reopened = open_zarr_root(archive, mode="r")
    if SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR in reopened.attrs:
        raise RuntimeError("Subject-mask stale activation lease was not cleared.")
    for path in target_paths:
        expected = path in active_paths
        if (
            reopened[path].attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR)
            is not expected
        ):
            raise RuntimeError(
                "Subject-mask stale activation readiness was not repaired."
            )
    _validate_live_bundle(archive, bundle_id=bundle_id)


def activate_subject_mask_bundle(
    *, analysis_zarr: Path, bundle_id: str
) -> dict[str, object]:
    """Atomically select one validated bundle without touching family selectors."""

    archive = analysis_zarr.expanduser().resolve()
    resolved_bundle = _require_run_id(bundle_id, name="bundle_id")
    lock_path = archive.parent / f".{archive.name}.subject_mask_bundle_activation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner = str(uuid4())
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _recover_stale_activation_lease(archive)
            manifest, _manifests = _validate_live_bundle(
                archive, bundle_id=resolved_bundle
            )
            root = open_zarr_root(archive, mode="a")
            if root.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR) is not None:
                raise RuntimeError(
                    "Another subject-mask bundle activation owns the lease."
                )
            root_attrs_before = copy.deepcopy(dict(root.attrs))
            generation = int(
                root.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR, 0)
            )
            next_generation = generation + 1
            lease = {
                "owner_uuid": owner,
                "bundle_id": resolved_bundle,
                "bundle_manifest_digest": manifest["payload_digest"],
                "next_generation": next_generation,
                "policy": SUBJECT_MASK_BUNDLE_ACTIVATION_POLICY,
            }
            root.attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR] = lease
            member_paths = tuple(
                manifest["payload"]["members"][role]["run_path"]
                for role in sorted(_MEMBER_SPECS)
            )
            ready_paths = (
                f"{SUBJECT_MASK_BUNDLE_FAMILY}/{resolved_bundle}",
                *member_paths,
            )
            prior_ready: dict[str, tuple[bool, Any]] = {}
            committed = False
            try:
                for path in ready_paths:
                    group = root[path]
                    prior_ready[path] = (
                        SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR in group.attrs,
                        copy.deepcopy(
                            group.attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR)
                        ),
                    )
                    group.attrs[SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR] = True
                consolidate_metadata_capture_expected_warnings(archive)
                _validate_live_bundle(archive, bundle_id=resolved_bundle)
                check = open_zarr_root(archive, mode="r")
                if check.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR) != lease:
                    raise RuntimeError("Subject-mask activation lease changed.")
                authority = {
                    "schema_id": "palette.subject_mask.bundle_authority",
                    "schema_version": 1,
                    "generation": next_generation,
                    "bundle_id": resolved_bundle,
                    "bundle_path": f"{SUBJECT_MASK_BUNDLE_FAMILY}/{resolved_bundle}",
                    "bundle_manifest_digest": manifest["payload_digest"],
                    "members": {
                        role: dict(manifest["payload"]["members"][role])
                        for role in sorted(_MEMBER_SPECS)
                    },
                    "activated_at_utc": utc_now(),
                    "activation_owner_uuid": owner,
                    "activation_policy": SUBJECT_MASK_BUNDLE_ACTIVATION_POLICY,
                }
                final_root = open_zarr_root(archive, mode="a")
                final_attrs = copy.deepcopy(dict(final_root.attrs))
                if final_attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR) != lease:
                    raise RuntimeError("Subject-mask activation lease was replaced.")
                final_attrs.pop(SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR, None)
                final_attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR] = (
                    next_generation
                )
                final_attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR] = authority
                # This attrs.put is the sole authority/selection commit.  No
                # scientific or selector metadata is mutated after it.
                final_root.attrs.put(final_attrs)
                reopened = open_zarr_root(archive, mode="r")
                if (
                    reopened.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR) != authority
                    or reopened.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR)
                    != next_generation
                    or SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR in reopened.attrs
                ):
                    raise RuntimeError("Subject-mask bundle authority did not persist.")
                committed = True
                return authority
            finally:
                if not committed:
                    observed = open_zarr_root(archive, mode="r")
                    authority = observed.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR)
                    if (
                        isinstance(authority, Mapping)
                        and authority.get("activation_owner_uuid") == owner
                        and authority.get("bundle_manifest_digest")
                        == manifest["payload_digest"]
                    ):
                        committed = True
                    else:
                        rollback = open_zarr_root(archive, mode="a")
                        for path, (present, value) in prior_ready.items():
                            group = rollback[path]
                            if present:
                                group.attrs[
                                    SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR
                                ] = value
                            elif (
                                SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR
                                in group.attrs
                            ):
                                del group.attrs[
                                    SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR
                                ]
                        rollback.attrs.put(root_attrs_before)
                        consolidate_metadata_capture_expected_warnings(archive)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    raise RuntimeError("Subject-mask bundle activation ended without a result.")


__all__ = [
    "SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR",
    "SUBJECT_MASK_BUNDLE_AUTHORITY_GENERATION_ATTR",
    "SUBJECT_MASK_BUNDLE_AUTHORITY_LEASE_ATTR",
    "SUBJECT_MASK_BUNDLE_FAMILY",
    "SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE",
    "SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID",
    "SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR",
    "activate_subject_mask_bundle",
    "build_subject_mask_bundle_manifest",
    "publish_subject_mask_bundle_candidate",
    "validate_subject_mask_bundle_candidate",
    "validate_subject_mask_bundle_manifest",
]
