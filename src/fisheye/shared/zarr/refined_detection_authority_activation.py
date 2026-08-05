"""Guarded production activation for immutable refined-detection authorities.

This module is intentionally separate from snapshot construction.  It promotes
one already-published, complete, selector-ineligible refined-v1 snapshot after
deep logical/physical validation.  The run-group ``stage_selector_eligible``
attribute is the literal final archive write.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.selector_activation import activate_selector_eligible_run
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
    REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE,
    REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE,
    RefinedDetectionLineageProfile,
    build_refined_detection_activation_candidate_manifest,
    build_refined_detection_authority_provenance,
    parse_refined_detection_clipped_binding,
    refined_detection_dimensions_from_manifest,
    refined_detection_logical_content_digest,
    validate_refined_detection_authority_provenance,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    refined_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.refined_detection_storage import (
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


REFINED_AUTHORITY_ACTIVATION_SCHEMA_ID = (
    "palette.refined_detection.authority_activation"
)
REFINED_AUTHORITY_ACTIVATION_SCHEMA_VERSION = 1
REFINED_AUTHORITY_ACTIVATION_POLICY_ATTR = (
    "refined_detection_authority_activation_policy"
)
REFINED_AUTHORITY_ACTIVATION_GENERATION_ATTR = (
    "refined_detection_authority_activation_generation"
)
REFINED_AUTHORITY_ACTIVATION_LEASE_ATTR = (
    "refined_detection_authority_activation_lease"
)
REFINED_AUTHORITY_ACTIVATION_POLICY = (
    "owner_generation_refined_detection_authority_v1"
)
REFINED_AUTHORITY_ACTIVATION_LEASE_SCHEMA_ID = (
    "palette.refined_detection.authority_activation_lease"
)


class RefinedDetectionAuthorityActivationError(RuntimeError):
    """Raised when an authority candidate cannot be activated safely."""


def _arrays(run: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = refined_detection_dimensions_from_manifest(manifest)
    return {
        path: run[path]
        for path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
    }


def _parent_evidence(
    parent: Any,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    parent_ref = manifest["payload"]["snapshot_lineage"]["parent_snapshot"]
    if parent_ref is None:
        return None, None
    parent_id = parent_ref["run_id"]
    if parent_id not in parent:
        raise RefinedDetectionAuthorityActivationError(
            "Refined successor authority lacks its persisted parent snapshot."
        )
    parent_run = parent[parent_id]
    parent_manifest = parent_run.attrs.get(
        REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE
    )
    if not isinstance(parent_manifest, Mapping):
        raise RefinedDetectionAuthorityActivationError(
            "Refined successor parent lacks its exact run_manifest."
        )
    return parent_manifest, _arrays(parent_run, parent_manifest)


def _validate_publication(
    archive: Path,
    *,
    run_id: str,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_run_eligibility: bool,
) -> dict[str, Any]:
    root = open_zarr_root(archive, mode="r")
    parent = root.get("refined_detect_runs")
    if parent is None or run_id not in parent:
        raise RefinedDetectionAuthorityActivationError(
            f"Refined detection run {run_id!r} does not exist."
        )
    run = parent[run_id]
    manifest = run.attrs.get(REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority candidate lacks its exact run_manifest."
        )
    manifest = dict(manifest)
    if expected_manifest is not None and manifest != expected_manifest:
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority candidate manifest drifted."
        )
    if manifest["payload"]["run_id"] != run_id:
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority candidate run_id differs from its group name."
        )
    if run.attrs.get("status") != "complete" or run.attrs.get(
        RUN_COMPLETION_STATUS_ATTR
    ) != RUN_STATUS_COMPLETE:
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority candidate is not strictly complete."
        )
    if run.attrs.get("stage_selector_eligible") is not expected_run_eligibility:
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority candidate eligibility state differs."
        )
    owner = run.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
    if not isinstance(owner, str) or not owner.strip():
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority candidate lacks its publication owner UUID."
        )

    dimensions = refined_detection_dimensions_from_manifest(manifest)
    profile = storage_profile_from_manifest(
        manifest["payload"]["storage_plan"]["storage_profile"]
    )
    plans = plan_refined_detection_storage(dimensions, profile=profile)
    direct, consolidated = refined_detection_metadata_declaration_maps(
        archive,
        run_id=run_id,
        plans=plans,
    )
    arrays = _arrays(run, manifest)
    parent_manifest, parent_arrays = _parent_evidence(parent, manifest)
    errors = validate_refined_detection_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
        parent_manifest=parent_manifest,
        parent_arrays=parent_arrays,
    )
    if errors:
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority publication validation failed: "
            + "; ".join(errors)
        )
    clipped_binding = None
    if (
        dimensions.lineage_profile
        is RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT
    ):
        clipped_binding = parse_refined_detection_clipped_binding(
            manifest["payload"]["logical_schema"]["clipped_binding"]
        )
    logical_digest = refined_detection_logical_content_digest(
        arrays,
        dimensions=dimensions,
        clipped_binding=clipped_binding,
    )
    return {
        "manifest": manifest,
        "manifest_digest": manifest["payload_digest"],
        "logical_content_digest": logical_digest,
        "publication_owner_uuid": owner,
        "dimensions": dimensions.as_manifest(),
        "storage_profile_id": profile.profile_id,
    }


def inspect_refined_detection_authority_candidate(
    *,
    analysis_zarr: Path,
    run_id: str,
) -> dict[str, Any]:
    """Deeply inspect one still-invisible activation candidate without writes."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir() or archive.suffix != ".zarr":
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    name = str(run_id).strip()
    if not name or "/" in name:
        raise ValueError("run_id must be one safe child-group name.")
    proof = _validate_publication(
        archive,
        run_id=name,
        expected_run_eligibility=False,
    )
    manifest = proof["manifest"]
    if manifest["payload"]["publication"]["stage_selector_eligible"] is not False:
        raise RefinedDetectionAuthorityActivationError(
            "Unactivated refined candidate manifest must declare false intent."
        )
    candidate = build_refined_detection_activation_candidate_manifest(manifest)
    root = open_zarr_root(archive, mode="r")
    parent = root["refined_detect_runs"]
    if REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE in parent.attrs or (
        REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE in parent.attrs
    ):
        raise RefinedDetectionAuthorityActivationError(
            "Initial activation requires no pre-existing refined authority."
        )
    return json_attr_safe(
        {
            "schema_id": "palette.refined_detection.authority_candidate_inspection",
            "schema_version": 1,
            "status": "ready",
            "analysis_zarr": str(archive),
            "recording_identity": str(root.attrs.get("recording_id") or ""),
            "run_id": name,
            "manifest_digest_before": proof["manifest_digest"],
            "activation_manifest_digest": candidate["payload_digest"],
            "logical_content_digest": proof["logical_content_digest"],
            "publication_owner_uuid": proof["publication_owner_uuid"],
            "dimensions": proof["dimensions"],
            "storage_profile_id": proof["storage_profile_id"],
            "intended_use": "analysis",
            "authority_absent": True,
            "run_selector_eligible": False,
        }
    )


def _restore_attribute(attrs: Any, name: str, snapshot: tuple[bool, Any]) -> None:
    present, value = snapshot
    if present:
        attrs[name] = copy.deepcopy(value)
    elif name in attrs:
        del attrs[name]


def activate_refined_detection_authority(
    *,
    analysis_zarr: Path,
    run_id: str,
    approved_by: str,
    review_method: str,
    approved_at_utc: str | None = None,
    git_sha: str | None = None,
    note: str = "",
    expected_inspection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Activate one exact analysis-only authority under the archive lock."""

    archive = analysis_zarr.expanduser().resolve()
    name = str(run_id).strip()
    observed = inspect_refined_detection_authority_candidate(
        analysis_zarr=archive,
        run_id=name,
    )
    if expected_inspection is not None and observed != dict(expected_inspection):
        raise RefinedDetectionAuthorityActivationError(
            "Frozen refined authority candidate inspection drifted."
        )
    approval_time = approved_at_utc or utc_now()
    committed = False
    staged = False
    manifest_snapshot: tuple[bool, Any] | None = None
    provenance_snapshot: tuple[bool, Any] | None = None
    consolidation: Mapping[str, Any] | None = None
    authority: Mapping[str, Any] | None = None
    candidate_manifest: Mapping[str, Any] | None = None

    with archive_metadata_publication_lock(archive):
        current = inspect_refined_detection_authority_candidate(
            analysis_zarr=archive,
            run_id=name,
        )
        if current != observed:
            raise RefinedDetectionAuthorityActivationError(
                "Refined authority candidate changed before lock acquisition."
            )
        root = open_zarr_root(archive, mode="a")
        parent = root["refined_detect_runs"]
        run = parent[name]
        manifest_snapshot = (
            REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE in run.attrs,
            copy.deepcopy(
                run.attrs.get(REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE)
            ),
        )
        provenance_snapshot = (
            REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE in parent.attrs,
            copy.deepcopy(
                parent.attrs.get(REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE)
            ),
        )
        candidate_manifest = build_refined_detection_activation_candidate_manifest(
            manifest_snapshot[1]
        )
        authority = build_refined_detection_authority_provenance(
            run_id=name,
            run_manifest_digest=candidate_manifest["payload_digest"],
            approved_by=approved_by,
            approved_at_utc=approval_time,
            review_method=review_method,
            intended_use="analysis",
            git_sha=git_sha,
            note=note,
        )
        authority_errors = validate_refined_detection_authority_provenance(authority)
        if authority_errors:
            raise RefinedDetectionAuthorityActivationError(
                "Generated refined authority provenance is invalid: "
                + "; ".join(authority_errors)
            )
        try:
            run.attrs[REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE] = candidate_manifest
            parent.attrs[
                REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE
            ] = authority
            staged = True
            consolidation = consolidate_metadata_capture_expected_warnings(archive)

            expected_proof = _validate_publication(
                archive,
                run_id=name,
                expected_manifest=candidate_manifest,
                expected_run_eligibility=False,
            )

            def proof_loader() -> tuple[Any, ...]:
                proof = _validate_publication(
                    archive,
                    run_id=name,
                    expected_manifest=candidate_manifest,
                    expected_run_eligibility=False,
                )
                fresh_parent = open_zarr_root(archive, mode="r")[
                    "refined_detect_runs"
                ]
                if (
                    fresh_parent.attrs.get(
                        REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE
                    )
                    != authority
                ):
                    raise RefinedDetectionAuthorityActivationError(
                        "Refined authority provenance changed during activation."
                    )
                return (
                    proof["manifest_digest"],
                    proof["logical_content_digest"],
                    proof["publication_owner_uuid"],
                    proof["storage_profile_id"],
                    authority["payload_digest"],
                )

            activate_selector_eligible_run(
                root,
                parent,
                run,
                parent_path="refined_detect_runs",
                run_path=f"refined_detect_runs/{name}",
                run_name=name,
                owner_attr=ATOMIC_PUBLICATION_OWNER_ATTR,
                expected_owner_uuid=str(expected_proof["publication_owner_uuid"]),
                policy_attr=REFINED_AUTHORITY_ACTIVATION_POLICY_ATTR,
                generation_attr=REFINED_AUTHORITY_ACTIVATION_GENERATION_ATTR,
                lease_attr=REFINED_AUTHORITY_ACTIVATION_LEASE_ATTR,
                policy=REFINED_AUTHORITY_ACTIVATION_POLICY,
                lease_schema_id=REFINED_AUTHORITY_ACTIVATION_LEASE_SCHEMA_ID,
                proof_loader=proof_loader,
                selector_attrs=(REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE,),
                parent_guard_attrs=(
                    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
                    "latest",
                    "latest_complete",
                    "latest_pending",
                ),
            )
            committed = True
        except BaseException:
            fresh = open_zarr_root(archive, mode="a")
            fresh_parent = fresh["refined_detect_runs"]
            fresh_run = fresh_parent[name]
            if (
                fresh_run.attrs.get("stage_selector_eligible") is True
                and fresh_parent.attrs.get(
                    REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE
                )
                == name
                and fresh_parent.attrs.get(
                    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE
                )
                == authority
            ):
                committed = True
            if not committed and staged:
                if manifest_snapshot is not None:
                    _restore_attribute(
                        fresh_run.attrs,
                        REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE,
                        manifest_snapshot,
                    )
                if provenance_snapshot is not None:
                    _restore_attribute(
                        fresh_parent.attrs,
                        REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
                        provenance_snapshot,
                    )
                consolidate_metadata_capture_expected_warnings(archive)
            if not committed:
                raise

    if not committed or candidate_manifest is None or authority is None:
        raise RefinedDetectionAuthorityActivationError(
            "Refined authority activation did not reach its commit point."
        )

    # Post-commit verification is read-only.  It cannot rewrite or roll back an
    # authority whose visibility bit has already committed.
    from fisheye.shared.zarr.refined_detection_crop_source import (
        bind_refined_detection_crop_source,
    )

    bound = bind_refined_detection_crop_source(archive)
    if bound.run_id != name or (
        bound.selection_mode != "approved_authoritative_refined_v1"
    ):
        raise RefinedDetectionAuthorityActivationError(
            "Production refined authority binding failed after commit."
        )
    return json_attr_safe(
        {
            "schema_id": REFINED_AUTHORITY_ACTIVATION_SCHEMA_ID,
            "schema_version": REFINED_AUTHORITY_ACTIVATION_SCHEMA_VERSION,
            "status": "complete",
            "activated_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "recording_identity": observed["recording_identity"],
            "run_id": name,
            "intended_use": "analysis",
            "manifest_digest_before": observed["manifest_digest_before"],
            "activated_manifest_digest": candidate_manifest["payload_digest"],
            "logical_content_digest": bound.logical_content_digest,
            "publication_owner_uuid": observed["publication_owner_uuid"],
            "authority_provenance": authority,
            "activation_policy": REFINED_AUTHORITY_ACTIVATION_POLICY,
            "selection_mode": bound.selection_mode,
            "consolidation": consolidation,
            "final_visibility_write": "stage_selector_eligible=true",
            "post_commit_archive_writes": 0,
            "registry_updated": False,
        }
    )


__all__ = [
    "REFINED_AUTHORITY_ACTIVATION_GENERATION_ATTR",
    "REFINED_AUTHORITY_ACTIVATION_LEASE_ATTR",
    "REFINED_AUTHORITY_ACTIVATION_POLICY",
    "REFINED_AUTHORITY_ACTIVATION_POLICY_ATTR",
    "REFINED_AUTHORITY_ACTIVATION_SCHEMA_ID",
    "REFINED_AUTHORITY_ACTIVATION_SCHEMA_VERSION",
    "RefinedDetectionAuthorityActivationError",
    "activate_refined_detection_authority",
    "inspect_refined_detection_authority_candidate",
]
