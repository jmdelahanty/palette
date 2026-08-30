"""Immutable operational selection of one reviewed arena-geometry candidate.

Candidate publication proves that geometry is readable.  This module records
the separate human/policy decision that makes exactly one candidate an
operational detection gate.  Candidates are never modified by selection.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import zarr

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    ACQUISITION_CANDIDATE_KIND,
    CANDIDATE_RUNS_PARENT,
    PALETTE_CANDIDATE_KIND,
    validate_arena_geometry_candidate_record,
)
from fisheye.analysis_workflows.materializers.arena_geometry_comparison import (
    COMPARISON_RUNS_PARENT,
    validate_arena_geometry_comparison_record,
)
from fisheye.shared.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.selector_activation import (
    SelectorActivationError,
    activate_selector_eligible_run,
)
from fisheye.shared.zarr.arena_geometry_selection import (
    COMPARISON_BOUND_SELECTION_POLICY,
    LEGACY_SELECTION_RECORD_SCHEMA_VERSION,
    MANUAL_PALETTE_SELECTION_POLICY,
    MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION,
    SELECTION_POLICY,
    SELECTION_RECORD_SCHEMA_ID,
    SELECTION_RECORD_SCHEMA_VERSION,
    SELECTION_RUN_SCHEMA_ID,
    SELECTION_RUN_SCHEMA_VERSION,
    SELECTION_RUNS_PARENT,
    validate_arena_geometry_selection_record,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)

SELECTION_PUBLISH_SCHEMA_ID = "palette.arena_geometry_selection_publish"
SELECTION_ALGORITHM_VERSION = 1
SELECTION_POLICY_ATTR = "arena_geometry_selection_policy"
SELECTION_GENERATION_ATTR = "arena_geometry_selection_generation"
SELECTION_LEASE_ATTR = "arena_geometry_selection_lease"


@dataclass(frozen=True)
class ArenaGeometrySelectionPlan:
    source_zarr: Path
    candidate_run: str
    candidate_record_sha256: str
    selection_id: str
    selection_record_sha256: str
    selection_record: Mapping[str, Any]
    target_run_path: Path
    run_provenance: Mapping[str, Any]


def _canonical_copy(value: Any) -> Any:
    return json.loads(strict_json_dumps(value))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _safe_name(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or "/" in text or text in {".", ".."}:
        raise ValueError(f"{label} must be one safe group name.")
    return text


def _candidate_snapshot(root: Any, candidate_run: str) -> dict[str, Any]:
    name = _safe_name(candidate_run, label="candidate_run")
    path = f"analysis/{CANDIDATE_RUNS_PARENT}/{name}"
    try:
        group = root[path]
    except KeyError as exc:
        raise ValueError(f"Arena-geometry candidate is missing: {path}") from exc
    attrs = dict(group.attrs)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not True
        or attrs.get("operational_selection_status") != "not_selected"
        or attrs.get("detection_gate_applied") is not False
    ):
        raise ValueError(
            f"Candidate {name!r} is not one complete, readable, unselected candidate."
        )
    record = attrs.get("candidate_record")
    if not isinstance(record, Mapping):
        raise ValueError(f"Candidate {name!r} lacks candidate_record.")
    validate_arena_geometry_candidate_record(record)
    digest = _payload_sha256(record)
    if attrs.get("candidate_record_sha256") != digest:
        raise ValueError(f"Candidate {name!r} record digest is invalid.")
    if attrs.get("candidate_id") != name:
        raise ValueError(f"Candidate {name!r} identity does not match its run name.")
    candidate_kind = record["candidate_kind"]
    physical = record.get("physical_inner_rim")
    observed = record.get("observed_boundary")
    if candidate_kind == ACQUISITION_CANDIDATE_KIND:
        if not isinstance(physical, Mapping) or observed is not None:
            raise ValueError("Acquisition candidate boundary semantics are invalid.")
        boundary_observation = {
            "role": "producer_physical_inner_rim",
            "observed_feature": "dish_inner_rim_water_side_edge",
            "boundary": _canonical_copy(physical),
        }
    elif candidate_kind == PALETTE_CANDIDATE_KIND:
        if not isinstance(observed, Mapping) or physical is not None:
            raise ValueError("Palette candidate boundary semantics are invalid.")
        boundary_observation = {
            "role": "independent_recording_image_observation",
            "observed_feature": observed.get("observed_feature"),
            "boundary": _canonical_copy(observed),
        }
    else:  # pragma: no cover - candidate validator already rejects this
        raise ValueError(f"Unsupported candidate kind: {candidate_kind!r}.")
    return {
        "run_name": name,
        "candidate_id": name,
        "candidate_kind": candidate_kind,
        "candidate_record_sha256": digest,
        "arena_binding": _canonical_copy(record["arena_binding"]),
        "coordinate_binding": _canonical_copy(record["coordinate_binding"]),
        "physical_inner_rim": (
            _canonical_copy(physical) if isinstance(physical, Mapping) else None
        ),
        "observed_boundary": (
            _canonical_copy(observed) if isinstance(observed, Mapping) else None
        ),
        "boundary_observation": boundary_observation,
        "valid_detection_region": _canonical_copy(record["valid_detection_region"]),
    }


def _comparison_snapshot(root: Any, comparison_run: str) -> dict[str, Any]:
    name = _safe_name(comparison_run, label="comparison_run")
    path = f"analysis/{COMPARISON_RUNS_PARENT}/{name}"
    try:
        group = root[path]
    except KeyError as exc:
        raise ValueError(f"Arena-geometry comparison is missing: {path}") from exc
    attrs = dict(group.attrs)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("candidate_selected") is not False
    ):
        raise ValueError(f"Comparison {name!r} is not complete immutable evidence.")
    record = attrs.get("comparison_record")
    if not isinstance(record, Mapping):
        raise ValueError(f"Comparison {name!r} lacks comparison_record.")
    validate_arena_geometry_comparison_record(record)
    digest = _payload_sha256(record)
    if attrs.get("comparison_record_sha256") != digest:
        raise ValueError(f"Comparison {name!r} record digest is invalid.")
    return {
        "run_name": name,
        "comparison_record_sha256": digest,
        "policy_id": record["policy"]["policy_id"],
        "policy_version": record["policy"]["policy_version"],
        "automatic_selection_promoted": record["policy"][
            "automatic_selection_promoted"
        ],
        "evidence_outcome": record["decision"]["evidence_outcome"],
        "workflow_action": record["decision"]["workflow_action"],
        "semantic_compatibility": record["observed_features"]["semantic_compatibility"],
        "candidate_bindings": _canonical_copy(record["candidate_bindings"]),
    }


def build_arena_geometry_selection_plan(
    source_zarr: str | Path,
    *,
    candidate_run: str,
    selected_by: str,
    decision_reason: str,
    decision_source: str = "manual_review",
    comparison_run: str | None = None,
    comparison_binding: Mapping[str, Any] | None = None,
) -> ArenaGeometrySelectionPlan:
    """Bind one exact candidate to an immutable operational selection."""

    zarr_path = Path(source_zarr).expanduser().resolve()
    root = open_zarr_root(zarr_path, mode="r")
    candidate = _candidate_snapshot(root, candidate_run)
    reviewer = str(selected_by).strip()
    reason = str(decision_reason).strip()
    source = str(decision_source).strip()
    if not reviewer or not reason or not source:
        raise ValueError(
            "Selection requires selected_by, decision_reason, and decision_source."
        )
    if comparison_run is not None and comparison_binding is not None:
        raise ValueError(
            "Use comparison_run for validated comparison evidence; do not also pass "
            "an unvalidated comparison_binding."
        )
    comparison = (
        _comparison_snapshot(root, comparison_run)
        if comparison_run is not None
        else None
    )
    if comparison is not None:
        compared_candidates = {
            binding["candidate_id"]
            for binding in comparison["candidate_bindings"].values()
        }
        if candidate["candidate_id"] not in compared_candidates:
            raise ValueError(
                "Selected candidate is not bound by the comparison artifact."
            )
        if source == "automatic_policy":
            if (
                comparison["evidence_outcome"] != "corroborated_pass"
                or comparison["workflow_action"] != "automatic_select_acquisition"
                or comparison["automatic_selection_promoted"] is not True
                or candidate["candidate_kind"] != ACQUISITION_CANDIDATE_KIND
            ):
                raise ValueError(
                    "Automatic selection requires a promoted corroborated acquisition pass."
                )
        elif source == "manual_review":
            if comparison["workflow_action"] == "fail":
                raise ValueError(
                    "A failed geometry comparison cannot be overridden by selecting one "
                    "of its existing candidates. Publish corrected evidence and compare "
                    "again."
                )
        else:
            raise ValueError(
                "Comparison-bound selection decision_source must be manual_review or "
                "automatic_policy."
            )
    elif source == "automatic_policy":
        raise ValueError("Automatic geometry selection requires a comparison artifact.")
    if comparison is not None:
        schema_version = SELECTION_RECORD_SCHEMA_VERSION
        selection_policy = COMPARISON_BOUND_SELECTION_POLICY
    elif candidate["candidate_kind"] == PALETTE_CANDIDATE_KIND:
        if source != "manual_review":
            raise ValueError(
                "A comparison-free Palette candidate requires explicit manual review."
            )
        schema_version = MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION
        selection_policy = MANUAL_PALETTE_SELECTION_POLICY
    else:
        schema_version = LEGACY_SELECTION_RECORD_SCHEMA_VERSION
        selection_policy = SELECTION_POLICY
    record = {
        "schema_id": SELECTION_RECORD_SCHEMA_ID,
        "schema_version": schema_version,
        "selection_policy": selection_policy,
        "selected_candidate": candidate,
        "decision": {
            "selected_by": reviewer,
            "decision_reason": reason,
            "decision_source": source,
            "comparison_binding": (
                _canonical_copy(comparison)
                if comparison is not None
                else (
                    _canonical_copy(comparison_binding)
                    if comparison_binding is not None
                    else None
                )
            ),
        },
        "operational_role": "bounding_box_centroid_detection_gating",
        "gate_semantics": {
            "method": "native_camera_circle_signed_distance_inclusive_v1",
            "signed_distance": "radius_px_minus_euclidean_radial_distance_px",
            "inside_predicate": "signed_distance_px_greater_than_or_equal_to_zero",
            "additional_palette_tolerance_px": 0.0,
        },
        "candidate_mutated": False,
        "legacy_dish_mask_projection_written": False,
    }
    normalized = _canonical_copy(record)
    validate_arena_geometry_selection_record(normalized)
    digest = _payload_sha256(normalized)
    selection_id = f"arena_geometry_selection_{digest[:20]}"
    provenance = build_writer_run_provenance(
        command="fisheye.utils.publish_arena_geometry_selection",
        params={
            "selection_id": selection_id,
            "candidate_run": candidate["run_name"],
            "candidate_record_sha256": candidate["candidate_record_sha256"],
            "decision_source": source,
            "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        },
        input_run_ids={
            "arena_geometry_candidate": candidate["run_name"],
            "candidate_record_sha256": candidate["candidate_record_sha256"],
            **(
                {
                    "arena_geometry_comparison": comparison["run_name"],
                    "comparison_record_sha256": comparison["comparison_record_sha256"],
                }
                if comparison is not None
                else {}
            ),
        },
        cwd=Path.cwd(),
        include_system_context=False,
    )
    return ArenaGeometrySelectionPlan(
        source_zarr=zarr_path,
        candidate_run=candidate["run_name"],
        candidate_record_sha256=candidate["candidate_record_sha256"],
        selection_id=selection_id,
        selection_record_sha256=digest,
        selection_record=normalized,
        target_run_path=(zarr_path / "analysis" / SELECTION_RUNS_PARENT / selection_id),
        run_provenance=provenance,
    )


def _selection_attrs(plan: ArenaGeometrySelectionPlan) -> dict[str, Any]:
    return {
        "schema_id": SELECTION_RUN_SCHEMA_ID,
        "schema_version": SELECTION_RUN_SCHEMA_VERSION,
        "selection_id": plan.selection_id,
        "selection_record": _canonical_copy(plan.selection_record),
        "selection_record_sha256": plan.selection_record_sha256,
        "selected_candidate_run": plan.candidate_run,
        "selected_candidate_record_sha256": plan.candidate_record_sha256,
        "run_provenance": _canonical_copy(plan.run_provenance),
        "operational_selection_status": "selected",
        "detection_gate_applied": False,
        "legacy_dish_mask_projection_written": False,
    }


def validate_arena_geometry_selection_run(
    run_path: str | Path,
    *,
    expected_plan: ArenaGeometrySelectionPlan,
    require_complete: bool = False,
    require_eligible: bool | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    path = Path(run_path).expanduser().resolve()
    try:
        group = open_zarr_root(path, mode="r")
        attrs = dict(group.attrs)
        for name, value in _selection_attrs(expected_plan).items():
            if name != "run_provenance" and attrs.get(name) != value:
                errors.append(f"{name} mismatch")
        record = attrs.get("selection_record")
        if isinstance(record, Mapping):
            validate_arena_geometry_selection_record(record)
            if _payload_sha256(record) != attrs.get("selection_record_sha256"):
                errors.append("selection record digest mismatch")
        else:
            errors.append("selection_record missing")
        provenance = validate_run_provenance(attrs.get("run_provenance"))
        if not provenance.valid:
            errors.extend(f"run provenance: {value}" for value in provenance.errors)
        if list(group.array_keys()) or list(group.group_keys()):
            errors.append("selection run must be metadata-only")
        status = attrs.get("palette_run_completion_status")
        if require_complete and status != "complete":
            errors.append("selection run is not complete")
        elif status not in {"running", "complete"}:
            errors.append("selection run has invalid completion status")
        if (
            require_eligible is not None
            and attrs.get("stage_selector_eligible") is not require_eligible
        ):
            errors.append("selection selector eligibility mismatch")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "selection_id": expected_plan.selection_id,
        "selection_record_sha256": expected_plan.selection_record_sha256,
        "run_path": str(path),
    }


def _revalidate_selection_source(plan: ArenaGeometrySelectionPlan) -> dict[str, Any]:
    root = open_zarr_root(plan.source_zarr, mode="r")
    candidate = _candidate_snapshot(root, plan.candidate_run)
    if candidate["candidate_record_sha256"] != plan.candidate_record_sha256:
        raise RuntimeError("Selected candidate changed during publication.")
    persisted = plan.selection_record["selected_candidate"]
    if candidate != persisted:
        raise RuntimeError("Selected candidate binding changed during publication.")
    comparison_binding = plan.selection_record["decision"].get("comparison_binding")
    comparison_audit: Mapping[str, Any] | None = None
    if plan.selection_record["schema_version"] == SELECTION_RECORD_SCHEMA_VERSION:
        if not isinstance(comparison_binding, Mapping):
            raise RuntimeError(
                "Comparison-bound selection lost its comparison binding."
            )
        current = _comparison_snapshot(root, str(comparison_binding["run_name"]))
        if current != comparison_binding:
            raise RuntimeError(
                "Comparison binding changed during selection publication."
            )
        comparison_audit = {
            "comparison_run": current["run_name"],
            "comparison_record_sha256": current["comparison_record_sha256"],
        }
    return {
        "status": "current",
        "candidate_run": plan.candidate_run,
        "candidate_record_sha256": plan.candidate_record_sha256,
        "comparison": comparison_audit,
    }


def publish_arena_geometry_selection(
    plan: ArenaGeometrySelectionPlan,
    *,
    scratch_root: str | Path,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Atomically publish and activate one immutable selection record."""

    if plan.target_run_path.exists():
        existing = validate_arena_geometry_selection_run(
            plan.target_run_path,
            expected_plan=plan,
            require_complete=True,
            require_eligible=True,
        )
        if not existing["valid"]:
            raise FileExistsError(
                f"Existing selection is not the expected run: {existing}"
            )
        return {"published": False, "status": "already_complete", **existing}

    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"palette-{plan.selection_id}-", dir=scratch
    ) as temporary:
        local_run = Path(temporary) / plan.selection_id
        local = zarr.open_group(str(local_run), mode="w", zarr_format=3)
        local.attrs.update(json_attr_safe(_selection_attrs(plan)))
        mark_run_started(
            local,
            run_name=plan.selection_id,
            stage="arena_geometry_selection",
        )

        def validate(path: Path) -> dict[str, Any]:
            return validate_arena_geometry_selection_run(path, expected_plan=plan)

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, SELECTION_RUNS_PARENT),)

        def complete(
            _root: zarr.Group, parent: zarr.Group, run_group: zarr.Group
        ) -> None:
            mark_run_complete(
                run_group,
                parent_group=parent,
                run_name=plan.selection_id,
                run_provenance=plan.run_provenance,
            )

        def verify(root: zarr.Group) -> None:
            parent = root[f"analysis/{SELECTION_RUNS_PARENT}"]
            run = parent[plan.selection_id]
            if (
                run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
                or parent.attrs.get("latest") == plan.selection_id
                or parent.attrs.get("latest_complete") == plan.selection_id
            ):
                raise RuntimeError("Selection became visible before activation.")

        def activate(
            root: zarr.Group, parent: zarr.Group, run_group: zarr.Group
        ) -> None:
            def proof() -> tuple[Any, ...]:
                current = root[f"analysis/{SELECTION_RUNS_PARENT}/{plan.selection_id}"]
                check = validate_arena_geometry_selection_run(
                    plan.target_run_path,
                    expected_plan=plan,
                    require_complete=True,
                    require_eligible=False,
                )
                if not check["valid"]:
                    raise RuntimeError(f"Selection activation proof failed: {check}")
                return (
                    current.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR),
                    current.attrs.get("palette_run_completion_status"),
                    current.attrs.get("selection_record_sha256"),
                    current.attrs.get("selected_candidate_record_sha256"),
                )

            try:
                activate_selector_eligible_run(
                    root,
                    parent,
                    run_group,
                    parent_path=f"analysis/{SELECTION_RUNS_PARENT}",
                    run_path=(f"analysis/{SELECTION_RUNS_PARENT}/{plan.selection_id}"),
                    run_name=plan.selection_id,
                    owner_attr=ATOMIC_PUBLICATION_OWNER_ATTR,
                    expected_owner_uuid=str(
                        run_group.attrs[ATOMIC_PUBLICATION_OWNER_ATTR]
                    ),
                    policy_attr=SELECTION_POLICY_ATTR,
                    generation_attr=SELECTION_GENERATION_ATTR,
                    lease_attr=SELECTION_LEASE_ATTR,
                    policy=SELECTION_POLICY,
                    lease_schema_id="palette.arena_geometry_selection_lease",
                    proof_loader=proof,
                )
            except SelectorActivationError as exc:
                raise RuntimeError(
                    f"Geometry selection activation failed: {exc}"
                ) from exc

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=local_run,
                target_run_path=plan.target_run_path,
                run_name=plan.selection_id,
                lock_suffix="arena-geometry-selection-publish",
                publish_schema_id=SELECTION_PUBLISH_SCHEMA_ID,
                policy="node_local_metadata_selection_atomic_publish_v1",
                rollback_policy="retain_failed_public_tombstone_restore_owned_selector_epoch",
                content_checksum=True,
                selector_owner_attr=SELECTION_LEASE_ATTR,
                selector_generation_attr=SELECTION_GENERATION_ATTR,
                owned_parent_attr_names=(
                    (
                        "latest_complete",
                        "latest",
                        SELECTION_POLICY_ATTR,
                        SELECTION_GENERATION_ATTR,
                        SELECTION_LEASE_ATTR,
                    ),
                ),
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            after_rename=lambda _root, _run, _physical_copy: {
                "source_revision_audit": _revalidate_selection_source(plan)
            },
            activate_run=activate,
            payload_metadata={
                "algorithm_version": SELECTION_ALGORITHM_VERSION,
                "selection_id": plan.selection_id,
                "candidate_run": plan.candidate_run,
                "candidate_record_sha256": plan.candidate_record_sha256,
            },
        )

    final = validate_arena_geometry_selection_run(
        plan.target_run_path,
        expected_plan=plan,
        require_complete=True,
        require_eligible=True,
    )
    if not final["valid"]:
        raise RuntimeError(f"Final selection validation failed: {final}")
    return {"published": True, "status": "complete", **publication, **final}


__all__ = [
    "ArenaGeometrySelectionPlan",
    "MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION",
    "SELECTION_RUNS_PARENT",
    "build_arena_geometry_selection_plan",
    "publish_arena_geometry_selection",
    "validate_arena_geometry_selection_record",
    "validate_arena_geometry_selection_run",
]
