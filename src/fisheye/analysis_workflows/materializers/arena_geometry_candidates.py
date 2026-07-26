"""Immutable acquisition-derived arena-geometry candidate publication.

Candidates preserve geometry and lineage for later comparison and selection.
Publishing a candidate never makes it an operational dish mask and never
updates the legacy ``analysis_metadata.attrs['dish_mask']`` projection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import zarr

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
)
from fisheye.shared.recording_geometry import (
    BoundRegisteredDishMask,
    RecordingGeometryError,
    RegisteredDishMask,
    bind_registered_dish_mask_to_source_camera_frame,
)
from fisheye.shared.recording_geometry_recovery import (
    RECOVERY_AUTHORITY,
    RECOVERY_REASON,
    VerifiedRecordingGeometryRecovery,
    registered_dish_mask_from_verified_recovery,
    validate_recording_geometry_recovery_receipt,
)
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


CANDIDATE_RECORD_SCHEMA_ID = "palette.arena_geometry_candidate_record"
CANDIDATE_RECORD_SCHEMA_VERSION = 1
CANDIDATE_RUN_SCHEMA_ID = "palette.arena_geometry_candidate_run"
CANDIDATE_RUN_SCHEMA_VERSION = 1
CANDIDATE_KIND = "acquisition_registered_dish"
CANDIDATE_RUNS_PARENT = "arena_geometry_runs"
PUBLISH_SCHEMA_ID = "palette.arena_geometry_candidate_publish"
PUBLISH_ALGORITHM_VERSION = 1
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArenaGeometryCandidatePlan:
    source_zarr: Path
    receipt_path: Path
    receipt_sha256: str
    candidate_id: str
    candidate_record_sha256: str
    candidate_record: Mapping[str, Any]
    run_name: str
    target_run_path: Path
    run_provenance: Mapping[str, Any]


def _canonical_copy(value: Any) -> Any:
    return json.loads(strict_json_dumps(value))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordingGeometryError(f"{label} must be a mapping.")
    return value


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordingGeometryError(f"{label} must be a nonempty string.")
    return value.strip()


def _required_sha256(value: Any, *, label: str) -> str:
    text = _required_text(value, label=label).lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise RecordingGeometryError(f"{label} must be a SHA-256 digest.")
    return text


def _required_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingGeometryError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise RecordingGeometryError(f"{label} must be finite.")
    return result


def _circle_record(circle: Any) -> dict[str, Any]:
    return {
        "type": "circle",
        "center_px": {
            "x": float(circle.center_x_native_px),
            "y": float(circle.center_y_native_px),
        },
        "radius_px": float(circle.radius_px),
    }


def build_acquisition_geometry_candidate_record(
    bound_mask: BoundRegisteredDishMask,
    *,
    recovery_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize one proven acquisition mask into candidate-only semantics."""

    if type(bound_mask) is not BoundRegisteredDishMask:
        raise RecordingGeometryError("A bound registered dish mask is required.")
    mask = bound_mask.mask
    record = {
        "schema_id": CANDIDATE_RECORD_SCHEMA_ID,
        "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "candidate_kind": CANDIDATE_KIND,
        "arena_binding": {
            "rig_id": mask.key.rig_id,
            "canvas_name": mask.key.canvas_name,
            "arena_id": mask.key.arena_id,
            "camera_serial": mask.key.camera_serial,
        },
        "physical_inner_rim": {
            "coordinate_space": mask.coordinate_space,
            "target_plane": mask.target_plane,
            "geometry": _circle_record(mask.physical_inner_rim),
        },
        "valid_detection_region": {
            "coordinate_space": mask.coordinate_space,
            "purpose": "bounding_box_centroid_detection_gating",
            "offset_direction": "outward",
            "geometry": _circle_record(mask.valid_detection_gate),
            "is_final_acquisition_tolerance": True,
            "additional_palette_tolerance_px": 0.0,
        },
        "coordinate_binding": {
            "space_id": mask.palette_space_id,
            "profile_id": mask.coordinate_profile_id,
            "pixel_convention": mask.pixel_convention,
            "units": "px",
            "origin": mask.origin,
            "positive_x": mask.positive_x,
            "positive_y": mask.positive_y,
            "native_width_px": mask.native_width_px,
            "native_height_px": mask.native_height_px,
            "pixel_frame_record_ref": bound_mask.pixel_frame_record_ref,
            "pixel_frame_record_sha256": bound_mask.pixel_frame_record_sha256,
        },
        "acquisition_source": {
            "source_kind": mask.source_kind,
            "artifact_id": mask.artifact_id,
            "source_observation_sha256": mask.source_observation_sha256,
            "registration_id": mask.registration_id,
            "registration_sha256": mask.registration_sha256,
            "source_contract_sha256": mask.source_contract_sha256,
            "materialized_asset_status": mask.materialized_asset_status.value,
            "citrus_registration_status": mask.citrus_registration_status.value,
            "selected_daily_registration_applied_by_citrus": (
                mask.selected_daily_registration_applied_by_citrus
            ),
            "producer_contract_linkage_status": (
                mask.producer_contract_linkage_status
            ),
            "source_valid_until_utc": mask.source_valid_until_utc,
            "producer_operator_accepted": mask.producer_operator_accepted,
            "producer_quality_flags": list(mask.producer_quality_flags),
            "recovery_binding": (
                _canonical_copy(recovery_binding)
                if recovery_binding is not None
                else None
            ),
        },
        "candidate_policy": {
            "publication_role": "candidate_only",
            "operationally_selected": False,
            "legacy_dish_mask_projection_written": False,
            "detection_gate_applied": False,
            "independent_palette_fit_required_before_operational_use": True,
        },
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    normalized = _canonical_copy(record)
    validate_acquisition_geometry_candidate_record(normalized)
    return normalized


def validate_acquisition_geometry_candidate_record(record: Mapping[str, Any]) -> None:
    """Validate scientific and coordinate invariants of one candidate record."""

    if record.get("schema_id") != CANDIDATE_RECORD_SCHEMA_ID or record.get(
        "schema_version"
    ) != CANDIDATE_RECORD_SCHEMA_VERSION:
        raise RecordingGeometryError("Unsupported arena-geometry candidate schema.")
    if record.get("candidate_kind") != CANDIDATE_KIND:
        raise RecordingGeometryError("Unsupported arena-geometry candidate kind.")
    arena = _required_mapping(record.get("arena_binding"), label="arena_binding")
    for name in ("rig_id", "canvas_name", "arena_id", "camera_serial"):
        _required_text(arena.get(name), label=f"arena_binding.{name}")

    physical = _required_mapping(record.get("physical_inner_rim"), label="physical_inner_rim")
    gate = _required_mapping(
        record.get("valid_detection_region"),
        label="valid_detection_region",
    )
    if physical.get("coordinate_space") != "camera_native_pixels" or gate.get(
        "coordinate_space"
    ) != "camera_native_pixels":
        raise RecordingGeometryError("Candidate circles must use native camera pixels.")
    if physical.get("target_plane") != "dish_top_rim":
        raise RecordingGeometryError("Physical rim must target dish_top_rim.")
    if gate.get("purpose") != "bounding_box_centroid_detection_gating" or gate.get(
        "offset_direction"
    ) != "outward":
        raise RecordingGeometryError("Detection gate semantics are invalid.")
    if gate.get("is_final_acquisition_tolerance") is not True or gate.get(
        "additional_palette_tolerance_px"
    ) != 0.0:
        raise RecordingGeometryError("Acquisition gate must not receive added Palette tolerance.")

    def circle(container: Mapping[str, Any], label: str) -> tuple[float, float, float]:
        geometry = _required_mapping(container.get("geometry"), label=f"{label}.geometry")
        if geometry.get("type") != "circle":
            raise RecordingGeometryError(f"{label} must be circular.")
        center = _required_mapping(geometry.get("center_px"), label=f"{label}.center_px")
        x = _required_finite(center.get("x"), label=f"{label}.center.x")
        y = _required_finite(center.get("y"), label=f"{label}.center.y")
        radius = _required_finite(geometry.get("radius_px"), label=f"{label}.radius")
        if radius <= 0:
            raise RecordingGeometryError(f"{label} radius must be positive.")
        return x, y, radius

    physical_x, physical_y, physical_radius = circle(physical, "physical_inner_rim")
    gate_x, gate_y, gate_radius = circle(gate, "valid_detection_region")
    if not math.isclose(physical_x, gate_x, abs_tol=1e-6) or not math.isclose(
        physical_y, gate_y, abs_tol=1e-6
    ):
        raise RecordingGeometryError("Physical rim and valid gate must be concentric.")
    if gate_radius < physical_radius:
        raise RecordingGeometryError("Valid gate cannot be smaller than the physical rim.")

    coordinate = _required_mapping(record.get("coordinate_binding"), label="coordinate_binding")
    if (
        coordinate.get("space_id") != "source_camera_image_px"
        or coordinate.get("profile_id") != "source_camera_image_px.top_left_y_down.v1"
        or coordinate.get("pixel_convention") != "continuous"
        or coordinate.get("units") != "px"
        or coordinate.get("origin") != "top_left"
        or coordinate.get("positive_x") != "right"
        or coordinate.get("positive_y") != "down"
    ):
        raise RecordingGeometryError("Candidate source-camera coordinate binding is invalid.")
    for name in ("native_width_px", "native_height_px"):
        value = coordinate.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RecordingGeometryError(f"coordinate_binding.{name} must be positive.")
    expected_ref = (
        f"/analysis/coordinate_frames/source_camera/{arena['camera_serial']}"
        "/continuous@pixel_frame_authority"
    )
    if coordinate.get("pixel_frame_record_ref") != expected_ref:
        raise RecordingGeometryError("Candidate does not bind the canonical continuous frame.")
    _required_sha256(
        coordinate.get("pixel_frame_record_sha256"),
        label="pixel_frame_record_sha256",
    )

    source = _required_mapping(record.get("acquisition_source"), label="acquisition_source")
    for name in ("source_kind", "artifact_id", "registration_id"):
        _required_text(source.get(name), label=f"acquisition_source.{name}")
    for name in ("source_observation_sha256", "source_contract_sha256"):
        _required_sha256(source.get(name), label=f"acquisition_source.{name}")
    if source.get("registration_sha256") is not None:
        _required_sha256(
            source.get("registration_sha256"),
            label="acquisition_source.registration_sha256",
        )
    recovery = source.get("recovery_binding")
    if recovery is not None:
        recovery_map = _required_mapping(recovery, label="recovery_binding")
        _required_sha256(recovery_map.get("receipt_sha256"), label="receipt_sha256")
        _required_sha256(recovery_map.get("target_h5_sha256"), label="target_h5_sha256")
        if recovery_map.get("authority") != RECOVERY_AUTHORITY or recovery_map.get(
            "reason"
        ) != RECOVERY_REASON:
            raise RecordingGeometryError("Recovery binding authority is invalid.")

    policy = _required_mapping(record.get("candidate_policy"), label="candidate_policy")
    expected_policy = {
        "publication_role": "candidate_only",
        "operationally_selected": False,
        "legacy_dish_mask_projection_written": False,
        "detection_gate_applied": False,
        "independent_palette_fit_required_before_operational_use": True,
    }
    if dict(policy) != expected_policy:
        raise RecordingGeometryError("Acquisition candidate policy is not fail-closed.")
    if record.get("canonicalization") != "canonical_json_sort_keys_v1":
        raise RecordingGeometryError("Candidate canonicalization is unsupported.")
    if _canonical_copy(record) != dict(record):
        raise RecordingGeometryError("Candidate record is not strict canonical JSON data.")


def _recovery_binding(verified: VerifiedRecordingGeometryRecovery) -> dict[str, Any]:
    target = _required_mapping(verified.receipt.get("target"), label="receipt.target")
    return {
        "receipt_schema_id": verified.receipt.get("schema_id"),
        "receipt_id": verified.receipt.get("receipt_id"),
        "receipt_sha256": verified.receipt_sha256,
        "authority": verified.receipt.get("authority"),
        "reason": verified.receipt.get("recovery_reason"),
        "target_h5_sha256": target.get("h5_sha256"),
        "target_session_uuid": target.get("session_uuid"),
        "h5_geometry_capture_status": target.get("h5_geometry_capture_status"),
        "producer_artifacts_mutated": False,
    }


def _bound_recovered_mask(
    source_zarr: Path,
    verified: VerifiedRecordingGeometryRecovery,
) -> BoundRegisteredDishMask:
    receipt_recording_root = verified.receipt_path.parent.parent
    zarr_recording_root = source_zarr.parent.parent
    if receipt_recording_root != zarr_recording_root:
        raise RecordingGeometryError(
            "Recovery receipt and analysis Zarr must be siblings in one recording root."
        )
    mask: RegisteredDishMask = registered_dish_mask_from_verified_recovery(verified)
    root = open_zarr_root(source_zarr, mode="r")
    _ownership, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=mask.key.camera_serial,
    )
    frame_path = (
        f"analysis/coordinate_frames/source_camera/{mask.key.camera_serial}/continuous"
    )
    try:
        frame_node = root[frame_path]
    except KeyError as exc:
        raise RecordingGeometryError(
            f"Analysis Zarr lacks canonical continuous source-camera authority {frame_path}."
        ) from exc
    source_frame = load_source_camera_pixel_frame_authority(
        frame_node,
        acquisition_frame=acquisition,
    )
    return bind_registered_dish_mask_to_source_camera_frame(mask, source_frame)


def _record_from_receipt_and_zarr(
    source_zarr: Path,
    receipt_path: Path,
) -> tuple[VerifiedRecordingGeometryRecovery, dict[str, Any], str]:
    verified = validate_recording_geometry_recovery_receipt(receipt_path)
    bound = _bound_recovered_mask(source_zarr, verified)
    record = build_acquisition_geometry_candidate_record(
        bound,
        recovery_binding=_recovery_binding(verified),
    )
    return verified, record, _payload_sha256(record)


def plan_recovered_acquisition_geometry_candidate(
    *,
    source_zarr: str | Path,
    receipt_path: str | Path,
) -> ArenaGeometryCandidatePlan:
    """Plan one deterministic candidate without writing the analysis Zarr."""

    zarr_path = Path(source_zarr).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    verified, record, digest = _record_from_receipt_and_zarr(zarr_path, receipt)
    candidate_id = f"arena-geometry-acquisition-{digest[:24]}"
    run_name = candidate_id
    params = {
        "algorithm_version": PUBLISH_ALGORITHM_VERSION,
        "candidate_id": candidate_id,
        "candidate_record_sha256": digest,
        "candidate_kind": CANDIDATE_KIND,
        "operational_selection": "not_performed",
    }
    provenance = build_writer_run_provenance(
        command="publish_acquisition_geometry_candidate",
        params=params,
        input_run_ids={},
        input_artifacts=(
            {
                "role": "recording_geometry_recovery_receipt",
                "path": str(receipt),
                "sha256": verified.receipt_sha256,
            },
            {
                "role": "orange_recording_geometry_contract",
                "path": str(verified.evidence.bundle_root / "recording_geometry_contract.json"),
                "sha256": verified.evidence.bundle_verification.contract_sha256,
            },
        ),
        include_system_context=False,
    )
    provenance_validation = validate_run_provenance(provenance)
    if not provenance_validation.valid:
        raise RuntimeError(
            "Acquisition candidate publication provenance is invalid: "
            f"{provenance_validation.errors}"
        )
    return ArenaGeometryCandidatePlan(
        source_zarr=zarr_path,
        receipt_path=receipt,
        receipt_sha256=verified.receipt_sha256,
        candidate_id=candidate_id,
        candidate_record_sha256=digest,
        candidate_record=record,
        run_name=run_name,
        target_run_path=zarr_path / "analysis" / CANDIDATE_RUNS_PARENT / run_name,
        run_provenance=provenance,
    )


def _candidate_attrs(plan: ArenaGeometryCandidatePlan) -> dict[str, Any]:
    return {
        "schema_id": CANDIDATE_RUN_SCHEMA_ID,
        "schema_version": CANDIDATE_RUN_SCHEMA_VERSION,
        "candidate_id": plan.candidate_id,
        "candidate_kind": CANDIDATE_KIND,
        "candidate_record": _canonical_copy(plan.candidate_record),
        "candidate_record_sha256": plan.candidate_record_sha256,
        "run_provenance": _canonical_copy(plan.run_provenance),
        "operational_selection_status": "not_selected",
        "legacy_dish_mask_projection_written": False,
        "detection_gate_applied": False,
    }


def validate_arena_geometry_candidate_run(
    run_path: str | Path,
    *,
    expected_plan: ArenaGeometryCandidatePlan,
    require_complete: bool = False,
    require_eligible: bool | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    path = Path(run_path).expanduser().resolve()
    try:
        group = open_zarr_root(path, mode="r")
        attrs = group.attrs
        expected = _candidate_attrs(expected_plan)
        for name, value in expected.items():
            if name == "run_provenance":
                continue
            if attrs.get(name) != value:
                errors.append(f"{name} mismatch")
        record = attrs.get("candidate_record")
        if isinstance(record, Mapping):
            try:
                validate_acquisition_geometry_candidate_record(record)
                if _payload_sha256(record) != attrs.get("candidate_record_sha256"):
                    errors.append("candidate record digest mismatch")
            except RecordingGeometryError as exc:
                errors.append(str(exc))
        else:
            errors.append("candidate_record missing")
        provenance = validate_run_provenance(attrs.get("run_provenance"))
        if not provenance.valid:
            errors.extend(f"run provenance: {item}" for item in provenance.errors)
        else:
            expected_provenance = expected_plan.run_provenance
            persisted_provenance = provenance.normalized or {}
            for name in (
                "command",
                "config_hash",
                "params",
                "input_run_ids",
                "input_artifacts",
            ):
                if persisted_provenance.get(name) != expected_provenance.get(name):
                    errors.append(f"run provenance {name} mismatch")
        if list(group.array_keys()) or list(group.group_keys()):
            errors.append("candidate run must be metadata-only")
        status = attrs.get("palette_run_completion_status")
        if require_complete and status != "complete":
            errors.append("candidate run is not complete")
        elif status not in {"running", "complete"}:
            errors.append("candidate run has invalid completion status")
        if require_eligible is not None and attrs.get("stage_selector_eligible") is not (
            require_eligible
        ):
            errors.append("candidate selector eligibility mismatch")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_id": expected_plan.candidate_id,
        "candidate_record_sha256": expected_plan.candidate_record_sha256,
        "run_path": str(path),
    }


def _materialize_local_run(plan: ArenaGeometryCandidatePlan, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing existing local candidate run: {path}")
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(json_attr_safe(_candidate_attrs(plan)))
    mark_run_started(
        group,
        run_name=plan.run_name,
        stage="arena_geometry_candidate",
    )
    validation = validate_arena_geometry_candidate_run(
        path,
        expected_plan=plan,
    )
    if not validation["valid"]:
        raise RuntimeError(f"Local acquisition candidate validation failed: {validation}")


def publish_arena_geometry_candidate(
    plan: ArenaGeometryCandidatePlan,
    *,
    scratch_root: str | Path,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Publish one metadata-only candidate atomically, without selecting it."""

    if plan.target_run_path.exists():
        existing = validate_arena_geometry_candidate_run(
            plan.target_run_path,
            expected_plan=plan,
            require_complete=True,
            require_eligible=True,
        )
        if not existing["valid"]:
            raise FileExistsError(
                f"Existing candidate path is not the expected immutable run: {existing}"
            )
        return {"published": False, "status": "already_complete", **existing}

    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"palette-{plan.run_name}-",
        dir=scratch,
    ) as temporary:
        local_run = Path(temporary) / plan.run_name
        _materialize_local_run(plan, local_run)

        def validate(path: Path) -> dict[str, Any]:
            return validate_arena_geometry_candidate_run(
                path,
                expected_plan=plan,
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, CANDIDATE_RUNS_PARENT),)

        def after_rename(_root: zarr.Group, _run: zarr.Group) -> dict[str, Any]:
            verified, current_record, current_digest = _record_from_receipt_and_zarr(
                plan.source_zarr,
                plan.receipt_path,
            )
            if (
                verified.receipt_sha256 != plan.receipt_sha256
                or current_digest != plan.candidate_record_sha256
                or current_record != plan.candidate_record
            ):
                raise RuntimeError("Acquisition geometry source changed during publication.")
            return {
                "source_revision_audit": {
                    "status": "current",
                    "receipt_sha256": verified.receipt_sha256,
                    "candidate_record_sha256": current_digest,
                }
            }

        def complete(
            _root: zarr.Group,
            parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            mark_run_complete(
                run_group,
                parent_group=parent,
                run_name=plan.run_name,
                run_provenance=plan.run_provenance,
            )

        def verify(root: zarr.Group) -> None:
            parent = root[f"analysis/{CANDIDATE_RUNS_PARENT}"]
            run_group = parent[plan.run_name]
            if (
                run_group.attrs.get("palette_run_completion_status") != "complete"
                or run_group.attrs.get("stage_selector_eligible") is not False
                or parent.attrs.get("latest") == plan.run_name
                or parent.attrs.get("latest_complete") == plan.run_name
            ):
                raise RuntimeError(
                    "Candidate must be complete and readable without becoming latest or selected."
                )

        def activate(
            _root: zarr.Group,
            _parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            if run_group.attrs.get("operational_selection_status") != "not_selected":
                raise RuntimeError("Candidate activation cannot perform operational selection.")
            run_group.attrs["stage_selector_eligible"] = True

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=local_run,
                target_run_path=plan.target_run_path,
                run_name=plan.run_name,
                lock_suffix="arena-geometry-candidate-publish",
                publish_schema_id=PUBLISH_SCHEMA_ID,
                policy="node_local_metadata_candidate_atomic_run_group_publish",
                rollback_policy=(
                    "retain_failed_public_tombstone_leave_parent_without_candidate_pointer"
                ),
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            activate_run=activate,
            after_rename=after_rename,
            payload_metadata={
                "algorithm_version": PUBLISH_ALGORITHM_VERSION,
                "candidate_id": plan.candidate_id,
                "candidate_record_sha256": plan.candidate_record_sha256,
                "selection_performed": False,
                "legacy_dish_mask_projection_written": False,
            },
        )

    final = validate_arena_geometry_candidate_run(
        plan.target_run_path,
        expected_plan=plan,
        require_complete=True,
        require_eligible=True,
    )
    if not final["valid"]:
        raise RuntimeError(f"Published acquisition candidate failed validation: {final}")
    return {
        "published": True,
        "status": "complete_candidate_not_selected",
        "publication": publication,
        **final,
    }


__all__ = [
    "ArenaGeometryCandidatePlan",
    "CANDIDATE_KIND",
    "CANDIDATE_RECORD_SCHEMA_ID",
    "CANDIDATE_RECORD_SCHEMA_VERSION",
    "CANDIDATE_RUNS_PARENT",
    "CANDIDATE_RUN_SCHEMA_ID",
    "CANDIDATE_RUN_SCHEMA_VERSION",
    "PUBLISH_ALGORITHM_VERSION",
    "PUBLISH_SCHEMA_ID",
    "build_acquisition_geometry_candidate_record",
    "plan_recovered_acquisition_geometry_candidate",
    "publish_arena_geometry_candidate",
    "validate_acquisition_geometry_candidate_record",
    "validate_arena_geometry_candidate_run",
]
