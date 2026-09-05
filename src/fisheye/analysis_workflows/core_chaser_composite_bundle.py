"""One core authority roster plus one exact-chaser extension bundle.

This module is an admission adapter for the existing ``validated_behavior/v1``
publisher.  It does not define a new scientific authority or publication
surface.  The bundle binds one completed core-workflow execution report and one
exact-chaser projection receipt, proves that every maintained chaser child
retains the selected core roster, and exposes a closed capability matrix to the
generic cohort engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CANONICAL_SWIM_BOUTS_CAPABILITY,
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CROSS_GRAIN_JOIN_AUTHORITY,
    KINEMATICS_SAMPLES_CAPABILITY,
    SUBJECT_BODY_FRAME_CAPABILITY,
)
from fisheye.analytics_exports.validated_behavior_core_chaser_contracts import (
    CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY,
    CORE_CHASER_EXPORT_PROFILE_ID,
    CORE_CHASER_TABLE_SPECS,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .core_authority_roster import validate_core_authority_roster
from .core_behavior_cohort_adapter import bind_core_behavior_cohort_sources
from .core_motion_source_handle import validate_core_motion_dependency_record
from .core_paradigm_authority import (
    core_paradigm_dependency_from_relative_manifest,
    validate_core_paradigm_dependency,
)
from .validated_behavior_cohort import (
    CAPABILITY_STATES,
    build_capability_contract,
    build_validated_behavior_bundle_set,
    validate_validated_behavior_bundle_set,
)
from .validated_behavior_cohort_adapters import (
    sha256_file,
    validate_membership_current_sources,
)
from .validated_behavior_source_admission import (
    CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
    EXACT_CHASER_ADMISSION_ROLE,
    validate_admission_receipt_binding,
)
from .validated_recording_behavior_bundle import (
    BASE_SCIENTIFIC_CHILD_KEYS,
    OPTIONAL_SCIENTIFIC_CHILD_KEYS,
    BoundExactChaserExtensionSources,
    bind_exact_chaser_extension_sources,
)

CORE_CHASER_BUNDLE_ADAPTER_ID = "core_behavior_plus_exact_chaser_v1"
CORE_CHASER_BUNDLE_SCHEMA_ID = "palette.analysis.core_chaser_composite_bundle"
CORE_CHASER_BUNDLE_SCHEMA_VERSION = 1
CORE_CHASER_BUNDLE_METHOD_ID = "one_core_roster_plus_exact_chaser_extension_v1"
CORE_CHASER_BUNDLE_STATUS = "complete_selector_ineligible_receipt_composition"
CORE_CHASER_CAPABILITY_PROFILE_ID = "core_behavior_plus_exact_chaser_sources_v1"
MAX_COMPOSITE_BUNDLE_BYTES = 2 * 1024 * 1024

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

_CHASER_EXTENSION_CAPABILITY_KEYS = (
    CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY,
    "semantic_epochs",
    "controller_trials",
    "generalized_bout_response",
    "escape_freeze",
    "epoch_behavior",
    "spatial_occupancy",
    "reviewed_arena_and_scale",
    "body_alignment_by_distance",
)
CORE_CHASER_CAPABILITY_KEYS = tuple(
    sorted({*CORE_BEHAVIOR_CAPABILITY_KEYS, *_CHASER_EXTENSION_CAPABILITY_KEYS})
)

_DIRECT_PARADIGM_CHILD_ROLES = MappingProxyType(
    {
        "controller": "keypoint",
        "keypoint_radial": "keypoint",
        "detection_radial": "detection",
        "body_alignment_by_distance": "keypoint",
        "gaze": "keypoint",
    }
)
_MOTION_CHILD_PATHS = MappingProxyType(
    {
        "bout": ("sources", "core_authority"),
        "escape": ("sources", "core_authority"),
        "epoch_behavior": ("sources", "core_motion"),
    }
)
_PUBLIC_CHILD_CAPABILITIES = MappingProxyType(
    {
        "semantic_epochs": "semantic_epochs",
        "controller_trials": "controller_trials",
        "generalized_bout_response": "generalized_bout_response",
        "escape_freeze": "escape_freeze",
        "epoch_behavior": "epoch_behavior",
        "spatial_occupancy": "spatial_occupancy",
        "body_alignment_by_distance": "body_alignment_by_distance",
    }
)
_INTERNAL_SOURCE_CAPABILITIES = (
    "fish_position_keypoint",
    "fish_position_detection",
    "anatomical_body_frame",
    "provider_motion",
    "canonical_swim_bouts",
    "semantic_epochs",
    "reviewed_arena_and_scale",
)
_SOURCE_BINDING_TYPES = MappingProxyType(
    {
        "fish_position_keypoint": "core_motion_on_exact_chaser_carrier_v1",
        "fish_position_detection": "core_motion_on_exact_chaser_carrier_v1",
        "chaser_observations_keypoint_projection": (
            "relative_frame_source_authority_v1"
        ),
        "chaser_observations_detection_projection": (
            "relative_frame_source_authority_v1"
        ),
        "anatomical_body_frame": "selected_core_subject_body_frame_v1",
        "row_axis_timing_and_scale": "paired_relative_frame_consensus_v1",
        "provider_motion": "selected_core_motion_authority_v1",
        "canonical_swim_bouts": "selected_core_swim_bout_authority_v1",
        "semantic_epochs": "exact_protocol_semantic_selection_v1",
        "reviewed_arena_and_scale": "spatial_radial_consensus_v1",
    }
)
_ALLOWED_SCIENTIFIC_CHILD_KEYS = frozenset(
    {*BASE_SCIENTIFIC_CHILD_KEYS, *OPTIONAL_SCIENTIFIC_CHILD_KEYS}
)

VALIDATION_POLICY = MappingProxyType(
    {
        "core_selection": "one_exact_execution_report_resolved_by_shared_core_binder",
        "extension_selection": "one_exact_chaser_projection_receipt",
        "extension_lineage": "every_maintained_child_names_selected_core_roster",
        "selector_resolution": False,
        "source_array_copy": False,
        "source_array_validation": "owned_by_strict_source_handles",
        "profile_composition": "collision_checked_core_plus_additive_chaser",
        "fallback": "prohibited",
    }
)
SAFETY = MappingProxyType(
    {
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "selector_activation": False,
        "source_mutation": False,
        "zarr_mutation": False,
    }
)


class CoreChaserCompositeBundleError(ValueError):
    """A proposed composite has missing or competing authority evidence."""


def _fail(message: str) -> None:
    raise CoreChaserCompositeBundleError(message)


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


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _DIGEST_RE.fullmatch(result) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _plain(body)
    return {**normalized, "record_sha256": canonical_json_sha256(normalized)}


def _validate_record_seal(value: object, *, field: str) -> dict[str, Any]:
    record = _plain(_mapping(value, field=field))
    persisted = _digest(
        record.pop("record_sha256", None), field=f"{field}.record_sha256"
    )
    if canonical_json_sha256(record) != persisted:
        _fail(f"{field} digest is stale.")
    return {**record, "record_sha256": persisted}


def _capability_reason_codes() -> dict[str, tuple[str | None, ...]]:
    result: dict[str, tuple[str | None, ...]] = {
        "complete": (None,),
        "inapplicable": ("member_not_admitted",),
        "invalid": ("invalid_source_authority", "core_authority_conflict"),
        "review_required": ("source_review_required",),
        "stale": ("source_stale",),
        "unavailable": (
            "blocked_by_invalid_membership",
            "blocked_by_unavailable_membership",
            "missing_exact_receipt",
        ),
    }
    if set(result) != set(CAPABILITY_STATES):  # pragma: no cover - constant guard
        _fail("Composite capability-state vocabulary is incomplete.")
    return result


def core_chaser_capability_contract() -> dict[str, Any]:
    """Return the closed public capability vocabulary for the profile."""

    expected_from_tables = {
        str(spec.required_capability)
        for spec in CORE_CHASER_TABLE_SPECS.values()
        if spec.required_capability is not None
    }
    if expected_from_tables | {CROSS_GRAIN_JOIN_AUTHORITY} != set(
        CORE_CHASER_CAPABILITY_KEYS
    ):
        _fail("Composite capability vocabulary differs from its table contracts.")
    return build_capability_contract(
        profile_id=CORE_CHASER_CAPABILITY_PROFILE_ID,
        keys=CORE_CHASER_CAPABILITY_KEYS,
        reason_codes_by_state=_capability_reason_codes(),
    )


def _receipt_binding(
    *,
    role: str,
    path: Path,
    receipt: Mapping[str, Any],
    recording_id: str,
    archive: Path,
) -> dict[str, Any]:
    binding = {
        "role": role,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "record_sha256": str(receipt["record_sha256"]),
        "schema_id": str(receipt["schema_id"]),
        "schema_version": int(receipt["schema_version"]),
    }
    return validate_admission_receipt_binding(
        binding,
        recording_id=recording_id,
        analysis_zarr=archive,
    )


def _require_relative_matches_core(
    dependency: Mapping[str, Any], *, roster: Mapping[str, Any]
) -> None:
    capabilities = roster["capability_bindings"]
    motion = capabilities[KINEMATICS_SAMPLES_CAPABILITY]["source_binding"]
    body = capabilities[SUBJECT_BODY_FRAME_CAPABILITY]["source_binding"]
    tracks = motion.get("tracks")
    if not isinstance(tracks, list) or len(tracks) != 1:
        _fail("Composite core motion authority must contain exactly one track.")
    track = _mapping(tracks[0], field="core motion selected track")
    if (
        dependency["core_authority_roster_sha256"] != roster["record_sha256"]
        or dependency["recording_id"] != roster["recording_id"]
        or dependency["selected_track_id"] != track.get("track_id")
        or dependency["core_motion_source_binding_sha256"]
        != motion.get("payload_sha256")
        or dependency["core_subject_body_frame_source_binding_sha256"]
        != body.get("payload_sha256")
    ):
        _fail("Chaser-relative source names another selected core authority.")


def _nested(value: Mapping[str, Any], path: tuple[str, ...], *, field: str) -> Any:
    current: Any = value
    for name in path:
        current = _mapping(current, field=field).get(name)
    return current


def _simple_binding(value: object, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    if not {"run_path", "manifest_sha256"}.issubset(record):
        _fail(f"{field} lacks one exact run/manifest binding.")
    return {
        "run_path": _text(record.get("run_path"), field=f"{field}.run_path"),
        "manifest_sha256": _digest(
            record.get("manifest_sha256"), field=f"{field}.manifest_sha256"
        ),
    }


def _require_extension_lineage(
    extension: BoundExactChaserExtensionSources,
    *,
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    dependencies: dict[str, Mapping[str, Any]] = {}
    for role in ("keypoint", "detection"):
        receipt = extension.relative_receipts[role]
        binding = extension.relative_bindings[role]
        dependency = core_paradigm_dependency_from_relative_manifest(
            receipt["run_manifest"],
            recording_id=extension.recording_id,
            analysis_zarr=extension.analysis_zarr,
            run_path=str(binding["run_path"]),
            manifest_sha256=str(binding["manifest_sha256"]),
            required=True,
        )
        assert dependency is not None
        _require_relative_matches_core(dependency, roster=roster)
        dependencies[role] = dependency
    shared_fields = (
        "core_authority_roster_sha256",
        "core_authority_consumption_receipt_sha256",
        "selected_track_id",
        "core_motion_source_binding_sha256",
        "core_subject_body_frame_source_binding_sha256",
    )
    if any(
        dependencies["keypoint"][field] != dependencies["detection"][field]
        for field in shared_fields
    ):
        _fail("Keypoint and detection projections bind different core authorities.")

    scientific = extension.scientific_manifests
    for child_key, role in _DIRECT_PARADIGM_CHILD_ROLES.items():
        if child_key not in scientific:
            continue
        observed = validate_core_paradigm_dependency(
            _mapping(scientific[child_key], field=f"{child_key} manifest").get(
                "core_authority"
            )
        )
        if _plain(observed) != _plain(dependencies[role]):
            _fail(f"Chaser child {child_key!r} binds another core projection.")

    spatial = _mapping(
        scientific["spatial_occupancy"].get("core_authority"),
        field="spatial core authority",
    )
    if set(spatial) != {
        "schema_id",
        "schema_version",
        "recording_id",
        "core_authority_roster_sha256",
        "core_authority_consumption_receipt_sha256",
        "selected_track_id",
        "provider_dependencies",
        "fallback",
    } or (
        spatial.get("schema_id") != "palette.core_behavior.paired_paradigm_dependency"
        or spatial.get("schema_version") != 1
        or spatial.get("recording_id") != extension.recording_id
        or spatial.get("fallback") != "prohibited"
    ):
        _fail("Spatial child lacks its exact paired core dependency.")
    raw_providers = spatial.get("provider_dependencies")
    if not isinstance(raw_providers, list) or len(raw_providers) != 2:
        _fail("Spatial core dependency must contain two provider projections.")
    observed_roles: list[str] = []
    for raw in raw_providers:
        provider = _mapping(raw, field="spatial provider dependency")
        if set(provider) != {"provider_role", "dependency"}:
            _fail("Spatial provider dependency field set is inexact.")
        role = _text(provider.get("provider_role"), field="spatial provider role")
        if role not in dependencies:
            _fail("Spatial provider dependency has an unsupported role.")
        dependency = validate_core_paradigm_dependency(provider.get("dependency"))
        if _plain(dependency) != _plain(dependencies[role]):
            _fail("Spatial provider binds another core projection.")
        observed_roles.append(role)
    if observed_roles != ["keypoint", "detection"]:
        _fail("Spatial provider dependency order is not canonical.")
    for field in shared_fields[:3]:
        if spatial.get(field) != dependencies["keypoint"][field]:
            _fail("Spatial paired dependency differs from the shared core authority.")

    motion_dependencies: dict[str, Mapping[str, Any]] = {}
    for child_key, path in _MOTION_CHILD_PATHS.items():
        if child_key not in scientific:
            continue
        value = _nested(scientific[child_key], path, field=f"{child_key} core source")
        motion_dependencies[child_key] = validate_core_motion_dependency_record(
            value, roster=roster
        )

    semantic = extension.semantic_binding
    controller = scientific["controller"]
    if _simple_binding(
        controller.get("source_relative_frame"),
        field="controller relative frame",
    ) != _simple_binding(
        extension.relative_bindings["keypoint"],
        field="keypoint relative binding",
    ) or _simple_binding(
        controller.get("semantic_selection"),
        field="controller semantic selection",
    ) != dict(
        semantic
    ):
        _fail("Controller child binds another relative or semantic source.")
    controller_payload = _digest(
        controller.get("payload_digest"), field="controller payload digest"
    )
    bout = scientific["bout"]
    bout_sources = _mapping(bout.get("sources"), field="bout sources")
    if (
        _simple_binding(bout_sources.get("relative_frame"), field="bout relative frame")
        != _simple_binding(
            extension.relative_bindings["keypoint"],
            field="keypoint relative binding",
        )
        or bout_sources.get("semantic_selection_manifest_sha256")
        != semantic["manifest_sha256"]
        or bout_sources.get("controller_trial_payload_sha256") != controller_payload
    ):
        _fail("Generalized bout-response dependency chain is stale or mixed.")
    bout_payload = _digest(bout.get("payload_digest"), field="bout payload digest")
    escape = scientific["escape"]
    escape_sources = _mapping(escape.get("sources"), field="escape sources")
    if (
        escape_sources.get("controller_trial_payload_sha256") != controller_payload
        or escape_sources.get("bout_response_payload_sha256") != bout_payload
    ):
        _fail("Escape/freeze dependency chain is stale or mixed.")
    if "epoch_behavior" in scientific:
        epoch_sources = _mapping(
            scientific["epoch_behavior"].get("sources"),
            field="epoch behavior sources",
        )
        if _simple_binding(
            epoch_sources.get("protocol_semantic_selection"),
            field="epoch semantic selection",
        ) != dict(semantic):
            _fail("Epoch behavior binds another semantic selection.")

    return _sealed(
        {
            "policy_id": "exact_chaser_children_share_one_core_roster_v1",
            "core_authority_roster_sha256": roster["record_sha256"],
            "relative_dependencies": _plain(dependencies),
            "motion_dependencies": _plain(motion_dependencies),
            "controller_payload_sha256": controller_payload,
            "bout_payload_sha256": bout_payload,
            "spatial_core_authority": _plain(spatial),
        }
    )


def _source_bindings(
    extension: BoundExactChaserExtensionSources,
    *,
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    relatives = extension.relative_evidence
    children = extension.relative_bindings
    core_capabilities = roster["capability_bindings"]
    motion = core_capabilities[KINEMATICS_SAMPLES_CAPABILITY]["source_binding"]
    body = core_capabilities[SUBJECT_BODY_FRAME_CAPABILITY]["source_binding"]
    bouts = core_capabilities[CANONICAL_SWIM_BOUTS_CAPABILITY]["source_binding"]
    semantic_manifest = extension.scientific_manifests["semantic_selection"]
    return {
        "fish_position_keypoint": {
            "binding_type": "core_motion_on_exact_chaser_carrier_v1",
            "authority": _plain(relatives["keypoint"]["fish"]),
            "sealed_by": _plain(children["keypoint"]),
        },
        "fish_position_detection": {
            "binding_type": "core_motion_on_exact_chaser_carrier_v1",
            "authority": _plain(relatives["detection"]["fish"]),
            "sealed_by": _plain(children["detection"]),
        },
        "chaser_observations_keypoint_projection": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": _plain(relatives["keypoint"]["chaser"]),
            "sealed_by": _plain(children["keypoint"]),
        },
        "chaser_observations_detection_projection": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": _plain(relatives["detection"]["chaser"]),
            "sealed_by": _plain(children["detection"]),
        },
        "anatomical_body_frame": {
            "binding_type": "selected_core_subject_body_frame_v1",
            "authority": _plain(relatives["keypoint"]["body"]),
            "source": _plain(body),
            "sealed_by": _plain(children["keypoint"]),
        },
        "row_axis_timing_and_scale": {
            "binding_type": "paired_relative_frame_consensus_v1",
            "authority": _plain(extension.relative_axis),
            "sealed_by": {
                "keypoint": _plain(children["keypoint"]),
                "detection": _plain(children["detection"]),
            },
        },
        "provider_motion": {
            "binding_type": "selected_core_motion_authority_v1",
            "source": _plain(motion),
            "authority": {
                "provider_id": str(motion["run_path"]),
                "provider_digest": str(motion["payload_sha256"]),
            },
            "sealed_by": roster["record_sha256"],
        },
        "canonical_swim_bouts": {
            "binding_type": "selected_core_swim_bout_authority_v1",
            "source": _plain(bouts),
            "sealed_by": roster["record_sha256"],
        },
        "semantic_epochs": {
            "binding_type": "exact_protocol_semantic_selection_v1",
            "source": _plain(semantic_manifest),
            "sealed_by": _plain(extension.exact_bindings["semantic_selection"]),
        },
        "reviewed_arena_and_scale": {
            "binding_type": "spatial_radial_consensus_v1",
            "authority": _plain(extension.reviewed_geometry),
            "sealed_by": {
                "spatial_occupancy": _plain(
                    extension.exact_bindings["spatial_occupancy"]
                ),
                "keypoint_radial": _plain(extension.exact_bindings["keypoint_radial"]),
                "detection_radial": _plain(
                    extension.exact_bindings["detection_radial"]
                ),
            },
        },
    }


def _complete_internal(scope: str, key: str) -> dict[str, Any]:
    return {
        "state": "complete",
        "reason_code": None,
        "detail": None,
        "binding_scope": scope,
        "binding_key": key,
    }


def _missing_internal() -> dict[str, Any]:
    return {
        "state": "unavailable",
        "reason_code": "missing_exact_receipt",
        "detail": "The selected exact projection has no child for this capability.",
        "binding_scope": None,
        "binding_key": None,
    }


def _public_capabilities(
    *,
    roster: Mapping[str, Any],
    extension: BoundExactChaserExtensionSources,
    lineage_proof: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": _plain(roster["capability_bindings"][key]),
        }
        for key in CORE_BEHAVIOR_CAPABILITY_KEYS
    }
    pair_binding = _sealed(
        {
            "profile_id": CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY,
            "core_authority_roster_sha256": roster["record_sha256"],
            "keypoint_relative": _plain(extension.relative_bindings["keypoint"]),
            "detection_relative": _plain(extension.relative_bindings["detection"]),
            "lineage_proof_sha256": lineage_proof["record_sha256"],
        }
    )
    result[CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY] = {
        "state": "complete",
        "reason_code": None,
        "detail": None,
        "binding": pair_binding,
    }
    for capability, child_key in _PUBLIC_CHILD_CAPABILITIES.items():
        binding = extension.scientific_child_bindings.get(child_key)
        if binding is None:
            result[capability] = {
                "state": "unavailable",
                "reason_code": "missing_exact_receipt",
                "detail": f"Exact chaser projection lacks {child_key!r}.",
                "binding": None,
            }
        else:
            result[capability] = {
                "state": "complete",
                "reason_code": None,
                "detail": None,
                "binding": _sealed(
                    {
                        "profile_id": f"core_bound_{capability}_v1",
                        "core_authority_roster_sha256": roster["record_sha256"],
                        "child": _plain(binding),
                        "lineage_proof_sha256": lineage_proof["record_sha256"],
                    }
                ),
            }
    result["reviewed_arena_and_scale"] = {
        "state": "complete",
        "reason_code": None,
        "detail": None,
        "binding": _sealed(
            {
                "profile_id": "core_bound_reviewed_arena_and_scale_v1",
                "core_authority_roster_sha256": roster["record_sha256"],
                "authority": _plain(extension.reviewed_geometry),
                "spatial_proof": _plain(extension.spatial_evidence),
            }
        ),
    }
    if set(result) != set(CORE_CHASER_CAPABILITY_KEYS):
        _fail("Composite public capability roster is incomplete or unexpected.")
    return result


def _internal_capabilities(
    extension: BoundExactChaserExtensionSources,
) -> dict[str, Any]:
    result = {
        key: _complete_internal("source_bindings", key)
        for key in _INTERNAL_SOURCE_CAPABILITIES
    }
    for child_key in BASE_SCIENTIFIC_CHILD_KEYS:
        result[child_key] = _complete_internal("scientific_child_bindings", child_key)
    for child_key in ("body_alignment_by_distance", "gaze"):
        result[child_key] = (
            _complete_internal("scientific_child_bindings", child_key)
            if child_key in extension.scientific_child_bindings
            else _missing_internal()
        )
    return result


def _resolve_content(
    core_execution_report_path: str | Path,
    chaser_projection_receipt_path: str | Path,
    *,
    expected_analysis_zarr: str | Path,
    expected_recording_id: str,
) -> dict[str, Any]:
    bound = bind_core_behavior_cohort_sources(
        core_execution_report_path,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
        export_profile_id=CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    )
    roster = validate_core_authority_roster(bound.core_authority_roster)
    motion_projection = roster["capability_bindings"][KINEMATICS_SAMPLES_CAPABILITY][
        "projection_contract"
    ]
    if motion_projection.get("sampling_stride_frames") != 1:
        _fail(
            "Composite chaser rows require the selected full-rate core-motion "
            "projection (sampling_stride_frames=1)."
        )
    extension = bind_exact_chaser_extension_sources(
        chaser_projection_receipt_path,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
    )
    lineage = _require_extension_lineage(extension, roster=roster)
    projection_binding = _receipt_binding(
        role=EXACT_CHASER_ADMISSION_ROLE,
        path=extension.projection_path,
        receipt=extension.projection,
        recording_id=extension.recording_id,
        archive=extension.analysis_zarr,
    )
    receipts = sorted(
        [_plain(bound.report_binding), projection_binding],
        key=lambda item: (item["role"], item["path"]),
    )
    source_bindings = _source_bindings(extension, roster=roster)
    public_capabilities = _public_capabilities(
        roster=roster, extension=extension, lineage_proof=lineage
    )
    return {
        "analysis_zarr": str(extension.analysis_zarr),
        "recording_id": extension.recording_id,
        "source_admission_receipts": receipts,
        "core_authority_roster": _plain(roster),
        "chaser_projection": {
            "receipt_path": str(extension.projection_path),
            "receipt_sha256": extension.projection["record_sha256"],
            "schema_id": extension.projection["schema_id"],
            "schema_version": extension.projection["schema_version"],
        },
        "source_bindings": source_bindings,
        "scientific_child_bindings": _plain(extension.scientific_child_bindings),
        "internal_capabilities": _internal_capabilities(extension),
        "capabilities": public_capabilities,
        "compatibility_proofs": {
            "extension_core_lineage": _plain(lineage),
            "paired_relative_frame_axis": _sealed(
                {
                    "policy_id": "exact_keypoint_detection_axis_consensus_v1",
                    "evidence": _plain(extension.relative_axis),
                }
            ),
            "spatial_radial_composition": _sealed(
                {
                    "policy_id": "exact_spatial_radial_consensus_v1",
                    "evidence": _plain(extension.spatial_evidence),
                }
            ),
        },
        "validation_policy": _plain(VALIDATION_POLICY),
        "safety": _plain(SAFETY),
    }


def build_core_chaser_composite_bundle(
    core_execution_report_path: str | Path,
    chaser_projection_receipt_path: str | Path,
    *,
    palette_commit: str,
    expected_analysis_zarr: str | Path,
    expected_recording_id: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build one immutable composite after full dynamic source admission."""

    commit = _text(palette_commit, field="palette_commit")
    if _COMMIT_RE.fullmatch(commit) is None:
        _fail("palette_commit must be one full lowercase Git object ID.")
    content = _resolve_content(
        core_execution_report_path,
        chaser_projection_receipt_path,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
    )
    body = {
        "schema_id": CORE_CHASER_BUNDLE_SCHEMA_ID,
        "schema_version": CORE_CHASER_BUNDLE_SCHEMA_VERSION,
        "method_id": CORE_CHASER_BUNDLE_METHOD_ID,
        "status": CORE_CHASER_BUNDLE_STATUS,
        **content,
        "software_authority": {"repository": "palette", "commit": commit},
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
    }
    result = {**body, "record_sha256": canonical_json_sha256(body)}
    validate_core_chaser_composite_bundle(result, validate_current_sources=False)
    return result


def _validate_receipt_binding(value: object, *, field: str) -> dict[str, Any]:
    record = _plain(_mapping(value, field=field))
    if set(record) != {
        "role",
        "path",
        "file_sha256",
        "record_sha256",
        "schema_id",
        "schema_version",
    }:
        _fail(f"{field} field set is inexact.")
    path = Path(_text(record.get("path"), field=f"{field}.path"))
    if not path.is_absolute() or str(path.resolve(strict=False)) != str(path):
        _fail(f"{field}.path must be canonical and absolute.")
    _digest(record.get("file_sha256"), field=f"{field}.file_sha256")
    _digest(record.get("record_sha256"), field=f"{field}.record_sha256")
    _text(record.get("schema_id"), field=f"{field}.schema_id")
    if type(record.get("schema_version")) is not int or record["schema_version"] <= 0:
        _fail(f"{field}.schema_version must be positive.")
    return record


def _validate_child_binding(value: object, *, field: str) -> dict[str, Any]:
    record = _plain(_mapping(value, field=field))
    if set(record) != {
        "receipt_path",
        "receipt_sha256",
        "run_path",
        "manifest_sha256",
        "payload_digest",
    }:
        _fail(f"{field} child binding is inexact.")
    _text(record.get("receipt_path"), field=f"{field}.receipt_path")
    _text(record.get("run_path"), field=f"{field}.run_path")
    for name in ("receipt_sha256", "manifest_sha256", "payload_digest"):
        _digest(record.get(name), field=f"{field}.{name}")
    return record


def _validate_source_bindings(
    value: object, *, roster: Mapping[str, Any]
) -> dict[str, Any]:
    bindings = _plain(_mapping(value, field="source_bindings"))
    if set(bindings) != set(_SOURCE_BINDING_TYPES):
        _fail("Composite source-binding inventory is inexact.")
    for key, binding_type in _SOURCE_BINDING_TYPES.items():
        binding = _mapping(bindings[key], field=f"source binding {key}")
        if binding.get("binding_type") != binding_type:
            _fail(f"Composite source binding {key!r} has another profile.")

    capabilities = roster["capability_bindings"]
    motion = capabilities[KINEMATICS_SAMPLES_CAPABILITY]["source_binding"]
    body = capabilities[SUBJECT_BODY_FRAME_CAPABILITY]["source_binding"]
    bouts = capabilities[CANONICAL_SWIM_BOUTS_CAPABILITY]["source_binding"]
    provider = _mapping(bindings["provider_motion"], field="provider motion binding")
    provider_authority = _mapping(
        provider.get("authority"), field="provider motion authority"
    )
    if (
        _plain(provider.get("source")) != _plain(motion)
        or provider.get("sealed_by") != roster["record_sha256"]
        or provider_authority.get("provider_id") != motion.get("run_path")
        or provider_authority.get("provider_digest") != motion.get("payload_sha256")
    ):
        _fail("Composite provider motion differs from the selected core roster.")
    anatomical = _mapping(
        bindings["anatomical_body_frame"], field="anatomical body binding"
    )
    if _plain(anatomical.get("source")) != _plain(body):
        _fail("Composite body frame differs from the selected core roster.")
    bout_binding = _mapping(
        bindings["canonical_swim_bouts"], field="canonical swim-bout binding"
    )
    if (
        _plain(bout_binding.get("source")) != _plain(bouts)
        or bout_binding.get("sealed_by") != roster["record_sha256"]
    ):
        _fail("Composite swim bouts differ from the selected core roster.")
    for key in ("fish_position_keypoint", "fish_position_detection"):
        authority = _mapping(
            _mapping(bindings[key], field=f"source binding {key}").get("authority"),
            field=f"source binding {key} authority",
        )
        if authority.get("source_authority_id") != motion.get("run_path"):
            _fail(f"Composite fish projection {key!r} names another core motion run.")
    return bindings


def _validate_internal_capabilities(
    value: object,
    *,
    source_bindings: Mapping[str, Any],
    scientific_children: Mapping[str, Any],
) -> dict[str, Any]:
    internal = _plain(_mapping(value, field="internal_capabilities"))
    expected = (
        set(_INTERNAL_SOURCE_CAPABILITIES)
        | set(BASE_SCIENTIFIC_CHILD_KEYS)
        | set(OPTIONAL_SCIENTIFIC_CHILD_KEYS)
    )
    if set(internal) != expected:
        _fail("Composite internal capability roster is inexact.")
    for key, raw in internal.items():
        item = _mapping(raw, field=f"internal capability {key}")
        if set(item) != {
            "state",
            "reason_code",
            "detail",
            "binding_scope",
            "binding_key",
        }:
            _fail(f"Internal capability {key!r} field set is inexact.")
        if item.get("state") == "complete":
            if (
                item.get("reason_code") is not None
                or item.get("detail") is not None
                or item.get("binding_scope")
                not in {"source_bindings", "scientific_child_bindings"}
                or item.get("binding_key") != key
            ):
                _fail(f"Internal capability {key!r} complete binding is invalid.")
            target = (
                source_bindings
                if item["binding_scope"] == "source_bindings"
                else scientific_children
            )
            if key not in target:
                _fail(f"Internal capability {key!r} binds an absent source.")
        elif (
            key not in OPTIONAL_SCIENTIFIC_CHILD_KEYS
            or item.get("state") != "unavailable"
            or item.get("reason_code") != "missing_exact_receipt"
            or type(item.get("detail")) is not str
            or item.get("binding_scope") is not None
            or item.get("binding_key") is not None
        ):
            _fail(f"Internal capability {key!r} disposition is invalid.")
    return internal


def _validate_capabilities(
    value: object, *, roster: Mapping[str, Any]
) -> dict[str, Any]:
    capabilities = _plain(_mapping(value, field="capabilities"))
    if set(capabilities) != set(CORE_CHASER_CAPABILITY_KEYS):
        _fail("Composite public capability roster is inexact.")
    reasons = _capability_reason_codes()
    for key, raw in capabilities.items():
        item = _mapping(raw, field=f"capability {key}")
        if set(item) != {"state", "reason_code", "detail", "binding"}:
            _fail(f"Capability {key!r} field set is inexact.")
        state = item.get("state")
        if state not in reasons or item.get("reason_code") not in reasons[state]:
            _fail(f"Capability {key!r} state/reason is invalid.")
        if item.get("detail") is not None and type(item.get("detail")) is not str:
            _fail(f"Capability {key!r} detail is invalid.")
        if state == "complete":
            if item.get("detail") is not None or not isinstance(
                item.get("binding"), Mapping
            ):
                _fail(f"Complete capability {key!r} lacks a binding.")
            if key not in CORE_BEHAVIOR_CAPABILITY_KEYS:
                _validate_record_seal(
                    item["binding"], field=f"capability {key} binding"
                )
        elif item.get("binding") is not None:
            _fail(f"Incomplete capability {key!r} cannot carry a binding.")
    for key in CORE_BEHAVIOR_CAPABILITY_KEYS:
        item = capabilities[key]
        if item["state"] != "complete" or _plain(item["binding"]) != _plain(
            roster["capability_bindings"][key]
        ):
            _fail(f"Core capability {key!r} differs from the selected roster.")
    required_extension = set(_CHASER_EXTENSION_CAPABILITY_KEYS).difference(
        {"body_alignment_by_distance"}
    )
    if any(capabilities[key]["state"] != "complete" for key in required_extension):
        _fail("Composite bundle lacks a required chaser extension capability.")
    optional_body = capabilities["body_alignment_by_distance"]
    if optional_body["state"] != "complete" and (
        optional_body["state"] != "unavailable"
        or optional_body["reason_code"] != "missing_exact_receipt"
    ):
        _fail("Optional body-alignment capability has an invalid disposition.")
    return capabilities


def validate_core_chaser_composite_bundle(
    value: object,
    *,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
    validate_current_sources: bool = False,
) -> Mapping[str, Any]:
    """Validate one sealed bundle and optionally re-admit its live sources."""

    record = _plain(_mapping(value, field="core-chaser composite bundle"))
    try:
        size = len(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise CoreChaserCompositeBundleError(
            f"Composite bundle is not strict JSON: {exc}"
        ) from exc
    if size > MAX_COMPOSITE_BUNDLE_BYTES:
        _fail("Composite bundle exceeds its bounded JSON contract.")
    persisted = _digest(record.pop("record_sha256", None), field="record_sha256")
    if canonical_json_sha256(record) != persisted:
        _fail("Composite bundle digest is stale.")
    required = {
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "analysis_zarr",
        "recording_id",
        "source_admission_receipts",
        "core_authority_roster",
        "chaser_projection",
        "source_bindings",
        "scientific_child_bindings",
        "internal_capabilities",
        "capabilities",
        "compatibility_proofs",
        "validation_policy",
        "safety",
        "software_authority",
        "created_at_utc",
    }
    if set(record) != required:
        _fail("Composite bundle field set is inexact.")
    if (
        record.get("schema_id") != CORE_CHASER_BUNDLE_SCHEMA_ID
        or record.get("schema_version") != CORE_CHASER_BUNDLE_SCHEMA_VERSION
        or record.get("method_id") != CORE_CHASER_BUNDLE_METHOD_ID
        or record.get("status") != CORE_CHASER_BUNDLE_STATUS
        or record.get("validation_policy") != dict(VALIDATION_POLICY)
        or record.get("safety") != dict(SAFETY)
    ):
        _fail("Composite bundle identity, policy, status, or safety is invalid.")
    archive = Path(_text(record.get("analysis_zarr"), field="analysis_zarr")).resolve()
    if str(archive) != record["analysis_zarr"]:
        _fail("Composite analysis_zarr path is not canonical and absolute.")
    recording_id = _text(record.get("recording_id"), field="recording_id")
    if (
        expected_analysis_zarr is not None
        and archive != Path(expected_analysis_zarr).expanduser().resolve()
    ):
        _fail("Composite bundle belongs to another analysis Zarr.")
    if expected_recording_id is not None and recording_id != expected_recording_id:
        _fail("Composite bundle belongs to another recording.")
    roster = validate_core_authority_roster(record.get("core_authority_roster"))
    if (
        roster["recording_id"] != recording_id
        or Path(roster["analysis_zarr"]) != archive
    ):
        _fail("Composite core roster belongs to another recording or archive.")
    receipts_raw = record.get("source_admission_receipts")
    if not isinstance(receipts_raw, list) or len(receipts_raw) != 2:
        _fail("Composite bundle requires exactly two source admission receipts.")
    receipts = [
        _validate_receipt_binding(item, field=f"source receipt {index}")
        for index, item in enumerate(receipts_raw)
    ]
    if [(item["role"], item["path"]) for item in receipts] != sorted(
        (item["role"], item["path"]) for item in receipts
    ) or {item["role"] for item in receipts} != {
        CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
        EXACT_CHASER_ADMISSION_ROLE,
    }:
        _fail("Composite source receipt roles or ordering are inexact.")
    by_role = {item["role"]: item for item in receipts}
    if by_role[CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE] != _plain(
        roster["execution_report_binding"]
    ):
        _fail("Composite core receipt differs from its authority roster.")
    projection = _mapping(record.get("chaser_projection"), field="chaser_projection")
    if set(projection) != {
        "receipt_path",
        "receipt_sha256",
        "schema_id",
        "schema_version",
    } or (
        projection.get("receipt_path") != by_role[EXACT_CHASER_ADMISSION_ROLE]["path"]
        or projection.get("receipt_sha256")
        != by_role[EXACT_CHASER_ADMISSION_ROLE]["record_sha256"]
        or projection.get("schema_id")
        != by_role[EXACT_CHASER_ADMISSION_ROLE]["schema_id"]
        or projection.get("schema_version")
        != by_role[EXACT_CHASER_ADMISSION_ROLE]["schema_version"]
    ):
        _fail("Composite chaser projection differs from its admission receipt.")
    children = _mapping(
        record.get("scientific_child_bindings"),
        field="scientific_child_bindings",
    )
    if set(BASE_SCIENTIFIC_CHILD_KEYS).difference(children) or not set(
        children
    ).issubset(_ALLOWED_SCIENTIFIC_CHILD_KEYS):
        _fail("Composite bundle lacks a required base chaser child.")
    for key, child in children.items():
        _validate_child_binding(child, field=f"scientific child {key}")
    source_bindings = _validate_source_bindings(
        record.get("source_bindings"), roster=roster
    )
    _validate_internal_capabilities(
        record.get("internal_capabilities"),
        source_bindings=source_bindings,
        scientific_children=children,
    )
    capabilities = _validate_capabilities(record.get("capabilities"), roster=roster)
    proofs = _mapping(record.get("compatibility_proofs"), field="compatibility proofs")
    if set(proofs) != {
        "extension_core_lineage",
        "paired_relative_frame_axis",
        "spatial_radial_composition",
    }:
        _fail("Composite compatibility-proof roster is inexact.")
    lineage = _mapping(
        proofs.get("extension_core_lineage"), field="extension core-lineage proof"
    )
    if lineage.get("core_authority_roster_sha256") != roster[
        "record_sha256"
    ] or canonical_json_sha256(
        {key: _plain(item) for key, item in lineage.items() if key != "record_sha256"}
    ) != lineage.get(
        "record_sha256"
    ):
        _fail("Composite extension core-lineage proof is stale.")
    for key, policy_id in (
        ("paired_relative_frame_axis", "exact_keypoint_detection_axis_consensus_v1"),
        ("spatial_radial_composition", "exact_spatial_radial_consensus_v1"),
    ):
        proof = _validate_record_seal(proofs[key], field=f"compatibility proof {key}")
        if proof.get("policy_id") != policy_id or "evidence" not in proof:
            _fail(f"Composite compatibility proof {key!r} is invalid.")
    software = _mapping(record.get("software_authority"), field="software_authority")
    if (
        set(software) != {"repository", "commit"}
        or software.get("repository") != "palette"
        or type(software.get("commit")) is not str
        or _COMMIT_RE.fullmatch(str(software["commit"])) is None
    ):
        _fail("Composite software authority is invalid.")
    _text(record.get("created_at_utc"), field="created_at_utc")
    normalized = {**record, "capabilities": capabilities, "record_sha256": persisted}
    if validate_current_sources:
        rebuilt = _resolve_content(
            by_role[CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE]["path"],
            by_role[EXACT_CHASER_ADMISSION_ROLE]["path"],
            expected_analysis_zarr=archive,
            expected_recording_id=recording_id,
        )
        for field in (
            "analysis_zarr",
            "recording_id",
            "source_admission_receipts",
            "core_authority_roster",
            "chaser_projection",
            "source_bindings",
            "scientific_child_bindings",
            "internal_capabilities",
            "capabilities",
            "compatibility_proofs",
            "validation_policy",
            "safety",
        ):
            if _plain(record[field]) != _plain(rebuilt[field]):
                _fail(f"Composite current sources changed at {field!r}.")
    return MappingProxyType(normalized)


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Composite bundle does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoreChaserCompositeBundleError(
            f"Cannot read composite bundle {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail("Composite bundle file must contain one JSON object.")
    return value


def read_core_chaser_composite_bundle(
    path: str | Path, **expected: Any
) -> Mapping[str, Any]:
    return validate_core_chaser_composite_bundle(
        _read_object(Path(path).expanduser().resolve()), **expected
    )


def ensure_core_chaser_composite_bundle(
    core_execution_report_path: str | Path,
    chaser_projection_receipt_path: str | Path,
    *,
    palette_commit: str,
    output_json: str | Path,
    expected_analysis_zarr: str | Path,
    expected_recording_id: str,
) -> dict[str, Any]:
    """Atomically create or exactly reuse one composite bundle generation."""

    output = Path(output_json).expanduser().resolve()
    if output.exists():
        current = read_core_chaser_composite_bundle(
            output,
            expected_analysis_zarr=expected_analysis_zarr,
            expected_recording_id=expected_recording_id,
            validate_current_sources=True,
        )
        receipts = {item["role"]: item for item in current["source_admission_receipts"]}
        if (
            Path(receipts[CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE]["path"])
            != Path(core_execution_report_path).expanduser().resolve()
            or Path(receipts[EXACT_CHASER_ADMISSION_ROLE]["path"])
            != Path(chaser_projection_receipt_path).expanduser().resolve()
            or current["software_authority"]["commit"] != palette_commit
        ):
            _fail("Existing composite bundle binds another source or commit.")
        return {**_plain(current), "bundle_path": str(output), "mode": "reused_exact"}
    bundle = build_core_chaser_composite_bundle(
        core_execution_report_path,
        chaser_projection_receipt_path,
        palette_commit=palette_commit,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
    )
    write_json_atomic(output, bundle, overwrite=False)
    return {**bundle, "bundle_path": str(output), "mode": "created"}


def _complete_bundle_member(
    bundle_path: str | Path,
    *,
    membership_member: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(bundle_path).expanduser().resolve()
    bundle = read_core_chaser_composite_bundle(
        source,
        expected_analysis_zarr=membership_member["analysis_zarr"],
        expected_recording_id=membership_member["recording_id"],
        validate_current_sources=True,
    )
    expected_receipts = sorted(
        [_plain(item) for item in membership_member["admission_receipts"]],
        key=lambda item: (item["role"], item["path"]),
    )
    if expected_receipts != list(bundle["source_admission_receipts"]):
        _fail("Composite bundle sources differ from membership admission receipts.")
    inventory = {
        "source_admission_receipts": expected_receipts,
        "core_authority_roster_sha256": bundle["core_authority_roster"][
            "record_sha256"
        ],
        "chaser_projection_receipt_sha256": bundle["chaser_projection"][
            "receipt_sha256"
        ],
        "capabilities": _plain(bundle["capabilities"]),
    }
    return {
        "recording_id": membership_member["recording_id"],
        "bundle_state": "complete",
        "reason_code": None,
        "bundle": {
            "adapter_id": CORE_CHASER_BUNDLE_ADAPTER_ID,
            "path": str(source),
            "file_sha256": sha256_file(source),
            "record_sha256": bundle["record_sha256"],
            "schema_id": CORE_CHASER_BUNDLE_SCHEMA_ID,
            "schema_version": CORE_CHASER_BUNDLE_SCHEMA_VERSION,
            "method_id": CORE_CHASER_BUNDLE_METHOD_ID,
            "status": CORE_CHASER_BUNDLE_STATUS,
            "receipt_bindings": expected_receipts,
            "binding_inventory_sha256": canonical_json_sha256(inventory),
        },
        "capabilities": _plain(bundle["capabilities"]),
    }


def _nonadmitted_member(member: Mapping[str, Any]) -> dict[str, Any]:
    state = str(member["membership_state"])
    if state == "admitted":
        _fail("Admitted composite members require one complete bundle.")
    if state == "invalid":
        capability_state, reason = "invalid", "invalid_source_authority"
    elif state == "excluded":
        capability_state, reason = "inapplicable", "member_not_admitted"
    else:
        capability_state, reason = (
            "unavailable",
            "blocked_by_unavailable_membership",
        )
    return {
        "recording_id": member["recording_id"],
        "bundle_state": state,
        "reason_code": member["reason_code"],
        "bundle": None,
        "capabilities": {
            key: {
                "state": capability_state,
                "reason_code": reason,
                "detail": member["disposition_evidence"]["detail"],
                "binding": None,
            }
            for key in CORE_CHASER_CAPABILITY_KEYS
        },
    }


def _bundle_profile(capability_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "adapter_id": CORE_CHASER_BUNDLE_ADAPTER_ID,
        "bundle_schema_id": CORE_CHASER_BUNDLE_SCHEMA_ID,
        "bundle_schema_version": CORE_CHASER_BUNDLE_SCHEMA_VERSION,
        "bundle_method_id": CORE_CHASER_BUNDLE_METHOD_ID,
        "bundle_status": CORE_CHASER_BUNDLE_STATUS,
        "capability_contract_sha256": capability_contract["record_sha256"],
        "export_profile_id": CORE_CHASER_EXPORT_PROFILE_ID,
        "publication_surface": "validated_behavior/v1",
        "core_authority_policy": "exactly_one_selected_roster_per_recording",
    }


def build_bundle_set_from_core_chaser_composite_bundles(
    *,
    bundle_set_id: str,
    membership: Mapping[str, Any],
    membership_path: str | Path,
    bundle_paths_by_recording: Mapping[str, str | Path],
    bundle_root: str | Path,
    palette_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Adapt prebuilt composite bundles into the generic bundle-set schema."""

    validated_membership = validate_membership_current_sources(membership)
    admitted = {
        member["recording_id"]
        for member in validated_membership["members"]
        if member["membership_state"] == "admitted"
    }
    if set(bundle_paths_by_recording) != admitted:
        _fail("Composite bundle paths must name every and only admitted recording.")
    contract = core_chaser_capability_contract()
    members = [
        (
            _complete_bundle_member(
                bundle_paths_by_recording[member["recording_id"]],
                membership_member=member,
            )
            if member["membership_state"] == "admitted"
            else _nonadmitted_member(member)
        )
        for member in validated_membership["members"]
    ]
    membership_file = Path(membership_path).expanduser().resolve()
    return build_validated_behavior_bundle_set(
        bundle_set_id=bundle_set_id,
        membership=validated_membership,
        membership_path=membership_file,
        membership_file_sha256=sha256_file(membership_file),
        bundle_root=bundle_root,
        bundle_profile=_bundle_profile(contract),
        capability_contract=contract,
        members=members,
        palette_commit=palette_commit,
        created_at_utc=created_at_utc,
    )


def validate_core_chaser_bundle_set_current_sources(
    value: object, *, membership: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate a generic bundle set and re-admit each composite source pair."""

    contract = core_chaser_capability_contract()
    bundle_set = validate_validated_behavior_bundle_set(
        value, membership=membership, capability_contract=contract
    )
    if _plain(bundle_set["bundle_profile"]) != _bundle_profile(contract):
        _fail("Bundle set does not declare the installed core-chaser profile.")
    members_by_id = {
        member["recording_id"]: member
        for member in validate_membership_current_sources(membership)["members"]
    }
    for member in bundle_set["members"]:
        membership_member = members_by_id[member["recording_id"]]
        if member["bundle_state"] != "complete":
            if member["bundle"] is not None:
                _fail("Incomplete composite member unexpectedly carries a bundle.")
            continue
        rebuilt = _complete_bundle_member(
            member["bundle"]["path"], membership_member=membership_member
        )
        for field in ("bundle_state", "reason_code", "bundle", "capabilities"):
            if _plain(member[field]) != _plain(rebuilt[field]):
                _fail(f"Composite bundle-set member changed at {field!r}.")
    return bundle_set


__all__ = [
    "CORE_CHASER_BUNDLE_ADAPTER_ID",
    "CORE_CHASER_BUNDLE_METHOD_ID",
    "CORE_CHASER_BUNDLE_SCHEMA_ID",
    "CORE_CHASER_BUNDLE_SCHEMA_VERSION",
    "CORE_CHASER_BUNDLE_STATUS",
    "CORE_CHASER_CAPABILITY_KEYS",
    "CORE_CHASER_CAPABILITY_PROFILE_ID",
    "CoreChaserCompositeBundleError",
    "build_bundle_set_from_core_chaser_composite_bundles",
    "build_core_chaser_composite_bundle",
    "core_chaser_capability_contract",
    "ensure_core_chaser_composite_bundle",
    "read_core_chaser_composite_bundle",
    "validate_core_chaser_bundle_set_current_sources",
    "validate_core_chaser_composite_bundle",
]
