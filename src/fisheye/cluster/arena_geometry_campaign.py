"""Plan and submit recording-level arena-geometry review campaigns.

This is the pre-review workflow only.  Each target publishes its immutable
acquisition candidate and independently generates a blind keyframe-only
Palette fit/reveal package.  The campaign stops before reviewed-candidate
publication, comparison, operational selection, or detection gating.  A final
serialized job refreshes registry projections from the immutable Zarr results.

The separate ``calibration`` subcommand only freezes or inspects existing
calibration evidence references. It writes no artifacts, submits no jobs, and
does not establish scientific acceptance or operational selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fisheye.cluster.arena_geometry_review import (
    ArenaGeometryProbeSource,
    ArenaGeometryReviewArrayWorkflowModule,
    ArenaGeometryReviewFragmentInputs,
    build_arena_geometry_review_array_fragment,
    compose_arena_geometry_workflow,
    validate_recording_level_probe_source,
)
from fisheye.cluster.clipped_inference import DEFAULT_REGISTRY
from fisheye.cluster.keypoints.common import (
    safe_component,
    validate_registered_analysis_zarr,
)
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfResources,
    LsfWorkflow,
    build_ssh_bsub_runner,
    submit_lsf_workflow,
    write_json_snapshot,
)

TARGET_MANIFEST_SCHEMA = "palette.arena_geometry_review_targets.v1"
PLAN_SCHEMA = "palette.arena_geometry_review_campaign_plan.v1"
DEFAULT_SUBMIT_HOST = "login1-citrus-poller"
CALIBRATION_PLAN_SCHEMA = "palette.arena_geometry_calibration_plan.v1"
CALIBRATION_READINESS_SCHEMA = "palette.arena_geometry_calibration_readiness.v1"
_CALIBRATION_ROLES = ("derivation", "validation")
_CALIBRATION_REFERENCE_LIMIT = 16 * 1024 * 1024


def _calibration_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} requires an actual lowercase SHA-256 digest.")
    return value


def _calibration_reference(value: Any) -> dict[str, Any]:
    """Normalize a reference without opening or statting its target."""
    if (
        not isinstance(value, Mapping)
        or not {"path", "sha256"} <= set(value)
        or set(value) - {"path", "sha256", "json_path"}
    ):
        raise ValueError(
            "Calibration artifact reference must name exact path and SHA-256."
        )
    path = value["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("Calibration artifact path must be absolute.")
    selected = value.get("json_path", [])
    if not isinstance(selected, list) or not all(
        isinstance(item, str) and item for item in selected
    ):
        raise ValueError("Calibration json_path must be a list of exact object keys.")
    return {
        "path": path,
        "sha256": _calibration_digest(value["sha256"], "artifact"),
        "json_path": list(selected),
    }


def _calibration_field(record: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        if not isinstance(record, Mapping) or key not in record:
            raise ValueError(f"Recorded calibration evidence lacks field {key!r}.")
        record = record[key]
    return record


class _CalibrationArtifactSet:
    """Whole-file identity, independent of JSON selectors or declared input role.

    Metadata checks recognize copies, symlinks, and hardlinks without reading
    protected content. File references still assume stable artifacts while an
    inspection executes; no file timestamp is treated as freeze evidence.
    """

    def __init__(self, references: Sequence[Mapping[str, Any]]) -> None:
        self.digests, self.paths, self.inodes = set(), set(), set()
        for reference in references:
            digest, path, inode = self._identity(reference)
            self.digests.add(digest)
            self.paths.add(path)
            if inode is not None:
                self.inodes.add(inode)

    @staticmethod
    def _identity(value: Any) -> tuple[str, str, tuple[int, int] | None]:
        reference = _calibration_reference(value)
        path = os.path.realpath(reference["path"])
        try:
            metadata = Path(path).stat()
        except FileNotFoundError:
            inode = None
        else:
            inode = (metadata.st_dev, metadata.st_ino)
        return reference["sha256"], path, inode

    def contains(self, reference: Mapping[str, Any]) -> bool:
        digest, path, inode = self._identity(reference)
        return (
            digest in self.digests
            or path in self.paths
            or (inode is not None and inode in self.inodes)
        )

    def reject(self, reference: Mapping[str, Any]) -> None:
        if self.contains(reference):
            raise ValueError(
                "Protected holdout/recording artifact role cannot be read through another input role."
            )


def _read_calibration_reference(
    value: Any,
    *,
    guard: Callable[[Mapping[str, Any]], None] | None = None,
    on_open: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    reference = _calibration_reference(value)
    if guard is not None:
        guard(reference)
    with Path(reference["path"]).open("rb") as handle:
        if on_open is not None:
            on_open(reference)
        raw = handle.read(_CALIBRATION_REFERENCE_LIMIT + 1)
    if len(raw) > _CALIBRATION_REFERENCE_LIMIT:
        raise ValueError(
            "Calibration references must be bounded JSON metadata artifacts."
        )
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Calibration artifact SHA-256 changed or is incorrect.")

    def reject(token: str) -> None:
        raise ValueError(f"Non-finite calibration JSON is forbidden: {token}")

    record = _calibration_field(
        json.loads(raw, parse_constant=reject), reference["json_path"]
    )
    if not isinstance(record, Mapping):
        raise ValueError("Calibration artifact must resolve to one JSON object.")
    return record


def _calibration_time(value: Any) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is None or result.utcoffset().total_seconds() != 0:
            raise ValueError
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Recorded calibration times must be explicit UTC timestamps."
        ) from exc
    return result


def _calibration_cell(value: Any) -> dict[str, Any]:
    from fisheye.shared.arena_geometry_auto_policy import (
        validate_geometry_policy_applicability,
    )

    if not isinstance(value, Mapping) or set(value) != {
        "applicability",
        "registration_sha256",
    }:
        raise ValueError(
            "A camera-registration cell requires exact applicability and registration."
        )
    validate_geometry_policy_applicability(value["applicability"])
    return {
        "applicability": dict(value["applicability"]),
        "registration_sha256": _calibration_digest(
            value["registration_sha256"], "registration"
        ),
    }


def build_geometry_calibration_plan(
    *,
    derivation_cohort: Mapping[str, Any],
    validation_cohort: Mapping[str, Any],
    recording_evidence: Sequence[Mapping[str, Any]],
    scientific_recipe_sha256: str,
    expected_camera_registration_cells: Mapping[str, Sequence[Mapping[str, Any]]],
    required_negative_controls: Sequence[Mapping[str, str]],
    diagnostic_catalog_reference: Mapping[str, Any] | None = None,
    cohort_profile: str = "explicit_frozen_geometry_cohorts_v1",
) -> dict[str, Any]:
    """Freeze a read-only readiness inventory without opening holdout evidence.

    Existing frozen-cohort manifests own membership. This planning record adds
    exact geometry evidence references and accounting, not acceptance. Neither
    metric/review paths nor diagnostic catalog contents are opened here.
    """
    from fisheye.cohorts.registry import validate_frozen_cohort
    from fisheye.shared.arena_geometry_auto_policy import SOURCE_BINDING_FIELDS
    from fisheye.shared.zarr.manifest_digest import (
        canonical_json_bytes,
        canonical_json_sha256,
    )

    recipe = _calibration_digest(scientific_recipe_sha256, "scientific recipe")
    if cohort_profile not in {
        "explicit_frozen_geometry_cohorts_v1",
        "goodbatbadbat_oq5_2026_08_12",
    }:
        raise ValueError("Unsupported calibration cohort profile.")
    if not isinstance(expected_camera_registration_cells, Mapping) or set(
        expected_camera_registration_cells
    ) != set(_CALIBRATION_ROLES):
        raise ValueError("Calibration requires exact expected cells for both roles.")
    # Membership metadata must not smuggle protected per-recording payloads
    # through a different json_path in the same file before roles are known.
    recording_artifacts = _CalibrationArtifactSet(
        [
            row[field]
            for row in recording_evidence
            if isinstance(row, Mapping)
            for field in (
                "shadow_evidence_reference",
                "historical_review_reference",
                "acquisition_candidate_reference",
            )
            if row.get(field) is not None
        ]
    )
    cohorts, members, cells, coverage = {}, {}, {}, {}
    for role, reference in (
        ("derivation", derivation_cohort),
        ("validation", validation_cohort),
    ):
        reference = _calibration_reference(reference)
        manifest = _read_calibration_reference(
            reference, guard=recording_artifacts.reject
        )
        errors = validate_frozen_cohort(manifest)
        if errors:
            raise ValueError("Invalid frozen cohort: " + "; ".join(errors))
        _calibration_time(manifest["created_utc"])
        member_list = sorted(manifest["members"], key=lambda row: row["recording_id"])
        for member in member_list:
            key = member["recording_id"]
            if key in members:
                raise ValueError(
                    "Derivation and validation recording membership overlaps."
                )
            for field in ("rig_id", "arena_id", "camera_id"):
                if not isinstance(member.get(field), str) or not member[field]:
                    raise ValueError(
                        "Geometry frozen cohort lacks exact rig/arena/camera identity."
                    )
            members[key] = (role, member)
        expected = [
            _calibration_cell(cell) for cell in expected_camera_registration_cells[role]
        ]
        cell_map = {canonical_json_sha256(cell): cell for cell in expected}
        if not cell_map or len(cell_map) != len(expected):
            raise ValueError(
                "Expected camera-registration cells are empty or duplicated."
            )
        cells[role] = dict(sorted(cell_map.items()))
        cohorts[role] = {
            "reference": reference,
            "manifest_sha256": manifest["manifest_sha256"],
            "created_utc": manifest["created_utc"],
            "recording_ids": [m["recording_id"] for m in member_list],
        }
        coverage[role] = {
            "expected_recordings": len(member_list),
            "expected_camera_registration_cell_count": len(cell_map),
        }
        if cohort_profile == "goodbatbadbat_oq5_2026_08_12":
            expected_count, expected_day = (
                (36, "2026-08-10") if role == "derivation" else (28, "2026-08-11")
            )
            cameras = {c["applicability"]["camera_serial"] for c in expected}
            registrations = {c["registration_sha256"] for c in expected}
            scopes = {canonical_json_sha256(c["applicability"]) for c in expected}
            actual_pairs = {
                (canonical_json_sha256(c["applicability"]), c["registration_sha256"])
                for c in expected
            }
            if (
                len(member_list) != expected_count
                or len(expected) != 12
                or len(cameras) != 4
                or len(registrations) != 3
                or len(scopes) != 4
                or actual_pairs
                != {
                    (scope, registration)
                    for scope in scopes
                    for registration in registrations
                }
                or any(
                    _calibration_time(m.get("recording_started_utc")).date().isoformat()
                    != expected_day
                    for m in member_list
                )
            ):
                raise ValueError(
                    "OQ5 requires complete August 10 36/12-cell and August 11 28/12-cell cohorts."
                )
    scope_sets = [
        {canonical_json_sha256(c["applicability"]) for c in cells[role].values()}
        for role in _CALIBRATION_ROLES
    ]
    registration_sets = [
        {c["registration_sha256"] for c in cells[role].values()}
        for role in _CALIBRATION_ROLES
    ]
    if scope_sets[0] != scope_sets[1]:
        raise ValueError(
            "Derivation and validation must cover the same exact camera/canvas/arena scopes."
        )
    if registration_sets[0] & registration_sets[1]:
        raise ValueError(
            "Derivation and validation must use independent registration snapshots."
        )
    inventory = {}
    detection_sources = set()
    fields = {
        "recording_id",
        "parent_recording_id",
        "applicability",
        "registration_sha256",
        "scientific_recipe_sha256",
        "expected_source_bindings",
        "shadow_evidence_reference",
        "historical_review_reference",
        "acquisition_candidate_reference",
    }
    for row in recording_evidence:
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError(
                "Calibration recording evidence has missing or unsupported fields."
            )
        key = row["recording_id"]
        if key not in members or key in inventory:
            raise ValueError(
                "Calibration inventory has an unexpected or duplicated frozen cohort member."
            )
        if row["parent_recording_id"] != key:
            raise ValueError(
                "Clip-derived parent observations cannot be counted as independent calibration recordings."
            )
        if row["scientific_recipe_sha256"] != recipe:
            raise ValueError("Calibration evidence uses the wrong scientific recipe.")
        source_bindings = row["expected_source_bindings"]
        if source_bindings is not None:
            if (
                not isinstance(source_bindings, Mapping)
                or set(source_bindings) != SOURCE_BINDING_FIELDS
            ):
                raise ValueError(
                    "Expected source bindings must name the exact shadow evidence sources."
                )
            for key_name, value in source_bindings.items():
                _calibration_digest(value, "expected_source_bindings." + key_name)
            if source_bindings["scientific_recipe_sha256"] != recipe:
                raise ValueError("Expected source bindings use the wrong recipe.")
            signature = source_bindings["detection_source_signature"]
            if signature in detection_sources:
                raise ValueError(
                    "An exact detection source cannot be reused across independent recording members."
                )
            detection_sources.add(signature)
        role, member = members[key]
        cell = _calibration_cell(
            {k: row[k] for k in ("applicability", "registration_sha256")}
        )
        cell_id = canonical_json_sha256(cell)
        if cell_id not in cells[role]:
            raise ValueError(
                "Calibration evidence belongs to an unexpected camera-registration cell."
            )
        scope = cell["applicability"]
        if (scope["rig_id"], scope["arena_id"], scope["camera_serial"]) != (
            member["rig_id"],
            member["arena_id"],
            member["camera_id"],
        ):
            raise ValueError(
                "Calibration evidence does not match its frozen cohort source."
            )
        inventory[key] = {
            **row,
            "role": role,
            "cell_id": cell_id,
            "shadow_evidence_reference": _calibration_reference(
                row["shadow_evidence_reference"]
            )
            if row["shadow_evidence_reference"] is not None
            else None,
            "historical_review_reference": _calibration_reference(
                row["historical_review_reference"]
            )
            if row["historical_review_reference"] is not None
            else None,
            "acquisition_candidate_reference": _calibration_reference(
                row["acquisition_candidate_reference"]
            )
            if row["acquisition_candidate_reference"] is not None
            else None,
        }
    controls = []
    for control in required_negative_controls:
        if (
            not isinstance(control, Mapping)
            or set(control) != {"control_id", "kind"}
            or not isinstance(control["control_id"], str)
            or not control["control_id"].strip()
            or control["kind"] not in {"real", "injected"}
        ):
            raise ValueError(
                "Required negative controls need exact IDs and real/injected kinds."
            )
        controls.append(dict(control))
    if len({c["control_id"] for c in controls}) != len(controls) or {
        c["kind"] for c in controls
    } != {"real", "injected"}:
        raise ValueError(
            "Freeze unique real and injected negative controls explicitly."
        )
    payload = {
        "schema": CALIBRATION_PLAN_SCHEMA,
        "status": "planned",
        "cohort_profile": cohort_profile,
        "cohorts": cohorts,
        "expected_camera_registration_cells": cells,
        "coverage": coverage,
        "recordings": [
            {
                "recording_id": key,
                "role": members[key][0],
                "evidence": inventory.get(key),
            }
            for key in sorted(members)
        ],
        "scientific_recipe_sha256": recipe,
        "required_negative_controls": sorted(controls, key=lambda c: c["control_id"]),
        "diagnostic_catalog_reference": _calibration_reference(
            diagnostic_catalog_reference
        )
        if diagnostic_catalog_reference is not None
        else None,
        "scientific_acceptance_created": False,
        "selection_performed": False,
        "validation_evidence_opened": False,
        "holdout_freshness": "unknown",
    }
    payload["plan_sha256"] = canonical_json_sha256(payload)
    return json.loads(canonical_json_bytes(payload))


def _calibration_policies(
    plan: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    *,
    read_reference: Callable[[Any], Mapping[str, Any]],
    inspected_at: datetime,
) -> dict[str, Any]:
    from fisheye.shared.arena_geometry_auto_policy import validate_geometry_auto_policy
    from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

    allowed_scopes = {
        canonical_json_sha256(cell["applicability"])
        for cell in plan["expected_camera_registration_cells"]["derivation"].values()
    }
    policies = {}
    for binding in references:
        if not isinstance(binding, Mapping) or set(binding) != {
            "policy_reference",
            "threshold_derivation_reference",
        }:
            raise ValueError(
                "A frozen policy requires exact policy and threshold-derivation references."
            )
        policy = read_reference(binding["policy_reference"])
        validate_geometry_auto_policy(policy)
        calibration = policy["calibration"]
        if calibration is None or policy["thresholds"] is None:
            raise ValueError("Readiness cannot invent missing calibrated thresholds.")
        derivation_reference = _calibration_reference(
            binding["threshold_derivation_reference"]
        )
        # This verifies the bound artifact bytes, not the scientific method by
        # which its author derived thresholds. Readiness never upgrades that
        # declaration to independently validated calibration or acceptance.
        read_reference(derivation_reference)
        if (
            calibration["threshold_derivation_sha256"] != derivation_reference["sha256"]
            or calibration["derivation_manifest_sha256"]
            != plan["cohorts"]["derivation"]["manifest_sha256"]
        ):
            raise ValueError(
                "Frozen policy does not bind the exact derivation cohort/artifact."
            )
        if calibration["scientific_recipe_sha256"] != plan[
            "scientific_recipe_sha256"
        ] or calibration["validation_manifest_sha256"] not in {
            None,
            plan["cohorts"]["validation"]["manifest_sha256"],
        }:
            raise ValueError(
                "Frozen policy has the wrong scientific recipe or validation cohort."
            )
        scope = canonical_json_sha256(calibration["applicability"])
        if scope not in allowed_scopes or scope in policies:
            raise ValueError("Frozen policy applicability is unexpected or ambiguous.")
        frozen = _calibration_time(calibration["frozen_at_utc"])
        if frozen > inspected_at:
            raise ValueError(
                "A future threshold freeze cannot authorize current holdout access."
            )
        if any(
            frozen < _calibration_time(cohort["created_utc"])
            for cohort in plan["cohorts"].values()
        ):
            raise ValueError(
                "Policy freeze predates its recorded cohort membership freeze."
            )
        policies[scope] = policy
    return policies


def _calibration_chronology(
    plan: Mapping[str, Any],
    policies: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    read_reference: Callable[[Any], Mapping[str, Any]],
    inspected_at: datetime,
) -> tuple[bool, str, list[str], list[dict[str, Any]]]:
    """Check actual recorded fields; never attest the absence of prior access."""
    validation_digest = plan["cohorts"]["validation"]["manifest_sha256"]
    policy_by_digest = {policy["digest"]: policy for policy in policies.values()}
    freeze_times, access_times, observed_times, checked = {}, [], [], []
    reasons = []
    freshness = "unknown"
    for event in events:
        if (
            not isinstance(event, Mapping)
            or set(event)
            != {"kind", "reference", "timestamp_path", "subject_digest_path"}
            or event["kind"]
            not in {
                "threshold_freeze",
                "validation_access",
                "aggregate_inspection",
                "adjudication",
                "threshold_tuning",
            }
        ):
            raise ValueError(
                "Chronology requires exact recorded event references, not an untouched boolean."
            )
        for key in ("timestamp_path", "subject_digest_path"):
            if (
                not isinstance(event[key], list)
                or not event[key]
                or not all(isinstance(value, str) and value for value in event[key])
            ):
                raise ValueError(
                    "Chronology field selectors must name exact recorded JSON keys."
                )
        record = read_reference(event["reference"])
        recorded_time = _calibration_field(record, event["timestamp_path"])
        at = _calibration_time(recorded_time)
        if at > inspected_at:
            raise ValueError(
                "A future chronology event cannot authorize current holdout access."
            )
        subject = _calibration_field(record, event["subject_digest_path"])
        _calibration_digest(subject, "chronology subject")
        kind = event["kind"]
        if kind == "threshold_freeze":
            if subject not in policy_by_digest or subject in freeze_times:
                raise ValueError(
                    "Chronology has an unknown or conflicting exact policy freeze."
                )
            if at != _calibration_time(
                policy_by_digest[subject]["calibration"]["frozen_at_utc"]
            ):
                raise ValueError(
                    "Recorded threshold freeze disagrees with the frozen policy."
                )
            freeze_times[subject] = at
        else:
            if subject != validation_digest:
                raise ValueError(
                    "Recorded holdout use concerns a different frozen cohort."
                )
            observed_times.append(at)
            if kind == "validation_access":
                access_times.append(at)
            if kind == "threshold_tuning":
                freshness = "fresh_holdout_required"
                reasons.append("holdout_used_for_threshold_tuning")
        checked.append(
            {
                "kind": kind,
                "reference": _calibration_reference(event["reference"]),
                "recorded_at_utc": recorded_time,
                "subject_sha256": subject,
                "assurance": "digest_bound_declared_event_not_independent_no_prior_access_proof",
            }
        )
    if not policies:
        reasons.append("frozen_policy_references_missing")
    if set(freeze_times) != set(policy_by_digest) or not freeze_times:
        reasons.append("recorded_threshold_freeze_evidence_missing")
    if not access_times:
        reasons.append("recorded_validation_access_evidence_missing")
    if (
        freeze_times
        and observed_times
        and max(freeze_times.values()) >= min(observed_times)
    ):
        reasons.append("thresholds_not_frozen_before_recorded_holdout_access")
    return not reasons, freshness, sorted(set(reasons)), checked


def _validate_calibration_candidate(
    candidate: Mapping[str, Any], item: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Bind candidate-level registration/scope, not a particular recording."""
    from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
        validate_acquisition_geometry_candidate_record,
    )
    from fisheye.shared.json_safety import strict_json_dumps
    from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

    validate_acquisition_geometry_candidate_record(candidate)
    candidate_digest = hashlib.sha256(
        strict_json_dumps(candidate).encode("utf-8")
    ).hexdigest()
    if (
        candidate_digest
        != item["expected_source_bindings"]["acquisition_candidate_record_sha256"]
    ):
        raise ValueError(
            "Candidate artifact does not match the exact bound acquisition candidate."
        )
    registration = candidate["acquisition_source"].get("registration_sha256")
    if (
        not isinstance(registration, str)
        or registration.strip().lower().removeprefix("sha256:")
        != item["registration_sha256"]
    ):
        raise ValueError(
            "Candidate registration does not match the declared camera-registration cell."
        )
    coordinate = candidate["coordinate_binding"]
    scope = {
        **{
            key: candidate["arena_binding"][key]
            for key in ("rig_id", "canvas_name", "arena_id", "camera_serial")
        },
        "coordinate_profile_id": coordinate["profile_id"],
        "native_width_px": coordinate["native_width_px"],
        "native_height_px": coordinate["native_height_px"],
    }
    if scope != item["applicability"]:
        raise ValueError(
            "Candidate applicability does not match the exact declared scope."
        )
    sources = item["expected_source_bindings"]
    if (
        candidate["acquisition_source"]["source_observation_sha256"]
        .strip()
        .lower()
        .removeprefix("sha256:")
        != sources["acquisition_observation_sha256"]
        or canonical_json_sha256(coordinate) != sources["coordinate_binding_sha256"]
        or candidate["valid_detection_region"]["geometry"]
        != evidence["acquisition_gate"]
    ):
        raise ValueError(
            "Metric acquisition observation, coordinate binding, or gate disagrees with the bound candidate."
        )


def _calibration_holdout_references(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    references = [
        row["evidence"][field]
        for row in plan["recordings"]
        if row["role"] == "validation" and row["evidence"] is not None
        for field in (
            "shadow_evidence_reference",
            "historical_review_reference",
            "acquisition_candidate_reference",
        )
        if row["evidence"][field] is not None
    ]
    if plan["diagnostic_catalog_reference"] is not None:
        references.append(plan["diagnostic_catalog_reference"])
    return references


def inspect_geometry_calibration_readiness(
    plan: Mapping[str, Any],
    *,
    phase: str = "derivation",
    policy_references: Sequence[Mapping[str, Any]] = (),
    chronology_events: Sequence[Mapping[str, Any]] = (),
    negative_control_references: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Account for every frozen member without treating presence as acceptance.

    The default derivation phase never reads validation metrics/reviews or
    chronology files. Explicit validation reads additionally require complete
    exact policy files and a recorded freeze-before-access chronology. That
    chronology is a bound declaration, not independent proof of an untouched
    holdout. Existing manual comparisons and control files remain distinct
    from policy-specific adjudication and validated fail-closed outcomes.
    """
    from fisheye.analysis_workflows.materializers.arena_geometry_comparison import (
        validate_arena_geometry_comparison_record,
    )
    from fisheye.shared.arena_geometry_auto_policy import (
        evaluate_geometry_auto_policy,
        validate_geometry_shadow_evidence,
    )
    from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

    if phase not in _CALIBRATION_ROLES:
        raise ValueError("Readiness phase must be derivation or explicit validation.")
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema") != CALIBRATION_PLAN_SCHEMA
        or plan.get("status") != "planned"
        or plan.get("scientific_acceptance_created") is not False
        or plan.get("selection_performed") is not False
        or plan.get("validation_evidence_opened") is not False
        or plan.get("plan_sha256")
        != canonical_json_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    ):
        raise ValueError("Calibration plan is malformed or its frozen digest changed.")
    # Rebuild from the exact recorded metadata inputs, rejecting plan tampering
    # or moved frozen-cohort artifacts without opening per-member evidence.
    rebuilt = build_geometry_calibration_plan(
        derivation_cohort=plan["cohorts"]["derivation"]["reference"],
        validation_cohort=plan["cohorts"]["validation"]["reference"],
        recording_evidence=[
            {k: v for k, v in row["evidence"].items() if k not in {"role", "cell_id"}}
            for row in plan["recordings"]
            if row["evidence"] is not None
        ],
        scientific_recipe_sha256=plan["scientific_recipe_sha256"],
        expected_camera_registration_cells={
            role: list(plan["expected_camera_registration_cells"][role].values())
            for role in _CALIBRATION_ROLES
        },
        required_negative_controls=plan["required_negative_controls"],
        diagnostic_catalog_reference=plan["diagnostic_catalog_reference"],
        cohort_profile=plan["cohort_profile"],
    )
    if rebuilt != dict(plan):
        raise ValueError(
            "Calibration plan does not reproduce its exact membership and references."
        )
    protected = _CalibrationArtifactSet(_calibration_holdout_references(plan))
    metric_artifacts = _CalibrationArtifactSet(
        [
            row["evidence"]["shadow_evidence_reference"]
            for row in plan["recordings"]
            if row["role"] == "validation"
            and row["evidence"] is not None
            and row["evidence"]["shadow_evidence_reference"] is not None
        ]
    )
    validation_unlocked, validation_opened, validation_evidence_opened = (
        False,
        False,
        False,
    )

    def record_open(reference: Mapping[str, Any]) -> None:
        nonlocal validation_opened, validation_evidence_opened
        validation_opened = validation_opened or metric_artifacts.contains(reference)
        validation_evidence_opened = validation_evidence_opened or protected.contains(
            reference
        )

    def read_reference(reference: Any) -> Mapping[str, Any]:
        return _read_calibration_reference(
            reference,
            guard=None if validation_unlocked else protected.reject,
            on_open=record_open,
        )

    inspected_at = datetime.now(timezone.utc)
    policies = _calibration_policies(
        plan,
        policy_references,
        read_reference=read_reference,
        inspected_at=inspected_at,
    )
    reasons = [
        "independent_holdout_freshness_not_established",
        "policy_specific_adjudication_not_validated",
        "negative_control_outcomes_not_validated",
        "member_source_provenance_not_independently_verified",
    ]
    chronology_consistent, freshness, history = False, "unknown", []
    if phase == "validation":
        chronology_consistent, freshness, chronology_reasons, history = (
            _calibration_chronology(
                plan,
                policies,
                chronology_events,
                read_reference=read_reference,
                inspected_at=inspected_at,
            )
        )
        reasons.extend(chronology_reasons)
        required_scopes = {
            canonical_json_sha256(c["applicability"])
            for c in plan["expected_camera_registration_cells"]["validation"].values()
        }
        if set(policies) != required_scopes:
            reasons.append("frozen_policy_applicability_coverage_incomplete")
        validation_unlocked = chronology_consistent and set(policies) == required_scopes
    coverage = {
        role: {
            **plan["coverage"][role],
            "missing_inventory": 0,
            "metric_references_valid": 0,
            "member_source_bindings_validated": 0,
            "independent_camera_registration_cells_validated": 0,
            "metrics_missing": 0,
            "metrics_invalid": 0,
            "locked": 0,
            "historical_review_references_valid": 0,
            "policy_adjudication_validated": 0,
            "camera_registration_cells": {
                key: {**cell, "recording_ids": [], "metric_references_valid": 0}
                for key, cell in plan["expected_camera_registration_cells"][
                    role
                ].items()
            },
        }
        for role in _CALIBRATION_ROLES
    }
    rows = []
    for member in plan["recordings"]:
        role, item = member["role"], member["evidence"]
        row = {
            "recording_id": member["recording_id"],
            "role": role,
            "status": "missing_inventory",
            "reason_codes": [],
            "historical_review_status": "not_read",
            "policy_adjudication_status": "not_validated",
            "member_source_provenance_status": "not_independently_verified",
            "candidate_binding_status": "not_read",
        }
        if item is None:
            coverage[role]["missing_inventory"] += 1
            row["reason_codes"].append("recording_evidence_inventory_missing")
            rows.append(row)
            continue
        cell = coverage[role]["camera_registration_cells"][item["cell_id"]]
        cell["recording_ids"].append(member["recording_id"])
        if role == "validation" and not validation_unlocked:
            coverage[role]["locked"] += 1
            row["status"] = "holdout_locked"
            row["reason_codes"].append("validation_evidence_not_opened")
            rows.append(row)
            continue
        metric_ref, expected_sources = (
            item["shadow_evidence_reference"],
            item["expected_source_bindings"],
        )
        if (
            metric_ref is None
            or expected_sources is None
            or item["acquisition_candidate_reference"] is None
        ):
            coverage[role]["metrics_missing"] += 1
            row["status"] = "metrics_missing"
            row["reason_codes"].append(
                "exact_metric_candidate_reference_or_source_bindings_missing"
            )
        else:
            try:
                evidence = read_reference(metric_ref)
                validate_geometry_shadow_evidence(evidence)
                if (
                    evidence["applicability"] != item["applicability"]
                    or evidence["source_bindings"] != expected_sources
                ):
                    raise ValueError(
                        "Metric evidence is stale or for another recording/source/recipe/applicability."
                    )
                candidate = read_reference(item["acquisition_candidate_reference"])
                _validate_calibration_candidate(candidate, item, evidence)
                row["candidate_binding_status"] = (
                    "exact_candidate_registration_and_scope_valid"
                )
                scope = canonical_json_sha256(item["applicability"])
                if scope in policies:
                    evaluation = evaluate_geometry_auto_policy(
                        policy=policies[scope], evidence=evidence
                    )
                    row["shadow_thresholds_satisfied"] = evaluation[
                        "thresholds_satisfied"
                    ]
                    row["shadow_evaluation_sha256"] = evaluation["digest"]
                coverage[role]["metric_references_valid"] += 1
                cell["metric_references_valid"] += 1
                row["status"] = "metric_reference_valid_member_source_unverified"
            except FileNotFoundError as exc:
                coverage[role]["metrics_missing"] += 1
                row["status"] = "metrics_missing"
                row["reason_codes"].append("metric_evidence_file_missing:" + str(exc))
            except (OSError, TypeError, ValueError) as exc:
                coverage[role]["metrics_invalid"] += 1
                row["status"] = "metrics_invalid"
                row["reason_codes"].append("metric_evidence_invalid:" + str(exc))
        review_ref = item["historical_review_reference"]
        if review_ref is None:
            row["historical_review_status"] = "missing"
        else:
            try:
                review = read_reference(review_ref)
                validate_arena_geometry_comparison_record(review)
                if (
                    expected_sources is None
                    or review["candidate_bindings"]["acquisition"][
                        "candidate_record_sha256"
                    ]
                    != expected_sources["acquisition_candidate_record_sha256"]
                ):
                    raise ValueError(
                        "Historical comparison is bound to another acquisition candidate."
                    )
                coverage[role]["historical_review_references_valid"] += 1
                row["historical_review_status"] = (
                    "comparison_present_not_policy_specific_adjudication"
                )
            except (OSError, TypeError, ValueError) as exc:
                row["historical_review_status"] = "invalid"
                row["reason_codes"].append(
                    "historical_review_reference_invalid:" + str(exc)
                )
        rows.append(row)
    controls = negative_control_references or {}
    required_ids = {c["control_id"] for c in plan["required_negative_controls"]}
    if not isinstance(controls, Mapping) or set(controls) - required_ids:
        raise ValueError(
            "Negative-control result references must use the frozen exact control IDs."
        )
    control_results, missing_ids = [], []
    for control in plan["required_negative_controls"]:
        key = control["control_id"]
        result = {**control, "status": "missing_result_reference"}
        if key not in controls:
            missing_ids.append(key)
        elif not validation_unlocked:
            result["status"] = "result_reference_frozen_not_opened"
            result["reference"] = _calibration_reference(controls[key])
        else:
            try:
                artifact = read_reference(controls[key])
                if artifact.get("control_id") != key:
                    raise ValueError("Control artifact has the wrong control ID.")
                result["reference"] = _calibration_reference(controls[key])
                result["status"] = "present_outcome_not_independently_validated"
            except (OSError, TypeError, ValueError) as exc:
                result["status"] = "invalid_result_reference"
                result["reason"] = str(exc)
        control_results.append(result)
    for role in _CALIBRATION_ROLES:
        role_cells = coverage[role]["camera_registration_cells"]
        coverage[role]["cells_without_valid_metric_references"] = sorted(
            key
            for key, cell in role_cells.items()
            if cell["metric_references_valid"] == 0
        )
        coverage[role]["accounted_recordings"] = sum(
            coverage[role][name]
            for name in (
                "missing_inventory",
                "metric_references_valid",
                "metrics_missing",
                "metrics_invalid",
                "locked",
            )
        )
    result = {
        "schema": CALIBRATION_READINESS_SCHEMA,
        "status": "readiness",
        "phase": phase,
        "inspected_at_utc": inspected_at.isoformat().replace("+00:00", "Z"),
        "plan_sha256": plan["plan_sha256"],
        "coverage": coverage,
        "recordings": rows,
        "negative_controls": {
            "required_count": len(required_ids),
            "missing_result_ids": sorted(missing_ids),
            "records": control_results,
            "validated_outcome_count": 0,
        },
        "policy_digests": sorted(policy["digest"] for policy in policies.values()),
        "recorded_chronology": history,
        "declared_chronology_consistent": chronology_consistent,
        "holdout_freshness": freshness,
        "validation_metrics_opened": validation_opened,
        "validation_evidence_opened": validation_evidence_opened,
        "diagnostic_catalog_reference": plan["diagnostic_catalog_reference"],
        "promotion_readiness": "not_established",
        "reason_codes": sorted(set(reasons)),
        "scientific_acceptance_created": False,
        "selection_performed": False,
        "false_automatic_pass_count": None,
    }
    result["readiness_sha256"] = canonical_json_sha256(result)
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _contained_file(path: Path, recording_dir: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(recording_dir)
    except ValueError as exc:
        raise ValueError(f"{label} must belong to the recording directory.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


@dataclass(frozen=True)
class ArenaGeometryTarget:
    target_id: str
    recording_id: str
    recording_dir: Path
    analysis_zarr: Path
    video_path: Path | None = None
    summary_path: Path | None = None
    keyframe_path: Path | None = None
    recovery_receipt_path: Path | None = None
    acquisition_observation_path: Path | None = None
    geometry_source: str = "recovery-receipt"
    geometry_camera_serial: str | None = None
    geometry_arena_id: str | None = None
    citrus_h5_path: Path | None = None

    def __post_init__(self) -> None:
        recording = self.recording_dir.expanduser().resolve()
        if not recording.is_dir():
            raise FileNotFoundError(f"Recording directory not found: {recording}")
        target_id = safe_component(self.target_id, default="target", max_length=80)
        recording_id = str(self.recording_id).strip()
        if not recording_id:
            raise ValueError(f"Target {target_id!r} has no recording_id.")
        analysis = self.analysis_zarr.expanduser().resolve()
        try:
            analysis.relative_to(recording)
        except ValueError as exc:
            raise ValueError("Analysis Zarr must belong to the recording.") from exc
        if not (analysis / "zarr.json").is_file():
            raise FileNotFoundError(f"Analysis target is not Zarr v3: {analysis}")
        observation = self.acquisition_observation_path
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "recording_id", recording_id)
        object.__setattr__(self, "recording_dir", recording)
        object.__setattr__(self, "analysis_zarr", analysis)
        probe_source = ArenaGeometryProbeSource(
            video_path=self.video_path,
            summary_path=self.summary_path,
            keyframe_path=self.keyframe_path,
            recording_dir=(
                recording
                if all(
                    value is None
                    for value in (
                        self.video_path,
                        self.summary_path,
                        self.keyframe_path,
                    )
                )
                else None
            ),
            acquisition_observation_path=self.acquisition_observation_path,
        )
        validate_recording_level_probe_source(recording, probe_source)
        object.__setattr__(self, "video_path", probe_source.video_path)
        object.__setattr__(self, "summary_path", probe_source.summary_path)
        object.__setattr__(self, "keyframe_path", probe_source.keyframe_path)
        geometry_source = str(self.geometry_source).strip()
        if geometry_source not in {
            "producer-folder",
            "citrus-h5",
            "recovery-receipt",
        }:
            raise ValueError(f"Unsupported geometry source: {geometry_source!r}.")
        receipt = None
        if geometry_source == "recovery-receipt":
            if self.recovery_receipt_path is None:
                raise ValueError("Recovery geometry requires recovery_receipt.")
            receipt = _contained_file(
                self.recovery_receipt_path,
                recording,
                label="geometry recovery receipt",
            )
        elif self.recovery_receipt_path is not None:
            raise ValueError(
                "Producer-native geometry must not declare recovery_receipt."
            )
        camera_serial = (
            str(self.geometry_camera_serial).strip()
            if self.geometry_camera_serial is not None
            else ""
        )
        arena_id = (
            str(self.geometry_arena_id).strip()
            if self.geometry_arena_id is not None
            else ""
        )
        if geometry_source != "recovery-receipt" and not (camera_serial and arena_id):
            raise ValueError(
                "Producer-native geometry requires geometry_camera_serial and "
                "geometry_arena_id."
            )
        citrus_h5 = None
        if geometry_source == "citrus-h5":
            if self.citrus_h5_path is None:
                raise ValueError("citrus-h5 geometry requires citrus_h5.")
            citrus_h5 = _contained_file(
                self.citrus_h5_path,
                recording,
                label="recording-bound Citrus H5",
            )
        elif self.citrus_h5_path is not None:
            raise ValueError("citrus_h5 is only valid for citrus-h5 geometry.")
        object.__setattr__(self, "recovery_receipt_path", receipt)
        object.__setattr__(self, "geometry_source", geometry_source)
        object.__setattr__(self, "geometry_camera_serial", camera_serial or None)
        object.__setattr__(self, "geometry_arena_id", arena_id or None)
        object.__setattr__(self, "citrus_h5_path", citrus_h5)
        if observation is not None:
            object.__setattr__(
                self,
                "acquisition_observation_path",
                _contained_file(
                    observation,
                    recording,
                    label="acquisition rim observation",
                ),
            )

    def probe_source(self) -> ArenaGeometryProbeSource:
        return ArenaGeometryProbeSource(
            video_path=self.video_path,
            summary_path=self.summary_path,
            keyframe_path=self.keyframe_path,
            recording_dir=(self.recording_dir if self.video_path is None else None),
            acquisition_observation_path=self.acquisition_observation_path,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "recording_id": self.recording_id,
            "recording_dir": str(self.recording_dir),
            "analysis_zarr": str(self.analysis_zarr),
            "probe_source": self.probe_source().to_json(),
            "video": str(self.video_path) if self.video_path is not None else None,
            "summary": (
                str(self.summary_path) if self.summary_path is not None else None
            ),
            "keyframes": (
                str(self.keyframe_path) if self.keyframe_path is not None else None
            ),
            "geometry_source": self.geometry_source,
            "geometry_camera_serial": self.geometry_camera_serial,
            "geometry_arena_id": self.geometry_arena_id,
            "recovery_receipt": (
                str(self.recovery_receipt_path)
                if self.recovery_receipt_path is not None
                else None
            ),
            "citrus_h5": (
                str(self.citrus_h5_path) if self.citrus_h5_path is not None else None
            ),
            "acquisition_observation": (
                str(self.acquisition_observation_path)
                if self.acquisition_observation_path is not None
                else None
            ),
        }


def load_target_manifest(path: Path) -> tuple[ArenaGeometryTarget, ...]:
    payload = _read_json_object(path.expanduser().resolve())
    if payload.get("schema") != TARGET_MANIFEST_SCHEMA:
        raise ValueError(f"Target manifest schema must be {TARGET_MANIFEST_SCHEMA!r}.")
    rows = payload.get("targets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Target manifest requires a non-empty targets list.")
    targets: list[ArenaGeometryTarget] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Target row {index} is not an object.")
        recording_dir = Path(str(row.get("recording_dir") or ""))
        observation = row.get("acquisition_observation")
        recovery = row.get("recovery_receipt")
        citrus_h5 = row.get("citrus_h5")
        video = row.get("video")
        summary = row.get("summary")
        keyframes = row.get("keyframes")
        targets.append(
            ArenaGeometryTarget(
                target_id=str(row.get("target_id") or recording_dir.name),
                recording_id=str(row.get("recording_id") or ""),
                recording_dir=recording_dir,
                analysis_zarr=Path(str(row.get("analysis_zarr") or "")),
                video_path=Path(str(video)) if video else None,
                summary_path=Path(str(summary)) if summary else None,
                keyframe_path=Path(str(keyframes)) if keyframes else None,
                recovery_receipt_path=(Path(str(recovery)) if recovery else None),
                acquisition_observation_path=(
                    Path(str(observation)) if observation else None
                ),
                geometry_source=str(row.get("geometry_source") or "recovery-receipt"),
                geometry_camera_serial=(
                    str(row.get("geometry_camera_serial"))
                    if row.get("geometry_camera_serial") is not None
                    else None
                ),
                geometry_arena_id=(
                    str(row.get("geometry_arena_id"))
                    if row.get("geometry_arena_id") is not None
                    else None
                ),
                citrus_h5_path=(Path(str(citrus_h5)) if citrus_h5 else None),
            )
        )
    if len({target.target_id for target in targets}) != len(targets):
        raise ValueError("Target ids must be unique.")
    if len({target.analysis_zarr for target in targets}) != len(targets):
        raise ValueError("Analysis Zarr targets must be unique.")
    return tuple(targets)


def _repo_commit(repo: Path) -> str:
    resolved = repo.expanduser().resolve()
    status = subprocess.run(
        ["git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        text=True,
        capture_output=True,
    )
    if status.stdout.strip():
        raise ValueError(f"Palette repo must be clean: {resolved}")
    commit = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("Palette repo did not resolve one full commit SHA.")
    return commit


@dataclass(frozen=True)
class ArenaGeometryCampaignPlan:
    run_label: str
    workflow_id: str
    repo: Path
    repo_commit: str
    registry: Path
    run_root: Path
    probe_queue: str
    targets: tuple[ArenaGeometryTarget, ...]
    module: ArenaGeometryReviewArrayWorkflowModule
    workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "run_label": self.run_label,
            "workflow_id": self.workflow_id,
            "repo": str(self.repo),
            "repo_commit": self.repo_commit,
            "registry": str(self.registry),
            "run_root": str(self.run_root),
            "probe_queue": self.probe_queue,
            "target_count": len(self.targets),
            "targets": [
                {**target.to_json(), "outputs": output.to_json()}
                for target, output in zip(
                    self.targets, self.module.outputs, strict=True
                )
            ],
            "execution_mode": "lsf_arrays",
            "human_review_barrier": True,
            "post_review_publication": "not_scheduled",
            "candidate_comparison": "not_scheduled",
            "operational_selection": "not_scheduled",
            "detection_gating": "not_scheduled",
            "registry_update": True,
            "lsf_workflow": self.workflow.to_json(),
        }


def build_plan(
    *,
    targets: Sequence[ArenaGeometryTarget],
    run_label: str,
    repo: Path,
    registry_path: Path,
    run_root: Path,
    acquisition_array_concurrency: int = 8,
    probe_array_concurrency: int = 4,
    probe_queue: str = "gpu_l4",
) -> ArenaGeometryCampaignPlan:
    if not targets:
        raise ValueError("Arena-geometry campaign requires at least one target.")
    label = safe_component(run_label, default="arena_geometry", max_length=80)
    workflow_id = f"arena_geometry_{label}"
    resolved_repo = repo.expanduser().resolve()
    resolved_registry = registry_path.expanduser().resolve()
    resolved_run_root = run_root.expanduser().resolve()
    queue = str(probe_queue).strip()
    if queue not in {"gpu_l4", "gpu_t4"}:
        raise ValueError("Geometry probe queue must be gpu_l4 or gpu_t4.")
    commit = _repo_commit(resolved_repo)
    fragment_inputs: list[ArenaGeometryReviewFragmentInputs] = []
    for target in targets:
        validate_registered_analysis_zarr(
            registry_path=resolved_registry,
            recording_id=target.recording_id,
            analysis_zarr=target.analysis_zarr,
        )
        fragment_inputs.append(
            ArenaGeometryReviewFragmentInputs(
                workflow_id=workflow_id,
                target_id=target.target_id,
                recording_dir=target.recording_dir,
                analysis_zarr=target.analysis_zarr,
                recovery_receipt_path=target.recovery_receipt_path,
                geometry_source=target.geometry_source,
                geometry_camera_serial=target.geometry_camera_serial,
                geometry_arena_id=target.geometry_arena_id,
                citrus_h5_path=target.citrus_h5_path,
                source=target.probe_source(),
                repo=resolved_repo,
                run_root=resolved_run_root,
                registry_path=resolved_registry,
                probe_resources=LsfResources(
                    queue=queue,
                    ncores=8,
                    mem_gb=32,
                    gpus=1,
                    walltime="1:00",
                    span_hosts=1,
                ),
            )
        )
    module = build_arena_geometry_review_array_fragment(
        tuple(fragment_inputs),
        acquisition_max_concurrent=int(acquisition_array_concurrency),
        probe_max_concurrent=int(probe_array_concurrency),
    )
    workflow = compose_arena_geometry_workflow(
        workflow_id=workflow_id,
        modules=(module,),
    )
    return ArenaGeometryCampaignPlan(
        run_label=label,
        workflow_id=workflow_id,
        repo=resolved_repo,
        repo_commit=commit,
        registry=resolved_registry,
        run_root=resolved_run_root,
        probe_queue=queue,
        targets=tuple(targets),
        module=module,
        workflow=workflow,
    )


def materialize_plan_bundle(plan: ArenaGeometryCampaignPlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = _read_json_object(plan_path)
        if existing != payload:
            raise FileExistsError(f"Run root contains a different plan: {plan_path}")
        if (
            not lsf_path.is_file()
            or _read_json_object(lsf_path) != plan.workflow.to_json()
        ):
            raise FileExistsError(f"Run root has mismatched LSF evidence: {lsf_path}")
        return existing
    for name in ("logs", "status", "arena_geometry"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.workflow.to_json())
    return payload


def apply_plan(
    plan: ArenaGeometryCampaignPlan,
    *,
    runner: CommandRunner,
) -> dict[str, Any]:
    submission_path = plan.run_root / "lsf_submission.json"
    if submission_path.exists():
        raise FileExistsError(f"Submission evidence already exists: {submission_path}")
    materialize_plan_bundle(plan)
    return submit_lsf_workflow(
        plan.workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission_path,
        runner=runner,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--acquisition-array-concurrency", type=int, default=8)
    parser.add_argument("--probe-array-concurrency", type=int, default=4)
    parser.add_argument(
        "--probe-queue",
        choices=("gpu_l4", "gpu_t4"),
        default="gpu_l4",
    )
    parser.add_argument(
        "--submit-host",
        default=os.environ.get("PALETTE_LSF_SUBMIT_HOST", DEFAULT_SUBMIT_HOST),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _calibration_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only geometry calibration planning; stdout only, no submissions or acceptance."
    )
    parser.add_argument("operation", choices=("freeze", "readiness"))
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Freeze keyword inputs, or an exact saved plan for readiness.",
    )
    parser.add_argument("--phase", choices=_CALIBRATION_ROLES, default="derivation")
    parser.add_argument(
        "--readiness-inputs",
        type=Path,
        help="Optional explicit policy/chronology/control reference keyword inputs.",
    )
    parser.add_argument(
        "--readiness-inputs-sha256",
        help="Required exact raw-file SHA-256 whenever --readiness-inputs is supplied.",
    )
    args = parser.parse_args(argv)
    if (args.readiness_inputs is None) != (args.readiness_inputs_sha256 is None):
        parser.error(
            "--readiness-inputs and --readiness-inputs-sha256 are required together"
        )
    payload = _read_json_object(args.input_json)
    if args.operation == "freeze":
        if args.readiness_inputs is not None or args.phase != "derivation":
            parser.error("freeze cannot request validation or readiness evidence")
        result = build_geometry_calibration_plan(**payload)
    else:
        inputs = (
            _read_calibration_reference(
                {
                    "path": str(args.readiness_inputs.absolute()),
                    "sha256": args.readiness_inputs_sha256,
                },
                guard=_CalibrationArtifactSet(
                    _calibration_holdout_references(payload)
                ).reject,
            )
            if args.readiness_inputs is not None
            else {}
        )
        if set(inputs) - {
            "policy_references",
            "chronology_events",
            "negative_control_references",
        }:
            raise ValueError("Unsupported calibration readiness inputs.")
        result = inspect_geometry_calibration_readiness(
            payload, phase=args.phase, **inputs
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    actual = list(argv) if argv is not None else sys.argv[1:]
    if actual[:1] == ["calibration"]:
        return _calibration_main(actual[1:])
    args = _parser().parse_args(argv)
    targets = load_target_manifest(args.manifest)
    plan = build_plan(
        targets=targets,
        run_label=args.run_label,
        repo=args.repo,
        registry_path=args.registry,
        run_root=args.run_root,
        acquisition_array_concurrency=args.acquisition_array_concurrency,
        probe_array_concurrency=args.probe_array_concurrency,
        probe_queue=args.probe_queue,
    )
    result = (
        apply_plan(plan, runner=build_ssh_bsub_runner(args.submit_host))
        if args.apply
        else materialize_plan_bundle(plan)
    )
    summary = {
        "status": "submitted" if args.apply else "dry_run",
        "plan_path": str(plan.run_root / "plan.json"),
        "lsf_plan_path": str(plan.run_root / "lsf_plan.json"),
        "submission_path": (
            str(plan.run_root / "lsf_submission.json") if args.apply else None
        ),
        "target_count": len(plan.targets),
        "job_count": len(plan.workflow.jobs),
        "execution_mode": "lsf_arrays",
        "probe_queue": plan.probe_queue,
        "human_review_barrier": True,
        "post_review_publication": "not_scheduled",
        "operational_selection": "not_scheduled",
        "detection_gating": "not_scheduled",
        "registry_update": True,
        "result": result if args.apply else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"{summary['status']}: {summary['target_count']} targets, "
            f"{summary['job_count']} jobs; stopped before human review"
        )
        print(f"Plan: {summary['plan_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
