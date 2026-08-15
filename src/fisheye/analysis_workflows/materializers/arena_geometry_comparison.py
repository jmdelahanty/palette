"""Immutable comparison of acquisition and independently fitted dish geometry.

The comparison is evidence only.  It never selects geometry, edits either
candidate, filters detections, or advances a production selector.  Numerical
automatic-selection thresholds remain deliberately unpromoted until the
approved canary and holdout have been adjudicated.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    ACQUISITION_CANDIDATE_KIND,
    CANDIDATE_RUNS_PARENT,
    PALETTE_CANDIDATE_KIND,
    validate_arena_geometry_candidate_record,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.detection_tables import (
    resolve_detection_instance_table,
    resolve_detection_source_pixel_authority,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID,
    canonical_detection_dimensions_from_manifest,
    require_active_coordinate_canonical_detection,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


COMPARISON_RECORD_SCHEMA_ID = "palette.arena_geometry_comparison_record"
COMPARISON_RECORD_SCHEMA_VERSION = 1
COMPARISON_RUN_SCHEMA_ID = "palette.arena_geometry_comparison_run"
COMPARISON_RUN_SCHEMA_VERSION = 1
COMPARISON_RUNS_PARENT = "arena_geometry_comparison_runs"
COMPARISON_PUBLISH_SCHEMA_ID = "palette.arena_geometry_comparison_publish"
COMPARISON_ALGORITHM_VERSION = 1

MANUAL_REVIEW_POLICY_ID = "manual_review_only_v1"
CORROBORATED_ACQUISITION_POLICY_ID = "corroborated_acquisition_v1"
SUPPORTED_POLICY_IDS = frozenset(
    {MANUAL_REVIEW_POLICY_ID, CORROBORATED_ACQUISITION_POLICY_ID}
)
SEMANTIC_COMPATIBILITY_STATES = frozenset(
    {
        "same_feature_confirmed",
        "different_feature_confirmed",
        "projected_edges_unresolved",
    }
)

_UNPROMOTED_THRESHOLDS: dict[str, float | int | None] = {
    "maximum_center_displacement_px": None,
    "maximum_center_displacement_dish_top_rim_mm": None,
    "maximum_same_feature_radius_difference_px": None,
    "maximum_same_feature_boundary_separation_px": None,
    "minimum_circle_iou": None,
    "maximum_gate_disagreement_fraction": None,
    "minimum_angular_edge_support_fraction": None,
    "maximum_radial_residual_px": None,
    "maximum_between_window_center_variation_px": None,
    "maximum_between_window_radius_variation_px": None,
    "minimum_acquisition_boundary_edge_support_fraction": None,
}

_CANARY_MEASUREMENTS = (
    "freeze_2026_08_10_goodbatbadbat_derivation_manifest",
    "adjudicate_same_feature_and_projected_edge_semantics",
    "measure_all_four_cameras_and_registration_cells",
    "freeze_thresholds_before_2026_08_11_holdout",
    "obtain_zero_false_automatic_passes_on_complete_holdout",
    "pass_real_and_injected_fail_closed_controls",
)


@dataclass(frozen=True)
class ArenaGeometryComparisonPlan:
    source_zarr: Path
    acquisition_candidate_run: str
    acquisition_candidate_record_sha256: str
    palette_candidate_run: str
    palette_candidate_record_sha256: str
    detect_source_group_path: str | None
    detect_source_signature: str | None
    comparison_id: str
    comparison_record_sha256: str
    comparison_record: Mapping[str, Any]
    target_run_path: Path
    run_provenance: Mapping[str, Any]


def _canonical_copy(value: Any) -> Any:
    return json.loads(strict_json_dumps(value))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _sha256_hex(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string.")
    digest = value.strip().lower().removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} is not a valid SHA-256 digest.")
    return digest


def _safe_name(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or "/" in text or text in {".", ".."}:
        raise ValueError(f"{label} must be one safe group name.")
    return text


def _safe_group_path(value: object) -> str:
    text = str(value or "").strip().strip("/")
    parts = text.split("/") if text else []
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("detect_source_group_path must be a safe relative group path.")
    return "/".join(parts)


def _candidate_snapshot(
    root: Any,
    *,
    run_name: str,
    expected_kind: str,
) -> dict[str, Any]:
    name = _safe_name(run_name, label=f"{expected_kind}_candidate_run")
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
        raise ValueError(f"Candidate {name!r} is not complete immutable evidence.")
    record = attrs.get("candidate_record")
    if not isinstance(record, Mapping):
        raise ValueError(f"Candidate {name!r} lacks candidate_record.")
    validate_arena_geometry_candidate_record(record)
    if record.get("candidate_kind") != expected_kind:
        raise ValueError(f"Candidate {name!r} has the wrong candidate kind.")
    digest = _payload_sha256(record)
    if attrs.get("candidate_record_sha256") != digest:
        raise ValueError(f"Candidate {name!r} record digest is invalid.")
    if attrs.get("candidate_id") != name:
        raise ValueError(f"Candidate {name!r} identity disagrees with its run name.")
    return {
        "run_name": name,
        "candidate_id": name,
        "candidate_kind": expected_kind,
        "candidate_record_sha256": digest,
        "record": _canonical_copy(record),
    }


def _circle(container: Mapping[str, Any], *, label: str) -> tuple[float, float, float]:
    geometry = container.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("type") != "circle":
        raise ValueError(f"{label} is not a circle.")
    center = geometry.get("center_px")
    if not isinstance(center, Mapping):
        raise ValueError(f"{label} lacks center_px.")
    values = (float(center.get("x")), float(center.get("y")), float(geometry.get("radius_px")))
    if not all(math.isfinite(value) for value in values) or values[2] <= 0.0:
        raise ValueError(f"{label} must be finite with a positive radius.")
    return values


def _circle_record(values: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "type": "circle",
        "center_px": {"x": values[0], "y": values[1]},
        "radius_px": values[2],
    }


def _circle_iou(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    x1, y1, r1 = first
    x2, y2, r2 = second
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance >= r1 + r2:
        intersection = 0.0
    elif distance <= abs(r1 - r2):
        intersection = math.pi * min(r1, r2) ** 2
    else:
        alpha = math.acos((distance * distance + r1 * r1 - r2 * r2) / (2.0 * distance * r1))
        beta = math.acos((distance * distance + r2 * r2 - r1 * r1) / (2.0 * distance * r2))
        lens = 0.5 * math.sqrt(
            max(
                0.0,
                (-distance + r1 + r2)
                * (distance + r1 - r2)
                * (distance - r1 + r2)
                * (distance + r1 + r2),
            )
        )
        intersection = r1 * r1 * alpha + r2 * r2 * beta - lens
    union = math.pi * r1 * r1 + math.pi * r2 * r2 - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(strict_json_dumps(list(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _detection_snapshot(
    root: Any,
    *,
    source_group_path: str,
    coordinate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = _safe_group_path(source_group_path)
    try:
        active_manifest = require_active_coordinate_canonical_detection(
            root,
            group_path=path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Detection source {path!r} is not the active canonical-v3 authority: {exc}"
        ) from exc
    try:
        group = root[path]
    except KeyError as exc:
        raise ValueError(f"Detection source is missing: {path}") from exc
    attrs = dict(group.attrs)
    if attrs.get("palette_run_completion_status") != "complete":
        raise ValueError(f"Detection source {path!r} is not complete.")
    table = resolve_detection_instance_table(group)
    for name in ("instance_key", "frame_indices", "bbox_norm_coords"):
        if name not in table:
            raise ValueError(f"Detection source {path!r} lacks {name}.")
    keys = np.asarray(table["instance_key"][:], dtype=np.uint64).reshape(-1)
    frames = np.asarray(table["frame_indices"][:], dtype=np.int64).reshape(-1)
    boxes = np.asarray(table["bbox_norm_coords"][:], dtype=np.float64)
    if boxes.shape != (len(keys), 4) or frames.shape != keys.shape:
        raise ValueError("Detection source row arrays do not have identical coverage.")
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Detection source instance_key values are not unique.")
    width_px = int(coordinate_binding["native_width_px"])
    height_px = int(coordinate_binding["native_height_px"])
    manifest = attrs.get("run_manifest")
    manifest_digest: str | None = None
    if (
        isinstance(manifest, Mapping)
        and manifest.get("schema_id") == CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID
    ):
        dimensions = canonical_detection_dimensions_from_manifest(manifest)
        if int(dimensions.n_instances) != len(keys):
            raise ValueError(
                "Canonical detection manifest instance count differs from its "
                "persisted table."
            )
        observed_width = int(dimensions.source_width)
        observed_height = int(dimensions.source_height)
        manifest_digest = _sha256_hex(
            manifest.get("payload_digest"),
            label="canonical detection manifest payload digest",
        )
        if manifest_digest != active_manifest["payload_digest"]:
            raise ValueError(
                "Detection source canonical manifest changed during comparison preflight."
            )
    else:
        observed_width = int(attrs.get("source_video_width") or attrs.get("width") or 0)
        observed_height = int(
            attrs.get("source_video_height") or attrs.get("height") or 0
        )
    if (observed_width, observed_height) != (width_px, height_px):
        raise ValueError(
            "Detection source and geometry comparison native extents disagree."
        )
    descriptor = table["bbox_norm_coords"].attrs.get("coordinate_descriptor")
    if descriptor is not None:
        if not isinstance(descriptor, Mapping) or (
            descriptor.get("geometry_type") != "bbox_cxcywh"
            or tuple(descriptor.get("component_units") or ()) != ("normalized",) * 4
        ):
            raise ValueError(
                "Detection source bbox_norm_coords coordinate descriptor is "
                "incompatible."
            )
    elif manifest_digest is None:
        raise ValueError(
            "Detection source bbox_norm_coords lacks coordinate authority."
        )
    pixel_authority = resolve_detection_source_pixel_authority(attrs)
    expected_authority = {
        "record_ref": coordinate_binding.get("pixel_frame_record_ref"),
        "record_sha256": _sha256_hex(
            coordinate_binding.get("pixel_frame_record_sha256"),
            label="geometry source-camera pixel authority digest",
        ),
    }
    if pixel_authority != expected_authority:
        raise ValueError(
            "Detection source and comparison candidates do not share the exact "
            "persisted source-camera pixel authority."
        )
    centers = np.column_stack((boxes[:, 0] * width_px, boxes[:, 1] * height_px))
    signature_payload = {
        "group_path": path,
        "row_count": len(keys),
        "instance_key_sha256": _array_digest(keys),
        "frame_indices_sha256": _array_digest(frames),
        "bbox_norm_coords_sha256": _array_digest(boxes),
        "source_video_width": observed_width,
        "source_video_height": observed_height,
        "source_pixel_authority": expected_authority,
        "run_provenance": attrs.get("run_provenance"),
    }
    if manifest_digest is not None:
        signature_payload["canonical_run_manifest_payload_digest"] = manifest_digest
    return {
        "group_path": path,
        "row_count": len(keys),
        "centers": centers,
        "signature": _payload_sha256(signature_payload),
        "signature_payload": _canonical_copy(signature_payload),
    }


def _gate_disagreement(
    detections: Mapping[str, Any],
    *,
    acquisition_gate: tuple[float, float, float],
    palette_gate: tuple[float, float, float],
) -> dict[str, Any]:
    centers = np.asarray(detections["centers"], dtype=np.float64)
    acquisition_inside = (
        acquisition_gate[2]
        - np.hypot(centers[:, 0] - acquisition_gate[0], centers[:, 1] - acquisition_gate[1])
    ) >= 0.0
    palette_inside = (
        palette_gate[2]
        - np.hypot(centers[:, 0] - palette_gate[0], centers[:, 1] - palette_gate[1])
    ) >= 0.0
    acquisition_only = int(np.count_nonzero(acquisition_inside & ~palette_inside))
    palette_only = int(np.count_nonzero(palette_inside & ~acquisition_inside))
    total = int(len(centers))
    return {
        "status": "measured",
        "source_group_path": detections["group_path"],
        "source_signature": detections["signature"],
        "source_signature_payload": detections["signature_payload"],
        "row_count": total,
        "both_inside_count": int(np.count_nonzero(acquisition_inside & palette_inside)),
        "both_outside_count": int(np.count_nonzero(~acquisition_inside & ~palette_inside)),
        "acquisition_only_count": acquisition_only,
        "palette_only_count": palette_only,
        "exclusive_disagreement_count": acquisition_only + palette_only,
        "exclusive_disagreement_fraction": (
            float((acquisition_only + palette_only) / total) if total else 0.0
        ),
        "boundary_inclusion": "inclusive",
        "additional_palette_tolerance_px": 0.0,
    }


def _policy_record(policy_id: str) -> dict[str, Any]:
    if policy_id not in SUPPORTED_POLICY_IDS:
        raise ValueError(f"Unsupported geometry comparison policy: {policy_id!r}.")
    promoted = policy_id == MANUAL_REVIEW_POLICY_ID
    return {
        "policy_id": policy_id,
        "policy_version": 1,
        "automatic_selection_promoted": False,
        "manual_review_policy_available": promoted,
        "thresholds": _canonical_copy(_UNPROMOTED_THRESHOLDS),
        "threshold_source": (
            "not_applicable_manual_review_only"
            if promoted
            else "unpromoted_pending_frozen_canary_and_holdout"
        ),
        "remaining_canary_measurements": (
            [] if promoted else list(_CANARY_MEASUREMENTS)
        ),
        "state_to_action": {
            "corroborated_pass": "review_required_until_policy_promotion",
            "review_required": "review",
            "offline_fit_failed_but_acquisition_geometry_valid": "review_no_selection",
            "semantic_feature_incompatible": "review_no_same_feature_claim",
            "producer_geometry_invalid": "fail",
            "coordinate_or_extent_mismatch": "fail",
            "comparison_failed": "fail",
        },
        "acquisition_only_fallback_allowed": False,
    }


def _decision_record(
    *,
    policy: Mapping[str, Any],
    semantic_state: str,
    bindings_match: bool,
) -> dict[str, Any]:
    if not bindings_match:
        outcome = "coordinate_or_extent_mismatch"
        action = "fail"
        reasons = ["candidate_recording_or_coordinate_bindings_disagree"]
    elif semantic_state == "different_feature_confirmed":
        outcome = "semantic_feature_incompatible"
        action = "review"
        reasons = ["direct_physical_boundary_comparison_is_not_valid"]
    else:
        outcome = "review_required"
        action = "review"
        reasons = [
            (
                "manual_review_policy_requires_explicit_selection"
                if policy["policy_id"] == MANUAL_REVIEW_POLICY_ID
                else "automatic_thresholds_are_not_promoted"
            )
        ]
        if semantic_state == "projected_edges_unresolved":
            reasons.append("same_feature_correspondence_is_unresolved")
    return {
        "evidence_outcome": outcome,
        "workflow_action": action,
        "reason_codes": reasons,
        "candidate_selected": False,
        "candidates_mutated": False,
        "raw_detections_mutated": False,
    }


def build_arena_geometry_comparison_plan(
    source_zarr: str | Path,
    *,
    acquisition_candidate_run: str,
    palette_candidate_run: str,
    semantic_compatibility: str,
    policy_id: str = MANUAL_REVIEW_POLICY_ID,
    semantic_review: Mapping[str, Any] | None = None,
    detect_source_group_path: str | None = None,
    acquisition_boundary_edge_support: Mapping[str, Any] | None = None,
) -> ArenaGeometryComparisonPlan:
    """Build one content-addressed comparison without writing the archive."""

    semantic_state = str(semantic_compatibility).strip()
    if semantic_state not in SEMANTIC_COMPATIBILITY_STATES:
        raise ValueError(f"Unsupported semantic compatibility: {semantic_state!r}.")
    if semantic_state in {"same_feature_confirmed", "different_feature_confirmed"}:
        review = _canonical_copy(semantic_review)
        if not isinstance(review, Mapping):
            raise ValueError(f"{semantic_state} requires explicit reviewed evidence.")
        for name in ("reviewer", "reviewed_at_utc", "evidence_reason"):
            if not str(review.get(name) or "").strip():
                raise ValueError(f"semantic_review lacks {name}.")
        if set(review) != {"reviewer", "reviewed_at_utc", "evidence_reason"}:
            raise ValueError("semantic_review has unsupported fields.")
    else:
        review = _canonical_copy(semantic_review) if semantic_review is not None else None

    archive = Path(source_zarr).expanduser().resolve()
    root = open_zarr_root(archive, mode="r")
    acquisition = _candidate_snapshot(
        root,
        run_name=acquisition_candidate_run,
        expected_kind=ACQUISITION_CANDIDATE_KIND,
    )
    palette = _candidate_snapshot(
        root,
        run_name=palette_candidate_run,
        expected_kind=PALETTE_CANDIDATE_KIND,
    )
    acquisition_record = acquisition["record"]
    palette_record = palette["record"]
    acquisition_physical = _circle(
        acquisition_record["physical_inner_rim"], label="acquisition physical inner rim"
    )
    palette_observed = _circle(
        palette_record["observed_boundary"], label="Palette observed boundary"
    )
    acquisition_gate = _circle(
        acquisition_record["valid_detection_region"], label="acquisition detection gate"
    )
    palette_gate = _circle(
        palette_record["valid_detection_region"], label="Palette diagnostic gate"
    )
    acquisition_coordinate = acquisition_record["coordinate_binding"]
    palette_coordinate = palette_record["coordinate_binding"]
    acquisition_arena = acquisition_record["arena_binding"]
    palette_arena = palette_record["arena_binding"]
    bindings_match = (
        acquisition_coordinate == palette_coordinate and acquisition_arena == palette_arena
    )
    center_displacement = math.hypot(
        palette_observed[0] - acquisition_physical[0],
        palette_observed[1] - acquisition_physical[1],
    )
    gate_iou = _circle_iou(acquisition_gate, palette_gate)
    if semantic_state == "same_feature_confirmed":
        signed_radius_difference = palette_observed[2] - acquisition_physical[2]
        same_feature_metrics: Mapping[str, Any] | None = {
            "signed_radius_difference_px": signed_radius_difference,
            "absolute_radius_difference_px": abs(signed_radius_difference),
            "maximum_boundary_separation_px": center_displacement
            + abs(signed_radius_difference),
        }
    else:
        same_feature_metrics = None

    fit_source = palette_record["palette_fit_source"]
    windows = fit_source["windows"]
    fit_evidence = {
        "fit_report_sha256": fit_source["fit_report_sha256"],
        "fit_method": fit_source["fit_method"],
        "fit_evidence_contract": _canonical_copy(fit_source["fit_evidence_contract"]),
        "temporal_stability_px": _canonical_copy(fit_source["temporal_stability_px"]),
        "windows": {
            name: {
                "angular_support_fraction": windows[name]["angular_support_fraction"],
                "radial_residual_px": windows[name]["radial_residual_px"],
                "median_radial_gradient": windows[name]["median_radial_gradient"],
                "selected_candidate_id": windows[name]["selected_candidate_id"],
                "selection_reason": windows[name]["selection_reason"],
                "frozen_candidate_count": len(windows[name]["frozen_candidates"]),
            }
            for name in ("early", "middle", "late")
        },
    }
    edge_support = _canonical_copy(
        acquisition_boundary_edge_support
        if acquisition_boundary_edge_support is not None
        else fit_source.get("acquisition_boundary_edge_support")
        or {"status": "not_measured", "reason": "recording_image_probe_not_supplied"}
    )
    if not isinstance(edge_support, Mapping) or not str(edge_support.get("status") or ""):
        raise ValueError("acquisition_boundary_edge_support requires an explicit status.")
    if edge_support.get("status") == "measured":
        measured_circle = _circle(
            {"geometry": edge_support.get("geometry")},
            label="acquisition boundary edge-support geometry",
        )
        if measured_circle != acquisition_physical:
            raise ValueError(
                "Acquisition boundary edge support measured a different physical circle."
            )
        if _sha256_hex(
            edge_support.get("source_observation_sha256"),
            label="acquisition boundary edge-support observation digest",
        ) != _sha256_hex(
            acquisition_record["acquisition_source"]["source_observation_sha256"],
            label="acquisition candidate source-observation digest",
        ):
            raise ValueError(
                "Acquisition boundary edge support does not bind the producer observation."
            )

    detections = None
    operational = {
        "status": "not_measured",
        "reason": "exact_detection_source_not_supplied",
    }
    source_path: str | None = None
    source_signature: str | None = None
    if detect_source_group_path is not None:
        detections = _detection_snapshot(
            root,
            source_group_path=detect_source_group_path,
            coordinate_binding=acquisition_coordinate,
        )
        source_path = str(detections["group_path"])
        source_signature = str(detections["signature"])
        operational = _gate_disagreement(
            detections,
            acquisition_gate=acquisition_gate,
            palette_gate=palette_gate,
        )

    policy = _policy_record(policy_id)
    decision = _decision_record(
        policy=policy,
        semantic_state=semantic_state,
        bindings_match=bindings_match,
    )
    record = {
        "schema_id": COMPARISON_RECORD_SCHEMA_ID,
        "schema_version": COMPARISON_RECORD_SCHEMA_VERSION,
        "algorithm_version": COMPARISON_ALGORITHM_VERSION,
        "candidate_bindings": {
            "acquisition": {
                key: acquisition[key]
                for key in (
                    "run_name",
                    "candidate_id",
                    "candidate_kind",
                    "candidate_record_sha256",
                )
            },
            "palette": {
                key: palette[key]
                for key in (
                    "run_name",
                    "candidate_id",
                    "candidate_kind",
                    "candidate_record_sha256",
                )
            },
        },
        "recording_binding": {
            "arena_binding": _canonical_copy(acquisition_arena),
            "coordinate_binding": _canonical_copy(acquisition_coordinate),
            "bindings_match_exactly": bindings_match,
        },
        "observed_features": {
            "acquisition": "dish_inner_rim_water_side_edge",
            "palette": palette_record["observed_boundary"]["observed_feature"],
            "semantic_compatibility": semantic_state,
            "semantic_review": review,
        },
        "geometry": {
            "acquisition_physical_inner_rim": _circle_record(acquisition_physical),
            "palette_observed_boundary": _circle_record(palette_observed),
            "acquisition_valid_detection_region": _circle_record(acquisition_gate),
            "palette_diagnostic_detection_region": _circle_record(palette_gate),
            "center_displacement_native_px": center_displacement,
            "center_displacement_dish_top_rim_mm": None,
            "same_feature_physical_boundary_metrics": same_feature_metrics,
            "detection_gate_circle_iou": gate_iou,
            "detection_gate_mask_disagreement_fraction": 1.0 - gate_iou,
        },
        "independent_fit_evidence": fit_evidence,
        "acquisition_boundary_edge_support": edge_support,
        "operational_gate_disagreement": operational,
        "policy": policy,
        "decision": decision,
        "immutability": {
            "candidates_mutated": False,
            "raw_detections_mutated": False,
            "selection_performed": False,
        },
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    normalized = _canonical_copy(record)
    validate_arena_geometry_comparison_record(normalized)
    digest = _payload_sha256(normalized)
    comparison_id = f"arena_geometry_comparison_{digest[:20]}"
    provenance = build_writer_run_provenance(
        command="fisheye.utils.publish_arena_geometry_comparison",
        params={
            "comparison_id": comparison_id,
            "comparison_record_sha256": digest,
            "policy_id": policy_id,
            "semantic_compatibility": semantic_state,
        },
        input_run_ids={
            "acquisition_candidate": acquisition["run_name"],
            "acquisition_candidate_record_sha256": acquisition["candidate_record_sha256"],
            "palette_candidate": palette["run_name"],
            "palette_candidate_record_sha256": palette["candidate_record_sha256"],
            **(
                {
                    "detection_source": source_path,
                    "detection_source_signature": source_signature,
                }
                if source_path is not None
                else {}
            ),
        },
        cwd=Path.cwd(),
        include_system_context=False,
    )
    return ArenaGeometryComparisonPlan(
        source_zarr=archive,
        acquisition_candidate_run=acquisition["run_name"],
        acquisition_candidate_record_sha256=acquisition["candidate_record_sha256"],
        palette_candidate_run=palette["run_name"],
        palette_candidate_record_sha256=palette["candidate_record_sha256"],
        detect_source_group_path=source_path,
        detect_source_signature=source_signature,
        comparison_id=comparison_id,
        comparison_record_sha256=digest,
        comparison_record=normalized,
        target_run_path=archive / "analysis" / COMPARISON_RUNS_PARENT / comparison_id,
        run_provenance=provenance,
    )


def validate_arena_geometry_comparison_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("schema_id") != COMPARISON_RECORD_SCHEMA_ID
        or record.get("schema_version") != COMPARISON_RECORD_SCHEMA_VERSION
        or record.get("algorithm_version") != COMPARISON_ALGORITHM_VERSION
    ):
        raise ValueError("Unsupported arena-geometry comparison record.")
    bindings = record.get("candidate_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {"acquisition", "palette"}:
        raise ValueError("Comparison candidate bindings are incomplete.")
    for role, expected_kind in (
        ("acquisition", ACQUISITION_CANDIDATE_KIND),
        ("palette", PALETTE_CANDIDATE_KIND),
    ):
        candidate = bindings.get(role)
        if not isinstance(candidate, Mapping) or candidate.get("candidate_kind") != expected_kind:
            raise ValueError(f"Comparison {role} candidate binding is invalid.")
        if candidate.get("run_name") != candidate.get("candidate_id"):
            raise ValueError(f"Comparison {role} candidate identity disagrees.")
        for name in ("run_name", "candidate_record_sha256"):
            if not str(candidate.get(name) or "").strip():
                raise ValueError(f"Comparison {role} candidate lacks {name}.")
    features = record.get("observed_features")
    if not isinstance(features, Mapping):
        raise ValueError("Comparison lacks observed-feature semantics.")
    semantic_state = features.get("semantic_compatibility")
    if semantic_state not in SEMANTIC_COMPATIBILITY_STATES:
        raise ValueError("Comparison semantic compatibility is unsupported.")
    geometry = record.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Comparison lacks geometry metrics.")
    same_feature = geometry.get("same_feature_physical_boundary_metrics")
    if (semantic_state == "same_feature_confirmed") != isinstance(same_feature, Mapping):
        raise ValueError("Physical radius metrics require confirmed same-feature evidence.")
    policy = record.get("policy")
    if not isinstance(policy, Mapping) or policy.get("policy_id") not in SUPPORTED_POLICY_IDS:
        raise ValueError("Comparison policy is unsupported.")
    if policy.get("automatic_selection_promoted") is not False:
        raise ValueError("Version-1 automatic selection thresholds are not promoted.")
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, Mapping) or dict(thresholds) != _UNPROMOTED_THRESHOLDS:
        raise ValueError("Comparison thresholds do not match the unpromoted policy.")
    decision = record.get("decision")
    if not isinstance(decision, Mapping) or decision.get("candidate_selected") is not False:
        raise ValueError("Comparison may not select a candidate.")
    immutability = record.get("immutability")
    if not isinstance(immutability, Mapping) or dict(immutability) != {
        "candidates_mutated": False,
        "raw_detections_mutated": False,
        "selection_performed": False,
    }:
        raise ValueError("Comparison immutability declaration is invalid.")
    if record.get("canonicalization") != "canonical_json_sort_keys_v1":
        raise ValueError("Comparison canonicalization is unsupported.")
    if _canonical_copy(record) != dict(record):
        raise ValueError("Comparison record is not strict canonical JSON data.")


def _run_attrs(plan: ArenaGeometryComparisonPlan) -> dict[str, Any]:
    return {
        "schema_id": COMPARISON_RUN_SCHEMA_ID,
        "schema_version": COMPARISON_RUN_SCHEMA_VERSION,
        "comparison_id": plan.comparison_id,
        "comparison_record": _canonical_copy(plan.comparison_record),
        "comparison_record_sha256": plan.comparison_record_sha256,
        "acquisition_candidate_run": plan.acquisition_candidate_run,
        "acquisition_candidate_record_sha256": plan.acquisition_candidate_record_sha256,
        "palette_candidate_run": plan.palette_candidate_run,
        "palette_candidate_record_sha256": plan.palette_candidate_record_sha256,
        "detect_source_group_path": plan.detect_source_group_path,
        "detect_source_signature": plan.detect_source_signature,
        "run_provenance": _canonical_copy(plan.run_provenance),
        "comparison_status": plan.comparison_record["decision"]["evidence_outcome"],
        "review_required": plan.comparison_record["decision"]["workflow_action"] == "review",
        "stage_selector_eligible": False,
        "candidate_selected": False,
        "candidates_mutated": False,
        "raw_detections_mutated": False,
    }


def validate_arena_geometry_comparison_run(
    run_path: str | Path,
    *,
    expected_plan: ArenaGeometryComparisonPlan,
    require_complete: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    path = Path(run_path).expanduser().resolve()
    try:
        group = open_zarr_root(path, mode="r")
        attrs = dict(group.attrs)
        for name, value in _run_attrs(expected_plan).items():
            if name != "run_provenance" and attrs.get(name) != value:
                errors.append(f"{name} mismatch")
        record = attrs.get("comparison_record")
        if isinstance(record, Mapping):
            validate_arena_geometry_comparison_record(record)
            if _payload_sha256(record) != attrs.get("comparison_record_sha256"):
                errors.append("comparison record digest mismatch")
        else:
            errors.append("comparison_record missing")
        provenance = validate_run_provenance(attrs.get("run_provenance"))
        if not provenance.valid:
            errors.extend(f"run provenance: {item}" for item in provenance.errors)
        if list(group.array_keys()) or list(group.group_keys()):
            errors.append("comparison run must be metadata-only")
        status = attrs.get("palette_run_completion_status")
        if require_complete and status != "complete":
            errors.append("comparison run is not complete")
        elif status not in {"running", "complete"}:
            errors.append("comparison run has invalid completion status")
        if attrs.get("stage_selector_eligible") is not False:
            errors.append("comparison evidence must remain selector-ineligible")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "comparison_id": expected_plan.comparison_id,
        "comparison_record_sha256": expected_plan.comparison_record_sha256,
        "run_path": str(path),
    }


def _revalidate_sources(plan: ArenaGeometryComparisonPlan) -> dict[str, Any]:
    root = open_zarr_root(plan.source_zarr, mode="r")
    acquisition = _candidate_snapshot(
        root,
        run_name=plan.acquisition_candidate_run,
        expected_kind=ACQUISITION_CANDIDATE_KIND,
    )
    palette = _candidate_snapshot(
        root,
        run_name=plan.palette_candidate_run,
        expected_kind=PALETTE_CANDIDATE_KIND,
    )
    if (
        acquisition["candidate_record_sha256"]
        != plan.acquisition_candidate_record_sha256
        or palette["candidate_record_sha256"] != plan.palette_candidate_record_sha256
    ):
        raise RuntimeError("Geometry candidates changed during comparison publication.")
    if plan.detect_source_group_path is not None:
        coordinate = acquisition["record"]["coordinate_binding"]
        detection = _detection_snapshot(
            root,
            source_group_path=plan.detect_source_group_path,
            coordinate_binding=coordinate,
        )
        if detection["signature"] != plan.detect_source_signature:
            raise RuntimeError("Detection source changed during comparison publication.")
    return {
        "status": "current",
        "acquisition_candidate_record_sha256": acquisition["candidate_record_sha256"],
        "palette_candidate_record_sha256": palette["candidate_record_sha256"],
        "detect_source_signature": plan.detect_source_signature,
    }


def publish_arena_geometry_comparison(
    plan: ArenaGeometryComparisonPlan,
    *,
    scratch_root: str | Path,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Publish immutable selector-ineligible comparison evidence."""

    if plan.target_run_path.exists():
        existing = validate_arena_geometry_comparison_run(
            plan.target_run_path,
            expected_plan=plan,
            require_complete=True,
        )
        if not existing["valid"]:
            raise FileExistsError(f"Existing comparison is not the expected run: {existing}")
        return {"published": False, "status": "already_complete", **existing}
    _revalidate_sources(plan)
    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"palette-{plan.comparison_id}-", dir=scratch
    ) as temporary:
        local_run = Path(temporary) / plan.comparison_id
        local = zarr.open_group(str(local_run), mode="w", zarr_format=3)
        local.attrs.update(json_attr_safe(_run_attrs(plan)))
        mark_run_started(local, run_name=plan.comparison_id, stage="arena_geometry_comparison")

        def validate(path: Path) -> dict[str, Any]:
            return validate_arena_geometry_comparison_run(path, expected_plan=plan)

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, COMPARISON_RUNS_PARENT),)

        def complete(_root: zarr.Group, parent: zarr.Group, run: zarr.Group) -> None:
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=plan.comparison_id,
                run_provenance=plan.run_provenance,
            )

        def verify(root: zarr.Group) -> None:
            parent = root[f"analysis/{COMPARISON_RUNS_PARENT}"]
            run = parent[plan.comparison_id]
            if (
                run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
                or parent.attrs.get("latest") == plan.comparison_id
                or parent.attrs.get("latest_complete") == plan.comparison_id
            ):
                raise RuntimeError(
                    "Comparison evidence became selector-visible during publication."
                )

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=local_run,
                target_run_path=plan.target_run_path,
                run_name=plan.comparison_id,
                lock_suffix="arena-geometry-comparison-publish",
                publish_schema_id=COMPARISON_PUBLISH_SCHEMA_ID,
                policy="node_local_metadata_only_atomic_publish_v1",
                rollback_policy="retain_failed_public_tombstone",
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            after_rename=lambda _root, _run: {
                "source_revision_audit": _revalidate_sources(plan)
            },
            payload_metadata={"algorithm_version": COMPARISON_ALGORITHM_VERSION},
        )
    final = validate_arena_geometry_comparison_run(
        plan.target_run_path,
        expected_plan=plan,
        require_complete=True,
    )
    if not final["valid"]:
        raise RuntimeError(f"Final comparison validation failed: {final}")
    return {"published": True, "status": "complete", "publication": publication, **final}


__all__ = [
    "ArenaGeometryComparisonPlan",
    "COMPARISON_RUNS_PARENT",
    "CORROBORATED_ACQUISITION_POLICY_ID",
    "MANUAL_REVIEW_POLICY_ID",
    "SEMANTIC_COMPATIBILITY_STATES",
    "build_arena_geometry_comparison_plan",
    "publish_arena_geometry_comparison",
    "validate_arena_geometry_comparison_record",
    "validate_arena_geometry_comparison_run",
]
