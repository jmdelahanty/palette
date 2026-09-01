"""Immutable provenance helpers for exact-chaser display projections."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

TRACE_DISPLAY_ALGORITHM = "source_order_bucket_first_last_min_max_missing_break_v1"
TRACE_MAX_POINTS = 6_000
TRAJECTORY_DISPLAY_ALGORITHM = "source_order_uniform_plus_coordinate_extrema_v1"
TRAJECTORY_MAX_POINTS = 15_000


def _verification(handle: Any) -> Mapping[str, Any]:
    """Return explicit loader authority, tolerating older deep-audit fixtures."""

    deep = getattr(handle, "deep_audited", False) is True
    return {
        "verification_mode": getattr(
            handle, "verification_mode", "deep_audit" if deep else "unverified"
        ),
        "receipt_digest": getattr(handle, "receipt_digest", None),
        "verified_array_names": list(getattr(handle, "verified_array_names", ())),
    }


def plain(value: Any) -> Any:
    """Return JSON-compatible display provenance without mutating the source."""

    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def freeze(value: Any) -> Any:
    """Recursively freeze projection metadata exposed to renderers."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def build_projection_provenance(
    *,
    spatial: Any,
    radials: Sequence[Any],
    relative_bindings: Sequence[Mapping[str, Any]],
    relative_binding_proofs: Sequence[Any],
    controller_trials: Any | None = None,
    generalized_bout_response: Any | None = None,
    escape_freeze: Any | None = None,
    gaze_tracking: Any | None = None,
    epoch_behavior: Any | None = None,
    body_alignment_by_distance: Any | None = None,
    relatives: Sequence[Any] | None = None,
    chaser_appearance: Any | None = None,
    projection_verification_mode: str = "deep_audit",
    projection_receipt_sha256: str | None = None,
    validated_behavior_bundle_path: str | None = None,
    validated_behavior_bundle_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Build the readable, immutable identity record shared by exact views."""

    return freeze(
        {
            "recording_id": spatial.recording_id,
            "bundle_run_path": spatial.run_path,
            "bundle_manifest_sha256": spatial.manifest_sha256,
            "projection_verification": {
                "verification_mode": projection_verification_mode,
                "projection_receipt_sha256": projection_receipt_sha256,
                "displayed_array_policy": (
                    "manifest_declared_content_sha256_per_consumed_array"
                ),
            },
            "validated_recording_behavior_bundle": (
                {
                    "bundle_path": validated_behavior_bundle_path,
                    "bundle_sha256": validated_behavior_bundle_sha256,
                    "role": "exact_source_choice_and_compatibility_authority",
                }
                if validated_behavior_bundle_path is not None
                else None
            ),
            "spatial_verification": _verification(spatial),
            "radial_run_paths": [radial.run_path for radial in radials],
            "radial_manifest_sha256": [radial.manifest_sha256 for radial in radials],
            "radial_verification": [_verification(radial) for radial in radials],
            "relative_verification": [
                {
                    "run_path": relative.run_path,
                    "verification_mode": relative.verification_mode,
                    "receipt_digest": relative.receipt_digest,
                    "verified_array_names": list(relative.verified_array_names),
                }
                for relative in (relatives or ())
            ],
            "relative_bindings": [plain(value) for value in relative_bindings],
            "relative_binding_proofs": [
                plain(proof.provenance_record()) for proof in relative_binding_proofs
            ],
            "chaser_appearance_binding": (
                plain(chaser_appearance.provenance_record())
                if chaser_appearance is not None
                else None
            ),
            "controller_trial_binding": (
                {
                    "run_path": controller_trials.run_path,
                    "manifest_sha256": controller_trials.manifest_sha256,
                    "scientific_payload_sha256": (
                        controller_trials.scientific_payload_sha256
                    ),
                    "source_relative_frame": plain(
                        controller_trials.scientific_manifest.get(
                            "source_relative_frame"
                        )
                    ),
                    "semantic_selection": plain(
                        controller_trials.scientific_manifest.get("semantic_selection")
                    ),
                    "deep_audited": controller_trials.deep_audited,
                    **_verification(controller_trials),
                }
                if controller_trials is not None
                else None
            ),
            "generalized_bout_response_binding": (
                {
                    "run_path": generalized_bout_response.run_path,
                    "manifest_sha256": generalized_bout_response.manifest_sha256,
                    "scientific_payload_sha256": (
                        generalized_bout_response.scientific_payload_sha256
                    ),
                    "sources": plain(
                        generalized_bout_response.scientific_manifest.get("sources")
                    ),
                    "body_extension_present": generalized_bout_response.scientific_manifest.get(
                        "scientific_schema", {}
                    ).get(
                        "body_extension_present"
                    ),
                    "deep_audited": generalized_bout_response.deep_audited,
                    **_verification(generalized_bout_response),
                }
                if generalized_bout_response is not None
                else None
            ),
            "escape_freeze_binding": (
                {
                    "run_path": escape_freeze.run_path,
                    "manifest_sha256": escape_freeze.manifest_sha256,
                    "scientific_payload_sha256": (
                        escape_freeze.scientific_payload_sha256
                    ),
                    "sources": plain(escape_freeze.scientific_manifest.get("sources")),
                    "parameters": plain(
                        escape_freeze.scientific_manifest.get("parameters")
                    ),
                    "method_id": escape_freeze.scientific_manifest.get(
                        "scientific_schema", {}
                    ).get("method_id"),
                    "deep_audited": escape_freeze.deep_audited,
                    **_verification(escape_freeze),
                }
                if escape_freeze is not None
                else None
            ),
            "gaze_tracking_binding": (
                {
                    "run_path": gaze_tracking.run_path,
                    "manifest_sha256": gaze_tracking.manifest_sha256,
                    "scientific_payload_sha256": (
                        gaze_tracking.scientific_payload_sha256
                    ),
                    "sources": plain(gaze_tracking.scientific_manifest.get("sources")),
                    "parameters": plain(
                        gaze_tracking.scientific_manifest.get("parameters")
                    ),
                    "method_id": gaze_tracking.scientific_manifest.get(
                        "scientific_schema", {}
                    ).get("method_id"),
                    "deep_audited": gaze_tracking.deep_audited,
                    **_verification(gaze_tracking),
                }
                if gaze_tracking is not None
                else None
            ),
            "epoch_behavior_binding": (
                {
                    "run_path": epoch_behavior.run_path,
                    "manifest_sha256": epoch_behavior.manifest_sha256,
                    "payload_digest": epoch_behavior.payload_digest,
                    "sources": plain(epoch_behavior.manifest.get("sources")),
                    "parameters": plain(epoch_behavior.manifest.get("parameters")),
                    "deep_audited": epoch_behavior.deep_audited,
                    **_verification(epoch_behavior),
                }
                if epoch_behavior is not None
                else None
            ),
            "body_alignment_by_distance_binding": (
                {
                    "run_path": body_alignment_by_distance.run_path,
                    "manifest_sha256": body_alignment_by_distance.manifest_sha256,
                    "scientific_payload_sha256": (
                        body_alignment_by_distance.scientific_payload_sha256
                    ),
                    "sources": plain(
                        body_alignment_by_distance.scientific_manifest.get("sources")
                    ),
                    "distance_bin_recipe": plain(
                        body_alignment_by_distance.scientific_manifest.get(
                            "distance_bin_recipe"
                        )
                    ),
                    "deep_audited": body_alignment_by_distance.deep_audited,
                    **_verification(body_alignment_by_distance),
                }
                if body_alignment_by_distance is not None
                else None
            ),
            "adapter_semantics": (
                "read_only_exact_children_no_selector_no_interpolation"
            ),
            "display_trace_algorithm": TRACE_DISPLAY_ALGORITHM,
            "display_trace_max_points_per_series": TRACE_MAX_POINTS,
            "display_trajectory_algorithm": TRAJECTORY_DISPLAY_ALGORITHM,
            "display_trajectory_max_points_per_panel": TRAJECTORY_MAX_POINTS,
        }
    )


__all__ = [
    "TRACE_DISPLAY_ALGORITHM",
    "TRACE_MAX_POINTS",
    "TRAJECTORY_DISPLAY_ALGORITHM",
    "TRAJECTORY_MAX_POINTS",
    "build_projection_provenance",
    "freeze",
    "plain",
]
