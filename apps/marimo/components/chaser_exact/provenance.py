"""Immutable provenance helpers for exact-chaser display projections."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

TRACE_DISPLAY_ALGORITHM = "source_order_bucket_first_last_min_max_missing_break_v1"
TRACE_MAX_POINTS = 6_000
TRAJECTORY_DISPLAY_ALGORITHM = "source_order_uniform_plus_coordinate_extrema_v1"
TRAJECTORY_MAX_POINTS = 15_000


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
) -> Mapping[str, Any]:
    """Build the readable, immutable identity record shared by exact views."""

    return freeze(
        {
            "recording_id": spatial.recording_id,
            "bundle_run_path": spatial.run_path,
            "bundle_manifest_sha256": spatial.manifest_sha256,
            "radial_run_paths": [radial.run_path for radial in radials],
            "radial_manifest_sha256": [radial.manifest_sha256 for radial in radials],
            "relative_bindings": [plain(value) for value in relative_bindings],
            "relative_binding_proofs": [
                plain(proof.provenance_record()) for proof in relative_binding_proofs
            ],
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
                }
                if escape_freeze is not None
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
