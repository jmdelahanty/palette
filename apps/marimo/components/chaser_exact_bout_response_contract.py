"""Closed scientific contract shared by bout-response discovery and loading."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

BOUT_RESPONSE_PARENT = "analysis/generalized_chaser_bout_response_runs"
SEMANTIC_PARENT = "analysis/protocol_semantic_chaser_selection_runs"
MOTION_PARENT = "analysis/track_kinematics_runs/provider"
SWIM_BOUT_PARENT = "analysis/swim_bout_runs"

EXPECTED_POLICY = {
    "bout_signal": "one_explicit_default_signal_only",
    "bout_attachment": "exact_acquisition_frame_identity",
    "trial_attachment": "onset_row_exact_controller_trial_membership",
    "trial_envelope": "retained_for_visualization_and_censoring_not_event_membership",
    "rate_denominator": "valid_transition_time_in_distance_band",
    "directed_metrics": "optional_body_frame_extension_no_motion_heading_fallback",
    "unattached_bouts": "retained_with_reason_code",
}
EXPECTED_REGISTRIES = {
    "semantic_role": {
        "1": "chaser_pre",
        "2": "chaser_training",
        "3": "chaser_post",
    },
    "attachment_reason": {
        "0": "valid_or_trial_optional",
        "1": "frame_unavailable",
        "2": "outside_semantic_selection",
        "3": "controller_trial_unavailable_at_onset",
    },
}
FORBIDDEN_SELECTORS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "current_run",
        "selected",
        "authoritative",
        "authoritative_run",
        "default",
    }
)


class ExactBoutResponseContractError(ValueError):
    """One closed generalized bout-response identity or policy is invalid."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactBoutResponseContractError(f"{label} must be one object.")
    return value


def require_digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactBoutResponseContractError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def exact_child_path(value: Any, *, parent: str, label: str) -> tuple[str, str]:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactBoutResponseContractError(f"{label} must be one exact child path.")
    prefix = f"{parent}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name in {".", ".."}
        or name.casefold() in FORBIDDEN_SELECTORS
    ):
        raise ExactBoutResponseContractError(
            f"{label} must be one exact child below {parent!r}."
        )
    return value, name


def normalize_motion_binding(value: Any) -> Mapping[str, Any]:
    motion = require_mapping(value, label="bout-response motion source")
    if set(motion) != {"run_path", "manifest_sha256", "relative_frame_projection"}:
        raise ExactBoutResponseContractError(
            "Bout-response motion source has an unsupported field set."
        )
    run_path, _ = exact_child_path(
        motion.get("run_path"), parent=MOTION_PARENT, label="motion run"
    )
    projection = require_mapping(
        motion.get("relative_frame_projection"), label="motion frame projection"
    )
    expected_fields = {
        "schema_id",
        "schema_version",
        "join_key",
        "join_policy",
        "provider_frame_count",
        "relative_frame_count",
        "matched_relative_frame_count",
        "missing_relative_frame_count",
        "provider_only_frame_count",
        "provider_frame_ids_sha256",
        "relative_frame_ids_sha256",
        "provider_row_index_by_relative_frame_sha256",
        "provider_frame_present_sha256",
        "fallback",
    }
    if set(projection) != expected_fields:
        raise ExactBoutResponseContractError(
            "Bout-response motion frame projection has an unsupported field set."
        )
    for field in (
        "provider_frame_ids_sha256",
        "relative_frame_ids_sha256",
        "provider_row_index_by_relative_frame_sha256",
        "provider_frame_present_sha256",
    ):
        require_digest(projection.get(field), label=f"motion projection {field}")
    try:
        provider_count = int(projection.get("provider_frame_count", -1))
        relative_count = int(projection.get("relative_frame_count", -1))
        matched_count = int(projection.get("matched_relative_frame_count", -1))
        missing_count = int(projection.get("missing_relative_frame_count", -1))
        provider_only_count = int(projection.get("provider_only_frame_count", -1))
    except (TypeError, ValueError) as exc:
        raise ExactBoutResponseContractError(
            "Bout-response motion projection counts are invalid."
        ) from exc
    if (
        projection.get("schema_id")
        != "palette.provider_motion.relative_frame_projection"
        or projection.get("schema_version") != 1
        or projection.get("join_key") != "exact_acquisition_frame_id"
        or projection.get("join_policy")
        != "left_join_missing_provider_rows_invalid_no_interpolation"
        or projection.get("fallback") != "prohibited"
        or min(
            provider_count,
            relative_count,
            matched_count,
            missing_count,
            provider_only_count,
        )
        < 0
        or matched_count + missing_count != relative_count
        or matched_count + provider_only_count != provider_count
    ):
        raise ExactBoutResponseContractError(
            "Bout-response motion frame projection is incompatible."
        )
    return {
        "run_path": run_path,
        "manifest_sha256": require_digest(
            motion.get("manifest_sha256"), label="motion manifest digest"
        ),
        "relative_frame_projection": dict(projection),
    }


def normalize_swim_bout_binding(value: Any) -> Mapping[str, Any]:
    binding = require_mapping(value, label="bout-response swim-bout source")
    if set(binding) != {"run_path", "lineage_sha256", "signal_id", "signal_level"}:
        raise ExactBoutResponseContractError(
            "Bout-response swim-bout source has an unsupported field set."
        )
    run_path, _ = exact_child_path(
        binding.get("run_path"), parent=SWIM_BOUT_PARENT, label="swim-bout run"
    )
    signal_id = binding.get("signal_id")
    signal_level = binding.get("signal_level")
    if type(signal_id) is not int or signal_id < 0:
        raise ExactBoutResponseContractError(
            "Bout-response source_signal_id must be one non-negative integer."
        )
    if (
        type(signal_level) is not str
        or not signal_level
        or signal_level != signal_level.strip()
    ):
        raise ExactBoutResponseContractError(
            "Bout-response signal level must be one exact non-empty string."
        )
    return {
        "run_path": run_path,
        "lineage_sha256": require_digest(
            binding.get("lineage_sha256"), label="swim-bout lineage digest"
        ),
        "signal_id": signal_id,
        "signal_level": signal_level,
    }


def distance_edges(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 65:
        raise ExactBoutResponseContractError(
            "Bout-response distance-bin edges are invalid."
        )
    edges: list[float] = []
    for index, item in enumerate(value):
        if item is None:
            if index != len(value) - 1:
                raise ExactBoutResponseContractError(
                    "Only the final bout-response distance-bin edge may be open."
                )
            edges.append(float("inf"))
        elif type(item) in {int, float} and np.isfinite(float(item)):
            edges.append(float(item))
        else:
            raise ExactBoutResponseContractError(
                "Bout-response distance-bin edges are invalid."
            )
    if np.any(np.diff(np.asarray(edges, dtype=np.float64)) <= 0):
        raise ExactBoutResponseContractError(
            "Bout-response distance-bin edges are not increasing."
        )
    return tuple(edges)


def validate_scientific_manifest(
    value: Any,
    *,
    expected_scientific_payload_sha256: str,
    expected_n_frames: int,
    expected_n_chasers: int,
    expected_relative_binding: Mapping[str, Any],
    expected_semantic_manifest_sha256: str,
    expected_controller_payload_sha256: str,
) -> Mapping[str, Any]:
    """Validate one complete version-1 scientific manifest and source join."""

    scientific = require_mapping(value, label="bout-response scientific manifest")
    expected_payload = require_digest(
        expected_scientific_payload_sha256,
        label="bout-response scientific payload digest",
    )
    body = dict(scientific)
    observed_payload = body.pop("payload_digest", None)
    if (
        observed_payload != expected_payload
        or canonical_json_sha256(_plain(body)) != observed_payload
    ):
        raise ExactBoutResponseContractError(
            "Bout-response scientific payload digest is stale."
        )
    schema = require_mapping(
        scientific.get("scientific_schema"), label="scientific schema"
    )
    if (
        set(schema)
        != {
            "schema_id",
            "schema_version",
            "method_id",
            "row_unit",
            "summary_unit",
            "body_extension_present",
        }
        or schema.get("schema_id")
        != "palette.analysis.generalized_chaser_bout_response"
        or schema.get("schema_version") != 1
        or schema.get("method_id")
        != "exact_signal_bout_x_chaser_distance_motion_with_body_extension_v1"
        or schema.get("row_unit") != "selected_swim_bout_x_chaser"
        or schema.get("summary_unit") != "semantic_role_x_chaser_x_distance_band"
        or type(schema.get("body_extension_present")) is not bool
    ):
        raise ExactBoutResponseContractError(
            "Bout-response scientific schema is incompatible."
        )
    policy = require_mapping(scientific.get("policy"), label="policy")
    registries = require_mapping(
        scientific.get("identity_registries"), label="identity registries"
    )
    if dict(policy) != EXPECTED_POLICY or dict(registries) != EXPECTED_REGISTRIES:
        raise ExactBoutResponseContractError(
            "Bout-response policy or identity registries are incompatible."
        )
    edges = distance_edges(scientific.get("distance_bin_edges_mm"))
    dimensions = require_mapping(scientific.get("dimensions"), label="dimensions")
    try:
        n_frames = int(dimensions.get("n_frames", -1))
        n_chasers = int(dimensions.get("n_chasers", -1))
        n_bouts = int(dimensions.get("n_bouts", -1))
        n_rows = int(dimensions.get("n_bout_chaser_rows", -1))
        n_summary = int(dimensions.get("n_summary_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ExactBoutResponseContractError(
            "Bout-response dimensions are invalid."
        ) from exc
    if (
        n_frames != expected_n_frames
        or n_chasers != expected_n_chasers
        or n_bouts < 0
        or n_rows != n_bouts * n_chasers
        or n_summary != 3 * n_chasers * (len(edges) - 1)
    ):
        raise ExactBoutResponseContractError(
            "Bout-response dimensions are incompatible."
        )
    sources = require_mapping(scientific.get("sources"), label="sources")
    if set(sources) != {
        "relative_frame",
        "motion",
        "swim_bouts",
        "semantic_selection_manifest_sha256",
        "controller_trial_payload_sha256",
    }:
        raise ExactBoutResponseContractError(
            "Bout-response sources have an unsupported field set."
        )
    try:
        relative_proof = require_same_exact_relative_frame_child(
            expected_relative_binding,
            sources.get("relative_frame"),
            expected_label="spatial keypoint relative-frame binding",
            observed_label="bout-response relative-frame binding",
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactBoutResponseContractError(str(exc)) from exc
    semantic_digest = require_digest(
        sources.get("semantic_selection_manifest_sha256"),
        label="bout-response semantic-selection digest",
    )
    controller_payload = require_digest(
        sources.get("controller_trial_payload_sha256"),
        label="bout-response controller-trial payload digest",
    )
    if (
        semantic_digest != expected_semantic_manifest_sha256
        or controller_payload != expected_controller_payload_sha256
    ):
        raise ExactBoutResponseContractError(
            "Bout-response semantic or controller source differs from its bundle."
        )
    motion = normalize_motion_binding(sources.get("motion"))
    swim_bouts = normalize_swim_bout_binding(sources.get("swim_bouts"))
    if int(motion["relative_frame_projection"]["relative_frame_count"]) != n_frames:
        raise ExactBoutResponseContractError(
            "Bout-response motion projection uses another relative-frame length."
        )
    return {
        "source_relative_frame": dict(relative_proof.normalized_identity),
        "source_motion": motion,
        "source_swim_bouts": swim_bouts,
        "semantic_selection_manifest_sha256": semantic_digest,
        "controller_trial_payload_sha256": controller_payload,
        "body_extension_present": schema["body_extension_present"],
        "n_frames": n_frames,
        "n_chasers": n_chasers,
        "n_bouts": n_bouts,
        "n_rows": n_rows,
        "n_summary": n_summary,
        "distance_edges_mm": edges,
    }


def normalized_relative_binding(value: Any, *, label: str) -> Mapping[str, str]:
    try:
        return dict(
            validate_exact_relative_frame_binding(
                value, label=label
            ).normalized_identity
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactBoutResponseContractError(str(exc)) from exc


__all__ = [
    "BOUT_RESPONSE_PARENT",
    "EXPECTED_POLICY",
    "EXPECTED_REGISTRIES",
    "ExactBoutResponseContractError",
    "SEMANTIC_PARENT",
    "distance_edges",
    "exact_child_path",
    "normalize_motion_binding",
    "normalize_swim_bout_binding",
    "normalized_relative_binding",
    "require_digest",
    "require_mapping",
    "validate_scientific_manifest",
]
