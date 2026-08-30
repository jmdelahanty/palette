"""Exact generalized bout-response option binding and audited loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
)

from ..chaser_exact_bout_response_contract import (
    BOUT_RESPONSE_PARENT,
    SEMANTIC_PARENT,
    ExactBoutResponseContractError,
    exact_child_path,
    normalize_motion_binding,
    normalize_swim_bout_binding,
    normalized_relative_binding,
    require_digest,
    require_mapping,
    validate_scientific_manifest,
)
from ..registry import InteractiveSpecOption
from .provenance import freeze, plain


class ExactBoutResponseProjectionError(ValueError):
    """A bout-response option or source is not one compatible exact child."""


def _projection_error(exc: Exception) -> ExactBoutResponseProjectionError:
    return ExactBoutResponseProjectionError(str(exc))


def option_bout_response_binding(option: InteractiveSpecOption) -> Mapping[str, Any]:
    """Validate the closed synthesized binding for one bout-response view."""

    try:
        analysis_bindings = require_mapping(
            option.spec.get("analysis_bindings"), label="analysis bindings"
        )
        binding = require_mapping(
            analysis_bindings.get("generalized_bout_response"),
            label="generalized bout-response analysis binding",
        )
        if set(binding) != {
            "run_path",
            "manifest_sha256",
            "scientific_payload_sha256",
            "source_relative_frame",
            "source_motion",
            "source_swim_bouts",
            "semantic_selection_manifest_sha256",
            "controller_trial_payload_sha256",
            "body_extension_present",
        }:
            raise ExactBoutResponseContractError(
                "Generalized bout-response binding has an unsupported field set."
            )
        run_path, _ = exact_child_path(
            binding.get("run_path"),
            parent=BOUT_RESPONSE_PARENT,
            label="generalized bout-response run",
        )
        body_present = binding.get("body_extension_present")
        if type(body_present) is not bool:
            raise ExactBoutResponseContractError(
                "Bout-response body-extension declaration must be boolean."
            )
        return freeze(
            {
                "run_path": run_path,
                "manifest_sha256": require_digest(
                    binding.get("manifest_sha256"),
                    label="bout-response manifest digest",
                ),
                "scientific_payload_sha256": require_digest(
                    binding.get("scientific_payload_sha256"),
                    label="bout-response scientific payload digest",
                ),
                "source_relative_frame": normalized_relative_binding(
                    binding.get("source_relative_frame"),
                    label="bout-response option relative-frame binding",
                ),
                "source_motion": normalize_motion_binding(binding.get("source_motion")),
                "source_swim_bouts": normalize_swim_bout_binding(
                    binding.get("source_swim_bouts")
                ),
                "semantic_selection_manifest_sha256": require_digest(
                    binding.get("semantic_selection_manifest_sha256"),
                    label="bout-response semantic-selection digest",
                ),
                "controller_trial_payload_sha256": require_digest(
                    binding.get("controller_trial_payload_sha256"),
                    label="bout-response controller-trial payload digest",
                ),
                "body_extension_present": body_present,
            }
        )
    except ExactBoutResponseContractError as exc:
        raise _projection_error(exc) from exc


def load_exact_bout_response(
    archive: Path,
    option: InteractiveSpecOption,
    *,
    spatial: Any,
    expected_relative_binding: Mapping[str, Any],
    relative: Any,
    controller_trials: Any,
    direct_validation_receipt: str | Path | None = None,
    required_array_names: Collection[str] | None = None,
) -> Any:
    """Load one verified bout-response successor and check every source join."""

    binding = option_bout_response_binding(option)
    try:
        require_same_exact_relative_frame_child(
            expected_relative_binding,
            binding["source_relative_frame"],
            expected_label="spatial keypoint relative-frame binding",
            observed_label="bout-response option relative-frame binding",
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactBoutResponseProjectionError(str(exc)) from exc
    try:
        spatial_sources = require_mapping(
            spatial.scientific_manifest.get("sources"), label="spatial sources"
        )
        semantic = require_mapping(
            spatial_sources.get("protocol_semantic_selection"),
            label="spatial semantic selection",
        )
        exact_child_path(
            semantic.get("run_path"),
            parent=SEMANTIC_PARENT,
            label="semantic selection",
        )
        semantic_digest = require_digest(
            semantic.get("manifest_sha256"),
            label="spatial semantic-selection digest",
        )
    except ExactBoutResponseContractError as exc:
        raise _projection_error(exc) from exc
    if binding["semantic_selection_manifest_sha256"] != semantic_digest:
        raise ExactBoutResponseProjectionError(
            "Bout-response option uses another semantic selection."
        )
    if (
        controller_trials.successor_kind != "controller_chase_trials"
        or binding["controller_trial_payload_sha256"]
        != controller_trials.scientific_payload_sha256
    ):
        raise ExactBoutResponseProjectionError(
            "Bout-response option uses another controller-trial payload."
        )
    controller_trials.require_verified_authority()
    try:
        run_path, run_name = exact_child_path(
            binding["run_path"],
            parent=BOUT_RESPONSE_PARENT,
            label="generalized bout-response run",
        )
    except ExactBoutResponseContractError as exc:
        raise _projection_error(exc) from exc
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="generalized_chaser_bout_response",
        run_name=run_name,
        expected_recording_id=spatial.recording_id,
        deep_audit=direct_validation_receipt is None,
        direct_validation_receipt=direct_validation_receipt,
        required_array_names=required_array_names,
    )
    if (
        handle.run_path != run_path
        or handle.manifest_sha256 != binding["manifest_sha256"]
        or handle.scientific_payload_sha256 != binding["scientific_payload_sha256"]
    ):
        raise ExactBoutResponseProjectionError(
            "Generalized bout-response successor changed after metadata discovery."
        )
    try:
        verified = validate_scientific_manifest(
            handle.scientific_manifest,
            expected_scientific_payload_sha256=handle.scientific_payload_sha256,
            expected_n_frames=relative.n_frames,
            expected_n_chasers=relative.n_chasers,
            expected_relative_binding=expected_relative_binding,
            expected_semantic_manifest_sha256=semantic_digest,
            expected_controller_payload_sha256=(
                controller_trials.scientific_payload_sha256
            ),
        )
    except ExactBoutResponseContractError as exc:
        raise _projection_error(exc) from exc
    for name in (
        "source_relative_frame",
        "source_motion",
        "source_swim_bouts",
        "semantic_selection_manifest_sha256",
        "controller_trial_payload_sha256",
        "body_extension_present",
    ):
        if plain(verified[name]) != plain(binding[name]):
            raise ExactBoutResponseProjectionError(
                "Bout-response scientific sources differ from the selected exact bundle."
            )
    return handle


__all__ = [
    "ExactBoutResponseProjectionError",
    "load_exact_bout_response",
    "option_bout_response_binding",
]
