"""Exact escape/freeze option binding and audited loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)

from ..chaser_exact_bout_response_contract import exact_child_path
from ..chaser_exact_escape_freeze_contract import (
    ESCAPE_FREEZE_PARENT,
    ExactEscapeFreezeContractError,
    normalize_escape_motion_binding,
    normalize_parameters,
    require_digest,
    require_mapping,
    validate_scientific_manifest,
)
from ..registry import InteractiveSpecOption
from .provenance import freeze, plain


class ExactEscapeFreezeProjectionError(ValueError):
    """An escape/freeze option or source is not one compatible exact child."""


def _projection_error(exc: Exception) -> ExactEscapeFreezeProjectionError:
    return ExactEscapeFreezeProjectionError(str(exc))


def _nonnegative_int(value: Any, *, label: str, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ExactEscapeFreezeContractError(
            f"{label} must be one {'positive' if positive else 'non-negative'} integer."
        )
    return value


def option_escape_freeze_binding(
    option: InteractiveSpecOption,
) -> Mapping[str, Any]:
    """Validate the closed synthesized binding for one escape/freeze view."""

    try:
        analysis_bindings = require_mapping(
            option.spec.get("analysis_bindings"), label="analysis bindings"
        )
        binding = require_mapping(
            analysis_bindings.get("escape_freeze"),
            label="escape/freeze analysis binding",
        )
        if set(binding) != {
            "run_path",
            "manifest_sha256",
            "scientific_payload_sha256",
            "source_motion",
            "controller_trial_payload_sha256",
            "bout_response_payload_sha256",
            "classifier_parameters",
            "n_trials",
            "n_events",
            "n_sweep_rows",
        }:
            raise ExactEscapeFreezeContractError(
                "Escape/freeze binding has an unsupported field set."
            )
        run_path, _ = exact_child_path(
            binding.get("run_path"),
            parent=ESCAPE_FREEZE_PARENT,
            label="escape/freeze run",
        )
        bout_binding = require_mapping(
            analysis_bindings.get("generalized_bout_response"),
            label="generalized bout-response analysis binding",
        )
        bout_motion = require_mapping(
            bout_binding.get("source_motion"), label="bout-response motion source"
        )
        return freeze(
            {
                "run_path": run_path,
                "manifest_sha256": require_digest(
                    binding.get("manifest_sha256"),
                    label="escape/freeze manifest digest",
                ),
                "scientific_payload_sha256": require_digest(
                    binding.get("scientific_payload_sha256"),
                    label="escape/freeze scientific payload digest",
                ),
                "source_motion": normalize_escape_motion_binding(
                    binding.get("source_motion"), expected_bout_motion=bout_motion
                ),
                "controller_trial_payload_sha256": require_digest(
                    binding.get("controller_trial_payload_sha256"),
                    label="escape/freeze controller payload digest",
                ),
                "bout_response_payload_sha256": require_digest(
                    binding.get("bout_response_payload_sha256"),
                    label="escape/freeze bout-response payload digest",
                ),
                "classifier_parameters": normalize_parameters(
                    binding.get("classifier_parameters")
                ),
                "n_trials": _nonnegative_int(
                    binding.get("n_trials"),
                    label="escape/freeze trial count",
                    positive=True,
                ),
                "n_events": _nonnegative_int(
                    binding.get("n_events"), label="escape/freeze event count"
                ),
                "n_sweep_rows": _nonnegative_int(
                    binding.get("n_sweep_rows"), label="escape/freeze sweep-row count"
                ),
            }
        )
    except (TypeError, ValueError) as exc:
        raise _projection_error(exc) from exc


def load_exact_escape_freeze(
    archive: Path,
    option: InteractiveSpecOption,
    *,
    spatial: Any,
    controller_trials: Any,
    generalized_bout_response: Any,
    direct_validation_receipt: str | Path | None = None,
    required_array_names: Collection[str] | None = None,
) -> Any:
    """Load one verified escape/freeze successor and check exact payload joins."""

    binding = option_escape_freeze_binding(option)
    if (
        controller_trials.successor_kind != "controller_chase_trials"
        or controller_trials.scientific_payload_sha256
        != binding["controller_trial_payload_sha256"]
    ):
        raise ExactEscapeFreezeProjectionError(
            "Escape/freeze option uses another controller-trial payload."
        )
    controller_trials.require_verified_authority()
    if (
        generalized_bout_response.successor_kind != "generalized_chaser_bout_response"
        or generalized_bout_response.scientific_payload_sha256
        != binding["bout_response_payload_sha256"]
    ):
        raise ExactEscapeFreezeProjectionError(
            "Escape/freeze option uses another generalized bout-response payload."
        )
    generalized_bout_response.require_verified_authority()
    try:
        bout_sources = require_mapping(
            generalized_bout_response.scientific_manifest.get("sources"),
            label="bout-response sources",
        )
        bout_motion = require_mapping(
            bout_sources.get("motion"), label="bout-response motion source"
        )
        normalized_motion = normalize_escape_motion_binding(
            binding["source_motion"], expected_bout_motion=bout_motion
        )
        run_path, run_name = exact_child_path(
            binding["run_path"],
            parent=ESCAPE_FREEZE_PARENT,
            label="escape/freeze run",
        )
    except (TypeError, ValueError) as exc:
        raise _projection_error(exc) from exc
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_escape_freeze",
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
        raise ExactEscapeFreezeProjectionError(
            "Escape/freeze successor changed after metadata discovery."
        )
    try:
        controller_dimensions = require_mapping(
            controller_trials.scientific_manifest.get("dimensions"),
            label="controller-trial dimensions",
        )
        n_trials = int(controller_dimensions.get("n_trials", -1))
        verified = validate_scientific_manifest(
            handle.scientific_manifest,
            expected_scientific_payload_sha256=handle.scientific_payload_sha256,
            expected_controller_payload_sha256=(
                controller_trials.scientific_payload_sha256
            ),
            expected_bout_response_payload_sha256=(
                generalized_bout_response.scientific_payload_sha256
            ),
            expected_bout_motion=bout_motion,
            expected_n_trials=n_trials,
        )
    except (TypeError, ValueError) as exc:
        raise _projection_error(exc) from exc
    expected = {
        "source_motion": normalized_motion,
        "controller_trial_payload_sha256": binding["controller_trial_payload_sha256"],
        "bout_response_payload_sha256": binding["bout_response_payload_sha256"],
        "classifier_parameters": binding["classifier_parameters"],
        "n_trials": binding["n_trials"],
        "n_events": binding["n_events"],
        "n_sweep_rows": binding["n_sweep_rows"],
    }
    for name, expected_value in expected.items():
        if plain(verified[name]) != plain(expected_value):
            raise ExactEscapeFreezeProjectionError(
                "Escape/freeze scientific evidence differs from the selected bundle."
            )
    return handle


__all__ = [
    "ExactEscapeFreezeProjectionError",
    "load_exact_escape_freeze",
    "option_escape_freeze_binding",
]
