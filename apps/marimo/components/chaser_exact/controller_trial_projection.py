"""Exact controller-trial option binding and deep-audited projection loader."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Collection, Mapping

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
)

from ..registry import InteractiveSpecOption

CONTROLLER_TRIAL_PARENT = "analysis/controller_chase_trial_runs"
SEMANTIC_PARENT = "analysis/protocol_semantic_chaser_selection_runs"


class ExactControllerTrialProjectionError(ValueError):
    """A controller-trial option or source is not one compatible exact child."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactControllerTrialProjectionError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactControllerTrialProjectionError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_child_path(value: Any, *, parent: str, label: str) -> tuple[str, str]:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactControllerTrialProjectionError(
            f"{label} must be one exact child path."
        )
    prefix = f"{parent}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name in {".", ".."}
        or name.casefold()
        in {
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
    ):
        raise ExactControllerTrialProjectionError(
            f"{label} must be one exact child below {parent!r}."
        )
    return value, name


def _exact_binding_identity(
    value: Any,
    *,
    parent: str,
    label: str,
) -> Mapping[str, str]:
    binding = _mapping(value, label=label)
    run_path, _ = _exact_child_path(
        binding.get("run_path"), parent=parent, label=f"{label} run"
    )
    return MappingProxyType(
        {
            "run_path": run_path,
            "manifest_sha256": _digest(
                binding.get("manifest_sha256"), label=f"{label} digest"
            ),
        }
    )


def option_controller_trial_binding(
    option: InteractiveSpecOption,
) -> Mapping[str, Any]:
    """Validate the closed synthesized binding for one controller-trial view."""

    analysis_bindings = _mapping(
        option.spec.get("analysis_bindings"), label="analysis bindings"
    )
    binding = _mapping(
        analysis_bindings.get("controller_trials"),
        label="controller-trial analysis binding",
    )
    if set(binding) != {
        "run_path",
        "manifest_sha256",
        "scientific_payload_sha256",
        "source_relative_frame",
        "semantic_selection",
    }:
        raise ExactControllerTrialProjectionError(
            "Controller-trial analysis binding has an unsupported field set."
        )
    run_path, _ = _exact_child_path(
        binding.get("run_path"),
        parent=CONTROLLER_TRIAL_PARENT,
        label="controller-trial run",
    )
    try:
        source_relative_frame = MappingProxyType(
            dict(
                validate_exact_relative_frame_binding(
                    binding.get("source_relative_frame"),
                    label="controller-trial option relative-frame binding",
                ).normalized_identity
            )
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactControllerTrialProjectionError(str(exc)) from exc
    return MappingProxyType(
        {
            "run_path": run_path,
            "manifest_sha256": _digest(
                binding.get("manifest_sha256"),
                label="controller-trial manifest digest",
            ),
            "scientific_payload_sha256": _digest(
                binding.get("scientific_payload_sha256"),
                label="controller-trial scientific payload digest",
            ),
            "source_relative_frame": source_relative_frame,
            "semantic_selection": _exact_binding_identity(
                binding.get("semantic_selection"),
                parent=SEMANTIC_PARENT,
                label="controller-trial option semantic selection",
            ),
        }
    )


def load_exact_controller_trials(
    archive: Path,
    option: InteractiveSpecOption,
    *,
    spatial: Any,
    expected_relative_binding: Mapping[str, Any],
    relative: Any,
    direct_validation_receipt: str | Path | None = None,
    required_array_names: Collection[str] | None = None,
) -> Any:
    """Load one verified controller successor and check its exact source join."""

    binding = option_controller_trial_binding(option)
    expected_relative = validate_exact_relative_frame_binding(
        expected_relative_binding,
        label="spatial keypoint relative-frame binding",
    )
    require_same_exact_relative_frame_child(
        expected_relative.normalized_identity,
        binding["source_relative_frame"],
        expected_label="spatial keypoint relative-frame binding",
        observed_label="controller-trial option relative-frame binding",
    )
    spatial_sources = _mapping(
        spatial.scientific_manifest.get("sources"), label="spatial sources"
    )
    expected_semantic = _exact_binding_identity(
        spatial_sources.get("protocol_semantic_selection"),
        parent=SEMANTIC_PARENT,
        label="spatial semantic selection",
    )
    if dict(binding["semantic_selection"]) != dict(expected_semantic):
        raise ExactControllerTrialProjectionError(
            "Controller-trial option uses another semantic selection."
        )
    run_path, run_name = _exact_child_path(
        binding["run_path"],
        parent=CONTROLLER_TRIAL_PARENT,
        label="controller-trial run",
    )
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="controller_chase_trials",
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
        raise ExactControllerTrialProjectionError(
            "Controller-trial successor changed after metadata discovery."
        )
    scientific = _mapping(
        handle.scientific_manifest, label="controller-trial scientific manifest"
    )
    schema = _mapping(
        scientific.get("scientific_schema"), label="controller-trial schema"
    )
    policy = _mapping(scientific.get("policy"), label="controller-trial policy")
    dimensions = _mapping(
        scientific.get("dimensions"), label="controller-trial dimensions"
    )
    if (
        schema.get("schema_id") != "palette.analysis.controller_chase_trials"
        or schema.get("schema_version") != 1
        or schema.get("method_id") != "exact_logged_trial_id_active_membership_v1"
        or policy.get("fallback") != "prohibited_fail_closed"
        or policy.get("legacy_contiguous_interval_reconstruction") != "rejected"
        or int(dimensions.get("n_frames", 0)) != relative.n_frames
        or int(dimensions.get("n_chasers", 0)) != relative.n_chasers
        or int(dimensions.get("n_source_rows", 0))
        != relative.n_frames * relative.n_chasers
        or not 0 < int(dimensions.get("n_trials", 0)) <= 32
    ):
        raise ExactControllerTrialProjectionError(
            "Controller-trial schema, policy, or dimensions are incompatible."
        )
    scientific_relative = validate_exact_relative_frame_binding(
        scientific.get("source_relative_frame"),
        label="controller-trial scientific relative-frame binding",
    )
    require_same_exact_relative_frame_child(
        expected_relative.normalized_identity,
        scientific_relative.normalized_identity,
        expected_label="spatial keypoint relative-frame binding",
        observed_label="controller-trial scientific relative-frame binding",
    )
    scientific_semantic = _exact_binding_identity(
        scientific.get("semantic_selection"),
        parent=SEMANTIC_PARENT,
        label="controller-trial scientific semantic selection",
    )
    if dict(scientific_semantic) != dict(expected_semantic):
        raise ExactControllerTrialProjectionError(
            "Controller-trial scientific source uses another semantic selection."
        )
    return handle


__all__ = [
    "ExactControllerTrialProjectionError",
    "load_exact_controller_trials",
    "option_controller_trial_binding",
]
