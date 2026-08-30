"""Verified read-only projection for exact body-frame gaze tracking."""

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

from ..chaser_exact_gaze_contract import (
    GAZE_TRACKING_PARENT,
    ExactGazeTrackingContractError,
    validate_gaze_scientific_manifest,
)
from .provenance import freeze, plain


class ExactGazeTrackingProjectionError(ValueError):
    """The selected gaze successor cannot produce a verified projection."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactGazeTrackingProjectionError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactGazeTrackingProjectionError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_run_path(value: Any) -> tuple[str, str]:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactGazeTrackingProjectionError("Gaze run path must be one exact child.")
    prefix = f"{GAZE_TRACKING_PARENT}/"
    name = value.removeprefix(prefix)
    if not value.startswith(prefix) or not name or "/" in name:
        raise ExactGazeTrackingProjectionError(
            "Gaze run path must be one exact non-selector child."
        )
    return value, name


def option_gaze_tracking_binding(option: Any) -> Mapping[str, Any]:
    """Validate and freeze the gaze binding discovered for an exact bundle."""

    bindings = _mapping(option.spec.get("analysis_bindings"), label="analysis bindings")
    value = _mapping(bindings.get("gaze_tracking"), label="gaze binding")
    if set(value) != {
        "run_path",
        "manifest_sha256",
        "scientific_payload_sha256",
        "source_relative_frame",
        "source_eye_orientation",
        "source_radial_geometry",
        "semantic_selection_manifest_sha256",
        "parameters",
    }:
        raise ExactGazeTrackingProjectionError(
            "Gaze analysis binding has an unsupported field set."
        )
    run_path, _ = _exact_run_path(value.get("run_path"))
    result = {
        "run_path": run_path,
        "manifest_sha256": _digest(
            value.get("manifest_sha256"), label="gaze manifest digest"
        ),
        "scientific_payload_sha256": _digest(
            value.get("scientific_payload_sha256"),
            label="gaze scientific payload digest",
        ),
        "source_relative_frame": dict(
            _mapping(value.get("source_relative_frame"), label="gaze relative source")
        ),
        "source_eye_orientation": dict(
            _mapping(value.get("source_eye_orientation"), label="gaze eye source")
        ),
        "source_radial_geometry": dict(
            _mapping(value.get("source_radial_geometry"), label="gaze radial source")
        ),
        "semantic_selection_manifest_sha256": _digest(
            value.get("semantic_selection_manifest_sha256"),
            label="gaze semantic-selection digest",
        ),
        "parameters": dict(_mapping(value.get("parameters"), label="gaze parameters")),
    }
    return freeze(result)


def load_exact_gaze_tracking(
    archive: Path,
    option: Any,
    *,
    spatial: Any,
    radial: Any,
    expected_relative_binding: Mapping[str, Any],
    relative: Any,
    direct_validation_receipt: str | Path | None = None,
    required_array_names: Collection[str] | None = None,
) -> Any:
    """Load one verified gaze successor and recheck every exact source join."""

    binding = option_gaze_tracking_binding(option)
    try:
        require_same_exact_relative_frame_child(
            expected_relative_binding,
            binding["source_relative_frame"],
            expected_label="spatial keypoint relative-frame binding",
            observed_label="gaze relative-frame binding",
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactGazeTrackingProjectionError(str(exc)) from exc
    sources = _mapping(
        spatial.scientific_manifest.get("sources"), label="spatial sources"
    )
    semantic = _mapping(
        sources.get("protocol_semantic_selection"),
        label="spatial semantic selection",
    )
    semantic_digest = _digest(
        semantic.get("manifest_sha256"), label="spatial semantic-selection digest"
    )
    if binding["semantic_selection_manifest_sha256"] != semantic_digest:
        raise ExactGazeTrackingProjectionError(
            "Gaze option uses another semantic selection."
        )
    providers = sources.get("position_providers")
    if (
        not isinstance(providers, (list, tuple))
        or len(providers) != 2
        or not isinstance(providers[0], Mapping)
    ):
        raise ExactGazeTrackingProjectionError(
            "Spatial bundle lacks the keypoint radial binding required by gaze."
        )
    spatial_radial_binding = _mapping(
        providers[0].get("radial_near_field"), label="spatial keypoint radial binding"
    )
    radial_sources = _mapping(
        radial.scientific_manifest.get("sources"), label="keypoint radial sources"
    )
    expected_radial_binding = {
        **dict(spatial_radial_binding),
        "scientific_payload_sha256": radial.scientific_payload_sha256,
        "arena_geometry_and_scale": radial_sources.get("arena_geometry_and_scale"),
        "arena": radial.scientific_manifest.get("arena"),
    }
    run_path, run_name = _exact_run_path(binding["run_path"])
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_gaze_tracking",
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
        raise ExactGazeTrackingProjectionError(
            "Gaze successor changed after metadata discovery."
        )
    try:
        verified = validate_gaze_scientific_manifest(
            handle.scientific_manifest,
            expected_scientific_payload_sha256=handle.scientific_payload_sha256,
            expected_n_frames=relative.n_frames,
            expected_n_chasers=relative.n_chasers,
            expected_relative_binding=expected_relative_binding,
            expected_semantic_manifest_sha256=semantic_digest,
            expected_radial_binding=expected_radial_binding,
        )
    except ExactGazeTrackingContractError as exc:
        raise ExactGazeTrackingProjectionError(str(exc)) from exc
    for key in (
        "source_relative_frame",
        "source_eye_orientation",
        "source_radial_geometry",
        "semantic_selection_manifest_sha256",
        "parameters",
    ):
        if plain(verified[key]) != plain(binding[key]):
            raise ExactGazeTrackingProjectionError(
                "Gaze scientific sources differ from the selected exact bundle."
            )
    return handle


__all__ = [
    "ExactGazeTrackingProjectionError",
    "load_exact_gaze_tracking",
    "option_gaze_tracking_binding",
]
