"""Closed metadata contract for the exact keypoint body-bearing capability."""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
)

BODY_BEARING_ARRAY_PATHS = (
    "body/body_bearing_deg",
    "body/body_bearing_valid",
)


class ExactBodyBearingContractError(ValueError):
    """The keypoint relative frame cannot supply accepted body bearing."""


def compatible_body_bearing_binding(
    relative_manifest: Mapping[str, Any],
    *,
    expected_relative_binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return a closed binding only for a complete declared body extension."""

    schema = relative_manifest.get("schema_binding")
    declarations = relative_manifest.get("array_declarations")
    if (
        not isinstance(schema, Mapping)
        or schema.get("body_extension_present") is not True
        or not isinstance(declarations, list)
    ):
        return None
    paths = {
        item.get("path")
        for item in declarations
        if isinstance(item, Mapping) and type(item.get("path")) is str
    }
    if not set(BODY_BEARING_ARRAY_PATHS).issubset(paths):
        return None
    try:
        normalized = validate_exact_relative_frame_binding(
            expected_relative_binding,
            label="body-bearing keypoint relative-frame binding",
        ).normalized_identity
    except ExactRelativeFrameBindingError:
        return None
    return {
        "source_relative_frame": dict(normalized),
        "array_paths": list(BODY_BEARING_ARRAY_PATHS),
        "body_axis_authority": "accepted_keypoint_body_extension",
        "position_substitution": "prohibited",
    }


def option_body_bearing_binding(option: Any) -> Mapping[str, Any]:
    """Validate the closed body-bearing binding carried by one explorer option."""

    bindings = option.spec.get("analysis_bindings")
    value = bindings.get("body_bearing") if isinstance(bindings, Mapping) else None
    if not isinstance(value, Mapping) or set(value) != {
        "source_relative_frame",
        "array_paths",
        "body_axis_authority",
        "position_substitution",
    }:
        raise ExactBodyBearingContractError(
            "Exact body-bearing analysis binding is absent or malformed."
        )
    if tuple(value.get("array_paths", ())) != BODY_BEARING_ARRAY_PATHS:
        raise ExactBodyBearingContractError(
            "Exact body-bearing array roster is incompatible."
        )
    if (
        value.get("body_axis_authority") != "accepted_keypoint_body_extension"
        or value.get("position_substitution") != "prohibited"
    ):
        raise ExactBodyBearingContractError(
            "Exact body-bearing authority or fallback policy is incompatible."
        )
    try:
        normalized = validate_exact_relative_frame_binding(
            value.get("source_relative_frame"),
            label="body-bearing option relative-frame binding",
        ).normalized_identity
        spatial_sources = option.spec.get("source_paths", {}).get("position_providers")
        if not isinstance(spatial_sources, list) or not spatial_sources:
            raise ExactBodyBearingContractError(
                "Exact body-bearing option lacks its keypoint provider binding."
            )
        keypoint = spatial_sources[0]
        if (
            not isinstance(keypoint, Mapping)
            or keypoint.get("provider_role") != "keypoint"
        ):
            raise ExactBodyBearingContractError(
                "Exact body-bearing option lacks a keypoint provider."
            )
        require_same_exact_relative_frame_child(
            keypoint.get("relative_frame"),
            normalized,
            expected_label="spatial keypoint relative-frame binding",
            observed_label="body-bearing relative-frame binding",
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactBodyBearingContractError(str(exc)) from exc
    return value


__all__ = [
    "BODY_BEARING_ARRAY_PATHS",
    "ExactBodyBearingContractError",
    "compatible_body_bearing_binding",
    "option_body_bearing_binding",
]
