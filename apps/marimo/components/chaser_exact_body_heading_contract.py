"""Closed metadata contract for exact anatomical fish heading."""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
)

BODY_HEADING_ARRAY_PATHS = (
    "body/body_source_row_id",
    "body/body_source_row_valid",
    "body/body_heading_deg",
    "body/body_heading_valid",
    "body/body_heading_reason_code",
)
BODY_HEADING_FRAME_COLLAPSE_POLICY = (
    "exact_equality_across_flattened_chaser_rows_then_one_row_per_acquisition_frame"
)


class ExactBodyHeadingContractError(ValueError):
    """The keypoint relative frame cannot supply exact anatomical heading."""


def compatible_body_heading_binding(
    relative_manifest: Mapping[str, Any],
    *,
    expected_relative_binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return a closed binding only when all heading evidence is declared."""

    schema = relative_manifest.get("schema_binding")
    declarations = relative_manifest.get("array_declarations")
    if (
        not isinstance(schema, Mapping)
        or schema.get("body_extension_present") is not True
        or not isinstance(declarations, list)
    ):
        return None
    authorities = relative_manifest.get("source_authorities")
    body_authority = (
        authorities.get("body_frame") if isinstance(authorities, Mapping) else None
    )
    if not isinstance(body_authority, Mapping) or any(
        type(body_authority.get(name)) is not str or not body_authority.get(name)
        for name in (
            "source_authority_id",
            "source_digest",
            "provider_id",
            "provider_digest",
        )
    ):
        return None
    paths = {
        item.get("path")
        for item in declarations
        if isinstance(item, Mapping) and type(item.get("path")) is str
    }
    if not set(BODY_HEADING_ARRAY_PATHS).issubset(paths):
        return None
    try:
        normalized = validate_exact_relative_frame_binding(
            expected_relative_binding,
            label="body-heading keypoint relative-frame binding",
        ).normalized_identity
    except ExactRelativeFrameBindingError:
        return None
    return {
        "source_relative_frame": dict(normalized),
        "array_paths": list(BODY_HEADING_ARRAY_PATHS),
        "body_axis_authority": "accepted_keypoint_body_extension",
        "frame_collapse_policy": BODY_HEADING_FRAME_COLLAPSE_POLICY,
        "position_substitution": "prohibited",
        "motion_heading_fallback": "prohibited",
    }


def option_body_heading_binding(option: Any) -> Mapping[str, Any]:
    """Validate one explorer option's closed body-heading binding."""

    bindings = option.spec.get("analysis_bindings")
    value = bindings.get("body_heading") if isinstance(bindings, Mapping) else None
    expected_keys = {
        "source_relative_frame",
        "array_paths",
        "body_axis_authority",
        "frame_collapse_policy",
        "position_substitution",
        "motion_heading_fallback",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ExactBodyHeadingContractError(
            "Exact body-heading analysis binding is absent or malformed."
        )
    if tuple(value.get("array_paths", ())) != BODY_HEADING_ARRAY_PATHS:
        raise ExactBodyHeadingContractError(
            "Exact body-heading array roster is incompatible."
        )
    if (
        value.get("body_axis_authority") != "accepted_keypoint_body_extension"
        or value.get("frame_collapse_policy") != BODY_HEADING_FRAME_COLLAPSE_POLICY
        or value.get("position_substitution") != "prohibited"
        or value.get("motion_heading_fallback") != "prohibited"
    ):
        raise ExactBodyHeadingContractError(
            "Exact body-heading authority, collapse, or fallback policy is incompatible."
        )
    try:
        normalized = validate_exact_relative_frame_binding(
            value.get("source_relative_frame"),
            label="body-heading option relative-frame binding",
        ).normalized_identity
        spatial_sources = option.spec.get("source_paths", {}).get("position_providers")
        if not isinstance(spatial_sources, list) or not spatial_sources:
            raise ExactBodyHeadingContractError(
                "Exact body-heading option lacks its keypoint provider binding."
            )
        keypoint = spatial_sources[0]
        if (
            not isinstance(keypoint, Mapping)
            or keypoint.get("provider_role") != "keypoint"
        ):
            raise ExactBodyHeadingContractError(
                "Exact body-heading option lacks a keypoint provider."
            )
        require_same_exact_relative_frame_child(
            keypoint.get("relative_frame"),
            normalized,
            expected_label="spatial keypoint relative-frame binding",
            observed_label="body-heading relative-frame binding",
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactBodyHeadingContractError(str(exc)) from exc
    return value


__all__ = [
    "BODY_HEADING_ARRAY_PATHS",
    "BODY_HEADING_FRAME_COLLAPSE_POLICY",
    "ExactBodyHeadingContractError",
    "compatible_body_heading_binding",
    "option_body_heading_binding",
]
