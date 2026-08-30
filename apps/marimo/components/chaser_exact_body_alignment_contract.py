"""Closed scientific contract for persisted anatomical alignment by distance."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    ANGLE_CONVENTION_ID,
    DISTANCE_BIN_POLICY_ID,
    METHOD_ID,
    PERSISTED_ARRAY_NAMES,
    SCHEMA_ID,
    SCHEMA_VERSION,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


BODY_ALIGNMENT_PARENT = "analysis/chaser_body_alignment_by_distance_runs"
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


class ExactBodyAlignmentContractError(ValueError):
    """A body-alignment successor identity, source, or policy is invalid."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactBodyAlignmentContractError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactBodyAlignmentContractError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalized_epoch_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ExactBodyAlignmentContractError(
            "Body alignment must bind exactly three semantic epoch records."
        )
    result = []
    for index, record in enumerate(value, start=1):
        item = _mapping(record, label=f"epoch record {index}")
        if "epoch_role_code" in item and item.get("epoch_role_code") != index:
            raise ExactBodyAlignmentContractError(
                "Body-alignment epoch role codes are not ordered 1, 2, 3."
            )
        normalized = {
            "analysis_role": item.get("analysis_role"),
            "window_id": item.get("window_id"),
            "source_label": item.get("source_label"),
            "start_frame": item.get("start_frame"),
            "end_frame_exclusive": item.get("end_frame_exclusive"),
            "source_interval_sha256": item.get("source_interval_sha256"),
        }
        if (
            normalized["analysis_role"]
            != ("chaser_pre", "chaser_training", "chaser_post")[index - 1]
            or type(normalized["window_id"]) is not int
            or type(normalized["start_frame"]) is not int
            or type(normalized["end_frame_exclusive"]) is not int
            or normalized["end_frame_exclusive"] <= normalized["start_frame"]
        ):
            raise ExactBodyAlignmentContractError(
                "Body-alignment semantic epoch identity is malformed."
            )
        _digest(
            normalized["source_interval_sha256"],
            label=f"epoch record {index} source interval",
        )
        result.append(normalized)
    return result


def validate_body_alignment_scientific_manifest(
    value: Any,
    *,
    expected_scientific_payload_sha256: str,
    expected_n_frames: int,
    expected_n_chasers: int,
    expected_relative_binding: Mapping[str, Any],
    expected_semantic_binding: Mapping[str, Any],
    expected_fish_position_authority: Mapping[str, Any],
    expected_body_frame_authority: Mapping[str, Any],
    expected_scale_policy: Mapping[str, Any],
    expected_epoch_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Validate the complete v1 scientific manifest without opening arrays."""

    scientific = _mapping(value, label="body-alignment scientific manifest")
    payload = _digest(
        expected_scientific_payload_sha256,
        label="body-alignment scientific payload digest",
    )
    body = dict(scientific)
    observed_payload = body.pop("payload_digest", None)
    if observed_payload != payload or canonical_json_sha256(_plain(body)) != payload:
        raise ExactBodyAlignmentContractError(
            "Body-alignment scientific payload digest is stale."
        )
    if (
        dict(_mapping(scientific.get("scientific_schema"), label="scientific schema"))
        != {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
        }
        or scientific.get("method_id") != METHOD_ID
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment scientific schema or method is incompatible."
        )
    if (
        scientific.get("selector_eligible") is not False
        or scientific.get("selection") != "none"
        or scientific.get("production_authority") is not False
        or scientific.get("registry_update") is not False
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment successor is not selector-ineligible."
        )

    dimensions = _mapping(scientific.get("dimensions"), label="dimensions")
    required_dimensions = {
        "n_frames",
        "n_chasers",
        "n_frame_rows",
        "n_epochs",
        "n_distance_bins",
        "n_summary_rows",
    }
    if set(dimensions) != required_dimensions:
        raise ExactBodyAlignmentContractError(
            "Body-alignment dimensions have an unsupported field set."
        )
    n_frames = int(dimensions.get("n_frames", 0))
    n_chasers = int(dimensions.get("n_chasers", 0))
    n_bins = int(dimensions.get("n_distance_bins", 0))
    n_summary = int(dimensions.get("n_summary_rows", 0))
    if (
        n_frames != expected_n_frames
        or n_chasers != expected_n_chasers
        or int(dimensions.get("n_frame_rows", 0)) != n_frames * n_chasers
        or int(dimensions.get("n_epochs", 0)) != 3
        or n_bins <= 0
        or n_summary != 3 * n_chasers * n_bins
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment dimensions differ from the exact relative child."
        )

    sources = _mapping(scientific.get("sources"), label="sources")
    if set(sources) != {
        "relative_frame",
        "protocol_semantic_selection",
        "fish_position_authority",
        "body_frame_authority",
        "scale_policy",
    }:
        raise ExactBodyAlignmentContractError(
            "Body-alignment source roster is incompatible."
        )
    try:
        require_same_exact_relative_frame_child(
            expected_relative_binding,
            _mapping(sources.get("relative_frame"), label="relative source"),
            expected_label="spatial keypoint relative-frame binding",
            observed_label="body-alignment relative-frame binding",
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactBodyAlignmentContractError(str(exc)) from exc
    if _plain(sources.get("protocol_semantic_selection")) != _plain(
        expected_semantic_binding
    ):
        raise ExactBodyAlignmentContractError(
            "Body alignment uses another semantic selection."
        )
    if _plain(sources.get("fish_position_authority")) != _plain(
        expected_fish_position_authority
    ):
        raise ExactBodyAlignmentContractError(
            "Body alignment uses another fish-position authority."
        )
    if _plain(sources.get("body_frame_authority")) != _plain(
        expected_body_frame_authority
    ):
        raise ExactBodyAlignmentContractError(
            "Body alignment uses another anatomical body-frame authority."
        )
    if _plain(sources.get("scale_policy")) != _plain(expected_scale_policy):
        raise ExactBodyAlignmentContractError(
            "Body alignment uses another physical scale policy."
        )
    if sources["scale_policy"].get("unit") != "mm":
        raise ExactBodyAlignmentContractError(
            "Body-alignment distance is not sealed in millimetres."
        )

    position_provider = _mapping(
        scientific.get("position_provider"), label="position provider"
    )
    if (
        position_provider.get("provider_id")
        != expected_fish_position_authority.get("provider_id")
        or position_provider.get("provider_digest")
        != expected_fish_position_authority.get("provider_digest")
        or position_provider.get("distance_surface")
        != "base/relative_distance_physical"
        or position_provider.get("body_origin_distance_substitution") != "prohibited"
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment position-provider policy is incompatible."
        )
    convention = _mapping(
        scientific.get("coordinate_and_angle_convention"), label="angle convention"
    )
    if (
        convention.get("convention_id") != ANGLE_CONVENTION_ID
        or convention.get("camera_axes") != "+x_right_+y_down"
        or convention.get("heading_world_projection") != "atan2(-forward_y,forward_x)"
        or convention.get("body_bearing") != "atan2(anatomical_left,anatomical_forward)"
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment coordinate or angle convention is incompatible."
        )

    recipe = dict(
        _mapping(scientific.get("distance_bin_recipe"), label="distance-bin recipe")
    )
    recipe_digest = _digest(
        recipe.pop("recipe_sha256", None), label="distance-bin recipe digest"
    )
    if canonical_json_sha256(recipe) != recipe_digest:
        raise ExactBodyAlignmentContractError("Distance-bin recipe digest is stale.")
    edges = np.asarray(recipe.get("edges_mm"), dtype=np.float64)
    width = float(recipe.get("bin_width_mm", math.nan))
    if (
        recipe.get("policy_id") != DISTANCE_BIN_POLICY_ID
        or recipe.get("interval_policy") != "half_open_except_final_closed"
        or recipe.get("zero_anchored") is not True
        or edges.shape != (n_bins + 1,)
        or edges[0] != 0.0
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or not math.isfinite(width)
        or width <= 0.0
        or not np.allclose(np.diff(edges), width, rtol=0.0, atol=1e-12)
        or recipe.get("edge_array_sha256") != array_values_sha256(edges)
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment persisted distance-bin recipe is incompatible."
        )

    observed_epochs = _normalized_epoch_records(scientific.get("epoch_records"))
    expected_epochs = _normalized_epoch_records(expected_epoch_records)
    if observed_epochs != expected_epochs:
        raise ExactBodyAlignmentContractError(
            "Body alignment uses different semantic epoch records."
        )
    if scientific.get("epoch_records_sha256") != canonical_json_sha256(
        _plain(scientific.get("epoch_records"))
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment epoch-record digest is stale."
        )

    declarations = scientific.get("array_declarations")
    if not isinstance(declarations, (tuple, list)):
        raise ExactBodyAlignmentContractError(
            "Body alignment lacks sealed array declarations."
        )
    declaration_by_path = {
        item.get("path"): item for item in declarations if isinstance(item, Mapping)
    }
    if len(declaration_by_path) != len(declarations) or set(declaration_by_path) != set(
        PERSISTED_ARRAY_NAMES
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment persisted array inventory is incompatible."
        )
    for path, declaration in declaration_by_path.items():
        _digest(declaration.get("content_sha256"), label=f"{path} content digest")
        shape = declaration.get("shape")
        expected_first = (
            n_frames * n_chasers
            if path.startswith("frame_")
            else n_bins + 1
            if path == "distance_bin_edges_mm"
            else n_summary
        )
        if not isinstance(shape, (tuple, list)) or list(shape) != [expected_first]:
            raise ExactBodyAlignmentContractError(
                f"Body-alignment array {path!r} has an incompatible shape."
            )

    denominators = _mapping(scientific.get("denominators"), label="denominators")
    if (
        denominators.get("viewer_rebinning") != "prohibited"
        or denominators.get("interpolation") != "prohibited"
        or denominators.get("missing_body_policy")
        != "retained_invalid_no_motion_heading_fallback"
    ):
        raise ExactBodyAlignmentContractError(
            "Body-alignment denominator or fallback policy is incompatible."
        )
    return {
        "source_relative_frame": sources["relative_frame"],
        "source_protocol_semantic_selection": sources["protocol_semantic_selection"],
        "source_fish_position_authority": sources["fish_position_authority"],
        "source_body_frame_authority": sources["body_frame_authority"],
        "distance_bin_recipe": scientific["distance_bin_recipe"],
        "dimensions": dimensions,
        "epoch_records": scientific["epoch_records"],
        "identity_registries": scientific["identity_registries"],
    }


__all__ = [
    "BODY_ALIGNMENT_PARENT",
    "FORBIDDEN_SELECTORS",
    "ExactBodyAlignmentContractError",
    "validate_body_alignment_scientific_manifest",
]
