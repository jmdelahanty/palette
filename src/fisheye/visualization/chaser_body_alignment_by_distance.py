"""Shared validation view for persisted anatomical alignment summaries.

Static publication and interactive exploration deliberately enter through this
same parser.  It validates the sealed distance-bin grid, row keys, support
conservation, and statistic domains; neither caller is allowed to regroup or
rebin frame evidence.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    ALIGNMENT_REASON_CODES,
    ANGLE_CONVENTION_ID,
    DISTANCE_BIN_POLICY_ID,
    METHOD_ID,
    PERSISTED_ARRAY_NAMES,
    SCHEMA_ID,
    SCHEMA_VERSION,
    SUMMARY_VIEW_ARRAY_NAMES,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Body-alignment {field} is not one SHA-256 digest.")
    return value


def _manifest_self_contract(scientific: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the closed scientific contract available from the child itself."""

    body = _plain(scientific)
    payload = body.pop("payload_digest", None)
    if _digest(payload, field="payload") != canonical_json_sha256(body):
        raise ValueError("Body-alignment scientific payload digest is stale.")
    if (
        _plain(scientific.get("scientific_schema"))
        != {"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION}
        or scientific.get("method_id") != METHOD_ID
        or scientific.get("selector_eligible") is not False
        or scientific.get("selection") != "none"
        or scientific.get("production_authority") is not False
        or scientific.get("registry_update") is not False
    ):
        raise ValueError("Body-alignment scientific identity or safety is invalid.")

    sources = scientific.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "relative_frame",
        "protocol_semantic_selection",
        "fish_position_authority",
        "body_frame_authority",
        "scale_policy",
    }:
        raise ValueError("Body-alignment source roster is incompatible.")
    for name in ("relative_frame", "protocol_semantic_selection"):
        binding = sources.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"Body-alignment {name} binding is absent.")
        minimal_fields = {"run_path", "manifest_sha256"}
        receipt_fields = {
            "run_path",
            "manifest_sha256",
            "verification_mode",
            "validation_receipt_sha256",
        }
        allowed_fields = (
            (minimal_fields, receipt_fields)
            if name == "relative_frame"
            else (minimal_fields,)
        )
        if set(binding) not in allowed_fields:
            raise ValueError(f"Body-alignment {name} binding is not closed.")
        if type(binding.get("run_path")) is not str or not binding["run_path"]:
            raise ValueError(f"Body-alignment {name} run path is invalid.")
        _digest(binding.get("manifest_sha256"), field=f"{name} manifest")
        if "validation_receipt_sha256" in binding:
            _digest(
                binding.get("validation_receipt_sha256"),
                field=f"{name} validation receipt",
            )
            if (
                binding.get("verification_mode")
                != "receipt_bound_targeted_array_rehash_v1"
            ):
                raise ValueError(
                    f"Body-alignment {name} verification mode is unsupported."
                )
    fish = sources.get("fish_position_authority")
    body_frame = sources.get("body_frame_authority")
    scale = sources.get("scale_policy")
    if (
        not isinstance(fish, Mapping)
        or not isinstance(body_frame, Mapping)
        or not isinstance(scale, Mapping)
        or scale.get("unit") != "mm"
    ):
        raise ValueError("Body-alignment provider or scale authority is invalid.")
    for name, authority in (("fish", fish), ("body", body_frame)):
        if type(authority.get("provider_id")) is not str:
            raise ValueError(f"Body-alignment {name} provider ID is invalid.")
        _digest(authority.get("provider_digest"), field=f"{name} provider")

    position = scientific.get("position_provider")
    if (
        not isinstance(position, Mapping)
        or position.get("provider_id") != fish.get("provider_id")
        or position.get("provider_digest") != fish.get("provider_digest")
        or position.get("distance_surface") != "base/relative_distance_physical"
        or position.get("body_origin_distance_substitution") != "prohibited"
    ):
        raise ValueError("Body-alignment position-provider policy is invalid.")
    convention = scientific.get("coordinate_and_angle_convention")
    if (
        not isinstance(convention, Mapping)
        or convention.get("convention_id") != ANGLE_CONVENTION_ID
        or convention.get("camera_axes") != "+x_right_+y_down"
        or convention.get("heading_world_projection") != "atan2(-forward_y,forward_x)"
        or convention.get("body_bearing") != "atan2(anatomical_left,anatomical_forward)"
        or convention.get("alignment_cos") != "cos(deg2rad(body_bearing_deg))"
        or convention.get("lateral_sin") != "sin(deg2rad(body_bearing_deg))"
    ):
        raise ValueError("Body-alignment angle convention is incompatible.")
    denominators = scientific.get("denominators")
    if (
        not isinstance(denominators, Mapping)
        or denominators.get("candidate_row_count")
        != "epoch_and_occurrence_and_base_physical_distance_valid_in_persisted_bin"
        or denominators.get("joint_valid_row_count")
        != "candidate_and_anatomical_body_bearing_valid"
        or denominators.get("distance_invalid_policy")
        != "retained_in_epoch_counts_not_assigned_to_a_distance_bin"
        or denominators.get("body_valid_distance_invalid_policy")
        != "retained_in_epoch_distance_invalid_body_valid_count_and_never_binned"
        or denominators.get("missing_body_policy")
        != "retained_invalid_no_motion_heading_fallback"
        or denominators.get("interpolation") != "prohibited"
        or denominators.get("viewer_rebinning") != "prohibited"
    ):
        raise ValueError("Body-alignment denominator or fallback policy is invalid.")

    declarations = scientific.get("array_declarations")
    if not isinstance(declarations, (tuple, list)):
        raise ValueError("Body-alignment array declarations are absent.")
    declaration_by_path = {
        item.get("path"): item for item in declarations if isinstance(item, Mapping)
    }
    if len(declaration_by_path) != len(declarations) or set(declaration_by_path) != set(
        PERSISTED_ARRAY_NAMES
    ):
        raise ValueError("Body-alignment array declaration roster is incompatible.")
    for path, declaration in declaration_by_path.items():
        if (
            set(declaration) != {"path", "dtype", "shape", "content_sha256"}
            or type(declaration.get("dtype")) is not str
            or not isinstance(declaration.get("shape"), (tuple, list))
        ):
            raise ValueError(f"Body-alignment array declaration {path!r} is malformed.")
        _digest(declaration.get("content_sha256"), field=f"{path} content")
    return sources


def _array(handle: Any, name: str, *, size: int | None = None) -> np.ndarray:
    try:
        values = np.asarray(handle.array(name)).reshape(-1)
    except KeyError as exc:
        raise ValueError(
            f"Body alignment lacks required persisted array {name!r}."
        ) from exc
    if size is not None and values.shape != (size,):
        raise ValueError(f"Body-alignment array {name!r} has the wrong row count.")
    return values


def validate_persisted_body_alignment_summary(handle: Any) -> Mapping[str, Any]:
    """Return one read-only, conserved summary view from an exact source handle."""

    handle.require_verified_arrays(SUMMARY_VIEW_ARRAY_NAMES)
    scientific = handle.scientific_manifest
    _manifest_self_contract(scientific)
    dimensions = scientific.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "n_frames",
        "n_chasers",
        "n_frame_rows",
        "n_epochs",
        "n_distance_bins",
        "n_summary_rows",
    }:
        raise ValueError("Body alignment lacks exact dimensions.")
    n_frames = int(dimensions.get("n_frames", 0))
    n_bins = int(dimensions.get("n_distance_bins", 0))
    n_chasers = int(dimensions.get("n_chasers", 0))
    n_rows = int(dimensions.get("n_summary_rows", 0))
    if (
        n_frames <= 0
        or n_bins <= 0
        or n_chasers <= 0
        or int(dimensions.get("n_frame_rows", 0)) != n_frames * n_chasers
        or int(dimensions.get("n_epochs", 0)) != 3
        or n_rows != 3 * n_chasers * n_bins
    ):
        raise ValueError("Body-alignment summary dimensions are inconsistent.")

    edges = _array(handle, "distance_bin_edges_mm")
    if (
        edges.shape != (n_bins + 1,)
        or edges[0] != 0.0
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("Body-alignment persisted distance edges are invalid.")
    recipe = dict(scientific["distance_bin_recipe"])
    recipe_digest = _digest(
        recipe.pop("recipe_sha256", None), field="distance-bin recipe"
    )
    recipe_edges = np.asarray(recipe.get("edges_mm"), dtype=np.float64)
    width = float(recipe.get("bin_width_mm", np.nan))
    if (
        canonical_json_sha256(recipe) != recipe_digest
        or recipe.get("policy_id") != DISTANCE_BIN_POLICY_ID
        or recipe.get("edge_count") != n_bins + 1
        or recipe.get("source_scope")
        != "all_occurrence_member_distance_valid_semantic_epoch_rows"
        or recipe.get("interval_policy") != "half_open_except_final_closed"
        or recipe.get("zero_anchored") is not True
        or recipe_edges.shape != edges.shape
        or not np.array_equal(recipe_edges, edges)
        or not np.isfinite(width)
        or width <= 0.0
        or not np.allclose(np.diff(edges), width, rtol=0.0, atol=1e-12)
        or recipe.get("edge_array_sha256") != array_values_sha256(edges)
    ):
        raise ValueError("Body-alignment distance-bin recipe is incompatible.")

    epoch_records = scientific.get("epoch_records")
    if (
        not isinstance(epoch_records, (tuple, list))
        or len(epoch_records) != 3
        or scientific.get("epoch_records_sha256")
        != canonical_json_sha256(_plain(epoch_records))
    ):
        raise ValueError("Body-alignment epoch records are incompatible.")
    expected_windows: dict[int, int] = {}
    previous_end: int | None = None
    for code, (record, role) in enumerate(
        zip(
            epoch_records,
            ("chaser_pre", "chaser_training", "chaser_post"),
            strict=True,
        ),
        start=1,
    ):
        if not isinstance(record, Mapping):
            raise ValueError("Body-alignment epoch record is malformed.")
        start = record.get("start_frame")
        end = record.get("end_frame_exclusive")
        window = record.get("window_id")
        if (
            record.get("epoch_role_code") != code
            or record.get("analysis_role") != role
            or type(window) is not int
            or type(start) is not int
            or type(end) is not int
            or end <= start
            or (previous_end is not None and start < previous_end)
        ):
            raise ValueError("Body-alignment epoch record is malformed.")
        _digest(
            record.get("source_interval_sha256"),
            field=f"epoch {code} source interval",
        )
        expected_windows[code] = window
        previous_end = end

    declarations = {item["path"]: item for item in scientific["array_declarations"]}
    for path, declaration in declarations.items():
        expected_size = (
            n_frames * n_chasers
            if path.startswith("frame_")
            else n_bins + 1
            if path == "distance_bin_edges_mm"
            else n_rows
        )
        if list(declaration["shape"]) != [expected_size]:
            raise ValueError(
                f"Body-alignment array declaration {path!r} has the wrong shape."
            )
    reasons = scientific.get("reason_codes")
    if (
        not isinstance(reasons, Mapping)
        or _plain(reasons.get("alignment"))
        != {str(code): label for code, label in ALIGNMENT_REASON_CODES.items()}
        or reasons.get("source_reason_codes")
        != "retained_verbatim_from_relative_frame_arrays"
        or not isinstance(scientific.get("identity_registries"), Mapping)
    ):
        raise ValueError("Body-alignment reason or identity registries are invalid.")
    arrays = {
        name: _array(handle, name, size=n_rows)
        for name in SUMMARY_VIEW_ARRAY_NAMES
        if name != "distance_bin_edges_mm"
    }
    integer_count_names = (
        "summary_candidate_row_count",
        "summary_joint_valid_row_count",
        "summary_body_source_missing_row_count",
        "summary_body_heading_invalid_row_count",
        "summary_body_bearing_invalid_row_count",
        "summary_other_alignment_invalid_row_count",
        "summary_epoch_occurrence_row_count",
        "summary_epoch_chaser_absent_row_count",
        "summary_epoch_distance_valid_row_count",
        "summary_epoch_distance_invalid_row_count",
        "summary_epoch_distance_invalid_body_valid_row_count",
    )
    if any(np.any(arrays[name] < 0) for name in integer_count_names):
        raise ValueError("Body-alignment persisted support contains a negative count.")

    role_code = arrays["summary_epoch_role_code"].astype(np.int64)
    identity = arrays["summary_chaser_identity_code"].astype(np.int64)
    bin_index = arrays["summary_distance_bin_index"].astype(np.int64)
    identities = tuple(sorted(int(value) for value in np.unique(identity)))
    if len(identities) != n_chasers or set(np.unique(role_code)) != {1, 2, 3}:
        raise ValueError("Body-alignment epoch or chaser identities are incomplete.")
    registries = scientific["identity_registries"]
    chaser_registry = registries.get("chaser")
    behavior_registry = registries.get("behavior_role")
    observed_behavior_roles = {
        int(value) for value in np.unique(arrays["summary_chaser_behavior_role_code"])
    }
    if (
        not isinstance(chaser_registry, Mapping)
        or not isinstance(behavior_registry, Mapping)
        or any(
            str(identity_code) not in chaser_registry for identity_code in identities
        )
        or any(
            str(behavior_code) not in behavior_registry
            for behavior_code in observed_behavior_roles
        )
    ):
        raise ValueError("Body-alignment summary identities lack registry labels.")
    for epoch_code, expected_window in expected_windows.items():
        observed_windows = np.unique(
            arrays["summary_epoch_window_id"][role_code == epoch_code]
        )
        if not np.array_equal(observed_windows, [expected_window]):
            raise ValueError(
                "Body-alignment summary window differs from its epoch record."
            )
    expected_keys = {
        (epoch, chaser, distance_bin)
        for epoch in (1, 2, 3)
        for chaser in identities
        for distance_bin in range(n_bins)
    }
    observed_keys = list(zip(role_code, identity, bin_index, strict=True))
    if len(set(observed_keys)) != n_rows or set(observed_keys) != expected_keys:
        raise ValueError("Body-alignment summary keys are missing or duplicated.")

    start = arrays["summary_distance_bin_start_mm"].astype(np.float64)
    end = arrays["summary_distance_bin_end_mm"].astype(np.float64)
    center = arrays["summary_distance_bin_center_mm"].astype(np.float64)
    if (
        np.any(~np.isfinite(start))
        or np.any(~np.isfinite(end))
        or np.any(~np.isfinite(center))
        or not np.array_equal(start, edges[bin_index])
        or not np.array_equal(end, edges[bin_index + 1])
        or not np.allclose(center, (start + end) / 2.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Body-alignment summary rows differ from persisted bin edges.")

    candidate = arrays["summary_candidate_row_count"].astype(np.int64)
    joint = arrays["summary_joint_valid_row_count"].astype(np.int64)
    invalid_in_bin = sum(
        arrays[name].astype(np.int64)
        for name in (
            "summary_body_source_missing_row_count",
            "summary_body_heading_invalid_row_count",
            "summary_body_bearing_invalid_row_count",
            "summary_other_alignment_invalid_row_count",
        )
    )
    if np.any(joint > candidate) or np.any(joint + invalid_in_bin != candidate):
        raise ValueError(
            "Body-alignment jointly valid and invalid reasons do not conserve bins."
        )

    for epoch, chaser in ((e, c) for e in (1, 2, 3) for c in identities):
        member = (role_code == epoch) & (identity == chaser)
        order = np.argsort(bin_index[member], kind="stable")
        indices = np.flatnonzero(member)[order]
        if not np.array_equal(bin_index[indices], np.arange(n_bins)):
            raise ValueError("Body-alignment distance bins are not contiguous.")
        repeated_names = (
            "summary_epoch_window_id",
            "summary_chaser_behavior_role_code",
            "summary_epoch_occurrence_row_count",
            "summary_epoch_chaser_absent_row_count",
            "summary_epoch_distance_valid_row_count",
            "summary_epoch_distance_invalid_row_count",
            "summary_epoch_distance_invalid_body_valid_row_count",
        )
        if any(np.unique(arrays[name][indices]).size != 1 for name in repeated_names):
            raise ValueError(
                "Body-alignment epoch/chaser support is not stable across bins."
            )
        distance_valid_total = int(
            arrays["summary_epoch_distance_valid_row_count"][indices[0]]
        )
        distance_invalid_total = int(
            arrays["summary_epoch_distance_invalid_row_count"][indices[0]]
        )
        occurrence_total = int(arrays["summary_epoch_occurrence_row_count"][indices[0]])
        distance_invalid_body_valid = int(
            arrays["summary_epoch_distance_invalid_body_valid_row_count"][indices[0]]
        )
        if (
            int(np.sum(candidate[indices])) != distance_valid_total
            or distance_valid_total + distance_invalid_total != occurrence_total
            or distance_invalid_body_valid > distance_invalid_total
        ):
            raise ValueError(
                "Body-alignment distance validity does not conserve epoch occurrence rows."
            )
    for epoch in (1, 2, 3):
        epoch_totals = []
        for chaser in identities:
            member = (role_code == epoch) & (identity == chaser)
            first = np.flatnonzero(member)[0]
            epoch_totals.append(
                int(arrays["summary_epoch_occurrence_row_count"][first])
                + int(arrays["summary_epoch_chaser_absent_row_count"][first])
            )
        if len(set(epoch_totals)) != 1 or epoch_totals[0] <= 0:
            raise ValueError(
                "Body-alignment epoch frame support differs across chasers."
            )

    statistic_names = (
        "summary_mean_alignment_cos",
        "summary_alignment_cos_p25",
        "summary_alignment_cos_p50",
        "summary_alignment_cos_p75",
        "summary_mean_abs_bearing_deg",
        "summary_abs_bearing_p25_deg",
        "summary_abs_bearing_p50_deg",
        "summary_abs_bearing_p75_deg",
        "summary_circular_mean_bearing_deg",
        "summary_circular_resultant_length",
    )
    empty = joint == 0
    if any(np.any(np.isfinite(arrays[name][empty])) for name in statistic_names):
        raise ValueError("An empty body-alignment bin contains a finite statistic.")
    populated = ~empty
    finite_required = tuple(
        name for name in statistic_names if name != "summary_circular_mean_bearing_deg"
    )
    if any(np.any(~np.isfinite(arrays[name][populated])) for name in finite_required):
        raise ValueError("A populated body-alignment bin lacks a finite statistic.")
    circular_mean = arrays["summary_circular_mean_bearing_deg"]
    resultant = arrays["summary_circular_resultant_length"]
    undefined = populated & ~np.isfinite(circular_mean)
    if np.any(resultant[undefined] > 1e-12):
        raise ValueError("A circular mean is undefined despite nonzero resultant.")
    if (
        np.any(np.abs(arrays["summary_mean_alignment_cos"][populated]) > 1.0 + 1e-12)
        or any(
            np.any(np.abs(arrays[name][populated]) > 1.0 + 1e-12)
            for name in (
                "summary_alignment_cos_p25",
                "summary_alignment_cos_p50",
                "summary_alignment_cos_p75",
            )
        )
        or np.any(arrays["summary_mean_abs_bearing_deg"][populated] < 0.0)
        or np.any(arrays["summary_mean_abs_bearing_deg"][populated] > 180.0)
        or any(
            np.any(arrays[name][populated] < 0.0)
            or np.any(arrays[name][populated] > 180.0)
            for name in (
                "summary_abs_bearing_p25_deg",
                "summary_abs_bearing_p50_deg",
                "summary_abs_bearing_p75_deg",
            )
        )
        or np.any(circular_mean[np.isfinite(circular_mean)] < -180.0)
        or np.any(circular_mean[np.isfinite(circular_mean)] > 180.0)
        or np.any(resultant[populated] < 0.0)
        or np.any(resultant[populated] > 1.0 + 1e-12)
    ):
        raise ValueError(
            "Body-alignment persisted statistics are outside their ranges."
        )
    if (
        np.any(
            arrays["summary_alignment_cos_p25"][populated]
            > arrays["summary_alignment_cos_p50"][populated]
        )
        or np.any(
            arrays["summary_alignment_cos_p50"][populated]
            > arrays["summary_alignment_cos_p75"][populated]
        )
        or np.any(
            arrays["summary_abs_bearing_p25_deg"][populated]
            > arrays["summary_abs_bearing_p50_deg"][populated]
        )
        or np.any(
            arrays["summary_abs_bearing_p50_deg"][populated]
            > arrays["summary_abs_bearing_p75_deg"][populated]
        )
    ):
        raise ValueError("Body-alignment persisted quantiles are not ordered.")
    return _freeze({"distance_bin_edges_mm": edges, "identities": identities, **arrays})


__all__ = ["validate_persisted_body_alignment_summary"]
