"""Palette's recompute of Orange's dish-rim auto-accept gate (``orange_dish_rim_auto_accept_v1``).

A rim observation v4 whose ``acceptance.basis`` is ``quality_gate`` was accepted
by Orange's gate, not by a person. Palette admits it only after it has checked
the gate's evaluation itself (agent-contracts ``orange-palette/dish_rim_registration_v4``,
#133 + #134, merged ``6695514``):

- the evaluation validates against the pinned schema;
- every derived number (the radius band from the fish-plane scale model, the
  centre shift, the radius change and every mm value) is recomputed from the
  recorded inputs and must match what Orange recorded;
- each check's verdict is recomputed with Palette's own copy of the gate's
  limits, never the limits the record carries;
- ``failures`` must list exactly the codes of the checks that failed, and
  ``outcome`` must agree with it;
- the drift anchor must be the camera's last operator-reviewed fit (Jeremy's
  drift rule, 2026-11-15) and must be the same fit in ``drift`` and in
  ``checks.change_from_anchor``.

Edge support and the Hough pass index are measurements of the source frames,
which Palette does not have. Palette rechecks their verdicts against the
limits but cannot remeasure them. Palette also cannot open the anchor
observation (only its artifact id and sha256 are recorded) or the fish-plane
scale file; both stay bound by sha256.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from importlib.resources import files as resource_files
from typing import Any, Mapping

GATE_EVALUATION_SCHEMA_ID = "orange.calibration.dish_rim_gate_evaluation"
GATE_EVALUATION_SCHEMA_VERSION = 1
GATE_ID = "orange_dish_rim_auto_accept_v1"
GATE_VERSION = 1
DRIFT_RULE = "compare_to_last_operator_reviewed_fit"
ANCHOR_BASIS_OPERATOR_REVIEW = "operator_review"

# Byte-identical to agent-contracts 3480e3e
# orange-palette/dish_rim_registration_v4/schemas/ (see contracts/README.md).
_SCHEMA_FILE = "orange_calibration_dish_rim_gate_evaluation_v1.schema.json"
_SCHEMA_SHA256 = "518c33d65259503d72f684e29c6fdb7ea375c08db6effb610756eeca42a9c506"
# The evaluation's ``requester`` is ``{"$ref": "orange.requester.v1"}``.
_REQUESTER_SCHEMA_FILE = "orange_requester_v1.schema.json"
_REQUESTER_SCHEMA_SHA256 = "58819f17c23feb0bb526c7254d088af24434899ced57b3093c3c8a64bdc4a687"

# Gate v1 limits, confirmed by Jeremy 2026-11-15. Palette decides with these;
# a record that carries different limits fails the schema (a different limit
# is a different gate version).
TOLERANCE_FRACTION = 0.02
MAX_CENTER_SHIFT_PX = 135.0
MAX_RADIUS_CHANGE_PX = 35.0
MIN_EDGE_FRACTION = 0.4
EDGE_POINT_COUNT = 720
REQUIRED_HOUGH_PASS_INDEX = 1
INNER_DIAMETER_MM = 80.0
WALL_HEIGHT_MM = 2.8
WATER_REFRACTIVE_INDEX = 1.333

# The order Orange lists failures in (the evaluation schema's enum).
FAILURE_CODES = (
    "fit_not_finite",
    "radius_outside_physical_range",
    "anchor_fit_missing",
    "center_shift_exceeds_limit",
    "radius_change_exceeds_limit",
    "edge_support_not_measured",
    "edge_support_below_limit",
    "fallback_hough_pass",
    "rig_config_not_pinned",
    "anchor_not_operator_reviewed",
)
_CHANGE_FAILURES = frozenset({
    "anchor_fit_missing",
    "center_shift_exceeds_limit",
    "radius_change_exceeds_limit",
    "anchor_not_operator_reviewed",
})

# Recorded derived values are float64 results of the same arithmetic; allow
# only rounding differences.
_REL_TOL = 1e-9
_ABS_TOL = 1e-9


class DishRimGateError(ValueError):
    """A gate evaluation is malformed, inconsistent, or does not accept the fit."""


def _pinned_schema(file_name: str, digest: str) -> Mapping[str, Any]:
    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(file_name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise DishRimGateError("packaged_contract_drift:" + file_name)
    return json.loads(data)


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    requester = _pinned_schema(_REQUESTER_SCHEMA_FILE, _REQUESTER_SCHEMA_SHA256)
    # Only the pinned requester schema resolves; nothing is fetched.
    registry = Registry().with_resource(requester["$id"], Resource.from_contents(requester))
    return Draft202012Validator(_pinned_schema(_SCHEMA_FILE, _SCHEMA_SHA256), registry=registry)


def _require_schema(evaluation: Mapping[str, Any]) -> None:
    from jsonschema.exceptions import best_match

    error = best_match(_validator().iter_errors(evaluation))
    if error is not None:
        where = "/".join(str(part) for part in error.absolute_path) or "(root)"
        raise DishRimGateError(f"Gate evaluation fails its schema at {where}: {error.message}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _agrees(label: str, recorded: Any, recomputed: float) -> None:
    if not _finite(recorded) or not math.isclose(
        float(recorded), recomputed, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
    ):
        raise DishRimGateError(
            f"Gate evaluation {label} is {recorded!r}; Palette recomputes {recomputed!r}."
        )


def _scale_at(z_mm: float, scale: Mapping[str, Any]) -> float:
    """px/mm at height z: 1/s(z) = 1/s_floor - (z - p_floor - e)/k."""

    inverse = 1.0 / float(scale["s_floor_px_per_mm"]) - (
        z_mm - float(scale["p_floor_mm"]) - float(scale["e_mm"])
    ) / float(scale["k_px"])
    if not inverse > 0:
        raise DishRimGateError("Gate evaluation fish-plane scale model gives no positive scale at the rim.")
    return 1.0 / inverse


def _radius_failures(evaluation: Mapping[str, Any]) -> tuple[list[str], float]:
    check = evaluation["checks"]["radius_in_physical_range"]
    dish = evaluation["inputs"]["dish"]
    scale = evaluation["inputs"]["fish_plane_scale"]
    wall = float(dish["wall_height_mm"])
    index = float(dish["water_refractive_index"])
    p_floor = float(scale["p_floor_mm"])
    z_low = p_floor + wall * (1.0 - 1.0 / index)
    z_high = p_floor + wall
    _agrees("radius_in_physical_range.z_low_mm", check["z_low_mm"], z_low)
    _agrees("radius_in_physical_range.z_high_mm", check["z_high_mm"], z_high)
    low = _scale_at(z_low, scale)
    high = _scale_at(z_high, scale)
    _agrees("radius_in_physical_range.px_per_mm_low", check["px_per_mm_low"], low)
    _agrees("radius_in_physical_range.px_per_mm_high", check["px_per_mm_high"], high)
    inner_radius = float(dish["inner_diameter_mm"]) / 2.0
    expected_low = inner_radius * low
    expected_high = inner_radius * high
    accept_low = expected_low * (1.0 - TOLERANCE_FRACTION)
    accept_high = expected_high * (1.0 + TOLERANCE_FRACTION)
    for name, value in (
        ("expected_low_px", expected_low),
        ("expected_high_px", expected_high),
        ("accept_low_px", accept_low),
        ("accept_high_px", accept_high),
        ("accept_low_mm", accept_low / low),
        ("accept_high_mm", accept_high / high),
    ):
        _agrees(f"radius_in_physical_range.{name}", check[name], value)
    mean_scale = (low + high) / 2.0
    radius = float(evaluation["fit"]["radius_px"])
    _agrees("radius_in_physical_range.radius_px", check["radius_px"], radius)
    _agrees("radius_in_physical_range.radius_mm", check["radius_mm"], radius / mean_scale)
    failures = [] if accept_low <= radius <= accept_high else ["radius_outside_physical_range"]
    return failures, mean_scale


def _anchor_identity(anchor: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(anchor[key] for key in ("artifact_id", "observation_sha256", "created_utc", "acceptance_basis"))


def _change_failures(evaluation: Mapping[str, Any], mean_scale: float) -> list[str]:
    check = evaluation["checks"]["change_from_anchor"]
    drift = evaluation["drift"]
    _agrees("change_from_anchor.px_per_mm", check["px_per_mm"], mean_scale)
    _agrees("change_from_anchor.max_center_shift_mm", check["max_center_shift_mm"], MAX_CENTER_SHIFT_PX / mean_scale)
    _agrees("change_from_anchor.max_radius_change_mm", check["max_radius_change_mm"], MAX_RADIUS_CHANGE_PX / mean_scale)
    anchor = check["anchor"]
    if anchor is None:
        if drift["anchor"] is not None or drift["chain_depth"] is not None:
            raise DishRimGateError("Gate evaluation drift names an anchor that change_from_anchor lacks.")
        return ["anchor_fit_missing"]
    if drift["anchor"] is None or _anchor_identity(drift["anchor"]) != _anchor_identity(anchor):
        raise DishRimGateError("Gate evaluation drift anchor is not the change_from_anchor fit.")
    if drift["chain_depth"] is None:
        raise DishRimGateError("Gate evaluation drift has an anchor but no chain_depth.")
    failures = []
    if anchor["acceptance_basis"] != ANCHOR_BASIS_OPERATOR_REVIEW:
        failures.append("anchor_not_operator_reviewed")
    fit = evaluation["fit"]
    shift = math.hypot(
        float(fit["center_px"]["x"]) - float(anchor["center_px"]["x"]),
        float(fit["center_px"]["y"]) - float(anchor["center_px"]["y"]),
    )
    change = abs(float(fit["radius_px"]) - float(anchor["radius_px"]))
    _agrees("change_from_anchor.center_shift_px", check["center_shift_px"], shift)
    _agrees("change_from_anchor.center_shift_mm", check["center_shift_mm"], shift / mean_scale)
    _agrees("change_from_anchor.radius_change_px", check["radius_change_px"], change)
    _agrees("change_from_anchor.radius_change_mm", check["radius_change_mm"], change / mean_scale)
    if shift > MAX_CENTER_SHIFT_PX:
        failures.append("center_shift_exceeds_limit")
    if change > MAX_RADIUS_CHANGE_PX:
        failures.append("radius_change_exceeds_limit")
    return failures


def recompute_gate_failures(evaluation: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate one gate evaluation and return the failures Palette recomputes.

    Raises :class:`DishRimGateError` when the record is malformed, when a
    derived value disagrees with Palette's recompute, or when a check's
    ``passed``, ``failures`` or ``outcome`` disagrees with Palette's verdict.
    """

    _require_schema(evaluation)
    if (evaluation["gate_id"], evaluation["gate_version"]) != (GATE_ID, GATE_VERSION):
        raise DishRimGateError("Gate evaluation is not orange_dish_rim_auto_accept_v1.")
    if evaluation["drift"]["rule"] != DRIFT_RULE:
        raise DishRimGateError("Gate evaluation does not use the last-operator-reviewed drift rule.")
    fit = evaluation["fit"]
    if not all(_finite(value) for value in (fit["center_px"]["x"], fit["center_px"]["y"], fit["radius_px"])):
        # Orange refuses such a fit (fit_not_finite); Palette cannot recompute
        # the other checks from it, so it refuses the record outright.
        raise DishRimGateError("Gate evaluation fit is not finite.")

    failures: set[str] = set()
    radius_failures, mean_scale = _radius_failures(evaluation)
    change_failures = _change_failures(evaluation, mean_scale)
    edge = evaluation["checks"]["edge_support"]
    if edge["fraction"] is None:
        edge_failures = ["edge_support_not_measured"]
    elif not edge["fraction"] >= MIN_EDGE_FRACTION:
        edge_failures = ["edge_support_below_limit"]
    else:
        edge_failures = []
    hough = evaluation["checks"]["first_hough_pass"]
    hough_failures = [] if hough["hough_pass_index"] == REQUIRED_HOUGH_PASS_INDEX else ["fallback_hough_pass"]
    failures.update(radius_failures, change_failures, edge_failures, hough_failures)
    if evaluation["inputs"]["rig_config"]["status"] != "pinned":
        failures.add("rig_config_not_pinned")

    for name, check_failures in (
        ("radius_in_physical_range", radius_failures),
        ("change_from_anchor", change_failures),
        ("edge_support", edge_failures),
        ("first_hough_pass", hough_failures),
    ):
        if evaluation["checks"][name]["passed"] is not (not check_failures):
            raise DishRimGateError(
                f"Gate evaluation {name}.passed is {evaluation['checks'][name]['passed']!r}; "
                f"Palette recomputes {not check_failures!r}."
            )
    recorded = evaluation["failures"]
    if len(recorded) != len(set(recorded)) or set(recorded) != failures:
        raise DishRimGateError(
            f"Gate evaluation failures {list(recorded)!r} differ from Palette's recompute "
            f"{[code for code in FAILURE_CODES if code in failures]!r}."
        )
    outcome = "refused" if failures else "accepted"
    if evaluation["outcome"] != outcome:
        raise DishRimGateError(
            f"Gate evaluation outcome is {evaluation['outcome']!r}; Palette recomputes {outcome!r}."
        )
    return tuple(code for code in FAILURE_CODES if code in failures)


def verify_gate_accepted_fit(
    evaluation: Any,
    *,
    camera_serial: str,
    fit_center_x_px: float,
    fit_center_y_px: float,
    fit_radius_px: float,
    image_width_px: int,
    image_height_px: int,
) -> None:
    """Require a gate evaluation that Palette recomputes as accepting this fit.

    The fit is the observation's ``accepted_inner_rim_boundary`` circle and the
    image is the observation's camera frame: the gate must have evaluated
    exactly that circle, for that camera.
    """

    if not isinstance(evaluation, Mapping):
        raise DishRimGateError("Gate-accepted rim observation lacks acceptance.gate.")
    if recompute_gate_failures(evaluation):
        raise DishRimGateError("Gate evaluation does not accept the fit.")
    if str(evaluation["camera_serial"]) != camera_serial:
        raise DishRimGateError("Gate evaluation camera_serial is not the observation's camera.")
    fit = evaluation["fit"]
    if (fit["image_width_px"], fit["image_height_px"]) != (image_width_px, image_height_px):
        raise DishRimGateError("Gate evaluation image size is not the observation's camera frame.")
    if (
        float(fit["center_px"]["x"]),
        float(fit["center_px"]["y"]),
        float(fit["radius_px"]),
    ) != (float(fit_center_x_px), float(fit_center_y_px), float(fit_radius_px)):
        raise DishRimGateError("Gate evaluation fit is not the observation's accepted inner rim.")
