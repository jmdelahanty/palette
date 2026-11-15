"""Palette's recompute of orange_dish_rim_auto_accept_v1 against the agent-contracts fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files as resource_files
from pathlib import Path

import pytest

from fisheye.shared import dish_rim_gate
from fisheye.shared.dish_rim_gate import (
    FAILURE_CODES,
    DishRimGateError,
    recompute_gate_failures,
    verify_gate_accepted_fit,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "orange_dish_rim_v4"
VALID_EVALUATIONS = sorted((FIXTURES / "valid").glob("evaluation_*.json"))
INVALID_EVALUATIONS = sorted((FIXTURES / "invalid").glob("evaluation_*.json"))


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_bytes())


def _accepted() -> dict:
    return _load("valid/evaluation_accepted.json")


def _fit_kwargs(evaluation: dict) -> dict:
    fit = evaluation["fit"]
    return {
        "camera_serial": evaluation["camera_serial"],
        "fit_center_x_px": fit["center_px"]["x"],
        "fit_center_y_px": fit["center_px"]["y"],
        "fit_radius_px": fit["radius_px"],
        "image_width_px": fit["image_width_px"],
        "image_height_px": fit["image_height_px"],
    }


def test_vendored_schemas_are_the_pinned_bytes() -> None:
    contracts = resource_files("fisheye.shared").joinpath("contracts")
    for name, digest in (
        (dish_rim_gate._SCHEMA_FILE, dish_rim_gate._SCHEMA_SHA256),
        (dish_rim_gate._REQUESTER_SCHEMA_FILE, dish_rim_gate._REQUESTER_SCHEMA_SHA256),
    ):
        assert hashlib.sha256(contracts.joinpath(name).read_bytes()).hexdigest() == digest


def test_fixture_counts_match_the_contract() -> None:
    assert [p.name for p in VALID_EVALUATIONS] == [
        "evaluation_accepted.json",
        "evaluation_refused_outer_wall.json",
    ]
    assert len(INVALID_EVALUATIONS) == 12


@pytest.mark.parametrize("path", VALID_EVALUATIONS, ids=lambda p: p.name)
def test_palette_recomputes_orange_gate_output(path: Path) -> None:
    # Every derived number Orange's C++ gate recorded is reproduced from the
    # recorded inputs, and the failures match in content and order.
    evaluation = json.loads(path.read_bytes())
    assert list(recompute_gate_failures(evaluation)) == evaluation["failures"]


def test_refused_fixture_names_four_failures() -> None:
    assert recompute_gate_failures(_load("valid/evaluation_refused_outer_wall.json")) == (
        "radius_outside_physical_range",
        "radius_change_exceeds_limit",
        "edge_support_below_limit",
        "fallback_hough_pass",
    )


@pytest.mark.parametrize("path", INVALID_EVALUATIONS, ids=lambda p: p.name)
def test_every_invalid_contract_fixture_is_refused(path: Path) -> None:
    with pytest.raises(DishRimGateError):
        recompute_gate_failures(json.loads(path.read_bytes()))


@pytest.mark.parametrize(
    "path",
    [
        ("checks", "radius_in_physical_range", "px_per_mm_low"),
        ("checks", "radius_in_physical_range", "expected_high_px"),
        ("checks", "radius_in_physical_range", "accept_low_px"),
        ("checks", "radius_in_physical_range", "radius_mm"),
        ("checks", "radius_in_physical_range", "z_low_mm"),
        ("checks", "change_from_anchor", "px_per_mm"),
        ("checks", "change_from_anchor", "center_shift_px"),
        ("checks", "change_from_anchor", "radius_change_mm"),
        ("checks", "change_from_anchor", "max_center_shift_mm"),
    ],
)
def test_a_derived_value_palette_cannot_reproduce_is_refused(path: tuple[str, ...]) -> None:
    evaluation = _accepted()
    node = evaluation
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]] * (1 + 1e-6)
    with pytest.raises(DishRimGateError, match=path[-1]):
        recompute_gate_failures(evaluation)


def test_changing_an_input_breaks_the_recorded_band() -> None:
    # The fish-plane scale is bound by sha256; a record whose numbers came from
    # a different model than the one it names does not recompute.
    evaluation = _accepted()
    evaluation["inputs"]["fish_plane_scale"]["k_px"] *= 1.01
    with pytest.raises(DishRimGateError, match="px_per_mm_low"):
        recompute_gate_failures(evaluation)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e["checks"]["edge_support"].__setitem__("fraction", 0.39), "edge_support.passed"),
        (lambda e: e["checks"]["edge_support"].__setitem__("fraction", None), "edge_support.passed"),
        (lambda e: e["checks"]["first_hough_pass"].__setitem__("hough_pass_index", 2), "first_hough_pass.passed"),
        (lambda e: e["checks"]["first_hough_pass"].__setitem__("hough_pass_index", 0), "first_hough_pass.passed"),
    ],
)
def test_palette_decides_with_its_own_limits(mutate, message: str) -> None:
    # Each record still passes the schema (passed stays true, outcome accepted);
    # Palette's verdict differs from Orange's, so the record is refused.
    evaluation = _accepted()
    mutate(evaluation)
    dish_rim_gate._require_schema(evaluation)
    with pytest.raises(DishRimGateError, match=message):
        recompute_gate_failures(evaluation)


def test_shift_and_radius_limits_are_inclusive_at_the_limit() -> None:
    # Orange's limits are "at most"; a fit exactly at a limit passes.
    evaluation = _accepted()
    check = evaluation["checks"]["change_from_anchor"]
    anchor = check["anchor"]
    fit = evaluation["fit"]
    fit["radius_px"] = anchor["radius_px"] - 35.0  # inside the physical band
    evaluation["checks"]["radius_in_physical_range"]["radius_px"] = fit["radius_px"]
    scale = check["px_per_mm"]
    evaluation["checks"]["radius_in_physical_range"]["radius_mm"] = fit["radius_px"] / scale
    fit["center_px"] = {"x": anchor["center_px"]["x"] + 135.0, "y": anchor["center_px"]["y"]}
    check.update(
        center_shift_px=135.0,
        center_shift_mm=135.0 / scale,
        radius_change_px=35.0,
        radius_change_mm=35.0 / scale,
    )
    assert recompute_gate_failures(evaluation) == ()


def test_jeremys_drift_rule_refuses_a_gate_accepted_anchor() -> None:
    # The anchor must be the last operator-reviewed fit, never an earlier gate
    # acceptance. The schema allows such a record; Palette's recompute does not.
    evaluation = _accepted()
    evaluation["checks"]["change_from_anchor"]["anchor"]["acceptance_basis"] = "quality_gate"
    evaluation["drift"]["anchor"]["acceptance_basis"] = "quality_gate"
    dish_rim_gate._require_schema(evaluation)
    with pytest.raises(DishRimGateError, match="change_from_anchor.passed"):
        recompute_gate_failures(evaluation)


def test_drift_anchor_must_be_the_compared_fit() -> None:
    evaluation = _accepted()
    evaluation["drift"]["anchor"]["observation_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(DishRimGateError, match="drift anchor"):
        recompute_gate_failures(evaluation)


def test_refused_failures_must_be_exact() -> None:
    evaluation = _load("valid/evaluation_refused_outer_wall.json")
    evaluation["failures"] = evaluation["failures"] + ["rig_config_not_pinned"]
    with pytest.raises(DishRimGateError, match="differ from Palette's recompute"):
        recompute_gate_failures(evaluation)
    evaluation = _load("valid/evaluation_refused_outer_wall.json")
    evaluation["failures"] = evaluation["failures"][:-1]
    with pytest.raises(DishRimGateError, match="differ from Palette's recompute"):
        recompute_gate_failures(evaluation)


def test_failure_codes_are_the_schema_enum_in_order() -> None:
    schema = json.loads(
        resource_files("fisheye.shared").joinpath("contracts").joinpath(dish_rim_gate._SCHEMA_FILE).read_bytes()
    )
    assert tuple(schema["properties"]["failures"]["items"]["enum"]) == FAILURE_CODES


def test_non_finite_fit_is_refused() -> None:
    evaluation = _accepted()
    evaluation["fit"]["radius_px"] = float("nan")
    with pytest.raises(DishRimGateError):
        recompute_gate_failures(evaluation)


def test_verify_accepts_exactly_the_evaluated_fit() -> None:
    evaluation = _accepted()
    verify_gate_accepted_fit(evaluation, **_fit_kwargs(evaluation))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"camera_serial": "2010094"}, "camera_serial"),
        ({"fit_radius_px": 2158.81}, "accepted inner rim"),
        ({"fit_center_x_px": 2245.9}, "accepted inner rim"),
        ({"image_height_px": 4096}, "image size"),
    ],
)
def test_verify_refuses_a_different_fit(override: dict, message: str) -> None:
    evaluation = _accepted()
    with pytest.raises(DishRimGateError, match=message):
        verify_gate_accepted_fit(evaluation, **{**_fit_kwargs(evaluation), **override})


def test_verify_refuses_a_refused_evaluation_and_a_missing_one() -> None:
    refused = _load("valid/evaluation_refused_outer_wall.json")
    with pytest.raises(DishRimGateError, match="does not accept"):
        verify_gate_accepted_fit(refused, **_fit_kwargs(refused))
    with pytest.raises(DishRimGateError, match="lacks acceptance.gate"):
        verify_gate_accepted_fit(None, **_fit_kwargs(_accepted()))


@pytest.mark.parametrize(
    "name",
    ["valid/observation_v4_quality_gate.json", "valid/contract_v3_quality_gate_selection.json"],
)
def test_gates_embedded_in_contract_fixtures_recompute(name: str) -> None:
    document = _load(name)
    if "acceptance" in document:
        gates = [document["acceptance"]["gate"]]
    else:
        gates = [
            document["cameras"]["2010093"]["physical_registration"]["recording_snapshot_entry"]["acceptance"]["gate"],
            document["physical_registration_geometry"]["cameras"]["2010093"]["recording_snapshot_entry"][
                "acceptance"
            ]["gate"],
        ]
    for gate in gates:
        assert recompute_gate_failures(copy.deepcopy(gate)) == ()
