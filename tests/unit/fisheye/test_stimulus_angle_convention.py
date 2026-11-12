"""Stimulus angle convention v1 and its legacy predecessor (agent-contracts PR 91)."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files

import numpy as np
import pytest

from fisheye.shared.heading_convention import (
    ANGLE_CONVENTION,
    LEGACY_CANVAS_ANGLE,
    STIMULUS_ANGLE_CONVENTION,
    legacy_canvas_angle_to_degrees,
    stimulus_angle_vector,
    wrap_delta_degrees,
)

CONTRACTS = resource_files("fisheye.shared").joinpath("contracts")
PINNED = {
    # agent-contracts PR 91 head 4d0fa15 (merged f3d97d4), citrus-palette/angles/
    "citrus_palette_stimulus_angle_convention_v1.json": "a99d82e708445db38f48919ba7ed76933b89cdb1f501d1d42b58394327e1b23c",
    "citrus_palette_stimulus_angle_cases_v1.json": "b05f4b19739c1d6ce64e1a47d30b7f0ab53910a10900418fea8b7514a10452a8",
}


def _load(name: str) -> dict:
    data = CONTRACTS.joinpath(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED[name]
    return json.loads(data)


DEFINITION = _load("citrus_palette_stimulus_angle_convention_v1.json")
CASES = _load("citrus_palette_stimulus_angle_cases_v1.json")
TOLERANCE = CASES["tolerance_deg"]


def test_ids_match_the_shared_definition():
    assert DEFINITION["angle"]["id"] == STIMULUS_ANGLE_CONVENTION == ANGLE_CONVENTION
    legacy = DEFINITION["legacy"]
    assert legacy["id"] == LEGACY_CANVAS_ANGLE
    assert legacy["definition"].startswith("0 = +x (right), +90 = down on the raster, positive clockwise")
    assert legacy["conversion"].startswith("theta_v1 = -theta_legacy")
    assert legacy["applies_to_catalogs"] == [
        "historical_v1", "chaser_v2", "bounding_boxes_v2", "frame_identity_v1"
    ]


@pytest.mark.parametrize("case", CASES["legacy_to_v1"], ids=lambda case: case["name"])
def test_legacy_to_v1_cases(case):
    converted = legacy_canvas_angle_to_degrees(case["legacy_deg"])
    assert abs(wrap_delta_degrees(converted - case["expect_v1_deg"])) <= TOLERANCE
    # Same raster, same vector.
    legacy = stimulus_angle_vector(case["legacy_deg"], LEGACY_CANVAS_ANGLE)
    assert np.allclose(stimulus_angle_vector(converted), legacy, atol=CASES["tolerance_vector"])


def test_conversion_is_not_normalised():
    assert legacy_canvas_angle_to_degrees(180.0) == -180.0
    assert legacy_canvas_angle_to_degrees(270.0) == -270.0
    assert legacy_canvas_angle_to_degrees(-180.0) == 180.0
    out = legacy_canvas_angle_to_degrees(np.asarray([90.0, -31.8], dtype=np.float32), dtype=np.float32)
    assert out.dtype == np.float32 and out.tolist() == pytest.approx([-90.0, 31.8])


@pytest.mark.parametrize("case", CASES["angle_to_vector"], ids=lambda case: case["name"])
def test_angle_to_vector_cases(case):
    assert np.allclose(
        stimulus_angle_vector(case["angle_deg"]), case["expect_vector"], atol=CASES["tolerance_vector"]
    )


@pytest.mark.parametrize("case", CASES["wrap_delta"], ids=lambda case: str(case["d"]))
def test_wrap_delta_cases(case):
    assert wrap_delta_degrees(case["d"]) == case["expect"]


def test_cardinals():
    assert np.allclose(stimulus_angle_vector(90.0), (0.0, -1.0))  # up
    assert np.allclose(stimulus_angle_vector(90.0, LEGACY_CANVAS_ANGLE), (0.0, 1.0))  # down
    with pytest.raises(ValueError):
        stimulus_angle_vector(0.0, "unknown")
