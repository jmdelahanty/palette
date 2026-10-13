"""The shared heading module reproduces the agent-contracts reference cases."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files

import numpy as np
import pytest

from fisheye.shared.heading_convention import (
    ANGLE_CONVENTION,
    HANDEDNESS,
    heading_degrees,
    CHASER_CENTRED_BEARING,
    MEGABOUTS_HEAD_YAW,
    chaser_centred_bearing_degrees,
    image_angle_degrees,
    left_axis,
    megabouts_head_yaw_radians,
    wrap_delta_degrees,
    wrap_delta_radians,
)

CONTRACTS = resource_files("fisheye.shared").joinpath("contracts")
PINNED = {
    # agent-contracts citrus-palette/heading at 6b63aa2
    "citrus_palette_heading_convention_v1.json": "95a1076107e778e7c7b9df79726c8904835226945f983590476fb687ed014e30",
    "citrus_palette_heading_cases_v1.json": "9af3436fd63eeb00ea0efe2283f098b2e9e6a9905556498761a02f9bf34d6d06",
}


def _load(name: str) -> dict:
    data = CONTRACTS.joinpath(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED[name]
    return json.loads(data)


def test_ids_match_the_shared_definition():
    definition = _load("citrus_palette_heading_convention_v1.json")
    assert definition["angle"]["id"] == ANGLE_CONVENTION
    assert definition["left_right"]["id"] == HANDEDNESS


CASES = [case for case in _load("citrus_palette_heading_cases_v1.json")["cases"] if "expect_angle_deg" in case]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_reference_cases(case):
    bladder = np.asarray(case["swim_bladder"], dtype=np.float64)
    origin = (np.asarray(case["eye_left"], dtype=np.float64) + np.asarray(case["eye_right"], dtype=np.float64)) / 2
    forward = origin - bladder
    assert np.allclose(origin, case["expect_origin"])
    assert abs(wrap_delta_degrees(heading_degrees(*forward) - case["expect_angle_deg"])) < 1e-9
    unit = forward / np.linalg.norm(forward)
    assert np.allclose(left_axis(*unit), case["expect_left_axis"])
    eye_offset = np.asarray(case["eye_left"], dtype=np.float64) - np.asarray(case["eye_right"], dtype=np.float64)
    assert float(np.dot(eye_offset, left_axis(*forward))) > 0


def test_both_ends_of_the_range_are_one_heading():
    assert heading_degrees(-10.0, 0.0) == -180.0  # IEEE atan2(-0.0, negative)
    assert wrap_delta_degrees(180.0 - -180.0) == 0.0
    assert wrap_delta_degrees(179.0 - -179.0) == -2.0


def test_radian_wrap_matches_the_degree_wrap():
    assert wrap_delta_radians(np.pi) == -np.pi
    assert wrap_delta_radians(-np.pi) == -np.pi
    assert wrap_delta_radians(2.0 * np.pi) == 0.0
    assert np.isnan(wrap_delta_radians(np.nan))


def test_dtype_selects_the_arithmetic_precision():
    forward = np.asarray([[0.3, -0.7]], dtype=np.float32)
    native = np.rad2deg(np.arctan2(-forward[:, 1], forward[:, 0]))
    shared = heading_degrees(forward[:, 0], forward[:, 1], dtype=np.float32)
    assert shared.dtype == np.float32 and shared.tobytes() == native.tobytes()
    assert heading_degrees(forward[:, 0], forward[:, 1]).dtype == np.float64
    assert wrap_delta_degrees(np.float32(190.0), dtype=np.float32).dtype == np.float32
    assert image_angle_degrees is heading_degrees


def test_named_foreign_conventions():
    assert MEGABOUTS_HEAD_YAW != CHASER_CENTRED_BEARING
    # tail base straight below the head in the image (y down): the
    # tail-base -> head vector points up, atan2(-1, 0) without a y flip.
    assert megabouts_head_yaw_radians(0.0, 1.0) == -np.pi / 2
    # straight ahead of the chaser, then to its right (clockwise positive).
    assert chaser_centred_bearing_degrees(0.0, 5.0) == 0.0
    assert chaser_centred_bearing_degrees(5.0, 0.0) == 90.0
