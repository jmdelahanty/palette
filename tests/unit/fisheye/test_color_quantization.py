from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fisheye.shared.color_quantization import (
    QUANTIZATION_ID,
    lround,
    unit_rgb_to_hex,
    unit_rgba_to_rgba8,
    unit_to_u8,
)
from fisheye.shared.stimulus_vocabulary import QUANTIZATION_DEFAULT

# Byte copy of agent-contracts citrus-palette/stimulus_vocabulary/fixtures/
# resolved_color_examples.json at main 8927594.
FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "citrus_stimulus_vocabulary"
    / "resolved_color_examples.json"
)


def test_helper_implements_the_contract_default() -> None:
    assert QUANTIZATION_ID == QUANTIZATION_DEFAULT


@pytest.mark.parametrize(
    "example",
    [
        example
        for example in json.loads(FIXTURE.read_text())["examples"]
        if example.get("quantization", QUANTIZATION_ID) == QUANTIZATION_ID
        # Element derivations other than plain_v1 belong to trial-index v3
        # support, which Palette does not accept yet.
        and example.get("derivation", "plain_v1") == "plain_v1"
    ],
    ids=lambda example: example["name"],
)
def test_contract_fixture_rows_replay(example) -> None:
    rgba8 = unit_rgba_to_rgba8(example["authored"], brightness=example["brightness"])

    assert list(rgba8) == example["recorded"]["rgba8"]


def test_exact_tie_rounds_away_from_zero_not_to_even() -> None:
    # 0.5 is the only one/two-decimal channel that is an exact tie in binary32.
    assert 0.5 * 255.0 == 127.5
    assert round(127.5) == 128 and unit_to_u8(0.5) == 128
    assert round(76.5) == 76  # half-to-even, the rule Palette used to apply


@pytest.mark.parametrize(
    ("x", "expected"),
    ((0.3, 77), (0.7, 178), (0.1, 26), (0.9, 229), (0.75, 191)),
)
def test_decimal_is_quantised_as_binary32_not_double(x: float, expected: int) -> None:
    assert unit_to_u8(x) == expected


@pytest.mark.parametrize(("x", "double_code"), ((0.7, 179), (0.9, 230)))
def test_double_quantisation_would_disagree_with_citrus(x: float, double_code: int) -> None:
    assert x * 255.0 == math.floor(x * 255.0) + 0.5
    assert lround(x * 255.0) == double_code
    assert unit_to_u8(x) == double_code - 1


def _h5_column_example(prefix: str) -> dict:
    rows = json.loads(FIXTURE.read_text())["h5_per_frame_column_examples"]
    return next(row for row in rows if row["name"].startswith(prefix))


def test_stored_appearance_code_floats_recover_every_code() -> None:
    example = _h5_column_example("visual_appearance")
    assert [unit_to_u8(np.float32(k / 255)) for k in example["stored_codes"]] == example["recovered"]
    assert all(unit_to_u8(np.float32(k / 255)) == k for k in range(256))


def test_stored_grating_floats_recover_the_displayed_code() -> None:
    example = _h5_column_example("moving_grating")
    rgba8 = unit_rgba_to_rgba8(example["authored"], brightness=example["brightness"])
    assert list(rgba8) == example["recovered"]


def test_lround_is_exact_below_one_half() -> None:
    assert lround(0.49999999999999994) == 0
    assert lround(0.5) == 1
    assert lround(-76.5) == -77
    assert lround(-0.4) == 0


@pytest.mark.parametrize("x", (float("nan"), float("inf"), float("-inf")))
def test_non_finite_channels_quantise_to_zero(x: float) -> None:
    assert unit_to_u8(x) == 0


def test_out_of_range_channels_clamp() -> None:
    assert unit_to_u8(-0.2) == 0
    assert unit_to_u8(1.7) == 255


def test_hex_uses_the_contract_rule() -> None:
    assert unit_rgb_to_hex((0.3, 0.7, 0.0, 1.0)) == "#4db200"
