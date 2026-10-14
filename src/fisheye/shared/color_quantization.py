"""The one float-to-8-bit colour rule shared with Citrus.

agent-contracts ``citrus-palette/stimulus_vocabulary`` names it
``rgba8_round_half_up_v1``: ``x`` is the channel as Citrus holds it, IEEE
binary32, promoted exactly to double; ``v = x * 255`` as a double, rounded half
away from zero (C ``lround``), clamped to 0..255; non-finite ``x`` becomes 0.
The binary32 step matters: the float nearest 0.7 gives 178.499997 -> 178,
while the double 0.7 gives exactly 178.5 -> 179. Python's ``round()`` rounds
half to even (0.3 -> 76, Citrus 77), so never use it for colour codes.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

QUANTIZATION_ID = "rgba8_round_half_up_v1"


def lround(value: float) -> int:
    """C ``lround``: nearest integer, exact ties away from zero."""

    magnitude = abs(value)
    whole = math.floor(magnitude)
    if magnitude - whole >= 0.5:
        whole += 1
    return int(whole) if value >= 0 else -int(whole)


def unit_to_u8(channel: float) -> int:
    """Quantise one authored channel in [0, 1] to its displayed 8-bit code."""

    number = float(np.float32(channel))
    if not math.isfinite(number):
        return 0
    return max(0, min(255, lround(number * 255.0)))


def unit_rgba_to_rgba8(
    rgba: Sequence[float], *, brightness: float = 1.0
) -> tuple[int, int, int, int]:
    """Authored RGBA floats to rgba8; brightness scales RGB before quantising.

    Citrus multiplies channel by brightness as binary32 (``ApplyBrightness``),
    so the product is formed in float32 too.
    """

    red, green, blue, alpha = (np.float32(value) for value in rgba)
    scale = np.float32(brightness)
    return (
        unit_to_u8(red * scale),
        unit_to_u8(green * scale),
        unit_to_u8(blue * scale),
        unit_to_u8(alpha),
    )


def unit_rgb_to_hex(rgb: Sequence[float]) -> str:
    """``#rrggbb`` of the first three authored channels."""

    return "#{:02x}{:02x}{:02x}".format(*(unit_to_u8(value) for value in rgb[:3]))
