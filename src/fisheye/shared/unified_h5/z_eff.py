"""Per-arena z_eff under semantic recipe v5 (agent-contracts pairing revision 4).

Citrus #114 (``be2cd47``) computes one float32 z_eff per arena at START from
the rig file's measured media stack and records it, with every input and its
source, in the attrs-only group ``/geometry/calibration/z_eff``. The group is
uncataloged (like ``tank`` and ``arena_geometry``) and its attributes are not
covered by the sealed dependency manifest, so this module checks it in full:
a closed set of 25 scalar attributes (``rig_geometry_revision`` optional),
their types, the fixed source strings, the two derived differences bit for
bit, and the value against the digest-bound ``arena_config_json``, the chaser
rows and the rig pin. Required on every v5 recording, refused on any other.
"""

from __future__ import annotations

import math
import re

import h5py
import numpy as np

from fisheye.shared.protocol_semantic_contract import (
    PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY,
)

from .common import require
from .rig_provenance import GROUP as RIG_PIN
from .schema import read_json

GROUP = "/geometry/calibration/z_eff"
ARENA_CONFIG = "/geometry/calibration/arena_config_json"
CHASER_STATES = "/components/chaser/states"
FORMULA_ID = "citrus.z_eff.hung_dish_paraxial.v1"

FLOAT_ATTRIBUTES = frozenset(
    {
        "z_eff_mm",
        "geometric_eye_to_image_mm",
        "water_under_dish_mm",
        "holder_top_above_image_plane_mm",
        "seat_to_floor_top_mm",
        "dish_floor_above_image_plane_mm",
        "dish_base_thickness_mm",
        "refractive_index_acrylic",
        "refractive_index_water",
        "eye_height_above_floor_mm",
    }
)
FIXED_STRINGS = {
    "formula_id": FORMULA_ID,
    "holder_top_above_image_plane_mm_source": "rig_file:media_stack.support.holder_top_above_film_mm",
    "seat_to_floor_top_mm_source": "tank_design_file:dimensions.seat_to_floor_top_mm",
    "dish_floor_above_image_plane_mm_source": "derived:holder_top_above_image_plane_mm-seat_to_floor_top_mm",
    "refractive_index_acrylic_source": "tank_design_file:materials.tank_refractive_index",
    "refractive_index_water_source": "tank_design_file:materials.water_refractive_index",
    "eye_height_above_floor_mm_source": "tank_design_biological_assumption",
}
STRING_ATTRIBUTES = frozenset(FIXED_STRINGS) | {
    "rig_id",
    "rig_file_path",
    "rig_file_sha256",
    "tank_design_id",
    "tank_design_file_path",
    "tank_design_file_sha256",
    "arena_id",
}
OPTIONAL_INT32 = "rig_geometry_revision"
REQUIRED = FLOAT_ATTRIBUTES | STRING_ATTRIBUTES
FILE_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _scalar(attrs, name: str):
    raw = attrs.get_id(name)
    require(
        raw.get_space().get_simple_extent_type() == h5py.h5s.SCALAR,
        f"z_eff_attribute_not_scalar:{name}",
    )
    return raw.get_type()


def _float(attrs, name: str) -> float:
    kind = _scalar(attrs, name)
    require(
        kind.get_class() == h5py.h5t.FLOAT and kind.get_size() == 8,
        f"z_eff_attribute_type:{name}",
    )
    value = float(np.float64(attrs[name]))
    require(math.isfinite(value), f"z_eff_attribute_value:{name}")
    return value


def _string(attrs, name: str) -> str:
    kind = _scalar(attrs, name)
    require(
        kind.get_class() == h5py.h5t.STRING
        and kind.is_variable_str()
        and kind.get_cset() == h5py.h5t.CSET_UTF8,
        f"z_eff_attribute_type:{name}",
    )
    value = attrs[name]
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    require(type(value) is str and value != "", f"z_eff_attribute_value:{name}")
    return value


def _int32(attrs, name: str) -> int:
    kind = _scalar(attrs, name)
    require(
        kind.get_class() == h5py.h5t.INTEGER
        and kind.get_size() == 4
        and kind.get_sign() == h5py.h5t.SGN_2,
        f"z_eff_attribute_type:{name}",
    )
    return int(attrs[name])


def _rig_pin_text(attrs, name: str) -> str | None:
    if name not in attrs:
        return None
    value = attrs[name]
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return value if type(value) is str else None


def validate_z_eff(h5, semantic_normalization_policy: object) -> float | None:
    """Check ``/geometry/calibration/z_eff``; return z_eff_mm under recipe v5.

    Every other policy must not carry the group: one hash never names both an
    authored and a measured z_eff.
    """

    group = h5.get(GROUP)
    if semantic_normalization_policy != PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY:
        require(group is None, "z_eff_group_outside_recipe_v5")
        return None
    require(isinstance(group, h5py.Group), "z_eff_group_missing")
    require(len(group) == 0, "z_eff_group_not_attributes_only")
    attrs = group.attrs
    names = set(attrs)
    require(
        REQUIRED <= names and names <= REQUIRED | {OPTIONAL_INT32},
        "z_eff_attribute_set_not_closed",
    )
    floats = {name: _float(attrs, name) for name in FLOAT_ATTRIBUTES}
    strings = {name: _string(attrs, name) for name in STRING_ATTRIBUTES}
    revision = _int32(attrs, OPTIONAL_INT32) if OPTIONAL_INT32 in names else None
    for name, expected in FIXED_STRINGS.items():
        require(strings[name] == expected, f"z_eff_fixed_string:{name}")
    for name in ("rig_file_sha256", "tank_design_file_sha256"):
        require(FILE_SHA256.match(strings[name]) is not None, f"z_eff_sha256_form:{name}")

    z = floats["z_eff_mm"]
    require(z > 0 and float(np.float32(z)) == z, "z_eff_value_not_positive_float32")
    floor = floats["dish_floor_above_image_plane_mm"]
    require(
        floor == floats["holder_top_above_image_plane_mm"] - floats["seat_to_floor_top_mm"],
        "z_eff_dish_floor_not_derived",
    )
    require(
        floats["water_under_dish_mm"] == floor - floats["dish_base_thickness_mm"],
        "z_eff_water_under_dish_not_derived",
    )

    # The renderer's value: digest-bound through the geometry receipt.
    config = read_json(h5, ARENA_CONFIG)
    optics = config.get("dish_config", {})
    optics = optics.get("optics") if isinstance(optics, dict) else None
    require(isinstance(optics, dict), "z_eff_arena_config_optics_missing")
    for label, value in (
        ("calculated_z_eff_mm", config.get("calculated_z_eff_mm")),
        ("dish_config.optics.z_eff_current_mm", optics.get("z_eff_current_mm")),
        ("dish_config.optics.z_eff_larvae_mm", optics.get("z_eff_larvae_mm")),
    ):
        require(
            type(value) in (int, float) and float(value) == z,
            f"z_eff_arena_config_mismatch:{label}",
        )
    require(strings["arena_id"] == config.get("config_name"), "z_eff_arena_id_mismatch")
    session_arena = h5["/metadata/session"].attrs.get("arena_id")
    if session_arena is not None:
        session_arena = (
            session_arena.decode("utf-8") if isinstance(session_arena, bytes) else session_arena
        )
        require(session_arena == strings["arena_id"], "z_eff_session_arena_id_mismatch")

    if CHASER_STATES in h5:
        states = h5[CHASER_STATES]
        if "z_eff_mm" in (states.dtype.names or ()) and states.shape[0]:
            rows = states.fields("z_eff_mm")[()].astype(np.float64)
            require(bool(np.all(rows == z)), "z_eff_chaser_rows_mismatch")

    # A mismatch means the rig file changed between START and logger open.
    pin = h5.get(RIG_PIN)
    require(isinstance(pin, h5py.Group), "z_eff_rig_pin_missing")
    for name in ("rig_id", "rig_file_sha256"):
        require(_rig_pin_text(pin.attrs, name) == strings[name], f"z_eff_rig_pin_mismatch:{name}")
    if revision is not None:
        require(
            "rig_geometry_revision" in pin.attrs
            and int(pin.attrs["rig_geometry_revision"]) == revision,
            "z_eff_rig_pin_mismatch:rig_geometry_revision",
        )
    return z


__all__ = ["GROUP", "validate_z_eff"]
