"""Citrus's declared-orientation check (agent-contracts ``citrus-palette/coordinate_spaces_v1``).

Citrus checks every arena's active ``camera_native_px -> final_display_canvas_px``
homography against the handedness the pancakebatter rig file declares, at
promotion and at START, and seals what it checked in the attributes-only group
``/geometry/calibration/declared_orientation`` (palette#377 Q-C3/Q-C5, Jeremy
2026-11-15). Like ``z_eff`` the group is uncataloged under admission revision
7, so this module checks it in full and never trusts ``status`` alone:

- a closed set of scalar attributes with the README's HDF5 types, validated
  against the pinned schema;
- rules R1-R4, the parity arithmetic (``test_citrus_declared_orientation_check_v1.py``);
- consumer checks 1-4: the determinant's sign recomputed from the recording's
  own active homography with ``w(p) > 0`` at the four corners of the camera
  frame and at the arena centre mapped back; the xrandr reflection parsed
  from the sealed output block; the rig pin; the arena and camera.

The group is required for a recording whose session starts at or after its
rig's cutoff (``palette_citrus_declared_orientation_cutoffs_v1.json``, the
install time of the first Citrus build that writes it); before the cutoff it
is optional but checked when present.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from importlib.resources import files as resource_files
from typing import Any, Mapping

import h5py
import numpy as np

from fisheye.shared.selected_calibration import (
    SelectedCalibrationError,
    _parse_selected_output_block,
)

from .common import require
from .rig_provenance import GROUP as RIG_PIN
from .schema import read_json

GROUP = "/geometry/calibration/declared_orientation"
INPUT = "/geometry/correspondence/input"
ARENA_CONFIG = "/geometry/calibration/arena_config_json"
SESSION = "/metadata/session"
OUTPUT_BLOCK = "/metadata/display/selected_output_block"

_SCHEMA_FILE = "citrus_declared_orientation_check_v1.schema.json"
_SCHEMA_SHA256 = "1bf066219da31fc628f10912d67522c97261023c070a3a78b0b4e88bfba0fa82"
_CUTOFFS_FILE = "palette_citrus_declared_orientation_cutoffs_v1.json"
_CUTOFFS_SCHEMA_ID = "palette.citrus_declared_orientation_cutoffs"

INT32_ATTRIBUTES = frozenset(
    {
        "schema_version",
        "canvas_space_parity",
        "recorded_space_parity",
        "presentation_parity",
        "expected_determinant_sign",
        "measured_determinant_sign",
        "camera_native_width_px",
        "camera_native_height_px",
    }
)
OPTIONAL_INT32 = "rig_geometry_revision"
FLOAT64_ATTRIBUTES = frozenset({"homography_determinant"})
UNKNOWN = "unknown"


def _contract_bytes(name: str) -> bytes:
    return resource_files("fisheye.shared").joinpath("contracts").joinpath(name).read_bytes()


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator

    data = _contract_bytes(_SCHEMA_FILE)
    require(hashlib.sha256(data).hexdigest() == _SCHEMA_SHA256, "packaged_contract_drift:" + _SCHEMA_FILE)
    return Draft202012Validator(json.loads(data))


@lru_cache(maxsize=1)
def cutoffs() -> Mapping[str, Mapping[str, str]]:
    document = json.loads(_contract_bytes(_CUTOFFS_FILE))
    require(
        document.get("schema_id") == _CUTOFFS_SCHEMA_ID and document.get("schema_version") == 1,
        "declared_orientation_cutoffs_schema",
    )
    return document["cutoffs"]


def _session_start(text: str) -> datetime:
    """Citrus's session_start_iso8601_utc: ``%Y-%m-%dT%H-%M-%SZ``, hyphens in the time."""

    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H-%M-%SZ")
    except (TypeError, ValueError):
        parsed = None
    require(parsed is not None, "declared_orientation_session_start_unparseable")
    return parsed.replace(tzinfo=timezone.utc)


def _rfc3339(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _decoded(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _attribute(attrs, name: str) -> Any:
    raw = attrs.get_id(name)
    require(
        raw.get_space().get_simple_extent_type() == h5py.h5s.SCALAR,
        f"declared_orientation_attribute_not_scalar:{name}",
    )
    kind = raw.get_type()
    if name in INT32_ATTRIBUTES or name == OPTIONAL_INT32:
        require(
            kind.get_class() == h5py.h5t.INTEGER and kind.get_size() == 4 and kind.get_sign() == h5py.h5t.SGN_2,
            f"declared_orientation_attribute_type:{name}",
        )
        return int(attrs[name])
    if name in FLOAT64_ATTRIBUTES:
        require(
            kind.get_class() == h5py.h5t.FLOAT and kind.get_size() == 8,
            f"declared_orientation_attribute_type:{name}",
        )
        return float(np.float64(attrs[name]))
    require(
        kind.get_class() == h5py.h5t.STRING and kind.is_variable_str() and kind.get_cset() == h5py.h5t.CSET_UTF8,
        f"declared_orientation_attribute_type:{name}",
    )
    value = _decoded(attrs[name])
    require(type(value) is str, f"declared_orientation_attribute_type:{name}")
    return value


def _product(factors):
    if any(f == UNKNOWN for f in factors):
        return UNKNOWN
    result = 1
    for factor in factors:
        result *= factor
    return result


def _mirror_parity(count: str):
    return UNKNOWN if count == UNKNOWN else (-1 if int(count) % 2 else 1)


def _flag_parity(flag: str):
    return UNKNOWN if flag == UNKNOWN else (-1 if flag == "true" else 1)


def rule_errors(record: Mapping[str, Any]) -> list[str]:
    """Rules R1-R4: the reference in the contract's test file, plus a finite determinant."""

    errors = []
    h = 1 if record["expected_camera_to_canvas_handedness"] == "preserved" else -1
    xrandr = {"none": 1, "xy": 1, "x": -1, "y": -1}[record["xrandr_reflection"]]
    if record["expected_determinant_sign"] != h * record["presentation_parity"] * xrandr:
        errors.append("R1")
    det = record["homography_determinant"]
    if (
        not math.isfinite(det)
        or det == 0
        or record["measured_determinant_sign"] != (-1 if det < 0 else 1)
        or record["measured_determinant_sign"] != record["expected_determinant_sign"]
    ):
        errors.append("R2")
    if h != record["recorded_space_parity"] * record["canvas_space_parity"]:
        errors.append("R3")
    canvas = _product(
        [
            {"front": 1, "rear": -1}.get(record["projection_mode"], UNKNOWN),
            {"none": 1, "horizontal": -1, "vertical": -1, "both": 1}.get(
                record["projector_firmware_flip"], UNKNOWN
            ),
            _mirror_parity(record["projection_optical_mirror_count"]),
        ]
    )
    recorded = _product(
        [
            {"dorsal": 1, "ventral": -1}.get(record["viewing_side"], UNKNOWN),
            _mirror_parity(record["camera_optical_mirror_count"]),
            _flag_parity(record["camera_reverse_x"]),
            _flag_parity(record["camera_reverse_y"]),
        ]
    )
    if canvas != record["canvas_space_parity"] or recorded != record["recorded_space_parity"]:
        errors.append("R4")
    return errors


def validate_record(record: Mapping[str, Any]) -> None:
    """The schema and R1-R4 for one record read from the group's attributes."""

    from jsonschema.exceptions import best_match

    error = best_match(_validator().iter_errors(dict(record)))
    require(error is None, "declared_orientation_schema:" + ("/".join(map(str, error.absolute_path)) if error else ""))
    errors = rule_errors(record)
    require(not errors, "declared_orientation_rule:" + ",".join(errors))


def _active_determinant(h5, *, camera: str, width: int, height: int) -> float:
    """Consumer check 1: sign(det H) where w(p) > 0 over the camera frame.

    H is the recording's active camera_native_px -> final_display_canvas_px
    homography. w is affine in p, so positive at the four continuous-edge
    corners means positive over the frame; the arena centre (canvas) mapped
    back through H^-1 is the reference point (palette#377 section 3).
    """

    matrix = np.asarray(h5[f"/geometry/calibration/{camera}/homography_matrix"][()], dtype=np.float64)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), "declared_orientation_homography")
    w_x, w_y, w_0 = matrix[2]

    def w(x: float, y: float) -> float:
        return w_x * x + w_y * y + w_0

    for x, y in ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height)):
        require(w(x, y) > 0, "declared_orientation_w_not_positive")
    attrs, renderer = h5[INPUT].attrs, h5[INPUT + "/renderer"].attrs
    centre = np.array(
        [
            float(attrs["effective_arena_origin_x_px"]) + float(renderer["texture_width_px"]) / 2.0,
            float(attrs["effective_arena_origin_y_px"]) + float(renderer["texture_height_px"]) / 2.0,
            1.0,
        ]
    )
    back = np.linalg.inv(matrix) @ centre
    require(math.isfinite(back[2]) and back[2] != 0, "declared_orientation_w_not_positive")
    require(w(back[0] / back[2], back[1] / back[2]) > 0, "declared_orientation_w_not_positive")
    det = float(np.linalg.det(matrix))
    require(math.isfinite(det) and det != 0, "declared_orientation_homography")
    return det


def _xrandr_reflection(h5) -> str:
    """Consumer check 2: Palette's own parse of the sealed output block.

    Palette admits only an unrotated, unreflected output (palette#377 §6a), so
    the only reflection it can parse is ``none``.
    """

    require(OUTPUT_BLOCK in h5, "declared_orientation_output_block_missing")
    try:
        _parse_selected_output_block(_decoded(h5[OUTPUT_BLOCK][()]))
    except SelectedCalibrationError:
        require(False, "declared_orientation_output_block_unadmitted")
    return "none"


def required_by_cutoff(h5) -> bool:
    """Whether this recording's session starts at or after its rig's cutoff."""

    session = h5[SESSION].attrs
    rig_id = _decoded(session.get("rig_id"))
    entry = cutoffs().get(rig_id) if isinstance(rig_id, str) else None
    if entry is None:
        return False
    start = _session_start(_decoded(session.get("session_start_iso8601_utc")))
    if start < _rfc3339(entry["cutoff_utc"]):
        return False
    # The cutoff is a time on this host's clock; another host needs its own.
    require(_decoded(session.get("hostname")) == entry["hostname"], "declared_orientation_cutoff_host_mismatch")
    return True


def validate_declared_orientation(h5) -> Mapping[str, Any] | None:
    """Check ``/geometry/calibration/declared_orientation``; return its record."""

    group = h5.get(GROUP)
    if group is None:
        require(not required_by_cutoff(h5), "declared_orientation_missing")
        return None
    require(isinstance(group, h5py.Group) and len(group) == 0, "declared_orientation_not_attributes_only")
    record = {name: _attribute(group.attrs, name) for name in group.attrs}
    validate_record(record)

    camera = _decoded(h5[INPUT].attrs["camera_serial"])
    require(record["camera_serial"] == camera, "declared_orientation_camera_mismatch")
    config = read_json(h5, ARENA_CONFIG)
    require(record["arena_id"] == config.get("config_name"), "declared_orientation_arena_mismatch")
    require(
        record["arena_id"] == _decoded(h5[SESSION].attrs.get("arena_id")),
        "declared_orientation_session_arena_mismatch",
    )
    attrs = h5[INPUT].attrs
    require(
        (record["camera_native_width_px"], record["camera_native_height_px"])
        == (int(attrs["native_source_width_px"]), int(attrs["native_source_height_px"])),
        "declared_orientation_camera_frame_mismatch",
    )
    det = _active_determinant(
        h5,
        camera=camera,
        width=record["camera_native_width_px"],
        height=record["camera_native_height_px"],
    )
    require(
        (-1 if det < 0 else 1) == record["measured_determinant_sign"],
        "declared_orientation_determinant_mismatch",
    )
    # The record names this homography: its determinant to rounding.
    require(
        math.isclose(record["homography_determinant"], det, rel_tol=1e-9, abs_tol=0.0),
        "declared_orientation_determinant_value_mismatch",
    )
    require(_xrandr_reflection(h5) == record["xrandr_reflection"], "declared_orientation_xrandr_mismatch")

    # Consumer check 3: the same rig pin as z_eff. A mismatch means the rig
    # file changed between START and logger open.
    pin = h5.get(RIG_PIN)
    require(isinstance(pin, h5py.Group), "declared_orientation_rig_pin_missing")
    for name in ("rig_id", "rig_file_sha256"):
        require(_decoded(pin.attrs.get(name)) == record[name], f"declared_orientation_rig_pin_mismatch:{name}")
    if OPTIONAL_INT32 in record:
        require(
            OPTIONAL_INT32 in pin.attrs and int(pin.attrs[OPTIONAL_INT32]) == record[OPTIONAL_INT32],
            "declared_orientation_rig_pin_mismatch:rig_geometry_revision",
        )
    return record


__all__ = ["GROUP", "cutoffs", "required_by_cutoff", "rule_errors", "validate_declared_orientation", "validate_record"]
