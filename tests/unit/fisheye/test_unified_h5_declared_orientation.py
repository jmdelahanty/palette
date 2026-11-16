"""Citrus declared-orientation check v1 (agent-contracts #135, 3480e3e) in unified H5 admission."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files
from pathlib import Path

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import declared_orientation as mod
from fisheye.shared.unified_h5.common import UnifiedH5ContractError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_declared_orientation_v1"
INT32 = mod.INT32_ATTRIBUTES | {mod.OPTIONAL_INT32}
# Real Citrus output header (2026_11_06 omnifin0 H5): unrotated, unreflected.
OUTPUT_BLOCK = (
    "DP-3 connected 1920x1080+3840+0 (normal left inverted right x axis y axis) 0mm x 0mm\n"
    "   1920x1080     60.00*+\n"
)
# camera -> canvas keeps x and reflects y (det < 0), as measured on omnifin0.
HOMOGRAPHY = np.array([[0.08, 0.0, 100.0], [0.0, -0.08, 1000.0], [0.0, 0.0, 1.0]])


def _fixture(kind: str, name: str) -> dict:
    return json.loads((FIXTURES / kind / name).read_bytes())


def _omnifin0() -> dict:
    record = _fixture("valid", "omnifin0_rev16_cam2010093.json")
    record["homography_determinant"] = float(np.linalg.det(HOMOGRAPHY))
    return record


def _write_attrs(group: h5py.Group, record: dict) -> None:
    for name, value in record.items():
        if name in INT32:
            group.attrs[name] = np.int32(value)
        elif name in mod.FLOAT64_ATTRIBUTES:
            group.attrs[name] = np.float64(value)
        else:
            group.attrs[name] = value


def _write_h5(
    path: Path,
    *,
    record: dict | None,
    session_start: str = "2026-11-16T10-00-00Z",
    hostname: str = "pancake0.hhmi.org",
    homography: np.ndarray = HOMOGRAPHY,
    output_block: str = OUTPUT_BLOCK,
    pin: dict | None = None,
) -> Path:
    with h5py.File(path, "w") as h5:
        session = h5.create_group("metadata/session")
        session.attrs["session_start_iso8601_utc"] = session_start
        session.attrs["hostname"] = hostname
        session.attrs["rig_id"] = "omnifin0"
        session.attrs["arena_id"] = "arena_1"
        h5.create_dataset("metadata/display/selected_output_block", data=output_block, dtype=h5py.string_dtype())
        rig = h5.create_group("metadata/rig/pancakebatter_rig_file")
        for name, value in (
            pin
            or {
                "rig_id": "omnifin0",
                "rig_file_sha256": "sha256:1934857fe156e7d3aa3873602760b67ce3ae1e74be181a8ddbd0e9df3f0c0332",
                "rig_geometry_revision": 16,
            }
        ).items():
            rig.attrs[name] = value
        inputs = h5.create_group("geometry/correspondence/input")
        inputs.attrs["camera_serial"] = "2010093"
        inputs.attrs["native_source_width_px"] = 4512
        inputs.attrs["native_source_height_px"] = 4512
        inputs.attrs["effective_arena_origin_x_px"] = 0.0
        inputs.attrs["effective_arena_origin_y_px"] = 0.0
        renderer = inputs.create_group("renderer")
        renderer.attrs["texture_width_px"] = 1080.0
        renderer.attrs["texture_height_px"] = 1080.0
        h5.create_dataset("geometry/calibration/2010093/homography_matrix", data=homography.astype("<f8"))
        h5.create_dataset(
            "geometry/calibration/arena_config_json",
            data=json.dumps({"config_name": "arena_1"}),
            dtype=h5py.string_dtype(),
        )
        if record is not None:
            _write_attrs(h5.create_group("geometry/calibration/declared_orientation"), record)
    return path


def _validate(path: Path):
    with h5py.File(path, "r") as h5:
        return mod.validate_declared_orientation(h5)


def test_vendored_contract_is_the_pinned_bytes() -> None:
    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(mod._SCHEMA_FILE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == mod._SCHEMA_SHA256


def test_omnifin0_cutoff_is_the_install_notice() -> None:
    assert mod.cutoffs()["omnifin0"]["cutoff_utc"] == "2026-11-15T23:17:22Z"
    assert mod.cutoffs()["omnifin0"]["citrus_commit"] == "027d292a41db771a6ba14fda7db839c7c1fffec9"


@pytest.mark.parametrize("path", sorted((FIXTURES / "valid").glob("*.json")), ids=lambda p: p.name)
def test_valid_contract_fixtures_pass_schema_and_rules(path: Path) -> None:
    mod.validate_record(json.loads(path.read_bytes()))


@pytest.mark.parametrize("path", sorted((FIXTURES / "invalid").glob("*.json")), ids=lambda p: p.name)
def test_invalid_contract_fixtures_are_refused(path: Path) -> None:
    with pytest.raises(UnifiedH5ContractError):
        mod.validate_record(json.loads(path.read_bytes()))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_determinant_fails_r2(value: float) -> None:
    record = _fixture("valid", "ventral_mirror_reverse_x_reflected.json")
    record["homography_determinant"] = value
    assert mod.rule_errors(record) == ["R2"]


def test_sealed_omnifin0_record_is_admitted(tmp_path: Path) -> None:
    record = _validate(_write_h5(tmp_path / "a.h5", record=_omnifin0()))
    assert record["status"] == "agrees"
    assert record["measured_determinant_sign"] == -1


@pytest.mark.parametrize(
    ("start", "required"),
    [
        ("2026-11-15T23-17-21Z", False),
        ("2026-11-15T23-17-22Z", True),  # at the cutoff second: required
        ("2026-11-16T10-00-00Z", True),
    ],
)
def test_group_is_required_from_the_cutoff(tmp_path: Path, start: str, required: bool) -> None:
    path = _write_h5(tmp_path / "a.h5", record=None, session_start=start)
    if required:
        with pytest.raises(UnifiedH5ContractError, match="declared_orientation_missing"):
            _validate(path)
    else:
        assert _validate(path) is None


def test_group_before_the_cutoff_is_still_checked(tmp_path: Path) -> None:
    # 119a8e8 (2026_11_15_17_39_20) wrote the group before omnifin0's cutoff.
    record = _omnifin0()
    record["expected_determinant_sign"] = 1
    path = _write_h5(tmp_path / "a.h5", record=record, session_start="2026-11-15T17-39-20Z")
    with pytest.raises(UnifiedH5ContractError, match="declared_orientation_rule"):
        _validate(path)


@pytest.mark.parametrize(
    "start", ["2026-11-16T10:00:00Z", "2026-11-16T10-00-00.5Z", "2026-11-16 10-00-00Z"]
)
def test_session_start_is_parsed_exactly(tmp_path: Path, start: str) -> None:
    path = _write_h5(tmp_path / "a.h5", record=None, session_start=start)
    with pytest.raises(UnifiedH5ContractError, match="session_start_unparseable"):
        _validate(path)


def test_cutoff_is_bound_to_its_host(tmp_path: Path) -> None:
    # Without the group, the requirement is decided on the cutoff host's clock;
    # a recording from another host cannot be placed against it.
    path = _write_h5(tmp_path / "a.h5", record=None, hostname="pancake1.hhmi.org")
    with pytest.raises(UnifiedH5ContractError, match="cutoff_host_mismatch"):
        _validate(path)


def test_unlisted_rig_does_not_require_the_group(tmp_path: Path) -> None:
    path = _write_h5(tmp_path / "a.h5", record=None)
    with h5py.File(path, "a") as h5:
        h5["metadata/session"].attrs["rig_id"] = "otherrig"
    assert _validate(path) is None


def _mutated(**changes) -> dict:
    record = _omnifin0()
    record.update(changes)
    return record


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        # Consumer check 1: the recording's own homography decides the sign.
        ({"homography": np.array([[0.08, 0, 100.0], [0, 0.08, 10.0], [0, 0, 1.0]])}, "determinant"),
        # w < 0 over the frame: -H is the same mapping with the opposite det sign.
        ({"homography": -HOMOGRAPHY}, "w_not_positive"),
        # w crosses zero inside the frame (horizon through the image).
        ({"homography": np.array([[0.08, 0, 100.0], [0, -0.08, 1000.0], [-0.001, 0, 1.0]])}, "w_not_positive"),
        # Consumer check 2: Palette parses the sealed output block itself.
        ({"output_block": OUTPUT_BLOCK.replace(" 0mm x 0mm", " X axis (normal left inverted right x axis y axis) 0mm x 0mm").replace("+0 (normal", "+0 normal")}, "output_block"),
        # Consumer check 3: the rig pin.
        ({"pin": {"rig_id": "omnifin0", "rig_file_sha256": "sha256:" + "0" * 64, "rig_geometry_revision": 16}}, "rig_pin_mismatch:rig_file_sha256"),
        ({"pin": {"rig_id": "omnifin0", "rig_file_sha256": "sha256:1934857fe156e7d3aa3873602760b67ce3ae1e74be181a8ddbd0e9df3f0c0332", "rig_geometry_revision": 15}}, "rig_geometry_revision"),
    ],
)
def test_consumer_checks_refuse(tmp_path: Path, kwargs: dict, message: str) -> None:
    path = _write_h5(tmp_path / "a.h5", record=_omnifin0(), **kwargs)
    with pytest.raises(UnifiedH5ContractError, match=message):
        _validate(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"camera_serial": "2010094"}, "camera_mismatch"),
        ({"arena_id": "arena_2"}, "arena_mismatch"),
        ({"camera_native_width_px": 4096}, "camera_frame_mismatch"),
        ({"homography_determinant": -0.5}, "determinant_value_mismatch"),
        ({"status": "refused"}, "declared_orientation_schema"),
        ({"xrandr_reflection": "x", "expected_determinant_sign": 1, "measured_determinant_sign": 1, "homography_determinant": 0.0064}, "determinant"),
    ],
)
def test_record_must_describe_this_recording(tmp_path: Path, changes: dict, message: str) -> None:
    path = _write_h5(tmp_path / "a.h5", record=_mutated(**changes))
    with pytest.raises(UnifiedH5ContractError, match=message):
        _validate(path)


def test_attribute_types_are_enforced(tmp_path: Path) -> None:
    path = _write_h5(tmp_path / "a.h5", record=_omnifin0())
    with h5py.File(path, "a") as h5:
        group = h5["geometry/calibration/declared_orientation"]
        del group.attrs["canvas_space_parity"]
        group.attrs["canvas_space_parity"] = np.int64(1)
    with pytest.raises(UnifiedH5ContractError, match="attribute_type:canvas_space_parity"):
        _validate(path)


def test_group_is_attributes_only(tmp_path: Path) -> None:
    path = _write_h5(tmp_path / "a.h5", record=_omnifin0())
    with h5py.File(path, "a") as h5:
        h5["geometry/calibration/declared_orientation"].create_dataset("extra", data=1)
    with pytest.raises(UnifiedH5ContractError, match="not_attributes_only"):
        _validate(path)
