"""Citrus pose observations v2: the shared heading convention, NaN when not valid.

Fixtures are Citrus's frozen reference bytes (agent-contracts
citrus-pose-observation-v2, 6b63aa2), pinned by digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5 import pose
from fisheye.shared.unified_h5.integrity import validate_internal_integrity

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_pose_observation_v2"
DIGESTS = {
    "pose_complete.h5": "7ca61b9277633e677dfedbb12a05a77b8cc16d4e20ba3b7504ecbefe961cc550",
    "pose_failed.h5": "31199ab626fd05f68d1612190b4dcc2d93f352db11b1b55b9e836456a0084e40",
    "pose_absent.h5": "6df3fc7115592d5f795f4fa2833981d38861fd0db0a3543659a3bedf72f3c171",
}
OBJECTS = "/observations/pose/objects"


def test_vendored_fixtures_are_the_published_bytes():
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


def test_a_complete_v2_pose_is_admitted_with_its_rows():
    with h5py.File(FIXTURES / "pose_complete.h5", "r") as h5:
        assert validate_internal_integrity(h5).pose_rows == 4 + 3 + 15


def test_an_absent_v2_pose_is_admitted_without_one():
    with h5py.File(FIXTURES / "pose_absent.h5", "r") as h5:
        assert validate_internal_integrity(h5).pose_rows is None


def test_a_declared_v2_pose_that_failed_refuses_the_file():
    with h5py.File(FIXTURES / "pose_failed.h5", "r") as h5:
        with pytest.raises(UnifiedH5ContractError, match="unified_session_not_complete"):
            validate_internal_integrity(h5)


def _copy(tmp_path: Path) -> Path:
    return Path(shutil.copy(FIXTURES / "pose_complete.h5", tmp_path / "pose_complete.h5"))


def _check(path: Path):
    with h5py.File(path, "r") as h5:
        data = json.loads(h5["/metadata/component_outcomes_json"][()])
        outcomes = {row["component_id"]: row for row in data["components"]}
        return pose.validate_pose_observations(h5, outcomes, {pose.COMPONENT})


def _edit_object(path: Path, row: int, **values) -> None:
    with h5py.File(path, "r+") as h5:
        objects = h5[OBJECTS]
        record = objects[row]
        for name, value in values.items():
            record[name] = value
        objects[row] = record


def test_the_fixture_rows_follow_the_shared_convention(tmp_path):
    path = _copy(tmp_path)
    assert _check(path) == 22
    with h5py.File(path, "r") as h5:
        objects = h5[OBJECTS][()]
    assert objects["heading_status"].tolist() == [7, 0, 7]
    assert np.isnan(objects[1]["heading_angle_degrees"])


def test_image_left_is_admitted_with_either_sign(tmp_path):
    path = _copy(tmp_path)
    for angle in (-180.0, 180.0):
        _edit_object(path, 0, heading_direction_x=-1.0, heading_direction_y=0.0, heading_angle_degrees=angle)
        assert _check(path) == 22


@pytest.mark.parametrize(
    ("row", "values", "reason"),
    [
        # v1's clockwise angle for a fish facing image-up.
        (0, {"heading_direction_x": 0.0, "heading_direction_y": -1.0, "heading_angle_degrees": 270.0 - 360.0},
         "pose_heading_angle_disagrees_with_direction"),
        (0, {"heading_direction_x": 2.0}, "pose_heading_direction_not_unit"),
        (0, {"heading_status": 9}, "pose_heading_not_nan_when_not_valid"),
        (1, {"heading_angle_degrees": 0.0}, "pose_heading_not_nan_when_not_valid"),
        (0, {"heading_origin_x_px": np.nan}, "pose_heading_not_finite_when_valid"),
        (0, {"heading_status": 10}, "pose_heading_status_unknown"),
    ],
)
def test_heading_rows_that_break_the_convention_are_refused(tmp_path, row, values, reason):
    path = _copy(tmp_path)
    _edit_object(path, row, **values)
    with pytest.raises(UnifiedH5ContractError, match=reason):
        _check(path)


def test_an_unknown_contract_version_is_refused(tmp_path):
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        contract = json.loads(h5[pose.CONTRACT][()])
        contract["schema_version"] = 3
        del h5[pose.CONTRACT]
        h5.create_dataset(pose.CONTRACT, data=json.dumps(contract), dtype=h5py.string_dtype("utf-8"))
    with pytest.raises(UnifiedH5ContractError, match="pose_contract_version_unknown"):
        _check(path)


def test_v1_tables_under_a_v2_contract_are_refused(tmp_path):
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5[OBJECTS].attrs["schema_version"] = "1"
    with pytest.raises(UnifiedH5ContractError, match="pose_table_attributes_mismatch:objects"):
        _check(path)


# Bound-token rule (agent-contracts citrus-pose-observation-v2 amendment
# 2026-10-13, PR #99, merged 08689f0d): on a bound v2 session every update carries the bound token.
FIXTURE_ROOT = FIXTURES.parent
BOUND_V2 = FIXTURE_ROOT / "citrus_unified_h5_frame_identity_v1" / "frame_identity_v1_pose_complete.h5"
BOUND_V1 = FIXTURE_ROOT / "citrus_unified_h5_bounding_boxes_v2" / "bounding_boxes_v2_capacity_pose_on.h5"
UPDATES = "/observations/pose/updates"
FOREIGN = b"sha256:" + b"0" * 64


def _copy_of(source: Path, tmp_path: Path) -> Path:
    return Path(shutil.copy(source, tmp_path / source.name))


def _set_token(path: Path, row: int, token: bytes) -> None:
    with h5py.File(path, "r+") as h5:
        updates = h5[UPDATES]
        record = updates[row]
        record["recording_identity_token"] = token
        updates[row] = record


def _tokens(path: Path) -> set[bytes]:
    with h5py.File(path, "r") as h5:
        return set(h5[UPDATES].fields("recording_identity_token")[:])


def test_a_bound_v2_session_whose_rows_carry_the_bound_token_is_admitted(tmp_path):
    path = _copy_of(BOUND_V2, tmp_path)
    with h5py.File(path, "r") as h5:
        bound = json.loads(h5[pose.BINDING][()])["recording_identity_token"].encode()
    assert _tokens(path) == {bound}
    assert _check(path) is not None


def test_a_bound_v2_row_from_another_recording_is_refused(tmp_path):
    path = _copy_of(BOUND_V2, tmp_path)
    _set_token(path, 0, FOREIGN)
    with pytest.raises(
        UnifiedH5ContractError,
        match="^pose_update_foreign_recording_token:foreign_rows=1:first_row=0$",
    ):
        _check(path)


def test_an_advisory_v2_session_admits_foreign_tokens(tmp_path):
    path = _copy(tmp_path)
    _set_token(path, 1, FOREIGN)
    assert len(_tokens(path)) == 2
    assert _check(path) == 22


def test_a_bound_v2_session_without_a_bound_token_is_refused(tmp_path):
    path = _copy_of(BOUND_V2, tmp_path)
    with h5py.File(path, "r+") as h5:
        del h5[pose.BINDING]
    with pytest.raises(UnifiedH5ContractError, match="pose_bound_recording_token_missing"):
        _check(path)


@pytest.mark.parametrize(
    ("source", "association_source"),
    [
        (BOUND_V2, "mutable_latest_recording_pointer"),
        (FIXTURES / "pose_complete.h5", pose.BOUND_SOURCE),
    ],
)
def test_a_status_and_source_that_disagree_are_refused(tmp_path, source, association_source):
    path = _copy_of(source, tmp_path)
    with h5py.File(path, "r+") as h5:
        h5[pose.RECORDING].attrs["association_source"] = association_source
    with pytest.raises(UnifiedH5ContractError, match="pose_association_declaration_inconsistent"):
        _check(path)


def test_an_empty_bound_updates_table_is_left_to_the_receipt(tmp_path):
    with h5py.File(BOUND_V2, "r") as source:
        dtype = source[UPDATES].dtype
        contract = json.loads(source[pose.CONTRACT][()])
    with h5py.File(tmp_path / "empty.h5", "w") as h5:
        h5.create_group(pose.RECORDING).attrs["association_source"] = pose.BOUND_SOURCE
        updates = h5.create_dataset(UPDATES, shape=(0,), dtype=dtype)
        pose._check_v2_bound_token(h5, contract, updates)


def test_a_bound_v1_session_with_a_stale_row_is_not_refused(tmp_path):
    path = _copy_of(BOUND_V1, tmp_path)
    with h5py.File(path, "r") as h5:
        contract = json.loads(h5[pose.CONTRACT][()])
    assert (contract["schema_version"], contract["association_status"]) == (1, "bound")
    _set_token(path, 0, FOREIGN)
    assert _check(path) is not None

