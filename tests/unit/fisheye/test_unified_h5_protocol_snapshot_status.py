"""A protocol its producer refused is named as such, before the angle check."""

from __future__ import annotations

import io
import json

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5.common import UnifiedH5ContractError
from fisheye.shared.unified_h5.schema import (
    PROTOCOL_AUTHORED,
    PROTOCOL_SEMANTIC_STATUS,
    require_stimulus_angle_convention,
    require_valid_protocol_snapshot,
)

ANGLE = "atan2_negative_y_x_degrees"
# What Citrus ed33e0e writes for a sham chaser in a step with proximity feedback
# (sample plan_dish_group_with_members_sham_proximity_feedback, 8d10f272…).
SHAM_STATUS = {
    "reasons": [{
        "code": "chaser_role_sham_with_proximity_feedback",
        "message": "chasers[1] is a sham in a step with proximity feedback, ...",
        "path": "steps[0].parameters.chasers[1].role",
    }],
    "schema_id": "citrus.protocol.semantic.status",
    "schema_version": 1,
    "status": "unsupported",
}


def _text(group, name, value):
    group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))


def _h5(contract_status="valid", status=None, semantic=True):
    h5 = h5py.File(io.BytesIO(), "w")
    session = h5.create_group("/metadata/session")
    session.attrs["stimulus_angle_convention"] = ANGLE
    authored = h5.create_group(PROTOCOL_AUTHORED)
    authored.attrs["schema_id"] = "citrus.protocol.snapshot"
    authored.attrs["schema_version"] = np.int32(2)
    authored.attrs["contract_status"] = contract_status
    for name in ("protocol_definition_json", "protocol_trial_index_json") + (
        ("protocol_semantic_json",) if semantic else ()
    ):
        _text(authored, name, json.dumps({"stimulus_angle_convention": ANGLE}))
    if status is not None:
        _text(authored, "protocol_semantic_status_json", status)
    return h5


def test_a_valid_snapshot_passes():
    require_valid_protocol_snapshot(_h5())


def test_a_file_without_a_protocol_group_is_left_to_validate_protocol():
    require_valid_protocol_snapshot(h5py.File(io.BytesIO(), "w"))


def test_a_producer_refused_protocol_names_its_reasons():
    h5 = _h5("unsupported", json.dumps(SHAM_STATUS), semantic=False)
    with pytest.raises(UnifiedH5ContractError) as refused:
        require_valid_protocol_snapshot(h5)
    assert str(refused.value) == (
        "native_protocol_snapshot_status:contract_status='unsupported'; "
        "producer reasons: chaser_role_sham_with_proximity_feedback"
    )


def test_the_same_file_used_to_be_refused_as_an_angle_mismatch():
    """Why the snapshot check runs first: the angle check sees no semantic identity."""

    h5 = _h5("unsupported", json.dumps(SHAM_STATUS), semantic=False)
    with pytest.raises(UnifiedH5ContractError, match="^stimulus_angle_convention_mismatch$"):
        require_stimulus_angle_convention(h5, {"stimulus_angle_convention": ANGLE})


@pytest.mark.parametrize(
    "status",
    [
        None,  # no status document
        "{not json",  # unreadable
        json.dumps({"reasons": "chaser_role_sham_with_proximity_feedback"}),  # not a list
        json.dumps({"reasons": [{"code": "Bad Code; injected"}, {"code": 7}, "x"]}),  # unsafe codes
    ],
)
def test_a_refused_protocol_without_readable_reasons_still_refuses(status):
    h5 = _h5("unsupported", status, semantic=False)
    with pytest.raises(UnifiedH5ContractError) as refused:
        require_valid_protocol_snapshot(h5)
    assert str(refused.value) == "native_protocol_snapshot_status:contract_status='unsupported'"


def test_at_most_eight_reason_codes_are_named():
    status = {"reasons": [{"code": f"reason_{i}"} for i in range(12)]}
    h5 = _h5("invalid", json.dumps(status))
    with pytest.raises(UnifiedH5ContractError) as refused:
        require_valid_protocol_snapshot(h5)
    assert str(refused.value).endswith("reason_0, " + ", ".join(f"reason_{i}" for i in range(1, 8)))


def test_a_wrong_snapshot_schema_id_is_refused_like_before():
    h5 = _h5()
    h5[PROTOCOL_AUTHORED].attrs["schema_id"] = "citrus.protocol.other"
    with pytest.raises(UnifiedH5ContractError, match="^native_protocol_snapshot_status:contract_status='valid'$"):
        require_valid_protocol_snapshot(h5)


def test_the_status_path_is_the_authored_group_member():
    assert PROTOCOL_SEMANTIC_STATUS == "/protocol/authored/protocol_semantic_status_json"
