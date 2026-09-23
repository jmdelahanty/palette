"""Native paths, existing protocol semantic/execution contract owners."""

import numpy as np

from fisheye.shared.protocol_execution_contract import validate_protocol_execution_index
from fisheye.shared.protocol_semantic_contract import (
    validate_protocol_semantic_snapshot,
)

from .common import require, text
from .hdf5_types import dataset_bytes
from .schema import read_json


def validate_protocol(h5):
    authored, executed = h5["/protocol/authored"], h5["/protocol/executed"]
    for group in (authored, executed):
        version = group.attrs.get_id("schema_version")
        require(
            version.shape == () and version.dtype == np.dtype("<i4"),
            "native_protocol_version_type",
        )
    require(
        authored.attrs.get("schema_id") == "citrus.protocol.snapshot"
        and authored.attrs.get("contract_status") == "valid",
        "native_protocol_snapshot_status",
    )
    # Parse first with the bounded duplicate-key/nonfinite parser, even for v1.
    for path in (
        "protocol_definition_json",
        "protocol_semantic_json",
        "protocol_trial_index_json",
    ):
        read_json(h5, authored.name + "/" + path)
    read_json(h5, executed.name + "/execution_index_json")

    def value(group, name):
        return text(dataset_bytes(group[name]), name)

    snapshot = validate_protocol_semantic_snapshot(
        semantic_hash=value(authored, "protocol_semantic_hash"),
        semantic_json=value(authored, "protocol_semantic_json"),
        trial_index_json=value(authored, "protocol_trial_index_json"),
        trial_index_hash=value(authored, "protocol_trial_index_hash"),
        snapshot_schema_version=int(authored.attrs["schema_version"]),
        snapshot_policy_id=text(authored.attrs["policy_id"], "protocol_policy_id"),
    )
    execution = validate_protocol_execution_index(
        execution_json=value(executed, "execution_index_json"),
        execution_hash=value(executed, "execution_index_hash"),
        snapshot=snapshot,
    )
    require(
        executed.attrs.get("schema_id") == "citrus.protocol.execution_index"
        and executed.attrs.get("schema_version") == 1
        and executed.attrs.get("policy_id")
        == "citrus.protocol.execution_index.half_open_stimulus_frames.v1"
        and executed.attrs.get("status") == execution.status,
        "native_execution_attributes_mismatch",
    )
    return snapshot, execution
