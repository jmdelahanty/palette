"""Producer-declared parent recording context (citrus.parent_recording_context v1/v2).

The subtype is a free producer label, independent of behavior_mode; context v2
may omit it, meaning "not specified". Palette never fills, trims or remaps it.
"""

from __future__ import annotations

import pytest

from fisheye.shared import recording_transfer_snapshot as transfer
from fisheye.shared.recording_manifest_context import (
    PRODUCER_CONTEXT_SOURCE,
    recording_manifest_context_issues,
    validate_recording_manifest_context,
)

CONTEXT_V1 = {
    "schema_id": "citrus.parent_recording_context",
    "schema_version": 1,
    "recording_type": "behavior",
    "recording_subtype": "dish_stimulus",
    "behavior_mode": "embedded",
    "recording_intent": "stimulus_experiment",
    "data_origin": "acquired",
}
CONTEXT_V2 = {k: v for k, v in CONTEXT_V1.items() if k != "recording_subtype"} | {
    "schema_version": 2
}


def _manifest(**fields) -> dict:
    return {
        "context_source": PRODUCER_CONTEXT_SOURCE,
        "recording_context_schema_version": 2,
        "artifact_schema_id": "orange_transfer_parent_v1",
        "recording_type": "behavior",
        "behavior_mode": "free",
        "recording_intent": "recording_only",
        "data_origin": "acquired",
        **fields,
    }


def test_subtype_free_producer_manifest_is_valid():
    validate_recording_manifest_context(_manifest())


def test_subtype_is_not_tied_to_behavior_mode_for_producer_context():
    validate_recording_manifest_context(
        _manifest(recording_subtype="dish_stimulus", behavior_mode="embedded")
    )


@pytest.mark.parametrize(
    "fields, code",
    [
        ({"recording_subtype": None}, "invalid_recording_subtype"),
        ({"recording_subtype": ""}, "invalid_recording_subtype"),
        ({"recording_subtype": "free "}, "invalid_recording_subtype"),
        ({"recording_context_schema_version": 1}, "missing_required_field"),
        ({"recording_context_schema_version": 3}, "invalid_recording_context_version"),
        ({"recording_type": "stimulus"}, "invalid_recording_type"),
        ({"behavior_mode": "swimming"}, "invalid_behavior_mode"),
        ({"recording_intent": None}, "invalid_recording_intent"),
        ({"data_origin": "simulated"}, "invalid_data_origin"),
    ],
)
def test_invalid_producer_manifest_context(fields, code):
    assert code in {c for c, _ in recording_manifest_context_issues(_manifest(**fields))}


def test_operator_manifests_keep_the_legacy_rules():
    legacy = {
        "recording_type": "behavior",
        "recording_subtype": "dish_stimulus",
        "behavior_mode": "embedded",
        "artifact_schema_id": "behavior_v1",
    }
    codes = {code for code, _ in recording_manifest_context_issues(legacy)}
    assert {"invalid_recording_subtype", "behavior_mode_mismatch"} <= codes


def test_parent_contexts_accept_v1_and_subtype_free_v2_per_camera():
    manifest = {"recording_contexts": {"A": dict(CONTEXT_V1), "B": dict(CONTEXT_V2)}}
    contexts = transfer.parent_contexts(manifest, ["A", "B"])
    assert contexts["A"]["recording_subtype"] == "dish_stimulus"
    assert "recording_subtype" not in contexts["B"]


@pytest.mark.parametrize("cameras", [["A"], ["A", "B", "C"]])
def test_parent_contexts_cover_exactly_every_camera(cameras):
    manifest = {"recording_contexts": {"A": dict(CONTEXT_V1), "B": dict(CONTEXT_V1)}}
    with pytest.raises(transfer.TransferSnapshotError, match="every camera"):
        transfer.parent_contexts(manifest, cameras)


@pytest.mark.parametrize(
    "label", ["x" * 1025, "tab\there", " lead", "trail ", "﻿bom", "nul\x00"]
)
def test_producer_labels_are_never_trimmed_or_repaired(label):
    context = dict(CONTEXT_V1, recording_subtype=label)
    with pytest.raises(transfer.TransferSnapshotError):
        transfer.parent_contexts({"recording_contexts": {"A": context}}, ["A"])


def test_utf8_byte_budget_is_not_a_character_count():
    context = dict(CONTEXT_V1, recording_subtype="é" * 600)  # 600 chars, 1200 bytes
    with pytest.raises(transfer.TransferSnapshotError, match="byte budget"):
        transfer.parent_contexts({"recording_contexts": {"A": context}}, ["A"])


def _observation(camera: str) -> dict:
    return {"observation_identity": {"identity": {"camera": {"source_camera_stream_id": camera}}}}


def test_recording_only_parent_cannot_carry_a_citrus_observation():
    manifest = {
        "recording_contexts": {"A": dict(CONTEXT_V1, recording_intent="recording_only")},
        "observation_contexts": [_observation("A")],
    }
    with pytest.raises(transfer.TransferSnapshotError, match="recording_only"):
        transfer.parent_contexts(manifest, ["A"])
    manifest["recording_contexts"]["A"]["recording_intent"] = "stimulus_experiment"
    transfer.parent_contexts(manifest, ["A"])


def test_binding_files_without_a_finalized_projection_are_refused(tmp_path):
    refs = {"recording_observation_bindings/finalized_collection.json": {}}
    with pytest.raises(transfer.TransferSnapshotError, match="without finalized"):
        transfer.require_observation_binding_transfer_admission(tmp_path, {}, refs)
    transfer.require_observation_binding_transfer_admission(tmp_path, {}, {})
