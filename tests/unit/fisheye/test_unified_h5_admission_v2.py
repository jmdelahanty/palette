"""Admission v2: chaser table v2 and streamed correspondence v2.

The fixture is an unmodified Citrus e1bbc75 / Orange a481250 synthetic
round-trip H5 (one camera) with its external finalization receipt.
"""

from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import (
    UnifiedH5ContractError,
    validate_unified_h5_artifact,
)
from fisheye.shared.unified_h5.correspondence import (
    LIVE_CHASERS,
    PREFLIGHT,
    validate_component_correspondence,
)
from fisheye.shared.unified_h5.schema import (
    CHASER_STATES,
    CONTRACT_HASHES,
    contract,
    table_schema,
    validate_catalog_table,
)
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture, receipt_for

NAME = "admission_v2_roundtrip"


def _canonical_sha256(value) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_packaged_contracts_match_the_admission_pins():
    admission = contract("unified_h5_admission_v2.json")
    pins = admission["correspondence_definition_pins"]
    for name, pin in (
        ("experimental_h5_correspondence_tables_v2.json", pins["catalog"]),
        ("experimental_h5_correspondence_input_v2.schema.json", pins["input_schema"]),
        ("experimental_h5_correspondence_receipt_v2.schema.json", pins["receipt_schema"]),
        (
            "experimental_h5_capacity_preflight_v1.schema.json",
            admission["capacity_preflight_definition_pin"],
        ),
        ("unified_h5_producer_policy_d544b081.json", admission["producer_policy_pin"]),
        (
            "experimental_h5_core_chaser_v2.json",
            admission["core_definition_pins"]["current"],
        ),
        ("experimental_h5_core_v1.json", admission["core_definition_pins"]["historical_v1"]),
    ):
        key = "catalog_file_sha256" if "catalog_file_sha256" in pin else "file_sha256"
        assert "sha256:" + CONTRACT_HASHES[name] == pin[key], name
    for table in pins["catalog"]["tables"]:
        definition = next(
            t
            for t in contract("experimental_h5_correspondence_tables_v2.json")["tables"]
            if t["path"] == table["path"]
        )
        assert _canonical_sha256(definition) == table["definition_canonical_json_sha256"]
    chaser_pin = admission["core_definition_pins"]["current"]["chaser_table"]
    assert _canonical_sha256(table_schema(CHASER_STATES, 2)) == (
        chaser_pin["definition_canonical_json_sha256"]
    )


def test_chaser_v1_and_v2_differ_only_in_version_and_camera_meaning():
    v1, v2 = table_schema(CHASER_STATES, 1), table_schema(CHASER_STATES, 2)
    assert (v1["schema_version"], v2["schema_version"]) == (1, 2)
    changed = [
        a["name"] for a, b in zip(v1["fields"], v2["fields"]) if a != b
    ]
    assert changed == ["target_source_camera_id"]
    assert "Zero is valid when target_source_camera_id_valid=1" in (
        v2["fields"][10]["meaning"]
    )


def test_real_v2_roundtrip_is_admitted(tmp_path):
    path = emit_fixture(tmp_path, NAME)
    with h5py.File(path, "r") as source:
        evidence = validate_unified_h5_artifact(
            source, source_h5=path, finalization_receipt=receipt_for(NAME)
        )
        assert evidence.profile == "unified_experimental_h5_v1"
        assert evidence.frame_count == 4
        assert evidence.component_rows["chaser"] == 4
        assert not evidence.selector_eligible
        summary = validate_component_correspondence(source)
        assert summary.mapped_frame_count == 4 and summary.component_rows == {"chaser": 4}


def _mutable_copy(tmp_path):
    return emit_fixture(tmp_path, NAME)


def test_a_file_cannot_mix_chaser_v1_with_correspondence_v2(tmp_path):
    path = _mutable_copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5[CHASER_STATES].attrs.modify("schema_version", np.uint64(1))
    with h5py.File(path, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="correspondence_revision_mismatch"
    ):
        validate_component_correspondence(h5)


def test_preflight_must_name_the_pinned_producer_policy(tmp_path):
    path = _mutable_copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        document = json.loads(h5[PREFLIGHT][()])
        document["admission_policy_sha256"] = "0" * 64
        raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        dtype, attrs = h5[PREFLIGHT].dtype, dict(h5[PREFLIGHT].attrs)
        del h5[PREFLIGHT]
        h5.create_dataset(PREFLIGHT, data=raw, dtype=dtype)
        for key, value in attrs.items():
            h5[PREFLIGHT].attrs[key] = value
    with h5py.File(path, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="capacity_preflight"
    ):
        validate_component_correspondence(h5)


def test_live_chaser_validity_fill_is_enforced_by_the_catalog(tmp_path):
    path = _mutable_copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        rows = h5[LIVE_CHASERS][()]
        rows[0]["target_recording_frame_valid"] = 0  # absent target, nonzero payload
        h5[LIVE_CHASERS][...] = rows
    with h5py.File(path, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="invalid_value_fill"
    ):
        validate_catalog_table(h5, LIVE_CHASERS)


def test_tampered_live_identity_is_refused(tmp_path):
    path = _mutable_copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        rows = h5[LIVE_CHASERS][()]
        rows[1]["source_recording_frame_id"] = 3  # disagrees with its live frame
        h5[LIVE_CHASERS][...] = rows
    with h5py.File(path, "r") as h5, pytest.raises(UnifiedH5ContractError):
        validate_component_correspondence(h5)
