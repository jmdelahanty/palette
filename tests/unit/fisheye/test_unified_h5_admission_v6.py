"""Unified H5 admission revision 6: phototaxis_v1 (agent-contracts PR 101).

Revision 6 adds one core table, ``/components/phototaxis/states``, to the
angle_v1 catalog, and is the first revision that admits a reactive
heading-split trial-index step (agent-contracts #98).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import pytest

from fisheye.shared.heading_convention import STIMULUS_ANGLE_CONVENTION
from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.correspondence import (
    BINDING,
    PREFLIGHT,
    validate_capacity_preflight,
    validate_component_correspondence,
)
from fisheye.shared.unified_h5.geometry import validate_geometry
from fisheye.shared.unified_h5.integrity import validate_internal_integrity
from fisheye.shared.unified_h5.phototaxis import validate_phototaxis_states
from fisheye.shared.unified_h5.protocol import validate_protocol
from fisheye.shared.unified_h5.rows import validate_table_relations
from fisheye.shared.unified_h5.schema import (
    ADMISSIONS_BY_PREFLIGHT_VERSION,
    CONTRACT_HASHES,
    admitted_binding,
    catalog_table_versions,
    contract,
    read_json,
    table_schema,
)
from fisheye.shared.protocol_semantic_contract import ProtocolSemanticContractError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
PHOTOTAXIS = FIXTURES / "citrus_unified_h5_phototaxis_v1"
COMPLETE = "phototaxis_v1_complete.h5"
PARTIAL = "phototaxis_v1_partial_adoption_refused.h5"
ANGLE_ID = "phototaxis_v1_catalog_id_angle_v1_refused.h5"
WRONG_STEP = "phototaxis_v1_states_wrong_step_refused.h5"
HASH_MISMATCH = "phototaxis_v1_states_hash_mismatch_refused.h5"
MISSING_POSE_ROW = "phototaxis_v1_states_missing_pose_row_refused.h5"
# agent-contracts PR 101 head b069be2, generated at Citrus d6754a3 (commit C).
DIGESTS = {
    COMPLETE: "e9777bdacd4ac5aa1398b5ef6066abc2636407de74e187629e3e66522b6836dd",
    PARTIAL: "bfb8c1618812ddf47bd395f94633512680e7bb0f7e3be99deae292082dac56ef",
    ANGLE_ID: "0b4efec0a4de3bff09963b33bf1cad2fc68d360580ea109c449fc1be748811cc",
    WRONG_STEP: "b201c091000debc3a2d60e833805f17e4bb68d418858e557a16ad3c1d7326446",
    HASH_MISMATCH: "b510722d846857ee1e5be1f53ea3e1059e5519d54fe01f4b0c8e9407058f1cb7",
    MISSING_POSE_ROW: "dcfaecf80186929565e88ed97d30cd3c18a750a185c71788955d281ee68a181e",
}
ANGLE_COMPLETE = FIXTURES / "citrus_unified_h5_angle_v1" / "angle_v1_complete.h5"
CATALOG = "experimental_h5_core_phototaxis_v1.json"
ANGLE_CATALOG = "experimental_h5_core_angle_v1.json"
POLICY = "unified_h5_producer_policy_012cd25f.json"
PREFLIGHT_V4 = "experimental_h5_capacity_preflight_v4.schema.json"
STATES = "/components/phototaxis/states"
V5 = ADMISSIONS_BY_PREFLIGHT_VERSION[3]
V6 = ADMISSIONS_BY_PREFLIGHT_VERSION[4]


def _canonical_sha256(value) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _binding(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        document = json.loads(h5[PREFLIGHT][()])
        return admitted_binding(h5, document["admission_policy_sha256"], document)


# --- pins -------------------------------------------------------------------


def test_vendored_witnesses_are_the_published_bytes():
    readme = (PHOTOTAXIS / "README.md").read_text()
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((PHOTOTAXIS / name).read_bytes()).hexdigest() == digest
        assert digest in readme


def test_preflight_v4_selects_revision_6():
    assert V6 == "unified_h5_admission_v6.json"
    assert contract(V6)["schema_version"] == 6
    assert contract(V6)["versions"]["capacity_preflight"]["schema_version"] == 4


def test_packaged_contracts_match_the_revision_6_pins():
    admission = contract(V6)
    pins = admission["core_definition_pins"]
    assert pins["selection"]["current_producer_catalog"] == "phototaxis_v1"
    pin = pins["phototaxis_v1"]
    assert pin["catalog_id"] == "phototaxis_v1"
    assert pin["stimulus_angle_convention"] == STIMULUS_ANGLE_CONVENTION
    assert "phototaxis_semantics" in pin
    assert "sha256:" + CONTRACT_HASHES[CATALOG] == pin["catalog_file_sha256"]
    assert _canonical_sha256(contract(CATALOG)) == pin["catalog_canonical_json_sha256"]
    # angle_v1's map plus exactly the states table.
    assert pin["table_version_map"] == {**catalog_table_versions(ANGLE_CATALOG), STATES: 1}
    assert pin["table_version_map"] == catalog_table_versions(CATALOG)
    for path, expected in pin["table_definitions"].items():
        definition = table_schema(path, expected["schema_version"])
        definition["path"] = path
        assert definition["id"] == expected["schema_id"]
        assert _canonical_sha256(definition) == expected["definition_canonical_json_sha256"], path
    binding = admission["admitted_catalog_bindings"]["phototaxis_v1"]
    assert binding["catalog_id"] == "phototaxis_v1"
    assert binding["citrus_commit"] == "d6754a3e6c620d3acc4c473b47db29bfcfc5a0e8"
    assert "sha256:" + CONTRACT_HASHES[POLICY] == binding["producer_policy"]["file_sha256"]
    assert _canonical_sha256(contract(POLICY)) == binding["producer_policy"]["canonical_json_sha256"]
    for pinned in (binding["capacity_preflight_schema"], admission["capacity_preflight_definition_pin"]):
        assert "sha256:" + CONTRACT_HASHES[PREFLIGHT_V4] == pinned["file_sha256"]
        assert _canonical_sha256(contract(PREFLIGHT_V4)) == pinned["canonical_json_sha256"]
    join = admission["correspondence_scope"]["component_current_source_join"]
    assert STATES in join["state_tables"]
    # Revision 5's pins and bindings carry over unchanged.
    v5 = contract(V5)
    for name in ("historical_v1", "chaser_v2", "bounding_boxes_v2", "frame_identity_v1", "angle_v1"):
        assert pins[name] == v5["core_definition_pins"][name], name
    for name in ("chaser_v2", "bounding_boxes_v2", "frame_identity_v1", "angle_v1"):
        assert admission["admitted_catalog_bindings"][name] == v5["admitted_catalog_bindings"][name], name


def test_the_states_contract_matches_the_catalog_table():
    states = contract("reactive_phototaxis_states_v1.json")
    definition = table_schema(STATES, 1)
    assert states["itemsize"] == definition["itemsize"] == 184
    assert [(f["name"], f["dtype"], f.get("count", 1)) for f in states["fields"]] == [
        (f["name"], f["dtype"], f.get("count", 1)) for f in definition["fields"]
    ]


def test_only_phototaxis_v1_admits_heading_split():
    assert _binding(PHOTOTAXIS / COMPLETE)["admits_heading_split"] is True
    assert _binding(ANGLE_COMPLETE)["admits_heading_split"] is False


# --- the witnesses ------------------------------------------------------------


def _stages(path: Path):
    with h5py.File(path, "r") as h5:
        integrity = validate_internal_integrity(h5)
        binding = validate_capacity_preflight(h5)
        validate_table_relations(h5, integrity.table_descriptors)
        correspondence = validate_component_correspondence(h5)
        snapshot, execution = validate_protocol(h5)
        validate_geometry(h5, read_json(h5, BINDING, canonical=True), integrity.table_descriptors)
        rows = validate_phototaxis_states(
            h5, snapshot, execution, camera_serial=correspondence.camera_serial
        )
        states = h5[STATES][()]
    return binding, snapshot, rows, states


def test_the_complete_witness_is_admitted():
    binding, snapshot, rows, states = _stages(PHOTOTAXIS / COMPLETE)
    assert binding["catalog"] == "phototaxis_v1"
    assert binding["admission"] == V6
    assert binding["preflight_schema"] == PREFLIGHT_V4
    assert binding["stimulus_angle_convention"] == STIMULUS_ANGLE_CONVENTION
    assert binding["admits_heading_split"] is True
    assert snapshot.steps[0].stimulus_mode == "PHOTOTAXIS"
    assert rows == 4
    assert states["stimulus_frame_num"].tolist() == [0, 1, 2, 3]
    assert states["input_used"].tolist() == [1, 2, 2, 2]
    assert states["held_age_updates"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        (WRONG_STEP, "phototaxis_step_not_active"),
        (HASH_MISMATCH, "phototaxis_pose_identity_mismatch"),
        (MISSING_POSE_ROW, "phototaxis_pose_row_missing"),
    ],
)
def test_states_witnesses_are_refused_by_the_states_contract(name, reason):
    # Mutated after sealing, so integrity refuses them first; the states
    # contract must refuse them on its own as well.
    with pytest.raises(UnifiedH5ContractError, match="canonical_dependency_mismatch"):
        _stages(PHOTOTAXIS / name)
    with h5py.File(PHOTOTAXIS / name, "r") as h5:
        correspondence = validate_component_correspondence(h5)
        snapshot, execution = validate_protocol(h5)
        with pytest.raises(UnifiedH5ContractError, match=reason):
            validate_phototaxis_states(
                h5, snapshot, execution, camera_serial=correspondence.camera_serial
            )


def test_the_complete_witness_heading_split_step_is_refused_without_revision_6():
    with h5py.File(PHOTOTAXIS / COMPLETE, "r") as h5:
        with pytest.raises(ProtocolSemanticContractError, match="no admission revision before 6"):
            validate_protocol(h5, STIMULUS_ANGLE_CONVENTION, False)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        # States schema_version 2 exists in no pinned catalog, so Palette refuses
        # it before catalog selection (the witness README names the later
        # catalog_id_mismatch; either way it never falls back to angle_v1).
        (PARTIAL, "unsupported_table_version:/components/phototaxis/states|catalog_id_mismatch"),
        (ANGLE_ID, "catalog_id_mismatch"),
    ],
)
def test_refused_witnesses_are_refused(name, reason):
    with h5py.File(PHOTOTAXIS / name, "r") as h5:
        document = json.loads(h5[PREFLIGHT][()])
        with pytest.raises(UnifiedH5ContractError, match=reason):
            admitted_binding(h5, document["admission_policy_sha256"], document)
        with pytest.raises(UnifiedH5ContractError):
            validate_internal_integrity(h5)
