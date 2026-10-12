"""Unified H5 admission revision 5: catalog angle_v1, stimulus angle convention v1.

agent-contracts PR 92 (head aa337ec; admission, policy 9af921ca, witnesses) and
PR 91 (head 4d0fa15; the angle convention), Citrus PR 27 at c9d52be (catalog
and capacity preflight v3). All three were unmerged at pin time. A capacity
preflight v3 file is selected by revision 5's table-version map plus
catalog_id; the selected binding fixes the stimulus angle convention, and the
file's /metadata/session and protocol declarations must agree with it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from fisheye.shared.heading_convention import (
    LEGACY_CANVAS_ANGLE,
    STIMULUS_ANGLE_CONVENTION,
    stimulus_angle_vector,
    wrap_delta_degrees,
)
from fisheye.shared.unified_h5 import UnifiedH5ContractError, validate_unified_h5_artifact
from fisheye.shared.unified_h5.correspondence import (
    BINDING,
    PREFLIGHT,
    RECEIPT,
    validate_capacity_preflight,
    validate_component_correspondence,
)
from fisheye.shared.unified_h5.geometry import validate_geometry
from fisheye.shared.unified_h5.integrity import validate_internal_integrity
from fisheye.shared.unified_h5.protocol import validate_protocol
from fisheye.shared.unified_h5.rows import validate_table_relations
from fisheye.shared.unified_h5.schema import (
    ADMISSIONS_BY_PREFLIGHT_VERSION,
    CONTRACT_HASHES,
    CORRESPONDENCE_RECEIPT_SCHEMAS,
    admitted_binding,
    catalog_table_versions,
    contract,
    read_json,
    table_schema,
)
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture, receipt_for

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
ANGLE = FIXTURES / "citrus_unified_h5_angle_v1"
COMPLETE = "angle_v1_complete.h5"
PARTIAL = "angle_v1_partial_adoption_refused.h5"
MISSING = "angle_v1_missing_session_convention_refused.h5"
LEGACY = "legacy_frame_identity_v1_with_session_convention_refused.h5"
DIGESTS = {
    COMPLETE: "aee0217b984a7ad3c57d7a2a7a8db21f7bd746c05c77d6f769bb6cc157d54df1",
    PARTIAL: "db818b7aae5a6aa3babb9879ac530a9d3c98a8842347793b2c91a55e3565f482",
    MISSING: "6924a717fa130ecae3d90405f00d1488fa113a4d5b017d50fb67eaffe17a6ce8",
    LEGACY: "34025b7b4cf37e068726c612d1066563edd016019ee97d4476f57d25c24db313",
}
FRAME_IDENTITY_COMPLETE = FIXTURES / "citrus_unified_h5_frame_identity_v1" / "frame_identity_v1_complete.h5"
BOUNDING_BOXES_V2 = FIXTURES / "citrus_unified_h5_bounding_boxes_v2" / "bounding_boxes_v2_capacity_pose_off.h5"
ANGLE_CATALOG = "experimental_h5_core_angle_v1.json"
FRAME_IDENTITY_CATALOG = "experimental_h5_core_frame_identity_v1.json"
CHANGED = {"/components/moving_grating/states": 3, "/components/independent_motion_grid/states": 2}
V4 = ADMISSIONS_BY_PREFLIGHT_VERSION[2]
V5 = ADMISSIONS_BY_PREFLIGHT_VERSION[3]
FRAME_IDENTITY_POLICY = "3c9866eda2f78c4aa37d1b68a9f4b6245aab8415b3ce8e3412cb3cbf5ab7eb78"
SESSION = "/metadata/session"
GRATING = "/components/moving_grating/states"


def _canonical_sha256(value) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _copy(tmp_path: Path, name: str = COMPLETE) -> Path:
    return Path(shutil.copy(ANGLE / name, tmp_path / name))


def _rewrite_json(path: Path, dataset: str, **changes) -> None:
    with h5py.File(path, "r+") as h5:
        document = json.loads(h5[dataset][()])
        for key, value in changes.items():
            if value is None:
                document.pop(key, None)
            else:
                document[key] = value
        dtype = h5[dataset].dtype
        del h5[dataset]
        h5.create_dataset(
            dataset, data=json.dumps(document, sort_keys=True, separators=(",", ":")), dtype=dtype
        )


def _binding(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        document = json.loads(h5[PREFLIGHT][()])
        return admitted_binding(h5, document["admission_policy_sha256"], document)


# --- pins -------------------------------------------------------------------


def test_vendored_witnesses_are_the_published_bytes():
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((ANGLE / name).read_bytes()).hexdigest() == digest
        assert digest in (ANGLE / "README.md").read_text()


def test_preflight_versions_select_revisions_3_4_5():
    assert ADMISSIONS_BY_PREFLIGHT_VERSION == {
        1: "unified_h5_admission_v3.json",
        2: "unified_h5_admission_v4.json",
        3: "unified_h5_admission_v5.json",
    }


def test_packaged_contracts_match_the_revision_5_pins():
    admission = contract(V5)
    assert admission["schema_version"] == 5
    assert admission["versions"]["capacity_preflight"]["schema_version"] == 3
    assert admission["versions"]["stimulus_angle"]["contract_id"] == "stimulus_angle_convention"
    pins = admission["core_definition_pins"]
    assert pins["selection"]["key"] == "table_version_map"
    pin = pins["angle_v1"]
    assert pin["admitted_by_this_revision"] is True
    assert pin["stimulus_angle_convention"] == STIMULUS_ANGLE_CONVENTION
    assert "sha256:" + CONTRACT_HASHES[ANGLE_CATALOG] == pin["catalog_file_sha256"]
    catalog = contract(ANGLE_CATALOG)
    assert _canonical_sha256(catalog) == pin["catalog_canonical_json_sha256"]
    expected_map = {**catalog_table_versions(FRAME_IDENTITY_CATALOG), **CHANGED}
    assert pin["table_version_map"] == expected_map == catalog_table_versions(ANGLE_CATALOG)
    for path, expected in pin["table_definitions"].items():
        definition = table_schema(path, expected["schema_version"])
        definition["path"] = path
        assert definition["id"] == expected["schema_id"]
        assert _canonical_sha256(definition) == expected["definition_canonical_json_sha256"], path
    binding = admission["admitted_catalog_bindings"]["angle_v1"]
    assert binding["catalog_id"] == "angle_v1"
    assert binding["citrus_commit"] == "c9d52be6dddf07da8f13607f31ae3bcf2aca9d93"
    assert "sha256:" + CONTRACT_HASHES["unified_h5_producer_policy_9af921ca.json"] == binding["producer_policy"]["file_sha256"]
    assert _canonical_sha256(contract("unified_h5_producer_policy_9af921ca.json")) == (
        binding["producer_policy"]["canonical_json_sha256"]
    )
    preflight = "experimental_h5_capacity_preflight_v3.schema.json"
    for pinned in (binding["capacity_preflight_schema"], admission["capacity_preflight_definition_pin"]):
        assert "sha256:" + CONTRACT_HASHES[preflight] == pinned["file_sha256"]
        assert _canonical_sha256(contract(preflight)) == pinned["canonical_json_sha256"]
    receipt = admission["correspondence_definition_pins"]["receipt_schema"]
    assert receipt["schema_version"] == 3
    assert "sha256:" + CONTRACT_HASHES[CORRESPONDENCE_RECEIPT_SCHEMAS[3]] == receipt["file_sha256"]
    # Revision 4's pins carry over, plus the legacy convention; its bindings are identical.
    v4 = contract(V4)
    for name in ("historical_v1", "chaser_v2", "bounding_boxes_v2", "frame_identity_v1"):
        new = dict(pins[name])
        assert new.pop("stimulus_angle_convention") == LEGACY_CANVAS_ANGLE
        assert new == v4["core_definition_pins"][name], name
    for name in ("chaser_v2", "bounding_boxes_v2", "frame_identity_v1"):
        assert admission["admitted_catalog_bindings"][name] == v4["admitted_catalog_bindings"][name]


def test_angle_v1_changes_only_the_angle_tables_meaning():
    for path, version in CHANGED.items():
        old, new = table_schema(path, version - 1), table_schema(path, version)
        assert [(f["name"], f["dtype"], f["offset"]) for f in old["fields"]] == [
            (f["name"], f["dtype"], f["offset"]) for f in new["fields"]
        ]
        assert old != new
    grating = {f["name"]: f["meaning"] for f in table_schema(GRATING, 3)["fields"]}
    assert "counter-clockwise" in grating["motion_direction_degrees_canvas"]


# --- the witnesses ------------------------------------------------------------


def test_the_complete_witness_is_admitted_with_the_v1_convention():
    with h5py.File(ANGLE / COMPLETE, "r") as h5:
        integrity = validate_internal_integrity(h5)
        binding = validate_capacity_preflight(h5)
        validate_table_relations(h5, integrity.table_descriptors)
        correspondence = validate_component_correspondence(h5)
        validate_protocol(h5)
        validate_protocol(h5, binding["stimulus_angle_convention"])
        validate_geometry(h5, read_json(h5, BINDING, canonical=True), integrity.table_descriptors)
        assert json.loads(h5[RECEIPT][()])["schema_version"] == 3
        assert correspondence.mapped_frame_count == h5["/frames/stimulus"].shape[0]
        rows = h5[GRATING][()]
        assert h5[GRATING].attrs["schema_version"] == 3
        assert "angle_zero_direction" not in h5["/components/chaser/states"].attrs
    assert binding["catalog"] == "angle_v1"
    assert binding["admission"] == V5
    assert binding["preflight_schema"] == "experimental_h5_capacity_preflight_v3.schema.json"
    assert binding["correspondence_receipt_version"] == 3
    assert binding["stimulus_angle_convention"] == STIMULUS_ANGLE_CONVENTION
    # Grating motion 90 is up on the y-down canvas; the render axis is motion + 180
    # at positive speed (the drift rule).
    convention = binding["stimulus_angle_convention"]
    assert rows["motion_direction_degrees_canvas"].tolist() == [90.0, 90.0]
    x, y = stimulus_angle_vector(rows["motion_direction_degrees_canvas"], convention)
    assert np.allclose(x, 0.0) and np.allclose(y, -1.0)
    assert (rows["speed_px_per_sec"] > 0).all()
    assert np.allclose(
        wrap_delta_degrees(rows["render_axis_degrees_canvas"] - rows["motion_direction_degrees_canvas"]), -180.0
    )


def test_the_partial_adoption_witness_is_a_catalog_id_mismatch():
    with h5py.File(ANGLE / PARTIAL, "r") as h5:
        with pytest.raises(UnifiedH5ContractError, match="catalog_id_mismatch"):
            validate_capacity_preflight(h5)
        with pytest.raises(UnifiedH5ContractError):
            validate_internal_integrity(h5)


def test_the_partial_adoption_witness_never_falls_back_to_frame_identity(tmp_path):
    """Relabelled as frame_identity_v1 its map matches, but its declarations refuse it."""
    copy = _copy(tmp_path, PARTIAL)
    _rewrite_json(
        copy, PREFLIGHT, schema_version=2, catalog_id="frame_identity_v1",
        admission_policy_sha256=FRAME_IDENTITY_POLICY,
    )
    assert _binding(copy)["stimulus_angle_convention"] == LEGACY_CANVAS_ANGLE
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="stimulus_angle_convention_unexpected"
    ):
        validate_capacity_preflight(h5)


def test_the_missing_session_convention_witness_is_refused():
    assert _binding(ANGLE / MISSING)["catalog"] == "angle_v1"
    with h5py.File(ANGLE / MISSING, "r") as h5:
        assert "stimulus_angle_convention" not in h5[SESSION].attrs
        with pytest.raises(UnifiedH5ContractError, match="stimulus_angle_convention_mismatch"):
            validate_capacity_preflight(h5)
        with pytest.raises(UnifiedH5ContractError, match="stimulus_angle_convention_mismatch"):
            validate_protocol(h5)


def test_a_legacy_binding_with_the_session_attribute_is_refused():
    binding = _binding(ANGLE / LEGACY)
    assert binding["catalog"] == "frame_identity_v1"
    assert binding["stimulus_angle_convention"] == LEGACY_CANVAS_ANGLE
    with h5py.File(ANGLE / LEGACY, "r") as h5:
        assert h5[SESSION].attrs["stimulus_angle_convention"] == STIMULUS_ANGLE_CONVENTION
        with pytest.raises(UnifiedH5ContractError, match="stimulus_angle_convention_unexpected"):
            validate_capacity_preflight(h5)


# --- mutated copies -------------------------------------------------------------


def test_a_session_attribute_naming_the_legacy_convention_is_refused(tmp_path):
    copy = _copy(tmp_path)
    with h5py.File(copy, "r+") as h5:
        h5[SESSION].attrs["stimulus_angle_convention"] = LEGACY_CANVAS_ANGLE
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="stimulus_angle_convention_mismatch"
    ):
        validate_capacity_preflight(h5)


@pytest.mark.parametrize(
    "dataset",
    [
        "/protocol/authored/protocol_definition_json",
        "/protocol/authored/protocol_trial_index_json",
        "/protocol/authored/protocol_semantic_json",
    ],
)
@pytest.mark.parametrize("value", [None, LEGACY_CANVAS_ANGLE])
def test_a_protocol_that_does_not_declare_the_convention_is_refused(tmp_path, dataset, value):
    copy = _copy(tmp_path)
    _rewrite_json(copy, dataset, stimulus_angle_convention=value)
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="stimulus_angle_convention_mismatch"
    ):
        validate_capacity_preflight(h5)


def test_a_legacy_binding_with_a_protocol_declaration_is_refused(tmp_path):
    copy = Path(shutil.copy(FRAME_IDENTITY_COMPLETE, tmp_path / "fi.h5"))
    _rewrite_json(
        copy, "/protocol/authored/protocol_definition_json",
        stimulus_angle_convention=STIMULUS_ANGLE_CONVENTION,
    )
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="stimulus_angle_convention_unexpected"
    ):
        validate_capacity_preflight(h5)


@pytest.mark.parametrize("catalog_id", ["frame_identity_v1", "angle_v2", None])
def test_a_preflight_v3_catalog_id_other_than_angle_v1_is_refused(tmp_path, catalog_id):
    copy = _copy(tmp_path)
    _rewrite_json(copy, PREFLIGHT, catalog_id=catalog_id)
    with h5py.File(copy, "r") as h5, pytest.raises(UnifiedH5ContractError, match="catalog_id_mismatch"):
        validate_capacity_preflight(h5)


def test_the_frame_identity_policy_under_preflight_v3_is_refused(tmp_path):
    copy = _copy(tmp_path)
    _rewrite_json(copy, PREFLIGHT, admission_policy_sha256=FRAME_IDENTITY_POLICY)
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="capacity_preflight_producer_policy"
    ):
        validate_capacity_preflight(h5)


# --- the protocol contract follows the binding -----------------------------------


def test_the_angle_v1_protocol_is_refused_under_the_legacy_convention():
    with h5py.File(ANGLE / COMPLETE, "r") as h5, pytest.raises(Exception, match="semantic.v1"):
        validate_protocol(h5, LEGACY_CANVAS_ANGLE)


def test_a_legacy_protocol_is_refused_under_the_v1_convention():
    with h5py.File(FRAME_IDENTITY_COMPLETE, "r") as h5:
        validate_protocol(h5)
        with pytest.raises(Exception, match="semantic.v2"):
            validate_protocol(h5, STIMULUS_ANGLE_CONVENTION)


# --- revisions 3 and 4 keep the legacy convention ----------------------------------


@pytest.mark.parametrize(
    ("path", "catalog"),
    [(FRAME_IDENTITY_COMPLETE, "frame_identity_v1"), (BOUNDING_BOXES_V2, "bounding_boxes_v2")],
)
def test_earlier_revisions_resolve_the_legacy_convention(path, catalog):
    with h5py.File(path, "r") as h5:
        binding = validate_capacity_preflight(h5)
    assert binding["catalog"] == catalog
    assert binding["stimulus_angle_convention"] == LEGACY_CANVAS_ANGLE


def test_the_admission_result_carries_the_convention(tmp_path):
    name = "admission_v2_roundtrip"
    path = emit_fixture(tmp_path, name)
    with h5py.File(path, "r") as source:
        evidence = validate_unified_h5_artifact(
            source, source_h5=path, finalization_receipt=receipt_for(name)
        )
    assert evidence.stimulus_angle_convention == LEGACY_CANVAS_ANGLE
    claims = evidence.manifest_claims()
    assert claims["stimulus_angle_convention"] == LEGACY_CANVAS_ANGLE
    assert claims["catalog"] == evidence.catalog
