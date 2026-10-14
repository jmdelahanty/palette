"""Unified H5 admission revision 7: video_overlay_v1 and pose observation v3.

agent-contracts PR 104 (head d2876ba), Citrus PR 42 commits a3849567 (catalog),
1855e374 (policy pin) and 20f27395 (witnesses). Revision 7 adds
``/components/video_tracking_overlay/states`` and its settings JSON to
phototaxis_v1, and requires pose observation v3 (skeleton sidecar in the
file, headings recomputable from keypoints) under that binding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.correspondence import (
    BINDING,
    PREFLIGHT,
    validate_capacity_preflight,
    validate_component_correspondence,
)
from fisheye.shared.unified_h5.geometry import validate_geometry
from fisheye.shared.unified_h5.integrity import validate_internal_integrity
from fisheye.shared.unified_h5.pose import CONTRACT as POSE_CONTRACT
from fisheye.shared.unified_h5.pose import GROUP as POSE_GROUP
from fisheye.shared.unified_h5.pose import _json as pose_json
from fisheye.shared.unified_h5.pose_v3 import check_v3_headings
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
from fisheye.shared.unified_h5.video_overlay import validate_video_overlay_states

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
OVERLAY = FIXTURES / "citrus_unified_h5_video_overlay_v1"
COMPLETE = "video_overlay_v1_complete.h5"
RIG = "video_overlay_v1_rig_homography_complete.h5"
HORIZON = "video_overlay_v1_probe_horizon_complete.h5"
# agent-contracts PR 104 head d2876ba, generated at Citrus 20f27395 (commit C).
DIGESTS = {
    COMPLETE: "8057641421531d4b12ba829f877ccebda0d2f614b59db653b92228049de082f8",
    RIG: "0fcfd8aeec0ea591d8597a31671766370836bd7cf40acaa37549c85c6dd58c53",
    HORIZON: "55ef06963664c24adb2a22ab6ca27d0f44def05cd8546c09860bce8151b631ee",
    "video_overlay_v1_partial_adoption_refused.h5": "262d9522768221d808a283eaa05593ecf58390e9a210c762d20aecfda660e62a",
    "video_overlay_v1_catalog_id_phototaxis_v1_refused.h5": "348f19afd227db6bbe719c704c776962ae3cea474883d6d1fc393a1050915c16",
    "video_overlay_v1_states_position_mismatch_refused.h5": "9378a79ef34176383373e345e53bc67b531f22e069c72f5362a188556254cd1b",
    "video_overlay_v1_states_missing_box_row_refused.h5": "a0ac0dcc67362a9df7254295d591ebaceea7d624f2e571bed84dc7aed4d8f576",
    "video_overlay_v1_states_coverage_gap_refused.h5": "ebc255acd3e83b164e6fd4cc47d85a1a982b651740393a3f36dfc6a0a470d721",
    "video_overlay_v1_states_arrow_flag_refused.h5": "ad3d67f080e1cc864245ece774da41c80e89256253e52bbda2544a9b5ba4ab65",
    "video_overlay_v1_pose_heading_not_reproduced_refused.h5": "796a910a2467534495fabe659abf862fa66c4f4f4637429b4e8230482477676d",
    "video_overlay_v1_pose_sidecar_bytes_refused.h5": "66cbf42be3c82641d4294c6d9c6a8fa4d69067922194bd5f89ff6e92bab404ed",
    "video_overlay_v1_rig_drawn_without_video_frame_refused.h5": "4b064cee9710441b56f5d4b46e2fb03309b36d8ca46898f997c2fbce4230bd92",
    "video_overlay_v1_rig_position_off_by_one_refused.h5": "611bad844e661ecc306a2f782bb97829edcad646e4235793f665c285d8e208b9",
    "video_overlay_v1_horizon_arrow_drawn_refused.h5": "10ee0e457819d831771abe111ddd53442665f0e00e6524224d5c072723cd434c",
}
CATALOG = "experimental_h5_core_video_overlay_v1.json"
PHOTOTAXIS_CATALOG = "experimental_h5_core_phototaxis_v1.json"
POLICY = "unified_h5_producer_policy_543102f7.json"
PREFLIGHT_V5 = "experimental_h5_capacity_preflight_v5.schema.json"
STATES = "/components/video_tracking_overlay/states"
V6 = ADMISSIONS_BY_PREFLIGHT_VERSION[4]
V7 = ADMISSIONS_BY_PREFLIGHT_VERSION[5]


def _canonical_sha256(value) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


# --- pins -------------------------------------------------------------------


def test_vendored_witnesses_are_the_published_bytes():
    readme = (OVERLAY / "README.md").read_text()
    assert sorted(p.name for p in OVERLAY.glob("*.h5")) == sorted(DIGESTS)
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((OVERLAY / name).read_bytes()).hexdigest() == digest
        assert digest in readme


def test_preflight_v5_selects_revision_7():
    assert V7 == "unified_h5_admission_v7.json"
    assert contract(V7)["schema_version"] == 7
    assert contract(V7)["versions"]["capacity_preflight"]["schema_version"] == 5


def test_packaged_contracts_match_the_revision_7_pins():
    admission = contract(V7)
    pins = admission["core_definition_pins"]
    assert pins["selection"]["current_producer_catalog"] == "video_overlay_v1"
    pin = pins["video_overlay_v1"]
    assert "phototaxis_semantics" in pin and "video_overlay_semantics" in pin
    assert "sha256:" + CONTRACT_HASHES[CATALOG] == pin["catalog_file_sha256"]
    assert _canonical_sha256(contract(CATALOG)) == pin["catalog_canonical_json_sha256"]
    assert pin["table_version_map"] == {**catalog_table_versions(PHOTOTAXIS_CATALOG), STATES: 1}
    assert pin["table_version_map"] == catalog_table_versions(CATALOG)
    for path, expected in pin["table_definitions"].items():
        definition = table_schema(path, expected["schema_version"])
        definition["path"] = path
        assert definition["id"] == expected["schema_id"]
        assert _canonical_sha256(definition) == expected["definition_canonical_json_sha256"], path
    binding = admission["admitted_catalog_bindings"]["video_overlay_v1"]
    assert binding["citrus_commit"] == "20f27395e159fbdfe03caafbf5e8607b992ddcd8"
    assert "sha256:" + CONTRACT_HASHES[POLICY] == binding["producer_policy"]["file_sha256"]
    assert _canonical_sha256(contract(POLICY)) == binding["producer_policy"]["canonical_json_sha256"]
    for pinned in (binding["capacity_preflight_schema"], admission["capacity_preflight_definition_pin"]):
        assert "sha256:" + CONTRACT_HASHES[PREFLIGHT_V5] == pinned["file_sha256"]
        assert _canonical_sha256(contract(PREFLIGHT_V5)) == pinned["canonical_json_sha256"]
    assert STATES in admission["correspondence_scope"]["component_current_source_join"]["state_tables"]
    v6 = contract(V6)
    for name in ("historical_v1", "chaser_v2", "bounding_boxes_v2", "frame_identity_v1", "angle_v1", "phototaxis_v1"):
        assert pins[name] == v6["core_definition_pins"][name], name
    for name in ("chaser_v2", "bounding_boxes_v2", "frame_identity_v1", "angle_v1", "phototaxis_v1"):
        assert admission["admitted_catalog_bindings"][name] == v6["admitted_catalog_bindings"][name], name


def test_the_states_contract_matches_the_catalog_table():
    states = contract("video_tracking_overlay_states_v1.json")
    definition = table_schema(STATES, 1)
    assert states["itemsize"] == definition["itemsize"] == 160
    assert [(f["name"], f["dtype"], f.get("count", 1)) for f in states["fields"]] == [
        (f["name"], f["dtype"], f.get("count", 1)) for f in definition["fields"]
    ]


# --- the witnesses ------------------------------------------------------------


def _binding(h5) -> dict:
    document = json.loads(h5[PREFLIGHT][()])
    return admitted_binding(h5, document["admission_policy_sha256"], document)


def _stages(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        integrity = validate_internal_integrity(h5)
        binding = validate_capacity_preflight(h5)
        validate_table_relations(h5, integrity.table_descriptors)
        correspondence = validate_component_correspondence(h5)
        validate_protocol(h5)
        validate_geometry(h5, read_json(h5, BINDING, canonical=True), integrity.table_descriptors)
        outcomes = {value["component_id"]: value for value in integrity.component_outcomes}
        overlay = validate_video_overlay_states(
            h5, binding, camera_serial=correspondence.camera_serial, outcomes=outcomes
        )
        return {"binding": binding, "overlay": overlay, "pose_rows": integrity.pose_rows,
                "states": h5[STATES][()]}


@pytest.mark.parametrize("name", [COMPLETE, RIG, HORIZON])
def test_the_complete_witnesses_are_admitted(name):
    result = _stages(OVERLAY / name)
    binding = result["binding"]
    assert binding["catalog"] == "video_overlay_v1"
    assert binding["admission"] == V7
    assert binding["preflight_schema"] == PREFLIGHT_V5
    assert binding["admits_heading_split"] is True
    assert binding["admits_video_overlay"] is True
    assert result["overlay"] == {"rows": 4, "superseded_box_rows": 0}


def test_the_witnesses_exercise_drawn_rows_and_the_probe_horizon():
    rig = _stages(OVERLAY / RIG)["states"]
    assert rig["drawn"].tolist() == [1, 1, 1, 0]
    assert rig["arrow_drawn"].tolist() == [1, 0, 0, 0]
    assert rig["position_arena_local_px"][0].tolist() == [1514.5, 441.0]
    horizon = _stages(OVERLAY / HORIZON)["states"]
    assert horizon["heading_valid"][0] == 1 and horizon["midline_valid"][0] == 0
    assert horizon["drawn"][0] == 1 and horizon["arrow_drawn"][0] == 0


def test_the_pose_statuses_are_recomputed():
    with h5py.File(OVERLAY / COMPLETE, "r") as h5:
        assert h5[POSE_GROUP + "/objects"]["heading_status"].tolist() == [7, 9, 7, 5, 6, 9, 3]
        check_v3_headings(
            h5, pose_json(h5, POSE_CONTRACT),
            h5[POSE_GROUP + "/updates"], h5[POSE_GROUP + "/objects"], h5[POSE_GROUP + "/keypoints"],
        )


OVERLAY_REFUSALS = [
    ("video_overlay_v1_states_position_mismatch_refused.h5", "video_overlay_position_mismatch:camera_px"),
    ("video_overlay_v1_states_missing_box_row_refused.h5", "video_overlay_box_row_missing"),
    ("video_overlay_v1_states_coverage_gap_refused.h5", "video_overlay_coverage_gap"),
    ("video_overlay_v1_states_arrow_flag_refused.h5", "video_overlay_arrow_flag"),
    ("video_overlay_v1_rig_drawn_without_video_frame_refused.h5", "video_overlay_drawn_flag"),
    ("video_overlay_v1_rig_position_off_by_one_refused.h5", "video_overlay_position_mismatch:arena_local_px"),
    ("video_overlay_v1_horizon_arrow_drawn_refused.h5", "video_overlay_arrow_flag"),
]


@pytest.mark.parametrize(("name", "reason"), OVERLAY_REFUSALS)
def test_states_witnesses_are_refused_by_the_states_contract(name, reason):
    # Mutated after sealing, so integrity refuses them first; the states
    # contract must refuse them on its own as well.
    with pytest.raises(UnifiedH5ContractError, match="canonical_dependency_mismatch"):
        _stages(OVERLAY / name)
    with h5py.File(OVERLAY / name, "r") as h5:
        binding = _binding(h5)
        serial = validate_component_correspondence(h5).camera_serial
        with pytest.raises(UnifiedH5ContractError, match=reason):
            validate_video_overlay_states(h5, binding, camera_serial=serial)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("video_overlay_v1_pose_heading_not_reproduced_refused.h5", "pose_heading_not_reproduced"),
        ("video_overlay_v1_pose_sidecar_bytes_refused.h5", "pose_sidecar_digest_mismatch"),
    ],
)
def test_pose_witnesses_are_refused_by_the_pose_v3_contract(name, reason):
    with pytest.raises(UnifiedH5ContractError, match="internal_dependency_mismatch"):
        _stages(OVERLAY / name)
    with h5py.File(OVERLAY / name, "r") as h5:
        with pytest.raises(UnifiedH5ContractError, match=reason):
            check_v3_headings(
                h5, pose_json(h5, POSE_CONTRACT),
                h5[POSE_GROUP + "/updates"], h5[POSE_GROUP + "/objects"], h5[POSE_GROUP + "/keypoints"],
            )


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        # States schema_version 2 exists in no pinned catalog, so Palette refuses
        # it before catalog selection; it never falls back to phototaxis_v1.
        ("video_overlay_v1_partial_adoption_refused.h5", "unsupported_table_version:/components/video_tracking_overlay/states"),
        ("video_overlay_v1_catalog_id_phototaxis_v1_refused.h5", "catalog_id_mismatch"),
    ],
)
def test_refused_witnesses_are_refused(name, reason):
    with h5py.File(OVERLAY / name, "r") as h5:
        with pytest.raises(UnifiedH5ContractError, match=reason):
            _binding(h5)
        with pytest.raises(UnifiedH5ContractError):
            validate_internal_integrity(h5)


def test_a_revision_6_binding_never_admits_the_overlay():
    with h5py.File(OVERLAY / COMPLETE, "r") as h5:
        binding = {**_binding(h5), "admits_video_overlay": False}
        with pytest.raises(UnifiedH5ContractError, match="video_overlay_not_admitted_by_revision"):
            validate_video_overlay_states(h5, binding, camera_serial="CAM-42")


def test_a_phototaxis_v1_witness_has_no_overlay_and_is_unaffected():
    path = FIXTURES / "citrus_unified_h5_phototaxis_v1" / "phototaxis_v1_complete.h5"
    with h5py.File(path, "r") as h5:
        binding = _binding(h5)
        assert binding["admits_video_overlay"] is False
        assert validate_video_overlay_states(h5, binding, camera_serial="CAM-42") == {
            "rows": 0, "superseded_box_rows": 0,
        }
