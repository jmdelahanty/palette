"""Protocol semantic recipe v3 (agent-contracts citrus-palette/angles pairing revision 2).

Citrus #86 (recipe 2e870c8) admits ``citrus.protocol.semantic.v3`` alongside
v2 under the angle_v1 binding, both with trial index v3. The cases are
Citrus's ``protocol_identity_dump`` output, vendored byte for byte.
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files as resource_files

import pytest

from fisheye.shared.heading_convention import LEGACY_CANVAS_ANGLE, STIMULUS_ANGLE_CONVENTION
from fisheye.shared.protocol_semantic_contract import (
    PROTOCOL_SEMANTIC_RECIPE_V3_NORMALIZATION_POLICY,
    STIMULUS_MODE,
    ProtocolSemanticContractError,
    semantic_v2_relabels_to_v3,
    validate_protocol_semantic_snapshot,
)

CONTRACTS = resource_files("fisheye.shared").joinpath("contracts")
PINNED = {
    # agent-contracts PR 121 head b63c483, citrus-palette/angles/
    "citrus_palette_semantic_policy_pairing_v2.json": "c0fdcab1f0c647adb2368bc499a3b0684e19f46271e38f3856d119f1dab363d6",
    "citrus_palette_semantic_hash_v3_cases.json": "5ab72f82da84bfe788b8cdb8527b8228803bd0186f0d8d59c5afcb32c1f68697",
}


def _load(name: str) -> dict:
    data = CONTRACTS.joinpath(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED[name]
    return json.loads(data)


PAIRING = _load("citrus_palette_semantic_policy_pairing_v2.json")
CASES = {case["case"]: case for case in _load("citrus_palette_semantic_hash_v3_cases.json")["cases"]}
# Cases whose steps the generic trial-index builder below can describe: the
# heading-split case needs a reactive binding and the empty protocol has no
# steps (refused by every recipe), so both are covered by the pure checks only.
SNAPSHOT_CASES = (
    "batman_composable",
    "good_cop_bad_cop",
    "good_cop_bad_cop_demo",
    "goodbatbadbat",
    "snow",
    "v3case_appearance_drive_loom_none",
)
CHASER_CASE = "goodbatbadbat"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _family(mode_id: int) -> str:
    return {4: "solid_color", 12: "chaser", 22: "solid_color"}.get(
        mode_id, STIMULUS_MODE[mode_id].lower()
    )


def _trial_index(semantic_json: str, semantic_hash: str) -> str:
    steps = []
    for index, step in enumerate(json.loads(semantic_json)["identity"]["steps"]):
        duration = step["duration"]
        mode_id = step["stimulus_mode_id"]
        steps.append(
            {
                "duration_s": duration["value"] * float(duration["scale"]),
                "features": (
                    {"color_name": "black", "resolved_color": {"color_space": "srgb", "rgba8": [0, 0, 0, 255]}}
                    if mode_id == 4
                    else {}
                ),
                "index_status": "detailed",
                "post_stimulus_iti_s": 0.0,
                "step_index": index,
                "stimulus_family": _family(mode_id),
                "stimulus_mode": STIMULUS_MODE[mode_id],
                "stimulus_mode_id": mode_id,
            }
        )
    return _json(
        {
            "normalization_policy": "citrus.protocol.trial_index.v3",
            "protocol_semantic_hash": semantic_hash,
            "schema_id": "citrus.protocol.trial_index",
            "schema_version": 3,
            "steps": steps,
            "stimulus_angle_convention": STIMULUS_ANGLE_CONVENTION,
        }
    )


def _validate(semantic_json: str, *, convention: str = STIMULUS_ANGLE_CONVENTION):
    semantic_hash = _hash(semantic_json)
    trial_json = _trial_index(semantic_json, semantic_hash)
    return validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
        trial_index_hash=_hash(trial_json),
        snapshot_schema_version=2,
        snapshot_policy_id="citrus.protocol.snapshot.v2",
        stimulus_angle_convention=convention,
    )


def _mutated_v3(case: str, mutate) -> str:
    semantic = json.loads(CASES[case]["v3"]["canonical_json"])
    mutate(semantic)
    return _json(semantic)


def _chaser_parameters(semantic: dict) -> dict:
    return next(
        step["parameters"] for step in semantic["identity"]["steps"] if step["stimulus_mode_id"] == 12
    )


def test_pairing_revision_admits_v2_and_v3_under_angle_v1_only():
    pairings = {p["stimulus_angle_convention"]: p for p in PAIRING["pairings"]}
    assert pairings[STIMULUS_ANGLE_CONVENTION]["semantic_normalization_policies"] == [
        "citrus.protocol.semantic.v2",
        PROTOCOL_SEMANTIC_RECIPE_V3_NORMALIZATION_POLICY,
    ]
    assert pairings[LEGACY_CANVAS_ANGLE]["semantic_normalization_policies"] == ["citrus.protocol.semantic.v1"]


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_case_hashes_are_over_the_exact_bytes(case, version):
    record = CASES[case][version]
    assert _hash(record["canonical_json"]) == record["protocol_semantic_hash"]


@pytest.mark.parametrize("case", SNAPSHOT_CASES)
def test_v3_is_admitted_under_angle_v1(case):
    snapshot = _validate(CASES[case]["v3"]["canonical_json"])
    assert snapshot.semantic_hash == CASES[case]["v3"]["protocol_semantic_hash"]
    assert snapshot.semantic_payload["normalization_policy"] == PROTOCOL_SEMANTIC_RECIPE_V3_NORMALIZATION_POLICY


@pytest.mark.parametrize("case", SNAPSHOT_CASES)
def test_v2_is_still_admitted_under_angle_v1(case):
    snapshot = _validate(CASES[case]["v2"]["canonical_json"])
    assert snapshot.semantic_hash == CASES[case]["v2"]["protocol_semantic_hash"]


def test_v3_is_refused_under_the_legacy_binding():
    with pytest.raises(ProtocolSemanticContractError, match="normalization_policy must be .citrus.protocol.semantic.v1."):
        _validate(CASES["snow"]["v3"]["canonical_json"], convention=LEGACY_CANVAS_ANGLE)


def test_an_unknown_policy_is_refused():
    text = _mutated_v3("snow", lambda s: s.update(normalization_policy="citrus.protocol.semantic.v4"))
    with pytest.raises(ProtocolSemanticContractError, match="normalization_policy must be"):
        _validate(text)


def test_v2_chaser_steps_are_not_held_to_the_v3_fields():
    # A v2 identity never carries z_eff/target_radius; only v3 requires them.
    assert "z_eff" not in _chaser_parameters(json.loads(CASES[CHASER_CASE]["v2"]["canonical_json"]))
    _validate(CASES[CHASER_CASE]["v2"]["canonical_json"])


def test_z_eff_null_is_admitted():
    assert _chaser_parameters(json.loads(CASES["good_cop_bad_cop_demo"]["v3"]["canonical_json"]))["z_eff"] is None
    _validate(CASES["good_cop_bad_cop_demo"]["v3"]["canonical_json"])


@pytest.mark.parametrize(
    "case,mutate,match",
    [
        (CHASER_CASE, lambda s: _chaser_parameters(s).pop("z_eff"), "z_eff is required"),
        (CHASER_CASE, lambda s: _chaser_parameters(s)["z_eff"].update(value=0), "z_eff must be one positive mm"),
        (CHASER_CASE, lambda s: _chaser_parameters(s)["z_eff"].update(unit="px"), "z_eff must be one positive mm"),
        (CHASER_CASE, lambda s: _chaser_parameters(s).pop("target_radius"), "target_radius must be"),
        (CHASER_CASE, lambda s: _chaser_parameters(s)["target_radius"].update(value=-1), "target_radius must be"),
        (
            "batman_composable",
            lambda s: _chaser_parameters(s)["chasers"][0]["behavior_program"]["phases"][0].pop("target_reference"),
            "target_reference is required on a chase phase",
        ),
        (
            "v3case_appearance_drive_loom_none",
            lambda s: _chaser_parameters(s)["chasers"][1].pop("loom"),
            "loom is required with an appearance_drive",
        ),
    ],
    ids=["z_eff_missing", "z_eff_zero", "z_eff_unit", "target_radius_missing", "target_radius_negative",
         "chase_target_reference_missing", "appearance_drive_without_loom"],
)
def test_malformed_v3_chaser_fields_are_refused(case, mutate, match):
    with pytest.raises(ProtocolSemanticContractError, match=match):
        _validate(_mutated_v3(case, mutate))


@pytest.mark.parametrize("case", sorted(CASES))
def test_relabel_rule_pools_exactly_the_label_only_cases(case):
    record = CASES[case]
    pools = semantic_v2_relabels_to_v3(
        record["v2"]["canonical_json"].encode("utf-8"),
        record["v2"]["protocol_semantic_hash"],
        record["v3"]["protocol_semantic_hash"],
    )
    assert pools is record["label_only"]


def test_relabel_rule_refuses_bytes_that_do_not_match_their_hash():
    record = CASES["snow"]
    tampered = record["v2"]["canonical_json"].encode("utf-8") + b" "
    assert not semantic_v2_relabels_to_v3(
        tampered, record["v2"]["protocol_semantic_hash"], record["v3"]["protocol_semantic_hash"]
    )


def test_relabel_rule_refuses_a_policy_member_that_occurs_twice():
    semantic = json.loads(CASES["snow"]["v2"]["canonical_json"])
    semantic["identity"]["note"] = {"normalization_policy": "citrus.protocol.semantic.v2"}
    text = _json(semantic).encode("utf-8")
    relabeled = text.replace(b'"citrus.protocol.semantic.v2"', b'"citrus.protocol.semantic.v3"')
    assert not semantic_v2_relabels_to_v3(text, _hash(text.decode()), _hash(relabeled.decode()))


def test_relabel_rule_takes_recorded_bytes_only():
    record = CASES["snow"]
    with pytest.raises(TypeError):
        semantic_v2_relabels_to_v3(
            record["v2"]["canonical_json"], record["v2"]["protocol_semantic_hash"], record["v3"]["protocol_semantic_hash"]
        )
    with pytest.raises(ProtocolSemanticContractError):
        semantic_v2_relabels_to_v3(b"{}", "not-a-hash", record["v3"]["protocol_semantic_hash"])


def test_v3_cases_do_not_change_the_v2_fixture_contract():
    # The v2 empty-protocol case from agent-contracts PR 91 keeps its hash.
    empty = copy.deepcopy(json.loads(CASES["v3case_empty"]["v2"]["canonical_json"]))
    assert empty["normalization_policy"] == "citrus.protocol.semantic.v2"
    assert CASES["v3case_empty"]["v2"]["protocol_semantic_hash"] == (
        "sha256:ba17eeca5b173894eb4a19db9903cee2f4b948d84b2920e00662751c7c4000fc"
    )
