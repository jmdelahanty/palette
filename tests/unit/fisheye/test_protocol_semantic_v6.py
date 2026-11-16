"""Protocol semantic recipe v6 / trial index v5 (pairing revision 5, Citrus #127).

v6 is v5 with every chaser of every chaser step carrying a hashed ``role``,
"test" or "sham"; v6 is the only policy paired with trial index v5, whose
chasers carry the same role. Palette re-checks the roles at intake
(R-role, S1-S6, T1) and pools v5 into v6 only through the crosswalk. The cases
are Citrus's ``protocol_identity_dump`` output, vendored byte for byte from
agent-contracts #137.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files

import h5py
import pytest

from fisheye.shared.heading_convention import LEGACY_CANVAS_ANGLE, STIMULUS_ANGLE_CONVENTION
from fisheye.shared.protocol_semantic_contract import (
    CHASER_ROLE_INTAKE_CODES,
    PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY,
    PROTOCOL_SEMANTIC_RECIPE_V6_NORMALIZATION_POLICY,
    PROTOCOL_TRIAL_INDEX_V5_SCHEMA,
    ProtocolSemanticContractError,
    chaser_role_intake_errors,
    semantic_v5_crosswalks_to_v6,
    validate_protocol_semantic_snapshot,
)
from fisheye.shared.unified_h5.common import UnifiedH5ContractError
from fisheye.shared.unified_h5.z_eff import validate_z_eff

CONTRACTS = resource_files("fisheye.shared").joinpath("contracts")
PINNED = {
    # agent-contracts PR #137 merge 06b30766, citrus-palette/angles/
    "citrus_palette_semantic_policy_pairing_v5.json": "510fccc8d155b2674b0993679f2b82ebe2b3fdb324af0bc6e2e7a67f8761c6f3",
    "citrus_palette_semantic_hash_v6_cases.json": "0c2f308687878ffce4a0ff48312500839c17ed10c71eff1d11451b1cce51c22b",
    # agent-contracts PR #139 head ac4830dc (pending merge; re-vendor from the
    # merge commit before review), citrus-palette/angles/; Citrus #127 b1391c6
    "citrus_protocol_trial_index_v5.schema.json": "f5f3047d746884ec09e26d414278305f20d1d5b56f2f91630a1e9fb39521c85c",
}


def _load(name: str) -> dict:
    data = CONTRACTS.joinpath(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED[name]
    return json.loads(data)


PAIRING = _load("citrus_palette_semantic_policy_pairing_v5.json")
FIXTURE = _load("citrus_palette_semantic_hash_v6_cases.json")
CASES = {case["case"]: case for case in FIXTURE["cases"]}
ROLE_CASES = {case["case"]: case for case in FIXTURE["role_cases"]}
VERSIONS = ("v5_120", "v5_240", "v6_120", "v6_240")
HEADING_SPLIT_CASES = {"demo_chaser_then_phototaxis_seed1", "phototaxis_heading_split_30trials_seed1"}
ADMITTED_ROLE_CASES = ("admitted_test_and_sham", "admitted_tests_without_role")
SHAM_HASH_120 = "sha256:61e8d1fa334fe6ac4fc4b80645925b1e3924a423605cb5039204384d346d89d9"
# Citrus's load refusal for each intake rule it backs (role_rules).
CITRUS_RULES = {
    "chaser_role_sham_differs_from_background": "S1",
    "chaser_role_sham_with_background_edge_falloff": "S2",
    "chaser_role_sham_not_passive": "S3",
    "chaser_role_sham_with_appearance_profile": "S4",
    "chaser_role_sham_with_proximity_feedback": "S5",
    "chaser_role_sham_with_danger_zone": "S6",
    "chaser_role_test_equals_background": "T1",
}
V6 = PROTOCOL_SEMANTIC_RECIPE_V6_NORMALIZATION_POLICY


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trial_object(value) -> dict:
    return json.loads(value) if isinstance(value, str) else json.loads(json.dumps(value))


def _validate(semantic_json: str, trial_index, *, convention: str = STIMULUS_ANGLE_CONVENTION, case: str = ""):
    semantic_hash = _hash(semantic_json)
    trial_json = _json({**_trial_object(trial_index), "protocol_semantic_hash": semantic_hash})
    return validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
        trial_index_hash=_hash(trial_json),
        snapshot_schema_version=2,
        snapshot_policy_id="citrus.protocol.snapshot.v2",
        stimulus_angle_convention=convention,
        admits_heading_split=case in HEADING_SPLIT_CASES,
    )


def _sham() -> tuple[str, dict]:
    entry = ROLE_CASES["admitted_test_and_sham"]["v6_120"]
    return entry["canonical_json"], _trial_object(entry["protocol_trial_index_json"])


def _chaser_step(steps: list) -> dict:
    return next(s for s in steps if s["stimulus_mode_id"] == 12)


def test_trial_index_v5_schema_pin_matches_the_reader():
    assert PROTOCOL_TRIAL_INDEX_V5_SCHEMA == (
        "citrus_protocol_trial_index_v5.schema.json",
        PINNED["citrus_protocol_trial_index_v5.schema.json"],
    )
    assert PAIRING["semantic_v6"]["trial_index"]["schema"].count(PROTOCOL_TRIAL_INDEX_V5_SCHEMA[1]) == 1


def test_pairing_revision_5_pairs_v6_with_trial_index_v5_only():
    assert PAIRING["revision"] == 5
    angle = [p for p in PAIRING["pairings"] if p["stimulus_angle_convention"] == STIMULUS_ANGLE_CONVENTION]
    assert [p["semantic_normalization_policies"] for p in angle][-2:] == [
        [PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY],
        [V6],
    ]
    v6 = PAIRING["semantic_v6"]
    assert [f["path"] for f in v6["newly_hashed"]] == ["identity.steps[].parameters.chasers[].role"]
    assert v6["removed"] == []
    assert (v6["trial_index"]["schema_version"], v6["trial_index"]["normalization_policy"]) == (
        5,
        "citrus.protocol.trial_index.v5",
    )
    assert sorted(r["code"] for r in v6["role_rules"]["citrus_load_refusals"]) == sorted(CITRUS_RULES)
    assert PAIRING["relabel_pooling_rule"]["rule_v5_to_v6"].startswith("crosswalk.")
    assert PAIRING["h5"].startswith("none")
    assert set(CHASER_ROLE_INTAKE_CODES) == {"R-role", "S1", "S2", "S3", "S4", "S5", "S6", "T1"}
    for rule in CHASER_ROLE_INTAKE_CODES:
        assert rule + "," in v6["role_rules"]["intake_recheck"] or rule + " " in v6["role_rules"]["intake_recheck"]


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_case_hashes_are_over_the_exact_bytes(case, version):
    record = CASES[case][version]
    assert _hash(record["canonical_json"]) == record["protocol_semantic_hash"]


@pytest.mark.parametrize("version", ["v6_120", "v6_240"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_v6_is_admitted_with_citrus_trial_index_v5(case, version):
    record = CASES[case][version]
    snapshot = _validate(record["canonical_json"], record["protocol_trial_index_json"], case=case)
    assert snapshot.semantic_hash == record["protocol_semantic_hash"]
    assert snapshot.semantic_payload["normalization_policy"] == V6
    assert snapshot.trial_index_schema_version == 5
    assert snapshot.stimulus_render_rate_hz == int(version[-3:])


@pytest.mark.parametrize("version", ["v5_120", "v5_240"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_v5_cases_still_pass_with_trial_index_v4(case, version):
    record = CASES[case][version]
    snapshot = _validate(record["canonical_json"], record["protocol_trial_index_json"], case=case)
    assert snapshot.trial_index_schema_version == 4


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_protocol_hash_changes_between_v5_and_v6(case):
    assert len({CASES[case][version]["protocol_semantic_hash"] for version in VERSIONS}) == 4


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("semantic_version,trial_version", [("v6_120", "v5_120"), ("v5_120", "v6_120")])
def test_mixed_semantic_and_trial_index_versions_are_refused(case, semantic_version, trial_version):
    with pytest.raises(ProtocolSemanticContractError, match="trial_index_json.schema_version"):
        _validate(
            CASES[case][semantic_version]["canonical_json"],
            CASES[case][trial_version]["protocol_trial_index_json"],
            case=case,
        )


def test_v6_is_refused_under_the_legacy_binding():
    record = CASES["batman_composable"]["v6_120"]
    with pytest.raises(ProtocolSemanticContractError, match="normalization_policy must be .citrus.protocol.semantic.v1."):
        _validate(record["canonical_json"], record["protocol_trial_index_json"], convention=LEGACY_CANVAS_ANGLE)


def test_an_unknown_policy_is_refused():
    semantic = json.loads(CASES["batman_composable"]["v6_120"]["canonical_json"])
    semantic["normalization_policy"] = "citrus.protocol.semantic.v7"
    with pytest.raises(ProtocolSemanticContractError, match="normalization_policy must be .citrus.protocol.semantic.v2."):
        _validate(_json(semantic), CASES["batman_composable"]["v6_120"]["protocol_trial_index_json"])


@pytest.mark.parametrize("version", ["v6_120", "v6_240"])
@pytest.mark.parametrize("case", ADMITTED_ROLE_CASES)
def test_role_cases_are_admitted(case, version):
    entry = ROLE_CASES[case][version]
    snapshot = _validate(entry["canonical_json"], entry["protocol_trial_index_json"])
    roles = [c["role"] for c in _chaser_step(snapshot.trial_index_payload["steps"])["features"]["chasers"]]
    assert roles == ["test", "test", "test", "sham"]


@pytest.mark.parametrize("version", ["v6_120", "v6_240"])
def test_a_missing_role_hashes_as_test(version):
    a, b = (ROLE_CASES[case][version] for case in ADMITTED_ROLE_CASES)
    assert a["canonical_json"] == b["canonical_json"]
    assert a["protocol_semantic_hash"] == b["protocol_semantic_hash"]
    assert ROLE_CASES["admitted_test_and_sham"]["v6_120"]["protocol_semantic_hash"] == SHAM_HASH_120


@pytest.mark.parametrize("case", sorted(set(ROLE_CASES) - set(ADMITTED_ROLE_CASES)))
def test_citrus_refused_role_cases_produce_no_identity(case):
    record = ROLE_CASES[case]
    assert record["expected"] == "refused"
    assert record["case"] == "refused_unknown_role" or record["refusal_code"] in CITRUS_RULES
    for version in ("v6_120", "v6_240"):
        assert record[version]["exit"] == 1
        assert "protocol_semantic_hash" not in record[version]


def test_the_role_absent_trial_index_is_refused():
    refused = FIXTURE["intake_refused_cases"][0]
    assert refused["case"] == "refused_trial_index_v5_role_absent"
    assert [e.split(" ")[0] for e in chaser_role_intake_errors(_trial_object(refused["protocol_trial_index_json"]))] == [
        "R-role"
    ]
    with pytest.raises(ProtocolSemanticContractError, match="protocol_chaser_role_invalid"):
        _validate(refused["canonical_json"], refused["protocol_trial_index_json"])


def _element(features, k):
    return features["appearance"]["elements"][f"chaser_{k}"]


def _features_mutations():
    """agent-contracts ``check_intake_mutations``: each changes one field of the
    admitted test+sham trial index and fails exactly one rule."""

    _, trial = _sham()
    chasers = _chaser_step(trial["steps"])["features"]["chasers"]
    sham = next(c["chaser_index"] for c in chasers if c["role"] == "sham")
    test = next(c["chaser_index"] for c in chasers if c["role"] == "test")

    def recolor_sham(f):
        rgba = _element(f, sham)["resolved_color"]["rgba8"]
        rgba[0] = (rgba[0] + 1) % 256

    def test_to_background(f):
        _element(f, test)["resolved_color"]["rgba8"] = list(
            f["appearance"]["elements"]["background"]["resolved_color"]["rgba8"]
        )

    return [
        ("R-role", "sham_role_absent", lambda f: f["chasers"][sham].pop("role")),
        ("R-role", "sham_role_control", lambda f: f["chasers"][sham].__setitem__("role", "control")),
        ("S1", "sham_recolored", recolor_sham),
        ("S2", "edge_falloff", lambda f: f.__setitem__("background_edge_falloff", {"profile": "edge_minimum_floor_half_v1"})),
        ("S3", "sham_not_passive", lambda f: f["chasers"][sham].__setitem__("behavior_mode_id", 2)),
        ("S4", "sham_appearance", lambda f: f["chasers"][sham].__setitem__("appearance", {"profile_digest": "x"})),
        ("S4", "sham_contrast_profile", lambda f: _element(f, sham).__setitem__("contrast_profile_digest", "x")),
        ("S5", "proximity_enabled", lambda f: f.__setitem__("proximity_feedback", {"enabled": True})),
        ("S5", "proximity_absent", lambda f: f.pop("proximity_feedback")),
        ("S6", "danger_zone_drawn", lambda f: f.__setitem__("danger_zone", {"enabled": True, "draw": True})),
        ("S6", "danger_zone_element", lambda f: f["appearance"]["elements"].__setitem__("danger_zone", {})),
        ("T1", "test_equals_background", test_to_background),
    ]


MUTATIONS = _features_mutations()


@pytest.mark.parametrize("rule,label,mutate", MUTATIONS, ids=[m[1] for m in MUTATIONS])
def test_each_intake_rule_refuses_its_own_mutation_only(rule, label, mutate):
    semantic_json, trial = _sham()
    mutate(_chaser_step(trial["steps"])["features"])
    assert [e.split(" ")[0] for e in chaser_role_intake_errors(trial)] == [rule]
    with pytest.raises(ProtocolSemanticContractError, match=CHASER_ROLE_INTAKE_CODES[rule]):
        _validate(semantic_json, trial)


def test_twelve_mutations_cover_every_rule():
    assert len(MUTATIONS) == 12
    assert {rule for rule, _, _ in MUTATIONS} == set(CHASER_ROLE_INTAKE_CODES)
    assert {rule for rule, _, _ in MUTATIONS} >= set(CITRUS_RULES.values())


def test_an_enabled_but_undrawn_danger_zone_is_admitted():
    semantic_json, trial = _sham()
    _chaser_step(trial["steps"])["features"]["danger_zone"] = {"enabled": True, "draw": False}
    assert chaser_role_intake_errors(trial) == []
    _validate(semantic_json, trial)


def test_a_sham_without_displayed_codes_is_refused():
    semantic_json, trial = _sham()
    del _chaser_step(trial["steps"])["features"]["appearance"]["elements"]["background"]
    with pytest.raises(ProtocolSemanticContractError, match="protocol_chaser_role_displayed_code_missing"):
        _validate(semantic_json, trial)


def test_semantic_and_trial_roles_must_agree():
    semantic_json, trial = _sham()
    semantic = json.loads(semantic_json)
    _chaser_step(semantic["identity"]["steps"])["parameters"]["chasers"][3]["role"] = "test"
    with pytest.raises(ProtocolSemanticContractError, match="protocol_chaser_role_semantic_differs"):
        _validate(_json(semantic), trial)


def test_a_semantic_role_must_be_test_or_sham():
    semantic_json, trial = _sham()
    semantic = json.loads(semantic_json)
    _chaser_step(semantic["identity"]["steps"])["parameters"]["chasers"][0].pop("role")
    with pytest.raises(ProtocolSemanticContractError, match="protocol_chaser_role_invalid: semantic"):
        _validate(_json(semantic), trial)


def test_role_is_refused_before_v6():
    record = CASES["goodbatbadbat"]["v5_120"]
    semantic = json.loads(record["canonical_json"])
    _chaser_step(semantic["identity"]["steps"])["parameters"]["chasers"][0]["role"] = "test"
    with pytest.raises(ProtocolSemanticContractError, match="allowed only on a chaser step under recipe v6"):
        _validate(_json(semantic), record["protocol_trial_index_json"])
    trial = _trial_object(record["protocol_trial_index_json"])
    _chaser_step(trial["steps"])["features"]["chasers"][0]["role"] = "test"
    with pytest.raises(ProtocolSemanticContractError, match="allowed only on a chaser step under recipe v6"):
        _validate(record["canonical_json"], trial)


def test_trial_index_v5_is_checked_against_citrus_schema():
    semantic_json, trial = _sham()
    _chaser_step(trial["steps"])["features"]["chasers"][0].pop("radius_mm")
    with pytest.raises(ProtocolSemanticContractError, match="protocol_trial_index_v5_schema"):
        _validate(semantic_json, trial)


# --- relabel_pooling_rule.rule_v5_to_v6 ------------------------------------


@pytest.mark.parametrize("rate", [120, 240])
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_crosswalk_reproduces_every_v6_hash_at_its_own_rate(case, rate):
    v5, v6 = CASES[case][f"v5_{rate}"], CASES[case][f"v6_{rate}"]
    other = CASES[case][f"v6_{360 - rate}"]
    data = v5["canonical_json"].encode("utf-8")
    assert semantic_v5_crosswalks_to_v6(data, v5["protocol_semantic_hash"], v6["protocol_semantic_hash"])
    assert not semantic_v5_crosswalks_to_v6(data, v5["protocol_semantic_hash"], other["protocol_semantic_hash"])


def test_no_v5_recording_joins_a_group_with_a_sham():
    v5 = CASES["good_cop_bad_cop"]["v5_120"]
    assert not semantic_v5_crosswalks_to_v6(
        v5["canonical_json"].encode("utf-8"), v5["protocol_semantic_hash"], SHAM_HASH_120
    )


def test_the_crosswalk_uses_only_recorded_canonical_v5_bytes():
    v5, v6 = CASES["goodbatbadbat"]["v5_120"], CASES["goodbatbadbat"]["v6_120"]
    data = v5["canonical_json"].encode("utf-8")
    assert not semantic_v5_crosswalks_to_v6(data, _hash("x"), v6["protocol_semantic_hash"])
    spaced = json.dumps(json.loads(data), sort_keys=True).encode("utf-8")
    assert not semantic_v5_crosswalks_to_v6(spaced, "sha256:" + hashlib.sha256(spaced).hexdigest(), v6["protocol_semantic_hash"])
    v6_data = v6["canonical_json"].encode("utf-8")
    assert not semantic_v5_crosswalks_to_v6(v6_data, v6["protocol_semantic_hash"], v6["protocol_semantic_hash"])
    with pytest.raises(TypeError):
        semantic_v5_crosswalks_to_v6(v5["canonical_json"], v5["protocol_semantic_hash"], v6["protocol_semantic_hash"])


# --- /geometry/calibration/z_eff --------------------------------------------


def test_v6_requires_the_z_eff_group(tmp_path):
    with h5py.File(tmp_path / "v6.h5", "w") as h5:
        with pytest.raises(UnifiedH5ContractError, match="z_eff_group_missing"):
            validate_z_eff(h5, V6)
