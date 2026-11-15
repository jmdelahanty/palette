"""Protocol semantic recipe v5 / trial index v4 (pairing revision 4, Citrus #114).

v5 is v4 with, on every chaser step, ``z_eff`` removed and
``z_eff_source = "arena_media_stack"`` hashed; v5 is the only policy paired
with trial index v4. Citrus computes z_eff per arena at START and records it
in the attrs-only ``/geometry/calibration/z_eff`` group (``validate_z_eff``).
The cases are Citrus's ``protocol_identity_dump`` output, vendored byte for
byte from agent-contracts #124.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files

import h5py
import numpy as np
import pytest

from fisheye.shared.heading_convention import LEGACY_CANVAS_ANGLE, STIMULUS_ANGLE_CONVENTION
from fisheye.shared.protocol_semantic_contract import (
    PROTOCOL_SEMANTIC_RECIPE_V4_NORMALIZATION_POLICY,
    PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY,
    ProtocolSemanticContractError,
    semantic_v3_relabels_to_v4,
    validate_protocol_semantic_snapshot,
)
from fisheye.shared.unified_h5.common import UnifiedH5ContractError
from fisheye.shared.unified_h5.rig_provenance import GROUP as RIG_PIN
from fisheye.shared.unified_h5.z_eff import FIXED_STRINGS, GROUP, validate_z_eff

CONTRACTS = resource_files("fisheye.shared").joinpath("contracts")
PINNED = {
    # agent-contracts PR #124 merge e35dd7a, citrus-palette/angles/
    "citrus_palette_semantic_policy_pairing_v4.json": "403a07615481cea8f80108b66cab0b5b04cf1772867aadf8cfbfec66dca4cd04",
    "citrus_palette_semantic_hash_v5_cases.json": "4dbcdcfbb717daacb1286704a8bdea86da7501b7cf81e2d4f7a4ae3353bd9f79",
}


def _load(name: str) -> dict:
    data = CONTRACTS.joinpath(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED[name]
    return json.loads(data)


PAIRING = _load("citrus_palette_semantic_policy_pairing_v4.json")
CASES = {case["case"]: case for case in _load("citrus_palette_semantic_hash_v5_cases.json")["cases"]}
VERSIONS = ("v4_120", "v4_240", "v5_120", "v5_240")
CHASER_CASES = sorted(
    case
    for case, record in CASES.items()
    if any(s["stimulus_mode_id"] == 12 for s in json.loads(record["v5_120"]["canonical_json"])["identity"]["steps"])
)
# Palette has always refused a protocol with no steps; Citrus dumps one anyway.
EMPTY_CASE = "synthetic_empty"
ADMITTED_CASES = sorted(set(CASES) - {EMPTY_CASE})
HEADING_SPLIT_CASES = {"demo_chaser_then_phototaxis_seed1", "phototaxis_gray035_1h_then_60x10s_seed1"}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate(semantic_json: str, trial_index: dict, *, convention: str = STIMULUS_ANGLE_CONVENTION, case: str = ""):
    semantic_hash = _hash(semantic_json)
    trial_json = _json({**trial_index, "protocol_semantic_hash": semantic_hash})
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


def _semantic(case: str, version: str) -> dict:
    return json.loads(CASES[case][version]["canonical_json"])


def _trial(case: str, version: str) -> dict:
    return json.loads(json.dumps(CASES[case][version]["protocol_trial_index_json"]))


def _chaser_step(steps: list, key: str, value) -> dict:
    return next(s for s in steps if s.get(key) == value)


def test_pairing_revision_4_pairs_v5_with_trial_index_v4_only():
    assert PAIRING["revision"] == 4
    angle = [p for p in PAIRING["pairings"] if p["stimulus_angle_convention"] == STIMULUS_ANGLE_CONVENTION]
    assert [p["semantic_normalization_policies"] for p in angle] == [
        ["citrus.protocol.semantic.v2", "citrus.protocol.semantic.v3", PROTOCOL_SEMANTIC_RECIPE_V4_NORMALIZATION_POLICY],
        [PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY],
    ]
    assert PAIRING["relabel_pooling_rule"]["rule_v4_to_v5"].startswith("none")
    assert PAIRING["semantic_v5"]["trial_index"]["schema_version"] == 4
    assert set(PAIRING["h5_z_eff"]["attributes"]) == set(FIXED_STRINGS) | {
        "z_eff_mm", "geometric_eye_to_image_mm", "water_under_dish_mm", "holder_top_above_image_plane_mm",
        "seat_to_floor_top_mm", "dish_floor_above_image_plane_mm", "rig_id", "rig_file_path", "rig_file_sha256",
        "rig_geometry_revision", "tank_design_id", "tank_design_file_path", "tank_design_file_sha256",
        "dish_base_thickness_mm", "refractive_index_acrylic", "refractive_index_water", "eye_height_above_floor_mm",
        "arena_id",
    }
    for name, value in FIXED_STRINGS.items():
        assert f'"{value}"' in PAIRING["h5_z_eff"]["attributes"][name]


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_case_hashes_are_over_the_exact_bytes(case, version):
    record = CASES[case][version]
    assert _hash(record["canonical_json"]) == record["protocol_semantic_hash"]


@pytest.mark.parametrize("version", ["v5_120", "v5_240"])
@pytest.mark.parametrize("case", ADMITTED_CASES)
def test_v5_is_admitted_with_citrus_trial_index_v4(case, version):
    record = CASES[case][version]
    snapshot = _validate(record["canonical_json"], record["protocol_trial_index_json"], case=case)
    assert snapshot.semantic_hash == record["protocol_semantic_hash"]
    assert snapshot.semantic_payload["normalization_policy"] == PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY
    assert snapshot.trial_index_schema_version == 4
    assert snapshot.stimulus_render_rate_hz == int(version[-3:])


@pytest.mark.parametrize("version", ["v4_120", "v4_240"])
@pytest.mark.parametrize("case", ADMITTED_CASES)
def test_v4_cases_still_pass_with_trial_index_v3(case, version):
    record = CASES[case][version]
    snapshot = _validate(record["canonical_json"], record["protocol_trial_index_json"], case=case)
    assert snapshot.trial_index_schema_version == 3


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_protocol_hash_changes_between_v4_and_v5(case):
    hashes = {CASES[case][version]["protocol_semantic_hash"] for version in VERSIONS}
    assert len(hashes) == 4


@pytest.mark.parametrize("case", ADMITTED_CASES)
@pytest.mark.parametrize("semantic_version,trial_version", [("v5_120", "v4_120"), ("v4_120", "v5_120")])
def test_mixed_semantic_and_trial_index_versions_are_refused(case, semantic_version, trial_version):
    with pytest.raises(ProtocolSemanticContractError, match="trial_index_json.schema_version"):
        _validate(CASES[case][semantic_version]["canonical_json"], _trial(case, trial_version), case=case)


@pytest.mark.parametrize("version", ["v5_120", "v5_240"])
def test_the_empty_protocol_is_refused(version):
    record = CASES[EMPTY_CASE][version]
    with pytest.raises(ProtocolSemanticContractError, match="at least one step"):
        _validate(record["canonical_json"], record["protocol_trial_index_json"])


def test_v5_is_refused_under_the_legacy_binding():
    record = CASES["solid_gray016_1h"]["v5_240"]
    with pytest.raises(ProtocolSemanticContractError, match="normalization_policy must be .citrus.protocol.semantic.v1."):
        _validate(record["canonical_json"], record["protocol_trial_index_json"], convention=LEGACY_CANVAS_ANGLE)


def test_v5_requires_the_render_rate():
    semantic = _semantic("solid_gray016_1h", "v5_120")
    semantic["identity"].pop("stimulus_render_rate_hz")
    with pytest.raises(ProtocolSemanticContractError, match="stimulus_render_rate_hz must be one JSON integer"):
        _validate(_json(semantic), _trial("solid_gray016_1h", "v5_120"))


def _v5_chaser_mutation(case, mutate_parameters=None, mutate_target=None):
    semantic, trial = _semantic(case, "v5_240"), _trial(case, "v5_240")
    if mutate_parameters:
        mutate_parameters(_chaser_step(semantic["identity"]["steps"], "stimulus_mode_id", 12)["parameters"])
    if mutate_target:
        mutate_target(_chaser_step(trial["steps"], "stimulus_mode", "CHASER")["features"]["target"])
    return _json(semantic), trial


@pytest.mark.parametrize("case", CHASER_CASES)
@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.update(z_eff={"scale": "1e-3", "unit": "mm", "value": 14942}), "z_eff must be absent"),
        (lambda p: p.update(z_eff=None), "z_eff must be absent"),
        (lambda p: p.pop("z_eff_source"), "z_eff_source must be"),
        (lambda p: p.update(z_eff_source="authored"), "z_eff_source must be"),
        (lambda p: p.pop("target_radius"), "target_radius must be"),
    ],
    ids=["authored_z_eff", "null_z_eff", "no_source", "wrong_source", "v3_check_kept"],
)
def test_malformed_v5_chaser_parameters_are_refused(case, mutate, match):
    semantic_json, trial = _v5_chaser_mutation(case, mutate_parameters=mutate)
    with pytest.raises(ProtocolSemanticContractError, match=match):
        _validate(semantic_json, trial, case=case)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda t: t.update(z_eff_mm=14.942), "without z_eff_mm under trial index v4"),
        (lambda t: t.pop("z_eff_source"), "target.z_eff_source must be"),
        (lambda t: t.update(z_eff_source="authored"), "target.z_eff_source must be"),
    ],
    ids=["authored_z_eff_mm", "no_source", "wrong_source"],
)
def test_malformed_trial_index_v4_chaser_target_is_refused(mutate, match):
    semantic_json, trial = _v5_chaser_mutation("goodbatbadbat", mutate_target=mutate)
    with pytest.raises(ProtocolSemanticContractError, match=match):
        _validate(semantic_json, trial)


def test_z_eff_source_is_refused_before_v5():
    semantic = _semantic("goodbatbadbat", "v4_240")
    _chaser_step(semantic["identity"]["steps"], "stimulus_mode_id", 12)["parameters"]["z_eff_source"] = "arena_media_stack"
    with pytest.raises(ProtocolSemanticContractError, match="z_eff_source is allowed only under recipe v5"):
        _validate(_json(semantic), _trial("goodbatbadbat", "v4_240"))
    trial = _trial("goodbatbadbat", "v4_240")
    _chaser_step(trial["steps"], "stimulus_mode", "CHASER")["features"]["target"]["z_eff_source"] = "arena_media_stack"
    with pytest.raises(ProtocolSemanticContractError, match="allowed only under trial index v4"):
        _validate(CASES["goodbatbadbat"]["v4_240"]["canonical_json"], trial)


@pytest.mark.parametrize("case", sorted(CASES))
def test_no_relabel_reaches_v5(case):
    # The revision-3 rule ends at v4; a v4 hash never relabels into a v5 group.
    record = CASES[case]
    v4 = record["v4_120"]["canonical_json"].encode("utf-8")
    assert not semantic_v3_relabels_to_v4(
        v4, record["v4_120"]["protocol_semantic_hash"], record["v5_120"]["protocol_semantic_hash"]
    )


# --- /geometry/calibration/z_eff -------------------------------------------

Z = float(np.float32(9.213892936706543))
RIG_SHA = "sha256:" + "ab" * 32


def _z_eff_attrs() -> dict:
    holder, seat, base = 8.13, 0.0, 3.0
    floor = holder - seat
    return {
        **FIXED_STRINGS,
        "z_eff_mm": Z,
        "geometric_eye_to_image_mm": floor - base + 1.4,
        "water_under_dish_mm": floor - base,
        "holder_top_above_image_plane_mm": holder,
        "seat_to_floor_top_mm": seat,
        "dish_floor_above_image_plane_mm": floor,
        "dish_base_thickness_mm": base,
        "refractive_index_acrylic": 1.49,
        "refractive_index_water": 1.333,
        "eye_height_above_floor_mm": 1.4,
        "rig_id": "omnifin0",
        "rig_file_path": "rigs/omnifin0/rig.yml",
        "rig_file_sha256": RIG_SHA,
        "tank_design_id": "palm2",
        "tank_design_file_path": "tank_designs/palm2.json",
        "tank_design_file_sha256": "sha256:" + "cd" * 32,
        "arena_id": "arena_3",
    }


def _write_attr(attrs, name, value):
    if isinstance(value, str):
        attrs.create(name, value, dtype=h5py.string_dtype("utf-8"))
    elif isinstance(value, float):
        attrs.create(name, value, dtype="<f8")
    else:
        attrs.create(name, value, dtype="<i4")


def _z_eff_h5(tmp_path, *, attrs=None, config=None, chaser_rows=None, pin=None, revision=15):
    handle = h5py.File(tmp_path / "z_eff.h5", "w")
    handle.create_group("/metadata/session")
    group = handle.create_group(GROUP)
    for name, value in (_z_eff_attrs() if attrs is None else attrs).items():
        _write_attr(group.attrs, name, value)
    if revision is not None:
        _write_attr(group.attrs, "rig_geometry_revision", revision)
    config = config or {
        "config_name": "arena_3",
        "calculated_z_eff_mm": Z,
        "dish_config": {"optics": {"z_eff_current_mm": Z, "z_eff_larvae_mm": Z, "z_eff_adult_mm": 1.0}},
    }
    handle.create_dataset(
        "/geometry/calibration/arena_config_json", data=_json(config), dtype=h5py.string_dtype("utf-8")
    )
    if chaser_rows is not None:
        handle.create_dataset(
            "/components/chaser/states",
            data=np.array([(i, value) for i, value in enumerate(chaser_rows)], dtype=[("i", "<i4"), ("z_eff_mm", "<f4")]),
        )
    pin_group = handle.create_group(RIG_PIN)
    for name, value in (pin or {"rig_id": "omnifin0", "rig_file_sha256": RIG_SHA, "rig_geometry_revision": 15}).items():
        _write_attr(pin_group.attrs, name, value)
    return handle


V5 = PROTOCOL_SEMANTIC_RECIPE_V5_NORMALIZATION_POLICY


def test_a_consistent_z_eff_group_is_admitted(tmp_path):
    with _z_eff_h5(tmp_path, chaser_rows=[Z, Z]) as h5:
        assert validate_z_eff(h5, V5) == Z


def test_rig_geometry_revision_is_optional(tmp_path):
    with _z_eff_h5(tmp_path, revision=None) as h5:
        assert validate_z_eff(h5, V5) == Z


def test_v5_without_the_group_is_refused(tmp_path):
    with _z_eff_h5(tmp_path) as h5:
        del h5[GROUP]
        with pytest.raises(UnifiedH5ContractError, match="z_eff_group_missing"):
            validate_z_eff(h5, V5)


@pytest.mark.parametrize("policy", ["citrus.protocol.semantic.v2", PROTOCOL_SEMANTIC_RECIPE_V4_NORMALIZATION_POLICY])
def test_the_group_is_refused_before_v5_and_ignored_when_absent(tmp_path, policy):
    with _z_eff_h5(tmp_path) as h5:
        with pytest.raises(UnifiedH5ContractError, match="z_eff_group_outside_recipe_v5"):
            validate_z_eff(h5, policy)
        del h5[GROUP]
        assert validate_z_eff(h5, policy) is None


def _without(name):
    attrs = _z_eff_attrs()
    attrs.pop(name)
    return attrs


@pytest.mark.parametrize(
    "attrs,code",
    [
        (_without("arena_id"), "z_eff_attribute_set_not_closed"),
        ({**_z_eff_attrs(), "z_eff_adult_mm": 1.0}, "z_eff_attribute_set_not_closed"),
        ({**_z_eff_attrs(), "formula_id": "citrus.z_eff.other.v1"}, "z_eff_fixed_string:formula_id"),
        (
            {**_z_eff_attrs(), "refractive_index_water_source": "authored"},
            "z_eff_fixed_string:refractive_index_water_source",
        ),
        ({**_z_eff_attrs(), "rig_file_sha256": "ab" * 32}, "z_eff_sha256_form:rig_file_sha256"),
        ({**_z_eff_attrs(), "z_eff_mm": 9.2138929}, "z_eff_value_not_positive_float32"),
        ({**_z_eff_attrs(), "dish_floor_above_image_plane_mm": 8.0}, "z_eff_dish_floor_not_derived"),
        ({**_z_eff_attrs(), "water_under_dish_mm": 5.0}, "z_eff_water_under_dish_not_derived"),
        ({**_z_eff_attrs(), "arena_id": "arena_4"}, "z_eff_arena_id_mismatch"),
        ({**_z_eff_attrs(), "rig_id": "omnifin1"}, "z_eff_rig_pin_mismatch:rig_id"),
    ],
    ids=[
        "missing_attr", "extra_attr", "formula", "source", "sha_form", "not_float32",
        "floor", "water", "arena_id", "rig_id",
    ],
)
def test_inconsistent_z_eff_attributes_are_refused(tmp_path, attrs, code):
    with _z_eff_h5(tmp_path, attrs=attrs) as h5:
        with pytest.raises(UnifiedH5ContractError, match=code):
            validate_z_eff(h5, V5)


def test_a_float32_number_attribute_is_refused(tmp_path):
    with _z_eff_h5(tmp_path) as h5:
        h5[GROUP].attrs.create("dish_base_thickness_mm", 3.0, dtype="<f4")
        with pytest.raises(UnifiedH5ContractError, match="z_eff_attribute_type:dish_base_thickness_mm"):
            validate_z_eff(h5, V5)


def test_a_child_dataset_is_refused(tmp_path):
    with _z_eff_h5(tmp_path) as h5:
        h5[GROUP].create_dataset("values", data=[Z])
        with pytest.raises(UnifiedH5ContractError, match="z_eff_group_not_attributes_only"):
            validate_z_eff(h5, V5)


@pytest.mark.parametrize("key", ["calculated_z_eff_mm", "z_eff_current_mm", "z_eff_larvae_mm"])
def test_arena_config_must_carry_the_same_value(tmp_path, key):
    config = {
        "config_name": "arena_3",
        "calculated_z_eff_mm": Z,
        "dish_config": {"optics": {"z_eff_current_mm": Z, "z_eff_larvae_mm": Z}},
    }
    if key == "calculated_z_eff_mm":
        config[key] = 14.942
    else:
        config["dish_config"]["optics"][key] = 14.942
    with _z_eff_h5(tmp_path, config=config) as h5:
        with pytest.raises(UnifiedH5ContractError, match="z_eff_arena_config_mismatch"):
            validate_z_eff(h5, V5)


def test_chaser_rows_must_carry_the_same_value(tmp_path):
    with _z_eff_h5(tmp_path, chaser_rows=[Z, 14.942]) as h5:
        with pytest.raises(UnifiedH5ContractError, match="z_eff_chaser_rows_mismatch"):
            validate_z_eff(h5, V5)


def test_rig_pin_must_exist_and_agree(tmp_path):
    with _z_eff_h5(tmp_path, pin={"rig_id": "omnifin0", "rig_file_sha256": RIG_SHA, "rig_geometry_revision": 14}) as h5:
        with pytest.raises(UnifiedH5ContractError, match="z_eff_rig_pin_mismatch:rig_geometry_revision"):
            validate_z_eff(h5, V5)
        del h5[RIG_PIN]
        with pytest.raises(UnifiedH5ContractError, match="z_eff_rig_pin_missing"):
            validate_z_eff(h5, V5)


def test_session_arena_id_must_agree_and_the_session_group_must_exist(tmp_path):
    with _z_eff_h5(tmp_path) as h5:
        h5["/metadata/session"].attrs.create("arena_id", "arena_4", dtype=h5py.string_dtype("utf-8"))
        with pytest.raises(UnifiedH5ContractError, match="z_eff_session_arena_id_mismatch"):
            validate_z_eff(h5, V5)
        h5["/metadata/session"].attrs.create("arena_id", "arena_3", dtype=h5py.string_dtype("utf-8"))
        assert validate_z_eff(h5, V5) == Z
        del h5["/metadata/session"]
        with pytest.raises(UnifiedH5ContractError, match="z_eff_session_missing"):
            validate_z_eff(h5, V5)
