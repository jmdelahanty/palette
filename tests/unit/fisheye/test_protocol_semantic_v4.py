"""Protocol semantic recipe v4 (pairing revision 3, Citrus #102).

v4 is v3 plus one session-level ``identity.stimulus_render_rate_hz``, a JSON
integer present on every v4 identity, because the chaser draws per update tick
and one hash must never name a 120 Hz and a 240 Hz stimulus. The cases are
Citrus's ``protocol_identity_dump`` output, vendored byte for byte, with the
trial indexes Citrus built for each rate. The H5 rate attributes must agree
with the identity (``validate_render_rate``).
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
    ProtocolSemanticContractError,
    semantic_v3_relabels_to_v4,
    validate_protocol_semantic_snapshot,
)
from fisheye.shared.unified_h5.common import UnifiedH5ContractError
from fisheye.shared.unified_h5.protocol import RENDER_RATE_SOURCE, validate_render_rate

CONTRACTS = resource_files("fisheye.shared").joinpath("contracts")
PINNED = {
    # Citrus PR #102 head 010e066, docs/protocol_semantic_v4/
    "citrus_palette_semantic_policy_pairing_v3.json": "fa61f869654b8f6c7877a9de6632af94c31d9e32e3c69f4e94971695b1d02bd8",
    "citrus_palette_semantic_hash_v4_cases.json": "f62e6f89fa5ac8cf3ef558fd4964c4bbea9efa71dbf2e32aef3272eac18a3d05",
}


def _load(name: str) -> dict:
    data = CONTRACTS.joinpath(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED[name]
    return json.loads(data)


PAIRING = _load("citrus_palette_semantic_policy_pairing_v3.json")
CASES = {case["case"]: case for case in _load("citrus_palette_semantic_hash_v4_cases.json")["cases"]}
RATES = {"v4_120": 120, "v4_240": 240}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# These cases carry a reactive heading-split phototaxis step, which only a
# revision 6 binding admits.
HEADING_SPLIT_CASES = {"demo_chaser_then_phototaxis_seed1", "phototaxis_gray035_1h_then_60x10s_seed1"}


def _validate(
    semantic_json: str,
    trial_index: dict,
    *,
    convention: str = STIMULUS_ANGLE_CONVENTION,
    admits_heading_split: bool = False,
):
    semantic_hash = _hash(semantic_json)
    trial_index = {**trial_index, "protocol_semantic_hash": semantic_hash}
    trial_json = _json(trial_index)
    return validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
        trial_index_hash=_hash(trial_json),
        snapshot_schema_version=2,
        snapshot_policy_id="citrus.protocol.snapshot.v2",
        stimulus_angle_convention=convention,
        admits_heading_split=admits_heading_split,
    )


def _mutated(case: str, version: str, mutate) -> str:
    semantic = json.loads(CASES[case][version]["canonical_json"])
    mutate(semantic)
    return _json(semantic)


def test_pairing_revision_admits_v2_v3_and_v4_under_angle_v1_only():
    pairings = {p["stimulus_angle_convention"]: p for p in PAIRING["pairings"]}
    assert pairings[STIMULUS_ANGLE_CONVENTION]["semantic_normalization_policies"] == [
        "citrus.protocol.semantic.v2",
        "citrus.protocol.semantic.v3",
        PROTOCOL_SEMANTIC_RECIPE_V4_NORMALIZATION_POLICY,
    ]
    assert pairings[LEGACY_CANVAS_ANGLE]["semantic_normalization_policies"] == ["citrus.protocol.semantic.v1"]
    assert PAIRING["semantic_v4"]["newly_hashed"][0]["path"] == "identity.stimulus_render_rate_hz"


@pytest.mark.parametrize("version", ["v3", "v4_120", "v4_240"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_case_hashes_are_over_the_exact_bytes(case, version):
    record = CASES[case][version]
    assert _hash(record["canonical_json"]) == record["protocol_semantic_hash"]


@pytest.mark.parametrize("version", ["v4_120", "v4_240"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_v4_is_admitted_with_citrus_trial_index(case, version):
    record = CASES[case][version]
    assert record["protocol_trial_index_json"]["protocol_semantic_hash"] == record["protocol_semantic_hash"]
    snapshot = _validate(
        record["canonical_json"],
        record["protocol_trial_index_json"],
        admits_heading_split=case in HEADING_SPLIT_CASES,
    )
    assert snapshot.semantic_hash == record["protocol_semantic_hash"]
    assert snapshot.semantic_payload["normalization_policy"] == PROTOCOL_SEMANTIC_RECIPE_V4_NORMALIZATION_POLICY
    assert snapshot.stimulus_render_rate_hz == RATES[version]


@pytest.mark.parametrize("case", sorted(CASES))
def test_one_protocol_has_a_different_hash_at_each_rate(case):
    hashes = {CASES[case][version]["protocol_semantic_hash"] for version in ("v3", "v4_120", "v4_240")}
    assert len(hashes) == 3


@pytest.mark.parametrize("case", sorted(CASES))
def test_v3_identity_reports_no_render_rate(case):
    record = CASES[case]["v4_120"]
    trial = record["protocol_trial_index_json"]
    snapshot = _validate(
        CASES[case]["v3"]["canonical_json"], trial, admits_heading_split=case in HEADING_SPLIT_CASES
    )
    assert snapshot.stimulus_render_rate_hz is None


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda s: s["identity"].pop("stimulus_render_rate_hz"), "stimulus_render_rate_hz must be one JSON integer"),
        (lambda s: s["identity"].update(stimulus_render_rate_hz=120.0), "must be one JSON integer"),
        (lambda s: s["identity"].update(stimulus_render_rate_hz="120"), "must be one JSON integer"),
        (lambda s: s["identity"].update(stimulus_render_rate_hz=True), "must be one JSON integer"),
        (lambda s: s["identity"].update(stimulus_render_rate_hz=0), "must be one JSON integer"),
        (lambda s: s["identity"].update(stimulus_render_rate_hz=144), "must be one JSON integer"),
    ],
    ids=["missing", "float", "string", "bool", "zero", "unadmitted_rate"],
)
def test_malformed_v4_render_rate_is_refused(mutate, match):
    trial = CASES["solid_gray016_1h"]["v4_120"]["protocol_trial_index_json"]
    with pytest.raises(ProtocolSemanticContractError, match=match):
        _validate(_mutated("solid_gray016_1h", "v4_120", mutate), trial)


def test_a_render_rate_under_v3_is_refused():
    trial = CASES["solid_gray016_1h"]["v4_120"]["protocol_trial_index_json"]
    text = _mutated("solid_gray016_1h", "v3", lambda s: s["identity"].update(stimulus_render_rate_hz=120))
    with pytest.raises(ProtocolSemanticContractError, match="allowed only under recipe v4"):
        _validate(text, trial)


def test_v4_is_refused_under_the_legacy_binding():
    record = CASES["solid_gray016_1h"]["v4_240"]
    with pytest.raises(ProtocolSemanticContractError, match="normalization_policy must be .citrus.protocol.semantic.v1."):
        _validate(record["canonical_json"], record["protocol_trial_index_json"], convention=LEGACY_CANVAS_ANGLE)


def test_an_unknown_policy_is_refused():
    trial = CASES["solid_gray016_1h"]["v4_120"]["protocol_trial_index_json"]
    text = _mutated("solid_gray016_1h", "v4_120", lambda s: s.update(normalization_policy="citrus.protocol.semantic.v5"))
    with pytest.raises(ProtocolSemanticContractError, match="normalization_policy must be"):
        _validate(text, trial)


def test_v4_chaser_steps_keep_the_v3_checks():
    def drop_z_eff(semantic):
        step = next(s for s in semantic["identity"]["steps"] if s["stimulus_mode_id"] == 12)
        step["parameters"].pop("z_eff")

    trial = CASES["goodbatbadbat"]["v4_240"]["protocol_trial_index_json"]
    with pytest.raises(ProtocolSemanticContractError, match="z_eff is required"):
        _validate(_mutated("goodbatbadbat", "v4_240", drop_z_eff), trial)


@pytest.mark.parametrize("case", sorted(CASES))
def test_relabel_rule_pools_v3_into_v4_at_120_only(case):
    record = CASES[case]
    v3 = record["v3"]["canonical_json"].encode("utf-8")
    assert semantic_v3_relabels_to_v4(v3, record["v3"]["protocol_semantic_hash"], record["v4_120"]["protocol_semantic_hash"])
    assert not semantic_v3_relabels_to_v4(v3, record["v3"]["protocol_semantic_hash"], record["v4_240"]["protocol_semantic_hash"])
    assert record["relabel_v3_to_v4_at_120_matches"] is True


def test_live_solid_gray_hash_is_the_v3_case():
    # 2026-11-06 14_17_17 recorded this v3 hash; its v4 at 120 is a separate group.
    record = CASES["solid_gray016_1h"]
    assert record["v3"]["protocol_semantic_hash"].startswith("sha256:032a71cd")
    assert record["v4_120"]["protocol_semantic_hash"].startswith("sha256:a50ddd9c")
    assert record["v4_240"]["protocol_semantic_hash"].startswith("sha256:19ffd9f1")


def test_relabel_rule_refuses_tampered_bytes_and_a_repeated_member():
    record = CASES["solid_gray016_1h"]
    v3_hash, v4_hash = record["v3"]["protocol_semantic_hash"], record["v4_120"]["protocol_semantic_hash"]
    assert not semantic_v3_relabels_to_v4(record["v3"]["canonical_json"].encode() + b" ", v3_hash, v4_hash)
    semantic = json.loads(record["v3"]["canonical_json"])
    semantic["identity"]["note"] = {"a": {}, "normalization_policy": "citrus.protocol.semantic.v3"}
    text = _json(semantic).encode("utf-8")
    assert text.count(b'},"normalization_policy":"citrus.protocol.semantic.v3"') == 2
    assert not semantic_v3_relabels_to_v4(text, _hash(text.decode()), v4_hash)


def test_relabel_rule_takes_recorded_bytes_only():
    record = CASES["solid_gray016_1h"]
    with pytest.raises(TypeError):
        semantic_v3_relabels_to_v4(
            record["v3"]["canonical_json"], record["v3"]["protocol_semantic_hash"], record["v4_120"]["protocol_semantic_hash"]
        )
    with pytest.raises(ProtocolSemanticContractError):
        semantic_v3_relabels_to_v4(b"{}", "not-a-hash", record["v4_120"]["protocol_semantic_hash"])


# --- H5 rate attributes agree with the identity -------------------------------------------


def _h5(tmp_path, *, session: dict, timing: dict | None = None):
    handle = h5py.File(tmp_path / "rates.h5", "w")
    group = handle.create_group("/metadata/session")
    for name, value in session.items():
        group.attrs[name] = value
    if timing is not None:
        display = handle.create_group("/timing/display_submissions")
        for name, value in timing.items():
            display.attrs[name] = value
    return handle


def _rates(rate: float, refresh: float = 239.96) -> dict:
    return {
        "stimulus_frame_rate_nominal_hz": np.float64(rate),
        "display_submission_rate_target_hz": np.float64(rate),
        "display_refresh_rate_reported_hz": np.float64(refresh),
    }


def _declared(declared: float) -> dict:
    return {"display_refresh_rate_declared_hz": np.float64(declared), "stimulus_render_rate_source": RENDER_RATE_SOURCE}


@pytest.mark.parametrize("rate,refresh", [(120, 240.0), (240, 240.0), (120, 120.0)])
def test_consistent_v4_rates_are_admitted(tmp_path, rate, refresh):
    with _h5(
        tmp_path,
        session={**_rates(rate, refresh - 0.04), **_declared(refresh)},
        timing={"display_submission_rate_target_hz": np.float64(rate)},
    ) as h5:
        assert validate_render_rate(h5, rate) == rate


def test_historical_120_hz_file_is_admitted(tmp_path):
    with _h5(tmp_path, session=_rates(120, 119.88), timing={"display_submission_rate_target_hz": np.float64(120)}) as h5:
        assert validate_render_rate(h5, None) == 120


def test_old_file_without_rate_attributes_is_admitted(tmp_path):
    with _h5(tmp_path, session={}) as h5:
        assert validate_render_rate(h5, None) == 120


@pytest.mark.parametrize(
    "identity_rate,session,timing,code",
    [
        (None, _rates(240), None, "native_render_rate_mismatch:stimulus_frame_rate_nominal_hz"),
        (240, _rates(120), None, "native_render_rate_mismatch:stimulus_frame_rate_nominal_hz"),
        (240, {**_rates(240), "display_submission_rate_target_hz": np.float64(120)}, None,
         "native_render_rate_mismatch:display_submission_rate_target_hz"),
        (240, _rates(240), {"display_submission_rate_target_hz": np.float64(120)},
         "native_render_rate_mismatch:display_submissions"),
        (240, {"display_submission_rate_target_hz": np.float64(240)}, None,
         "native_render_rate_attribute_missing:stimulus_frame_rate_nominal_hz"),
        (240, {**_rates(240), "stimulus_frame_rate_nominal_hz": np.float32(240)}, None,
         "native_render_rate_attribute_type:stimulus_frame_rate_nominal_hz"),
        (240, {**_rates(240), "stimulus_frame_rate_nominal_hz": np.float64("nan")}, None,
         "native_render_rate_attribute_value:stimulus_frame_rate_nominal_hz"),
    ],
    ids=["240_file_as_pre_v4", "120_file_as_v4_240", "session_target_disagrees", "timing_target_disagrees",
         "v4_missing_nominal", "float32_attribute", "nan_attribute"],
)
def test_inconsistent_rates_are_refused(tmp_path, identity_rate, session, timing, code):
    with _h5(tmp_path, session=session, timing=timing) as h5:
        with pytest.raises(UnifiedH5ContractError, match=f"^{code}$"):
            validate_render_rate(h5, identity_rate)


@pytest.mark.parametrize(
    "rate,session,code",
    [
        (120, {**_rates(120, 119.88), **_declared(119.88)}, "native_render_rate_incompatible_with_refresh"),
        (240, {**_rates(240, 120.0), **_declared(120.0)}, "native_render_rate_incompatible_with_refresh"),
        (120, {**_rates(120, 60.0), **_declared(240.0)}, "native_display_refresh_mismatch"),
        (120, {**_rates(120), "display_refresh_rate_declared_hz": np.float64(240.0)}, "native_render_rate_source"),
        (120, {**_rates(120), **_declared(240.0), "stimulus_render_rate_source": "local_config"}, "native_render_rate_source"),
        (120, {**_rates(120), "stimulus_render_rate_source": RENDER_RATE_SOURCE},
         "native_render_rate_source_without_declared_refresh"),
    ],
    ids=["refresh_119_88_on_120", "refresh_below_render", "reported_differs_from_declared", "declared_without_source",
         "unknown_source", "source_without_declared"],
)
def test_declared_refresh_must_fit_the_render_rate(tmp_path, rate, session, code):
    with _h5(tmp_path, session=session) as h5:
        with pytest.raises(UnifiedH5ContractError, match=f"^{code}$"):
            validate_render_rate(h5, rate)
