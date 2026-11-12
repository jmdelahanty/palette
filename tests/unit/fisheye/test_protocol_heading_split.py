"""Reactive heading-split trial-index steps (agent-contracts #98 revision 5)."""

from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fisheye.shared.heading_convention import STIMULUS_ANGLE_CONVENTION
from fisheye.shared.protocol_semantic_contract import (
    ProtocolSemanticContractError,
    validate_protocol_semantic_snapshot,
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _reactive(**changes: object) -> dict[str, object]:
    reactive: dict[str, object] = {
        "module": "PhototaxisHeadingSplit",
        "heading_definition_ref": "agent-contracts:citrus-palette/heading/heading_convention_v1.json",
        "camera_view_assumed": "dorsal_unmirrored",
        "side_rule_id": "projected_left_axis_v1",
        "hysteresis_frame": "camera",
        "fill_style": "two_tone",
        "minimum_keypoint_confidence": 0.5,
        "minimum_heading_displacement_degrees": 10.0,
        "maximum_held_heading_updates": 10,
        "unknown_heading_fallback": "static_fill",
        "left_side": "side_a",
    }
    reactive.update(changes)
    return reactive


def _heading_split_features(**reactive_changes: object) -> dict[str, object]:
    return {
        "fill": {
            "type": "heading_split",
            "style": "two_tone",
            "domain": "experimental_area",
            "outside_area_policy": "side_b_v1",
        },
        "reactive": _reactive(**reactive_changes),
    }


STATIC_FEATURES = {
    "fill": {"type": "two_tone", "domain": "experimental_area", "outside_area_policy": "side_b_v1"}
}


# Verbatim from Citrus c0c0607 tools/protocol_identity_dump, protocol B
# (phototaxis_heading_split_30trials_seed1.json) semantic step 1 and its trial
# index reactive block.
CITRUS_SEMANTIC_PARAMETERS = json.loads(
    '{"brightness":{"scale":"1e-6","unit":"scalar","value":1000000},'
    '"circle_radius_fraction":{"scale":"1e-6","unit":"fraction","value":500000},"fill_type_id":2,'
    '"half_side_id":0,"heading_split":{"fill_style_id":0,"left_side_id":0,"maximum_held_heading_updates":60,'
    '"minimum_heading_displacement_degrees":{"scale":"1e-6","unit":"scalar","value":5000000},'
    '"minimum_keypoint_confidence":{"scale":"1e-6","unit":"fraction","value":500000},'
    '"module":"PhototaxisHeadingSplit","unknown_heading_fallback_id":2},"quarter_id":0,"region_type_id":0,'
    '"side_a_color":{"color_space":"srgb","rgba8":[255,255,255,255]},'
    '"side_b_color":{"color_space":"srgb","rgba8":[0,0,0,255]},"type":"ProtocolPhototaxisParams"}'
)
CITRUS_REACTIVE = json.loads(
    '{"camera_view_assumed":"dorsal_unmirrored","fill_style":"two_tone",'
    '"heading_definition_ref":"agent-contracts:citrus-palette/heading/heading_convention_v1.json",'
    '"hysteresis_frame":"camera","left_side":"side_a","maximum_held_heading_updates":60,'
    '"minimum_heading_displacement_degrees":5.0,"minimum_keypoint_confidence":0.5,'
    '"module":"PhototaxisHeadingSplit","side_rule_id":"projected_left_axis_v1",'
    '"unknown_heading_fallback":"uniform_side_b"}'
)


def _semantic_parameters(reactive: object) -> dict[str, object]:
    """Citrus's semantic parameters re-derived from a (valid) reactive block."""

    parameters = json.loads(json.dumps(CITRUS_SEMANTIC_PARAMETERS))
    if not isinstance(reactive, dict):
        return parameters
    split = parameters["heading_split"]
    ids = {
        "fill_style": ("fill_style_id", ["two_tone", "gradient"]),
        "left_side": ("left_side_id", ["side_a", "side_b"]),
        "unknown_heading_fallback": ("unknown_heading_fallback_id", ["static_fill", "uniform_side_a", "uniform_side_b"]),
    }
    for key, (field, names) in ids.items():
        if reactive.get(key) in names:
            split[field] = names.index(reactive[key])
    if "maximum_held_heading_updates" in reactive:
        split["maximum_held_heading_updates"] = reactive["maximum_held_heading_updates"]
    for key in ("minimum_keypoint_confidence", "minimum_heading_displacement_degrees"):
        value = reactive.get(key)
        if type(value) in (int, float):
            split[key]["value"] = round(value * 10**6)
    if type(reactive.get("module")) is str:
        split["module"] = reactive["module"]
    return parameters


def _validate(
    features: dict[str, object],
    *,
    mode_id: int = 18,
    admits: bool = True,
    trial_version: int = 3,
    parameters: object = "auto",
):
    mode, family = {18: ("PHOTOTAXIS", "phototaxis"), 17: ("STATIC_GRID", "static_grid")}[mode_id]
    if parameters == "auto":
        reactive = features.get("reactive")
        heading_split = isinstance(features.get("fill"), dict) and features["fill"].get("type") == "heading_split"
        parameters = _semantic_parameters(reactive) if heading_split and mode_id == 18 else {}
    angle_v1 = trial_version == 3
    semantic = {
        "identity": {
            "iti_stimulus_mode_id": 99,
            "steps": [
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 30000},
                    "parameters": parameters,
                    "post_stimulus_iti": {"scale": "1e-3", "unit": "s", "value": 0},
                    "stimulus_mode_id": mode_id,
                }
            ],
        },
        "normalization_policy": "citrus.protocol.semantic.v2" if angle_v1 else "citrus.protocol.semantic.v1",
        "schema_id": "citrus.protocol.semantic",
        "schema_version": 1,
    }
    if angle_v1:
        semantic["stimulus_angle_convention"] = STIMULUS_ANGLE_CONVENTION
    semantic_json = _json(semantic)
    semantic_hash = "sha256:" + sha256(semantic_json.encode()).hexdigest()
    trial = {
        "normalization_policy": f"citrus.protocol.trial_index.v{trial_version}",
        "protocol_semantic_hash": semantic_hash,
        "schema_id": "citrus.protocol.trial_index",
        "schema_version": trial_version,
        "steps": [
            {
                "duration_s": 30.0,
                "features": features,
                "index_status": "detailed",
                "post_stimulus_iti_s": 0.0,
                "step_index": 0,
                "stimulus_family": family,
                "stimulus_mode": mode,
                "stimulus_mode_id": mode_id,
            }
        ],
    }
    if angle_v1:
        trial["stimulus_angle_convention"] = STIMULUS_ANGLE_CONVENTION
    trial_json = _json(trial)
    kwargs = {}
    if angle_v1:
        kwargs["stimulus_angle_convention"] = STIMULUS_ANGLE_CONVENTION
    return validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
        trial_index_hash="sha256:" + sha256(trial_json.encode()).hexdigest(),
        snapshot_schema_version=2,
        snapshot_policy_id="citrus.protocol.snapshot.v2",
        admits_heading_split=admits,
        **kwargs,
    )


def test_admitted_heading_split_step_validates() -> None:
    snapshot = _validate(_heading_split_features())
    assert snapshot.steps[0].stimulus_mode == "PHOTOTAXIS"


@pytest.mark.parametrize("held", [1, 10, "never"])
def test_hold_limit_is_an_integer_or_never(held: object) -> None:
    _validate(_heading_split_features(maximum_held_heading_updates=held))


def test_static_phototaxis_is_unaffected_by_the_admission_flag() -> None:
    for admits in (False, True):
        _validate(STATIC_FEATURES, admits=admits)


@pytest.mark.parametrize(
    "features",
    [
        _heading_split_features(),
        {"fill": {**STATIC_FEATURES["fill"], "style": "two_tone"}},
        {**STATIC_FEATURES, "reactive": _reactive()},
    ],
    ids=["heading_split", "style_only", "reactive_only"],
)
def test_revisions_before_6_refuse_any_reactive_declaration(features: dict[str, object]) -> None:
    with pytest.raises(ProtocolSemanticContractError, match="no admission revision before 6"):
        _validate(features, admits=False)


def _without(mapping: dict[str, object], key: str) -> dict[str, object]:
    return {k: v for k, v in mapping.items() if k != key}


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ({"fill": {**STATIC_FEATURES["fill"], "style": "two_tone"}}, "allowed only when features.fill.type is 'heading_split'"),
        ({**STATIC_FEATURES, "reactive": _reactive()}, "allowed only when features.fill.type is 'heading_split'"),
        ({"fill": _heading_split_features()["fill"]}, "requires features.fill.style and features.reactive"),
        ({**_heading_split_features(), "fill": _without(_heading_split_features()["fill"], "style")}, "requires features.fill.style"),
        ({**_heading_split_features(fill_style="gradient"), "fill": {**_heading_split_features()["fill"], "style": "gradient"}}, "reserved and not admitted"),
        ({**_heading_split_features(), "fill": {**_heading_split_features()["fill"], "style": "hatched"}}, "fill.style must be one of"),
        ({**_heading_split_features(), "fill": {**_heading_split_features()["fill"], "outside_area_policy": "masked_black_v1"}}, "fill.outside_area_policy"),
        (_heading_split_features(fill_style="gradient"), "reactive.fill_style"),
        (_heading_split_features(module="Other"), "reactive.module"),
        (_heading_split_features(camera_view_assumed="ventral"), "reactive.camera_view_assumed"),
        (_heading_split_features(side_rule_id="naive_v0"), "reactive.side_rule_id"),
        (_heading_split_features(hysteresis_frame="canvas"), "reactive.hysteresis_frame"),
        (_heading_split_features(minimum_keypoint_confidence=1.5), r"in \[0, 1\]"),
        (_heading_split_features(minimum_heading_displacement_degrees=-1.0), r"in \[0, 180\]"),
        (_heading_split_features(minimum_heading_displacement_degrees=True), "one JSON number"),
        (_heading_split_features(maximum_held_heading_updates=0), "greater than or equal to 1"),
        (_heading_split_features(maximum_held_heading_updates=True), "one exact integer"),
        (_heading_split_features(maximum_held_heading_updates="always"), "one exact integer"),
        (_heading_split_features(unknown_heading_fallback="black"), "unknown_heading_fallback must be one of"),
        (_heading_split_features(left_side="left"), "left_side must be one of"),
        ({**_heading_split_features(), "reactive": _without(_reactive(), "left_side")}, "exactly the keys"),
        ({**_heading_split_features(), "reactive": {**_reactive(), "extra": 1}}, "exactly the keys"),
        ({**_heading_split_features(), "reactive": "PhototaxisHeadingSplit"}, "must be one object"),
    ],
)
def test_admitted_heading_split_refuses_malformed_steps(features: dict[str, object], message: str) -> None:
    with pytest.raises(ProtocolSemanticContractError, match=message):
        _validate(features)


def test_heading_split_requires_phototaxis() -> None:
    with pytest.raises(ProtocolSemanticContractError, match="stimulus_mode_id for fill type heading_split"):
        _validate(_heading_split_features(), mode_id=17)


def test_heading_split_requires_trial_index_v3() -> None:
    with pytest.raises(ProtocolSemanticContractError, match="trial index schema_version"):
        _validate(_heading_split_features(), trial_version=2)


def test_citrus_bytes_validate() -> None:
    features = {**_heading_split_features(), "reactive": CITRUS_REACTIVE}
    _validate(features, parameters=CITRUS_SEMANTIC_PARAMETERS)


def _split(**changes: object) -> dict[str, object]:
    parameters = json.loads(json.dumps(CITRUS_SEMANTIC_PARAMETERS))
    for key, value in changes.items():
        if value is None:
            del parameters["heading_split"][key]
        else:
            parameters["heading_split"][key] = value
    return parameters


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({**CITRUS_SEMANTIC_PARAMETERS, "fill_type_id": 0}, "fill_type_id"),
        ({k: v for k, v in CITRUS_SEMANTIC_PARAMETERS.items() if k != "heading_split"}, "heading_split must have exactly the keys"),
        (_split(module="Other"), "heading_split.module"),
        (_split(maximum_held_heading_updates=61), "maximum_held_heading_updates"),
        (_split(maximum_held_heading_updates="never"), "maximum_held_heading_updates"),
        (_split(fill_style_id=1), "fill_style_id"),
        (_split(left_side_id=1), "left_side_id"),
        (_split(unknown_heading_fallback_id=0), "unknown_heading_fallback_id"),
        (_split(minimum_heading_displacement_degrees={"scale": "1e-6", "unit": "scalar", "value": 5000001}), "differs from the trial index"),
        (_split(minimum_keypoint_confidence={"scale": "1e-6", "unit": "scalar", "value": 500000}), "1e-6 fraction scaled integer"),
        (_split(minimum_keypoint_confidence={"scale": "1e-3", "unit": "fraction", "value": 500}), "1e-6 fraction scaled integer"),
        (_split(minimum_keypoint_confidence=0.5), "1e-6 fraction scaled integer"),
        (_split(module=None), "exactly the keys"),
        (_split(extra=1), "exactly the keys"),
    ],
)
def test_semantic_identity_must_match_the_trial_index(parameters: dict, message: str) -> None:
    features = {**_heading_split_features(), "reactive": CITRUS_REACTIVE}
    with pytest.raises(ProtocolSemanticContractError, match=message):
        _validate(features, parameters=parameters)


def test_semantic_only_heading_split_is_refused() -> None:
    with pytest.raises(ProtocolSemanticContractError, match="no admission revision before 6"):
        _validate(STATIC_FEATURES, admits=False, parameters=CITRUS_SEMANTIC_PARAMETERS)
    with pytest.raises(ProtocolSemanticContractError, match="allowed only when features.fill.type is 'heading_split'"):
        _validate(STATIC_FEATURES, parameters=CITRUS_SEMANTIC_PARAMETERS)
