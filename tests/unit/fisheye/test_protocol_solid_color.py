"""SOLID_COLOR (mode 22) trial-index steps (agent-contracts stimulus vocabulary)."""

from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fisheye.shared.citrus_enums import STIMULUS_MODE
from fisheye.shared.heading_convention import STIMULUS_ANGLE_CONVENTION
from fisheye.shared.protocol_semantic_contract import (
    ProtocolSemanticContractError,
    validate_protocol_semantic_snapshot,
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _features(rgba8: list[int], *, name: object = "auto", appearance: bool = True, **element) -> dict:
    features: dict[str, object] = {"resolved_color": {"color_space": "srgb", "rgba8": rgba8}}
    if name == "auto":
        name = {
            (0, 0, 0, 255): "black",
            (255, 255, 255, 255): "white",
        }.get(tuple(rgba8), "gray" if rgba8[0] == rgba8[1] == rgba8[2] and rgba8[3] == 255 else None)
    if name is not None:
        features["color_name"] = name
    if appearance:
        background = {
            "authored_color": {"color_space": "srgb", "rgba": [v / 255 for v in rgba8]},
            "brightness": 1,
            "derivation": "plain_v1",
            "quantization": "rgba8_round_half_up_v1",
            "resolved_color": {"color_space": "srgb", "rgba8": list(rgba8)},
            "contrast_profile_digest": None,
        }
        background.update(element)
        features["appearance"] = {"elements": {"background": background}}
    return features


def _parameters(rgba8: list[int]) -> dict:
    return {
        "color": {"color_space": "srgb", "rgba8": list(rgba8)},
        "color_type_id": 2,
        "type": "ProtocolSolidColorParams",
    }


def _validate(features: dict, *, family: str = "solid_color", parameters: object = "auto"):
    if parameters == "auto":
        rgba8 = features.get("resolved_color", {}).get("rgba8", [0, 0, 0, 255])
        parameters = _parameters(rgba8)
    semantic = {
        "identity": {
            "iti_stimulus_mode_id": 99,
            "steps": [
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 600000},
                    "parameters": parameters,
                    "post_stimulus_iti": {"scale": "1e-3", "unit": "s", "value": 0},
                    "stimulus_mode_id": 22,
                }
            ],
        },
        "normalization_policy": "citrus.protocol.semantic.v2",
        "schema_id": "citrus.protocol.semantic",
        "schema_version": 1,
        "stimulus_angle_convention": STIMULUS_ANGLE_CONVENTION,
    }
    semantic_json = _json(semantic)
    semantic_hash = "sha256:" + sha256(semantic_json.encode()).hexdigest()
    trial_json = _json(
        {
            "normalization_policy": "citrus.protocol.trial_index.v3",
            "protocol_semantic_hash": semantic_hash,
            "schema_id": "citrus.protocol.trial_index",
            "schema_version": 3,
            "stimulus_angle_convention": STIMULUS_ANGLE_CONVENTION,
            "steps": [
                {
                    "duration_s": 600.0,
                    "features": features,
                    "index_status": "detailed",
                    "post_stimulus_iti_s": 0.0,
                    "step_index": 0,
                    "stimulus_family": family,
                    "stimulus_mode": "SOLID_COLOR",
                    "stimulus_mode_id": 22,
                }
            ],
        }
    )
    return validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
        trial_index_hash="sha256:" + sha256(trial_json.encode()).hexdigest(),
        snapshot_schema_version=2,
        snapshot_policy_id="citrus.protocol.snapshot.v2",
        stimulus_angle_convention=STIMULUS_ANGLE_CONVENTION,
    )


def test_mode_22_is_solid_color() -> None:
    assert STIMULUS_MODE[22] == "SOLID_COLOR"


@pytest.mark.parametrize(
    "rgba8",
    [[0, 0, 0, 255], [255, 255, 255, 255], [128, 128, 128, 255], [0, 0, 77, 255], [10, 20, 30, 255]],
    ids=["black", "white", "gray", "dim_blue", "unnamed"],
)
def test_solid_color_steps_validate(rgba8: list[int]) -> None:
    (step,) = _validate(_features(rgba8)).steps
    assert step.stimulus_mode == "SOLID_COLOR"
    assert step.resolved_color_rgba8 == tuple(rgba8)
    assert step.display_context == "other"  # never the mode-4 solid_black baseline


def test_solid_color_without_appearance_validates() -> None:
    _validate(_features([0, 0, 77, 255], appearance=False))


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ({}, "requires features.resolved_color"),
        (_features([0, 0, 77, 254]), "must be opaque"),
        (_features([128, 128, 128, 128], name=None), "must be opaque"),
        (_features([0, 0, 0, 255], name=None), "color_name"),
        (_features([128, 128, 128, 255], name=None), "color_name"),
        (_features([0, 0, 77, 255], name="blue"), "color_name"),
        (_features([128, 128, 128, 255], name="grey"), "color_name"),
        (_features([255, 255, 255, 255], name="gray"), "color_name"),
        (_features([0, 0, 77, 255], resolved_color={"color_space": "srgb", "rgba8": [0, 0, 78, 255]}), "must equal features.resolved_color"),
        (_features([0, 0, 77, 255], brightness=0.5), "background.brightness"),
        (_features([0, 0, 77, 255], derivation="edge_minimum_floor_half_v1"), "background.derivation"),
        (_features([0, 0, 77, 255], quantization="rgba8_trunc_legacy"), "background.quantization"),
        (_features([0, 0, 77, 255], contrast_profile_digest="sha256:" + "0" * 64), "background.contrast_profile_digest"),
        ({**_features([0, 0, 77, 255]), "appearance": {"elements": {}}}, "must equal features.resolved_color"),
    ],
)
def test_solid_color_refuses_malformed_steps(features: dict, message: str) -> None:
    with pytest.raises(ProtocolSemanticContractError, match=message):
        _validate(features)


def test_solid_color_requires_the_solid_color_family() -> None:
    with pytest.raises(ProtocolSemanticContractError, match="stimulus_family"):
        _validate(_features([0, 0, 77, 255]), family="phototaxis")


# Verbatim from Citrus c0c0607 tools/protocol_identity_dump, protocol A step 0
# (gray 0.5) and step 2 (dim blue, no color_name).
CITRUS_GRAY_SEMANTIC_PARAMETERS = json.loads(
    '{"color":{"color_space":"srgb","rgba8":[128,128,128,255]},"color_type_id":2,"type":"ProtocolSolidColorParams"}'
)
CITRUS_GRAY_FEATURES = json.loads(
    '{"appearance":{"elements":{"background":{"authored_color":{"color_space":"srgb","rgba":[0.5,0.5,0.5,1.0]},'
    '"brightness":1.0,"contrast_profile_digest":null,"derivation":"plain_v1","quantization":"rgba8_round_half_up_v1",'
    '"resolved_color":{"color_space":"srgb","rgba8":[128,128,128,255]}}}},"color_name":"gray",'
    '"resolved_color":{"color_space":"srgb","rgba8":[128,128,128,255]}}'
)
CITRUS_DIM_BLUE_FEATURES = json.loads(
    '{"appearance":{"elements":{"background":{"authored_color":{"color_space":"srgb",'
    '"rgba":[0.10000000149011612,0.20000000298023224,0.699999988079071,1.0]},'
    '"brightness":1.0,"contrast_profile_digest":null,"derivation":"plain_v1","quantization":"rgba8_round_half_up_v1",'
    '"resolved_color":{"color_space":"srgb","rgba8":[26,51,178,255]}}}},'
    '"resolved_color":{"color_space":"srgb","rgba8":[26,51,178,255]}}'
)


def test_citrus_bytes_validate() -> None:
    (step,) = _validate(CITRUS_GRAY_FEATURES, parameters=CITRUS_GRAY_SEMANTIC_PARAMETERS).steps
    assert step.resolved_color_rgba8 == (128, 128, 128, 255)
    (step,) = _validate(CITRUS_DIM_BLUE_FEATURES).steps
    assert step.resolved_color_rgba8 == (26, 51, 178, 255)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"color_type_id": 0, "type": "ProtocolSolidColorParams"},
        {**_parameters([128, 128, 128, 255]), "color_type_id": 1},
        {**_parameters([128, 128, 128, 255]), "color_type_id": True},
        {**_parameters([128, 128, 128, 255]), "type": "ProtocolPhototaxisParams"},
        _parameters([128, 128, 129, 255]),
        {**_parameters([128, 128, 128, 255]), "brightness": 1},
        {**_parameters([128, 128, 128, 255]), "color": {"color_space": "display_p3", "rgba8": [128, 128, 128, 255]}},
        "ProtocolSolidColorParams",
    ],
    ids=["empty", "mode4_shape", "color_type_1", "color_type_bool", "wrong_type", "other_colour", "extra_key", "colour_space", "not_object"],
)
def test_semantic_parameters_must_match_the_displayed_colour(parameters: object) -> None:
    with pytest.raises(ProtocolSemanticContractError, match="SOLID_COLOR parameters must be exactly"):
        _validate(CITRUS_GRAY_FEATURES, parameters=parameters)
