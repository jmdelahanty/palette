from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.chaser_appearance import (
    APPEARANCE_POLICY_ID,
    ChaserAppearanceProjectionError,
    resolve_chaser_appearance_projection,
)


def _evidence() -> tuple[dict, dict]:
    protocol = {
        "steps": [
            {
                "parameters": {
                    "chasers": [
                        {
                            "chaser_index": 0,
                            "enable_chase": True,
                            "enable_random_movement": True,
                            "behavior_mode": 0,
                            "color_r": 0.0,
                            "color_g": 0.0,
                            "color_b": 1.0,
                            "color_a": 1.0,
                        },
                        {
                            "chaser_index": 1,
                            "enable_chase": False,
                            "enable_random_movement": False,
                            "behavior_mode": 1,
                            "color_r": 0.0,
                            "color_g": 0.0,
                            "color_b": 1.0,
                            "color_a": 1.0,
                        },
                    ]
                }
            }
        ]
    }
    occurrence = {
        "schema_id": "palette.chaser_relative_frame.chaser_occurrence_binding",
        "schema_version": 1,
        "recording_id": "recording-1",
        "occurrence_policy_id": "native_sample_declared_chaser_axis_v1",
        "chaser_identity_policy_id": "stimulus_run_scoped_chaser_index_v1",
        "source_stimulus_run_path": "analysis/stimulus_runs/stimulus-exact-v1",
        "source_protocol_sha256": canonical_json_sha256(protocol),
        "chasers": [
            {
                "chaser_index": 0,
                "identity": "stimulus-exact-v1:chaser_index:0",
                "behavior_role": "aggressive",
            },
            {
                "chaser_index": 1,
                "identity": "stimulus-exact-v1:chaser_index:1",
                "behavior_role": "inert",
            },
        ],
        "semantics": "exact native chaser axis",
    }
    manifest = {
        "recording_id": "recording-1",
        "dimensions": {"n_frames": 4, "n_chasers": 2, "n_rows": 8},
        "identity_registries": {
            "fish": {"1": "fish-1"},
            "chaser": {
                "1": "stimulus-exact-v1:chaser_index:0",
                "2": "stimulus-exact-v1:chaser_index:1",
            },
            "behavior_role": {"1": "aggressive", "2": "inert"},
            "active_state": {"0": "inactive", "1": "active"},
        },
        "context": {
            "chaser_occurrence": {
                "record": occurrence,
                "sha256": canonical_json_sha256(occurrence),
            }
        },
    }
    return manifest, protocol


def _resolve(manifest: dict, protocol: dict):
    return resolve_chaser_appearance_projection(
        relative_manifest=manifest,
        protocol_payload=protocol,
        identity_code_by_column=np.asarray([1, 2], dtype=np.uint16),
        behavior_role_code_by_column=np.asarray([1, 2], dtype=np.uint8),
        expected_recording_id="recording-1",
    )


def test_protocol_color_and_behavior_role_are_independent_channels() -> None:
    manifest, protocol = _evidence()

    projection = _resolve(manifest, protocol)

    assert projection.provenance_record()["appearance_policy_id"] == (
        APPEARANCE_POLICY_ID
    )
    aggressive, inert = projection.appearances
    assert aggressive.experimental_color_hex == "#0000ff"
    assert inert.experimental_color_hex == "#0000ff"
    assert aggressive.plotly_role_symbol == "star"
    assert inert.plotly_role_symbol == "circle"
    assert aggressive.behavior_role == "aggressive"
    assert inert.behavior_role == "inert"
    assert projection.provenance_record()["color_role_independence"] is True


def test_protocol_digest_mismatch_fails_closed() -> None:
    manifest, protocol = _evidence()
    protocol["steps"][0]["parameters"]["chasers"][0]["color_b"] = 0.5

    with pytest.raises(ChaserAppearanceProjectionError, match="differs"):
        _resolve(manifest, protocol)


def test_occurrence_and_array_role_mismatch_fails_closed() -> None:
    manifest, protocol = _evidence()

    with pytest.raises(ChaserAppearanceProjectionError, match="roles disagree"):
        resolve_chaser_appearance_projection(
            relative_manifest=manifest,
            protocol_payload=protocol,
            identity_code_by_column=np.asarray([1, 2], dtype=np.uint16),
            behavior_role_code_by_column=np.asarray([2, 1], dtype=np.uint8),
        )


def test_missing_protocol_color_does_not_fall_back() -> None:
    manifest, protocol = _evidence()
    modified = deepcopy(protocol)
    del modified["steps"][0]["parameters"]["chasers"][0]["color_b"]
    occurrence = manifest["context"]["chaser_occurrence"]["record"]
    occurrence["source_protocol_sha256"] = canonical_json_sha256(modified)
    manifest["context"]["chaser_occurrence"]["sha256"] = canonical_json_sha256(
        occurrence
    )

    with pytest.raises(ChaserAppearanceProjectionError, match="fallback is prohibited"):
        _resolve(manifest, modified)
