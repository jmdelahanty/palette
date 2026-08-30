from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.subject_mask_component_support import (
    SubjectMaskComponentSupportError,
    load_subject_mask_component_area_support_profile,
    require_subject_mask_component_area_support_profile,
)

MODEL_SHA256 = "217da20cd6ed780f5efe2c16add7cb932f40f08aac2f6e44795c0c381283839c"
MODEL_IDENTITY = {
    "registry_set_id": (
        "subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001"
    ),
    "registry_run_id": "subject_masks_union_all_components_v001",
    "artifact_sha256": MODEL_SHA256,
    "label_schema_id": "subject_v1_union",
}


def test_bundled_profile_resolves_exact_model_and_scales_area_floors() -> None:
    profile = require_subject_mask_component_area_support_profile(MODEL_IDENTITY)

    assert dict(profile.model_binding) == MODEL_IDENTITY
    assert profile.training_evidence["approved_source_count"] == 15
    assert profile.training_evidence["approved_row_count"] == 3153
    assert profile.reference_mask_shape_hw == (512, 512)
    assert {
        component: profile.minimum_area_px(component, mask_shape_hw=(384, 384))
        for component in (
            "subject_body",
            "eyes_union",
            "eye_left",
            "eye_right",
            "swim_bladder",
        )
    } == {
        "subject_body": 1095,
        "eyes_union": 52,
        "eye_left": 52,
        "eye_right": 52,
        "swim_bladder": 94,
    }


def test_profile_resolution_rejects_model_identity_mismatch() -> None:
    mismatched = dict(MODEL_IDENTITY)
    mismatched["registry_run_id"] = "different-run"

    with pytest.raises(
        SubjectMaskComponentSupportError,
        match="model binding differs",
    ):
        require_subject_mask_component_area_support_profile(mismatched)


def test_profile_parser_rejects_incomplete_component_coverage(tmp_path: Path) -> None:
    profile = require_subject_mask_component_area_support_profile(MODEL_IDENTITY)
    raw = dict(profile.payload)
    raw["component_families"] = list(raw["component_families"][:-1])
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        SubjectMaskComponentSupportError,
        match="coverage must be exact",
    ):
        load_subject_mask_component_area_support_profile(path)


def test_profile_parser_rejects_family_floor_not_equal_to_observed_minimum(
    tmp_path: Path,
) -> None:
    profile = require_subject_mask_component_area_support_profile(MODEL_IDENTITY)
    raw = json.loads(json.dumps(profile.payload))
    raw["component_families"][1]["minimum_area_px_reference"] = 93
    path = tmp_path / "bad-floor.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        SubjectMaskComponentSupportError,
        match="does not equal its observed source-label minimum",
    ):
        load_subject_mask_component_area_support_profile(path)
