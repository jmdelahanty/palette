from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from fisheye.shared.anatomy_profile import (
    AnatomyProfile,
    build_anatomy_profile_document,
    load_anatomy_profile,
    source_binding_sha256,
)
from fisheye.shared.traditional_heading_compatibility import (
    HEADING_SCALAR_CONVENTION,
    TRADITIONAL_V3_BINDING_ID,
    TraditionalHeadingCompatibilityError,
    load_traditional_v3_heading_compatibility,
    validate_traditional_v3_heading_compatibility,
)

_ROOT = Path(__file__).parents[3]
_POSE_PATH = _ROOT / "configs/fisheye/pose_schemas/traditional_v3.json"
_PROFILE_PATH = _ROOT / "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"


def _inputs() -> tuple[dict[str, object], AnatomyProfile, dict[str, object]]:
    pose = json.loads(_POSE_PATH.read_text(encoding="utf-8"))
    profile = load_anatomy_profile(_PROFILE_PATH)
    return pose, profile, profile.binding(TRADITIONAL_V3_BINDING_ID)


def _tampered_profile(
    mutate_binding=None,
) -> AnatomyProfile:
    payload = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    binding = next(
        item
        for item in payload["source_bindings"]
        if item["binding_id"] == TRADITIONAL_V3_BINDING_ID
    )
    if mutate_binding is not None:
        mutate_binding(binding)
    binding["binding_sha256"] = source_binding_sha256(binding)
    rebuilt = build_anatomy_profile_document(payload)
    return AnatomyProfile.from_mapping(rebuilt)


def test_package_configs_produce_immutable_compatibility_receipt() -> None:
    receipt = load_traditional_v3_heading_compatibility(
        pose_schema_path=_POSE_PATH,
        anatomy_profile_path=_PROFILE_PATH,
        source_binding_id=TRADITIONAL_V3_BINDING_ID,
    )

    assert receipt.schema_name == "traditional_v3"
    assert receipt.skeleton_id == "pose_skel_traditional_v3"
    assert receipt.recipe_id == "anterior_axis"
    assert receipt.scalar_convention == HEADING_SCALAR_CONVENTION
    assert receipt.validated_heading_computation()["direction_from"] == {
        "op": "keypoint",
        "label": "swim_bladder",
    }
    with pytest.raises(TypeError):
        receipt.heading_computation["enabled"] = False  # type: ignore[index]
    assert receipt.as_dict()["receipt_sha256"] == receipt.receipt_sha256


def test_explicit_pose_schema_and_binding_are_required() -> None:
    pose, profile, binding = _inputs()
    with pytest.raises(TraditionalHeadingCompatibilityError, match="traditional_v3"):
        validate_traditional_v3_heading_compatibility(
            pose_schema={**pose, "name": "traditional_v2"},
            anatomy_profile=profile,
            source_binding=binding,
        )


def test_tampered_pose_schema_copy_cannot_use_the_binding() -> None:
    pose, profile, binding = _inputs()
    pose["nodes"][0]["name"] = "renamed_swim_bladder"  # type: ignore[index]
    with pytest.raises(TraditionalHeadingCompatibilityError, match="match.*package"):
        validate_traditional_v3_heading_compatibility(
            pose_schema=pose,
            anatomy_profile=profile,
            source_binding=binding,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("enabled", False, "enabled"),
        ("direction_from", {"op": "keypoint", "label": "eye_left"}, "direction_from"),
        (
            "origin",
            {"op": "midpoint", "labels": ["eye_left"]},
            "origin.*labels",
        ),
        (
            "direction_to",
            {"op": "midpoint", "labels": ["eye_left", "swim_bladder"]},
            "controlled eye roles",
        ),
    ],
)
def test_tampered_inline_heading_is_fail_closed(field, value, message) -> None:
    pose, profile, binding = _inputs()
    pose["metadata"]["heading_computation"][field] = value  # type: ignore[index]
    with pytest.raises(TraditionalHeadingCompatibilityError, match=message):
        validate_traditional_v3_heading_compatibility(
            pose_schema=pose,
            anatomy_profile=profile,
            source_binding=binding,
        )


def test_stale_profile_and_binding_digests_are_rejected() -> None:
    pose, profile, binding = _inputs()
    stale_profile = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    stale_profile["description"] = "tampered profile"
    with pytest.raises(TraditionalHeadingCompatibilityError, match="stale digest"):
        validate_traditional_v3_heading_compatibility(
            pose_schema=pose,
            anatomy_profile=stale_profile,
            source_binding=binding,
        )

    stale_binding = deepcopy(binding)
    stale_binding["advertised_recipe_ids"] = ["eye_pair_midpoint"]
    with pytest.raises(TraditionalHeadingCompatibilityError, match="stale"):
        validate_traditional_v3_heading_compatibility(
            pose_schema=pose,
            anatomy_profile=profile,
            source_binding=stale_binding,
        )


def test_renamed_role_and_unadvertised_recipe_are_rejected_even_when_redigested() -> (
    None
):
    pose, _profile, _binding = _inputs()
    renamed_pose = deepcopy(pose)
    renamed_pose["metadata"]["heading_computation"]["dependent_keypoints"][
        0
    ] = "renamed_swim_bladder"
    with pytest.raises(TraditionalHeadingCompatibilityError, match="renamed"):
        validate_traditional_v3_heading_compatibility(
            pose_schema=renamed_pose,
            anatomy_profile=_profile,
            source_binding=_binding,
        )

    unadvertised = _tampered_profile(
        lambda binding: binding.update({"advertised_recipe_ids": ["eye_pair_midpoint"]})
    )
    with pytest.raises(TraditionalHeadingCompatibilityError, match="advertise"):
        validate_traditional_v3_heading_compatibility(
            pose_schema=pose,
            anatomy_profile=unadvertised,
            source_binding=unadvertised.binding(TRADITIONAL_V3_BINDING_ID),
        )


def test_loader_reads_tampered_profile_copy_without_writing_it(tmp_path: Path) -> None:
    tampered_path = tmp_path / "zebrafish_larva_v1.json"
    payload = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    payload["description"] = "tampered profile"
    original = json.dumps(payload, indent=2, sort_keys=True)
    tampered_path.write_text(original, encoding="utf-8")

    with pytest.raises(TraditionalHeadingCompatibilityError, match="stale digest"):
        load_traditional_v3_heading_compatibility(
            pose_schema_path=_POSE_PATH,
            anatomy_profile_path=tampered_path,
            source_binding_id=TRADITIONAL_V3_BINDING_ID,
        )
    assert tampered_path.read_text(encoding="utf-8") == original
