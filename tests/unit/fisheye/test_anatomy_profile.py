from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fisheye.shared.anatomy_profile import (
    AnatomyProfile,
    AnatomyProfileError,
    anatomy_profile_sha256,
    build_anatomy_profile_document,
    load_anatomy_profile,
    source_binding_sha256,
    source_schema_sha256,
    validate_source_binding,
)
from fisheye.shared.pose_schema import schema_payload_from_package


PROFILE_PATH = (
    Path(__file__).parents[3]
    / "configs"
    / "fisheye"
    / "anatomy_profiles"
    / "zebrafish_larva_v1.json"
)


def _profile_payload() -> dict[str, object]:
    return {
        "schema_id": "palette.anatomy_profile",
        "schema_version": 1,
        "profile_id": "zebrafish_larva_anatomy.v1",
        "profile_version": 1,
        "digest_algorithm": "sha256_canonical_json_v1",
        "roles": [
            {"role_id": "eye_left", "description": "left eye"},
            {"role_id": "eye_right", "description": "right eye"},
            {"role_id": "subject_body", "description": "body"},
            {"role_id": "swim_bladder", "description": "swim bladder"},
        ],
        "recipes": [
            {
                "recipe_id": "anterior_axis",
                "kind": "axis",
                "required_roles": ["eye_left", "eye_right", "swim_bladder"],
                "expression": {
                    "op": "axis",
                    "from": {"op": "role_point", "role_id": "swim_bladder"},
                    "to": {
                        "op": "midpoint",
                        "points": [
                            {"op": "role_point", "role_id": "eye_left"},
                            {"op": "role_point", "role_id": "eye_right"},
                        ],
                    },
                },
            },
            {
                "recipe_id": "eye_pair_midpoint",
                "kind": "point",
                "required_roles": ["eye_left", "eye_right"],
                "expression": {
                    "op": "midpoint",
                    "points": [
                        {"op": "role_point", "role_id": "eye_left"},
                        {"op": "role_point", "role_id": "eye_right"},
                    ],
                },
            },
            {
                "recipe_id": "head_triad_equal_mean",
                "kind": "point",
                "required_roles": ["eye_left", "eye_right", "swim_bladder"],
                "expression": {
                    "op": "mean_points",
                    "weighting": "equal_per_point",
                    "points": [
                        {"op": "role_point", "role_id": "eye_left"},
                        {"op": "role_point", "role_id": "eye_right"},
                        {"op": "role_point", "role_id": "swim_bladder"},
                    ],
                },
            },
        ],
        "source_bindings": [],
    }


def _digested_profile(payload: dict[str, object]) -> dict[str, object]:
    return build_anatomy_profile_document(payload)


def _source_schema(
    *, labels: list[str], modality: str = "subject_mask"
) -> dict[str, object]:
    schema: dict[str, object] = {
        "authority": "declared_schema",
        "schema_id": f"{modality}_schema_test",
        "schema_version": 1,
        "modality": modality,
        "labels": labels,
    }
    schema["schema_sha256"] = source_schema_sha256(schema)
    return schema


def _binding(
    profile: AnatomyProfile,
    *,
    source_schema: dict[str, object] | None = None,
    role_bindings: list[dict[str, str]] | None = None,
    advertised_recipe_ids: list[str] | None = None,
) -> dict[str, object]:
    binding: dict[str, object] = {
        "schema_id": "palette.anatomy_source_role_binding",
        "schema_version": 1,
        "binding_id": "test_binding_v1",
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "source_schema": source_schema
        or _source_schema(
            labels=["swim_bladder", "eye_left", "eye_right", "unused"],
        ),
        "role_bindings": role_bindings
        or [
            {"role_id": "eye_left", "source_label": "eye_left"},
            {"role_id": "eye_right", "source_label": "eye_right"},
            {"role_id": "swim_bladder", "source_label": "swim_bladder"},
        ],
        "advertised_recipe_ids": advertised_recipe_ids
        or ["anterior_axis", "eye_pair_midpoint", "head_triad_equal_mean"],
    }
    binding["binding_sha256"] = source_binding_sha256(binding)
    return binding


def test_zebrafish_profile_has_strict_roles_recipes_and_valid_bindings() -> None:
    profile = load_anatomy_profile(PROFILE_PATH)

    assert profile.profile_id == "zebrafish_larva_anatomy.v1"
    assert {role.role_id for role in profile.roles} == {
        "swim_bladder",
        "eye_left",
        "eye_right",
        "subject_body",
    }
    assert {recipe.recipe_id for recipe in profile.recipes} == {
        "head_triad_equal_mean",
        "eye_pair_midpoint",
        "subject_body_centroid",
        "anterior_axis",
    }
    assert {binding["source_schema"]["modality"] for binding in profile.source_bindings} == {
        "keypoint",
        "subject_mask",
    }
    keypoint_binding_v2 = profile.binding(
        "zebrafish_larva_keypoint_traditional_v2_v1"
    )
    assert keypoint_binding_v2["profile_id"] == "zebrafish_larva_anatomy.v1"
    assert (
        keypoint_binding_v2["source_schema"]["authority"]
        == "keypoint_skeleton_semantics"
    )
    assert keypoint_binding_v2["source_schema"]["skeleton_sha256"] == (
        "6eeb4bece23774d3c75e76664a414ea93e84d4f6102293f6cce6f065793954ef"
    )
    assert source_schema_sha256(keypoint_binding_v2["source_schema"]) == (
        keypoint_binding_v2["source_schema"]["skeleton_sha256"]
    )
    assert keypoint_binding_v2["source_schema"]["skeleton_document"][
        "skeleton_id"
    ] == "pose_skel_traditional_v2"
    assert keypoint_binding_v2["source_schema"]["skeleton_document"]["edges"] == [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 3],
        [0, 4],
    ]
    assert keypoint_binding_v2["source_local_compatibility"] == {
        "kind": "inline_pose_heading",
        "metadata_path": "metadata.heading_computation",
        "authority": "source_schema",
    }
    assert {
        item["role_id"]: item["source_label"]
        for item in keypoint_binding_v2["role_bindings"]
    } == {
        "swim_bladder": "swim_bladder",
        "eye_left": "eye_left",
        "eye_right": "eye_right",
    }

    keypoint_binding = profile.binding(
        "zebrafish_larva_keypoint_traditional_v3_v1"
    )
    _schema, package_payload = schema_payload_from_package("traditional_v3")
    assert keypoint_binding["profile_id"] == "zebrafish_larva_anatomy.v1"
    assert keypoint_binding["source_schema"]["authority"] == "pose_schema_package"
    assert keypoint_binding["source_schema"]["package_payload"] == package_payload
    assert keypoint_binding["source_local_compatibility"] == {
        "kind": "inline_pose_heading",
        "metadata_path": "metadata.heading_computation",
        "authority": "source_schema",
    }
    mask_binding = profile.binding("zebrafish_larva_subject_mask_lr_v1")
    assert mask_binding["profile_id"] == "zebrafish_larva_anatomy.v1"
    assert "subject_body_centroid" in mask_binding["advertised_recipe_ids"]
    assert "subject_body_centroid" not in keypoint_binding_v2["advertised_recipe_ids"]
    assert "subject_body_centroid" not in keypoint_binding["advertised_recipe_ids"]


def test_keypoint_binding_rejects_wrong_pose_schema_package_bytes(
    tmp_path: Path,
) -> None:
    profile = load_anatomy_profile(PROFILE_PATH)
    binding = profile.binding("zebrafish_larva_keypoint_traditional_v3_v1")
    package_path = (
        Path(__file__).parents[3]
        / "configs"
        / "fisheye"
        / "pose_schemas"
        / "traditional_v3.json"
    )
    wrong_package = json.loads(package_path.read_text(encoding="utf-8"))
    wrong_package["nodes"][0]["name"] = "wrong_swim_bladder"
    (tmp_path / "traditional_v3.json").write_text(
        json.dumps(wrong_package), encoding="utf-8"
    )

    with pytest.raises(
        AnatomyProfileError,
        match="does not match the exact current pose-schema package payload",
    ):
        validate_source_binding(
            profile,
            binding,
            pose_schema_base_dir=tmp_path,
        )


def test_keypoint_binding_rejects_redigested_wrong_embedded_package() -> None:
    profile = load_anatomy_profile(PROFILE_PATH)
    binding = profile.binding("zebrafish_larva_keypoint_traditional_v3_v1")
    binding["source_schema"]["package_payload"]["nodes"][0][
        "name"
    ] = "wrong_swim_bladder"
    binding["source_schema"]["package_sha256"] = source_schema_sha256(
        binding["source_schema"]
    )
    binding["binding_sha256"] = source_binding_sha256(binding)

    with pytest.raises(
        AnatomyProfileError,
        match="does not match the exact current pose-schema package payload",
    ):
        validate_source_binding(profile, binding)


def test_validated_profile_authority_cannot_be_mutated_in_place() -> None:
    profile = load_anatomy_profile(PROFILE_PATH)
    digest_before = anatomy_profile_sha256(profile)
    canonical_digest_before = source_schema_sha256(
        profile.binding("zebrafish_larva_keypoint_traditional_v3_v1")[
            "source_schema"
        ]
    )

    with pytest.raises(TypeError):
        profile.payload["profile_id"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.source_bindings[0]["profile_id"] = "mutated"  # type: ignore[index]
    recipe = profile.recipe("head_triad_equal_mean")
    with pytest.raises(TypeError):
        recipe.expression["points"][0]["role_id"] = "mutated"  # type: ignore[index]

    detached = profile.binding("zebrafish_larva_keypoint_traditional_v3_v1")
    detached["profile_id"] = "mutated"
    detached["source_schema"]["package_payload"]["nodes"][0][
        "name"
    ] = "mutated"

    resolved_again = profile.binding(
        "zebrafish_larva_keypoint_traditional_v3_v1"
    )
    assert resolved_again["profile_id"] == "zebrafish_larva_anatomy.v1"
    assert resolved_again["source_schema"]["package_payload"]["nodes"][0] == {
        "id": 0,
        "name": "swim_bladder",
    }
    assert anatomy_profile_sha256(profile) == digest_before
    assert (
        source_schema_sha256(resolved_again["source_schema"])
        == canonical_digest_before
    )


def test_reordered_source_labels_do_not_change_explicit_role_binding() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(
        profile,
        source_schema=_source_schema(
            labels=["unused", "eye_right", "swim_bladder", "eye_left"],
        ),
    )

    validated = validate_source_binding(profile, binding)

    assert {item["role_id"] for item in validated["role_bindings"]} == {
        "eye_left",
        "eye_right",
        "swim_bladder",
    }


def test_missing_role_rejects_advertised_recipe() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(
        profile,
        role_bindings=[
            {"role_id": "eye_left", "source_label": "eye_left"},
            {"role_id": "eye_right", "source_label": "eye_right"},
        ],
    )

    with pytest.raises(AnatomyProfileError, match="missing roles: swim_bladder"):
        validate_source_binding(profile, binding)


def test_duplicate_source_mapping_rejects_binding() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(
        profile,
        role_bindings=[
            {"role_id": "eye_left", "source_label": "eye_left"},
            {"role_id": "eye_right", "source_label": "eye_left"},
            {"role_id": "swim_bladder", "source_label": "swim_bladder"},
        ],
    )

    with pytest.raises(AnatomyProfileError, match="duplicate source-label mapping"):
        validate_source_binding(profile, binding)


def test_unknown_role_rejects_binding() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(
        profile,
        role_bindings=[
            {"role_id": "eye_left", "source_label": "eye_left"},
            {"role_id": "eye_right", "source_label": "eye_right"},
            {"role_id": "tail_tip", "source_label": "swim_bladder"},
        ],
    )

    with pytest.raises(AnatomyProfileError, match="unknown anatomy role 'tail_tip'"):
        validate_source_binding(profile, binding)


def test_unadvertised_recipe_capability_is_rejected() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(
        profile,
        advertised_recipe_ids=["head_triad_equal_mean"],
    )
    binding["role_bindings"] = [
        {"role_id": "eye_left", "source_label": "eye_left"},
        {"role_id": "eye_right", "source_label": "eye_right"},
    ]
    binding["binding_sha256"] = source_binding_sha256(binding)

    with pytest.raises(AnatomyProfileError, match="missing roles: swim_bladder"):
        validate_source_binding(profile, binding)


def test_eyes_union_cannot_satisfy_independent_eye_recipe() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(
        profile,
        source_schema=_source_schema(
            labels=["subject_body", "eyes_union", "swim_bladder"],
            modality="subject_mask",
        ),
        role_bindings=[
            {"role_id": "eye_left", "source_label": "eyes_union"},
            {"role_id": "eye_right", "source_label": "eyes_union"},
            {"role_id": "swim_bladder", "source_label": "swim_bladder"},
        ],
    )

    with pytest.raises(AnatomyProfileError, match="duplicate source-label mapping"):
        validate_source_binding(profile, binding)


@pytest.mark.parametrize("digest_field", ["profile_sha256", "binding_sha256"])
def test_stale_digests_fail_closed(digest_field: str) -> None:
    profile_payload = _digested_profile(_profile_payload())
    if digest_field == "profile_sha256":
        profile_payload["profile_sha256"] = "f" * 64
        with pytest.raises(AnatomyProfileError, match="stale digest"):
            AnatomyProfile.from_mapping(profile_payload)
        return

    profile = AnatomyProfile.from_mapping(profile_payload)
    binding = _binding(profile)
    binding["binding_sha256"] = "f" * 64
    with pytest.raises(AnatomyProfileError, match="stale digest"):
        validate_source_binding(profile, binding)


def test_source_schema_digest_is_also_checked() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(profile)
    binding["source_schema"] = copy.deepcopy(binding["source_schema"])
    binding["source_schema"]["schema_sha256"] = "f" * 64
    binding["binding_sha256"] = source_binding_sha256(binding)

    with pytest.raises(AnatomyProfileError, match="source_schema_sha256|stale digest"):
        validate_source_binding(profile, binding)


def test_inline_heading_compatibility_is_source_local_not_shared_profile() -> None:
    profile = AnatomyProfile.from_mapping(_digested_profile(_profile_payload()))
    binding = _binding(profile)
    binding["source_local_compatibility"] = {
        "kind": "inline_pose_heading",
        "metadata_path": "metadata.heading_computation",
        "authority": "source_schema",
    }
    binding["binding_sha256"] = source_binding_sha256(binding)

    validated = validate_source_binding(profile, binding)

    assert "profile_id" not in validated["source_local_compatibility"]
    assert validated["source_local_compatibility"]["authority"] == "source_schema"
