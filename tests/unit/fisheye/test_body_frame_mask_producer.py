"""Focused tests for the refined subject-mask body-frame adapter."""

from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.anatomy_profile import load_anatomy_profile, source_schema_sha256
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_mask_producer import (
    MASK_BODY_FRAME_RECIPE_ID,
    MaskBodyFrameProducerError,
    build_mask_body_frame_recipe,
    prepare_refined_subject_mask_body_frame,
)

PROFILE_PATH = "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"
BINDING_ID = "zebrafish_larva_subject_mask_lr_v1"


def _profile():
    return load_anatomy_profile(PROFILE_PATH)


def _source_metadata():
    profile = _profile()
    binding = profile.binding(BINDING_ID)
    schema = binding["source_schema"]
    return profile, schema, source_schema_sha256(schema)


def _arrays(*, reordered: bool = False, rows: int = 4):
    labels = ("subject_body", "eye_left", "eye_right", "swim_bladder")
    centroid = np.zeros((rows, 4, 2), dtype=np.float32)
    # In the declared order: body, left eye, right eye, swim bladder.
    centroid[:, 0] = [90.0, 100.0]
    centroid[:, 1] = [20.0, 10.0]
    centroid[:, 2] = [20.0, 30.0]
    centroid[:, 3] = [0.0, 20.0]
    valid = np.ones((rows, 4), dtype=bool)
    if reordered:
        order = [3, 0, 2, 1]
        labels = tuple(labels[index] for index in order)
        centroid = centroid[:, order, :]
        valid = valid[:, order]
    return labels, centroid, valid


def _prepare(*, reordered: bool = False, centroid=None, valid=None):
    profile, schema, schema_digest = _source_metadata()
    labels, default_centroid, default_valid = _arrays(reordered=reordered)
    centroid = default_centroid if centroid is None else centroid
    valid = default_valid if valid is None else valid
    rows = int(centroid.shape[0])
    return prepare_refined_subject_mask_body_frame(
        run_path="refined_subject_masks_runs/masks-001",
        run_manifest_digest="11" * 32,
        mask_schema_id=schema["schema_id"],
        mask_schema_version=schema["schema_version"],
        mask_schema_digest=schema_digest,
        anatomy_profile=profile,
        binding_id=BINDING_ID,
        component_labels=labels,
        component_centroid_xy=centroid,
        component_centroid_valid=valid,
        available_channels=np.ones(4, dtype=bool),
        instance_key=np.asarray([101, 102, 201, 301][:rows], dtype=np.uint64),
        frame_indices=np.asarray([0, 0, 2, 3][:rows], dtype=np.int64),
        source_acquisition_frame_index=np.asarray(
            [1000, 1000, 1002, 1003][:rows], dtype=np.int64
        ),
        source_row_ids=np.asarray([7, 8, 12, 19][:rows], dtype=np.int64),
        source_row_signature=np.arange(rows * 32, dtype=np.uint8).reshape(rows, 32),
        row_identity_digest="22" * 32,
        n_frames=4,
    )


def test_recipe_binds_modality_neutral_anterior_axis_and_schema_digest():
    _, schema, schema_digest = _source_metadata()
    recipe = build_mask_body_frame_recipe(
        anatomy_profile=_profile(),
        binding_id=BINDING_ID,
        mask_schema_id=schema["schema_id"],
        mask_schema_version=schema["schema_version"],
        mask_schema_digest=schema_digest,
    )

    assert recipe.recipe_id == MASK_BODY_FRAME_RECIPE_ID
    assert recipe.role_bindings == {
        "swim_bladder": "swim_bladder",
        "eye_left": "eye_left",
        "eye_right": "eye_right",
    }
    assert recipe.payload()["recipe_profile_recipe_id"] == "anterior_axis"
    assert recipe.recipe_digest == recipe.as_manifest()["recipe_digest"]


def test_reordered_components_resolve_by_controlled_labels():
    prepared = _prepare(reordered=True)

    assert prepared.source.as_manifest()["run_path"] == (
        "refined_subject_masks_runs/masks-001"
    )
    np.testing.assert_allclose(prepared.arrays["origin_xy"][0], [20.0, 20.0])
    np.testing.assert_allclose(prepared.arrays["forward_axis_xy"][0], [1.0, 0.0])
    np.testing.assert_allclose(prepared.arrays["left_axis_xy"][0], [0.0, -1.0])
    np.testing.assert_allclose(prepared.arrays["heading_deg"][0], 0.0)


def test_missing_role_fails_closed_without_guessing_component_index():
    profile, schema, schema_digest = _source_metadata()
    labels, centroid, valid = _arrays()
    labels = tuple(label for label in labels if label != "eye_right")
    centroid = centroid[:, :3, :]
    valid = valid[:, :3]
    with pytest.raises(MaskBodyFrameProducerError, match="component labels"):
        prepare_refined_subject_mask_body_frame(
            run_path="refined_subject_masks_runs/masks-001",
            run_manifest_digest="11" * 32,
            mask_schema_id=schema["schema_id"],
            mask_schema_version=schema["schema_version"],
            mask_schema_digest=schema_digest,
            anatomy_profile=profile,
            binding_id=BINDING_ID,
            component_labels=labels,
            component_centroid_xy=centroid,
            component_centroid_valid=valid,
            available_channels=np.ones(3, dtype=bool),
            instance_key=np.asarray([101, 102, 201, 301], dtype=np.uint64),
            frame_indices=np.asarray([0, 0, 2, 3], dtype=np.int64),
            source_acquisition_frame_index=np.asarray(
                [1000, 1000, 1002, 1003], dtype=np.int64
            ),
            source_row_ids=np.asarray([7, 8, 12, 19], dtype=np.int64),
            source_row_signature=np.zeros((4, 32), dtype=np.uint8),
            row_identity_digest="22" * 32,
            n_frames=4,
        )


@pytest.mark.parametrize("failure", ("invalid", "zero"))
def test_invalid_centroid_or_zero_length_axis_is_canonical_nan(failure):
    labels, centroid, valid = _arrays()
    if failure == "invalid":
        valid[1, labels.index("eye_left")] = False
        centroid[1, labels.index("eye_left")] = np.nan
    else:
        left = centroid[2, labels.index("eye_left")]
        right = centroid[2, labels.index("eye_right")]
        centroid[2, labels.index("swim_bladder")] = (left + right) / 2.0
    prepared = _prepare(centroid=centroid, valid=valid)

    expected_valid = (
        [True, False, True, True] if failure == "invalid" else [True, True, False, True]
    )
    assert prepared.arrays["axis_valid"].tolist() == expected_valid
    invalid = ~prepared.arrays["axis_valid"]
    assert np.all(np.isnan(prepared.arrays["origin_xy"][invalid]))
    assert np.all(np.isnan(prepared.arrays["forward_axis_xy"][invalid]))
    assert np.all(np.isnan(prepared.arrays["left_axis_xy"][invalid]))
    assert np.all(np.isnan(prepared.arrays["heading_deg"][invalid]))


def test_lineage_and_offsets_are_preserved_exactly_and_payload_is_nonpublishing():
    prepared = _prepare()

    np.testing.assert_array_equal(prepared.arrays["instance_key"], [101, 102, 201, 301])
    np.testing.assert_array_equal(prepared.arrays["frame_indices"], [0, 0, 2, 3])
    np.testing.assert_array_equal(prepared.arrays["frame_row_offsets"], [0, 2, 2, 3, 4])
    np.testing.assert_array_equal(
        prepared.arrays["source_keypoint_row_ids"], [7, 8, 12, 19]
    )
    np.testing.assert_array_equal(
        prepared.arrays["source_keypoint_row_signature"],
        prepared.source_arrays["source_row_signature"],
    )
    assert prepared.source.as_manifest()["source_row_ids_digest"] == sha256_array(
        np.asarray([7, 8, 12, 19], dtype=np.int64)
    )
    payload = prepared.as_publication_payload()
    assert payload["publication"]["selector_changes"] is False
    assert payload["publication"]["dense_masks_reopened"] is False
    assert payload["schema_compatibility"].startswith("source_keypoint_row_ids/")


def test_deterministic_geometry_and_source_digests():
    left = _prepare()
    right = _prepare()

    for name in left.arrays:
        np.testing.assert_array_equal(left.arrays[name], right.arrays[name])
    assert left.recipe.as_manifest() == right.recipe.as_manifest()
    assert left.source.as_manifest() == right.source.as_manifest()


def test_stale_schema_digest_and_non_refined_path_fail_closed():
    _, schema, schema_digest = _source_metadata()
    with pytest.raises(MaskBodyFrameProducerError, match="mask_schema_digest"):
        build_mask_body_frame_recipe(
            anatomy_profile=_profile(),
            binding_id=BINDING_ID,
            mask_schema_id=schema["schema_id"],
            mask_schema_version=schema["schema_version"],
            mask_schema_digest="33" * 32,
        )
    with pytest.raises(MaskBodyFrameProducerError, match="refined_subject_masks_runs"):
        # Construction is covered directly because the normal preparation path
        # binds a valid refined path.
        from fisheye.shared.zarr.body_frame_mask_producer import (
            MaskBodyFrameSourceReference,
        )

        MaskBodyFrameSourceReference(
            run_path="subject_mask_runs/raw",
            run_manifest_digest="11" * 32,
            mask_schema_id=schema["schema_id"],
            mask_schema_version=schema["schema_version"],
            mask_schema_digest=schema_digest,
            anatomy_profile_id=_profile().profile_id,
            anatomy_profile_version=1,
            anatomy_profile_digest="44" * 32,
            binding_id=BINDING_ID,
            binding_digest="55" * 32,
            row_identity_digest="66" * 32,
            instance_key_digest="77" * 32,
            frame_indices_digest="88" * 32,
            source_acquisition_frame_index_digest="99" * 32,
            source_row_ids_digest="aa" * 32,
            source_row_signature_digest="bb" * 32,
        )
