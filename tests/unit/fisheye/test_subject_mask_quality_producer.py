from __future__ import annotations

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_quality_producer import (
    COMPONENT_FLAG_MISSING,
    OBSERVATION_FLAG_EYE_LEFT_OUTSIDE_BODY,
    OBSERVATION_FLAG_EYE_PAIR_OVERLAP,
    OBSERVATION_FLAG_EYE_RIGHT_SWIM_BLADDER_OVERLAP,
    OBSERVATION_FLAG_MISSING_REQUIRED_COMPONENT,
    SUBJECT_V1_LR_COMPONENTS,
    SubjectV1LrObservationQualityPolicy,
    prepare_in_memory_observation_local_subject_mask_quality,
    quality_profile_for_policy,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_schema import SubjectMaskComponentRegistry


def _components() -> SubjectMaskComponentRegistry:
    return SubjectMaskComponentRegistry(SUBJECT_V1_LR_COMPONENTS)


def _source_reference() -> SubjectMaskQualitySourceReference:
    components = _components()
    return SubjectMaskQualitySourceReference(
        run_name="refined_subject_masks_001",
        manifest_digest="11" * 32,
        dense_array_values_sha256="22" * 32,
        component_registry_digest=canonical_json_sha256(components.as_manifest()),
    )


def _source_arrays() -> dict[str, np.ndarray]:
    masks = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    body, left, right, bladder = range(4)
    masks[:, body, 1:7, 1:7] = 1

    # Fully conforming: organs overlap the whole-body silhouette, but not one another.
    masks[0, left, 2, 2] = 1
    masks[0, right, 2, 5] = 1
    masks[0, bladder, 5, 3] = 1

    # Left eye outside the body; right eye and swim bladder overlap.
    masks[1, left, 0, 0] = 1
    masks[1, right, 3, 3] = 1
    masks[1, bladder, 3, 3] = 1

    # Both eye masks overlap.
    masks[2, left, 2, 2] = 1
    masks[2, right, 2, 2] = 1
    masks[2, bladder, 5, 3] = 1

    # A required component is absent.
    masks[3, left, 2, 2] = 1
    masks[3, bladder, 5, 3] = 1

    return {
        "masks_roi": masks,
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray(
            [0, 0, 2, 3], dtype=np.int64
        ),
        "available_channels": np.ones((4,), dtype=bool),
    }


def test_subject_v1_lr_profile_freezes_current_component_relations() -> None:
    policy = SubjectV1LrObservationQualityPolicy()
    profile = quality_profile_for_policy(policy)

    assert policy.payload()["component_semantics"]["subject_body"] == (
        "whole_subject_silhouette_including_organs"
    )
    assert policy.payload()["exclusive_pairs"] == [
        ["eye_left", "eye_right"],
        ["eye_left", "swim_bladder"],
        ["eye_right", "swim_bladder"],
    ]
    assert len(profile.component_metrics) == 8
    assert len(profile.observation_metrics) == 7
    assert profile.component_metrics[0].higher_is_worse is None
    assert profile.policy_digest == policy.policy_digest


def test_subject_mask_quality_relations_and_validity_are_exact() -> None:
    source_arrays = _source_arrays()
    masks_before = source_arrays["masks_roi"].copy()
    prepared = prepare_in_memory_observation_local_subject_mask_quality(
        source_arrays,
        n_frames=4,
        components=_components(),
        source=_source_reference(),
    )
    arrays = prepared.arrays

    np.testing.assert_array_equal(source_arrays["masks_roi"], masks_before)
    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 4]
    np.testing.assert_allclose(
        arrays["observation_metric_values"][0],
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert int(arrays["observation_quality_flags"][0]) == 0
    assert bool(arrays["proposed_observation_usable"][0])

    row_one_flags = int(arrays["observation_quality_flags"][1])
    assert row_one_flags & OBSERVATION_FLAG_EYE_LEFT_OUTSIDE_BODY
    assert row_one_flags & OBSERVATION_FLAG_EYE_RIGHT_SWIM_BLADDER_OVERLAP
    assert arrays["observation_metric_values"][1, 1] == np.float32(1.0)
    assert arrays["observation_metric_values"][1, 6] == np.float32(1.0)

    assert int(arrays["observation_quality_flags"][2]) & (
        OBSERVATION_FLAG_EYE_PAIR_OVERLAP
    )
    assert int(arrays["component_quality_flags"][3, 2]) & COMPONENT_FLAG_MISSING
    assert int(arrays["observation_quality_flags"][3]) & (
        OBSERVATION_FLAG_MISSING_REQUIRED_COMPONENT
    )
    assert not bool(arrays["proposed_component_usable"][3, 2])
    assert not bool(arrays["proposed_observation_usable"][3])

    # Counts and area remain valid for empty masks; ratios are canonical NaN.
    values = arrays["component_metric_values"][3, 2]
    valid = arrays["component_metric_valid"][3, 2]
    assert values[0] == np.float32(0.0) and bool(valid[0])
    assert values[1] == np.float32(0.0) and bool(valid[1])
    assert values[3] == np.float32(0.0) and bool(valid[3])
    assert np.isnan(values[2]) and not bool(valid[2])
    assert np.isnan(values[4]) and not bool(valid[4])


def test_relation_tolerances_are_digest_bound_policy_inputs() -> None:
    exact = SubjectV1LrObservationQualityPolicy()
    tolerant = SubjectV1LrObservationQualityPolicy(
        maximum_outside_body_fraction=0.01,
        maximum_exclusive_pair_overlap_fraction=0.01,
    )

    assert exact.policy_digest != tolerant.policy_digest
    assert exact.as_manifest()["policy_digest"] == exact.policy_digest
