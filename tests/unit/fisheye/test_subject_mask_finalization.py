from __future__ import annotations

import numpy as np
import pytest

from fisheye.refinement.subject_mask_finalization import (
    QUALITY_CLEANUP_APPLIED,
    QUALITY_NEEDS_REVIEW,
    ComponentFinalizationPolicy,
    finalize_component_mask,
)


def test_subject_body_finalization_fills_holes_and_removes_small_islands() -> None:
    surface = np.zeros((18, 18), dtype=np.float32)
    surface[4:14, 4:14] = 0.9
    surface[7:9, 7:9] = 0.0
    surface[1:3, 1:3] = 0.6

    policy = ComponentFinalizationPolicy(
        component_name="subject_body",
        threshold=0.5,
        closing_radius=0,
        fill_holes=True,
        min_component_area_px=5,
        keep_largest_component=True,
        max_removed_high_prob_mass_fraction=1.0,
        max_changed_area_fraction=1.0,
    )

    result = finalize_component_mask("subject_body", surface, policy=policy)

    assert result.mask.dtype == np.uint8
    assert result.mask[1:3, 1:3].sum() == 0
    assert result.mask[7:9, 7:9].all()
    assert result.quality_code == QUALITY_CLEANUP_APPLIED
    assert result.review_recommendation == "pending"
    assert "cleanup_filled_holes" in result.reason_tags
    assert "cleanup_removed_small_islands" in result.reason_tags
    assert not any(tag.startswith("needs_review") for tag in result.reason_tags)
    assert result.metrics["component_count_after"] == 1.0


def test_subject_body_finalization_routes_removed_high_probability_islands_to_review() -> (
    None
):
    surface = np.zeros((20, 20), dtype=np.float32)
    surface[5:15, 5:15] = 0.9
    surface[1:4, 1:4] = 0.95

    policy = ComponentFinalizationPolicy(
        component_name="subject_body",
        threshold=0.5,
        high_threshold=0.8,
        fill_holes=True,
        min_component_area_px=1,
        keep_largest_component=True,
        max_removed_high_prob_mass_fraction=0.001,
        max_changed_area_fraction=1.0,
    )

    result = finalize_component_mask("subject_body", surface, policy=policy)

    assert result.mask[1:4, 1:4].sum() == 0
    assert result.mask[5:15, 5:15].all()
    assert result.quality_code == QUALITY_NEEDS_REVIEW
    assert result.review_recommendation == "needs_review"
    assert "cleanup_kept_largest_component" in result.reason_tags
    assert "needs_review_removed_high_prob_island" in result.reason_tags
    assert result.metrics["removed_high_prob_area_px"] == 9.0
    assert result.quality_score >= 100.0


def test_subject_body_finalization_hysteresis_keeps_low_support_touching_high_seed_only() -> (
    None
):
    surface = np.zeros((12, 12), dtype=np.float32)
    surface[3:8, 3:8] = 0.4
    surface[5, 5] = 0.9
    surface[1:3, 9:11] = 0.4

    policy = ComponentFinalizationPolicy(
        component_name="subject_body",
        low_threshold=0.3,
        high_threshold=0.8,
        fill_holes=False,
        min_component_area_px=1,
        keep_largest_component=False,
    )

    result = finalize_component_mask("subject_body", surface, policy=policy)

    assert result.mask[3:8, 3:8].all()
    assert result.mask[1:3, 9:11].sum() == 0
    assert result.reason_tags == ("clean",)


def test_subject_body_finalization_rejects_policy_component_mismatch() -> None:
    policy = ComponentFinalizationPolicy(component_name="subject_body")

    with pytest.raises(ValueError, match="Policy component"):
        finalize_component_mask(
            "swim_bladder", np.zeros((4, 4), dtype=np.float32), policy=policy
        )


def test_subject_body_finalization_requires_explicit_component_policy() -> None:
    with pytest.raises(NotImplementedError, match="No default finalization policy"):
        finalize_component_mask("unknown_component", np.zeros((4, 4), dtype=np.float32))


def test_swim_bladder_finalization_fills_hole_and_keeps_one_component() -> None:
    surface = np.zeros((16, 16), dtype=np.float32)
    surface[5:11, 5:11] = 0.9
    surface[7, 7] = 0.0
    surface[1:3, 1:3] = 0.7

    result = finalize_component_mask("swim_bladder", surface)

    assert result.mask[7, 7] == 1
    assert result.mask[1:3, 1:3].sum() == 0
    assert result.metrics["component_count_after"] == 1.0
    assert "cleanup_filled_holes" in result.reason_tags
    assert "cleanup_kept_largest_component" in result.reason_tags


def test_eyes_union_finalization_preserves_two_eye_components_and_removes_tiny_island() -> (
    None
):
    surface = np.zeros((20, 20), dtype=np.float32)
    surface[4:8, 4:8] = 0.9
    surface[4:8, 12:16] = 0.9
    surface[18, 18] = 0.6

    result = finalize_component_mask("eyes_union", surface)

    assert result.mask[4:8, 4:8].all()
    assert result.mask[4:8, 12:16].all()
    assert result.mask[18, 18] == 0
    assert result.metrics["component_count_after"] == 2.0
    assert "cleanup_removed_small_islands" in result.reason_tags
    assert not any(tag.startswith("needs_review") for tag in result.reason_tags)


def test_eyes_union_finalization_routes_three_plausible_components_to_review() -> None:
    surface = np.zeros((24, 24), dtype=np.float32)
    surface[3:7, 3:7] = 0.9
    surface[3:7, 10:14] = 0.9
    surface[3:7, 17:21] = 0.9

    result = finalize_component_mask("eyes_union", surface)

    assert result.metrics["component_count_after"] == 2.0
    assert result.review_recommendation == "needs_review"
    assert "needs_review_multiple_components" in result.reason_tags


def test_model_supported_area_floor_rejects_tiny_component_with_explicit_reason() -> (
    None
):
    surface = np.zeros((16, 16), dtype=np.float32)
    surface[4:7, 5:9] = 0.9
    policy = ComponentFinalizationPolicy(
        component_name="swim_bladder",
        threshold=0.5,
        closing_radius=0,
        fill_holes=False,
        min_component_area_px=20,
        keep_largest_component=True,
        component_area_support_profile_id="training-support-v1",
        component_area_support_profile_digest="a" * 64,
        component_area_support_minimum_px=20,
    )

    result = finalize_component_mask("swim_bladder", surface, policy=policy)

    assert int(result.source_mask.sum()) == 12
    assert int(result.mask.sum()) == 0
    assert "cleanup_removed_below_model_supported_area" in result.reason_tags
    assert "needs_review_below_model_supported_area" in result.reason_tags
    assert result.review_recommendation == "needs_review"
