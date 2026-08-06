from __future__ import annotations

import pytest

from fisheye.shared.refined_subject_mask_mutation import (
    REFINED_SUBJECT_MASK_EDITABLE_DRAFT,
    REFINED_SUBJECT_MASK_LIFECYCLE_ATTR,
    REFINED_SUBJECT_MASK_SEALED_SNAPSHOT,
    RefinedSubjectMaskMutationError,
    refined_subject_mask_lifecycle_state,
    migrate_unapproved_refined_subject_mask_to_editable_draft,
    require_mutable_refined_subject_mask_group,
    stamp_refined_subject_mask_editable_draft,
    stamp_refined_subject_mask_sealed_snapshot,
)


class _Group:
    def __init__(self, attrs: dict[str, object]) -> None:
        self.attrs = attrs
        self.path = "refined_subject_masks_runs/r1"


class _Parent(dict[str, _Group]):
    def __init__(self, children: dict[str, _Group], attrs: dict[str, object]) -> None:
        super().__init__(children)
        self.attrs = attrs


class _Root(dict[str, _Parent]):
    pass


def _approved_attrs() -> dict[str, object]:
    return {
        "stage_selector_eligible": False,
        "mask_labels": ["subject_body", "eye_left"],
        "component_review_statuses": {
            "subject_body": {"state": "approved"},
            "eye_left": {"state": "approved"},
        },
        "refined_subject_mask_review_status": {"state": "approved"},
    }


def test_explicit_editable_canonical_draft_is_mutable() -> None:
    group = _Group(
        {
            "coordinate_contract": "canonical_v2",
            "refined_subject_mask_publication_owner": "a" * 32,
            "stage_selector_eligible": False,
        }
    )

    stamp_refined_subject_mask_editable_draft(group)

    assert refined_subject_mask_lifecycle_state(group) == (
        REFINED_SUBJECT_MASK_EDITABLE_DRAFT
    )
    assert require_mutable_refined_subject_mask_group(group) is group


def test_legacy_canonical_run_remains_fail_closed() -> None:
    group = _Group(
        {
            "coordinate_contract": "canonical_v2",
            "stage_selector_eligible": False,
        }
    )

    with pytest.raises(RefinedSubjectMaskMutationError, match="immutable"):
        require_mutable_refined_subject_mask_group(group)


def test_editable_draft_cannot_be_selector_eligible() -> None:
    group = _Group({"stage_selector_eligible": True})

    with pytest.raises(RefinedSubjectMaskMutationError, match="selector-ineligible"):
        stamp_refined_subject_mask_editable_draft(group)


def test_sealing_requires_all_component_reviews_approved() -> None:
    attrs = _approved_attrs()
    attrs["component_review_statuses"]["eye_left"] = {"state": "pending"}  # type: ignore[index]
    group = _Group(attrs)
    stamp_refined_subject_mask_editable_draft(group)

    with pytest.raises(RefinedSubjectMaskMutationError, match="not approved"):
        stamp_refined_subject_mask_sealed_snapshot(group)


def test_approved_draft_can_be_sealed_and_is_then_immutable() -> None:
    group = _Group(_approved_attrs())
    stamp_refined_subject_mask_editable_draft(group)

    stamp_refined_subject_mask_sealed_snapshot(group)

    assert group.attrs[REFINED_SUBJECT_MASK_LIFECYCLE_ATTR]["state"] == (  # type: ignore[index]
        REFINED_SUBJECT_MASK_SEALED_SNAPSHOT
    )
    with pytest.raises(RefinedSubjectMaskMutationError, match="immutable"):
        require_mutable_refined_subject_mask_group(group)


def test_unapproved_activated_run_can_be_demoted_without_array_writes() -> None:
    run = _Group(
        {
            "coordinate_contract": "canonical_v2",
            "refined_subject_mask_publication_owner": "a" * 32,
            "stage_selector_eligible": True,
            "refined_subject_mask_review_status": {"state": "pending"},
        }
    )
    parent = _Parent(
        {"r1": run},
        {"latest": "r1", "latest_complete": "r1"},
    )
    root = _Root({"refined_subject_masks_runs": parent})

    result = migrate_unapproved_refined_subject_mask_to_editable_draft(root, "r1")

    assert result["scientific_arrays_rewritten"] is False
    assert run.attrs["stage_selector_eligible"] is False
    assert refined_subject_mask_lifecycle_state(run) == (
        REFINED_SUBJECT_MASK_EDITABLE_DRAFT
    )
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    assert parent.attrs["latest_pending"] == "r1"


def test_approved_run_cannot_be_demoted() -> None:
    run = _Group(
        {
            "coordinate_contract": "canonical_v2",
            "refined_subject_mask_publication_owner": "a" * 32,
            "stage_selector_eligible": True,
            "refined_subject_mask_review_status": {"state": "approved"},
        }
    )
    root = _Root({"refined_subject_masks_runs": _Parent({"r1": run}, {})})

    with pytest.raises(RefinedSubjectMaskMutationError, match="Approved"):
        migrate_unapproved_refined_subject_mask_to_editable_draft(root, "r1")


def test_migration_requires_exact_boolean_selector_eligibility() -> None:
    run = _Group(
        {
            "coordinate_contract": "canonical_v2",
            "refined_subject_mask_publication_owner": "a" * 32,
            "stage_selector_eligible": 1,
            "refined_subject_mask_review_status": {"state": "pending"},
        }
    )
    root = _Root({"refined_subject_masks_runs": _Parent({"r1": run}, {})})

    with pytest.raises(RefinedSubjectMaskMutationError, match="exact boolean"):
        migrate_unapproved_refined_subject_mask_to_editable_draft(root, "r1")
