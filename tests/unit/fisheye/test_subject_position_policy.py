from __future__ import annotations

import pytest

from fisheye.shared.subject_position_expression import ESTIMATOR_PROFILE_RECORDS
from fisheye.shared.subject_position_policy import (
    SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID,
    canonicalize_subject_position_selection_policy,
    get_subject_position_selection_policy,
    subject_position_selection_policy_digest,
)


def test_canary_policy_has_no_default_fallback_or_selector_eligibility() -> None:
    policy = get_subject_position_selection_policy(
        SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID
    )

    assert policy["selection_mode"] == "explicit_provider_only"
    assert policy["default_estimator_id"] is None
    assert policy["fallback"] == "none"
    assert policy["selector_eligible"] is False
    assert policy["promotion_evidence"] == "required"
    assert policy["allowed_estimator_ids"] == sorted(ESTIMATOR_PROFILE_RECORDS)


def test_canary_policy_digest_is_canonical_and_detached() -> None:
    policy = get_subject_position_selection_policy(
        SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID
    )
    expected = subject_position_selection_policy_digest(policy)

    policy["selection_mode"] = "implicit_default"
    fresh = get_subject_position_selection_policy(
        SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID
    )

    assert subject_position_selection_policy_digest(fresh) == expected
    with pytest.raises(ValueError, match="exact Phase 2 canary policy"):
        canonicalize_subject_position_selection_policy(policy)


@pytest.mark.parametrize(
    "mutation",
    (
        {"default_estimator_id": "detection_bbox_centroid.v1"},
        {"fallback": "detection_bbox_centroid.v1"},
        {"selector_eligible": True},
        {"promotion_evidence": "optional"},
    ),
)
def test_canary_policy_rejects_silent_activation_or_fallback(
    mutation: dict[str, object],
) -> None:
    policy = get_subject_position_selection_policy(
        SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID
    )
    policy.update(mutation)

    with pytest.raises(ValueError, match="exact Phase 2 canary policy"):
        canonicalize_subject_position_selection_policy(policy)


def test_selection_policy_has_no_implicit_lookup_default() -> None:
    with pytest.raises(ValueError, match="Unknown subject-position selection policy"):
        get_subject_position_selection_policy("detection_compatibility.v1")
