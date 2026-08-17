"""Versioned provider-selection policy for Phase 2 subject-position canaries.

Estimator profiles define how one row is evaluated.  This module defines the
orthogonal policy for whether a materialized provider may be selected.  Phase
2 intentionally has no default provider and cannot activate a selector.
"""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Final, Mapping

from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
)
from fisheye.shared.subject_position_types import OBSERVATION_POSITION_ROW_AXIS
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)


SUBJECT_POSITION_SELECTION_POLICY_SCHEMA_ID: Final = (
    "palette.subject_position_provider_selection_policy"
)
SUBJECT_POSITION_SELECTION_POLICY_SCHEMA_VERSION: Final = 1
SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID: Final = (
    "subject_position_canary_no_default.v1"
)
CURRENT_CANONICAL_SOURCE_POLICY_ID: Final = (
    "current_complete_selector_eligible_canonical_coordinate_run.v1"
)

_INITIAL_ESTIMATOR_IDS: Final = tuple(
    sorted(
        (
            DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
            KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
            MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
            SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
        )
    )
)


def _canary_policy_record() -> dict[str, Any]:
    return {
        "schema_id": SUBJECT_POSITION_SELECTION_POLICY_SCHEMA_ID,
        "schema_version": SUBJECT_POSITION_SELECTION_POLICY_SCHEMA_VERSION,
        "policy_id": SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID,
        "policy_version": 1,
        "row_axis": OBSERVATION_POSITION_ROW_AXIS,
        "selection_mode": "explicit_provider_only",
        "allowed_estimator_ids": list(_INITIAL_ESTIMATOR_IDS),
        "default_estimator_id": None,
        "fallback": "none",
        "source_authority_policy_id": CURRENT_CANONICAL_SOURCE_POLICY_ID,
        "promotion_evidence": "required",
        "selector_eligible": False,
    }


SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY: Final[Mapping[str, Any]] = (
    MappingProxyType(
        {
            **_canary_policy_record(),
            "allowed_estimator_ids": _INITIAL_ESTIMATOR_IDS,
        }
    )
)


def canonicalize_subject_position_selection_policy(
    value: object,
) -> dict[str, Any]:
    """Require the exact Phase 2 no-default selection policy."""

    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError("Subject-position selection policy must be an object.")
    canonical = _canary_policy_record()
    if canonical_json_bytes(value) != canonical_json_bytes(canonical):
        raise ValueError(
            "Subject-position selection policy differs from the exact Phase 2 "
            "canary policy."
        )
    return deepcopy(canonical)


def subject_position_selection_policy_digest(value: object) -> str:
    """Return the canonical digest of one validated policy."""

    return canonical_json_sha256(
        canonicalize_subject_position_selection_policy(value)
    )


def get_subject_position_selection_policy(policy_id: str) -> dict[str, Any]:
    """Resolve the sole Phase 2 policy explicitly; no default is implied."""

    if policy_id != SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID:
        raise ValueError(f"Unknown subject-position selection policy {policy_id!r}.")
    return _canary_policy_record()


__all__ = [
    "CURRENT_CANONICAL_SOURCE_POLICY_ID",
    "SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY",
    "SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID",
    "SUBJECT_POSITION_SELECTION_POLICY_SCHEMA_ID",
    "SUBJECT_POSITION_SELECTION_POLICY_SCHEMA_VERSION",
    "canonicalize_subject_position_selection_policy",
    "get_subject_position_selection_policy",
    "subject_position_selection_policy_digest",
]
