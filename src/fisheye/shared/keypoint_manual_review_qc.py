"""Exact, replayable QC policy for manual keypoint review."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.keypoint_quality import (
    compute_geometry_metrics,
    resolve_head_triangle_for_labels,
)
from fisheye.shared.pose_schema import HEAD_TRIANGLE_KEYPOINT_LABELS
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


MANUAL_KEYPOINT_QC_POLICY_SCHEMA_ID = "palette.keypoint.manual_review_qc_policy"
MANUAL_KEYPOINT_QC_POLICY_SCHEMA_VERSION = 1
MANUAL_KEYPOINT_QC_EVALUATOR_ID = "palette.keypoint.manual_review_qc_evaluator"
MANUAL_KEYPOINT_QC_EVALUATOR_VERSION = 1
MANUAL_KEYPOINT_REVIEW_DERIVATION_SCHEMA_ID = (
    "palette.keypoint.manual_review_derivation"
)
MANUAL_KEYPOINT_REVIEW_DERIVATION_SCHEMA_VERSION = 1

DEFAULT_MANUAL_KEYPOINT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_MANUAL_KEYPOINT_MIN_TRIANGLE_ANGLE_DEG = 10.0
DEFAULT_MANUAL_KEYPOINT_MIN_TRIANGLE_AREA_PX2 = 100.0
DEFAULT_MANUAL_KEYPOINT_MAX_TRIANGLE_AREA_PX2: float | None = None
DEFAULT_MANUAL_KEYPOINT_REPLACEMENT_CONFIDENCE = 1.0

_SHA256_LENGTH = 64
_OUTPUT_RULES = {
    "refined_success": "any_valid_landmark",
    "confidence_valid": "all_landmarks_valid_finite_and_greater_equal_threshold",
    "geometry_valid": "head_triangle_finite_and_within_inclusive_bounds",
    "usable_keypoints": ("refined_success_and_confidence_valid_and_geometry_valid"),
}
_MANUAL_EDIT_SEMANTICS = {
    "replacement_confidence": DEFAULT_MANUAL_KEYPOINT_REPLACEMENT_CONFIDENCE,
    "cleared_landmark_confidence": "nan",
    "cleared_landmark_coordinates": "nan_nan",
    "validity_source": "finite_final_coordinates",
}


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value).strip()
    if len(text) != _SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
    return text


def _require_finite_nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True)
class ManualKeypointQcPolicy:
    """One exact skeleton-bound manual-review QC policy."""

    policy_id: str
    policy_version: int
    skeleton_id: str
    skeleton_digest: str
    keypoint_labels: tuple[str, ...]
    head_triangle_indices: tuple[int, int, int]
    confidence_threshold: float
    min_triangle_angle_deg: float
    min_triangle_area_px2: float
    max_triangle_area_px2: float | None
    replacement_confidence: float

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id).strip()
        skeleton_id = str(self.skeleton_id).strip()
        labels = tuple(str(value).strip() for value in self.keypoint_labels)
        indices = tuple(int(value) for value in self.head_triangle_indices)
        if not policy_id or not skeleton_id:
            raise ValueError("Manual keypoint QC policy and skeleton IDs are required.")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("Manual keypoint QC policy_version must be positive.")
        if (
            not labels
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                "Manual keypoint QC keypoint labels must be nonempty and unique."
            )
        if len(indices) != 3 or any(
            index < 0 or index >= len(labels) for index in indices
        ):
            raise ValueError("Manual keypoint QC head-triangle indices are invalid.")
        resolved = resolve_head_triangle_for_labels(
            labels,
            keypoint_count=len(labels),
            allow_legacy_3point_fallback=False,
        ).as_tuple
        if indices != resolved:
            raise ValueError(
                "Manual keypoint QC head-triangle indices differ from skeleton labels."
            )
        confidence_threshold = _require_finite_nonnegative(
            self.confidence_threshold, name="confidence_threshold"
        )
        min_angle = _require_finite_nonnegative(
            self.min_triangle_angle_deg, name="min_triangle_angle_deg"
        )
        min_area = _require_finite_nonnegative(
            self.min_triangle_area_px2, name="min_triangle_area_px2"
        )
        max_area = self.max_triangle_area_px2
        if max_area is not None:
            max_area = _require_finite_nonnegative(
                max_area, name="max_triangle_area_px2"
            )
            if max_area < min_area:
                raise ValueError("Maximum triangle area cannot be below its minimum.")
        replacement = _require_finite_nonnegative(
            self.replacement_confidence, name="replacement_confidence"
        )
        if replacement < confidence_threshold:
            raise ValueError(
                "Manual replacement confidence must satisfy its confidence threshold."
            )
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "skeleton_id", skeleton_id)
        object.__setattr__(
            self,
            "skeleton_digest",
            _require_sha256(self.skeleton_digest, name="skeleton_digest"),
        )
        object.__setattr__(self, "keypoint_labels", labels)
        object.__setattr__(self, "head_triangle_indices", indices)
        object.__setattr__(self, "confidence_threshold", confidence_threshold)
        object.__setattr__(self, "min_triangle_angle_deg", min_angle)
        object.__setattr__(self, "min_triangle_area_px2", min_area)
        object.__setattr__(self, "max_triangle_area_px2", max_area)
        object.__setattr__(self, "replacement_confidence", replacement)

    def _document_without_digest(self) -> dict[str, object]:
        return {
            "schema_id": MANUAL_KEYPOINT_QC_POLICY_SCHEMA_ID,
            "schema_version": MANUAL_KEYPOINT_QC_POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "evaluator": {
                "id": MANUAL_KEYPOINT_QC_EVALUATOR_ID,
                "version": MANUAL_KEYPOINT_QC_EVALUATOR_VERSION,
            },
            "skeleton": {
                "skeleton_id": self.skeleton_id,
                "skeleton_digest": self.skeleton_digest,
                "keypoint_labels": list(self.keypoint_labels),
                "head_triangle_labels": list(HEAD_TRIANGLE_KEYPOINT_LABELS),
                "head_triangle_indices": list(self.head_triangle_indices),
            },
            "thresholds": {
                "confidence_greater_equal": self.confidence_threshold,
                "min_triangle_angle_deg_inclusive": self.min_triangle_angle_deg,
                "min_triangle_area_px2_inclusive": self.min_triangle_area_px2,
                "max_triangle_area_px2_inclusive": self.max_triangle_area_px2,
            },
            "manual_edit_semantics": {
                **_MANUAL_EDIT_SEMANTICS,
                "replacement_confidence": self.replacement_confidence,
            },
            "output_rules": dict(_OUTPUT_RULES),
        }

    @property
    def policy_digest(self) -> str:
        return canonical_json_sha256(self._document_without_digest())

    def as_manifest(self) -> dict[str, object]:
        return {
            **self._document_without_digest(),
            "policy_digest": self.policy_digest,
        }


@dataclass(frozen=True)
class ManualKeypointQcResult:
    refined_success: bool
    confidence_valid: bool
    geometry_valid: bool
    usable_keypoints: bool
    triangle_area_px2: float
    minimum_triangle_angle_deg: float


def build_default_manual_keypoint_qc_policy(
    *,
    skeleton_id: str,
    skeleton_digest: str,
    keypoint_labels: Sequence[str],
) -> ManualKeypointQcPolicy:
    labels = tuple(str(value) for value in keypoint_labels)
    indices = resolve_head_triangle_for_labels(
        labels,
        keypoint_count=len(labels),
        allow_legacy_3point_fallback=False,
    ).as_tuple
    return ManualKeypointQcPolicy(
        policy_id="manual_pose_review_v1",
        policy_version=1,
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        keypoint_labels=labels,
        head_triangle_indices=indices,
        confidence_threshold=DEFAULT_MANUAL_KEYPOINT_CONFIDENCE_THRESHOLD,
        min_triangle_angle_deg=DEFAULT_MANUAL_KEYPOINT_MIN_TRIANGLE_ANGLE_DEG,
        min_triangle_area_px2=DEFAULT_MANUAL_KEYPOINT_MIN_TRIANGLE_AREA_PX2,
        max_triangle_area_px2=DEFAULT_MANUAL_KEYPOINT_MAX_TRIANGLE_AREA_PX2,
        replacement_confidence=DEFAULT_MANUAL_KEYPOINT_REPLACEMENT_CONFIDENCE,
    )


def manual_keypoint_qc_policy_from_manifest(
    value: object,
) -> ManualKeypointQcPolicy:
    expected = {
        "schema_id",
        "schema_version",
        "policy_id",
        "policy_version",
        "evaluator",
        "skeleton",
        "thresholds",
        "manual_edit_semantics",
        "output_rules",
        "policy_digest",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Manual keypoint QC policy envelope is not exact.")
    if (
        value.get("schema_id") != MANUAL_KEYPOINT_QC_POLICY_SCHEMA_ID
        or value.get("schema_version") != MANUAL_KEYPOINT_QC_POLICY_SCHEMA_VERSION
        or value.get("evaluator")
        != {
            "id": MANUAL_KEYPOINT_QC_EVALUATOR_ID,
            "version": MANUAL_KEYPOINT_QC_EVALUATOR_VERSION,
        }
        or value.get("output_rules") != _OUTPUT_RULES
    ):
        raise ValueError("Manual keypoint QC policy identity or rules are invalid.")
    skeleton = value.get("skeleton")
    thresholds = value.get("thresholds")
    semantics = value.get("manual_edit_semantics")
    if (
        not isinstance(skeleton, Mapping)
        or set(skeleton)
        != {
            "skeleton_id",
            "skeleton_digest",
            "keypoint_labels",
            "head_triangle_labels",
            "head_triangle_indices",
        }
        or skeleton.get("head_triangle_labels") != list(HEAD_TRIANGLE_KEYPOINT_LABELS)
        or not isinstance(thresholds, Mapping)
        or set(thresholds)
        != {
            "confidence_greater_equal",
            "min_triangle_angle_deg_inclusive",
            "min_triangle_area_px2_inclusive",
            "max_triangle_area_px2_inclusive",
        }
        or not isinstance(semantics, Mapping)
        or set(semantics) != set(_MANUAL_EDIT_SEMANTICS)
        or semantics.get("cleared_landmark_confidence") != "nan"
        or semantics.get("cleared_landmark_coordinates") != "nan_nan"
        or semantics.get("validity_source") != "finite_final_coordinates"
    ):
        raise ValueError("Manual keypoint QC policy nested document is not exact.")
    policy = ManualKeypointQcPolicy(
        policy_id=str(value.get("policy_id") or ""),
        policy_version=value.get("policy_version"),  # type: ignore[arg-type]
        skeleton_id=str(skeleton.get("skeleton_id") or ""),
        skeleton_digest=str(skeleton.get("skeleton_digest") or ""),
        keypoint_labels=tuple(skeleton.get("keypoint_labels") or ()),
        head_triangle_indices=tuple(skeleton.get("head_triangle_indices") or ()),  # type: ignore[arg-type]
        confidence_threshold=thresholds.get("confidence_greater_equal"),  # type: ignore[arg-type]
        min_triangle_angle_deg=thresholds.get("min_triangle_angle_deg_inclusive"),  # type: ignore[arg-type]
        min_triangle_area_px2=thresholds.get("min_triangle_area_px2_inclusive"),  # type: ignore[arg-type]
        max_triangle_area_px2=thresholds.get("max_triangle_area_px2_inclusive"),  # type: ignore[arg-type]
        replacement_confidence=semantics.get("replacement_confidence"),  # type: ignore[arg-type]
    )
    if dict(value) != policy.as_manifest():
        raise ValueError("Manual keypoint QC policy differs from canonical form.")
    return policy


def evaluate_manual_keypoint_qc(
    points_xy: Any,
    confidences: Any,
    *,
    policy: ManualKeypointQcPolicy,
) -> ManualKeypointQcResult:
    points = np.asarray(points_xy, dtype=np.float64)
    confidence = np.asarray(confidences, dtype=np.float64)
    expected_points = (len(policy.keypoint_labels), 2)
    expected_confidence = (len(policy.keypoint_labels),)
    if points.shape != expected_points or confidence.shape != expected_confidence:
        raise ValueError(
            "Manual keypoint QC input shapes differ from the bound skeleton."
        )
    point_valid = np.all(np.isfinite(points), axis=1)
    success = bool(np.any(point_valid))
    confidence_valid = bool(
        np.all(point_valid)
        and np.all(np.isfinite(confidence))
        and np.all(confidence >= policy.confidence_threshold)
    )
    triangle = compute_geometry_metrics(points[list(policy.head_triangle_indices)])
    max_ok = (
        policy.max_triangle_area_px2 is None
        or triangle.area <= policy.max_triangle_area_px2
    )
    geometry_valid = bool(
        np.isfinite(triangle.min_angle)
        and np.isfinite(triangle.area)
        and triangle.min_angle >= policy.min_triangle_angle_deg
        and triangle.area >= policy.min_triangle_area_px2
        and max_ok
    )
    return ManualKeypointQcResult(
        refined_success=success,
        confidence_valid=confidence_valid,
        geometry_valid=geometry_valid,
        usable_keypoints=success and confidence_valid and geometry_valid,
        triangle_area_px2=float(triangle.area),
        minimum_triangle_angle_deg=float(triangle.min_angle),
    )


def build_manual_keypoint_review_derivation(
    *,
    base_run_path: str,
    delta_run: str,
    generation: str,
    generation_sha256: str,
    overlay_sha256: str,
    partition_count: int,
    event_count: int,
    policy: ManualKeypointQcPolicy,
) -> dict[str, object]:
    document = {
        "schema_id": MANUAL_KEYPOINT_REVIEW_DERIVATION_SCHEMA_ID,
        "schema_version": MANUAL_KEYPOINT_REVIEW_DERIVATION_SCHEMA_VERSION,
        "base_run_path": str(base_run_path),
        "delta_run": str(delta_run),
        "generation": str(generation),
        "generation_sha256": _require_sha256(
            generation_sha256, name="generation_sha256"
        ),
        "overlay_sha256": _require_sha256(overlay_sha256, name="overlay_sha256"),
        "partition_count": int(partition_count),
        "event_count": int(event_count),
        "review_qc_policy_digest": policy.policy_digest,
        "review_qc_policy": policy.as_manifest(),
    }
    if (
        not document["base_run_path"]
        or not document["delta_run"]
        or not document["generation"]
    ):
        raise ValueError("Manual keypoint review derivation identities are required.")
    if document["partition_count"] < 0 or document["event_count"] < 0:
        raise ValueError("Manual keypoint review derivation counts cannot be negative.")
    return document


def validate_manual_keypoint_review_derivation(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("manual keypoint review derivation must be an object",)
    try:
        policy = manual_keypoint_qc_policy_from_manifest(value.get("review_qc_policy"))
        expected = build_manual_keypoint_review_derivation(
            base_run_path=str(value.get("base_run_path") or ""),
            delta_run=str(value.get("delta_run") or ""),
            generation=str(value.get("generation") or ""),
            generation_sha256=str(value.get("generation_sha256") or ""),
            overlay_sha256=str(value.get("overlay_sha256") or ""),
            partition_count=value.get("partition_count"),  # type: ignore[arg-type]
            event_count=value.get("event_count"),  # type: ignore[arg-type]
            policy=policy,
        )
    except (TypeError, ValueError) as exc:
        return (str(exc),)
    if dict(value) != expected:
        return ("manual keypoint review derivation is not canonical",)
    return ()


__all__ = [
    "DEFAULT_MANUAL_KEYPOINT_CONFIDENCE_THRESHOLD",
    "DEFAULT_MANUAL_KEYPOINT_MAX_TRIANGLE_AREA_PX2",
    "DEFAULT_MANUAL_KEYPOINT_MIN_TRIANGLE_ANGLE_DEG",
    "DEFAULT_MANUAL_KEYPOINT_MIN_TRIANGLE_AREA_PX2",
    "DEFAULT_MANUAL_KEYPOINT_REPLACEMENT_CONFIDENCE",
    "MANUAL_KEYPOINT_QC_POLICY_SCHEMA_ID",
    "MANUAL_KEYPOINT_QC_POLICY_SCHEMA_VERSION",
    "MANUAL_KEYPOINT_REVIEW_DERIVATION_SCHEMA_ID",
    "MANUAL_KEYPOINT_REVIEW_DERIVATION_SCHEMA_VERSION",
    "ManualKeypointQcPolicy",
    "ManualKeypointQcResult",
    "build_default_manual_keypoint_qc_policy",
    "build_manual_keypoint_review_derivation",
    "evaluate_manual_keypoint_qc",
    "manual_keypoint_qc_policy_from_manifest",
    "validate_manual_keypoint_review_derivation",
]
