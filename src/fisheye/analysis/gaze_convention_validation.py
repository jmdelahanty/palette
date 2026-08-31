"""Read-only validation gates for eye-gaze and chaser-bearing conventions.

This module deliberately validates the persisted geometric identities before a
consumer computes gaze-to-object error.  Compact eye-angle runs are sampled by
physical row windows so a million-frame recording does not require loading the
entire packed ``frame_angles`` or ``roi_angles`` table.

The canonical comparison fields are ``left/right_gaze_signed_deg`` and
egocentric chaser ``bearing_deg``.  Both are fish-body-frame angles with zero
forward and positive values toward anatomical left.  The per-eye
``left/right_eye_angle_deg`` fields are nasal-positive biological
presentations and must not be compared directly with object bearing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.eye_angle_io import resolve_eye_angle_run  # noqa: E402
from fisheye.analysis.eye_angle_schema import (  # noqa: E402
    eye_angle_dimensions_from_run_attrs,
    validate_eye_angle_compact_run,
)
from fisheye.analysis.eye_angle_storage import (  # noqa: E402
    validate_eye_angle_candidate_storage,
)
from fisheye.shared.eye_geometry_source import (  # noqa: E402
    EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
    EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
    EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID,
    EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_VERSION,
    EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_SCHEMA_VERSION,
    EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_SCOPE,
    MaskStoreChannelSelectionArray,
    resolve_eye_geometry_source,
)
from fisheye.shared.json_safety import decode_null_terminated_text  # noqa: E402
from fisheye.shared.mask_store import open_mask_store  # noqa: E402
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run  # noqa: E402
from fisheye.shared.subject_shape_coordinate_publication import (  # noqa: E402
    require_translation_only_subject_shape_placement,
)
from fisheye.shared.zarr.subject_shape_bundle_source import (  # noqa: E402
    BoundSubjectShapeBundleSource,
    require_bound_subject_shape_bundle_source,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256  # noqa: E402
from fisheye.shared.zarr.metadata_equivalence import (  # noqa: E402
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root  # noqa: E402
from fisheye.shared.zarr_run_completion import (  # noqa: E402
    is_run_complete_in_parent,
)


SCHEMA_ID = "palette.gaze_convention_validation.v1"
SCHEMA_VERSION = 1
EXPECTED_BODY_FRAME_CONVENTION = "math_ccw_degrees_after_y_flip"
EXPECTED_GAZE_SIGN_CONVENTION = "positive_anatomical_left"
DEFAULT_WINDOWS = 12
DEFAULT_ROWS_PER_WINDOW = 256
_STAGED_SUBJECT_SHAPE_SOURCE_AUTHORITY_MODE = "digest_bound_staged_subset"
_STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_FIELDS = {
    "schema_id",
    "schema_version",
    "authority_scope",
    "source_subject_shape_run",
    "source_subject_shape_run_ref",
    "row_count",
    "canonical_publication",
    "source_contract_attrs",
    "allowed_arrays",
    "closed_array_inventory",
    "normal_reader_authority",
    "candidate_admission",
    "record_sha256",
}


def wrap_degrees_signed(values: np.ndarray | float) -> np.ndarray:
    """Wrap degrees into ``[-180, 180)``."""

    array = np.asarray(values, dtype=np.float64)
    return ((array + 180.0) % 360.0) - 180.0


def body_frame_angles_from_vectors(
    vectors_xy: np.ndarray,
    forward_axis_xy: np.ndarray,
    left_axis_xy: np.ndarray,
) -> np.ndarray:
    """Return anatomical-left-positive body-frame angles for image-xy vectors."""

    vectors = np.asarray(vectors_xy, dtype=np.float64)
    forward = np.asarray(forward_axis_xy, dtype=np.float64)
    left = np.asarray(left_axis_xy, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("vectors_xy must have shape (row, 2).")
    if forward.shape != vectors.shape or left.shape != vectors.shape:
        raise ValueError("vectors and body-frame axes must have matching (row, 2) shapes.")
    forward_component = np.einsum("ij,ij->i", vectors, forward)
    left_component = np.einsum("ij,ij->i", vectors, left)
    return np.rad2deg(np.arctan2(left_component, forward_component))


def expected_gaze_from_major(
    left_major_signed_deg: np.ndarray,
    right_major_signed_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed gaze angles from resolved major-axis angles."""

    return (
        wrap_degrees_signed(np.asarray(left_major_signed_deg, dtype=np.float64) + 90.0),
        wrap_degrees_signed(np.asarray(right_major_signed_deg, dtype=np.float64) - 90.0),
    )


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    sample_count: int
    tolerance: Optional[float]
    metrics: Mapping[str, float | int | str | bool | None]
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "sample_count": int(self.sample_count),
            "tolerance": self.tolerance,
            "metrics": dict(self.metrics),
            "explanation": self.explanation,
        }


def _finite_error_stats(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    circular: bool,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    exp = np.asarray(expected, dtype=np.float64).reshape(-1)
    if obs.shape != exp.shape:
        raise ValueError(f"Observed and expected shapes differ: {obs.shape} != {exp.shape}.")
    finite = np.isfinite(obs) & np.isfinite(exp)
    error = np.full(obs.shape, np.nan, dtype=np.float64)
    if np.any(finite):
        delta = obs[finite] - exp[finite]
        error[finite] = wrap_degrees_signed(delta) if circular else delta
        absolute = np.abs(error[finite])
        stats: dict[str, float | int | None] = {
            "median_abs_error": float(np.median(absolute)),
            "p99_abs_error": float(np.quantile(absolute, 0.99)),
            "max_abs_error": float(np.max(absolute)),
            "finite_count": int(absolute.size),
        }
    else:
        stats = {
            "median_abs_error": None,
            "p99_abs_error": None,
            "max_abs_error": None,
            "finite_count": 0,
        }
    return error, stats


def _identity_check(
    name: str,
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance: float,
    circular: bool = True,
    explanation: str,
) -> ValidationCheck:
    _error, stats = _finite_error_stats(observed, expected, circular=circular)
    count = int(stats["finite_count"] or 0)
    p99 = stats["p99_abs_error"]
    passed = count > 0 and p99 is not None and float(p99) <= float(tolerance)
    return ValidationCheck(
        name=name,
        passed=passed,
        sample_count=count,
        tolerance=float(tolerance),
        metrics=stats,
        explanation=explanation,
    )


def validate_gaze_geometry_arrays(
    *,
    left_major_signed_deg: np.ndarray,
    right_major_signed_deg: np.ndarray,
    left_eye_angle_deg: np.ndarray,
    right_eye_angle_deg: np.ndarray,
    vergence_eye_angle_deg: np.ndarray,
    left_gaze_signed_deg: np.ndarray,
    right_gaze_signed_deg: np.ndarray,
    left_gaze_xy: np.ndarray,
    right_gaze_xy: np.ndarray,
    forward_axis_xy: np.ndarray,
    left_axis_xy: np.ndarray,
    heading_deg: np.ndarray,
    valid: Optional[np.ndarray] = None,
    angle_tolerance_deg: float = 0.01,
    axis_tolerance: float = 1.0e-4,
) -> tuple[ValidationCheck, ...]:
    """Validate body-frame, eye identity, sign, and gaze-vector identities."""

    forward = np.asarray(forward_axis_xy, dtype=np.float64)
    left_axis = np.asarray(left_axis_xy, dtype=np.float64)
    if forward.ndim != 2 or forward.shape[1] != 2 or left_axis.shape != forward.shape:
        raise ValueError("forward_axis_xy and left_axis_xy must have matching (row, 2) shapes.")
    row_count = int(forward.shape[0])
    valid_mask = np.ones(row_count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if valid_mask.shape[0] != row_count:
        raise ValueError("valid must have the same row count as the body-frame arrays.")

    finite_axes = valid_mask & np.all(np.isfinite(forward), axis=1) & np.all(np.isfinite(left_axis), axis=1)
    forward_norm = np.linalg.norm(forward, axis=1)
    left_norm = np.linalg.norm(left_axis, axis=1)
    dot = np.einsum("ij,ij->i", forward, left_axis)
    determinant = forward[:, 0] * left_axis[:, 1] - forward[:, 1] * left_axis[:, 0]
    reconstructed_heading = np.rad2deg(np.arctan2(-forward[:, 1], forward[:, 0]))

    def axis_check(name: str, error: np.ndarray, explanation: str) -> ValidationCheck:
        values = np.abs(np.asarray(error, dtype=np.float64)[finite_axes])
        metrics: dict[str, float | int | None] = {
            "median_abs_error": float(np.median(values)) if values.size else None,
            "p99_abs_error": float(np.quantile(values, 0.99)) if values.size else None,
            "max_abs_error": float(np.max(values)) if values.size else None,
            "finite_count": int(values.size),
        }
        p99 = metrics["p99_abs_error"]
        return ValidationCheck(
            name=name,
            passed=bool(values.size and p99 is not None and float(p99) <= axis_tolerance),
            sample_count=int(values.size),
            tolerance=float(axis_tolerance),
            metrics=metrics,
            explanation=explanation,
        )

    checks: list[ValidationCheck] = [
        axis_check(
            "body_axis_unit_length",
            np.maximum(np.abs(forward_norm - 1.0), np.abs(left_norm - 1.0)),
            "Forward and anatomical-left axes must be unit vectors.",
        ),
        axis_check(
            "body_axes_orthogonal",
            dot,
            "Forward and anatomical-left axes must be orthogonal.",
        ),
    ]

    handed_values = determinant[finite_axes]
    wrong_handed = int(np.sum(handed_values >= 0.0))
    checks.append(
        ValidationCheck(
            name="body_frame_image_xy_handedness",
            passed=bool(handed_values.size and wrong_handed == 0),
            sample_count=int(handed_values.size),
            tolerance=0.0,
            metrics={
                "wrong_handed_count": wrong_handed,
                "wrong_handed_fraction": float(wrong_handed / handed_values.size) if handed_values.size else None,
                "median_determinant": float(np.median(handed_values)) if handed_values.size else None,
            },
            explanation=(
                "In image xy, anatomical left is the clockwise perpendicular to forward; "
                "a non-negative determinant indicates a likely left/right identity or frame error."
            ),
        )
    )
    checks.append(
        _identity_check(
            "heading_from_forward_axis",
            np.asarray(heading_deg)[valid_mask],
            reconstructed_heading[valid_mask],
            tolerance=angle_tolerance_deg,
            explanation="Heading must be atan2(-forward_y, forward_x) in math-y-up degrees.",
        )
    )

    left_major = np.asarray(left_major_signed_deg, dtype=np.float64)
    right_major = np.asarray(right_major_signed_deg, dtype=np.float64)
    expected_left_gaze, expected_right_gaze = expected_gaze_from_major(left_major, right_major)
    checks.extend(
        (
            _identity_check(
                "left_eye_angle_nasal_sign",
                left_eye_angle_deg,
                -left_major,
                tolerance=angle_tolerance_deg,
                explanation="Left Bianco/Engert eye angle must be -left major-axis angle.",
            ),
            _identity_check(
                "right_eye_angle_nasal_sign",
                right_eye_angle_deg,
                right_major,
                tolerance=angle_tolerance_deg,
                explanation="Right Bianco/Engert eye angle must equal the right major-axis angle.",
            ),
            _identity_check(
                "vergence_eye_angle_sum",
                vergence_eye_angle_deg,
                np.asarray(left_eye_angle_deg) + np.asarray(right_eye_angle_deg),
                tolerance=angle_tolerance_deg,
                explanation="Signed convergence is the sum of the two nasal-positive eye angles.",
            ),
            _identity_check(
                "left_gaze_from_resolved_major",
                left_gaze_signed_deg,
                expected_left_gaze,
                tolerance=angle_tolerance_deg,
                explanation="Left directed gaze must be wrap(left major + 90 degrees).",
            ),
            _identity_check(
                "right_gaze_from_resolved_major",
                right_gaze_signed_deg,
                expected_right_gaze,
                tolerance=angle_tolerance_deg,
                explanation="Right directed gaze must be wrap(right major - 90 degrees).",
            ),
            _identity_check(
                "left_gaze_vector_body_angle",
                left_gaze_signed_deg,
                body_frame_angles_from_vectors(left_gaze_xy, forward, left_axis),
                tolerance=angle_tolerance_deg,
                explanation="The stored left ROI/image-space gaze vector must reconstruct its body-frame angle.",
            ),
            _identity_check(
                "right_gaze_vector_body_angle",
                right_gaze_signed_deg,
                body_frame_angles_from_vectors(right_gaze_xy, forward, left_axis),
                tolerance=angle_tolerance_deg,
                explanation="The stored right ROI/image-space gaze vector must reconstruct its body-frame angle.",
            ),
        )
    )
    return tuple(checks)


def _group_attrs(group: Any) -> dict[str, Any]:
    try:
        return dict(group.attrs)
    except Exception:
        return {}


def _decode_channel_names(index_group: zarr.Group, expected_count: int) -> list[str]:
    if "name" not in index_group:
        raise ValueError("Compact channel index is missing its name array.")
    values = np.asarray(index_group["name"][:])
    names = [str(decode_null_terminated_text(value)) for value in values]
    if len(names) != int(expected_count) or any(not name for name in names):
        raise ValueError(
            f"Channel index names {len(names)} fields for expected count {expected_count}."
        )
    return names


def _sample_slices(row_count: int, *, windows: int, rows_per_window: int) -> tuple[slice, ...]:
    if row_count <= 0:
        return ()
    windows = max(1, min(int(windows), row_count))
    width = max(1, min(int(rows_per_window), row_count))
    max_start = max(0, row_count - width)
    starts = np.rint(np.linspace(0, max_start, windows)).astype(np.int64)
    unique_starts = list(dict.fromkeys(int(value) for value in starts.tolist()))
    return tuple(slice(start, min(row_count, start + width)) for start in unique_starts)


def _read_packed_columns(
    data: zarr.Array,
    names: Sequence[str],
    requested: Sequence[str],
    slices: Sequence[slice],
) -> dict[str, np.ndarray]:
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"Packed array {data.path!r} is missing required channels: {missing}.")
    indices = [names.index(name) for name in requested]
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in requested}
    for row_slice in slices:
        # Packed scalar arrays use a single all-channel physical chunk.  Reading
        # one bounded row window and then selecting columns minimizes network
        # round trips without pretending the column table is independently chunked.
        block = np.asarray(data[row_slice, :])
        for name, channel_index in zip(requested, indices):
            pieces[name].append(np.asarray(block[:, channel_index]))
    return {
        name: np.concatenate(values, axis=0) if values else np.asarray([], dtype=data.dtype)
        for name, values in pieces.items()
    }


def _read_row_windows(data: Any, slices: Sequence[slice]) -> np.ndarray:
    values = [np.asarray(data[row_slice]) for row_slice in slices]
    if not values:
        shape = tuple(int(v) for v in getattr(data, "shape", ()))[1:]
        return np.empty((0, *shape), dtype=getattr(data, "dtype", np.float32))
    return np.concatenate(values, axis=0)


def _sample_row_indices(slices: Sequence[slice]) -> np.ndarray:
    pieces = [np.arange(int(row_slice.start or 0), int(row_slice.stop or 0), dtype=np.int64) for row_slice in slices]
    return np.concatenate(pieces) if pieces else np.asarray([], dtype=np.int64)


def _resolve_eye_run(
    root: zarr.Group,
    run_name: Optional[str],
    *,
    legacy_compatibility: bool = False,
    allow_ineligible_candidate: bool = False,
) -> tuple[str, zarr.Group]:
    if allow_ineligible_candidate:
        if type(run_name) is not str or not run_name.strip():
            raise ValueError(
                "Selector-ineligible eye-angle review requires one explicit run name."
            )
        normalized = run_name.strip().strip("/")
        prefix = "analysis/eye_angle_runs/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
        if (
            not normalized
            or "/" in normalized
            or normalized
            in {"latest", "latest_complete", "selected", "current", ".", ".."}
        ):
            raise ValueError(
                "Selector-ineligible eye-angle review requires one exact child name."
            )
        parent = root.get("analysis/eye_angle_runs")
        if parent is None or normalized not in parent:
            raise ValueError(f"Explicit eye-angle candidate {normalized!r} is absent.")
        run_group = parent[normalized]
        if not is_run_complete_in_parent(parent, run_group, legacy_default=False):
            raise ValueError("Explicit eye-angle candidate is not complete.")
        if run_group.attrs.get("stage_selector_eligible") is not False:
            raise ValueError(
                "Explicit candidate review requires stage_selector_eligible=false."
            )
        compact_issues = validate_eye_angle_compact_run(run_group)
        if compact_issues:
            raise ValueError(
                "Explicit eye-angle candidate is not exact compact-v7: "
                + "; ".join(
                    f"{issue.code}:{issue.path}:{issue.message}"
                    for issue in compact_issues
                )
            )
        dimensions = eye_angle_dimensions_from_run_attrs(_group_attrs(run_group))
        storage_issues = validate_eye_angle_candidate_storage(
            run_group,
            dimensions=dimensions,
        )
        if storage_issues:
            raise ValueError(
                "Explicit eye-angle candidate storage is invalid: "
                + "; ".join(
                    f"{issue.code}:{issue.path}:{issue.message}"
                    for issue in storage_issues
                )
            )
        return normalized, run_group
    run_group, resolved, _run_path = resolve_eye_angle_run(
        root,
        run_name,
        legacy_compatibility=legacy_compatibility,
    )
    return resolved, run_group


def _load_compact_sample(
    run_group: zarr.Group,
    *,
    windows: int,
    rows_per_window: int,
) -> dict[str, np.ndarray]:
    attrs = _group_attrs(run_group)
    layout = str(attrs.get("layout") or attrs.get("storage_layout") or "")
    if layout != "compact_dense_v2":
        raise ValueError(
            f"Convention validator requires compact_dense_v2; got {layout!r}. "
            "Recompute this validation canary with the modern eye-angle contract."
        )
    required_arrays = (
        "roi_angles",
        "roi_vectors",
        "roi_qa",
        "angle_channel_index",
        "vector_channel_index",
        "qa_channel_index",
        "support/body_frame/forward_axis_xy",
        "support/body_frame/left_axis_xy",
        "support/body_frame/heading_deg",
        "support/body_frame/origin_xy",
        "support/body_frame/valid",
        "support/frame_indices",
    )
    missing_arrays = [path for path in required_arrays if run_group.get(path) is None]
    if missing_arrays:
        raise ValueError(f"Eye-angle run is missing required arrays/groups: {missing_arrays}.")

    angle_data = run_group["roi_angles"]
    vector_data = run_group["roi_vectors"]
    qa_data = run_group["roi_qa"]
    row_count = int(angle_data.shape[0])
    if int(vector_data.shape[0]) != row_count or int(qa_data.shape[0]) != row_count:
        raise ValueError("Eye-angle packed ROI arrays have inconsistent row counts.")
    slices = _sample_slices(row_count, windows=windows, rows_per_window=rows_per_window)
    angle_names = _decode_channel_names(run_group["angle_channel_index"], int(angle_data.shape[1]))
    vector_names = _decode_channel_names(run_group["vector_channel_index"], int(vector_data.shape[1]))
    qa_names = _decode_channel_names(run_group["qa_channel_index"], int(qa_data.shape[1]))

    angle_fields = (
        "left_major_signed_deg",
        "right_major_signed_deg",
        "left_eye_angle_deg",
        "right_eye_angle_deg",
        "vergence_eye_angle_deg",
        "left_gaze_signed_deg",
        "right_gaze_signed_deg",
    )
    output = _read_packed_columns(angle_data, angle_names, angle_fields, slices)
    missing_vectors = [name for name in ("left_gaze_xy", "right_gaze_xy") if name not in vector_names]
    if missing_vectors:
        raise ValueError(f"Packed vector array is missing required channels: {missing_vectors}.")
    vector_indices = [vector_names.index("left_gaze_xy"), vector_names.index("right_gaze_xy")]
    vector_pieces: list[np.ndarray] = []
    for row_slice in slices:
        vector_pieces.append(np.asarray(vector_data[row_slice, :, :])[:, vector_indices, :])
    vectors = np.concatenate(vector_pieces, axis=0)
    output["left_gaze_xy"] = vectors[:, 0, :]
    output["right_gaze_xy"] = vectors[:, 1, :]

    qa_fields = ("valid_frame", "major_axis_marginal")
    output.update(_read_packed_columns(qa_data, qa_names, qa_fields, slices))
    for name in ("forward_axis_xy", "left_axis_xy", "heading_deg", "origin_xy", "valid"):
        output[name] = _read_row_windows(run_group[f"support/body_frame/{name}"], slices)
    output["frame_indices"] = _read_row_windows(run_group["support/frame_indices"], slices).astype(np.int64)
    output["sample_row_indices"] = _sample_row_indices(slices)
    output["row_count"] = np.asarray(row_count, dtype=np.int64)
    return output


def _metadata_checks(attrs: Mapping[str, Any]) -> tuple[ValidationCheck, ...]:
    body_convention = str(attrs.get("body_frame_angle_convention") or "")
    variant_schema = attrs.get("eye_angle_variant_schema")
    gaze_sign = ""
    gaze_frame = ""
    if isinstance(variant_schema, Mapping):
        representations = variant_schema.get("representations")
        if isinstance(representations, Mapping):
            gaze = representations.get("gaze")
            if isinstance(gaze, Mapping):
                gaze_sign = str(gaze.get("sign_convention") or "")
                gaze_frame = str(gaze.get("coordinate_frame") or "")
    return (
        ValidationCheck(
            name="body_frame_metadata_convention",
            passed=body_convention == EXPECTED_BODY_FRAME_CONVENTION,
            sample_count=1,
            tolerance=None,
            metrics={"observed": body_convention, "expected": EXPECTED_BODY_FRAME_CONVENTION},
            explanation="Eye body-frame metadata must explicitly declare math-y-up CCW degrees after y flip.",
        ),
        ValidationCheck(
            name="gaze_metadata_convention",
            passed=gaze_sign == EXPECTED_GAZE_SIGN_CONVENTION and gaze_frame == "fish_body_frame",
            sample_count=1,
            tolerance=None,
            metrics={
                "observed_sign": gaze_sign,
                "expected_sign": EXPECTED_GAZE_SIGN_CONVENTION,
                "observed_coordinate_frame": gaze_frame,
                "expected_coordinate_frame": "fish_body_frame",
            },
            explanation="Only common fish-body-frame gaze fields are valid inputs to chaser lock-on geometry.",
        ),
    )


def _resolve_keypoint_group(root: zarr.Group, run_attrs: Mapping[str, Any]) -> tuple[str, zarr.Group]:
    keypoint_run = resolve_source_keypoints_run(run_attrs)
    if not keypoint_run:
        raise ValueError("Eye-angle run does not identify its source keypoint run.")
    for parent_name in ("refined_keypoints_runs", "keypoints_runs"):
        parent = root.get(parent_name)
        if parent is not None and keypoint_run in parent:
            return str(keypoint_run), parent[str(keypoint_run)]
    raise ValueError(f"Source keypoint run {keypoint_run!r} is not present.")


def _resolve_roi_images(root: zarr.Group, keypoint_group: zarr.Group) -> Any | None:
    if "roi_images" in keypoint_group:
        return keypoint_group["roi_images"]
    crop_run = str(keypoint_group.attrs.get("source_crop_run") or "")
    if crop_run and root.get(f"crop_runs/{crop_run}/roi_images") is not None:
        return root[f"crop_runs/{crop_run}/roi_images"]
    # Geometry-only crop authorities intentionally omit pixel payloads.  The
    # convention review can still show masks, ellipse centers, and body/gaze
    # axes in their shared ROI coordinate system.
    return None


def _candidate_subject_shape_review_admission(
    run_attrs: Mapping[str, Any],
    *,
    subject_shape_run: str,
) -> Mapping[str, Any] | None:
    """Return the exact embedded candidate admission or fail closed.

    A selector-ineligible eye-angle candidate may have been computed from an
    exact selector-ineligible subject-shape candidate.  The eye-angle source
    contract carries the materializer's self-digested authority for that
    source.  Candidate review may reuse only its exact nested admission; it
    does not grant normal-reader or selector authority.
    """

    authority_mode = str(run_attrs.get("source_eye_geometry_authority_mode") or "")
    if authority_mode != _STAGED_SUBJECT_SHAPE_SOURCE_AUTHORITY_MODE:
        return None
    contracts = run_attrs.get("eye_angle_source_contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError(
            "Selector-ineligible eye-angle review lacks sealed source contracts."
        )
    eye_geometry = contracts.get("eye_geometry")
    if not isinstance(eye_geometry, Mapping):
        raise ValueError(
            "Selector-ineligible eye-angle review lacks an eye-geometry source contract."
        )
    authority = eye_geometry.get("source_authority")
    if not isinstance(authority, Mapping):
        raise ValueError(
            "Selector-ineligible eye-angle review lacks source geometry authority."
        )
    schema_version = authority.get("schema_version")
    if schema_version == EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_VERSION:
        # A selector-ineligible eye-angle storage candidate can still derive
        # from a normal canonical subject-shape publication.  The canonical
        # resolver below remains the correct authority for that case.
        return None
    body = dict(authority)
    persisted_digest = body.pop("record_sha256", None)
    admission = authority.get("candidate_admission")
    if (
        set(authority) != _STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_FIELDS
        or authority.get("schema_id")
        != EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID
        or schema_version
        != EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_SCHEMA_VERSION
        or authority.get("authority_scope")
        != EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_SCOPE
        or authority.get("source_subject_shape_run") != subject_shape_run
        or authority.get("source_subject_shape_run_ref")
        != f"/{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{subject_shape_run}"
        or authority.get("normal_reader_authority") is not False
        or type(persisted_digest) is not str
        or persisted_digest != canonical_json_sha256(body)
        or not isinstance(admission, Mapping)
        or admission.get("source_subject_shape_run") != subject_shape_run
        or admission.get("normal_reader_authority") is not False
        or admission.get("selector_activation") is not False
    ):
        raise ValueError(
            "Selector-ineligible eye-angle review source authority is invalid or stale."
        )
    if (
        eye_geometry.get("stage_group") != EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
        or eye_geometry.get("run_name") != subject_shape_run
    ):
        raise ValueError(
            "Selector-ineligible eye-angle review source contract names another run."
        )
    return admission


def _resolve_review_geometry(
    root: zarr.Group,
    run_attrs: Mapping[str, Any],
    *,
    allow_ineligible_candidate: bool = False,
):
    if type(allow_ineligible_candidate) is not bool:
        raise ValueError("Review candidate admission flag must be an exact bool.")
    stage = str(run_attrs.get("source_eye_geometry_stage") or "")
    run = str(run_attrs.get("source_eye_geometry_run") or "")
    if stage == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE:
        if allow_ineligible_candidate:
            admission = _candidate_subject_shape_review_admission(
                run_attrs,
                subject_shape_run=run,
            )
            if admission is not None:
                return resolve_eye_geometry_source(
                    root,
                    subject_shape_run=run,
                    _completed_ineligible_subject_shape_candidate=admission,
                )
        return resolve_eye_geometry_source(root, subject_shape_run=run)
    if stage == EYE_GEOMETRY_STAGE_REFINED_SUBJECT:
        return resolve_eye_geometry_source(root, refined_subject_run=run)
    refined = str(run_attrs.get("source_refined_subject_masks_run") or "")
    if refined:
        return resolve_eye_geometry_source(root, refined_subject_run=refined)
    raise ValueError("Eye-angle run does not identify a modern subject eye-geometry source.")


def _resolve_review_masks(root: zarr.Group, geometry: Any) -> tuple[Any, str]:
    """Resolve exact left/right masks aligned to the selected eye geometry."""

    masks = geometry.masks_roi
    mask_source_path = str(geometry.group_path)
    if masks is None:
        publication = getattr(
            geometry,
            "subject_shape_coordinate_publication",
            None,
        )
        publication_source = getattr(publication, "source", None)
        if isinstance(publication_source, BoundSubjectShapeBundleSource):
            bundle_source = require_bound_subject_shape_bundle_source(
                publication_source
            )
            mask_store = open_mask_store(
                bundle_source.authority.refined_run,
                source_path=bundle_source.authority.refined_run_path,
                prefer="dense",
            )
            eye_channels = tuple(
                mask_store.component_index(component)
                for component in ("eye_left", "eye_right")
            )
            masks = MaskStoreChannelSelectionArray(mask_store, eye_channels)
            mask_source_path = mask_store.storage_path
    if masks is None:
        refined_subject_run = str(geometry.source_refined_subject_run or "").strip()
        if not refined_subject_run:
            raise ValueError(
                "Subject-shape eye geometry does not identify its source "
                "refined-subject-mask run for review."
            )
        mask_geometry = resolve_eye_geometry_source(
            root,
            refined_subject_run=refined_subject_run,
        )
        masks = mask_geometry.masks_roi
        mask_source_path = str(mask_geometry.group_path)

    if masks is None:
        raise ValueError("Resolved eye geometry has no dense/decodable eye masks for review.")
    mask_shape = tuple(int(value) for value in getattr(masks, "shape", ()))
    ellipse_shape = tuple(int(value) for value in getattr(geometry.ellipse_params, "shape", ()))
    if len(mask_shape) != 4 or mask_shape[1] != 2:
        raise ValueError(
            f"Review eye masks must have shape (row, 2, y, x); got {mask_shape}."
        )
    if len(ellipse_shape) < 3 or ellipse_shape[1] != 2:
        raise ValueError(
            "Review eye ellipses must have shape (row, 2, parameter); "
            f"got {ellipse_shape}."
        )
    if mask_shape[0] != ellipse_shape[0]:
        raise ValueError(
            "Review masks and eye ellipses are not row-aligned: "
            f"{mask_source_path} has {mask_shape[0]} rows but "
            f"{geometry.group_path} has {ellipse_shape[0]}."
        )
    return masks, mask_source_path


def _resolve_review_roi_offsets(geometry: Any) -> np.ndarray | None:
    """Return exact ROI-to-source translations for source-frame geometry."""

    if str(getattr(geometry, "stage_group", "")) != EYE_GEOMETRY_STAGE_SUBJECT_SHAPE:
        return None
    publication = getattr(
        geometry,
        "subject_shape_coordinate_publication",
        None,
    )
    if publication is None:
        raise ValueError(
            "Subject-shape review geometry lacks its bound coordinate publication."
        )
    continuous, edge = require_translation_only_subject_shape_placement(
        publication.source
    )
    offsets = np.asarray(continuous, dtype=np.float64)
    edge_offsets = np.asarray(edge, dtype=np.float64)
    ellipse_rows = int(getattr(geometry.ellipse_params, "shape", (0,))[0])
    if (
        offsets.shape != (ellipse_rows, 2)
        or edge_offsets.shape != offsets.shape
        or not np.isfinite(offsets).all()
        or not np.array_equal(offsets, edge_offsets)
    ):
        raise ValueError(
            "Subject-shape review masks lack exact row-aligned translation-only placement."
        )
    return offsets


def _normalize_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.quantile(finite, [0.01, 0.99])
    if not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _review_selection(sample: Mapping[str, np.ndarray], panel_count: int) -> np.ndarray:
    valid = np.asarray(sample["valid_frame"], dtype=bool) & np.asarray(sample["valid"], dtype=bool)
    valid &= np.isfinite(np.asarray(sample["left_gaze_signed_deg"], dtype=np.float64))
    valid &= np.isfinite(np.asarray(sample["right_gaze_signed_deg"], dtype=np.float64))
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return np.asarray([], dtype=np.int64)
    score = np.maximum(
        np.abs(np.asarray(sample["left_gaze_signed_deg"])[candidates]),
        np.abs(np.asarray(sample["right_gaze_signed_deg"])[candidates]),
    )
    order = candidates[np.argsort(score)]
    count = min(int(panel_count), int(order.size))
    positions = np.rint(np.linspace(0, order.size - 1, count)).astype(np.int64)
    return order[positions]


def write_bounded_review_png(
    *,
    root: zarr.Group,
    run_group: zarr.Group,
    sample: Mapping[str, np.ndarray],
    output_path: Path,
    panel_count: int = 12,
    allow_ineligible_candidate: bool = False,
) -> tuple[list[int], str]:
    """Write a bounded eye-identity/gaze-vector overlay review grid."""

    attrs = _group_attrs(run_group)
    _keypoint_name, keypoint_group = _resolve_keypoint_group(root, attrs)
    roi_images = _resolve_roi_images(root, keypoint_group)
    geometry = _resolve_review_geometry(
        root,
        attrs,
        allow_ineligible_candidate=allow_ineligible_candidate,
    )
    masks, mask_source_path = _resolve_review_masks(root, geometry)
    roi_offsets = _resolve_review_roi_offsets(geometry)
    ellipse_params = geometry.ellipse_params
    selected = _review_selection(sample, panel_count)
    if selected.size == 0:
        raise ValueError("No valid sampled rows are available for a review overlay.")

    columns = min(4, int(selected.size))
    rows = int(np.ceil(selected.size / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.2 * rows), squeeze=False)
    review_rows: list[int] = []
    for panel_index, sample_index in enumerate(selected.tolist()):
        ax = axes.flat[panel_index]
        row_index = int(np.asarray(sample["sample_row_indices"])[sample_index])
        review_rows.append(row_index)
        eye_masks = np.asarray(masks[row_index], dtype=bool)
        if roi_images is None:
            image = np.zeros(eye_masks.shape[-2:], dtype=np.float32)
            pixel_mode = "mask-only geometry"
        else:
            image = _normalize_image(np.asarray(roi_images[row_index]))
            pixel_mode = "ROI pixels"
        rgb = np.repeat(image[..., None], 3, axis=2)
        colors = (np.asarray([0.92, 0.20, 0.25]), np.asarray([0.15, 0.52, 0.95]))
        for eye_index, color in enumerate(colors):
            if eye_index < eye_masks.shape[0]:
                mask = eye_masks[eye_index]
                rgb[mask] = 0.45 * rgb[mask] + 0.55 * color
        ax.imshow(np.clip(rgb, 0.0, 1.0))

        params = np.asarray(ellipse_params[row_index])
        gaze_vectors = (
            np.asarray(sample["left_gaze_xy"])[sample_index],
            np.asarray(sample["right_gaze_xy"])[sample_index],
        )
        scale = 0.22 * float(max(image.shape[:2]))
        for eye_index, (color, vector) in enumerate(zip(colors, gaze_vectors)):
            if eye_index >= params.shape[0] or not np.all(np.isfinite(params[eye_index, :2])):
                continue
            center = np.asarray(params[eye_index, :2], dtype=np.float64)
            if roi_offsets is not None:
                center = center - roi_offsets[row_index]
            if not np.all(np.isfinite(vector)):
                continue
            ax.arrow(
                float(center[0]),
                float(center[1]),
                float(vector[0]) * scale,
                float(vector[1]) * scale,
                width=0.35,
                head_width=2.5,
                color=color,
                length_includes_head=True,
            )

        # Eye-angle body-frame support is already stored in ROI-local pixels;
        # only subject-shape ellipse points require source-camera translation.
        origin = np.asarray(sample["origin_xy"])[sample_index]
        forward = np.asarray(sample["forward_axis_xy"])[sample_index]
        left = np.asarray(sample["left_axis_xy"])[sample_index]
        if np.all(np.isfinite(origin)) and np.all(np.isfinite(forward)):
            ax.arrow(*origin, *(forward * scale), width=0.3, head_width=2.2, color="#facc15", length_includes_head=True)
        if np.all(np.isfinite(origin)) and np.all(np.isfinite(left)):
            ax.arrow(*origin, *(left * scale * 0.7), width=0.25, head_width=2.0, color="#22c55e", length_includes_head=True)

        frame = int(np.asarray(sample["frame_indices"])[sample_index])
        left_gaze = float(np.asarray(sample["left_gaze_signed_deg"])[sample_index])
        right_gaze = float(np.asarray(sample["right_gaze_signed_deg"])[sample_index])
        marginal = bool(np.asarray(sample["major_axis_marginal"])[sample_index])
        ax.set_title(
            f"ROI {row_index:,} · frame {frame:,} · {pixel_mode}\n"
            f"L gaze {left_gaze:+.1f}° · R gaze {right_gaze:+.1f}° · marginal={marginal}",
            fontsize=9,
        )
        ax.set_xlim(-0.5, image.shape[1] - 0.5)
        ax.set_ylim(image.shape[0] - 0.5, -0.5)
        ax.axis("off")
    for ax in axes.flat[selected.size :]:
        ax.axis("off")
    fig.suptitle(
        "Eye convention gate · red/blue = labeled left/right eye · yellow = forward · green = anatomical left",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return review_rows, mask_source_path


def validate_eye_angle_run(
    zarr_path: Path,
    *,
    eye_angle_run: Optional[str] = None,
    windows: int = DEFAULT_WINDOWS,
    rows_per_window: int = DEFAULT_ROWS_PER_WINDOW,
    review_png: Optional[Path] = None,
    review_panels: int = 12,
    legacy_compatibility: bool = False,
    allow_ineligible_candidate: bool = False,
) -> dict[str, object]:
    """Validate one persisted eye-angle run without modifying its Zarr."""

    root = open_zarr_root(zarr_path, mode="r")
    resolved_run, run_group = _resolve_eye_run(
        root,
        eye_angle_run,
        legacy_compatibility=legacy_compatibility,
        allow_ineligible_candidate=allow_ineligible_candidate,
    )
    metadata_equivalence = None
    if allow_ineligible_candidate:
        metadata_equivalence = validate_direct_consolidated_subtree(
            zarr_path,
            subtree_path=f"analysis/eye_angle_runs/{resolved_run}",
        ).to_json()
    attrs = _group_attrs(run_group)
    sample = _load_compact_sample(run_group, windows=windows, rows_per_window=rows_per_window)
    valid = np.asarray(sample["valid_frame"], dtype=bool) & np.asarray(sample["valid"], dtype=bool)
    checks = list(_metadata_checks(attrs))
    checks.extend(
        validate_gaze_geometry_arrays(
            left_major_signed_deg=sample["left_major_signed_deg"],
            right_major_signed_deg=sample["right_major_signed_deg"],
            left_eye_angle_deg=sample["left_eye_angle_deg"],
            right_eye_angle_deg=sample["right_eye_angle_deg"],
            vergence_eye_angle_deg=sample["vergence_eye_angle_deg"],
            left_gaze_signed_deg=sample["left_gaze_signed_deg"],
            right_gaze_signed_deg=sample["right_gaze_signed_deg"],
            left_gaze_xy=sample["left_gaze_xy"],
            right_gaze_xy=sample["right_gaze_xy"],
            forward_axis_xy=sample["forward_axis_xy"],
            left_axis_xy=sample["left_axis_xy"],
            heading_deg=sample["heading_deg"],
            valid=valid,
        )
    )
    valid_count = int(np.sum(valid))
    marginal = np.asarray(sample["major_axis_marginal"], dtype=bool) & valid
    assumption = {
        "name": "ellipse_axis_direction_assumption",
        "passed": None,
        "review_required": True,
        "definition": (
            "Directed gaze is derived from the directionless ellipse major axis after resolving "
            "that axis into the fish-forward half-plane. Numeric identities cannot independently "
            "validate this biological direction assumption."
        ),
        "valid_sample_count": valid_count,
        "major_axis_marginal_count": int(np.sum(marginal)),
        "major_axis_marginal_fraction": float(np.sum(marginal) / valid_count) if valid_count else None,
        "stored_gaze_angle_source": attrs.get("gaze_angle_source"),
    }
    review_rows: list[int] = []
    review_mask_source_path: Optional[str] = None
    if review_png is not None:
        review_rows, review_mask_source_path = write_bounded_review_png(
            root=root,
            run_group=run_group,
            sample=sample,
            output_path=review_png,
            panel_count=review_panels,
            allow_ineligible_candidate=allow_ineligible_candidate,
        )
    passed = all(check.passed for check in checks)
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "zarr_path": str(zarr_path),
        "eye_angle_run": resolved_run,
        "eye_angle_run_path": f"analysis/eye_angle_runs/{resolved_run}",
        "eye_angle_admission": (
            "explicit_complete_selector_ineligible_storage_candidate_v1"
            if allow_ineligible_candidate
            else "canonical_selector_eligible_eye_angle_reader_v1"
        ),
        "direct_consolidated_metadata_equivalence": metadata_equivalence,
        "read_only": True,
        "sampling": {
            "row_axis": "keypoint_detection_rows",
            "total_rows": int(sample["row_count"]),
            "sample_rows": int(np.asarray(sample["sample_row_indices"]).size),
            "valid_sample_rows": valid_count,
            "window_count": int(windows),
            "rows_per_window": int(rows_per_window),
            "sample_row_first": int(np.asarray(sample["sample_row_indices"])[0]) if np.asarray(sample["sample_row_indices"]).size else None,
            "sample_row_last": int(np.asarray(sample["sample_row_indices"])[-1]) if np.asarray(sample["sample_row_indices"]).size else None,
        },
        "comparison_contract": {
            "object_angle_field": "egocentric_bearing/per_chaser/bearing_deg",
            "eye_angle_fields": ["left_gaze_signed_deg", "right_gaze_signed_deg"],
            "coordinate_frame": "fish_body_frame",
            "zero": "fish_forward",
            "positive": "anatomical_left",
            "explicitly_not_comparable_fields": ["left_eye_angle_deg", "right_eye_angle_deg"],
        },
        "checks": [check.to_dict() for check in checks],
        "direction_assumption": assumption,
        "review_png": str(review_png) if review_png is not None else None,
        "review_mask_source_path": review_mask_source_path,
        "review_row_indices": review_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation of eye identity, gaze signs, and body-frame geometry."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette analysis Zarr.")
    parser.add_argument("--eye-angle-run", help="Eye-angle run name; defaults to the stable complete selector.")
    parser.add_argument(
        "--legacy-eye-angle-compatibility",
        action="store_true",
        help="Permit statusless historical eye-angle runs; never permits selector-ineligible runs.",
    )
    parser.add_argument(
        "--allow-ineligible-candidate",
        action="store_true",
        help=(
            "Review one explicitly named complete access-aware storage candidate. "
            "The exact candidate contract and direct/consolidated metadata are "
            "validated; no selector alias or arbitrary ineligible run is accepted."
        ),
    )
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS, help="Bounded sample windows across the run.")
    parser.add_argument(
        "--rows-per-window",
        type=int,
        default=DEFAULT_ROWS_PER_WINDOW,
        help="Contiguous ROI rows read per sample window.",
    )
    parser.add_argument("--review-png", type=Path, help="Optional bounded eye-mask/gaze-vector review grid.")
    parser.add_argument("--review-panels", type=int, default=12, help="Maximum panels in --review-png.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report path; stdout is always printed.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return exit code 1 when a numeric/metadata gate fails.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_eye_angle_run(
        args.zarr_path,
        eye_angle_run=args.eye_angle_run,
        windows=args.windows,
        rows_per_window=args.rows_per_window,
        review_png=args.review_png,
        review_panels=args.review_panels,
        legacy_compatibility=bool(args.legacy_eye_angle_compatibility),
        allow_ineligible_candidate=bool(args.allow_ineligible_candidate),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if args.fail_on_error and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
