"""Persist anatomical fish--chaser alignment on exact semantic epochs.

This successor is intentionally a reduction of one exact keypoint
``chaser_relative_frame`` child.  It consumes the base physical distance and
the anatomical body extension from that *same* child, assigns the already
sealed protocol-semantic epochs, and persists both row evidence and fixed
distance-bin summaries.  It never substitutes motion heading, interpolates a
body row, or lets a viewer choose scientific bins.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.provider_chaser_position_suite import PositionSuiteEpoch
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)
from fisheye.analysis_workflows.core_paradigm_authority import (
    core_paradigm_dependency_from_relative_frame,
    validate_core_paradigm_source_dependency,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SCHEMA_ID = "palette.chaser.body_alignment_by_distance"
SCHEMA_VERSION = 1
METHOD_ID = "exact_anatomical_body_bearing_distance_summary_v1"
DISTANCE_BIN_POLICY_ID = "zero_anchored_fixed_width_covering_semantic_rows_v1"
DISTANCE_BIN_WIDTH_MM = 5.0
ANGLE_CONVENTION_ID = "atan2_anatomical_left_over_forward_degrees_v1"
ALIGNMENT_REASON_CODES = MappingProxyType(
    {
        0: "ok",
        1: "chaser_occurrence_absent",
        2: "base_physical_distance_invalid",
        3: "body_source_row_unavailable",
        4: "body_heading_invalid",
        5: "body_bearing_invalid",
        6: "joint_validity_failed_other",
    }
)
FRAME_ARRAY_NAMES = (
    "frame_acquisition_frame_id",
    "frame_epoch_role_code",
    "frame_epoch_window_id",
    "frame_selection_member",
    "frame_chaser_occurrence_member",
    "frame_chaser_identity_code",
    "frame_chaser_behavior_role_code",
    "frame_chaser_behavior_role_valid",
    "frame_relative_distance_physical",
    "frame_relative_physical_valid",
    "frame_relative_physical_reason_code",
    "frame_body_source_row_id",
    "frame_body_source_row_valid",
    "frame_body_heading_deg",
    "frame_body_heading_valid",
    "frame_body_heading_reason_code",
    "frame_body_bearing_deg",
    "frame_body_bearing_valid",
    "frame_body_bearing_reason_code",
    "frame_alignment_cos",
    "frame_lateral_sin",
    "frame_alignment_valid",
    "frame_alignment_reason_code",
)
SUMMARY_ARRAY_NAMES = (
    "summary_epoch_role_code",
    "summary_epoch_window_id",
    "summary_chaser_identity_code",
    "summary_chaser_behavior_role_code",
    "summary_distance_bin_index",
    "summary_distance_bin_start_mm",
    "summary_distance_bin_end_mm",
    "summary_distance_bin_center_mm",
    "summary_candidate_row_count",
    "summary_joint_valid_row_count",
    "summary_body_source_missing_row_count",
    "summary_body_heading_invalid_row_count",
    "summary_body_bearing_invalid_row_count",
    "summary_other_alignment_invalid_row_count",
    "summary_epoch_occurrence_row_count",
    "summary_epoch_chaser_absent_row_count",
    "summary_epoch_distance_valid_row_count",
    "summary_epoch_distance_invalid_row_count",
    "summary_epoch_distance_invalid_body_valid_row_count",
    "summary_mean_alignment_cos",
    "summary_alignment_cos_p25",
    "summary_alignment_cos_p50",
    "summary_alignment_cos_p75",
    "summary_mean_abs_bearing_deg",
    "summary_abs_bearing_p25_deg",
    "summary_abs_bearing_p50_deg",
    "summary_abs_bearing_p75_deg",
    "summary_circular_mean_bearing_deg",
    "summary_circular_resultant_length",
)
PERSISTED_ARRAY_NAMES = (
    *FRAME_ARRAY_NAMES,
    "distance_bin_edges_mm",
    *SUMMARY_ARRAY_NAMES,
)
SUMMARY_VIEW_ARRAY_NAMES = ("distance_bin_edges_mm", *SUMMARY_ARRAY_NAMES)


class ChaserBodyAlignmentByDistanceError(ValueError):
    """Raised when anatomical alignment cannot remain exact and auditable."""


def _fail(message: str) -> None:
    raise ChaserBodyAlignmentByDistanceError(message)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one exact non-empty string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
            "content_sha256": array_values_sha256(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    ]


@dataclass(frozen=True, slots=True)
class ChaserBodyAlignmentByDistanceInput:
    recording_id: str
    relative_frame_run_path: str
    relative_frame_manifest_sha256: str
    semantic_selection_run_path: str
    semantic_selection_manifest_sha256: str
    n_frames: int
    n_chasers: int
    acquisition_frame_id: np.ndarray
    selection_member: np.ndarray
    chaser_occurrence_member: np.ndarray
    chaser_identity_code: np.ndarray
    chaser_behavior_role_code: np.ndarray
    chaser_behavior_role_valid: np.ndarray
    relative_distance_physical: np.ndarray
    relative_physical_valid: np.ndarray
    relative_physical_reason_code: np.ndarray
    body_source_row_id: np.ndarray
    body_source_row_valid: np.ndarray
    body_heading_deg: np.ndarray
    body_heading_valid: np.ndarray
    body_heading_reason_code: np.ndarray
    body_bearing_deg: np.ndarray
    body_bearing_valid: np.ndarray
    body_bearing_reason_code: np.ndarray
    epochs: Sequence[PositionSuiteEpoch]
    fish_position_authority: Mapping[str, Any]
    body_frame_authority: Mapping[str, Any]
    identity_registries: Mapping[str, Any]
    scale_policy: Mapping[str, Any]
    relative_frame_verification_mode: str = "direct_deep_audit"
    relative_frame_validation_receipt_sha256: str | None = None
    distance_bin_width_mm: float = DISTANCE_BIN_WIDTH_MM
    core_authority_dependency: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PreparedChaserBodyAlignmentByDistance:
    recording_id: str
    n_frames: int
    n_chasers: int
    n_frame_rows: int
    n_summary_rows: int
    n_distance_bins: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown body-alignment array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _matrix(
    value: Any,
    *,
    shape: tuple[int, int],
    field: str,
    kind: str | None = None,
) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        _fail(f"{field} must have exact frame/chaser shape {shape!r}.")
    if kind is not None and result.dtype.kind not in kind:
        _fail(f"{field} has unsupported dtype {result.dtype.str!r}.")
    return result


def _frame_matrix(
    value: Any,
    *,
    n_frames: int,
    n_chasers: int,
    field: str,
    kind: str | None = None,
) -> np.ndarray:
    result = _matrix(
        value,
        shape=(n_frames, n_chasers),
        field=field,
        kind=kind,
    )
    reference = result[:, :1]
    repeated = (
        np.array_equal(result, np.broadcast_to(reference, result.shape), equal_nan=True)
        if result.dtype.kind == "f"
        else np.array_equal(result, np.broadcast_to(reference, result.shape))
    )
    if not repeated:
        _fail(f"{field} differs across flattened chaser rows.")
    return result[:, 0]


def _distance_edges(
    distance: np.ndarray,
    member: np.ndarray,
    *,
    width_mm: float,
) -> np.ndarray:
    if not math.isfinite(width_mm) or width_mm <= 0.0:
        _fail("distance_bin_width_mm must be finite and positive.")
    values = distance[member]
    maximum = float(np.max(values)) if values.size else 0.0
    count = max(1, int(math.ceil(maximum / width_mm)))
    if count > 4096:
        _fail("Distance-bin count exceeds the bounded successor contract.")
    edges = np.linspace(0.0, float(count) * width_mm, count + 1, dtype=np.float64)
    if values.size and float(np.max(values)) > float(edges[-1]) + 1e-9:
        _fail("Distance-bin edge construction failed to cover exact valid rows.")
    return edges


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return math.nan, math.nan, math.nan
    result = np.quantile(values, (0.25, 0.5, 0.75))
    return float(result[0]), float(result[1]), float(result[2])


def _circular_summary(degrees: np.ndarray) -> tuple[float, float]:
    if degrees.size == 0:
        return math.nan, math.nan
    radians = np.deg2rad(degrees)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    resultant = float(math.hypot(mean_sin, mean_cos))
    if resultant <= np.finfo(np.float64).eps:
        return math.nan, resultant
    return float(np.rad2deg(math.atan2(mean_sin, mean_cos))), resultant


def prepare_chaser_body_alignment_by_distance_successor(
    inputs: ChaserBodyAlignmentByDistanceInput,
) -> PreparedChaserBodyAlignmentByDistance:
    """Build immutable row evidence and epoch/chaser/distance-bin summaries."""

    if type(inputs) is not ChaserBodyAlignmentByDistanceInput:
        raise TypeError("inputs must be one ChaserBodyAlignmentByDistanceInput.")
    recording_id = _text(inputs.recording_id, field="recording_id")
    relative_path = _text(inputs.relative_frame_run_path, field="relative run path")
    relative_digest = _digest(
        inputs.relative_frame_manifest_sha256,
        field="relative manifest digest",
    )
    semantic_path = _text(
        inputs.semantic_selection_run_path,
        field="semantic selection run path",
    )
    semantic_digest = _digest(
        inputs.semantic_selection_manifest_sha256,
        field="semantic selection manifest digest",
    )
    try:
        core_authority = validate_core_paradigm_source_dependency(
            inputs.core_authority_dependency,
            recording_id=recording_id,
            source_relative_frame_run_path=relative_path,
            source_relative_frame_manifest_sha256=relative_digest,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Core-authority dependency is invalid: {exc}")
    n_frames = int(inputs.n_frames)
    n_chasers = int(inputs.n_chasers)
    if n_frames <= 0 or n_chasers <= 0:
        _fail("n_frames and n_chasers must be positive.")
    shape = (n_frames, n_chasers)

    frame_id = _frame_matrix(
        inputs.acquisition_frame_id,
        n_frames=n_frames,
        n_chasers=n_chasers,
        field="acquisition_frame_id",
        kind="iu",
    ).astype(np.int64, copy=False)
    if np.any(np.diff(frame_id) <= 0):
        _fail("acquisition_frame_id must be strictly increasing.")
    selected = _frame_matrix(
        inputs.selection_member,
        n_frames=n_frames,
        n_chasers=n_chasers,
        field="selection_member",
    )
    if selected.dtype != np.dtype(bool):
        _fail("selection_member must be boolean.")

    occurrence = _matrix(
        inputs.chaser_occurrence_member,
        shape=shape,
        field="chaser_occurrence_member",
    )
    role_valid = _matrix(
        inputs.chaser_behavior_role_valid,
        shape=shape,
        field="chaser_behavior_role_valid",
    )
    for field, values in (
        ("chaser_occurrence_member", occurrence),
        ("chaser_behavior_role_valid", role_valid),
    ):
        if values.dtype != np.dtype(bool):
            _fail(f"{field} must be boolean.")
    identity = _matrix(
        inputs.chaser_identity_code,
        shape=shape,
        field="chaser_identity_code",
        kind="iu",
    ).astype(np.uint16, copy=False)
    role = _matrix(
        inputs.chaser_behavior_role_code,
        shape=shape,
        field="chaser_behavior_role_code",
        kind="iu",
    ).astype(np.uint8, copy=False)
    if not np.all(identity == identity[:1]):
        _fail("Chaser identity changes along the exact frame axis.")
    if len(set(int(value) for value in identity[0])) != n_chasers:
        _fail("Chaser identities are not unique on the chaser axis.")

    distance = _matrix(
        inputs.relative_distance_physical,
        shape=shape,
        field="relative_distance_physical",
        kind="f",
    ).astype(np.float64, copy=False)
    distance_valid = _matrix(
        inputs.relative_physical_valid,
        shape=shape,
        field="relative_physical_valid",
    )
    distance_reason = _matrix(
        inputs.relative_physical_reason_code,
        shape=shape,
        field="relative_physical_reason_code",
        kind="iu",
    ).astype(np.uint16, copy=False)
    if distance_valid.dtype != np.dtype(bool):
        _fail("relative_physical_valid must be boolean.")
    if np.any(distance_valid & (~np.isfinite(distance) | (distance < 0.0))):
        _fail("Declared-valid base physical distances must be finite and nonnegative.")

    body_source = _matrix(
        inputs.body_source_row_id,
        shape=shape,
        field="body_source_row_id",
        kind="iu",
    ).astype(np.int64, copy=False)
    body_source_valid = _matrix(
        inputs.body_source_row_valid,
        shape=shape,
        field="body_source_row_valid",
    )
    heading = _matrix(
        inputs.body_heading_deg,
        shape=shape,
        field="body_heading_deg",
        kind="f",
    ).astype(np.float64, copy=False)
    heading_valid = _matrix(
        inputs.body_heading_valid,
        shape=shape,
        field="body_heading_valid",
    )
    heading_reason = _matrix(
        inputs.body_heading_reason_code,
        shape=shape,
        field="body_heading_reason_code",
        kind="iu",
    ).astype(np.uint16, copy=False)
    bearing = _matrix(
        inputs.body_bearing_deg,
        shape=shape,
        field="body_bearing_deg",
        kind="f",
    ).astype(np.float64, copy=False)
    bearing_valid = _matrix(
        inputs.body_bearing_valid,
        shape=shape,
        field="body_bearing_valid",
    )
    bearing_reason = _matrix(
        inputs.body_bearing_reason_code,
        shape=shape,
        field="body_bearing_reason_code",
        kind="iu",
    ).astype(np.uint16, copy=False)
    for field, values in (
        ("body_source_row_valid", body_source_valid),
        ("body_heading_valid", heading_valid),
        ("body_bearing_valid", bearing_valid),
    ):
        if values.dtype != np.dtype(bool):
            _fail(f"{field} must be boolean.")
    if np.any(body_source_valid & (body_source < 0)):
        _fail("Valid body source rows must be nonnegative.")
    if np.any(
        heading_valid & (~np.isfinite(heading) | (heading < -180.0) | (heading > 180.0))
    ):
        _fail("Declared-valid body headings must be finite in [-180, 180].")
    if np.any(
        bearing_valid & (~np.isfinite(bearing) | (bearing < -180.0) | (bearing > 180.0))
    ):
        _fail("Declared-valid body bearings must be finite in [-180, 180].")
    if np.any(bearing_valid & (~body_source_valid | ~heading_valid)):
        _fail("Valid body bearing lacks a valid source row or anatomical heading.")
    for field, values, kind in (
        ("body_source_row_id", body_source, "iu"),
        ("body_source_row_valid", body_source_valid, None),
        ("body_heading_deg", heading, "f"),
        ("body_heading_valid", heading_valid, None),
        ("body_heading_reason_code", heading_reason, "iu"),
    ):
        _frame_matrix(
            values,
            n_frames=n_frames,
            n_chasers=n_chasers,
            field=field,
            kind=kind,
        )

    epochs = tuple(inputs.epochs)
    if tuple(epoch.analysis_role for epoch in epochs) != CHASER_WINDOW_ROLES:
        _fail("Body alignment requires exact chaser pre/training/post epoch order.")
    epoch_code = np.zeros(n_frames, dtype=np.uint8)
    epoch_window = np.full(n_frames, -1, dtype=np.int64)
    epoch_member = np.zeros(n_frames, dtype=bool)
    epoch_masks: list[np.ndarray] = []
    previous_end: int | None = None
    for code, epoch in enumerate(epochs, start=1):
        if (
            type(epoch) is not PositionSuiteEpoch
            or epoch.start_frame < 0
            or epoch.end_frame <= epoch.start_frame
            or (previous_end is not None and epoch.start_frame < previous_end)
        ):
            _fail("Body-alignment semantic epoch bounds are invalid or overlap.")
        _digest(epoch.source_interval_sha256, field="epoch source interval digest")
        member = (frame_id >= int(epoch.start_frame)) & (
            frame_id < int(epoch.end_frame)
        )
        if not np.any(member):
            _fail(f"Semantic epoch {epoch.analysis_role!r} has no exact frame rows.")
        if np.any(member & ~selected):
            _fail(
                "A semantic epoch frame is absent from relative selection membership."
            )
        epoch_code[member] = code
        epoch_window[member] = int(epoch.window_id)
        epoch_member |= member
        epoch_masks.append(member)
        previous_end = int(epoch.end_frame)

    semantic_candidate = epoch_member[:, None] & occurrence
    edges = _distance_edges(
        distance,
        semantic_candidate & distance_valid,
        width_mm=float(inputs.distance_bin_width_mm),
    )
    n_bins = int(edges.size - 1)
    bin_recipe = {
        "policy_id": DISTANCE_BIN_POLICY_ID,
        "bin_width_mm": float(inputs.distance_bin_width_mm),
        "edge_count": int(edges.size),
        "edges_mm": edges.tolist(),
        "edge_array_sha256": array_values_sha256(edges),
        "source_scope": "all_occurrence_member_distance_valid_semantic_epoch_rows",
        "interval_policy": "half_open_except_final_closed",
        "zero_anchored": True,
    }
    bin_recipe_sha256 = canonical_json_sha256(bin_recipe)

    alignment_valid = occurrence & distance_valid & bearing_valid
    radians = np.deg2rad(bearing)
    alignment_cos = np.full(shape, np.nan, dtype=np.float64)
    lateral_sin = np.full(shape, np.nan, dtype=np.float64)
    alignment_cos[alignment_valid] = np.cos(radians[alignment_valid])
    lateral_sin[alignment_valid] = np.sin(radians[alignment_valid])
    alignment_reason = np.full(shape, 6, dtype=np.uint8)
    alignment_reason[~occurrence] = 1
    alignment_reason[occurrence & ~distance_valid] = 2
    alignment_reason[occurrence & distance_valid & ~body_source_valid] = 3
    alignment_reason[
        occurrence & distance_valid & body_source_valid & ~heading_valid
    ] = 4
    alignment_reason[
        occurrence & distance_valid & body_source_valid & heading_valid & ~bearing_valid
    ] = 5
    alignment_reason[alignment_valid] = 0

    summary: dict[str, list[Any]] = {name: [] for name in SUMMARY_ARRAY_NAMES}

    for epoch_index, epoch_mask in enumerate(epoch_masks, start=1):
        for chaser_index in range(n_chasers):
            epoch_column = epoch_mask
            occurrence_column = occurrence[:, chaser_index]
            candidate = epoch_column & occurrence_column
            candidate_roles = role[epoch_column, chaser_index]
            candidate_role_valid = role_valid[epoch_column, chaser_index]
            if not np.all(candidate_role_valid) or np.unique(candidate_roles).size != 1:
                _fail(
                    "Chaser behavior role is missing or changes within a semantic epoch."
                )
            role_code = int(candidate_roles[0])
            distance_ok = candidate & distance_valid[:, chaser_index]
            occurrence_count = int(np.count_nonzero(candidate))
            absent_count = int(np.count_nonzero(epoch_column & ~occurrence_column))
            distance_valid_count = int(np.count_nonzero(distance_ok))
            distance_invalid = candidate & ~distance_valid[:, chaser_index]
            distance_invalid_count = int(np.count_nonzero(distance_invalid))
            distance_invalid_body_valid_count = int(
                np.count_nonzero(distance_invalid & bearing_valid[:, chaser_index])
            )
            if distance_valid_count + distance_invalid_count != occurrence_count:
                _fail(
                    "Distance-valid and distance-invalid counts do not conserve occurrence rows."
                )

            for bin_index, (left, right) in enumerate(
                zip(edges[:-1], edges[1:], strict=True)
            ):
                in_bin = distance_ok & (distance[:, chaser_index] >= left)
                in_bin &= (
                    distance[:, chaser_index] <= right
                    if bin_index == n_bins - 1
                    else distance[:, chaser_index] < right
                )
                joint = in_bin & bearing_valid[:, chaser_index]
                reasons = alignment_reason[:, chaser_index]
                candidate_count = int(np.count_nonzero(in_bin))
                joint_count = int(np.count_nonzero(joint))
                source_missing_count = int(np.count_nonzero(in_bin & (reasons == 3)))
                heading_invalid_count = int(np.count_nonzero(in_bin & (reasons == 4)))
                bearing_invalid_count = int(np.count_nonzero(in_bin & (reasons == 5)))
                other_invalid_count = int(np.count_nonzero(in_bin & (reasons == 6)))
                if (
                    joint_count
                    + source_missing_count
                    + heading_invalid_count
                    + bearing_invalid_count
                    + other_invalid_count
                    != candidate_count
                ):
                    _fail(
                        "Alignment validity reasons do not conserve one distance bin."
                    )

                bearing_values = bearing[joint, chaser_index]
                alignment_values = alignment_cos[joint, chaser_index]
                abs_bearing = np.abs(bearing_values)
                align_q25, align_q50, align_q75 = _quantiles(alignment_values)
                abs_q25, abs_q50, abs_q75 = _quantiles(abs_bearing)
                circular_mean, circular_resultant = _circular_summary(bearing_values)
                values = {
                    "summary_epoch_role_code": epoch_index,
                    "summary_epoch_window_id": int(epochs[epoch_index - 1].window_id),
                    "summary_chaser_identity_code": int(identity[0, chaser_index]),
                    "summary_chaser_behavior_role_code": role_code,
                    "summary_distance_bin_index": bin_index,
                    "summary_distance_bin_start_mm": float(left),
                    "summary_distance_bin_end_mm": float(right),
                    "summary_distance_bin_center_mm": float((left + right) / 2.0),
                    "summary_candidate_row_count": candidate_count,
                    "summary_joint_valid_row_count": joint_count,
                    "summary_body_source_missing_row_count": source_missing_count,
                    "summary_body_heading_invalid_row_count": heading_invalid_count,
                    "summary_body_bearing_invalid_row_count": bearing_invalid_count,
                    "summary_other_alignment_invalid_row_count": other_invalid_count,
                    "summary_epoch_occurrence_row_count": occurrence_count,
                    "summary_epoch_chaser_absent_row_count": absent_count,
                    "summary_epoch_distance_valid_row_count": distance_valid_count,
                    "summary_epoch_distance_invalid_row_count": distance_invalid_count,
                    "summary_epoch_distance_invalid_body_valid_row_count": distance_invalid_body_valid_count,
                    "summary_mean_alignment_cos": (
                        float(np.mean(alignment_values)) if joint_count else math.nan
                    ),
                    "summary_alignment_cos_p25": align_q25,
                    "summary_alignment_cos_p50": align_q50,
                    "summary_alignment_cos_p75": align_q75,
                    "summary_mean_abs_bearing_deg": (
                        float(np.mean(abs_bearing)) if joint_count else math.nan
                    ),
                    "summary_abs_bearing_p25_deg": abs_q25,
                    "summary_abs_bearing_p50_deg": abs_q50,
                    "summary_abs_bearing_p75_deg": abs_q75,
                    "summary_circular_mean_bearing_deg": circular_mean,
                    "summary_circular_resultant_length": circular_resultant,
                }
                for name, value in values.items():
                    summary[name].append(value)

    if sum(summary["summary_candidate_row_count"]) != int(
        np.count_nonzero(semantic_candidate & distance_valid)
    ):
        _fail("Persisted distance bins do not conserve semantic candidate rows.")
    if sum(summary["summary_joint_valid_row_count"]) != int(
        np.count_nonzero(semantic_candidate & alignment_valid)
    ):
        _fail("Persisted distance bins do not conserve jointly valid alignment rows.")

    epoch_code_rows = np.broadcast_to(epoch_code[:, None], shape)
    epoch_window_rows = np.broadcast_to(epoch_window[:, None], shape)
    arrays: dict[str, np.ndarray] = {
        "frame_acquisition_frame_id": _readonly(
            np.broadcast_to(frame_id[:, None], shape).reshape(-1), dtype=np.int64
        ),
        "frame_epoch_role_code": _readonly(epoch_code_rows.reshape(-1), dtype=np.uint8),
        "frame_epoch_window_id": _readonly(
            epoch_window_rows.reshape(-1), dtype=np.int64
        ),
        "frame_selection_member": _readonly(
            np.broadcast_to(selected[:, None], shape).reshape(-1), dtype=bool
        ),
        "frame_chaser_occurrence_member": _readonly(occurrence.reshape(-1), dtype=bool),
        "frame_chaser_identity_code": _readonly(identity.reshape(-1), dtype=np.uint16),
        "frame_chaser_behavior_role_code": _readonly(role.reshape(-1), dtype=np.uint8),
        "frame_chaser_behavior_role_valid": _readonly(
            role_valid.reshape(-1), dtype=bool
        ),
        "frame_relative_distance_physical": _readonly(
            distance.reshape(-1), dtype=np.float64
        ),
        "frame_relative_physical_valid": _readonly(
            distance_valid.reshape(-1), dtype=bool
        ),
        "frame_relative_physical_reason_code": _readonly(
            distance_reason.reshape(-1), dtype=np.uint16
        ),
        "frame_body_source_row_id": _readonly(body_source.reshape(-1), dtype=np.int64),
        "frame_body_source_row_valid": _readonly(
            body_source_valid.reshape(-1), dtype=bool
        ),
        "frame_body_heading_deg": _readonly(heading.reshape(-1), dtype=np.float64),
        "frame_body_heading_valid": _readonly(heading_valid.reshape(-1), dtype=bool),
        "frame_body_heading_reason_code": _readonly(
            heading_reason.reshape(-1), dtype=np.uint16
        ),
        "frame_body_bearing_deg": _readonly(bearing.reshape(-1), dtype=np.float64),
        "frame_body_bearing_valid": _readonly(bearing_valid.reshape(-1), dtype=bool),
        "frame_body_bearing_reason_code": _readonly(
            bearing_reason.reshape(-1), dtype=np.uint16
        ),
        "frame_alignment_cos": _readonly(alignment_cos.reshape(-1), dtype=np.float64),
        "frame_lateral_sin": _readonly(lateral_sin.reshape(-1), dtype=np.float64),
        "frame_alignment_valid": _readonly(alignment_valid.reshape(-1), dtype=bool),
        "frame_alignment_reason_code": _readonly(
            alignment_reason.reshape(-1), dtype=np.uint8
        ),
        "distance_bin_edges_mm": _readonly(edges, dtype=np.float64),
    }
    integer_names = {
        "summary_epoch_role_code": np.uint8,
        "summary_epoch_window_id": np.int64,
        "summary_chaser_identity_code": np.uint16,
        "summary_chaser_behavior_role_code": np.uint8,
        "summary_distance_bin_index": np.int32,
        "summary_candidate_row_count": np.int64,
        "summary_joint_valid_row_count": np.int64,
        "summary_body_source_missing_row_count": np.int64,
        "summary_body_heading_invalid_row_count": np.int64,
        "summary_body_bearing_invalid_row_count": np.int64,
        "summary_other_alignment_invalid_row_count": np.int64,
        "summary_epoch_occurrence_row_count": np.int64,
        "summary_epoch_chaser_absent_row_count": np.int64,
        "summary_epoch_distance_valid_row_count": np.int64,
        "summary_epoch_distance_invalid_row_count": np.int64,
        "summary_epoch_distance_invalid_body_valid_row_count": np.int64,
    }
    for name, values in summary.items():
        arrays[name] = _readonly(values, dtype=integer_names.get(name, np.float64))
    if set(arrays) != set(PERSISTED_ARRAY_NAMES):  # pragma: no cover - internal guard
        _fail("Body-alignment persisted array inventory is incomplete.")
    readonly = MappingProxyType(arrays)

    fish_authority = _plain(inputs.fish_position_authority)
    body_authority = _plain(inputs.body_frame_authority)
    provider_id = _text(fish_authority.get("provider_id"), field="fish provider ID")
    provider_digest = _digest(
        fish_authority.get("provider_digest"), field="fish provider digest"
    )
    _text(body_authority.get("provider_id"), field="body-frame provider ID")
    _digest(body_authority.get("provider_digest"), field="body-frame provider digest")
    scale = _plain(inputs.scale_policy)
    if scale.get("unit") != "mm":
        _fail("Body alignment requires the relative child physical unit to be mm.")
    epoch_records = [
        {
            "epoch_role_code": index,
            "analysis_role": epoch.analysis_role,
            "window_id": int(epoch.window_id),
            "source_label": epoch.source_label,
            "start_frame": int(epoch.start_frame),
            "end_frame_exclusive": int(epoch.end_frame),
            "source_interval_sha256": epoch.source_interval_sha256,
        }
        for index, epoch in enumerate(epochs, start=1)
    ]
    relative_source_binding = {
        "run_path": relative_path,
        "manifest_sha256": relative_digest,
    }
    if inputs.relative_frame_validation_receipt_sha256 is not None:
        relative_source_binding.update(
            {
                "verification_mode": _text(
                    inputs.relative_frame_verification_mode,
                    field="relative-frame verification mode",
                ),
                "validation_receipt_sha256": _digest(
                    inputs.relative_frame_validation_receipt_sha256,
                    field="relative-frame validation receipt digest",
                ),
            }
        )
    body = {
        "scientific_schema": {"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION},
        "method_id": METHOD_ID,
        "recording_id": recording_id,
        **(
            {"core_authority": _plain(core_authority)}
            if core_authority is not None
            else {}
        ),
        "dimensions": {
            "n_frames": n_frames,
            "n_chasers": n_chasers,
            "n_frame_rows": n_frames * n_chasers,
            "n_epochs": len(epochs),
            "n_distance_bins": n_bins,
            "n_summary_rows": len(epochs) * n_chasers * n_bins,
        },
        "sources": {
            "relative_frame": relative_source_binding,
            "protocol_semantic_selection": {
                "run_path": semantic_path,
                "manifest_sha256": semantic_digest,
            },
            "fish_position_authority": fish_authority,
            "body_frame_authority": body_authority,
            "scale_policy": scale,
        },
        "position_provider": {
            "provider_id": provider_id,
            "provider_digest": provider_digest,
            "distance_surface": "base/relative_distance_physical",
            "body_origin_distance_substitution": "prohibited",
        },
        "coordinate_and_angle_convention": {
            "camera_axes": "+x_right_+y_down",
            "heading_world_projection": "atan2(-forward_y,forward_x)",
            "body_bearing": "atan2(anatomical_left,anatomical_forward)",
            "bearing_positive_direction": "toward_anatomical_left",
            "bearing_range_degrees": "[-180,180]",
            "alignment_cos": "cos(deg2rad(body_bearing_deg))",
            "lateral_sin": "sin(deg2rad(body_bearing_deg))",
            "convention_id": ANGLE_CONVENTION_ID,
        },
        "distance_bin_recipe": {**bin_recipe, "recipe_sha256": bin_recipe_sha256},
        "epoch_records": epoch_records,
        "epoch_records_sha256": canonical_json_sha256(epoch_records),
        "identity_registries": _plain(inputs.identity_registries),
        "reason_codes": {
            "alignment": {
                str(code): value for code, value in ALIGNMENT_REASON_CODES.items()
            },
            "source_reason_codes": "retained_verbatim_from_relative_frame_arrays",
        },
        "denominators": {
            "candidate_row_count": "epoch_and_occurrence_and_base_physical_distance_valid_in_persisted_bin",
            "joint_valid_row_count": "candidate_and_anatomical_body_bearing_valid",
            "distance_invalid_policy": "retained_in_epoch_counts_not_assigned_to_a_distance_bin",
            "body_valid_distance_invalid_policy": "retained_in_epoch_distance_invalid_body_valid_count_and_never_binned",
            "missing_body_policy": "retained_invalid_no_motion_heading_fallback",
            "interpolation": "prohibited",
            "viewer_rebinning": "prohibited",
        },
        "array_declarations": _array_declarations(readonly),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze({**body, "payload_digest": canonical_json_sha256(body)})
    return PreparedChaserBodyAlignmentByDistance(
        recording_id=recording_id,
        n_frames=n_frames,
        n_chasers=n_chasers,
        n_frame_rows=n_frames * n_chasers,
        n_summary_rows=len(summary["summary_epoch_role_code"]),
        n_distance_bins=n_bins,
        arrays=readonly,
        manifest=manifest,
    )


_REQUIRED_BASE_ARRAYS = (
    "acquisition_frame_id",
    "selection_member",
    "chaser_occurrence_member",
    "chaser_identity_code",
    "chaser_behavior_role_code",
    "chaser_behavior_role_valid",
    "relative_distance_physical",
    "relative_physical_valid",
    "relative_physical_reason_code",
)
_REQUIRED_BODY_ARRAYS = (
    "body_source_row_id",
    "body_source_row_valid",
    "body_heading_deg",
    "body_heading_valid",
    "body_heading_reason_code",
    "body_bearing_deg",
    "body_bearing_valid",
    "body_bearing_reason_code",
)


def chaser_body_alignment_input_from_handles(
    relative: Any,
    semantic_selection: Any,
    *,
    distance_bin_width_mm: float = DISTANCE_BIN_WIDTH_MM,
) -> ChaserBodyAlignmentByDistanceInput:
    """Bind exact current relative/body and semantic handles without fallback."""

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
        ChaserRelativeFrameTargetedSourceHandle,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    if type(relative) not in {
        ChaserRelativeFrameSourceHandle,
        ChaserRelativeFrameTargetedSourceHandle,
    }:
        raise TypeError("relative must be one strict relative-frame handle.")
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be one strict semantic handle.")
    core_authority = core_paradigm_dependency_from_relative_frame(relative)
    semantic_selection.assert_current()
    if relative.recording_id != semantic_selection.recording_id:
        _fail("Body-alignment sources belong to different recordings.")
    if relative.analysis_zarr_path != semantic_selection.analysis_zarr:
        _fail("Body-alignment sources belong to different archives.")
    if (
        type(relative) is ChaserRelativeFrameSourceHandle
        and not relative.body_available
    ):
        _fail("The exact keypoint relative-frame child has no body extension.")
    missing_base = set(_REQUIRED_BASE_ARRAYS).difference(relative.base_arrays)
    missing_body = set(_REQUIRED_BODY_ARRAYS).difference(relative.body_arrays or {})
    if missing_base or missing_body:
        _fail(
            "Exact body-alignment source lacks required arrays: "
            + ", ".join(
                sorted(
                    [
                        *(f"base/{name}" for name in missing_base),
                        *(f"body/{name}" for name in missing_body),
                    ]
                )
            )
        )
    manifest = relative.run_manifest
    scale = manifest.get("scale_policy")
    if not isinstance(scale, Mapping):
        _fail("Relative-frame manifest lacks its scale policy.")
    authorities = relative.source_authorities
    fish_authority = authorities.get("fish_position")
    body_authority = authorities.get("body_frame")
    if not isinstance(fish_authority, Mapping) or not isinstance(
        body_authority, Mapping
    ):
        _fail(
            "Relative-frame source lacks exact fish-position or body-frame authority."
        )
    verification_mode = str(relative.verification_mode)
    receipt_digest = (
        str(relative.receipt_digest)
        if getattr(relative, "receipt_digest", None) is not None
        else None
    )
    return ChaserBodyAlignmentByDistanceInput(
        recording_id=relative.recording_id,
        relative_frame_run_path=relative.run_path,
        relative_frame_manifest_sha256=relative.manifest_sha256,
        semantic_selection_run_path=semantic_selection.run_path,
        semantic_selection_manifest_sha256=semantic_selection.manifest_sha256,
        n_frames=relative.n_frames,
        n_chasers=relative.n_chasers,
        acquisition_frame_id=relative.base_frame_chaser("acquisition_frame_id"),
        selection_member=relative.base_frame_chaser("selection_member"),
        chaser_occurrence_member=relative.base_frame_chaser("chaser_occurrence_member"),
        chaser_identity_code=relative.base_frame_chaser("chaser_identity_code"),
        chaser_behavior_role_code=relative.base_frame_chaser(
            "chaser_behavior_role_code"
        ),
        chaser_behavior_role_valid=relative.base_frame_chaser(
            "chaser_behavior_role_valid"
        ),
        relative_distance_physical=relative.base_frame_chaser(
            "relative_distance_physical"
        ),
        relative_physical_valid=relative.base_frame_chaser("relative_physical_valid"),
        relative_physical_reason_code=relative.base_frame_chaser(
            "relative_physical_reason_code"
        ),
        body_source_row_id=relative.body_frame_chaser("body_source_row_id"),
        body_source_row_valid=relative.body_frame_chaser("body_source_row_valid"),
        body_heading_deg=relative.body_frame_chaser("body_heading_deg"),
        body_heading_valid=relative.body_frame_chaser("body_heading_valid"),
        body_heading_reason_code=relative.body_frame_chaser("body_heading_reason_code"),
        body_bearing_deg=relative.body_frame_chaser("body_bearing_deg"),
        body_bearing_valid=relative.body_frame_chaser("body_bearing_valid"),
        body_bearing_reason_code=relative.body_frame_chaser("body_bearing_reason_code"),
        epochs=semantic_selection.position_suite_epochs(),
        fish_position_authority=fish_authority,
        body_frame_authority=body_authority,
        identity_registries=(
            relative.identity_registries
            if type(relative) is ChaserRelativeFrameSourceHandle
            else manifest.get("identity_registries", {})
        ),
        scale_policy=scale,
        core_authority_dependency=core_authority,
        relative_frame_verification_mode=verification_mode,
        relative_frame_validation_receipt_sha256=receipt_digest,
        distance_bin_width_mm=distance_bin_width_mm,
    )


def prepare_chaser_body_alignment_by_distance_successor_from_handles(
    relative: Any,
    semantic_selection: Any,
    *,
    distance_bin_width_mm: float = DISTANCE_BIN_WIDTH_MM,
) -> PreparedChaserBodyAlignmentByDistance:
    return prepare_chaser_body_alignment_by_distance_successor(
        chaser_body_alignment_input_from_handles(
            relative,
            semantic_selection,
            distance_bin_width_mm=distance_bin_width_mm,
        )
    )


__all__ = [
    "ALIGNMENT_REASON_CODES",
    "ANGLE_CONVENTION_ID",
    "DISTANCE_BIN_POLICY_ID",
    "DISTANCE_BIN_WIDTH_MM",
    "FRAME_ARRAY_NAMES",
    "METHOD_ID",
    "PERSISTED_ARRAY_NAMES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SUMMARY_ARRAY_NAMES",
    "SUMMARY_VIEW_ARRAY_NAMES",
    "ChaserBodyAlignmentByDistanceError",
    "ChaserBodyAlignmentByDistanceInput",
    "PreparedChaserBodyAlignmentByDistance",
    "chaser_body_alignment_input_from_handles",
    "prepare_chaser_body_alignment_by_distance_successor",
    "prepare_chaser_body_alignment_by_distance_successor_from_handles",
]
