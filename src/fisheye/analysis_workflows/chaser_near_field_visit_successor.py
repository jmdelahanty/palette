"""Persist exact individual fish--chaser near-field visits and samples.

The aggregate radial/near-field successor remains unchanged.  This additive
successor binds that exact child, proves the shared state machine reproduces
its temporal summaries, and retains one row per real-chaser visit plus a flat
ragged sample table.  It never discovers a selector, interpolates a gap, or
drops a short visit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.provider_chaser_position_suite import PositionSuiteEpoch
from fisheye.analysis_workflows.chaser_radial_near_field_successor import (
    SCHEMA_ID as RADIAL_SCHEMA_ID,
    SCHEMA_VERSION as RADIAL_SCHEMA_VERSION,
    TIMING_POLICY_ID,
    VISIT_POLICY_ID,
)
from fisheye.analysis_workflows.chaser_relative_distance_view import (
    load_chaser_relative_distance_view,
)
from fisheye.analysis_workflows.core_paradigm_authority import (
    core_paradigm_dependency_from_relative_frame,
    validate_core_paradigm_source_dependency,
)
from fisheye.analysis_workflows.near_field_visit_state_machine import (
    LEFT_CENSOR_INVALID_GAP,
    LEFT_CENSOR_NONE,
    LEFT_CENSOR_PHASE_START,
    LEFT_CENSOR_REASONS,
    RIGHT_CENSOR_INVALID_GAP,
    RIGHT_CENSOR_NONE,
    RIGHT_CENSOR_PHASE_END,
    RIGHT_CENSOR_REASONS,
    ExactNearFieldVisitSegmentation,
    segment_exact_time_near_field_visits,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SCHEMA_ID = "palette.analysis.chaser_near_field_visits"
SCHEMA_VERSION = 1
METHOD_ID = "exact_real_chaser_hysteretic_visits_and_ragged_samples_v1"
REFERENCE_KIND_REAL_CHASER = 1
REFERENCE_KIND_REGISTRY = {REFERENCE_KIND_REAL_CHASER: "real_chaser"}
QUALITY_OK = 0
QUALITY_SHORT = 1
QUALITY_REGISTRY = {QUALITY_OK: "ok", QUALITY_SHORT: "short_visit_retained"}
MIN_VISIT_SAMPLE_COUNT = 5
CANONICAL_REASON_OK = 0
CANONICAL_REASON_REFERENCE_AT_ARENA_CENTER = 1
CANONICAL_REASON_REGISTRY = {
    CANONICAL_REASON_OK: "ok",
    CANONICAL_REASON_REFERENCE_AT_ARENA_CENTER: "reference_at_arena_center",
}
KEY_ENCODING = "lowercase_sha256_raw_32_bytes"
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SELECTOR_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "selected",
        "authoritative",
        "default",
    }
)

RADIAL_PARITY_INT_FIELDS = (
    "near_zone_entry_count",
    "near_zone_invalid_gap_count",
    "near_zone_censor_event_count",
    "near_zone_boundary_censor_event_count",
    "near_zone_invalid_gap_censor_event_count",
)
RADIAL_PARITY_FLOAT_FIELDS = (
    "near_zone_dwell_s",
    "near_zone_entry_rate_per_min_valid_time",
    "near_zone_valid_tracked_duration_s",
    "near_zone_complete_visit_median_dwell_s",
    "near_zone_complete_visit_total_dwell_s",
)
RADIAL_PARITY_ARRAY_NAMES = (
    "metric_epoch_window_id",
    "metric_chaser_column",
    "metric_chaser_identity_code",
    "metric_behavior_role_code",
    *(f"metric_{name}" for name in RADIAL_PARITY_INT_FIELDS),
    *(f"metric_{name}" for name in RADIAL_PARITY_FLOAT_FIELDS),
)
RELATIVE_FRAME_ARRAY_NAMES = (
    "acquisition_frame_id",
    "track_sample_id",
    "timestamp_ns",
    "timestamp_valid",
    "fish_source_row_id",
    "fish_source_row_valid",
    "fish_position_xy_px",
    "fish_position_valid",
    "chaser_source_row_id",
    "chaser_source_row_valid",
    "chaser_position_xy_px",
    "chaser_position_valid",
    "relative_vector_physical_xy",
    "relative_distance_physical",
    "relative_physical_valid",
    "selection_member",
    "chaser_occurrence_member",
    "chaser_behavior_role_code",
    "chaser_behavior_role_valid",
    "chaser_identity_code",
)
RELATIVE_FRAME_COLLAPSED_ARRAY_NAMES = (
    "acquisition_frame_id",
    "track_sample_id",
    "timestamp_ns",
    "timestamp_valid",
    "fish_source_row_id",
    "fish_source_row_valid",
    "fish_position_xy_px",
    "fish_position_valid",
    "selection_member",
)


class ChaserNearFieldVisitSuccessorError(ValueError):
    """Raised when individual visits cannot remain exact and auditable."""


def _fail(message: str) -> None:
    raise ChaserNearFieldVisitSuccessorError(message)


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


def _exact_run_path(value: object, *, field: str, parent: str) -> str:
    result = _text(value, field=field)
    prefix = f"{parent}/"
    name = result.removeprefix(prefix)
    if (
        not result.startswith(prefix)
        or not name
        or "/" in name
        or "\\" in name
        or name.casefold() in _SELECTOR_NAMES
        or _RUN_NAME_RE.fullmatch(name) is None
    ):
        _fail(f"{field} must name one exact concrete run below {parent!r}.")
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
            "dtype": np.asarray(array).dtype.str,
            "shape": list(np.asarray(array).shape),
            "content_sha256": array_values_sha256(np.asarray(array)),
        }
        for name, array in sorted(arrays.items())
    ]


def _registry(value: Mapping[str, str], *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{field} must be one non-empty registry.")
    result: dict[str, str] = {}
    for key, item in value.items():
        code = _text(key, field=f"{field} code")
        label = _text(item, field=f"{field}[{code}]")
        try:
            numeric = int(code)
        except ValueError as exc:
            raise ChaserNearFieldVisitSuccessorError(
                f"{field} code {code!r} is not an integer."
            ) from exc
        if numeric < 0 or str(numeric) != code:
            _fail(f"{field} code {code!r} is not canonical.")
        result[code] = label
    if len(set(result.values())) != len(result):
        _fail(f"{field} labels must be unique.")
    return result


def _vector(
    value: Any,
    *,
    length: int,
    dtype: Any,
    field: str,
) -> np.ndarray:
    result = np.asarray(value)
    expected = np.dtype(dtype)
    if result.shape != (length,) or result.dtype != expected:
        _fail(
            f"{field} must have exact shape {(length,)!r} and dtype "
            f"{expected.str!r}."
        )
    return result


def _matrix(
    value: Any,
    *,
    shape: tuple[int, ...],
    field: str,
    dtype: Any | None = None,
    kind: str | None = None,
) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        _fail(f"{field} must have exact shape {shape!r}.")
    if dtype is not None and result.dtype != np.dtype(dtype):
        _fail(f"{field} must have exact dtype {np.dtype(dtype).str!r}.")
    if kind is not None and result.dtype.kind not in kind:
        _fail(f"{field} has unsupported dtype {result.dtype.str!r}.")
    return result


@dataclass(frozen=True, slots=True)
class ChaserNearFieldVisitInput:
    recording_id: str
    source_relative_frame_run_path: str
    source_relative_frame_manifest_sha256: str
    source_semantic_selection_run_path: str
    source_semantic_selection_manifest_sha256: str
    source_radial_near_field_run_path: str
    source_radial_near_field_manifest_sha256: str
    radial_near_field_manifest: Mapping[str, Any]
    radial_metric_arrays: Mapping[str, np.ndarray]
    fish_position_authority: Mapping[str, Any]
    timing_authority: Mapping[str, Any]
    n_frames: int
    n_chasers: int
    acquisition_frame_id: np.ndarray
    track_sample_id: np.ndarray
    timestamp_ns_session: np.ndarray
    timestamp_valid: np.ndarray
    fish_source_row_id: np.ndarray
    fish_source_row_valid: np.ndarray
    fish_xy_px: np.ndarray
    fish_valid: np.ndarray
    chaser_source_row_id: np.ndarray
    chaser_source_row_valid: np.ndarray
    chaser_xy_px: np.ndarray
    chaser_valid: np.ndarray
    relative_vector_mm: np.ndarray
    distance_mm: np.ndarray
    distance_valid: np.ndarray
    selection_member: np.ndarray
    chaser_occurrence_member: np.ndarray
    chaser_role_code: np.ndarray
    chaser_role_valid: np.ndarray
    chaser_identity_code: np.ndarray
    role_registry: Mapping[str, str]
    chaser_registry: Mapping[str, str]
    epochs: Sequence[PositionSuiteEpoch]
    arena_center_xy_px: np.ndarray
    arena_radius_px: float
    mm_per_pixel: float
    minimum_quality_sample_count: int = MIN_VISIT_SAMPLE_COUNT
    core_authority_dependency: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PreparedChaserNearFieldVisits:
    recording_id: str
    n_visits: int
    n_samples: int
    n_summary_rows: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown near-field visit array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _validate_radial_manifest(
    inputs: ChaserNearFieldVisitInput,
) -> tuple[dict[str, Any], float, float, float]:
    radial = _plain(inputs.radial_near_field_manifest)
    if not isinstance(radial, dict):
        _fail("radial_near_field_manifest must be one exact object.")
    payload = _digest(radial.get("payload_digest"), field="radial payload digest")
    body = dict(radial)
    body.pop("payload_digest")
    if canonical_json_sha256(body) != payload:
        _fail("Radial scientific manifest self digest is stale.")
    scientific = radial.get("scientific_schema")
    if scientific != {
        "schema_id": RADIAL_SCHEMA_ID,
        "schema_version": RADIAL_SCHEMA_VERSION,
    }:
        _fail("Radial source does not carry the exact supported scientific schema.")
    if radial.get("recording_id") != inputs.recording_id:
        _fail("Radial source belongs to another recording.")
    sources = radial.get("sources")
    if not isinstance(sources, Mapping):
        _fail("Radial source manifest lacks exact dependencies.")
    expected = {
        "relative_frame": {
            "run_path": inputs.source_relative_frame_run_path,
            "manifest_sha256": inputs.source_relative_frame_manifest_sha256,
        },
        "protocol_semantic_selection": {
            "run_path": inputs.source_semantic_selection_run_path,
            "manifest_sha256": inputs.source_semantic_selection_manifest_sha256,
        },
    }
    for key, value in expected.items():
        if _plain(sources.get(key)) != value:
            _fail(f"Radial source binds a different {key} dependency.")
    if _plain(sources.get("fish_position")) != _plain(inputs.fish_position_authority):
        _fail("Radial source binds a different fish-position authority.")
    if _plain(sources.get("timing")) != _plain(inputs.timing_authority):
        _fail("Radial source binds a different timing authority.")
    registries = radial.get("identity_registries")
    if not isinstance(registries, Mapping):
        _fail("Radial source lacks exact identity registries.")
    if _plain(registries.get("behavior_role")) != _plain(inputs.role_registry):
        _fail("Radial and relative behavior-role registries differ.")
    if _plain(registries.get("chaser")) != _plain(inputs.chaser_registry):
        _fail("Radial and relative chaser registries differ.")
    policies = radial.get("policies")
    if not isinstance(policies, Mapping) or policies.get("visit") != VISIT_POLICY_ID:
        _fail("Radial source lacks the exact supported visit policy.")
    if policies.get("temporal_integration") != TIMING_POLICY_ID:
        _fail("Radial source lacks the exact supported timing policy.")
    config = radial.get("config")
    if not isinstance(config, Mapping):
        _fail("Radial source lacks its exact configuration.")
    radii: list[float] = []
    for key in (
        "near_zone_radius_mm",
        "near_entry_radius_mm",
        "near_exit_radius_mm",
    ):
        value = float(config.get(key, math.nan))
        if not math.isfinite(value) or value < 0.0:
            _fail(f"Radial configuration {key!r} is invalid.")
        radii.append(value)
    if radii[2] <= radii[1]:
        _fail("Radial exit radius is not greater than its entry radius.")
    position = radial.get("position_provider")
    fish = inputs.fish_position_authority
    if not isinstance(position, Mapping) or (
        position.get("provider_id") != fish.get("provider_id")
        or position.get("provider_digest") != fish.get("provider_digest")
    ):
        _fail("Radial position-provider identity differs from the relative source.")
    arena = radial.get("arena")
    if not isinstance(arena, Mapping):
        _fail("Radial source lacks reviewed arena geometry.")
    center = np.asarray(inputs.arena_center_xy_px, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        _fail("arena_center_xy_px must be one finite XY point.")
    if not np.array_equal(
        center,
        np.asarray([arena.get("center_x_px"), arena.get("center_y_px")]),
    ):
        _fail("Input arena centre differs from the radial source.")
    if float(arena.get("radius_px", math.nan)) != float(inputs.arena_radius_px):
        _fail("Input arena radius differs from the radial source.")
    if not math.isfinite(inputs.mm_per_pixel) or inputs.mm_per_pixel <= 0.0:
        _fail("mm_per_pixel must be finite and positive.")
    radial_radius_mm = float(arena.get("radius_mm", math.nan))
    if not math.isclose(
        radial_radius_mm,
        float(inputs.arena_radius_px) * float(inputs.mm_per_pixel),
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        _fail("Relative-frame scale differs from radial reviewed arena scale.")
    return radial, radii[0], radii[1], radii[2]


def _radial_rows(
    inputs: ChaserNearFieldVisitInput,
) -> tuple[dict[tuple[int, int], int], dict[str, np.ndarray]]:
    source = inputs.radial_metric_arrays
    if not isinstance(source, Mapping) or set(source) != set(RADIAL_PARITY_ARRAY_NAMES):
        _fail("Radial parity array roster is inexact.")
    arrays = {name: np.asarray(source[name]) for name in RADIAL_PARITY_ARRAY_NAMES}
    lengths = {array.shape[0] for array in arrays.values() if array.ndim == 1}
    if len(lengths) != 1 or any(array.ndim != 1 for array in arrays.values()):
        _fail("Radial parity arrays must share one one-dimensional row axis.")
    row_count = next(iter(lengths), 0)
    if row_count != len(inputs.epochs) * inputs.n_chasers:
        _fail("Radial metric row count differs from epoch-by-chaser cardinality.")
    integer_names = {
        "metric_epoch_window_id",
        "metric_chaser_column",
        "metric_chaser_identity_code",
        "metric_behavior_role_code",
        *(f"metric_{name}" for name in RADIAL_PARITY_INT_FIELDS),
    }
    for name, array in arrays.items():
        expected_kind = "iu" if name in integer_names else "f"
        if array.dtype.kind not in expected_kind or array.dtype.hasobject:
            _fail(f"Radial parity array {name!r} has an invalid dtype.")
    lookup: dict[tuple[int, int], int] = {}
    for row in range(row_count):
        key = (
            int(arrays["metric_epoch_window_id"][row]),
            int(arrays["metric_chaser_column"][row]),
        )
        if key in lookup:
            _fail("Radial metric rows contain a duplicate epoch/chaser key.")
        lookup[key] = row
    return lookup, arrays


def _same_scalar(left: object, right: object) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(left_float) and math.isnan(right_float):
        return True
    return left_float == right_float


def _assert_radial_parity(
    segmented: ExactNearFieldVisitSegmentation,
    *,
    arrays: Mapping[str, np.ndarray],
    row: int,
) -> None:
    values: Mapping[str, object] = {
        "near_zone_entry_count": segmented.entry_count,
        "near_zone_invalid_gap_count": segmented.invalid_gap_count,
        "near_zone_censor_event_count": segmented.censor_event_count,
        "near_zone_boundary_censor_event_count": (
            segmented.boundary_censor_event_count
        ),
        "near_zone_invalid_gap_censor_event_count": (
            segmented.invalid_gap_censor_event_count
        ),
        "near_zone_dwell_s": segmented.near_dwell_s,
        "near_zone_entry_rate_per_min_valid_time": (segmented.entry_rate_per_min),
        "near_zone_valid_tracked_duration_s": (segmented.valid_tracked_duration_s),
        "near_zone_complete_visit_median_dwell_s": (
            segmented.complete_visit_median_dwell_s
        ),
        "near_zone_complete_visit_total_dwell_s": (
            segmented.complete_visit_total_dwell_s
        ),
    }
    for field, value in values.items():
        persisted = arrays[f"metric_{field}"][row]
        if not _same_scalar(persisted, value):
            _fail(
                f"Individual visit state machine differs from radial aggregate "
                f"field {field!r}."
            )


def _path_length(xy: np.ndarray) -> float:
    if xy.shape[0] <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def _displacement(xy: np.ndarray) -> float:
    if xy.shape[0] <= 1:
        return 0.0
    return float(np.linalg.norm(xy[-1] - xy[0]))


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan, math.nan
    result = np.quantile(array, (0.25, 0.5, 0.75))
    return float(result[0]), float(result[1]), float(result[2])


def _event_near_dwell_s(
    event_indices: np.ndarray,
    *,
    exit_index: int | None,
    timestamp: np.ndarray,
    distance: np.ndarray,
    near_zone_mm: float,
) -> float:
    dwell = 0.0
    for left, right in zip(event_indices[:-1], event_indices[1:], strict=True):
        if distance[left] <= near_zone_mm:
            dwell += float(timestamp[right] - timestamp[left]) / 1e9
    if exit_index is not None and event_indices.size:
        left = int(event_indices[-1])
        if distance[left] <= near_zone_mm:
            dwell += float(timestamp[exit_index] - timestamp[left]) / 1e9
    return dwell


def _columnize(
    visit_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    visit_types: Mapping[str, Any] = {
        "visit_row_id": np.int64,
        "visit_key_sha256_bytes": np.uint8,
        "visit_ordinal": np.int32,
        "visit_epoch_role_code": np.uint8,
        "visit_epoch_window_id": np.int64,
        "visit_reference_kind_code": np.uint8,
        "visit_chaser_column": np.int32,
        "visit_chaser_identity_code": np.uint16,
        "visit_behavior_role_code": np.uint8,
        "visit_sample_offset": np.int64,
        "visit_sample_count": np.int64,
        "visit_first_relative_frame_row": np.int64,
        "visit_first_acquisition_frame_id": np.int64,
        "visit_first_timestamp_ns_session": np.int64,
        "visit_entry_relative_frame_row": np.int64,
        "visit_entry_acquisition_frame_id": np.int64,
        "visit_entry_timestamp_ns_session": np.int64,
        "visit_entry_observed": bool,
        "visit_last_inside_relative_frame_row": np.int64,
        "visit_last_inside_acquisition_frame_id": np.int64,
        "visit_last_inside_timestamp_ns_session": np.int64,
        "visit_exit_relative_frame_row": np.int64,
        "visit_exit_acquisition_frame_id": np.int64,
        "visit_exit_timestamp_ns_session": np.int64,
        "visit_exit_observed": bool,
        "visit_complete": bool,
        "visit_left_censor_reason_code": np.uint8,
        "visit_right_censor_reason_code": np.uint8,
        "visit_censor_side_count": np.uint8,
        "visit_quality_code": np.uint8,
        "visit_observed_active_duration_s": np.float64,
        "visit_complete_dwell_s": np.float64,
        "visit_near_zone_dwell_s": np.float64,
        "visit_min_distance_mm": np.float64,
        "visit_max_distance_mm": np.float64,
        "visit_cpa_sample_ordinal": np.int32,
        "visit_cpa_acquisition_frame_id": np.int64,
        "visit_cpa_timestamp_ns_session": np.int64,
        "visit_fish_path_length_mm": np.float64,
        "visit_relative_path_length_mm": np.float64,
        "visit_fish_displacement_mm": np.float64,
        "visit_relative_displacement_mm": np.float64,
    }
    sample_types: Mapping[str, Any] = {
        "sample_visit_row_id": np.int64,
        "sample_ordinal_within_visit": np.int32,
        "sample_relative_frame_row": np.int64,
        "sample_acquisition_frame_id": np.int64,
        "sample_track_sample_id": np.int64,
        "sample_timestamp_ns_session": np.int64,
        "sample_fish_source_row_id": np.int64,
        "sample_chaser_source_row_id": np.int64,
        "sample_fish_x_mm": np.float64,
        "sample_fish_y_mm": np.float64,
        "sample_reference_x_mm": np.float64,
        "sample_reference_y_mm": np.float64,
        "sample_relative_x_mm": np.float64,
        "sample_relative_y_mm": np.float64,
        "sample_canonical_x_mm": np.float64,
        "sample_canonical_y_mm": np.float64,
        "sample_canonical_valid": bool,
        "sample_canonical_reason_code": np.uint8,
        "sample_distance_mm": np.float64,
        "sample_near_zone_member": bool,
        "sample_time_from_first_sample_s": np.float64,
        "sample_time_from_observed_entry_s": np.float64,
    }
    summary_types: Mapping[str, Any] = {
        "summary_row_id": np.int64,
        "summary_epoch_role_code": np.uint8,
        "summary_epoch_window_id": np.int64,
        "summary_chaser_column": np.int32,
        "summary_chaser_identity_code": np.uint16,
        "summary_behavior_role_code": np.uint8,
        "summary_total_visit_count": np.int64,
        "summary_observed_entry_visit_count": np.int64,
        "summary_complete_visit_count": np.int64,
        "summary_censored_visit_count": np.int64,
        "summary_boundary_censor_event_count": np.int64,
        "summary_invalid_gap_censor_event_count": np.int64,
        "summary_invalid_gap_count": np.int64,
        "summary_short_visit_count": np.int64,
        "summary_visit_sample_count": np.int64,
        "summary_near_zone_dwell_s": np.float64,
        "summary_visit_assigned_near_zone_dwell_s": np.float64,
        "summary_unassigned_near_zone_dwell_s": np.float64,
        "summary_valid_tracked_duration_s": np.float64,
        "summary_entry_rate_per_min_valid_time": np.float64,
        "summary_complete_visit_total_dwell_s": np.float64,
        "summary_complete_visit_mean_dwell_s": np.float64,
        "summary_complete_visit_p25_dwell_s": np.float64,
        "summary_complete_visit_median_dwell_s": np.float64,
        "summary_complete_visit_p75_dwell_s": np.float64,
        "summary_visit_min_distance_p25_mm": np.float64,
        "summary_visit_min_distance_median_mm": np.float64,
        "summary_visit_min_distance_p75_mm": np.float64,
    }
    for name, dtype in visit_types.items():
        if name == "visit_key_sha256_bytes":
            value = np.asarray([row[name] for row in visit_rows], dtype=dtype).reshape(
                len(visit_rows), 32
            )
        else:
            value = np.asarray([row[name] for row in visit_rows], dtype=dtype)
        arrays[name] = _readonly(value)
    for name, dtype in sample_types.items():
        arrays[name] = _readonly([row[name] for row in sample_rows], dtype=dtype)
    for name, dtype in summary_types.items():
        arrays[name] = _readonly([row[name] for row in summary_rows], dtype=dtype)
    return arrays


def _validate_ragged_tables(arrays: Mapping[str, np.ndarray]) -> None:
    visit_ids = np.asarray(arrays["visit_row_id"], dtype=np.int64)
    offsets = np.asarray(arrays["visit_sample_offset"], dtype=np.int64)
    counts = np.asarray(arrays["visit_sample_count"], dtype=np.int64)
    sample_visit = np.asarray(arrays["sample_visit_row_id"], dtype=np.int64)
    sample_ordinal = np.asarray(arrays["sample_ordinal_within_visit"], dtype=np.int64)
    if not np.array_equal(visit_ids, np.arange(visit_ids.size, dtype=np.int64)):
        _fail("Visit row IDs are not one canonical dense axis.")
    expected_offset = 0
    for row, (offset, count) in enumerate(zip(offsets, counts, strict=True)):
        if int(offset) != expected_offset or int(count) <= 0:
            _fail("Visit sample offsets are not gapless positive ragged slices.")
        stop = int(offset + count)
        if stop > sample_visit.size:
            _fail("Visit sample slice exceeds the flattened sample table.")
        if not np.all(sample_visit[int(offset) : stop] == row):
            _fail("Flattened sample rows reference the wrong visit row.")
        if not np.array_equal(
            sample_ordinal[int(offset) : stop],
            np.arange(int(count), dtype=np.int64),
        ):
            _fail("Within-visit sample ordinals are not canonical.")
        expected_offset = stop
    if expected_offset != sample_visit.size:
        _fail("Visit ragged slices do not cover the exact sample table.")
    keys = np.asarray(arrays["visit_key_sha256_bytes"], dtype=np.uint8)
    if keys.shape != (visit_ids.size, 32):
        _fail("Visit-key byte rows have an invalid shape.")
    if visit_ids.size and np.unique(keys, axis=0).shape[0] != visit_ids.size:
        _fail("Visit-key byte rows are not unique.")


def prepare_chaser_near_field_visit_successor(
    inputs: ChaserNearFieldVisitInput,
) -> PreparedChaserNearFieldVisits:
    """Build immutable real-chaser visit rows and flattened exact samples."""

    if type(inputs) is not ChaserNearFieldVisitInput:
        raise TypeError("inputs must be one ChaserNearFieldVisitInput.")
    recording_id = _text(inputs.recording_id, field="recording_id")
    relative_path = _exact_run_path(
        inputs.source_relative_frame_run_path,
        field="relative-frame run path",
        parent="analysis/chaser_relative_frame_runs",
    )
    relative_digest = _digest(
        inputs.source_relative_frame_manifest_sha256,
        field="relative-frame manifest digest",
    )
    semantic_path = _exact_run_path(
        inputs.source_semantic_selection_run_path,
        field="semantic-selection run path",
        parent="analysis/protocol_semantic_chaser_selection_runs",
    )
    semantic_digest = _digest(
        inputs.source_semantic_selection_manifest_sha256,
        field="semantic-selection manifest digest",
    )
    radial_path = _exact_run_path(
        inputs.source_radial_near_field_run_path,
        field="radial/near-field run path",
        parent="analysis/chaser_radial_near_field_runs",
    )
    radial_digest = _digest(
        inputs.source_radial_near_field_manifest_sha256,
        field="radial/near-field manifest digest",
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
    if type(inputs.n_frames) is not int or inputs.n_frames <= 0:
        _fail("n_frames must be one positive exact integer.")
    if type(inputs.n_chasers) is not int or inputs.n_chasers <= 0:
        _fail("n_chasers must be one positive exact integer.")
    if (
        type(inputs.minimum_quality_sample_count) is not int
        or inputs.minimum_quality_sample_count <= 0
        or inputs.minimum_quality_sample_count > 65535
    ):
        _fail("minimum_quality_sample_count is invalid.")
    radial, near_zone_mm, enter_mm, exit_mm = _validate_radial_manifest(inputs)
    if _plain(radial.get("core_authority")) != _plain(core_authority):
        _fail("Radial and visit successors bind different core authorities.")
    provider_id = _text(
        inputs.fish_position_authority.get("provider_id"),
        field="fish-position provider ID",
    )
    provider_digest = _digest(
        inputs.fish_position_authority.get("provider_digest"),
        field="fish-position provider digest",
    )
    lookup, radial_arrays = _radial_rows(inputs)
    n = inputs.n_frames
    m = inputs.n_chasers
    frame = _vector(
        inputs.acquisition_frame_id,
        length=n,
        dtype=np.int64,
        field="acquisition_frame_id",
    )
    track = _vector(
        inputs.track_sample_id,
        length=n,
        dtype=np.int64,
        field="track_sample_id",
    )
    timestamp = _vector(
        inputs.timestamp_ns_session,
        length=n,
        dtype=np.int64,
        field="timestamp_ns_session",
    )
    timestamp_valid = _vector(
        inputs.timestamp_valid,
        length=n,
        dtype=bool,
        field="timestamp_valid",
    )
    fish_source_row = _vector(
        inputs.fish_source_row_id,
        length=n,
        dtype=np.int64,
        field="fish_source_row_id",
    )
    fish_source_valid = _vector(
        inputs.fish_source_row_valid,
        length=n,
        dtype=bool,
        field="fish_source_row_valid",
    )
    fish_xy = _matrix(
        inputs.fish_xy_px, shape=(n, 2), field="fish_xy_px", kind="f"
    ).astype(np.float64, copy=False)
    fish_valid = _vector(inputs.fish_valid, length=n, dtype=bool, field="fish_valid")
    chaser_source_row = _matrix(
        inputs.chaser_source_row_id,
        shape=(n, m),
        field="chaser_source_row_id",
        dtype=np.int64,
    )
    chaser_source_valid = _matrix(
        inputs.chaser_source_row_valid,
        shape=(n, m),
        field="chaser_source_row_valid",
        dtype=bool,
    )
    chaser_xy = _matrix(
        inputs.chaser_xy_px,
        shape=(n, m, 2),
        field="chaser_xy_px",
        kind="f",
    ).astype(np.float64, copy=False)
    chaser_valid = _matrix(
        inputs.chaser_valid,
        shape=(n, m),
        field="chaser_valid",
        dtype=bool,
    )
    relative = _matrix(
        inputs.relative_vector_mm,
        shape=(n, m, 2),
        field="relative_vector_mm",
        kind="f",
    ).astype(np.float64, copy=False)
    distance = _matrix(
        inputs.distance_mm, shape=(n, m), field="distance_mm", kind="f"
    ).astype(np.float64, copy=False)
    distance_valid = _matrix(
        inputs.distance_valid,
        shape=(n, m),
        field="distance_valid",
        dtype=bool,
    )
    selected = _vector(
        inputs.selection_member,
        length=n,
        dtype=bool,
        field="selection_member",
    )
    occurrence = _matrix(
        inputs.chaser_occurrence_member,
        shape=(n, m),
        field="chaser_occurrence_member",
        dtype=bool,
    )
    role = _matrix(
        inputs.chaser_role_code,
        shape=(n, m),
        field="chaser_role_code",
        dtype=np.uint8,
    )
    role_valid = _matrix(
        inputs.chaser_role_valid,
        shape=(n, m),
        field="chaser_role_valid",
        dtype=bool,
    )
    identity = _matrix(
        inputs.chaser_identity_code,
        shape=(n, m),
        field="chaser_identity_code",
        dtype=np.uint16,
    )
    if np.any(np.diff(frame) <= 0):
        _fail("Acquisition-frame IDs must be strictly increasing.")
    if np.any(frame < 0):
        _fail("Acquisition-frame IDs must be non-negative.")
    if np.any(track < 0) or np.unique(track).size != n:
        _fail(
            "track_sample_id must identify one non-negative exact source row per frame."
        )
    valid_timestamps = timestamp[timestamp_valid]
    if valid_timestamps.size < 2 or np.any(np.diff(valid_timestamps) <= 0):
        _fail("Valid exact session timestamps must be strictly increasing.")
    if np.any(timestamp_valid & (timestamp < 0)):
        _fail("Valid session timestamps must be non-negative.")
    if n and np.any(identity != identity[:1, :]):
        _fail("Chaser identity codes must be stable along each chaser column.")
    roles = _registry(inputs.role_registry, field="behavior-role registry")
    chasers = _registry(inputs.chaser_registry, field="chaser registry")
    for code in np.unique(identity):
        if str(int(code)) not in chasers:
            _fail("Chaser identity array contains an undeclared code.")
    for code in np.unique(role[role_valid]):
        if str(int(code)) not in roles:
            _fail("Behavior-role array contains an undeclared valid code.")
    usable = distance_valid & selected[:, None] & occurrence & role_valid
    if np.any(
        usable
        & ~(
            fish_valid[:, None]
            & fish_source_valid[:, None]
            & chaser_valid
            & chaser_source_valid
        )
    ):
        _fail("A valid visit distance lacks exact fish/chaser source-row evidence.")
    if np.any(usable & ((fish_source_row < 0)[:, None] | (chaser_source_row < 0))):
        _fail("A valid visit distance has a negative fish/chaser source-row identity.")
    finite_positions = (
        np.isfinite(fish_xy).all(axis=1)[:, None]
        & np.isfinite(chaser_xy).all(axis=2)
        & np.isfinite(relative).all(axis=2)
        & np.isfinite(distance)
    )
    if np.any(usable & ~finite_positions):
        _fail("A valid visit distance has non-finite coordinate evidence.")
    expected_relative = (chaser_xy - fish_xy[:, None, :]) * float(inputs.mm_per_pixel)
    if np.any(
        usable
        & ~np.isclose(
            relative,
            expected_relative,
            rtol=1e-5,
            atol=1e-5,
            equal_nan=False,
        ).all(axis=2)
    ):
        _fail(
            "Relative vectors disagree with the sealed fish/chaser positions and scale."
        )
    norms = np.linalg.norm(relative, axis=2)
    if np.any(
        usable & ~np.isclose(norms, distance, rtol=1e-5, atol=1e-5, equal_nan=False)
    ):
        _fail("Relative vectors and distances disagree on valid visit rows.")

    epochs = tuple(inputs.epochs)
    if not epochs or any(type(epoch) is not PositionSuiteEpoch for epoch in epochs):
        _fail("epochs must contain exact PositionSuiteEpoch records.")
    epoch_codes = {
        label: code
        for code, label in enumerate(
            dict.fromkeys(epoch.analysis_role for epoch in epochs), start=1
        )
    }
    if len(epoch_codes) > np.iinfo(np.uint8).max:
        _fail("Epoch-role cardinality exceeds the persisted uint8 registry.")
    if len({epoch.window_id for epoch in epochs}) != len(epochs):
        _fail("Epoch window IDs must be unique.")
    expected_epoch_records = [
        {
            "analysis_role": epoch.analysis_role,
            "window_id": epoch.window_id,
            "source_label": epoch.source_label,
            "start_frame": epoch.start_frame,
            "end_frame_exclusive": epoch.end_frame,
            "source_interval_sha256": epoch.source_interval_sha256,
        }
        for epoch in epochs
    ]
    if _plain(radial.get("epoch_records")) != expected_epoch_records:
        _fail("Radial source binds different semantic epoch records.")
    arena_center_mm = np.asarray(inputs.arena_center_xy_px, dtype=np.float64) * float(
        inputs.mm_per_pixel
    )
    fish_mm = fish_xy * float(inputs.mm_per_pixel)
    chaser_mm = chaser_xy * float(inputs.mm_per_pixel)

    visit_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for epoch in epochs:
        epoch_mask = (frame >= epoch.start_frame) & (frame < epoch.end_frame)
        epoch_rows = np.flatnonzero(epoch_mask)
        if epoch_rows.size == 0:
            _fail("A declared semantic epoch has no relative-frame rows.")
        for column in range(m):
            radial_key = (int(epoch.window_id), column)
            if radial_key not in lookup:
                _fail("Radial source lacks one exact epoch/chaser row.")
            radial_row = lookup[radial_key]
            chaser_code = int(identity[0, column])
            radial_chaser = int(
                radial_arrays["metric_chaser_identity_code"][radial_row]
            )
            if radial_chaser != chaser_code:
                _fail("Radial and relative chaser identities differ.")
            behavior_code = int(radial_arrays["metric_behavior_role_code"][radial_row])
            if str(behavior_code) not in roles:
                _fail("Radial metric row contains an undeclared behavior-role code.")
            candidate = selected & occurrence[:, column] & role_valid[:, column]
            role_rows = epoch_mask & candidate
            if np.any(role_rows & (role[:, column] != behavior_code)):
                _fail("Behavior role changes within one radial epoch/chaser row.")
            segmented = segment_exact_time_near_field_visits(
                frame_id=frame[epoch_rows],
                timestamp_ns=timestamp[epoch_rows],
                timestamp_valid=timestamp_valid[epoch_rows],
                distance_mm=distance[epoch_rows, column],
                distance_valid=(candidate & distance_valid[:, column])[epoch_rows],
                near_zone_mm=near_zone_mm,
                enter_mm=enter_mm,
                exit_mm=exit_mm,
            )
            _assert_radial_parity(segmented, arrays=radial_arrays, row=radial_row)
            summary_visit_rows: list[dict[str, Any]] = []
            for event in segmented.visits:
                first = int(epoch_rows[event.first_sample_index])
                last = int(epoch_rows[event.last_inside_index])
                entry = (
                    None
                    if event.entry_index is None
                    else int(epoch_rows[event.entry_index])
                )
                exit_index = (
                    None
                    if event.exit_index is None
                    else int(epoch_rows[event.exit_index])
                )
                member_rows = np.arange(first, last + 1, dtype=np.int64)
                if member_rows.size != event.sample_count or np.any(
                    ~(candidate & distance_valid[:, column])[member_rows]
                ):
                    _fail(
                        "Visit ragged membership differs from state-machine evidence."
                    )
                if member_rows.size > 1 and (
                    np.any(np.diff(frame[member_rows]) != 1)
                    or np.any(np.diff(timestamp[member_rows]) <= 0)
                ):
                    _fail("A visit sample span crosses a temporal evidence gap.")
                if np.any(role[member_rows, column] != behavior_code):
                    _fail("A visit crosses behavior-role identity.")
                sample_offset = len(sample_rows)
                relative_fish = -relative[member_rows, column, :]
                fish_path = fish_mm[member_rows]
                reference_path = chaser_mm[member_rows, column, :]
                canonical = np.full_like(relative_fish, np.nan)
                canonical_valid = np.zeros(member_rows.size, dtype=bool)
                canonical_reason = np.full(
                    member_rows.size,
                    CANONICAL_REASON_REFERENCE_AT_ARENA_CENTER,
                    dtype=np.uint8,
                )
                to_center = arena_center_mm[None, :] - reference_path
                center_distance = np.linalg.norm(to_center, axis=1)
                canonical_valid = center_distance > 1e-9
                canonical_reason[canonical_valid] = CANONICAL_REASON_OK
                theta = np.arctan2(to_center[:, 1], to_center[:, 0])
                cosine = np.cos(theta)
                sine = np.sin(theta)
                canonical[canonical_valid, 0] = (
                    relative_fish[canonical_valid, 0] * cosine[canonical_valid]
                    + relative_fish[canonical_valid, 1] * sine[canonical_valid]
                )
                canonical[canonical_valid, 1] = (
                    -relative_fish[canonical_valid, 0] * sine[canonical_valid]
                    + relative_fish[canonical_valid, 1] * cosine[canonical_valid]
                )
                entry_timestamp = timestamp[entry] if entry is not None else None
                for ordinal, source_row in enumerate(member_rows):
                    sample_rows.append(
                        {
                            "sample_visit_row_id": len(visit_rows),
                            "sample_ordinal_within_visit": ordinal,
                            "sample_relative_frame_row": int(source_row),
                            "sample_acquisition_frame_id": int(frame[source_row]),
                            "sample_track_sample_id": int(track[source_row]),
                            "sample_timestamp_ns_session": int(timestamp[source_row]),
                            "sample_fish_source_row_id": int(
                                fish_source_row[source_row]
                            ),
                            "sample_chaser_source_row_id": int(
                                chaser_source_row[source_row, column]
                            ),
                            "sample_fish_x_mm": float(fish_path[ordinal, 0]),
                            "sample_fish_y_mm": float(fish_path[ordinal, 1]),
                            "sample_reference_x_mm": float(reference_path[ordinal, 0]),
                            "sample_reference_y_mm": float(reference_path[ordinal, 1]),
                            "sample_relative_x_mm": float(relative_fish[ordinal, 0]),
                            "sample_relative_y_mm": float(relative_fish[ordinal, 1]),
                            "sample_canonical_x_mm": float(canonical[ordinal, 0]),
                            "sample_canonical_y_mm": float(canonical[ordinal, 1]),
                            "sample_canonical_valid": bool(canonical_valid[ordinal]),
                            "sample_canonical_reason_code": int(
                                canonical_reason[ordinal]
                            ),
                            "sample_distance_mm": float(distance[source_row, column]),
                            "sample_near_zone_member": bool(
                                distance[source_row, column] <= near_zone_mm
                            ),
                            "sample_time_from_first_sample_s": float(
                                timestamp[source_row] - timestamp[first]
                            )
                            / 1e9,
                            "sample_time_from_observed_entry_s": (
                                math.nan
                                if entry_timestamp is None
                                else float(timestamp[source_row] - entry_timestamp)
                                / 1e9
                            ),
                        }
                    )
                event_distance = distance[member_rows, column]
                cpa_ordinal = int(np.argmin(event_distance))
                cpa_row = int(member_rows[cpa_ordinal])
                key_body = {
                    "schema_id": SCHEMA_ID,
                    "schema_version": SCHEMA_VERSION,
                    "recording_id": recording_id,
                    "relative_frame": {
                        "run_path": relative_path,
                        "manifest_sha256": relative_digest,
                    },
                    "radial_near_field": {
                        "run_path": radial_path,
                        "manifest_sha256": radial_digest,
                    },
                    "epoch_window_id": int(epoch.window_id),
                    "epoch_source_interval_sha256": epoch.source_interval_sha256,
                    "reference_kind": "real_chaser",
                    "chaser_column": column,
                    "chaser_identity_code": chaser_code,
                    "provider_id": provider_id,
                    "provider_digest": provider_digest,
                    "visit_ordinal": event.ordinal,
                    "first_acquisition_frame_id": int(frame[first]),
                    "last_inside_acquisition_frame_id": int(frame[last]),
                }
                key = canonical_json_sha256(key_body)
                if key in seen_keys:
                    _fail("Deterministic visit-key collision detected.")
                seen_keys.add(key)
                complete_dwell_s = (
                    math.nan
                    if entry is None or exit_index is None
                    else float(timestamp[exit_index] - timestamp[entry]) / 1e9
                )
                event_near_dwell = _event_near_dwell_s(
                    member_rows,
                    exit_index=exit_index,
                    timestamp=timestamp,
                    distance=distance[:, column],
                    near_zone_mm=near_zone_mm,
                )
                row = {
                    "visit_row_id": len(visit_rows),
                    "visit_key_sha256_bytes": np.frombuffer(
                        bytes.fromhex(key), dtype=np.uint8
                    ),
                    "visit_ordinal": event.ordinal,
                    "visit_epoch_role_code": epoch_codes[epoch.analysis_role],
                    "visit_epoch_window_id": int(epoch.window_id),
                    "visit_reference_kind_code": REFERENCE_KIND_REAL_CHASER,
                    "visit_chaser_column": column,
                    "visit_chaser_identity_code": chaser_code,
                    "visit_behavior_role_code": behavior_code,
                    "visit_sample_offset": sample_offset,
                    "visit_sample_count": member_rows.size,
                    "visit_first_relative_frame_row": first,
                    "visit_first_acquisition_frame_id": int(frame[first]),
                    "visit_first_timestamp_ns_session": int(timestamp[first]),
                    "visit_entry_relative_frame_row": -1 if entry is None else entry,
                    "visit_entry_acquisition_frame_id": (
                        -1 if entry is None else int(frame[entry])
                    ),
                    "visit_entry_timestamp_ns_session": (
                        -1 if entry is None else int(timestamp[entry])
                    ),
                    "visit_entry_observed": event.entry_observed,
                    "visit_last_inside_relative_frame_row": last,
                    "visit_last_inside_acquisition_frame_id": int(frame[last]),
                    "visit_last_inside_timestamp_ns_session": int(timestamp[last]),
                    "visit_exit_relative_frame_row": (
                        -1 if exit_index is None else exit_index
                    ),
                    "visit_exit_acquisition_frame_id": (
                        -1 if exit_index is None else int(frame[exit_index])
                    ),
                    "visit_exit_timestamp_ns_session": (
                        -1 if exit_index is None else int(timestamp[exit_index])
                    ),
                    "visit_exit_observed": event.exit_observed,
                    "visit_complete": event.complete,
                    "visit_left_censor_reason_code": (event.left_censor_reason_code),
                    "visit_right_censor_reason_code": (event.right_censor_reason_code),
                    "visit_censor_side_count": int(
                        event.left_censor_reason_code != LEFT_CENSOR_NONE
                    )
                    + int(event.right_censor_reason_code != RIGHT_CENSOR_NONE),
                    "visit_quality_code": (
                        QUALITY_OK
                        if member_rows.size >= inputs.minimum_quality_sample_count
                        else QUALITY_SHORT
                    ),
                    "visit_observed_active_duration_s": float(
                        timestamp[last] - timestamp[first]
                    )
                    / 1e9,
                    "visit_complete_dwell_s": complete_dwell_s,
                    "visit_near_zone_dwell_s": event_near_dwell,
                    "visit_min_distance_mm": float(event_distance[cpa_ordinal]),
                    "visit_max_distance_mm": float(np.max(event_distance)),
                    "visit_cpa_sample_ordinal": cpa_ordinal,
                    "visit_cpa_acquisition_frame_id": int(frame[cpa_row]),
                    "visit_cpa_timestamp_ns_session": int(timestamp[cpa_row]),
                    "visit_fish_path_length_mm": _path_length(fish_path),
                    "visit_relative_path_length_mm": _path_length(relative_fish),
                    "visit_fish_displacement_mm": _displacement(fish_path),
                    "visit_relative_displacement_mm": _displacement(relative_fish),
                }
                visit_rows.append(row)
                summary_visit_rows.append(row)

            observed_entries = sum(
                bool(row["visit_entry_observed"]) for row in summary_visit_rows
            )
            boundary_censors = sum(
                int(row["visit_left_censor_reason_code"]) == LEFT_CENSOR_PHASE_START
                for row in summary_visit_rows
            ) + sum(
                int(row["visit_right_censor_reason_code"]) == RIGHT_CENSOR_PHASE_END
                for row in summary_visit_rows
            )
            gap_censors = sum(
                int(row["visit_left_censor_reason_code"]) == LEFT_CENSOR_INVALID_GAP
                for row in summary_visit_rows
            ) + sum(
                int(row["visit_right_censor_reason_code"]) == RIGHT_CENSOR_INVALID_GAP
                for row in summary_visit_rows
            )
            if (
                observed_entries != segmented.entry_count
                or boundary_censors != segmented.boundary_censor_event_count
                or gap_censors != segmented.invalid_gap_censor_event_count
                or boundary_censors + gap_censors != segmented.censor_event_count
            ):
                _fail("Persisted visit rows do not conserve state-machine events.")
            complete_dwell = [
                float(row["visit_complete_dwell_s"])
                for row in summary_visit_rows
                if bool(row["visit_complete"])
            ]
            dwell_p25, dwell_p50, dwell_p75 = _quantiles(complete_dwell)
            minima = [float(row["visit_min_distance_mm"]) for row in summary_visit_rows]
            min_p25, min_p50, min_p75 = _quantiles(minima)
            assigned_near = float(
                sum(float(row["visit_near_zone_dwell_s"]) for row in summary_visit_rows)
            )
            unassigned_near = segmented.near_dwell_s - assigned_near
            if unassigned_near < -1e-9:
                _fail("Visit-assigned near dwell exceeds the radial aggregate.")
            if abs(unassigned_near) <= 1e-9:
                unassigned_near = 0.0
            summary_rows.append(
                {
                    "summary_row_id": len(summary_rows),
                    "summary_epoch_role_code": epoch_codes[epoch.analysis_role],
                    "summary_epoch_window_id": int(epoch.window_id),
                    "summary_chaser_column": column,
                    "summary_chaser_identity_code": chaser_code,
                    "summary_behavior_role_code": behavior_code,
                    "summary_total_visit_count": len(summary_visit_rows),
                    "summary_observed_entry_visit_count": segmented.entry_count,
                    "summary_complete_visit_count": sum(
                        bool(row["visit_complete"]) for row in summary_visit_rows
                    ),
                    "summary_censored_visit_count": sum(
                        not bool(row["visit_complete"]) for row in summary_visit_rows
                    ),
                    "summary_boundary_censor_event_count": (
                        segmented.boundary_censor_event_count
                    ),
                    "summary_invalid_gap_censor_event_count": (
                        segmented.invalid_gap_censor_event_count
                    ),
                    "summary_invalid_gap_count": segmented.invalid_gap_count,
                    "summary_short_visit_count": sum(
                        int(row["visit_quality_code"]) == QUALITY_SHORT
                        for row in summary_visit_rows
                    ),
                    "summary_visit_sample_count": sum(
                        int(row["visit_sample_count"]) for row in summary_visit_rows
                    ),
                    "summary_near_zone_dwell_s": segmented.near_dwell_s,
                    "summary_visit_assigned_near_zone_dwell_s": assigned_near,
                    "summary_unassigned_near_zone_dwell_s": unassigned_near,
                    "summary_valid_tracked_duration_s": (
                        segmented.valid_tracked_duration_s
                    ),
                    "summary_entry_rate_per_min_valid_time": (
                        segmented.entry_rate_per_min
                    ),
                    "summary_complete_visit_total_dwell_s": (
                        segmented.complete_visit_total_dwell_s
                    ),
                    "summary_complete_visit_mean_dwell_s": (
                        float(np.mean(complete_dwell)) if complete_dwell else math.nan
                    ),
                    "summary_complete_visit_p25_dwell_s": dwell_p25,
                    "summary_complete_visit_median_dwell_s": dwell_p50,
                    "summary_complete_visit_p75_dwell_s": dwell_p75,
                    "summary_visit_min_distance_p25_mm": min_p25,
                    "summary_visit_min_distance_median_mm": min_p50,
                    "summary_visit_min_distance_p75_mm": min_p75,
                }
            )

    arrays = _columnize(visit_rows, sample_rows, summary_rows)
    _validate_ragged_tables(arrays)
    readonly = MappingProxyType(
        {name: _readonly(value) for name, value in arrays.items()}
    )
    epoch_registry = {str(code): label for label, code in epoch_codes.items()}
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
            "n_frames": n,
            "n_chasers": m,
            "n_visits": len(visit_rows),
            "n_samples": len(sample_rows),
            "n_summary_rows": len(summary_rows),
        },
        "sources": {
            "relative_frame": {
                "run_path": relative_path,
                "manifest_sha256": relative_digest,
            },
            "protocol_semantic_selection": {
                "run_path": semantic_path,
                "manifest_sha256": semantic_digest,
            },
            "radial_near_field": {
                "run_path": radial_path,
                "manifest_sha256": radial_digest,
                "scientific_payload_sha256": radial["payload_digest"],
            },
            "fish_position": _plain(inputs.fish_position_authority),
            "timing": _plain(inputs.timing_authority),
            "arena_geometry_and_scale": _plain(
                radial["sources"]["arena_geometry_and_scale"]
            ),
        },
        "position_provider": _plain(radial["position_provider"]),
        "policies": {
            "visit": VISIT_POLICY_ID,
            "temporal_integration": TIMING_POLICY_ID,
            "epoch_membership": "acquisition_frame_id_in_exact_half_open_interval_v1",
            "visit_membership": (
                "first_strict_inside_sample_through_last_hysteresis_inside_sample_"
                "inclusive_exit_sample_separate_v1"
            ),
            "gap": "invalid_or_nonadjacent_rows_close_and_censor_no_interpolation_v1",
            "short_visit": "retain_all_visits_and_mark_quality_v1",
            "reference": "real_chaser_per_frame_exact_position_v1",
            "key": "deterministic_source_epoch_reference_ordinal_and_bounds_sha256_v1",
        },
        "config": {
            "near_zone_radius_mm": near_zone_mm,
            "near_entry_radius_mm": enter_mm,
            "near_exit_radius_mm": exit_mm,
            "minimum_quality_sample_count": inputs.minimum_quality_sample_count,
        },
        "arena": {
            **_plain(radial["arena"]),
            "mm_per_pixel": float(inputs.mm_per_pixel),
        },
        "identity_registries": {
            "epoch_role": epoch_registry,
            "behavior_role": roles,
            "chaser": chasers,
            "reference_kind": {
                str(key): value for key, value in REFERENCE_KIND_REGISTRY.items()
            },
            "left_censor_reason": {
                str(key): value for key, value in LEFT_CENSOR_REASONS.items()
            },
            "right_censor_reason": {
                str(key): value for key, value in RIGHT_CENSOR_REASONS.items()
            },
            "quality": {str(key): value for key, value in QUALITY_REGISTRY.items()},
            "canonical_reason": {
                str(key): value for key, value in CANONICAL_REASON_REGISTRY.items()
            },
        },
        "epoch_records": expected_epoch_records,
        "table_semantics": {
            "visits": "one row per epoch_window_x_real_chaser_x_visit_ordinal",
            "samples": "flat ragged exact valid visit-member rows",
            "summaries": "one row per epoch_window_x_real_chaser",
            "visit_key_encoding": KEY_ENCODING,
            "sentinel": "unobserved_entry_or_exit_integer_identity_is_minus_one",
            "complete": "observed_entry_and_observed_exit_only",
            "censoring": "left_and_right_censor_axes_are_retained_separately",
        },
        "coordinate_conventions": {
            "source": "continuous_pixel_xy_top_left_y_down_scaled_to_mm",
            "relative": "fish_minus_real_chaser_in_source_xy_axes",
            "canonical": (
                "per_sample_real_chaser_at_origin_arena_center_on_positive_x_"
                "source_handedness_retained"
            ),
            "canonical_invalid": "reference_at_reviewed_arena_center",
        },
        "radial_aggregate_parity": {
            "status": "exact",
            "fields": [
                *RADIAL_PARITY_INT_FIELDS,
                *RADIAL_PARITY_FLOAT_FIELDS,
            ],
            "summary_row_count": len(summary_rows),
            "comparison": "exact_scalar_equality_nan_equal_v1",
        },
        "array_declarations": _array_declarations(readonly),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze({**body, "payload_digest": canonical_json_sha256(body)})
    return PreparedChaserNearFieldVisits(
        recording_id=recording_id,
        n_visits=len(visit_rows),
        n_samples=len(sample_rows),
        n_summary_rows=len(summary_rows),
        arrays=readonly,
        manifest=manifest,
    )


def chaser_near_field_visit_input_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    radial_near_field: Any,
    *,
    minimum_quality_sample_count: int = MIN_VISIT_SAMPLE_COUNT,
) -> ChaserNearFieldVisitInput:
    """Bind strict current handles without selector or provider discovery."""

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
        ChaserRelativeFrameTargetedSourceHandle,
    )
    from fisheye.analysis_workflows.composable_chaser_successor_publication import (
        ComposableChaserSuccessorSourceHandle,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    relative_types = (
        ChaserRelativeFrameSourceHandle,
        ChaserRelativeFrameTargetedSourceHandle,
    )
    if type(relative_frame) not in relative_types:
        raise TypeError(
            "relative_frame must be one strict full or receipt-targeted handle."
        )
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be one strict loader-minted handle.")
    if type(radial_near_field) is not ComposableChaserSuccessorSourceHandle:
        raise TypeError("radial_near_field must be one strict loader-minted handle.")
    if radial_near_field.successor_kind != "chaser_radial_near_field":
        _fail("radial_near_field handle names another successor kind.")
    core_authority = core_paradigm_dependency_from_relative_frame(relative_frame)
    semantic_selection.assert_current()
    radial_near_field.assert_current()
    radial_near_field.require_verified_arrays(RADIAL_PARITY_ARRAY_NAMES)
    archives = {
        relative_frame.analysis_zarr_path,
        semantic_selection.analysis_zarr,
        radial_near_field.analysis_zarr,
    }
    if len(archives) != 1:
        _fail("Visit dependencies belong to different analysis archives.")
    if (
        len(
            {
                relative_frame.recording_id,
                semantic_selection.recording_id,
                radial_near_field.recording_id,
            }
        )
        != 1
    ):
        _fail("Visit dependencies belong to different recordings.")
    if type(relative_frame) is ChaserRelativeFrameSourceHandle:
        view = load_chaser_relative_distance_view(relative_frame)
        n_frames = view.n_frames
        n_chasers = view.n_chasers
        frame_array = view.frame_array
        pair_array = view.pair_array
    else:
        missing = set(RELATIVE_FRAME_ARRAY_NAMES).difference(relative_frame.base_arrays)
        missing_collapsed = set(RELATIVE_FRAME_COLLAPSED_ARRAY_NAMES).difference(
            relative_frame.frame_arrays
        )
        if missing or missing_collapsed:
            _fail(
                "Receipt-targeted relative handle lacks required visit arrays: "
                + repr(sorted(missing | missing_collapsed))
            )
        n_frames = relative_frame.n_frames
        n_chasers = relative_frame.n_chasers
        frame_array = relative_frame.frame_array
        pair_array = relative_frame.base_frame_chaser
    scale = relative_frame.run_manifest.get("scale_policy")
    if not isinstance(scale, Mapping) or scale.get("unit") != "mm":
        _fail("Relative-frame physical coordinates are not explicitly millimetres.")
    pixels_per_mm = float(scale.get("pixels_per_unit", math.nan))
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0.0:
        _fail("Relative-frame millimetre scale is invalid.")
    radial_manifest = radial_near_field.scientific_manifest
    arena = radial_manifest.get("arena")
    if not isinstance(arena, Mapping):
        _fail("Radial source lacks reviewed arena geometry.")
    registries = radial_manifest.get("identity_registries")
    if not isinstance(registries, Mapping):
        _fail("Radial source lacks exact identity registries.")
    return ChaserNearFieldVisitInput(
        recording_id=relative_frame.recording_id,
        source_relative_frame_run_path=relative_frame.run_path,
        source_relative_frame_manifest_sha256=relative_frame.manifest_sha256,
        source_semantic_selection_run_path=semantic_selection.run_path,
        source_semantic_selection_manifest_sha256=semantic_selection.manifest_sha256,
        source_radial_near_field_run_path=radial_near_field.run_path,
        source_radial_near_field_manifest_sha256=radial_near_field.manifest_sha256,
        radial_near_field_manifest=radial_manifest,
        radial_metric_arrays={
            name: radial_near_field.array(name) for name in RADIAL_PARITY_ARRAY_NAMES
        },
        fish_position_authority=relative_frame.source_authorities["fish_position"],
        timing_authority=relative_frame.run_manifest["timing_policy"],
        n_frames=n_frames,
        n_chasers=n_chasers,
        acquisition_frame_id=frame_array("acquisition_frame_id"),
        track_sample_id=frame_array("track_sample_id"),
        timestamp_ns_session=frame_array("timestamp_ns"),
        timestamp_valid=frame_array("timestamp_valid"),
        fish_source_row_id=frame_array("fish_source_row_id"),
        fish_source_row_valid=frame_array("fish_source_row_valid"),
        fish_xy_px=frame_array("fish_position_xy_px"),
        fish_valid=frame_array("fish_position_valid"),
        chaser_source_row_id=pair_array("chaser_source_row_id"),
        chaser_source_row_valid=pair_array("chaser_source_row_valid"),
        chaser_xy_px=pair_array("chaser_position_xy_px"),
        chaser_valid=pair_array("chaser_position_valid"),
        relative_vector_mm=pair_array("relative_vector_physical_xy"),
        distance_mm=pair_array("relative_distance_physical"),
        distance_valid=pair_array("relative_physical_valid"),
        selection_member=frame_array("selection_member"),
        chaser_occurrence_member=pair_array("chaser_occurrence_member"),
        chaser_role_code=pair_array("chaser_behavior_role_code"),
        chaser_role_valid=pair_array("chaser_behavior_role_valid"),
        chaser_identity_code=pair_array("chaser_identity_code"),
        role_registry=registries["behavior_role"],
        chaser_registry=registries["chaser"],
        epochs=semantic_selection.position_suite_epochs(),
        arena_center_xy_px=np.asarray(
            [arena["center_x_px"], arena["center_y_px"]], dtype=np.float64
        ),
        arena_radius_px=float(arena["radius_px"]),
        mm_per_pixel=1.0 / pixels_per_mm,
        core_authority_dependency=core_authority,
        minimum_quality_sample_count=minimum_quality_sample_count,
    )


def prepare_chaser_near_field_visit_successor_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    radial_near_field: Any,
    **kwargs: Any,
) -> PreparedChaserNearFieldVisits:
    return prepare_chaser_near_field_visit_successor(
        chaser_near_field_visit_input_from_handles(
            relative_frame,
            semantic_selection,
            radial_near_field,
            **kwargs,
        )
    )


__all__ = [
    "CANONICAL_REASON_REGISTRY",
    "ChaserNearFieldVisitInput",
    "ChaserNearFieldVisitSuccessorError",
    "KEY_ENCODING",
    "METHOD_ID",
    "MIN_VISIT_SAMPLE_COUNT",
    "PreparedChaserNearFieldVisits",
    "QUALITY_REGISTRY",
    "RADIAL_PARITY_ARRAY_NAMES",
    "RELATIVE_FRAME_ARRAY_NAMES",
    "RELATIVE_FRAME_COLLAPSED_ARRAY_NAMES",
    "REFERENCE_KIND_REGISTRY",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "chaser_near_field_visit_input_from_handles",
    "prepare_chaser_near_field_visit_successor",
    "prepare_chaser_near_field_visit_successor_from_handles",
]
