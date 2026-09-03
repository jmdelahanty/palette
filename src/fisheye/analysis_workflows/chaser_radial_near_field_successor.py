"""Exact-time, provider-generic fish--chaser distance analytics successor.

The product consumes one immutable chaser-relative-frame run and one immutable
protocol-semantic selection.  The fish position authority is not selected or
ranked here: detection/bounding-box centroids, anatomical keypoint aggregates,
and future providers are peers when their source handle satisfies the same
coordinate, scale, row-axis, and timing contracts.

Spatial metrics reuse the reviewed moving-chaser geometric null.  Durations and
visit rates are recomputed from exact session timestamps; nominal FPS is never
used for temporal metrics.  Session time is acquisition evidence and does not
claim camera-exposure or physical-presentation synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.provider_chaser_position_suite import (
    CircularArena,
    PositionSuiteConfig,
    PositionSuiteEpoch,
    compute_provider_chaser_position_suite,
)
from fisheye.analysis_workflows.chaser_relative_distance_view import (
    load_chaser_relative_distance_view,
)
from fisheye.analysis_workflows.near_field_visit_state_machine import (
    ExactNearFieldVisitError,
    segment_exact_time_near_field_visits,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.analysis.chaser_radial_near_field_successor"
SCHEMA_VERSION = 1
METHOD_ID = "provider_generic_radial_near_field_exact_session_time_v1"
TIMING_POLICY_ID = "adjacent_acquisition_frame_exact_session_interval_v1"
VISIT_POLICY_ID = "exact_session_time_hysteresis_5mm_6mm_gap_censored_v1"


class ChaserRadialNearFieldSuccessorError(ValueError):
    """Raised when a radial/near-field successor input is incomplete."""


def _fail(message: str) -> None:
    raise ChaserRadialNearFieldSuccessorError(message)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{name} must be one exact non-empty string.")
    return value


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
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
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_or_nan(value: object) -> float:
    if value is None:
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


@dataclass(frozen=True, slots=True)
class ChaserRadialNearFieldInput:
    recording_id: str
    source_relative_frame_run_path: str
    source_relative_frame_manifest_sha256: str
    source_semantic_selection_run_path: str
    source_semantic_selection_manifest_sha256: str
    fish_position_authority: Mapping[str, Any]
    timing_authority: Mapping[str, Any]
    arena_geometry_authority: Mapping[str, Any]
    n_frames: int
    n_chasers: int
    acquisition_frame_id: np.ndarray
    timestamp_ns_session: np.ndarray
    timestamp_valid: np.ndarray
    fish_xy_px: np.ndarray
    fish_valid: np.ndarray
    chaser_xy_px: np.ndarray
    chaser_valid: np.ndarray
    distance_px: np.ndarray
    distance_px_valid: np.ndarray
    distance_mm: np.ndarray
    distance_mm_valid: np.ndarray
    selection_member: np.ndarray
    chaser_occurrence_member: np.ndarray
    chaser_role_codes: np.ndarray
    chaser_role_valid: np.ndarray
    chaser_identity_codes: np.ndarray
    role_registry: Mapping[str, str]
    chaser_registry: Mapping[str, str]
    epochs: Sequence[PositionSuiteEpoch]
    arena: CircularArena
    mm_per_pixel: float
    config: PositionSuiteConfig = PositionSuiteConfig()


@dataclass(frozen=True, slots=True)
class PreparedChaserRadialNearField:
    recording_id: str
    n_epoch_chaser_rows: int
    n_radial_rows: int
    n_cdf_rows: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown radial/near-field array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


@dataclass(frozen=True, slots=True)
class _ExactVisitResult:
    near_dwell_s: float
    valid_tracked_duration_s: float
    entry_count: int
    entry_rate_per_min: float
    complete_visit_median_dwell_s: float
    complete_visit_total_dwell_s: float
    invalid_gap_count: int
    censor_event_count: int
    boundary_censor_event_count: int
    invalid_gap_censor_event_count: int


def _exact_time_visits(
    *,
    frame_id: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_valid: np.ndarray,
    distance_mm: np.ndarray,
    distance_valid: np.ndarray,
    near_zone_mm: float,
    enter_mm: float,
    exit_mm: float,
) -> _ExactVisitResult:
    """Integrate tracked intervals and hysteretic visits without bridging gaps."""
    try:
        segmented = segment_exact_time_near_field_visits(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            timestamp_valid=timestamp_valid,
            distance_mm=distance_mm,
            distance_valid=distance_valid,
            near_zone_mm=near_zone_mm,
            enter_mm=enter_mm,
            exit_mm=exit_mm,
        )
    except ExactNearFieldVisitError as exc:
        _fail(str(exc))
    return _ExactVisitResult(
        near_dwell_s=segmented.near_dwell_s,
        valid_tracked_duration_s=segmented.valid_tracked_duration_s,
        entry_count=segmented.entry_count,
        entry_rate_per_min=segmented.entry_rate_per_min,
        complete_visit_median_dwell_s=(segmented.complete_visit_median_dwell_s),
        complete_visit_total_dwell_s=(segmented.complete_visit_total_dwell_s),
        invalid_gap_count=segmented.invalid_gap_count,
        censor_event_count=segmented.censor_event_count,
        boundary_censor_event_count=segmented.boundary_censor_event_count,
        invalid_gap_censor_event_count=(segmented.invalid_gap_censor_event_count),
    )


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


def _code_registry(values: Sequence[str]) -> tuple[dict[str, int], dict[str, str]]:
    ordered = tuple(dict.fromkeys(str(value) for value in values))
    return (
        {value: index for index, value in enumerate(ordered)},
        {str(index): value for index, value in enumerate(ordered)},
    )


def _table_arrays(
    *,
    metrics: Sequence[Mapping[str, Any]],
    radial: Sequence[Mapping[str, Any]],
    cdf: Sequence[Mapping[str, Any]],
    role_code: Mapping[str, int],
    behavior_code: Mapping[str, int],
    identity_code: Mapping[str, int],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    metric_int = (
        "epoch_window_id", "epoch_start_frame", "epoch_end_frame_exclusive",
        "chaser_column", "chaser_identity_code", "behavior_role_code",
        "candidate_frame_count", "valid_distance_frame_count",
        "same_quadrant_valid_frame_count", "near_zone_frame_count",
        "near_zone_entry_count", "near_zone_invalid_gap_count",
        "near_zone_censor_event_count", "near_zone_boundary_censor_event_count",
        "near_zone_invalid_gap_censor_event_count", "wall_excluded_valid_frame_count",
    )
    metric_float = (
        "valid_distance_fraction", "distance_mean_mm", "distance_p05_mm",
        "distance_p25_mm", "distance_p50_mm", "distance_p75_mm", "distance_p95_mm",
        "same_quadrant_fraction_valid", "same_quadrant_fraction_candidate",
        "near_zone_fraction_valid", "near_zone_fraction_candidate", "near_zone_dwell_s",
        "near_zone_expected_fraction_geometric", "near_zone_enrichment_geometric",
        "near_zone_entry_rate_per_min_valid_time", "near_zone_valid_tracked_duration_s",
        "near_zone_complete_visit_median_dwell_s",
        "near_zone_complete_visit_total_dwell_s", "fish_arena_radius_mean_mm",
        "fish_arena_radius_p50_mm", "fish_wall_distance_mean_mm",
        "fish_wall_distance_p50_mm",
    )
    arrays["metric_epoch_role_code"] = _readonly(
        [role_code[str(row["analysis_role"])] for row in metrics], dtype=np.uint8
    )
    for name in metric_int:
        arrays[f"metric_{name}"] = _readonly([int(row[name]) for row in metrics], dtype=np.int64)
    for name in metric_float:
        arrays[f"metric_{name}"] = _readonly(
            [_finite_or_nan(row[name]) for row in metrics], dtype=np.float64
        )

    arrays["radial_epoch_role_code"] = _readonly(
        [role_code[str(row["analysis_role"])] for row in radial], dtype=np.uint8
    )
    arrays["radial_behavior_role_code"] = _readonly(
        [behavior_code[str(row["behavior_role"])] for row in radial], dtype=np.uint8
    )
    arrays["radial_chaser_identity_code"] = _readonly(
        [identity_code[str(row["chaser_identity"])] for row in radial], dtype=np.uint16
    )
    arrays["radial_epoch_window_id"] = _readonly(
        [int(row["epoch_window_id"]) for row in radial], dtype=np.int64
    )
    for name in ("observed_count", "wall_excluded_observed_count"):
        arrays[f"radial_{name}"] = _readonly([int(row[name]) for row in radial], dtype=np.int64)
    for name in (
        "bin_start_mm", "bin_end_mm", "observed_fraction",
        "expected_available_area_mm2_frames", "expected_fraction_geometric",
        "selection_index_geometric", "wall_excluded_observed_fraction",
        "wall_excluded_expected_available_area_mm2_frames",
        "wall_excluded_expected_fraction_geometric",
        "wall_excluded_selection_index_geometric",
    ):
        arrays[f"radial_{name}"] = _readonly(
            [_finite_or_nan(row[name]) for row in radial], dtype=np.float64
        )

    arrays["cdf_epoch_role_code"] = _readonly(
        [role_code[str(row["analysis_role"])] for row in cdf], dtype=np.uint8
    )
    arrays["cdf_behavior_role_code"] = _readonly(
        [behavior_code[str(row["behavior_role"])] for row in cdf], dtype=np.uint8
    )
    arrays["cdf_chaser_identity_code"] = _readonly(
        [identity_code[str(row["chaser_identity"])] for row in cdf], dtype=np.uint16
    )
    arrays["cdf_epoch_window_id"] = _readonly(
        [int(row["epoch_window_id"]) for row in cdf], dtype=np.int64
    )
    arrays["cdf_threshold_mm"] = _readonly(
        [float(row["threshold_mm"]) for row in cdf], dtype=np.float64
    )
    arrays["cdf_fraction_at_or_below"] = _readonly(
        [_finite_or_nan(row["fraction_at_or_below"]) for row in cdf], dtype=np.float64
    )
    return arrays


def prepare_chaser_radial_near_field_successor(
    inputs: ChaserRadialNearFieldInput,
) -> PreparedChaserRadialNearField:
    """Compute compact spatial summaries and exact-session-time near visits."""

    if type(inputs) is not ChaserRadialNearFieldInput:
        raise TypeError("inputs must be one ChaserRadialNearFieldInput.")
    recording_id = _text(inputs.recording_id, name="recording_id")
    _text(inputs.source_relative_frame_run_path, name="source relative-frame path")
    _digest(inputs.source_relative_frame_manifest_sha256, name="relative-frame manifest")
    _text(inputs.source_semantic_selection_run_path, name="semantic selection path")
    _digest(inputs.source_semantic_selection_manifest_sha256, name="semantic manifest")
    fish_authority = _plain(inputs.fish_position_authority)
    provider_id = _text(fish_authority.get("provider_id"), name="fish position provider_id")
    provider_digest = _digest(
        fish_authority.get("provider_digest"), name="fish position provider digest"
    )
    timing = _plain(inputs.timing_authority)
    if timing.get("timestamp_field") != "timestamp_ns_session":
        _fail("Temporal summaries require exact timestamp_ns_session evidence.")
    _text(timing.get("timing_authority_id"), name="timing authority ID")
    _digest(timing.get("timing_digest"), name="timing authority digest")
    geometry = _plain(inputs.arena_geometry_authority)
    _digest(geometry.get("selection_record_sha256"), name="arena selection digest")
    _digest(geometry.get("physical_authority_sha256"), name="physical authority digest")
    if fish_authority.get("coordinate_authority_id") != geometry.get(
        "pixel_frame_record_ref"
    ):
        _fail("Fish positions and reviewed arena use different coordinate authorities.")
    if type(inputs.n_frames) is not int or inputs.n_frames <= 1:
        _fail("At least two exact relative frames are required.")
    if type(inputs.n_chasers) is not int or inputs.n_chasers <= 0:
        _fail("At least one chaser column is required.")

    frame = np.asarray(inputs.acquisition_frame_id)
    timestamp = np.asarray(inputs.timestamp_ns_session)
    timestamp_valid = np.asarray(inputs.timestamp_valid)
    if (
        frame.dtype != np.dtype(np.int64)
        or timestamp.dtype != np.dtype(np.int64)
        or timestamp_valid.dtype != np.dtype(bool)
        or frame.shape != (inputs.n_frames,)
        or timestamp.shape != frame.shape
        or timestamp_valid.shape != frame.shape
    ):
        _fail("Frame/timestamp axes must be exact int64/int64/bool vectors.")
    if np.any(np.diff(frame) <= 0):
        _fail("Acquisition frame IDs must be strictly increasing.")
    valid_timestamps = timestamp[timestamp_valid]
    if valid_timestamps.size < 2 or np.any(np.diff(valid_timestamps) <= 0):
        _fail("Valid exact session timestamps must be strictly increasing.")

    suite = compute_provider_chaser_position_suite(
        frame_ids=frame,
        fish_xy_px=inputs.fish_xy_px,
        fish_valid=inputs.fish_valid,
        chaser_xy_px=inputs.chaser_xy_px,
        chaser_valid=inputs.chaser_valid,
        distance_px=inputs.distance_px,
        distance_px_valid=inputs.distance_px_valid,
        distance_mm=inputs.distance_mm,
        distance_mm_valid=inputs.distance_mm_valid,
        selection_member=inputs.selection_member,
        chaser_occurrence_member=inputs.chaser_occurrence_member,
        chaser_role_codes=inputs.chaser_role_codes,
        chaser_role_valid=inputs.chaser_role_valid,
        chaser_identity_codes=inputs.chaser_identity_codes,
        role_registry=inputs.role_registry,
        chaser_registry=inputs.chaser_registry,
        epochs=inputs.epochs,
        arena=inputs.arena,
        mm_per_pixel=inputs.mm_per_pixel,
        fps=1.0,  # spatial engine placeholder; all temporal fields are replaced below
        config=inputs.config,
    )
    selected = np.asarray(inputs.selection_member, dtype=bool)
    occurrence = np.asarray(inputs.chaser_occurrence_member, dtype=bool)
    role_ok = np.asarray(inputs.chaser_role_valid, dtype=bool)
    distance_ok = np.asarray(inputs.distance_mm_valid, dtype=bool)
    distance_mm = np.asarray(inputs.distance_mm, dtype=np.float64)
    by_epoch = {epoch.analysis_role: epoch for epoch in inputs.epochs}
    for record in suite["per_epoch_chaser_metrics"]:
        epoch = by_epoch[str(record["analysis_role"])]
        column = int(record["chaser_column"])
        epoch_mask = (frame >= epoch.start_frame) & (frame < epoch.end_frame)
        candidate = epoch_mask & selected & occurrence[:, column] & role_ok[:, column]
        exact = _exact_time_visits(
            frame_id=frame[epoch_mask],
            timestamp_ns=timestamp[epoch_mask],
            timestamp_valid=timestamp_valid[epoch_mask],
            distance_mm=distance_mm[epoch_mask, column],
            distance_valid=(candidate & distance_ok[:, column])[epoch_mask],
            near_zone_mm=inputs.config.near_zone_radius_mm,
            enter_mm=inputs.config.near_entry_radius_mm,
            exit_mm=inputs.config.near_exit_radius_mm,
        )
        record.update(
            {
                "near_zone_dwell_s": exact.near_dwell_s,
                "near_zone_entry_count": exact.entry_count,
                "near_zone_entry_rate_per_min_valid_time": exact.entry_rate_per_min,
                "near_zone_valid_tracked_duration_s": exact.valid_tracked_duration_s,
                "near_zone_complete_visit_median_dwell_s": exact.complete_visit_median_dwell_s,
                "near_zone_complete_visit_total_dwell_s": exact.complete_visit_total_dwell_s,
                "near_zone_invalid_gap_count": exact.invalid_gap_count,
                "near_zone_censor_event_count": exact.censor_event_count,
                "near_zone_boundary_censor_event_count": exact.boundary_censor_event_count,
                "near_zone_invalid_gap_censor_event_count": exact.invalid_gap_censor_event_count,
            }
        )

    metrics_by_role = {
        (str(row["analysis_role"]), str(row["behavior_role"])): row
        for row in suite["per_epoch_chaser_metrics"]
    }
    for contrast in suite["role_contrasts"]:
        if contrast["metric"] not in {
            "near_zone_dwell_s",
            "near_zone_entry_rate_per_min_valid_time",
        }:
            continue
        left = metrics_by_role[(str(contrast["analysis_role"]), inputs.config.treatment_role)][contrast["metric"]]
        right = metrics_by_role[(str(contrast["analysis_role"]), inputs.config.baseline_role)][contrast["metric"]]
        contrast["treatment_value"] = left
        contrast["baseline_value"] = right
        contrast["treatment_minus_baseline"] = (
            None
            if left is None or right is None or not math.isfinite(float(left)) or not math.isfinite(float(right))
            else float(left) - float(right)
        )

    suite["schema_id"] = SCHEMA_ID
    suite["schema_version"] = SCHEMA_VERSION
    suite["method_id"] = METHOD_ID
    suite.pop("fps_hz", None)
    suite["policies"]["visit"] = VISIT_POLICY_ID
    suite["policies"]["temporal_integration"] = TIMING_POLICY_ID
    suite["temporal_metric_timebase"] = {
        "timestamp_field": "timestamp_ns_session",
        "integration": "left_hold_over_adjacent_valid_acquisition_frame_intervals",
        "gap_policy": "invalid_or_nonadjacent_intervals_excluded_and_active_visits_censored",
        "visit_duration": "exact_entry_timestamp_to_exact_exit_timestamp",
        "rate_denominator": "sum_of_exact_contiguous_valid_tracked_intervals",
        "physical_presentation_verified": False,
    }
    suite["interpretation_caveats"].append(
        "Exact seconds use Citrus session-monotonic acquisition timestamps; they do not establish camera-exposure or physical-presentation latency."
    )

    metrics = suite["per_epoch_chaser_metrics"]
    radial = suite["radial_occupancy"]
    cdf = suite["distance_cdf"]
    epoch_codes, epoch_registry = _code_registry(
        [epoch.analysis_role for epoch in inputs.epochs]
    )
    behavior_codes = {str(value): int(key) for key, value in inputs.role_registry.items()}
    identity_codes = {str(value): int(key) for key, value in inputs.chaser_registry.items()}
    arrays = _table_arrays(
        metrics=metrics,
        radial=radial,
        cdf=cdf,
        role_code=epoch_codes,
        behavior_code=behavior_codes,
        identity_code=identity_codes,
    )
    readonly = MappingProxyType({name: _readonly(value) for name, value in arrays.items()})
    body = {
        "scientific_schema": {"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION},
        "method_id": METHOD_ID,
        "recording_id": recording_id,
        "dimensions": {
            "n_frames": inputs.n_frames,
            "n_chasers": inputs.n_chasers,
            "n_epoch_chaser_rows": len(metrics),
            "n_radial_rows": len(radial),
            "n_cdf_rows": len(cdf),
        },
        "sources": {
            "relative_frame": {
                "run_path": inputs.source_relative_frame_run_path,
                "manifest_sha256": inputs.source_relative_frame_manifest_sha256,
            },
            "protocol_semantic_selection": {
                "run_path": inputs.source_semantic_selection_run_path,
                "manifest_sha256": inputs.source_semantic_selection_manifest_sha256,
            },
            "fish_position": fish_authority,
            "timing": timing,
            "arena_geometry_and_scale": geometry,
        },
        "position_provider": {
            "provider_id": provider_id,
            "provider_digest": provider_digest,
            "status": "first_class_explicit_authority",
            "provider_selection": "none",
        },
        "policies": _plain(suite["policies"]),
        "config": _plain(suite["config"]),
        "arena": _plain(suite["arena"]),
        "temporal_metric_timebase": _plain(suite["temporal_metric_timebase"]),
        "identity_registries": {
            "epoch_role": epoch_registry,
            "behavior_role": {str(key): str(value) for key, value in inputs.role_registry.items()},
            "chaser": {str(key): str(value) for key, value in inputs.chaser_registry.items()},
        },
        "epoch_records": _plain(suite["epoch_roles"]),
        "source_distance_surface": {
            "per_frame_distance_array": "relative_frame:base/relative_distance_physical",
            "per_frame_distance_validity_array": "relative_frame:base/relative_physical_valid",
            "unit": "mm",
            "copy_policy": "referenced_by_exact_immutable_digest_not_duplicated",
            "summary_fields": [
                "distance_mean_mm", "distance_p05_mm", "distance_p25_mm",
                "distance_p50_mm", "distance_p75_mm", "distance_p95_mm",
            ],
        },
        "interpretation_caveats": _plain(suite["interpretation_caveats"]),
        "array_declarations": _array_declarations(readonly),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze({**body, "payload_digest": canonical_json_sha256(body)})
    return PreparedChaserRadialNearField(
        recording_id=recording_id,
        n_epoch_chaser_rows=len(metrics),
        n_radial_rows=len(radial),
        n_cdf_rows=len(cdf),
        arrays=readonly,
        manifest=manifest,
    )


def chaser_radial_near_field_input_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    *,
    arena: CircularArena,
    mm_per_pixel: float,
    arena_geometry_authority: Mapping[str, Any],
    config: PositionSuiteConfig = PositionSuiteConfig(),
) -> ChaserRadialNearFieldInput:
    """Bind strict current handles without selecting or special-casing a provider."""

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    if type(relative_frame) is not ChaserRelativeFrameSourceHandle:
        raise TypeError("relative_frame must be one strict loader-minted handle.")
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be one strict loader-minted handle.")
    relative_frame.assert_current()
    semantic_selection.assert_current()
    if relative_frame.analysis_zarr_path != semantic_selection.analysis_zarr:
        _fail("Relative-frame and semantic sources belong to different archives.")
    if relative_frame.recording_id != semantic_selection.recording_id:
        _fail("Relative-frame and semantic sources belong to different recordings.")
    view = load_chaser_relative_distance_view(relative_frame)
    scale = relative_frame.run_manifest.get("scale_policy")
    if not isinstance(scale, Mapping) or scale.get("unit") != "mm":
        _fail("Relative-frame physical distances are not explicitly millimetres.")
    pixels_per_mm = float(scale.get("pixels_per_unit", math.nan))
    if not math.isfinite(pixels_per_mm) or not math.isclose(
        pixels_per_mm * float(mm_per_pixel), 1.0, rel_tol=1e-6, abs_tol=1e-9
    ):
        _fail("Reviewed physical scale differs from the relative-frame scale.")
    timing = relative_frame.run_manifest.get("timing_policy")
    if not isinstance(timing, Mapping):
        _fail("Relative-frame timing authority is absent.")
    return ChaserRadialNearFieldInput(
        recording_id=relative_frame.recording_id,
        source_relative_frame_run_path=relative_frame.run_path,
        source_relative_frame_manifest_sha256=relative_frame.manifest_sha256,
        source_semantic_selection_run_path=semantic_selection.run_path,
        source_semantic_selection_manifest_sha256=semantic_selection.manifest_sha256,
        fish_position_authority=relative_frame.source_authorities["fish_position"],
        timing_authority=timing,
        arena_geometry_authority=arena_geometry_authority,
        n_frames=view.n_frames,
        n_chasers=view.n_chasers,
        acquisition_frame_id=view.frame_array("acquisition_frame_id"),
        timestamp_ns_session=view.frame_array("timestamp_ns"),
        timestamp_valid=view.frame_array("timestamp_valid"),
        fish_xy_px=view.frame_array("fish_position_xy_px"),
        fish_valid=view.frame_array("fish_position_valid"),
        chaser_xy_px=view.pair_array("chaser_position_xy_px"),
        chaser_valid=view.pair_array("chaser_position_valid"),
        distance_px=view.pair_array("relative_distance_px"),
        distance_px_valid=view.pair_array("relative_px_valid"),
        distance_mm=view.pair_array("relative_distance_physical"),
        distance_mm_valid=view.pair_array("relative_physical_valid"),
        selection_member=view.frame_array("selection_member"),
        chaser_occurrence_member=view.pair_array("chaser_occurrence_member"),
        chaser_role_codes=view.pair_array("chaser_behavior_role_code"),
        chaser_role_valid=view.pair_array("chaser_behavior_role_valid"),
        chaser_identity_codes=view.pair_array("chaser_identity_code"),
        role_registry=view.registries.behavior_role,
        chaser_registry=view.registries.chaser_identity,
        epochs=semantic_selection.position_suite_epochs(),
        arena=arena,
        mm_per_pixel=float(mm_per_pixel),
        config=config,
    )


def prepare_chaser_radial_near_field_successor_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    **kwargs: Any,
) -> PreparedChaserRadialNearField:
    return prepare_chaser_radial_near_field_successor(
        chaser_radial_near_field_input_from_handles(
            relative_frame, semantic_selection, **kwargs
        )
    )


__all__ = [
    "METHOD_ID",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "TIMING_POLICY_ID",
    "VISIT_POLICY_ID",
    "ChaserRadialNearFieldInput",
    "ChaserRadialNearFieldSuccessorError",
    "PreparedChaserRadialNearField",
    "chaser_radial_near_field_input_from_handles",
    "prepare_chaser_radial_near_field_successor",
    "prepare_chaser_radial_near_field_successor_from_handles",
]
