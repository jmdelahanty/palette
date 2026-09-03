"""Validated read model for exact persisted near-field visit trajectories.

This module is deliberately a view over ``chaser_near_field_visits``.  It does
not segment distances, bridge gaps, drop short visits, or infer chaser roles.
The same read model can therefore back static plots and the Marimo explorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.chaser_near_field_visit_successor import (
    METHOD_ID,
    QUALITY_OK,
    QUALITY_SHORT,
    REFERENCE_KIND_REAL_CHASER,
    SCHEMA_ID,
    SCHEMA_VERSION,
)
from fisheye.analysis_workflows.chaser_radial_near_field_successor import (
    TIMING_POLICY_ID,
    VISIT_POLICY_ID,
)
from fisheye.analysis_workflows.near_field_visit_state_machine import (
    LEFT_CENSOR_NONE,
    RIGHT_CENSOR_NONE,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


VISIT_TRAJECTORY_ARRAY_NAMES = (
    "visit_row_id",
    "visit_key_sha256_bytes",
    "visit_ordinal",
    "visit_epoch_role_code",
    "visit_epoch_window_id",
    "visit_reference_kind_code",
    "visit_chaser_identity_code",
    "visit_behavior_role_code",
    "visit_sample_offset",
    "visit_sample_count",
    "visit_entry_observed",
    "visit_exit_observed",
    "visit_complete",
    "visit_left_censor_reason_code",
    "visit_right_censor_reason_code",
    "visit_quality_code",
    "visit_min_distance_mm",
    "visit_cpa_sample_ordinal",
    "visit_observed_active_duration_s",
    "visit_complete_dwell_s",
    "sample_visit_row_id",
    "sample_ordinal_within_visit",
    "sample_acquisition_frame_id",
    "sample_timestamp_ns_session",
    "sample_canonical_x_mm",
    "sample_canonical_y_mm",
    "sample_canonical_valid",
    "sample_distance_mm",
    "sample_near_zone_member",
    "sample_time_from_first_sample_s",
    "sample_time_from_observed_entry_s",
    "summary_epoch_role_code",
    "summary_epoch_window_id",
    "summary_chaser_identity_code",
    "summary_behavior_role_code",
    "summary_total_visit_count",
    "summary_complete_visit_count",
    "summary_censored_visit_count",
    "summary_short_visit_count",
)

VISIT_TRAJECTORY_ARRAY_DTYPES = {
    "visit_row_id": np.int64,
    "visit_key_sha256_bytes": np.uint8,
    "visit_ordinal": np.int32,
    "visit_epoch_role_code": np.uint8,
    "visit_epoch_window_id": np.int64,
    "visit_reference_kind_code": np.uint8,
    "visit_chaser_identity_code": np.uint16,
    "visit_behavior_role_code": np.uint8,
    "visit_sample_offset": np.int64,
    "visit_sample_count": np.int64,
    "visit_entry_observed": bool,
    "visit_exit_observed": bool,
    "visit_complete": bool,
    "visit_left_censor_reason_code": np.uint8,
    "visit_right_censor_reason_code": np.uint8,
    "visit_quality_code": np.uint8,
    "visit_min_distance_mm": np.float64,
    "visit_cpa_sample_ordinal": np.int32,
    "visit_observed_active_duration_s": np.float64,
    "visit_complete_dwell_s": np.float64,
    "sample_visit_row_id": np.int64,
    "sample_ordinal_within_visit": np.int32,
    "sample_acquisition_frame_id": np.int64,
    "sample_timestamp_ns_session": np.int64,
    "sample_canonical_x_mm": np.float64,
    "sample_canonical_y_mm": np.float64,
    "sample_canonical_valid": bool,
    "sample_distance_mm": np.float64,
    "sample_near_zone_member": bool,
    "sample_time_from_first_sample_s": np.float64,
    "sample_time_from_observed_entry_s": np.float64,
    "summary_epoch_role_code": np.uint8,
    "summary_epoch_window_id": np.int64,
    "summary_chaser_identity_code": np.uint16,
    "summary_behavior_role_code": np.uint8,
    "summary_total_visit_count": np.int64,
    "summary_complete_visit_count": np.int64,
    "summary_censored_visit_count": np.int64,
    "summary_short_visit_count": np.int64,
}


class ChaserNearFieldVisitViewError(ValueError):
    """Raised when a persisted visit publication cannot support this view."""


def _fail(message: str) -> None:
    raise ChaserNearFieldVisitViewError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"Near-field visit {field} is not one SHA-256 digest.")
    return value


def _array(handle: Any, name: str) -> np.ndarray:
    value = np.asarray(handle.array(name))
    if value.dtype.hasobject:
        _fail(f"Near-field visit array {name!r} has an object dtype.")
    if value.dtype != np.dtype(VISIT_TRAJECTORY_ARRAY_DTYPES[name]):
        _fail(f"Near-field visit array {name!r} has an unsupported dtype.")
    return value


def _registry(scientific: Mapping[str, Any], name: str) -> dict[int, str]:
    registries = scientific.get("identity_registries")
    if not isinstance(registries, Mapping):
        _fail("Near-field visit manifest lacks identity registries.")
    value = registries.get(name)
    if not isinstance(value, Mapping) or not value:
        _fail(f"Near-field visit manifest lacks registry {name!r}.")
    result: dict[int, str] = {}
    for raw_code, raw_label in value.items():
        try:
            code = int(raw_code)
        except (TypeError, ValueError) as exc:
            raise ChaserNearFieldVisitViewError(
                f"Registry {name!r} contains a non-integer code."
            ) from exc
        if str(code) != str(raw_code) or type(raw_label) is not str or not raw_label:
            _fail(f"Registry {name!r} contains a non-canonical entry.")
        result[code] = raw_label
    return result


def _vector(handle: Any, name: str, length: int) -> np.ndarray:
    value = _array(handle, name)
    if value.shape != (length,):
        _fail(f"Near-field visit array {name!r} has the wrong row count.")
    return value


@dataclass(frozen=True, slots=True)
class NearFieldVisitPanel:
    epoch_role_code: int
    epoch_role: str
    epoch_window_id: int
    chaser_identity_code: int
    chaser_identity: str
    behavior_role_code: int
    behavior_role: str
    total_visit_count: int
    complete_visit_count: int
    censored_visit_count: int
    short_visit_count: int


@dataclass(frozen=True, slots=True)
class NearFieldVisitTrajectory:
    visit_row_id: int
    visit_key_sha256: str
    visit_ordinal: int
    epoch_role_code: int
    epoch_window_id: int
    chaser_identity_code: int
    behavior_role_code: int
    entry_observed: bool
    exit_observed: bool
    complete: bool
    left_censor_reason: str
    right_censor_reason: str
    quality: str
    observed_active_duration_s: float
    complete_dwell_s: float
    min_distance_mm: float
    cpa_sample_ordinal: int
    acquisition_frame_id: np.ndarray
    timestamp_ns_session: np.ndarray
    canonical_x_mm: np.ndarray
    canonical_y_mm: np.ndarray
    canonical_valid: np.ndarray
    distance_mm: np.ndarray
    near_zone_member: np.ndarray
    time_from_first_sample_s: np.ndarray
    time_from_observed_entry_s: np.ndarray


@dataclass(frozen=True, slots=True)
class NearFieldVisitTrajectoryView:
    recording_id: str
    source_run_path: str
    source_manifest_sha256: str
    source_scientific_payload_sha256: str
    position_provider_id: str
    position_provider_digest: str
    verification_mode: str
    validation_receipt_sha256: str | None
    near_zone_radius_mm: float
    near_entry_radius_mm: float
    near_exit_radius_mm: float
    coordinate_convention: str
    panels: tuple[NearFieldVisitPanel, ...]
    visits: tuple[NearFieldVisitTrajectory, ...]


def _readonly_slice(value: np.ndarray, start: int, stop: int) -> np.ndarray:
    result = np.array(value[start:stop], copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_manifest(
    handle: Any,
) -> tuple[Mapping[str, Any], dict[str, dict[int, str]]]:
    if getattr(handle, "successor_kind", None) != "chaser_near_field_visits":
        _fail("Visit trajectory source is not a near-field visit successor.")
    handle.require_verified_arrays(VISIT_TRAJECTORY_ARRAY_NAMES)
    scientific = handle.scientific_manifest
    if not isinstance(scientific, Mapping):
        _fail("Near-field visit scientific manifest is absent.")
    body = _plain(scientific)
    payload_digest = body.pop("payload_digest", None)
    if _digest(payload_digest, field="scientific payload") != canonical_json_sha256(
        body
    ):
        _fail("Near-field visit scientific payload digest is stale.")
    if getattr(handle, "scientific_payload_sha256", None) != payload_digest:
        _fail("Near-field visit outer and scientific payload digests disagree.")
    if scientific.get("scientific_schema") != {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
    }:
        _fail("Near-field visit scientific schema is unsupported.")
    if scientific.get("method_id") != METHOD_ID:
        _fail("Near-field visit method is unsupported.")
    if scientific.get("recording_id") != handle.recording_id:
        _fail("Near-field visit manifest belongs to another recording.")
    if (
        scientific.get("selector_eligible") is not False
        or scientific.get("selection") != "none"
        or scientific.get("production_authority") is not False
        or scientific.get("registry_update") is not False
    ):
        _fail("Near-field visit scientific safety flags are invalid.")
    sources = scientific.get("sources")
    expected_sources = {
        "relative_frame",
        "protocol_semantic_selection",
        "radial_near_field",
        "fish_position",
        "timing",
        "arena_geometry_and_scale",
    }
    if not isinstance(sources, Mapping) or set(sources) != expected_sources:
        _fail("Near-field visit source roster is incompatible.")
    for name in ("relative_frame", "protocol_semantic_selection"):
        binding = sources.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"run_path", "manifest_sha256"}
            or type(binding.get("run_path")) is not str
            or not binding["run_path"]
        ):
            _fail(f"Near-field visit {name} binding is invalid.")
        _digest(binding.get("manifest_sha256"), field=f"{name} manifest")
    radial = sources.get("radial_near_field")
    if (
        not isinstance(radial, Mapping)
        or set(radial) != {"run_path", "manifest_sha256", "scientific_payload_sha256"}
        or type(radial.get("run_path")) is not str
        or not radial["run_path"]
    ):
        _fail("Near-field visit radial source binding is invalid.")
    _digest(radial.get("manifest_sha256"), field="radial source manifest")
    _digest(
        radial.get("scientific_payload_sha256"),
        field="radial scientific payload",
    )
    fish_position = sources.get("fish_position")
    position_provider = scientific.get("position_provider")
    if (
        not isinstance(fish_position, Mapping)
        or type(fish_position.get("provider_id")) is not str
        or not fish_position["provider_id"]
        or not isinstance(position_provider, Mapping)
        or position_provider.get("provider_id") != fish_position["provider_id"]
        or position_provider.get("status") != "first_class_explicit_authority"
        or position_provider.get("provider_selection") != "none"
    ):
        _fail("Near-field visit position-provider authority is invalid.")
    provider_digest = _digest(
        fish_position.get("provider_digest"), field="fish-position provider"
    )
    if position_provider.get("provider_digest") != provider_digest:
        _fail("Near-field visit position-provider digests disagree.")
    coordinate = scientific.get("coordinate_conventions")
    if not isinstance(coordinate, Mapping) or coordinate.get("canonical") != (
        "per_sample_real_chaser_at_origin_arena_center_on_positive_x_"
        "source_handedness_retained"
    ):
        _fail("Near-field visit canonical coordinate convention is unsupported.")
    policies = scientific.get("policies")
    expected_policies = {
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
    }
    if not isinstance(policies, Mapping) or _plain(policies) != expected_policies:
        _fail("Near-field visit gap policy is unsupported.")
    declarations = scientific.get("array_declarations")
    if not isinstance(declarations, (tuple, list)):
        _fail("Near-field visit array declarations are absent.")
    declaration_by_path = {
        item.get("path"): item for item in declarations if isinstance(item, Mapping)
    }
    if len(declaration_by_path) != len(declarations) or not set(
        VISIT_TRAJECTORY_ARRAY_NAMES
    ).issubset(declaration_by_path):
        _fail("Near-field visit array declaration roster is incompatible.")
    for name in VISIT_TRAJECTORY_ARRAY_NAMES:
        declaration = declaration_by_path[name]
        if (
            set(declaration) != {"path", "dtype", "shape", "content_sha256"}
            or declaration.get("dtype")
            != np.dtype(VISIT_TRAJECTORY_ARRAY_DTYPES[name]).str
            or not isinstance(declaration.get("shape"), (tuple, list))
        ):
            _fail(f"Near-field visit declaration {name!r} is incompatible.")
        _digest(declaration.get("content_sha256"), field=f"{name} content")
    registries = {
        name: _registry(scientific, name)
        for name in (
            "epoch_role",
            "behavior_role",
            "chaser",
            "reference_kind",
            "left_censor_reason",
            "right_censor_reason",
            "quality",
        )
    }
    if registries["reference_kind"].get(REFERENCE_KIND_REAL_CHASER) != "real_chaser":
        _fail("Near-field visit reference-kind registry is unsupported.")
    return scientific, registries


def validated_near_field_visit_trajectory_view(
    handle: Any,
) -> NearFieldVisitTrajectoryView:
    """Build a lossless trajectory view from a verified exact successor handle."""

    scientific, registries = _validate_manifest(handle)
    dimensions = scientific.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "n_frames",
        "n_chasers",
        "n_visits",
        "n_samples",
        "n_summary_rows",
    }:
        _fail("Near-field visit dimensions are absent.")
    if any(type(dimensions.get(name)) is not int for name in dimensions):
        _fail("Near-field visit dimensions are not exact integers.")
    n_frames = dimensions["n_frames"]
    n_chasers = dimensions["n_chasers"]
    n_visits = dimensions["n_visits"]
    n_samples = dimensions["n_samples"]
    n_summary = dimensions["n_summary_rows"]
    if n_frames <= 0 or n_chasers <= 0 or min(n_visits, n_samples, n_summary) < 0:
        _fail("Near-field visit dimensions are invalid.")

    visit = {
        name: _vector(handle, name, n_visits)
        for name in VISIT_TRAJECTORY_ARRAY_NAMES
        if name.startswith("visit_") and name != "visit_key_sha256_bytes"
    }
    keys = _array(handle, "visit_key_sha256_bytes")
    if keys.shape != (n_visits, 32) or keys.dtype != np.dtype(np.uint8):
        _fail("Near-field visit keys do not have exact raw SHA-256 encoding.")
    if n_visits and np.unique(keys, axis=0).shape[0] != n_visits:
        _fail("Near-field visit keys are not unique.")
    sample = {
        name: _vector(handle, name, n_samples)
        for name in VISIT_TRAJECTORY_ARRAY_NAMES
        if name.startswith("sample_")
    }
    summary = {
        name: _vector(handle, name, n_summary)
        for name in VISIT_TRAJECTORY_ARRAY_NAMES
        if name.startswith("summary_")
    }

    row_ids = visit["visit_row_id"].astype(np.int64, copy=False)
    if not np.array_equal(row_ids, np.arange(n_visits, dtype=np.int64)):
        _fail("Near-field visit rows are not one dense canonical axis.")
    if np.any(
        visit["visit_reference_kind_code"].astype(np.int64)
        != REFERENCE_KIND_REAL_CHASER
    ):
        _fail("Visit trajectory view only admits persisted real-chaser references.")

    config = scientific.get("config")
    if not isinstance(config, Mapping):
        _fail("Near-field visit configuration is absent.")
    near_zone = float(config.get("near_zone_radius_mm", math.nan))
    enter = float(config.get("near_entry_radius_mm", math.nan))
    exit_radius = float(config.get("near_exit_radius_mm", math.nan))
    if not all(math.isfinite(value) and value >= 0.0 for value in (near_zone, enter)):
        _fail("Near-field visit radii are invalid.")
    if not math.isfinite(exit_radius) or exit_radius <= enter:
        _fail("Near-field visit exit radius must exceed its entry radius.")
    minimum_quality_count = config.get("minimum_quality_sample_count")
    if type(minimum_quality_count) is not int or minimum_quality_count <= 0:
        _fail("Near-field visit quality threshold is invalid.")

    offsets = visit["visit_sample_offset"].astype(np.int64, copy=False)
    counts = visit["visit_sample_count"].astype(np.int64, copy=False)
    sample_visit = sample["sample_visit_row_id"].astype(np.int64, copy=False)
    sample_ordinal = sample["sample_ordinal_within_visit"].astype(np.int64, copy=False)
    expected_offset = 0
    trajectories: list[NearFieldVisitTrajectory] = []
    for row in range(n_visits):
        offset = int(offsets[row])
        count = int(counts[row])
        stop = offset + count
        if offset != expected_offset or count <= 0 or stop > n_samples:
            _fail("Near-field visit ragged slices are not gapless and positive.")
        if not np.all(sample_visit[offset:stop] == row) or not np.array_equal(
            sample_ordinal[offset:stop], np.arange(count, dtype=np.int64)
        ):
            _fail("Near-field visit sample membership is inconsistent.")

        canonical_valid = sample["sample_canonical_valid"][offset:stop].astype(
            bool, copy=False
        )
        canonical_x = sample["sample_canonical_x_mm"][offset:stop].astype(
            np.float64, copy=False
        )
        canonical_y = sample["sample_canonical_y_mm"][offset:stop].astype(
            np.float64, copy=False
        )
        finite_xy = np.isfinite(canonical_x) & np.isfinite(canonical_y)
        if not np.array_equal(finite_xy, canonical_valid):
            _fail("Canonical visit coordinate validity and finiteness disagree.")
        distance = sample["sample_distance_mm"][offset:stop].astype(
            np.float64, copy=False
        )
        if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
            _fail("Persisted visit distances must be finite and non-negative.")
        time_first = sample["sample_time_from_first_sample_s"][offset:stop].astype(
            np.float64, copy=False
        )
        frame_id = sample["sample_acquisition_frame_id"][offset:stop].astype(
            np.int64, copy=False
        )
        timestamp = sample["sample_timestamp_ns_session"][offset:stop].astype(
            np.int64, copy=False
        )
        if (
            np.any(~np.isfinite(time_first))
            or not math.isclose(float(time_first[0]), 0.0, abs_tol=1e-12)
            or (count > 1 and np.any(np.diff(time_first) <= 0.0))
        ):
            _fail("Persisted within-visit time is not exact and monotonic.")
        if count > 1 and (
            np.any(np.diff(frame_id) != 1) or np.any(np.diff(timestamp) <= 0)
        ):
            _fail("A persisted visit sample slice crosses a frame or timing gap.")
        timestamp_time = (timestamp - timestamp[0]).astype(np.float64) / 1e9
        if not np.allclose(time_first, timestamp_time, rtol=0.0, atol=1e-12):
            _fail("Persisted visit-relative time disagrees with session timestamps.")
        near_member = sample["sample_near_zone_member"][offset:stop].astype(
            bool, copy=False
        )
        if not np.array_equal(near_member, distance <= near_zone):
            _fail("Persisted near-zone membership disagrees with distance.")
        if np.any(
            canonical_valid
            & ~np.isclose(
                np.hypot(canonical_x, canonical_y),
                distance,
                rtol=1e-5,
                atol=1e-5,
            )
        ):
            _fail("Canonical visit coordinates disagree with persisted distance.")
        cpa = int(visit["visit_cpa_sample_ordinal"][row])
        if cpa < 0 or cpa >= count or cpa != int(np.argmin(distance)):
            _fail("Persisted visit CPA ordinal disagrees with sample distances.")
        if not math.isclose(
            float(visit["visit_min_distance_mm"][row]),
            float(distance[cpa]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            _fail("Persisted visit minimum distance disagrees with its CPA sample.")

        epoch_code = int(visit["visit_epoch_role_code"][row])
        chaser_code = int(visit["visit_chaser_identity_code"][row])
        behavior_code = int(visit["visit_behavior_role_code"][row])
        left_code = int(visit["visit_left_censor_reason_code"][row])
        right_code = int(visit["visit_right_censor_reason_code"][row])
        quality_code = int(visit["visit_quality_code"][row])
        for registry_name, code in (
            ("epoch_role", epoch_code),
            ("chaser", chaser_code),
            ("behavior_role", behavior_code),
            ("left_censor_reason", left_code),
            ("right_censor_reason", right_code),
            ("quality", quality_code),
        ):
            if code not in registries[registry_name]:
                _fail(f"Visit row references an unknown {registry_name} code.")
        entry_observed = bool(visit["visit_entry_observed"][row])
        exit_observed = bool(visit["visit_exit_observed"][row])
        complete = bool(visit["visit_complete"][row])
        if complete != (entry_observed and exit_observed):
            _fail("Persisted visit completeness disagrees with its boundaries.")
        if entry_observed != (left_code == LEFT_CENSOR_NONE) or exit_observed != (
            right_code == RIGHT_CENSOR_NONE
        ):
            _fail("Persisted visit censor reasons disagree with its boundaries.")
        entry_time = sample["sample_time_from_observed_entry_s"][offset:stop].astype(
            np.float64, copy=False
        )
        if entry_observed:
            if not np.array_equal(entry_time, time_first):
                _fail("Observed-entry visit time disagrees with its first sample.")
        elif np.any(~np.isnan(entry_time)):
            _fail("Left-censored visit incorrectly claims an observed-entry time.")
        expected_quality = (
            QUALITY_OK if count >= minimum_quality_count else QUALITY_SHORT
        )
        if quality_code != expected_quality:
            _fail("Persisted visit quality disagrees with its sample count.")
        active_duration = float(visit["visit_observed_active_duration_s"][row])
        if not math.isclose(
            active_duration,
            float(time_first[-1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail("Persisted visit duration disagrees with its exact sample times.")
        complete_dwell = float(visit["visit_complete_dwell_s"][row])
        if complete:
            if not math.isfinite(complete_dwell) or complete_dwell < active_duration:
                _fail("Complete visit dwell is inconsistent with retained samples.")
        elif not math.isnan(complete_dwell):
            _fail("Censored visit must not claim a complete dwell duration.")

        trajectories.append(
            NearFieldVisitTrajectory(
                visit_row_id=row,
                visit_key_sha256=bytes(keys[row]).hex(),
                visit_ordinal=int(visit["visit_ordinal"][row]),
                epoch_role_code=epoch_code,
                epoch_window_id=int(visit["visit_epoch_window_id"][row]),
                chaser_identity_code=chaser_code,
                behavior_role_code=behavior_code,
                entry_observed=entry_observed,
                exit_observed=exit_observed,
                complete=complete,
                left_censor_reason=registries["left_censor_reason"][left_code],
                right_censor_reason=registries["right_censor_reason"][right_code],
                quality=registries["quality"][quality_code],
                observed_active_duration_s=active_duration,
                complete_dwell_s=complete_dwell,
                min_distance_mm=float(visit["visit_min_distance_mm"][row]),
                cpa_sample_ordinal=cpa,
                acquisition_frame_id=_readonly_slice(
                    sample["sample_acquisition_frame_id"], offset, stop
                ),
                timestamp_ns_session=_readonly_slice(
                    sample["sample_timestamp_ns_session"], offset, stop
                ),
                canonical_x_mm=_readonly_slice(canonical_x, 0, count),
                canonical_y_mm=_readonly_slice(canonical_y, 0, count),
                canonical_valid=_readonly_slice(canonical_valid, 0, count),
                distance_mm=_readonly_slice(distance, 0, count),
                near_zone_member=_readonly_slice(
                    sample["sample_near_zone_member"], offset, stop
                ),
                time_from_first_sample_s=_readonly_slice(time_first, 0, count),
                time_from_observed_entry_s=_readonly_slice(
                    sample["sample_time_from_observed_entry_s"], offset, stop
                ),
            )
        )
        expected_offset = stop
    if expected_offset != n_samples:
        _fail("Near-field visit ragged slices do not cover the exact sample table.")

    epoch_records = scientific.get("epoch_records")
    if not isinstance(epoch_records, (tuple, list)) or not epoch_records:
        _fail("Near-field visit epoch records are absent.")
    if n_summary != len(epoch_records) * n_chasers:
        _fail("Near-field visit summary cardinality disagrees with dimensions.")
    expected_epoch_windows: set[tuple[int, int]] = set()
    for record in epoch_records:
        if not isinstance(record, Mapping):
            _fail("Near-field visit epoch record is malformed.")
        role = record.get("analysis_role")
        window = record.get("window_id")
        if type(role) is not str or type(window) is not int:
            _fail("Near-field visit epoch identity is malformed.")
        matching_codes = [
            code for code, label in registries["epoch_role"].items() if label == role
        ]
        if len(matching_codes) != 1:
            _fail("Near-field visit epoch role has no unique registry code.")
        expected_epoch_windows.add((matching_codes[0], window))

    panels: list[NearFieldVisitPanel] = []
    seen_panel_keys: set[tuple[int, int, int]] = set()
    visit_counts: dict[tuple[int, int, int], list[int]] = {}
    for trajectory in trajectories:
        key = (
            trajectory.epoch_role_code,
            trajectory.epoch_window_id,
            trajectory.chaser_identity_code,
        )
        counts_for_key = visit_counts.setdefault(key, [0, 0, 0])
        counts_for_key[0] += 1
        counts_for_key[1 if trajectory.complete else 2] += 1
    for row in range(n_summary):
        epoch_code = int(summary["summary_epoch_role_code"][row])
        window_id = int(summary["summary_epoch_window_id"][row])
        chaser_code = int(summary["summary_chaser_identity_code"][row])
        behavior_code = int(summary["summary_behavior_role_code"][row])
        key = (epoch_code, window_id, chaser_code)
        if key in seen_panel_keys:
            _fail("Near-field visit summaries contain a duplicate panel key.")
        seen_panel_keys.add(key)
        for registry_name, code in (
            ("epoch_role", epoch_code),
            ("chaser", chaser_code),
            ("behavior_role", behavior_code),
        ):
            if code not in registries[registry_name]:
                _fail(f"Summary row references an unknown {registry_name} code.")
        expected_counts = visit_counts.get(key, [0, 0, 0])
        persisted_counts = [
            int(summary["summary_total_visit_count"][row]),
            int(summary["summary_complete_visit_count"][row]),
            int(summary["summary_censored_visit_count"][row]),
        ]
        if persisted_counts != expected_counts:
            _fail("Near-field visit summaries do not conserve persisted visits.")
        if any(
            trajectory.behavior_role_code != behavior_code
            for trajectory in trajectories
            if (
                trajectory.epoch_role_code,
                trajectory.epoch_window_id,
                trajectory.chaser_identity_code,
            )
            == key
        ):
            _fail("A persisted visit disagrees with its summary behavior role.")
        short_count = sum(
            trajectory.quality == "short_visit_retained"
            for trajectory in trajectories
            if (
                trajectory.epoch_role_code,
                trajectory.epoch_window_id,
                trajectory.chaser_identity_code,
            )
            == key
        )
        if int(summary["summary_short_visit_count"][row]) != short_count:
            _fail("Near-field visit summaries do not conserve short visits.")
        panels.append(
            NearFieldVisitPanel(
                epoch_role_code=epoch_code,
                epoch_role=registries["epoch_role"][epoch_code],
                epoch_window_id=window_id,
                chaser_identity_code=chaser_code,
                chaser_identity=registries["chaser"][chaser_code],
                behavior_role_code=behavior_code,
                behavior_role=registries["behavior_role"][behavior_code],
                total_visit_count=persisted_counts[0],
                complete_visit_count=persisted_counts[1],
                censored_visit_count=persisted_counts[2],
                short_visit_count=short_count,
            )
        )
    if set(visit_counts).difference(seen_panel_keys):
        _fail("A persisted visit has no matching exact summary row.")
    if {(panel.epoch_role_code, panel.epoch_window_id) for panel in panels} != (
        expected_epoch_windows
    ):
        _fail("Near-field visit summaries disagree with semantic epoch records.")
    for key, grouped in visit_counts.items():
        ordinals = sorted(
            trajectory.visit_ordinal
            for trajectory in trajectories
            if (
                trajectory.epoch_role_code,
                trajectory.epoch_window_id,
                trajectory.chaser_identity_code,
            )
            == key
        )
        if ordinals != list(range(grouped[0])):
            _fail("Near-field visit ordinals are not dense within one panel.")

    return NearFieldVisitTrajectoryView(
        recording_id=handle.recording_id,
        source_run_path=handle.run_path,
        source_manifest_sha256=handle.manifest_sha256,
        source_scientific_payload_sha256=handle.scientific_payload_sha256,
        position_provider_id=str(scientific["position_provider"]["provider_id"]),
        position_provider_digest=str(
            scientific["position_provider"]["provider_digest"]
        ),
        verification_mode=handle.verification_mode,
        validation_receipt_sha256=handle.receipt_digest,
        near_zone_radius_mm=near_zone,
        near_entry_radius_mm=enter,
        near_exit_radius_mm=exit_radius,
        coordinate_convention=str(scientific["coordinate_conventions"]["canonical"]),
        panels=tuple(panels),
        visits=tuple(trajectories),
    )


__all__ = [
    "VISIT_TRAJECTORY_ARRAY_NAMES",
    "VISIT_TRAJECTORY_ARRAY_DTYPES",
    "ChaserNearFieldVisitViewError",
    "NearFieldVisitPanel",
    "NearFieldVisitTrajectory",
    "NearFieldVisitTrajectoryView",
    "validated_near_field_visit_trajectory_view",
]
