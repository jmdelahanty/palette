"""Exact-source gaze-to-chaser tracking successor.

Gaze and chaser bearing are compared only in the fish body frame.  The
successor requires independently valid eye-orientation and body-frame bearing
sources, keeps invalid rows explicit, and never substitutes world-frame gaze,
motion heading, or nasal-positive eye-angle fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.generalized_bout_response_successor import (
    ROLE_CODES,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.analysis.chaser_gaze_tracking"
SCHEMA_VERSION = 2
PREPARED_SCHEMA_ID = "palette.analysis.chaser_gaze_tracking.prepared_successor"
PREPARED_SCHEMA_VERSION = 1
METHOD_ID = "exact_eye_body_frame_gaze_vs_exact_chaser_body_bearing_v1"

EYE_LEFT = 1
EYE_RIGHT = 2


class GazeTrackingSuccessorError(ValueError):
    """Raised when an exact gaze successor cannot be prepared."""


def _fail(message: str) -> None:
    raise GazeTrackingSuccessorError(message)


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{name} must be one non-empty exact string.")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _vector(value: Any, *, name: str, dtype: Any, size: int) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype) or result.shape != (size,):
        _fail(
            f"{name} must have exact dtype {np.dtype(dtype).str!r} and "
            f"shape {(size,)!r}."
        )
    return result


def _float_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "f" or result.shape != shape:
        _fail(f"{name} must be one floating array with shape {shape!r}.")
    return np.asarray(result, dtype=np.float64)


def _wrap_deg(value: Any) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _linear_fit(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[int, float, float, float]:
    use = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    x_use, y_use = np.asarray(x, dtype=np.float64)[use], np.asarray(y, dtype=np.float64)[use]
    count = int(x_use.size)
    if count < 3 or float(np.ptp(x_use)) <= 1e-9:
        return count, math.nan, math.nan, math.nan
    centered_x = x_use - float(np.mean(x_use))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0:
        return count, math.nan, math.nan, math.nan
    gain = float(np.dot(centered_x, y_use - float(np.mean(y_use))) / denominator)
    intercept = float(np.mean(y_use) - gain * np.mean(x_use))
    correlation = float(np.corrcoef(x_use, y_use)[0, 1])
    return count, gain, intercept, correlation


@dataclass(frozen=True, slots=True)
class GazeTrackingInput:
    recording_id: str
    source_relative_frame_run_path: str
    source_relative_frame_manifest_sha256: str
    source_eye_run_path: str
    source_eye_manifest_sha256: str
    source_eye_convention_receipt_sha256: str
    source_eye_channel_policy: str
    source_semantic_selection_manifest_sha256: str
    n_frames: int
    n_chasers: int
    acquisition_frame_id_by_frame: np.ndarray
    timestamp_ns_by_frame: np.ndarray
    timestamp_valid_by_frame: np.ndarray
    semantic_role_code_by_frame: np.ndarray
    chaser_identity_code: np.ndarray
    distance_mm: np.ndarray
    distance_valid: np.ndarray
    chaser_bearing_deg: np.ndarray
    chaser_bearing_valid: np.ndarray
    gaze_signed_deg: np.ndarray
    gaze_valid: np.ndarray
    vergence_deg: np.ndarray
    vergence_valid: np.ndarray
    lock_threshold_deg: float = 10.0
    minimum_lock_duration_s: float = 0.1
    maximum_tracking_distance_mm: float = 50.0
    accessible_quantiles: tuple[float, float] = (0.025, 0.975)


@dataclass(frozen=True, slots=True)
class PreparedGazeTracking:
    recording_id: str
    n_gaze_rows: int
    n_summary_rows: int
    n_lock_events: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown gaze-tracking array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": np.asarray(values).dtype.str,
            "shape": list(np.asarray(values).shape),
            "content_sha256": array_values_sha256(np.asarray(values)),
        }
        for name, values in sorted(arrays.items())
    ]


def prepare_gaze_tracking_successor(source: GazeTrackingInput) -> PreparedGazeTracking:
    """Prepare frame/eye/chaser rows, summaries, and contiguous lock events."""

    if type(source) is not GazeTrackingInput:
        raise TypeError("source must be one GazeTrackingInput.")
    recording_id = _text(source.recording_id, name="recording_id")
    for name in (
        "source_relative_frame_run_path",
        "source_eye_run_path",
        "source_eye_channel_policy",
    ):
        _text(getattr(source, name), name=name)
    for name in (
        "source_relative_frame_manifest_sha256",
        "source_eye_manifest_sha256",
        "source_eye_convention_receipt_sha256",
        "source_semantic_selection_manifest_sha256",
    ):
        _digest(getattr(source, name), name=name)
    if type(source.n_frames) is not int or source.n_frames < 0:
        _fail("n_frames must be one non-negative exact integer.")
    if type(source.n_chasers) is not int or source.n_chasers <= 0:
        _fail("n_chasers must be one positive exact integer.")
    n_frames, n_chasers = source.n_frames, source.n_chasers
    n_rel = n_frames * n_chasers
    frame_ids = _vector(
        source.acquisition_frame_id_by_frame,
        name="acquisition_frame_id_by_frame",
        dtype=np.int64,
        size=n_frames,
    )
    if np.unique(frame_ids).size != n_frames:
        _fail("Acquisition frame identities are duplicated.")
    timestamp = _vector(
        source.timestamp_ns_by_frame,
        name="timestamp_ns_by_frame",
        dtype=np.int64,
        size=n_frames,
    )
    timestamp_valid = _vector(
        source.timestamp_valid_by_frame,
        name="timestamp_valid_by_frame",
        dtype=bool,
        size=n_frames,
    )
    role = _vector(
        source.semantic_role_code_by_frame,
        name="semantic_role_code_by_frame",
        dtype=np.uint8,
        size=n_frames,
    )
    if np.any(~np.isin(role, np.asarray([0, *ROLE_CODES.values()], dtype=np.uint8))):
        _fail("semantic_role_code_by_frame contains an unknown code.")
    codes = _vector(
        source.chaser_identity_code,
        name="chaser_identity_code",
        dtype=np.uint16,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    if n_frames and np.any(codes != codes[:1, :]):
        _fail("Chaser identity changed along the fixed chaser axis.")
    chaser_codes = (
        codes[0] if n_frames else np.arange(1, n_chasers + 1, dtype=np.uint16)
    )
    if np.unique(chaser_codes).size != n_chasers:
        _fail("Chaser identity codes are duplicated.")
    distance = _float_array(
        source.distance_mm,
        name="distance_mm",
        shape=(n_rel,),
    ).reshape(n_frames, n_chasers)
    distance_valid = _vector(
        source.distance_valid,
        name="distance_valid",
        dtype=bool,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    bearing = _float_array(
        source.chaser_bearing_deg,
        name="chaser_bearing_deg",
        shape=(n_rel,),
    ).reshape(n_frames, n_chasers)
    bearing_valid = _vector(
        source.chaser_bearing_valid,
        name="chaser_bearing_valid",
        dtype=bool,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    gaze = _float_array(
        source.gaze_signed_deg,
        name="gaze_signed_deg",
        shape=(n_frames, 2),
    )
    gaze_valid = np.asarray(source.gaze_valid)
    if gaze_valid.dtype != np.dtype(bool) or gaze_valid.shape != (n_frames, 2):
        _fail("gaze_valid must be exact bool with shape (n_frames, 2).")
    vergence = _float_array(
        source.vergence_deg,
        name="vergence_deg",
        shape=(n_frames,),
    )
    vergence_valid = _vector(
        source.vergence_valid,
        name="vergence_valid",
        dtype=bool,
        size=n_frames,
    )
    for values, valid, name in (
        (distance, distance_valid, "distance"),
        (bearing, bearing_valid, "bearing"),
        (gaze, gaze_valid, "gaze"),
        (vergence, vergence_valid, "vergence"),
    ):
        if np.any(valid & ~np.isfinite(values)):
            _fail(f"A valid {name} value is non-finite.")

    parameters: dict[str, float] = {}
    for name in (
        "lock_threshold_deg",
        "minimum_lock_duration_s",
        "maximum_tracking_distance_mm",
    ):
        value = getattr(source, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"{name} must be one positive finite number.")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            _fail(f"{name} must be one positive finite number.")
        parameters[name] = value
    quantiles = np.asarray(source.accessible_quantiles, dtype=np.float64)
    if (
        quantiles.shape != (2,)
        or np.any(~np.isfinite(quantiles))
        or not (0 <= quantiles[0] < quantiles[1] <= 1)
    ):
        _fail("accessible_quantiles must be an ordered pair in [0, 1].")
    eye_range = np.full((2, 2), np.nan, dtype=np.float64)
    for eye in range(2):
        values = gaze[:, eye][gaze_valid[:, eye]]
        if values.size:
            eye_range[eye] = np.quantile(values, quantiles)

    n_rows = n_frames * 2 * n_chasers
    frame_index = np.repeat(np.arange(n_frames, dtype=np.int64), 2 * n_chasers)
    eye_pos = np.tile(
        np.repeat(np.arange(2, dtype=np.int64), n_chasers), n_frames
    )
    chaser_pos = np.tile(np.arange(n_chasers, dtype=np.int64), n_frames * 2)
    row_gaze = gaze[frame_index, eye_pos]
    row_bearing = bearing[frame_index, chaser_pos]
    row_distance = distance[frame_index, chaser_pos]
    valid = (
        (role[frame_index] != 0)
        & gaze_valid[frame_index, eye_pos]
        & bearing_valid[frame_index, chaser_pos]
        & distance_valid[frame_index, chaser_pos]
        & (row_distance <= parameters["maximum_tracking_distance_mm"])
    )
    accessible = np.zeros(n_rows, dtype=bool)
    for eye in range(2):
        low, high = eye_range[eye]
        if np.isfinite(low) and np.isfinite(high):
            eye_rows = eye_pos == eye
            accessible[eye_rows] = (
                (row_bearing[eye_rows] >= low)
                & (row_bearing[eye_rows] <= high)
            )
    error = _wrap_deg(row_gaze - row_bearing)
    lock = valid & accessible & (np.abs(error) <= parameters["lock_threshold_deg"])

    arrays: dict[str, np.ndarray] = {
        "gaze_row_id": np.arange(n_rows, dtype=np.int64),
        "acquisition_frame_id": frame_ids[frame_index],
        "semantic_role_code": role[frame_index],
        "eye_code": (eye_pos + 1).astype(np.uint8),
        "chaser_identity_code": chaser_codes[chaser_pos],
        "distance_mm": row_distance.astype(np.float32),
        "bearing_deg": row_bearing.astype(np.float32),
        "gaze_signed_deg": row_gaze.astype(np.float32),
        "vergence_deg": vergence[frame_index].astype(np.float32),
        "valid": valid,
        "accessible": accessible,
        "gaze_error_deg": error.astype(np.float32),
        "lock_on": lock,
    }

    role_values = np.asarray(sorted(ROLE_CODES.values()), dtype=np.uint8)
    summary_count = int(role_values.size * 2 * n_chasers)
    summary: dict[str, np.ndarray] = {
        "summary_row_id": np.arange(summary_count, dtype=np.int64),
        "summary_role_code": np.zeros(summary_count, dtype=np.uint8),
        "summary_eye_code": np.zeros(summary_count, dtype=np.uint8),
        "summary_chaser_identity_code": np.zeros(summary_count, dtype=np.uint16),
        "summary_valid_sample_count": np.zeros(summary_count, dtype=np.int64),
        "summary_accessible_sample_count": np.zeros(summary_count, dtype=np.int64),
        "summary_lock_sample_count": np.zeros(summary_count, dtype=np.int64),
        "summary_lock_fraction": np.full(summary_count, np.nan, dtype=np.float64),
        "summary_median_abs_error_deg": np.full(summary_count, np.nan, dtype=np.float64),
        "summary_tracking_gain": np.full(summary_count, np.nan, dtype=np.float64),
        "summary_tracking_intercept_deg": np.full(summary_count, np.nan, dtype=np.float64),
        "summary_tracking_correlation": np.full(summary_count, np.nan, dtype=np.float64),
        "summary_regression_sample_count": np.zeros(summary_count, dtype=np.int64),
    }
    out = 0
    for role_code in role_values:
        for eye_code in (EYE_LEFT, EYE_RIGHT):
            for code in chaser_codes:
                mask = (
                    (arrays["semantic_role_code"] == role_code)
                    & (arrays["eye_code"] == eye_code)
                    & (arrays["chaser_identity_code"] == code)
                )
                use = mask & valid
                access = use & accessible
                summary["summary_role_code"][out] = role_code
                summary["summary_eye_code"][out] = eye_code
                summary["summary_chaser_identity_code"][out] = code
                summary["summary_valid_sample_count"][out] = int(np.count_nonzero(use))
                access_count = int(np.count_nonzero(access))
                lock_count = int(np.count_nonzero(mask & lock))
                summary["summary_accessible_sample_count"][out] = access_count
                summary["summary_lock_sample_count"][out] = lock_count
                if access_count:
                    summary["summary_lock_fraction"][out] = lock_count / access_count
                    summary["summary_median_abs_error_deg"][out] = float(
                        np.median(np.abs(error[access]))
                    )
                count, gain, intercept, correlation = _linear_fit(
                    row_bearing,
                    row_gaze,
                    access,
                )
                summary["summary_regression_sample_count"][out] = count
                summary["summary_tracking_gain"][out] = gain
                summary["summary_tracking_intercept_deg"][out] = intercept
                summary["summary_tracking_correlation"][out] = correlation
                out += 1
    arrays.update(summary)

    event_records: list[tuple[int, int, int, int, int, float, int, float]] = []
    for role_code in role_values:
        for eye_code in (EYE_LEFT, EYE_RIGHT):
            for code in chaser_codes:
                mask = (
                    (arrays["semantic_role_code"] == role_code)
                    & (arrays["eye_code"] == eye_code)
                    & (arrays["chaser_identity_code"] == code)
                    & lock
                )
                frame_mask = np.zeros(n_frames, dtype=bool)
                frame_mask[frame_index[mask]] = True
                padded = np.pad(frame_mask.astype(np.int8), (1, 1))
                starts = np.flatnonzero(np.diff(padded) == 1)
                ends = np.flatnonzero(np.diff(padded) == -1)
                for start, end in zip(starts.tolist(), ends.tolist()):
                    if end <= start:
                        continue
                    duration = math.nan
                    if timestamp_valid[start] and timestamp_valid[end - 1]:
                        duration = (timestamp[end - 1] - timestamp[start]) / 1e9
                    if not math.isfinite(duration):
                        continue
                    # A one-sample event has zero timestamp span. Preserve a
                    # conservative zero duration and apply the requested gate.
                    if duration < parameters["minimum_lock_duration_s"]:
                        continue
                    event_row_mask = mask & (frame_index >= start) & (frame_index < end)
                    event_records.append(
                        (
                            int(role_code),
                            int(eye_code),
                            int(code),
                            int(frame_ids[start]),
                            int(frame_ids[end - 1]),
                            float(duration),
                            int(end - start),
                            float(np.median(np.abs(error[event_row_mask]))),
                        )
                    )
    n_events = len(event_records)
    event_arrays = {
        "lock_event_row_id": np.arange(n_events, dtype=np.int64),
        "lock_event_role_code": np.asarray(
            [row[0] for row in event_records], dtype=np.uint8
        ),
        "lock_event_eye_code": np.asarray(
            [row[1] for row in event_records], dtype=np.uint8
        ),
        "lock_event_chaser_identity_code": np.asarray(
            [row[2] for row in event_records], dtype=np.uint16
        ),
        "lock_event_start_acquisition_frame_id": np.asarray(
            [row[3] for row in event_records], dtype=np.int64
        ),
        "lock_event_end_acquisition_frame_id_inclusive": np.asarray(
            [row[4] for row in event_records], dtype=np.int64
        ),
        "lock_event_duration_s": np.asarray(
            [row[5] for row in event_records], dtype=np.float64
        ),
        "lock_event_sample_count": np.asarray(
            [row[6] for row in event_records], dtype=np.int64
        ),
        "lock_event_median_abs_error_deg": np.asarray(
            [row[7] for row in event_records], dtype=np.float32
        ),
    }
    arrays.update(event_arrays)
    readonly = {name: _readonly(values) for name, values in arrays.items()}
    manifest_body = {
        "schema_id": PREPARED_SCHEMA_ID,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "scientific_schema": {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "row_unit": "acquisition_frame_x_eye_x_chaser",
            "summary_unit": "semantic_role_x_eye_x_chaser",
            "event_unit": "contiguous_lock_on_interval",
        },
        "recording_id": recording_id,
        "sources": {
            "relative_frame": {
                "run_path": source.source_relative_frame_run_path,
                "manifest_sha256": source.source_relative_frame_manifest_sha256,
            },
            "eye_orientation": {
                "run_path": source.source_eye_run_path,
                "manifest_sha256": source.source_eye_manifest_sha256,
                "convention_receipt_sha256": source.source_eye_convention_receipt_sha256,
                "channel_policy": source.source_eye_channel_policy,
            },
            "semantic_selection_manifest_sha256": (
                source.source_semantic_selection_manifest_sha256
            ),
        },
        "parameters": {
            **parameters,
            "accessible_quantiles": quantiles.tolist(),
            "empirical_eye_range_deg": eye_range.tolist(),
        },
        "dimensions": {
            "n_frames": n_frames,
            "n_chasers": n_chasers,
            "n_gaze_rows": n_rows,
            "n_summary_rows": summary_count,
            "n_lock_events": n_events,
        },
        "policy": {
            "gaze_field": "directed_left_right_gaze_signed_deg_in_fish_body_frame",
            "bearing_field": "exact_chaser_body_bearing_deg_anatomical_left_positive",
            "world_frame_gaze": "prohibited",
            "nasal_positive_eye_angle": "prohibited_for_object_bearing_comparison",
            "orientation_fallback": "prohibited",
            "invalid_rows": "retained_and_excluded_from_summaries",
            "cohort_inference_unit": "recording_fish",
        },
        "identity_registries": {
            "eye": {"1": "left", "2": "right"},
            "semantic_role": {str(value): name for name, value in ROLE_CODES.items()},
        },
        "array_declarations": _declarations(readonly),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze(
        {
            **manifest_body,
            "payload_digest": canonical_json_sha256(manifest_body),
        }
    )
    return PreparedGazeTracking(
        recording_id=recording_id,
        n_gaze_rows=n_rows,
        n_summary_rows=summary_count,
        n_lock_events=n_events,
        arrays=MappingProxyType(readonly),
        manifest=manifest,
    )


def gaze_tracking_input_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    eye_gaze: Any,
    *,
    lock_threshold_deg: float = 10.0,
    minimum_lock_duration_s: float = 0.1,
    maximum_tracking_distance_mm: float = 50.0,
    accessible_quantiles: tuple[float, float] = (0.025, 0.975),
) -> GazeTrackingInput:
    """Bind exact relative, semantic, and reviewed eye-gaze handles."""

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.eye_gaze_source_handle import (
        EyeGazeSourceHandle,
    )
    from fisheye.analysis_workflows.generalized_bout_response_successor import (
        semantic_role_codes_from_handles,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    if type(relative_frame) is not ChaserRelativeFrameSourceHandle:
        raise TypeError("relative_frame must be a strict loader-minted handle.")
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be a strict loader-minted handle.")
    if type(eye_gaze) is not EyeGazeSourceHandle:
        raise TypeError("eye_gaze must be a strict loader-minted reviewed handle.")
    relative_frame.assert_current()
    semantic_selection.assert_current()
    eye_gaze.assert_current()
    if not relative_frame.body_available:
        _fail("Gaze tracking requires the exact relative-frame body extension.")
    if not (
        relative_frame.analysis_zarr_path
        == semantic_selection.analysis_zarr
        == eye_gaze.analysis_zarr_path
    ):
        _fail("Gaze successor sources do not belong to one exact archive.")
    if not (
        relative_frame.recording_id
        == semantic_selection.recording_id
        == eye_gaze.recording_id
    ):
        _fail("Gaze successor sources belong to different recordings.")
    if relative_frame.run_manifest.get("scale_policy", {}).get("unit") != "mm":
        _fail("Chaser-relative physical distance is not explicitly in millimeters.")
    acquisition_matrix = relative_frame.base_frame_chaser("acquisition_frame_id")
    timestamp_matrix = relative_frame.base_frame_chaser("timestamp_ns")
    timestamp_valid_matrix = relative_frame.base_frame_chaser("timestamp_valid")
    if relative_frame.n_frames and (
        not np.all(acquisition_matrix == acquisition_matrix[:, :1])
        or not np.all(timestamp_matrix == timestamp_matrix[:, :1])
        or not np.all(timestamp_valid_matrix == timestamp_valid_matrix[:, :1])
    ):
        _fail("Relative-frame acquisition/timing evidence differs across chasers.")
    acquisition = np.asarray(acquisition_matrix[:, 0], dtype=np.int64)
    gaze, gaze_valid, vergence, vergence_valid = (
        eye_gaze.align_to_acquisition_frames(acquisition)
    )
    roles = semantic_role_codes_from_handles(relative_frame, semantic_selection)
    return GazeTrackingInput(
        recording_id=relative_frame.recording_id,
        source_relative_frame_run_path=relative_frame.run_path,
        source_relative_frame_manifest_sha256=relative_frame.manifest_sha256,
        source_eye_run_path=eye_gaze.run_path,
        source_eye_manifest_sha256=eye_gaze.logical_manifest_sha256,
        source_eye_convention_receipt_sha256=(
            eye_gaze.convention_receipt_sha256
        ),
        source_eye_channel_policy=(
            f"{eye_gaze.channel_variant}:"
            f"{','.join(eye_gaze.gaze_channel_names)}:"
            f"{eye_gaze.vergence_channel_name}"
        ),
        source_semantic_selection_manifest_sha256=(
            semantic_selection.manifest_sha256
        ),
        n_frames=relative_frame.n_frames,
        n_chasers=relative_frame.n_chasers,
        acquisition_frame_id_by_frame=acquisition,
        timestamp_ns_by_frame=np.asarray(timestamp_matrix[:, 0], dtype=np.int64),
        timestamp_valid_by_frame=np.asarray(
            timestamp_valid_matrix[:, 0], dtype=bool
        ),
        semantic_role_code_by_frame=roles,
        chaser_identity_code=relative_frame.base_array("chaser_identity_code"),
        distance_mm=np.asarray(
            relative_frame.base_array("relative_distance_physical"),
            dtype=np.float64,
        ),
        distance_valid=relative_frame.base_array("relative_physical_valid"),
        chaser_bearing_deg=np.asarray(
            relative_frame.body_array("body_bearing_deg"), dtype=np.float64
        ),
        chaser_bearing_valid=relative_frame.body_array("body_bearing_valid"),
        gaze_signed_deg=np.asarray(gaze, dtype=np.float64),
        gaze_valid=np.asarray(gaze_valid, dtype=bool),
        vergence_deg=np.asarray(vergence, dtype=np.float64),
        vergence_valid=np.asarray(vergence_valid, dtype=bool),
        lock_threshold_deg=lock_threshold_deg,
        minimum_lock_duration_s=minimum_lock_duration_s,
        maximum_tracking_distance_mm=maximum_tracking_distance_mm,
        accessible_quantiles=accessible_quantiles,
    )


def prepare_gaze_tracking_successor_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    eye_gaze: Any,
    **kwargs: Any,
) -> PreparedGazeTracking:
    """Prepare gaze tracking from exact current source handles."""

    return prepare_gaze_tracking_successor(
        gaze_tracking_input_from_handles(
            relative_frame,
            semantic_selection,
            eye_gaze,
            **kwargs,
        )
    )


__all__ = [
    "METHOD_ID",
    "PREPARED_SCHEMA_ID",
    "PREPARED_SCHEMA_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "GazeTrackingInput",
    "GazeTrackingSuccessorError",
    "PreparedGazeTracking",
    "gaze_tracking_input_from_handles",
    "prepare_gaze_tracking_successor",
    "prepare_gaze_tracking_successor_from_handles",
]
