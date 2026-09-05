"""Bundle-backed canonical bout and interval distribution inputs."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingDistributionMetricInput,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    RecordingDistributionScope,
    ScopeMaskProjection,
    exact_source_membership_masks,
    fully_contained_frame_event_masks,
    fully_contained_time_event_masks,
    validate_scope_registry,
)
from fisheye.group_statistics.validated_behavior_distributions import (
    derive_bout_heading_values,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DistributionMetricSpec,
)

from .provider_epoch_behavior_summary_source_handle import (
    load_provider_epoch_behavior_summary_source_handle,
)
from .recording_distribution_motion_adapter import (
    ProviderMotionDistributionContext,
    load_provider_motion_distribution_context,
)
from .recording_distribution_scope_adapters import (
    PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID,
)
from .recording_distribution_timebase_adapter import (
    RecordingSessionTimebase,
    require_scope_timebase_binding,
)
from .validated_recording_behavior_source import ValidatedRecordingBehaviorSource


_BOUT_VALUE_COLUMNS = frozenset(
    {
        "duration_s",
        "path_length_mm",
        "net_displacement_mm",
        "mean_speed_mm_s",
        "peak_speed_mm_s",
        "tortuosity",
        "net_heading_change_deg",
        "abs_net_heading_change_deg",
        "heading_path_deg",
    }
)
_BOUT_SOURCE_FIELD = MappingProxyType(
    {
        "duration_s": "duration_s",
        "path_length_mm": "path_length_mm",
        "net_displacement_mm": "net_displacement_mm",
        "mean_speed_mm_s": "mean_speed_mm_s",
        "peak_speed_mm_s": "peak_physical_speed_mm_s",
    }
)
_CANONICAL_BOUT_VALIDITY = "finite_nonnegative_canonical_bout_value_v1"
_HEADING_VALIDITY = "derived_angular_valid_and_epoch_crosschecked_v1"
_INTERVAL_VALIDITY = "producer_interval_canonical_axis_epoch_crosschecked_v1"
_EPOCH_BOUT_ARRAYS = (
    "bout_source_row",
    "bout_id",
    "bout_start_frame",
    "bout_end_frame",
    "bout_net_heading_change_deg",
    "abs_bout_net_heading_change_deg",
    "bout_heading_path_deg",
    "analysis_role",
    "source_interval_sha256",
)


class RecordingDistributionBoutAdapterError(ValueError):
    """The canonical bout source cannot supply exact distribution inputs."""


def _fail(message: str) -> None:
    raise RecordingDistributionBoutAdapterError(message)


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (
                bytes(value).rstrip(b"\x00").decode("utf-8")
                if isinstance(value, (bytes, np.bytes_))
                else str(value)
            )
            for value in np.asarray(values).reshape(-1)
        ],
        dtype=object,
    )


def _constant(value: Any, count: int) -> np.ndarray:
    result = np.empty(count, dtype=object)
    result[:] = value
    return result


def _bout_fields(bouts: np.ndarray) -> set[str]:
    fields = set(bouts.dtype.names or ())
    required = {
        "candidate_id",
        "signal_id",
        "track_id",
        "bout_id",
        "start_frame",
        "end_frame",
        "duration_s",
        "path_length_mm",
        "net_displacement_mm",
        "mean_speed_mm_s",
        "peak_physical_speed_mm_s",
    }
    if not required.issubset(fields):
        _fail(f"Canonical bout table lacks fields: {sorted(required - fields)!r}.")
    return fields


def _load_epoch_membership(
    source: ValidatedRecordingBehaviorSource,
    *,
    scopes: Sequence[RecordingDistributionScope],
    bouts: np.ndarray,
    net_heading: np.ndarray,
    heading_path: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    child = source.scientific_child("epoch_behavior")
    binding = child.binding
    paths = tuple(f"per_epoch_bouts/{name}" for name in _EPOCH_BOUT_ARRAYS)
    handle = load_provider_epoch_behavior_summary_source_handle(
        source.analysis_zarr,
        run_name=str(binding["run_path"]).rsplit("/", 1)[-1],
        expected_recording_id=source.recording_id,
        direct_validation_receipt=binding["receipt_path"],
        required_array_paths=paths,
    )
    observed = {
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "payload_digest": handle.payload_digest,
        "receipt_sha256": handle.receipt_digest,
    }
    expected = {
        "run_path": binding["run_path"],
        "manifest_sha256": binding["manifest_sha256"],
        "payload_digest": binding["payload_digest"],
        "receipt_sha256": binding["receipt_sha256"],
    }
    if observed != expected:
        _fail("Epoch behavior child differs from the validated bundle binding.")
    handle.require_verified_arrays(paths)
    arrays = {
        name: np.asarray(handle.array(f"per_epoch_bouts/{name}")).reshape(-1)
        for name in _EPOCH_BOUT_ARRAYS
    }
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        _fail("Epoch bout membership arrays do not share one row axis.")
    roles = _decode(arrays["analysis_role"])
    bounded_scopes = tuple(validate_scope_registry(scopes))[1:]
    interval_by_role = {
        scope.scope_id: str(scope.source_binding.get("source_interval_sha256") or "")
        for scope in bounded_scopes
    }
    if any(len(value) != 64 for value in interval_by_role.values()):
        _fail("Protocol-semantic scopes lack exact source-interval digests.")
    allowed_scope_ids = set(interval_by_role)
    unknown = sorted(set(roles.tolist()) - allowed_scope_ids)
    if unknown:
        _fail(f"Epoch bout membership names unknown scopes: {unknown!r}.")
    membership = np.empty(bouts.size, dtype=object)
    membership[:] = None
    source_rows = np.asarray(arrays["bout_source_row"], dtype=np.int64)
    if (
        np.any(source_rows < 0)
        or np.any(source_rows >= bouts.size)
        or np.unique(source_rows).size != source_rows.size
    ):
        _fail("Epoch bout source-row mapping is duplicated or out of range.")
    interval_digests = _decode(arrays["source_interval_sha256"])
    for index, source_row in enumerate(source_rows):
        row = int(source_row)
        role = str(roles[index])
        if interval_digests[index] != interval_by_role[role]:
            _fail(
                "Epoch bout membership binds a different semantic interval "
                f"for scope {role!r}."
            )
        if (
            int(arrays["bout_id"][index]) != int(bouts["bout_id"][row])
            or int(arrays["bout_start_frame"][index])
            != int(bouts["start_frame"][row])
            or int(arrays["bout_end_frame"][index]) != int(bouts["end_frame"][row])
        ):
            _fail("Epoch bout membership differs from the canonical bout row.")
        for calculated, persisted in (
            (net_heading[row], float(arrays["bout_net_heading_change_deg"][index])),
            (
                abs(net_heading[row]) if math.isfinite(net_heading[row]) else np.nan,
                float(arrays["abs_bout_net_heading_change_deg"][index]),
            ),
            (heading_path[row], float(arrays["bout_heading_path_deg"][index])),
        ):
            if not (
                (math.isnan(calculated) and math.isnan(persisted))
                or math.isclose(calculated, persisted, rel_tol=1e-7, abs_tol=1e-6)
            ):
                _fail("Derived bout heading differs from persisted epoch evidence.")
        membership[row] = roles[index]
    receipt = {
        **observed,
        "receipt_path": str(binding["receipt_path"]),
        "verified_array_paths": list(handle.verified_array_paths),
    }
    return membership, MappingProxyType(receipt)


def _event_projection(
    scopes: Sequence[RecordingDistributionScope],
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    start_time_ns: np.ndarray,
    end_time_ns: np.ndarray,
    time_valid: np.ndarray,
    exact_membership: np.ndarray | None,
    require_exact_protocol_membership: bool,
) -> ScopeMaskProjection:
    bounded = tuple(validate_scope_registry(scopes))[1:]
    if not bounded:
        return fully_contained_frame_event_masks(
            scopes,
            start_acquisition_frame_id=starts,
            end_acquisition_frame_id=ends,
        )
    providers = {scope.scope_provider_id for scope in bounded}
    axes = {scope.axis_kind for scope in bounded}
    if providers == {PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID}:
        if require_exact_protocol_membership:
            if exact_membership is None:
                _fail("Protocol-semantic bout scopes lack exact source membership.")
            return exact_source_membership_masks(
                scopes, source_scope_id=exact_membership
            )
        return fully_contained_frame_event_masks(
            scopes,
            start_acquisition_frame_id=starts,
            end_acquisition_frame_id=ends,
        )
    if axes == {"acquisition_frame"}:
        return fully_contained_frame_event_masks(
            scopes,
            start_acquisition_frame_id=starts,
            end_acquisition_frame_id=ends,
        )
    if axes == {"session_time_ns"}:
        return fully_contained_time_event_masks(
            scopes,
            start_timestamp_ns_session=start_time_ns,
            end_timestamp_ns_session=end_time_ns,
            timestamp_valid=time_valid,
        )
    _fail("One event adapter cannot mix incompatible bounded scope providers.")


def canonical_bout_distribution_inputs(
    source: ValidatedRecordingBehaviorSource,
    scopes: Sequence[RecordingDistributionScope],
    metric_specs: Sequence[DistributionMetricSpec],
    *,
    motion_context: ProviderMotionDistributionContext | None = None,
    session_timebase: RecordingSessionTimebase | None = None,
) -> tuple[ProviderMotionDistributionContext, tuple[RecordingDistributionMetricInput, ...], Mapping[str, Any]]:
    """Build canonical bout and IBI metrics with exact semantic membership."""

    if type(source) is not ValidatedRecordingBehaviorSource:
        _fail("Bout adapter requires one validated recording source.")
    ordered_scopes = validate_scope_registry(scopes)
    selected = tuple(
        spec
        for spec in metric_specs
        if spec.source_surface
        in {"bout_observations", "inter_bout_interval_observations"}
    )
    if not selected:
        _fail("No bout or inter-bout-interval metrics were selected.")
    unsupported = sorted(
        {
            spec.value_column
            for spec in selected
            if spec.source_surface == "bout_observations"
            and spec.value_column not in _BOUT_VALUE_COLUMNS
        }
    )
    if unsupported:
        _fail(f"Unsupported canonical bout value columns: {unsupported!r}.")
    context = motion_context or load_provider_motion_distribution_context(
        source,
        ordered_scopes,
        value_columns=(),
        session_timebase=session_timebase,
    )
    if session_timebase is not None and context.session_timebase is not session_timebase:
        _fail("Bout and provider-motion adapters received different timebases.")
    tables = source.canonical_swim_bout_tables()
    bouts = np.asarray(tables.bouts)
    _bout_fields(bouts)
    binding = source.bundle["source_bindings"]["canonical_swim_bouts"]["source"]
    for field, expected in (
        ("track_id", int(binding["track_id"])),
        ("candidate_id", int(binding["default_candidate_id"])),
        ("signal_id", int(binding["default_signal_id"])),
    ):
        if bouts.size and not np.all(np.asarray(bouts[field]) == expected):
            _fail(f"Canonical bout rows contain a foreign {field}.")
    starts = np.asarray(bouts["start_frame"], dtype=np.int64)
    ends = np.asarray(bouts["end_frame"], dtype=np.int64)
    motion_frames = np.asarray(
        context.arrays["source_acquisition_frame_index"], dtype=np.int64
    )
    net_heading, heading_path = derive_bout_heading_values(
        acquisition_frames=motion_frames,
        smoothed_heading_deg=context.arrays["smoothed_heading_degrees"],
        angular_sample_valid=context.arrays["angular_sample_valid"],
        bout_start_frames=starts,
        bout_end_frames=ends,
    )
    exact_membership = None
    epoch_receipt: Mapping[str, Any] = MappingProxyType({})
    bounded_providers = {scope.scope_provider_id for scope in ordered_scopes[1:]}
    if bounded_providers == {PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID}:
        exact_membership, epoch_receipt = _load_epoch_membership(
            source,
            scopes=ordered_scopes,
            bouts=bouts,
            net_heading=net_heading,
            heading_path=heading_path,
        )
    requires_session_time = require_scope_timebase_binding(
        ordered_scopes, context.session_timebase
    )
    if requires_session_time:
        if context.session_timebase is None:
            _fail("Session-time bout scopes require an exact recording timebase.")
        start_time_ns, start_time_valid = context.session_timebase.map_frames(starts)
        end_time_ns, end_time_valid = context.session_timebase.map_frames(ends)
    else:
        start_time_ns = np.zeros(starts.shape, dtype=np.int64)
        end_time_ns = np.zeros(ends.shape, dtype=np.int64)
        start_time_valid = np.zeros(starts.shape, dtype=bool)
        end_time_valid = np.zeros(ends.shape, dtype=bool)
    bout_projection = _event_projection(
        ordered_scopes,
        starts=starts,
        ends=ends,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        time_valid=start_time_valid & end_time_valid,
        exact_membership=exact_membership,
        require_exact_protocol_membership=True,
    )
    bout_fps = float(tables.run_attrs["fps"])
    if not math.isclose(bout_fps, context.fps, rel_tol=5e-5, abs_tol=5e-5):
        _fail("Canonical bouts and provider motion do not close one FPS.")
    bout_fields = set(bouts.dtype.names or ())
    if {"start_time_s", "end_time_s"}.issubset(bout_fields) and bouts.size:
        if not np.allclose(
            np.asarray(bouts["start_time_s"], dtype=np.float64),
            starts / bout_fps,
            rtol=5e-5,
            atol=1e-8,
        ) or not np.allclose(
            np.asarray(bouts["end_time_s"], dtype=np.float64),
            ends / bout_fps,
            rtol=5e-5,
            atol=1e-8,
        ):
            _fail("Canonical bout time fields disagree with their frame/FPS axis.")
    source_identity_values = {
        "source_run_path": str(binding["run_path"]),
        "source_lineage_sha256": str(binding["lineage_hash"]),
        "source_frame_axis_sha256": str(binding["frame_axis_sha256"]),
        "source_track_motion_manifest_sha256": str(
            binding["source_track_motion_manifest_sha256"]
        ),
        "source_track_motion_verification_digest": str(
            binding["source_track_motion_verification_digest"]
        ),
        "track_id": int(binding["track_id"]),
        "candidate_id": int(binding["default_candidate_id"]),
        "signal_id": int(binding["default_signal_id"]),
        "signal_level": str(binding["default_signal_level"]),
        "event_time_axis_policy_id": (
            "exact_acquisition_frame_to_session_timestamp_no_interpolation_v1"
            if requires_session_time
            else "not_required_for_frame_scopes"
        ),
        "scope_timebase_sha256": (
            context.session_timebase.binding["timebase_sha256"]
            if context.session_timebase is not None
            else "not_required_for_frame_scopes"
        ),
    }
    bout_identity_arrays = {
        name: _constant(value, bouts.size)
        for name, value in source_identity_values.items()
    }
    path = np.asarray(bouts["path_length_mm"], dtype=np.float64)
    displacement = np.asarray(bouts["net_displacement_mm"], dtype=np.float64)
    tortuosity = np.divide(
        path,
        displacement,
        out=np.full(path.shape, np.nan, dtype=np.float64),
        where=displacement > 1e-6,
    )
    derived = {
        "tortuosity": tortuosity,
        "net_heading_change_deg": net_heading,
        "abs_net_heading_change_deg": np.abs(net_heading),
        "heading_path_deg": heading_path,
    }
    result: list[RecordingDistributionMetricInput] = []
    for spec in selected:
        if spec.source_surface != "bout_observations":
            continue
        if spec.value_column in derived:
            values = derived[spec.value_column]
        else:
            values = np.asarray(bouts[_BOUT_SOURCE_FIELD[spec.value_column]])
        if spec.validity_policy_id == _CANONICAL_BOUT_VALIDITY:
            valid = np.isfinite(values) & (values >= 0)
            if spec.coverage_policy == "log10_cover_valid_positive_range":
                valid &= values > 0
        elif spec.validity_policy_id == _HEADING_VALIDITY:
            valid = np.isfinite(values)
        else:
            _fail(f"{spec.metric_id}: unsupported canonical bout validity policy.")
        result.append(
            RecordingDistributionMetricInput(
                spec=spec,
                values=values,
                valid=valid,
                scope_projection=bout_projection,
                source_identity_arrays=bout_identity_arrays,
                source_identity_fallback=source_identity_values,
                valid_duration_s_by_scope=context.valid_duration_s_by_scope,
            )
        )

    intervals = np.asarray(tables.inter_bout_intervals)
    interval_specs = tuple(
        spec
        for spec in selected
        if spec.source_surface == "inter_bout_interval_observations"
    )
    if interval_specs:
        required = {
            "valid",
            "prev_end_frame",
            "next_start_frame",
            "prev_end_time_s",
            "next_start_time_s",
            "interval_s",
        }
        fields = set(intervals.dtype.names or ())
        if not required.issubset(fields):
            _fail(f"Canonical interval table lacks fields: {sorted(required - fields)!r}.")
        previous_end = np.asarray(intervals["prev_end_frame"], dtype=np.int64)
        next_start = np.asarray(intervals["next_start_frame"], dtype=np.int64)
        if intervals.size and (
            not np.allclose(
                np.asarray(intervals["prev_end_time_s"], dtype=np.float64),
                previous_end / bout_fps,
                rtol=5e-5,
                atol=1e-8,
            )
            or not np.allclose(
                np.asarray(intervals["next_start_time_s"], dtype=np.float64),
                next_start / bout_fps,
                rtol=5e-5,
                atol=1e-8,
            )
        ):
            _fail("Canonical interval time fields disagree with their frame/FPS axis.")
        if requires_session_time:
            assert context.session_timebase is not None
            previous_time_ns, previous_time_valid = context.session_timebase.map_frames(
                previous_end
            )
            next_time_ns, next_time_valid = context.session_timebase.map_frames(
                next_start
            )
        else:
            previous_time_ns = np.zeros(previous_end.shape, dtype=np.int64)
            next_time_ns = np.zeros(next_start.shape, dtype=np.int64)
            previous_time_valid = np.zeros(previous_end.shape, dtype=bool)
            next_time_valid = np.zeros(next_start.shape, dtype=bool)
        interval_projection = _event_projection(
            ordered_scopes,
            starts=previous_end,
            ends=next_start,
            start_time_ns=previous_time_ns,
            end_time_ns=next_time_ns,
            time_valid=previous_time_valid & next_time_valid,
            exact_membership=None,
            require_exact_protocol_membership=False,
        )
        interval_identity_arrays = {
            name: _constant(value, intervals.size)
            for name, value in source_identity_values.items()
        }
        for spec in interval_specs:
            if (
                spec.value_column != "interval_s"
                or spec.validity_policy_id != _INTERVAL_VALIDITY
            ):
                _fail(f"{spec.metric_id}: unsupported interval metric contract.")
            values = np.asarray(intervals["interval_s"], dtype=np.float64)
            valid = (
                np.asarray(intervals["valid"], dtype=bool)
                & np.isfinite(values)
                & (values >= 0)
            )
            result.append(
                RecordingDistributionMetricInput(
                    spec=spec,
                    values=values,
                    valid=valid,
                    scope_projection=interval_projection,
                    source_identity_arrays=interval_identity_arrays,
                    source_identity_fallback=source_identity_values,
                    valid_duration_s_by_scope=context.valid_duration_s_by_scope,
                )
            )
    return context, tuple(result), epoch_receipt


__all__ = [
    "RecordingDistributionBoutAdapterError",
    "canonical_bout_distribution_inputs",
]
