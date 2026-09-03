"""Optional exact fish--chaser distance inputs for recording distributions."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    load_chaser_relative_frame_targeted_source_handle,
)
from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingDistributionMetricInput,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    RecordingDistributionScope,
    ScopeMaskProjection,
    sample_scope_masks,
    transition_scope_masks,
    validate_scope_registry,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DistributionMetricSpec,
)

from .recording_distribution_timebase_adapter import (
    RecordingSessionTimebase,
    require_scope_timebase_binding,
)
from .validated_recording_behavior_source import ValidatedRecordingBehaviorSource


DISTANCE_TIME_WEIGHT_POLICY_ID = (
    "exact_session_timestamp_delta_positive_relative_transition_v1"
)

_DISTANCE_VALIDITY_POLICY_ID = (
    "exact_occurrence_relative_physical_and_time_transition_valid_v1"
)
_PROVIDER_CAPABILITY = MappingProxyType(
    {
        "keypoint": "chaser_relative_keypoint",
        "detection": "chaser_relative_detection",
    }
)
_BASE_ARRAYS = (
    "acquisition_frame_delta",
    "acquisition_frame_id",
    "chaser_behavior_role_code",
    "chaser_behavior_role_valid",
    "chaser_identity_code",
    "chaser_occurrence_member",
    "relative_distance_physical",
    "relative_physical_valid",
    "relative_transition_valid",
    "row_valid",
    "selection_member",
    "timestamp_delta_ns",
    "timestamp_ns",
    "timestamp_valid",
)


class RecordingDistributionDistanceAdapterError(ValueError):
    """Exact chaser-relative evidence cannot satisfy the metric contract."""


def _fail(message: str) -> None:
    raise RecordingDistributionDistanceAdapterError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _registry(value: object, *, field: str) -> Mapping[int, str]:
    raw = _mapping(value, field=field)
    result: dict[int, str] = {}
    for key, item in raw.items():
        try:
            code = int(key)
        except (TypeError, ValueError) as exc:
            raise RecordingDistributionDistanceAdapterError(
                f"{field} has a non-integer code."
            ) from exc
        if (
            code < 0
            or type(item) is not str
            or not item
            or item != item.strip()
            or code in result
        ):
            _fail(f"{field} has an invalid or duplicate record.")
        result[code] = item
    if not result:
        _fail(f"{field} must not be empty.")
    return MappingProxyType(result)


def _decode_codes(
    values: np.ndarray, registry: Mapping[int, str], *, field: str
) -> np.ndarray:
    codes = np.asarray(values)
    if codes.ndim != 1 or codes.dtype.kind not in "iu":
        _fail(f"{field} codes must be one integer vector.")
    unknown = sorted(set(map(int, np.unique(codes))) - set(registry))
    if unknown:
        _fail(f"{field} codes are absent from the sealed registry: {unknown!r}.")
    return np.asarray([registry[int(code)] for code in codes], dtype=object)


def _constant(value: Any, count: int) -> np.ndarray:
    result = np.empty(count, dtype=object)
    result[:] = value
    return result


def _join_projections(
    projections: Sequence[ScopeMaskProjection],
) -> ScopeMaskProjection:
    values = tuple(projections)
    if not values:
        _fail("At least one scope projection is required.")
    policy = values[0].membership_policy_id
    roster = tuple(values[0].masks)
    if any(
        item.membership_policy_id != policy
        or tuple(item.masks) != roster
        or tuple(item.uncovered) != roster
        for item in values
    ):
        _fail("Distance providers do not share one exact scope projection contract.")
    return ScopeMaskProjection(
        masks=MappingProxyType(
            {
                scope_id: np.concatenate(
                    [np.asarray(item.masks[scope_id], dtype=bool) for item in values]
                )
                for scope_id in roster
            }
        ),
        uncovered=MappingProxyType(
            {
                scope_id: np.concatenate(
                    [
                        np.asarray(item.uncovered[scope_id], dtype=bool)
                        for item in values
                    ]
                )
                for scope_id in roster
            }
        ),
        membership_policy_id=policy,
    )


def _load_provider_rows(
    source: ValidatedRecordingBehaviorSource,
    scopes: Sequence[RecordingDistributionScope],
    *,
    provider_role: str,
    session_timebase: RecordingSessionTimebase | None,
) -> Mapping[str, Any]:
    try:
        capability = _PROVIDER_CAPABILITY[provider_role]
    except KeyError as exc:
        raise RecordingDistributionDistanceAdapterError(
            f"Unsupported chaser-distance provider role {provider_role!r}."
        ) from exc
    child = source.scientific_child(capability)
    binding = child.binding
    handle = load_chaser_relative_frame_targeted_source_handle(
        binding["receipt_path"],
        required_base_arrays=_BASE_ARRAYS,
        required_body_arrays=(),
        collapsed_frame_arrays=(),
        expected_analysis_zarr=source.analysis_zarr,
        expected_recording_id=source.recording_id,
        expected_run_name=str(binding["run_path"]).rsplit("/", 1)[-1],
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
        _fail(f"{provider_role} relative-frame evidence differs from the bundle.")
    arrays = handle.base_arrays
    count = handle.n_rows
    if any(np.asarray(arrays[name]).shape[0] != count for name in _BASE_ARRAYS):
        _fail(f"{provider_role} targeted arrays do not share one row axis.")
    registries = _mapping(
        handle.manifest.get("identity_registries"), field="identity_registries"
    )
    behavior_registry = _registry(
        registries.get("behavior_role"), field="behavior_role registry"
    )
    chaser_registry = _registry(
        registries.get("chaser"), field="chaser identity registry"
    )
    behavior = _decode_codes(
        arrays["chaser_behavior_role_code"],
        behavior_registry,
        field="behavior role",
    )
    identity = _decode_codes(
        arrays["chaser_identity_code"], chaser_registry, field="chaser identity"
    )
    authority = _mapping(
        handle.source_authorities.get("fish_position"),
        field="fish position authority",
    )
    provider_id = str(authority.get("provider_id") or "")
    provider_digest = str(authority.get("provider_digest") or "")
    if not provider_id or len(provider_digest) != 64:
        _fail(f"{provider_role} fish-position authority is incomplete.")

    frames = np.asarray(arrays["acquisition_frame_id"], dtype=np.int64)
    timestamps = np.asarray(arrays["timestamp_ns"], dtype=np.int64)
    timestamp_valid = np.asarray(arrays["timestamp_valid"], dtype=bool)
    if session_timebase is not None:
        expected_timestamps, expected_valid = session_timebase.map_frames(frames)
        if not np.array_equal(timestamp_valid, expected_valid) or not np.array_equal(
            timestamps[timestamp_valid], expected_timestamps[timestamp_valid]
        ):
            _fail(
                f"{provider_role} relative timestamps differ from the requested "
                "recording timebase."
            )
    sample_projection = sample_scope_masks(
        scopes,
        acquisition_frame_id=frames,
        timestamp_ns_session=timestamps,
        timestamp_valid=timestamp_valid,
    )
    transition_projection = transition_scope_masks(
        scopes,
        acquisition_frame_id=frames,
        acquisition_frame_delta=arrays["acquisition_frame_delta"],
        timestamp_ns_session=timestamps,
        timestamp_delta_ns=arrays["timestamp_delta_ns"],
        timestamp_valid=timestamp_valid,
    )
    base_valid = np.logical_and.reduce(
        [
            np.asarray(arrays[name], dtype=bool)
            for name in (
                "chaser_behavior_role_valid",
                "selection_member",
                "chaser_occurrence_member",
                "row_valid",
                "relative_physical_valid",
            )
        ]
    )
    delta_seconds = (
        np.asarray(arrays["timestamp_delta_ns"], dtype=np.float64)
        / 1_000_000_000.0
    )
    delta_valid = (
        timestamp_valid
        & np.asarray(arrays["relative_transition_valid"], dtype=bool)
        & np.isfinite(delta_seconds)
        & (delta_seconds > 0)
    )
    delta_seconds[~delta_valid] = np.nan
    identity_values = {
        "position_provider_id": provider_id,
        "position_provider_digest": provider_digest,
        "source_run_path": handle.run_path,
        "source_manifest_sha256": handle.manifest_sha256,
        "source_payload_digest": handle.payload_digest,
        "source_receipt_sha256": handle.receipt_digest,
        "time_weight_policy_id": DISTANCE_TIME_WEIGHT_POLICY_ID,
        "scope_timebase_sha256": (
            session_timebase.binding["timebase_sha256"]
            if session_timebase is not None
            else "not_required_for_frame_scopes"
        ),
    }
    identity_arrays = {
        name: _constant(value, count) for name, value in identity_values.items()
    }
    identity_arrays["chaser_identity_code"] = np.asarray(
        arrays["chaser_identity_code"]
    )
    identity_arrays["chaser_identity"] = identity
    return MappingProxyType(
        {
            "values": np.asarray(arrays["relative_distance_physical"]),
            "valid": base_valid,
            "sample_projection": sample_projection,
            "transition_projection": transition_projection,
            "time_weights_s": delta_seconds,
            "provider_role": _constant(provider_role, count),
            "behavior_role": behavior,
            "identity_arrays": MappingProxyType(identity_arrays),
            "identity_fallback": MappingProxyType(
                {
                    **identity_values,
                    "chaser_identity_code": 0,
                    "chaser_identity": "empty_source_no_chaser_rows",
                }
            ),
            "binding": MappingProxyType(observed),
        }
    )


def chaser_distance_distribution_inputs(
    source: ValidatedRecordingBehaviorSource,
    scopes: Sequence[RecordingDistributionScope],
    metric_specs: Sequence[DistributionMetricSpec],
    *,
    provider_roles: Sequence[str] = ("keypoint", "detection"),
    session_timebase: RecordingSessionTimebase | None = None,
) -> tuple[tuple[RecordingDistributionMetricInput, ...], tuple[Mapping[str, Any], ...]]:
    """Build optional distance metrics across exact available providers."""

    if type(source) is not ValidatedRecordingBehaviorSource:
        _fail("Distance adapter requires one validated recording source.")
    ordered_scopes = validate_scope_registry(scopes)
    require_scope_timebase_binding(ordered_scopes, session_timebase)
    if session_timebase is not None and type(session_timebase) is not RecordingSessionTimebase:
        _fail("session_timebase must be one exact RecordingSessionTimebase.")
    selected = tuple(
        spec for spec in metric_specs if spec.source_surface == "chaser_relative_samples"
    )
    if not selected:
        _fail("No chaser-distance metric specifications were selected.")
    if any(
        spec.value_column != "relative_distance_mm"
        or spec.validity_policy_id != _DISTANCE_VALIDITY_POLICY_ID
        for spec in selected
    ):
        _fail("A selected chaser-distance metric has an unsupported contract.")
    if isinstance(provider_roles, (str, bytes)):
        _fail("provider_roles must select unique keypoint/detection providers.")
    requested = tuple(provider_roles)
    if (
        not requested
        or any(type(role) is not str for role in requested)
        or len(set(requested)) != len(requested)
        or set(requested) - set(_PROVIDER_CAPABILITY)
    ):
        _fail("provider_roles must select unique keypoint/detection providers.")
    rows = []
    for role in requested:
        capability = _PROVIDER_CAPABILITY[role]
        if source.capability_record(capability)["state"] == "complete":
            rows.append(
                _load_provider_rows(
                    source,
                    ordered_scopes,
                    provider_role=role,
                    session_timebase=session_timebase,
                )
            )
    if not rows:
        _fail("No requested exact chaser-relative provider is available.")

    sample_projection = _join_projections(
        [row["sample_projection"] for row in rows]
    )
    transition_projection = _join_projections(
        [row["transition_projection"] for row in rows]
    )
    values = np.concatenate([np.asarray(row["values"]) for row in rows])
    valid = np.concatenate([np.asarray(row["valid"], dtype=bool) for row in rows])
    weights = np.concatenate(
        [np.asarray(row["time_weights_s"], dtype=np.float64) for row in rows]
    )
    groups = {
        "provider_role": np.concatenate(
            [np.asarray(row["provider_role"], dtype=object) for row in rows]
        ),
        "behavior_role": np.concatenate(
            [np.asarray(row["behavior_role"], dtype=object) for row in rows]
        ),
    }
    identity_names = tuple(rows[0]["identity_arrays"])
    if any(tuple(row["identity_arrays"]) != identity_names for row in rows):
        _fail("Distance providers do not expose one source-identity schema.")
    identities = {
        name: np.concatenate(
            [np.asarray(row["identity_arrays"][name]) for row in rows]
        )
        for name in identity_names
    }
    fallback = {
        name: rows[0]["identity_fallback"][name] for name in identity_names
    }
    inputs = tuple(
        RecordingDistributionMetricInput(
            spec=spec,
            values=values,
            valid=valid,
            scope_projection=sample_projection,
            group_arrays=groups,
            source_identity_arrays=identities,
            source_identity_fallback=fallback,
            time_weights_s=weights,
            time_scope_projection=transition_projection,
        )
        for spec in selected
    )
    return inputs, tuple(row["binding"] for row in rows)


__all__ = [
    "DISTANCE_TIME_WEIGHT_POLICY_ID",
    "RecordingDistributionDistanceAdapterError",
    "chaser_distance_distribution_inputs",
]
