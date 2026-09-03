"""Compose validated providers into one recording-local distribution result."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fisheye.group_statistics.recording_behavior_distribution_specs import (
    DEFAULT_RECORDING_DISTRIBUTION_METRICS,
)
from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingBehaviorDistributionConfig,
    RecordingBehaviorDistributionResult,
    compute_recording_behavior_distributions,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    RecordingDistributionScope,
    validate_scope_registry,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DistributionMetricSpec,
    validate_distribution_metric_specs,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_file

from .recording_distribution_bout_adapter import (
    canonical_bout_distribution_inputs,
)
from .recording_distribution_distance_adapter import (
    chaser_distance_distribution_inputs,
)
from .recording_distribution_motion_adapter import (
    load_provider_motion_distribution_context,
    provider_motion_distribution_inputs,
)
from .recording_distribution_timebase_adapter import RecordingSessionTimebase
from .validated_recording_behavior_source import ValidatedRecordingBehaviorSource


WORKFLOW_SCHEMA_ID = "palette.analysis.recording_behavior_distribution_workflow"
WORKFLOW_SCHEMA_VERSION = 1


class RecordingBehaviorDistributionWorkflowError(ValueError):
    """The selected bundle capabilities cannot close the requested product."""


def _fail(message: str) -> None:
    raise RecordingBehaviorDistributionWorkflowError(message)


@dataclass(frozen=True, slots=True)
class PreparedRecordingBehaviorDistribution:
    """Pure reduced output plus the exact adapter evidence used to construct it."""

    result: RecordingBehaviorDistributionResult
    adapter_evidence: Mapping[str, Any]
    omitted_metrics: tuple[Mapping[str, Any], ...]


def prepare_recording_behavior_distribution(
    source: ValidatedRecordingBehaviorSource,
    *,
    distribution_run_id: str,
    scopes: Sequence[RecordingDistributionScope],
    metric_specs: Sequence[
        DistributionMetricSpec
    ] = DEFAULT_RECORDING_DISTRIBUTION_METRICS,
    session_timebase: RecordingSessionTimebase | None = None,
    chaser_provider_roles: Sequence[str] = ("keypoint", "detection"),
    require_all_metrics: bool = False,
) -> PreparedRecordingBehaviorDistribution:
    """Reduce exact bundle sources without selecting or mutating any source run."""

    if type(source) is not ValidatedRecordingBehaviorSource:
        _fail("Workflow requires one validated recording source.")
    if type(require_all_metrics) is not bool:
        _fail("require_all_metrics must be the exact boolean.")
    ordered_scopes = validate_scope_registry(scopes)
    requested_specs = validate_distribution_metric_specs(metric_specs)
    inputs = []
    evidence: dict[str, Any] = {}
    omitted: list[Mapping[str, Any]] = []

    motion_specs = tuple(
        spec
        for spec in requested_specs
        if spec.source_surface == "provider_motion_samples"
    )
    event_specs = tuple(
        spec
        for spec in requested_specs
        if spec.source_surface
        in {"bout_observations", "inter_bout_interval_observations"}
    )
    motion_context = None
    if motion_specs:
        motion_context, motion_inputs = provider_motion_distribution_inputs(
            source,
            ordered_scopes,
            motion_specs,
            session_timebase=session_timebase,
        )
        inputs.extend(motion_inputs)
    elif event_specs:
        motion_context = load_provider_motion_distribution_context(
            source,
            ordered_scopes,
            value_columns=(),
            session_timebase=session_timebase,
        )
    if motion_context is not None:
        evidence["provider_motion"] = {
            "run_path": motion_context.projection.run_path,
            "manifest_sha256": motion_context.projection.manifest_sha256,
            "verification_digest": motion_context.projection.verification_digest,
            "track_id": motion_context.projection.track_id,
            "verified_array_sha256": dict(motion_context.projection.array_sha256),
        }

    if event_specs:
        assert motion_context is not None
        _context, event_inputs, epoch_receipt = canonical_bout_distribution_inputs(
            source,
            ordered_scopes,
            event_specs,
            motion_context=motion_context,
            session_timebase=session_timebase,
        )
        inputs.extend(event_inputs)
        evidence["canonical_swim_bouts"] = {
            "source": dict(
                source.bundle["source_bindings"]["canonical_swim_bouts"][
                    "source"
                ]
            ),
            "epoch_membership_receipt": dict(epoch_receipt),
        }

    distance_specs = tuple(
        spec
        for spec in requested_specs
        if spec.source_surface == "chaser_relative_samples"
    )
    if distance_specs:
        if isinstance(chaser_provider_roles, (str, bytes)):
            _fail(
                "chaser_provider_roles must select unique keypoint/detection "
                "providers."
            )
        requested_chaser_roles = tuple(chaser_provider_roles)
        if (
            not requested_chaser_roles
            or any(
                type(role) is not str or role not in {"keypoint", "detection"}
                for role in requested_chaser_roles
            )
            or len(set(requested_chaser_roles)) != len(requested_chaser_roles)
        ):
            _fail(
                "chaser_provider_roles must select unique keypoint/detection "
                "providers."
            )
        available_roles = tuple(
            role
            for role in requested_chaser_roles
            if source.capability_record(f"chaser_relative_{role}")["state"]
            == "complete"
        )
        if available_roles:
            distance_inputs, distance_bindings = chaser_distance_distribution_inputs(
                source,
                ordered_scopes,
                distance_specs,
                provider_roles=available_roles,
                session_timebase=session_timebase,
            )
            inputs.extend(distance_inputs)
            evidence["chaser_relative_distance"] = {
                "provider_roles": list(available_roles),
                "source_bindings": [dict(value) for value in distance_bindings],
            }
        else:
            if require_all_metrics:
                _fail("Requested chaser-distance metrics have no complete provider.")
            omitted.extend(
                MappingProxyType(
                    {
                        "metric_id": spec.metric_id,
                        "reason_code": "optional_chaser_relative_capability_unavailable",
                    }
                )
                for spec in distance_specs
            )

    supported_surfaces = {
        "provider_motion_samples",
        "bout_observations",
        "inter_bout_interval_observations",
        "chaser_relative_samples",
    }
    unsupported = sorted(
        {
            spec.source_surface
            for spec in requested_specs
            if spec.source_surface not in supported_surfaces
        }
    )
    if unsupported:
        _fail(f"Unsupported recording metric surfaces: {unsupported!r}.")
    if not inputs:
        _fail("No requested metric has one complete recording source.")
    published_ids = {item.spec.metric_id for item in inputs}
    expected_published = {
        spec.metric_id for spec in requested_specs
    } - {str(row["metric_id"]) for row in omitted}
    if published_ids != expected_published:
        _fail("Metric adapters did not close the requested publication roster.")

    source_record = {
        "schema_id": WORKFLOW_SCHEMA_ID,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "analysis_zarr": str(source.analysis_zarr),
        "bundle_path": str(source.bundle_path),
        "bundle_file_sha256": sha256_file(source.bundle_path),
        "bundle_record_sha256": source.bundle_sha256,
        "requested_metric_ids": [spec.metric_id for spec in requested_specs],
        "published_metric_ids": sorted(published_ids),
        "omitted_metrics": [dict(row) for row in omitted],
        "session_timebase": (
            None if session_timebase is None else dict(session_timebase.binding)
        ),
        "adapter_evidence": evidence,
    }
    config = RecordingBehaviorDistributionConfig(
        distribution_run_id=distribution_run_id,
        recording_id=source.recording_id,
        scopes=ordered_scopes,
        source_record=source_record,
    )
    result = compute_recording_behavior_distributions(config, tuple(inputs))
    return PreparedRecordingBehaviorDistribution(
        result=result,
        adapter_evidence=MappingProxyType(evidence),
        omitted_metrics=tuple(omitted),
    )


__all__ = [
    "PreparedRecordingBehaviorDistribution",
    "RecordingBehaviorDistributionWorkflowError",
    "WORKFLOW_SCHEMA_ID",
    "WORKFLOW_SCHEMA_VERSION",
    "prepare_recording_behavior_distribution",
]
