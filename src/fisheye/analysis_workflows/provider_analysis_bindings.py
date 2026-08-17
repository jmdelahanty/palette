"""Verified adapters from Phase 3 sources into Phase 4A offer contracts.

These helpers are the only bridge in this phase between loader-minted source
handles and the pure provider/offer records.  They never resolve selectors and
they keep incomplete recording or timing authority visible as blocked offer
readiness instead of inventing a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fisheye.analysis_workflows.body_frame_source_handle import (
    BodyFrameSourceHandle,
    load_body_frame_source_handle,
    require_body_frame_source_handle,
)
from fisheye.analysis_workflows.provider_analysis_offers import (
    AnalysisOffer,
    ProviderAnalysisOfferError,
    ProviderIdentity,
    ProviderKind,
    ProviderRequirements,
    ProviderRole,
    ScientificReadiness,
    TemporalSelectionIdentity,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    ProviderTrackMotionSourceHandle,
    require_provider_track_motion_source_handle,
)
from fisheye.analysis_workflows.provider_recording_timing_authority import (
    ProviderRecordingTimingAuthority,
    ProviderRecordingTimingAuthorityError,
    load_provider_recording_timing_authority,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
    load_subject_position_source_handle,
    require_subject_position_source_handle,
)

_POSITION_KIND_BY_MODALITY = {
    "detection": ProviderKind.DETECTION,
    "keypoint": ProviderKind.KEYPOINT,
    "subject_mask": ProviderKind.SUBJECT_MASK,
}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderAnalysisOfferError(f"{name} must be one exact object.")
    return value


def _recording_timing_authority(
    analysis_zarr_path: str | Path,
) -> ProviderRecordingTimingAuthority | None:
    try:
        return load_provider_recording_timing_authority(
            analysis_zarr_path,
            required=False,
            use_consolidated=True,
        )
    except ProviderRecordingTimingAuthorityError as exc:
        raise ProviderAnalysisOfferError(str(exc)) from exc


def position_provider_identity(
    value: SubjectPositionSourceHandle,
) -> ProviderIdentity:
    """Reopen one exact observation-position run and bind its offer identity."""

    supplied = require_subject_position_source_handle(value)
    current = load_subject_position_source_handle(
        supplied.analysis_zarr_path,
        supplied.run_path,
        expected_selector_eligible=supplied.selector_eligible,
        expected_manifest_sha256=supplied.manifest_sha256,
    )
    modality = current.estimator_record.get("source_modality")
    if modality not in _POSITION_KIND_BY_MODALITY:
        raise ProviderAnalysisOfferError(
            f"Unsupported subject-position modality: {modality!r}."
        )
    timing = _recording_timing_authority(current.analysis_zarr_path)
    if timing is not None:
        timing.validate_source_frame_indices(
            current.source_acquisition_frame_index[:],
            name="subject-position source_acquisition_frame_index",
        )
    return ProviderIdentity(
        role=ProviderRole.POSITION,
        kind=_POSITION_KIND_BY_MODALITY[str(modality)],
        modality=f"{modality}.v1",
        provider_id=str(current.estimator_record["estimator_id"]),
        run_path=current.run_path,
        recording_id=None if timing is None else timing.recording_id,
        manifest_sha256=current.manifest_sha256,
        decoded_content_sha256=current.decoded_content_sha256,
        coordinate_authority_sha256=current.coordinate_sha256,
        timing_authority_sha256=None if timing is None else timing.sha256,
        validity_array_names=("valid",),
    )


def body_frame_provider_identity(value: BodyFrameSourceHandle) -> ProviderIdentity:
    """Reopen one exact keypoint body-frame run and bind its offer identity."""

    supplied = require_body_frame_source_handle(value)
    current = load_body_frame_source_handle(
        supplied.analysis_zarr_path,
        run_path=supplied.run_path,
        expected_selector_eligible=supplied.selector_eligible,
    )
    if current.verification_digest != supplied.verification_digest:
        raise ProviderAnalysisOfferError(
            "Body-frame source changed after its offer input was sealed."
        )
    timing = _recording_timing_authority(current.analysis_zarr_path)
    if timing is not None:
        timing.validate_source_frame_indices(
            current.arrays["frame_indices"],
            name="body-frame frame_indices",
        )
    return ProviderIdentity(
        role=ProviderRole.BODY_FRAME,
        kind=ProviderKind.KEYPOINT,
        modality="keypoint_body_frame.v1",
        provider_id=current.recipe_id,
        run_path=current.run_path,
        recording_id=None if timing is None else timing.recording_id,
        manifest_sha256=str(current.run_manifest["payload_digest"]),
        decoded_content_sha256=current.verification_digest,
        coordinate_authority_sha256=current.recipe_digest,
        timing_authority_sha256=None if timing is None else timing.sha256,
        validity_array_names=("axis_valid",),
    )


def provider_motion_identity(
    value: ProviderTrackMotionSourceHandle,
) -> ProviderIdentity:
    """Bind one exact motion successor without upgrading compatibility timing."""

    current = require_provider_track_motion_source_handle(value)
    source = _mapping(
        current.source_authority_record,
        name="provider-motion source authority",
    )
    position = _mapping(
        source.get("position_source"),
        name="provider-motion position source",
    )
    timing = _recording_timing_authority(current.analysis_zarr_path)
    if timing is not None:
        timing.validate_source_frame_indices(
            current.source_acquisition_frame_index,
            name="provider-motion source_acquisition_frame_index",
        )
        parameters = _mapping(
            current.computation_record.get("parameters"),
            name="provider-motion computation parameters",
        )
        fps = parameters.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or float(fps) != timing.nominal_fps
        ):
            raise ProviderAnalysisOfferError(
                "Provider-motion FPS differs from the recording timing authority."
            )
    return ProviderIdentity(
        role=ProviderRole.MOTION,
        kind=ProviderKind.TRACK_MOTION,
        modality="track_motion.v1",
        provider_id=str(current.computation_record["computation_id"]),
        run_path=current.run_path,
        recording_id=None if timing is None else timing.recording_id,
        manifest_sha256=current.provider_manifest_sha256,
        decoded_content_sha256=current.verification_digest,
        coordinate_authority_sha256=str(position["coordinate_sha256"]),
        timing_authority_sha256=None if timing is None else timing.sha256,
        validity_array_names=(
            "position_source_valid",
            "body_frame_source_valid",
            "linear_sample_valid",
            "angular_sample_valid",
            "transition_valid",
        ),
    )


def temporal_selection_identity(
    value: ResolvedEpochSelection,
) -> TemporalSelectionIdentity:
    """Bind one loader-minted resolved v2 epoch projection."""

    if type(value) is not ResolvedEpochSelection:
        raise ProviderAnalysisOfferError(
            "A verified ResolvedEpochSelection is required."
        )
    value.assert_verified()
    recording_id = value.source_timeline_identity.get("recording_id")
    if type(recording_id) is not str or not recording_id:
        raise ProviderAnalysisOfferError(
            "Resolved epoch selection lacks its exact recording identity."
        )
    return TemporalSelectionIdentity(
        selection_id="stimulus_epoch_compatibility.v1",
        run_path=value.run_path,
        recording_id=recording_id,
        source_timeline_sha256=value.source_timeline_digest,
        resolved_sha256=value.selection_digest,
        timing_authority_sha256=value.recording_timing_authority_sha256,
    )


def build_provider_analysis_offer(
    *,
    analysis_class_id: str,
    analysis_class_version: int,
    computation_id: str,
    computation_version: int,
    temporal_selection: TemporalSelectionIdentity,
    provider_requirements: ProviderRequirements,
) -> AnalysisOffer:
    """Build one offer with readiness derived from exact authority gaps."""

    known_recordings = {
        provider.recording_id
        for provider in (
            provider_requirements.position,
            provider_requirements.body_frame,
            provider_requirements.motion,
        )
        if provider is not None and provider.recording_id is not None
    }
    if any(item != temporal_selection.recording_id for item in known_recordings):
        raise ProviderAnalysisOfferError(
            "Provider and temporal selection recording identities disagree."
        )
    known_timing_authorities = {
        provider.timing_authority_sha256
        for provider in (
            provider_requirements.position,
            provider_requirements.body_frame,
            provider_requirements.motion,
        )
        if provider is not None and provider.timing_authority_sha256 is not None
    }
    if len(known_timing_authorities) > 1:
        raise ProviderAnalysisOfferError("Provider timing authority digests disagree.")
    if (
        temporal_selection.timing_authority_sha256 is not None
        and known_timing_authorities
        and temporal_selection.timing_authority_sha256 not in known_timing_authorities
    ):
        raise ProviderAnalysisOfferError(
            "Provider and temporal selection timing authority digests disagree."
        )
    providers = tuple(
        provider
        for provider in (
            provider_requirements.position,
            provider_requirements.body_frame,
            provider_requirements.motion,
        )
        if provider is not None
    )
    if any(provider.recording_id is None for provider in providers):
        readiness = ScientificReadiness.BLOCKED_RECORDING_AUTHORITY
    elif temporal_selection.timing_authority_sha256 is None or any(
        provider.timing_authority_sha256 is None for provider in providers
    ):
        readiness = ScientificReadiness.BLOCKED_TEMPORAL_AUTHORITY
    else:
        readiness = ScientificReadiness.READY
    return AnalysisOffer(
        analysis_class_id=analysis_class_id,
        analysis_class_version=analysis_class_version,
        computation_id=computation_id,
        computation_version=computation_version,
        temporal_selection=temporal_selection,
        provider_requirements=provider_requirements,
        scientific_readiness=readiness,
    )


__all__ = [
    "body_frame_provider_identity",
    "build_provider_analysis_offer",
    "position_provider_identity",
    "provider_motion_identity",
    "temporal_selection_identity",
]
