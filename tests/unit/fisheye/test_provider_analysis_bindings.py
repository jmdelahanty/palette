from __future__ import annotations

from dataclasses import replace

import pytest
import zarr

from fisheye.analysis_workflows.provider_analysis_bindings import (
    body_frame_provider_identity,
    build_provider_analysis_offer,
    position_provider_identity,
    provider_motion_identity,
    temporal_selection_identity,
)
from fisheye.analysis_workflows.provider_analysis_offers import (
    ProviderAnalysisOfferError,
    ProviderIdentity,
    ProviderKind,
    ProviderRequirements,
    ProviderRole,
    ScientificReadiness,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    load_provider_track_motion_source_handle,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    resolve_exact_stimulus_epoch_selection,
)
from tests.unit.fisheye.test_position_body_frame_motion import _handles
from tests.unit.fisheye.test_provider_track_motion_source_handle import (
    _publish_fixture,
)
from tests.unit.fisheye.test_provider_recording_timing_authority import (
    _install_clock_authority,
)

pytest_plugins = ("tests.unit.fisheye.test_stimulus_epoch_consumer",)


def test_exact_position_and_body_frame_handles_expose_current_authority_gaps(
    tmp_path,
) -> None:
    position, body_frame = _handles(tmp_path)

    position_identity = position_provider_identity(position)
    body_identity = body_frame_provider_identity(body_frame)

    assert position_identity.provider_id == "detection_bbox_centroid.v1"
    assert position_identity.recording_id is None
    assert position_identity.timing_authority_sha256 is None
    assert body_identity.provider_id.endswith("_v1")
    assert body_identity.recording_id is None
    assert body_identity.validity_array_names == ("axis_valid",)


def test_current_motion_handle_maps_to_blocked_not_authoritative_identity(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch)
    handle = load_provider_track_motion_source_handle(archive, plan.run_path)

    identity = provider_motion_identity(handle)

    assert identity.role is ProviderRole.MOTION
    assert identity.provider_id == "track_motion_provider_successor.v1"
    assert identity.recording_id is None
    assert identity.timing_authority_sha256 is None
    assert "linear_sample_valid" in identity.validity_array_names
    assert "angular_sample_valid" in identity.validity_array_names
    assert "transition_valid" in identity.validity_array_names


def test_existing_position_and_body_frame_runs_late_bind_same_authority(
    tmp_path,
) -> None:
    position, body_frame = _handles(tmp_path)
    _install_clock_authority(
        position.analysis_zarr_path,
        tmp_path,
        frame_count=3,
        fps=10.0,
    )

    position_identity = position_provider_identity(position)
    body_identity = body_frame_provider_identity(body_frame)

    assert position_identity.recording_id == "recording-001"
    assert body_identity.recording_id == "recording-001"
    assert position_identity.timing_authority_sha256
    assert (
        position_identity.timing_authority_sha256
        == body_identity.timing_authority_sha256
    )


def test_existing_motion_run_late_binds_matching_fps_and_rejects_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch)
    _install_clock_authority(
        archive,
        tmp_path,
        frame_count=3,
        fps=10.0,
    )
    matching = load_provider_track_motion_source_handle(archive, plan.run_path)

    identity = provider_motion_identity(matching)

    assert identity.recording_id == "recording-001"
    assert identity.timing_authority_sha256

    direct = zarr.open_group(
        str(archive), mode="a", zarr_format=3, use_consolidated=False
    )
    metadata = dict(direct.attrs["source_video_metadata"])
    metadata["fps"] = 11.0
    direct.attrs["source_video_metadata"] = metadata
    direct.attrs["fps"] = 11.0
    direct["raw_video"].attrs["fps"] = 11.0
    zarr.consolidate_metadata(str(archive))
    mismatched = load_provider_track_motion_source_handle(archive, plan.run_path)
    with pytest.raises(ProviderAnalysisOfferError, match="FPS differs"):
        provider_motion_identity(mismatched)


def test_exact_epoch_selection_maps_to_temporal_identity(
    published_candidate,
) -> None:
    resolved = resolve_exact_stimulus_epoch_selection(
        published_candidate,
        run_name="candidate",
    )

    identity = temporal_selection_identity(resolved)

    assert identity.run_path == resolved.run_path
    assert identity.recording_id == "recording_1"
    assert identity.source_timeline_sha256 == resolved.source_timeline_digest
    assert identity.resolved_sha256 == resolved.selection_digest


def _provider(*, recording_id: str | None, timing: str | None) -> ProviderIdentity:
    return ProviderIdentity(
        role=ProviderRole.POSITION,
        kind=ProviderKind.DETECTION,
        modality="detection.v1",
        provider_id="detection_bbox_centroid.v1",
        run_path="analysis/subject_position_runs/observation/position_001",
        recording_id=recording_id,
        manifest_sha256="a" * 64,
        decoded_content_sha256="b" * 64,
        coordinate_authority_sha256="c" * 64,
        timing_authority_sha256=timing,
        validity_array_names=("valid",),
    )


def _selection(published_candidate):  # type: ignore[no-untyped-def]
    return temporal_selection_identity(
        resolve_exact_stimulus_epoch_selection(
            published_candidate,
            run_name="candidate",
        )
    )


def test_offer_builder_derives_recording_then_timing_then_ready_states(
    published_candidate,
) -> None:
    selection = _selection(published_candidate)
    missing_recording = build_provider_analysis_offer(
        analysis_class_id="occupancy",
        analysis_class_version=2,
        computation_id="provider_occupancy",
        computation_version=1,
        temporal_selection=selection,
        provider_requirements=ProviderRequirements(
            position=_provider(recording_id=None, timing=None)
        ),
    )
    assert missing_recording.scientific_readiness is (
        ScientificReadiness.BLOCKED_RECORDING_AUTHORITY
    )

    missing_timing = build_provider_analysis_offer(
        analysis_class_id="occupancy",
        analysis_class_version=2,
        computation_id="provider_occupancy",
        computation_version=1,
        temporal_selection=selection,
        provider_requirements=ProviderRequirements(
            position=_provider(recording_id=selection.recording_id, timing=None)
        ),
    )
    assert missing_timing.scientific_readiness is (
        ScientificReadiness.BLOCKED_TEMPORAL_AUTHORITY
    )

    ready = build_provider_analysis_offer(
        analysis_class_id="occupancy",
        analysis_class_version=2,
        computation_id="provider_occupancy",
        computation_version=1,
        temporal_selection=replace(
            selection,
            timing_authority_sha256="d" * 64,
        ),
        provider_requirements=ProviderRequirements(
            position=_provider(
                recording_id=selection.recording_id,
                timing="d" * 64,
            )
        ),
    )
    assert ready.scientific_readiness is ScientificReadiness.READY


def test_offer_builder_rejects_cross_timing_authority(published_candidate) -> None:
    selection = replace(
        _selection(published_candidate),
        timing_authority_sha256="e" * 64,
    )
    with pytest.raises(ProviderAnalysisOfferError, match="timing authority digests"):
        build_provider_analysis_offer(
            analysis_class_id="occupancy",
            analysis_class_version=2,
            computation_id="provider_occupancy",
            computation_version=1,
            temporal_selection=selection,
            provider_requirements=ProviderRequirements(
                position=_provider(
                    recording_id=selection.recording_id,
                    timing="d" * 64,
                )
            ),
        )


def test_offer_builder_rejects_cross_recording_provider(published_candidate) -> None:
    selection = _selection(published_candidate)
    with pytest.raises(ProviderAnalysisOfferError, match="recording identities"):
        build_provider_analysis_offer(
            analysis_class_id="occupancy",
            analysis_class_version=2,
            computation_id="provider_occupancy",
            computation_version=1,
            temporal_selection=selection,
            provider_requirements=ProviderRequirements(
                position=_provider(recording_id="other-recording", timing="d" * 64)
            ),
        )
