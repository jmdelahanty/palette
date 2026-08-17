from __future__ import annotations

from copy import deepcopy

import pytest

from fisheye.analysis_workflows.provider_analysis_offers import (
    AnalysisOffer,
    ProviderAnalysisOfferError,
    ProviderIdentity,
    ProviderKind,
    ProviderRequirements,
    ProviderRole,
    ScientificReadiness,
    TemporalSelectionIdentity,
    require_analysis_offer,
    require_provider_identity,
)


def _position_provider() -> ProviderIdentity:
    return ProviderIdentity(
        role=ProviderRole.POSITION,
        kind=ProviderKind.DETECTION,
        modality="detection.v1",
        provider_id="detection_bbox_centroid.v1",
        run_path="analysis/subject_position_runs/observation/position_canary_001",
        recording_id="recording-001",
        manifest_sha256="a" * 64,
        decoded_content_sha256="b" * 64,
        coordinate_authority_sha256="c" * 64,
        timing_authority_sha256="9" * 64,
        validity_array_names=("valid",),
    )


def _body_frame_provider() -> ProviderIdentity:
    return ProviderIdentity(
        role=ProviderRole.BODY_FRAME,
        kind=ProviderKind.KEYPOINT,
        modality="keypoint.v1",
        provider_id="keypoint_anatomical_triad.v1",
        run_path="analysis/body_frame_runs/body_frame_canary_001",
        recording_id="recording-001",
        manifest_sha256="d" * 64,
        decoded_content_sha256="e" * 64,
        coordinate_authority_sha256="f" * 64,
        timing_authority_sha256="9" * 64,
        validity_array_names=("axis_valid", "heading_valid"),
    )


def _temporal_selection() -> TemporalSelectionIdentity:
    return TemporalSelectionIdentity(
        selection_id="stimulus_epoch.v2",
        run_path="analysis/stimulus_epoch_runs/epoch_canary_001",
        recording_id="recording-001",
        source_timeline_sha256="2" * 64,
        resolved_sha256="1" * 64,
        timing_authority_sha256="9" * 64,
    )


def _offer(*, requirements: ProviderRequirements | None = None) -> AnalysisOffer:
    return AnalysisOffer(
        analysis_class_id="occupancy",
        analysis_class_version=1,
        computation_id="provider_occupancy",
        computation_version=2,
        temporal_selection=_temporal_selection(),
        provider_requirements=(
            ProviderRequirements(position=_position_provider())
            if requirements is None
            else requirements
        ),
        scientific_readiness=ScientificReadiness.READY,
    )


def test_offer_round_trips_as_strict_digest_bound_record() -> None:
    offer = _offer(
        requirements=ProviderRequirements(
            position=_position_provider(), body_frame=_body_frame_provider()
        )
    )

    envelope = offer.as_envelope()
    parsed = require_analysis_offer(envelope, expected_digest=offer.sha256)

    assert parsed == offer
    assert envelope["sha256"] == offer.sha256
    assert envelope["record"]["selector_eligible"] is False
    assert envelope["record"]["readiness"] == {"scientific": "ready"}
    assert envelope["record"]["provider_requirements"]["record"]["required_roles"] == [
        "position",
        "body_frame",
    ]


def test_digest_tampering_fails_without_rehashing() -> None:
    envelope = _offer().as_envelope()
    tampered = deepcopy(envelope)
    tampered["record"]["computation_version"] = 99

    with pytest.raises(ProviderAnalysisOfferError, match="digest"):
        require_analysis_offer(tampered)


def test_expected_digest_prevents_rehashed_tampering() -> None:
    offer = _offer()
    tampered = deepcopy(offer.as_envelope())
    tampered["record"]["computation_version"] = 99
    tampered["sha256"] = AnalysisOffer.from_record(tampered["record"]).sha256

    with pytest.raises(ProviderAnalysisOfferError, match="differs from expectation"):
        require_analysis_offer(tampered, expected_digest=offer.sha256)


@pytest.mark.parametrize(
    "run_path",
    [
        "analysis/subject_position_runs/observation/latest",
        "analysis/subject_position_runs/observation/latest_complete",
        "analysis/subject_position_runs/observation/../position_canary_001",
        "analysis/subject_position_runs/observation/position//canary",
        "analysis/subject_position_runs/observation/./position_canary_001",
        "../analysis/subject_position_runs/observation/position_canary_001",
    ],
)
def test_selector_alias_and_traversal_paths_fail_closed(run_path: str) -> None:
    with pytest.raises(
        ProviderAnalysisOfferError, match="canonical|selector|traversal"
    ):
        ProviderIdentity(
            role=ProviderRole.POSITION,
            kind=ProviderKind.DETECTION,
            modality="detection.v1",
            provider_id="detection_bbox_centroid.v1",
            run_path=run_path,
            recording_id="recording-001",
            manifest_sha256="a" * 64,
            decoded_content_sha256="b" * 64,
            coordinate_authority_sha256="c" * 64,
            timing_authority_sha256=None,
            validity_array_names=("valid",),
        )


def test_missing_validity_array_fails_closed() -> None:
    with pytest.raises(ProviderAnalysisOfferError, match="validity_array_names"):
        ProviderIdentity(
            role=ProviderRole.POSITION,
            kind=ProviderKind.DETECTION,
            modality="detection.v1",
            provider_id="detection_bbox_centroid.v1",
            run_path="analysis/subject_position_runs/observation/position_001",
            recording_id="recording-001",
            manifest_sha256="a" * 64,
            decoded_content_sha256="b" * 64,
            coordinate_authority_sha256="c" * 64,
            timing_authority_sha256=None,
            validity_array_names=(),
        )


@pytest.mark.parametrize(
    ("role", "kind"),
    [
        ("unknown", "detection"),
        ("position", "unknown"),
        (ProviderRole.POSITION, ProviderKind.TRACK_MOTION),
        (ProviderRole.BODY_FRAME, ProviderKind.DETECTION),
        (ProviderRole.MOTION, ProviderKind.KEYPOINT),
    ],
)
def test_unknown_or_mismatched_provider_role_and_kind_fail_closed(
    role: object, kind: object
) -> None:
    with pytest.raises(ProviderAnalysisOfferError, match="Unknown|incompatible"):
        ProviderIdentity(
            role=role,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            modality="provider.v1",
            provider_id="provider_recipe.v1",
            run_path="analysis/providers/position_001",
            recording_id="recording-001",
            manifest_sha256="a" * 64,
            decoded_content_sha256="b" * 64,
            coordinate_authority_sha256="c" * 64,
            timing_authority_sha256=("d" * 64 if role == ProviderRole.MOTION else None),
            validity_array_names=("valid",),
        )


def test_position_and_body_frame_requirements_are_independent() -> None:
    position_only = ProviderRequirements(position=_position_provider())
    body_only = ProviderRequirements(body_frame=_body_frame_provider())

    assert position_only.required_roles == ("position",)
    assert position_only.body_frame is None
    assert body_only.required_roles == ("body_frame",)
    assert body_only.position is None
    assert (
        _offer(requirements=position_only).record["provider_requirements"]["record"][
            "body_frame"
        ]
        is None
    )


def test_requirement_role_mismatch_and_empty_requirements_fail_closed() -> None:
    with pytest.raises(ProviderAnalysisOfferError, match="cannot satisfy"):
        ProviderRequirements(body_frame=_position_provider())  # type: ignore[arg-type]
    with pytest.raises(ProviderAnalysisOfferError, match="at least one"):
        ProviderRequirements()


def test_ready_motion_offer_requires_timing_but_blocked_offer_can_describe_gap() -> (
    None
):
    motion = ProviderIdentity(
        role=ProviderRole.MOTION,
        kind=ProviderKind.TRACK_MOTION,
        modality="track_motion.v1",
        provider_id="provider_track_motion.v1",
        run_path="analysis/track_kinematics_runs/provider/motion_001",
        recording_id="recording-001",
        manifest_sha256="a" * 64,
        decoded_content_sha256="b" * 64,
        coordinate_authority_sha256="c" * 64,
        timing_authority_sha256=None,
        validity_array_names=("linear_sample_valid", "angular_sample_valid"),
    )
    requirements = ProviderRequirements(motion=motion)
    with pytest.raises(ProviderAnalysisOfferError, match="timing authority"):
        AnalysisOffer(
            analysis_class_id="speed",
            analysis_class_version=1,
            computation_id="provider_speed",
            computation_version=1,
            temporal_selection=_temporal_selection(),
            provider_requirements=requirements,
            scientific_readiness=ScientificReadiness.READY,
        )
    blocked = AnalysisOffer(
        analysis_class_id="speed",
        analysis_class_version=1,
        computation_id="provider_speed",
        computation_version=1,
        temporal_selection=_temporal_selection(),
        provider_requirements=requirements,
        scientific_readiness=ScientificReadiness.BLOCKED_TEMPORAL_AUTHORITY,
    )
    assert blocked.scientific_readiness is (
        ScientificReadiness.BLOCKED_TEMPORAL_AUTHORITY
    )
    assert _position_provider().timing_authority_sha256 is not None


def test_existing_underscore_version_suffix_is_supported() -> None:
    provider = _body_frame_provider()
    provider = ProviderIdentity(
        **{
            **provider.__dict__,
            "provider_id": "keypoint_eye_midpoint_head_axis_camera_xy_v1",
        }
    )
    assert provider.provider_id.endswith("_v1")


def test_ready_offer_rejects_cross_recording_provider() -> None:
    provider = ProviderIdentity(
        **{
            **_position_provider().__dict__,
            "recording_id": "recording-002",
        }
    )
    with pytest.raises(ProviderAnalysisOfferError, match="recording identity"):
        _offer(requirements=ProviderRequirements(position=provider))


def test_offer_rejects_selector_eligibility_and_fallback_fields() -> None:
    record = _offer().record
    record["selector_eligible"] = True
    with pytest.raises(ProviderAnalysisOfferError, match="selector-ineligible"):
        AnalysisOffer.from_record(record)

    record = _offer().record
    record["fallback"] = "detection.v1"
    with pytest.raises(ProviderAnalysisOfferError, match="inexact field set"):
        AnalysisOffer.from_record(record)


def test_provider_identity_requires_lowercase_versioned_ids_and_digests() -> None:
    with pytest.raises(ProviderAnalysisOfferError, match="versioned identifier"):
        ProviderIdentity(
            role=ProviderRole.POSITION,
            kind=ProviderKind.DETECTION,
            modality="Detection.v1",
            provider_id="detection_bbox_centroid.v1",
            run_path="analysis/subject_position_runs/observation/position_001",
            recording_id="recording-001",
            manifest_sha256="a" * 64,
            decoded_content_sha256="b" * 64,
            coordinate_authority_sha256="c" * 64,
            timing_authority_sha256=None,
            validity_array_names=("valid",),
        )

    record = _position_provider().record
    record["manifest_sha256"] = "A" * 64
    with pytest.raises(ProviderAnalysisOfferError, match="lowercase SHA-256"):
        require_provider_identity(
            {"record": record, "sha256": _position_provider().sha256}
        )
