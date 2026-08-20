from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_relative_frame import (
    AcquisitionFrameKeys,
    BodyFrameInput,
    ChaserObservations,
    ChaserRelativeFrameInput,
    CoordinatePolicy,
    ProviderSourceAuthority,
    ScalePolicy,
    TimingPolicy,
    compute_chaser_relative_frame,
)
from fisheye.analysis_workflows.chaser_relative_frame_storage import (
    MAX_CONTEXT_RECORD_BYTES,
    ChaserRelativeFramePublicationContext,
    ChaserRelativeFrameStorageError,
    prepare_chaser_relative_frame,
    validate_prepared_chaser_relative_frame,
)
from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    BEHAVIORAL_DENOMINATOR,
    PROJECTION_RECORD_SCHEMA_ID,
    PROJECTION_RECORD_SCHEMA_VERSION,
    PROXY_POLICY_ID,
    TEMPORAL_ALIGNMENT_CLASS,
    TEMPORAL_ALIGNMENT_REQUIREMENT,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _authority(provider_id: str) -> ProviderSourceAuthority:
    return ProviderSourceAuthority(
        recording_id="recording-1",
        source_authority_id=f"{provider_id}-source",
        source_digest=f"{provider_id}-source-digest",
        provider_id=provider_id,
        provider_digest=f"{provider_id}-digest",
        coordinate_authority_id="camera-native-v1",
        scale_authority_id="scale-v1",
        timing_authority_id="camera-time-v1",
        row_axis_authority_id="camera-rows-v1",
        row_axis_authority_digest="camera-rows-digest",
    )


def _result(*, body: bool = True, timestamps: bool = True, chasers: int = 2):
    n = 3
    keys = AcquisitionFrameKeys(
        recording_id="recording-1",
        acquisition_frame_id=np.asarray([10, 11, 12], dtype=np.int64),
        track_sample_id=np.asarray([20, 21, 22], dtype=np.int64),
        row_axis_authority_id="camera-rows-v1",
        row_axis_authority_digest="camera-rows-digest",
        timestamp_ns=(
            np.asarray([100, 200, 300], dtype=np.int64) if timestamps else None
        ),
    )
    fish_xy = np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.float64)
    chaser_xy = np.zeros((n, chasers, 2), dtype=np.float64)
    for column in range(chasers):
        chaser_xy[:, column] = fish_xy + np.asarray([column + 3, 4])
    roles = np.empty((n, chasers), dtype="<U16")
    for column in range(chasers):
        roles[:, column] = "aggressive" if column == 0 else "inert"
    body_input = None
    if body:
        body_input = BodyFrameInput(
            frame_keys=keys,
            origin_xy=fish_xy + np.asarray([1.0, 0.0]),
            forward_axis_xy=np.repeat([[1.0, 0.0]], n, axis=0),
            left_axis_xy=np.repeat([[0.0, -1.0]], n, axis=0),
            axis_valid=np.ones(n, dtype=bool),
            source_row_index=np.arange(30, 33, dtype=np.int64),
            authority=_authority("body-provider"),
        )
    inputs = ChaserRelativeFrameInput(
        frame_keys=keys,
        fish_xy=fish_xy,
        fish_valid=np.ones(n, dtype=bool),
        fish_source_row_index=np.arange(n, dtype=np.int64),
        fish_authority=_authority("fish-provider"),
        chasers=ChaserObservations(
            identities=tuple(f"chaser-{index}" for index in range(chasers)),
            behavior_roles=roles,
            xy=chaser_xy,
            valid=np.ones((n, chasers), dtype=bool),
            source_row_index=np.arange(n * chasers, dtype=np.int64).reshape(
                n, chasers
            ),
            authority=_authority("chaser-provider"),
            trial_ids=np.asarray(
                [[0] * chasers, [0] * chasers, [1] * chasers], dtype=np.int64
            ),
            active=np.ones((n, chasers), dtype=bool),
        ),
        selection_membership=np.asarray([True, True, False], dtype=bool),
        occurrence_membership=np.ones((n, chasers), dtype=bool),
        coordinate_policy=CoordinatePolicy(
            coordinate_authority_id="camera-native-v1",
            coordinate_frame="source_camera_pixels",
        ),
        scale_policy=ScalePolicy(
            scale_authority_id="scale-v1",
            scale_digest="scale-digest",
            pixels_per_unit=10.0,
        ),
        timing_policy=TimingPolicy(
            timing_authority_id="camera-time-v1",
            timing_digest="camera-time-digest",
            recording_id="recording-1",
        ),
        body_frame=body_input,
    )
    return compute_chaser_relative_frame(inputs)


def _record(schema_id: str, **values: object) -> dict[str, object]:
    return {"schema_id": schema_id, "schema_version": 1, **values}


def _proxy_projection_record(**replacements: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_id": PROJECTION_RECORD_SCHEMA_ID,
        "schema_version": PROJECTION_RECORD_SCHEMA_VERSION,
        "recording_id": "recording-1",
        "policy_id": PROXY_POLICY_ID,
        "temporal_alignment_requirement": TEMPORAL_ALIGNMENT_REQUIREMENT,
        "temporal_alignment_class": TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": False,
        "presentation_timestamp_available": False,
        "camera_presentation_clock_transform_available": False,
        "camera_exposure_reference": "unknown",
        "scientific_use_class": "exploratory_proxy",
        "behavioral_denominator": BEHAVIORAL_DENOMINATOR,
        "native_sample_axis": "stimulus_samples",
        "native_sample_rows_preserved": True,
        "source_acquisition_frame_field": "source_acquisition_frame_index",
        "selection_order": [
            "timestamp_ns_session",
            "stimulus_frame_num",
            "source_stimulus_run_row_index",
            "source_stimulus_source_row_index",
            "source_sample_row_index",
        ],
        "complete_sample_rule": (
            "all_declared_chasers_valid_and_finite_in_one_native_sample"
        ),
        "missing_frame_rule": "no_carry_forward",
        "native_sample_count": 5,
        "unique_acquisition_frame_count": 3,
        "selected_acquisition_frame_count": 3,
        "chaser_count": 2,
        "candidate_sample_row_index_is_zero_based": True,
        "source_authority_id": "native-source-authority-v1",
        "source_authority_digest": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "source_verification_digest": "c" * 64,
        "source_run_path": "analysis/provider_chaser_distance_candidate_runs/native",
    }
    record.update(replacements)
    return record


def _proxy_publication_binding(
    projection: dict[str, object],
    **replacements: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_id": "palette.chaser_input_provenance_proxy_publication_binding",
        "schema_version": 1,
        "recording_id": projection["recording_id"],
        "run_path": "analysis/chaser_input_provenance_proxy_runs/proxy_v1",
        "manifest_sha256": "d" * 64,
        "acquisition_projection_record_sha256": canonical_json_sha256(projection),
        "policy_id": projection["policy_id"],
        "temporal_alignment_class": projection["temporal_alignment_class"],
        "source_run_path": projection["source_run_path"],
        "source_manifest_sha256": projection["source_manifest_sha256"],
        "source_verification_digest": projection["source_verification_digest"],
        "n_frames": projection["unique_acquisition_frame_count"],
        "n_candidates": projection["native_sample_count"],
        "n_chasers": projection["chaser_count"],
        "selector_eligible": False,
        "selection": "none",
    }
    record.update(replacements)
    return record


def _context(**replacements: object) -> ChaserRelativeFramePublicationContext:
    values: dict[str, object] = {
        "fish_identity": "fish-1",
        "subject_identity_record": _record(
            "palette.subject_identity",
            recording_id="recording-1",
            subject_id="fish-1",
            identity_policy_id="acquisition_subject_uuid_v1",
        ),
        "temporal_selection_record": _record(
            "palette.resolved_temporal_selection",
            recording_id="recording-1",
            selection_id="all_chaser_epochs_v1",
            intervals_sha256="selection-digest",
        ),
        "chaser_occurrence_record": _record(
            "palette.chaser_occurrence_projection",
            recording_id="recording-1",
            occurrence_policy_id="logged_chaser_occurrence_v1",
            source_sha256="occurrence-digest",
        ),
        "acquisition_projection_record": _record(
            "palette.chaser_acquisition_projection",
            recording_id="recording-1",
            policy_id="sealed_fixture_projection_v1",
            source_sample_sha256="sample-digest",
            contributor_arrays=["source_sample_start", "source_sample_stop"],
        ),
        "analysis_profile_record": _record(
            "palette.chaser_analysis_profile",
            profile_id="chaser_behavior_full_v3",
            profile_digest="profile-digest",
        ),
    }
    values.update(replacements)
    projection = values["acquisition_projection_record"]
    if (
        isinstance(projection, dict)
        and projection.get("policy_id") == PROXY_POLICY_ID
        and "acquisition_projection_publication_record" not in replacements
    ):
        values["acquisition_projection_publication_record"] = (
            _proxy_publication_binding(projection)
        )
    return ChaserRelativeFramePublicationContext(**values)


def _result_bound_to_proxy(
    context: ChaserRelativeFramePublicationContext,
):
    result = _result()
    projection = context.acquisition_projection_record
    publication = context.acquisition_projection_publication_record
    assert publication is not None
    authority = replace(
        result.chaser_authority,
        source_authority_id=publication["run_path"],
        source_digest=publication["manifest_sha256"],
        provider_id=projection["policy_id"],
        provider_digest=publication["acquisition_projection_record_sha256"],
    )
    return replace(result, chaser_authority=authority)


def test_prepare_flattens_frame_major_and_validates_body_extension() -> None:
    prepared = prepare_chaser_relative_frame(_result(), context=_context())

    assert prepared.dimensions.n_rows == 6
    assert prepared.base_arrays["acquisition_frame_id"].tolist() == [
        10,
        10,
        11,
        11,
        12,
        12,
    ]
    assert prepared.base_arrays["chaser_identity_code"].tolist() == [1, 2] * 3
    assert prepared.base_arrays["selection_member"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert prepared.body_arrays is not None
    np.testing.assert_allclose(
        prepared.body_arrays["body_relative_vector_px_xy"][:2],
        [[2, 4], [3, 4]],
    )
    assert prepared.manifest["selector_eligible"] is False
    assert prepared.manifest["selection"] == "none"
    assert prepared.manifest["metadata_policy"]["row_evidence"] == (
        "typed_arrays_only"
    )
    assert not prepared.base_arrays["relative_distance_px"].flags.writeable
    assert all(
        array.dtype.kind not in {"O", "U", "S"}
        for array in list(prepared.base_arrays.values())
        + list(prepared.body_arrays.values())
    )


def test_proxy_projection_context_preserves_explicit_caveats_and_source_binding() -> None:
    context = _context(acquisition_projection_record=_proxy_projection_record())
    prepared = prepare_chaser_relative_frame(
        _result_bound_to_proxy(context), context=context
    )
    projection = prepared.manifest["context"]["acquisition_projection"]["record"]

    assert projection["temporal_alignment_requirement"] == (
        "input_provenance_proxy_allowed"
    )
    assert projection["physical_presentation_verified"] is False
    assert projection["behavioral_denominator"] == "unique_input_acquisition_frames"
    assert projection["missing_frame_rule"] == "no_carry_forward"
    publication = prepared.manifest["context"][
        "acquisition_projection_publication"
    ]["record"]
    assert publication["run_path"].endswith("/proxy_v1")
    assert publication["acquisition_projection_record_sha256"] == (
        canonical_json_sha256(projection)
    )


def test_proxy_projection_context_requires_exact_published_run_binding() -> None:
    projection = _proxy_projection_record()
    with pytest.raises(ChaserRelativeFrameStorageError, match="published proxy"):
        _context(
            acquisition_projection_record=projection,
            acquisition_projection_publication_record=None,
        )
    with pytest.raises(ChaserRelativeFrameStorageError, match="manifest_sha256"):
        _context(
            acquisition_projection_record=projection,
            acquisition_projection_publication_record=_proxy_publication_binding(
                projection,
                manifest_sha256="not-a-digest",
            ),
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"physical_presentation_verified": True}, "physical_presentation_verified"),
        ({"behavioral_denominator": "native_samples"}, "behavioral_denominator"),
        ({"selected_acquisition_frame_count": 4}, "counts"),
        ({"source_verification_digest": "not-a-digest"}, "source_verification_digest"),
    ],
)
def test_proxy_projection_context_rejects_semantic_or_source_tampering(
    replacement: dict[str, object], message: str
) -> None:
    with pytest.raises(ChaserRelativeFrameStorageError, match=message):
        _context(
            acquisition_projection_record=_proxy_projection_record(**replacement)
        )


@pytest.mark.parametrize(
    "replacement",
    [
        {
            "unique_acquisition_frame_count": 2,
            "selected_acquisition_frame_count": 2,
        },
        {"selected_acquisition_frame_count": 2},
        {"chaser_count": 1},
    ],
)
def test_proxy_projection_must_match_relative_frame_axes_and_complete_rows(
    replacement: dict[str, object],
) -> None:
    context = _context(
        acquisition_projection_record=_proxy_projection_record(**replacement)
    )
    with pytest.raises(ChaserRelativeFrameStorageError, match="does not match"):
        prepare_chaser_relative_frame(
            _result_bound_to_proxy(context), context=context
        )


def test_proxy_projection_rejects_unbound_chaser_authority() -> None:
    context = _context(acquisition_projection_record=_proxy_projection_record())
    with pytest.raises(ChaserRelativeFrameStorageError, match="chaser authority"):
        prepare_chaser_relative_frame(_result(), context=context)


def test_position_only_result_omits_body_extension() -> None:
    prepared = prepare_chaser_relative_frame(
        _result(body=False), context=_context()
    )

    assert prepared.body_arrays is None
    assert prepared.manifest["schema_binding"]["body_extension_present"] is False


def test_missing_timestamps_are_explicit_not_fabricated() -> None:
    prepared = prepare_chaser_relative_frame(
        _result(timestamps=False), context=_context()
    )

    assert (prepared.base_arrays["timestamp_ns"] == -1).all()
    assert not prepared.base_arrays["timestamp_valid"].any()
    registry = prepared.manifest["reason_codes"]
    reasons = {
        registry[str(code)]
        for code in prepared.base_arrays["timestamp_reason_code"].tolist()
    }
    assert reasons == {"timestamp_unavailable"}


def test_manifest_is_deterministic_and_binds_array_content() -> None:
    first = prepare_chaser_relative_frame(_result(), context=_context())
    second = prepare_chaser_relative_frame(_result(), context=_context())
    changed = prepare_chaser_relative_frame(
        _result(),
        context=_context(
            analysis_profile_record=_record(
                "palette.chaser_analysis_profile",
                profile_id="chaser_behavior_full_v3",
                profile_digest="different-profile-digest",
            )
        ),
    )

    assert first.payload_digest == second.payload_digest
    assert changed.payload_digest != first.payload_digest
    declarations = first.manifest["array_declarations"]
    assert {item["path"] for item in declarations} == {
        *(f"base/{path}" for path in first.base_arrays),
        *(f"body/{path}" for path in first.body_arrays or {}),
    }
    assert all(len(item["content_sha256"]) == 64 for item in declarations)


def test_recording_mismatch_fails_closed() -> None:
    bad = _record(
        "palette.resolved_temporal_selection",
        recording_id="other-recording",
        selection_id="all_chaser_epochs_v1",
    )

    with pytest.raises(ChaserRelativeFrameStorageError, match="recording_id"):
        prepare_chaser_relative_frame(
            _result(), context=_context(temporal_selection_record=bad)
        )


def test_valid_position_without_source_row_fails_closed() -> None:
    result = _result()
    source_rows = result.fish_source_row_index.copy()
    source_rows[1] = -1
    broken = replace(result, fish_source_row_index=source_rows)

    with pytest.raises(ChaserRelativeFrameStorageError, match="source-row"):
        prepare_chaser_relative_frame(broken, context=_context())


def test_empty_chaser_axis_is_inapplicable_not_successful_zero_rows() -> None:
    with pytest.raises(ChaserRelativeFrameStorageError, match="non-empty chaser"):
        prepare_chaser_relative_frame(
            _result(chasers=0, body=False), context=_context()
        )


def test_context_records_are_readable_and_bounded() -> None:
    huge = _record(
        "palette.resolved_temporal_selection",
        recording_id="recording-1",
        selection_id="all_chaser_epochs_v1",
        accidental_row_payload="x" * MAX_CONTEXT_RECORD_BYTES,
    )

    with pytest.raises(ChaserRelativeFrameStorageError, match="bounded metadata"):
        _context(temporal_selection_record=huge)


def test_context_requires_readable_profile_and_projection_policy() -> None:
    with pytest.raises(ChaserRelativeFrameStorageError, match="profile_id"):
        _context(
            analysis_profile_record=_record(
                "palette.chaser_analysis_profile", profile_digest="opaque-only"
            )
        )
    with pytest.raises(ChaserRelativeFrameStorageError, match="policy_id"):
        _context(
            acquisition_projection_record=_record(
                "palette.chaser_acquisition_projection",
                recording_id="recording-1",
                source_sample_sha256="opaque-only",
            )
        )


def test_arena_geometry_and_transform_bind_as_one_optional_capability() -> None:
    geometry = _record(
        "palette.arena_geometry",
        recording_id="recording-1",
        geometry_id="registered-arena-v1",
    )
    transform = _record(
        "palette.arena_to_source_camera_transform",
        recording_id="recording-1",
        transform_id="arena-to-source-camera-v1",
    )
    prepared = prepare_chaser_relative_frame(
        _result(),
        context=_context(
            arena_geometry_record=geometry,
            arena_to_source_camera_transform_record=transform,
        ),
    )
    assert prepared.manifest["context"]["arena_geometry"]["record"] == geometry
    assert (
        prepared.manifest["context"]["arena_to_source_camera_transform"]["record"]
        == transform
    )

    with pytest.raises(ChaserRelativeFrameStorageError, match="supplied together"):
        _context(arena_geometry_record=geometry)


def test_prepared_validator_rejects_stale_manifest_and_array_content() -> None:
    prepared = prepare_chaser_relative_frame(_result(), context=_context())
    receipt = validate_prepared_chaser_relative_frame(prepared)
    assert receipt["n_rows"] == 6
    assert receipt["selector_eligible"] is False

    stale_manifest = dict(prepared.manifest)
    stale_manifest["candidate_state"] = "tampered"
    with pytest.raises(ChaserRelativeFrameStorageError, match="payload_digest"):
        validate_prepared_chaser_relative_frame(
            replace(prepared, manifest=stale_manifest)
        )

    changed_arrays = dict(prepared.base_arrays)
    values = changed_arrays["relative_distance_px"].copy()
    values[0] += 1.0
    changed_arrays["relative_distance_px"] = values
    with pytest.raises(ChaserRelativeFrameStorageError):
        validate_prepared_chaser_relative_frame(
            replace(prepared, base_arrays=changed_arrays)
        )
