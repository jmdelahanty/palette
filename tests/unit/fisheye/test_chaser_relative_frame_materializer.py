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
    ChaserRelativeFramePublicationContext,
    prepare_chaser_relative_frame,
)
from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    ChaserRelativeFrameValidationReceiptError,
    ensure_chaser_relative_frame_validation_receipt,
    load_chaser_relative_frame_targeted_source_handle,
)
from fisheye.analysis_workflows.materializers.chaser_relative_frame import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    PARENT_PATH,
    ChaserRelativeFrameMaterializationError,
    materialize_chaser_relative_frame,
    validate_chaser_relative_frame_run,
)
from fisheye.shared.zarr_io import open_zarr_root
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


def _record(schema_id: str, **values: object) -> dict[str, object]:
    return {"schema_id": schema_id, "schema_version": 1, **values}


def _context() -> ChaserRelativeFramePublicationContext:
    return ChaserRelativeFramePublicationContext(
        fish_identity="fish-1",
        subject_identity_record=_record(
            "palette.subject_identity",
            recording_id="recording-1",
            subject_id="fish-1",
        ),
        temporal_selection_record=_record(
            "palette.temporal_selection",
            recording_id="recording-1",
            selection_id="all-chaser-frames-v1",
        ),
        chaser_occurrence_record=_record(
            "palette.chaser_occurrence",
            recording_id="recording-1",
            occurrence_policy_id="logged-occurrence-v1",
        ),
        acquisition_projection_record=_record(
            "palette.acquisition_projection",
            recording_id="recording-1",
            policy_id="sealed-projection-v1",
        ),
        analysis_profile_record=_record(
            "palette.chaser_profile",
            profile_id="chaser_behavior_full_v3",
        ),
    )


def _prepared(*, body: bool) -> object:
    n_frames = 3
    keys = AcquisitionFrameKeys(
        recording_id="recording-1",
        acquisition_frame_id=np.asarray([10, 11, 12], dtype=np.int64),
        track_sample_id=np.asarray([20, 21, 22], dtype=np.int64),
        row_axis_authority_id="camera-rows-v1",
        row_axis_authority_digest="camera-rows-digest",
        timestamp_ns=np.asarray([100, 200, 300], dtype=np.int64),
    )
    fish_xy = np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.float64)
    chaser_xy = fish_xy[:, None, :] + np.asarray([[[3.0, 4.0]]])
    chaser = ChaserObservations(
        identities=("chaser-1",),
        behavior_roles=np.full((n_frames, 1), "aggressive", dtype="<U16"),
        xy=chaser_xy,
        valid=np.ones((n_frames, 1), dtype=bool),
        source_row_index=np.arange(n_frames, dtype=np.int64)[:, None],
        authority=_authority("chaser-provider"),
        trial_ids=np.zeros((n_frames, 1), dtype=np.int64),
        active=np.ones((n_frames, 1), dtype=bool),
    )
    body_input = None
    if body:
        body_input = BodyFrameInput(
            frame_keys=keys,
            origin_xy=fish_xy + np.asarray([1.0, 0.0]),
            forward_axis_xy=np.repeat([[1.0, 0.0]], n_frames, axis=0),
            left_axis_xy=np.repeat([[0.0, -1.0]], n_frames, axis=0),
            axis_valid=np.ones(n_frames, dtype=bool),
            source_row_index=np.arange(30, 33, dtype=np.int64),
            authority=_authority("body-provider"),
        )
    result = compute_chaser_relative_frame(
        ChaserRelativeFrameInput(
            frame_keys=keys,
            fish_xy=fish_xy,
            fish_valid=np.ones(n_frames, dtype=bool),
            fish_source_row_index=np.arange(n_frames, dtype=np.int64),
            fish_authority=_authority("fish-provider"),
            chasers=chaser,
            selection_membership=np.ones(n_frames, dtype=bool),
            occurrence_membership=np.ones((n_frames, 1), dtype=bool),
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
    )
    return prepare_chaser_relative_frame(result, context=_context())


def _archive(tmp_path, *, selector_value: str = "old-run"):
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording-1"
    parent = root.require_group(PARENT_PATH)
    for name in (
        "latest",
        "latest_complete",
        "latest_provider",
        "latest_any",
        "latest_materialized",
        "latest_composite",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
        "active_run",
        "active",
        "current_run",
        "current",
        "default_run",
        "default",
        "selected_run",
        "selected",
        "publication_generation",
        "publication_policy",
    ):
        parent.attrs[name] = f"{selector_value}-{name}"
    return archive


def _publish(tmp_path, *, body: bool = True, run_name: str = "candidate-v1"):
    prepared = _prepared(body=body)
    archive = _archive(tmp_path)
    result = materialize_chaser_relative_frame(
        archive,
        prepared=prepared,
        scratch_root=tmp_path / "scratch",
        run_name=run_name,
        copy_backend="python",
        apply=True,
    )
    return archive, prepared, result


def test_materializes_typed_base_and_body_arrays(tmp_path):
    archive, prepared, result = _publish(tmp_path, body=True)

    assert result["status"] == "complete"
    run = open_zarr_root(archive, mode="r", use_consolidated=True)[
        f"{PARENT_PATH}/candidate-v1"
    ]
    assert set(run.group_keys()) == {"base", "body"}
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["selector_eligible"] is False
    assert run.attrs["selection"] == "none"
    assert run.attrs[MANIFEST_DIGEST_ATTR] == canonical_json_sha256(
        dict(prepared.manifest)
    )
    assert run["base/acquisition_frame_id"].dtype == np.dtype("int64")
    assert run["body/body_bearing_deg"].dtype == np.dtype("float32")


def test_position_only_candidate_has_no_body_extension(tmp_path):
    archive, _prepared_value, _result = _publish(tmp_path, body=False)

    run = open_zarr_root(archive, mode="r", use_consolidated=True)[
        f"{PARENT_PATH}/candidate-v1"
    ]
    assert set(run.group_keys()) == {"base"}
    assert run.attrs["selection"] == "none"


def test_publication_does_not_update_parent_selectors(tmp_path):
    archive, _prepared_value, _result = _publish(tmp_path)

    parent = open_zarr_root(archive, mode="r", use_consolidated=False)[PARENT_PATH]
    for name in (
        "latest",
        "latest_complete",
        "latest_provider",
        "latest_any",
        "latest_materialized",
        "latest_composite",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
        "active_run",
        "active",
        "current_run",
        "current",
        "default_run",
        "default",
        "selected_run",
        "selected",
        "publication_generation",
        "publication_policy",
    ):
        assert parent.attrs[name] == f"old-run-{name}"


def test_existing_target_fails_closed(tmp_path):
    archive, prepared, _result = _publish(tmp_path)

    with pytest.raises(FileExistsError):
        materialize_chaser_relative_frame(
            archive,
            prepared=prepared,
            scratch_root=tmp_path / "second-scratch",
            run_name="candidate-v1",
            copy_backend="python",
            apply=True,
        )


def test_source_recording_identity_mismatch_fails_closed(tmp_path):
    prepared = _prepared(body=True)
    archive = _archive(tmp_path)
    root = open_zarr_root(archive, mode="a")
    root.attrs["recording_id"] = "different-recording"

    with pytest.raises(ChaserRelativeFrameMaterializationError, match="recording_id"):
        materialize_chaser_relative_frame(
            archive,
            prepared=prepared,
            scratch_root=tmp_path / "mismatch-scratch",
            run_name="candidate-v1",
            copy_backend="python",
            apply=True,
        )


def test_manifest_or_payload_tamper_fails_validation(tmp_path):
    archive, prepared, _result = _publish(tmp_path)
    run_path = archive / PARENT_PATH / "candidate-v1"
    root = open_zarr_root(run_path, mode="a")
    values = np.asarray(root["base/relative_distance_px"][...])
    values[0] += 1
    root["base/relative_distance_px"][...] = values
    validation = validate_chaser_relative_frame_run(
        run_path,
        expected_manifest=prepared.manifest,
    )
    assert validation["valid"] is False
    assert any("content digest mismatch" in error for error in validation["errors"])

    root.attrs[MANIFEST_ATTR] = {"tampered": True}
    manifest_validation = validate_chaser_relative_frame_run(
        run_path,
        expected_manifest=prepared.manifest,
    )
    assert manifest_validation["valid"] is False
    assert any("manifest" in error for error in manifest_validation["errors"])


def test_direct_and_consolidated_declarations_are_equivalent(tmp_path):
    archive, prepared, _result = _publish(tmp_path)
    run_path = archive / PARENT_PATH / "candidate-v1"

    direct = validate_chaser_relative_frame_run(
        run_path,
        expected_manifest=prepared.manifest,
        use_consolidated=False,
    )
    consolidated = validate_chaser_relative_frame_run(
        run_path,
        expected_manifest=prepared.manifest,
        use_consolidated=True,
    )
    assert direct["valid"] is True
    assert consolidated["valid"] is True
    assert direct["manifest_sha256"] == consolidated["manifest_sha256"]


def test_reusable_receipt_targets_arrays_without_archive_root_reparse(tmp_path):
    archive, prepared, _result = _publish(tmp_path, body=True)
    receipt_path = tmp_path / "receipts" / "relative.json"

    created = ensure_chaser_relative_frame_validation_receipt(
        archive,
        run_name="candidate-v1",
        palette_commit="a" * 40,
        output_json=receipt_path,
        expected_recording_id="recording-1",
    )
    reused = ensure_chaser_relative_frame_validation_receipt(
        archive,
        run_name="candidate-v1",
        palette_commit="a" * 40,
        output_json=receipt_path,
        expected_recording_id="recording-1",
    )
    # Reuse is deliberately independent of the archive-root consolidated
    # metadata document; only the sealed immutable child is reopened.
    (archive / "zarr.json").write_text("{not valid root metadata", encoding="utf-8")
    handle = load_chaser_relative_frame_targeted_source_handle(
        receipt_path,
        expected_analysis_zarr=archive,
        expected_recording_id="recording-1",
        expected_run_name="candidate-v1",
    )

    assert created["mode"] == "created"
    assert reused["mode"] == "reused_exact"
    assert created["manifest_sha256"] == canonical_json_sha256(
        created["run_manifest"]
    )
    assert created["validation_policy"]["archive_root_consolidated_metadata_reparse"] is False
    assert handle.verification_mode == "receipt_bound_targeted_array_rehash_v1"
    np.testing.assert_array_equal(
        handle.frame_array("acquisition_frame_id"),
        np.asarray([10, 11, 12], dtype=np.int64),
    )
    assert handle.receipt_digest == created["record_sha256"]

    run = open_zarr_root(
        archive / PARENT_PATH / "candidate-v1",
        mode="a",
        use_consolidated=False,
    )
    fish = np.asarray(run["base/fish_position_xy_px"][...])
    fish[0, 0] += 1.0
    run["base/fish_position_xy_px"][...] = fish
    with pytest.raises(
        ChaserRelativeFrameValidationReceiptError,
        match="content digest changed",
    ):
        load_chaser_relative_frame_targeted_source_handle(receipt_path)
