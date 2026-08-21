from __future__ import annotations

from dataclasses import replace
from copy import deepcopy

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
from fisheye.analysis_workflows.materializers.chaser_relative_frame import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    PARENT_PATH,
    materialize_chaser_relative_frame,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    ChaserRelativeFrameBodyUnavailableError,
    ChaserRelativeFrameSourceHandleError,
    load_chaser_relative_frame_source_handle,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_chaser_relative_frame_storage import (
    _context as _storage_context,
    _proxy_projection_record,
    _result_bound_to_proxy,
)


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


def _prepared(*, body: bool = True, timestamps: bool = True):
    n_frames = 3
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
    chaser_xy = fish_xy[:, None, :] + np.asarray([[[3.0, 4.0]]])
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
            chasers=ChaserObservations(
                identities=("chaser-1",),
                behavior_roles=np.full((n_frames, 1), "aggressive", dtype="<U16"),
                xy=chaser_xy,
                valid=np.ones((n_frames, 1), dtype=bool),
                source_row_index=np.arange(n_frames, dtype=np.int64)[:, None],
                authority=_authority("chaser-provider"),
                trial_ids=np.zeros((n_frames, 1), dtype=np.int64),
                active=np.ones((n_frames, 1), dtype=bool),
            ),
            selection_membership=np.ones(n_frames, dtype=bool),
            occurrence_membership=np.ones((n_frames, 1), dtype=bool),
            coordinate_policy=CoordinatePolicy(
                coordinate_authority_id="camera-native-v1",
                coordinate_frame="source_camera_continuous_pixel_xy",
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


def _publish(tmp_path, *, body: bool = True, timestamps: bool = True):
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording-1"
    root.require_group(PARENT_PATH)
    prepared = _prepared(body=body, timestamps=timestamps)
    materialize_chaser_relative_frame(
        archive,
        prepared=prepared,
        scratch_root=tmp_path / "scratch",
        run_name="candidate-v1",
        copy_backend="python",
        apply=True,
    )
    return archive


def _publish_proxy_bound(tmp_path, *, timestamps: bool = True):
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording-1"
    root.require_group(PARENT_PATH)
    context = _storage_context(
        acquisition_projection_record=_proxy_projection_record()
    )
    result = _result_bound_to_proxy(context)
    if not timestamps:
        result = replace(
            result,
            frame_keys=replace(result.frame_keys, timestamp_ns=None),
            timing_policy=replace(
                result.timing_policy,
                timestamp_field=None,
                policy_id="acquisition_frame_domain_without_camera_timestamps_v1",
            ),
        )
    prepared = prepare_chaser_relative_frame(
        result,
        context=context,
    )
    materialize_chaser_relative_frame(
        archive,
        prepared=prepared,
        scratch_root=tmp_path / "scratch",
        run_name="candidate-v1",
        copy_backend="python",
        apply=True,
    )
    return archive


def _run_attrs(archive, *, mode="r"):
    return open_zarr_root(archive, mode=mode, use_consolidated=False)[
        f"{PARENT_PATH}/candidate-v1"
    ]


def test_loads_sealed_body_candidate_and_zero_copy_reshapes(tmp_path):
    archive = _publish(tmp_path, body=True)

    handle = load_chaser_relative_frame_source_handle(
        archive,
        run_name="candidate-v1",
        expected_recording_id="recording-1",
    )

    assert handle.run_path == f"{PARENT_PATH}/candidate-v1"
    assert (handle.n_frames, handle.n_chasers, handle.n_rows) == (3, 1, 3)
    assert handle.body_available
    assert handle.base_array("acquisition_frame_id").flags.writeable is False
    reshaped = handle.base_frame_chaser("relative_vector_px_xy")
    assert reshaped.shape == (3, 1, 2)
    np.testing.assert_allclose(reshaped[:, 0], [[3, 4], [3, 4], [3, 4]])
    assert reshaped.base is not None
    assert handle.metadata_equivalence["subtree_path"] == handle.run_path
    handle.assert_current()


def test_position_only_handle_seals_base_and_rejects_body_access(tmp_path):
    archive = _publish(tmp_path, body=False)

    handle = load_chaser_relative_frame_source_handle(
        archive, run_name="candidate-v1", expected_recording_id="recording-1"
    )

    assert not handle.body_available
    with pytest.raises(ChaserRelativeFrameBodyUnavailableError, match="position-only"):
        handle.body_array("body_bearing_deg")
    with pytest.raises(ChaserRelativeFrameBodyUnavailableError):
        handle.body_frame_chaser("body_bearing_deg")


def test_proxy_bound_handle_preserves_exact_publication_authority(tmp_path):
    archive = _publish_proxy_bound(tmp_path)

    handle = load_chaser_relative_frame_source_handle(
        archive,
        run_name="candidate-v1",
        expected_recording_id="recording-1",
    )

    projection = handle.context["acquisition_projection"]["record"]
    publication = handle.context["acquisition_projection_publication"]["record"]
    assert projection["policy_id"] == (
        "latest_logged_cpu_state_per_input_acquisition_proxy_v1"
    )
    assert publication["run_path"] == (
        "analysis/chaser_input_provenance_proxy_runs/proxy_v1"
    )
    assert publication["selector_eligible"] is False
    assert handle.source_authorities["chaser_position"]["source_authority_id"] == (
        publication["run_path"]
    )
    handle.assert_current()


@pytest.mark.parametrize(
    "run_name",
    ["latest", "default", "analysis/candidate-v1", "../candidate-v1", "candidate/v1"],
)
def test_rejects_selector_aliases_and_paths(tmp_path, run_name):
    archive = _publish(tmp_path)

    with pytest.raises(ChaserRelativeFrameSourceHandleError):
        load_chaser_relative_frame_source_handle(archive, run_name=run_name)


def test_rejects_wrong_recording_and_payload_tamper(tmp_path):
    archive = _publish(tmp_path)
    with pytest.raises(ChaserRelativeFrameSourceHandleError, match="recording_id"):
        load_chaser_relative_frame_source_handle(
            archive, run_name="candidate-v1", expected_recording_id="other"
        )

    run = _run_attrs(archive, mode="a")
    values = np.asarray(run["base/relative_distance_px"][...])
    values[0] += 1
    run["base/relative_distance_px"][...] = values
    consolidate_metadata_capture_expected_warnings(archive)
    with pytest.raises(ChaserRelativeFrameSourceHandleError, match="content digest"):
        load_chaser_relative_frame_source_handle(archive, run_name="candidate-v1")


def test_rejects_stale_consolidated_metadata_and_reordered_declarations(tmp_path):
    archive = _publish(tmp_path)
    run = _run_attrs(archive, mode="a")
    values = np.asarray(run["base/relative_distance_px"][...])
    values[0] += 1
    run["base/relative_distance_px"][...] = values
    run.attrs["layout"] = "stale-layout"
    with pytest.raises(ChaserRelativeFrameSourceHandleError, match="consolidated"):
        load_chaser_relative_frame_source_handle(archive, run_name="candidate-v1")

    # Restore the payload, then make a validly re-consolidated but noncanonical
    # declaration order.  This proves the handle rejects reordering rather than
    # accepting it merely because all arrays are present.
    run["base/relative_distance_px"][...] = _prepared(body=True).base_arrays[
        "relative_distance_px"
    ]
    run.attrs["layout"] = "frame_x_chaser_sparse_rows_v1"
    manifest = deepcopy(dict(run.attrs[MANIFEST_ATTR]))
    manifest["array_declarations"] = list(reversed(manifest["array_declarations"]))
    payload = dict(manifest)
    payload.pop("payload_digest")
    manifest["payload_digest"] = canonical_json_sha256(payload)
    run.attrs[MANIFEST_ATTR] = manifest
    run.attrs[MANIFEST_DIGEST_ATTR] = canonical_json_sha256(manifest)
    provenance = deepcopy(dict(run.attrs["run_provenance"]))
    provenance["input_run_ids"]["prepared_chaser_relative_frame"] = manifest[
        "payload_digest"
    ]
    run.attrs["run_provenance"] = provenance
    consolidate_metadata_capture_expected_warnings(archive)
    with pytest.raises(ChaserRelativeFrameSourceHandleError, match="reordered"):
        load_chaser_relative_frame_source_handle(archive, run_name="candidate-v1")


def test_rejects_invalid_provenance(tmp_path):
    archive = _publish(tmp_path)
    # The unmodified materializer output is accepted before the negative
    # mutation below; this binds the test to the actual writer contract.
    accepted = load_chaser_relative_frame_source_handle(
        archive, run_name="candidate-v1", expected_recording_id="recording-1"
    )
    assert accepted.run_provenance["input_run_ids"]["prepared_chaser_relative_frame"] == (
        accepted.run_manifest["payload_digest"]
    )

    run = _run_attrs(archive, mode="a")
    provenance = deepcopy(dict(run.attrs["run_provenance"]))
    provenance["input_run_ids"]["prepared_chaser_relative_frame"] = "0" * 64
    run.attrs["run_provenance"] = provenance
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(ChaserRelativeFrameSourceHandleError, match="payload digest"):
        load_chaser_relative_frame_source_handle(archive, run_name="candidate-v1")
