from __future__ import annotations

from copy import deepcopy

import pytest

from fisheye.analysis_workflows.analysis_candidate_invocation import (
    CandidateInvocationContract,
    build_exact_tabular_invocation,
    build_eye_angle_invocation,
    build_stimulus_epoch_invocation,
    build_track_flat_invocation,
    require_candidate_invocation_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _rehash(value: dict[str, object]) -> None:
    value["payload_digest"] = canonical_json_sha256(value["payload"])


def test_exact_tabular_invocation_is_closed_and_digest_bound() -> None:
    invocation = build_exact_tabular_invocation(
        storage_profile_id="published_http_v1",
        copy_backend="python",
        keep_scratch=False,
    )
    require_candidate_invocation_manifest(
        invocation,
        expected_contract=CandidateInvocationContract.EXACT_TABULAR_V1,
        expected_profile_id="published_http_v1",
    )

    changed_backend = deepcopy(invocation)
    changed_backend["payload"]["parameters"]["copy_backend"] = "shell"
    _rehash(changed_backend)
    with pytest.raises(ValueError, match="copy_backend"):
        require_candidate_invocation_manifest(changed_backend)

    unexpected = deepcopy(invocation)
    unexpected["payload"]["parameters"]["apply"] = True
    _rehash(unexpected)
    with pytest.raises(ValueError, match="field set"):
        require_candidate_invocation_manifest(unexpected)

    wrong_bool = deepcopy(invocation)
    wrong_bool["payload"]["parameters"]["keep_scratch"] = 0
    _rehash(wrong_bool)
    with pytest.raises(TypeError, match="exact bool"):
        require_candidate_invocation_manifest(wrong_bool)


def test_track_flat_invocation_binds_authority_and_flat_projection() -> None:
    invocation = build_track_flat_invocation(
        source_motion_authority_sha256="a" * 64,
        storage_profile_id="published_http_v1",
        copy_backend="rsync",
        keep_scratch=False,
    )
    require_candidate_invocation_manifest(
        invocation,
        expected_contract="track_flat_v1",
        expected_profile_id="published_http_v1",
    )

    changed_authority = deepcopy(invocation)
    changed_authority["payload"]["parameters"][
        "source_motion_authority_sha256"
    ] = "not-a-digest"
    _rehash(changed_authority)
    with pytest.raises(ValueError, match="source_motion_authority_sha256"):
        require_candidate_invocation_manifest(changed_authority)

    changed_bundle = deepcopy(invocation)
    changed_bundle["payload"]["parameters"]["physical_bundle_mode"] = "included"
    _rehash(changed_bundle)
    with pytest.raises(ValueError, match="physical_bundle_mode"):
        require_candidate_invocation_manifest(changed_bundle)

    bool_version = deepcopy(invocation)
    bool_version["payload"]["parameters"]["source_schema_version"] = True
    _rehash(bool_version)
    with pytest.raises(ValueError, match="source_schema_version"):
        require_candidate_invocation_manifest(bool_version)

    online_scope = deepcopy(invocation)
    online_scope["payload"]["parameters"]["source_run_type"] = "online"
    _rehash(online_scope)
    with pytest.raises(ValueError, match="must be offline"):
        require_candidate_invocation_manifest(online_scope)


def test_eye_angle_invocation_binds_sources_compute_and_transfer() -> None:
    invocation = build_eye_angle_invocation(
        subject_shape_run="subject_shape_v4",
        keypoint_run="refined_keypoints_v2",
        storage_profile_id="eye_angle_access_aware_candidate_v1",
        chunk_rows=16_384,
        angle_chunk_rows=16_384,
        angle_chunk_columns=3,
        output_shard_rows=131_072,
        angle_shard_columns=3,
        execution_backend="serial_driver",
        scheduler="processes",
        num_workers=1,
        shard_workers=1,
        native_threads=1,
        fps=None,
        smoothing_window=None,
        copy_backend="python",
        keep_scratch=False,
        check_capacity=True,
    )
    require_candidate_invocation_manifest(
        invocation,
        expected_contract="eye_angles_v1",
        expected_profile_id="eye_angle_access_aware_candidate_v1",
    )
    assert invocation["payload"]["parameters"]["fps_source"] == (
        "authoritative_recording_metadata"
    )

    descendant = deepcopy(invocation)
    descendant["payload"]["parameters"]["subject_shape_run"] = "subject_shape_v4/arrays"
    _rehash(descendant)
    with pytest.raises(ValueError, match="run name"):
        require_candidate_invocation_manifest(descendant)

    wrong_backend = deepcopy(invocation)
    wrong_backend["payload"]["parameters"]["execution_backend"] = "dask_worker_chunks"
    _rehash(wrong_backend)
    with pytest.raises(ValueError, match="serial_driver"):
        require_candidate_invocation_manifest(wrong_backend)

    wrong_fps = deepcopy(invocation)
    wrong_fps["payload"]["parameters"]["fps_source"] = "explicit_override"
    wrong_fps["payload"]["parameters"]["fps"] = float("nan")
    wrong_fps["payload_digest"] = "0" * 64
    with pytest.raises(ValueError, match="strict JSON|payload digest"):
        require_candidate_invocation_manifest(wrong_fps)


def test_stimulus_epoch_invocation_binds_migration_and_staged_source() -> None:
    invocation = build_stimulus_epoch_invocation(
        source_stimulus_fingerprint="a" * 64,
        source_epoch_lineage_hash="b" * 64,
        storage_profile_id="published_http_v1",
        copy_backend="python",
        keep_scratch=False,
    )
    require_candidate_invocation_manifest(
        invocation,
        expected_contract=CandidateInvocationContract.STIMULUS_EPOCHS_V1,
        expected_profile_id="published_http_v1",
    )

    changed_source = deepcopy(invocation)
    changed_source["payload"]["parameters"]["source_schema_version"] = True
    _rehash(changed_source)
    with pytest.raises(ValueError, match="source schema identity"):
        require_candidate_invocation_manifest(changed_source)

    changed_fingerprint = deepcopy(invocation)
    changed_fingerprint["payload"]["parameters"][
        "source_stimulus_fingerprint"
    ] = "not-a-digest"
    _rehash(changed_fingerprint)
    with pytest.raises(ValueError, match="source_stimulus_fingerprint"):
        require_candidate_invocation_manifest(changed_fingerprint)

    changed_lineage = deepcopy(invocation)
    changed_lineage["payload"]["parameters"][
        "source_epoch_lineage_hash"
    ] = "not-a-digest"
    _rehash(changed_lineage)
    with pytest.raises(ValueError, match="source_epoch_lineage_hash"):
        require_candidate_invocation_manifest(changed_lineage)

    bypassed_staging = deepcopy(invocation)
    bypassed_staging["payload"]["parameters"]["source_staging_mode"] = "direct"
    _rehash(bypassed_staging)
    with pytest.raises(ValueError, match="source_staging_mode"):
        require_candidate_invocation_manifest(bypassed_staging)


def test_invocation_contract_and_profile_bindings_fail_closed() -> None:
    invocation = build_exact_tabular_invocation(
        storage_profile_id="published_http_v1",
        copy_backend="python",
        keep_scratch=False,
    )
    with pytest.raises(ValueError, match="contract differs"):
        require_candidate_invocation_manifest(
            invocation,
            expected_contract="track_flat_v1",
        )
    with pytest.raises(ValueError, match="storage profile differs"):
        require_candidate_invocation_manifest(
            invocation,
            expected_profile_id="scratch_compute_v1",
        )
