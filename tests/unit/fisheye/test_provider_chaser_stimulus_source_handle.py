from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr

from tests.unit.fisheye.test_provider_chaser_distance_candidates import _candidate
from fisheye.analysis.provider_chaser_distance_candidates import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    _materialize_local,
)
from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    select_chaser_input_provenance_proxy,
)
from fisheye.analysis_workflows.provider_chaser_stimulus_source_handle import (
    ProviderChaserStimulusSourceHandleError,
    _validate_native_layout,
    load_provider_chaser_stimulus_source_handle,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings


def _published_candidate(tmp_path: Path) -> tuple[Path, str]:
    candidate = _candidate(tmp_path)
    arrays = dict(candidate.arrays)
    arrays["samples/source_acquisition_frame_index"] = np.asarray(
        [0, 0, 1, 2], dtype=np.int64
    )
    arrays["samples/timestamp_ns"] = np.asarray(
        [0, 0, 100_000_000, 200_000_000], dtype=np.int64
    )
    for path in (
        "positions/source_position_run_row_index",
        "positions/source_position_source_row_index",
        "positions/source_position_instance_key",
        "positions/source_position_failure_reason_code",
        "positions/fish_position_source_camera_xy",
        "positions/fish_position_arena_xy",
        "positions/fish_valid",
    ):
        arrays[path] = np.array(arrays[path], copy=True)
        arrays[path][1] = arrays[path][0]
    fish_xy = arrays["positions/fish_position_arena_xy"]
    fish_valid = arrays["positions/fish_valid"]
    chaser_xy = arrays["positions/chaser_position_arena_xy"]
    chaser_valid = arrays["positions/chaser_valid"]
    distance_px = np.full(chaser_valid.shape, np.nan, dtype=np.float32)
    for column in range(chaser_xy.shape[1]):
        valid = fish_valid & chaser_valid[:, column]
        delta = chaser_xy[:, column, :] - fish_xy
        distance_px[valid, column] = np.linalg.norm(delta[valid], axis=1).astype(
            np.float32
        )
    arrays["distances/distance_px"] = distance_px
    arrays["distances/distance_mm"] = (
        distance_px / np.float32(candidate.pixels_per_mm_projector)
    ).astype(np.float32)
    nearest_index = np.full(fish_valid.shape, -1, dtype=np.int16)
    nearest_distance = np.full(fish_valid.shape, np.nan, dtype=np.float32)
    any_finite = np.isfinite(arrays["distances/distance_mm"]).any(axis=1)
    if np.any(any_finite):
        filled = np.where(
            np.isfinite(arrays["distances/distance_mm"]),
            arrays["distances/distance_mm"],
            np.inf,
        )
        nearest_column = np.argmin(filled[any_finite], axis=1)
        nearest_index[any_finite] = arrays["chasers/chaser_index"][nearest_column]
        nearest_distance[any_finite] = filled[
            any_finite, nearest_column
        ].astype(np.float32)
    arrays["distances/nearest_chaser_index"] = nearest_index
    arrays["distances/nearest_distance_mm"] = nearest_distance
    authority = {
        "schema_id": "palette.provider_chaser_distance_candidate_source_authority",
        "schema_version": 1,
        "recording_id": candidate.recording_id,
        "position": {
            "run_path": candidate.position_run_path,
            "manifest_sha256": "a" * 64,
            "decoded_content_sha256": "b" * 64,
            "estimator_id": candidate.position_estimator_id,
            "estimator_sha256": "c" * 64,
            "policy_sha256": "d" * 64,
            "source_sha256": "e" * 64,
            "anatomy_sha256": "f" * 64,
            "coordinate_sha256": "0" * 64,
            "source_camera_frame": {"frame_id": "camera-native"},
        },
        "stimulus": {
            "run_path": candidate.stimulus_run_path,
            "row_identity": {"record_ref": "rows", "record_sha256": "1" * 64},
            "temporal_authority": {"record_ref": "time", "record_sha256": "2" * 64},
            "surface_manifest": {"record_ref": "surface", "record_sha256": "3" * 64},
            "output_manifest": {"record_ref": "output", "record_sha256": "4" * 64},
            "transform_manifest": {"record_ref": "transform", "record_sha256": "5" * 64},
            "source_camera_frame": {"frame_id": "camera-native"},
        },
        "stimulus_epoch": {
            "run_path": candidate.stimulus_epoch_run_path,
            "schema_id": "palette.stimulus_epoch",
            "schema_version": 2,
            "manifest_sha256": "6" * 64,
            "metadata_equivalence": {"declarations_sha256": "7" * 64},
        },
        "acquisition_frame_authority": {"recording_id": candidate.recording_id},
        "total_frames": candidate.total_frames,
        "stimulus_sample_count": 4,
        "fps": candidate.fps,
        "fps_authority": {"recording_id": candidate.recording_id},
        "pixels_per_mm_projector": candidate.pixels_per_mm_projector,
        "temporal_join_policy": "preserve_unique_stimulus_frame_num_then_join_exact_source_acquisition_frame_index_v1",
        "numeric_transform": "source_camera_to_selected_canvas_then_inverse_arena_to_canvas_v1",
    }
    candidate = replace(candidate, arrays=arrays, source_authority=authority)
    archive = tmp_path / "candidate.zarr"
    _materialize_local(candidate, local_zarr=archive)
    root = zarr.open_group(str(archive), mode="a", zarr_format=3, use_consolidated=False)
    root.attrs["recording_id"] = candidate.recording_id
    consolidate_metadata_capture_expected_warnings(archive)
    return archive, candidate.run_name


def _rewrite_manifest(archive: Path, run_name: str, mutate) -> None:
    root = zarr.open_group(str(archive), mode="a", zarr_format=3, use_consolidated=False)
    run = root[f"analysis/provider_chaser_distance_candidate_runs/{run_name}"]
    manifest = dict(run.attrs[MANIFEST_ATTR])
    payload = dict(manifest["payload"])
    mutate(payload)
    manifest["payload"] = payload
    manifest["payload_digest"] = canonical_json_sha256(payload)
    run.attrs[MANIFEST_ATTR] = manifest
    run.attrs[MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]
    consolidate_metadata_capture_expected_warnings(archive)


def test_published_writer_contract_fixture_exposes_native_samples_without_camera_deduplication(
    tmp_path: Path,
) -> None:
    archive, run_name = _published_candidate(tmp_path)

    handle = load_provider_chaser_stimulus_source_handle(
        archive,
        run_name=run_name,
        expected_recording_id="recording-1",
    )

    assert handle.selector_eligible is False
    assert handle.run_path.endswith(f"/{run_name}")
    assert handle.dimensions.n_samples == 4
    assert handle.dimensions.n_chasers == 2
    np.testing.assert_array_equal(handle.stimulus_frame_num, [0, 1, 2, 3])
    # The source candidate is stimulus-sample evidence.  This assertion is
    # intentionally about preserving repeated acquisition lineage, not about
    # turning the samples into an acquisition-frame table.
    np.testing.assert_array_equal(handle.source_acquisition_frame_index, [0, 0, 1, 2])
    assert handle.chaser_position_arena_xy.shape == (4, 2, 2)
    assert handle.chaser_valid.shape == (4, 2)
    assert handle.source_stimulus_source_row_index.shape == (4, 2)
    assert handle.fish_source_position_run_row_index.shape == (4,)
    assert handle.source_stimulus_run_path.endswith("stimulus_v1")
    assert handle.metadata_equivalence["subtree_path"] == handle.run_path
    assert handle.stimulus_frame_num.flags.writeable is False
    assert handle.stimulus_frame_num.flags.c_contiguous is True
    with pytest.raises(ValueError):
        handle.stimulus_frame_num[0] = 10
    handle.assert_current()

    proxy = select_chaser_input_provenance_proxy(handle)
    np.testing.assert_array_equal(proxy.acquisition_frame_index, [0, 1, 2])
    np.testing.assert_array_equal(proxy.candidate_sample_count, [2, 1, 1])
    np.testing.assert_array_equal(proxy.selected_stimulus_frame_num, [1, 2, 3])
    assert proxy.source_manifest_sha256 == handle.manifest_sha256
    assert proxy.source_verification_digest == handle.verification_digest


@pytest.mark.parametrize(
    "run_name",
    [
        "latest",
        "latest_complete",
        "newest",
        "../candidate_v1",
        "analysis/provider_chaser_distance_candidate_runs/candidate_v1",
        "candidate_v1/child",
        "candidate_v1/",
        "/candidate_v1",
    ],
)
def test_bare_run_name_rejects_selectors_traversal_and_nested_paths(
    tmp_path: Path,
    run_name: str,
) -> None:
    archive, _ = _published_candidate(tmp_path)
    with pytest.raises(ProviderChaserStimulusSourceHandleError):
        load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)


def test_wrong_recording_identity_fails_closed(tmp_path: Path) -> None:
    archive, run_name = _published_candidate(tmp_path)
    root = zarr.open_group(str(archive), mode="a", zarr_format=3, use_consolidated=False)
    root.attrs["recording_id"] = "other-recording"
    consolidate_metadata_capture_expected_warnings(archive)
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="does not match"):
        load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)


def test_array_payload_tampering_is_rejected(tmp_path: Path) -> None:
    archive, run_name = _published_candidate(tmp_path)
    root = zarr.open_group(str(archive), mode="a", zarr_format=3, use_consolidated=False)
    run = root[f"analysis/provider_chaser_distance_candidate_runs/{run_name}"]
    run["samples/stimulus_frame_num"][0] = 99
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="content digest"):
        load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)


def test_stale_consolidated_metadata_is_rejected(tmp_path: Path) -> None:
    archive, run_name = _published_candidate(tmp_path)
    root = zarr.open_group(str(archive), mode="a", zarr_format=3, use_consolidated=False)
    run = root[f"analysis/provider_chaser_distance_candidate_runs/{run_name}"]
    run.attrs["row_axis"] = "acquisition_frames"
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="direct/consolidated"):
        load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)


def test_reordered_declarations_are_rejected(tmp_path: Path) -> None:
    archive, run_name = _published_candidate(tmp_path)

    def reorder(payload: dict) -> None:
        payload["arrays"] = list(reversed(payload["arrays"]))

    _rewrite_manifest(archive, run_name, reorder)
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="reordered"):
        load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)


def test_missing_or_extra_declarations_are_rejected(tmp_path: Path) -> None:
    archive, run_name = _published_candidate(tmp_path)

    def remove_required(payload: dict) -> None:
        payload["arrays"] = [
            item
            for item in payload["arrays"]
            if item["path"] != "samples/stimulus_frame_num"
        ]

    _rewrite_manifest(archive, run_name, remove_required)
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="omit required"):
        load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)

    archive2, run_name2 = _published_candidate(tmp_path / "extra")

    def add_extra(payload: dict) -> None:
        payload["arrays"].append(
            {
                "path": "samples/not_declared_by_schema",
                "dtype": "<i8",
                "shape": [1],
                "sha256": "0" * 64,
            }
        )

    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="array path"):
        _rewrite_manifest(archive2, run_name2, add_extra)
        load_provider_chaser_stimulus_source_handle(archive2, run_name=run_name2)


def test_parent_selector_attribute_is_rejected(tmp_path: Path) -> None:
    archive, run_name = _published_candidate(tmp_path)
    root = zarr.open_group(str(archive), mode="a", zarr_format=3, use_consolidated=False)
    parent = root["analysis/provider_chaser_distance_candidate_runs"]
    parent.attrs["latest"] = run_name
    consolidate_metadata_capture_expected_warnings(archive)
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="selector"):
        load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)


def test_assert_current_rejects_post_seal_mutation(tmp_path: Path) -> None:
    archive, run_name = _published_candidate(tmp_path)
    handle = load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)
    root = zarr.open_group(str(archive), mode="a", zarr_format=3, use_consolidated=False)
    root[f"analysis/provider_chaser_distance_candidate_runs/{run_name}"][
        "samples/timestamp_ns"
    ][0] = 42
    with pytest.raises(ProviderChaserStimulusSourceHandleError):
        handle.assert_current()


def test_native_lineage_rejects_wrong_validity_and_duplicate_source_rows(
    tmp_path: Path,
) -> None:
    archive, run_name = _published_candidate(tmp_path)
    handle = load_provider_chaser_stimulus_source_handle(archive, run_name=run_name)

    fish_bad = dict(handle.arrays)
    fish_bad["positions/fish_valid"] = handle.fish_valid.astype(np.uint8)
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="exact bool"):
        _validate_native_layout(
            fish_bad,
            total_frames=handle.dimensions.total_frames,
            authority=handle.authorities,
        )

    rows_bad = dict(handle.arrays)
    rows_bad["samples/source_stimulus_run_row_index"] = (
        handle.source_stimulus_run_row_index.copy()
    )
    rows_bad["samples/source_stimulus_run_row_index"].flat[1] = 0
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="not unique"):
        _validate_native_layout(
            rows_bad,
            total_frames=handle.dimensions.total_frames,
            authority=handle.authorities,
        )

    contradictory = dict(handle.arrays)
    contradictory["positions/source_position_source_row_index"] = (
        handle.fish_source_position_source_row_index.copy()
    )
    contradictory["positions/source_position_source_row_index"][1] = 999
    with pytest.raises(ProviderChaserStimulusSourceHandleError, match="contradictory"):
        _validate_native_layout(
            contradictory,
            total_frames=handle.dimensions.total_frames,
            authority=handle.authorities,
        )
