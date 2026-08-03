from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import zarr

from fisheye.analysis import track_kinematics as track_mod
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.analysis_workflows.materializers.track_kinematics_candidate import (
    materialize_track_kinematics_flat_candidate,
)
from fisheye.diagnostics import (
    benchmark_track_kinematics_v2_candidate as benchmark,
)
from fisheye.shared.coordinate_identity import (
    resolve_source_acquisition_frame_indices,
)
from fisheye.shared.observation_coordinate_publication import (
    load_persisted_detection_observation_geometry,
    load_persisted_source_camera_position_surface,
    publish_crop_observation_geometry,
)
from tests.unit.fisheye.test_detect_yolo_sharding import (
    _complete_canonical_detection_observation,
)

SOURCE_RUN_NAME = "sealed_source_v1"
SOURCE_RUN_PATH = f"analysis/track_kinematics_runs/offline/{SOURCE_RUN_NAME}"


def _canonical_parameters() -> dict[str, object]:
    return {
        "fps": 10.0,
        "smoothing_seconds": 0.1,
        "smoothing_method": "moving_average",
        "smoothing_alignment": "centered",
        "savgol_polyorder": None,
        "distance_interpolation_seconds": 0.0,
        "coordinate_space": "source_camera_image_px",
        "hysteresis_enabled": False,
        "hysteresis_high_px": None,
        "hysteresis_low_px": None,
        "hysteresis_min_frames": None,
        "hysteresis_band_policy": "reset",
    }


def _build_canonical_sealed_source(tmp_path: Path) -> Path:
    """Build one genuine public real-Zarr track-motion source fixture."""

    root, detection_run, _declaration = _complete_canonical_detection_observation(
        tmp_path,
        row_count=2,
    )
    detection = load_persisted_detection_observation_geometry(
        root,
        "detect_runs/nonempty",
    )

    crop = root.require_group("crop_runs").create_group("canonical_track_source")
    source_rows = np.asarray([0, 1], dtype=np.int64)
    crop.create_array("detection_indices", data=source_rows)
    crop.create_array(
        "instance_key",
        data=np.asarray(detection_run["instance_key"][:])[source_rows],
    )
    crop.create_array(
        "source_acquisition_frame_index",
        data=np.asarray(detection_run["source_acquisition_frame_index"][:])[
            source_rows
        ],
    )
    for name in ("bbox_norm_coords", "bbox_img_xyxy", "centers_img_xy"):
        crop.create_array(name, data=np.asarray(detection_run[name][:])[source_rows])
    publish_crop_observation_geometry(
        crop,
        crop["instance_key"],
        crop["detection_indices"],
        crop["source_acquisition_frame_index"],
        crop["bbox_norm_coords"],
        crop["bbox_img_xyxy"],
        crop["centers_img_xy"],
        source_geometry=detection,
    )
    crop.attrs.update(
        {
            "coordinate_contract": "canonical_v2",
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    position = load_persisted_source_camera_position_surface(
        root,
        "crop_runs/canonical_track_source",
    )

    keypoints = root.require_group("keypoints_runs").create_group("kp_1")
    heading = keypoints.create_array(
        "heading",
        data=np.asarray([0.0, 10.0], dtype=np.float32),
    )
    heading_usable = keypoints.create_array(
        "heading_usable",
        data=np.ones(2, dtype=bool),
    )
    keypoint_key = keypoints.create_array(
        "instance_key",
        data=np.asarray(crop["instance_key"][:]),
    )

    tracking = root.require_group("tracking_runs").create_group("trk_1")
    tracking.create_array("track_ids", data=np.asarray([7, 7], dtype=np.int32))
    tracking.create_array("arena_ids", data=np.asarray([3, 3], dtype=np.int32))
    tracking.create_array(
        "instance_key",
        data=np.asarray(crop["instance_key"][:]),
    )
    tracking.create_array("track_ids_present", data=np.asarray([7], dtype=np.int32))
    tracking.create_array("track_arena_ids", data=np.asarray([3], dtype=np.int32))

    input_authority = track_mod.build_track_motion_input_authority(
        root,
        source_positions=position.coordinates,
        mode="offline_exact_sources_v1",
        heading_node=heading,
        keypoint_usability_node=heading_usable,
        keypoint_row_key_node=keypoint_key,
        tracking_group=tracking,
    )
    source_rows = np.asarray([0, 1], dtype=np.int64)
    frames = resolve_source_acquisition_frame_indices(
        position.temporal_authority,
        source_rows,
    )
    tracks, summaries = track_mod.build_track_datasets(
        track_ids=np.asarray([7, 7], dtype=np.int64),
        frames=frames,
        positions_px=np.asarray(position.coordinates.coordinate_node[:]),
        headings_deg=np.asarray(heading[:]),
        keypoint_success=np.asarray(heading_usable[:]),
        detection_source=None,
        fps=10.0,
        smooth_seconds=0.1,
        pixel_to_mm=None,
        smoothing_alignment="centered",
        source_row_index=source_rows,
        source_temporal_authority=position.temporal_authority,
    )

    parent = root.require_group("analysis").require_group("track_kinematics_runs")
    offline = parent.require_group("offline")
    run = offline.create_group(SOURCE_RUN_NAME)
    run.attrs[track_mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR] = str(uuid.uuid4())
    track_mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=position.temporal_authority,
        positions_px_source=position.coordinates,
        input_authority=input_authority,
        track_id_to_arena_id={7: 3},
    )
    parameters = _canonical_parameters()
    inputs = {
        "detection_path": "detect_runs/nonempty",
        "position_source_path": "crop_runs/canonical_track_source/centers_img_xy",
        "position_source_rowset_path": "crop_runs/canonical_track_source",
        "position_source_kind": "canonical_crop_rows_source_camera_centers",
        "keypoint_path": "keypoints_runs/kp_1",
        "crop_run": "canonical_track_source",
        "tracking_path": "tracking_runs/trk_1",
    }
    run.attrs.update(
        track_mod._track_kinematics_contract_attrs(  # noqa: SLF001
            run_type="offline",
            method="track_kinematics_offline",
            parameters=parameters,
            inputs=inputs,
        )
    )
    run.attrs.update(
        {
            "inputs": copy.deepcopy(inputs),
            "fps": 10.0,
            "smoothing_seconds": 0.1,
            "smoothing_method": "moving_average",
            "smoothing_alignment": "centered",
            "savgol_polyorder": None,
            "distance_interpolation_seconds": 0.0,
            "hysteresis_enabled": False,
            "hysteresis_high_px": None,
            "hysteresis_low_px": None,
            "hysteresis_min_frames": None,
            "hysteresis_band_policy": "reset",
            "provenance": {
                "stage": "track_kinematics",
                "parameters": copy.deepcopy(parameters),
                "inputs": copy.deepcopy(inputs),
            },
            "run_provenance": {
                "schema": "palette.run_provenance.v1",
                "git_sha": "a" * 40,
                "config_hash": track_mod.sha256_payload(parameters),
                "params": copy.deepcopy(parameters),
                "input_run_ids": copy.deepcopy(inputs),
                "command": "test_benchmark_track_kinematics_v2_candidate",
                "fisheye_version": None,
            },
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    track_mod._seal_and_load_track_motion_run_before_selection(
        root, run
    )  # noqa: SLF001
    # Sealing deliberately resolves a fresh run handle.  Reacquire that handle
    # before the selector transition so a stale attrs cache cannot replace the
    # just-persisted full-motion manifest.
    run = root[SOURCE_RUN_PATH]
    run.attrs["stage_selector_eligible"] = True
    parent.attrs.update(
        {
            "latest": f"offline/{SOURCE_RUN_NAME}",
            "latest_complete": f"offline/{SOURCE_RUN_NAME}",
            "latest_offline": SOURCE_RUN_NAME,
        }
    )
    offline.attrs.update(
        {
            "latest": SOURCE_RUN_NAME,
            "latest_complete": SOURCE_RUN_NAME,
        }
    )
    zarr.consolidate_metadata(root.store)
    return Path(root.store_path.store.root)


@pytest.fixture(scope="module")
def canonical_candidate_pair(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("track-v2-read-benchmark")
    archive = _build_canonical_sealed_source(directory)
    result = materialize_track_kinematics_flat_candidate(
        archive,
        source_run=SOURCE_RUN_NAME,
        run_name="flat_candidate_v2",
        scratch_root=directory / "candidate-scratch",
        copy_backend="python",
        apply=True,
    )
    assert result["status"] == "complete"
    return archive


def test_real_zarr_fixture_is_genuinely_public_and_has_explicit_physical_state(
    tmp_path: Path,
) -> None:
    archive = _build_canonical_sealed_source(tmp_path)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    loaded = load_track_kinematics_track(
        root,
        run_name=SOURCE_RUN_NAME,
        scope="offline",
        track_id=7,
    )

    assert loaded.authority_status == "verified_canonical_track_motion_v1"
    np.testing.assert_array_equal(loaded.frame_indices, np.asarray([0, 1]))
    run = root[SOURCE_RUN_PATH]
    assert run["tracks/id_7"].attrs["physical_outputs_status"] == (
        "omitted_no_compatible_typed_physical_frame"
    )
    assert "positions_mm" not in run["tracks/id_7"]


def test_pair_validation_is_complete_no_physical_and_nonpromoting(
    canonical_candidate_pair: Path,
) -> None:
    result = benchmark.validate_pair(
        canonical_candidate_pair,
        source_run_name=SOURCE_RUN_NAME,
        candidate_run_name="flat_candidate_v2",
    )
    benchmark._require_pair_validation(result)
    dimensions = result["source_schema"]["payload"]["dimensions"]

    assert result["complete_decoded_equality"] is True
    assert result["source_consumer"]["public_consumer_implemented"] is True
    assert result["candidate_consumer"]["public_consumer_implemented"] is False
    assert result["candidate_consumer"]["diagnostic_consumer_implemented"] is True
    assert dimensions["track_count"] == 1
    assert dimensions["arena_inventory_present"] is True
    assert dimensions["physical_surfaces_present"] is False
    assert dimensions["physical_bundle_benchmarked"] is False
    assert result["source_schema"]["payload"]["array_count"] == 71
    assert len(result["logical_hashes"]) == 74
    assert dimensions["source_inventory_formula"].endswith(
        "arrays_per_track=69+(35 if physical_surfaces_present else 0)"
    )
    assert dimensions["candidate_inventory_formula"].endswith(
        "arrays_per_track=72+(35 if physical_surfaces_present else 0)"
    )
    assert result["publication_physical_copy_replayed"] is False
    assert result["profile_promoted"] is False
    assert result["selector_eligible_candidate"] is False
    root = zarr.open_group(
        str(canonical_candidate_pair), mode="r", use_consolidated=True
    )
    with pytest.raises(ValueError, match="schema|canonical|selector"):
        load_track_kinematics_track(
            root,
            run_name="flat_candidate_v2",
            scope="offline",
            track_id=7,
        )


def test_one_repetition_matrix_uses_fresh_children_and_preserves_archive(
    canonical_candidate_pair: Path,
    tmp_path: Path,
) -> None:
    result = benchmark.run_benchmark_matrix(
        canonical_candidate_pair,
        source_run_name=SOURCE_RUN_NAME,
        candidate_run_name="flat_candidate_v2",
        output=tmp_path / "track-v2-read-benchmark-output",
        repetitions=1,
        seed=17,
        window_rows=2,
        windows_per_array=2,
    )
    benchmark.require_matrix_result(result)
    payload = result["payload"]
    trials = payload["trials"]

    assert len(trials) == 2
    assert len({trial["payload"]["runtime"]["process_id"] for trial in trials}) == 2
    assert all(
        trial["payload"]["runtime"]["parent_process_id"] == payload["driver_process_id"]
        for trial in trials
    )
    assert payload["archive_read_only"] is True
    assert payload["selectors_unchanged"] is True
    assert payload["physical_surfaces_present"] is False
    assert payload["physical_bundle_benchmarked"] is False
    assert (tmp_path / "track-v2-read-benchmark-output" / "matrix.json").is_file()

    tampered = copy.deepcopy(result)
    tampered_pair = tampered["payload"]["pair_validation"]
    tampered_pair["metadata_equivalence"]["candidate"]["declarations_sha256"] = "b" * 64
    for index, trial in enumerate(tampered["payload"]["trials"]):
        trial["payload"]["validation"]["receipt"] = copy.deepcopy(tampered_pair)
        tampered["payload"]["trials"][index] = benchmark._strict_envelope(
            benchmark.TRIAL_SCHEMA_ID,
            trial["payload"],
        )
    tampered["payload"]["summary"] = benchmark._matrix_summary(
        tampered["payload"]["trials"]
    )
    tampered = benchmark._strict_envelope(
        benchmark.MATRIX_SCHEMA_ID,
        tampered["payload"],
    )
    with pytest.raises(ValueError, match="diverge from the live"):
        benchmark.require_matrix_result(tampered)

    coordinated_storage = copy.deepcopy(result)
    for index, trial in enumerate(coordinated_storage["payload"]["trials"]):
        run_tree = trial["payload"]["storage"]["run_tree"]
        run_tree["payload_file_count"] += 1
        run_tree["file_count"] += 1
        coordinated_storage["payload"]["trials"][index] = (
            benchmark._strict_envelope(  # noqa: SLF001
                benchmark.TRIAL_SCHEMA_ID,
                trial["payload"],
            )
        )
    coordinated_storage["payload"]["summary"] = (
        benchmark._matrix_summary(  # noqa: SLF001
            coordinated_storage["payload"]["trials"]
        )
    )
    coordinated_storage = benchmark._strict_envelope(  # noqa: SLF001
        benchmark.MATRIX_SCHEMA_ID,
        coordinated_storage["payload"],
    )
    with pytest.raises(ValueError, match="storage observations diverge from the live"):
        benchmark.require_matrix_result(coordinated_storage)

    coordinated_primary = copy.deepcopy(result)
    for index, trial in enumerate(coordinated_primary["payload"]["trials"]):
        receipt = trial["payload"]["primary_access"]["receipt"]
        receipt["arrays"][0]["payload_sha256"] = "f" * 64
        receipt["payload_sha256"] = benchmark._primary_projection_digest(
            receipt["arrays"]
        )  # noqa: SLF001
        coordinated_primary["payload"]["trials"][index] = (
            benchmark._strict_envelope(  # noqa: SLF001
                benchmark.TRIAL_SCHEMA_ID,
                trial["payload"],
            )
        )
    coordinated_primary["payload"]["summary"] = (
        benchmark._matrix_summary(  # noqa: SLF001
            coordinated_primary["payload"]["trials"]
        )
    )
    coordinated_primary = benchmark._strict_envelope(  # noqa: SLF001
        benchmark.MATRIX_SCHEMA_ID,
        coordinated_primary["payload"],
    )
    with pytest.raises(ValueError, match="primary workload.*live replay"):
        benchmark.require_matrix_result(coordinated_primary)


def test_trial_rejects_false_physical_scope_and_replayed_parent_pid(
    canonical_candidate_pair: Path,
) -> None:
    role = benchmark._trial_order(seed=17, repetition_index=0)[0]
    trial = benchmark.run_trial(
        canonical_candidate_pair,
        source_run_name=SOURCE_RUN_NAME,
        candidate_run_name="flat_candidate_v2",
        role=role,
        repetition_index=0,
        order_position=0,
        seed=17,
        driver_process_id=os.getppid(),
        window_rows=2,
        windows_per_array=2,
    )
    tampered = copy.deepcopy(trial)
    tampered["payload"]["physical_surfaces_present"] = True
    tampered = benchmark._strict_envelope(
        benchmark.TRIAL_SCHEMA_ID,
        tampered["payload"],
    )
    with pytest.raises(ValueError, match="scope/nonpromotion"):
        benchmark.require_trial_result(tampered)

    tampered = copy.deepcopy(trial)
    tampered["payload"]["environment"]["unexpected_authority"] = True
    tampered = benchmark._strict_envelope(  # noqa: SLF001
        benchmark.TRIAL_SCHEMA_ID,
        tampered["payload"],
    )
    with pytest.raises(ValueError, match="environment field set"):
        benchmark.require_trial_result(tampered)

    tampered = copy.deepcopy(trial)
    tampered["payload"]["storage"]["archive_guard"]["storage"][
        "unexpected_authority"
    ] = 1
    tampered = benchmark._strict_envelope(  # noqa: SLF001
        benchmark.TRIAL_SCHEMA_ID,
        tampered["payload"],
    )
    with pytest.raises(ValueError, match="storage-stat field set"):
        benchmark.require_trial_result(tampered)

    tampered = copy.deepcopy(trial)
    tampered["payload"]["storage"]["run_tree"]["role"] = (
        "candidate" if role == "source" else "source"
    )
    tampered = benchmark._strict_envelope(  # noqa: SLF001
        benchmark.TRIAL_SCHEMA_ID,
        tampered["payload"],
    )
    with pytest.raises(ValueError, match="storage identity binding"):
        benchmark.require_trial_result(tampered)

    with pytest.raises(ValueError, match="live parent"):
        benchmark.run_trial(
            canonical_candidate_pair,
            source_run_name=SOURCE_RUN_NAME,
            candidate_run_name="flat_candidate_v2",
            role=role,
            repetition_index=0,
            order_position=0,
            seed=17,
            driver_process_id=max(os.getpid(), os.getppid()) + 10_000,
            window_rows=2,
            windows_per_array=2,
        )


def test_coordinated_rehash_cannot_claim_physical_bundle(
    canonical_candidate_pair: Path,
) -> None:
    pair = benchmark.validate_pair(
        canonical_candidate_pair,
        source_run_name=SOURCE_RUN_NAME,
        candidate_run_name="flat_candidate_v2",
    )
    tampered = copy.deepcopy(pair)
    source_schema = tampered["source_schema"]
    source_schema["payload"]["dimensions"]["physical_surfaces_present"] = True
    source_schema["payload_digest"] = benchmark.canonical_json_sha256(
        source_schema["payload"]
    )
    tampered["physical_surfaces_present"] = True
    with pytest.raises(ValueError, match="correctness/nonpromotion"):
        benchmark._require_pair_validation(tampered)

    tampered = copy.deepcopy(pair)
    storage = tampered["candidate_storage_receipt"]
    storage["payload"]["arrays"][0]["plan"]["chunk_nbytes"] += 1
    storage["payload_digest"] = benchmark.canonical_json_sha256(storage["payload"])
    with pytest.raises(ValueError, match="executable byte planning"):
        benchmark._require_pair_validation(tampered)

    tampered = copy.deepcopy(pair)
    tampered["publication_physical_copy_replayed"] = True
    tampered["publication_physical_copy_receipt_role"] = (
        "authoritative_replayed_physical_copy"
    )
    with pytest.raises(ValueError, match="correctness/nonpromotion"):
        benchmark._require_pair_validation(tampered)


def test_atomic_receipt_extra_field_and_descendant_symlink_fail_closed(
    canonical_candidate_pair: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "track-symlink-attack.zarr"
    shutil.copytree(canonical_candidate_pair, copied)
    root = zarr.open_group(str(copied), mode="a", use_consolidated=False)
    source = root[benchmark._source_run_path(SOURCE_RUN_NAME)]
    candidate = root[benchmark._source_run_path("flat_candidate_v2")]
    hashes = benchmark.source_flat_projection_hashes(
        source,
        benchmark.build_flat_candidate_declarations(source),
    )
    receipt = copy.deepcopy(candidate.attrs["cluster_output_staging"])
    receipt["unexpected_authority"] = True
    candidate.attrs["cluster_output_staging"] = receipt
    with pytest.raises(ValueError, match="field set"):
        benchmark._require_atomic_publication_receipt(
            candidate,
            archive=copied,
            source_name=SOURCE_RUN_NAME,
            candidate_name="flat_candidate_v2",
            expected_hashes=hashes,
            expected_parent_attrs=benchmark._parent_attrs_snapshot(
                root
            ),  # noqa: SLF001
        )

    candidate_metadata = copied.joinpath(
        *benchmark._source_run_path("flat_candidate_v2").split("/"),
        "zarr.json",
    )
    retained = tmp_path / "candidate-zarr.json"
    candidate_metadata.rename(retained)
    candidate_metadata.symlink_to(retained)
    with pytest.raises(ValueError, match="forbidden symlink"):
        benchmark.validate_pair(
            copied,
            source_run_name=SOURCE_RUN_NAME,
            candidate_run_name="flat_candidate_v2",
        )

    dependency_copy = tmp_path / "track-dependency-symlink-attack.zarr"
    shutil.copytree(canonical_candidate_pair, dependency_copy)
    pair = benchmark.validate_pair(
        canonical_candidate_pair,
        source_run_name=SOURCE_RUN_NAME,
        candidate_run_name="flat_candidate_v2",
    )
    dependency = dependency_copy.joinpath(*pair["dependencies"][0].split("/"))
    dependency_metadata = dependency / "zarr.json"
    retained_dependency = tmp_path / "dependency-zarr.json"
    dependency_metadata.rename(retained_dependency)
    dependency_metadata.symlink_to(retained_dependency)
    with pytest.raises(ValueError, match="Source dependency.*forbidden symlink"):
        benchmark.validate_pair(
            dependency_copy,
            source_run_name=SOURCE_RUN_NAME,
            candidate_run_name="flat_candidate_v2",
        )


@pytest.mark.parametrize(
    ("backend", "verification"),
    (
        ("python", "rsync_checksum_dry_run"),
        ("rsync", "sha256_all_physical_files"),
        ("invented", "sha256_all_physical_files"),
    ),
)
def test_atomic_receipt_rejects_backend_verification_mismatch(
    canonical_candidate_pair: Path,
    tmp_path: Path,
    backend: str,
    verification: str,
) -> None:
    copied = tmp_path / f"track-copy-pair-{backend}-{verification}.zarr"
    shutil.copytree(canonical_candidate_pair, copied)
    root = zarr.open_group(str(copied), mode="a", use_consolidated=False)
    candidate = root[benchmark._source_run_path("flat_candidate_v2")]
    source = root[benchmark._source_run_path(SOURCE_RUN_NAME)]
    receipt = copy.deepcopy(candidate.attrs["cluster_output_staging"])
    receipt["physical_copy"]["backend"] = backend
    receipt["physical_copy"]["verification"] = verification
    candidate.attrs["cluster_output_staging"] = receipt
    hashes = benchmark.source_flat_projection_hashes(
        source,
        benchmark.build_flat_candidate_declarations(source),
    )

    with pytest.raises(ValueError, match="physical-copy evidence"):
        benchmark._require_atomic_publication_receipt(  # noqa: SLF001
            candidate,
            archive=canonical_candidate_pair,
            source_name=SOURCE_RUN_NAME,
            candidate_name="flat_candidate_v2",
            expected_hashes=hashes,
            expected_parent_attrs=benchmark._parent_attrs_snapshot(
                root
            ),  # noqa: SLF001
        )


def test_atomic_receipt_rejects_coordinated_forged_parent_pointer_snapshot(
    canonical_candidate_pair: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "track-forged-parent-pointer.zarr"
    shutil.copytree(canonical_candidate_pair, copied)
    root = zarr.open_group(str(copied), mode="a", use_consolidated=False)
    candidate = root[benchmark._source_run_path("flat_candidate_v2")]
    receipt = copy.deepcopy(candidate.attrs["cluster_output_staging"])
    for field in ("parent_attrs_before", "parent_attrs_after"):
        receipt[field][benchmark.PARENT_PATH]["latest"] = "offline/forged_source"
    candidate.attrs["cluster_output_staging"] = receipt
    source = root[benchmark._source_run_path(SOURCE_RUN_NAME)]
    hashes = benchmark.source_flat_projection_hashes(
        source,
        benchmark.build_flat_candidate_declarations(source),
    )

    with pytest.raises(ValueError, match="live selected source authority"):
        benchmark._require_atomic_publication_receipt(  # noqa: SLF001
            candidate,
            archive=canonical_candidate_pair,
            source_name=SOURCE_RUN_NAME,
            candidate_name="flat_candidate_v2",
            expected_hashes=hashes,
            expected_parent_attrs=benchmark._parent_attrs_snapshot(
                root
            ),  # noqa: SLF001
        )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is unavailable")
def test_genuine_rsync_publication_receipt_pair_is_accepted(tmp_path: Path) -> None:
    archive = _build_canonical_sealed_source(tmp_path)
    result = materialize_track_kinematics_flat_candidate(
        archive,
        source_run=SOURCE_RUN_NAME,
        run_name="flat_candidate_rsync",
        scratch_root=tmp_path / "candidate-rsync-scratch",
        copy_backend="rsync",
        apply=True,
    )
    assert result["status"] == "complete"

    pair = benchmark.validate_pair(
        archive,
        source_run_name=SOURCE_RUN_NAME,
        candidate_run_name="flat_candidate_rsync",
    )
    benchmark._require_pair_validation(pair)  # noqa: SLF001
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    receipt = root[benchmark._source_run_path("flat_candidate_rsync")].attrs[
        "cluster_output_staging"
    ]
    assert receipt["physical_copy"]["backend"] == "rsync"
    assert receipt["physical_copy"]["verification"] == "rsync_checksum_dry_run"
