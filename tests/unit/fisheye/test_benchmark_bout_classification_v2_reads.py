from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
import pytest
import zarr

from fisheye.analysis.bout_classification_runs import (
    load_bout_classification_table,
    resolve_bout_classification_run,
)
from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PLAN_DIGEST_ATTR,
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis.megabouts_classifier import (
    MegaboutsClassificationResult,
    write_megabouts_classification_run,
)
from fisheye.analysis.megabouts_classifier_inputs import MegaboutsClassifierInputPack
from fisheye.analysis_workflows.bout_classification_candidate_execution import (
    bout_classification_logical_manifest_sha256,
    build_bout_classification_coordinate_evidence,
    build_bout_classification_execution_suite,
    build_bout_classification_scientific_identity,
    compute_bout_classification_logical_hashes,
    reconstruct_bout_classification_writer_inputs,
    require_bout_classification_execution_suite,
)
from fisheye.analysis_workflows.analysis_candidate_execution import (
    PhysicalIOScope,
    build_candidate_execution_request,
    require_candidate_execution_receipt,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    build_bout_classification_invocation,
)
from fisheye.analysis_workflows.materializers.bout_classification_candidate import (
    materialize_bout_classification_candidate,
)
from fisheye.diagnostics import (
    benchmark_bout_classification_v2_reads as benchmark,
)
from fisheye.diagnostics.bout_classification_candidate_execution import (
    BoutClassificationCandidateExecutionFailed,
    bout_classification_selector_snapshot_sha256,
    require_bout_classification_execution_attempt,
    run_bout_classification_candidate_fresh_process,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1

SOURCE = "selected_public_v2"
CANDIDATE = "byte_candidate_v2"


def _pack() -> MegaboutsClassifierInputPack:
    n_bouts = 5
    window_frames = 4
    valid = np.asarray([True, False, True, False, True], dtype=bool)
    return MegaboutsClassifierInputPack(
        tail_array=np.zeros((n_bouts, 10, window_frames), dtype=np.float32),
        traj_array=np.zeros((n_bouts, 3, window_frames), dtype=np.float32),
        tail_valid=np.repeat(valid[:, None], window_frames, axis=1),
        traj_valid=np.repeat(valid[:, None], window_frames, axis=1),
        traj_reference_valid=valid.copy(),
        source_bout_id=np.asarray([11, 12, 13, 14, 15], dtype=np.int64),
        source_start_frame=np.asarray([0, 10, 20, 30, 40], dtype=np.int64),
        source_end_frame=np.asarray([3, 13, 23, 33, 43], dtype=np.int64),
        window_start_frame=np.asarray([0, 10, 20, 30, 40], dtype=np.int64),
        window_end_frame=np.asarray([3, 13, 23, 33, 43], dtype=np.int64),
        tail_valid_fraction=valid.astype(np.float32),
        traj_valid_fraction=valid.astype(np.float32),
        max_consecutive_tail_invalid=np.asarray([0, 4, 0, 4, 0], dtype=np.int32),
        max_consecutive_traj_invalid=np.asarray([0, 4, 0, 4, 0], dtype=np.int32),
        valid_bout=valid,
        failure_reason=np.asarray(
            ["ok", "tail_invalid", "ok", "trajectory_invalid", "ok"],
            dtype=object,
        ),
        source_refs={
            "tail_posture_view_run": "analysis/tail_posture_view_runs/posture",
            "tail_frame_indices": "analysis/tail_posture_view_runs/posture/source_acquisition_frame_index",
            "tail_instance_key": "analysis/tail_posture_view_runs/posture/instance_key",
            "tail_angle_rad": "analysis/tail_posture_view_runs/posture/tail_angle_rad",
            "tail_valid": "analysis/tail_posture_view_runs/posture/valid",
            "tail_posture_publication_manifest_ref": (
                "analysis/tail_posture_view_runs/posture"
                "@tail_coordinate_publication_manifest"
            ),
            "tail_posture_publication_manifest_sha256": "a" * 64,
            "tail_posture_source_subject_shape_run": (
                "analysis/subject_shape_runs/shape"
            ),
            "tail_posture_source_subject_shape_publication_manifest_ref": (
                "analysis/subject_shape_runs/shape"
                "@subject_shape_publication_manifest"
            ),
            "tail_posture_source_subject_shape_publication_manifest_sha256": ("b" * 64),
            "track_kinematics_run": "analysis/track_kinematics_runs/offline/tk",
            "track_group": "analysis/track_kinematics_runs/offline/tk/tracks/id_0",
            "track_frame_indices": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/source_acquisition_frame_index",
            "track_source_instance_key": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/source_instance_key",
            "positions_mm": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/positions_mm",
            "positions_mm_coordinate_descriptor_sha256": "c" * 64,
            "heading": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/smoothed_heading_radians",
            "sample_valid": "analysis/track_kinematics_runs/offline/tk/tracks/id_0/sample_valid",
            "track_motion_manifest_sha256": "d" * 64,
            "swim_bout_run": "analysis/swim_bout_runs/bouts",
            "swim_bout_level": "analysis/swim_bout_runs/bouts/speed_filtered",
            "bouts": "analysis/swim_bout_runs/bouts/speed_filtered/bouts",
            "swim_bout_source_track_motion_manifest_sha256": "d" * 64,
        },
        parameters={
            "adapter_method": "palette_megabouts_classifier_input_dry_run",
            "source_mode": "palette_bouts",
            "tail_posture_view_run": "posture",
            "track_kinematics_scope": "offline",
            "track_kinematics_run": "tk",
            "track_id": 0,
            "swim_bout_run": "bouts",
            "speed_level": "speed_filtered",
            "swim_bout_candidate_id": 0,
            "swim_bout_signal_id": 0,
            "heading_source": "smoothed_heading_radians",
            "fps": 60.0,
            "bout_duration_s": window_frames / 60.0,
            "bout_duration_frames": window_frames,
            "window_policy": "start_frame_fixed_duration",
            "tail_track_join_policy": (
                "posture_instance_key_to_track_source_instance_key_then_exact_"
                "acquisition_frame_v1"
            ),
            "classifier_input_mode": "palette_prepared_fixed_windows",
            "megabouts_preprocessing": False,
            "megabouts_segmentation": False,
            "traj_alignment": "onset_translation_rotation",
            "traj_reference_index": 0,
            "min_tail_valid_fraction": 0.9,
            "min_traj_valid_fraction": 0.9,
            "max_consecutive_invalid_frames": 1,
            "requires_traj_reference_valid": True,
        },
    )


def _result() -> MegaboutsClassificationResult:
    return MegaboutsClassificationResult(
        classified_indices=np.asarray([0, 2, 4], dtype=np.int64),
        classif_results={
            "cat": np.asarray([2, 3, 4], dtype=np.int32),
            "subcat": np.asarray([12, 13, 14], dtype=np.int32),
            "sign": np.asarray([-1, 1, -1], dtype=np.int32),
            "proba": np.asarray([0.875, 0.75, 0.625], dtype=np.float32),
            "first_half_beat": np.asarray([1, 2, 1], dtype=np.int32),
        },
        runtime=None,
    )


def _build_pair_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "bout_pair.zarr"
    root = zarr.open_group(archive, mode="w", zarr_format=3)
    write_megabouts_classification_run(
        root,
        run_name=SOURCE,
        pack=_pack(),
        result=_result(),
    )
    write_megabouts_classification_run(
        root,
        run_name=CANDIDATE,
        pack=_pack(),
        result=_result(),
        storage_profile=PUBLISHED_HTTP_V1,
    )
    zarr.consolidate_metadata(archive)
    return archive


def _resign(envelope: dict[str, object]) -> dict[str, object]:
    envelope["payload_digest"] = canonical_json_sha256(envelope["payload"])
    return envelope


def _coordinated_pair_rewrite(
    matrix: dict[str, object], pair: dict[str, object]
) -> dict[str, object]:
    payload = matrix["payload"]
    assert isinstance(payload, dict)
    payload["pair_validation"] = pair
    guard = payload["archive_read_only_guard"]
    assert isinstance(guard, dict)
    guard["before"] = copy.deepcopy(pair)
    guard["after"] = copy.deepcopy(pair)
    for trial in payload["trials"]:
        trial_payload = trial["payload"]
        trial_payload["pair_payload_digest"] = pair["payload_digest"]
        trial_payload["workload_payload_digest"] = pair["payload"]["workload"][
            "payload_digest"
        ]
        _resign(trial)
    return _resign(matrix)


@pytest.fixture()
def pair_archive(tmp_path: Path) -> Path:
    return _build_pair_archive(tmp_path)


def test_typed_execution_suite_replays_exact_direct_writer_payload(
    pair_archive: Path,
) -> None:
    root = zarr.open_group(pair_archive, mode="a", use_consolidated=False)
    source = root[f"analysis/bout_classification_runs/{SOURCE}"]
    pack, result = reconstruct_bout_classification_writer_inputs(source)

    replay_name = "typed_writer_replay_v2"
    write_megabouts_classification_run(
        root,
        run_name=replay_name,
        pack=pack,
        result=result,
        storage_profile=PUBLISHED_HTTP_V1,
    )
    replay = root[f"analysis/bout_classification_runs/{replay_name}"]

    assert compute_bout_classification_logical_hashes(replay) == (
        compute_bout_classification_logical_hashes(source)
    )
    assert build_bout_classification_scientific_identity(replay) == (
        build_bout_classification_scientific_identity(source)
    )
    coordinate = build_bout_classification_coordinate_evidence(source, replay)
    assert coordinate["status"] == "verified_bound_source"
    assert coordinate["coordinate_gate_passed"] is True

    suite = build_bout_classification_execution_suite(
        source,
        seed=23,
        repetitions=2,
    )
    require_bout_classification_execution_suite("bout_classification", suite)
    assert len(suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]) == 20


def test_atomic_typed_materializer_preserves_source_and_selectors(
    pair_archive: Path,
    tmp_path: Path,
) -> None:
    before = zarr.open_group(pair_archive, mode="r", use_consolidated=False)[
        "analysis/bout_classification_runs"
    ]
    latest_before = before.attrs["latest"]
    latest_complete_before = before.attrs["latest_complete"]

    result = materialize_bout_classification_candidate(
        pair_archive,
        source_run=SOURCE,
        run_name="atomic_typed_candidate_v2",
        scratch_root=tmp_path / "bout-typed-scratch",
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["publication"]["physical_copy"]["verification"] == (
        "sha256_all_physical_files"
    )
    assert result["publication"]["physical_copy"]["content_sha256"]
    assert result["local_direct_consolidated_array_count"] == 20
    assert result["published_direct_consolidated_array_count"] == 20
    root = zarr.open_group(pair_archive, mode="r", use_consolidated=False)
    parent = root["analysis/bout_classification_runs"]
    assert parent.attrs["latest"] == latest_before
    assert parent.attrs["latest_complete"] == latest_complete_before
    source = parent[SOURCE]
    candidate = parent["atomic_typed_candidate_v2"]
    assert candidate.attrs["stage_selector_eligible"] is False
    assert compute_bout_classification_logical_hashes(candidate) == (
        compute_bout_classification_logical_hashes(source)
    )


@pytest.mark.parametrize("valid_source_identity", [True, False])
def test_fresh_process_executes_or_fails_closed(
    tmp_path: Path,
    valid_source_identity: bool,
) -> None:
    benchmark_root = tmp_path / ".palette_benchmarks" / "bout-execution"
    benchmark_root.mkdir(parents=True)
    archive = _build_pair_archive(benchmark_root)
    root = zarr.open_group(archive, mode="r", use_consolidated=False)
    source_path = f"analysis/bout_classification_runs/{SOURCE}"
    source = root[source_path]
    source_identity = bout_classification_logical_manifest_sha256(source)
    scientific_identity = canonical_json_sha256(
        build_bout_classification_scientific_identity(source)
    )
    candidate_name = (
        "typed_fresh_candidate_valid"
        if valid_source_identity
        else "typed_fresh_candidate_invalid"
    )
    candidate_path = f"analysis/bout_classification_runs/{candidate_name}"
    probe = benchmark_root / "protected.json"
    probe.write_text("{}\n", encoding="utf-8")
    scratch = Path("/tmp") / f"palette-bout-runner-{uuid.uuid4().hex}"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["bout_classification"]
    request = build_candidate_execution_request(
        execution_id=f"bout_{uuid.uuid4().hex}",
        adapter_manifest=adapter.as_manifest(),
        invocation=build_bout_classification_invocation(
            source_scientific_identity_sha256=scientific_identity,
            storage_profile_id="published_http_v1",
            copy_backend="python",
            keep_scratch=False,
            check_capacity=False,
        ),
        benchmark_suite=build_bout_classification_execution_suite(
            source,
            repetitions=1,
        ),
        archive_path=archive,
        source_run_path=source_path,
        candidate_run_path=candidate_path,
        scratch_root=scratch,
        source_identity_sha256=(source_identity if valid_source_identity else "f" * 64),
        palette_commit=commit,
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256=(bout_classification_selector_snapshot_sha256(archive)),
        registry_probe_path=probe,
        production_profiles_probe_path=probe,
    )
    workflow = benchmark_root / f"driver-{valid_source_identity}"
    evidence_dir = workflow / "evidence"
    evidence_dir.mkdir(parents=True)
    request_path = workflow / "request.json"
    receipt_path = evidence_dir / "receipt.json"
    attempt_path = evidence_dir / "attempt.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    if valid_source_identity:
        evidence = run_bout_classification_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )
        require_candidate_execution_receipt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        assert evidence["payload"]["logical_equality"] == {
            "contract_id": "bout_classification_v2_arrays_v1",
            "compared_array_count": 20,
            "source_logical_manifest_sha256": source_identity,
            "candidate_logical_manifest_sha256": source_identity,
            "equal": True,
        }
        assert evidence["payload"]["coordinate_evidence"]["status"] == (
            "verified_bound_source"
        )
        assert evidence["payload"]["publication_gate_passed"] is False
        assert receipt_path.is_file()
        assert not attempt_path.exists()
        candidate = zarr.open_group(archive, mode="r", use_consolidated=False)[
            candidate_path
        ]
        assert candidate.attrs["stage_selector_eligible"] is False
        assert candidate.attrs["storage_candidate_profile_promoted"] is False
    else:
        with pytest.raises(
            BoutClassificationCandidateExecutionFailed,
            match="see attempt record",
        ) as raised:
            run_bout_classification_candidate_fresh_process(
                request_path,
                receipt_path=receipt_path,
                attempt_path=attempt_path,
            )
        evidence = raised.value.attempt
        require_bout_classification_execution_attempt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        assert evidence["payload"]["failure_phase"] == "runner_preflight"
        assert evidence["payload"]["target_state"]["exists"] is False
        assert attempt_path.is_file()
        assert not receipt_path.exists()
    assert not scratch.exists()


def test_real_writers_public_source_private_candidate_and_five_process_matrix(
    pair_archive: Path, tmp_path: Path
) -> None:
    pair = benchmark.build_pair_validation(
        pair_archive,
        source_run=SOURCE,
        candidate_run=CANDIDATE,
        window_rows=3,
    )
    benchmark.require_pair_validation(pair, replay_archive=True)
    payload = pair["payload"]
    assert payload["schema_contract"]["array_count"] == 20
    assert payload["logical_equality"] is True
    assert payload["scientific_identity"]["equal"] is True
    assert (
        payload["scientific_identity"]["source"]
        == payload["scientific_identity"]["candidate"]
    )
    assert payload["profile_promoted"] is False
    assert payload["lifecycle"]["candidate_selector_eligible"] is False
    assert payload["consumers"] == {
        "source": benchmark.SOURCE_CONSUMER,
        "source_public_consumer_implemented": True,
        "candidate": benchmark.CANDIDATE_CONSUMER,
        "candidate_public_consumer_implemented": False,
        "candidate_diagnostic_consumer_implemented": True,
    }
    assert len(payload["logical_arrays"]) == 20
    assert all(
        array["declared_access_pattern"] == "eager"
        for array in payload["workload"]["payload"]["arrays"]
    )

    direct = zarr.open_group(pair_archive, mode="r", use_consolidated=False)
    source, resolved, _path = resolve_bout_classification_run(direct, SOURCE)
    assert resolved == SOURCE
    assert load_bout_classification_table(source).shape == (5,)
    parent = direct["analysis/bout_classification_runs"]
    assert parent.attrs["latest"] == SOURCE
    assert parent.attrs["latest_complete"] == SOURCE
    assert parent[CANDIDATE].attrs["stage_selector_eligible"] is False

    output = tmp_path / "bout_benchmark_matrix"
    matrix = benchmark.run_benchmark_matrix(
        pair_archive,
        source_run=SOURCE,
        candidate_run=CANDIDATE,
        output_dir=output,
        repetitions=benchmark.DEFAULT_REPETITIONS,
        window_rows=3,
    )
    benchmark.require_matrix_result(matrix, replay_archive=True)
    matrix_payload = matrix["payload"]
    assert matrix_payload["balanced_read_matrix_complete"] is True
    assert len(matrix_payload["trials"]) == 10
    assert (
        len({trial["payload"]["child_pid"] for trial in matrix_payload["trials"]}) == 10
    )
    for trial in matrix_payload["trials"]:
        trial_payload = trial["payload"]
        assert len(trial_payload["array_reads"]) == 20
        assert trial_payload["aggregate_read"]["eager_operation_count"] == 20
        assert trial_payload["aggregate_read"]["windowed_operation_count"] == 60
        assert trial_payload["physical_io"]["transferred_bytes"] is None
    assert matrix_payload["profile_promoted"] is False
    assert matrix_payload["candidate_selector_eligible"] is False
    assert matrix_payload["evidence_boundaries"] == {
        "writer_phase_measured": False,
        "publication_phase_measured": False,
        "physical_io_measured": False,
        "representative_scale_executed": False,
        "promotion_gate_executed": False,
        "runtime_observations_attested": False,
    }


def test_rejects_outer_consumer_promotion_and_physical_claim_rewrites(
    pair_archive: Path, tmp_path: Path
) -> None:
    matrix = benchmark.run_benchmark_matrix(
        pair_archive,
        source_run=SOURCE,
        candidate_run=CANDIDATE,
        output_dir=tmp_path / "benchmark_claim_attacks",
        repetitions=1,
        window_rows=3,
    )
    for mutate in (
        lambda payload: payload.__setitem__("profile_promoted", True),
        lambda payload: payload.__setitem__("candidate_selector_eligible", True),
        lambda payload: payload["consumers"].__setitem__(
            "candidate_public_consumer_implemented", True
        ),
        lambda payload: payload["physical_io"].__setitem__("transferred_bytes", 1),
        lambda payload: payload["evidence_boundaries"].__setitem__(
            "publication_phase_measured", True
        ),
        lambda payload: payload["evidence_boundaries"].__setitem__(
            "runtime_observations_attested", True
        ),
    ):
        attack = copy.deepcopy(matrix)
        mutate(attack["payload"])
        _resign(attack)
        with pytest.raises(ValueError):
            benchmark.require_matrix_result(attack, replay_archive=False)


@pytest.mark.parametrize(
    "target",
    [
        "workload",
        "storage",
        "metadata",
        "publication",
        "source_ref",
        "parameter",
        "profile",
    ],
)
def test_live_replay_rejects_coordinated_nested_resigning(
    pair_archive: Path, tmp_path: Path, target: str
) -> None:
    matrix = benchmark.run_benchmark_matrix(
        pair_archive,
        source_run=SOURCE,
        candidate_run=CANDIDATE,
        output_dir=tmp_path / f"benchmark_nested_attack_{target}",
        repetitions=1,
        window_rows=3,
    )
    attack = copy.deepcopy(matrix)
    pair = copy.deepcopy(attack["payload"]["pair_validation"])
    pair_payload = pair["payload"]
    if target == "workload":
        workload = pair_payload["workload"]
        workload["payload"]["arrays"][0]["windowed_read_spans"] = [[1, 2]]
        _resign(workload)
    elif target == "storage":
        receipt = pair_payload["candidate_storage_receipt"]
        receipt["payload"]["arrays"][0]["plan"]["chunk_shape"][0] += 1
        _resign(receipt)
    elif target == "metadata":
        pair_payload["metadata_equivalence"]["candidate"]["declarations_sha256"] = (
            "0" * 64
        )
    else:
        if target == "publication":
            pair_payload["lifecycle"]["publication_generation"] += 1
        elif target in {"source_ref", "parameter"}:
            identity = pair_payload["scientific_identity"]
            for role in ("source", "candidate"):
                envelope = identity[role]
                if target == "source_ref":
                    envelope["payload"]["source_refs"][
                        "tail_posture_publication_manifest_sha256"
                    ] = ("e" * 64)
                else:
                    envelope["payload"]["parameters"]["fps"] = 61.0
                _resign(envelope)
            identity["shared_payload_digest"] = identity["source"]["payload_digest"]
        else:
            receipt = pair_payload["candidate_storage_receipt"]
            receipt["payload"]["storage_profile"]["profile_id"] = "scratch_compute_v1"
            _resign(receipt)
    _resign(pair)
    _coordinated_pair_rewrite(attack, pair)
    with pytest.raises(ValueError, match="live archive|replay"):
        benchmark.require_matrix_result(attack, replay_archive=True)


def test_rejects_forged_live_selector_rehashed_storage_and_metadata_drift(
    pair_archive: Path, tmp_path: Path
) -> None:
    selector_archive = tmp_path / "selector_attack.zarr"
    shutil.copytree(pair_archive, selector_archive)
    root = zarr.open_group(
        selector_archive, mode="a", zarr_format=3, use_consolidated=False
    )
    parent = root["analysis/bout_classification_runs"]
    parent.attrs["latest"] = CANDIDATE
    parent.attrs["latest_complete"] = CANDIDATE
    zarr.consolidate_metadata(selector_archive)
    with pytest.raises(ValueError, match="selected complete authority"):
        benchmark.build_pair_validation(
            selector_archive, source_run=SOURCE, candidate_run=CANDIDATE
        )

    storage_archive = tmp_path / "storage_attack.zarr"
    shutil.copytree(pair_archive, storage_archive)
    root = zarr.open_group(
        storage_archive, mode="a", zarr_format=3, use_consolidated=False
    )
    candidate = root[f"analysis/bout_classification_runs/{CANDIDATE}"]
    receipt = candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]
    receipt["payload"]["arrays"][0]["plan"]["chunk_shape"][0] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    candidate.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = receipt["payload_digest"]
    zarr.consolidate_metadata(storage_archive)
    with pytest.raises(ValueError, match="storage|validation"):
        benchmark.build_pair_validation(
            storage_archive, source_run=SOURCE, candidate_run=CANDIDATE
        )

    metadata_archive = tmp_path / "metadata_attack.zarr"
    shutil.copytree(pair_archive, metadata_archive)
    root = zarr.open_group(
        metadata_archive, mode="a", zarr_format=3, use_consolidated=False
    )
    root[f"analysis/bout_classification_runs/{SOURCE}"].attrs[
        "classifier_name"
    ] = "tampered_after_consolidation"
    with pytest.raises(Exception, match="metadata|declaration|differ"):
        benchmark.build_pair_validation(
            metadata_archive, source_run=SOURCE, candidate_run=CANDIDATE
        )


def test_rejects_wrong_profile_dependency_identity_and_hb1_arithmetic(
    pair_archive: Path, tmp_path: Path
) -> None:
    profile_archive = tmp_path / "profile_attack.zarr"
    shutil.copytree(pair_archive, profile_archive)
    root = zarr.open_group(
        profile_archive, mode="a", zarr_format=3, use_consolidated=False
    )
    candidate = root[f"analysis/bout_classification_runs/{CANDIDATE}"]
    candidate.attrs["analysis_storage_profile_id"] = "scratch_compute_v1"
    zarr.consolidate_metadata(profile_archive)
    with pytest.raises(ValueError, match="profile"):
        benchmark.build_pair_validation(
            profile_archive, source_run=SOURCE, candidate_run=CANDIDATE
        )

    identity_archive = tmp_path / "identity_attack.zarr"
    shutil.copytree(pair_archive, identity_archive)
    root = zarr.open_group(
        identity_archive, mode="a", zarr_format=3, use_consolidated=False
    )
    candidate = root[f"analysis/bout_classification_runs/{CANDIDATE}"]
    refs = candidate.attrs["source_refs"]
    refs["track_motion_manifest_sha256"] = "e" * 64
    refs["swim_bout_source_track_motion_manifest_sha256"] = "e" * 64
    candidate.attrs["source_refs"] = refs
    zarr.consolidate_metadata(identity_archive)
    with pytest.raises(ValueError, match="scientific and dependency identities"):
        benchmark.build_pair_validation(
            identity_archive, source_run=SOURCE, candidate_run=CANDIDATE
        )

    hb1_archive = tmp_path / "hb1_attack.zarr"
    shutil.copytree(pair_archive, hb1_archive)
    root = zarr.open_group(hb1_archive, mode="a", zarr_format=3, use_consolidated=False)
    for run_name in (SOURCE, CANDIDATE):
        run = root[f"analysis/bout_classification_runs/{run_name}"]
        run["per_bout/HB1_frame"][0] = 3
    zarr.consolidate_metadata(hb1_archive)
    with pytest.raises(ValueError, match="HB1_frame must equal"):
        benchmark.build_pair_validation(
            hb1_archive, source_run=SOURCE, candidate_run=CANDIDATE
        )


def test_rejects_aliases_extra_arrays_symlink_archives_and_unsafe_output(
    pair_archive: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="explicit immutable"):
        benchmark.build_pair_validation(
            pair_archive, source_run="latest", candidate_run=CANDIDATE
        )
    with pytest.raises(ValueError, match="disjoint"):
        benchmark.run_benchmark_matrix(
            pair_archive,
            source_run=SOURCE,
            candidate_run=CANDIDATE,
            output_dir=pair_archive / "benchmark_output",
            repetitions=1,
        )
    alias = tmp_path / "archive_alias.zarr"
    alias.symlink_to(pair_archive, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        benchmark.build_pair_validation(
            alias, source_run=SOURCE, candidate_run=CANDIDATE
        )

    extra_archive = tmp_path / "extra_array.zarr"
    shutil.copytree(pair_archive, extra_archive)
    root = zarr.open_group(
        extra_archive, mode="a", zarr_format=3, use_consolidated=False
    )
    root[f"analysis/bout_classification_runs/{CANDIDATE}/per_bout"].create_array(
        "legacy_alias", data=np.zeros(5, dtype=np.int32)
    )
    zarr.consolidate_metadata(extra_archive)
    with pytest.raises(ValueError, match="validation|inventory"):
        benchmark.build_pair_validation(
            extra_archive, source_run=SOURCE, candidate_run=CANDIDATE
        )


@pytest.mark.parametrize("windows_per_array", [1, 2])
def test_rejects_incomplete_window_workloads_in_api_and_cli(
    pair_archive: Path, tmp_path: Path, windows_per_array: int
) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 3"):
        benchmark.build_pair_validation(
            pair_archive,
            source_run=SOURCE,
            candidate_run=CANDIDATE,
            windows_per_array=windows_per_array,
        )

    output = tmp_path / f"incomplete_cli_{windows_per_array}"
    with pytest.raises(SystemExit):
        benchmark.main(
            [
                "matrix",
                str(pair_archive),
                "--source-run",
                SOURCE,
                "--candidate-run",
                CANDIDATE,
                "--output",
                str(output),
                "--windows-per-array",
                str(windows_per_array),
            ]
        )
    assert not output.exists()
