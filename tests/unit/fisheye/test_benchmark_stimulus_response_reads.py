from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.stimulus_response import (
    ProtocolStep,
    _write_stimulus_response_compact_v3,
)
from fisheye.analysis.stimulus_response_storage import (
    STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR,
    consolidate_and_validate_stimulus_response_metadata,
)
from fisheye.diagnostics import benchmark_stimulus_response_reads as benchmark
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import ZarrMetadataEquivalenceError
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from fisheye.shared.zarr.stimulus_response_schema import (
    STIMULUS_RESPONSE_LAYOUT,
    STIMULUS_RESPONSE_SCHEMA_ID,
    STIMULUS_RESPONSE_SCHEMA_VERSION,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)


def _global() -> dict[str, np.ndarray]:
    return {
        "fish_id": np.asarray([4, 9], dtype=np.int32),
        "total_distance_mm": np.asarray([1.0, 2.0], dtype=np.float32),
        "mean_speed_mm_s": np.asarray([3.0, 4.0], dtype=np.float32),
        "total_active_s": np.asarray([5.0, 6.0], dtype=np.float32),
        "fraction_moving": np.asarray([0.5, 0.75], dtype=np.float32),
    }


def _step_metrics() -> dict[str, np.ndarray]:
    return {
        "fish_id": np.asarray([4, 9], dtype=np.int32),
        "total_distance_mm": np.asarray([1.0, 2.0], dtype=np.float32),
        "mean_speed_mm_s": np.asarray([3.0, 4.0], dtype=np.float32),
        "median_speed_mm_s": np.asarray([3.0, 4.0], dtype=np.float32),
        "max_speed_mm_s": np.asarray([5.0, 6.0], dtype=np.float32),
        "fraction_moving": np.asarray([0.5, 0.75], dtype=np.float32),
        "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
    }


def _archive(tmp_path: Path, *, frames: int = 256) -> Path:
    archive = tmp_path / "stimulus_response_analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    parent = root.require_group("analysis").require_group("stimulus_response_runs")
    parent.attrs.update(
        {
            "latest": "source",
            "latest_complete": "source",
            "latest_pending": None,
        }
    )
    frame_annotations = {
        "step_index": np.zeros(frames, dtype=np.int32),
        "stimulus_mode_id": np.full(frames, 7, dtype=np.int32),
    }
    for run_name, profile, eligible in (
        ("source", None, True),
        ("candidate", PUBLISHED_HTTP_V1, False),
    ):
        run = parent.create_group(run_name)
        run.attrs.update(
            {
                "schema_id": STIMULUS_RESPONSE_SCHEMA_ID,
                "schema_version": STIMULUS_RESPONSE_SCHEMA_VERSION,
                "layout": STIMULUS_RESPONSE_LAYOUT,
                RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
                RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
                RUN_NAME_ATTR: run_name,
                "stage_selector_eligible": eligible,
            }
        )
        _write_stimulus_response_compact_v3(
            run,
            global_metrics=_global(),
            steps=[ProtocolStep(0, "baseline", "SOLID_BLACK", 7, 0, 20, 2.0)],
            step_metrics=[_step_metrics()],
            frame_annotations=frame_annotations,
            step_bout_metrics=None,
            step_grating_data=None,
            step_concentric_data=None,
            step_loom_data=None,
            global_omr_metrics=None,
            storage_profile=profile,
        )
    consolidate_and_validate_stimulus_response_metadata(
        root,
        run_path="analysis/stimulus_response_runs/candidate",
    )
    return archive


def _preflight(archive: Path, *, repetitions: int = 1) -> dict[str, object]:
    return benchmark._preflight(
        archive,
        source_run_name="source",
        candidate_run_name="candidate",
        seed=benchmark.DEFAULT_SEED,
        repetitions=repetitions,
    )


def _rehash(envelope: dict[str, object]) -> None:
    envelope["payload_digest"] = canonical_json_sha256(envelope["payload"])


def test_preflight_binds_executable_schema_storage_and_logical_content(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)

    result = _preflight(archive, repetitions=5)
    workload = result["workload"]
    assert isinstance(workload, dict)
    benchmark.require_workload(workload)
    payload = workload["payload"]
    assert payload["source_array_schema_manifest"]["schema_version"] == 1
    assert payload["candidate_array_schema_manifest"]["schema_version"] == 2
    assert payload["candidate_selector_eligible"] is False
    assert payload["promotion_authorized"] is False
    assert payload["access"]["operation_count"] == len(payload["expected_arrays"])
    assert (
        payload["candidate_storage_receipt"]["payload_digest"]
        == result["candidate_storage_receipt_payload_digest"]
    )


def test_single_trials_use_role_specific_strict_readers_and_equal_values(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    workload = _preflight(archive)["workload"]
    assert isinstance(workload, dict)
    order = benchmark._trial_order(seed=benchmark.DEFAULT_SEED, repetition_index=0)
    results = {}
    for position, role in enumerate(order):
        results[role] = benchmark.run_single_trial(
            archive,
            source_run="source",
            candidate_run="candidate",
            role=role,
            repetition_index=0,
            order_position=position,
            seed=benchmark.DEFAULT_SEED,
            cache_state="pytest_uncontrolled_os_cache",
            workload=workload,
        )
        benchmark.require_trial_result(results[role], workload=workload)

    assert results["source"]["payload"]["validation"]["consumer_path"] == (
        "strict_compact_v3_source"
    )
    assert results["candidate"]["payload"]["validation"]["consumer_path"] == (
        "strict_byte_planned_compact_v3_candidate"
    )
    assert (
        results["source"]["payload"]["full_scan"]["arrays"]
        == results["candidate"]["payload"]["full_scan"]["arrays"]
    )
    assert results["candidate"]["payload"]["physical_io"]["transferred_bytes"] is None


def test_matrix_uses_distinct_processes_and_preserves_archive_metadata(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    output = tmp_path / ".palette_benchmarks" / "stimulus-response-v3"

    result = benchmark.run_benchmark_matrix(
        archive,
        source_run="source",
        candidate_run="candidate",
        output_dir=output,
        cache_state="pytest_uncontrolled_os_cache",
        seed=benchmark.DEFAULT_SEED,
        repetitions=1,
    )

    benchmark.require_matrix_result(result)
    payload = result["payload"]
    assert payload["correctness"]["all_passed"] is True
    assert payload["archive_read_only_metadata_guard"]["unchanged"] is True
    assert payload["balanced_fresh_process_matrix_complete"] is False
    pids = [trial["payload"]["process_id"] for trial in payload["trials"]]
    assert len(set(pids)) == 2
    assert payload["driver_process_id"] not in pids
    assert (output / "read_workload.json").is_file()
    assert (output / "matrix_result.json").is_file()


def test_workload_rejects_coordinated_receipt_and_schema_tampering(
    tmp_path: Path,
) -> None:
    workload = _preflight(_archive(tmp_path))["workload"]
    assert isinstance(workload, dict)

    receipt = copy.deepcopy(workload)
    payload = receipt["payload"]
    payload["candidate_storage_receipt"]["payload"]["arrays"][0]["plan"]["chunk_shape"][
        0
    ] += 1
    payload["candidate_storage_receipt"]["payload_digest"] = canonical_json_sha256(
        payload["candidate_storage_receipt"]["payload"]
    )
    payload["candidate_storage_receipt_payload_digest"] = payload[
        "candidate_storage_receipt"
    ]["payload_digest"]
    _rehash(receipt)
    with pytest.raises(ValueError, match="receipt"):
        benchmark.require_workload(receipt)

    schema = copy.deepcopy(workload)
    schema["payload"]["candidate_array_schema_manifest"]["arrays"][0][
        "physical_policy_owner"
    ] = "tampered.owner"
    _rehash(schema)
    with pytest.raises(ValueError, match="schema manifest"):
        benchmark.require_workload(schema)

    metadata = copy.deepcopy(workload)
    metadata_payload = metadata["payload"]
    declarations = metadata_payload["candidate_metadata_declarations"]
    array_path = next(
        path
        for path, declaration in declarations.items()
        if declaration["node_type"] == "array"
    )
    declarations[array_path]["chunk_grid"]["configuration"]["chunk_shape"][0] += 1
    normalized = benchmark._candidate_normalized_metadata_document(
        declarations,
        run_path=metadata_payload["candidate_run_path"],
        bundles=metadata_payload["bundles"],
    )
    equivalence = metadata_payload["candidate_metadata_equivalence_receipt"]
    equivalence["payload"]["normalized_metadata_sha256"] = canonical_json_sha256(
        normalized
    )
    equivalence["payload_digest"] = canonical_json_sha256(equivalence["payload"])
    metadata_payload["candidate_metadata_equivalence_payload_digest"] = equivalence[
        "payload_digest"
    ]
    declarations[metadata_payload["candidate_run_path"]]["attributes"][
        STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR
    ] = equivalence
    metadata_payload["candidate_metadata_equivalence"]["declarations_sha256"] = (
        canonical_json_sha256(declarations)
    )
    _rehash(metadata)
    with pytest.raises(ValueError, match="executable storage plan"):
        benchmark.require_workload(metadata)


def test_rehashed_trial_and_matrix_cannot_authorize_promotion_or_io(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    workload = _preflight(archive)["workload"]
    assert isinstance(workload, dict)
    role = benchmark._trial_order(seed=benchmark.DEFAULT_SEED, repetition_index=0)[0]
    trial = benchmark.run_single_trial(
        archive,
        source_run="source",
        candidate_run="candidate",
        role=role,
        repetition_index=0,
        order_position=0,
        seed=benchmark.DEFAULT_SEED,
        cache_state="pytest",
        workload=workload,
    )

    promoted = copy.deepcopy(trial)
    promoted["payload"]["promotion_authorized"] = True
    _rehash(promoted)
    with pytest.raises(ValueError, match="nonpromotion"):
        benchmark.require_trial_result(promoted, workload=workload)

    fabricated = copy.deepcopy(trial)
    fabricated["payload"]["physical_io"]["transferred_bytes"] = 1
    _rehash(fabricated)
    with pytest.raises(ValueError, match="must not fabricate"):
        benchmark.require_trial_result(fabricated, workload=workload)

    wrong_order = copy.deepcopy(trial)
    wrong_order["payload"]["order_position"] = 2
    _rehash(wrong_order)
    with pytest.raises(ValueError, match="deterministic order"):
        benchmark.require_trial_result(wrong_order, workload=workload)

    matrix = benchmark.run_benchmark_matrix(
        archive,
        source_run="source",
        candidate_run="candidate",
        output_dir=tmp_path / ".palette_benchmarks" / "tamper-matrix",
        cache_state="pytest",
        repetitions=1,
    )
    matrix["payload"]["promotion_authorized"] = True
    _rehash(matrix)
    with pytest.raises(ValueError, match="nonpromotion"):
        benchmark.require_matrix_result(matrix)


def test_preflight_rejects_stale_consolidated_metadata_and_value_mismatch(
    tmp_path: Path,
) -> None:
    stale = _archive(tmp_path / "stale")
    root_metadata_path = stale / "zarr.json"
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    source_inline = root_metadata["consolidated_metadata"]["metadata"][
        "analysis/stimulus_response_runs/source"
    ]
    source_inline["attributes"]["tampered_consolidated_only"] = True
    root_metadata_path.write_text(
        json.dumps(root_metadata, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        ZarrMetadataEquivalenceError,
        match="metadata|consolidated|stale",
    ):
        _preflight(stale)

    mismatch = _archive(tmp_path / "mismatch")
    direct = zarr.open_group(str(mismatch), mode="a", use_consolidated=False)
    source = direct["analysis/stimulus_response_runs/source"]
    source["global_per_fish/mean_speed_mm_s"][0] = np.float32(999.0)
    zarr.consolidate_metadata(direct.store)
    with pytest.raises(
        ValueError, match="decoded array values differ|reader results differ"
    ):
        _preflight(mismatch)


def test_unsafe_names_outputs_and_symlinks_fail_closed(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    with pytest.raises(ValueError, match="explicit immutable"):
        benchmark._preflight(
            archive,
            source_run_name="latest",
            candidate_run_name="candidate",
            seed=31,
            repetitions=1,
        )
    with pytest.raises(ValueError, match="must differ"):
        benchmark._preflight(
            archive,
            source_run_name="source",
            candidate_run_name="source",
            seed=31,
            repetitions=1,
        )
    with pytest.raises(ValueError, match="benchmark-only"):
        benchmark._safe_output(tmp_path / "results", archive=archive.resolve())
    symlink = archive / "analysis/stimulus_response_runs/linked-run"
    symlink.symlink_to("source", target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        benchmark._safe_persisted_run_path(
            archive,
            run_name="linked-run",
        )


def test_five_repetition_order_is_balanced_and_deterministic() -> None:
    orders = [
        benchmark._trial_order(seed=benchmark.DEFAULT_SEED, repetition_index=index)
        for index in range(5)
    ]
    assert orders == [
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
    ]
