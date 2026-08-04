from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis import bout_kinematics_schema, swim_bout_schema
from fisheye.analysis.detection_occupancy_runs import (
    OccupancyWindow,
    build_detection_occupancy_result,
    build_session_occupancy_result,
    write_detection_occupancy_run,
    write_session_occupancy_run,
)
from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis_workflows.storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
)
from fisheye.analysis_workflows.materializers.exact_tabular_candidate import (
    materialize_exact_tabular_candidate,
)
from fisheye.diagnostics import benchmark_exact_tabular_candidates as benchmark
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _create_array(
    group: zarr.Group,
    path: str,
    dtype: str | None,
    axes: tuple[str, ...],
) -> None:
    parent = group
    parts = path.split("/")
    for name in parts[:-1]:
        parent = parent.require_group(name)
    resolved = np.dtype("S64" if dtype is None else dtype)
    first_extent = (
        2 if axes[0] == "detector_signal" else (7 if axes[0] == "frame" else 3)
    )
    shape = (first_extent,) if len(axes) == 1 else (first_extent, 7)
    values = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    parent.create_array(parts[-1], data=values.astype(resolved))


def _set_columnar_attrs(
    group: zarr.Group,
    specs: dict[str, object],
    table_paths: tuple[str, ...],
) -> None:
    for table_path in table_paths:
        prefix = table_path + "/"
        fields = [
            (path[len(prefix) :], spec)
            for path, spec in specs.items()
            if path.startswith(prefix) and "/" not in path[len(prefix) :]
        ]
        if not fields:
            continue
        table = group[table_path]
        table.attrs["storage_layout"] = "columnar"
        table.attrs["field_names"] = [name for name, _spec in fields]
        table.attrs["field_dtypes"] = {
            name: spec.logical_dtype for name, spec in fields
        }


def _archive(path: Path, *, family: str) -> Path:
    if family in {"detection_occupancy", "session_occupancy"}:
        return _occupancy_archive(path, family=family)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    analysis = root.create_group("analysis")
    if family == "swim_bouts":
        schema = swim_bout_schema
        parent = analysis.create_group("swim_bout_runs")
        required = schema._required_specs()
        table_paths = schema._COLUMNAR_TABLE_PATHS
        attrs = {
            "schema_id": schema.SWIM_BOUT_RUN_SCHEMA_ID,
            "schema_version": schema.SWIM_BOUT_RUN_SCHEMA_VERSION,
            "layout": schema.SWIM_BOUT_LAYOUT,
        }
        writer = schema.write_swim_bout_array_manifest
    else:
        schema = bout_kinematics_schema
        parent = analysis.create_group("bout_kinematics_runs")
        required = schema._required_specs()
        table_paths = schema._COLUMNAR_TABLE_PATHS
        attrs = {
            "schema_id": schema.BOUT_KINEMATICS_RUN_SCHEMA_ID,
            "schema_version": schema.BOUT_KINEMATICS_RUN_SCHEMA_VERSION,
            "layout": schema.BOUT_KINEMATICS_LAYOUT,
        }
        writer = schema.write_bout_kinematics_array_manifest
    parent.attrs.update(
        {
            "latest": "source",
            "latest_complete": "source",
            "palette_completion_epoch": 1,
        }
    )
    run = parent.create_group("source")
    run.attrs.update(
        {
            **attrs,
            "palette_run_name": "source",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "provenance": {"stage": family},
        }
    )
    for spec in required.values():
        _create_array(run, spec.path, spec.dtype, spec.axes)
    _set_columnar_attrs(run, required, table_paths)
    writer(run)
    result = materialize_exact_tabular_candidate(
        path,
        family_id=family,
        source_run="source",
        run_name="candidate",
        scratch_root=path.parent / f"scratch-{family}",
        copy_backend="python",
        apply=True,
    )
    assert result["status"] == "complete"
    return path


def _occupancy_archive(path: Path, *, family: str) -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "occupancy_benchmark",
            "width": 100,
            "height": 80,
            "fps": 10.0,
            "total_frames": 8,
        }
    )
    instances = (
        root.require_group("refined_detect_runs")
        .require_group("refined_1")
        .require_group("instances")
    )
    frames = np.asarray([0, 1, 2, 4, 5, 7], dtype=np.int64)
    centers = np.asarray(
        [
            [10.0, 10.0],
            [75.0, 10.0],
            [10.0, 60.0],
            [75.0, 60.0],
            [50.0, 10.0],
            [10.0, 40.0],
        ],
        dtype=np.float32,
    )
    boxes = np.column_stack(
        [
            centers[:, 0] / np.float32(100.0),
            centers[:, 1] / np.float32(80.0),
            np.full(centers.shape[0], 0.02, dtype=np.float32),
            np.full(centers.shape[0], 0.025, dtype=np.float32),
        ]
    ).astype(np.float32)
    instances.create_array("frame_indices", data=frames)
    instances.create_array("bbox_norm_coords", data=boxes)
    instances.create_array(
        "confidence_scores",
        data=np.linspace(0.5, 1.0, frames.size, dtype=np.float32),
    )
    if family == "detection_occupancy":
        epoch_windows = (
            OccupancyWindow(0, "first", 0, 3, 0.0, 0.4, 0.4),
            OccupancyWindow(1, "second", 4, 7, 0.4, 0.8, 0.4),
        )
        epoch_parent = root.require_group("analysis").require_group(
            "stimulus_epoch_runs"
        )
        epoch_parent.attrs.update(
            {"latest": "epoch_source", "latest_complete": "epoch_source"}
        )
        epoch_run = epoch_parent.create_group("epoch_source")
        epoch_run.attrs.update(
            {
                "schema_id": "palette.stimulus_epoch_run.v2",
                "schema_version": 2,
                "palette_run_completion_status": "complete",
                "stage_selector_eligible": True,
            }
        )
        epoch_arrays = epoch_run.create_group("windows")
        labels = np.zeros((2, 96), dtype=np.uint8)
        for index, label in enumerate(("first", "second")):
            encoded = label.encode("utf-8")
            labels[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        epoch_arrays.create_array("window_id", data=np.asarray([0, 1], dtype=np.int32))
        epoch_arrays.create_array("label_bytes", data=labels)
        epoch_arrays.create_array(
            "start_frame", data=np.asarray([0, 4], dtype=np.int64)
        )
        epoch_arrays.create_array("end_frame", data=np.asarray([3, 7], dtype=np.int64))
        epoch_arrays.create_array(
            "start_time_s", data=np.asarray([0.0, 0.4], dtype=np.float64)
        )
        epoch_arrays.create_array(
            "end_time_s", data=np.asarray([0.4, 0.8], dtype=np.float64)
        )
        epoch_arrays.create_array(
            "duration_s", data=np.asarray([0.4, 0.4], dtype=np.float64)
        )
        result = build_detection_occupancy_result(
            path,
            run_name="source",
            stimulus_epoch_run="epoch_source",
            epoch_windows=epoch_windows,
            detection_path="refined_detect_runs/refined_1/instances",
            bin_size=50,
            smooth_sigma=0.0,
        )
        write_detection_occupancy_run(path, result, write_png=False)
    else:
        result = build_session_occupancy_result(
            path,
            run_name="source",
            detection_path="refined_detect_runs/refined_1/instances",
            bin_size=50,
            smooth_sigma=0.0,
        )
        write_session_occupancy_run(path, result, write_png=False)
    published = materialize_exact_tabular_candidate(
        path,
        family_id=family,
        source_run="source",
        run_name="candidate",
        scratch_root=path.parent / f"scratch-{family}",
        copy_backend="python",
        apply=True,
    )
    assert published["status"] == "complete"
    return path


@pytest.mark.parametrize(
    "family",
    [
        "swim_bouts",
        "bout_kinematics",
        "detection_occupancy",
        "session_occupancy",
    ],
)
def test_preflight_binds_both_families_to_executable_candidate_receipt(
    tmp_path: Path,
    family: str,
) -> None:
    archive = _archive(tmp_path / f"{family}.zarr", family=family)

    result = benchmark._preflight(
        archive,
        family=benchmark._family(family),
        source_run_name="source",
        candidate_run_name="candidate",
        seed=17,
        repetitions=5,
    )

    suite = result["suite"]
    assert suite["payload"]["repetitions"] == 5
    assert suite["payload"]["family_id"] == family
    assert (
        suite["payload"]["storage_plan_receipt"]["payload_digest"]
        == result["candidate_storage_receipt_payload_digest"]
    )
    assert result["source_validation"]["selector_eligible"] is True
    assert result["candidate_validation"]["selector_eligible"] is False


def test_matrix_uses_fresh_processes_and_preserves_archive_metadata(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "swim.zarr", family="swim_bouts")
    output = tmp_path / ".palette_benchmarks" / "swim-read-matrix"

    result = benchmark.run_benchmark_matrix(
        archive,
        family_id="swim_bouts",
        source_run="source",
        candidate_run="candidate",
        output_dir=output,
        cache_state="pytest_uncontrolled_os_cache",
        seed=17,
        repetitions=1,
    )

    benchmark.require_matrix_result(result)
    normalized = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
        "swim_bouts"
    ].validated_matrix_identity(result)
    payload = result["payload"]
    assert payload["correctness"]["all_passed"] is True
    assert payload["archive_read_only_guard"]["unchanged"] is True
    assert payload["balanced_read_matrix_complete"] is False
    assert normalized["balanced_repetitions"] == "failed"
    assert normalized["decoded_equality"] == "passed"
    assert normalized["metadata_equivalence"] == "passed"
    assert normalized["physical_io"] == "unavailable"
    assert normalized["crimson_consumer"] == "not_recorded"
    assert [trial["payload"]["role"] for trial in payload["trials"]] == [
        "candidate",
        "source",
    ]
    assert all(
        trial["payload"]["physical_io"]["transferred_bytes"] is None
        for trial in payload["trials"]
    )
    assert (output / "analysis_benchmark_suite.json").is_file()
    assert (output / "matrix_result.json").is_file()
    assert len(list((output / "trials").glob("*.json"))) == 2


def test_rehashed_candidate_receipt_tampering_fails_closed(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "tampered.zarr", family="swim_bouts")
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate = root["analysis/swim_bout_runs/candidate"]
    receipt = copy.deepcopy(candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR])
    receipt["payload"]["storage_profile"]["target_chunk_bytes"] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt

    with pytest.raises(ValueError, match="Invalid swim_bouts candidate run"):
        benchmark._preflight(
            archive,
            family=benchmark._family("swim_bouts"),
            source_run_name="source",
            candidate_run_name="candidate",
            seed=17,
            repetitions=5,
        )


def test_trial_evidence_rejects_fabricated_physical_transfer_metrics(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "trial.zarr", family="bout_kinematics")
    preflight = benchmark._preflight(
        archive,
        family=benchmark._family("bout_kinematics"),
        source_run_name="source",
        candidate_run_name="candidate",
        seed=17,
        repetitions=1,
    )
    trial = benchmark.run_single_trial(
        archive,
        family_id="bout_kinematics",
        source_run="source",
        candidate_run="candidate",
        role="candidate",
        repetition_index=0,
        order_position=0,
        seed=17,
        cache_state="pytest_uncontrolled_os_cache",
        suite_manifest=preflight["suite"],
    )
    tampered = copy.deepcopy(trial)
    tampered["payload"]["physical_io"]["transferred_bytes"] = 123
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="must not fabricate"):
        benchmark.require_trial_result(tampered)


def test_trial_rejects_stale_consolidated_candidate_metadata(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "stale.zarr", family="bout_kinematics")
    preflight = benchmark._preflight(
        archive,
        family=benchmark._family("bout_kinematics"),
        source_run_name="source",
        candidate_run_name="candidate",
        seed=17,
        repetitions=1,
    )
    direct = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    direct["analysis/bout_kinematics_runs/candidate"].attrs[
        "post_consolidation_tamper"
    ] = True

    with pytest.raises(RuntimeError, match="Direct/consolidated declaration differs"):
        benchmark.run_single_trial(
            archive,
            family_id="bout_kinematics",
            source_run="source",
            candidate_run="candidate",
            role="candidate",
            repetition_index=0,
            order_position=0,
            seed=17,
            cache_state="pytest_uncontrolled_os_cache",
            suite_manifest=preflight["suite"],
        )


def test_alias_and_output_path_safety_fail_before_writing(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "safe.zarr", family="bout_kinematics")

    with pytest.raises(ValueError, match="explicit immutable run name"):
        benchmark.run_benchmark_matrix(
            archive,
            family_id="bout_kinematics",
            source_run="latest",
            candidate_run="candidate",
            output_dir=tmp_path / ".palette_benchmarks" / "alias",
            cache_state="pytest",
            repetitions=1,
        )
    with pytest.raises(ValueError, match="outside the source archive"):
        benchmark.run_benchmark_matrix(
            archive,
            family_id="bout_kinematics",
            source_run="source",
            candidate_run="candidate",
            output_dir=archive / "benchmark-output",
            cache_state="pytest",
            repetitions=1,
        )
    unsafe = tmp_path / "results"
    with pytest.raises(ValueError, match="benchmark-only"):
        benchmark.run_benchmark_matrix(
            archive,
            family_id="bout_kinematics",
            source_run="source",
            candidate_run="candidate",
            output_dir=unsafe,
            cache_state="pytest",
            repetitions=1,
        )
    assert not unsafe.exists()


def test_five_repetition_order_is_deterministic_and_rotated() -> None:
    first = [
        benchmark._trial_order(seed=17, repetition_index=index) for index in range(5)
    ]
    second = [
        benchmark._trial_order(seed=17, repetition_index=index) for index in range(5)
    ]

    assert first == second
    assert first == [
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
    ]
