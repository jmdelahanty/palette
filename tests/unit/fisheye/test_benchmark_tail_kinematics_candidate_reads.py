from __future__ import annotations

from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis.tail_kinematics_schema import (
    TailKinematicsDimensions,
    stamp_tail_kinematics_array_schema,
    tail_kinematics_array_shapes_and_dtypes,
)
from fisheye.analysis.tail_kinematics_storage import (
    build_tail_kinematics_storage_receipt,
    create_tail_kinematics_arrays_from_receipt,
    persist_tail_kinematics_storage_receipt,
)
from fisheye.diagnostics import benchmark_tail_kinematics_candidate_reads as mod
from fisheye.shared import tail_coordinate_publication as tail_publication
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def _reason_bytes(rows: int) -> np.ndarray:
    result = np.zeros((rows, 64), dtype=np.uint8)
    result[:, :2] = np.frombuffer(b"ok", dtype=np.uint8)
    return result


def _values(dimensions: TailKinematicsDimensions) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, (path, (shape, dtype)) in enumerate(
        sorted(tail_kinematics_array_shapes_and_dtypes(dimensions).items())
    ):
        dtype = np.dtype(dtype)
        if path == "instance_key":
            values = np.arange(1000, 1000 + dimensions.n_rows, dtype=dtype)
        elif path == "source_crop_row_ids":
            values = np.arange(2000, 2000 + dimensions.n_rows, dtype=dtype)
        elif path == "source_acquisition_frame_index":
            values = np.arange(dimensions.n_rows, dtype=dtype)
        elif path == "valid":
            values = np.ones(shape, dtype=dtype)
        elif path == "failure_reason_bytes":
            values = _reason_bytes(dimensions.n_rows)
        elif path == "tail_angle_sample_s":
            values = np.linspace(0.0, 1.0, dimensions.n_tail_samples, dtype=dtype)
        elif path.endswith("row_revision_available"):
            values = np.ones(shape, dtype=dtype)
        elif dtype.kind == "b":
            values = np.ones(shape, dtype=dtype)
        elif dtype.kind in "iu":
            values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        else:
            values = (
                np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10.0
                + np.float32(index)
            ).astype(dtype)
        result[path] = values
    return result


def _parent_and_leaf(group: zarr.Group, path: str) -> tuple[zarr.Group, str]:
    parts = path.split("/")
    parent = group
    for part in parts[:-1]:
        parent = parent.require_group(part)
    return parent, parts[-1]


def _write_source_arrays(group: zarr.Group, values: dict[str, np.ndarray]) -> None:
    for path, data in sorted(values.items()):
        parent, leaf = _parent_and_leaf(group, path)
        chunks = tuple(max(1, min(int(size), 4)) for size in data.shape)
        parent.create_array(leaf, data=data, chunks=chunks, overwrite=False)


def _write_candidate_arrays(
    group: zarr.Group,
    dimensions: TailKinematicsDimensions,
    values: dict[str, np.ndarray],
) -> None:
    receipt = build_tail_kinematics_storage_receipt(
        dimensions, profile=PUBLISHED_HTTP_V1
    )
    create_tail_kinematics_arrays_from_receipt(
        group, receipt=receipt, dimensions=dimensions
    )
    for path, data in values.items():
        node = group
        for part in path.split("/"):
            node = node[part]
        node[:] = data
    stamp_tail_kinematics_array_schema(group, dimensions, byte_planner_adopted=True)
    persist_tail_kinematics_storage_receipt(group, receipt)


def _build_archive(
    path: Path,
    *,
    include_revision_bundle: bool,
) -> tuple[str, str]:
    source_name = "source_tail"
    candidate_name = "candidate_tail"
    dimensions = TailKinematicsDimensions(
        n_rows=12,
        n_tail_samples=5,
        n_components=2 if include_revision_bundle else None,
    )
    values = _values(dimensions)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    parent = root.require_group("analysis").require_group("tail_kinematics_runs")
    parent.attrs.update({"latest": source_name, "latest_complete": source_name})
    source = parent.create_group(source_name)
    candidate = parent.create_group(candidate_name)
    _write_source_arrays(source, values)
    _write_candidate_arrays(candidate, dimensions, values)
    owner = "11111111-1111-4111-8111-111111111111"
    common = {
        "schema_id": "analysis.tail_kinematics_runs",
        "schema_version": 2,
        "palette_run_completion_status": "complete",
        "fps": 700.0,
        "tail_coordinate_publication_kind": "tail_kinematics",
    }
    source.attrs.update({**common, "stage_selector_eligible": True})
    candidate.attrs.update(
        {
            **common,
            "stage_selector_eligible": False,
            "byte_planner_adopted": True,
            "storage_candidate_status": "unpromoted_selector_ineligible",
            "tail_publication_owner_uuid": owner,
            "valid_row_count": dimensions.n_rows,
            "invalid_row_count": 0,
            "output_row_chunk": 4,
            "requested_output_shard_rows": 16,
            "effective_output_shard_rows": 16,
            "output_shard_rows": 16,
            "output_shard_count": 1,
            "completed_worker_task_count": 0,
        }
    )
    validation = {
        "valid": True,
        "errors": [],
        "row_count": dimensions.n_rows,
        "sample_count": dimensions.n_tail_samples,
        "valid_row_count": dimensions.n_rows,
        "invalid_row_count": 0,
        "output_row_chunk": 4,
        "requested_output_shard_rows": 16,
        "effective_output_shard_rows": 16,
        "output_shard_rows": 16,
        "output_shard_count": 1,
        "completed_worker_task_count": 0,
    }
    staged_zarr = path.parent / "scratch" / "source-subset.zarr"
    local_run = staged_zarr / "analysis" / "tail_kinematics_runs" / candidate_name
    canonical_publication = {
        "manifest_ref": "/analysis/subject_shape_runs/shape_fixture/publication_manifest",
        "manifest_sha256": "a" * 64,
        "row_identity_ref": "/analysis/subject_shape_runs/shape_fixture/row_identity",
        "row_identity_sha256": "1" * 64,
        "tail_sample_axis_ref": (
            "/analysis/subject_shape_runs/shape_fixture/tail_sample_axis"
        ),
        "tail_sample_axis_sha256": "2" * 64,
        "tail_curvature_semantics_ref": (
            "/analysis/subject_shape_runs/shape_fixture/tail_curvature_semantics"
        ),
        "tail_curvature_semantics_sha256": "3" * 64,
        "body_frame_ref": "/analysis/subject_shape_runs/shape_fixture/body_frame",
        "body_frame_sha256": "4" * 64,
    }
    source_contract_attrs = {
        "schema_id": "analysis.subject_shape_runs",
        "schema_version": 4,
        "source_refined_subject_masks_run": "refined_masks_fixture",
        "body_frame_schema_id": "fish_anatomical_body_frame",
    }
    allowed_arrays = {
        relative_ref: {
            "array_ref": (f"/analysis/subject_shape_runs/shape_fixture/{relative_ref}"),
            "relative_ref": relative_ref,
            "dtype": "<f4",
            "shape": [dimensions.n_rows],
            "content_sha256": "5" * 64,
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        }
        for relative_ref in mod._ORDERED_STAGED_SOURCE_REFS
        if relative_ref in mod._REQUIRED_STAGED_SOURCE_REFS
    }
    staged_authority_record = {
        "schema_id": mod.TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_ID,
        "schema_version": (mod.TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCHEMA_VERSION),
        "authority_scope": mod.TAIL_KINEMATICS_STAGED_SOURCE_AUTHORITY_SCOPE,
        "source_subject_shape_run": "shape_fixture",
        "source_subject_shape_run_ref": "/analysis/subject_shape_runs/shape_fixture",
        "row_count": dimensions.n_rows,
        "source_sample_count": dimensions.n_tail_samples,
        "canonical_publication": canonical_publication,
        "source_contract_attrs": source_contract_attrs,
        "allowed_arrays": allowed_arrays,
        "closed_array_inventory": True,
        "normal_reader_authority": False,
    }
    staged_authority = {
        **staged_authority_record,
        "record_sha256": mod.canonical_json_sha256(staged_authority_record),
    }
    identity_attrs = {
        "method": mod.TAIL_KINEMATICS_METHOD,
        "method_version": mod.TAIL_KINEMATICS_METHOD_VERSION,
        "row_axis": "observation_instance",
        "source_subject_shape_run": "shape_fixture",
        "source_subject_shape_path": "analysis/subject_shape_runs/shape_fixture",
        "source_subject_shape_publication_manifest_sha256": "a" * 64,
        "source_subject_shape_authority_mode": "canonical_publication",
        "source_subject_shape_authority_sha256": staged_authority["record_sha256"],
        "source_subject_shape_authority": staged_authority,
        "source_refined_subject_masks_run": "refined_masks_fixture",
        "source_refined_subject_masks_revision_snapshot": include_revision_bundle,
        "source_tail_geometry_kind": mod.SOURCE_TAIL_GEOMETRY_KIND,
        "body_frame_convention": "fish_anatomical_body_frame",
        "body_frame_source": "analysis/subject_shape_runs/shape_fixture/body_frame",
        "tail_angle_reference_axis": "caudal_axis=-forward_axis",
        "tail_angle_positive_direction": "anatomical_left",
        "tail_angle_units_primary": "rad",
        "tail_sample_domain": "tail_segment_normalized_arclength",
        "tail_angle_sample_count": dimensions.n_tail_samples,
        "source_geometry_tail_sample_count": dimensions.n_tail_samples,
        "curvature_source": "subject_shape.tail_curvature_px_inv",
        "acquisition_frame_index_source": "source_acquisition_frame_index",
        "row_lineage_copied": list(mod.ROW_LINEAGE_NAMES),
        "row_lineage_missing": [],
        "compute_kernel": mod.TAIL_KINEMATICS_COMPUTE_KERNEL,
        "source_refs": {
            "subject_shape_run": "analysis/subject_shape_runs/shape_fixture",
            "subject_shape_body_component": (
                "analysis/subject_shape_runs/shape_fixture/components/subject_body"
            ),
            "subject_shape_body_frame": (
                "analysis/subject_shape_runs/shape_fixture/body_frame"
            ),
        },
    }
    source.attrs.update(identity_attrs)
    candidate.attrs.update(
        {
            **identity_attrs,
            "source_subject_shape_authority_mode": "digest_bound_staged_subset",
        }
    )
    source_contract = {
        **source_contract_attrs,
        "canonical_publication_manifest_sha256": "a" * 64,
        "staged_source_authority_sha256": staged_authority["record_sha256"],
    }
    source_staging = {
        "schema_id": "palette.tail_kinematics_source_staging.v1",
        "status": "complete",
        "started_at_utc": "2026-08-03T12:00:00+00:00",
        "completed_at_utc": "2026-08-03T12:00:01+00:00",
        "duration_seconds": 1.0,
        "mib_per_second": 1.0,
        "copy_backend": "python",
        "host": "fixture-host",
        "lsb_jobid": None,
        "source_zarr": str(path.resolve()),
        "staged_zarr": str(staged_zarr.resolve()),
        "shape_run": "shape_fixture",
        "row_count": dimensions.n_rows,
        "selected_paths": [
            f"analysis/subject_shape_runs/shape_fixture/{relative_ref}"
            for relative_ref in mod._ORDERED_STAGED_SOURCE_REFS
            if relative_ref in allowed_arrays
        ],
        "source_metadata_sha256": "8" * 64,
        "source_contract": source_contract,
        "staged_source_authority_sha256": staged_authority["record_sha256"],
        "staged_source_authority": staged_authority,
        "inventory": {
            "valid": True,
            "expected_file_count": 1,
            "observed_file_count": 1,
            "expected_bytes": 1,
            "observed_bytes": 1,
            "expected_inventory_sha256": "9" * 64,
            "observed_inventory_sha256": "9" * 64,
            "missing": [],
            "size_mismatches": [],
        },
        "capacity": {
            "check_enabled": False,
            "free_bytes_before_copy": 1,
            "required_bytes_estimate": 2,
            "estimated_output_bytes": 1,
            "margin_bytes": 0,
        },
    }
    parent_snapshot = {mod.PARENT_PATH: dict(parent.attrs)}
    candidate.attrs["cluster_output_staging"] = {
        "staged_zarr": str(staged_zarr.resolve()),
        "source_run_path": str(local_run.resolve()),
        "copy_backend": "python",
        "source_staging": source_staging,
        "schema_id": mod.PUBLISH_SCHEMA_ID,
        "publisher_contract": {
            "schema_id": mod.ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
            "schema_version": mod.ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
        },
        "policy": "node_local_source_and_output_atomic_run_group_publish",
        "serialization_policy": mod.SERIALIZATION_POLICY,
        "rollback_policy": mod._ROLLBACK_POLICY,
        "published_at_utc": "2026-08-03T12:00:02+00:00",
        "host": "fixture-host",
        "lsb_jobid": None,
        "source_zarr": str(path.resolve()),
        "publication_source_run_path": str(local_run.resolve()),
        "target_run_path": str(
            path.resolve() / "analysis" / "tail_kinematics_runs" / candidate_name
        ),
        "publication_owner_attr": tail_publication.TAIL_PUBLICATION_OWNER_ATTR,
        "publication_owner_uuid": owner,
        "failed_public_child_policy": (
            "retain_owner_bound_selector_ineligible_tombstone"
        ),
        "hidden_temporary_policy": "same_parent_hidden_sibling_then_os_replace",
        "copy_duration_seconds": 0.5,
        "physical_copy": {
            "backend": "python",
            "verification": "sha256_all_physical_files",
            "content_sha256": "a" * 64,
            "inventory_sha256": "b" * 64,
            "file_count": 2,
            "physical_bytes": 1024,
        },
        "local_validation": validation,
        "temporary_validation": validation,
        "pre_pointer_validation": validation,
        "final_validation": validation,
        "parent_attrs_before": parent_snapshot,
        "parent_attrs_after": parent_snapshot,
    }
    zarr.consolidate_metadata(root.store)
    return source_name, candidate_name


@pytest.fixture
def fake_consumers(monkeypatch):
    calls = {"source_publication": 0, "candidate_publication": 0, "window": 0}
    source_manifest = SimpleNamespace(record_sha256="a" * 64)

    def source_publication(_root, run_path):
        calls["source_publication"] += 1
        assert run_path.endswith("/source_tail")
        return SimpleNamespace(
            manifest=SimpleNamespace(record_sha256="b" * 64),
            source=SimpleNamespace(manifest=source_manifest),
        )

    def candidate_publication(_root, run_path, **kwargs):
        calls["candidate_publication"] += 1
        assert run_path.endswith("/candidate_tail")
        assert kwargs == {
            "expected_selector_eligible": False,
            "expected_kind": "tail_kinematics",
            "require_complete": True,
        }
        return SimpleNamespace(
            manifest=SimpleNamespace(record_sha256="c" * 64),
            source=SimpleNamespace(manifest=source_manifest),
        )

    def window(root, **parameters):
        calls["window"] += 1
        assert parameters["run_name"] == "source_tail"
        run = root["analysis/tail_kinematics_runs/source_tail"]
        frames = np.asarray(run["source_acquisition_frame_index"][:], dtype=np.int64)
        count = min(frames.size, 4)
        frames = frames[:count]
        scalars = {
            name: np.asarray(run[name][:count], dtype=np.float32)
            for name in parameters["scalar_series"]
        }
        return SimpleNamespace(
            frame_indices=frames,
            time_seconds=frames.astype(np.float64) / 700.0,
            valid=np.asarray(run["valid"][:count], dtype=bool),
            angle_deg=np.asarray(run["tail_angle_deg"][:count], dtype=np.float32),
            scalar_series=scalars,
        )

    monkeypatch.setattr(
        mod.tail_publication,
        "load_tail_kinematics_coordinate_publication",
        source_publication,
    )
    monkeypatch.setattr(
        mod.tail_publication, "_load_tail_coordinate_publication", candidate_publication
    )
    monkeypatch.setattr(mod, "load_tail_kinematics_window", window)
    return calls


@pytest.mark.parametrize(
    ("include_revision_bundle", "expected_count"),
    ((False, 21), (True, 23)),
)
def test_validate_pair_closes_core_and_complete_revision_bundle(
    tmp_path: Path,
    fake_consumers,
    include_revision_bundle: bool,
    expected_count: int,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(
        archive, include_revision_bundle=include_revision_bundle
    )

    receipt = mod.validate_pair(
        archive, source_run_name=source, candidate_run_name=candidate
    )
    payload = mod._require_pair_receipt(receipt)

    assert payload["array_count"] == expected_count
    assert payload["optional_revision_bundle"] is include_revision_bundle
    assert len(payload["logical_hashes"]) == expected_count
    assert payload["selector_eligible_candidate"] is False
    assert payload["profile_promoted"] is False
    assert payload["promotion_authorized"] is False
    assert payload["physical_io"] is None
    assert payload["candidate_consumer"]["public_consumer_implemented"] is False
    assert fake_consumers["source_publication"] == 1
    assert fake_consumers["candidate_publication"] == 1


def test_source_trial_uses_public_reader_candidate_remains_diagnostic(
    tmp_path: Path,
    fake_consumers,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=True)

    source_trial = mod.run_trial(
        archive,
        source_run_name=source,
        candidate_run_name=candidate,
        role="source",
        repetition_index=0,
        seed=7,
        window_rows=4,
        driver_pid=os.getppid(),
    )
    candidate_trial = mod.run_trial(
        archive,
        source_run_name=source,
        candidate_run_name=candidate,
        role="candidate",
        repetition_index=0,
        seed=7,
        window_rows=4,
        driver_pid=os.getppid(),
    )
    source_payload = mod._require_envelope(source_trial, schema_id=mod.TRIAL_SCHEMA_ID)
    candidate_payload = mod._require_envelope(
        candidate_trial, schema_id=mod.TRIAL_SCHEMA_ID
    )
    public_payload = mod._require_envelope(
        source_payload["public_consumer_result"],
        schema_id=mod.PUBLIC_WORKLOAD_SCHEMA_ID,
    )

    assert public_payload["consumer"] == "load_tail_kinematics_window"
    assert candidate_payload["public_consumer_result"] is None
    assert source_payload["workload_result"]["physical_io"] is None
    assert candidate_payload["profile_promoted"] is False
    assert fake_consumers["window"] == 1


def _spawn_trial(
    archive: Path,
    *,
    source: str,
    candidate: str,
    role: str,
    seed: int,
    window_rows: int,
) -> tuple[dict, dict]:
    context = mp.get_context("fork")
    queue = context.Queue()
    driver_pid = os.getpid()

    def target() -> None:
        queue.put(
            mod.run_trial(
                archive,
                source_run_name=source,
                candidate_run_name=candidate,
                role=role,
                repetition_index=0,
                seed=seed,
                window_rows=window_rows,
                driver_pid=driver_pid,
            )
        )

    process = context.Process(target=target)
    process.start()
    result = queue.get(timeout=30)
    process.join(timeout=30)
    assert process.exitcode == 0
    payload = mod._require_envelope(result, schema_id=mod.TRIAL_SCHEMA_ID)
    assert payload["pid"] == process.pid
    return result, {
        "repetition_index": 0,
        "role": role,
        "spawned_pid": process.pid,
        "driver_pid": driver_pid,
        "return_code": 0,
    }


def _matrix_fixture(
    archive: Path,
    *,
    source: str,
    candidate: str,
    seed: int = 11,
    window_rows: int = 3,
) -> dict:
    trials: list[dict] = []
    receipts: list[dict] = []
    for role in ("source", "candidate"):
        trial, receipt = _spawn_trial(
            archive,
            source=source,
            candidate=candidate,
            role=role,
            seed=seed,
            window_rows=window_rows,
        )
        trials.append(trial)
        receipts.append(receipt)
    timing = {
        role: [
            mod._require_envelope(trial, schema_id=mod.TRIAL_SCHEMA_ID)["timing"][
                "workload_seconds"
            ]
            for trial in trials
            if mod._require_envelope(trial, schema_id=mod.TRIAL_SCHEMA_ID)["role"]
            == role
        ]
        for role in ("source", "candidate")
    }
    payload = {
        "benchmark_id": mod.BENCHMARK_ID,
        "family_id": mod.FAMILY_ID,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "archive": str(archive.resolve()),
        "source_run_name": source,
        "candidate_run_name": candidate,
        "repetitions": 1,
        "seed": seed,
        "window_rows": window_rows,
        "driver_pid": os.getpid(),
        "trials": trials,
        "process_receipts": receipts,
        "aggregate": {
            role: {
                "median_workload_seconds": statistics.median(values),
                "trial_count": len(values),
            }
            for role, values in timing.items()
        },
        "environment": {
            "python": "fixture",
            "platform": "fixture",
            "thread_environment": dict(mod.STORAGE_BENCHMARK_THREAD_ENVIRONMENT),
            "palette_git": {},
        },
        "archive_guard": mod._archive_guard(archive.resolve()),
        "physical_io": None,
        "physical_io_measured": False,
        "profile_promoted": False,
        "promotion_authorized": False,
        "promotion_decision": "not_evaluated_diagnostic_only",
    }
    return mod._strict_envelope(mod.MATRIX_SCHEMA_ID, payload)


def _resign_trial(payload: dict) -> dict:
    return mod._strict_envelope(mod.TRIAL_SCHEMA_ID, payload)


def _resign_matrix(payload: dict) -> dict:
    return mod._strict_envelope(mod.MATRIX_SCHEMA_ID, payload)


def test_live_matrix_rejects_coordinated_resigned_workload_subset(
    tmp_path: Path,
    fake_consumers,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=False)
    matrix = _matrix_fixture(archive, source=source, candidate=candidate)
    mod.validate_matrix(matrix, replay_live=True)

    forged = deepcopy(matrix)
    matrix_payload = dict(forged["payload"])
    trial_payload = dict(matrix_payload["trials"][1]["payload"])
    workload_payload = dict(trial_payload["workload"]["payload"])
    workload_payload["accesses"] = workload_payload["accesses"][:-1]
    trial_payload["workload"] = mod._strict_envelope(
        mod.WORKLOAD_SCHEMA_ID, workload_payload
    )
    trial_payload["workload_result"] = {
        **trial_payload["workload_result"],
        "workload_payload_digest": trial_payload["workload"]["payload_digest"],
    }
    matrix_payload["trials"][1] = _resign_trial(trial_payload)
    forged = _resign_matrix(matrix_payload)
    with pytest.raises(ValueError, match="live declaration reconstruction"):
        mod.validate_matrix(forged, replay_live=True)


def test_live_matrix_rejects_coordinated_resigned_pair_receipt(
    tmp_path: Path,
    fake_consumers,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=True)
    matrix = _matrix_fixture(archive, source=source, candidate=candidate)
    matrix_payload = deepcopy(matrix["payload"])
    for index, trial in enumerate(matrix_payload["trials"]):
        trial_payload = deepcopy(trial["payload"])
        pair_payload = deepcopy(trial_payload["pair_validation"]["payload"])
        pair_payload["source_coordinate_publication_sha256"] = "f" * 64
        trial_payload["pair_validation"] = mod._strict_envelope(
            mod.PAIR_SCHEMA_ID, pair_payload
        )
        matrix_payload["trials"][index] = mod._strict_envelope(
            mod.TRIAL_SCHEMA_ID, trial_payload
        )
    forged = mod._strict_envelope(mod.MATRIX_SCHEMA_ID, matrix_payload)
    with pytest.raises(ValueError, match="live pair validation"):
        mod.validate_matrix(forged, replay_live=True)


def test_live_matrix_rejects_coordinated_resigned_scientific_identity(
    tmp_path: Path,
    fake_consumers,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=True)
    matrix = _matrix_fixture(archive, source=source, candidate=candidate)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    forged_refined_run = "forged_refined_masks"
    for run_name in (source, candidate):
        root[mod._run_path(run_name)].attrs[
            "source_refined_subject_masks_run"
        ] = forged_refined_run
    zarr.consolidate_metadata(root.store)

    matrix_payload = deepcopy(matrix["payload"])
    for index, trial in enumerate(matrix_payload["trials"]):
        trial_payload = deepcopy(trial["payload"])
        pair_payload = deepcopy(trial_payload["pair_validation"]["payload"])
        identity = deepcopy(pair_payload["stable_scientific_identity"])
        identity["source_identity"][
            "source_refined_subject_masks_run"
        ] = forged_refined_run
        pair_payload["stable_scientific_identity"] = identity
        pair_payload["stable_scientific_identity_sha256"] = mod.canonical_json_sha256(
            identity
        )
        trial_payload["pair_validation"] = mod._strict_envelope(
            mod.PAIR_SCHEMA_ID, pair_payload
        )
        matrix_payload["trials"][index] = mod._strict_envelope(
            mod.TRIAL_SCHEMA_ID, trial_payload
        )
    forged = mod._strict_envelope(mod.MATRIX_SCHEMA_ID, matrix_payload)
    with pytest.raises(ValueError, match="source authority binding"):
        mod.validate_matrix(forged, replay_live=True)


@pytest.mark.parametrize(
    ("profile_field", "forged_value", "message"),
    (
        (
            "max_payload_objects",
            PUBLISHED_HTTP_V1.max_payload_objects + 1,
            "registered storage profile",
        ),
        ("codec_profile_id", "forged_zstd_fast_v1", "registered storage profile"),
    ),
)
def test_same_id_changed_profile_budget_or_codec_fails_closed(
    tmp_path: Path,
    fake_consumers,
    profile_field: str,
    forged_value: object,
    message: str,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=False)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate_group = root[mod._run_path(candidate)]
    receipt = deepcopy(candidate_group.attrs[mod.ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR])
    assert receipt["payload"]["storage_profile"]["profile_id"] == "published_http_v1"
    receipt["payload"]["storage_profile"][profile_field] = forged_value
    receipt["payload_digest"] = mod.canonical_json_sha256(receipt["payload"])
    candidate_group.attrs[mod.ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    candidate_group.attrs["analysis_storage_plan_payload_sha256"] = receipt[
        "payload_digest"
    ]
    zarr.consolidate_metadata(root.store)
    with pytest.raises(ValueError, match=message):
        mod.validate_pair(archive, source_run_name=source, candidate_run_name=candidate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("order", "identity"),
        ("seed", "identity"),
        ("pid", "process"),
    ),
)
def test_matrix_identity_order_and_pid_tampering_fail_closed(
    tmp_path: Path,
    fake_consumers,
    mutation: str,
    message: str,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=False)
    matrix = _matrix_fixture(archive, source=source, candidate=candidate)
    payload = deepcopy(matrix["payload"])
    if mutation == "order":
        payload["trials"].reverse()
    elif mutation == "seed":
        trial_payload = deepcopy(payload["trials"][0]["payload"])
        trial_payload["seed"] += 1
        payload["trials"][0] = mod._strict_envelope(mod.TRIAL_SCHEMA_ID, trial_payload)
    else:
        payload["process_receipts"][0]["spawned_pid"] += 1000
    forged = mod._strict_envelope(mod.MATRIX_SCHEMA_ID, payload)
    with pytest.raises(ValueError, match=message):
        mod.validate_matrix(forged, replay_live=False)


def test_candidate_receipt_and_partial_revision_bundle_fail_closed(
    tmp_path: Path,
    fake_consumers,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=True)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate_group = root[mod._run_path(candidate)]
    receipt = deepcopy(candidate_group.attrs["cluster_output_staging"])
    receipt["physical_copy"]["verification"] = "not_a_real_verification"
    candidate_group.attrs["cluster_output_staging"] = receipt
    zarr.consolidate_metadata(root.store)
    with pytest.raises(ValueError, match="physical-copy evidence"):
        mod.validate_pair(archive, source_run_name=source, candidate_run_name=candidate)

    repaired = deepcopy(candidate_group.attrs["cluster_output_staging"])
    repaired["physical_copy"]["verification"] = "sha256_all_physical_files"
    repaired["unexpected"] = True
    candidate_group.attrs["cluster_output_staging"] = repaired
    zarr.consolidate_metadata(root.store)
    with pytest.raises(ValueError, match="field set"):
        mod.validate_pair(archive, source_run_name=source, candidate_run_name=candidate)

    del repaired["unexpected"]
    repaired["physical_copy"]["content_sha256"] = "invalid"
    candidate_group.attrs["cluster_output_staging"] = repaired
    zarr.consolidate_metadata(root.store)
    with pytest.raises(ValueError, match="physical-copy evidence"):
        mod.validate_pair(archive, source_run_name=source, candidate_run_name=candidate)

    repaired["physical_copy"]["content_sha256"] = "a" * 64
    candidate_group.attrs["cluster_output_staging"] = repaired
    del root[mod._run_path(source)][
        "source_refined_subject_masks/row_revision_available"
    ]
    zarr.consolidate_metadata(root.store)
    with pytest.raises(ValueError, match="optional bundle is partial"):
        mod.validate_pair(archive, source_run_name=source, candidate_run_name=candidate)


def test_output_guard_and_rotation_are_fail_closed(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    zarr.consolidate_metadata(root.store)
    with pytest.raises(ValueError, match="outside"):
        mod._safe_output(archive.resolve(), archive / "evidence.json")
    archive_alias = tmp_path / "archive-alias.zarr"
    archive_alias.symlink_to(archive, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        mod._safe_archive(archive_alias)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(output_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        mod._safe_output(archive.resolve(), output_alias / "matrix.json")
    guarded_tree = archive / "guarded"
    guarded_tree.mkdir()
    external = tmp_path / "external.bin"
    external.write_bytes(b"external")
    (guarded_tree / "alias.bin").symlink_to(external)
    with pytest.raises(ValueError, match="forbidden symlink"):
        mod._guard_archive_tree(archive.resolve(), "guarded", label="Fixture")
    assert mod._trial_order(0) == ("source", "candidate")
    assert mod._trial_order(1) == ("candidate", "source")


def test_trial_json_is_strict_json(
    tmp_path: Path,
    fake_consumers,
) -> None:
    archive = tmp_path / "fixture.zarr"
    source, candidate = _build_archive(archive, include_revision_bundle=False)
    trial = mod.run_trial(
        archive,
        source_run_name=source,
        candidate_run_name=candidate,
        role="candidate",
        repetition_index=0,
        seed=3,
        window_rows=3,
        driver_pid=os.getppid(),
    )
    encoded = json.dumps(trial, sort_keys=True, allow_nan=False)
    assert json.loads(encoded)["payload"]["physical_io"] is None
