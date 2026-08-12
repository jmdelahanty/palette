from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fisheye.cluster.subject_masks import full_duration_canary as canary
from fisheye.diagnostics import benchmark_subject_mask_single_clip_inference as bench
from fisheye.shared.gpu_runtime_telemetry import GpuRuntimeTelemetrySampler
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


class _FakeRoot:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}


class _FakeProcess:
    def __init__(self, *, stdout: Any, stderr: Any) -> None:
        stdout.write(
            "2026/08/10 17:00:00.000, 0, GPU-test, NVIDIA L4, "
            "80, 40, 15, 4096, 65, 58, 1600, 6250\n"
        )
        stdout.flush()
        stderr.flush()
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: int) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_plan(tmp_path: Path) -> Path:
    root = tmp_path / ".palette_benchmarks" / "source_canary"
    root.mkdir(parents=True)
    analysis = root / "analysis.zarr"
    analysis.mkdir()
    raw_dimensions = canary.SubjectMaskDimensions(
        n_frames=100,
        n_rois=100,
        n_channels=3,
        roi_height=512,
        roi_width=512,
    )
    refined_dimensions = canary.SubjectMaskDimensions(
        n_frames=100,
        n_rois=100,
        n_channels=4,
        roi_height=512,
        roi_width=512,
    )
    refined_components = canary.SubjectMaskComponentRegistry(
        ("subject_body", "eye_left", "eye_right", "swim_bladder")
    )
    sampled_contour_profile = canary.default_subject_mask_sampled_contour_profile(
        refined_components
    )
    payload: dict[str, Any] = {
        "schema_id": canary.PLAN_SCHEMA_ID,
        "schema_version": 5,
        "status": "planned",
        "classification": canary.BENCHMARK_CLASSIFICATION,
        "created_at_utc": "2026-08-10T20:00:00+00:00",
        "workflow_id": "source_canary",
        "run_root": str(root),
        "repo": {
            "path": "/groups/palette-source",
            "commit": "1" * 40,
            "branch": "HEAD",
            "dirty": False,
        },
        "recording": {
            "recording_id": "recording-1",
            "camera_identity": "camera-1",
            "n_frames": 100,
            "n_rows": 100,
            "video_source": {"window_index_sha256": "2" * 64},
        },
        "references": {
            "analysis_zarr": str(analysis),
            "crop": {
                "source_archive": str(analysis),
                "parent": "crop_runs",
                "run": "crop_v2",
                "manifest": {
                    "schema_id": "palette.crop_geometry.run_manifest",
                    "schema_version": 2,
                    "payload_digest": "3" * 64,
                },
                "copy": {},
            },
            "refined_keypoints": {
                "source_archive": str(analysis),
                "parent": "refined_keypoints_runs",
                "run": "keypoints_v2",
                "manifest": {
                    "schema_id": "palette.keypoint.refined.run_manifest",
                    "schema_version": 2,
                    "payload_digest": "4" * 64,
                },
                "copy": {},
            },
        },
        "model": {
            "path": "/groups/models/model.pt",
            "size_bytes": 100,
            "mtime_ns": 1,
            "device": 1,
            "inode": 2,
            "sha256": "5" * 64,
        },
        "final_layout": {
            "schema_id": "palette.subject_mask.final_layout_work_plan",
            "schema_version": 1,
            "ownership_policy": (
                "complete_final_outer_row_units_per_worker_with_"
                "deterministic_boundary_rebuild_v1"
            ),
            "raw": canary.subject_mask_final_layout_payload_plan(
                kind="raw_probability_uint8",
                dimensions=raw_dimensions,
            ),
            "refined": canary.subject_mask_final_layout_payload_plan(
                kind="refined_dense_core",
                dimensions=refined_dimensions,
            ),
            "sampled_contours": canary.plan_subject_mask_sampled_contour_storage(
                refined_dimensions,
                components=refined_components,
                contour_profile=sampled_contour_profile,
            ).as_manifest(),
        },
        "windows": [
            {
                "window_index": 0,
                "window_id": "clip_000000",
                "row_start": 0,
                "row_stop": 100,
                "row_count": 100,
                "start_frame": 0,
                "end_frame": 100,
                "frame_count": 100,
                "camera_identity": "camera-1",
                "source_video_path": "/groups/video.mp4",
                "source_file": {
                    "path": "/groups/video.mp4",
                    "size_bytes": 1000,
                    "mtime_ns": 2,
                },
                "raw_run": "raw_clip_0",
                "raw_attempt_id": "6" * 36,
                "refined_run": "refined_clip_0",
                "refined_attempt_id": "7" * 36,
            }
        ],
        "outputs": {
            "raw_run": "raw",
            "refined_run": "refined",
            "quality_run": "quality",
            "cache_run": "cache",
            "bundle_id": "bundle",
            "result_path": str(root / "result.json"),
        },
        "execution": {
            "inference": {
                "device": "0",
                "batch_size": 128,
                "probability_dtype": "uint8",
                "inner_chunk_rows": 32,
                "outer_shard_rows": 2048,
            },
            "refinement": {
                "chunk_rows": 256,
                "dense_mask_chunk_rows": 256,
                "workers": 16,
                "metric_level": "cheap",
            },
            "publication": {
                "core_physical_unit_workers": 4,
                "core_validation_mode": (
                    canary.SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE.value
                ),
                "logical_identity_unit_rows": (
                    canary.SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
                ),
                "ownership_policy": (
                    "bounded_threaded_disjoint_whole_physical_row_bands_v1"
                ),
            },
        },
        "safety": {
            "production_registry_used": False,
            "production_selector_mutation_allowed": False,
            "bundle_activation_allowed": False,
            "all_outputs_below_run_root": True,
            "worker_writes_are_node_local_until_atomic_bundle_publish": True,
            "window_rows_are_exact_nonoverlapping_complete": True,
            "final_layout_units_are_selector_ineligible_transport": True,
            "receipt_bound_composable_dense_identity_required": True,
            "finalizer_full_dense_decode_hash_allowed": False,
            "worker_sampled_contours_required": True,
            "full_ragged_contours_allowed": False,
        },
        "inference_reuse": None,
    }
    payload["plan_digest"] = canonical_json_sha256(payload)
    plan_path = root / "plan.json"
    _write_json(plan_path, payload)
    canary.load_plan(plan_path)
    return plan_path


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    source_plan = _source_plan(tmp_path)
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(bench.zarr, "open_group", lambda *_args, **_kwargs: _FakeRoot())
    monkeypatch.setattr(
        bench,
        "_repo_identity",
        lambda path: {
            "path": str(Path(path).resolve()),
            "commit": "8" * 40,
            "branch": "HEAD",
            "dirty": False,
        },
    )
    monkeypatch.setattr(
        bench,
        "_copy_reference_run",
        lambda **kwargs: {
            "source_path": str(kwargs["source_archive"]),
            "destination_path": str(kwargs["target_archive"]),
            "tree": {"files": 1, "bytes": 1},
            "tree_sha256": "9" * 64,
            "duration_seconds": 0.1,
        },
    )
    root = tmp_path / ".palette_benchmarks" / "single_clip_matrix"
    matrix = bench.prepare_matrix(
        source_plan_path=source_plan,
        window_index=0,
        run_root=root,
        palette_repo=repository,
        matrix_id="single_clip_matrix",
        repetitions=3,
        after_job_id="153303424",
    )
    return root / "matrix_manifest.json", matrix


def _telemetry(path: Path) -> None:
    def popen(_command: list[str], **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(stdout=kwargs["stdout"], stderr=kwargs["stderr"])

    sampler = GpuRuntimeTelemetrySampler(
        output_path=path,
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        executable_resolver=lambda _name: "/usr/bin/nvidia-smi",
        popen_factory=popen,
    )
    sampler.start().stop(workload_outcome="success")


def _complete_trials(matrix: dict[str, Any]) -> None:
    content_arrays = {
        "mask_probs_roi": {
            "dtype": "uint8",
            "shape": [100, 3, 32, 32],
            "units_digest": "a" * 64,
        },
        "metrics/area_px": {
            "dtype": "float32",
            "shape": [100, 3],
            "units_digest": "b" * 64,
        },
    }
    for task in matrix["tasks"]:
        bundle = Path(task["bundle_path"])
        run = bundle / "archive.zarr" / task["run_path"]
        profile = bool(task["synchronized_stage_profiling"])
        duration = 120.0 if profile else 100.0
        _write_json(
            run / "zarr.json",
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "profile_timings_enabled": profile,
                    "inference_duration_seconds": duration,
                    "timing_profile": {"enabled": True} if profile else None,
                },
            },
        )
        receipt_payload = {"arrays": content_arrays}
        _write_json(
            run / "worker_semantic_receipt.json",
            {
                "schema_id": "palette.subject_mask.worker_semantic_receipt",
                "schema_version": 1,
                "payload": receipt_payload,
                "payload_digest": canonical_json_sha256(receipt_payload),
            },
        )
        result = {
            "schema_id": canary.WORKER_RESULT_SCHEMA_ID,
            "schema_version": canary.WORKER_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "stage": "inference",
            "plan_digest": task["task_plan_digest"],
            "window_id": task["window_id"],
            "compute_duration_seconds": duration + 60.0,
            "copy_duration_seconds": 2.0,
            "performance_phase_durations_seconds": {
                "reference_archive_stage": 5.0,
                "video_copy": 10.0,
                "model_copy": 1.0,
                "crop_pixel_materialization": 20.0,
                "inference_cli": duration,
                "local_proof": 14.0,
                "final_layout_unit": 10.0,
                "worker_pre_bundle_total": duration + 60.0,
            },
        }
        _write_json(Path(task["result_path"]), result)
        _telemetry(bundle / "performance" / "gpu_runtime.json")


def test_prepare_matrix_balances_candidates_and_freezes_external_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, prepared = _prepare(tmp_path, monkeypatch)

    loaded = bench.load_matrix(matrix_path)
    assert loaded == prepared
    assert len(loaded["tasks"]) == 6
    assert [task["candidate_id"] for task in loaded["tasks"]] == [
        "async_no_synchronized_stage_profile",
        "async_synchronized_stage_profile",
        "async_synchronized_stage_profile",
        "async_no_synchronized_stage_profile",
        "async_no_synchronized_stage_profile",
        "async_synchronized_stage_profile",
    ]
    assert loaded["scheduler"]["after_condition"] == "done(153303424)"
    assert all(Path(task["task_plan_path"]).is_file() for task in loaded["tasks"])


def test_workflow_runs_fresh_trials_serially_after_live_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, _prepared = _prepare(tmp_path, monkeypatch)

    workflow = bench.build_workflow(matrix_path)

    trial_array, aggregate = workflow.jobs
    assert trial_array.execution_group is not None
    assert trial_array.execution_group.max_concurrent == 1
    assert len(trial_array.execution_group.tasks) == 6
    assert trial_array.resources.extra_lsf_args == ("-w", "done(153303424)")
    assert aggregate.dependency is not None
    assert aggregate.dependency.upstream_job_keys == ("single_clip_inference_trials",)


def test_aggregate_requires_exact_output_equality_and_reports_profile_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, prepared = _prepare(tmp_path, monkeypatch)
    _complete_trials(prepared)

    aggregate = bench.aggregate_matrix(matrix_path=matrix_path)

    assert aggregate["correctness"] == {
        "exact_decoded_array_receipt_signatures_equal": True,
        "content_signature_digest": canonical_json_sha256(
            aggregate["trials"][0]["content_signature"]
        ),
        "complete_gpu_telemetry_for_every_trial": True,
    }
    assert aggregate["candidates"]["async_no_synchronized_stage_profile"][
        "median_inference_rows_per_second"
    ] == pytest.approx(1.0)
    assert aggregate["comparisons"][
        "synchronized_profile_throughput_change_percent"
    ] == pytest.approx(-16.6666666667)
    assert aggregate["decision"]["profile_or_writer_promoted"] is False


def test_aggregate_fails_closed_on_cross_trial_payload_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, prepared = _prepare(tmp_path, monkeypatch)
    _complete_trials(prepared)
    changed = prepared["tasks"][-1]
    receipt_path = (
        Path(changed["bundle_path"])
        / "archive.zarr"
        / changed["run_path"]
        / "worker_semantic_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["payload"]["arrays"]["mask_probs_roi"]["units_digest"] = "c" * 64
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    _write_json(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="signatures differ"):
        bench.aggregate_matrix(matrix_path=matrix_path)
