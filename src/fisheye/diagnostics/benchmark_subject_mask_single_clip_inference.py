"""Controlled single-clip subject-mask inference performance matrix.

The matrix reuses the production-path full-duration canary worker, but gives
every trial an isolated benchmark-only archive and immutable plan.  It is
intended to measure observer overhead and accelerator/runtime behavior without
changing a recording archive, selector, registry row, or scientific authority.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping, Sequence
from uuid import uuid5

import zarr

from fisheye.cluster.clipped_lsf import (
    build_execution_task,
    build_job,
    build_task_group_job,
)
from fisheye.cluster.lsf import (
    LsfExecutionMode,
    LsfResources,
    LsfWorkflow,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.backend import build_ssh_bsub_runner
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
    RUNTIME_USER_TOKEN,
)
from fisheye.cluster.subject_masks.full_duration_canary import (
    DEFAULT_GPU_TELEMETRY_INTERVAL_SECONDS,
    PLAN_SCHEMA_VERSION,
    WORKER_RESULT_SCHEMA_ID,
    WORKER_RESULT_SCHEMA_VERSION,
    _ATTEMPT_NAMESPACE,
    _copy_reference_run,
    _repo_identity,
    _require_benchmark_root,
    _strict_json,
    _write_json_atomic,
    load_plan as load_canary_plan,
)
from fisheye.shared.gpu_runtime_telemetry import (
    GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
    GPU_RUNTIME_TELEMETRY_SCHEMA_ID,
    GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION,
    require_gpu_runtime_telemetry,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

MATRIX_SCHEMA_ID = "palette.subject_mask.single_clip_inference_matrix"
MATRIX_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_ID = "palette.subject_mask.single_clip_inference_aggregate"
AGGREGATE_SCHEMA_VERSION = 1
FAMILY = "subject_mask_single_clip_inference_benchmark"
CLASSIFICATION = "selector_ineligible_single_clip_performance_benchmark"
CANDIDATES = (
    {
        "candidate_id": "async_no_synchronized_stage_profile",
        "synchronized_stage_profiling": False,
    },
    {
        "candidate_id": "async_synchronized_stage_profile",
        "synchronized_stage_profiling": True,
    },
)
PERFORMANCE_PHASES = (
    "reference_archive_stage",
    "video_copy",
    "model_copy",
    "crop_pixel_materialization",
    "inference_cli",
    "local_proof",
    "worker_pre_bundle_total",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    if not safe:
        raise ValueError("Benchmark identifier does not contain a safe component.")
    return safe[:96]


def _require_job_id(value: str | int) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ValueError("External LSF dependency must be one positive job ID.")
    return text


def _write_task_plan(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(payload)
    document.pop("plan_digest", None)
    document["plan_digest"] = canonical_json_sha256(document)
    _write_json_atomic(path, document)
    return document


def _balanced_tasks(repetitions: int) -> list[tuple[int, Mapping[str, Any]]]:
    if repetitions <= 0:
        raise ValueError("Benchmark repetitions must be positive.")
    tasks: list[tuple[int, Mapping[str, Any]]] = []
    for repetition in range(repetitions):
        order = CANDIDATES if repetition % 2 == 0 else tuple(reversed(CANDIDATES))
        tasks.extend((repetition, candidate) for candidate in order)
    return tasks


def prepare_matrix(
    *,
    source_plan_path: Path,
    window_index: int,
    run_root: Path,
    palette_repo: Path,
    matrix_id: str,
    repetitions: int = 3,
    after_job_id: str | int,
    gpu_telemetry_interval_seconds: int = DEFAULT_GPU_TELEMETRY_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Freeze a balanced matrix and copy its exact maintained references."""

    output = _require_benchmark_root(run_root)
    if output.exists():
        raise FileExistsError(output)
    if gpu_telemetry_interval_seconds <= 0:
        raise ValueError("GPU telemetry interval must be positive.")
    source_plan_file = source_plan_path.expanduser().resolve()
    source_plan = load_canary_plan(source_plan_file)
    selected = [
        dict(window)
        for window in source_plan["windows"]
        if int(window["window_index"]) == int(window_index)
    ]
    if len(selected) != 1 or int(selected[0]["row_count"]) <= 0:
        raise ValueError("Benchmark requires one nonempty source window.")
    selected_window = selected[0]
    repo_identity = _repo_identity(palette_repo)
    matrix_name = _safe_component(matrix_id)
    external_job_id = _require_job_id(after_job_id)
    output.mkdir(parents=True)
    for child in ("logs", "status", "trials"):
        (output / child).mkdir()

    task_records: list[dict[str, Any]] = []
    for task_index, (repetition, candidate) in enumerate(
        _balanced_tasks(repetitions), start=1
    ):
        candidate_id = str(candidate["candidate_id"])
        task_id = f"r{repetition + 1:02d}_{candidate_id}"
        task_root = output / "trials" / task_id
        task_root.mkdir(parents=True)
        for child in ("logs", "status", "workers", "bundles", "publish"):
            (task_root / child).mkdir()
        analysis = task_root / "analysis.zarr"
        target_root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
        target_root.attrs.update(
            {
                "recording_id": source_plan["recording"]["recording_id"],
                "benchmark_classification": CLASSIFICATION,
                "stage_selector_eligible": False,
                "source_frame_count": int(source_plan["recording"]["n_frames"]),
            }
        )
        references: dict[str, Any] = {"analysis_zarr": str(analysis)}
        source_analysis = Path(source_plan["references"]["analysis_zarr"])
        for name in ("crop", "refined_keypoints"):
            reference = dict(source_plan["references"][name])
            reference["source_archive"] = str(source_analysis)
            reference["copy"] = _copy_reference_run(
                source_archive=source_analysis,
                parent=str(reference["parent"]),
                run_name=str(reference["run"]),
                target_archive=analysis,
            )
            references[name] = reference

        window = deepcopy(selected_window)
        raw_run = _safe_component(f"subject_mask_perf_{matrix_name}_{task_id}")
        window["raw_run"] = raw_run
        window["raw_attempt_id"] = str(
            uuid5(
                _ATTEMPT_NAMESPACE,
                f"{matrix_name}:{task_id}:{window['window_id']}:raw",
            )
        )
        task_plan = deepcopy(source_plan)
        task_plan.pop("plan_digest", None)
        task_plan.update(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "created_at_utc": _utc_now(),
                "workflow_id": _safe_component(f"{matrix_name}_{task_id}"),
                "run_root": str(task_root),
                "repo": repo_identity,
                "references": references,
                "windows": [window],
                "outputs": {
                    "raw_run": raw_run,
                    "refined_run": _safe_component(f"unused_refined_{task_id}"),
                    "quality_run": _safe_component(f"unused_quality_{task_id}"),
                    "bundle_id": _safe_component(f"unused_bundle_{task_id}"),
                    "result_path": str(task_root / "unused_result.json"),
                },
            }
        )
        inference = task_plan["execution"]["inference"]
        inference.update(
            {
                "synchronized_stage_profiling": bool(
                    candidate["synchronized_stage_profiling"]
                ),
                "gpu_runtime_telemetry": {
                    "enabled": True,
                    "sample_interval_seconds": int(gpu_telemetry_interval_seconds),
                    "schema_id": GPU_RUNTIME_TELEMETRY_SCHEMA_ID,
                    "schema_version": GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION,
                    "identity_policy": GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
                },
            }
        )
        task_plan["safety"].update(
            {
                "production_registry_used": False,
                "production_selector_mutation_allowed": False,
                "bundle_activation_allowed": False,
                "all_outputs_below_run_root": True,
            }
        )
        task_plan_path = task_root / "plan.json"
        task_plan = _write_task_plan(task_plan_path, task_plan)
        loaded = load_canary_plan(task_plan_path)
        if loaded["plan_digest"] != task_plan["plan_digest"]:
            raise RuntimeError("Prepared task plan failed an exact reload.")
        result_path = (
            task_root
            / "bundles"
            / "inference"
            / str(window["window_id"])
            / "result.json"
        )
        task_records.append(
            {
                "task_index": task_index,
                "task_id": task_id,
                "candidate_id": candidate_id,
                "repetition": repetition + 1,
                "synchronized_stage_profiling": bool(
                    candidate["synchronized_stage_profiling"]
                ),
                "task_plan_path": str(task_plan_path),
                "task_plan_digest": task_plan["plan_digest"],
                "window_index": int(window["window_index"]),
                "window_id": str(window["window_id"]),
                "row_count": int(window["row_count"]),
                "run_path": f"subject_mask_shard_runs/{raw_run}",
                "result_path": str(result_path),
                "bundle_path": str(result_path.parent),
            }
        )

    payload: dict[str, Any] = {
        "schema_id": MATRIX_SCHEMA_ID,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "status": "planned",
        "classification": CLASSIFICATION,
        "created_at_utc": _utc_now(),
        "matrix_id": matrix_name,
        "run_root": str(output),
        "palette_repo": repo_identity,
        "source": {
            "plan_path": str(source_plan_file),
            "plan_digest": source_plan["plan_digest"],
            "worker_palette_commit": source_plan["repo"]["commit"],
            "window_index": int(window_index),
            "window_id": str(selected_window["window_id"]),
            "row_count": int(selected_window["row_count"]),
            "model": source_plan["model"],
        },
        "candidate_order_policy": "alternating_balanced_fresh_process_v1",
        "candidates": [dict(value) for value in CANDIDATES],
        "repetitions": int(repetitions),
        "tasks": task_records,
        "scheduler": {
            "after_job_id": external_job_id,
            "after_condition": f"done({external_job_id})",
            "max_concurrent": 1,
            "queue": "gpu_l4",
            "fresh_process_per_trial": True,
        },
        "outputs": {
            "aggregate_path": str(output / "aggregate.json"),
            "lsf_plan_path": str(output / "lsf_plan.json"),
            "submission_path": str(output / "submission.json"),
        },
        "safety": {
            "benchmark_only": True,
            "selector_eligible": False,
            "production_archive_mutation_allowed": False,
            "registry_mutation_allowed": False,
            "candidate_difference_only_synchronized_stage_profiling": True,
        },
    }
    payload["payload_digest"] = canonical_json_sha256(payload)
    _write_json_atomic(output / "matrix_manifest.json", payload)
    return payload


def load_matrix(path: Path) -> dict[str, Any]:
    matrix_path = path.expanduser().resolve()
    payload = _strict_json(matrix_path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_id") != MATRIX_SCHEMA_ID
        or payload.get("schema_version") != MATRIX_SCHEMA_VERSION
        or payload.get("status") != "planned"
        or payload.get("classification") != CLASSIFICATION
    ):
        raise ValueError("Unsupported single-clip inference matrix.")
    digest = payload.pop("payload_digest", None)
    observed = canonical_json_sha256(payload)
    payload["payload_digest"] = digest
    if digest != observed:
        raise ValueError("Single-clip inference matrix digest differs.")
    run_root = _require_benchmark_root(Path(str(payload.get("run_root") or "")))
    if matrix_path != run_root / "matrix_manifest.json":
        raise ValueError("Matrix manifest escapes its declared run root.")
    if payload.get("safety") != {
        "benchmark_only": True,
        "selector_eligible": False,
        "production_archive_mutation_allowed": False,
        "registry_mutation_allowed": False,
        "candidate_difference_only_synchronized_stage_profiling": True,
    }:
        raise ValueError("Single-clip matrix safety envelope differs.")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 2 * int(payload["repetitions"]):
        raise ValueError("Single-clip matrix task count differs.")
    if [int(task["task_index"]) for task in tasks] != list(range(1, len(tasks) + 1)):
        raise ValueError("Single-clip matrix task indices are not contiguous.")
    for task in tasks:
        task_plan = load_canary_plan(Path(task["task_plan_path"]))
        if task_plan["plan_digest"] != task["task_plan_digest"]:
            raise ValueError("Matrix task plan binding differs.")
        observed_profile = task_plan["execution"]["inference"][
            "synchronized_stage_profiling"
        ]
        if observed_profile is not task["synchronized_stage_profiling"]:
            raise ValueError("Matrix task profiling policy differs.")
    return payload


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("A percentile requires at least one value.")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _receipt_content_signature(receipt_path: Path) -> dict[str, Any]:
    receipt = _strict_json(receipt_path)
    payload = receipt.get("payload") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_id") != "palette.subject_mask.worker_semantic_receipt"
        or receipt.get("schema_version") != 1
        or not isinstance(payload, Mapping)
        or receipt.get("payload_digest") != canonical_json_sha256(payload)
    ):
        raise ValueError(f"Worker semantic receipt digest differs: {receipt_path}")
    arrays = payload.get("arrays")
    if not isinstance(arrays, Mapping) or "mask_probs_roi" not in arrays:
        raise ValueError("Worker semantic receipt lacks mask_probs_roi evidence.")
    for path, evidence in arrays.items():
        if (
            not isinstance(path, str)
            or not isinstance(evidence, Mapping)
            or not isinstance(evidence.get("dtype"), str)
            or not isinstance(evidence.get("shape"), list)
            or not isinstance(evidence.get("units_digest"), str)
        ):
            raise ValueError("Worker semantic array receipt differs.")
    return {
        str(path): {
            "dtype": evidence["dtype"],
            "shape": evidence["shape"],
            "units_digest": evidence["units_digest"],
        }
        for path, evidence in sorted(arrays.items())
    }


def _trial_result(task: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(task["result_path"])
    result = _strict_json(result_path)
    if (
        not isinstance(result, dict)
        or result.get("schema_id") != WORKER_RESULT_SCHEMA_ID
        or result.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION
        or result.get("status") != "complete"
        or result.get("stage") != "inference"
        or result.get("plan_digest") != task["task_plan_digest"]
    ):
        raise ValueError(f"Benchmark worker result differs: {result_path}")
    bundle = Path(task["bundle_path"])
    run_path = str(task["run_path"])
    run_metadata = _strict_json(bundle / "archive.zarr" / run_path / "zarr.json")
    attrs = run_metadata.get("attributes")
    if not isinstance(attrs, Mapping):
        raise ValueError("Benchmark output run lacks direct attributes.")
    profile_enabled = attrs.get("profile_timings_enabled")
    if profile_enabled is not task["synchronized_stage_profiling"]:
        raise ValueError("Output profiling attribute differs from its candidate.")
    receipt_path = bundle / "archive.zarr" / run_path / "worker_semantic_receipt.json"
    content_signature = _receipt_content_signature(receipt_path)
    telemetry_path = bundle / "performance" / "gpu_runtime.json"
    telemetry = _strict_json(telemetry_path)
    require_gpu_runtime_telemetry(telemetry)
    if telemetry["status"] != "complete":
        raise ValueError("A performance benchmark requires complete GPU telemetry.")
    rows = int(task["row_count"])
    inference_seconds = float(attrs["inference_duration_seconds"])
    compute_seconds = float(result["compute_duration_seconds"])
    phases = result.get("performance_phase_durations_seconds")
    if not isinstance(phases, Mapping) or set(phases) != set(PERFORMANCE_PHASES):
        raise ValueError("Benchmark worker performance phase timings differ.")
    phase_durations = {name: float(phases[name]) for name in PERFORMANCE_PHASES}
    if any(value < 0.0 for value in phase_durations.values()):
        raise ValueError("Benchmark worker performance phase duration is negative.")
    return {
        "task_id": task["task_id"],
        "candidate_id": task["candidate_id"],
        "repetition": int(task["repetition"]),
        "row_count": rows,
        "inference_duration_seconds": inference_seconds,
        "inference_rows_per_second": rows / inference_seconds,
        "worker_compute_duration_seconds": compute_seconds,
        "worker_compute_rows_per_second": rows / compute_seconds,
        "performance_phase_durations_seconds": phase_durations,
        "atomic_bundle_copy_duration_seconds": float(result["copy_duration_seconds"]),
        "profile_timings_enabled": bool(profile_enabled),
        "timing_profile": attrs.get("timing_profile"),
        "gpu_runtime_summary": telemetry["summary"],
        "gpu_runtime_payload_digest": telemetry["payload_digest"],
        "content_signature": content_signature,
        "worker_result_digest": canonical_json_sha256(result),
    }


def aggregate_matrix(*, matrix_path: Path) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    output_path = Path(matrix["outputs"]["aggregate_path"])
    if output_path.exists():
        raise FileExistsError(output_path)
    trials = [_trial_result(task) for task in matrix["tasks"]]
    signatures = {canonical_json_sha256(trial["content_signature"]) for trial in trials}
    if len(signatures) != 1:
        raise RuntimeError("Decoded array receipt signatures differ across trials.")
    candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_id = str(candidate["candidate_id"])
        selected = [trial for trial in trials if trial["candidate_id"] == candidate_id]
        if len(selected) != int(matrix["repetitions"]):
            raise RuntimeError(f"Candidate {candidate_id} lacks complete repetitions.")
        rates = [trial["inference_rows_per_second"] for trial in selected]
        durations = [trial["inference_duration_seconds"] for trial in selected]
        candidates[candidate_id] = {
            "repetitions": len(selected),
            "median_inference_rows_per_second": median(rates),
            "p95_inference_duration_seconds": _percentile(durations, 0.95),
            "median_worker_compute_rows_per_second": median(
                trial["worker_compute_rows_per_second"] for trial in selected
            ),
            "median_atomic_bundle_copy_duration_seconds": median(
                trial["atomic_bundle_copy_duration_seconds"] for trial in selected
            ),
            "median_performance_phase_durations_seconds": {
                name: median(
                    trial["performance_phase_durations_seconds"][name]
                    for trial in selected
                )
                for name in PERFORMANCE_PHASES
            },
            "trial_ids": [trial["task_id"] for trial in selected],
        }
    baseline = candidates["async_no_synchronized_stage_profile"]
    synchronized = candidates["async_synchronized_stage_profile"]
    baseline_rate = float(baseline["median_inference_rows_per_second"])
    synchronized_rate = float(synchronized["median_inference_rows_per_second"])
    result: dict[str, Any] = {
        "schema_id": AGGREGATE_SCHEMA_ID,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "complete",
        "classification": CLASSIFICATION,
        "finished_at_utc": _utc_now(),
        "matrix_manifest_digest": matrix["payload_digest"],
        "source": matrix["source"],
        "trials": trials,
        "candidates": candidates,
        "comparisons": {
            "synchronized_profile_rate_ratio_to_unsynchronized": (
                synchronized_rate / baseline_rate
            ),
            "synchronized_profile_throughput_change_percent": (
                (synchronized_rate / baseline_rate - 1.0) * 100.0
            ),
        },
        "correctness": {
            "exact_decoded_array_receipt_signatures_equal": True,
            "content_signature_digest": next(iter(signatures)),
            "complete_gpu_telemetry_for_every_trial": True,
        },
        "historical_context_not_a_gate": {
            "l4_rows_per_second": 85.1,
            "a6000_rows_per_second_range": [195.0, 214.0],
            "comparison_policy": "context_only_different_hardware_and_runtime",
        },
        "decision": {
            "profile_or_writer_promoted": False,
            "production_state_changed": False,
            "next_step": "use phase attribution to choose a bounded L4/A6000 follow-up",
        },
    }
    result["payload_digest"] = canonical_json_sha256(result)
    _write_json_atomic(output_path, result)
    return result


def build_workflow(matrix_path: Path) -> LsfWorkflow:
    matrix = load_matrix(matrix_path)
    repo = Path(matrix["palette_repo"]["path"])
    run_root = Path(matrix["run_root"])
    scratch_template = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}_"
        f"{RUNTIME_JOB_INDEX_TOKEN}/subject_mask_single_clip_benchmark"
    )
    tasks = [
        build_execution_task(
            run_root=run_root,
            task_key=f"trial:{task['task_id']}",
            stage="subject_mask_single_clip_inference_trial",
            command=(
                "scripts/py",
                "-m",
                "fisheye.cluster.subject_masks.full_duration_canary",
                "inference-worker",
                "--plan",
                str(task["task_plan_path"]),
                "--window-index",
                str(task["window_index"]),
                "--scratch-root",
                scratch_template,
            ),
            expected_outputs=(Path(task["result_path"]),),
            cleanup_paths=(scratch_template,),
            array_indexed=True,
        )
        for task in matrix["tasks"]
    ]
    dependency = str(matrix["scheduler"]["after_condition"])
    trial_array = build_task_group_job(
        workflow_id=str(matrix["matrix_id"]),
        family=FAMILY,
        repo=repo,
        run_root=run_root,
        job_key="single_clip_inference_trials",
        stage="subject_mask_single_clip_inference_trial",
        tasks=tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=1,
        resources=LsfResources(
            queue="gpu_l4",
            ncores=8,
            mem_gb=64,
            gpus=1,
            walltime="2:00",
            extra_lsf_args=("-w", dependency),
        ),
    )
    aggregate_scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "subject_mask_single_clip_aggregate"
    )
    aggregate = build_job(
        workflow_id=str(matrix["matrix_id"]),
        family=FAMILY,
        repo=repo,
        run_root=run_root,
        job_key="single_clip_inference_aggregate",
        stage="subject_mask_single_clip_inference_aggregate",
        command=(
            "scripts/py",
            "-m",
            "fisheye.diagnostics.benchmark_subject_mask_single_clip_inference",
            "aggregate",
            "--matrix",
            str(matrix_path),
        ),
        resources=LsfResources(queue="short", ncores=1, mem_gb=8, walltime="0:30"),
        upstream=("single_clip_inference_trials",),
        expected_outputs=(Path(matrix["outputs"]["aggregate_path"]),),
        cleanup_paths=(aggregate_scratch,),
    )
    return LsfWorkflow(
        workflow_id=str(matrix["matrix_id"]),
        family=FAMILY,
        jobs=(trial_array, aggregate),
        metadata={
            "classification": CLASSIFICATION,
            "external_dependency": dependency,
            "matrix_manifest_digest": matrix["payload_digest"],
        },
    )


def submit_matrix(*, matrix_path: Path, submit_host: str) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    workflow = build_workflow(matrix_path)
    return submit_lsf_workflow(
        workflow,
        cwd=Path(matrix["palette_repo"]["path"]),
        plan_path=Path(matrix["outputs"]["lsf_plan_path"]),
        submission_path=Path(matrix["outputs"]["submission_path"]),
        runner=build_ssh_bsub_runner(submit_host),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-plan", required=True, type=Path)
    prepare.add_argument("--window-index", required=True, type=int)
    prepare.add_argument("--run-root", required=True, type=Path)
    prepare.add_argument("--palette-repo", required=True, type=Path)
    prepare.add_argument("--matrix-id", required=True)
    prepare.add_argument("--repetitions", type=int, default=3)
    prepare.add_argument("--after-job-id", required=True)
    prepare.add_argument(
        "--gpu-telemetry-interval-seconds",
        type=int,
        default=DEFAULT_GPU_TELEMETRY_INTERVAL_SECONDS,
    )
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--matrix", required=True, type=Path)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--matrix", required=True, type=Path)
    submit.add_argument("--submit-host", default="login1-citrus-poller")
    render = subparsers.add_parser("render")
    render.add_argument("--matrix", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        payload = prepare_matrix(
            source_plan_path=args.source_plan,
            window_index=args.window_index,
            run_root=args.run_root,
            palette_repo=args.palette_repo,
            matrix_id=args.matrix_id,
            repetitions=args.repetitions,
            after_job_id=args.after_job_id,
            gpu_telemetry_interval_seconds=args.gpu_telemetry_interval_seconds,
        )
    elif args.command == "aggregate":
        payload = aggregate_matrix(matrix_path=args.matrix)
    elif args.command == "submit":
        payload = submit_matrix(matrix_path=args.matrix, submit_host=args.submit_host)
    else:
        matrix = load_matrix(args.matrix)
        workflow = build_workflow(args.matrix)
        write_json_snapshot(
            Path(matrix["outputs"]["lsf_plan_path"]), workflow.to_json()
        )
        payload = workflow.to_json()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
