#!/usr/bin/env python3
"""Validate and aggregate completed canonical detection benchmark blocks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.shared.zarr.benchmark_matrix import (
    require_storage_benchmark_matrix_manifest,
)
from fisheye.shared.zarr.detection_benchmark_access import (
    require_detection_consumer_workloads,
)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot reduce an empty metric sequence.")
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "minimum": min(values),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "maximum": max(values),
    }


def _read_trial_seconds(
    report: dict[str, Any],
    *,
    field: str,
    pass_index: int,
) -> float:
    trials = report.get(field)
    if not isinstance(trials, list) or len(trials) <= pass_index:
        raise ValueError(f"Read report lacks {field}[{pass_index}].")
    trial = trials[pass_index]
    if not isinstance(trial, dict):
        raise ValueError(f"Read report {field}[{pass_index}] is not an object.")
    return float(trial["seconds"])


def _consumer_metrics(
    report: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    workloads = report.get("consumer_workloads")
    if not isinstance(workloads, dict):
        raise ValueError("Read report lacks consumer workloads.")
    require_detection_consumer_workloads(workloads)
    metrics: dict[str, float] = {}
    for workload_name in (
        "eager_frame_row_offsets",
        "random_frame_slices",
        "random_observation_ranges",
        "sequential_frame_windows",
    ):
        workload = workloads[workload_name]
        for pass_index, cache_name in ((0, "first"), (1, "warm")):
            trial = workload["passes"][pass_index]
            base = f"{prefix}.{workload_name}.{cache_name}"
            metrics[f"{base}.consumer_seconds"] = float(
                trial["consumer_seconds"]
            )
            metrics[f"{base}.read_seconds"] = float(trial["read_seconds"])
            for field in (
                "p95_frame_seconds",
                "p95_range_seconds",
                "p95_window_seconds",
                "frames_per_second",
            ):
                if trial.get(field) is not None:
                    metrics[f"{base}.{field}"] = float(trial[field])
    return metrics


def _candidate_observation(
    record: dict[str, Any],
    *,
    repetition_index: int,
) -> dict[str, object]:
    local = _read_json(Path(str(record["local_write_report"])).resolve())
    local_reads = _read_json(Path(str(record["local_read_report"])).resolve())
    publication = _read_json(Path(str(record["publication_report"])).resolve())
    prfs = _read_json(Path(str(record["prfs_read_report"])).resolve())
    if local_reads.get("schema_version") != 2 or prfs.get("schema_version") != 2:
        raise ValueError("Candidate read reports must use schema version 2.")
    if not all(bool(item.get("exact")) for item in local_reads.get("arrays", [])):
        raise RuntimeError("Local read-array validation failed.")
    if not all(bool(item.get("exact")) for item in prfs.get("arrays", [])):
        raise RuntimeError("PRFS read-array validation failed.")

    timing = local["timing"]
    physical = local["physical"]
    publication_timing = publication["timing"]
    metrics = {
        "local.pipeline_seconds": float(timing["total_seconds"]),
        "local.candidate_subprocess_seconds": float(record["subprocess_seconds"]),
        "local.read_subprocess_seconds": float(
            record["local_read_subprocess_seconds"]
        ),
        "publication.copy_seconds": float(publication_timing["copy_seconds"]),
        "publication.total_seconds": float(
            publication_timing["publication_seconds"]
        ),
        "physical.payload_object_count": float(
            physical["payload_file_count"]
        ),
        "physical.apparent_bytes": float(physical["apparent_bytes"]),
        "local.direct_open.first_seconds": _read_trial_seconds(
            local_reads,
            field="direct_open_trials",
            pass_index=0,
        ),
        "local.direct_open.warm_seconds": _read_trial_seconds(
            local_reads,
            field="direct_open_trials",
            pass_index=1,
        ),
        "local.consolidated_open.first_seconds": _read_trial_seconds(
            local_reads,
            field="consolidated_open_trials",
            pass_index=0,
        ),
        "local.consolidated_open.warm_seconds": _read_trial_seconds(
            local_reads,
            field="consolidated_open_trials",
            pass_index=1,
        ),
        "prfs.read_subprocess_seconds": float(
            record["prfs_read_subprocess_seconds"]
        ),
        "prfs.direct_open.first_seconds": _read_trial_seconds(
            prfs,
            field="direct_open_trials",
            pass_index=0,
        ),
        "prfs.direct_open.warm_seconds": _read_trial_seconds(
            prfs,
            field="direct_open_trials",
            pass_index=1,
        ),
        "prfs.consolidated_open.first_seconds": _read_trial_seconds(
            prfs,
            field="consolidated_open_trials",
            pass_index=0,
        ),
        "prfs.consolidated_open.warm_seconds": _read_trial_seconds(
            prfs,
            field="consolidated_open_trials",
            pass_index=1,
        ),
    }
    metrics.update(_consumer_metrics(local_reads, prefix="local"))
    metrics.update(_consumer_metrics(prfs, prefix="prfs"))
    return {
        "repetition_index": repetition_index,
        "position": int(record["position"]),
        "metrics": metrics,
    }


def finalize_benchmark(
    *,
    matrix_path: Path,
    workflow_root: Path,
    output: Path,
) -> dict[str, object]:
    matrix = _read_json(matrix_path.expanduser().resolve())
    require_storage_benchmark_matrix_manifest(matrix)
    root = workflow_root.expanduser().resolve()
    output_path = output.expanduser().resolve()
    if not output_path.is_relative_to(root):
        raise ValueError("Aggregate output must be below the workflow root.")
    if output_path.exists():
        raise FileExistsError(f"Aggregate output already exists: {output_path}")

    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in matrix.get("candidates", [])
    }
    block_summaries: list[dict[str, object]] = []
    all_published: list[str] = []
    observations_by_candidate: dict[str, list[dict[str, object]]] = {
        candidate_id: [] for candidate_id in candidates
    }
    for repetition in matrix.get("repetitions", []):
        scale_id = str(repetition["scale_id"])
        repetition_index = int(repetition["repetition_index"])
        block_path = (
            root
            / "reports"
            / "blocks"
            / f"{scale_id}_repetition_{repetition_index:03d}.json"
        )
        block = _read_json(block_path)
        if block.get("status") != "complete" or block.get("fixture_unchanged") is not True:
            raise RuntimeError(f"Benchmark block is not complete and exact: {block_path}")
        expected_ids = [str(trial["candidate_id"]) for trial in repetition["trials"]]
        actual_records = block.get("candidates")
        if not isinstance(actual_records, list):
            raise ValueError(f"Benchmark block lacks candidate records: {block_path}")
        actual_ids = [str(record["candidate_id"]) for record in actual_records]
        if actual_ids != expected_ids:
            raise RuntimeError(f"Benchmark block candidate order mismatch: {block_path}")
        for record in actual_records:
            candidate_id = str(record["candidate_id"])
            if candidate_id not in candidates:
                raise RuntimeError(f"Unknown candidate in block: {candidate_id}")
            if record.get("physical_fingerprint") != candidates[candidate_id].get(
                "physical_fingerprint"
            ):
                raise RuntimeError(f"Candidate fingerprint mismatch: {candidate_id}")
            published = Path(str(record["published_candidate"])).resolve()
            if not published.is_relative_to(root) or not (published / "zarr.json").is_file():
                raise RuntimeError(f"Published candidate is missing or unsafe: {published}")
            for field in (
                "local_write_report",
                "local_read_report",
                "publication_report",
                "prfs_read_report",
            ):
                evidence = Path(str(record[field])).resolve()
                if not evidence.is_relative_to(root) or not evidence.is_file():
                    raise RuntimeError(f"Candidate evidence is missing: {evidence}")
            if record.get("prfs_reads", {}).get("all_exact") is not True:
                raise RuntimeError(f"PRFS read validation failed: {candidate_id}")
            if (
                record.get("prfs_reads", {}).get("consumer_workloads_exact")
                is not True
            ):
                raise RuntimeError(
                    f"PRFS consumer-read validation failed: {candidate_id}"
                )
            observations_by_candidate[candidate_id].append(
                _candidate_observation(
                    record,
                    repetition_index=repetition_index,
                )
            )
            all_published.append(str(published))
        block_summaries.append(
            {
                "scale_id": scale_id,
                "repetition_index": repetition_index,
                "block_report": str(block_path),
                "candidate_count": len(actual_records),
                "total_seconds": block.get("total_seconds"),
            }
        )

    candidate_reductions: list[dict[str, object]] = []
    for candidate_id, observations in observations_by_candidate.items():
        if not observations:
            raise RuntimeError(f"Candidate has no observations: {candidate_id}")
        metric_names = set(observations[0]["metrics"])
        if any(set(observation["metrics"]) != metric_names for observation in observations):
            raise RuntimeError(f"Candidate metric keys drifted: {candidate_id}")
        reductions = {
            metric_name: _metric_summary(
                [
                    float(observation["metrics"][metric_name])
                    for observation in observations
                ]
            )
            for metric_name in sorted(metric_names)
        }
        candidate = candidates[candidate_id]
        candidate_reductions.append(
            {
                "candidate_id": candidate_id,
                "physical_fingerprint": candidate["physical_fingerprint"],
                "request": candidate["request"],
                "repetition_count": len(observations),
                "repetition_indices": [
                    observation["repetition_index"]
                    for observation in observations
                ],
                "observations": observations,
                "metric_reductions": reductions,
            }
        )

    minimum_repetitions = int(
        matrix.get("performance_tolerances", {}).get(
            "minimum_balanced_repetitions_for_reduction",
            1,
        )
    )
    balanced_repetition_count = min(
        item["repetition_count"] for item in candidate_reductions
    )
    aggregate = {
        "schema_id": "palette.canonical_detection_storage_benchmark_aggregate",
        "schema_version": 2,
        "status": "complete",
        "matrix": str(matrix_path.expanduser().resolve()),
        "matrix_fingerprint": matrix.get("matrix_fingerprint"),
        "workflow_root": str(root),
        "blocks": block_summaries,
        "published_candidates": all_published,
        "candidate_reductions": candidate_reductions,
        "selection": {
            "performed": False,
            "reason": (
                "insufficient_balanced_repetitions"
                if balanced_repetition_count < minimum_repetitions
                else "profile_selection_requires_separate_review"
            ),
            "balanced_repetition_count": balanced_repetition_count,
            "minimum_balanced_repetitions": minimum_repetitions,
        },
        "summary": {
            "block_count": len(block_summaries),
            "published_candidate_count": len(all_published),
            "registry_updates": 0,
            "selector_updates": 0,
            "training_artifacts": 0,
            "profile_promoted": False,
            "balanced_repetition_count": balanced_repetition_count,
            "candidate_reduction_count": len(candidate_reductions),
        },
    }
    write_json_snapshot(output_path, aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = finalize_benchmark(
        matrix_path=args.matrix,
        workflow_root=args.workflow_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
