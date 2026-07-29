"""Cross-workflow reduction and gates for canonical detection benchmarks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.shared.zarr.benchmark_matrix import (
    require_storage_benchmark_matrix_manifest,
)


CROSS_WORKFLOW_AGGREGATE_SCHEMA_ID = (
    "palette.canonical_detection_storage_cross_workflow_aggregate"
)
CROSS_WORKFLOW_AGGREGATE_SCHEMA_VERSION = 1
REQUIRED_PRFS_LATENCY_METRICS = (
    "prfs.read_subprocess_seconds",
    "prfs.eager_frame_row_offsets.first.consumer_seconds",
    "prfs.eager_frame_row_offsets.warm.consumer_seconds",
    "prfs.random_frame_slices.first.p95_frame_seconds",
    "prfs.random_frame_slices.warm.p95_frame_seconds",
    "prfs.random_observation_ranges.first.p95_range_seconds",
    "prfs.random_observation_ranges.warm.p95_range_seconds",
    "prfs.sequential_frame_windows.first.consumer_seconds",
    "prfs.sequential_frame_windows.warm.consumer_seconds",
)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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


def _compatibility_projection(matrix: Mapping[str, Any]) -> dict[str, object]:
    return {
        "seed": matrix.get("seed"),
        "scales": matrix.get("scales"),
        "workloads": matrix.get("workloads"),
        "candidates": matrix.get("candidates"),
        "correctness_gates": matrix.get("correctness_gates"),
        "performance_tolerances": matrix.get("performance_tolerances"),
    }


def _peak_rss_by_observation(
    *,
    workflow_root: Path,
    matrix: Mapping[str, Any],
) -> dict[tuple[str, int], float]:
    peaks: dict[tuple[str, int], float] = {}
    for repetition in matrix["repetitions"]:
        scale_id = str(repetition["scale_id"])
        repetition_index = int(repetition["repetition_index"])
        block_path = (
            workflow_root
            / "reports"
            / "blocks"
            / f"{scale_id}_repetition_{repetition_index:03d}.json"
        )
        block = _read_json(block_path)
        if (
            block.get("status") != "complete"
            or block.get("fixture_unchanged") is not True
        ):
            raise RuntimeError(f"Benchmark block is not complete: {block_path}")
        expected_ids = [
            str(trial["candidate_id"]) for trial in repetition["trials"]
        ]
        records = block.get("candidates")
        if not isinstance(records, list):
            raise ValueError(f"Benchmark block lacks candidates: {block_path}")
        if [str(record["candidate_id"]) for record in records] != expected_ids:
            raise RuntimeError(f"Benchmark block order drifted: {block_path}")
        for record in records:
            candidate_id = str(record["candidate_id"])
            local_report_path = Path(str(record["local_write_report"])).resolve()
            if (
                not local_report_path.is_relative_to(workflow_root)
                or not local_report_path.is_file()
            ):
                raise RuntimeError(
                    f"Local write evidence is missing or unsafe: {local_report_path}"
                )
            local_report = _read_json(local_report_path)
            values = [
                float(envelope["summary"]["peak_rss_bytes"])
                for envelope in local_report["common_benchmark_envelopes"]
                if envelope.get("phase") == "write"
                and envelope.get("summary", {}).get("peak_rss_bytes") is not None
            ]
            if not values:
                raise ValueError(
                    f"Local write evidence lacks peak RSS: {local_report_path}"
                )
            peaks[(candidate_id, repetition_index)] = max(values)
    return peaks


def _require_metric(
    reduction: Mapping[str, Any],
    metric_name: str,
    statistic: str,
) -> float:
    metrics = reduction.get("metric_reductions")
    if not isinstance(metrics, Mapping):
        raise ValueError("Candidate reduction lacks metrics.")
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping) or statistic not in metric:
        raise ValueError(
            f"Candidate reduction lacks {metric_name}.{statistic}."
        )
    return float(metric[statistic])


def _ratio_check(
    *,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    metric_name: str,
    statistic: str,
    maximum_ratio: float,
) -> dict[str, object]:
    candidate_value = _require_metric(candidate, metric_name, statistic)
    control_value = _require_metric(control, metric_name, statistic)
    if control_value <= 0:
        raise ValueError(f"Control metric must be positive: {metric_name}")
    ratio = candidate_value / control_value
    return {
        "metric": metric_name,
        "statistic": statistic,
        "candidate_value": candidate_value,
        "control_value": control_value,
        "ratio_to_control": ratio,
        "maximum_ratio": float(maximum_ratio),
        "passed": ratio <= float(maximum_ratio),
    }


def evaluate_detection_candidate_gates(
    candidate_reductions: Sequence[Mapping[str, Any]],
    *,
    performance_tolerances: Mapping[str, Any],
) -> dict[str, object]:
    """Apply the predeclared detection reduction metrics and ratios."""

    control_label = str(performance_tolerances["control"])
    controls = [
        reduction
        for reduction in candidate_reductions
        if reduction.get("request", {}).get("label") == control_label
    ]
    if len(controls) != 1:
        raise ValueError("Reduction requires exactly one declared control.")
    control = controls[0]
    median_write_limit = float(
        performance_tolerances["max_median_write_time_ratio_to_control"]
    )
    median_publish_limit = float(
        performance_tolerances["max_median_publish_time_ratio_to_control"]
    )
    median_read_limit = float(
        performance_tolerances[
            "max_median_required_read_latency_ratio_to_control"
        ]
    )
    p95_read_limit = float(
        performance_tolerances[
            "max_p95_required_read_latency_ratio_to_control"
        ]
    )
    peak_rss_limit = float(
        performance_tolerances["max_peak_rss_ratio_to_control"]
    )

    results: list[dict[str, object]] = []
    for candidate in candidate_reductions:
        scalar_checks = [
            _ratio_check(
                candidate=candidate,
                control=control,
                metric_name="local.pipeline_seconds",
                statistic="median",
                maximum_ratio=median_write_limit,
            ),
            _ratio_check(
                candidate=candidate,
                control=control,
                metric_name="publication.total_seconds",
                statistic="median",
                maximum_ratio=median_publish_limit,
            ),
            _ratio_check(
                candidate=candidate,
                control=control,
                metric_name="local.peak_rss_bytes",
                statistic="median",
                maximum_ratio=peak_rss_limit,
            ),
        ]
        median_read_checks = [
            _ratio_check(
                candidate=candidate,
                control=control,
                metric_name=metric_name,
                statistic="median",
                maximum_ratio=median_read_limit,
            )
            for metric_name in REQUIRED_PRFS_LATENCY_METRICS
        ]
        p95_read_checks = [
            _ratio_check(
                candidate=candidate,
                control=control,
                metric_name=metric_name,
                statistic="p95",
                maximum_ratio=p95_read_limit,
            )
            for metric_name in REQUIRED_PRFS_LATENCY_METRICS
        ]
        passed = all(
            bool(check["passed"])
            for check in scalar_checks + median_read_checks + p95_read_checks
        )
        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "request": candidate["request"],
                "passed": passed,
                "scalar_checks": scalar_checks,
                "median_required_read_latency_checks": median_read_checks,
                "p95_required_read_latency_checks": p95_read_checks,
            }
        )
    return {
        "schema_id": "palette.canonical_detection_storage_reduction_gates",
        "schema_version": 1,
        "control_candidate_id": control["candidate_id"],
        "control_request_label": control_label,
        "required_prfs_latency_metrics": list(REQUIRED_PRFS_LATENCY_METRICS),
        "candidates": results,
    }


def reduce_detection_benchmark_workflows(
    *,
    workflow_roots: Sequence[Path],
    benchmark_root: Path,
    output: Path,
) -> dict[str, object]:
    """Combine nonoverlapping compatible workflow aggregates and apply gates."""

    if not workflow_roots:
        raise ValueError("Reduction requires at least one workflow.")
    benchmark = benchmark_root.expanduser().resolve()
    required_output_root = (
        benchmark / "canonical_detection_storage" / "reductions"
    ).resolve()
    output_path = output.expanduser().resolve()
    if (
        output_path == required_output_root
        or not output_path.is_relative_to(required_output_root)
    ):
        raise ValueError(
            f"Cross-workflow output must be below {required_output_root}."
        )
    if output_path.exists():
        raise FileExistsError(f"Cross-workflow output exists: {output_path}")

    source_manifests: list[dict[str, object]] = []
    baseline_projection: dict[str, object] | None = None
    baseline_matrix: dict[str, Any] | None = None
    observations_by_candidate: dict[str, list[dict[str, object]]] = {}
    candidate_models: dict[str, Mapping[str, Any]] = {}
    seen_blocks: set[tuple[str, int]] = set()
    benchmark_commits: set[str] = set()
    fixture_ids: set[str] = set()
    fixture_manifest_digests: set[str] = set()

    for raw_root in workflow_roots:
        root = raw_root.expanduser().resolve()
        if not root.is_relative_to(benchmark):
            raise ValueError(f"Workflow is outside the benchmark root: {root}")
        matrix_path = root / "matrix.json"
        aggregate_path = root / "aggregate.json"
        matrix = _read_json(matrix_path)
        require_storage_benchmark_matrix_manifest(matrix)
        aggregate = _read_json(aggregate_path)
        if (
            aggregate.get("schema_id")
            != "palette.canonical_detection_storage_benchmark_aggregate"
            or aggregate.get("schema_version") != 2
            or aggregate.get("status") != "complete"
        ):
            raise ValueError(f"Workflow lacks a complete v2 aggregate: {root}")
        if (
            aggregate.get("matrix_fingerprint")
            != matrix.get("matrix_fingerprint")
        ):
            raise RuntimeError(f"Workflow matrix fingerprint drifted: {root}")
        projection = _compatibility_projection(matrix)
        if baseline_projection is None:
            baseline_projection = projection
            baseline_matrix = matrix
            candidate_models = {
                str(candidate["candidate_id"]): candidate
                for candidate in matrix["candidates"]
            }
            observations_by_candidate = {
                candidate_id: [] for candidate_id in candidate_models
            }
        elif projection != baseline_projection:
            raise RuntimeError(f"Benchmark workflow is not compatible: {root}")

        peaks = _peak_rss_by_observation(
            workflow_root=root,
            matrix=matrix,
        )
        for block in aggregate["blocks"]:
            block_key = (str(block["scale_id"]), int(block["repetition_index"]))
            if block_key in seen_blocks:
                raise RuntimeError(f"Duplicate benchmark block: {block_key!r}")
            seen_blocks.add(block_key)
            expected_block_path = (
                root
                / "reports"
                / "blocks"
                / f"{block_key[0]}_repetition_{block_key[1]:03d}.json"
            ).resolve()
            reported_block_path = Path(str(block["block_report"])).resolve()
            if (
                reported_block_path != expected_block_path
                or not reported_block_path.is_file()
            ):
                raise RuntimeError(
                    "Aggregate block evidence is missing or unsafe: "
                    f"{reported_block_path}"
                )
            block_report = _read_json(reported_block_path)
            benchmark_commits.add(str(block_report["palette"]["commit"]))
            fixture_ids.add(str(block_report["fixture_id"]))
            fixture_manifest_digests.add(
                str(block_report["fixture_manifest_sha256"])
            )

        for reduction in aggregate["candidate_reductions"]:
            candidate_id = str(reduction["candidate_id"])
            if candidate_id not in candidate_models:
                raise RuntimeError(f"Unknown candidate reduction: {candidate_id}")
            candidate = candidate_models[candidate_id]
            if (
                reduction.get("physical_fingerprint")
                != candidate.get("physical_fingerprint")
                or reduction.get("request") != candidate.get("request")
            ):
                raise RuntimeError(f"Candidate reduction drifted: {candidate_id}")
            for raw_observation in reduction["observations"]:
                observation = dict(raw_observation)
                repetition_index = int(observation["repetition_index"])
                if any(
                    int(existing["repetition_index"]) == repetition_index
                    for existing in observations_by_candidate[candidate_id]
                ):
                    raise RuntimeError(
                        f"Duplicate candidate repetition: {candidate_id} "
                        f"repetition {repetition_index}"
                    )
                metrics = dict(observation["metrics"])
                metrics["local.peak_rss_bytes"] = peaks[
                    (candidate_id, repetition_index)
                ]
                observation["metrics"] = metrics
                observation["source_workflow"] = str(root)
                observations_by_candidate[candidate_id].append(observation)

        source_manifests.append(
            {
                "workflow_root": str(root),
                "matrix": str(matrix_path),
                "matrix_fingerprint": matrix["matrix_fingerprint"],
                "aggregate": str(aggregate_path),
                "repetition_indices": sorted(
                    int(block["repetition_index"])
                    for block in aggregate["blocks"]
                ),
            }
        )

    if baseline_matrix is None:  # pragma: no cover - roots length is validated
        raise RuntimeError("Cross-workflow reduction did not load a matrix.")
    if len(benchmark_commits) != 1:
        raise RuntimeError("Benchmark workflows used different Palette commits.")
    if len(fixture_ids) != 1 or len(fixture_manifest_digests) != 1:
        raise RuntimeError("Benchmark workflows used different fixtures.")

    candidate_reductions: list[dict[str, object]] = []
    for candidate_id, observations in observations_by_candidate.items():
        observations.sort(key=lambda item: int(item["repetition_index"]))
        if not observations:
            raise RuntimeError(f"Candidate has no observations: {candidate_id}")
        candidate = candidate_models[candidate_id]
        scale_id = str(candidate["scale_id"])
        expected_repetition_indices = sorted(
            repetition_index
            for block_scale_id, repetition_index in seen_blocks
            if block_scale_id == scale_id
        )
        actual_repetition_indices = [
            int(observation["repetition_index"])
            for observation in observations
        ]
        if actual_repetition_indices != expected_repetition_indices:
            raise RuntimeError(
                f"Candidate repetition coverage drifted: {candidate_id}"
            )
        metric_names = set(observations[0]["metrics"])
        if any(
            set(observation["metrics"]) != metric_names
            for observation in observations
        ):
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
        candidate_reductions.append(
            {
                "candidate_id": candidate_id,
                "physical_fingerprint": candidate["physical_fingerprint"],
                "request": candidate["request"],
                "repetition_count": len(observations),
                "repetition_indices": [
                    int(observation["repetition_index"])
                    for observation in observations
                ],
                "observations": observations,
                "metric_reductions": reductions,
            }
        )

    minimum_repetitions = int(
        baseline_matrix["performance_tolerances"][
            "minimum_balanced_repetitions_for_reduction"
        ]
    )
    balanced_repetition_count = min(
        int(reduction["repetition_count"])
        for reduction in candidate_reductions
    )
    gates = None
    selection: dict[str, object]
    if balanced_repetition_count < minimum_repetitions:
        selection = {
            "performed": False,
            "reason": "insufficient_balanced_repetitions",
            "balanced_repetition_count": balanced_repetition_count,
            "minimum_balanced_repetitions": minimum_repetitions,
            "selected_candidate_id": None,
        }
    else:
        gates = evaluate_detection_candidate_gates(
            candidate_reductions,
            performance_tolerances=baseline_matrix["performance_tolerances"],
        )
        passing_ids = [
            str(candidate["candidate_id"])
            for candidate in gates["candidates"]
            if candidate["passed"] is True
        ]
        reductions_by_id = {
            str(reduction["candidate_id"]): reduction
            for reduction in candidate_reductions
        }
        selected_id = (
            min(
                passing_ids,
                key=lambda candidate_id: (
                    _require_metric(
                        reductions_by_id[candidate_id],
                        "physical.payload_object_count",
                        "median",
                    ),
                    _require_metric(
                        reductions_by_id[candidate_id],
                        "prfs.read_subprocess_seconds",
                        "median",
                    ),
                    candidate_id,
                ),
            )
            if passing_ids
            else None
        )
        selection = {
            "performed": True,
            "reason": (
                "fewest_objects_then_prfs_reader_wall_time_among_gate_passing_candidates"
                if selected_id is not None
                else "no_candidate_passed_all_reduction_gates"
            ),
            "balanced_repetition_count": balanced_repetition_count,
            "minimum_balanced_repetitions": minimum_repetitions,
            "passing_candidate_ids": passing_ids,
            "selected_candidate_id": selected_id,
            "profile_promoted": False,
            "next_stage_only": True,
        }

    result = {
        "schema_id": CROSS_WORKFLOW_AGGREGATE_SCHEMA_ID,
        "schema_version": CROSS_WORKFLOW_AGGREGATE_SCHEMA_VERSION,
        "status": "complete",
        "sources": source_manifests,
        "compatibility": {
            "exact_projection_match": True,
            "benchmark_commit": next(iter(benchmark_commits)),
            "fixture_id": next(iter(fixture_ids)),
            "fixture_manifest_sha256": next(iter(fixture_manifest_digests)),
            "block_keys": [
                {"scale_id": scale_id, "repetition_index": repetition_index}
                for scale_id, repetition_index in sorted(seen_blocks)
            ],
        },
        "gate_contract": {
            "control": baseline_matrix["performance_tolerances"]["control"],
            "required_prfs_latency_metrics": list(
                REQUIRED_PRFS_LATENCY_METRICS
            ),
            "performance_tolerances": baseline_matrix[
                "performance_tolerances"
            ],
            "metadata_open_timings_reported_but_not_reduction_gates": True,
        },
        "candidate_reductions": candidate_reductions,
        "gates": gates,
        "selection": selection,
        "summary": {
            "source_workflow_count": len(source_manifests),
            "block_count": len(seen_blocks),
            "candidate_count": len(candidate_reductions),
            "balanced_repetition_count": balanced_repetition_count,
            "profile_promoted": False,
            "registry_updates": 0,
            "selector_updates": 0,
            "training_artifacts": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_snapshot(output_path, result)
    return result


__all__ = [
    "CROSS_WORKFLOW_AGGREGATE_SCHEMA_ID",
    "CROSS_WORKFLOW_AGGREGATE_SCHEMA_VERSION",
    "REQUIRED_PRFS_LATENCY_METRICS",
    "evaluate_detection_candidate_gates",
    "reduce_detection_benchmark_workflows",
]
