from __future__ import annotations

import json
from pathlib import Path

from fisheye.shared.zarr.benchmark_matrix import (
    BenchmarkLayout,
    BenchmarkScale,
    StorageCandidateRequest,
)
from fisheye.shared.zarr.detection_benchmark_matrix import (
    INITIAL_DETECTION_PERFORMANCE_TOLERANCES,
    plan_canonical_detection_benchmark_matrix,
)
from fisheye.shared.zarr.detection_benchmark_reduction import (
    REQUIRED_PRFS_LATENCY_METRICS,
    evaluate_detection_candidate_gates,
    reduce_detection_benchmark_workflows,
)
from fisheye.shared.zarr.storage_profiles import MIB


def _metric(median: float, p95: float | None = None) -> dict[str, float | int]:
    resolved_p95 = float(median if p95 is None else p95)
    return {
        "count": 5,
        "minimum": float(median),
        "median": float(median),
        "p95": resolved_p95,
        "maximum": resolved_p95,
    }


def _candidate_reduction(
    candidate_id: str,
    label: str,
    *,
    write: float,
    publish: float,
    rss: float,
    read_median: float,
    read_p95: float,
) -> dict[str, object]:
    metrics = {
        "local.pipeline_seconds": _metric(write),
        "publication.total_seconds": _metric(publish),
        "local.peak_rss_bytes": _metric(rss),
    }
    metrics.update(
        {
            metric_name: _metric(read_median, read_p95)
            for metric_name in REQUIRED_PRFS_LATENCY_METRICS
        }
    )
    return {
        "candidate_id": candidate_id,
        "request": {"label": label},
        "metric_reductions": metrics,
    }


def test_detection_reduction_gates_use_predeclared_control_ratios() -> None:
    control = _candidate_reduction(
        "control",
        "regular__chunk_1048576",
        write=10,
        publish=10,
        rss=100,
        read_median=10,
        read_p95=10,
    )
    passing = _candidate_reduction(
        "passing",
        "sharded__passing",
        write=12.5,
        publish=8,
        rss=100,
        read_median=11,
        read_p95=12,
    )
    failing = _candidate_reduction(
        "failing",
        "sharded__failing",
        write=10,
        publish=10,
        rss=100,
        read_median=11.01,
        read_p95=10,
    )

    result = evaluate_detection_candidate_gates(
        (control, passing, failing),
        performance_tolerances=INITIAL_DETECTION_PERFORMANCE_TOLERANCES,
    )

    by_id = {item["candidate_id"]: item for item in result["candidates"]}
    assert by_id["control"]["passed"] is True
    assert by_id["passing"]["passed"] is True
    assert by_id["failing"]["passed"] is False
    failed_metrics = {
        item["metric"]
        for item in by_id["failing"][
            "median_required_read_latency_checks"
        ]
        if item["passed"] is False
    }
    assert failed_metrics == set(REQUIRED_PRFS_LATENCY_METRICS)


def _write_workflow(
    benchmark_root: Path,
    *,
    workflow_id: str,
    repetition_index: int,
    metric_value: float,
    peak_rss: int,
) -> Path:
    workflow = (
        benchmark_root
        / "canonical_detection_storage"
        / "workflows"
        / workflow_id
    )
    matrix = plan_canonical_detection_benchmark_matrix(
        matrix_id=workflow_id,
        scales=(
            BenchmarkScale.from_mapping(
                "frames_5",
                {
                    "n_frames": 5,
                    "n_instances": 4,
                    "source_width": 640,
                    "source_height": 480,
                },
            ),
        ),
        destination_root=workflow / "candidates",
        repetitions=1,
        repetition_start=repetition_index,
        candidate_requests=(
            StorageCandidateRequest(
                layout=BenchmarkLayout.REGULAR,
                target_chunk_bytes=MIB,
            ),
        ),
    ).as_manifest()
    workflow.mkdir(parents=True)
    matrix_path = workflow / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    candidate = matrix["candidates"][0]
    candidate_id = str(candidate["candidate_id"])
    local_write = workflow / "evidence" / f"{candidate_id}.local-write.json"
    local_write.parent.mkdir()
    local_write.write_text(
        json.dumps(
            {
                "common_benchmark_envelopes": [
                    {
                        "phase": "write",
                        "summary": {"peak_rss_bytes": peak_rss},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    block_path = (
        workflow
        / "reports"
        / "blocks"
        / f"frames_5_repetition_{repetition_index:03d}.json"
    )
    block_path.parent.mkdir(parents=True)
    block_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "fixture_unchanged": True,
                "fixture_id": "fixture-v1",
                "fixture_manifest_sha256": "fixture-sha",
                "palette": {"commit": "benchmark-commit"},
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "local_write_report": str(local_write),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    aggregate = {
        "schema_id": "palette.canonical_detection_storage_benchmark_aggregate",
        "schema_version": 2,
        "status": "complete",
        "matrix_fingerprint": matrix["matrix_fingerprint"],
        "blocks": [
            {
                "scale_id": "frames_5",
                "repetition_index": repetition_index,
                "block_report": str(block_path),
            }
        ],
        "candidate_reductions": [
            {
                "candidate_id": candidate_id,
                "physical_fingerprint": candidate["physical_fingerprint"],
                "request": candidate["request"],
                "observations": [
                    {
                        "repetition_index": repetition_index,
                        "position": 0,
                        "metrics": {"example.seconds": metric_value},
                    }
                ],
            }
        ],
    }
    (workflow / "aggregate.json").write_text(
        json.dumps(aggregate),
        encoding="utf-8",
    )
    return workflow


def test_cross_workflow_reducer_combines_nonoverlapping_repetitions(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "benchmarks"
    first = _write_workflow(
        benchmark_root,
        workflow_id="first",
        repetition_index=1,
        metric_value=1.0,
        peak_rss=100,
    )
    second = _write_workflow(
        benchmark_root,
        workflow_id="second",
        repetition_index=2,
        metric_value=3.0,
        peak_rss=200,
    )
    output = (
        benchmark_root
        / "canonical_detection_storage"
        / "reductions"
        / "combined"
        / "aggregate.json"
    )

    result = reduce_detection_benchmark_workflows(
        workflow_roots=(first, second),
        benchmark_root=benchmark_root,
        output=output,
    )

    reduction = result["candidate_reductions"][0]
    assert result["summary"]["block_count"] == 2
    assert result["summary"]["balanced_repetition_count"] == 2
    assert result["selection"]["reason"] == "insufficient_balanced_repetitions"
    assert reduction["repetition_indices"] == [1, 2]
    assert reduction["metric_reductions"]["example.seconds"]["median"] == 2.0
    assert reduction["metric_reductions"]["local.peak_rss_bytes"][
        "median"
    ] == 150.0
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_cross_workflow_reducer_rejects_aggregate_block_path_escape(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "benchmarks"
    workflow = _write_workflow(
        benchmark_root,
        workflow_id="unsafe",
        repetition_index=1,
        metric_value=1.0,
        peak_rss=100,
    )
    aggregate_path = workflow / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    aggregate["blocks"][0]["block_report"] = str(outside)
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    output = (
        benchmark_root
        / "canonical_detection_storage"
        / "reductions"
        / "unsafe"
        / "aggregate.json"
    )
    try:
        reduce_detection_benchmark_workflows(
            workflow_roots=(workflow,),
            benchmark_root=benchmark_root,
            output=output,
        )
    except RuntimeError as error:
        assert "block evidence is missing or unsafe" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("Unsafe aggregate evidence path was accepted.")
