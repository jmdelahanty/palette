"""Benchmark clipped ROI-cache construction concurrency on one NVIDIA GPU.

The benchmark deliberately reuses the production clipped-collection bundle
runner.  Each trial therefore exercises independent PyNvVideoCodec decoder
sessions, node-local cache construction, manifest-last NRS publication, and
the production cache validation path.

Only bounded benchmark payloads are created.  They are removed after their
bundle summaries have been captured unless ``--keep-payloads`` is requested.
Canonical analysis Zarrs and production ROI caches are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "palette.clipped_roi_cache_nvdec_benchmark.v1"
TELEMETRY_FIELDS = (
    "timestamp",
    "gpu_index",
    "gpu_name",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "decoder_utilization_percent",
    "memory_used_mib",
    "power_draw_watts",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _number(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "not supported", "[not supported]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_telemetry(path: Path) -> dict[str, Any]:
    """Summarize headerless ``nvidia-smi --query-gpu`` CSV samples."""

    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8", errors="replace") as stream:
            for values in csv.reader(stream):
                if len(values) < len(TELEMETRY_FIELDS):
                    continue
                row: dict[str, object] = {}
                for field, value in zip(
                    TELEMETRY_FIELDS,
                    values[: len(TELEMETRY_FIELDS)],
                    strict=True,
                ):
                    cleaned = value.strip()
                    row[field] = (
                        cleaned
                        if field in {"timestamp", "gpu_name"}
                        else _number(cleaned)
                    )
                rows.append(row)

    decoder_values = [
        float(row["decoder_utilization_percent"])
        for row in rows
        if row.get("decoder_utilization_percent") is not None
    ]
    gpu_values = [
        float(row["gpu_utilization_percent"])
        for row in rows
        if row.get("gpu_utilization_percent") is not None
    ]
    memory_values = [
        float(row["memory_used_mib"])
        for row in rows
        if row.get("memory_used_mib") is not None
    ]

    def stats(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "p95": None, "max": None}
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": ordered[p95_index],
            "max": max(values),
        }

    return {
        "sample_count": len(rows),
        "decoder_utilization_percent": stats(decoder_values),
        "gpu_utilization_percent": stats(gpu_values),
        "memory_used_mib": stats(memory_values),
    }


def parse_gnu_time(path: Path) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "user_seconds": None,
        "system_seconds": None,
        "cpu_percent": None,
        "max_rss_kib": None,
    }
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        label, separator, value = line.partition(":")
        if not separator:
            continue
        label = label.strip()
        value = value.strip()
        try:
            if label == "User time (seconds)":
                result["user_seconds"] = float(value)
            elif label == "System time (seconds)":
                result["system_seconds"] = float(value)
            elif label == "Percent of CPU this job got":
                result["cpu_percent"] = float(value.rstrip("%"))
            elif label == "Maximum resident set size (kbytes)":
                result["max_rss_kib"] = int(value)
        except ValueError:
            continue
    return result


def summarize_bundle(
    bundle: Mapping[str, Any],
    *,
    trial_seconds: float,
    telemetry: Mapping[str, Any],
    resource_usage: Mapping[str, Any],
) -> dict[str, Any]:
    children = [
        child
        for child in bundle.get("children", [])
        if isinstance(child, Mapping) and child.get("status") == "ok"
    ]
    decoded_frames = 0
    decode_seconds = 0.0
    builder_seconds = 0.0
    publish_seconds = 0.0
    rows = 0
    payload_bytes = 0
    child_metrics: list[dict[str, Any]] = []
    for child in children:
        builder = child.get("builder") if isinstance(child.get("builder"), Mapping) else {}
        timing = builder.get("timing") if isinstance(builder.get("timing"), Mapping) else {}
        publisher = (
            child.get("publisher")
            if isinstance(child.get("publisher"), Mapping)
            else {}
        )
        row_index = (
            child.get("row_index")
            if isinstance(child.get("row_index"), Mapping)
            else {}
        )
        child_frames = int(timing.get("decoded_frames") or 0)
        child_decode_seconds = float(timing.get("decode_seconds_total") or 0.0)
        child_builder_seconds = float(timing.get("duration_seconds") or 0.0)
        child_publish_seconds = float(publisher.get("payload_copy_seconds") or 0.0)
        child_rows = int(row_index.get("row_count") or timing.get("rows") or 0)
        child_bytes = int(child.get("published_bin_size_bytes") or 0)
        decoded_frames += child_frames
        decode_seconds += child_decode_seconds
        builder_seconds += child_builder_seconds
        publish_seconds += child_publish_seconds
        rows += child_rows
        payload_bytes += child_bytes
        child_metrics.append(
            {
                "clip_id": child.get("clip_id"),
                "rows": child_rows,
                "decoded_frames": child_frames,
                "decode_seconds": child_decode_seconds,
                "builder_seconds": child_builder_seconds,
                "publish_seconds": child_publish_seconds,
                "payload_bytes": child_bytes,
                "decode_frames_per_second": (
                    child_frames / child_decode_seconds
                    if child_decode_seconds > 0
                    else None
                ),
            }
        )

    return {
        "bundle_status": bundle.get("status"),
        "host": bundle.get("host"),
        "max_workers": int(bundle.get("max_workers") or 0),
        "requested_child_count": int(bundle.get("requested_child_count") or 0),
        "completed_child_count": int(bundle.get("completed_child_count") or 0),
        "trial_seconds": trial_seconds,
        "rows": rows,
        "payload_bytes": payload_bytes,
        "decoded_frames": decoded_frames,
        "sum_decode_seconds": decode_seconds,
        "sum_builder_seconds": builder_seconds,
        "sum_publish_seconds": publish_seconds,
        "aggregate_rows_per_second": rows / trial_seconds if trial_seconds > 0 else None,
        "aggregate_decoded_frames_per_second": (
            decoded_frames / trial_seconds if trial_seconds > 0 else None
        ),
        "weighted_child_decode_frames_per_second": (
            decoded_frames / decode_seconds if decode_seconds > 0 else None
        ),
        "telemetry": dict(telemetry),
        "resource_usage": dict(resource_usage),
        "children": child_metrics,
    }


def aggregate_trials(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for trial in trials:
        if trial.get("status") != "complete":
            continue
        grouped.setdefault(int(trial["max_workers"]), []).append(trial)

    aggregates: list[dict[str, Any]] = []
    for workers, items in sorted(grouped.items()):
        throughputs = [float(item["aggregate_rows_per_second"]) for item in items]
        frame_rates = [
            float(item["aggregate_decoded_frames_per_second"]) for item in items
        ]
        wall_times = [float(item["trial_seconds"]) for item in items]
        decoder_means = [
            float(value)
            for item in items
            for value in [
                (
                    item.get("telemetry", {})
                    .get("decoder_utilization_percent", {})
                    .get("mean")
                )
            ]
            if value is not None
        ]
        max_rss_values = [
            int(value)
            for item in items
            for value in [item.get("resource_usage", {}).get("max_rss_kib")]
            if value is not None
        ]
        aggregates.append(
            {
                "max_workers": workers,
                "trial_count": len(items),
                "median_trial_seconds": statistics.median(wall_times),
                "median_aggregate_rows_per_second": statistics.median(throughputs),
                "median_aggregate_decoded_frames_per_second": statistics.median(
                    frame_rates
                ),
                "median_decoder_utilization_percent": (
                    statistics.median(decoder_means) if decoder_means else None
                ),
                "max_rss_kib": max(max_rss_values) if max_rss_values else None,
            }
        )
    return aggregates


def recommend_concurrency(aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not aggregates:
        return {
            "fastest_max_workers": None,
            "efficient_max_workers": None,
            "efficiency_threshold_fraction": 0.95,
        }
    fastest = max(
        aggregates, key=lambda item: float(item["median_aggregate_rows_per_second"])
    )
    fastest_rate = float(fastest["median_aggregate_rows_per_second"])
    threshold = 0.95 * fastest_rate
    efficient = min(
        (
            item
            for item in aggregates
            if float(item["median_aggregate_rows_per_second"]) >= threshold
        ),
        key=lambda item: int(item["max_workers"]),
    )
    return {
        "fastest_max_workers": int(fastest["max_workers"]),
        "fastest_median_rows_per_second": fastest_rate,
        "efficient_max_workers": int(efficient["max_workers"]),
        "efficient_median_rows_per_second": float(
            efficient["median_aggregate_rows_per_second"]
        ),
        "efficiency_threshold_fraction": 0.95,
        "interpretation": (
            "efficient_max_workers is the smallest tested concurrency within 5% "
            "of the fastest median end-to-end row throughput"
        ),
    }


def _render_reports(report: Mapping[str, Any], report_path: Path) -> None:
    aggregates = list(report.get("aggregates", []))
    trials = list(report.get("trials", []))

    csv_path = report_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "trial_id",
                "repetition",
                "max_workers",
                "trial_seconds",
                "rows",
                "aggregate_rows_per_second",
                "aggregate_decoded_frames_per_second",
                "decoder_utilization_mean_percent",
                "decoder_utilization_p95_percent",
                "max_rss_kib",
                "payload_cleanup",
            ]
        )
        for trial in trials:
            decoder = (
                trial.get("telemetry", {}).get("decoder_utilization_percent", {})
            )
            usage = trial.get("resource_usage", {})
            writer.writerow(
                [
                    trial.get("trial_id"),
                    trial.get("repetition"),
                    trial.get("max_workers"),
                    trial.get("trial_seconds"),
                    trial.get("rows"),
                    trial.get("aggregate_rows_per_second"),
                    trial.get("aggregate_decoded_frames_per_second"),
                    decoder.get("mean"),
                    decoder.get("p95"),
                    usage.get("max_rss_kib"),
                    trial.get("payload_cleanup"),
                ]
            )

    markdown_path = report_path.with_suffix(".md")
    lines = [
        "# Clipped ROI-cache NVDEC concurrency benchmark",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Host: `{report.get('host')}`",
        f"- GPU inventory: `{report.get('gpu_inventory')}`",
        f"- Bounded rows per clip: `{report.get('limit_rows')}`",
        f"- Clips per trial: `{len(report.get('clip_ids', []))}`",
        "- NVDEC telemetry is the aggregate decoder-utilization value reported by nvidia-smi.",
        "",
        "| Workers | Trials | Median wall (s) | Median rows/s | Median decoded frames/s | Median NVDEC util. (%) | Max RSS (GiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        decoder = item.get("median_decoder_utilization_percent")
        rss = item.get("max_rss_kib")
        lines.append(
            "| {workers} | {trials} | {wall:.3f} | {rows:.3f} | {frames:.3f} | {decoder} | {rss} |".format(
                workers=item["max_workers"],
                trials=item["trial_count"],
                wall=float(item["median_trial_seconds"]),
                rows=float(item["median_aggregate_rows_per_second"]),
                frames=float(item["median_aggregate_decoded_frames_per_second"]),
                decoder=(f"{float(decoder):.2f}" if decoder is not None else "n/a"),
                rss=(f"{int(rss) / 1024 / 1024:.2f}" if rss is not None else "n/a"),
            )
        )
    recommendation = report.get("recommendation", {})
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Fastest tested concurrency: `{recommendation.get('fastest_max_workers')}` workers.",
            f"- Smallest concurrency within 5% of fastest: `{recommendation.get('efficient_max_workers')}` workers.",
            "- Use the efficient value as the default unless queue occupancy or downstream bundle latency makes the absolute fastest value preferable.",
            "",
            "The benchmark uses one L4 allocation and independent decoder processes. It does not infer per-engine utilization; NVIDIA's driver schedules sessions across the four physical NVDEC engines.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _drop_client_cache(paths: Iterable[Path]) -> dict[str, Any]:
    supported = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
    outcomes: list[dict[str, str]] = []
    for path in paths:
        if not supported:
            outcomes.append({"path": str(path), "status": "unsupported"})
            continue
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
            outcomes.append({"path": str(path), "status": "requested"})
        except OSError as exc:
            outcomes.append(
                {"path": str(path), "status": "failed", "error": str(exc)}
            )
    return {
        "method": "posix_fadvise_POSIX_FADV_DONTNEED",
        "scope": "best-effort compute-node client page cache; not server cache",
        "outcomes": outcomes,
    }


def _gpu_inventory() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,pci.bus_id",
        "--format=csv,noheader",
    ]
    return subprocess.check_output(command, text=True, timeout=30).strip()


def _start_telemetry(path: Path, error_path: Path) -> tuple[subprocess.Popen[str], Any, Any]:
    output_stream = path.open("w", encoding="utf-8")
    error_stream = error_path.open("w", encoding="utf-8")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,utilization.decoder,memory.used,power.draw",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    process = subprocess.Popen(
        command,
        stdout=output_stream,
        stderr=error_stream,
        text=True,
    )
    return process, output_stream, error_stream


def _stop_telemetry(process: subprocess.Popen[str], *streams: Any) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    for stream in streams:
        stream.close()


def _bundle_path(trials_root: Path, trial_id: str) -> Path:
    candidates = sorted(
        trials_root.glob(
            f"clipped_collection_flat_roi_cache_bundle_{trial_id}_*/*.bundle.json"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one bundle summary for {trial_id}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _cleanup_trial_payload(public_root: Path, trial_id: str) -> str:
    trial_root = public_root / trial_id
    resolved_root = public_root.resolve(strict=False)
    resolved_trial = trial_root.resolve(strict=False)
    if resolved_root not in resolved_trial.parents:
        raise RuntimeError(f"Unsafe trial cleanup path: {resolved_trial}")
    if trial_root.exists():
        shutil.rmtree(trial_root)
        return "removed"
    return "absent"


def _parse_trial_order(values: Sequence[str], clip_count: int) -> list[list[int]]:
    orders: list[list[int]] = []
    for value in values:
        try:
            order = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid --trial-order: {value}") from exc
        if not order or any(item < 1 or item > clip_count for item in order):
            raise ValueError(
                f"Every concurrency must be between 1 and the {clip_count} selected clips: {value}"
            )
        if len(set(order)) != len(order):
            raise ValueError(f"Concurrency values must be unique within an order: {value}")
        orders.append(order)
    return orders


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Refusing to run the NVDEC benchmark outside an LSF job")
    if len(args.clip_id) != len(args.source_video):
        raise ValueError("Provide exactly one --source-video for every --clip-id")

    zarr_path = args.zarr.resolve()
    recording_frame_index = args.recording_frame_index.resolve()
    bundle_script = args.bundle_script.resolve()
    run_dir = args.run_dir.resolve()
    public_root = args.public_root.resolve()
    source_videos = [path.resolve() for path in args.source_video]
    orders = _parse_trial_order(args.trial_order, len(args.clip_id))
    for path in [zarr_path, recording_frame_index, bundle_script, *source_videos]:
        if not path.exists():
            raise FileNotFoundError(path)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Benchmark run directory is not empty: {run_dir}")
    if public_root.exists() and any(public_root.iterdir()):
        raise FileExistsError(f"Benchmark public root is not empty: {public_root}")

    trials_root = run_dir / "trials"
    telemetry_root = run_dir / "telemetry"
    trials_root.mkdir(parents=True, exist_ok=True)
    telemetry_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "lsb_jobid": os.environ.get("LSB_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": _gpu_inventory(),
        "zarr_path": str(zarr_path),
        "collection_id": args.collection_id,
        "recording_frame_index": str(recording_frame_index),
        "clip_ids": list(args.clip_id),
        "source_videos": [str(path) for path in source_videos],
        "limit_rows": args.limit_rows,
        "gpu_chunk_frames": args.gpu_chunk_frames,
        "trial_orders": orders,
        "public_root": str(public_root),
        "keep_payloads": args.keep_payloads,
        "trials": [],
        "aggregates": [],
        "recommendation": {},
    }
    _atomic_json(report_path, report)

    try:
        for repetition, order in enumerate(orders, start=1):
            for ordinal, workers in enumerate(order, start=1):
                trial_id = f"r{repetition:02d}_o{ordinal:02d}_w{workers:02d}"
                trial_public_root = public_root / trial_id
                telemetry_path = telemetry_root / f"{trial_id}.csv"
                telemetry_error_path = telemetry_root / f"{trial_id}.err"
                stdout_path = run_dir / f"{trial_id}.out"
                stderr_path = run_dir / f"{trial_id}.err"
                time_path = run_dir / f"{trial_id}.time.txt"
                cache_drop = _drop_client_cache(source_videos)
                command = [
                    "/usr/bin/time",
                    "-v",
                    "-o",
                    str(time_path),
                    "bash",
                    str(bundle_script),
                    "--zarr",
                    str(zarr_path),
                    "--collection-id",
                    args.collection_id,
                    "--recording-frame-index",
                    str(recording_frame_index),
                    "--public-cache-dir",
                    str(trial_public_root),
                    "--log-dir",
                    str(trials_root),
                    "--run-id",
                    trial_id,
                    "--run-label",
                    f"{args.benchmark_id}_{trial_id}",
                    "--max-workers",
                    str(workers),
                    "--limit-rows",
                    str(args.limit_rows),
                    "--gpu-chunk-frames",
                    str(args.gpu_chunk_frames),
                    "--progress-interval-s",
                    "60",
                    "--gpus",
                    "0",
                    "--run-direct",
                ]
                for clip_id in args.clip_id:
                    command.extend(["--clip-id", clip_id])

                print(
                    json.dumps(
                        {
                            "event": "trial_started",
                            "trial_id": trial_id,
                            "max_workers": workers,
                            "started_at_utc": _utc_now(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                telemetry_process, telemetry_out, telemetry_err = _start_telemetry(
                    telemetry_path, telemetry_error_path
                )
                started = time.monotonic()
                with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr_stream:
                    completed = subprocess.run(
                        command,
                        cwd=bundle_script.parent.parent,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        text=True,
                        check=False,
                    )
                trial_seconds = time.monotonic() - started
                _stop_telemetry(telemetry_process, telemetry_out, telemetry_err)

                trial: dict[str, Any] = {
                    "trial_id": trial_id,
                    "repetition": repetition,
                    "ordinal": ordinal,
                    "max_workers": workers,
                    "status": "failed" if completed.returncode else "complete",
                    "returncode": completed.returncode,
                    "trial_seconds": trial_seconds,
                    "cache_drop": cache_drop,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "telemetry_path": str(telemetry_path),
                    "time_path": str(time_path),
                }
                if completed.returncode == 0:
                    bundle_path = _bundle_path(trials_root, trial_id)
                    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                    summary = summarize_bundle(
                        bundle,
                        trial_seconds=trial_seconds,
                        telemetry=parse_telemetry(telemetry_path),
                        resource_usage=parse_gnu_time(time_path),
                    )
                    trial.update(summary)
                    trial["bundle_path"] = str(bundle_path)
                else:
                    trial["telemetry"] = parse_telemetry(telemetry_path)
                    trial["resource_usage"] = parse_gnu_time(time_path)

                trial["payload_cleanup"] = (
                    "retained"
                    if args.keep_payloads
                    else _cleanup_trial_payload(public_root, trial_id)
                )
                report["trials"].append(trial)
                report["aggregates"] = aggregate_trials(report["trials"])
                report["recommendation"] = recommend_concurrency(report["aggregates"])
                _atomic_json(report_path, report)
                _render_reports(report, report_path)
                print(
                    json.dumps(
                        {
                            "event": "trial_finished",
                            "trial_id": trial_id,
                            "status": trial["status"],
                            "max_workers": workers,
                            "trial_seconds": trial_seconds,
                            "rows_per_second": trial.get("aggregate_rows_per_second"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Trial {trial_id} failed with exit code {completed.returncode}; "
                        f"see {stderr_path}"
                    )

        signatures = {
            (int(trial["rows"]), int(trial["payload_bytes"]))
            for trial in report["trials"]
            if trial.get("status") == "complete"
        }
        report["data_consistency"] = {
            "status": "ok" if len(signatures) == 1 else "mismatch",
            "row_and_payload_byte_signatures": sorted([list(item) for item in signatures]),
            "note": "Every concurrency must produce the same bounded rows and payload bytes.",
        }
        if len(signatures) != 1:
            raise RuntimeError(
                f"Benchmark data consistency failed; observed signatures: {signatures}"
            )
        report["status"] = "complete"
        report["finished_at_utc"] = _utc_now()
    except Exception as exc:
        report["status"] = "failed"
        report["finished_at_utc"] = _utc_now()
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["aggregates"] = aggregate_trials(report["trials"])
        report["recommendation"] = recommend_concurrency(report["aggregates"])
        _atomic_json(report_path, report)
        _render_reports(report, report_path)
        if not args.keep_payloads:
            try:
                public_root.rmdir()
            except OSError:
                pass
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--recording-frame-index", type=Path, required=True)
    parser.add_argument("--clip-id", action="append", required=True)
    parser.add_argument("--source-video", action="append", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument(
        "--bundle-script",
        type=Path,
        default=Path("scripts/submit_clipped_collection_flat_roi_cache_bundle_bsub.sh"),
    )
    parser.add_argument("--limit-rows", type=int, default=8192)
    parser.add_argument("--gpu-chunk-frames", type=int, default=32)
    parser.add_argument(
        "--trial-order",
        action="append",
        default=[],
        help="Comma-separated concurrency order; repeat for repeated passes",
    )
    parser.add_argument("--keep-payloads", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.trial_order:
        args.trial_order = ["4,1,8,2,6", "6,2,8,1,4"]
    if args.limit_rows < 1:
        parser.error("--limit-rows must be positive")
    if args.gpu_chunk_frames < 1:
        parser.error("--gpu-chunk-frames must be positive")
    try:
        report = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["recommendation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
