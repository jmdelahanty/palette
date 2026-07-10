"""Run a command while recording process-tree CPU and memory telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
from typing import Optional, Sequence

import psutil


RESOURCE_TELEMETRY_SCHEMA = "palette_process_tree_resource_telemetry_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resource_cpu_seconds(usage: resource.struct_rusage) -> tuple[float, float]:
    return float(usage.ru_utime), float(usage.ru_stime)


def _tree_snapshot(root: psutil.Process) -> dict[str, object]:
    processes: list[psutil.Process] = []
    try:
        processes.append(root)
        processes.extend(root.children(recursive=True))
    except (psutil.Error, OSError):
        pass

    cpu_by_process: dict[tuple[int, float], float] = {}
    rss_bytes = 0
    thread_count = 0
    for process in processes:
        try:
            identity = (int(process.pid), float(process.create_time()))
            cpu_times = process.cpu_times()
            cpu_by_process[identity] = float(cpu_times.user) + float(cpu_times.system)
            rss_bytes += int(process.memory_info().rss)
            thread_count += int(process.num_threads())
        except (psutil.Error, OSError):
            continue
    return {
        "cpu_by_process": cpu_by_process,
        "rss_bytes": int(rss_bytes),
        "process_count": int(len(cpu_by_process)),
        "thread_count": int(thread_count),
    }


def _pump_output(stream: object, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_handle:
        for line in stream:  # type: ignore[union-attr]
            text = str(line)
            sys.stdout.write(text)
            sys.stdout.flush()
            log_handle.write(text)
            log_handle.flush()


def run_with_resource_telemetry(
    command: Sequence[str],
    *,
    summary_json: Path,
    samples_jsonl: Path,
    stdout_log: Path,
    requested_workers: int,
    allocated_slots: int,
    sample_interval_seconds: float = 2.0,
) -> dict[str, object]:
    if not command:
        raise ValueError("A command is required.")
    interval = max(0.05, float(sample_interval_seconds))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(
        [str(value) for value in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if process.stdout is None:
        raise RuntimeError("Could not capture command output.")
    output_thread = threading.Thread(
        target=_pump_output,
        args=(process.stdout, stdout_log),
        name="resource-telemetry-output-pump",
        daemon=True,
    )
    output_thread.start()

    root = psutil.Process(process.pid)
    previous_cpu_by_process: dict[tuple[int, float], float] = {}
    previous_elapsed = 0.0
    sampled_cpu_seconds = 0.0
    sample_count = 0
    peak_rss_bytes = 0
    peak_process_count = 0
    peak_thread_count = 0
    peak_effective_cpu_cores = 0.0
    termination_reason: str | None = None
    try:
        with samples_jsonl.open("w", encoding="utf-8") as samples_handle:
            while True:
                snapshot = _tree_snapshot(root)
                elapsed = max(0.0, float(time.perf_counter() - started))
                cpu_by_process = dict(snapshot["cpu_by_process"])
                cpu_delta = 0.0
                for identity, cpu_seconds in cpu_by_process.items():
                    previous = float(previous_cpu_by_process.get(identity, 0.0))
                    cpu_delta += max(0.0, float(cpu_seconds) - previous)
                sampled_cpu_seconds += cpu_delta
                elapsed_delta = max(0.0, elapsed - previous_elapsed)
                effective_cpu_cores = cpu_delta / elapsed_delta if elapsed_delta > 0.0 else 0.0
                record = {
                    "sample_index": int(sample_count),
                    "timestamp_utc": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "cpu_seconds_delta": float(cpu_delta),
                    "effective_cpu_cores": float(effective_cpu_cores),
                    "process_tree_rss_bytes": int(snapshot["rss_bytes"]),
                    "process_count": int(snapshot["process_count"]),
                    "thread_count": int(snapshot["thread_count"]),
                }
                samples_handle.write(json.dumps(record, sort_keys=True) + "\n")
                samples_handle.flush()
                sample_count += 1
                peak_rss_bytes = max(peak_rss_bytes, int(snapshot["rss_bytes"]))
                peak_process_count = max(peak_process_count, int(snapshot["process_count"]))
                peak_thread_count = max(peak_thread_count, int(snapshot["thread_count"]))
                peak_effective_cpu_cores = max(peak_effective_cpu_cores, float(effective_cpu_cores))
                previous_cpu_by_process = cpu_by_process
                previous_elapsed = elapsed

                if process.poll() is not None:
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        termination_reason = "keyboard_interrupt"
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    exit_code = int(process.wait())
    output_thread.join(timeout=30.0)
    duration = max(0.0, float(time.perf_counter() - started))
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    before_user, before_system = _resource_cpu_seconds(usage_before)
    after_user, after_system = _resource_cpu_seconds(usage_after)
    user_cpu_seconds = max(0.0, after_user - before_user)
    system_cpu_seconds = max(0.0, after_system - before_system)
    total_cpu_seconds = user_cpu_seconds + system_cpu_seconds
    average_effective_cpu_cores = total_cpu_seconds / duration if duration > 0.0 else 0.0
    requested_workers = max(1, int(requested_workers))
    allocated_slots = max(1, int(allocated_slots))
    summary: dict[str, object] = {
        "schema": RESOURCE_TELEMETRY_SCHEMA,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "duration_seconds": duration,
        "exit_code": exit_code,
        "status": "interrupted" if termination_reason else ("ok" if exit_code == 0 else "error"),
        "termination_reason": termination_reason,
        "command": [str(value) for value in command],
        "requested_workers": requested_workers,
        "allocated_slots": allocated_slots,
        "cpu_seconds": {
            "user": user_cpu_seconds,
            "system": system_cpu_seconds,
            "total": total_cpu_seconds,
        },
        "average_effective_cpu_cores": average_effective_cpu_cores,
        "cpu_efficiency_percent_of_requested_workers": (
            100.0 * average_effective_cpu_cores / float(requested_workers)
        ),
        "cpu_efficiency_percent_of_allocated_slots": (
            100.0 * average_effective_cpu_cores / float(allocated_slots)
        ),
        "sample_interval_seconds": interval,
        "sample_count": int(sample_count),
        "sampled_cpu_seconds": float(sampled_cpu_seconds),
        "sampled_cpu_coverage_fraction": (
            float(sampled_cpu_seconds / total_cpu_seconds) if total_cpu_seconds > 0.0 else None
        ),
        "peak_sampled_effective_cpu_cores": float(peak_effective_cpu_cores),
        "peak_process_tree_rss_bytes": int(peak_rss_bytes),
        "peak_process_count": int(peak_process_count),
        "peak_thread_count": int(peak_thread_count),
        "samples_jsonl": str(samples_jsonl),
        "stdout_log": str(stdout_log),
        "host": os.uname().nodename,
        "root_pid": int(process.pid),
    }
    temp_summary = summary_json.with_suffix(summary_json.suffix + ".tmp")
    temp_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_summary.replace(summary_json)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--requested-workers", type=int, required=True)
    parser.add_argument("--allocated-slots", type=int, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Pass a command after --.")
    summary = run_with_resource_telemetry(
        command,
        summary_json=args.summary_json,
        samples_jsonl=args.samples_jsonl,
        stdout_log=args.stdout_log,
        requested_workers=args.requested_workers,
        allocated_slots=args.allocated_slots,
        sample_interval_seconds=args.sample_interval_seconds,
    )
    print(json.dumps(summary, sort_keys=True))
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
