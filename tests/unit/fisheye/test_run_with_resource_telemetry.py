from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from fisheye.diagnostics.run_with_resource_telemetry import run_with_resource_telemetry


def test_run_with_resource_telemetry_writes_summary_samples_and_stdout(tmp_path: Path) -> None:
    summary_path = tmp_path / "resources.json"
    samples_path = tmp_path / "resources.jsonl"
    stdout_path = tmp_path / "command.stdout"

    summary = run_with_resource_telemetry(
        [sys.executable, "-c", "sum(i * i for i in range(2_000_000)); print('telemetry-ok')"],
        summary_json=summary_path,
        samples_jsonl=samples_path,
        stdout_log=stdout_path,
        requested_workers=2,
        allocated_slots=4,
        sample_interval_seconds=0.05,
    )

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    assert summary == persisted
    assert persisted["status"] == "ok"
    assert persisted["exit_code"] == 0
    assert persisted["requested_workers"] == 2
    assert persisted["allocated_slots"] == 4
    assert persisted["cpu_seconds"]["total"] > 0.0
    assert persisted["average_effective_cpu_cores"] > 0.0
    assert persisted["sample_count"] == len(samples)
    assert samples
    assert max(int(item["process_count"]) for item in samples) >= 1
    assert "telemetry-ok" in stdout_path.read_text(encoding="utf-8")


def test_run_with_resource_telemetry_persists_nonzero_exit(tmp_path: Path) -> None:
    summary_path = tmp_path / "resources.json"
    summary = run_with_resource_telemetry(
        [sys.executable, "-c", "raise SystemExit(3)"],
        summary_json=summary_path,
        samples_jsonl=tmp_path / "resources.jsonl",
        stdout_log=tmp_path / "command.stdout",
        requested_workers=1,
        allocated_slots=1,
        sample_interval_seconds=0.05,
    )

    assert summary["status"] == "error"
    assert summary["exit_code"] == 3
    assert json.loads(summary_path.read_text(encoding="utf-8"))["exit_code"] == 3


def test_resource_telemetry_cli_persists_keyboard_interrupt(tmp_path: Path) -> None:
    summary_path = tmp_path / "resources.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fisheye.diagnostics.run_with_resource_telemetry",
            "--summary-json",
            str(summary_path),
            "--samples-jsonl",
            str(tmp_path / "resources.jsonl"),
            "--stdout-log",
            str(tmp_path / "command.stdout"),
            "--requested-workers",
            "1",
            "--allocated-slots",
            "1",
            "--sample-interval-seconds",
            "0.05",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=15.0) != 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "interrupted"
    assert summary["termination_reason"] == "keyboard_interrupt"
    assert summary["exit_code"] != 0
