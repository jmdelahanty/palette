"""Continuous report-only NVIDIA GPU telemetry for bounded workloads.

The documents produced here describe execution performance only.  They are
deliberately excluded from scientific identity, array digests, selectors, and
registry authority.  Callers should collect into node-local scratch and attach
the completed document to an immutable result bundle as a sidecar.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import time
from typing import Any, TextIO

from fisheye.shared.json_safety import json_attr_safe_mapping, write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

GPU_RUNTIME_TELEMETRY_SCHEMA_ID = "palette.gpu_runtime_telemetry"
GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION = 1
GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY = (
    "performance_only_excluded_from_scientific_identity_and_payload_digests"
)

_QUERY_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "utilization.decoder",
    "memory.used",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)
_SAMPLE_FIELDS = (
    "device_timestamp",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "decoder_utilization_percent",
    "memory_used_mib",
    "power_draw_watts",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
)
_METRIC_FIELDS = _SAMPLE_FIELDS[4:]
_ROOT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "identity_policy",
        "status",
        "reason",
        "started_at_utc",
        "finished_at_utc",
        "wall_seconds",
        "sample_interval_seconds",
        "sampler",
        "execution",
        "workload_outcome",
        "workload_error_type",
        "samples",
        "summary",
        "stderr_tail",
        "payload_digest",
    }
)
_SAMPLER_FIELDS = frozenset(
    {
        "backend",
        "command",
        "device_selector",
        "device_selector_source",
        "query_fields",
        "process_returncode",
    }
)
_SUMMARY_FIELDS = frozenset(
    {"sample_count", "metrics", "first_device_timestamp", "last_device_timestamp"}
)
_STATS_FIELDS = frozenset({"count", "mean", "median", "p95", "max"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {
        "n/a",
        "na",
        "not supported",
        "[not supported]",
        "-",
    }:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _device_selector(
    environment: Mapping[str, str],
) -> tuple[str | None, str]:
    # LSF records the physical host allocation in CUDA_VISIBLE_DEVICES_ORIG,
    # but presents the job-local CUDA/nvidia-smi namespace through
    # CUDA_VISIBLE_DEVICES.  On hosts that remap a physical device (for
    # example, physical GPU 5 to job-local GPU 0), passing the *_ORIG value to
    # nvidia-smi exits without samples even though inference is healthy.  Use
    # the same namespace as the CUDA workload and retain *_ORIG as provenance.
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES_ORIG"):
        value = str(environment.get(name, "")).strip()
        if value and value not in {"-1", "NoDevFiles"}:
            return value, name
    return None, "none"


def _stats(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
    }


def _parse_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not path.is_file():
        return samples
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for values in csv.reader(stream):
            if len(values) < len(_QUERY_FIELDS):
                continue
            raw = [str(value).strip() for value in values[: len(_QUERY_FIELDS)]]
            gpu_index = _integer(raw[1])
            if gpu_index is None or not raw[2] or not raw[3]:
                continue
            samples.append(
                {
                    "device_timestamp": raw[0],
                    "gpu_index": gpu_index,
                    "gpu_uuid": raw[2],
                    "gpu_name": raw[3],
                    "gpu_utilization_percent": _number(raw[4]),
                    "memory_utilization_percent": _number(raw[5]),
                    "decoder_utilization_percent": _number(raw[6]),
                    "memory_used_mib": _number(raw[7]),
                    "power_draw_watts": _number(raw[8]),
                    "temperature_c": _number(raw[9]),
                    "sm_clock_mhz": _number(raw[10]),
                    "memory_clock_mhz": _number(raw[11]),
                }
            )
    return samples


def _summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        name: _stats(
            [float(sample[name]) for sample in samples if sample.get(name) is not None]
        )
        for name in _METRIC_FIELDS
    }
    return {
        "sample_count": len(samples),
        "metrics": metrics,
        "first_device_timestamp": (
            str(samples[0]["device_timestamp"]) if samples else None
        ),
        "last_device_timestamp": (
            str(samples[-1]["device_timestamp"]) if samples else None
        ),
    }


def _timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be one ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone.")
    return parsed


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite and nonnegative.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return number


def require_gpu_runtime_telemetry(value: Mapping[str, Any]) -> None:
    """Deeply validate one report-only GPU runtime telemetry document."""

    if not isinstance(value, Mapping) or set(value) != _ROOT_FIELDS:
        raise ValueError("GPU runtime telemetry root field set differs.")
    if (
        value["schema_id"] != GPU_RUNTIME_TELEMETRY_SCHEMA_ID
        or value["schema_version"] != GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION
        or value["identity_policy"] != GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY
    ):
        raise ValueError("GPU runtime telemetry contract identity differs.")
    if value["status"] not in {"complete", "unavailable", "error"}:
        raise ValueError("GPU runtime telemetry status differs.")
    if value["reason"] is not None and (
        type(value["reason"]) is not str or not value["reason"]
    ):
        raise ValueError("GPU runtime telemetry reason is invalid.")
    started = _timestamp(value["started_at_utc"], label="telemetry started_at_utc")
    finished = _timestamp(value["finished_at_utc"], label="telemetry finished_at_utc")
    if finished < started:
        raise ValueError("GPU runtime telemetry finishes before it starts.")
    _finite_nonnegative(value["wall_seconds"], label="telemetry wall_seconds")
    interval = _finite_nonnegative(
        value["sample_interval_seconds"], label="telemetry sample interval"
    )
    if interval <= 0.0:
        raise ValueError("GPU runtime telemetry sample interval must be positive.")
    if value["workload_outcome"] not in {"success", "error"}:
        raise ValueError("GPU telemetry workload outcome differs.")
    if value["workload_error_type"] is not None and (
        type(value["workload_error_type"]) is not str
        or not value["workload_error_type"]
    ):
        raise ValueError("GPU telemetry workload error type is invalid.")
    if (
        value["workload_outcome"] == "success"
        and value["workload_error_type"] is not None
    ):
        raise ValueError("Successful GPU telemetry workload records an error type.")
    sampler = value["sampler"]
    if not isinstance(sampler, Mapping) or set(sampler) != _SAMPLER_FIELDS:
        raise ValueError("GPU runtime sampler field set differs.")
    if sampler["backend"] != "nvidia_smi_query_loop_v1":
        raise ValueError("GPU runtime sampler backend differs.")
    if not isinstance(sampler["command"], list) or not all(
        type(item) is str for item in sampler["command"]
    ):
        raise ValueError("GPU runtime sampler command differs.")
    if (
        sampler["device_selector"] is not None
        and type(sampler["device_selector"]) is not str
    ):
        raise ValueError("GPU runtime sampler device selector differs.")
    if sampler["device_selector_source"] not in {
        "CUDA_VISIBLE_DEVICES_ORIG",
        "CUDA_VISIBLE_DEVICES",
        "none",
    }:
        raise ValueError("GPU runtime sampler selector source differs.")
    if sampler["query_fields"] != list(_QUERY_FIELDS):
        raise ValueError("GPU runtime sampler query fields differ.")
    if (
        sampler["process_returncode"] is not None
        and type(sampler["process_returncode"]) is not int
    ):
        raise ValueError("GPU runtime sampler return code differs.")
    if not isinstance(value["execution"], Mapping):
        raise ValueError("GPU runtime telemetry execution context differs.")
    if not isinstance(value["stderr_tail"], list) or not all(
        type(item) is str for item in value["stderr_tail"]
    ):
        raise ValueError("GPU runtime telemetry stderr tail differs.")

    samples = value["samples"]
    if not isinstance(samples, list):
        raise ValueError("GPU runtime telemetry samples differ.")
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != set(_SAMPLE_FIELDS):
            raise ValueError("GPU runtime telemetry sample field set differs.")
        if (
            type(sample["device_timestamp"]) is not str
            or not sample["device_timestamp"]
        ):
            raise ValueError("GPU runtime device timestamp differs.")
        if type(sample["gpu_index"]) is not int or sample["gpu_index"] < 0:
            raise ValueError("GPU runtime sample index differs.")
        for name in ("gpu_uuid", "gpu_name"):
            if type(sample[name]) is not str or not sample[name]:
                raise ValueError(f"GPU runtime sample {name} differs.")
        for name in _METRIC_FIELDS:
            metric = sample[name]
            if metric is not None:
                _finite_nonnegative(metric, label=f"GPU runtime sample {name}")

    summary = value["summary"]
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_FIELDS:
        raise ValueError("GPU runtime telemetry summary fields differ.")
    expected_summary = _summary(samples)
    if summary != expected_summary:
        raise ValueError("GPU runtime telemetry summary differs from samples.")
    if value["status"] == "complete" and not samples:
        raise ValueError("Complete GPU runtime telemetry has no samples.")
    if value["status"] != "complete" and samples:
        raise ValueError("Incomplete GPU runtime telemetry unexpectedly has samples.")
    payload = dict(value)
    digest = payload.pop("payload_digest")
    if type(digest) is not str or digest != canonical_json_sha256(payload):
        raise ValueError("GPU runtime telemetry payload digest differs.")


class GpuRuntimeTelemetrySampler:
    """Manage one bounded ``nvidia-smi`` sampling subprocess."""

    def __init__(
        self,
        *,
        output_path: Path,
        sample_interval_seconds: int = 1,
        execution_context: Mapping[str, Any] | None = None,
        environment: Mapping[str, str] | None = None,
        executable_resolver: Callable[[str], str | None] = shutil.which,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if type(sample_interval_seconds) is not int or sample_interval_seconds <= 0:
            raise ValueError(
                "GPU telemetry sample interval must be a positive integer."
            )
        self.output_path = output_path.expanduser().resolve()
        self.sample_interval_seconds = sample_interval_seconds
        self.environment = dict(os.environ if environment is None else environment)
        self.executable_resolver = executable_resolver
        self.popen_factory = popen_factory
        self.execution_context = json_attr_safe_mapping(execution_context or {})
        self.execution_context.update(
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "lsb_jobid": self.environment.get("LSB_JOBID"),
                "lsb_jobindex": self.environment.get("LSB_JOBINDEX"),
                "lsb_jobname": self.environment.get("LSB_JOBNAME"),
                "lsb_queue": self.environment.get("LSB_QUEUE"),
                "cuda_visible_devices": self.environment.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "cuda_visible_devices_orig": self.environment.get(
                    "CUDA_VISIBLE_DEVICES_ORIG"
                ),
            }
        )
        self._started_at_utc: str | None = None
        self._started_perf: float | None = None
        self._process: Any | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._raw_path = self.output_path.with_name(f".{self.output_path.name}.raw.csv")
        self._error_path = self.output_path.with_name(
            f".{self.output_path.name}.stderr"
        )
        self._command: list[str] = []
        self._device_selector, self._selector_source = _device_selector(
            self.environment
        )
        self._start_error: str | None = None
        self._stopped = False

    def start(self) -> "GpuRuntimeTelemetrySampler":
        if self._started_at_utc is not None:
            raise RuntimeError("GPU runtime telemetry sampler already started.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._started_at_utc = _utc_now()
        self._started_perf = time.perf_counter()
        executable = self.executable_resolver("nvidia-smi")
        if executable is None:
            self._start_error = "nvidia-smi not found in PATH"
            return self
        command = [
            executable,
            f"--query-gpu={','.join(_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
            "-l",
            str(self.sample_interval_seconds),
        ]
        if self._device_selector is not None:
            command.extend(["-i", self._device_selector])
        self._command = command
        try:
            self._stdout = self._raw_path.open("w", encoding="utf-8")
            self._stderr = self._error_path.open("w", encoding="utf-8")
            self._process = self.popen_factory(
                command,
                stdout=self._stdout,
                stderr=self._stderr,
                text=True,
                env=self.environment,
            )
        except Exception as exc:  # performance evidence must not abort inference
            self._start_error = f"{type(exc).__name__}: {exc}"
            self._close_streams()
        return self

    def _close_streams(self) -> None:
        for stream in (self._stdout, self._stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def stop(
        self,
        *,
        workload_outcome: str,
        workload_error_type: str | None = None,
    ) -> dict[str, Any]:
        if self._started_at_utc is None or self._started_perf is None:
            raise RuntimeError("GPU runtime telemetry sampler was not started.")
        if self._stopped:
            raise RuntimeError("GPU runtime telemetry sampler already stopped.")
        if workload_outcome not in {"success", "error"}:
            raise ValueError("GPU telemetry workload outcome must be success or error.")
        if workload_outcome == "success" and workload_error_type is not None:
            raise ValueError(
                "Successful GPU telemetry workload cannot have an error type."
            )
        self._stopped = True
        process_returncode: int | None = None
        stop_error: str | None = None
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=10)
                process_returncode = self._process.poll()
            except Exception as exc:  # retain a parseable report on sampler failure
                stop_error = f"{type(exc).__name__}: {exc}"
        self._close_streams()
        samples = _parse_samples(self._raw_path)
        stderr_tail: list[str] = []
        if self._error_path.is_file():
            stderr_tail = self._error_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-20:]
        reason = self._start_error or stop_error
        if samples:
            status = "complete"
            reason = None
        elif self._start_error and "not found" in self._start_error:
            status = "unavailable"
        else:
            status = "error"
            if reason is None:
                reason = "nvidia-smi produced no valid telemetry samples"
        payload: dict[str, Any] = {
            "schema_id": GPU_RUNTIME_TELEMETRY_SCHEMA_ID,
            "schema_version": GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION,
            "identity_policy": GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
            "status": status,
            "reason": reason,
            "started_at_utc": self._started_at_utc,
            "finished_at_utc": _utc_now(),
            "wall_seconds": float(time.perf_counter() - self._started_perf),
            "sample_interval_seconds": self.sample_interval_seconds,
            "sampler": {
                "backend": "nvidia_smi_query_loop_v1",
                "command": list(self._command),
                "device_selector": self._device_selector,
                "device_selector_source": self._selector_source,
                "query_fields": list(_QUERY_FIELDS),
                "process_returncode": process_returncode,
            },
            "execution": self.execution_context,
            "workload_outcome": workload_outcome,
            "workload_error_type": workload_error_type,
            "samples": samples,
            "summary": _summary(samples),
            "stderr_tail": stderr_tail,
        }
        payload["payload_digest"] = canonical_json_sha256(payload)
        require_gpu_runtime_telemetry(payload)
        write_json_atomic(self.output_path, payload, overwrite=False)
        for path in (self._raw_path, self._error_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return payload


__all__ = [
    "GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY",
    "GPU_RUNTIME_TELEMETRY_SCHEMA_ID",
    "GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION",
    "GpuRuntimeTelemetrySampler",
    "require_gpu_runtime_telemetry",
]
