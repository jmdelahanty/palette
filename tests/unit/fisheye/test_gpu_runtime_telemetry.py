from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import zarr

from fisheye.cluster.subject_masks import full_duration_canary as canary
from fisheye.shared.gpu_runtime_telemetry import (
    GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
    GpuRuntimeTelemetrySampler,
    require_gpu_runtime_telemetry,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

_SAMPLES = """\
2026/08/10 16:00:00.000, 3, GPU-abc, NVIDIA L4, 75, 40, 10, 4096, 61.5, 58, 1500, 6250
2026/08/10 16:00:01.000, 3, GPU-abc, NVIDIA L4, 95, 50, 20, 6144, 70.5, 60, 1800, 6250
2026/08/10 16:00:02.000, 3, GPU-abc, NVIDIA L4, 85, 45, 15, 5120, 66.0, 59, 1650, 6250
"""


class _FakeProcess:
    def __init__(self, *, stdout: Any, stderr: Any) -> None:
        stdout.write(_SAMPLES)
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


def _sampler(path: Path) -> tuple[GpuRuntimeTelemetrySampler, list[str]]:
    observed_command: list[str] = []

    def popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        observed_command[:] = command
        return _FakeProcess(stdout=kwargs["stdout"], stderr=kwargs["stderr"])

    sampler = GpuRuntimeTelemetrySampler(
        output_path=path,
        sample_interval_seconds=1,
        execution_context={"workflow_id": "fixture", "window_id": "clip_0"},
        environment={
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES_ORIG": "3",
            "LSB_JOBID": "123",
            "LSB_JOBINDEX": "7",
        },
        executable_resolver=lambda _name: "/usr/bin/nvidia-smi",
        popen_factory=popen,
    )
    return sampler, observed_command


def test_sampler_persists_exact_trace_summary_and_job_local_gpu_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "gpu_runtime.json"
    sampler, command = _sampler(output)

    report = sampler.start().stop(workload_outcome="success")
    persisted = json.loads(output.read_text(encoding="utf-8"))

    require_gpu_runtime_telemetry(persisted)
    assert persisted == report
    assert report["status"] == "complete"
    assert report["summary"]["sample_count"] == 3
    assert report["summary"]["metrics"]["gpu_utilization_percent"] == {
        "count": 3,
        "mean": 85.0,
        "median": 85.0,
        "p95": 95.0,
        "max": 95.0,
    }
    assert report["execution"]["lsb_jobid"] == "123"
    assert report["sampler"]["device_selector"] == "0"
    assert report["sampler"]["device_selector_source"] == "CUDA_VISIBLE_DEVICES"
    assert report["execution"]["cuda_visible_devices"] == "0"
    assert report["execution"]["cuda_visible_devices_orig"] == "3"
    assert command[-2:] == ["-i", "0"]
    assert not (tmp_path / ".gpu_runtime.json.raw.csv").exists()
    assert not (tmp_path / ".gpu_runtime.json.stderr").exists()


def test_sampler_falls_back_to_original_gpu_selector_without_job_local_value(
    tmp_path: Path,
) -> None:
    observed_command: list[str] = []

    def popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        observed_command[:] = command
        return _FakeProcess(stdout=kwargs["stdout"], stderr=kwargs["stderr"])

    sampler = GpuRuntimeTelemetrySampler(
        output_path=tmp_path / "gpu_runtime.json",
        environment={"CUDA_VISIBLE_DEVICES_ORIG": "3"},
        executable_resolver=lambda _name: "/usr/bin/nvidia-smi",
        popen_factory=popen,
    )

    report = sampler.start().stop(workload_outcome="success")

    require_gpu_runtime_telemetry(report)
    assert report["sampler"]["device_selector"] == "3"
    assert report["sampler"]["device_selector_source"] == (
        "CUDA_VISIBLE_DEVICES_ORIG"
    )
    assert observed_command[-2:] == ["-i", "3"]


def test_missing_nvidia_smi_is_valid_missing_performance_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "gpu_runtime.json"
    sampler = GpuRuntimeTelemetrySampler(
        output_path=output,
        executable_resolver=lambda _name: None,
        environment={},
    )

    report = sampler.start().stop(
        workload_outcome="error", workload_error_type="SyntheticFailure"
    )

    require_gpu_runtime_telemetry(report)
    assert report["status"] == "unavailable"
    assert report["summary"]["sample_count"] == 0
    assert report["reason"] == "nvidia-smi not found in PATH"
    assert report["workload_error_type"] == "SyntheticFailure"


def test_validator_rejects_recomputed_digest_metric_tampering(tmp_path: Path) -> None:
    sampler, _command = _sampler(tmp_path / "gpu_runtime.json")
    report = sampler.start().stop(workload_outcome="success")
    tampered = copy.deepcopy(report)
    tampered["samples"][0]["gpu_utilization_percent"] = 1.0
    payload = dict(tampered)
    payload.pop("payload_digest")
    tampered["payload_digest"] = canonical_json_sha256(payload)

    with pytest.raises(ValueError, match="summary differs"):
        require_gpu_runtime_telemetry(tampered)


def test_worker_bundle_atomically_carries_report_only_gpu_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_archive = tmp_path / "local.zarr"
    root = zarr.open_group(str(local_archive), mode="w", zarr_format=3)
    run = root.create_group("subject_mask_shard_runs").create_group("raw_window_0")
    run.attrs[canary.RUN_COMPLETION_STATUS_ATTR] = canary.RUN_STATUS_COMPLETE
    run.attrs["stage_selector_eligible"] = False
    monkeypatch.setattr(
        canary,
        "_worker_evidence",
        lambda _archive, _run: {
            "scientific_identity": {"digest": "1" * 64},
            "attempt": {"payload_digest": "2" * 64},
            "receipt": {"payload_digest": "3" * 64},
        },
    )
    telemetry_path = tmp_path / "scratch" / "gpu_runtime.json"
    sampler, _command = _sampler(telemetry_path)
    telemetry = sampler.start().stop(workload_outcome="success")
    destination = tmp_path / "bundles" / "window_0"
    result = {
        "schema_id": canary.WORKER_RESULT_SCHEMA_ID,
        "schema_version": canary.WORKER_RESULT_SCHEMA_VERSION,
        "status": "complete",
    }

    published = canary._publish_worker_bundle(
        local_archive=local_archive,
        parent="subject_mask_shard_runs",
        run_name="raw_window_0",
        bundle=destination,
        result=result,
        gpu_runtime_telemetry_path=telemetry_path,
    )

    sidecar = destination / "performance" / "gpu_runtime.json"
    copied = json.loads(sidecar.read_text(encoding="utf-8"))
    require_gpu_runtime_telemetry(copied)
    assert copied == telemetry
    receipt = published["performance_telemetry"]
    assert receipt["identity_policy"] == GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY
    assert receipt["scientific_identity_included"] is False
    assert receipt["gpu_runtime"]["capture_status"] == "complete"
    assert receipt["gpu_runtime"]["payload_digest"] == telemetry["payload_digest"]
    assert receipt["gpu_runtime"]["size_bytes"] == sidecar.stat().st_size
    run_metadata = (
        destination
        / "archive.zarr"
        / "subject_mask_shard_runs"
        / "raw_window_0"
        / "zarr.json"
    ).read_text(encoding="utf-8")
    assert "gpu_runtime" not in run_metadata
    assert not any(
        child.name.startswith(".window_0") for child in destination.parent.iterdir()
    )


def test_invalid_gpu_sidecar_does_not_block_scientific_worker_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_archive = tmp_path / "local.zarr"
    root = zarr.open_group(str(local_archive), mode="w", zarr_format=3)
    run = root.create_group("subject_mask_shard_runs").create_group("raw_window_0")
    run.attrs[canary.RUN_COMPLETION_STATUS_ATTR] = canary.RUN_STATUS_COMPLETE
    run.attrs["stage_selector_eligible"] = False
    monkeypatch.setattr(
        canary,
        "_worker_evidence",
        lambda _archive, _run: {
            "scientific_identity": {"digest": "1" * 64},
            "attempt": {"payload_digest": "2" * 64},
            "receipt": {"payload_digest": "3" * 64},
        },
    )
    telemetry_path = tmp_path / "scratch" / "gpu_runtime.json"
    telemetry_path.parent.mkdir()
    telemetry_path.write_text('{"tampered":true}\n', encoding="utf-8")
    destination = tmp_path / "bundles" / "window_0"

    published = canary._publish_worker_bundle(
        local_archive=local_archive,
        parent="subject_mask_shard_runs",
        run_name="raw_window_0",
        bundle=destination,
        result={
            "schema_id": canary.WORKER_RESULT_SCHEMA_ID,
            "schema_version": canary.WORKER_RESULT_SCHEMA_VERSION,
            "status": "complete",
        },
        gpu_runtime_telemetry_path=telemetry_path,
    )

    assert destination.is_dir()
    assert not (destination / "performance").exists()
    assert published["performance_telemetry"]["gpu_runtime"]["status"] == "missing"
    assert (
        "root field set differs"
        in published["performance_telemetry"]["gpu_runtime"]["reason"]
    )
