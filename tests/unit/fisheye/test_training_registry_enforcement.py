"""Enforcement corrections for training registry writes, exit status and metrics.

Covers the 2026-11-17 cluster-training review findings:
- training honors PALETTE_DISABLE_REGISTRY_WRITES and never opens the registry
  when it is set (opening initialises the schema, which is a write);
- training refuses to write the registry from inside an LSF job;
- registry write failures fail the run instead of being warned about;
- detect records the metrics of best.pt (the registered weights), not of the
  last epoch;
- a run with no best.pt or no bound metrics is not recorded as success;
- a config-load failure exits non-zero;
- invocation records can carry the LSF allocation and accelerator identity.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from rich.console import Console

from fisheye.registry.write_policy import (
    REGISTRY_WRITES_DISABLED_ENV,
    registry_writes_disabled,
    running_inside_lsf_job,
)
from fisheye.shared import system_metadata
from fisheye.training import train_detection as td
from fisheye.training import training_run_shared
from fisheye.training.training_run_shared import (
    TrainingRegistryWriteError,
    record_registry_training_run,
)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def _record(registry_path: Path, *, status: str = "in_progress") -> None:
    record_registry_training_run(
        args=SimpleNamespace(registry=registry_path),
        console=_console(),
        invocation_payload={"tool": "test"},
        run_id="detect_run",
        set_id="detect_set",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status=status,
        final_metrics={"stage": "preflight"},
    )


@pytest.fixture
def clean_scheduler_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv(REGISTRY_WRITES_DISABLED_ENV, raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    return monkeypatch


# --- write policy -----------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_registry_writes_disabled_accepts_truthy_values(value: str) -> None:
    assert registry_writes_disabled({REGISTRY_WRITES_DISABLED_ENV: value})


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_registry_writes_disabled_rejects_other_values(value: str) -> None:
    assert not registry_writes_disabled({REGISTRY_WRITES_DISABLED_ENV: value})
    assert not registry_writes_disabled({})


def test_running_inside_lsf_job_reads_lsb_jobid() -> None:
    assert running_inside_lsf_job({"LSB_JOBID": "153285074"})
    assert not running_inside_lsf_job({"LSB_JOBID": "  "})
    assert not running_inside_lsf_job({})


# --- training registry record -----------------------------------------------


def test_record_writes_registry_when_allowed(tmp_path: Path, clean_scheduler_env) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _record(registry_path)
    assert registry_path.exists()


def test_record_skips_without_opening_registry_when_writes_disabled(
    tmp_path: Path, clean_scheduler_env
) -> None:
    clean_scheduler_env.setenv(REGISTRY_WRITES_DISABLED_ENV, "1")
    registry_path = tmp_path / "registry.sqlite"
    _record(registry_path, status="success")
    assert not registry_path.exists()


def test_record_refuses_inside_lsf_job(tmp_path: Path, clean_scheduler_env) -> None:
    clean_scheduler_env.setenv("LSB_JOBID", "153285074")
    registry_path = tmp_path / "registry.sqlite"
    with pytest.raises(TrainingRegistryWriteError, match="inside an LSF job"):
        _record(registry_path)
    assert not registry_path.exists()


def test_record_inside_lsf_job_with_writes_disabled_is_a_clean_skip(
    tmp_path: Path, clean_scheduler_env
) -> None:
    clean_scheduler_env.setenv("LSB_JOBID", "153285074")
    clean_scheduler_env.setenv(REGISTRY_WRITES_DISABLED_ENV, "1")
    registry_path = tmp_path / "registry.sqlite"
    _record(registry_path)
    assert not registry_path.exists()


def test_record_raises_when_registry_write_fails(tmp_path: Path, clean_scheduler_env) -> None:
    class _BrokenRegistry:
        def __init__(self, path) -> None:
            self.path = path

        def record_training_run(self, **kwargs) -> None:
            raise RuntimeError("database is locked")

        def close(self) -> None:
            pass

    clean_scheduler_env.setattr(training_run_shared, "Registry", _BrokenRegistry)
    with pytest.raises(TrainingRegistryWriteError, match="database is locked"):
        _record(tmp_path / "registry.sqlite", status="success")


# --- invocation record -------------------------------------------------------


def test_invocation_record_default_shape_is_unchanged(clean_scheduler_env) -> None:
    clean_scheduler_env.setenv("LSB_JOBID", "153285074")
    record = system_metadata.build_invocation_record(
        tool="test", args=None, argv=[], include_git=False, include_environment=False
    )
    assert set(record["platform"]) == {"hostname", "username", "system", "release"}
    assert "accelerator" not in record


def test_invocation_record_can_include_lsf_allocation(clean_scheduler_env) -> None:
    clean_scheduler_env.setenv("LSB_JOBID", "153285074")
    clean_scheduler_env.setenv("LSB_QUEUE", "gpu_l4")
    clean_scheduler_env.setenv("LSB_GPU_REQ", "num=1")
    record = system_metadata.build_invocation_record(
        tool="test",
        args=None,
        argv=[],
        include_git=False,
        include_environment=False,
        include_scheduler=True,
    )
    lsf = record["platform"]["lsf"]
    assert lsf["job_id"] == "153285074"
    assert lsf["queue"] == "gpu_l4"
    assert lsf["gpu_request"] == "num=1"


def test_invocation_record_scheduler_absent_outside_lsf(clean_scheduler_env) -> None:
    clean_scheduler_env.delenv("SLURM_JOB_ID", raising=False)
    record = system_metadata.build_invocation_record(
        tool="test",
        args=None,
        argv=[],
        include_git=False,
        include_environment=False,
        include_scheduler=True,
    )
    assert "lsf" not in record["platform"]


def test_invocation_record_can_include_accelerator(clean_scheduler_env) -> None:
    clean_scheduler_env.setattr(
        system_metadata,
        "get_gpu_info",
        lambda: {"available": True, "devices": [{"name": "NVIDIA L4", "uuid": "GPU-1"}]},
    )
    record = system_metadata.build_invocation_record(
        tool="test",
        args=None,
        argv=[],
        include_git=False,
        include_environment=False,
        include_accelerator=True,
    )
    assert record["accelerator"]["devices"][0]["name"] == "NVIDIA L4"


# --- detect metrics bound to best.pt ----------------------------------------

_COLUMNS = [
    "epoch",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

# Shaped like detect v004: best.pt is epoch 27 (0.71357), training ran to 37.
_RESULTS = pd.DataFrame(
    [
        [26, 0.97369, 0.97369, 0.97833, 0.64270],
        [27, 0.97678, 0.97583, 0.98086, 0.71357],
        [36, 0.97867, 0.97867, 0.98223, 0.68222],
        [37, 0.97530, 0.97440, 0.97827, 0.70829],
    ],
    columns=_COLUMNS,
)

_BEST_TRAIN_METRICS = {
    "metrics/precision(B)": 0.97678,
    "metrics/recall(B)": 0.97583,
    "metrics/mAP50(B)": 0.98086,
    "metrics/mAP50-95(B)": 0.71357,
    "val/box_loss": 1.05027,
    "fitness": 0.71357,
}


def _write_checkpoint(path: Path, train_metrics: dict | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"epoch": -1, "model": None}
    if train_metrics is not None:
        payload["train_metrics"] = train_metrics
    torch.save(payload, path)
    return path


def test_best_checkpoint_metrics_are_best_epoch_not_last(tmp_path: Path) -> None:
    weights = _write_checkpoint(tmp_path / "weights" / "best.pt", _BEST_TRAIN_METRICS)

    metrics = td._best_checkpoint_validation_metrics(weights_path=weights, results_df=_RESULTS)

    assert metrics["mAP50_95"] == pytest.approx(0.71357)
    assert metrics["mAP50"] == pytest.approx(0.98086)
    assert metrics["best_epoch"] == 27
    assert metrics["fitness"] == pytest.approx(0.71357)
    assert metrics["metrics_checkpoint"] == "best.pt"
    last = td._results_row_box_metrics(_RESULTS.iloc[-1])
    assert last["mAP50_95"] == pytest.approx(0.70829)
    assert metrics["mAP50_95"] != pytest.approx(last["mAP50_95"])


def test_best_checkpoint_metrics_unmatched_epoch_is_none(tmp_path: Path) -> None:
    weights = _write_checkpoint(
        tmp_path / "best.pt", {**_BEST_TRAIN_METRICS, "metrics/mAP50-95(B)": 0.5}
    )
    metrics = td._best_checkpoint_validation_metrics(weights_path=weights, results_df=_RESULTS)
    assert metrics["best_epoch"] is None
    assert metrics["mAP50_95"] == pytest.approx(0.5)


def test_best_checkpoint_metrics_require_train_metrics(tmp_path: Path) -> None:
    weights = _write_checkpoint(tmp_path / "best.pt", None)
    with pytest.raises(ValueError, match="no train_metrics"):
        td._best_checkpoint_validation_metrics(weights_path=weights, results_df=_RESULTS)


def test_best_checkpoint_metrics_require_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        td._best_checkpoint_validation_metrics(
            weights_path=tmp_path / "missing.pt", results_df=_RESULTS
        )


# --- completion gate and exit status ----------------------------------------


def test_completion_errors_empty_for_complete_run(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "weights" / "best.pt", _BEST_TRAIN_METRICS)
    _RESULTS.to_csv(tmp_path / "results.csv", index=False)
    assert td._detection_completion_errors(
        run_dir=tmp_path, final_validation_metrics={"mAP50_95": 0.7}
    ) == []


def test_completion_errors_report_missing_weights_and_metrics(tmp_path: Path) -> None:
    errors = td._detection_completion_errors(run_dir=tmp_path, final_validation_metrics=None)
    assert any("best.pt missing" in message for message in errors)
    assert any("results.csv missing" in message for message in errors)
    assert any("could not be determined" in message for message in errors)


def test_main_returns_nonzero_on_config_load_failure(
    tmp_path: Path, clean_scheduler_env
) -> None:
    clean_scheduler_env.setattr(td, "build_invocation_record", lambda **kwargs: {"tool": "test"})
    args = SimpleNamespace(
        config_path=str(tmp_path / "missing.yaml"),
        manifest=None,
        set_id=None,
        run_name="detect_config_failure",
        profile=False,
        log_registry=False,
        registry=None,
        allow_source_mismatch=False,
    )
    assert td.main(args) == 1


def test_main_records_failed_status_on_config_load_failure(
    tmp_path: Path, clean_scheduler_env
) -> None:
    clean_scheduler_env.setattr(td, "build_invocation_record", lambda **kwargs: {"tool": "test"})
    recorded: list[dict] = []
    clean_scheduler_env.setattr(
        td, "_record_registry_training_run", lambda **kwargs: recorded.append(kwargs)
    )
    args = SimpleNamespace(
        config_path=str(tmp_path / "missing.yaml"),
        manifest=None,
        set_id=None,
        run_name="detect_config_failure",
        profile=False,
        log_registry=True,
        registry=tmp_path / "registry.sqlite",
        allow_source_mismatch=False,
    )
    assert td.main(args) == 1
    assert [entry["status"] for entry in recorded] == ["failed"]
    assert recorded[0]["final_metrics"]["stage"] == "config_load"
