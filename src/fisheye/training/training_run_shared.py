"""Shared helpers for training run snapshotting and registry lifecycle logging."""

from __future__ import annotations

import json
import math
import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from .export_shared import write_registry_model_exports
from ..registry.db import Registry, RegistryPaths
from ..registry.write_policy import (
    REGISTRY_WRITES_DISABLED_ENV,
    registry_writes_disabled,
    running_inside_lsf_job,
)


class TrainingRegistryWriteError(RuntimeError):
    """A training run's registry lifecycle record could not be written."""


# --- Ultralytics best.pt metric binding -------------------------------------
#
# The registry registers ``best.pt``, so the metrics recorded with it must be
# best.pt's own. Ultralytics embeds the best epoch's validation metrics in the
# checkpoint as ``train_metrics``; ``results.csv``'s last row is a different
# epoch whenever early stopping or a later regression occurs.
#
# Registry rows written before 2026-11-17 hold LAST-epoch values under the
# same keys (precision/recall/mAP50/mAP50_95[/pose_*]); rows written by this
# code carry ``metrics_source == "best.pt:train_metrics"``. Compare across
# that boundary only after the historical backfill, or by checking the key.

ULTRALYTICS_BOX_METRIC_COLUMNS: dict[str, str] = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}
ULTRALYTICS_POSE_METRIC_COLUMNS: dict[str, str] = {
    **ULTRALYTICS_BOX_METRIC_COLUMNS,
    "pose_mAP50": "metrics/mAP50(P)",
    "pose_mAP50_95": "metrics/mAP50-95(P)",
}
BEST_CHECKPOINT_METRICS_SOURCE = "best.pt:train_metrics"
# results.csv and the checkpoint's train_metrics carry the same rounded values.
_EPOCH_MATCH_TOLERANCE = 5e-6


def _finite_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def results_row_epoch(row: Any) -> Optional[int]:
    """The 1-based ``epoch`` of an Ultralytics ``results.csv`` row."""
    value = _finite_float_or_none(row.get("epoch") if hasattr(row, "get") else None)
    return None if value is None else int(value)


def results_row_metrics(row: Any, columns: dict[str, str]) -> dict[str, Optional[float]]:
    """Read ``columns`` (registry key -> results.csv column) from one row."""
    getter = row.get if hasattr(row, "get") else (lambda _key: None)
    return {key: _finite_float_or_none(getter(column)) for key, column in columns.items()}


def load_checkpoint_train_metrics(weights_path: Path) -> Optional[dict[str, Any]]:
    import torch  # deferred: this module is imported by non-torch callers

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return None
    train_metrics = checkpoint.get("train_metrics")
    return dict(train_metrics) if isinstance(train_metrics, dict) else None


def best_checkpoint_validation_metrics(
    *,
    weights_path: Path,
    results_df: Any,
    columns: dict[str, str],
) -> dict[str, Any]:
    """Return the validation metrics of the epoch saved as ``best.pt``.

    ``best_epoch`` is the 1-based ``results.csv`` epoch whose metrics match the
    checkpoint's (first match, as Ultralytics keeps the first epoch reaching
    the best fitness); ``None`` if no row matches.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"best.pt not found: {weights_path}")
    train_metrics = load_checkpoint_train_metrics(weights_path)
    if not train_metrics:
        raise ValueError(f"{weights_path} carries no train_metrics; cannot bind metrics to it")
    missing = [column for column in columns.values() if column not in train_metrics]
    if missing:
        raise ValueError(f"{weights_path} train_metrics lacks {missing}")
    metrics = {key: float(train_metrics[column]) for key, column in columns.items()}

    best_epoch = None
    for _, row in results_df.iterrows():
        row_metrics = results_row_metrics(row, columns)
        if all(
            row_metrics[key] is not None
            and abs(row_metrics[key] - metrics[key]) <= _EPOCH_MATCH_TOLERANCE
            for key in metrics
        ):
            best_epoch = results_row_epoch(row)
            break

    return {
        **metrics,
        "fitness": _finite_float_or_none(train_metrics.get("fitness")),
        "best_epoch": best_epoch,
        "metrics_checkpoint": "best.pt",
        "metrics_source": BEST_CHECKPOINT_METRICS_SOURCE,
    }


def ultralytics_completion_errors(
    *, run_dir: Path, final_validation_metrics: Optional[dict[str, Any]]
) -> list[str]:
    """Reasons a finished Ultralytics training run must not be recorded as success."""
    errors: list[str] = []
    best_path = Path(run_dir) / "weights" / "best.pt"
    if not best_path.is_file():
        errors.append(f"best.pt missing at {best_path}")
    results_csv = Path(run_dir) / "results.csv"
    if not results_csv.is_file():
        errors.append(f"results.csv missing at {results_csv}")
    if final_validation_metrics is None:
        errors.append("validation metrics bound to best.pt could not be determined")
    return errors


def safe_sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists() or not path.is_file():
        return None
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot_training_inputs(
    *,
    run_dir: Path,
    config_path: Optional[Path],
    manifest_path: Optional[Path],
    invocation_payload: Optional[dict[str, Any]],
) -> list[Path]:
    """Copy immutable run inputs into run_dir/inputs for reproducibility."""
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if config_path is not None and config_path.exists():
        dest = inputs_dir / config_path.name
        shutil.copy2(config_path, dest)
        written.append(dest)
    if manifest_path is not None and manifest_path.exists():
        dest = inputs_dir / manifest_path.name
        shutil.copy2(manifest_path, dest)
        written.append(dest)
    if invocation_payload:
        dest = inputs_dir / "train_invocation.json"
        dest.write_text(json.dumps(invocation_payload, indent=2), encoding="utf-8")
        written.append(dest)

    return written


def record_registry_training_run(
    *,
    args: Any,
    console: Console,
    invocation_payload: Optional[dict[str, Any]],
    run_id: str,
    set_id: Optional[str],
    config_path: Optional[Path],
    manifest_path: Optional[Path],
    model_path: Optional[Path],
    metrics_path: Optional[Path],
    status: str,
    final_metrics: Optional[dict[str, Any]],
    pose_schema: Optional[dict[str, Any]] = None,
    export_artifacts: Optional[dict[str, Any]] = None,
    expected_manifest_sha256: Optional[str] = None,
) -> None:
    actual_manifest_sha256 = safe_sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        expected_digest = str(expected_manifest_sha256).strip().lower()
        if actual_manifest_sha256 != expected_digest:
            raise ValueError(
                "Pose training manifest content changed after its schema authority "
                "was resolved."
            )
    if registry_writes_disabled():
        # Checked before Registry() is constructed: opening a registry
        # initialises its schema, which is itself a write.
        console.print(
            "[yellow]Registry writes disabled:[/yellow] "
            f"{REGISTRY_WRITES_DISABLED_ENV} is set; training run {run_id!r} "
            f"status {status!r} was not recorded in the registry."
        )
        return
    if running_inside_lsf_job():
        raise TrainingRegistryWriteError(
            "Refusing to write the registry from inside an LSF job "
            f"(LSB_JOBID={os.environ.get('LSB_JOBID')}); cluster training records "
            "are registered on the writer host. Pass --no-log-registry or set "
            f"{REGISTRY_WRITES_DISABLED_ENV}=1."
        )
    registry = None
    try:
        registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        skeleton_id = None
        if isinstance(pose_schema, dict):
            skeleton_id = registry.upsert_pose_skeleton_spec(
                kpt_shape=pose_schema.get("kpt_shape"),
                keypoint_labels=pose_schema.get("keypoint_labels"),
                edges=pose_schema.get("skeleton"),
                name="pose_schema",
            )
        registry.record_training_run(
            run_id=run_id,
            set_id=set_id,
            config_path=config_path,
            manifest_path=manifest_path,
            skeleton_id=skeleton_id,
            model_path=model_path,
            metrics_path=metrics_path,
            config_sha256=safe_sha256_file(config_path),
            manifest_sha256=actual_manifest_sha256,
            model_sha256=safe_sha256_file(model_path),
            metrics_sha256=safe_sha256_file(metrics_path),
            status=status,
            final_metrics=final_metrics,
            invocation=invocation_payload,
        )
        if export_artifacts:
            if skeleton_id:
                export_artifacts["skeleton_id"] = skeleton_id
            if isinstance(pose_schema, dict):
                export_artifacts["pose_schema"] = dict(pose_schema)
            write_registry_model_exports(
                registry=registry,
                run_id=run_id,
                export_artifacts=export_artifacts,
            )
        console.print(f"[green]✓ Registry updated:[/green] {registry_path}")
    except Exception as exc:
        console.print(f"[bold red]✗ Registry update failed:[/bold red] {exc}")
        raise TrainingRegistryWriteError(
            f"Could not record training run {run_id!r} (status {status!r}) in the registry: {exc}"
        ) from exc
    finally:
        if registry is not None:
            try:
                registry.close()
            except Exception:
                pass
