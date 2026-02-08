#!/usr/bin/env python3
"""Backfill a completed training run into the registry."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils.system import build_invocation_record


def _safe_sha256_file(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_final_metrics(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if not path or not path.exists():
        return None
    last_row = None
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row
    if not last_row:
        return None
    def _as_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None
    metrics = {
        "precision": _as_float(last_row.get("metrics/precision(B)")),
        "recall": _as_float(last_row.get("metrics/recall(B)")),
        "mAP50": _as_float(last_row.get("metrics/mAP50(B)")),
        "mAP50_95": _as_float(last_row.get("metrics/mAP50-95(B)")),
    }
    return metrics


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Ultralytics run directory (contains weights/results.csv).")
    parser.add_argument("--config", type=Path, help="Config YAML used for training.")
    parser.add_argument("--manifest", type=Path, help="Manifest JSON used for training.")
    parser.add_argument("--set-id", type=str, help="Training set ID to associate.")
    parser.add_argument("--run-id", type=str, help="Override run_id (defaults to run_dir name).")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path.")
    parser.add_argument("--status", type=str, default="success", help="Status to record.")
    parser.add_argument("--model-path", type=Path, help="Override model path.")
    parser.add_argument("--metrics-path", type=Path, help="Override metrics CSV path.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    run_id = args.run_id or run_dir.name
    config_path = args.config.expanduser().resolve() if args.config else None
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else None
    model_path = args.model_path or (run_dir / "weights" / "best.pt")
    metrics_path = args.metrics_path or (run_dir / "results.csv")
    if model_path and not model_path.exists():
        model_path = None
    if metrics_path and not metrics_path.exists():
        metrics_path = None

    final_metrics = _load_final_metrics(metrics_path)
    if final_metrics is not None:
        final_metrics["stage"] = "completed"
        final_metrics.setdefault("status_detail", "registered_backfill")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    invocation = build_invocation_record(tool="fisheye.utils.register_training_run", args=args)
    registry.record_training_run(
        run_id=run_id,
        set_id=args.set_id,
        config_path=config_path,
        manifest_path=manifest_path,
        skeleton_id=None,
        model_path=model_path,
        metrics_path=metrics_path,
        config_sha256=_safe_sha256_file(config_path),
        manifest_sha256=_safe_sha256_file(manifest_path),
        model_sha256=_safe_sha256_file(model_path),
        metrics_sha256=_safe_sha256_file(metrics_path),
        status=args.status,
        final_metrics=final_metrics,
        invocation=invocation,
    )
    registry.close()
    print(f"Registered run {run_id} in {registry_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
