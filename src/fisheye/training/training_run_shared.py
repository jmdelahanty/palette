"""Shared helpers for training run snapshotting and registry lifecycle logging."""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from .export_shared import write_registry_model_exports
from ..registry.db import Registry, RegistryPaths


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
) -> None:
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
            manifest_sha256=safe_sha256_file(manifest_path),
            model_sha256=safe_sha256_file(model_path),
            metrics_sha256=safe_sha256_file(metrics_path),
            status=status,
            final_metrics=final_metrics,
            invocation=invocation_payload,
        )
        if status == "success" and export_artifacts:
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
        console.print(f"[yellow]Registry update skipped:[/yellow] {exc}")
    finally:
        if registry is not None:
            try:
                registry.close()
            except Exception:
                pass
