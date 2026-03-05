from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import zarr
from rich.console import Console

from ..registry.db import (
    DatasetMetadata,
    Registry,
    RegistryPaths,
    extract_dataset_metadata,
    resolve_dataset_id,
)
from ..registry.status_ledger import upsert_recording_step_status
from ..registry.step_cascade import invalidate_downstream_steps

RegistryInput = Optional[Union[Registry, Path, str]]
ResolveDatasetIdFn = Callable[[zarr.Group, Path], Tuple[str, Optional[str]]]
UpsertStepStatusFn = Callable[..., None]
InvalidateStepsFn = Callable[..., None]


def safe_zarr_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _resolve_registry_input(
    registry: RegistryInput,
    *,
    auto_registry_from_env: bool,
    require_env_registry_exists: bool,
) -> tuple[Optional[Registry], bool]:
    if isinstance(registry, Registry):
        return registry, False

    if registry is not None and not isinstance(registry, (Path, str)):
        return registry, False  # type: ignore[return-value]

    if registry is not None:
        path = Path(registry).expanduser().resolve()
        return Registry(path), True

    if not auto_registry_from_env:
        return None, False

    inferred = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    if require_env_registry_exists and not inferred.exists():
        return None, False
    return Registry(inferred), True


def emit_stage_completion(
    root: Optional[zarr.Group],
    zarr_path: Path,
    *,
    step_name: str,
    status: str,
    source: str,
    run_name: Optional[str] = None,
    method: Optional[str] = None,
    coverage_pct: Optional[float] = None,
    review_status_json: Optional[Mapping[str, Any]] = None,
    details_json: Optional[Mapping[str, Any]] = None,
    console: Optional[Console] = None,
    warning_label: Optional[str] = None,
    registry: RegistryInput = None,
    auto_registry_from_env: bool = True,
    require_env_registry_exists: bool = True,
    invalidate_on_ok: bool = True,
    trigger_run_name: Optional[str] = None,
    metadata: Optional[DatasetMetadata] = None,
    upsert_dataset_row: bool = True,
    resolve_dataset_id_fn: ResolveDatasetIdFn = resolve_dataset_id,
    upsert_step_status_fn: UpsertStepStatusFn = upsert_recording_step_status,
    invalidate_steps_fn: InvalidateStepsFn = invalidate_downstream_steps,
) -> bool:
    registry_db: Optional[Registry] = None
    close_registry = False
    try:
        registry_db, close_registry = _resolve_registry_input(
            registry,
            auto_registry_from_env=auto_registry_from_env,
            require_env_registry_exists=require_env_registry_exists,
        )
        if registry_db is None:
            return False

        resolved_path = Path(zarr_path).expanduser().resolve()
        if metadata is None:
            if root is None:
                raise ValueError("root is required when metadata is not provided")
            metadata = extract_dataset_metadata(
                root,
                resolved_path,
                resolve_dataset_id_fn=resolve_dataset_id_fn,
            )
        if upsert_dataset_row:
            registry_db.upsert_dataset(
                metadata.dataset_id,
                session_uuid=metadata.session_uuid,
                zarr_path=resolved_path,
                recording_id=metadata.recording_id,
                zarr_use=metadata.zarr_use,
                zarr_purpose=metadata.zarr_purpose,
            )
        upsert_step_status_fn(
            registry_db,
            dataset_id=metadata.dataset_id,
            recording_id=metadata.recording_id,
            step_name=step_name,
            status=status,
            run_name=run_name,
            method=method,
            coverage_pct=coverage_pct,
            review_status_json=dict(review_status_json) if isinstance(review_status_json, Mapping) else None,
            details_json=dict(details_json) if isinstance(details_json, Mapping) else None,
            source=source,
            zarr_mtime_ns=safe_zarr_mtime_ns(resolved_path),
        )
        if invalidate_on_ok and status == "ok":
            invalidate_steps_fn(
                registry_db,
                dataset_id=metadata.dataset_id,
                step_name=step_name,
                source=source,
                recording_id=metadata.recording_id,
                trigger_run_name=trigger_run_name if trigger_run_name is not None else run_name,
            )
        return True
    except Exception as exc:
        if console is not None:
            target = warning_label or step_name
            console.print(
                "[yellow]Warning:[/yellow] failed to write recording step status "
                f"for {target}: {exc}"
            )
        return False
    finally:
        if close_registry and registry_db is not None:
            registry_db.close()
