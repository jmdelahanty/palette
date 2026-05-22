from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import zarr
from rich.console import Console

from .db import (
    DatasetMetadata,
    Registry,
    RegistryPaths,
    extract_dataset_metadata,
    resolve_dataset_id,
)
from .stage_catalog import canonical_stage_id
from .status_ledger import upsert_recording_step_status
from .step_cascade import invalidate_downstream_steps
from ..shared.zarr.stage_arrays import STAGES, StageSpec, validate_run
from ..shared.zarr_helpers import open_zarr_group_direct
from ..shared.zarr_run_completion import is_run_complete

RegistryInput = Optional[Union[Registry, Path, str]]
ResolveDatasetIdFn = Callable[[zarr.Group, Path], Tuple[str, Optional[str]]]
UpsertStepStatusFn = Callable[..., None]
InvalidateStepsFn = Callable[..., None]


_STEP_RUN_PARENTS = {
    "detect": ("detect_runs",),
    "detect_quality": ("detect_runs",),
    "refined_detect": ("refined_detect_runs", "refined_runs"),
    "refine": ("refined_detect_runs", "refined_runs"),
    "crop": ("crop_runs",),
    "keypoints": ("keypoints_runs",),
    "refined_keypoints": ("refined_keypoints_runs", "keypoints_refined_runs"),
    "keypoints_refine": ("refined_keypoints_runs", "keypoints_refined_runs"),
    "eye_masks": ("eye_masks_runs",),
    "refined_eye_masks": ("refined_eye_masks_runs",),
    "subject_masks": ("subject_mask_runs",),
    "refined_subject_masks": ("refined_subject_masks_runs",),
    "arena_assignment": ("arena_assignment_runs",),
    "assign_ids": ("arena_assignment_runs",),
    "track": ("tracking_runs",),
    "tracks": ("tracking_runs",),
}

_STAGE_ARRAY_SPEC_ALIASES = {
    # Defensive aliases for legacy callers and tests; current production
    # writers mostly use canonical stage names.
    "refine": "refined_detect",
    "keypoints_refine": "refined_keypoints",
    "assign_ids": "arena_assignment",
    "track": "tracking",
    "tracks": "tracking",
}

# Stage-array specs were originally diagnostic contracts and still need a
# writer-by-writer audit before every stage can be fail-closed. Completion-marker
# enforcement remains hard for every ok run; array validation is hard only for
# the staged allowlist below. Add canonical stage names here after real-run
# shadow telemetry confirms that the current writers always emit the required
# arrays in ``shared.zarr.stage_arrays``.
_ENFORCE_STAGE_ARRAY_VALIDATION_FOR = frozenset({"detect_quality"})


def safe_zarr_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _resolve_completion_run_group(
    root: Optional[zarr.Group],
    *,
    step_name: str,
    run_name: Optional[str],
    zarr_path: Optional[Path] = None,
    completion_group_path: Optional[str] = None,
) -> Optional[zarr.Group]:
    if root is None or not run_name:
        return None

    def _open_direct_child(*parts: str) -> Optional[zarr.Group]:
        if zarr_path is None:
            return None
        try:
            return open_zarr_group_direct(Path(zarr_path).joinpath(*parts), mode="r")
        except Exception:
            return None

    if completion_group_path:
        normalized_path = str(completion_group_path).strip().strip("/")
        path_parts = [part for part in normalized_path.split("/") if part]
        if any(part == ".." for part in path_parts):
            return None
        try:
            group: Any = root
            for part in path_parts:
                group = group[part]
            return group
        except Exception:
            pass
        direct_group = _open_direct_child(*path_parts)
        if direct_group is not None:
            return direct_group

    if step_name == "detect_quality":
        try:
            detect_parent = root.get("detect_runs")
        except Exception:
            detect_parent = None
        if detect_parent is not None:
            try:
                detect_names = detect_parent.group_keys()
            except Exception:
                try:
                    detect_names = detect_parent.keys()
                except Exception:
                    detect_names = []
            for detect_name in detect_names:
                try:
                    quality_parent = detect_parent[detect_name].get("quality_reports")
                except Exception:
                    quality_parent = None
                if quality_parent is None:
                    continue
                try:
                    if run_name in quality_parent:
                        return quality_parent[run_name]
                except Exception:
                    pass
                direct_child = _open_direct_child(
                    "detect_runs",
                    str(detect_name),
                    "quality_reports",
                    str(run_name),
                )
                if direct_child is not None:
                    return direct_child
    for parent_name in _STEP_RUN_PARENTS.get(step_name, (f"{step_name}_runs",)):
        try:
            parent = root.get(parent_name)
        except Exception:
            parent = None
        if parent is None:
            continue
        try:
            if run_name in parent:
                return parent[run_name]
        except Exception:
            try:
                candidate = parent.get(run_name)
            except Exception:
                candidate = None
            if candidate is not None:
                return candidate
        direct_child = _open_direct_child(str(parent_name), str(run_name))
        if direct_child is not None:
            return direct_child
    return None


def _stage_array_spec_for_completion(step_name: str) -> Optional[StageSpec]:
    stage_name = _STAGE_ARRAY_SPEC_ALIASES.get(step_name, step_name)
    if stage_name in STAGES:
        return STAGES[stage_name]
    try:
        canonical_name = canonical_stage_id(step_name)
    except KeyError:
        canonical_name = step_name
    stage_name = _STAGE_ARRAY_SPEC_ALIASES.get(canonical_name, canonical_name)
    return STAGES.get(stage_name)


def _validate_completion_run_group(
    run_group: zarr.Group,
    *,
    step_name: str,
    run_name: str,
) -> dict[str, Any]:
    if not is_run_complete(run_group, legacy_default=True):
        raise RuntimeError(
            f"Refusing to mark {step_name} run {run_name!r} ok because "
            "the Zarr run-completion marker is not complete"
        )

    spec = _stage_array_spec_for_completion(step_name)
    if spec is None:
        return {
            "stage_array_validation_status": "no_spec",
            "stage_array_validation_enforced": False,
            "stage_array_validation_warnings": [
                f"no StageSpec for step {step_name!r}; skipped array validation"
            ],
        }

    enforced = spec.stage_name in _ENFORCE_STAGE_ARRAY_VALIDATION_FOR
    result = validate_run(run_group, spec)
    details: dict[str, Any] = {
        "stage_array_validation_status": "ok" if result.valid else "invalid",
        "stage_array_validation_stage": spec.stage_name,
        "stage_array_validation_enforced": enforced,
    }
    if result.errors:
        details["stage_array_validation_errors"] = list(result.errors)
    if result.warnings:
        details["stage_array_validation_warnings"] = list(result.warnings)

    if enforced and not result.valid:
        errors = "; ".join(result.errors)
        raise RuntimeError(
            f"Refusing to mark {step_name} run {run_name!r} ok because "
            f"required Zarr arrays failed validation: {errors}"
        )
    return details


def _merge_validation_details(
    details_json: Optional[Mapping[str, Any]],
    validation_details: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    details = dict(details_json) if isinstance(details_json, Mapping) else {}
    # The live validator is authoritative for these keys; overwrite any stale
    # caller-supplied validation payload from an earlier attempt.
    details.update(validation_details)
    return details or None


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
    completion_group_path: Optional[str] = None,
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
        validation_details: dict[str, Any] = {}
        if status == "ok" and run_name:
            if root is None:
                raise RuntimeError(
                    f"Refusing to mark {step_name} run {run_name!r} ok because "
                    "root is required to verify Zarr run completion"
                )
            completion_group = _resolve_completion_run_group(
                root,
                step_name=step_name,
                run_name=run_name,
                zarr_path=resolved_path,
                completion_group_path=completion_group_path,
            )
            if completion_group is None:
                raise RuntimeError(
                    f"Refusing to mark {step_name} run {run_name!r} ok because "
                    "the Zarr run group could not be resolved"
                )
            validation_details = _validate_completion_run_group(
                completion_group,
                step_name=step_name,
                run_name=run_name,
            )
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
            details_json=_merge_validation_details(details_json, validation_details),
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
