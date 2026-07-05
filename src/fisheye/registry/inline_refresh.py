"""Best-effort inline registry refresh helpers for stage writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import zarr
from rich.console import Console

from fisheye.shared.type_conversions import normalize_attr as _normalize_text

from .db import Registry
from .extractors.detect_performance import (
    _build_detection_data_profile_row,
    _coerce_mapping,
    _get_group,
)


def _open_root(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="r", consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _select_latest_profile_run(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = sorted(str(name) for name in parent.group_keys())
    except Exception:
        names = sorted(str(name) for name in parent.keys())
    return names[-1] if names else None


def _dataset_row_for_zarr(
    registry: Registry,
    zarr_path: Path,
    *,
    dataset_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if dataset_id:
        row = registry.conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ? LIMIT 1;",
            (str(dataset_id),),
        ).fetchone()
        return dict(row) if row is not None else None

    candidates = [str(zarr_path)]
    try:
        resolved = str(zarr_path.resolve())
    except OSError:
        resolved = None
    if resolved and resolved not in candidates:
        candidates.append(resolved)
    for candidate in candidates:
        row = registry.conn.execute(
            "SELECT * FROM datasets WHERE zarr_path = ? LIMIT 1;",
            (candidate,),
        ).fetchone()
        if row is not None:
            return dict(row)
    return None


def sync_latest_detection_profile_for_zarr(
    registry: Registry,
    zarr_path: Path,
    *,
    root: Optional[zarr.Group] = None,
    dataset_id: Optional[str] = None,
    profile_run: Optional[str] = None,
    apply: bool = True,
) -> Dict[str, Any]:
    """Sync one Zarr's detection profile into ``detection_data_profile`` rows.

    Approval/finalization-time inline projection: the detection profile is
    materialized into ``analysis/detection_profile_runs`` at detect-review
    approval-for-training, and this helper projects the latest (or a named) run
    into the registry so users need not run a separate sync CLI. Bulk / repair
    projection is now the job of ``Registry.reconcile_dataset_from_root`` (which
    re-derives all detection profile runs); this helper preserves the historical
    single-run upsert behavior for the inline approval path.
    """

    row = _dataset_row_for_zarr(registry, zarr_path, dataset_id=dataset_id)
    if row is None:
        return {
            "status": "missing_dataset",
            "zarr_path": str(zarr_path),
            "profile_run": profile_run,
            "reason": "zarr_path is not registered",
        }

    resolved_dataset_id = _normalize_text(row.get("dataset_id"))
    if not resolved_dataset_id:
        return {
            "status": "error",
            "zarr_path": str(zarr_path),
            "profile_run": profile_run,
            "reason": "registry dataset row has no dataset_id",
        }

    opened_root = root if root is not None else _open_root(zarr_path)
    analysis = _get_group(opened_root, "analysis")
    runs_parent = _get_group(analysis, "detection_profile_runs") if analysis is not None else None
    if runs_parent is None:
        return {
            "status": "missing_profile",
            "dataset_id": resolved_dataset_id,
            "zarr_path": str(zarr_path),
            "profile_run": profile_run,
            "reason": "analysis/detection_profile_runs missing",
        }

    selected_run = profile_run or _select_latest_profile_run(runs_parent)
    run_group = _get_group(runs_parent, str(selected_run)) if selected_run is not None else None
    summary = _coerce_mapping(run_group.attrs.get("profile_summary")) if run_group is not None else None
    if summary is None:
        return {
            "status": "missing_profile",
            "dataset_id": resolved_dataset_id,
            "zarr_path": str(zarr_path),
            "profile_run": selected_run,
            "reason": "profile_summary missing or invalid",
        }

    try:
        zarr_mtime_ns: Optional[int] = int(Path(zarr_path).stat().st_mtime_ns)
    except OSError:
        zarr_mtime_ns = None

    payload = _build_detection_data_profile_row(
        dataset_id=resolved_dataset_id,
        profile_run=str(selected_run),
        summary=summary,
        run_attrs=dict(run_group.attrs),
        fallback_recording_id=_normalize_text(row.get("recording_id")),
        fallback_zarr_use=_normalize_text(row.get("zarr_use")),
        fallback_genotype=_normalize_text(row.get("genotype")),
        fallback_dpf_at_acquisition=(
            int(row.get("dpf_at_acquisition"))
            if str(row.get("dpf_at_acquisition") or "").strip().lstrip("-").isdigit()
            else None
        ),
        zarr_mtime_ns=zarr_mtime_ns,
    )

    if apply:
        registry.upsert_detection_data_profile(**payload)
        status = "updated"
    else:
        status = "would_upsert"
    return {
        "status": status,
        "dataset_id": resolved_dataset_id,
        "zarr_path": str(zarr_path),
        "profile_run": selected_run,
        "payload": payload,
    }


def refresh_keypoint_performance_details(
    *,
    root: zarr.Group,
    zarr_path: Path,
    run_name: str,
    registry_path: Path,
    console: Optional[Console],
) -> Dict[str, Any]:
    """Refresh keypoint performance rows and return status details for ledgers."""

    registry: Optional[Registry] = None
    try:
        registry = Registry(registry_path)
        dataset_id, row_count = registry.refresh_keypoint_performance_from_root(root, zarr_path)
        refreshed_row = registry.conn.execute(
            """
            SELECT 1
            FROM keypoint_performance
            WHERE dataset_id = ? AND keypoint_run = ?
            LIMIT 1;
            """,
            (dataset_id, str(run_name)),
        ).fetchone()
        return {
            "keypoint_performance_refresh_status": "ok",
            "keypoint_performance_refresh_dataset_id": dataset_id,
            "keypoint_performance_refresh_rows": int(row_count),
            "keypoint_performance_refresh_run": str(run_name),
            "keypoint_performance_refresh_run_present": refreshed_row is not None,
        }
    except Exception as exc:
        if console is not None:
            console.print(
                "[yellow]Warning:[/yellow] failed to refresh keypoint_performance "
                f"for keypoint run {run_name!r}: {exc}"
            )
        return {
            "keypoint_performance_refresh_status": "error",
            "keypoint_performance_refresh_run": str(run_name),
            "keypoint_performance_refresh_reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if registry is not None:
            registry.close()
