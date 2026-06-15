"""Best-effort inline registry refresh helpers for stage writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import zarr
from rich.console import Console

from .db import Registry


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
