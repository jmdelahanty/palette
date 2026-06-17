"""Optional registry sync for saved dish-mask tuning metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from fisheye.registry.db import Registry
from fisheye.registry.status_ledger import upsert_recording_step_status


@dataclass(frozen=True)
class DishMaskRegistrySyncResult:
    synced: bool
    status: str
    message: str
    registry_path: str
    zarr_path: str
    dataset_id: Optional[str] = None
    recording_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_path_text(path: Path) -> set[str]:
    expanded = path.expanduser()
    values = {str(path), str(expanded)}
    try:
        values.add(str(expanded.resolve(strict=False)))
    except OSError:
        pass
    try:
        values.add(str(expanded.absolute()))
    except OSError:
        pass
    return {value for value in values if value}


def _same_path(left: str, right: Path) -> bool:
    right_values = _normalize_path_text(right)
    if left in right_values:
        return True
    try:
        return Path(left).expanduser().resolve(strict=False) == right.expanduser().resolve(strict=False)
    except OSError:
        return False


def _resolve_dataset_row(registry: Registry, zarr_path: Path) -> Optional[Any]:
    candidates = sorted(_normalize_path_text(zarr_path))
    placeholders = ",".join("?" for _ in candidates)
    rows = registry.conn.execute(
        f"""
        SELECT dataset_id, recording_id, zarr_path, status
        FROM datasets
        WHERE zarr_path IN ({placeholders})
        ORDER BY
            CASE WHEN status IS NULL OR lower(status) NOT IN ('missing', 'deleted') THEN 0 ELSE 1 END,
            last_seen_utc DESC,
            created_utc DESC,
            dataset_id ASC;
        """,
        candidates,
    ).fetchall()
    if rows:
        return rows[0]

    # Registry paths are intended to be absolute, but older rows can differ by
    # symlink expansion or relative spelling. Fall back to a bounded scan.
    rows = registry.conn.execute(
        """
        SELECT dataset_id, recording_id, zarr_path, status
        FROM datasets
        ORDER BY
            CASE WHEN status IS NULL OR lower(status) NOT IN ('missing', 'deleted') THEN 0 ELSE 1 END,
            last_seen_utc DESC,
            created_utc DESC,
            dataset_id ASC;
        """
    ).fetchall()
    for row in rows:
        if _same_path(str(row["zarr_path"]), zarr_path):
            return row
    return None


def sync_dish_mask_registry_status(
    zarr_path: str | Path,
    registry_path: str | Path,
    *,
    method: Optional[str] = None,
    source: str = "dish_mask_save",
    details: Optional[Mapping[str, Any]] = None,
) -> DishMaskRegistrySyncResult:
    """Mark ``dish_mask`` OK for the dataset that owns ``zarr_path``.

    This is intentionally optional. The Zarr attribute remains the source data;
    the registry row is a freshness/status index for workflow selection.
    """

    zarr = Path(zarr_path).expanduser()
    registry_file = Path(registry_path).expanduser()
    registry = Registry(registry_file)
    try:
        row = _resolve_dataset_row(registry, zarr)
        if row is None:
            return DishMaskRegistrySyncResult(
                synced=False,
                status="dataset_not_found",
                message=f"No registry dataset row matches zarr_path={zarr}",
                registry_path=str(registry_file),
                zarr_path=str(zarr),
            )

        try:
            zarr_mtime_ns = zarr.stat().st_mtime_ns
        except OSError:
            zarr_mtime_ns = None

        payload: dict[str, Any] = {
            "reason": "dish_mask_saved",
            "zarr_path": str(zarr),
        }
        if details:
            payload.update(dict(details))

        upsert_recording_step_status(
            registry,
            dataset_id=str(row["dataset_id"]),
            recording_id=row["recording_id"],
            step_name="dish_mask",
            status="ok",
            method=method,
            details_json=payload,
            source=source,
            zarr_mtime_ns=zarr_mtime_ns,
        )
        return DishMaskRegistrySyncResult(
            synced=True,
            status="synced",
            message="dish_mask status marked ok",
            registry_path=str(registry_file),
            zarr_path=str(zarr),
            dataset_id=str(row["dataset_id"]),
            recording_id=row["recording_id"],
        )
    finally:
        registry.close()


__all__ = [
    "DishMaskRegistrySyncResult",
    "sync_dish_mask_registry_status",
]
