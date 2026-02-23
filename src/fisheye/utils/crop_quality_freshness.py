"""Freshness checks for registry crop-quality rows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def _coerce_mtime_ns(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_crop_quality_row_fresh(
    *,
    zarr_path: Path,
    zarr_mtime_ns: object,
) -> Tuple[bool, Optional[str]]:
    """Compare stored row mtime with current filesystem mtime."""
    stored_mtime_ns = _coerce_mtime_ns(zarr_mtime_ns)
    if stored_mtime_ns is None:
        return False, "missing_registry_zarr_mtime_ns"

    try:
        fs_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except OSError:
        return False, "zarr_path_not_statable"

    if fs_mtime_ns != stored_mtime_ns:
        return False, f"zarr_mtime_mismatch(fs={fs_mtime_ns},registry={stored_mtime_ns})"
    return True, None

