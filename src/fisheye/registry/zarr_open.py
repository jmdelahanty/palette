"""Registry-local Zarr opening policy.

Registry mutation and refresh paths inspect archives that may still be
mutable, so they must never trust consolidated metadata implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def import_zarr() -> Any:
    """Import Zarr lazily so SQL-only registry commands stay lightweight."""

    try:
        import zarr
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise ModuleNotFoundError(
            "zarr is required for scan/register operations. "
            "Install zarr to read Zarr archives."
        ) from exc
    return zarr


def open_zarr_group_non_consolidated(
    zarr_path: Path,
    *,
    mode: str = "r",
) -> Any:
    """Open a Zarr root without trusting possibly stale metadata."""

    zarr = import_zarr()
    try:
        return zarr.open_group(
            str(zarr_path), mode=mode, use_consolidated=False
        )
    except TypeError:
        return zarr.open_group(
            str(zarr_path), mode=mode, consolidated=False
        )


__all__ = ["import_zarr", "open_zarr_group_non_consolidated"]
