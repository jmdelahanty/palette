"""
Utility helpers for opening Palette Zarr archives.

Supports both legacy Zarr v2 (DirectoryStore) and newer Zarr v3 (LocalStore)
layouts so that callers don't need to worry about the underlying storage
format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import zarr

try:
    from zarr.storage import LocalStore  # Zarr v3 store implementation
except ImportError:  # pragma: no cover - LocalStore is unavailable on Zarr < 3
    LocalStore = None  # type: ignore

PathLike = Union[str, Path]


def open_zarr_root(
    zarr_path: PathLike,
    mode: str = "r",
    *,
    cache_metadata: Optional[bool] = None,
) -> zarr.Group:
    """
    Open a Palette Zarr archive, automatically handling v2 and v3 layouts.

    Parameters
    ----------
    zarr_path:
        Filesystem path to the `.zarr` directory.
    mode:
        Access mode passed through to Zarr (default "r").
    cache_metadata:
        Optional flag to propagate to Zarr's open call. Left as a keyword-only
        argument so we can extend behaviour later without breaking callers.
    """
    zarr_path = Path(zarr_path)
    last_error: Optional[Exception] = None
    open_kwargs = {"mode": mode}
    if cache_metadata is not None:
        open_kwargs["cache_metadata"] = cache_metadata

    # First try the Zarr v3 LocalStore (used by Palette schema >= 3).
    if LocalStore is not None:
        try:
            store = LocalStore(str(zarr_path))
            return zarr.open_group(store=store, **open_kwargs)
        except Exception as exc:  # pragma: no cover - requires zarr v3 datasets
            last_error = exc

    # Fallback to the default directory-based loader for legacy archives.
    try:
        return zarr.open_group(str(zarr_path), **open_kwargs)
    except Exception as exc:
        if last_error is not None:
            raise exc from last_error
        raise
