"""Utility helpers for opening Palette Zarr-v3 archives."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import zarr

PathLike = Union[str, Path]
_ZARR_V2_METADATA_NAMES = (".zarray", ".zattrs", ".zgroup", ".zmetadata")


def open_zarr_root(
    zarr_path: PathLike,
    mode: str = "r",
) -> zarr.Group:
    """Open a Palette Zarr-v3 archive without stale consolidated metadata.

    Parameters
    ----------
    zarr_path:
        Filesystem path to the `.zarr` directory.
    mode:
        Access mode passed through to Zarr (default "r").
    """
    zarr_path = Path(zarr_path)
    v2_metadata = [
        name for name in _ZARR_V2_METADATA_NAMES if (zarr_path / name).exists()
    ]
    if v2_metadata:
        raise ValueError(
            "Zarr format 2 is unsupported; refusing to open a store containing "
            f"{', '.join(v2_metadata)}: {zarr_path}"
        )
    group = zarr.open_group(
        str(zarr_path),
        mode=mode,
        zarr_format=3,
        use_consolidated=False,
    )
    try:
        setattr(group, "_palette_fs_path", str(zarr_path.expanduser().resolve()))
        setattr(group, "_palette_open_mode", str(mode))
    except Exception:
        pass
    return group
