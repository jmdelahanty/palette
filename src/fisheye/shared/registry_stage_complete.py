"""Backward-compatible import shim for registry stage-completion helpers.

New code should import from :mod:`fisheye.registry.stage_complete`. Keeping
this shim avoids breaking older scripts while the module boundary is cleaned up.
"""

from __future__ import annotations

from fisheye.registry.stage_complete import (  # noqa: F401
    DatasetMetadata,
    RegistryInput,
    emit_stage_completion,
    extract_dataset_metadata,
    safe_zarr_mtime_ns,
)

__all__ = [
    "DatasetMetadata",
    "RegistryInput",
    "emit_stage_completion",
    "extract_dataset_metadata",
    "safe_zarr_mtime_ns",
]
