"""Small reusable runtime helpers for storage benchmark adapters."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import resource

import numpy as np
import zarr


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def storage_stats(path: Path) -> dict[str, int]:
    stats = {
        "file_count": 0,
        "metadata_file_count": 0,
        "payload_file_count": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
    }
    for root, _directories, filenames in os.walk(path):
        for filename in filenames:
            item = Path(root) / filename
            result = item.stat()
            stats["file_count"] += 1
            if filename == "zarr.json":
                stats["metadata_file_count"] += 1
            else:
                stats["payload_file_count"] += 1
            stats["apparent_bytes"] += int(result.st_size)
            stats["allocated_bytes"] += int(result.st_blocks * 512)
    return stats


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if platform.system() != "Darwin" else value


def local_environment_manifest() -> dict[str, object]:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "storage_tier": "local_tmp",
        "cache_state": "uncontrolled_exploratory_smoke",
        "request_counting": "unavailable_local_filesystem",
    }


__all__ = [
    "local_environment_manifest",
    "peak_rss_bytes",
    "sha256_array",
    "sha256_file",
    "storage_stats",
    "utc_now",
]
