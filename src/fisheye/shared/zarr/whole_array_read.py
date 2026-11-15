"""Read a small Zarr v3 array whole, opening each stored file once.

``array[:]`` on a ``LocalStore`` issues one ``get`` per chunk, and for a
sharded array one ranged ``get`` per shard index and per group of inner
chunks, each of which opens the file again. On NFS every open is a
close-to-open round trip (about 11-20 ms measured on /groups), so a 3 MB
sharded column costs about 0.6 s while the same bytes read in about 25 ms
from local disk.

:func:`read_whole_array` reads every stored file of the array once into a
``MemoryStore`` and decodes it there with the array's own metadata, so the
bytes, codecs, fill values and dtype are exactly those ``array[:]`` uses.
Anything it cannot prove equivalent falls back to ``array[:]``: a store that
is not a ``LocalStore``, a symlink or other non-regular entry in the array
directory, or more stored bytes than ``max_stored_bytes``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore, MemoryStore

DEFAULT_MAX_STORED_BYTES = 64 * 1024 * 1024
_METADATA_NAME = "zarr.json"


def _stored_files(array_dir: Path, max_stored_bytes: int) -> dict[str, bytes] | None:
    """Every stored chunk/shard file below ``array_dir`` keyed like the store, or
    ``None`` when the directory holds anything but regular files and
    directories, or more than ``max_stored_bytes``."""

    files: dict[str, bytes] = {}
    total = 0
    pending = [(array_dir, "")]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                key = prefix + entry.name
                if entry.is_symlink():
                    return None
                if entry.is_dir(follow_symlinks=False):
                    pending.append((Path(entry.path), key + "/"))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    return None
                if key == _METADATA_NAME:
                    continue
                with open(entry.path, "rb") as handle:
                    data = handle.read(max_stored_bytes - total + 1)
                total += len(data)
                if total > max_stored_bytes:
                    return None
                files[key] = data
    return files


def read_whole_array(
    array: Any, *, max_stored_bytes: int = DEFAULT_MAX_STORED_BYTES
) -> np.ndarray:
    """``np.asarray(array[:])``, opening each stored file once when possible."""

    store_path = getattr(array, "store_path", None)
    store = getattr(store_path, "store", None)
    metadata = getattr(array, "metadata", None)
    if (
        not isinstance(store, LocalStore)
        or getattr(metadata, "zarr_format", None) != 3
    ):
        return np.asarray(array[:])
    array_dir = Path(store.root).joinpath(*filter(None, str(store_path.path).split("/")))
    if not array_dir.is_dir():
        return np.asarray(array[:])
    files = _stored_files(array_dir, max_stored_bytes)
    if files is None:
        return np.asarray(array[:])
    prototype = default_buffer_prototype()
    contents = dict(metadata.to_buffer_dict(prototype))
    contents.update(
        {key: prototype.buffer.from_bytes(data) for key, data in files.items()}
    )
    in_memory = zarr.open_array(store=MemoryStore(contents), mode="r", zarr_format=3)
    return np.asarray(in_memory[:])


__all__ = ["DEFAULT_MAX_STORED_BYTES", "read_whole_array"]
