"""Filesystem primitives for immutable in-archive coordinate successors."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil


def metadata_tree_sha256(path: Path) -> str:
    """Hash all direct Zarr metadata without reading array payload objects."""

    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("zarr.json")):
        digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_metadata_and_link_payload(source: Path, target: Path) -> dict[str, int]:
    """Copy mutable metadata while hard-linking immutable payload files."""

    if target.exists():
        raise FileExistsError(f"Immutable successor target exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    linked = 0
    copied = 0
    try:
        for source_path in sorted(source.rglob("*")):
            relative = source_path.relative_to(source)
            target_path = target / relative
            if source_path.is_dir():
                target_path.mkdir(exist_ok=False)
            elif source_path.is_file():
                if source_path.name == "zarr.json" or source_path.suffix == ".json":
                    shutil.copy2(source_path, target_path)
                    copied += 1
                else:
                    os.link(source_path, target_path)
                    linked += 1
            else:
                raise ValueError(f"Unsupported source entry: {source_path}")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"metadata_files_copied": copied, "payload_files_hardlinked": linked}


__all__ = ["copy_metadata_and_link_payload", "metadata_tree_sha256"]
