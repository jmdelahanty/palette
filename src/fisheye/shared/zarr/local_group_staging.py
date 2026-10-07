"""Stage one Zarr group of an archive on local disk and sync its changes back.

Many small reads and read-modify-writes are slow on NFS. A writer that holds
the group's write lock can instead copy the group to node-local scratch, do
its work there, and sync back only the files that changed:

* changed and new files are written through a temporary file and renamed;
  files the local work removed are deleted;
* ``protected`` subtrees (for example dense ``masks_roi`` pixels) must be
  byte-identical after the local work, or the sync refuses before writing;
* the group's own ``zarr.json`` (its attributes, where writers keep their
  stale/complete flags) is written last, so an interrupted sync leaves the
  archive's flags as they were before the sync;
* every written file is read back from the archive and checked by SHA-256.

What changed is judged against digests of the local copy taken at staging
time (no second scan of the archive over NFS), so the caller must hold the
group's write lock from staging to sync. Only the group's own ``zarr.json``
may change on the archive meanwhile (for example a stale flag the caller
sets before syncing); it is always rewritten last.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

GROUP_METADATA = "zarr.json"


class StagedSyncRefused(RuntimeError):
    """The local copy changed something the sync must not write back."""


@dataclass(frozen=True)
class StagedGroup:
    archive: Path
    group_path: str
    local_root: Path
    initial: dict[str, str]

    @property
    def archive_group(self) -> Path:
        return self.archive / self.group_path

    @property
    def local_group(self) -> Path:
        return self.local_root / self.group_path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(base: Path) -> dict[str, str]:
    return {
        str(PurePosixPath(*path.relative_to(base).parts)): _digest(path)
        for path in base.rglob("*")
        if path.is_file()
    }


def stage_group(archive: Path, group_path: str, scratch_dir: Path) -> StagedGroup:
    """Copy ``archive/group_path`` (and its parents' group metadata) under ``scratch_dir``."""

    archive = Path(archive).expanduser().resolve()
    group_path = str(PurePosixPath(group_path))
    local_root = Path(scratch_dir) / archive.name
    if local_root.exists():
        raise FileExistsError(f"Staging target already exists: {local_root}")
    parts = PurePosixPath(group_path).parts
    for depth in range(len(parts)):
        parent = archive.joinpath(*parts[:depth])
        target = local_root.joinpath(*parts[:depth])
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / GROUP_METADATA, target / GROUP_METADATA)
    shutil.copytree(archive / group_path, local_root / group_path)
    return StagedGroup(
        archive=archive, group_path=group_path, local_root=local_root, initial=_files(local_root / group_path),
    )


def sync_group_back(staged: StagedGroup, *, protected: tuple[str, ...] = ()) -> dict[str, int]:
    """Write the local group's changes back to the archive; verify by digest."""

    before = staged.initial
    after = _files(staged.local_group)
    for prefix in protected:
        mine = {k: v for k, v in after.items() if k == prefix or k.startswith(prefix + "/")}
        theirs = {k: v for k, v in before.items() if k == prefix or k.startswith(prefix + "/")}
        if mine != theirs:
            raise StagedSyncRefused(f"Protected subtree {prefix!r} changed in the staged copy; not syncing.")
    changed = sorted(k for k, v in after.items() if before.get(k) != v and k != GROUP_METADATA)
    removed = sorted(k for k in before if k not in after)
    for rel in changed:
        _write_file(staged.local_group / rel, staged.archive_group / rel)
    for rel in removed:
        (staged.archive_group / rel).unlink()
    _write_file(staged.local_group / GROUP_METADATA, staged.archive_group / GROUP_METADATA)
    changed.append(GROUP_METADATA)
    for directory in sorted({p for p in staged.archive_group.rglob("*") if p.is_dir()}, reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    for rel in changed:
        if _digest(staged.archive_group / rel) != after[rel]:
            raise RuntimeError(f"Synced file differs on the archive after writing: {rel}")
    if any((staged.archive_group / rel).exists() for rel in removed):
        raise RuntimeError("A file removed in the staged copy still exists on the archive.")
    return {"files_written": len(changed), "files_removed": len(removed), "files_total": len(after)}


def _write_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.staging-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


__all__ = ["GROUP_METADATA", "StagedGroup", "StagedSyncRefused", "stage_group", "sync_group_back"]
