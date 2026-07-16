"""Shared fail-closed publication for completed node-local Zarr run groups.

Analysis-specific materializers own computation, scientific validation, run
completion, and pointer semantics.  This module owns the mechanical transaction:
serialize publication, copy to a hidden sibling, verify the physical copy,
atomically rename it, snapshot parent attributes, and roll back both the new run
and every parent snapshot after any post-rename failure.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import zarr

from ...shared.json_safety import json_attr_safe
from ...shared.zarr_io import open_zarr_root


ATOMIC_RUN_PUBLISHER_SCHEMA_ID = "palette.atomic_run_group_publisher"
ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION = 1
SERIALIZATION_POLICY = "per_recording_advisory_file_lock"

RunValidator = Callable[[Path], Mapping[str, Any]]
PrepareParents = Callable[[zarr.Group], Sequence[zarr.Group]]
CompleteRun = Callable[[zarr.Group, zarr.Group, zarr.Group], None]
VerifyPointers = Callable[[zarr.Group], None]
AfterRename = Callable[[zarr.Group, zarr.Group], Mapping[str, Any] | None]


@dataclass(frozen=True)
class TreeFile:
    relative_path: str
    size_bytes: int


@dataclass(frozen=True)
class TreeInventory:
    files: tuple[TreeFile, ...]
    inventory_sha256: str
    content_sha256: str | None

    @property
    def physical_bytes(self) -> int:
        return int(sum(item.size_bytes for item in self.files))

    def to_json(self) -> dict[str, Any]:
        return {
            "file_count": len(self.files),
            "physical_bytes": self.physical_bytes,
            "inventory_sha256": self.inventory_sha256,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class AtomicRunPublishSpec:
    source_zarr: Path
    local_run_path: Path
    target_run_path: Path
    run_name: str
    lock_suffix: str
    publish_schema_id: str
    policy: str
    rollback_policy: str
    content_checksum: bool = False

    @property
    def lock_path(self) -> Path:
        return self.source_zarr.parent / (
            f".{self.source_zarr.name}.{self.lock_suffix}.lock"
        )

    @property
    def temporary_path(self) -> Path:
        return self.target_run_path.parent / (
            f".{self.run_name}.publish_tmp.{os.getpid()}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inventory_digest(files: Sequence[TreeFile]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(item.size_bytes)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def tree_inventory(root: Path, *, hash_content: bool) -> TreeInventory:
    files = tuple(
        TreeFile(
            relative_path=path.relative_to(root).as_posix(),
            size_bytes=int(path.stat().st_size),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    content_sha256: str | None = None
    if hash_content:
        digest = hashlib.sha256()
        for item in files:
            digest.update(item.relative_path.encode("utf-8"))
            digest.update(b"\0")
            with (root / item.relative_path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\n")
        content_sha256 = digest.hexdigest()
    return TreeInventory(
        files=files,
        inventory_sha256=_inventory_digest(files),
        content_sha256=content_sha256,
    )


def _copy_tree(source: Path, target: Path, *, backend: str) -> None:
    if backend == "python":
        shutil.copytree(source, target)
        return
    if backend == "rsync":
        target.mkdir(parents=True)
        subprocess.run(
            ["rsync", "--archive", f"{source}/", f"{target}/"],
            check=True,
        )
        return
    raise ValueError(f"Unsupported copy backend: {backend!r}.")


def _copy_and_verify(
    source: Path,
    target: Path,
    *,
    backend: str,
    content_checksum: bool,
) -> dict[str, Any]:
    source_inventory = tree_inventory(source, hash_content=content_checksum)
    _copy_tree(source, target, backend=backend)
    if content_checksum and backend == "rsync":
        check = subprocess.run(
            [
                "rsync",
                "--archive",
                "--dry-run",
                "--checksum",
                "--delete",
                "--itemize-changes",
                f"{source}/",
                f"{target}/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if check.stdout.strip():
            raise RuntimeError(
                "Rsync checksum validation found physical copy differences: "
                f"{check.stdout}"
            )
        target_inventory = tree_inventory(target, hash_content=False)
        verification = "rsync_checksum_dry_run"
    else:
        target_inventory = tree_inventory(target, hash_content=content_checksum)
        verification = (
            "sha256_all_physical_files"
            if content_checksum
            else "relative_path_and_size_inventory"
        )

    if source_inventory.files != target_inventory.files:
        raise RuntimeError("Published temporary physical inventory differs from local run.")
    if source_inventory.inventory_sha256 != target_inventory.inventory_sha256:
        raise RuntimeError("Published temporary inventory digest differs from local run.")
    if (
        content_checksum
        and target_inventory.content_sha256 is not None
        and source_inventory.content_sha256 != target_inventory.content_sha256
    ):
        raise RuntimeError("Published temporary content digest differs from local run.")
    return {
        "backend": backend,
        "verification": verification,
        **source_inventory.to_json(),
    }


def _group_path(group: zarr.Group) -> str:
    return str(getattr(group, "path", "")).strip("/")


def _parent_paths(parents: Sequence[zarr.Group]) -> tuple[str, ...]:
    paths = tuple(_group_path(parent) for parent in parents)
    if len(set(paths)) != len(paths):
        raise ValueError(f"prepare_parents returned duplicate group paths: {paths}")
    return paths


def _require_parent_paths(
    parents: Sequence[zarr.Group],
    *,
    expected: Sequence[str],
) -> None:
    observed = _parent_paths(parents)
    if tuple(observed) != tuple(expected):
        raise RuntimeError(
            "prepare_parents returned inconsistent groups during publication: "
            f"expected={tuple(expected)}, observed={observed}"
        )


def _resolve_group(root: zarr.Group, path: str) -> zarr.Group:
    if not path:
        return root
    group = root[path]
    if not isinstance(group, zarr.Group):
        raise TypeError(f"Parent snapshot path is not a Zarr group: {path}")
    return group


def _restore_parent_attrs(
    root: zarr.Group,
    snapshots: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    for path, attrs in snapshots:
        group = _resolve_group(root, path)
        group.attrs.clear()
        group.attrs.update(dict(attrs))


def _require_valid(validation: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    payload = dict(validation)
    if not bool(payload.get("valid")):
        raise RuntimeError(f"{label} validation failed: {payload}")
    return payload


def atomic_publish_run_group(
    spec: AtomicRunPublishSpec,
    *,
    copy_backend: str,
    validate_run: RunValidator,
    prepare_parents: PrepareParents,
    complete_run: CompleteRun,
    verify_pointers: VerifyPointers,
    payload_metadata: Mapping[str, Any] | None = None,
    after_rename: AfterRename | None = None,
) -> dict[str, Any]:
    """Publish one completed local run as a fail-closed transaction."""

    if not spec.source_zarr.is_dir():
        raise FileNotFoundError(f"Authoritative source Zarr not found: {spec.source_zarr}")
    if spec.target_run_path.name != spec.run_name:
        raise ValueError(
            "target_run_path must end in run_name: "
            f"target={spec.target_run_path}, run_name={spec.run_name!r}"
        )
    try:
        spec.target_run_path.resolve().relative_to(spec.source_zarr.resolve())
    except ValueError as exc:
        raise ValueError(
            "target_run_path must be inside the authoritative source Zarr."
        ) from exc
    if not spec.local_run_path.is_dir():
        raise FileNotFoundError(
            f"Local materialized run not found: {spec.local_run_path}"
        )
    local_validation = _require_valid(
        validate_run(spec.local_run_path),
        label="Local run",
    )

    spec.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _atomic_publish_locked(
                spec,
                copy_backend=copy_backend,
                validate_run=validate_run,
                prepare_parents=prepare_parents,
                complete_run=complete_run,
                verify_pointers=verify_pointers,
                payload_metadata=payload_metadata,
                after_rename=after_rename,
                local_validation=local_validation,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _atomic_publish_locked(
    spec: AtomicRunPublishSpec,
    *,
    copy_backend: str,
    validate_run: RunValidator,
    prepare_parents: PrepareParents,
    complete_run: CompleteRun,
    verify_pointers: VerifyPointers,
    payload_metadata: Mapping[str, Any] | None,
    after_rename: AfterRename | None,
    local_validation: Mapping[str, Any],
) -> dict[str, Any]:
    root = open_zarr_root(spec.source_zarr, mode="a")
    parents = tuple(prepare_parents(root))
    if not parents:
        raise ValueError("prepare_parents must return at least the target parent group.")
    parent_paths = _parent_paths(parents)
    target_parent = parents[-1]
    if spec.run_name in target_parent or spec.target_run_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {spec.target_run_path}"
        )
    parent_snapshots = tuple(
        (path, dict(parent.attrs)) for path, parent in zip(parent_paths, parents)
    )
    snapshot_payload = {
        path or "/": json_attr_safe(dict(attrs))
        for path, attrs in parent_snapshots
    }

    spec.target_run_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = spec.temporary_path
    if temporary.exists():
        raise FileExistsError(
            f"Refusing existing publish temporary path: {temporary}"
        )

    published = False
    try:
        copy_started = time.perf_counter()
        physical_copy = _copy_and_verify(
            spec.local_run_path,
            temporary,
            backend=copy_backend,
            content_checksum=bool(spec.content_checksum),
        )
        temporary_validation = _require_valid(
            validate_run(temporary),
            label="Temporary run",
        )
        os.replace(temporary, spec.target_run_path)
        published = True

        root = open_zarr_root(spec.source_zarr, mode="a")
        parents = tuple(prepare_parents(root))
        _require_parent_paths(parents, expected=parent_paths)
        target_parent = parents[-1]
        run_group = target_parent[spec.run_name]
        if not isinstance(run_group, zarr.Group):
            raise TypeError("Published target is not a Zarr group.")

        post_rename_metadata = (
            dict(after_rename(root, run_group) or {})
            if after_rename is not None
            else {}
        )
        payload = {
            **dict(payload_metadata or {}),
            **post_rename_metadata,
            "schema_id": spec.publish_schema_id,
            "publisher_contract": {
                "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
                "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
            },
            "policy": spec.policy,
            "serialization_policy": SERIALIZATION_POLICY,
            "rollback_policy": spec.rollback_policy,
            "published_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "lsb_jobid": os.environ.get("LSB_JOBID"),
            "source_zarr": str(spec.source_zarr),
            "publication_source_run_path": str(spec.local_run_path),
            "target_run_path": str(spec.target_run_path),
            "hidden_temporary_policy": "same_parent_hidden_sibling_then_os_replace",
            "copy_duration_seconds": float(time.perf_counter() - copy_started),
            "physical_copy": physical_copy,
            "parent_attrs_before": snapshot_payload,
            "local_validation": dict(local_validation),
            "temporary_validation": temporary_validation,
        }
        run_group.attrs["cluster_output_staging"] = json_attr_safe(payload)

        pre_pointer_validation = _require_valid(
            validate_run(spec.target_run_path),
            label="Published pre-pointer",
        )
        payload["pre_pointer_validation"] = pre_pointer_validation
        run_group.attrs["cluster_output_staging"] = json_attr_safe(payload)

        complete_run(root, target_parent, run_group)
        final_validation = _require_valid(
            validate_run(spec.target_run_path),
            label="Published final run",
        )
        pointer_root = open_zarr_root(spec.source_zarr, mode="r")
        verify_pointers(pointer_root)
        payload["final_validation"] = final_validation

        final_root = open_zarr_root(spec.source_zarr, mode="a")
        final_parents = tuple(prepare_parents(final_root))
        _require_parent_paths(final_parents, expected=parent_paths)
        payload["parent_attrs_after"] = {
            path or "/": json_attr_safe(dict(parent.attrs))
            for path, parent in zip(parent_paths, final_parents)
        }
        final_run = final_parents[-1][spec.run_name]
        final_run.attrs["cluster_output_staging"] = json_attr_safe(payload)
        return json_attr_safe(payload)
    except BaseException:
        if published and spec.target_run_path.exists():
            shutil.rmtree(spec.target_run_path)
        rollback_root = open_zarr_root(spec.source_zarr, mode="a")
        _restore_parent_attrs(rollback_root, parent_snapshots)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "ATOMIC_RUN_PUBLISHER_SCHEMA_ID",
    "ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION",
    "AtomicRunPublishSpec",
    "SERIALIZATION_POLICY",
    "atomic_publish_run_group",
    "tree_inventory",
]
