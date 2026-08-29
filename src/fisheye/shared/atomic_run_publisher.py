"""Shared fail-closed publication for completed node-local Zarr run groups.

Analysis-specific materializers own computation, scientific validation, run
completion, and pointer semantics.  This module owns the mechanical transaction:
serialize publication, copy to a hidden sibling, verify the physical copy,
atomically rename it, snapshot parent attributes, and leave an immutable failed
public tombstone while conditionally restoring owned parent mutations after any
post-rename failure.
"""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import zarr

from .json_safety import json_attr_safe
from .runtime_telemetry import PhaseTelemetry
from .zarr_helpers import (
    ARCHIVE_METADATA_PUBLICATION_LOCK_SUFFIX,
    archive_metadata_publication_lock,
    archive_metadata_publication_lock_path,
)
from .zarr_io import open_zarr_root


ATOMIC_RUN_PUBLISHER_SCHEMA_ID = "palette.atomic_run_group_publisher"
ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION = 1
SERIALIZATION_POLICY = "per_recording_advisory_file_lock"
DEFAULT_COPY_INTEGRITY_POLICY = "content_checksum_required_v1"
DEFAULT_COPY_CONTENT_CHECKSUM = True
ATOMIC_PUBLICATION_OWNER_ATTR = "atomic_publication_owner_uuid"
ATOMIC_PUBLICATION_TOMBSTONE_ATTR = "atomic_publication_tombstone"

# This order is a report-only execution contract, not part of scientific
# identity.  Successful candidate publication executes every phase except the
# checksum dry run, which is present only for the rsync+content-checksum path.
# Keep it explicit so candidate execution receipts can reject fabricated,
# reordered, or silently dropped mechanical-publication timing.
ATOMIC_RUN_PUBLISHER_PHASE_ORDER = (
    "local_run_validation",
    "publication_lock_wait",
    "initial_authoritative_open_and_parent_prepare",
    "parent_attribute_snapshot",
    "source_physical_inventory",
    "physical_tree_copy",
    "rsync_checksum_dry_run",
    "target_physical_inventory",
    "physical_inventory_comparison",
    "hidden_target_owner_stamp_and_verification",
    "hidden_target_validation",
    "atomic_rename",
    "renamed_target_owner_verification",
    "renamed_authoritative_open_and_parent_prepare",
    "post_rename_binding",
    "initial_publication_metadata_write",
    "pre_pointer_validation",
    "pre_pointer_metadata_write",
    "completion_and_pointer_publication",
    "final_run_validation",
    "pointer_verification",
    "final_authoritative_open_and_parent_snapshot",
    "final_publication_metadata_write",
    "activation_preflight",
    "selector_activation",
    "publication_lock_release",
)
ATOMIC_RUN_PUBLISHER_OPTIONAL_SUCCESS_PHASES = frozenset(
    {"rsync_checksum_dry_run"}
)

RunValidator = Callable[[Path], Mapping[str, Any]]
PrepareParents = Callable[[zarr.Group], Sequence[zarr.Group]]
CompleteRun = Callable[[zarr.Group, zarr.Group, zarr.Group], None]
ActivateRun = Callable[[zarr.Group, zarr.Group, zarr.Group], None]
RollbackActivation = Callable[[], None]
VerifyPointers = Callable[[zarr.Group], None]
AfterRename = Callable[
    [zarr.Group, zarr.Group, Mapping[str, Any]],
    Mapping[str, Any] | None,
]
RepairFailedPublicationVisibility = Callable[[Path], None]


# Consolidated metadata belongs to the complete archive, not to one run family.
# Every publication through this shared primitive must therefore serialize on
# one recording-wide lock even when callers retain stage-specific labels for
# provenance and diagnostics.
ARCHIVE_PUBLICATION_LOCK_SUFFIX = ARCHIVE_METADATA_PUBLICATION_LOCK_SUFFIX


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
    # Maintained publishers fail safe: Python copies hash every physical file,
    # while rsync copies use a checksum dry-run. Historical compatibility
    # callers must opt out explicitly and retain path/size verification.
    content_checksum: bool = DEFAULT_COPY_CONTENT_CHECKSUM
    # Exact-manifest publishers may seal the immutable run metadata before
    # import and persist this operational receipt outside the run.  They still
    # receive the complete returned receipt, but opt out of post-seal attrs.
    persist_run_receipt: bool = True
    publication_owner_attr: str | None = None
    selector_owner_attr: str | None = None
    selector_generation_attr: str | None = None
    owned_parent_attr_names: tuple[tuple[str, ...], ...] = ()
    publication_attempt_uuid: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    @property
    def lock_path(self) -> Path:
        return archive_metadata_publication_lock_path(self.source_zarr)

    @property
    def temporary_path(self) -> Path:
        try:
            attempt = uuid.UUID(str(self.publication_attempt_uuid)).hex
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                "publication_attempt_uuid must be one canonical UUID token."
            ) from exc
        host = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in socket.gethostname()
        ).strip("_") or "unknown-host"
        job = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(os.environ.get("LSB_JOBID") or "local")
        ).strip("_") or "local"
        return self.target_run_path.parent / (
            f".{self.run_name}.publish_tmp.{host}.{job}.{os.getpid()}."
            f"{attempt}"
        )


@dataclass(frozen=True)
class _AttrState:
    present: bool
    value: Any


@dataclass(frozen=True)
class _OwnedParentMutation:
    parent_path: str
    attr_name: str
    previous: _AttrState
    attempted: _AttrState


@dataclass(frozen=True)
class _OwnedParentRollbackReceipt:
    lease_parent_path: str
    lease_attr: str
    lease: _AttrState
    generation_attr: str | None
    generation: _AttrState | None
    mutations: tuple[_OwnedParentMutation, ...]


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
    telemetry: PhaseTelemetry,
) -> dict[str, Any]:
    with telemetry.phase("source_physical_inventory"):
        source_inventory = tree_inventory(source, hash_content=content_checksum)
    with telemetry.phase("physical_tree_copy"):
        _copy_tree(source, target, backend=backend)
    if content_checksum and backend == "rsync":
        with telemetry.phase("rsync_checksum_dry_run"):
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
        with telemetry.phase("target_physical_inventory"):
            target_inventory = tree_inventory(target, hash_content=False)
        verification = "rsync_checksum_dry_run"
    else:
        with telemetry.phase("target_physical_inventory"):
            target_inventory = tree_inventory(target, hash_content=content_checksum)
        verification = (
            "sha256_all_physical_files"
            if content_checksum
            else "relative_path_and_size_inventory"
        )

    with telemetry.phase("physical_inventory_comparison"):
        if source_inventory.files != target_inventory.files:
            raise RuntimeError(
                "Published temporary physical inventory differs from local run."
            )
        if source_inventory.inventory_sha256 != target_inventory.inventory_sha256:
            raise RuntimeError(
                "Published temporary inventory digest differs from local run."
            )
        if (
            content_checksum
            and target_inventory.content_sha256 is not None
            and source_inventory.content_sha256 != target_inventory.content_sha256
        ):
            raise RuntimeError(
                "Published temporary content digest differs from local run."
            )
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


def _attr_state(attrs: Any, name: str) -> _AttrState:
    return _AttrState(
        present=name in attrs,
        value=copy.deepcopy(attrs.get(name)),
    )


def _states_equal(left: _AttrState, right: _AttrState) -> bool:
    return left.present is right.present and (
        not left.present or left.value == right.value
    )


def _restore_attr_state(attrs: Any, name: str, state: _AttrState) -> None:
    if state.present:
        attrs[name] = copy.deepcopy(state.value)
    elif name in attrs:
        del attrs[name]


def _capture_owned_parent_rollback_receipt(
    root: zarr.Group,
    snapshots: Sequence[tuple[str, Mapping[str, Any]]],
    owned_names: Sequence[Sequence[str]],
    *,
    lease_parent_path: str,
    lease_attr: str,
    generation_attr: str | None,
    publication_owner: str,
) -> _OwnedParentRollbackReceipt | None:
    """Capture exact values only while this attempt still owns its lease epoch."""

    if len(snapshots) != len(owned_names):
        raise ValueError(
            "Owned parent rollback names must align with parent snapshots."
        )
    lease_parent = _resolve_group(root, lease_parent_path)
    lease = _attr_state(lease_parent.attrs, lease_attr)
    if (
        not lease.present
        or not isinstance(lease.value, Mapping)
        or lease.value.get("owner_uuid") != publication_owner
    ):
        return None

    generation: _AttrState | None = None
    if generation_attr is not None:
        generation = _attr_state(lease_parent.attrs, generation_attr)
        if not generation.present or type(generation.value) is not int:
            return None
        leased_generation = lease.value.get("next_generation")
        if leased_generation is not None and generation.value != leased_generation:
            return None

    mutations: list[_OwnedParentMutation] = []
    for (path, snapshot), names in zip(snapshots, owned_names, strict=True):
        attrs = _resolve_group(root, path).attrs
        for name in names:
            previous = _AttrState(
                present=name in snapshot,
                value=copy.deepcopy(snapshot.get(name)),
            )
            attempted = _attr_state(attrs, name)
            if _states_equal(previous, attempted):
                continue
            mutations.append(
                _OwnedParentMutation(
                    parent_path=path,
                    attr_name=name,
                    previous=previous,
                    attempted=attempted,
                )
            )

    # Keep ownership metadata alive until all selector/guard values are restored.
    # Generation is restored immediately before the lease; the lease itself is
    # always the final mutation.
    mutations.sort(
        key=lambda mutation: (
            2
            if mutation.parent_path == lease_parent_path
            and mutation.attr_name == lease_attr
            else 1
            if generation_attr is not None
            and mutation.parent_path == lease_parent_path
            and mutation.attr_name == generation_attr
            else 0
        )
    )
    return _OwnedParentRollbackReceipt(
        lease_parent_path=lease_parent_path,
        lease_attr=lease_attr,
        lease=lease,
        generation_attr=generation_attr,
        generation=generation,
        mutations=tuple(mutations),
    )


def _restore_owned_parent_attrs(
    root: zarr.Group,
    receipt: _OwnedParentRollbackReceipt,
) -> None:
    """Restore an exact mutation log, stopping immediately on ownership loss."""

    errors: list[str] = []
    expected_generation = receipt.generation
    for mutation in receipt.mutations:
        try:
            lease_parent = _resolve_group(root, receipt.lease_parent_path)
            if not _states_equal(
                _attr_state(lease_parent.attrs, receipt.lease_attr),
                receipt.lease,
            ):
                break
            if receipt.generation_attr is not None:
                assert expected_generation is not None
                if not _states_equal(
                    _attr_state(lease_parent.attrs, receipt.generation_attr),
                    expected_generation,
                ):
                    break

            attrs = _resolve_group(root, mutation.parent_path).attrs
            current = _attr_state(attrs, mutation.attr_name)
            if _states_equal(current, mutation.previous):
                if (
                    receipt.generation_attr is not None
                    and mutation.parent_path == receipt.lease_parent_path
                    and mutation.attr_name == receipt.generation_attr
                ):
                    expected_generation = mutation.previous
                continue
            if not _states_equal(current, mutation.attempted):
                break

            _restore_attr_state(attrs, mutation.attr_name, mutation.previous)
            if not _states_equal(
                _attr_state(attrs, mutation.attr_name),
                mutation.previous,
            ):
                errors.append(
                    f"verify /{mutation.parent_path}@{mutation.attr_name}: "
                    "persisted value differs"
                )
                break
            if (
                receipt.generation_attr is not None
                and mutation.parent_path == receipt.lease_parent_path
                and mutation.attr_name == receipt.generation_attr
            ):
                expected_generation = mutation.previous
        except BaseException as exc:  # pragma: no cover - hostile store
            errors.append(
                f"restore /{mutation.parent_path}@{mutation.attr_name}: {exc}"
            )
            break
    if errors:
        raise RuntimeError(
            f"Owned parent selector rollback was incomplete: {errors!r}."
        )


def _persist_failed_public_tombstone(
    spec: AtomicRunPublishSpec,
    *,
    owner_attr: str,
    publication_owner: str,
    failure: BaseException,
    rollback_errors: list[str],
) -> None:
    """Leave one immutable, owner-bound failed child at its public run path."""

    tombstone = {
        "schema_id": "palette.atomic_publication_tombstone",
        "schema_version": 1,
        "failed_at_utc": _utc_now(),
        "publication_owner_attr": owner_attr,
        "publication_owner_uuid": publication_owner,
        "run_name": spec.run_name,
        "run_path": str(spec.target_run_path),
        "public_path_retained": True,
        "selector_eligible": False,
        "retry_policy": "new_immutable_run_name_required",
        "failure_type": type(failure).__name__,
        "failure": str(failure),
    }
    try:
        run = _fresh_owned_public_target(
            spec,
            owner_attr=owner_attr,
            publication_owner=publication_owner,
            mode="a",
        )
        if run is None:
            return
        run.attrs["stage_selector_eligible"] = False

        run = _fresh_owned_public_target(
            spec,
            owner_attr=owner_attr,
            publication_owner=publication_owner,
            mode="a",
        )
        if run is None:
            return
        if "palette_run_completed_at_utc" in run.attrs:
            del run.attrs["palette_run_completed_at_utc"]

        run = _fresh_owned_public_target(
            spec,
            owner_attr=owner_attr,
            publication_owner=publication_owner,
            mode="a",
        )
        if run is None:
            return
        run.attrs["palette_run_completion_status"] = "failed"

        run = _fresh_owned_public_target(
            spec,
            owner_attr=owner_attr,
            publication_owner=publication_owner,
            mode="a",
        )
        if run is None:
            return
        if "publication_status" in run.attrs:
            run.attrs["publication_status"] = "failed"

        run = _fresh_owned_public_target(
            spec,
            owner_attr=owner_attr,
            publication_owner=publication_owner,
            mode="a",
        )
        if run is None:
            return
        run.attrs["palette_run_error"] = str(failure)

        run = _fresh_owned_public_target(
            spec,
            owner_attr=owner_attr,
            publication_owner=publication_owner,
            mode="a",
        )
        if run is None:
            return
        run.attrs[ATOMIC_PUBLICATION_TOMBSTONE_ATTR] = json_attr_safe(tombstone)

        check = _fresh_owned_public_target(
            spec,
            owner_attr=owner_attr,
            publication_owner=publication_owner,
            mode="r",
        )
        if check is None:
            return
        if (
            check.attrs.get("stage_selector_eligible") is not False
            or check.attrs.get("palette_run_completion_status") != "failed"
            or "palette_run_completed_at_utc" in check.attrs
            or check.attrs.get(ATOMIC_PUBLICATION_TOMBSTONE_ATTR)
            != json_attr_safe(tombstone)
        ):
            raise RuntimeError("failed public tombstone did not persist exactly")
    except BaseException as exc:  # pragma: no cover - hostile store
        rollback_errors.append(f"persist failed public tombstone: {exc}")


def _persisted_publication_owner(path: Path, attr_name: str) -> Any:
    group = open_zarr_root(path, mode="r")
    return group.attrs.get(attr_name)


def _target_group_path(spec: AtomicRunPublishSpec) -> str:
    return spec.target_run_path.resolve().relative_to(
        spec.source_zarr.resolve()
    ).as_posix()


def _fresh_owned_public_target(
    spec: AtomicRunPublishSpec,
    *,
    owner_attr: str,
    publication_owner: str,
    mode: str,
) -> zarr.Group | None:
    """Fresh-resolve one exact public child and prove its original owner."""

    root = open_zarr_root(spec.source_zarr, mode=mode)
    expected_path = _target_group_path(spec)
    try:
        run = _resolve_group(root, expected_path)
    except (KeyError, FileNotFoundError):
        return None
    if (
        _group_path(run) != expected_path
        or run.attrs.get(owner_attr) != publication_owner
    ):
        return None
    return run


def _verify_failed_publication_visibility(
    spec: AtomicRunPublishSpec,
    *,
    owner_attr: str,
    publication_owner: str,
    expected_attrs: Mapping[str, Any],
) -> None:
    """Require the failed tombstone to agree in direct and inline metadata."""

    direct_root = zarr.open_group(
        str(spec.source_zarr),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )
    consolidated_root = zarr.open_group(
        str(spec.source_zarr),
        mode="r",
        zarr_format=3,
        use_consolidated=True,
    )
    target_path = _target_group_path(spec)
    direct = _resolve_group(direct_root, target_path)
    consolidated = _resolve_group(consolidated_root, target_path)
    direct_attrs = dict(direct.attrs)
    consolidated_attrs = dict(consolidated.attrs)
    expected = copy.deepcopy(dict(expected_attrs))
    for label, attrs in (
        ("direct", direct_attrs),
        ("consolidated", consolidated_attrs),
    ):
        if attrs != expected:
            raise RuntimeError(f"{label} failed tombstone metadata changed")
        if attrs.get(owner_attr) != publication_owner:
            raise RuntimeError(f"{label} failed tombstone owner differs")
        if attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError(f"{label} failed tombstone is selector eligible")
        if attrs.get("palette_run_completion_status") != "failed":
            raise RuntimeError(f"{label} failed tombstone is not failed")
        if ATOMIC_PUBLICATION_TOMBSTONE_ATTR not in attrs:
            raise RuntimeError(f"{label} failed tombstone receipt is missing")


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
    activate_run: ActivateRun | None = None,
    rollback_activation: RollbackActivation | None = None,
    repair_failed_publication_visibility: (
        RepairFailedPublicationVisibility | None
    ) = None,
    accept_persisted_activation_on_callback_error: bool = True,
) -> dict[str, Any]:
    """Publish one completed local run as a fail-closed transaction.

    ``after_rename`` receives the exact verified physical-copy receipt before
    it performs any authoritative binding. Stage publishers can therefore seal
    transfer evidence without reconstructing or re-running the copy gate.

    By default, an activation callback that raises after persisting the exact
    final eligibility write is treated as an interrupted acknowledgement of a
    successful commit. Callers whose activation callback also owns fallible
    consolidated-visibility validation must disable that recovery so callback
    failure enters the stage-specific rollback and tombstone path.
    """

    telemetry = PhaseTelemetry(
        materializer="atomic_run_publisher",
        context={
            "publish_schema_id": spec.publish_schema_id,
            "copy_backend": copy_backend,
            "content_checksum": bool(spec.content_checksum),
            "archive_publication_lock": str(spec.lock_path),
            "caller_lock_suffix": spec.lock_suffix,
        },
    )

    if rollback_activation is not None and (
        spec.selector_owner_attr is not None
        or spec.selector_generation_attr is not None
        or bool(spec.owned_parent_attr_names)
    ):
        raise ValueError(
            "A stage-specific rollback_activation receipt and generic pre-copy "
            "parent rollback are mutually exclusive; choose one exact selector "
            "rollback authority."
        )
    if type(accept_persisted_activation_on_callback_error) is not bool:
        raise TypeError(
            "accept_persisted_activation_on_callback_error must be one boolean."
        )
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
    with telemetry.phase("local_run_validation"):
        local_validation = _require_valid(
            validate_run(spec.local_run_path),
            label="Local run",
        )

    lock_context = archive_metadata_publication_lock(spec.source_zarr)
    with telemetry.phase("publication_lock_wait"):
        lock_context.__enter__()
    try:
        result = _atomic_publish_locked(
            spec,
            copy_backend=copy_backend,
            validate_run=validate_run,
            prepare_parents=prepare_parents,
            complete_run=complete_run,
            verify_pointers=verify_pointers,
            payload_metadata=payload_metadata,
            after_rename=after_rename,
            activate_run=activate_run,
            rollback_activation=rollback_activation,
            repair_failed_publication_visibility=(
                repair_failed_publication_visibility
            ),
            accept_persisted_activation_on_callback_error=(
                accept_persisted_activation_on_callback_error
            ),
            local_validation=local_validation,
            telemetry=telemetry,
        )
    finally:
        with telemetry.phase("publication_lock_release"):
            lock_context.__exit__(None, None, None)
    result["runtime_telemetry"] = telemetry.to_json()
    return result


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
    activate_run: ActivateRun | None,
    rollback_activation: RollbackActivation | None,
    repair_failed_publication_visibility: (
        RepairFailedPublicationVisibility | None
    ),
    accept_persisted_activation_on_callback_error: bool,
    local_validation: Mapping[str, Any],
    telemetry: PhaseTelemetry,
) -> dict[str, Any]:
    publication_owner_attr = (
        spec.publication_owner_attr or ATOMIC_PUBLICATION_OWNER_ATTR
    )
    expected_publication_owner: str
    if spec.publication_owner_attr is not None:
        expected_publication_owner = _persisted_publication_owner(
            spec.local_run_path,
            spec.publication_owner_attr,
        )
        if not isinstance(expected_publication_owner, str) or not expected_publication_owner:
            raise RuntimeError(
                "Atomic publication source lacks its required publication owner."
            )
    else:
        expected_publication_owner = str(uuid.uuid4())
    with telemetry.phase("initial_authoritative_open_and_parent_prepare"):
        root = open_zarr_root(spec.source_zarr, mode="a")
        parents = tuple(prepare_parents(root))
    if not parents:
        raise ValueError("prepare_parents must return at least the target parent group.")
    parent_paths = _parent_paths(parents)
    if spec.owned_parent_attr_names and len(spec.owned_parent_attr_names) != len(
        parents
    ):
        raise ValueError(
            "owned_parent_attr_names must align exactly with prepare_parents."
        )
    if spec.selector_owner_attr is None:
        if spec.selector_generation_attr is not None or spec.owned_parent_attr_names:
            raise ValueError(
                "Owned parent rollback declarations require selector_owner_attr."
            )
    else:
        if not spec.owned_parent_attr_names:
            raise ValueError(
                "selector_owner_attr requires an explicit owned parent mutation surface."
            )
        first_owned = spec.owned_parent_attr_names[0]
        if spec.selector_owner_attr not in first_owned:
            raise ValueError(
                "The selector lease attribute must belong to the first parent mutation surface."
            )
        if (
            spec.selector_generation_attr is not None
            and spec.selector_generation_attr not in first_owned
        ):
            raise ValueError(
                "The selector generation attribute must belong to the first parent "
                "mutation surface."
            )
    target_parent = parents[-1]
    if spec.run_name in target_parent or spec.target_run_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {spec.target_run_path}"
        )
    with telemetry.phase("parent_attribute_snapshot"):
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

    final_activation_attempted = False
    result: dict[str, Any] = {}
    try:
        copy_started = time.perf_counter()
        physical_copy = _copy_and_verify(
            spec.local_run_path,
            temporary,
            backend=copy_backend,
            content_checksum=bool(spec.content_checksum),
            telemetry=telemetry,
        )
        # Before a tree can appear at a public canonical child path, persist one
        # exact attempt owner and fail-closed selector eligibility in the hidden
        # sibling. This also owner-binds older materializers that do not yet
        # carry a stage-specific publication-owner attribute.
        with telemetry.phase("hidden_target_owner_stamp_and_verification"):
            temporary_group = open_zarr_root(temporary, mode="a")
            temporary_group.attrs[publication_owner_attr] = expected_publication_owner
            temporary_group.attrs["stage_selector_eligible"] = False
            temporary_check = open_zarr_root(temporary, mode="r")
            if (
                temporary_check.attrs.get(publication_owner_attr)
                != expected_publication_owner
                or temporary_check.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Hidden publication target did not persist its exact owner and "
                    "selector-ineligible state."
                )
        with telemetry.phase("hidden_target_validation"):
            temporary_validation = _require_valid(
                validate_run(temporary),
                label="Temporary run",
            )
        with telemetry.phase("atomic_rename"):
            os.replace(temporary, spec.target_run_path)

        # The owner is selected from the local source before copying.  Prove
        # the renamed tree still carries that exact owner before invoking any
        # callback or mutating authoritative metadata.
        with telemetry.phase("renamed_target_owner_verification"):
            renamed_owner = _persisted_publication_owner(
                spec.target_run_path,
                publication_owner_attr,
            )
            if renamed_owner != expected_publication_owner:
                raise RuntimeError(
                    "Renamed atomic-publication target changed publication "
                    "owner before authoritative callbacks."
                )

        with telemetry.phase("renamed_authoritative_open_and_parent_prepare"):
            root = open_zarr_root(spec.source_zarr, mode="a")
            parents = tuple(prepare_parents(root))
            _require_parent_paths(parents, expected=parent_paths)
            target_parent = parents[-1]
            run_group = target_parent[spec.run_name]
            if not isinstance(run_group, zarr.Group):
                raise TypeError("Published target is not a Zarr group.")

        with telemetry.phase("post_rename_binding"):
            post_rename_metadata = (
                dict(after_rename(root, run_group, dict(physical_copy)) or {})
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
            "publication_owner_attr": publication_owner_attr,
            "publication_owner_uuid": expected_publication_owner,
            "failed_public_child_policy": (
                "retain_owner_bound_selector_ineligible_tombstone"
            ),
            "hidden_temporary_policy": "same_parent_hidden_sibling_then_os_replace",
            "copy_duration_seconds": float(time.perf_counter() - copy_started),
            "physical_copy": physical_copy,
            "parent_attrs_before": snapshot_payload,
            "local_validation": dict(local_validation),
            "temporary_validation": temporary_validation,
        }
        with telemetry.phase("initial_publication_metadata_write"):
            if spec.persist_run_receipt:
                run_group.attrs["cluster_output_staging"] = json_attr_safe(payload)

        with telemetry.phase("pre_pointer_validation"):
            pre_pointer_validation = _require_valid(
                validate_run(spec.target_run_path),
                label="Published pre-pointer",
            )
        payload["pre_pointer_validation"] = pre_pointer_validation
        with telemetry.phase("pre_pointer_metadata_write"):
            if spec.persist_run_receipt:
                run_group.attrs["cluster_output_staging"] = json_attr_safe(payload)

        with telemetry.phase("completion_and_pointer_publication"):
            complete_run(root, target_parent, run_group)
        with telemetry.phase("final_run_validation"):
            final_validation = _require_valid(
                validate_run(spec.target_run_path),
                label="Published final run",
            )
        with telemetry.phase("pointer_verification"):
            pointer_root = open_zarr_root(spec.source_zarr, mode="r")
            verify_pointers(pointer_root)
        payload["final_validation"] = final_validation

        with telemetry.phase("final_authoritative_open_and_parent_snapshot"):
            final_root = open_zarr_root(spec.source_zarr, mode="a")
            final_parents = tuple(prepare_parents(final_root))
            _require_parent_paths(final_parents, expected=parent_paths)
            payload["parent_attrs_after"] = {
                path or "/": json_attr_safe(dict(parent.attrs))
                for path, parent in zip(parent_paths, final_parents)
            }
            final_run = final_parents[-1][spec.run_name]
        with telemetry.phase("final_publication_metadata_write"):
            if spec.persist_run_receipt:
                final_run.attrs["cluster_output_staging"] = json_attr_safe(payload)
        result = json_attr_safe(payload)
        if activate_run is not None:
            # Reopen after the final publisher-owned metadata write. A stale
            # handle or same-name replacement is never activation authority.
            with telemetry.phase("activation_preflight"):
                activation_root = open_zarr_root(spec.source_zarr, mode="a")
                activation_parents = tuple(prepare_parents(activation_root))
                _require_parent_paths(activation_parents, expected=parent_paths)
                activation_run = activation_parents[-1][spec.run_name]
                expected_run_path = _target_group_path(spec)
                if (
                    not isinstance(activation_run, zarr.Group)
                    or _group_path(activation_run) != expected_run_path
                    or activation_run.attrs.get(publication_owner_attr)
                    != expected_publication_owner
                    or activation_run.attrs.get("palette_run_completion_status")
                    != "complete"
                    or activation_run.attrs.get("stage_selector_eligible") is not False
                ):
                    raise RuntimeError(
                        "Final atomic-publication activation lost its exact owned, "
                        "selector-ineligible canonical child."
                    )
            # This callback owns the literal selector-eligibility commit. All
            # fallible publisher validation and metadata writes are complete.
            final_activation_attempted = True
            with telemetry.phase("selector_activation"):
                activate_run(
                    activation_root,
                    activation_parents[-1],
                    activation_run,
                )
        return result
    except BaseException as exc:
        # A literal final-write activator may lose only its acknowledgement.
        # Visibility-aware activators disable this recovery because a callback
        # error means their complete publication contract did not pass.
        if (
            final_activation_attempted
            and accept_persisted_activation_on_callback_error
        ):
            try:
                committed = _fresh_owned_public_target(
                    spec,
                    owner_attr=publication_owner_attr,
                    publication_owner=expected_publication_owner,
                    mode="r",
                )
                if (
                    committed is not None
                    and committed.attrs.get("stage_selector_eligible") is True
                    and committed.attrs.get("palette_run_completion_status")
                    == "complete"
                ):
                    # The callback's literal final write persisted before its
                    # acknowledgement failed. This fresh exact-owner proof is
                    # the terminal success path; perform no further work.
                    return result
            except BaseException:
                pass
        rollback_errors: list[str] = []
        if rollback_activation is not None:
            try:
                rollback_activation()
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(
                    f"cancel deferred selector activation: {rollback_exc}"
                )
        # ``os.replace`` can complete the rename and still raise. Inspect any
        # public target that appeared, but never delete, rename, or reuse it.
        # Exact ownership authorizes only an in-place failed/ineligible
        # tombstone; a foreign replacement is left completely untouched.
        owned_public_tombstoned = False
        expected_failed_publication_attrs: dict[str, Any] | None = None
        if spec.target_run_path.exists():
            try:
                target_owned = (
                    _persisted_publication_owner(
                        spec.target_run_path,
                        publication_owner_attr,
                    )
                    == expected_publication_owner
                )
            except BaseException as owner_exc:
                target_owned = False
                rollback_errors.append(
                    f"resolve failed target publication owner: {owner_exc}"
                )
            if target_owned:
                _persist_failed_public_tombstone(
                    spec,
                    owner_attr=publication_owner_attr,
                    publication_owner=expected_publication_owner,
                    failure=exc,
                    rollback_errors=rollback_errors,
                )
                try:
                    failed = _fresh_owned_public_target(
                        spec,
                        owner_attr=publication_owner_attr,
                        publication_owner=expected_publication_owner,
                        mode="r",
                    )
                    owned_public_tombstoned = bool(
                        failed is not None
                        and failed.attrs.get("stage_selector_eligible") is False
                        and failed.attrs.get("palette_run_completion_status")
                        == "failed"
                        and ATOMIC_PUBLICATION_TOMBSTONE_ATTR in failed.attrs
                    )
                    if owned_public_tombstoned and failed is not None:
                        expected_failed_publication_attrs = copy.deepcopy(
                            dict(failed.attrs)
                        )
                except BaseException as tombstone_check_exc:  # pragma: no cover
                    rollback_errors.append(
                        "verify failed public tombstone: "
                        f"{tombstone_check_exc}"
                    )

        if spec.selector_owner_attr is not None and spec.owned_parent_attr_names:
            # Only an exact persisted selector lease plus an explicit mutation
            # surface authorizes parent rollback.
            try:
                rollback_root = open_zarr_root(spec.source_zarr, mode="a")
                try:
                    receipt = _capture_owned_parent_rollback_receipt(
                        rollback_root,
                        parent_snapshots,
                        spec.owned_parent_attr_names,
                        lease_parent_path=parent_paths[0],
                        lease_attr=spec.selector_owner_attr,
                        generation_attr=spec.selector_generation_attr,
                        publication_owner=expected_publication_owner,
                    )
                except BaseException as receipt_exc:  # pragma: no cover - hostile store
                    receipt = None
                    rollback_errors.append(
                        f"capture selector rollback receipt: {receipt_exc}"
                    )
                if receipt is not None:
                    try:
                        _restore_owned_parent_attrs(
                            rollback_root,
                            receipt,
                        )
                    except BaseException as restore_exc:  # pragma: no cover - hostile store
                        rollback_errors.append(str(restore_exc))
            except BaseException as root_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"open rollback root: {root_exc}")
        # Without both declarations, parent state has no rollback authority.
        # Leave it untouched: the retained child is selector-ineligible, so a
        # stale pointer fails closed instead of risking successor clobber.

        # A callback may have refreshed archive-wide consolidated metadata
        # immediately before failing. Once direct metadata is tombstoned and
        # any owned parent mutations are restored, give the stage one exact
        # under-lock opportunity to make its consolidated view fail closed too.
        if (
            owned_public_tombstoned
            and expected_failed_publication_attrs is not None
            and repair_failed_publication_visibility is not None
        ):
            try:
                repair_failed_publication_visibility(spec.target_run_path)
                _verify_failed_publication_visibility(
                    spec,
                    owner_attr=publication_owner_attr,
                    publication_owner=expected_publication_owner,
                    expected_attrs=expected_failed_publication_attrs,
                )
            except BaseException as repair_exc:  # pragma: no cover - hostile store
                rollback_errors.append(
                    "repair failed publication visibility: " f"{repair_exc}"
                )

        if temporary.exists():
            try:
                shutil.rmtree(temporary)
            except BaseException as temporary_exc:  # pragma: no cover - hostile store
                rollback_errors.append(
                    f"delete publication temporary: {temporary_exc}"
                )

        if rollback_errors:
            raise RuntimeError(
                "Atomic publication failed and rollback was incomplete: "
                f"{rollback_errors!r}."
            ) from exc
        raise


__all__ = [
    "ARCHIVE_PUBLICATION_LOCK_SUFFIX",
    "ATOMIC_PUBLICATION_OWNER_ATTR",
    "ATOMIC_PUBLICATION_TOMBSTONE_ATTR",
    "ATOMIC_RUN_PUBLISHER_SCHEMA_ID",
    "ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION",
    "DEFAULT_COPY_CONTENT_CHECKSUM",
    "DEFAULT_COPY_INTEGRITY_POLICY",
    "AtomicRunPublishSpec",
    "SERIALIZATION_POLICY",
    "atomic_publish_run_group",
    "tree_inventory",
]
