"""Atomic publisher for one sealed derived chaser component.

Scientific component writers build and seal a complete node-local component.
This materializer owns the destination transaction: same-filesystem hidden
copy, validation, immutable rename, and completion receipt. Digest-bound
selector activation is a separate explicitly requested final callback; the
default publication remains selector-ineligible.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import zarr

from fisheye.analysis.chaser_component_publication import (
    COMPONENT_MANIFEST_ATTR,
    COMPONENT_MANIFEST_DIGEST_ATTR,
    COMPONENT_SELECTOR_ATTR,
    COMPONENT_SELECTOR_DIGEST_ATTR,
    RETRY_TRANSIENT_ATTRIBUTE_NAMES,
    ChaserComponentContract,
    ChaserComponentPublicationError,
    build_chaser_component_selector,
    chaser_component_retry_equivalence_sha256,
    component_record_sha256,
    persist_chaser_component_selector,
    validate_chaser_component_manifest,
)
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr_helpers import archive_metadata_publication_lock
from fisheye.shared.zarr_io import open_zarr_root

from .atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    ATOMIC_PUBLICATION_TOMBSTONE_ATTR,
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)


CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_ID = (
    "palette.chaser_component_atomic_publication"
)
CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_VERSION = 1
CHASER_COMPONENT_RECOVERY_SCHEMA_ID = (
    "palette.chaser_component_publication_recovery"
)
CHASER_COMPONENT_RECOVERY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ChaserComponentPublishRequest:
    """Exact inputs for one immutable component publication."""

    source_zarr: Path
    local_component_path: Path
    base_run_name: str
    base_run_path: str
    relative_path: str
    contract: ChaserComponentContract
    copy_backend: str = "python"
    content_checksum: bool = True
    activate_selector: bool = False


def _restore_attr(attrs: Any, name: str, *, present: bool, value: Any) -> None:
    if present:
        attrs[name] = copy.deepcopy(value)
    elif name in attrs:
        del attrs[name]


def _normalized_request(
    request: ChaserComponentPublishRequest,
) -> tuple[str, str, str]:
    expected = request.contract.normalized_component(
        relative_path=request.relative_path
    )
    family = expected["component_family"]
    component_name = expected["component_name"]
    base_path = "/".join(
        part for part in str(request.base_run_path).strip("/").split("/") if part
    )
    expected_base_path = (
        f"analysis/chaser_distance_runs/{str(request.base_run_name).strip()}"
    )
    if base_path != expected_base_path:
        raise ValueError(
            "Chaser component base run path must match the exact requested run name."
        )
    return base_path, family, component_name


def _recover_completed_ineligible_component(
    request: ChaserComponentPublishRequest,
    *,
    source_zarr: Path,
    base_path: str,
    family: str,
    component_name: str,
    local_manifest: dict[str, Any],
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Reconstruct acknowledgement for one exact already-committed candidate."""

    target_group_path = f"{base_path}/{family}/{component_name}"
    with archive_metadata_publication_lock(source_zarr):
        root = open_zarr_root(source_zarr, mode="r")
        snapshot = load_chaser_distance_run(
            root,
            run_name=request.base_run_name,
        )
        if snapshot.run_path != base_path:
            raise RuntimeError(
                "Verified chaser-distance base changed before receipt recovery."
            )
        try:
            component = root[target_group_path]
        except Exception as exc:
            raise FileExistsError(
                "Existing chaser component disappeared before receipt recovery."
            ) from exc
        if not isinstance(component, zarr.Group):
            raise FileExistsError(
                "Existing chaser component recovery target is not a group."
            )
        owner = component.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
        try:
            canonical_owner = str(uuid.UUID(str(owner)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise FileExistsError(
                "Existing chaser component lacks one valid publication owner."
            ) from exc
        if (
            owner != canonical_owner
            or component.attrs.get("palette_run_completion_status") != "complete"
            or component.attrs.get("stage_selector_eligible") is not False
            or ATOMIC_PUBLICATION_TOMBSTONE_ATTR in component.attrs
        ):
            raise FileExistsError(
                "Existing chaser component is not one completed, "
                "selector-ineligible, untombstoned publication."
            )
        try:
            manifest = validate_chaser_component_manifest(
                component,
                snapshot=snapshot,
                expected_relative_path=request.relative_path,
                expected_contract=request.contract,
            )
        except ChaserComponentPublicationError as exc:
            raise FileExistsError(
                "Existing chaser component is not one valid publication of "
                "the requested contract."
            ) from exc
        manifest_sha256 = component_record_sha256(manifest)
        local_retry_sha256 = chaser_component_retry_equivalence_sha256(
            local_manifest
        )
        existing_retry_sha256 = chaser_component_retry_equivalence_sha256(
            manifest
        )
        if existing_retry_sha256 != local_retry_sha256:
            raise FileExistsError(
                "Existing chaser component scientific payload or semantics differ "
                "from the local retry."
            )
        validation = {
            "valid": True,
            "component_manifest_sha256": manifest_sha256,
            "array_count": len(manifest["payload"]["arrays"]),
            "group_count": len(manifest["payload"]["groups"]),
        }
        receipt = {
            "schema_id": CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_ID,
            "schema_version": CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_VERSION,
            "base_run_path": base_path,
            "component_family": family,
            "component_name": component_name,
            "component_manifest_sha256": manifest_sha256,
            "selector_activation_requested": False,
            "publication_owner_uuid": canonical_owner,
            "recovered_existing": True,
            "recovery": {
                "schema_id": CHASER_COMPONENT_RECOVERY_SCHEMA_ID,
                "schema_version": CHASER_COMPONENT_RECOVERY_SCHEMA_VERSION,
                "policy": (
                    "exact_science_complete_ineligible_untombstoned_reopen_v1"
                ),
                "target_group_path": target_group_path,
                "local_manifest_sha256": expected_manifest_sha256,
                "existing_manifest_sha256": manifest_sha256,
                "retry_equivalence_sha256": existing_retry_sha256,
                "ignored_attribute_names": sorted(
                    RETRY_TRANSIENT_ATTRIBUTE_NAMES
                ),
            },
            "final_validation": validation,
        }
        return json_attr_safe(receipt)


def publish_sealed_chaser_component(
    request: ChaserComponentPublishRequest,
) -> dict[str, Any]:
    """Publish one sealed node-local component; activate only when requested."""

    base_path, family, component_name = _normalized_request(request)
    source_zarr = request.source_zarr.expanduser().resolve()
    local_component = request.local_component_path.expanduser().resolve()
    target_path = source_zarr / base_path / family / component_name

    authoritative_root = open_zarr_root(source_zarr, mode="r")
    source_snapshot = load_chaser_distance_run(
        authoritative_root,
        run_name=request.base_run_name,
    )
    if source_snapshot.run_path != base_path:
        raise RuntimeError(
            "Verified chaser-distance base run changed before publication."
        )

    local_group = open_zarr_root(local_component, mode="r")
    local_manifest = validate_chaser_component_manifest(
        local_group,
        snapshot=source_snapshot,
        expected_relative_path=request.relative_path,
        expected_contract=request.contract,
    )
    local_manifest_digest = component_record_sha256(local_manifest)
    expected_selector = build_chaser_component_selector(
        snapshot=source_snapshot,
        relative_path=request.relative_path,
        component_manifest_sha256=local_manifest_digest,
    )
    expected_selector_digest = component_record_sha256(expected_selector)

    selector_state: dict[str, tuple[bool, Any]] = {}
    selector_attempted = False

    def snapshot() -> Any:
        root = open_zarr_root(source_zarr, mode="r")
        selected = load_chaser_distance_run(root, run_name=request.base_run_name)
        if selected.run_path != base_path:
            raise RuntimeError(
                "Verified chaser-distance base run changed during publication."
            )
        return selected

    def validate_path(path: Path) -> dict[str, Any]:
        component = open_zarr_root(path, mode="r")
        manifest = validate_chaser_component_manifest(
            component,
            snapshot=snapshot(),
            expected_relative_path=request.relative_path,
            expected_contract=request.contract,
            expected_manifest_sha256=local_manifest_digest,
        )
        return {
            "valid": True,
            "component_manifest_sha256": component_record_sha256(manifest),
            "array_count": len(manifest["payload"]["arrays"]),
            "group_count": len(manifest["payload"]["groups"]),
        }

    def prepare_parents(root: zarr.Group) -> tuple[zarr.Group]:
        run = root[base_path]
        if not isinstance(run, zarr.Group):
            raise TypeError("Verified chaser-distance base path is not a group.")
        parent = run.require_group(family)
        return (parent,)

    def complete_run(
        _root: zarr.Group,
        _parent: zarr.Group,
        component: zarr.Group,
    ) -> None:
        component.attrs["palette_run_completion_status"] = "complete"
        component.attrs["palette_run_completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()

    def verify_pre_activation(root: zarr.Group) -> None:
        component = root[f"{base_path}/{family}/{component_name}"]
        if not isinstance(component, zarr.Group):
            raise TypeError("Published chaser component is not a group.")
        if component.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError("Chaser component became eligible before activation.")
        validate_chaser_component_manifest(
            component,
            snapshot=load_chaser_distance_run(
                root,
                run_name=request.base_run_name,
            ),
            expected_relative_path=request.relative_path,
            expected_contract=request.contract,
            expected_manifest_sha256=local_manifest_digest,
        )

    def activate_run(
        root: zarr.Group,
        parent: zarr.Group,
        component: zarr.Group,
    ) -> None:
        nonlocal selector_attempted
        fresh_snapshot = load_chaser_distance_run(
            root,
            run_name=request.base_run_name,
        )
        if fresh_snapshot.run_path != base_path:
            raise RuntimeError("Base authority changed before component activation.")
        for name in (COMPONENT_SELECTOR_ATTR, COMPONENT_SELECTOR_DIGEST_ATTR):
            selector_state[name] = (
                name in parent.attrs,
                copy.deepcopy(parent.attrs.get(name)),
            )
        selector_attempted = True
        selector, selector_digest = persist_chaser_component_selector(
            parent,
            component=component,
            snapshot=fresh_snapshot,
            relative_path=request.relative_path,
        )
        if selector != expected_selector or selector_digest != expected_selector_digest:
            raise RuntimeError(
                "Component selector differs from the validated candidate."
            )
        # Literal commit point. No fallible validation or metadata write follows.
        component.attrs["stage_selector_eligible"] = True

    def rollback_activation() -> None:
        if not selector_attempted:
            return
        root = open_zarr_root(source_zarr, mode="a")
        parent = root[f"{base_path}/{family}"]
        current_selector = parent.attrs.get(COMPONENT_SELECTOR_ATTR)
        current_digest = parent.attrs.get(COMPONENT_SELECTOR_DIGEST_ATTR)
        if current_selector == expected_selector:
            present, value = selector_state[COMPONENT_SELECTOR_ATTR]
            _restore_attr(
                parent.attrs,
                COMPONENT_SELECTOR_ATTR,
                present=present,
                value=value,
            )
        if current_digest == expected_selector_digest:
            present, value = selector_state[COMPONENT_SELECTOR_DIGEST_ATTR]
            _restore_attr(
                parent.attrs,
                COMPONENT_SELECTOR_DIGEST_ATTR,
                present=present,
                value=value,
            )

    spec = AtomicRunPublishSpec(
        source_zarr=source_zarr,
        local_run_path=local_component,
        target_run_path=target_path,
        run_name=component_name,
        lock_suffix=f"chaser-component-{family}",
        publish_schema_id=CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_ID,
        policy="immutable_component_then_digest_bound_selector",
        rollback_policy=(
            "retain_owner_bound_ineligible_tombstone_and_conditionally_restore_selector"
        ),
        content_checksum=bool(request.content_checksum),
    )
    try:
        result = atomic_publish_run_group(
            spec,
            copy_backend=request.copy_backend,
            validate_run=validate_path,
            prepare_parents=prepare_parents,
            complete_run=complete_run,
            verify_pointers=verify_pre_activation,
            activate_run=activate_run if request.activate_selector else None,
            rollback_activation=(
                rollback_activation if request.activate_selector else None
            ),
            payload_metadata={
                "schema_version": CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_VERSION,
                "base_run_path": base_path,
                "component_family": family,
                "component_name": component_name,
                "component_manifest_attr": COMPONENT_MANIFEST_ATTR,
                "component_manifest_digest_attr": COMPONENT_MANIFEST_DIGEST_ATTR,
                "component_manifest_sha256": local_manifest_digest,
                "component_selector_sha256": expected_selector_digest,
                "selector_activation_requested": bool(request.activate_selector),
            },
        )
    except FileExistsError:
        if request.activate_selector:
            raise
        result = _recover_completed_ineligible_component(
            request,
            source_zarr=source_zarr,
            base_path=base_path,
            family=family,
            component_name=component_name,
            local_manifest=dict(local_manifest),
            expected_manifest_sha256=local_manifest_digest,
        )

    return result


__all__ = [
    "CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_ID",
    "CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_VERSION",
    "CHASER_COMPONENT_RECOVERY_SCHEMA_ID",
    "CHASER_COMPONENT_RECOVERY_SCHEMA_VERSION",
    "ChaserComponentPublishRequest",
    "publish_sealed_chaser_component",
]
