"""Atomic publisher for one sealed derived chaser component.

Scientific component writers build and seal a complete node-local component.
This materializer owns the destination transaction: same-filesystem hidden
copy, validation, immutable rename, and completion receipt. Digest-bound
selector activation is a separate explicitly requested final callback; the
default publication remains selector-ineligible.
"""

from __future__ import annotations

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
    validate_chaser_component_selector,
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
CHASER_COMPONENT_ACTIVATION_LEASE_ATTR = (
    "chaser_component_publication_activation_lease"
)
CHASER_COMPONENT_ACTIVATION_GENERATION_ATTR = (
    "chaser_component_publication_activation_generation"
)
CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_ID = (
    "palette.chaser_component_publication_activation_lease"
)
CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_VERSION = 1

_ACTIVATION_LEASE_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "owner_uuid",
        "next_generation",
        "component_path",
        "component_manifest_sha256",
        "selector_sha256",
        "record_sha256",
    }
)


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


def _canonical_uuid(value: Any, *, label: str) -> str:
    try:
        canonical = str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be one canonical UUID.") from exc
    if value != canonical:
        raise ValueError(f"{label} must be one canonical UUID.")
    return canonical


def _activation_generation(parent: zarr.Group) -> int:
    value = parent.attrs.get(CHASER_COMPONENT_ACTIVATION_GENERATION_ATTR, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            "Chaser component activation generation must be one nonnegative integer."
        )
    return int(value)


def _lower_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _build_activation_lease(
    *,
    owner_uuid: str,
    next_generation: int,
    component_path: str,
    component_manifest_sha256: str,
    selector_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_id": CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_ID,
        "schema_version": CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_VERSION,
        "owner_uuid": _canonical_uuid(
            owner_uuid,
            label="activation lease owner",
        ),
        "next_generation": int(next_generation),
        "component_path": component_path,
        "component_manifest_sha256": _lower_sha256(
            component_manifest_sha256,
            label="activation component manifest digest",
        ),
        "selector_sha256": _lower_sha256(
            selector_sha256,
            label="activation selector digest",
        ),
    }
    if body["next_generation"] <= 0:
        raise ValueError("Activation lease generation must be positive.")
    return {
        **body,
        "record_sha256": component_record_sha256(body),
    }


def _validate_activation_lease(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ACTIVATION_LEASE_FIELDS:
        raise RuntimeError("Chaser component activation lease has unexpected fields.")
    body = {key: value[key] for key in _ACTIVATION_LEASE_FIELDS - {"record_sha256"}}
    if (
        body["schema_id"] != CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_ID
        or body["schema_version"]
        != CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_VERSION
        or isinstance(body["next_generation"], bool)
        or not isinstance(body["next_generation"], int)
        or body["next_generation"] <= 0
        or not isinstance(body["component_path"], str)
        or not isinstance(body["component_manifest_sha256"], str)
        or not isinstance(body["selector_sha256"], str)
    ):
        raise RuntimeError("Chaser component activation lease is malformed.")
    _canonical_uuid(body["owner_uuid"], label="activation lease owner")
    _lower_sha256(
        body["component_manifest_sha256"],
        label="activation component manifest digest",
    )
    _lower_sha256(body["selector_sha256"], label="activation selector digest")
    _lower_sha256(value["record_sha256"], label="activation lease record digest")
    if component_record_sha256(body) != value["record_sha256"]:
        raise RuntimeError("Chaser component activation lease digest changed.")
    return dict(value)


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


def _recover_or_complete_selector_activation(
    request: ChaserComponentPublishRequest,
    *,
    source_zarr: Path,
    base_path: str,
    family: str,
    component_name: str,
    local_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Recover only the exact activation epoch durably leased by this child."""

    target_group_path = f"{base_path}/{family}/{component_name}"
    with archive_metadata_publication_lock(source_zarr):
        root = open_zarr_root(source_zarr, mode="a")
        snapshot = load_chaser_distance_run(root, run_name=request.base_run_name)
        if snapshot.run_path != base_path:
            raise RuntimeError(
                "Verified chaser-distance base changed before activation recovery."
            )
        try:
            parent = root[f"{base_path}/{family}"]
            component = parent[component_name]
        except Exception as exc:
            raise FileExistsError(
                "Existing chaser component disappeared before activation recovery."
            ) from exc
        if not isinstance(parent, zarr.Group) or not isinstance(component, zarr.Group):
            raise FileExistsError(
                "Existing chaser activation recovery paths are not groups."
            )
        owner_uuid = _canonical_uuid(
            component.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR),
            label="existing component owner",
        )
        if (
            component.attrs.get("palette_run_completion_status") != "complete"
            or ATOMIC_PUBLICATION_TOMBSTONE_ATTR in component.attrs
            or type(component.attrs.get("stage_selector_eligible")) is not bool
        ):
            raise FileExistsError(
                "Existing chaser component is not one complete, untombstoned "
                "activation candidate."
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
                "Existing chaser activation candidate fails its requested contract."
            ) from exc
        if chaser_component_retry_equivalence_sha256(
            manifest
        ) != chaser_component_retry_equivalence_sha256(local_manifest):
            raise FileExistsError(
                "Existing chaser activation candidate differs scientifically "
                "from the local retry."
            )

        manifest_sha256 = component_record_sha256(manifest)
        expected_selector = build_chaser_component_selector(
            snapshot=snapshot,
            relative_path=request.relative_path,
            component_manifest_sha256=manifest_sha256,
        )
        expected_selector_sha256 = component_record_sha256(expected_selector)
        lease = _validate_activation_lease(
            parent.attrs.get(CHASER_COMPONENT_ACTIVATION_LEASE_ATTR)
        )
        generation = _activation_generation(parent)
        expected_lease_fields = {
            "owner_uuid": owner_uuid,
            "next_generation": generation,
            "component_path": request.relative_path,
            "component_manifest_sha256": manifest_sha256,
            "selector_sha256": expected_selector_sha256,
        }
        if any(lease.get(key) != value for key, value in expected_lease_fields.items()):
            raise FileExistsError(
                "Existing chaser activation lease belongs to another component "
                "or generation."
            )

        eligibility = component.attrs["stage_selector_eligible"]
        completed_by_recovery = False
        if eligibility is False:
            selector, selector_sha256 = persist_chaser_component_selector(
                parent,
                component=component,
                snapshot=snapshot,
                relative_path=request.relative_path,
            )
            if (
                selector != expected_selector
                or selector_sha256 != expected_selector_sha256
            ):
                raise RuntimeError(
                    "Recovered selector differs from its durable activation lease."
                )
            # Literal activation commit. Any later acknowledgement failure is
            # recoverable from the same generation-bound lease.
            component.attrs["stage_selector_eligible"] = True
            completed_by_recovery = True
        else:
            validate_chaser_component_selector(
                parent,
                component=component,
                snapshot=snapshot,
                expected_family=family,
            )
            if (
                parent.attrs.get(COMPONENT_SELECTOR_ATTR) != expected_selector
                or parent.attrs.get(COMPONENT_SELECTOR_DIGEST_ATTR)
                != expected_selector_sha256
            ):
                raise FileExistsError(
                    "Eligible chaser component is not selected by its exact lease."
                )

        validation = {
            "valid": True,
            "component_manifest_sha256": manifest_sha256,
            "array_count": len(manifest["payload"]["arrays"]),
            "group_count": len(manifest["payload"]["groups"]),
        }
        return json_attr_safe(
            {
                "schema_id": CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_ID,
                "schema_version": CHASER_COMPONENT_ATOMIC_PUBLISH_SCHEMA_VERSION,
                "base_run_path": base_path,
                "component_family": family,
                "component_name": component_name,
                "component_manifest_sha256": manifest_sha256,
                "selector_activation_requested": True,
                "selector_activation_recovered": True,
                "publication_owner_uuid": owner_uuid,
                "recovered_existing": True,
                "recovery": {
                    "schema_id": CHASER_COMPONENT_RECOVERY_SCHEMA_ID,
                    "schema_version": CHASER_COMPONENT_RECOVERY_SCHEMA_VERSION,
                    "policy": "generation_leased_selector_activation_v1",
                    "target_group_path": target_group_path,
                    "activation_generation": generation,
                    "activation_completed_by_recovery": completed_by_recovery,
                    "activation_lease_sha256": lease["record_sha256"],
                },
                "final_validation": validation,
            }
        )


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
        fresh_snapshot = load_chaser_distance_run(
            root,
            run_name=request.base_run_name,
        )
        if fresh_snapshot.run_path != base_path:
            raise RuntimeError("Base authority changed before component activation.")
        owner_uuid = _canonical_uuid(
            component.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR),
            label="published component owner",
        )
        next_generation = _activation_generation(parent) + 1
        lease = _build_activation_lease(
            owner_uuid=owner_uuid,
            next_generation=next_generation,
            component_path=request.relative_path,
            component_manifest_sha256=local_manifest_digest,
            selector_sha256=expected_selector_digest,
        )
        parent.attrs[CHASER_COMPONENT_ACTIVATION_LEASE_ATTR] = json_attr_safe(
            lease
        )
        parent.attrs[CHASER_COMPONENT_ACTIVATION_GENERATION_ATTR] = (
            next_generation
        )
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
        selector_owner_attr=(
            CHASER_COMPONENT_ACTIVATION_LEASE_ATTR
            if request.activate_selector
            else None
        ),
        selector_generation_attr=(
            CHASER_COMPONENT_ACTIVATION_GENERATION_ATTR
            if request.activate_selector
            else None
        ),
        owned_parent_attr_names=(
            (
                COMPONENT_SELECTOR_ATTR,
                COMPONENT_SELECTOR_DIGEST_ATTR,
                CHASER_COMPONENT_ACTIVATION_LEASE_ATTR,
                CHASER_COMPONENT_ACTIVATION_GENERATION_ATTR,
            ),
        )
        if request.activate_selector
        else (),
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
            result = _recover_or_complete_selector_activation(
                request,
                source_zarr=source_zarr,
                base_path=base_path,
                family=family,
                component_name=component_name,
                local_manifest=dict(local_manifest),
            )
        else:
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
    "CHASER_COMPONENT_ACTIVATION_GENERATION_ATTR",
    "CHASER_COMPONENT_ACTIVATION_LEASE_ATTR",
    "CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_ID",
    "CHASER_COMPONENT_ACTIVATION_LEASE_SCHEMA_VERSION",
    "CHASER_COMPONENT_RECOVERY_SCHEMA_ID",
    "CHASER_COMPONENT_RECOVERY_SCHEMA_VERSION",
    "ChaserComponentPublishRequest",
    "publish_sealed_chaser_component",
]
