"""Plan exact geometry-only crop successors across refined-detection snapshots.

Crop runs remain complete immutable snapshots.  This module does not append to
an existing run and does not activate selectors.  It classifies the rows of a
prepared target snapshot against one fully validated parent publication so a
publisher or DAG can record which observations were reused, added, changed, or
retired.

The persisted crop ``source_row_signature`` deliberately binds one exact
refined snapshot.  That is the correct scientific identity, but it is too
strict for cross-snapshot reuse planning because every successor has a new run
and snapshot identity.  The reconciliation signature below instead binds the
stable refined lineage and compares every row-local geometry field.  It is an
optimization receipt, never a replacement for the persisted authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.keyed_delta import (
    ACTION_CODE_MAP,
    REASON_CODE_MAP,
    KeyedDeltaPlan,
    build_keyed_delta_plan,
)
from fisheye.shared.row_source_signature import build_row_source_signatures
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_manifest import (
    CropPixelAuthority,
    CropRefinedSourceIdentity,
    crop_pixel_authority_from_manifest,
    crop_refined_source_identity_from_manifest,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropGeometryPolicy,
    crop_geometry_policy_from_manifest,
)
from fisheye.shared.zarr.crop_shadow import (
    DEFAULT_CROP_GEOMETRY_SHADOW_ROOT,
    CropGeometryShadowPublication,
    PreparedCropGeometrySnapshot,
    publish_selector_ineligible_crop_geometry_snapshot,
    validate_crop_geometry_shadow_publication,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile


CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_ID = (
    "palette.crop_geometry.successor_reconciliation"
)
CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_VERSION = 1
CROP_GEOMETRY_SUCCESSOR_SIGNATURE_STAGE = "crop_geometry_successor"
CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_ID = (
    "palette.crop_geometry.successor_publication"
)
CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_VERSION = 1
CROP_GEOMETRY_SUCCESSOR_PUBLICATION_RECEIPT_NAME = (
    "crop_successor_publication_receipt.json"
)
_RECEIPT_INLINE_KEY_LIMIT = 64
_RECONCILIATION_CONTENT_PATHS = tuple(
    path
    for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths
    if path not in {"frame_row_offsets", "source_row_signature"}
)


class CropGeometrySuccessorError(ValueError):
    """Raised when a crop successor cannot be proven against its parent."""


@dataclass(frozen=True)
class CropGeometrySuccessorPlan:
    """One exact keyed reconciliation from a parent crop to its successor."""

    parent_crop_run_id: str
    parent_crop_manifest_digest: str
    parent_source: CropRefinedSourceIdentity
    target_source: CropRefinedSourceIdentity
    keyed_plan: KeyedDeltaPlan
    receipt: Mapping[str, Any]

    @property
    def reused_instance_keys(self) -> np.ndarray:
        mask = self.keyed_plan.action_codes == ACTION_CODE_MAP["copy"]
        return self.keyed_plan.target_instance_keys[mask]

    @property
    def added_instance_keys(self) -> np.ndarray:
        mask = self.keyed_plan.reason_codes == REASON_CODE_MAP["added"]
        return self.keyed_plan.target_instance_keys[mask]

    @property
    def changed_instance_keys(self) -> np.ndarray:
        mask = np.isin(
            self.keyed_plan.reason_codes,
            (
                REASON_CODE_MAP["source_changed"],
                REASON_CODE_MAP["signature_spec_changed"],
            ),
        )
        return self.keyed_plan.target_instance_keys[mask]

    @property
    def retired_instance_keys(self) -> np.ndarray:
        return self.keyed_plan.omitted_instance_keys


@dataclass(frozen=True)
class CropGeometrySuccessorPublication:
    """One selector-ineligible successor plus its reconciliation evidence."""

    publication: CropGeometryShadowPublication
    plan: CropGeometrySuccessorPlan
    receipt: Mapping[str, Any]


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _key_set_receipt(values: np.ndarray) -> dict[str, object]:
    keys = np.sort(np.asarray(values, dtype=np.uint64).reshape(-1))
    receipt: dict[str, object] = {
        "count": int(keys.shape[0]),
        "sha256": sha256_array(keys),
    }
    if keys.shape[0] <= _RECEIPT_INLINE_KEY_LIMIT:
        receipt["values"] = [int(value) for value in keys]
    return receipt


def _parent_contract(
    publication: CropGeometryShadowPublication,
) -> tuple[CropRefinedSourceIdentity, CropPixelAuthority, CropGeometryPolicy]:
    errors = validate_crop_geometry_shadow_publication(publication)
    if errors:
        raise CropGeometrySuccessorError(
            "Parent crop publication is invalid: " + "; ".join(errors)
        )
    payload = publication.manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise CropGeometrySuccessorError("Parent crop manifest lacks its payload.")
    try:
        source = crop_refined_source_identity_from_manifest(
            payload["source_refined_snapshot"]
        )
        pixels = crop_pixel_authority_from_manifest(
            payload["source_pixel_authority"]
        )
        logical = payload["logical_schema"]
        if not isinstance(logical, Mapping):
            raise TypeError("logical_schema must be an object")
        policy = crop_geometry_policy_from_manifest(logical["crop_policy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CropGeometrySuccessorError(
            f"Parent crop contract cannot be reconstructed: {exc}"
        ) from exc
    return source, pixels, policy


def _require_immediate_refined_successor(
    *,
    parent: CropRefinedSourceIdentity,
    target: PreparedCropGeometrySnapshot,
) -> None:
    target_source = target.source
    if target_source.recording_identity != parent.recording_identity:
        raise CropGeometrySuccessorError(
            "Target crop source belongs to a different recording."
        )
    if target_source.lineage_id != parent.lineage_id:
        raise CropGeometrySuccessorError(
            "Target crop source belongs to a different refined lineage."
        )
    if target_source.snapshot_id == parent.snapshot_id:
        raise CropGeometrySuccessorError(
            "Target crop source must use a new refined snapshot identity."
        )
    payload = target.source_manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise CropGeometrySuccessorError(
            "Target refined source manifest lacks its payload."
        )
    lineage = payload.get("snapshot_lineage")
    if not isinstance(lineage, Mapping):
        raise CropGeometrySuccessorError(
            "Target refined source manifest lacks snapshot_lineage."
        )
    expected_parent = {
        "run_id": parent.run_id,
        "run_manifest_digest": parent.run_manifest_digest,
    }
    if lineage.get("parent_snapshot") != expected_parent:
        raise CropGeometrySuccessorError(
            "Target refined source does not bind the parent crop's immediate "
            "refined snapshot."
        )
    if (
        payload.get("run_id") != target_source.run_id
        or target.source_manifest.get("payload_digest")
        != target_source.run_manifest_digest
        or lineage.get("lineage_id") != target_source.lineage_id
        or lineage.get("snapshot_id") != target_source.snapshot_id
    ):
        raise CropGeometrySuccessorError(
            "Target prepared source identity differs from its refined manifest."
        )


def _require_target_source_rows(target: PreparedCropGeometrySnapshot) -> None:
    comparisons = {
        "instance_key": "instances/instance_key",
        "source_refined_row_ids": "instances/refined_row_ids",
        "frame_indices": "instances/frame_indices",
        "source_acquisition_frame_index": (
            "instances/source_acquisition_frame_index"
        ),
        "frame_row_offsets": "instances/frame_row_offsets",
        "bbox_norm_coords": "instances/bbox_norm_coords",
        "bbox_img_xyxy": "instances/bbox_img_xyxy",
        "centers_img_xy": "instances/centers_img_xy",
    }
    missing = [
        source_path
        for source_path in comparisons.values()
        if source_path not in target.source_arrays
    ]
    if missing:
        raise CropGeometrySuccessorError(
            f"Target refined source evidence is missing {missing!r}."
        )
    for crop_path, source_path in comparisons.items():
        if not np.array_equal(
            _array_values(target.arrays[crop_path]),
            _array_values(target.source_arrays[source_path]),
        ):
            raise CropGeometrySuccessorError(
                f"Target crop {crop_path!r} differs from {source_path!r}."
            )


def _reconciliation_signatures(
    arrays: Mapping[str, Any],
    *,
    source: CropRefinedSourceIdentity,
    policy: CropGeometryPolicy,
    pixel_authority: CropPixelAuthority,
) -> Any:
    missing = {"instance_key", *_RECONCILIATION_CONTENT_PATHS} - set(arrays)
    if missing:
        raise CropGeometrySuccessorError(
            f"Crop successor arrays are missing {sorted(missing)!r}."
        )
    return build_row_source_signatures(
        stage=CROP_GEOMETRY_SUCCESSOR_SIGNATURE_STAGE,
        instance_keys=_array_values(arrays["instance_key"]),
        content_components={
            path: _array_values(arrays[path])
            for path in _RECONCILIATION_CONTENT_PATHS
        },
        compatibility_context={
            "schema_id": CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_ID,
            "schema_version": CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_VERSION,
            "crop_schema": {
                "id": CROP_GEOMETRY_SCHEMA_V1.schema_id,
                "version": CROP_GEOMETRY_SCHEMA_V1.schema_version,
            },
            "crop_policy_digest": policy.payload_digest,
            "recording_identity": source.recording_identity,
            "source_refined_lineage_id": source.lineage_id,
            "source_pixel_authority_id": pixel_authority.authority_id,
            "source_pixel_authority_manifest_digest": (
                pixel_authority.authority_manifest_digest
            ),
            "snapshot_identity_role": (
                "excluded_from_reuse_signature_but_bound_by_successor_receipt"
            ),
        },
    )


def plan_crop_geometry_successor(
    parent: CropGeometryShadowPublication,
    target: PreparedCropGeometrySnapshot,
) -> CropGeometrySuccessorPlan:
    """Classify one complete target against its immediate parent crop.

    The returned plan is read-only evidence.  It neither writes the target nor
    changes selectors, registries, or parent data.
    """

    if not isinstance(parent, CropGeometryShadowPublication):
        raise TypeError("parent must be a CropGeometryShadowPublication.")
    if not isinstance(target, PreparedCropGeometrySnapshot):
        raise TypeError("target must be a PreparedCropGeometrySnapshot.")
    parent_source, parent_pixels, parent_policy = _parent_contract(parent)
    _require_immediate_refined_successor(parent=parent_source, target=target)
    if target.pixel_authority != parent_pixels:
        raise CropGeometrySuccessorError(
            "Crop successor v1 requires the exact same source-pixel authority."
        )
    if target.policy != parent_policy:
        raise CropGeometrySuccessorError(
            "Crop successor v1 requires the exact same crop policy."
        )
    if (
        target.dimensions.n_frames != parent.dimensions.n_frames
        or target.dimensions.source_width != parent.dimensions.source_width
        or target.dimensions.source_height != parent.dimensions.source_height
    ):
        raise CropGeometrySuccessorError(
            "Crop successor frame domain or source dimensions changed."
        )
    CROP_GEOMETRY_SCHEMA_V1.require(
        target.arrays,
        dimensions=target.dimensions,
        policy=target.policy,
    )
    _require_target_source_rows(target)

    parent_signatures = _reconciliation_signatures(
        parent.arrays,
        source=parent_source,
        policy=parent_policy,
        pixel_authority=parent_pixels,
    )
    target_signatures = _reconciliation_signatures(
        target.arrays,
        source=target.source,
        policy=target.policy,
        pixel_authority=target.pixel_authority,
    )
    keyed_plan = build_keyed_delta_plan(
        target_instance_keys=_array_values(target.arrays["instance_key"]),
        target_source_signatures=target_signatures.signatures,
        target_signature_spec_digest=target_signatures.spec.spec_digest,
        source_instance_keys=_array_values(parent.arrays["instance_key"]),
        source_row_signatures=parent_signatures.signatures,
        source_signature_spec_digest=parent_signatures.spec.spec_digest,
    )
    reused = keyed_plan.target_instance_keys[
        keyed_plan.action_codes == ACTION_CODE_MAP["copy"]
    ]
    added = keyed_plan.target_instance_keys[
        keyed_plan.reason_codes == REASON_CODE_MAP["added"]
    ]
    changed = keyed_plan.target_instance_keys[
        np.isin(
            keyed_plan.reason_codes,
            (
                REASON_CODE_MAP["source_changed"],
                REASON_CODE_MAP["signature_spec_changed"],
            ),
        )
    ]
    retired = keyed_plan.omitted_instance_keys
    receipt = {
        "schema_id": CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_ID,
        "schema_version": CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_VERSION,
        "parent_crop_run_id": parent.run_id,
        "parent_crop_manifest_digest": parent.manifest["payload_digest"],
        "parent_refined_run_id": parent_source.run_id,
        "parent_refined_manifest_digest": parent_source.run_manifest_digest,
        "parent_refined_logical_content_digest": (
            parent_source.logical_content_digest
        ),
        "parent_refined_snapshot_id": parent_source.snapshot_id,
        "target_refined_run_id": target.source.run_id,
        "target_refined_manifest_digest": target.source.run_manifest_digest,
        "target_refined_logical_content_digest": (
            target.source.logical_content_digest
        ),
        "target_refined_snapshot_id": target.source.snapshot_id,
        "refined_lineage_id": target.source.lineage_id,
        "reconciliation_signature_spec_digest": (
            target_signatures.spec.spec_digest
        ),
        "keyed_plan": keyed_plan.summary(),
        "instance_keys": {
            "reused": _key_set_receipt(reused),
            "added": _key_set_receipt(added),
            "changed": _key_set_receipt(changed),
            "retired": _key_set_receipt(retired),
        },
        "publication_authorized": False,
        "selector_activation": "none_plan_only",
        "production_state_changes": [],
    }
    return CropGeometrySuccessorPlan(
        parent_crop_run_id=parent.run_id,
        parent_crop_manifest_digest=parent.manifest["payload_digest"],
        parent_source=parent_source,
        target_source=target.source,
        keyed_plan=keyed_plan,
        receipt=receipt,
    )


def validate_crop_geometry_successor_publication_receipt(
    receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the strict safety envelope of one persisted successor receipt."""

    errors: list[str] = []
    expected_outer = {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }
    if set(receipt) != expected_outer:
        errors.append("crop successor receipt has an unexpected field set")
    if (
        receipt.get("schema_id")
        != CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_ID
        or receipt.get("schema_version")
        != CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_VERSION
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("crop successor receipt schema header mismatch")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "crop successor receipt payload must be an object")
    expected_payload = {
        "status",
        "selector_eligible",
        "registry_registered",
        "parent_crop",
        "output_crop",
        "reconciliation",
        "storage_profile_id",
        "selector_activation",
        "production_state_changes",
    }
    if set(payload) != expected_payload:
        errors.append("crop successor receipt payload has an unexpected field set")
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"crop successor receipt payload is not canonical JSON: {exc}")
    else:
        if receipt.get("payload_digest") != expected_digest:
            errors.append("crop successor receipt payload digest mismatch")
    if payload.get("status") != "complete":
        errors.append("crop successor receipt status is not complete")
    if payload.get("selector_eligible") is not False:
        errors.append("crop successor receipt must remain selector-ineligible")
    if payload.get("registry_registered") is not False:
        errors.append("crop successor receipt must remain unregistered")
    if payload.get("selector_activation") != "none_direct_path_only":
        errors.append("crop successor receipt selector activation is invalid")
    if payload.get("production_state_changes") != []:
        errors.append("crop successor receipt reports production-state changes")
    parent_crop = payload.get("parent_crop")
    output_crop = payload.get("output_crop")
    reconciliation = payload.get("reconciliation")
    if not isinstance(parent_crop, Mapping) or set(parent_crop) != {
        "run_id",
        "run_manifest_digest",
    }:
        errors.append("crop successor parent_crop binding is invalid")
    if not isinstance(output_crop, Mapping) or set(output_crop) != {
        "path",
        "run_id",
        "run_manifest_digest",
        "logical_content_digest",
    }:
        errors.append("crop successor output_crop binding is invalid")
    if not isinstance(reconciliation, Mapping):
        errors.append("crop successor reconciliation must be an object")
    else:
        if (
            reconciliation.get("schema_id")
            != CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_ID
            or reconciliation.get("schema_version")
            != CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_VERSION
            or reconciliation.get("publication_authorized") is not False
            or reconciliation.get("selector_activation") != "none_plan_only"
            or reconciliation.get("production_state_changes") != []
        ):
            errors.append("crop successor reconciliation safety envelope is invalid")
        if isinstance(parent_crop, Mapping) and (
            parent_crop.get("run_id")
            != reconciliation.get("parent_crop_run_id")
            or parent_crop.get("run_manifest_digest")
            != reconciliation.get("parent_crop_manifest_digest")
        ):
            errors.append("crop successor parent and reconciliation bindings differ")
    return tuple(errors)


def publish_selector_ineligible_crop_geometry_successor(
    parent: CropGeometryShadowPublication,
    target: PreparedCropGeometrySnapshot,
    *,
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_CROP_GEOMETRY_SHADOW_ROOT,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "crop_geometry_successor",
    coordinate_catalog: bool = True,
) -> CropGeometrySuccessorPublication:
    """Publish a complete successor after exact read-only reconciliation.

    This narrow publisher creates a new standalone immutable Zarr.  It never
    imports into a recording archive and never changes selectors or registries.
    The production DAG may later use the same validated output as the local
    source of an atomic selector-ineligible import.
    """

    plan = plan_crop_geometry_successor(parent, target)
    publication = publish_selector_ineligible_crop_geometry_snapshot(
        target,
        destination=destination,
        run_id=run_id,
        shadow_root=shadow_root,
        profile=profile,
        created_by=created_by,
        coordinate_catalog=coordinate_catalog,
    )
    errors = validate_crop_geometry_shadow_publication(publication)
    if errors:
        raise CropGeometrySuccessorError(
            "Published crop successor is invalid: " + "; ".join(errors)
        )
    payload = {
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "parent_crop": {
            "run_id": plan.parent_crop_run_id,
            "run_manifest_digest": plan.parent_crop_manifest_digest,
        },
        "output_crop": {
            "path": str(publication.output_path),
            "run_id": publication.run_id,
            "run_manifest_digest": publication.manifest["payload_digest"],
            "logical_content_digest": publication.receipt[
                "logical_content_digest"
            ],
        },
        "reconciliation": dict(plan.receipt),
        "storage_profile_id": profile.profile_id,
        "selector_activation": "none_direct_path_only",
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_ID,
        "schema_version": CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    receipt_errors = validate_crop_geometry_successor_publication_receipt(receipt)
    if receipt_errors:
        raise CropGeometrySuccessorError(
            "Crop successor receipt is invalid: " + "; ".join(receipt_errors)
        )
    receipt_path = (
        publication.output_path / CROP_GEOMETRY_SUCCESSOR_PUBLICATION_RECEIPT_NAME
    )
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(
            receipt,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        handle.write("\n")
    return CropGeometrySuccessorPublication(
        publication=publication,
        plan=plan,
        receipt=receipt,
    )


__all__ = [
    "CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_ID",
    "CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_VERSION",
    "CROP_GEOMETRY_SUCCESSOR_PUBLICATION_RECEIPT_NAME",
    "CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_ID",
    "CROP_GEOMETRY_SUCCESSOR_PUBLICATION_SCHEMA_VERSION",
    "CropGeometrySuccessorError",
    "CropGeometrySuccessorPlan",
    "CropGeometrySuccessorPublication",
    "plan_crop_geometry_successor",
    "publish_selector_ineligible_crop_geometry_successor",
    "validate_crop_geometry_successor_publication_receipt",
]
