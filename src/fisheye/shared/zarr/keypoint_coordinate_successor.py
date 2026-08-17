"""Republish immutable raw keypoints with the canonical-v2 coordinate graph.

This migration never reruns inference and never advances a selector.  Payload
objects are hard-linked from one sealed raw-keypoint bundle member, while all
metadata objects are copied before the successor is modified.  The original
bundle authority remains the scientific authority for the observation values;
the new child supplies only a freshly validated coordinate publication.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import zarr

from fisheye.shared.artifact_fingerprint import CONTENT_FINGERPRINT_SCHEME
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.keypoint_coordinate_publication import (
    KEYPOINT_COORDINATE_CONTEXT_ATTR,
    KEYPOINT_COORDINATE_DERIVATION_ATTR,
    KEYPOINT_PUBLICATION_OWNER_ATTR,
    load_persisted_ineligible_keypoint_coordinate_surfaces,
    prepare_keypoint_coordinate_context,
    publish_keypoint_coordinate_surfaces,
    require_bound_ineligible_keypoint_coordinate_surfaces,
)
from fisheye.shared.model_input_transform import model_input_transform_from_attrs
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.coordinate_successor_authority import (
    KEYPOINT_COORDINATE_SUCCESSOR_KIND,
    build_coordinate_successor_authority,
    load_coordinate_successor_authority,
    stamp_coordinate_successor_authority,
)
from fisheye.shared.zarr.coordinate_successor_files import (
    copy_metadata_and_link_payload,
    metadata_tree_sha256,
)
from fisheye.shared.zarr.keypoint_bundle_activation import (
    KEYPOINT_BUNDLE_AUTHORITY_ATTR,
    resolve_active_keypoint_bundle_from_root,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    build_keypoint_coordinate_successor_manifest,
    keypoint_preprocessing_from_manifest,
    validate_keypoint_publication,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_publication import (
    keypoint_metadata_declaration_maps,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
)
from fisheye.shared.zarr.keypoint_schema import KEYPOINT_SCHEMA_V2, KeypointDimensions
from fisheye.shared.zarr.keypoint_storage import plan_keypoint_storage
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)


KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID = (
    "palette.keypoint_coordinate_successor.production_publication"
)
KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION = 1
KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY = (
    "sealed_raw_bundle_member_hardlink_payload_canonical_v2_metadata_v1"
)
SELECTOR_ACTIVATION_DEFERRED = "deferred_separate_reviewed_change"

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


def _require_run_id(value: object, *, label: str) -> str:
    result = str(value or "").strip()
    if not _RUN_ID.fullmatch(result):
        raise ValueError(f"{label} must be one safe nonempty run ID.")
    return result


def _selector_snapshot(root: Any) -> dict[str, Any]:
    family = root["keypoints_runs"]
    return {
        "family": {
            name: {"present": name in family.attrs, "value": family.attrs.get(name)}
            for name in _SELECTOR_ATTRS
        },
        "root_current_keypoint_group_path": {
            "present": "current_keypoint_group_path" in root.attrs,
            "value": root.attrs.get("current_keypoint_group_path"),
        },
        "root_bundle_authority_digest": (
            canonical_json_sha256(root.attrs[KEYPOINT_BUNDLE_AUTHORITY_ATTR])
            if isinstance(root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR), Mapping)
            else None
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_artifact(path: Path, *, pose_binding: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Keypoint model artifact not found: {resolved}")
    model = pose_binding.get("model")
    expected = model.get("sha256") if isinstance(model, Mapping) else None
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError("Pose binding lacks an exact keypoint-model SHA-256.")
    actual = _sha256_file(resolved)
    if actual != expected:
        raise ValueError("Keypoint model bytes differ from the bound pose schema.")
    stat = resolved.stat()
    return {
        "role": "keypoint_model",
        "path": str(resolved),
        "fingerprint_scheme": CONTENT_FINGERPRINT_SCHEME,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": actual,
        "source": "coordinate_successor_revalidation",
        "pose_schema_binding": copy.deepcopy(dict(pose_binding)),
    }


def _source_bundle_authority(
    root: Any,
    *,
    run_path: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = resolve_active_keypoint_bundle_from_root(root)
    if not isinstance(resolved, Mapping) or not isinstance(
        resolved.get("authority"), Mapping
    ):
        raise ValueError("Archive lacks a committed keypoint bundle authority.")
    authority = dict(resolved["authority"])
    members = authority.get("members")
    raw = members.get("raw_keypoints") if isinstance(members, Mapping) else None
    logical = source_manifest.get("payload", {}).get("logical_content", {})
    if (
        not isinstance(raw, Mapping)
        or raw.get("run_path") != run_path
        or raw.get("manifest_payload_digest") != source_manifest.get("payload_digest")
        or raw.get("manifest_document_digest")
        != canonical_json_sha256(source_manifest)
        or raw.get("logical_content_digest") != logical.get("digest")
    ):
        raise ValueError(
            "Committed keypoint bundle authority does not bind the exact source run."
        )
    return authority


def _dimensions(payload: Mapping[str, Any]) -> KeypointDimensions:
    raw = payload["logical_schema"]["dimensions"]
    return KeypointDimensions(
        n_frames=raw["n_frames"],
        n_instances=raw["n_instances"],
        n_keypoints=raw["n_keypoints"],
        source_width=raw["source_width"],
        source_height=raw["source_height"],
    )


def _submitted_input_mode(preprocessing: Any) -> str:
    """Resolve the actual model submission mode, not the outer cache reader."""

    value = preprocessing.document.get("model_input_mode")
    if value is None and preprocessing.input_mode in {"numpy-list", "tensor"}:
        value = preprocessing.input_mode
    if value not in {"numpy-list", "tensor"}:
        raise ValueError(
            "Keypoint preprocessing lacks an exact submitted model input mode."
        )
    return str(value)


def inspect_keypoint_coordinate_successor_source(
    *,
    analysis_zarr: Path,
    source_run_id: str,
    successor_run_id: str,
    keypoint_model_path: Path,
) -> dict[str, Any]:
    """Validate one source and produce a read-only successor plan."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    source_id = _require_run_id(source_run_id, label="source_run_id")
    successor_id = _require_run_id(successor_run_id, label="successor_run_id")
    if source_id == successor_id:
        raise ValueError("Keypoint coordinate successor cannot replace its source.")
    source_path = archive / "keypoints_runs" / source_id
    target_path = archive / "keypoints_runs" / successor_id
    if not source_path.is_dir():
        raise FileNotFoundError(f"Raw keypoint source not found: {source_path}")
    if target_path.exists():
        raise FileExistsError(f"Immutable successor target exists: {target_path}")

    root = open_zarr_root(archive, mode="r")
    source = root[f"keypoints_runs/{source_id}"]
    manifest = source.attrs.get(KEYPOINT_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError("Raw keypoint source lacks an immutable run manifest.")
    errors = validate_keypoint_run_manifest(manifest)
    if errors:
        raise ValueError("Raw keypoint source manifest is invalid: " + "; ".join(errors))
    payload = manifest["payload"]
    if payload.get("run_id") != source_id:
        raise ValueError("Raw keypoint source manifest binds another run ID.")
    if (
        source.attrs.get("status") != RUN_STATUS_COMPLETE
        or source.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or source.attrs.get("stage_selector_eligible") is not False
        or source.attrs.get("production_candidate") is not True
    ):
        raise ValueError("Raw keypoint source is not one complete sealed candidate.")
    authority = _source_bundle_authority(
        root,
        run_path=f"keypoints_runs/{source_id}",
        source_manifest=manifest,
    )
    dimensions = _dimensions(payload)
    profile = storage_profile_from_manifest(
        payload["storage_plan"]["storage_profile"]
    )
    plans = plan_keypoint_storage(dimensions, profile=profile)
    direct, consolidated = keypoint_metadata_declaration_maps(
        archive, run_id=source_id, plans=plans
    )
    crop_manifest, crop_arrays = _source_crop_manifest_and_arrays(root, payload)
    source_errors = validate_keypoint_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays={name: source[name] for name in KEYPOINT_SCHEMA_V2.binding_paths},
        source_crop_arrays=crop_arrays,
        source_crop_manifest=crop_manifest,
    )
    if source_errors:
        raise ValueError(
            "Raw keypoint source publication is invalid: "
            + "; ".join(source_errors)
        )
    preprocessing = keypoint_preprocessing_from_manifest(payload["preprocessing"])
    transform_value = preprocessing.document.get("model_input_transform")
    if not isinstance(transform_value, Mapping):
        raise ValueError("Raw keypoint preprocessing lacks model_input_transform.")
    transform = model_input_transform_from_attrs(dict(transform_value))
    artifact = _model_artifact(
        keypoint_model_path,
        pose_binding=payload["pose_model_schema_binding"],
    )
    crop = payload["source_crop_snapshot"]
    return json_attr_safe(
        {
            "schema_id": "palette.keypoint_coordinate_successor.source_inspection",
            "schema_version": 1,
            "status": "ready",
            "analysis_zarr": str(archive),
            "source_run_id": source_id,
            "source_run_path": f"keypoints_runs/{source_id}",
            "successor_run_id": successor_id,
            "successor_run_path": f"keypoints_runs/{successor_id}",
            "source_manifest_digest": canonical_json_sha256(manifest),
            "source_manifest_payload_digest": manifest["payload_digest"],
            "source_logical_content_digest": payload["logical_content"]["digest"],
            "source_metadata_tree_sha256": metadata_tree_sha256(source_path),
            "source_authority_digest": canonical_json_sha256(authority),
            "source_crop_path": crop["run_path"],
            "preprocessing_input_mode": _submitted_input_mode(preprocessing),
            "model_input_transform": transform.to_attrs(),
            "model_artifact": artifact,
            "selectors_before": _selector_snapshot(root),
            "selector_eligible": False,
            "selector_activation": SELECTOR_ACTIVATION_DEFERRED,
            "registry_updated": False,
        }
    )


def _coordinate_record_pointers(surfaces: Any) -> dict[str, dict[str, str]]:
    values = {
        "context": surfaces.context.context_record,
        "row_identity": surfaces.context.row_identity,
        "temporal_authority": surfaces.context.temporal_authority,
        "derivation": surfaces.derivation,
    }
    return {
        name: {
            "record_ref": value.record_ref,
            "record_sha256": value.record_sha256,
        }
        for name, value in values.items()
    }


def _source_crop_manifest_and_arrays(root: Any, payload: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    crop_snapshot = payload["source_crop_snapshot"]
    crop_path = str(crop_snapshot["run_path"])
    crop = root[crop_path]
    manifest = crop.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Bound crop source lacks its run manifest.")
    arrays = {
        name: crop[name]
        for name in (
            "instance_key",
            "frame_indices",
            "source_acquisition_frame_index",
            "source_row_signature",
            "roi_coordinates_full",
            "roi_sizes_full",
        )
    }
    return manifest, arrays


def publish_keypoint_coordinate_successor(
    *,
    analysis_zarr: Path,
    source_run_id: str,
    successor_run_id: str,
    keypoint_model_path: Path,
) -> dict[str, Any]:
    """Publish one complete selector-ineligible coordinate successor."""

    initial = inspect_keypoint_coordinate_successor_source(
        analysis_zarr=analysis_zarr,
        source_run_id=source_run_id,
        successor_run_id=successor_run_id,
        keypoint_model_path=keypoint_model_path,
    )
    archive = Path(initial["analysis_zarr"])
    source_id = str(initial["source_run_id"])
    successor_id = str(initial["successor_run_id"])
    source_path = archive / "keypoints_runs" / source_id
    target_path = archive / "keypoints_runs" / successor_id
    copy_receipt: dict[str, int] | None = None

    with archive_metadata_publication_lock(archive):
        checked = inspect_keypoint_coordinate_successor_source(
            analysis_zarr=archive,
            source_run_id=source_id,
            successor_run_id=successor_id,
            keypoint_model_path=keypoint_model_path,
        )
        if checked["selectors_before"] != initial["selectors_before"]:
            raise RuntimeError("Keypoint selectors changed after successor planning.")
        root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
        source = root[f"keypoints_runs/{source_id}"]
        source_manifest = copy.deepcopy(dict(source.attrs[KEYPOINT_RUN_MANIFEST_ATTRIBUTE]))
        source_authority = _source_bundle_authority(
            root,
            run_path=f"keypoints_runs/{source_id}",
            source_manifest=source_manifest,
        )
        try:
            copy_receipt = copy_metadata_and_link_payload(source_path, target_path)
            run = root[f"keypoints_runs/{successor_id}"]
            attrs = dict(run.attrs)
            for name in (
                KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
                KEYPOINT_COORDINATE_CONTEXT_ATTR,
                KEYPOINT_COORDINATE_DERIVATION_ATTR,
                "coordinate_successor_authority",
                "coordinate_successor_authority_sha256",
                RUN_COMPLETED_AT_ATTR,
                "palette_run_failed_at_utc",
                "palette_run_error",
            ):
                attrs.pop(name, None)
            owner = uuid4().hex
            attrs.update(
                {
                    "status": "running",
                    "stage_selector_eligible": False,
                    "production_candidate": True,
                    "production_selector_activation": SELECTOR_ACTIVATION_DEFERRED,
                    "coordinate_contract": "coordinate_successor_preparing",
                    "coordinate_successor_policy": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY,
                    "coordinate_successor_source_run_path": f"keypoints_runs/{source_id}",
                    ATOMIC_PUBLICATION_OWNER_ATTR: owner,
                    KEYPOINT_PUBLICATION_OWNER_ATTR: owner,
                }
            )
            run.attrs.put(attrs)
            mark_run_started(run, run_name=successor_id, stage="keypoints")

            payload = source_manifest["payload"]
            preprocessing = keypoint_preprocessing_from_manifest(payload["preprocessing"])
            transform = model_input_transform_from_attrs(
                dict(preprocessing.document["model_input_transform"])
            )
            artifact = _model_artifact(
                keypoint_model_path,
                pose_binding=payload["pose_model_schema_binding"],
            )
            prepare_keypoint_coordinate_context(
                root,
                f"keypoints_runs/{successor_id}",
                crop_path=str(payload["source_crop_snapshot"]["run_path"]),
                model_input_transform=transform,
                preprocessing_input_mode=_submitted_input_mode(preprocessing),
                model_artifact=artifact,
            )
            surfaces = publish_keypoint_coordinate_surfaces(
                root, f"keypoints_runs/{successor_id}"
            )
            authority = build_coordinate_successor_authority(
                kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
                source_family="keypoints_runs",
                source_run_path=f"keypoints_runs/{source_id}",
                source_manifest=source_manifest,
                source_authority_kind="committed_root_keypoint_bundle_authority_v1",
                source_authority=source_authority,
                successor_family="keypoints_runs",
                successor_run_path=f"keypoints_runs/{successor_id}",
                payload_equivalence={
                    "policy": "same_filesystem_hardlink_payload_exact_logical_digest_v1",
                    "source_logical_content_digest": payload["logical_content"]["digest"],
                    **dict(copy_receipt),
                },
                coordinate_records=_coordinate_record_pointers(surfaces),
            )
            stamp_coordinate_successor_authority(run, authority)
            run.attrs["status"] = RUN_STATUS_COMPLETE
            mark_run_complete(run, run_name=successor_id)

            consolidate_metadata_capture_expected_warnings(archive)
            dimensions = _dimensions(payload)
            profile = storage_profile_from_manifest(
                payload["storage_plan"]["storage_profile"]
            )
            plans = plan_keypoint_storage(dimensions, profile=profile)
            direct, consolidated = keypoint_metadata_declaration_maps(
                archive, run_id=successor_id, plans=plans
            )
            successor_manifest = build_keypoint_coordinate_successor_manifest(
                source_manifest,
                run_id=successor_id,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
            )
            run.attrs[KEYPOINT_RUN_MANIFEST_ATTRIBUTE] = successor_manifest
            consolidate_metadata_capture_expected_warnings(archive)

            published = open_zarr_root(archive, mode="r")
            if _selector_snapshot(published) != initial["selectors_before"]:
                raise RuntimeError("Keypoint selectors changed during successor publication.")
            published_run = published[f"keypoints_runs/{successor_id}"]
            published_surfaces = require_bound_ineligible_keypoint_coordinate_surfaces(
                load_persisted_ineligible_keypoint_coordinate_surfaces(
                    published, f"keypoints_runs/{successor_id}"
                )
            )
            load_coordinate_successor_authority(
                published_run,
                expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
                expected_successor_run_path=f"keypoints_runs/{successor_id}",
            )
            direct, consolidated = keypoint_metadata_declaration_maps(
                archive, run_id=successor_id, plans=plans
            )
            crop_manifest, crop_arrays = _source_crop_manifest_and_arrays(
                published, payload
            )
            errors = validate_keypoint_publication(
                successor_manifest,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
                arrays={name: published_run[name] for name in KEYPOINT_SCHEMA_V2.binding_paths},
                source_crop_arrays=crop_arrays,
                source_crop_manifest=crop_manifest,
            )
            if errors:
                raise RuntimeError(
                    "Published keypoint successor failed validation: " + "; ".join(errors)
                )
            del published_surfaces
            if metadata_tree_sha256(source_path) != initial["source_metadata_tree_sha256"]:
                raise RuntimeError("Immutable source keypoint metadata changed.")
        except BaseException as exc:
            if target_path.exists():
                try:
                    failed_root = zarr.open_group(
                        str(archive), mode="a", use_consolidated=False
                    )
                    failed = failed_root[f"keypoints_runs/{successor_id}"]
                    failed.attrs["status"] = "failed"
                    failed.attrs["stage_selector_eligible"] = False
                    mark_run_failed(failed, run_name=successor_id, error=str(exc))
                    consolidate_metadata_capture_expected_warnings(archive)
                except BaseException:
                    pass
            raise

    return json_attr_safe(
        {
            "schema_id": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID,
            "schema_version": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "published_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "source_run_path": f"keypoints_runs/{source_id}",
            "successor_run_path": f"keypoints_runs/{successor_id}",
            "source_manifest_digest": initial["source_manifest_digest"],
            "source_metadata_tree_sha256": initial["source_metadata_tree_sha256"],
            "copy": copy_receipt,
            "coordinate_contract": "canonical_v2",
            "selector_eligible": False,
            "selectors_before": initial["selectors_before"],
            "selectors_after": initial["selectors_before"],
            "selector_activation": SELECTOR_ACTIVATION_DEFERRED,
            "registry_updated": False,
        }
    )


__all__ = [
    "KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY",
    "KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID",
    "KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION",
    "inspect_keypoint_coordinate_successor_source",
    "publish_keypoint_coordinate_successor",
]
