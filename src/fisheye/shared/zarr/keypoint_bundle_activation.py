"""Atomic authority selection for one sealed keypoint-v2 four-surface chain.

The raw, quality, refined, and body-frame run manifests are immutable and bind
their exact group/array metadata declarations.  Activation therefore must not
rewrite per-run lifecycle attributes or family selectors.  Instead, this
module validates all four sealed candidates and publishes one root authority
record.  Consolidating that root record is the sole published visibility
commit for bundle-aware consumers.
"""

from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np
import zarr

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.body_frame_manifest import (
    BODY_FRAME_RUN_MANIFEST_ATTRIBUTE,
    body_frame_metadata_declarations_digest,
    validate_body_frame_run_manifest,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_RUN_MANIFEST_ATTRIBUTE,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    keypoint_crop_source_from_manifest,
    keypoint_metadata_declarations_digest,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
)
from fisheye.shared.zarr.keypoint_quality_manifest import (
    KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE,
    keypoint_quality_metadata_declarations_digest,
    validate_keypoint_quality_run_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    refined_keypoint_metadata_declarations_digest,
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent


KEYPOINT_BUNDLE_AUTHORITY_ATTR = "keypoint_bundle_authority"
KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR = "keypoint_bundle_authority_generation"
KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR = "keypoint_bundle_authority_lease"
KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID = "palette.keypoint.bundle_authority"
KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_VERSION = 1
KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_ID = "palette.keypoint.bundle_activation_plan"
KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_VERSION = 1
KEYPOINT_BUNDLE_ACTIVATION_POLICY = (
    "sealed_four_surface_root_authority_then_consolidated_visibility_v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OWNER = re.compile(r"^[0-9a-f]{32}$")
_ROLE_SPECS: Mapping[
    str,
    tuple[str, str, Callable[[Mapping[str, Any]], tuple[str, ...]], Callable[..., str]],
] = {
    "raw_keypoints": (
        "keypoints_runs",
        KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
        validate_keypoint_run_manifest,
        keypoint_metadata_declarations_digest,
    ),
    "keypoint_quality": (
        "keypoint_quality_runs",
        KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE,
        validate_keypoint_quality_run_manifest,
        keypoint_quality_metadata_declarations_digest,
    ),
    "refined_keypoints": (
        "refined_keypoints_runs",
        REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
        validate_refined_keypoint_run_manifest,
        refined_keypoint_metadata_declarations_digest,
    ),
    "body_frame": (
        "analysis/body_frame_runs",
        BODY_FRAME_RUN_MANIFEST_ATTRIBUTE,
        validate_body_frame_run_manifest,
        body_frame_metadata_declarations_digest,
    ),
}


class KeypointBundleActivationError(ValueError):
    """Raised when a four-surface keypoint authority cannot be selected."""


def _fail(message: str) -> None:
    raise KeypointBundleActivationError(message)


def _require_run_id(value: object, *, name: str) -> str:
    result = str(value or "").strip()
    if not result or "/" in result:
        _fail(f"{name} must be one nonempty run name.")
    return result


def _decode_authority_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _get_group_path(parent: object, path: str) -> object | None:
    current = parent
    for part in [part for part in path.split("/") if part]:
        try:
            if part not in current:  # type: ignore[operator]
                return None
            current = current[part]  # type: ignore[index]
        except Exception:
            return None
    return current


def resolve_active_keypoint_bundle_from_root(
    root: object,
) -> dict[str, object] | None:
    """Resolve one committed root authority without legacy family selectors.

    Registry reconciliation receives an already-open root, including in-memory
    test roots, so this validator deliberately operates on the group protocol
    rather than reopening an archive path. Any present-but-invalid authority
    fails closed.
    """

    attrs = getattr(root, "attrs", {})
    if KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR in attrs:
        raise ValueError(
            "Keypoint bundle activation lease is present; registry reconciliation "
            "requires a committed authority generation."
        )
    authority = attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR)
    if authority is None:
        return None
    if not isinstance(authority, Mapping):
        raise ValueError("Keypoint bundle authority must be one object.")
    expected_fields = {
        "schema_id",
        "schema_version",
        "generation",
        "base_generation",
        "policy",
        "activation_plan_payload_digest",
        "prior_authority_present",
        "prior_authority_digest",
        "crop",
        "members",
        "activated_at_utc",
        "activation_owner_uuid",
    }
    generation = attrs.get(KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR)
    if (
        set(authority) != expected_fields
        or authority.get("schema_id") != KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID
        or authority.get("schema_version")
        != KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_VERSION
        or authority.get("policy") != KEYPOINT_BUNDLE_ACTIVATION_POLICY
        or type(generation) is not int
        or generation <= 0
        or authority.get("generation") != generation
        or type(authority.get("base_generation")) is not int
        or authority.get("base_generation") != generation - 1
    ):
        raise ValueError("Keypoint bundle authority schema or generation is invalid.")
    members = authority.get("members")
    crop = authority.get("crop")
    if not isinstance(members, Mapping) or set(members) != set(_ROLE_SPECS):
        raise ValueError("Keypoint bundle authority member roles are invalid.")
    if not isinstance(crop, Mapping):
        raise ValueError("Keypoint bundle authority crop member is invalid.")

    resolved: dict[str, object] = {
        "authority": dict(authority),
        "generation": generation,
        "activation_plan_payload_digest": authority.get(
            "activation_plan_payload_digest"
        ),
    }
    declarations: dict[str, Mapping[str, object]] = {"crop": crop}
    declarations.update(
        {
            role: member
            for role, member in members.items()
            if isinstance(member, Mapping)
        }
    )
    if set(declarations) != {"crop", *_ROLE_SPECS}:
        raise ValueError("Keypoint bundle authority member declarations are invalid.")
    parent_paths = {
        "crop": "crop_runs",
        **{role: spec[0] for role, spec in _ROLE_SPECS.items()},
    }
    for role, parent_path in parent_paths.items():
        member = declarations[role]
        if role != "crop" and member.get("role") != role:
            raise ValueError(
                f"Keypoint bundle authority {role} declaration has another role."
            )
        run_id = _decode_authority_text(member.get("run_id"))
        run_path = _decode_authority_text(member.get("run_path"))
        expected_path = f"{parent_path}/{run_id}" if run_id else None
        if not run_id or run_path != expected_path:
            raise ValueError(
                f"Keypoint bundle authority {role} path does not bind its run ID."
            )
        parent = _get_group_path(root, parent_path)
        group = _get_group_path(root, run_path)
        if parent is None or group is None:
            raise ValueError(
                f"Keypoint bundle authority member is absent: {run_path}."
            )
        group_attrs = getattr(group, "attrs", {})
        if (
            group_attrs.get("status") != "complete"
            or group_attrs.get("palette_run_completion_status") != "complete"
            or group_attrs.get("production_candidate") is not True
            or group_attrs.get("stage_selector_eligible") is not False
        ):
            raise ValueError(
                f"Keypoint bundle authority member is not sealed: {run_path}."
            )
        manifest = group_attrs.get("run_manifest")
        if (
            not isinstance(manifest, Mapping)
            or member.get("manifest_payload_digest")
            != manifest.get("payload_digest")
            or member.get("manifest_document_digest")
            != canonical_json_sha256(manifest)
        ):
            raise ValueError(
                f"Keypoint bundle authority member manifest changed: {run_path}."
            )
        resolved[role] = {
            "run_id": run_id,
            "run_path": run_path,
            "parent": parent,
            "group": group,
        }
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    import json

    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _metadata_maps(
    archive: Path,
    *,
    run_path: str,
    array_paths: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    relative_paths = ("", *array_paths)
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        node = archive / run_path
        if relative:
            node /= relative
        direct[relative] = _read_json_object(node / "zarr.json")
    root_metadata = _read_json_object(archive / "zarr.json")
    envelope = root_metadata.get("consolidated_metadata")
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("kind") != "inline"
        or envelope.get("must_understand") is not False
        or not isinstance(envelope.get("metadata"), Mapping)
    ):
        _fail("Published archive lacks one valid inline consolidated generation.")
    flattened = envelope["metadata"]
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_path if not relative else f"{run_path}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            _fail(f"Consolidated metadata lacks {full_path!r}.")
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def _validate_logical_arrays(run: Any, manifest: Mapping[str, Any]) -> None:
    logical_content = manifest["payload"].get("logical_content")
    document = (
        logical_content.get("document")
        if isinstance(logical_content, Mapping)
        else None
    )
    declared = document.get("arrays") if isinstance(document, Mapping) else None
    if not isinstance(declared, Mapping) or not declared:
        _fail("Candidate manifest lacks logical array declarations.")
    for path, declaration in declared.items():
        if not isinstance(path, str) or not isinstance(declaration, Mapping):
            _fail("Candidate logical array declaration is malformed.")
        values = np.asarray(run[path][...])
        if (
            list(values.shape) != declaration.get("shape")
            or str(values.dtype) != declaration.get("dtype")
            or declaration.get("digest_algorithm")
            != "sha256_c_contiguous_bytes_v1"
            or sha256_array(values) != declaration.get("sha256")
        ):
            _fail(f"Candidate logical array changed: {run.path}/{path}.")


def _member_document(
    archive: Path,
    root: Any,
    *,
    role: str,
    run_id: str,
) -> dict[str, Any]:
    parent_path, manifest_attr, manifest_validator, metadata_digest = _ROLE_SPECS[
        role
    ]
    parent = root[parent_path]
    run_path = f"{parent_path}/{run_id}"
    run = root[run_path]
    manifest = run.attrs.get(manifest_attr)
    if not isinstance(manifest, Mapping):
        _fail(f"{run_path} lacks its exact run manifest.")
    errors = manifest_validator(manifest)
    if errors:
        _fail(f"{run_path} manifest is invalid: {'; '.join(errors)}")
    if manifest["payload"].get("run_id") != run_id:
        _fail(f"{run_path} manifest names another run.")
    owner = run.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
    if not isinstance(owner, str) or _OWNER.fullmatch(owner) is None:
        _fail(f"{run_path} lacks one exact production publication owner.")
    if (
        run.attrs.get("status") != "complete"
        or run.attrs.get("palette_run_completion_status") != "complete"
        or run.attrs.get("production_candidate") is not True
        or run.attrs.get("stage_selector_eligible") is not False
        or not is_run_complete_in_parent(parent, run)
    ):
        _fail(f"{run_path} is not one sealed selector-ineligible candidate.")
    validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    logical_arrays = manifest["payload"]["logical_content"]["document"]["arrays"]
    array_paths = tuple(sorted(str(path) for path in logical_arrays))
    direct, consolidated = _metadata_maps(
        archive,
        run_path=run_path,
        array_paths=array_paths,
    )
    observed_metadata_digest = metadata_digest(
        direct,
        consolidated_by_path=consolidated,
    )
    if observed_metadata_digest != manifest["payload"]["publication"].get(
        "metadata_declarations_digest"
    ):
        _fail(f"{run_path} metadata declaration digest changed.")
    _validate_logical_arrays(run, manifest)
    return {
        "role": role,
        "run_id": run_id,
        "run_path": run_path,
        "publication_owner_uuid": owner,
        "manifest_payload_digest": manifest["payload_digest"],
        "manifest_document_digest": canonical_json_sha256(manifest),
        "logical_content_digest": manifest["payload"]["logical_content"]["digest"],
    }


def _require_cross_bindings(
    *,
    crop_id: str,
    crop_manifest: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    crop_digest = canonical_json_sha256(crop_manifest)
    raw = manifests["raw_keypoints"]
    raw_digest = canonical_json_sha256(raw)
    quality = manifests["keypoint_quality"]
    quality_digest = canonical_json_sha256(quality)
    refined = manifests["refined_keypoints"]
    refined_digest = canonical_json_sha256(refined)
    body = manifests["body_frame"]
    expected_crop = keypoint_crop_source_from_manifest(crop_manifest).as_manifest()
    if raw["payload"].get("source_crop_snapshot") != expected_crop:
        _fail("Raw keypoint candidate does not bind the exact crop authority.")
    quality_source = quality["payload"].get("source_keypoint_snapshot")
    if (
        not isinstance(quality_source, Mapping)
        or quality_source.get("run_name") != raw["payload"].get("run_id")
        or quality_source.get("manifest_digest") != raw_digest
    ):
        _fail("Quality candidate does not bind the exact raw candidate.")
    sources = refined["payload"].get("source_bindings")
    if not isinstance(sources, Mapping):
        _fail("Refined candidate lacks source bindings.")
    expected_refined = {
        "raw_keypoint_snapshot": (raw["payload"].get("run_id"), raw_digest),
        "quality_snapshot": (quality["payload"].get("run_id"), quality_digest),
        "crop_snapshot": (crop_id, crop_digest),
    }
    for name, (run_id, digest) in expected_refined.items():
        value = sources.get(name)
        if (
            not isinstance(value, Mapping)
            or value.get("run_id") != run_id
            or value.get("manifest_digest") != digest
        ):
            _fail(f"Refined candidate {name} binding changed.")
    body_source = body["payload"].get("source_keypoint_snapshot")
    if (
        not isinstance(body_source, Mapping)
        or body_source.get("run_name") != refined["payload"].get("run_id")
        or body_source.get("manifest_digest") != refined_digest
    ):
        _fail("Body-frame candidate does not bind the exact refined candidate.")


def _plan_payload(
    analysis_zarr: Path,
    *,
    crop_run_id: str,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    expected_activation_lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    archive = analysis_zarr.expanduser().resolve()
    root = open_zarr_root(archive, mode="r")
    generation = root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR, 0)
    if type(generation) is not int or generation < 0:
        _fail("Keypoint bundle authority generation is invalid.")
    observed_lease = root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR)
    if expected_activation_lease is None:
        if KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR in root.attrs:
            _fail("A keypoint bundle activation lease is already present.")
    elif observed_lease != dict(expected_activation_lease):
        _fail("The owned keypoint bundle activation lease changed.")
    crop_id = _require_run_id(crop_run_id, name="crop_run_id")
    crop_parent = root["crop_runs"]
    crop_run = root[f"crop_runs/{crop_id}"]
    crop_manifest = crop_run.attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(crop_manifest, Mapping):
        _fail("Bound crop run lacks its exact manifest.")
    crop_errors = validate_crop_run_manifest(crop_manifest)
    if crop_errors or not is_run_complete_in_parent(crop_parent, crop_run):
        _fail("Bound crop authority is invalid or incomplete.")
    validate_direct_consolidated_subtree(
        archive, subtree_path=f"crop_runs/{crop_id}"
    )
    run_ids = {
        "raw_keypoints": _require_run_id(raw_run_id, name="raw_run_id"),
        "keypoint_quality": _require_run_id(quality_run_id, name="quality_run_id"),
        "refined_keypoints": _require_run_id(
            refined_run_id, name="refined_run_id"
        ),
        "body_frame": _require_run_id(
            body_frame_run_id, name="body_frame_run_id"
        ),
    }
    members: dict[str, dict[str, Any]] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    for role, run_id in run_ids.items():
        members[role] = _member_document(
            archive, root, role=role, run_id=run_id
        )
        parent_path, manifest_attr, _validator, _digest = _ROLE_SPECS[role]
        manifests[role] = root[f"{parent_path}/{run_id}"].attrs[manifest_attr]
    _require_cross_bindings(
        crop_id=crop_id,
        crop_manifest=crop_manifest,
        manifests=manifests,
    )
    authority = root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR)
    prior_authority_present = isinstance(authority, Mapping)
    prior_authority_digest = (
        canonical_json_sha256(authority) if prior_authority_present else None
    )
    return json_attr_safe(
        {
            "analysis_zarr": str(archive),
            "base_generation": generation,
            "next_generation": generation + 1,
            "prior_authority_present": prior_authority_present,
            "prior_authority_digest": prior_authority_digest,
            "crop": {
                "run_id": crop_id,
                "run_path": f"crop_runs/{crop_id}",
                "manifest_payload_digest": crop_manifest["payload_digest"],
                "manifest_document_digest": canonical_json_sha256(crop_manifest),
            },
            "members": members,
            "policy": KEYPOINT_BUNDLE_ACTIVATION_POLICY,
        }
    )


def build_keypoint_bundle_activation_plan(
    analysis_zarr: Path,
    *,
    crop_run_id: str,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
) -> dict[str, Any]:
    """Exhaustively validate and return one deterministic dry-run plan."""

    payload = _plan_payload(
        analysis_zarr,
        crop_run_id=crop_run_id,
        raw_run_id=raw_run_id,
        quality_run_id=quality_run_id,
        refined_run_id=refined_run_id,
        body_frame_run_id=body_frame_run_id,
    )
    return {
        "schema_id": KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_ID,
        "schema_version": KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def _build_plan_during_owned_lease(
    analysis_zarr: Path,
    *,
    crop_run_id: str,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    lease: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _plan_payload(
        analysis_zarr,
        crop_run_id=crop_run_id,
        raw_run_id=raw_run_id,
        quality_run_id=quality_run_id,
        refined_run_id=refined_run_id,
        body_frame_run_id=body_frame_run_id,
        expected_activation_lease=lease,
    )
    return {
        "schema_id": KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_ID,
        "schema_version": KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def _authority_from_plan(
    plan: Mapping[str, Any], *, owner: str
) -> dict[str, Any]:
    payload = plan["payload"]
    return json_attr_safe(
        {
            "schema_id": KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID,
            "schema_version": KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_VERSION,
            "generation": payload["next_generation"],
            "base_generation": payload["base_generation"],
            "policy": KEYPOINT_BUNDLE_ACTIVATION_POLICY,
            "activation_plan_payload_digest": plan["payload_digest"],
            "prior_authority_present": payload["prior_authority_present"],
            "prior_authority_digest": payload["prior_authority_digest"],
            "crop": payload["crop"],
            "members": payload["members"],
            "activated_at_utc": utc_now(),
            "activation_owner_uuid": owner,
        }
    )


def _same_active_candidate(
    authority: object, plan: Mapping[str, Any]
) -> bool:
    return bool(
        isinstance(authority, Mapping)
        and authority.get("schema_id") == KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID
        and authority.get("crop") == plan["payload"]["crop"]
        and authority.get("members") == plan["payload"]["members"]
    )


def activate_keypoint_bundle_from_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one reviewed plan; fail if any candidate or authority state changed."""

    if (
        not isinstance(plan, Mapping)
        or plan.get("schema_id") != KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_ID
        or plan.get("schema_version")
        != KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_VERSION
        or not isinstance(plan.get("payload"), Mapping)
        or plan.get("payload_digest")
        != canonical_json_sha256(plan.get("payload"))
    ):
        _fail("Activation plan envelope or digest is invalid.")
    payload = plan["payload"]
    archive = Path(str(payload["analysis_zarr"])).expanduser().resolve()
    run_ids = {
        role: str(payload["members"][role]["run_id"]) for role in _ROLE_SPECS
    }
    with archive_metadata_publication_lock(archive):
        expected = build_keypoint_bundle_activation_plan(
            archive,
            crop_run_id=str(payload["crop"]["run_id"]),
            raw_run_id=run_ids["raw_keypoints"],
            quality_run_id=run_ids["keypoint_quality"],
            refined_run_id=run_ids["refined_keypoints"],
            body_frame_run_id=run_ids["body_frame"],
        )
        root = open_zarr_root(archive, mode="r")
        if _same_active_candidate(root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR), plan):
            return {
                "status": "already_active",
                "authority": copy.deepcopy(
                    root.attrs[KEYPOINT_BUNDLE_AUTHORITY_ATTR]
                ),
                "activation_performed": False,
            }
        if expected != dict(plan):
            _fail("Activation plan is stale; live candidates or authority changed.")
        owner = uuid4().hex
        lease = {
            "owner_uuid": owner,
            "plan_payload_digest": plan["payload_digest"],
            "next_generation": payload["next_generation"],
            "policy": KEYPOINT_BUNDLE_ACTIVATION_POLICY,
        }
        mutable = open_zarr_root(archive, mode="a")
        before = copy.deepcopy(dict(mutable.attrs))
        mutable.attrs[KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR] = lease
        committed = False
        authority: dict[str, Any] | None = None
        try:
            # Zarr's root-attribute write replaces the inline consolidated
            # envelope. Publish a fail-closed lease generation before the
            # second exhaustive validation, then replace it with authority.
            consolidate_metadata_capture_expected_warnings(archive)
            authority = _authority_from_plan(plan, owner=owner)
            checked = open_zarr_root(archive, mode="r")
            if checked.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR) != lease:
                _fail("Keypoint bundle activation lease changed.")
            rechecked = _build_plan_during_owned_lease(
                archive,
                crop_run_id=str(payload["crop"]["run_id"]),
                raw_run_id=run_ids["raw_keypoints"],
                quality_run_id=run_ids["keypoint_quality"],
                refined_run_id=run_ids["refined_keypoints"],
                body_frame_run_id=run_ids["body_frame"],
                lease=lease,
            )
            if rechecked != dict(plan):
                _fail("Keypoint candidates changed after lease acquisition.")
            final_root = open_zarr_root(archive, mode="a")
            final_attrs = copy.deepcopy(dict(final_root.attrs))
            if final_attrs.get(KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR) != lease:
                _fail("Keypoint bundle activation lease was replaced.")
            final_attrs.pop(KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR, None)
            final_attrs[KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR] = payload[
                "next_generation"
            ]
            final_attrs[KEYPOINT_BUNDLE_AUTHORITY_ATTR] = authority
            final_root.attrs.put(final_attrs)
            consolidation = consolidate_metadata_capture_expected_warnings(archive)
            direct = open_zarr_root(archive, mode="r")
            consolidated = zarr.open_group(
                str(archive), mode="r", zarr_format=3, use_consolidated=True
            )
            for label, view in (("direct", direct), ("consolidated", consolidated)):
                if (
                    view.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR) != authority
                    or view.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR)
                    != payload["next_generation"]
                    or KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR in view.attrs
                ):
                    _fail(f"Keypoint bundle {label} authority did not persist.")
            committed = True
            return {
                "status": "activated",
                "authority": authority,
                "activation_performed": True,
                "consolidation": consolidation,
            }
        finally:
            if not committed:
                observed = open_zarr_root(archive, mode="r")
                if (
                    authority is not None
                    and observed.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR)
                    == authority
                ):
                    consolidate_metadata_capture_expected_warnings(archive)
                    committed = True
                else:
                    rollback = open_zarr_root(archive, mode="a")
                    if rollback.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR) == lease:
                        rollback.attrs.put(before)
                        consolidate_metadata_capture_expected_warnings(archive)


def validate_active_keypoint_bundle(analysis_zarr: Path) -> dict[str, Any]:
    """Revalidate and return the active four-surface authority."""

    archive = analysis_zarr.expanduser().resolve()
    root = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    authority = root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR)
    if not isinstance(authority, Mapping):
        _fail("No consolidated keypoint bundle authority is active.")
    expected_fields = {
        "schema_id",
        "schema_version",
        "generation",
        "base_generation",
        "policy",
        "activation_plan_payload_digest",
        "prior_authority_present",
        "prior_authority_digest",
        "crop",
        "members",
        "activated_at_utc",
        "activation_owner_uuid",
    }
    if set(authority) != expected_fields or (
        authority.get("schema_id") != KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID
        or authority.get("schema_version")
        != KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_VERSION
        or authority.get("policy") != KEYPOINT_BUNDLE_ACTIVATION_POLICY
    ):
        _fail("Active keypoint bundle authority schema is unsupported.")
    members = authority.get("members")
    crop = authority.get("crop")
    if not isinstance(members, Mapping) or not isinstance(crop, Mapping):
        _fail("Active keypoint bundle authority is malformed.")
    generation = authority.get("generation")
    base_generation = authority.get("base_generation")
    if (
        type(generation) is not int
        or type(base_generation) is not int
        or generation != base_generation + 1
        or root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR) != generation
        or KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR in root.attrs
        or not isinstance(authority.get("activation_owner_uuid"), str)
        or _OWNER.fullmatch(authority["activation_owner_uuid"]) is None
        or not isinstance(authority.get("activation_plan_payload_digest"), str)
        or _SHA256.fullmatch(authority["activation_plan_payload_digest"]) is None
    ):
        _fail("Active keypoint bundle authority generation is malformed.")
    original_plan_payload = {
        "analysis_zarr": str(archive),
        "base_generation": base_generation,
        "next_generation": generation,
        "prior_authority_present": authority.get("prior_authority_present"),
        "prior_authority_digest": authority.get("prior_authority_digest"),
        "crop": dict(crop),
        "members": {str(role): dict(value) for role, value in members.items()},
        "policy": KEYPOINT_BUNDLE_ACTIVATION_POLICY,
    }
    if canonical_json_sha256(original_plan_payload) != authority.get(
        "activation_plan_payload_digest"
    ):
        _fail("Active keypoint bundle authority does not bind its reviewed plan.")
    plan = build_keypoint_bundle_activation_plan(
        archive,
        crop_run_id=str(crop.get("run_id")),
        raw_run_id=str(members["raw_keypoints"]["run_id"]),
        quality_run_id=str(members["keypoint_quality"]["run_id"]),
        refined_run_id=str(members["refined_keypoints"]["run_id"]),
        body_frame_run_id=str(members["body_frame"]["run_id"]),
    )
    if not _same_active_candidate(authority, plan):
        _fail("Active keypoint bundle authority differs from live candidates.")
    return dict(authority)


__all__ = [
    "KEYPOINT_BUNDLE_ACTIVATION_PLAN_SCHEMA_ID",
    "KEYPOINT_BUNDLE_AUTHORITY_ATTR",
    "KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR",
    "KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID",
    "KeypointBundleActivationError",
    "activate_keypoint_bundle_from_plan",
    "build_keypoint_bundle_activation_plan",
    "resolve_active_keypoint_bundle_from_root",
    "validate_active_keypoint_bundle",
]
