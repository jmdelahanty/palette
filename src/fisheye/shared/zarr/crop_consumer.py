"""Small consumer binding for strict and historical crop runs.

The future-facing identity is the validated immutable run manifest.  Historical
``crop_signature``/``crop_revision`` pairs remain readable through an explicitly
labelled compatibility profile; they are never synthesized for strict runs.
"""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.shared.zarr.crop_manifest import (
    CROP_RUN_MANIFEST_ATTRIBUTE,
    CROP_RUN_MANIFEST_SCHEMA_ID,
    crop_geometry_policy_from_manifest,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes


CROP_RUN_REFERENCE_SCHEMA_ID = "palette.crop_geometry.run_reference"
CROP_RUN_REFERENCE_SCHEMA_VERSION = 1
CROP_RUN_REFERENCE_STRICT_PROFILE = "immutable_run_manifest_v1"
CROP_RUN_REFERENCE_LEGACY_PROFILE = "legacy_signature_revision_v1"
CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE = "legacy_unversioned_run_name_v1"


def _required_run_id(value: object) -> str:
    run_id = str(value).strip()
    if not run_id or run_id != value or "/" in run_id:
        raise ValueError("Crop run_id must be one exact nonempty group name.")
    return run_id


def _strict_manifest(
    crop_group: Any,
    *,
    run_id: str,
) -> Mapping[str, Any] | None:
    manifest = crop_group.attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE)
    if manifest is None:
        return None
    if not isinstance(manifest, Mapping):
        raise ValueError("Crop run_manifest must be an object when present.")
    errors = validate_crop_run_manifest(manifest)
    if errors:
        raise ValueError("Invalid crop run_manifest: " + "; ".join(errors))
    payload = manifest["payload"]
    if payload["run_id"] != run_id:
        raise ValueError("Crop run group name differs from run_manifest run_id.")
    return manifest


def build_crop_run_reference(
    crop_group: Any,
    *,
    run_id: str,
    allow_unversioned_legacy: bool = False,
) -> dict[str, object]:
    """Bind one crop run without weakening strict identity to legacy attrs."""

    normalized_run_id = _required_run_id(run_id)
    manifest = _strict_manifest(crop_group, run_id=normalized_run_id)
    if manifest is not None:
        logical_content = manifest["payload"]["logical_content"]
        reference: dict[str, object] = {
            "schema_id": CROP_RUN_REFERENCE_SCHEMA_ID,
            "schema_version": CROP_RUN_REFERENCE_SCHEMA_VERSION,
            "profile": CROP_RUN_REFERENCE_STRICT_PROFILE,
            "run_id": normalized_run_id,
            "run_manifest_schema_id": CROP_RUN_MANIFEST_SCHEMA_ID,
            "run_manifest_schema_version": manifest["schema_version"],
            "run_manifest_digest": manifest["payload_digest"],
            "logical_content_digest": logical_content["digest"],
        }
        canonical_json_bytes(reference)
        return reference

    signature = crop_group.attrs.get("crop_signature")
    revision = crop_group.attrs.get("crop_revision")
    if signature is None or signature == "" or revision is None or revision == "":
        if allow_unversioned_legacy:
            reference = {
                "schema_id": CROP_RUN_REFERENCE_SCHEMA_ID,
                "schema_version": CROP_RUN_REFERENCE_SCHEMA_VERSION,
                "profile": CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE,
                "run_id": normalized_run_id,
                "observed_crop_signature": signature,
                "observed_crop_revision": revision,
            }
            canonical_json_bytes(reference)
            return reference
        raise ValueError(
            "Crop run requires a valid immutable run_manifest or the explicitly "
            "historical crop_signature/crop_revision pair."
        )
    reference = {
        "schema_id": CROP_RUN_REFERENCE_SCHEMA_ID,
        "schema_version": CROP_RUN_REFERENCE_SCHEMA_VERSION,
        "profile": CROP_RUN_REFERENCE_LEGACY_PROFILE,
        "run_id": normalized_run_id,
        "crop_signature": signature,
        "crop_revision": revision,
    }
    canonical_json_bytes(reference)
    return reference


def validate_crop_run_reference(value: object) -> dict[str, object]:
    """Validate and return a canonical detached crop-run reference."""

    if not isinstance(value, Mapping):
        raise ValueError("Crop run reference must be an object.")
    reference = dict(value)
    if (
        reference.get("schema_id") != CROP_RUN_REFERENCE_SCHEMA_ID
        or reference.get("schema_version") != CROP_RUN_REFERENCE_SCHEMA_VERSION
    ):
        raise ValueError("Crop run reference schema identity mismatch.")
    run_id = _required_run_id(reference.get("run_id"))
    profile = reference.get("profile")
    if profile == CROP_RUN_REFERENCE_STRICT_PROFILE:
        expected = {
            "schema_id",
            "schema_version",
            "profile",
            "run_id",
            "run_manifest_schema_id",
            "run_manifest_schema_version",
            "run_manifest_digest",
            "logical_content_digest",
        }
        if set(reference) != expected:
            raise ValueError("Strict crop run reference has an unexpected field set.")
        if reference["run_manifest_schema_id"] != CROP_RUN_MANIFEST_SCHEMA_ID:
            raise ValueError("Strict crop run reference manifest schema mismatch.")
        for field in ("run_manifest_digest", "logical_content_digest"):
            digest = reference.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"Strict crop run reference {field} is not SHA-256.")
        if type(reference.get("run_manifest_schema_version")) is not int:
            raise ValueError("Strict crop manifest schema version must be an integer.")
    elif profile == CROP_RUN_REFERENCE_LEGACY_PROFILE:
        expected = {
            "schema_id",
            "schema_version",
            "profile",
            "run_id",
            "crop_signature",
            "crop_revision",
        }
        if set(reference) != expected:
            raise ValueError("Legacy crop run reference has an unexpected field set.")
        if (
            reference.get("crop_signature") in (None, "")
            or reference.get("crop_revision") in (None, "")
        ):
            raise ValueError("Legacy crop run reference is incomplete.")
    elif profile == CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE:
        expected = {
            "schema_id",
            "schema_version",
            "profile",
            "run_id",
            "observed_crop_signature",
            "observed_crop_revision",
        }
        if set(reference) != expected:
            raise ValueError(
                "Unversioned legacy crop reference has an unexpected field set."
            )
    else:
        raise ValueError(f"Unsupported crop run reference profile {profile!r}.")
    if reference["run_id"] != run_id:
        raise ValueError("Crop run reference run_id is not canonical.")
    canonical_json_bytes(reference)
    return reference


def strict_crop_fixed_roi_shape(
    crop_group: Any,
    *,
    run_id: str,
) -> tuple[int, int] | None:
    """Return ``(height, width)`` from a strict fixed-size policy, if present."""

    normalized_run_id = _required_run_id(run_id)
    manifest = _strict_manifest(crop_group, run_id=normalized_run_id)
    if manifest is None:
        return None
    policy = crop_geometry_policy_from_manifest(
        manifest["payload"]["logical_schema"]["crop_policy"]
    )
    if policy.fixed_size_wh is None:
        return None
    width, height = policy.fixed_size_wh
    return int(height), int(width)


def strict_crop_source_frame_shape(
    crop_group: Any,
    *,
    run_id: str,
) -> tuple[int, int] | None:
    """Return ``(height, width)`` from strict source-pixel authority."""

    normalized_run_id = _required_run_id(run_id)
    manifest = _strict_manifest(crop_group, run_id=normalized_run_id)
    if manifest is None:
        return None
    pixel_authority = manifest["payload"]["source_pixel_authority"]
    return (
        int(pixel_authority["source_height"]),
        int(pixel_authority["source_width"]),
    )


__all__ = [
    "CROP_RUN_REFERENCE_LEGACY_PROFILE",
    "CROP_RUN_REFERENCE_SCHEMA_ID",
    "CROP_RUN_REFERENCE_SCHEMA_VERSION",
    "CROP_RUN_REFERENCE_STRICT_PROFILE",
    "CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE",
    "build_crop_run_reference",
    "strict_crop_fixed_roi_shape",
    "strict_crop_source_frame_shape",
    "validate_crop_run_reference",
]
