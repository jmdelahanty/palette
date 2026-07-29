"""Small consumer binding for strict, current signed, and historical crop runs.

The future-facing identity is the validated immutable run manifest. Maintained
acquisition crop-video sources that have not adopted that envelope use a
distinct signed-current profile. Historical ``crop_signature``/``crop_revision``
pairs remain readable through an explicitly labelled compatibility profile;
they are never synthesized for strict runs.
"""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
    SOURCE_PIXELS_RAW_CAMERA_VIDEO,
    normalize_pixel_contract,
    orange_mono_pynvvc_luma_hybrid_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.row_source_signature import (
    RowSourceSignatureSpec,
    load_row_source_signature_spec,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_RUN_MANIFEST_ATTRIBUTE,
    CROP_RUN_MANIFEST_SCHEMA_ID,
    crop_geometry_policy_from_manifest,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.crop_pixel_authority import (
    CROP_SOURCE_VIDEO_DECODE_PROFILE,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes


CROP_RUN_REFERENCE_SCHEMA_ID = "palette.crop_geometry.run_reference"
CROP_RUN_REFERENCE_SCHEMA_VERSION = 1
CROP_RUN_REFERENCE_STRICT_PROFILE = "immutable_run_manifest_v1"
CROP_RUN_REFERENCE_SIGNED_PROFILE = "signed_current_source_v1"
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
            "Crop run requires a valid immutable run_manifest or a signed "
            "crop_signature/crop_revision pair. Unversioned historical runs "
            "require explicit compatibility opt-in."
        )
    source_contract = authoritative_crop_roi_pixel_contract(
        crop_group,
        run_id=normalized_run_id,
    )
    reference = {
        "schema_id": CROP_RUN_REFERENCE_SCHEMA_ID,
        "schema_version": CROP_RUN_REFERENCE_SCHEMA_VERSION,
        "profile": (
            CROP_RUN_REFERENCE_SIGNED_PROFILE
            if source_contract is not None
            else CROP_RUN_REFERENCE_LEGACY_PROFILE
        ),
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
    elif profile in {
        CROP_RUN_REFERENCE_SIGNED_PROFILE,
        CROP_RUN_REFERENCE_LEGACY_PROFILE,
    }:
        expected = {
            "schema_id",
            "schema_version",
            "profile",
            "run_id",
            "crop_signature",
            "crop_revision",
        }
        label = "Signed current" if profile == CROP_RUN_REFERENCE_SIGNED_PROFILE else "Legacy"
        if set(reference) != expected:
            raise ValueError(f"{label} crop run reference has an unexpected field set.")
        if reference.get("crop_signature") in (None, "") or reference.get(
            "crop_revision"
        ) in (None, ""):
            raise ValueError(f"{label} crop run reference is incomplete.")
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
            raise ValueError("Unversioned legacy crop reference has an unexpected field set.")
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
    policy = crop_geometry_policy_from_manifest(manifest["payload"]["logical_schema"]["crop_policy"])
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


def strict_crop_row_source_signature_spec(
    crop_group: Any,
    *,
    run_id: str,
) -> RowSourceSignatureSpec | None:
    """Return the exact strict crop row-signature spec, if one is present."""

    normalized_run_id = _required_run_id(run_id)
    manifest = _strict_manifest(crop_group, run_id=normalized_run_id)
    if manifest is None:
        return None
    row_signature = manifest["payload"].get("row_signature")
    if not isinstance(row_signature, Mapping):
        raise ValueError("Strict crop manifest row_signature must be an object.")
    if row_signature.get("array_path") != "source_row_signature":
        raise ValueError("Strict crop row signature binds the wrong array path.")
    spec = load_row_source_signature_spec(row_signature, prefix="")
    live_signature = crop_group.attrs.get("row_signature")
    if not isinstance(live_signature, Mapping) or dict(live_signature) != dict(row_signature):
        raise ValueError("Strict crop run row_signature attribute differs from its manifest.")
    return spec


def strict_crop_required_roi_pixel_contract(
    crop_group: Any,
    *,
    run_id: str,
) -> dict[str, Any] | None:
    """Return the only model-facing pixel contract allowed by a strict run."""

    normalized_run_id = _required_run_id(run_id)
    manifest = _strict_manifest(crop_group, run_id=normalized_run_id)
    if manifest is None:
        return None
    authority = manifest["payload"].get("source_pixel_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("Strict crop source_pixel_authority must be an object.")
    authority_id = str(authority.get("authority_id") or "")
    expected_suffix = f"#decode={CROP_SOURCE_VIDEO_DECODE_PROFILE}"
    if not authority_id.endswith(expected_suffix):
        raise ValueError("Strict crop source pixel authority uses an unsupported decode profile.")
    live_authority = crop_group.attrs.get("source_pixel_authority")
    if not isinstance(live_authority, Mapping) or dict(live_authority) != dict(authority):
        raise ValueError("Strict crop run source_pixel_authority attribute differs from its manifest.")
    return orange_mono_pynvvc_luma_pixel_contract(
        source_pixels=SOURCE_PIXELS_RAW_CAMERA_VIDEO,
    )


def authoritative_crop_roi_pixel_contract(
    crop_group: Any,
    *,
    run_id: str,
) -> dict[str, Any] | None:
    """Return the exact contract for a current authoritative pixel source.

    Strict crop-v2 geometry currently binds the full-frame camera video.
    Acquisition crop-video runs are an independent current source profile and
    bind their already-cropped video frames.  Older or otherwise ambiguous
    sources return ``None`` instead of having a pixel contract inferred.
    """

    strict_contract = strict_crop_required_roi_pixel_contract(
        crop_group,
        run_id=run_id,
    )
    if strict_contract is not None:
        return strict_contract

    source_pixels = str(
        crop_group.attrs.get("source_pixels") or crop_group.attrs.get("roi_pixel_provider") or ""
    ).strip()
    if source_pixels == SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME:
        expected = orange_mono_pynvvc_luma_hybrid_pixel_contract()
        declared = normalize_pixel_contract(crop_group.attrs.get("roi_pixel_contract"))
        if declared != expected:
            raise ValueError(
                "Hybrid acquisition/full-frame crop run does not declare its "
                "exact per-row pixel-source routing contract."
            )
        return expected
    if source_pixels != SOURCE_PIXELS_ACQUISITION_CROP_VIDEO:
        return None
    expected = orange_mono_pynvvc_luma_pixel_contract(
        source_pixels=SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    )
    declared = normalize_pixel_contract(crop_group.attrs.get("roi_pixel_contract"))
    if declared is not None and declared != expected:
        raise ValueError(
            "Acquisition crop-video run declares a pixel contract that differs "
            "from its authoritative PyNvVC luma source profile."
        )
    return expected


__all__ = [
    "CROP_RUN_REFERENCE_LEGACY_PROFILE",
    "CROP_RUN_REFERENCE_SCHEMA_ID",
    "CROP_RUN_REFERENCE_SCHEMA_VERSION",
    "CROP_RUN_REFERENCE_SIGNED_PROFILE",
    "CROP_RUN_REFERENCE_STRICT_PROFILE",
    "CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE",
    "authoritative_crop_roi_pixel_contract",
    "build_crop_run_reference",
    "strict_crop_fixed_roi_shape",
    "strict_crop_required_roi_pixel_contract",
    "strict_crop_row_source_signature_spec",
    "strict_crop_source_frame_shape",
    "validate_crop_run_reference",
]
