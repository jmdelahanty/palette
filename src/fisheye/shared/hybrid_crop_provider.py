"""Immutable row identity for hybrid acquisition/offline crop providers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
    normalize_pixel_contract,
    orange_mono_pynvvc_luma_hybrid_pixel_contract,
)
from fisheye.shared.row_source_signature import (
    RowSourceSignatureBatch,
    build_row_source_signatures,
    load_row_source_signature_spec,
    validate_row_source_signature_array,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


HYBRID_CROP_RUN_SCHEMA_ID = "palette.hybrid_acquisition_offline_crop_run.v3"
HYBRID_CROP_IDENTITY_SCHEMA_ID = "palette.hybrid_crop_provider_identity"
HYBRID_CROP_IDENTITY_SCHEMA_VERSION = 1
HYBRID_CROP_SIGNATURE_SCHEMA_ID = "palette.hybrid_crop_provider.signature"
HYBRID_CROP_SIGNATURE_SCHEMA_VERSION = 1
HYBRID_CROP_SIGNATURE_STAGE = "hybrid_crop_pixel_routing_v1"
HYBRID_CROP_REVISION_INITIAL = 1
HYBRID_CROP_PIXEL_FINGERPRINT_BASIS = (
    "sha256_provider_record_digest_and_exact_pixel_contract_v1"
)
HYBRID_CROP_ROWSET_FINGERPRINT_BASIS = (
    "sha256_domain_instance_keys_and_row_source_signatures_v1"
)

_CONTENT_COMPONENTS: tuple[tuple[str, Any, tuple[int, ...]], ...] = (
    ("bbox_norm_coords", np.float64, (4,)),
    ("crop_state_codes", np.int8, ()),
    ("frame_indices", np.int64, ()),
    ("roi_coordinates_full", np.int32, (2,)),
    ("roi_sizes_full", np.int32, (2,)),
    ("routing_reason_codes", np.int8, ()),
    ("source_acquisition_crop_row_indices", np.int64, ()),
    ("source_acquisition_crop_xywh", np.float64, (4,)),
    ("source_crop_video_frame_indices", np.int64, ()),
    ("source_crop_xywh", np.float64, (4,)),
    ("source_pixel_kind_codes", np.int8, ()),
    ("source_refined_row_ids", np.int64, ()),
    ("supplemental_cache_row_indices", np.int64, ()),
)
_COLLECTION_CONTENT_COMPONENTS: tuple[tuple[str, Any, tuple[int, ...]], ...] = (
    ("source_crop_video_member_indices", np.int32, ()),
    ("source_full_video_member_indices", np.int32, ()),
)

HYBRID_STRICT_CROP_GEOMETRY_PATHS = (
    "instance_key",
    "source_refined_row_ids",
    "frame_indices",
    "source_acquisition_frame_index",
    "roi_coordinates_full",
    "roi_sizes_full",
)


@dataclass(frozen=True)
class HybridCropProviderIdentity:
    """Signed identity persisted beside one immutable hybrid crop rowset."""

    row_signatures: RowSourceSignatureBatch
    crop_signature: Mapping[str, Any]
    source_pixel_fingerprint: str
    source_rowset_fingerprint: str

    def attrs(self) -> dict[str, Any]:
        return {
            **self.row_signatures.spec.to_attrs(),
            "source_pixel_fingerprint": self.source_pixel_fingerprint,
            "source_pixel_fingerprint_basis": HYBRID_CROP_PIXEL_FINGERPRINT_BASIS,
            "source_rowset_fingerprint": self.source_rowset_fingerprint,
            "source_rowset_fingerprint_basis": HYBRID_CROP_ROWSET_FINGERPRINT_BASIS,
            "crop_signature": dict(self.crop_signature),
            "crop_revision": HYBRID_CROP_REVISION_INITIAL,
        }


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value).strip()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


def _content_arrays(
    arrays: Mapping[str, Any],
    *,
    row_count: int,
) -> dict[str, np.ndarray]:
    content: dict[str, np.ndarray] = {}
    for name, dtype, trailing_shape in _CONTENT_COMPONENTS:
        if name not in arrays:
            raise ValueError(f"Hybrid crop provider identity is missing {name!r}.")
        values = np.asarray(arrays[name], dtype=dtype)
        expected = (row_count, *trailing_shape)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Hybrid crop provider identity {name!r} must have shape "
                f"{expected}, got {values.shape}."
            )
        content[name] = values
    collection_names = [
        name for name, _dtype, _shape in _COLLECTION_CONTENT_COMPONENTS
    ]
    collection_present = [name in arrays for name in collection_names]
    if any(collection_present) and not all(collection_present):
        raise ValueError(
            "Hybrid collection provider identity requires both crop-video and "
            "full-video member index arrays."
        )
    if all(collection_present):
        for name, dtype, trailing_shape in _COLLECTION_CONTENT_COMPONENTS:
            values = np.asarray(arrays[name], dtype=dtype)
            expected = (row_count, *trailing_shape)
            if tuple(values.shape) != expected:
                raise ValueError(
                    f"Hybrid crop provider identity {name!r} must have shape "
                    f"{expected}, got {values.shape}."
                )
            content[name] = values
    return content


def _rowset_fingerprint(
    instance_keys: np.ndarray,
    source_row_signatures: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"palette.hybrid_crop_provider.rowset.v1\x00")
    digest.update(np.asarray(instance_keys, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(source_row_signatures, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def build_hybrid_crop_provider_identity(
    arrays: Mapping[str, Any],
    *,
    provider_record_sha256: str,
    routing_policy_id: str,
    crop_policy_id: str,
    pixel_contract: Mapping[str, Any] | None = None,
) -> HybridCropProviderIdentity:
    """Build exact per-row signatures for one hybrid provider publication."""

    provider_digest = _require_sha256(
        provider_record_sha256,
        label="provider_record_sha256",
    )
    routing_policy = str(routing_policy_id).strip()
    crop_policy = str(crop_policy_id).strip()
    if not routing_policy or not crop_policy:
        raise ValueError("Hybrid crop routing and crop policy IDs must be non-empty.")
    contract = normalize_pixel_contract(
        pixel_contract or orange_mono_pynvvc_luma_hybrid_pixel_contract()
    )
    expected_contract = orange_mono_pynvvc_luma_hybrid_pixel_contract()
    if contract != expected_contract:
        raise ValueError(
            "Hybrid crop provider identity requires the exact Orange PyNvVC "
            "luma hybrid pixel contract."
        )

    keys = np.asarray(arrays.get("instance_key"), dtype=np.uint64).reshape(-1)
    content = _content_arrays(arrays, row_count=int(keys.shape[0]))
    pixel_descriptor = {
        "schema_id": HYBRID_CROP_IDENTITY_SCHEMA_ID,
        "schema_version": HYBRID_CROP_IDENTITY_SCHEMA_VERSION,
        "source_pixels": SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
        "provider_record_sha256": provider_digest,
        "routing_policy_id": routing_policy,
        "crop_policy_id": crop_policy,
        "pixel_contract": contract,
    }
    pixel_fingerprint = canonical_json_sha256(pixel_descriptor)
    signatures = build_row_source_signatures(
        stage=HYBRID_CROP_SIGNATURE_STAGE,
        instance_keys=keys,
        content_components=content,
        compatibility_context={
            **pixel_descriptor,
            "source_pixel_fingerprint": pixel_fingerprint,
        },
    )
    rowset_fingerprint = _rowset_fingerprint(keys, signatures.signatures)
    crop_signature = {
        "schema_id": HYBRID_CROP_SIGNATURE_SCHEMA_ID,
        "schema_version": HYBRID_CROP_SIGNATURE_SCHEMA_VERSION,
        "source_pixels": SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
        "provider_record_sha256": provider_digest,
        "source_pixel_fingerprint": pixel_fingerprint,
        "source_row_signature_spec_digest": signatures.spec.spec_digest,
        "source_rowset_fingerprint": rowset_fingerprint,
    }
    canonical_json_sha256(crop_signature)
    return HybridCropProviderIdentity(
        row_signatures=signatures,
        crop_signature=crop_signature,
        source_pixel_fingerprint=pixel_fingerprint,
        source_rowset_fingerprint=rowset_fingerprint,
    )


def validate_hybrid_crop_signed_identity(
    group: Any,
    *,
    expected_provider_record_sha256: str,
) -> dict[str, Any]:
    """Recompute and validate the signed identity of one hybrid crop group."""

    expected_provider = _require_sha256(
        expected_provider_record_sha256,
        label="expected_provider_record_sha256",
    )
    if str(group.attrs.get("schema_id") or "") != HYBRID_CROP_RUN_SCHEMA_ID:
        raise ValueError("Hybrid crop run schema identity is incompatible.")
    if str(group.attrs.get("source_pixels") or "") != SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME:
        raise ValueError("Hybrid crop run source-pixel identity is incompatible.")
    if str(group.attrs.get("provider_record_sha256") or "") != expected_provider:
        raise ValueError("Hybrid crop signed identity binds a different provider record.")

    routing_policy = str(group.attrs.get("source_pixel_routing_policy") or "").strip()
    crop_policy = str(group.attrs.get("crop_policy_id") or "").strip()
    required_arrays = ("instance_key", "source_row_signature", *[item[0] for item in _CONTENT_COMPONENTS])
    missing = [name for name in required_arrays if name not in group]
    if missing:
        raise ValueError(
            "Hybrid crop signed identity lacks required arrays: " + ", ".join(missing)
        )
    arrays = {name: group[name][:] for name in required_arrays if name != "source_row_signature"}
    collection_names = [
        name for name, _dtype, _shape in _COLLECTION_CONTENT_COMPONENTS
    ]
    collection_present = [name in group for name in collection_names]
    if any(collection_present) and not all(collection_present):
        raise ValueError(
            "Hybrid collection signed identity has an incomplete member-index binding."
        )
    if all(collection_present):
        arrays.update({name: group[name][:] for name in collection_names})
    identity = build_hybrid_crop_provider_identity(
        arrays,
        provider_record_sha256=expected_provider,
        routing_policy_id=routing_policy,
        crop_policy_id=crop_policy,
        pixel_contract=group.attrs.get("roi_pixel_contract"),
    )
    row_count = int(np.asarray(arrays["instance_key"]).shape[0])
    validate_row_source_signature_array(
        group["source_row_signature"],
        expected_row_count=row_count,
    )
    observed_signatures = np.asarray(group["source_row_signature"][:], dtype=np.uint8)
    if not np.array_equal(observed_signatures, identity.row_signatures.signatures):
        raise ValueError("Hybrid crop source_row_signature payload is stale.")
    stored_spec = load_row_source_signature_spec(group.attrs)
    if stored_spec.canonical_json != identity.row_signatures.spec.canonical_json:
        raise ValueError("Hybrid crop row-signature specification is stale.")

    expected_attrs = identity.attrs()
    for name in (
        "source_pixel_fingerprint",
        "source_pixel_fingerprint_basis",
        "source_rowset_fingerprint",
        "source_rowset_fingerprint_basis",
        "crop_signature",
        "crop_revision",
    ):
        if group.attrs.get(name) != expected_attrs[name]:
            raise ValueError(f"Hybrid crop signed identity attr {name!r} is stale.")
    return {
        "provider_record_sha256": expected_provider,
        "row_count": row_count,
        "source_row_signature_spec_digest": identity.row_signatures.spec.spec_digest,
        "source_pixel_fingerprint": identity.source_pixel_fingerprint,
        "source_rowset_fingerprint": identity.source_rowset_fingerprint,
        "crop_signature": dict(identity.crop_signature),
        "crop_revision": HYBRID_CROP_REVISION_INITIAL,
    }


def validate_hybrid_provider_strict_crop_geometry(
    provider: Any,
    crop: Any,
    *,
    expected_provider_record_sha256: str,
) -> dict[str, Any]:
    """Resolve one signed hybrid pixel provider against one strict crop-v2.

    This is the shared admission boundary for the split crop publication
    shape: pixels are supplied by a signed hybrid provider while coordinates
    are supplied by an independently sealed geometry-only crop publication.
    Consumers must not implement their own subsets of these checks.
    """

    signed = validate_hybrid_crop_signed_identity(
        provider,
        expected_provider_record_sha256=expected_provider_record_sha256,
    )
    crop_payload = getattr(crop, "manifest", {}).get("payload")
    crop_source = (
        crop_payload.get("source_refined_snapshot")
        if isinstance(crop_payload, Mapping)
        else None
    )
    if not isinstance(crop_source, Mapping) or provider.attrs.get(
        "source_refined_run_id"
    ) != crop_source.get("run_id"):
        raise ValueError(
            "Hybrid pixel provider and crop-v2 bind different refined sources."
        )

    crop_arrays = getattr(crop, "arrays", None)
    if not isinstance(crop_arrays, Mapping):
        raise ValueError("Strict crop-v2 publication lacks its validated arrays.")
    mismatched: list[str] = []
    for path in HYBRID_STRICT_CROP_GEOMETRY_PATHS:
        if (
            path not in provider
            or path not in crop_arrays
            or not np.array_equal(
                np.asarray(provider[path][...]),
                np.asarray(crop_arrays[path][...]),
            )
        ):
            mismatched.append(path)
    if mismatched:
        raise ValueError(
            "Hybrid pixel provider differs from crop-v2 geometry at: "
            + ", ".join(mismatched)
        )
    return {
        **signed,
        "geometry_crop_run": str(getattr(crop, "run_id", "")),
        "geometry_crop_manifest_digest": str(
            getattr(crop, "manifest", {}).get("payload_digest") or ""
        ),
        "ordered_geometry_coverage_exact": True,
        "exact_geometry_paths": list(HYBRID_STRICT_CROP_GEOMETRY_PATHS),
    }


__all__ = [
    "HYBRID_CROP_IDENTITY_SCHEMA_ID",
    "HYBRID_CROP_IDENTITY_SCHEMA_VERSION",
    "HYBRID_CROP_PIXEL_FINGERPRINT_BASIS",
    "HYBRID_CROP_REVISION_INITIAL",
    "HYBRID_CROP_ROWSET_FINGERPRINT_BASIS",
    "HYBRID_CROP_RUN_SCHEMA_ID",
    "HYBRID_CROP_SIGNATURE_SCHEMA_ID",
    "HYBRID_CROP_SIGNATURE_SCHEMA_VERSION",
    "HYBRID_STRICT_CROP_GEOMETRY_PATHS",
    "HybridCropProviderIdentity",
    "build_hybrid_crop_provider_identity",
    "validate_hybrid_crop_signed_identity",
    "validate_hybrid_provider_strict_crop_geometry",
]
