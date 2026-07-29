"""Stable row identity for maintained acquisition crop-video snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.instance_keys import (
    instance_key_attrs,
    mint_detection_instance_keys,
)
from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    normalize_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.row_source_signature import (
    RowSourceSignatureBatch,
    build_row_source_signatures,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


ACQUISITION_CROP_IDENTITY_SCHEMA_ID = "palette.analysis_acquisition_crop_identity"
ACQUISITION_CROP_IDENTITY_SCHEMA_VERSION = 1
ACQUISITION_CROP_SIGNATURE_SCHEMA_ID = (
    "palette.analysis_acquisition_crop_run.signature"
)
ACQUISITION_CROP_SIGNATURE_SCHEMA_VERSION = 1
ACQUISITION_CROP_SIGNATURE_STAGE = "crop"
ACQUISITION_CROP_REVISION_INITIAL = 1
ACQUISITION_CROP_INSTANCE_KEY_POLICY = "minted_at_acquisition_detection_origin"
ACQUISITION_CROP_PIXEL_FINGERPRINT_BASIS = (
    "sha256_canonical_source_descriptor_v1_not_video_content_hash"
)
ACQUISITION_CROP_ROWSET_FINGERPRINT_BASIS = (
    "sha256_instance_keys_and_row_source_signatures_v1"
)

_CONTENT_COMPONENTS: tuple[tuple[str, Any, tuple[int, ...]], ...] = (
    ("bbox_norm_coords", np.float32, (4,)),
    ("crop_state_codes", np.int8, ()),
    ("frame_indices", np.int64, ()),
    ("roi_coordinates_full", np.int32, (2,)),
    ("roi_sizes_full", np.int32, (2,)),
    ("source_crop_local_frame_ids", np.int64, ()),
    ("source_crop_meta_row_indices", np.int64, ()),
    ("source_crop_video_frame_indices", np.int64, ()),
    ("source_pixel_kind_codes", np.int8, ()),
)


@dataclass(frozen=True)
class AcquisitionCropIdentity:
    """Complete stable identity payload written by the acquisition producer."""

    instance_keys: np.ndarray
    row_signatures: RowSourceSignatureBatch
    crop_signature: Mapping[str, Any]
    source_pixel_fingerprint: str
    source_rowset_fingerprint: str
    recording_identity: str

    def attrs(self) -> dict[str, Any]:
        return {
            **instance_key_attrs(self.recording_identity),
            "instance_key_status": "present",
            "instance_key_policy": ACQUISITION_CROP_INSTANCE_KEY_POLICY,
            **self.row_signatures.spec.to_attrs(),
            "source_pixel_fingerprint": self.source_pixel_fingerprint,
            "source_pixel_fingerprint_basis": (
                ACQUISITION_CROP_PIXEL_FINGERPRINT_BASIS
            ),
            "source_rowset_fingerprint": self.source_rowset_fingerprint,
            "source_rowset_fingerprint_basis": (
                ACQUISITION_CROP_ROWSET_FINGERPRINT_BASIS
            ),
            "crop_signature": dict(self.crop_signature),
            "crop_revision": ACQUISITION_CROP_REVISION_INITIAL,
        }


def acquisition_crop_video_descriptor(path: str | Path) -> dict[str, Any]:
    """Describe the immutable source file without hashing a multi-gigabyte video."""

    source = Path(path).expanduser().resolve()
    stat = source.stat()
    if not source.is_file():
        raise ValueError(f"Acquisition crop video is not a file: {source}")
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _content_arrays(
    arrays: Mapping[str, Any],
    *,
    row_count: int,
) -> dict[str, np.ndarray]:
    content: dict[str, np.ndarray] = {}
    for name, dtype, trailing_shape in _CONTENT_COMPONENTS:
        if name not in arrays:
            raise ValueError(f"Acquisition crop identity is missing {name!r}.")
        values = np.asarray(arrays[name], dtype=dtype)
        expected = (row_count, *trailing_shape)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Acquisition crop identity {name!r} must have shape "
                f"{expected}, got {values.shape}."
            )
        content[name] = values
    return content


def _rowset_fingerprint(
    instance_keys: np.ndarray,
    source_row_signatures: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(instance_keys, dtype="<u8").tobytes(order="C"))
    digest.update(
        np.asarray(source_row_signatures, dtype=np.uint8).tobytes(order="C")
    )
    return digest.hexdigest()


def build_acquisition_crop_identity(
    arrays: Mapping[str, Any],
    *,
    recording_identity: str,
    source_video_descriptor: Mapping[str, Any],
    source_crop_meta_path: str | Path,
    source_width: int,
    source_height: int,
    pixel_contract: Mapping[str, Any] | None = None,
) -> AcquisitionCropIdentity:
    """Mint stable observation keys and exact reuse signatures for one crop run."""

    normalized_recording = str(recording_identity).strip()
    if not normalized_recording:
        raise ValueError("Acquisition crop recording_identity cannot be empty.")
    contract = normalize_pixel_contract(
        pixel_contract or orange_mono_pynvvc_luma_pixel_contract()
    )
    expected_contract = orange_mono_pynvvc_luma_pixel_contract()
    if contract != expected_contract:
        raise ValueError(
            "Acquisition crop identity requires the exact Orange PyNvVC luma "
            "pixel contract."
        )
    frames = np.asarray(arrays.get("frame_indices"), dtype=np.int64).reshape(-1)
    boxes = np.asarray(arrays.get("bbox_norm_coords"), dtype=np.float32).reshape(
        -1, 4
    )
    if boxes.shape[0] != frames.shape[0]:
        raise ValueError("Acquisition crop bbox/frame row counts differ.")
    content = _content_arrays(arrays, row_count=int(frames.shape[0]))
    keys = mint_detection_instance_keys(
        recording_identity=normalized_recording,
        frame_indices=frames,
        bbox_norm_coords=boxes,
    )
    descriptor = {
        "schema_id": ACQUISITION_CROP_IDENTITY_SCHEMA_ID,
        "schema_version": ACQUISITION_CROP_IDENTITY_SCHEMA_VERSION,
        "source_pixels": SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
        "recording_identity": normalized_recording,
        "source_crop_video": dict(source_video_descriptor),
        "source_crop_meta_path": str(Path(source_crop_meta_path).expanduser().resolve()),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "pixel_contract": contract,
    }
    if int(source_width) <= 0 or int(source_height) <= 0:
        raise ValueError("Acquisition crop source dimensions must be positive.")
    pixel_fingerprint = canonical_json_sha256(descriptor)
    signatures = build_row_source_signatures(
        stage=ACQUISITION_CROP_SIGNATURE_STAGE,
        instance_keys=keys,
        content_components=content,
        compatibility_context={
            **descriptor,
            "source_pixel_fingerprint": pixel_fingerprint,
        },
    )
    rowset_fingerprint = _rowset_fingerprint(keys, signatures.signatures)
    crop_signature = {
        "schema_id": ACQUISITION_CROP_SIGNATURE_SCHEMA_ID,
        "schema_version": ACQUISITION_CROP_SIGNATURE_SCHEMA_VERSION,
        "recording_identity": normalized_recording,
        "source_pixels": SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
        "source_pixel_fingerprint": pixel_fingerprint,
        "source_row_signature_spec_digest": signatures.spec.spec_digest,
        "source_rowset_fingerprint": rowset_fingerprint,
    }
    canonical_json_sha256(crop_signature)
    return AcquisitionCropIdentity(
        instance_keys=keys,
        row_signatures=signatures,
        crop_signature=crop_signature,
        source_pixel_fingerprint=pixel_fingerprint,
        source_rowset_fingerprint=rowset_fingerprint,
        recording_identity=normalized_recording,
    )


__all__ = [
    "ACQUISITION_CROP_IDENTITY_SCHEMA_ID",
    "ACQUISITION_CROP_IDENTITY_SCHEMA_VERSION",
    "ACQUISITION_CROP_INSTANCE_KEY_POLICY",
    "ACQUISITION_CROP_PIXEL_FINGERPRINT_BASIS",
    "ACQUISITION_CROP_REVISION_INITIAL",
    "ACQUISITION_CROP_ROWSET_FINGERPRINT_BASIS",
    "ACQUISITION_CROP_SIGNATURE_SCHEMA_ID",
    "ACQUISITION_CROP_SIGNATURE_SCHEMA_VERSION",
    "AcquisitionCropIdentity",
    "acquisition_crop_video_descriptor",
    "build_acquisition_crop_identity",
]
