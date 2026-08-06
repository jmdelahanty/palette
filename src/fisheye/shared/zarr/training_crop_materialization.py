"""Strict source binding for self-contained training crop pixels.

The geometry-only crop contract remains the authority for observation identity
and coordinate semantics.  This module binds a copied ``roi_images`` surface
to that authority without pretending the training archive owns a second
crop-v2 authority.  Pixel payloads are deliberately not re-hashed when a
consumer opens a multi-gigabyte training store; publication is responsible for
physical-copy checksums, while this binding revalidates the small identity
columns that join inference output back to the source observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct


TRAINING_CROP_MATERIALIZATION_SCHEMA_ID = (
    "palette.training_crop_materialization.v1"
)
TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE = (
    "training_crop_materialization_binding"
)
TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID = (
    "palette.training_crop_materialization_binding"
)
TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION = 1
TRAINING_CROP_MATERIALIZATION_PROVIDERS = (
    "source_video_pynvvc_luma",
    "verified_flat_roi_cache",
)

_IDENTITY_ARRAYS = (
    *CROP_GEOMETRY_SCHEMA_V1.binding_paths,
    "source_crop_row_ids",
    "source_frame_indices",
)
_OPTIONAL_CLIPPED_IDENTITY_ARRAYS = (
    "source_clip_local_frame_indices",
    "source_clip_indices",
)


class TrainingCropMaterializationError(RuntimeError):
    """Raised when training pixels are not exactly bound to crop geometry."""


@dataclass(frozen=True)
class BoundTrainingCropMaterialization:
    archive_path: Path
    run_id: str
    run_path: str
    run_group: zarr.Group
    roi_images: zarr.Array
    instance_key: zarr.Array
    source_frame_indices: zarr.Array
    binding: Mapping[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.roi_images.shape[0])

    @property
    def roi_shape(self) -> tuple[int, int]:
        return (int(self.roi_images.shape[1]), int(self.roi_images.shape[2]))


def _array_declaration(array: Any) -> dict[str, Any]:
    return {
        "dtype": np.dtype(array.dtype).str,
        "shape": [int(value) for value in array.shape],
    }


def _identity_digest(array: Any) -> str:
    return sha256_array(np.asarray(array[:]))


def build_training_crop_materialization_binding(
    run: zarr.Group,
) -> dict[str, Any]:
    """Build the exact persisted binding after all pixels have been written."""

    provider = str(run.attrs.get("training_materialization_provider") or "")
    if provider not in TRAINING_CROP_MATERIALIZATION_PROVIDERS:
        raise TrainingCropMaterializationError(
            f"Unsupported training materialization provider: {provider!r}."
        )
    if run.attrs.get("training_materialization_schema") != (
        TRAINING_CROP_MATERIALIZATION_SCHEMA_ID
    ):
        raise TrainingCropMaterializationError(
            "Training crop materialization schema identity is missing or wrong."
        )
    missing = [name for name in (*_IDENTITY_ARRAYS, "roi_images") if name not in run]
    if missing:
        raise TrainingCropMaterializationError(
            f"Training crop materialization is missing required arrays: {missing}."
        )

    roi_images = run["roi_images"]
    if np.dtype(roi_images.dtype) != np.dtype(np.uint8) or len(roi_images.shape) != 3:
        raise TrainingCropMaterializationError(
            "roi_images must be a rank-3 uint8 array."
        )
    row_count = int(roi_images.shape[0])
    declarations: dict[str, dict[str, Any]] = {}
    identity_sha256: dict[str, str] = {}
    for name in (*_IDENTITY_ARRAYS, *_OPTIONAL_CLIPPED_IDENTITY_ARRAYS):
        if name not in run:
            continue
        array = run[name]
        if int(array.shape[0]) != row_count and name != "frame_row_offsets":
            raise TrainingCropMaterializationError(
                f"{name} first-axis length differs from roi_images."
            )
        declarations[name] = _array_declaration(array)
        identity_sha256[name] = _identity_digest(array)
    declarations["roi_images"] = _array_declaration(roi_images)

    source_binding = run.attrs.get("source_crop_manifest_binding")
    if not isinstance(source_binding, Mapping):
        raise TrainingCropMaterializationError(
            "Training crop materialization lacks source_crop_manifest_binding."
        )
    provider_evidence: dict[str, Any]
    if provider == "verified_flat_roi_cache":
        provider_evidence = {
            "manifest_path": run.attrs.get("source_roi_cache_manifest"),
            "manifest_sha256": run.attrs.get("source_roi_cache_manifest_sha256"),
            "payload_sha256": run.attrs.get("source_roi_cache_payload_sha256"),
            "verified": run.attrs.get("source_roi_cache_verified"),
            "runtime_dependency": run.attrs.get("source_roi_cache_independence"),
        }
    else:
        provider_evidence = {
            "source_video_path": run.attrs.get("source_video_path"),
            "decode_backend": run.attrs.get("decode_backend"),
            "pixel_contract_name": run.attrs.get("roi_pixel_contract_name"),
        }

    payload = {
        "schema_id": TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID,
        "schema_version": TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION,
        "provider": provider,
        "stage_selector_eligible": False,
        "source": {
            "archive_path": str(run.attrs.get("source_crop_archive_path") or ""),
            "crop_run": str(run.attrs.get("source_crop_run") or ""),
            "crop_path": str(run.attrs.get("source_crop_path") or ""),
            "crop_manifest": dict(source_binding),
        },
        "dimensions": {
            "row_count": row_count,
            "roi_shape": [int(roi_images.shape[1]), int(roi_images.shape[2])],
            "source_height": int(run.attrs.get("height") or 0),
            "source_width": int(run.attrs.get("width") or 0),
        },
        "array_declarations": declarations,
        "identity_array_sha256": identity_sha256,
        "provider_evidence": provider_evidence,
        "pixel_payload_validation": (
            "physical_publication_checksum_plus_provider_evidence_v1"
        ),
    }
    return {
        "payload": payload,
        "payload_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
    }


def _parse_binding(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "payload",
        "payload_digest_algorithm",
        "payload_digest",
    }:
        raise TrainingCropMaterializationError(
            "Training crop binding envelope has an unexpected field set."
        )
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise TrainingCropMaterializationError(
            "Training crop binding payload must be an object."
        )
    if value.get("payload_digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        raise TrainingCropMaterializationError(
            "Training crop binding digest algorithm mismatch."
        )
    if value.get("payload_digest") != canonical_json_sha256(payload):
        raise TrainingCropMaterializationError(
            "Training crop binding payload digest mismatch."
        )
    return value


def bind_training_crop_materialization(
    archive_path: str | Path,
    *,
    run_id: str,
    require_consolidated: bool = True,
) -> BoundTrainingCropMaterialization:
    """Open one candidate input for keypoint or subject-mask inference."""

    path = Path(archive_path).expanduser().resolve()
    root = open_zarr_group_direct(path, mode="r")
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise TrainingCropMaterializationError(
            "Training crop materialization requires a training-purpose Zarr."
        )
    parent = root.get("crop_runs")
    if parent is None or str(run_id) not in parent:
        raise TrainingCropMaterializationError(
            f"Training crop run not found: {run_id!r}."
        )
    run = parent[str(run_id)]
    if run.attrs.get("status") != "completed":
        raise TrainingCropMaterializationError(
            "Training crop materialization is not completed."
        )
    if run.attrs.get("stage_selector_eligible") is not False:
        raise TrainingCropMaterializationError(
            "Training crop materialization must remain selector-ineligible."
        )
    persisted = _parse_binding(
        run.attrs.get(TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE)
    )
    recomputed = build_training_crop_materialization_binding(run)
    if dict(persisted) != recomputed:
        raise TrainingCropMaterializationError(
            "Training crop materialization binding differs from decoded arrays or attrs."
        )
    if require_consolidated:
        try:
            consolidated_root = zarr.open_group(
                str(path), mode="r", zarr_format=3, use_consolidated=True
            )
            consolidated_run = consolidated_root[f"crop_runs/{run_id}"]
        except Exception as exc:
            raise TrainingCropMaterializationError(
                f"Published training crop lacks readable consolidated metadata: {exc}"
            ) from exc
        if (
            consolidated_run.attrs.get(
                TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE
            )
            != persisted
        ):
            raise TrainingCropMaterializationError(
                "Direct and consolidated training crop bindings differ."
            )
        for name, expected in persisted["payload"]["array_declarations"].items():
            observed = _array_declaration(consolidated_run[name])
            if observed != expected:
                raise TrainingCropMaterializationError(
                    f"Direct and consolidated declaration differ for {name}."
                )
    payload = persisted["payload"]
    if payload.get("stage_selector_eligible") is not False:
        raise TrainingCropMaterializationError(
            "Training crop binding is not selector-ineligible."
        )
    keys = np.asarray(run["instance_key"][:], dtype=np.uint64).reshape(-1)
    if int(np.unique(keys).shape[0]) != int(keys.shape[0]):
        raise TrainingCropMaterializationError(
            "Training crop instance_key values are not unique."
        )
    return BoundTrainingCropMaterialization(
        archive_path=path,
        run_id=str(run_id),
        run_path=f"crop_runs/{run_id}",
        run_group=run,
        roi_images=run["roi_images"],
        instance_key=run["instance_key"],
        source_frame_indices=run["source_frame_indices"],
        binding=persisted,
    )


__all__ = [
    "TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE",
    "TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID",
    "TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION",
    "TRAINING_CROP_MATERIALIZATION_PROVIDERS",
    "TRAINING_CROP_MATERIALIZATION_SCHEMA_ID",
    "BoundTrainingCropMaterialization",
    "TrainingCropMaterializationError",
    "bind_training_crop_materialization",
    "build_training_crop_materialization_binding",
]
