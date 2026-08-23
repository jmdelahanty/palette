"""Subject-mask placement adapter for signed hybrid crop providers."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .hybrid_crop_provider import resolve_hybrid_crop_source_frame_shape

HYBRID_SUBJECT_MASK_PLACEMENT_NORMALIZATION_SCHEMA = (
    "palette.subject_mask.hybrid_crop_placement_normalization"
)
HYBRID_SUBJECT_MASK_PLACEMENT_NORMALIZATION_OPERATION = (
    "signed_hybrid_float64_to_float32_exact_roundtrip_v1"
)


def normalize_subject_mask_crop_placement(
    crop_group: object,
    *,
    crop_run: str,
    target_rows: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, object] | None]:
    """Return canonical float32 placement for an exact signed-hybrid selection.

    Ordinary canonical crop-v2 placement is already float32 and passes through.
    A float64 source is adapted only when it is a complete hybrid provider,
    every value survives an exact float32 round trip, and placement still
    equals the provider's integer ROI origins and sizes.
    """

    placement = np.asarray(values)
    if placement.dtype == np.dtype(np.float32):
        return np.ascontiguousarray(placement), None
    if placement.dtype != np.dtype(np.float64):
        return placement, None

    source_frame_shape = resolve_hybrid_crop_source_frame_shape(
        crop_group,
        run_id=str(crop_run),
    )
    if source_frame_shape is None:
        return placement, None
    rows = np.asarray(target_rows, dtype=np.int64).reshape(-1)
    if placement.shape != (int(rows.shape[0]), 4):
        raise ValueError(
            "Signed hybrid subject-mask source_crop_xywh must have shape [N,4]."
        )
    if not np.isfinite(placement).all():
        raise ValueError(
            "Signed hybrid subject-mask source_crop_xywh must contain only finite values."
        )

    normalized = np.ascontiguousarray(placement.astype(np.float32))
    if not np.array_equal(normalized.astype(np.float64), placement):
        raise ValueError(
            "Signed hybrid subject-mask source_crop_xywh cannot be represented "
            "exactly as canonical float32 placement."
        )

    roi_coordinates = np.asarray(crop_group["roi_coordinates_full"][rows])
    if (
        roi_coordinates.shape != (rows.shape[0], 2)
        or roi_coordinates.dtype != np.dtype(np.int32)
        or not np.array_equal(normalized[:, :2], roi_coordinates)
    ):
        raise ValueError(
            "Signed hybrid subject-mask placement differs from exact int32 ROI origins."
        )
    has_roi_sizes = "roi_sizes_full" in crop_group
    if has_roi_sizes:
        roi_sizes = np.asarray(crop_group["roi_sizes_full"][rows])
        if (
            roi_sizes.shape != (rows.shape[0], 2)
            or roi_sizes.dtype != np.dtype(np.int32)
            or not np.array_equal(normalized[:, 2:], roi_sizes)
        ):
            raise ValueError(
                "Signed hybrid subject-mask placement differs from exact int32 ROI sizes."
            )

    attrs = crop_group.attrs
    return normalized, {
        "schema_id": HYBRID_SUBJECT_MASK_PLACEMENT_NORMALIZATION_SCHEMA,
        "schema_version": 1,
        "operation": HYBRID_SUBJECT_MASK_PLACEMENT_NORMALIZATION_OPERATION,
        "source_crop_run": str(crop_run),
        "source_dtype": "float64",
        "output_dtype": "float32",
        "target_row_count": int(crop_group["source_crop_xywh"].shape[0]),
        "provider_record_sha256": str(attrs["provider_record_sha256"]),
        "source_frame_shape_hw": [
            int(source_frame_shape[0]),
            int(source_frame_shape[1]),
        ],
        "validation": {
            "provider_record_valid": True,
            "finite": True,
            "float32_roundtrip_exact": True,
            "roi_origins_exact": True,
            "roi_sizes_exact": bool(has_roi_sizes),
        },
    }


__all__ = [
    "HYBRID_SUBJECT_MASK_PLACEMENT_NORMALIZATION_OPERATION",
    "HYBRID_SUBJECT_MASK_PLACEMENT_NORMALIZATION_SCHEMA",
    "normalize_subject_mask_crop_placement",
]
