"""Shared ROI pixel-representation contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

ROI_PIXEL_CONTRACT_SCHEMA = "palette_roi_pixel_contract_v1"
ROI_IMAGE_REPRESENTATION = "uint8_grayscale_roi_v1"


def roi_pixel_contract(
    *,
    name: str,
    color_conversion: str,
    production_status: str,
    source_frame_representation: str | None = None,
) -> dict[str, Any]:
    """Build the canonical metadata contract for model-facing ROI pixels."""

    payload: dict[str, Any] = {
        "schema": ROI_PIXEL_CONTRACT_SCHEMA,
        "name": str(name),
        "image_representation": ROI_IMAGE_REPRESENTATION,
        "shape": "[roi, roi_height, roi_width]",
        "dtype": "uint8",
        "order": "C",
        "row_order": "crop_runs/<run> row order",
        "coordinates": "crop_runs/<run>/roi_coordinates_full top-left coordinates",
        "padding": "zero outside source-frame bounds",
        "color_conversion": str(color_conversion),
        "production_status": str(production_status),
    }
    if source_frame_representation is not None:
        payload["source_frame_representation"] = str(source_frame_representation)
    return payload


def flat_cache_pixel_contract_for_backend(decode_backend: str) -> dict[str, Any]:
    """Return the pixel contract written into flat ROI cache manifests."""

    if decode_backend == "pynvvc_luma":
        return roi_pixel_contract(
            name="nv12_luma_plane_uint8",
            color_conversion="raw NV12 Y/luma plane crop; no RGB reconstruction",
            production_status="provisional_until_crop_pixel_parity_passes",
            source_frame_representation="PyNvVideoCodec decoded NV12 surface",
        )
    if decode_backend == "read_slice":
        return roi_pixel_contract(
            name="crop_image_source_uint8_grayscale",
            color_conversion=(
                "delegates to CropImageSource; materialized crops are read as stored, "
                "geometry-only live reads use the configured live reader grayscale conversion"
            ),
            production_status="canonical_reader_path",
        )
    return roi_pixel_contract(
        name=f"{decode_backend}_uint8_grayscale",
        color_conversion="backend-specific grayscale conversion",
        production_status="unknown",
    )


def crop_run_pixel_contract(
    *,
    crop_storage_mode: str,
    video_source_type: str | None,
    acceleration: str | None,
) -> dict[str, Any]:
    """Return the best-known contract for pixels stored or implied by a crop run."""

    storage = str(crop_storage_mode or "").strip().lower()
    source = str(video_source_type or "").strip().lower()
    accel = str(acceleration or "").strip().lower()

    if storage == "geometry_only":
        return roi_pixel_contract(
            name="geometry_only_deferred_uint8_grayscale",
            color_conversion=(
                "no ROI pixels materialized in crop run; downstream reader/cache records "
                "the effective live or cache conversion"
            ),
            production_status="logical_contract_only",
        )

    if source == "external" and accel == "gpu":
        return roi_pixel_contract(
            name="decord_rgb_channel_mean_uint8",
            color_conversion="Decord GPU frame decoded to RGB-like channels, then channel mean to uint8",
            production_status="historical_materialized_gpu_contract",
            source_frame_representation="Decord GPU frame tensor [H,W,C]",
        )

    if source == "external" and accel == "cpu":
        return roi_pixel_contract(
            name="opencv_bgr2gray_uint8",
            color_conversion="OpenCV VideoCapture BGR frame converted with cv2.COLOR_BGR2GRAY",
            production_status="historical_materialized_cpu_contract",
            source_frame_representation="OpenCV BGR frame",
        )

    if source == "zarr":
        return roi_pixel_contract(
            name="raw_video_images_full_to_uint8_grayscale",
            color_conversion=(
                "raw_video/images_full cropped through the zarr crop path; grayscale source frames "
                "are preserved as uint8 and RGB-like source frames are converted to uint8 grayscale"
            ),
            production_status="canonical_zarr_crop_contract",
            source_frame_representation="raw_video/images_full",
        )

    return roi_pixel_contract(
        name="materialized_legacy_uint8_grayscale",
        color_conversion="legacy materialized ROI pixels read as stored",
        production_status="legacy_or_unknown_materialized_contract",
    )


def crop_image_source_live_pixel_contract(
    *,
    frame_source_kind: str,
    roi_live_acceleration_effective: str | None,
) -> dict[str, Any]:
    """Return the contract implied by a live ``CropImageSource`` read path."""

    frame_source = str(frame_source_kind or "").strip()
    accel = str(roi_live_acceleration_effective or "").strip().lower()
    if frame_source == "source_video_path" and accel == "gpu":
        return roi_pixel_contract(
            name="decord_gpu_channel_mean_uint8",
            color_conversion="Decord GPU external-video read, then channel mean to uint8",
            production_status="live_reader_contract",
            source_frame_representation="Decord GPU frame tensor [H,W,C]",
        )
    if frame_source == "source_video_path":
        return roi_pixel_contract(
            name="external_video_live_cpu_grayscale_uint8",
            color_conversion=(
                "external-video CPU live read; Decord RGB frames use weighted RGB grayscale, "
                "OpenCV fallback uses cv2.COLOR_BGR2GRAY"
            ),
            production_status="live_reader_contract_backend_resolved_at_read",
        )
    if frame_source == "raw_video/images_full":
        return roi_pixel_contract(
            name="raw_video_images_full_live_grayscale_uint8",
            color_conversion="raw_video/images_full live crop converted to uint8 grayscale when needed",
            production_status="live_reader_contract",
            source_frame_representation="raw_video/images_full",
        )
    if frame_source == "roi_images":
        return roi_pixel_contract(
            name="materialized_roi_images_uint8_grayscale",
            color_conversion="materialized crop_runs/<run>/roi_images read as stored",
            production_status="materialized_reader_contract",
        )
    return roi_pixel_contract(
        name="crop_image_source_uint8_grayscale",
        color_conversion="CropImageSource returned uint8 grayscale ROI pixels",
        production_status="reader_contract",
    )


def normalize_pixel_contract(value: Any) -> dict[str, Any] | None:
    """Normalize a stored pixel-contract attr from dict or JSON string form."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        payload = dict(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, Mapping):
            return None
        payload = dict(parsed)
    else:
        return None

    if not payload.get("schema"):
        payload["schema"] = ROI_PIXEL_CONTRACT_SCHEMA
    if not payload.get("image_representation"):
        payload["image_representation"] = ROI_IMAGE_REPRESENTATION
    return payload

