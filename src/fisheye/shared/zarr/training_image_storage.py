"""Physical access-unit and resize contract for sampled training images."""

from __future__ import annotations

from typing import Any, Sequence


SAMPLED_TRAINING_IMAGE_STORAGE_SCHEMA_ID = "palette.sampled_training_image_storage.v1"
SAMPLED_TRAINING_DOWNSAMPLE_TRANSFORM_SCHEMA_ID = (
    "palette.sampled_training_downsample_transform.v1"
)


def sampled_training_image_chunk_shape(
    shape: Sequence[int],
) -> tuple[int, int, int]:
    """Return the complete single-frame chunk required by training images."""

    normalized = tuple(int(value) for value in shape)
    if len(normalized) != 3:
        raise ValueError(
            "Sampled training image arrays must have shape [frame, height, width]."
        )
    _frames, height, width = normalized
    if height <= 0 or width <= 0:
        raise ValueError("Sampled training image height and width must be positive.")
    return (1, height, width)


def compute_letterbox_dimensions(
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
) -> tuple[int, int, int, int, int, int]:
    """Return resized H/W followed by top/bottom/left/right padding."""

    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("Source and target image dimensions must be positive.")
    scale = min(target_h / source_h, target_w / source_w)
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    pad_top = (target_h - resized_h) // 2
    pad_bottom = target_h - resized_h - pad_top
    pad_left = (target_w - resized_w) // 2
    pad_right = target_w - resized_w - pad_left
    return resized_h, resized_w, pad_top, pad_bottom, pad_left, pad_right


def sampled_training_downsample_transform(
    *,
    source_hw: Sequence[int],
    target_hw: Sequence[int],
    method: str,
    preserve_aspect: bool,
) -> dict[str, Any]:
    """Build the exact persisted source-frame to ``images_ds`` transform."""

    source = tuple(int(value) for value in source_hw)
    target = tuple(int(value) for value in target_hw)
    if len(source) != 2 or len(target) != 2:
        raise ValueError("source_hw and target_hw must both be [height, width].")
    source_h, source_w = source
    target_h, target_w = target
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("Source and target image dimensions must be positive.")

    if preserve_aspect:
        (
            resized_h,
            resized_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        ) = compute_letterbox_dimensions(source_h, source_w, target_h, target_w)
        mode = "aspect_preserving_letterbox"
    else:
        resized_h, resized_w = target_h, target_w
        pad_top = pad_bottom = pad_left = pad_right = 0
        mode = "direct_resize"

    return {
        "schema_id": SAMPLED_TRAINING_DOWNSAMPLE_TRANSFORM_SCHEMA_ID,
        "source_shape_hw": [source_h, source_w],
        "stored_shape_hw": [target_h, target_w],
        "resized_shape_hw": [resized_h, resized_w],
        "padding_tblr": [pad_top, pad_bottom, pad_left, pad_right],
        "mode": mode,
        "interpolation": str(method),
        "padding_value_uint8": 0,
    }


__all__ = [
    "SAMPLED_TRAINING_DOWNSAMPLE_TRANSFORM_SCHEMA_ID",
    "SAMPLED_TRAINING_IMAGE_STORAGE_SCHEMA_ID",
    "compute_letterbox_dimensions",
    "sampled_training_downsample_transform",
    "sampled_training_image_chunk_shape",
]
