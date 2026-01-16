"""Segmentation utilities for FishEye."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .eye_segmentation import EyeSegmentationConfig, segment_eye_masks
    from .eye_segmentation_yolo import segment_eye_masks_yolo

__all__ = ["EyeSegmentationConfig", "segment_eye_masks", "segment_eye_masks_yolo"]


def __getattr__(name: str):
    if name in {"EyeSegmentationConfig", "segment_eye_masks"}:
        from . import eye_segmentation as _eye_segmentation

        return getattr(_eye_segmentation, name)
    if name == "segment_eye_masks_yolo":
        from . import eye_segmentation_yolo as _eye_segmentation_yolo

        return _eye_segmentation_yolo.segment_eye_masks_yolo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
