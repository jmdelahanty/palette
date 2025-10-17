"""Segmentation utilities for FishEye."""

from .eye_segmentation import EyeSegmentationConfig, segment_eye_masks
from .eye_segmentation_yolo import segment_eye_masks_yolo

__all__ = ["EyeSegmentationConfig", "segment_eye_masks", "segment_eye_masks_yolo"]
