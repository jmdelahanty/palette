"""Segmentation utilities for FishEye."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .subject_segmentation import SubjectSegmentationConfig, segment_subject_masks
    from .swim_bladder_segmentation import (
        SwimBladderSegmentationConfig,
        segment_swim_bladder_masks,
    )

__all__ = [
    "SubjectSegmentationConfig",
    "segment_subject_masks",
    "SwimBladderSegmentationConfig",
    "segment_swim_bladder_masks",
]


def __getattr__(name: str):
    if name in {"SubjectSegmentationConfig", "segment_subject_masks"}:
        from . import subject_segmentation as _subject_segmentation

        return getattr(_subject_segmentation, name)
    if name in {"SwimBladderSegmentationConfig", "segment_swim_bladder_masks"}:
        from . import swim_bladder_segmentation as _swim_bladder_segmentation

        return getattr(_swim_bladder_segmentation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
