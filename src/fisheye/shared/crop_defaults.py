"""Shared workflow defaults for zebrafish crop geometry.

These values are workflow defaults, not schema invariants.  Persisted crop
artifacts must continue to declare their exact per-run geometry policy.
"""

from __future__ import annotations


DEFAULT_ZEBRAFISH_CROP_SIZE_PX = 384
DEFAULT_ZEBRAFISH_CROP_SIZE_HW = (
    DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
)
DEFAULT_ZEBRAFISH_CROP_SIZE_WH = (
    DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
)
DEFAULT_ZEBRAFISH_CROP_PURPOSE = "zebrafish_pose_subject_mask_input"


__all__ = [
    "DEFAULT_ZEBRAFISH_CROP_PURPOSE",
    "DEFAULT_ZEBRAFISH_CROP_SIZE_HW",
    "DEFAULT_ZEBRAFISH_CROP_SIZE_PX",
    "DEFAULT_ZEBRAFISH_CROP_SIZE_WH",
]
