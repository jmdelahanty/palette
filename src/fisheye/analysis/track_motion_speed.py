"""Verified frame-aligned speed inputs for downstream behavioral analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from fisheye.analysis.track_kinematics_io import load_track_kinematics_track


SMOOTHED_SPEED_SOURCE = "track_motion.movement/speed/smoothed/mm"


@dataclass(frozen=True)
class VerifiedFrameSpeed:
    """Dense camera-frame speed plus the authority that produced it."""

    values_mm_s: np.ndarray
    source: str
    authority: dict[str, Any]


def load_verified_smoothed_frame_speed(
    root: zarr.Group,
    total_frames: int,
) -> VerifiedFrameSpeed:
    """Load canonical offline smoothed speed onto a dense camera-frame axis.

    Only samples whose source row and transition are both valid are exposed.
    Missing frames and transitions across tracking gaps remain NaN so downstream
    summaries cannot turn a teleport across a gap into physical motion.
    """

    frame_count = int(total_frames)
    if frame_count <= 0:
        raise ValueError("Verified frame speed requires a positive frame extent.")
    track = load_track_kinematics_track(
        root,
        run_name="latest",
        scope="offline",
        track_id=0,
        required_speed_levels=("smoothed",),
    )
    if track.source_acquisition_frame_index is None:
        raise ValueError(
            f"Verified track motion {track.track_path} has no acquisition-frame identity."
        )
    if track.sample_valid is None or track.transition_valid is None:
        raise ValueError(
            f"Verified track motion {track.track_path} lacks sample/transition validity."
        )

    frames = np.asarray(
        track.source_acquisition_frame_index,
        dtype=np.int64,
    ).reshape(-1)
    speed = np.asarray(
        track.speed_mm_by_level["smoothed"],
        dtype=np.float64,
    ).reshape(-1)
    sample_valid = np.asarray(track.sample_valid, dtype=bool).reshape(-1)
    transition_valid = np.asarray(track.transition_valid, dtype=bool).reshape(-1)
    if not (
        frames.shape == speed.shape == sample_valid.shape == transition_valid.shape
    ):
        raise ValueError(
            f"Verified track motion {track.track_path} has inconsistent frame, "
            "speed, and validity lengths."
        )
    if np.any(frames < 0) or np.any(frames >= frame_count):
        raise ValueError(
            f"Verified acquisition-frame identities in {track.track_path} exceed "
            f"the chaser-distance extent [0, {frame_count})."
        )
    if np.unique(frames).shape[0] != frames.shape[0]:
        raise ValueError(
            f"Verified track motion {track.track_path} repeats acquisition-frame identities."
        )

    dense = np.full(frame_count, np.nan, dtype=np.float64)
    usable = sample_valid & transition_valid & np.isfinite(speed)
    dense[frames[usable]] = speed[usable]
    return VerifiedFrameSpeed(
        values_mm_s=dense,
        source=SMOOTHED_SPEED_SOURCE,
        authority=track.authority_record(),
    )
