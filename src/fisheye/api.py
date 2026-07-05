"""Curated public Palette API surface."""

from __future__ import annotations

from fisheye.cli.palette import (
    ApproveRequest,
    ArtifactsRequest,
    CropRequest,
    DetectRequest,
    KeypointsRequest,
    PlanRequest,
    StatusRequest,
    approve,
    artifacts,
    crop,
    detect,
    keypoints,
    plan,
    status,
)
from fisheye.shared.recording import Recording, RecordingRead, open_recording
from fisheye.shared.run_resolution import RunResolution, RunResolutionResult, resolve_run

__all__ = [
    "ApproveRequest",
    "ArtifactsRequest",
    "CropRequest",
    "DetectRequest",
    "KeypointsRequest",
    "PlanRequest",
    "Recording",
    "RecordingRead",
    "RunResolution",
    "RunResolutionResult",
    "StatusRequest",
    "approve",
    "artifacts",
    "crop",
    "detect",
    "keypoints",
    "open_recording",
    "plan",
    "resolve_run",
    "status",
]
