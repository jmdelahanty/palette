"""Construction boundary for disposable detection candidates."""

from __future__ import annotations

from typing import Any


def build_detection_candidate(**kwargs: Any) -> str:
    """Build one disposable candidate without selecting or publishing it.

    The lazy import keeps Ultralytics and its settings initialization out of
    help, planning, and artifact-inspection processes.
    """

    from fisheye.detection.detect_yolo import detect_yolo

    return detect_yolo(**kwargs)


__all__ = ["build_detection_candidate"]
