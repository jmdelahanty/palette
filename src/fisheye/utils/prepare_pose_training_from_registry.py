#!/usr/bin/env python3
"""Compatibility entrypoint for pose preflight (alias of keypoint preflight)."""

from __future__ import annotations

from fisheye.utils.prepare_keypoint_training_from_registry import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
