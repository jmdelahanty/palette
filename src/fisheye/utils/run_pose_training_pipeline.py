#!/usr/bin/env python3
"""Compatibility entrypoint for pose pipeline (alias of keypoint pipeline)."""

from __future__ import annotations

from fisheye.utils.run_keypoint_training_pipeline import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
