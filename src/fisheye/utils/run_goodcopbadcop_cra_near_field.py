"""Compatibility entry point for the generic chaser near-field occupancy runner."""

# ruff: noqa: F401,F403

from fisheye.utils.run_chaser_near_field_occupancy import *  # noqa: F403
from fisheye.utils.run_chaser_near_field_occupancy import (
    _explicit_zarr_targets,
    _filesystem_targets,
    _query_targets,
)

if __name__ == "__main__":
    raise SystemExit(main())  # noqa: F405
