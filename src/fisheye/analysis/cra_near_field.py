"""Compatibility imports for :mod:`fisheye.analysis.chaser_near_field_occupancy`."""

from fisheye.analysis.chaser_near_field_occupancy import *  # noqa: F403
from fisheye.analysis.chaser_near_field_occupancy import (  # noqa: F401
    ChaserNearFieldIdentity as CRANearFieldObject,
    ChaserNearFieldOccupancyResult as CRANearFieldResult,
    ChaserNearFieldPhase as CRANearFieldPhase,
    _available_annulus_area_mm2,
    _rectangle_annulus_area_mm2,
    build_chaser_near_field_occupancy_result,
    write_chaser_near_field_occupancy_component as write_cra_near_field_component,
)


def build_cra_near_field_result(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    """Map the historical source argument onto the generic component input."""

    legacy = kwargs.pop("cra_primary_endpoint_component", None)
    if legacy is not None:
        kwargs.setdefault(
            "quadrant_occupancy_component",
            "latest" if legacy == "object_relative_pre_post_v1" else legacy,
        )
    return build_chaser_near_field_occupancy_result(*args, **kwargs)
