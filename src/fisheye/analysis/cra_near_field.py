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
    """Map the historical source argument onto the explicit legacy input policy."""

    legacy = kwargs.pop("cra_primary_endpoint_component", None)
    if legacy is not None:
        if kwargs.get("quadrant_occupancy_dependency_handle") is not None:
            raise ValueError(
                "cra_primary_endpoint_component is a legacy compatibility input and "
                "cannot be combined with an exact dependency handle."
            )
        requested_compatibility = kwargs.get(
            "legacy_quadrant_occupancy_component_compatibility"
        )
        if requested_compatibility is False:
            raise ValueError(
                "cra_primary_endpoint_component explicitly requires legacy quadrant "
                "occupancy compatibility."
            )
        kwargs.setdefault(
            "quadrant_occupancy_component",
            "latest" if legacy == "object_relative_pre_post_v1" else legacy,
        )
        kwargs["legacy_quadrant_occupancy_component_compatibility"] = True
    return build_chaser_near_field_occupancy_result(*args, **kwargs)
