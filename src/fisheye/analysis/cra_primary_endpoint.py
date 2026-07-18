"""Compatibility imports for :mod:`fisheye.analysis.chaser_quadrant_occupancy`.

New analyses must use the generic module directly. This shim does not provide
compatibility for historical persisted schemas or exported tables.
"""

from fisheye.analysis.chaser_quadrant_occupancy import *  # noqa: F403
from fisheye.analysis.chaser_quadrant_occupancy import (  # noqa: F401
    ChaserQuadrantOccupancyResult as CRAPrimaryEndpointResult,
    ChaserQuadrantPhase as CRAPhaseWindow,
    ChaserQuadrantRole as CRAObjectRole,
    build_chaser_quadrant_occupancy_result,
    resolve_chaser_roles_from_protocol_payload as resolve_object_roles_from_protocol_payload,
    write_chaser_quadrant_occupancy_component as write_cra_primary_endpoint_component,
)
from fisheye.analysis.chaser_profiles import default_goodcopbadcop_source_profile_path


def build_cra_primary_endpoint_result(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    """Build the generic schema using the historical adapter default."""

    kwargs.setdefault("protocol_profile", default_goodcopbadcop_source_profile_path())
    return build_chaser_quadrant_occupancy_result(*args, **kwargs)
