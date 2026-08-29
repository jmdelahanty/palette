"""Compatibility facade for the modular exact-chaser explorer provider.

New code should import the provider adapter or focused modules below
``apps.marimo.components.chaser_exact``. This facade preserves the supported
pre-package imports while callers migrate without a flag day.
"""

from .chaser_exact.distance_traces import (
    _trace_display_projection,
    build_exact_distance_traces_output,
)
from .chaser_exact.controller_trials import build_exact_controller_trials_output
from .chaser_exact.projection import (
    ExactChaserSelectionIdentity,
    ExactChaserSuccessorProjection,
    RelativeFrameProjection,
    _verify_bundle_children,
)
from .chaser_exact.provider import (
    ANALYSIS_IDS,
    EXACT_CHASER_PROVIDER_ADAPTER,
    ExactChaserProviderAdapter,
    available_exact_chaser_successor_analysis_ids,
    load_exact_chaser_successor_projection,
)
from .chaser_exact.radial_near_field import build_exact_radial_near_field_output
from .chaser_exact.spatial_occupancy import build_exact_spatial_occupancy_output
from .chaser_exact.trajectory_overlays import (
    _trajectory_display_indices,
    build_exact_trajectory_overlays_output,
)

__all__ = [
    "ANALYSIS_IDS",
    "EXACT_CHASER_PROVIDER_ADAPTER",
    "ExactChaserProviderAdapter",
    "ExactChaserSelectionIdentity",
    "ExactChaserSuccessorProjection",
    "RelativeFrameProjection",
    "available_exact_chaser_successor_analysis_ids",
    "build_exact_distance_traces_output",
    "build_exact_controller_trials_output",
    "build_exact_radial_near_field_output",
    "build_exact_spatial_occupancy_output",
    "build_exact_trajectory_overlays_output",
    "load_exact_chaser_successor_projection",
]
