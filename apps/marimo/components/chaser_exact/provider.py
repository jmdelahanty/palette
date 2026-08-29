"""Closed routing adapter for the exact-chaser Marimo provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..registry import InteractiveSpecOption
from .controller_trial_projection import (
    ExactControllerTrialProjectionError,
    option_controller_trial_binding,
)
from .controller_trials import build_exact_controller_trials_output
from .distance_traces import build_exact_distance_traces_output
from .projection import (
    ExactChaserSelectionIdentity,
    ExactChaserSuccessorProjection,
    _option_bundle,
    build_exact_chaser_selection_identity,
    load_exact_chaser_projection,
)
from .radial_near_field import build_exact_radial_near_field_output
from .spatial_occupancy import build_exact_spatial_occupancy_output
from .trajectory_overlays import build_exact_trajectory_overlays_output

EXACT_CHASER_PROVIDER_ID = "stimulus_chaser_exact_successors"


class ExactChaserProviderError(ValueError):
    """Base class for closed exact-chaser provider routing failures."""


class ExactChaserUnknownAnalysisError(ExactChaserProviderError):
    """An analysis ID is outside the provider's closed route table."""


class ExactChaserAnalysisUnavailableError(ExactChaserProviderError):
    """A known analysis uses a shared shell renderer or lacks a projection."""


class ExactChaserStaleSelectionError(ExactChaserProviderError):
    """A completed projection belongs to an earlier reactive selection."""


Renderer = Callable[[Any, Any, ExactChaserSuccessorProjection], Any]


@dataclass(frozen=True)
class ExactChaserAnalysisRoute:
    analysis_id: str
    display_parameter_version: str
    load_relative: bool
    renderer: Renderer | None
    load_controller_trials: bool = False


_ROUTES: Mapping[str, ExactChaserAnalysisRoute] = MappingProxyType(
    {
        "radial_near_field": ExactChaserAnalysisRoute(
            analysis_id="radial_near_field",
            display_parameter_version="exact-radial-near-field-display-v1",
            load_relative=False,
            renderer=build_exact_radial_near_field_output,
        ),
        "distance_traces": ExactChaserAnalysisRoute(
            analysis_id="distance_traces",
            display_parameter_version="exact-distance-trace-display-v1",
            load_relative=True,
            renderer=build_exact_distance_traces_output,
        ),
        "trajectory_overlays": ExactChaserAnalysisRoute(
            analysis_id="trajectory_overlays",
            display_parameter_version="exact-trajectory-overlay-display-v1",
            load_relative=True,
            renderer=build_exact_trajectory_overlays_output,
        ),
        "spatial_occupancy": ExactChaserAnalysisRoute(
            analysis_id="spatial_occupancy",
            display_parameter_version="exact-spatial-occupancy-display-v1",
            load_relative=False,
            renderer=build_exact_spatial_occupancy_output,
        ),
        "controller_trials": ExactChaserAnalysisRoute(
            analysis_id="controller_trials",
            display_parameter_version="exact-controller-trial-display-v1",
            load_relative=True,
            renderer=build_exact_controller_trials_output,
            load_controller_trials=True,
        ),
        "provenance": ExactChaserAnalysisRoute(
            analysis_id="provenance",
            display_parameter_version="shared-spec-provenance-display-v1",
            load_relative=False,
            renderer=None,
        ),
    }
)

ANALYSIS_IDS = tuple(_ROUTES)


@dataclass(frozen=True)
class ExactChaserProviderAdapter:
    """One stable metadata/load/control/render boundary for exact successors."""

    provider_id: str = EXACT_CHASER_PROVIDER_ID

    def initial_source_label(self, source_labels: Sequence[str]) -> str | None:
        """Default only when discovery exposes one unambiguous exact bundle."""

        labels = tuple(source_labels)
        if len(labels) == 1:
            return labels[0]
        return None

    def route(self, analysis_id: str) -> ExactChaserAnalysisRoute:
        try:
            return _ROUTES[analysis_id]
        except KeyError as exc:
            raise ExactChaserUnknownAnalysisError(
                f"Unsupported exact chaser analysis {analysis_id!r}."
            ) from exc

    def available_analysis_ids(
        self,
        zarr_path: Path | str,
        option: InteractiveSpecOption,
    ) -> tuple[str, ...]:
        """List capability IDs using discovery metadata only."""

        del zarr_path
        _option_bundle(option)
        if option.spec.get("bundle_status") != "exact_selector_ineligible":
            return ()
        available = [value for value in ANALYSIS_IDS if value != "controller_trials"]
        try:
            option_controller_trial_binding(option)
        except ExactControllerTrialProjectionError:
            return tuple(available)
        available.insert(4, "controller_trials")
        return tuple(available)

    def requires_projection(self, analysis_id: str) -> bool:
        """Return whether the provider, rather than the shared shell, renders it."""

        return self.route(analysis_id).renderer is not None

    def build_controls(self, analysis_id: str) -> None:
        """Validate the route and report that current exact views need no controls."""

        self.route(analysis_id)
        return None

    def selection_identity(
        self,
        zarr_path: Path | str,
        option: InteractiveSpecOption,
        *,
        analysis_id: str,
    ) -> ExactChaserSelectionIdentity:
        route = self.route(analysis_id)
        return build_exact_chaser_selection_identity(
            zarr_path,
            option,
            analysis_id=analysis_id,
            display_parameter_version=route.display_parameter_version,
        )

    def load_projection(
        self,
        zarr_path: Path | str,
        option: InteractiveSpecOption,
        *,
        analysis_id: str,
    ) -> ExactChaserSuccessorProjection:
        route = self.route(analysis_id)
        identity = self.selection_identity(zarr_path, option, analysis_id=analysis_id)
        return load_exact_chaser_projection(
            zarr_path,
            option,
            selection_identity=identity,
            load_relative=route.load_relative,
            load_controller_trials=route.load_controller_trials,
        )

    def require_current_projection(
        self,
        projection: ExactChaserSuccessorProjection,
        *,
        zarr_path: Path | str,
        option: InteractiveSpecOption,
        analysis_id: str,
    ) -> None:
        expected = self.selection_identity(zarr_path, option, analysis_id=analysis_id)
        if (
            projection.analysis_id != analysis_id
            or projection.selection_identity != expected
        ):
            raise ExactChaserStaleSelectionError(
                "Exact chaser projection belongs to an earlier archive, source, "
                "analysis, renderer, manifest, or display-parameter selection."
            )

    def render(
        self,
        mo: Any,
        go: Any,
        projection: ExactChaserSuccessorProjection,
        *,
        zarr_path: Path | str,
        option: InteractiveSpecOption,
        analysis_id: str,
    ) -> Any:
        route = self.route(analysis_id)
        if route.renderer is None:
            raise ExactChaserAnalysisUnavailableError(
                f"Exact chaser analysis {analysis_id!r} is rendered by the shared "
                "recording-explorer shell."
            )
        self.require_current_projection(
            projection,
            zarr_path=zarr_path,
            option=option,
            analysis_id=analysis_id,
        )
        return route.renderer(mo, go, projection)


EXACT_CHASER_PROVIDER_ADAPTER = ExactChaserProviderAdapter()


def available_exact_chaser_successor_analysis_ids(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
) -> tuple[str, ...]:
    """Compatibility function for the pre-package component API."""

    return EXACT_CHASER_PROVIDER_ADAPTER.available_analysis_ids(zarr_path, option)


def load_exact_chaser_successor_projection(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    analysis_id: str,
) -> ExactChaserSuccessorProjection:
    """Compatibility function for the pre-package component API."""

    return EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        zarr_path, option, analysis_id=analysis_id
    )


__all__ = [
    "ANALYSIS_IDS",
    "EXACT_CHASER_PROVIDER_ADAPTER",
    "EXACT_CHASER_PROVIDER_ID",
    "ExactChaserAnalysisRoute",
    "ExactChaserAnalysisUnavailableError",
    "ExactChaserProviderAdapter",
    "ExactChaserProviderError",
    "ExactChaserStaleSelectionError",
    "ExactChaserUnknownAnalysisError",
    "available_exact_chaser_successor_analysis_ids",
    "load_exact_chaser_successor_projection",
]
