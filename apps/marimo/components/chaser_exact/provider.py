"""Closed routing adapter for the exact-chaser Marimo provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from fisheye.analysis_workflows.validated_recording_behavior_source import (
    ValidatedRecordingBehaviorSource,
)

from ..registry import InteractiveSpecOption
from ..chaser_exact_body_bearing_contract import (
    ExactBodyBearingContractError,
    option_body_bearing_binding,
)
from ..chaser_exact_body_heading_contract import (
    ExactBodyHeadingContractError,
    option_body_heading_binding,
)
from .body_bearing import build_exact_body_bearing_output
from .body_bearing_distance import build_exact_body_bearing_distance_output
from .body_heading import build_exact_body_heading_output
from .body_alignment import build_exact_body_alignment_output
from .body_alignment_projection import (
    ExactBodyAlignmentProjectionError,
    option_body_alignment_binding,
)
from .bout_response import build_exact_bout_response_output
from .bout_response_projection import (
    ExactBoutResponseProjectionError,
    option_bout_response_binding,
)
from .controller_trial_projection import (
    ExactControllerTrialProjectionError,
    option_controller_trial_binding,
)
from .controller_trials import build_exact_controller_trials_output
from .distance_traces import build_exact_distance_traces_output
from .distance_distributions import build_exact_distance_distributions_output
from .escape_freeze import build_exact_escape_freeze_output
from .escape_freeze_projection import (
    ExactEscapeFreezeProjectionError,
    option_escape_freeze_binding,
)
from .epoch_behavior import build_exact_epoch_behavior_output
from .epoch_behavior_projection import (
    ExactEpochBehaviorProjectionError,
    option_epoch_behavior_binding,
)
from .gaze_tracking import build_exact_gaze_tracking_output
from .gaze_tracking_projection import (
    ExactGazeTrackingProjectionError,
    option_gaze_tracking_binding,
)
from .projection import (
    ExactChaserSelectionIdentity,
    ExactChaserSuccessorProjection,
    _option_bundle,
    build_exact_chaser_selection_identity,
    load_exact_chaser_projection,
)
from .radial_near_field import build_exact_radial_near_field_output
from .same_quadrant import build_exact_same_quadrant_output
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
    load_relative_arrays: bool = True
    load_chaser_appearance: bool = False
    load_keypoint_body_bearing: bool = False
    load_keypoint_body_heading: bool = False
    load_controller_trials: bool = False
    load_generalized_bout_response: bool = False
    load_escape_freeze: bool = False
    load_gaze_tracking: bool = False
    load_epoch_behavior: bool = False
    load_body_alignment_by_distance: bool = False


_ROUTES: Mapping[str, ExactChaserAnalysisRoute] = MappingProxyType(
    {
        "radial_near_field": ExactChaserAnalysisRoute(
            analysis_id="radial_near_field",
            display_parameter_version="exact-radial-near-field-display-v1",
            load_relative=False,
            renderer=build_exact_radial_near_field_output,
        ),
        "distance_distributions": ExactChaserAnalysisRoute(
            analysis_id="distance_distributions",
            display_parameter_version="exact-distance-distributions-display-v1",
            load_relative=False,
            renderer=build_exact_distance_distributions_output,
        ),
        "same_quadrant_occupancy": ExactChaserAnalysisRoute(
            analysis_id="same_quadrant_occupancy",
            display_parameter_version="exact-same-quadrant-display-v1",
            load_relative=False,
            renderer=build_exact_same_quadrant_output,
        ),
        "distance_traces": ExactChaserAnalysisRoute(
            analysis_id="distance_traces",
            display_parameter_version="exact-distance-trace-display-v1",
            load_relative=True,
            renderer=build_exact_distance_traces_output,
        ),
        "body_bearing_polar": ExactChaserAnalysisRoute(
            analysis_id="body_bearing_polar",
            display_parameter_version="exact-body-bearing-polar-display-v1",
            load_relative=True,
            load_keypoint_body_bearing=True,
            renderer=build_exact_body_bearing_output,
        ),
        "body_bearing_distance": ExactChaserAnalysisRoute(
            analysis_id="body_bearing_distance",
            display_parameter_version=("exact-body-bearing-distance-display-v1"),
            load_relative=True,
            load_keypoint_body_bearing=True,
            renderer=build_exact_body_bearing_distance_output,
        ),
        "fish_heading": ExactChaserAnalysisRoute(
            analysis_id="fish_heading",
            display_parameter_version="exact-body-heading-display-v1",
            load_relative=True,
            load_keypoint_body_heading=True,
            renderer=build_exact_body_heading_output,
        ),
        "trajectory_overlays": ExactChaserAnalysisRoute(
            analysis_id="trajectory_overlays",
            display_parameter_version="exact-trajectory-overlay-display-v2",
            load_relative=True,
            load_chaser_appearance=True,
            renderer=build_exact_trajectory_overlays_output,
        ),
        "spatial_occupancy": ExactChaserAnalysisRoute(
            analysis_id="spatial_occupancy",
            display_parameter_version="exact-spatial-occupancy-display-v4",
            load_relative=True,
            load_chaser_appearance=True,
            renderer=build_exact_spatial_occupancy_output,
        ),
        "controller_trials": ExactChaserAnalysisRoute(
            analysis_id="controller_trials",
            display_parameter_version="exact-controller-trial-display-v1",
            load_relative=True,
            renderer=build_exact_controller_trials_output,
            load_controller_trials=True,
        ),
        "generalized_bout_response": ExactChaserAnalysisRoute(
            analysis_id="generalized_bout_response",
            display_parameter_version="exact-generalized-bout-response-display-v1",
            load_relative=True,
            load_relative_arrays=False,
            renderer=build_exact_bout_response_output,
            load_controller_trials=True,
            load_generalized_bout_response=True,
        ),
        "escape_freeze": ExactChaserAnalysisRoute(
            analysis_id="escape_freeze",
            display_parameter_version="exact-escape-freeze-display-v1",
            load_relative=True,
            load_relative_arrays=False,
            renderer=build_exact_escape_freeze_output,
            load_controller_trials=True,
            load_generalized_bout_response=True,
            load_escape_freeze=True,
        ),
        "gaze_tracking": ExactChaserAnalysisRoute(
            analysis_id="gaze_tracking",
            display_parameter_version="exact-gaze-tracking-display-v1",
            load_relative=True,
            load_relative_arrays=False,
            renderer=build_exact_gaze_tracking_output,
            load_gaze_tracking=True,
        ),
        "epoch_behavior": ExactChaserAnalysisRoute(
            analysis_id="epoch_behavior",
            display_parameter_version="exact-semantic-epoch-behavior-display-v1",
            load_relative=False,
            renderer=build_exact_epoch_behavior_output,
            load_epoch_behavior=True,
        ),
        "body_alignment_by_distance": ExactChaserAnalysisRoute(
            analysis_id="body_alignment_by_distance",
            display_parameter_version="exact-body-alignment-distance-display-v1",
            load_relative=True,
            load_relative_arrays=False,
            renderer=build_exact_body_alignment_output,
            load_body_alignment_by_distance=True,
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
        available = {
            value
            for value in ANALYSIS_IDS
            if value
            not in {
                "body_bearing_polar",
                "body_bearing_distance",
                "fish_heading",
                "controller_trials",
                "generalized_bout_response",
                "escape_freeze",
                "gaze_tracking",
                "epoch_behavior",
                "body_alignment_by_distance",
            }
        }
        try:
            option_body_bearing_binding(option)
        except ExactBodyBearingContractError:
            pass
        else:
            available.add("body_bearing_polar")
            available.add("body_bearing_distance")
        try:
            option_body_heading_binding(option)
        except ExactBodyHeadingContractError:
            pass
        else:
            available.add("fish_heading")
        try:
            option_gaze_tracking_binding(option)
        except ExactGazeTrackingProjectionError:
            pass
        else:
            available.add("gaze_tracking")
        try:
            option_epoch_behavior_binding(option)
        except ExactEpochBehaviorProjectionError:
            pass
        else:
            available.add("epoch_behavior")
        try:
            option_body_alignment_binding(option)
        except ExactBodyAlignmentProjectionError:
            pass
        else:
            available.add("body_alignment_by_distance")
        try:
            option_controller_trial_binding(option)
        except ExactControllerTrialProjectionError:
            return tuple(value for value in ANALYSIS_IDS if value in available)
        available.add("controller_trials")
        try:
            option_bout_response_binding(option)
        except ExactBoutResponseProjectionError:
            return tuple(value for value in ANALYSIS_IDS if value in available)
        available.add("generalized_bout_response")
        try:
            option_escape_freeze_binding(option)
        except ExactEscapeFreezeProjectionError:
            return tuple(value for value in ANALYSIS_IDS if value in available)
        available.add("escape_freeze")
        return tuple(value for value in ANALYSIS_IDS if value in available)

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
        projection_receipt_path: str | Path | None = None,
        validated_behavior_source: ValidatedRecordingBehaviorSource | None = None,
    ) -> ExactChaserSelectionIdentity:
        route = self.route(analysis_id)
        behavior_bundle_path = None
        behavior_bundle_sha256 = None
        if validated_behavior_source is not None and analysis_id == "epoch_behavior":
            validated_behavior_source.require_analysis_zarr(zarr_path)
            validated_behavior_source.scientific_child("epoch_behavior")
            projection_receipt_path = (
                validated_behavior_source.exact_projection_receipt_path(
                    explicit_path=projection_receipt_path
                )
            )
            behavior_bundle_path = validated_behavior_source.bundle_path
            behavior_bundle_sha256 = validated_behavior_source.bundle_sha256
        return build_exact_chaser_selection_identity(
            zarr_path,
            option,
            analysis_id=analysis_id,
            display_parameter_version=route.display_parameter_version,
            projection_receipt_path=projection_receipt_path,
            validated_behavior_bundle_path=behavior_bundle_path,
            validated_behavior_bundle_sha256=behavior_bundle_sha256,
        )

    def load_projection(
        self,
        zarr_path: Path | str,
        option: InteractiveSpecOption,
        *,
        analysis_id: str,
        projection_receipt_path: str | Path | None = None,
        validated_behavior_source: ValidatedRecordingBehaviorSource | None = None,
    ) -> ExactChaserSuccessorProjection:
        route = self.route(analysis_id)
        if route.load_keypoint_body_bearing:
            option_body_bearing_binding(option)
        if route.load_keypoint_body_heading:
            option_body_heading_binding(option)
        identity = self.selection_identity(
            zarr_path,
            option,
            analysis_id=analysis_id,
            projection_receipt_path=projection_receipt_path,
            validated_behavior_source=validated_behavior_source,
        )
        return load_exact_chaser_projection(
            zarr_path,
            option,
            selection_identity=identity,
            load_relative=route.load_relative,
            load_relative_arrays=route.load_relative_arrays,
            load_chaser_appearance=route.load_chaser_appearance,
            load_keypoint_body_bearing=route.load_keypoint_body_bearing,
            load_keypoint_body_heading=route.load_keypoint_body_heading,
            load_controller_trials=route.load_controller_trials,
            load_generalized_bout_response=(route.load_generalized_bout_response),
            load_escape_freeze=route.load_escape_freeze,
            load_gaze_tracking=route.load_gaze_tracking,
            load_epoch_behavior=route.load_epoch_behavior,
            load_body_alignment_by_distance=(route.load_body_alignment_by_distance),
        )

    def require_current_projection(
        self,
        projection: ExactChaserSuccessorProjection,
        *,
        zarr_path: Path | str,
        option: InteractiveSpecOption,
        analysis_id: str,
        projection_receipt_path: str | Path | None = None,
        validated_behavior_source: ValidatedRecordingBehaviorSource | None = None,
    ) -> None:
        expected = self.selection_identity(
            zarr_path,
            option,
            analysis_id=analysis_id,
            projection_receipt_path=projection_receipt_path,
            validated_behavior_source=validated_behavior_source,
        )
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
        projection_receipt_path: str | Path | None = None,
        validated_behavior_source: ValidatedRecordingBehaviorSource | None = None,
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
            projection_receipt_path=projection_receipt_path,
            validated_behavior_source=validated_behavior_source,
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
    projection_receipt_path: str | Path | None = None,
    validated_behavior_source: ValidatedRecordingBehaviorSource | None = None,
) -> ExactChaserSuccessorProjection:
    """Compatibility function for the pre-package component API."""

    return EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        zarr_path,
        option,
        analysis_id=analysis_id,
        projection_receipt_path=projection_receipt_path,
        validated_behavior_source=validated_behavior_source,
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
