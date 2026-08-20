"""Typed spatial authority for canonical stimulus-response calculations.

The historical stimulus-response path searched several calibration groups and
then applied a direction-neutral matrix helper.  Canonical stimulus imports
already publish an exact arena -> selected canvas -> source camera chain and a
selected source-camera physical scale.  This module binds those authorities to
the physical authority used by the selected track-motion run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.directed_transform_chain import (
    apply_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    apply_bound_directed_transform_v2,
)
from fisheye.shared.stimulus_coordinate_contract import (
    BoundStimulusCoordinateEvidence,
    StimulusCoordinateContractError,
    canonical_mapping_digest,
    load_bound_stimulus_coordinate_evidence,
)
from fisheye.shared.stimulus_physical_coordinate import (
    BoundStimulusPhysicalCoordinateAuthority,
    StimulusPhysicalCoordinateError,
    load_stimulus_physical_coordinate_authority,
    require_bound_stimulus_physical_coordinate_authority,
)


STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_ID = (
    "palette.stimulus_response.coordinate_lineage"
)
STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_VERSION = 1


class StimulusResponseCoordinateAuthorityError(ValueError):
    """Raised when response inputs do not share one exact coordinate authority."""


def _pointer(record_ref: str, record_sha256: str) -> dict[str, str]:
    return {
        "record_ref": str(record_ref),
        "record_sha256": str(record_sha256),
    }


def _physical_identity(
    authority: BoundStimulusPhysicalCoordinateAuthority,
) -> dict[str, Any]:
    bound = require_bound_stimulus_physical_coordinate_authority(authority)
    return {
        "stimulus_run": bound.stimulus_run,
        "camera_id": bound.camera_id,
        "authority_manifest": _pointer(
            bound.manifest.record_ref,
            bound.manifest.record_sha256,
        ),
        "physical_frame": _pointer(
            bound.physical_frame.record_ref,
            bound.physical_frame.record_sha256,
        ),
        "source_camera_frame": _pointer(
            bound.source_camera_frame.record_ref,
            bound.source_camera_frame.record_sha256,
        ),
        "selected_calibration": _pointer(
            bound.selected_calibration.manifest_record_ref,
            bound.selected_calibration.manifest_sha256,
        ),
        "mm_per_pixel": float(bound.mm_per_pixel),
    }


def _coordinate_lineage_record(
    *,
    stimulus_run: str,
    evidence: BoundStimulusCoordinateEvidence,
    physical: BoundStimulusPhysicalCoordinateAuthority,
) -> dict[str, Any]:
    frame_transform = evidence.frame_transform
    selected = frame_transform.selected_calibration
    arena = frame_transform.arena_relative_frame
    canvas = frame_transform.selected_canvas_frame
    camera = frame_transform.source_camera_frame
    transform_chain = [
        _pointer(record.record_ref, record.record_sha256)
        for record in frame_transform.transform_chain.transform_records
    ]
    record = {
        "schema_id": STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_ID,
        "schema_version": STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_VERSION,
        "source_stimulus_run_ref": f"/analysis/stimulus_runs/{stimulus_run}",
        "stimulus_coordinate_output_manifest": _pointer(
            evidence.output_manifest.record_ref,
            evidence.output_manifest.record_sha256,
        ),
        "stimulus_frame_transform_manifest": _pointer(
            frame_transform.manifest.record_ref,
            frame_transform.manifest.record_sha256,
        ),
        "selected_calibration": _pointer(
            selected.manifest_record_ref,
            selected.manifest_sha256,
        ),
        "arena_geometry": _pointer(
            evidence.arena_reference.record_ref,
            evidence.arena_reference.record_sha256,
        ),
        "arena_frame": _pointer(arena.record_ref, arena.record_sha256),
        "selected_canvas_frame": _pointer(canvas.record_ref, canvas.record_sha256),
        "source_camera_frame": _pointer(camera.record_ref, camera.record_sha256),
        "arena_to_source_camera_transform_chain": transform_chain,
        "physical_authority": _physical_identity(physical),
        "coordinate_policy": (
            "typed_arena_or_canvas_to_source_camera_then_exact_mm_scale_v1"
        ),
    }
    record["record_sha256"] = canonical_mapping_digest(record)
    return record


def _stimulus_groups(
    root_node: zarr.Group,
    stimulus_run: str,
) -> tuple[zarr.Group, zarr.Group]:
    path = f"analysis/stimulus_runs/{stimulus_run}"
    try:
        run_group = root_node[path]
        chaser_group = run_group["tracking_data/chaser_states"]
    except Exception as exc:
        raise StimulusResponseCoordinateAuthorityError(
            f"Canonical stimulus run /{path} lacks tracking_data/chaser_states."
        ) from exc
    if not isinstance(run_group, zarr.Group) or not isinstance(chaser_group, zarr.Group):
        raise StimulusResponseCoordinateAuthorityError(
            f"Canonical stimulus run /{path} has invalid coordinate groups."
        )
    return run_group, chaser_group


def _load_stimulus_evidence(
    root_node: zarr.Group,
    stimulus_run: str,
) -> BoundStimulusCoordinateEvidence:
    run_group, chaser_group = _stimulus_groups(root_node, stimulus_run)
    try:
        return load_bound_stimulus_coordinate_evidence(
            run_group,
            chaser_group,
            root_node=root_node,
        )
    except StimulusCoordinateContractError as exc:
        raise StimulusResponseCoordinateAuthorityError(
            f"Stimulus run {stimulus_run!r} lacks canonical coordinate evidence: {exc}"
        ) from exc


def _load_physical_authority(
    root_node: zarr.Group,
    stimulus_run: str,
) -> BoundStimulusPhysicalCoordinateAuthority:
    try:
        authority = load_stimulus_physical_coordinate_authority(
            root_node,
            stimulus_run=stimulus_run,
        )
    except StimulusPhysicalCoordinateError as exc:
        raise StimulusResponseCoordinateAuthorityError(
            f"Stimulus run {stimulus_run!r} has invalid physical authority: {exc}"
        ) from exc
    if authority is None:
        raise StimulusResponseCoordinateAuthorityError(
            f"Stimulus run {stimulus_run!r} has no source-camera physical authority."
        )
    return require_bound_stimulus_physical_coordinate_authority(authority)


@dataclass(frozen=True)
class StimulusResponseCoordinateAuthority:
    """Freshly validated response-space context for one selected stimulus run."""

    stimulus_run: str
    record: Mapping[str, Any]
    evidence: BoundStimulusCoordinateEvidence = field(repr=False, compare=False)
    physical: BoundStimulusPhysicalCoordinateAuthority = field(
        repr=False,
        compare=False,
    )
    _root_node: zarr.Group = field(repr=False, compare=False)

    @property
    def mm_per_pixel(self) -> float:
        return float(self.physical.mm_per_pixel)

    @property
    def pixels_per_mm_projector(self) -> float | None:
        value = self.evidence.frame_transform.selected_calibration.pixels_per_mm_projector
        return float(value) if value is not None else None

    @property
    def z_eff_mm(self) -> float | None:
        value = self.evidence.frame_transform.selected_calibration.z_eff_mm
        return float(value) if value is not None else None

    @property
    def arena_width_px(self) -> float:
        return float(self.evidence.arena_reference.width)

    @property
    def arena_height_px(self) -> float:
        return float(self.evidence.arena_reference.height)

    def assert_verified(self) -> None:
        current_evidence = _load_stimulus_evidence(
            self._root_node,
            self.stimulus_run,
        )
        current_physical = _load_physical_authority(
            self._root_node,
            self.stimulus_run,
        )
        current = _coordinate_lineage_record(
            stimulus_run=self.stimulus_run,
            evidence=current_evidence,
            physical=current_physical,
        )
        if dict(self.record) != current:
            raise StimulusResponseCoordinateAuthorityError(
                "Stimulus-response coordinate authority changed after binding."
            )

    def assert_track_physical_authority(
        self,
        track_physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
    ) -> None:
        """Require a track run to use this exact source-camera physical frame."""

        if track_physical_authority is None:
            raise StimulusResponseCoordinateAuthorityError(
                "Track motion has no bound source-camera physical authority."
            )
        try:
            track_physical = require_bound_stimulus_physical_coordinate_authority(
                track_physical_authority
            )
        except StimulusPhysicalCoordinateError as exc:
            raise StimulusResponseCoordinateAuthorityError(
                f"Track physical authority is invalid: {exc}"
            ) from exc
        if _physical_identity(track_physical) != _physical_identity(self.physical):
            raise StimulusResponseCoordinateAuthorityError(
                "Track positions and selected stimulus run do not share the exact "
                "source-camera physical authority."
            )

    def arena_to_source_camera_mm(self, points_xy: Any) -> np.ndarray:
        return self.arena_to_source_camera_px(points_xy) * self.mm_per_pixel

    def arena_to_source_camera_px(self, points_xy: Any) -> np.ndarray:
        """Apply the exact published arena-to-source-camera transform chain."""

        camera_px = apply_bound_directed_transform_chain(
            points_xy,
            self.evidence.frame_transform.transform_chain,
        )
        return np.asarray(camera_px, dtype=np.float64)

    def selected_canvas_to_source_camera_mm(self, points_xy: Any) -> np.ndarray:
        return self.selected_canvas_to_source_camera_px(points_xy) * self.mm_per_pixel

    def selected_canvas_to_source_camera_px(self, points_xy: Any) -> np.ndarray:
        """Apply the exact selected-canvas-to-source-camera transform."""

        camera_px = apply_bound_directed_transform_v2(
            points_xy,
            self.evidence.frame_transform.canvas_to_source_camera,
        )
        return np.asarray(camera_px, dtype=np.float64)

    def arena_center_mm(self) -> tuple[float, float]:
        center_arena = np.asarray(
            [0.5 * self.arena_width_px, 0.5 * self.arena_height_px],
            dtype=np.float64,
        )
        center_mm = self.arena_to_source_camera_mm(center_arena)
        return float(center_mm[0]), float(center_mm[1])

    def arena_axis_extent_mm(self, direction_xy: Any) -> float:
        direction = np.asarray(direction_xy, dtype=np.float64)
        if (
            direction.shape != (2,)
            or not np.isfinite(direction).all()
            or float(np.linalg.norm(direction)) <= 0.0
        ):
            raise StimulusResponseCoordinateAuthorityError(
                "Arena-axis direction must be one finite nonzero XY vector."
            )
        direction = direction / np.linalg.norm(direction)
        corners = np.asarray(
            [
                [0.0, 0.0],
                [self.arena_width_px, 0.0],
                [0.0, self.arena_height_px],
                [self.arena_width_px, self.arena_height_px],
            ],
            dtype=np.float64,
        )
        corners_mm = self.arena_to_source_camera_mm(corners)
        projections = corners_mm @ direction
        return 0.5 * float(np.max(projections) - np.min(projections))


def load_stimulus_response_coordinate_authority(
    root_node: zarr.Group,
    *,
    stimulus_run: str,
    track_physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
) -> StimulusResponseCoordinateAuthority:
    """Bind one stimulus run to the exact physical frame used by track motion."""

    if not isinstance(stimulus_run, str) or not stimulus_run.strip():
        raise StimulusResponseCoordinateAuthorityError(
            "A non-empty selected stimulus run is required."
        )
    evidence = _load_stimulus_evidence(root_node, stimulus_run)
    physical = _load_physical_authority(root_node, stimulus_run)
    if track_physical_authority is None:
        raise StimulusResponseCoordinateAuthorityError(
            "Stimulus response requires track positions bound to a physical authority."
        )
    record = _coordinate_lineage_record(
        stimulus_run=stimulus_run,
        evidence=evidence,
        physical=physical,
    )
    result = StimulusResponseCoordinateAuthority(
        stimulus_run=stimulus_run,
        record=record,
        evidence=evidence,
        physical=physical,
        _root_node=root_node,
    )
    result.assert_verified()
    result.assert_track_physical_authority(track_physical_authority)
    return result


__all__ = [
    "STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_ID",
    "STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_VERSION",
    "StimulusResponseCoordinateAuthority",
    "StimulusResponseCoordinateAuthorityError",
    "load_stimulus_response_coordinate_authority",
]
