"""Composable exact-epoch chaser locations for spatial display overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.visualization.chaser_appearance import ChaserAppearance

from .projection import ExactChaserSuccessorProjection

CHASER_LOCATION_DISPLAY_RECIPE = (
    "exact_logged_chaser_epoch_median_protocol_color_role_glyph_v1"
)
STATIC_LOCATION_EPOCH_ROLES = ("chaser_pre", "chaser_post")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Chaser-location projection lacks {label}.")
    return value


@dataclass(frozen=True, slots=True)
class ExactChaserEpochLocation:
    """A display-only median of exact logged positions in one semantic epoch."""

    epoch_index: int
    analysis_role: str
    appearance: ChaserAppearance
    x_mm: float
    y_mm: float
    sample_count: int
    median_drift_mm: float
    p95_drift_mm: float
    maximum_drift_mm: float

    def provenance_record(self) -> dict[str, Any]:
        return {
            "epoch_index": self.epoch_index,
            "analysis_role": self.analysis_role,
            "identity_code": self.appearance.identity_code,
            "chaser_index": self.appearance.chaser_index,
            "identity": self.appearance.identity,
            "behavior_role": self.appearance.behavior_role,
            "experimental_color_rgba": list(self.appearance.experimental_color_rgba),
            "role_glyph": self.appearance.plotly_role_symbol,
            "median_position_mm": [self.x_mm, self.y_mm],
            "sample_count": self.sample_count,
            "median_drift_mm": self.median_drift_mm,
            "p95_drift_mm": self.p95_drift_mm,
            "maximum_drift_mm": self.maximum_drift_mm,
        }


def exact_static_chaser_epoch_locations(
    projection: ExactChaserSuccessorProjection,
) -> tuple[ExactChaserEpochLocation, ...]:
    """Project pre/post logged chaser rows without interpolation or role-color inference."""

    if projection.relatives is None or projection.chaser_appearance is None:
        raise ValueError(
            "Exact chaser locations require relative-frame arrays and appearance binding."
        )
    keypoint = projection.relatives[0]
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    positions = np.asarray(
        keypoint.frame_chaser("chaser_position_xy_px"), dtype=np.float64
    )
    valid = np.asarray(
        keypoint.frame_chaser("chaser_position_valid"), dtype=bool
    ) & np.asarray(keypoint.frame_chaser("chaser_occurrence_member"), dtype=bool)
    identity = np.asarray(keypoint.frame_chaser("chaser_identity_code"), dtype=np.int64)
    role = np.asarray(
        keypoint.frame_chaser("chaser_behavior_role_code"), dtype=np.int64
    )
    if positions.shape != (keypoint.n_frames, keypoint.n_chasers, 2):
        raise ValueError("Exact logged chaser positions have an invalid shape.")
    if not np.all(identity == identity[:1]) or not np.all(role == role[:1]):
        raise ValueError("Exact chaser identity or behavior role changes by frame.")
    appearance_by_code = projection.chaser_appearance.by_identity_code()
    if set(appearance_by_code) != set(int(value) for value in identity[0]):
        raise ValueError("Appearance identities differ from relative-frame columns.")

    scientific = projection.spatial.scientific_manifest
    arena = _mapping(scientific.get("arena"), label="reviewed arena")
    center_x = float(arena.get("center_x_px", np.nan))
    center_y = float(arena.get("center_y_px", np.nan))
    mm_per_pixel = float(arena.get("mm_per_pixel", np.nan))
    if not np.all(np.isfinite([center_x, center_y, mm_per_pixel])) or mm_per_pixel <= 0:
        raise ValueError("Reviewed arena lacks a finite physical pixel scale.")

    locations: list[ExactChaserEpochLocation] = []
    observed_roles: set[str] = set()
    for epoch_index, record in enumerate(projection.epoch_records):
        analysis_role = str(record.get("analysis_role") or "")
        if analysis_role not in STATIC_LOCATION_EPOCH_ROLES:
            continue
        if analysis_role in observed_roles:
            raise ValueError("Static chaser-location epoch roles are duplicated.")
        observed_roles.add(analysis_role)
        start = int(record["start_frame"])
        stop = int(record["end_frame_exclusive"])
        epoch_member = selected & (frame_id >= start) & (frame_id < stop)
        if not np.any(epoch_member):
            raise ValueError("An exact static chaser epoch has no selected rows.")
        for column in range(keypoint.n_chasers):
            appearance = appearance_by_code[int(identity[0, column])]
            if appearance.behavior_role_code != int(role[0, column]):
                raise ValueError(
                    "Appearance behavior role differs from relative-frame evidence."
                )
            member = epoch_member & valid[:, column]
            member &= np.all(np.isfinite(positions[:, column]), axis=1)
            values = positions[member, column]
            if not values.size:
                raise ValueError(
                    "An exact static epoch lacks valid logged chaser positions."
                )
            median_px = np.median(values, axis=0)
            centered_mm = (
                median_px - np.asarray([center_x, center_y], dtype=np.float64)
            ) * mm_per_pixel
            drift_mm = np.linalg.norm(values - median_px, axis=1) * mm_per_pixel
            locations.append(
                ExactChaserEpochLocation(
                    epoch_index=epoch_index,
                    analysis_role=analysis_role,
                    appearance=appearance,
                    x_mm=float(centered_mm[0]),
                    y_mm=float(centered_mm[1]),
                    sample_count=int(values.shape[0]),
                    median_drift_mm=float(np.median(drift_mm)),
                    p95_drift_mm=float(np.quantile(drift_mm, 0.95)),
                    maximum_drift_mm=float(np.max(drift_mm)),
                )
            )
    if observed_roles != set(STATIC_LOCATION_EPOCH_ROLES):
        raise ValueError("Exact spatial display requires one pre and one post epoch.")
    return tuple(locations)


__all__ = [
    "CHASER_LOCATION_DISPLAY_RECIPE",
    "STATIC_LOCATION_EPOCH_ROLES",
    "ExactChaserEpochLocation",
    "exact_static_chaser_epoch_locations",
]
