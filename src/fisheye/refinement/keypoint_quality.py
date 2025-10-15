"""
Geometry analysis helpers for keypoint refinement.

Shared utilities for computing triangle metrics formed by bladder/eye keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class KeypointGeometryMetrics:
    """Triangle geometry derived from three keypoints."""

    area: float
    angles: np.ndarray  # degrees, shape (3,)
    min_angle: float
    max_angle: float
    edge_lengths: np.ndarray  # shape (3,)


def _triangle_area(vertices: np.ndarray) -> float:
    """Compute triangle area using the cross product."""
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])


def _edge_lengths(vertices: np.ndarray) -> np.ndarray:
    """Compute edge lengths for triangle defined by vertices."""
    # Edges: (p1-p2), (p1-p3), (p2-p3)
    e0 = np.linalg.norm(vertices[0] - vertices[1])
    e1 = np.linalg.norm(vertices[0] - vertices[2])
    e2 = np.linalg.norm(vertices[1] - vertices[2])
    return np.array([e0, e1, e2], dtype=np.float64)


def _triangle_angles(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute triangle interior angles (degrees) via the Law of Cosines.

    Returns tuple of (angles, min_angle, max_angle).
    """
    angles = np.full(3, np.nan, dtype=np.float64)

    # Avoid division by zero when triangle collapses.
    def safe_acos(val: float) -> float:
        return np.arccos(np.clip(val, -1.0, 1.0))

    # Using standard triangle notation:
    # angle at vertex 0 is opposite edge length edges[2] (between vertices 1 and 2)
    # angle at vertex 1 opposite edges[1], angle at vertex 2 opposite edges[0]
    a, b, c = edges
    if b > 0 and c > 0:
        cos_a0 = (b**2 + c**2 - a**2) / (2 * b * c)
        angles[0] = safe_acos(cos_a0)
    if a > 0 and c > 0:
        cos_a1 = (a**2 + c**2 - b**2) / (2 * a * c)
        angles[1] = safe_acos(cos_a1)
    if a > 0 and b > 0:
        cos_a2 = (a**2 + b**2 - c**2) / (2 * a * b)
        angles[2] = safe_acos(cos_a2)

    angles_deg = np.rad2deg(angles)
    finite_angles = angles_deg[np.isfinite(angles_deg)]
    if finite_angles.size == 0:
        min_angle = float("nan")
        max_angle = float("nan")
    else:
        min_angle = float(np.min(finite_angles))
        max_angle = float(np.max(finite_angles))

    return angles_deg, min_angle, max_angle


def compute_geometry_metrics(vertices: np.ndarray) -> KeypointGeometryMetrics:
    """Compute core triangle metrics for a set of three (x, y) vertices."""
    if vertices.shape != (3, 2):
        raise ValueError(f"Expected vertices with shape (3, 2), got {vertices.shape}")

    if np.isnan(vertices).any():
        return KeypointGeometryMetrics(
            area=float("nan"),
            angles=np.full(3, np.nan, dtype=np.float64),
            min_angle=float("nan"),
            max_angle=float("nan"),
            edge_lengths=np.full(3, np.nan, dtype=np.float64),
        )

    edges = _edge_lengths(vertices)
    area = _triangle_area(vertices)
    angles, min_angle, max_angle = _triangle_angles(vertices, edges)

    return KeypointGeometryMetrics(
        area=float(area),
        angles=angles,
        min_angle=float(min_angle),
        max_angle=float(max_angle),
        edge_lengths=edges,
    )
