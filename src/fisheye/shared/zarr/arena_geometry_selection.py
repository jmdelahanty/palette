"""Canonical contract and reader for active arena-geometry selections.

Selection publication lives in the analysis-workflow layer, but operational
consumers also need one shared, fail-closed way to resolve the selected native
camera geometry.  This module owns that lower-layer read contract so tracking
and detection gating cannot grow independent selector semantics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_group

SELECTION_RECORD_SCHEMA_ID = "palette.arena_geometry_selection_record"
SELECTION_RECORD_SCHEMA_VERSION = 2
LEGACY_SELECTION_RECORD_SCHEMA_VERSION = 1
MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION = 3
SELECTION_RUN_SCHEMA_ID = "palette.arena_geometry_selection_run"
SELECTION_RUN_SCHEMA_VERSION = 1
SELECTION_RUNS_PARENT = "arena_geometry_selection"
SELECTION_POLICY = "reviewed_candidate_exact_binding_v1"
COMPARISON_BOUND_SELECTION_POLICY = "comparison_bound_reviewed_candidate_v2"
MANUAL_PALETTE_SELECTION_POLICY = "manual_reviewed_palette_candidate_exact_binding_v3"

_ACQUISITION_CANDIDATE_KIND = "acquisition_registered_dish"
_PALETTE_CANDIDATE_KIND = "palette_recording_image_fit"


@dataclass(frozen=True)
class ActiveArenaGeometryCircle:
    """One exact active native-camera circle and its immutable selection identity."""

    selection_run: str
    selection_record_sha256: str
    candidate_run: str
    camera_serial: str
    center_x_px: float
    center_y_px: float
    radius_px: float
    native_width_px: int
    native_height_px: int


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _required_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be one positive integer.")
    return int(value)


def _required_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def validate_arena_geometry_selection_record(record: Mapping[str, Any]) -> None:
    """Validate the canonical immutable selection-record contract."""

    schema_version = record.get("schema_version")
    expected_policy = {
        LEGACY_SELECTION_RECORD_SCHEMA_VERSION: SELECTION_POLICY,
        SELECTION_RECORD_SCHEMA_VERSION: COMPARISON_BOUND_SELECTION_POLICY,
        MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION: (
            MANUAL_PALETTE_SELECTION_POLICY
        ),
    }.get(schema_version)
    if (
        record.get("schema_id") != SELECTION_RECORD_SCHEMA_ID
        or expected_policy is None
        or record.get("selection_policy") != expected_policy
    ):
        raise ValueError("Unsupported arena-geometry selection record.")
    selected = record.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise ValueError("Selection lacks selected_candidate.")
    for name in (
        "run_name",
        "candidate_id",
        "candidate_kind",
        "candidate_record_sha256",
    ):
        if not str(selected.get(name) or "").strip():
            raise ValueError(f"Selection candidate lacks {name}.")
    if selected.get("run_name") != selected.get("candidate_id"):
        raise ValueError("Selection candidate run and identity disagree.")
    for name in ("arena_binding", "coordinate_binding", "valid_detection_region"):
        if not isinstance(selected.get(name), Mapping):
            raise ValueError(f"Selection candidate lacks {name}.")
    if schema_version == LEGACY_SELECTION_RECORD_SCHEMA_VERSION:
        if not isinstance(selected.get("physical_inner_rim"), Mapping):
            raise ValueError("Legacy selection candidate lacks physical_inner_rim.")
    else:
        boundary = selected.get("boundary_observation")
        if not isinstance(boundary, Mapping):
            raise ValueError("Selection candidate lacks boundary_observation.")
        kind = selected.get("candidate_kind")
        if kind == _ACQUISITION_CANDIDATE_KIND:
            if (
                not isinstance(selected.get("physical_inner_rim"), Mapping)
                or selected.get("observed_boundary") is not None
            ):
                raise ValueError("Selected acquisition boundary semantics are invalid.")
        elif kind == _PALETTE_CANDIDATE_KIND:
            if (
                not isinstance(selected.get("observed_boundary"), Mapping)
                or selected.get("physical_inner_rim") is not None
            ):
                raise ValueError("Selected Palette boundary semantics are invalid.")
        else:
            raise ValueError("Selection candidate kind is unsupported.")
    decision = record.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("Selection lacks decision metadata.")
    for name in ("selected_by", "decision_reason", "decision_source"):
        if not str(decision.get(name) or "").strip():
            raise ValueError(f"Selection decision lacks {name}.")
    comparison = decision.get("comparison_binding")
    if schema_version == SELECTION_RECORD_SCHEMA_VERSION:
        if not isinstance(comparison, Mapping):
            raise ValueError("Version-2 selection requires comparison binding.")
        for name in (
            "run_name",
            "comparison_record_sha256",
            "policy_id",
            "evidence_outcome",
            "workflow_action",
            "semantic_compatibility",
        ):
            if not str(comparison.get(name) or "").strip():
                raise ValueError(f"Selection comparison binding lacks {name}.")
    elif schema_version == MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION:
        if selected.get("candidate_kind") != _PALETTE_CANDIDATE_KIND:
            raise ValueError("Version-3 selection requires a Palette candidate.")
        if decision.get("decision_source") != "manual_review":
            raise ValueError("Version-3 selection requires explicit manual review.")
        if comparison is not None:
            raise ValueError("Version-3 selection cannot claim comparison evidence.")
    if record.get("candidate_mutated") is not False:
        raise ValueError("Selection must not claim candidate mutation.")
    if record.get("legacy_dish_mask_projection_written") is not False:
        raise ValueError("Arena-geometry selection cannot write a legacy projection.")


def resolve_active_arena_geometry_circle(
    root: Any,
) -> ActiveArenaGeometryCircle | None:
    """Resolve the active selected circle, or ``None`` if no modern family exists.

    Once the modern selection parent exists, an incomplete, malformed, or
    mismatched selector is an error.  Consumers must not silently fall back to
    legacy ``analysis_metadata.dish_mask`` in that state.
    """

    if "analysis" not in root or SELECTION_RUNS_PARENT not in root["analysis"]:
        return None
    parent = root["analysis"][SELECTION_RUNS_PARENT]
    run_name, run = resolve_latest_complete_run_group(parent, legacy_default=False)
    if run_name is None or run is None:
        raise ValueError(
            "Arena-geometry selection parent lacks one matching active complete "
            "selector."
        )
    if not run_name or "/" in run_name or run_name in {".", ".."}:
        raise ValueError("Active arena-geometry selection has an unsafe run name.")

    attrs = dict(run.attrs)
    if (
        attrs.get("schema_id") != SELECTION_RUN_SCHEMA_ID
        or attrs.get("schema_version") != SELECTION_RUN_SCHEMA_VERSION
        or attrs.get("selection_id") != run_name
        or attrs.get("operational_selection_status") != "selected"
        or attrs.get("detection_gate_applied") is not False
    ):
        raise ValueError("Active arena-geometry selection run is invalid.")
    record = attrs.get("selection_record")
    if not isinstance(record, Mapping):
        raise ValueError("Active arena-geometry selection lacks selection_record.")
    validate_arena_geometry_selection_record(record)
    digest = _payload_sha256(record)
    if attrs.get("selection_record_sha256") != digest:
        raise ValueError("Active arena-geometry selection record digest is invalid.")

    selected = record["selected_candidate"]
    binding = selected["coordinate_binding"]
    region = selected["valid_detection_region"]
    geometry = region.get("geometry")
    arena_binding = selected["arena_binding"]
    if not isinstance(geometry, Mapping) or not isinstance(arena_binding, Mapping):
        raise ValueError("Selected arena geometry lacks circle or arena binding.")
    center = geometry.get("center_px")
    if not isinstance(center, Mapping):
        raise ValueError("Selected arena geometry lacks a pixel center.")
    if (
        region.get("coordinate_space") != "camera_native_pixels"
        or geometry.get("type") != "circle"
        or binding.get("space_id") != "source_camera_image_px"
        or binding.get("profile_id") != "source_camera_image_px.top_left_y_down.v1"
        or binding.get("units") != "px"
        or binding.get("origin") != "top_left"
        or binding.get("positive_x") != "right"
        or binding.get("positive_y") != "down"
    ):
        raise ValueError(
            "Selected arena geometry is not native-camera circle geometry."
        )

    camera_serial = str(arena_binding.get("camera_serial") or "").strip()
    root_camera = str(
        root.attrs.get("camera_serial") or root.attrs.get("camera_id") or ""
    ).strip()
    if not camera_serial or (root_camera and root_camera != camera_serial):
        raise ValueError(
            "Selected arena geometry camera binding does not match archive."
        )

    radius = _required_finite(geometry.get("radius_px"), label="circle radius")
    if radius <= 0:
        raise ValueError("circle radius must be positive.")
    return ActiveArenaGeometryCircle(
        selection_run=run_name,
        selection_record_sha256=digest,
        candidate_run=str(selected["run_name"]),
        camera_serial=camera_serial,
        center_x_px=_required_finite(center.get("x"), label="circle center x"),
        center_y_px=_required_finite(center.get("y"), label="circle center y"),
        radius_px=radius,
        native_width_px=_required_positive_int(
            binding.get("native_width_px"), label="native width"
        ),
        native_height_px=_required_positive_int(
            binding.get("native_height_px"), label="native height"
        ),
    )


__all__ = [
    "ActiveArenaGeometryCircle",
    "COMPARISON_BOUND_SELECTION_POLICY",
    "LEGACY_SELECTION_RECORD_SCHEMA_VERSION",
    "MANUAL_PALETTE_SELECTION_POLICY",
    "MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION",
    "SELECTION_POLICY",
    "SELECTION_RECORD_SCHEMA_ID",
    "SELECTION_RECORD_SCHEMA_VERSION",
    "SELECTION_RUN_SCHEMA_ID",
    "SELECTION_RUN_SCHEMA_VERSION",
    "SELECTION_RUNS_PARENT",
    "resolve_active_arena_geometry_circle",
    "validate_arena_geometry_selection_record",
]
