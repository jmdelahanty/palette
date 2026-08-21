"""Immutable acquisition-derived arena-geometry candidate publication.

Candidates preserve geometry and lineage for later comparison and selection.
Publishing a candidate never makes it an operational dish mask and never
updates the legacy ``analysis_metadata.attrs['dish_mask']`` projection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import zarr

from fisheye.analysis_workflows.materializers.arena_geometry_fit_review import (
    FIT_REVIEW_RUNS_PARENT,
    load_arena_geometry_fit_review_evidence,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
)
from fisheye.shared.recording_geometry import (
    BoundRegisteredDishMask,
    GeometryLoadPolicy,
    MaskGeometryStatus,
    RecordingGeometryError,
    RegisteredDishMask,
    bind_registered_dish_mask_to_source_camera_frame,
    load_registered_dish_masks_from_citrus_h5,
    load_registered_dish_masks_from_recording_folder,
)
from fisheye.shared.recording_geometry_recovery import (
    RECOVERY_AUTHORITY,
    RECOVERY_REASON,
    VerifiedRecordingGeometryRecovery,
    registered_dish_mask_from_verified_recovery,
    validate_recording_geometry_recovery_receipt,
)
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)

CANDIDATE_RECORD_SCHEMA_ID = "palette.arena_geometry_candidate_record"
CANDIDATE_RECORD_SCHEMA_VERSION = 1
CANDIDATE_RUN_SCHEMA_ID = "palette.arena_geometry_candidate_run"
CANDIDATE_RUN_SCHEMA_VERSION = 1
ACQUISITION_CANDIDATE_KIND = "acquisition_registered_dish"
PALETTE_CANDIDATE_KIND = "palette_recording_image_fit"
CLIPPED_ACQUISITION_FRAME_AUTHORITY_KIND = (
    "palette_acquisition_camera_frame_v1"
)
LEGACY_CLIPPED_SNAPSHOT_FRAME_AUTHORITY_KIND = (
    "orange_recording_snapshot_coordinate_frame_v1"
)
# Compatibility export for callers written before Palette image candidates.
CANDIDATE_KIND = ACQUISITION_CANDIDATE_KIND
CANDIDATE_RUNS_PARENT = "arena_geometry_runs"
PUBLISH_SCHEMA_ID = "palette.arena_geometry_candidate_publish"
PUBLISH_ALGORITHM_VERSION = 1
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArenaGeometryCandidatePlan:
    source_zarr: Path
    receipt_path: Path
    receipt_sha256: str
    candidate_id: str
    candidate_record_sha256: str
    candidate_record: Mapping[str, Any]
    run_name: str
    target_run_path: Path
    run_provenance: Mapping[str, Any]
    candidate_kind: str = ACQUISITION_CANDIDATE_KIND


def _canonical_copy(value: Any) -> Any:
    return json.loads(strict_json_dumps(value))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordingGeometryError(f"{label} must be a mapping.")
    return value


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordingGeometryError(f"{label} must be a nonempty string.")
    return value.strip()


def _required_sha256(value: Any, *, label: str) -> str:
    text = _required_text(value, label=label).lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise RecordingGeometryError(f"{label} must be a SHA-256 digest.")
    return text


def _required_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingGeometryError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise RecordingGeometryError(f"{label} must be finite.")
    return result


def _circle_record(circle: Any) -> dict[str, Any]:
    return {
        "type": "circle",
        "center_px": {
            "x": float(circle.center_x_native_px),
            "y": float(circle.center_y_native_px),
        },
        "radius_px": float(circle.radius_px),
    }


def build_acquisition_geometry_candidate_record(
    bound_mask: BoundRegisteredDishMask,
    *,
    recovery_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize one proven acquisition mask into candidate-only semantics."""

    if type(bound_mask) is not BoundRegisteredDishMask:
        raise RecordingGeometryError("A bound registered dish mask is required.")
    mask = bound_mask.mask
    record = {
        "schema_id": CANDIDATE_RECORD_SCHEMA_ID,
        "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "candidate_kind": CANDIDATE_KIND,
        "arena_binding": {
            "rig_id": mask.key.rig_id,
            "canvas_name": mask.key.canvas_name,
            "arena_id": mask.key.arena_id,
            "camera_serial": mask.key.camera_serial,
        },
        "physical_inner_rim": {
            "coordinate_space": mask.coordinate_space,
            "target_plane": mask.target_plane,
            "geometry": _circle_record(mask.physical_inner_rim),
        },
        "valid_detection_region": {
            "coordinate_space": mask.coordinate_space,
            "purpose": "bounding_box_centroid_detection_gating",
            "offset_direction": "outward",
            "geometry": _circle_record(mask.valid_detection_gate),
            "is_final_acquisition_tolerance": True,
            "additional_palette_tolerance_px": 0.0,
        },
        "coordinate_binding": {
            "space_id": mask.palette_space_id,
            "profile_id": mask.coordinate_profile_id,
            "pixel_convention": mask.pixel_convention,
            "units": "px",
            "origin": mask.origin,
            "positive_x": mask.positive_x,
            "positive_y": mask.positive_y,
            "native_width_px": mask.native_width_px,
            "native_height_px": mask.native_height_px,
            "pixel_frame_record_ref": bound_mask.pixel_frame_record_ref,
            "pixel_frame_record_sha256": bound_mask.pixel_frame_record_sha256,
        },
        "acquisition_source": {
            "source_kind": mask.source_kind,
            "artifact_id": mask.artifact_id,
            "source_observation_sha256": mask.source_observation_sha256,
            "registration_id": mask.registration_id,
            "registration_sha256": mask.registration_sha256,
            "source_contract_sha256": mask.source_contract_sha256,
            "materialized_asset_status": mask.materialized_asset_status.value,
            "citrus_registration_status": mask.citrus_registration_status.value,
            "selected_daily_registration_applied_by_citrus": (
                mask.selected_daily_registration_applied_by_citrus
            ),
            "producer_contract_linkage_status": (mask.producer_contract_linkage_status),
            "source_valid_until_utc": mask.source_valid_until_utc,
            "producer_operator_accepted": mask.producer_operator_accepted,
            "producer_quality_flags": list(mask.producer_quality_flags),
            "recovery_binding": (
                _canonical_copy(recovery_binding)
                if recovery_binding is not None
                else None
            ),
        },
        "candidate_policy": {
            "publication_role": "candidate_only",
            "operationally_selected": False,
            "legacy_dish_mask_projection_written": False,
            "detection_gate_applied": False,
            "independent_palette_fit_required_before_operational_use": True,
        },
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    normalized = _canonical_copy(record)
    validate_acquisition_geometry_candidate_record(normalized)
    return normalized


def validate_acquisition_geometry_candidate_record(record: Mapping[str, Any]) -> None:
    """Validate scientific and coordinate invariants of one candidate record."""

    if (
        record.get("schema_id") != CANDIDATE_RECORD_SCHEMA_ID
        or record.get("schema_version") != CANDIDATE_RECORD_SCHEMA_VERSION
    ):
        raise RecordingGeometryError("Unsupported arena-geometry candidate schema.")
    if record.get("candidate_kind") != CANDIDATE_KIND:
        raise RecordingGeometryError("Unsupported arena-geometry candidate kind.")
    arena = _required_mapping(record.get("arena_binding"), label="arena_binding")
    for name in ("rig_id", "canvas_name", "arena_id", "camera_serial"):
        _required_text(arena.get(name), label=f"arena_binding.{name}")

    physical = _required_mapping(
        record.get("physical_inner_rim"), label="physical_inner_rim"
    )
    gate = _required_mapping(
        record.get("valid_detection_region"),
        label="valid_detection_region",
    )
    if (
        physical.get("coordinate_space") != "camera_native_pixels"
        or gate.get("coordinate_space") != "camera_native_pixels"
    ):
        raise RecordingGeometryError("Candidate circles must use native camera pixels.")
    if physical.get("target_plane") != "dish_top_rim":
        raise RecordingGeometryError("Physical rim must target dish_top_rim.")
    if (
        gate.get("purpose") != "bounding_box_centroid_detection_gating"
        or gate.get("offset_direction") != "outward"
    ):
        raise RecordingGeometryError("Detection gate semantics are invalid.")
    if (
        gate.get("is_final_acquisition_tolerance") is not True
        or gate.get("additional_palette_tolerance_px") != 0.0
    ):
        raise RecordingGeometryError(
            "Acquisition gate must not receive added Palette tolerance."
        )

    def circle(container: Mapping[str, Any], label: str) -> tuple[float, float, float]:
        geometry = _required_mapping(
            container.get("geometry"), label=f"{label}.geometry"
        )
        if geometry.get("type") != "circle":
            raise RecordingGeometryError(f"{label} must be circular.")
        center = _required_mapping(
            geometry.get("center_px"), label=f"{label}.center_px"
        )
        x = _required_finite(center.get("x"), label=f"{label}.center.x")
        y = _required_finite(center.get("y"), label=f"{label}.center.y")
        radius = _required_finite(geometry.get("radius_px"), label=f"{label}.radius")
        if radius <= 0:
            raise RecordingGeometryError(f"{label} radius must be positive.")
        return x, y, radius

    physical_x, physical_y, physical_radius = circle(physical, "physical_inner_rim")
    gate_x, gate_y, gate_radius = circle(gate, "valid_detection_region")
    if not math.isclose(physical_x, gate_x, abs_tol=1e-6) or not math.isclose(
        physical_y, gate_y, abs_tol=1e-6
    ):
        raise RecordingGeometryError("Physical rim and valid gate must be concentric.")
    if gate_radius < physical_radius:
        raise RecordingGeometryError(
            "Valid gate cannot be smaller than the physical rim."
        )

    coordinate = _required_mapping(
        record.get("coordinate_binding"), label="coordinate_binding"
    )
    if (
        coordinate.get("space_id") != "source_camera_image_px"
        or coordinate.get("profile_id") != "source_camera_image_px.top_left_y_down.v1"
        or coordinate.get("pixel_convention") != "continuous"
        or coordinate.get("units") != "px"
        or coordinate.get("origin") != "top_left"
        or coordinate.get("positive_x") != "right"
        or coordinate.get("positive_y") != "down"
    ):
        raise RecordingGeometryError(
            "Candidate source-camera coordinate binding is invalid."
        )
    for name in ("native_width_px", "native_height_px"):
        value = coordinate.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RecordingGeometryError(f"coordinate_binding.{name} must be positive.")
    expected_ref = (
        f"/analysis/coordinate_frames/source_camera/{arena['camera_serial']}"
        "/continuous@pixel_frame_authority"
    )
    if coordinate.get("pixel_frame_record_ref") != expected_ref:
        raise RecordingGeometryError(
            "Candidate does not bind the canonical continuous frame."
        )
    _required_sha256(
        coordinate.get("pixel_frame_record_sha256"),
        label="pixel_frame_record_sha256",
    )

    source = _required_mapping(
        record.get("acquisition_source"), label="acquisition_source"
    )
    for name in ("source_kind", "artifact_id", "registration_id"):
        _required_text(source.get(name), label=f"acquisition_source.{name}")
    for name in ("source_observation_sha256", "source_contract_sha256"):
        _required_sha256(source.get(name), label=f"acquisition_source.{name}")
    if source.get("registration_sha256") is not None:
        _required_sha256(
            source.get("registration_sha256"),
            label="acquisition_source.registration_sha256",
        )
    recovery = source.get("recovery_binding")
    if recovery is not None:
        recovery_map = _required_mapping(recovery, label="recovery_binding")
        _required_sha256(recovery_map.get("receipt_sha256"), label="receipt_sha256")
        _required_sha256(recovery_map.get("target_h5_sha256"), label="target_h5_sha256")
        if (
            recovery_map.get("authority") != RECOVERY_AUTHORITY
            or recovery_map.get("reason") != RECOVERY_REASON
        ):
            raise RecordingGeometryError("Recovery binding authority is invalid.")

    policy = _required_mapping(record.get("candidate_policy"), label="candidate_policy")
    expected_policy = {
        "publication_role": "candidate_only",
        "operationally_selected": False,
        "legacy_dish_mask_projection_written": False,
        "detection_gate_applied": False,
        "independent_palette_fit_required_before_operational_use": True,
    }
    if dict(policy) != expected_policy:
        raise RecordingGeometryError("Acquisition candidate policy is not fail-closed.")
    if record.get("canonicalization") != "canonical_json_sort_keys_v1":
        raise RecordingGeometryError("Candidate canonicalization is unsupported.")
    if _canonical_copy(record) != dict(record):
        raise RecordingGeometryError(
            "Candidate record is not strict canonical JSON data."
        )


def _circle_mapping(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], tuple[float, float, float]]:
    geometry = _required_mapping(value, label=label)
    if geometry.get("type") != "circle":
        raise RecordingGeometryError(f"{label} must be circular.")
    center = _required_mapping(geometry.get("center_px"), label=f"{label}.center_px")
    x = _required_finite(center.get("x"), label=f"{label}.center_px.x")
    y = _required_finite(center.get("y"), label=f"{label}.center_px.y")
    radius = _required_finite(geometry.get("radius_px"), label=f"{label}.radius_px")
    if radius <= 0:
        raise RecordingGeometryError(f"{label}.radius_px must be positive.")
    return (
        {
            "type": "circle",
            "center_px": {"x": x, "y": y},
            "radius_px": radius,
        },
        (x, y, radius),
    )


def _source_camera_candidate_binding(
    source_zarr: Path,
    *,
    expected_camera_serial: str,
    fit_source: Mapping[str, Any] | None = None,
    arena_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str], Mapping[str, Any]]:
    root = open_zarr_root(source_zarr, mode="r")
    attrs = dict(root.attrs)
    source_mode = str((fit_source or {}).get("mode") or "single_video")
    if source_mode == "clipped_recording":
        cameras = [str(value) for value in attrs.get("camera_serials", [])]
        if cameras != [expected_camera_serial]:
            raise RecordingGeometryError(
                "Palette fit camera does not match the clipped analysis Zarr camera."
            )
        if attrs.get("source_layout") != "rolling_clips":
            raise RecordingGeometryError(
                "Clipped Palette fits require a rolling-clips analysis Zarr."
            )
        source = _required_mapping(fit_source, label="fit_report.source")
        recording_id = _required_text(
            source.get("recording_id"), label="fit_report.source.recording_id"
        )
        if str(attrs.get("recording_id") or "") != recording_id:
            raise RecordingGeometryError(
                "Palette fit recording does not match the clipped analysis Zarr."
            )
        clip_index_path = (
            Path(
                _required_text(
                    attrs.get("recording_clip_index_json"),
                    label="root.recording_clip_index_json",
                )
            )
            .expanduser()
            .resolve()
        )
        if not clip_index_path.is_file():
            raise RecordingGeometryError(
                f"Clipped recording index is missing: {clip_index_path}"
            )
        expected_clip_index_sha256 = _required_sha256(
            source.get("recording_clip_index_sha256"),
            label="fit_report.source.recording_clip_index_sha256",
        ).removeprefix("sha256:")
        if _file_sha256(clip_index_path) != expected_clip_index_sha256:
            raise RecordingGeometryError(
                "Clipped recording index changed after the Palette fit."
            )
        expected_frame_count = int(source.get("frame_count") or 0)
        expected_first = int(source.get("first_recording_frame_id") or 0)
        expected_last = int(source.get("last_recording_frame_id") or 0)
        expected_clip_count = int(source.get("clip_count") or 0)
        expected_session_id = _required_text(
            source.get("session_id"), label="fit_report.source.session_id"
        )
        if (
            expected_frame_count <= 0
            or expected_last - expected_first + 1 != expected_frame_count
            or expected_clip_count <= 0
            or int(attrs.get("clip_count") or 0) != expected_clip_count
            or str(attrs.get("session_id") or "") != expected_session_id
            or int(attrs.get("recording_frame_index_row_count") or 0)
            != expected_frame_count
            or int(attrs.get("recording_frame_id_min") or 0) != expected_first
            or int(attrs.get("recording_frame_id_max") or 0) != expected_last
        ):
            raise RecordingGeometryError(
                "Palette fit frame domain does not match the clipped analysis Zarr."
            )

        recording_root = source_zarr.parent.parent.resolve()
        snapshot_path = (
            recording_root
            / "raw"
            / "recording_geometry_bundle"
            / "recording_snapshot.json"
        )
        if not snapshot_path.is_file():
            raise RecordingGeometryError(
                f"Clipped recording geometry snapshot is missing: {snapshot_path}"
            )
        snapshot_sha256 = _file_sha256(snapshot_path)
        if snapshot_sha256 != _required_sha256(
            source.get("recording_geometry_snapshot_sha256"),
            label="fit_report.source.recording_geometry_snapshot_sha256",
        ).removeprefix("sha256:"):
            raise RecordingGeometryError(
                "Recording geometry snapshot changed after the Palette fit."
            )
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordingGeometryError(
                "Clipped recording geometry snapshot is invalid JSON."
            ) from exc
        runtime = _required_mapping(
            _required_mapping(snapshot, label="recording_snapshot").get(
                "camera_runtime"
            ),
            label="recording_snapshot.camera_runtime",
        )
        camera_runtime = _required_mapping(
            runtime.get(expected_camera_serial),
            label=f"recording_snapshot.camera_runtime.{expected_camera_serial}",
        )
        frame = _required_mapping(
            camera_runtime.get("coordinate_frame"),
            label="recording_snapshot camera coordinate_frame",
        )
        extent = _required_mapping(
            frame.get("extent"), label="recording_snapshot coordinate extent"
        )
        width = int(extent.get("width_px") or 0)
        height = int(extent.get("height_px") or 0)
        if (
            width <= 0
            or height <= 0
            or frame.get("coordinate_space") != "camera_native_pixels"
            or frame.get("units") != "pixels"
            or _required_mapping(
                frame.get("origin"), label="recording_snapshot coordinate origin"
            ).get("name")
            != "top_left_pixel"
        ):
            raise RecordingGeometryError(
                "Clipped recording snapshot has an unsupported camera coordinate frame."
            )
        _ownership, acquisition = load_persisted_acquisition_camera_authority(
            root,
            expected_camera_id=expected_camera_serial,
        )
        acquisition.assert_verified()
        if (
            acquisition.record.camera_id != expected_camera_serial
            or acquisition.width != width
            or acquisition.height != height
        ):
            raise RecordingGeometryError(
                "Persisted acquisition camera authority does not match the clipped "
                "recording snapshot."
            )
        calibration = root.get("analysis/calibration")
        if calibration is not None:
            calibration_attrs = dict(calibration.attrs)
            if (
                str(calibration_attrs.get("active_camera_id") or "")
                != expected_camera_serial
                or int(calibration_attrs.get("native_width_px") or 0) != width
                or int(calibration_attrs.get("native_height_px") or 0) != height
            ):
                raise RecordingGeometryError(
                    "Imported calibration does not match the clipped camera frame."
                )

        requested_arena = dict(arena_binding or {})
        arena = {
            "rig_id": _required_text(
                attrs.get("rig_id") or requested_arena.get("rig_id"),
                label="clipped arena rig_id",
            ),
            "canvas_name": _required_text(
                attrs.get("canvas_name") or requested_arena.get("canvas_name"),
                label="clipped arena canvas_name",
            ),
            "arena_id": _required_text(
                attrs.get("arena_id") or requested_arena.get("arena_id"),
                label="clipped arena arena_id",
            ),
            "camera_serial": expected_camera_serial,
        }
        for name in ("rig_id", "canvas_name", "arena_id"):
            root_value = attrs.get(name)
            supplied_value = requested_arena.get(name)
            if (
                root_value is not None
                and supplied_value is not None
                and str(root_value) != str(supplied_value)
            ):
                raise RecordingGeometryError(
                    f"Explicit clipped {name} conflicts with the analysis Zarr."
                )
        coordinate = {
            "space_id": "source_camera_image_px",
            "profile_id": "source_camera_image_px.top_left_y_down.v1",
            "pixel_convention": "continuous",
            "units": "px",
            "origin": "top_left",
            "positive_x": "right",
            "positive_y": "down",
            "native_width_px": width,
            "native_height_px": height,
            "source_camera_frame_authority_kind": (
                CLIPPED_ACQUISITION_FRAME_AUTHORITY_KIND
            ),
            "pixel_frame_record_ref": acquisition.record_ref,
            "pixel_frame_record_sha256": acquisition.record_sha256,
        }
        collection = {
            "mode": "clipped_recording",
            "recording_id": recording_id,
            "recording_clip_index_path": str(clip_index_path),
            "recording_clip_index_sha256": expected_clip_index_sha256,
            "recording_geometry_snapshot_path": str(snapshot_path),
            "recording_geometry_snapshot_sha256": snapshot_sha256,
            "frame_count": expected_frame_count,
            "first_recording_frame_id": expected_first,
            "last_recording_frame_id": expected_last,
        }
        return coordinate, arena, collection
    if source_mode != "single_video":
        raise RecordingGeometryError(
            f"Unsupported Palette fit source mode: {source_mode!r}."
        )
    camera_serial = str(attrs.get("camera_id") or "")
    if camera_serial != expected_camera_serial:
        raise RecordingGeometryError(
            "Palette fit camera does not match the analysis Zarr camera."
        )
    arena = {
        "rig_id": _required_text(attrs.get("rig_id"), label="root.rig_id"),
        "canvas_name": _required_text(
            attrs.get("canvas_name"), label="root.canvas_name"
        ),
        "arena_id": _required_text(attrs.get("arena_id"), label="root.arena_id"),
        "camera_serial": camera_serial,
    }
    source_video_metadata = _required_mapping(
        attrs.get("source_video_metadata"), label="root.source_video_metadata"
    )
    _ownership, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=camera_serial,
    )
    frame_path = f"analysis/coordinate_frames/source_camera/{camera_serial}/continuous"
    try:
        frame_node = root[frame_path]
    except KeyError as exc:
        raise RecordingGeometryError(
            f"Analysis Zarr lacks canonical continuous source-camera authority {frame_path}."
        ) from exc
    source_frame = load_source_camera_pixel_frame_authority(
        frame_node,
        acquisition_frame=acquisition,
    )
    source_frame.assert_verified()
    endpoint = source_frame.endpoint
    coordinate = {
        "space_id": endpoint.space_id,
        "profile_id": "source_camera_image_px.top_left_y_down.v1",
        "pixel_convention": endpoint.pixel_convention,
        "units": endpoint.units,
        "origin": "top_left",
        "positive_x": "right",
        "positive_y": "down",
        "native_width_px": endpoint.width,
        "native_height_px": endpoint.height,
        "pixel_frame_record_ref": endpoint.record_ref,
        "pixel_frame_record_sha256": endpoint.record_sha256,
    }
    return coordinate, arena, source_video_metadata


def build_reviewed_palette_geometry_candidate_record(
    *,
    source_zarr: str | Path,
    fit_report_path: str | Path | None = None,
    montage_path: str | Path | None = None,
    fit_review_run: str | None = None,
    review: Mapping[str, Any],
    arena_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one reviewed blind fit without reinterpreting acquisition geometry."""

    zarr_path = Path(source_zarr).expanduser().resolve()
    embedded_evidence = None
    if fit_review_run is not None:
        if fit_report_path is not None or montage_path is not None:
            raise RecordingGeometryError(
                "Choose the embedded fit-review run or external fit files, not both."
            )
        embedded_evidence = load_arena_geometry_fit_review_evidence(
            zarr_path,
            run_name=_required_text(fit_review_run, label="fit_review_run"),
        )
        report_bytes = embedded_evidence.fit_report_bytes
        montage_bytes = embedded_evidence.montage_bytes
        reveal_bytes = embedded_evidence.acquisition_reveal_bytes
        report_ref = embedded_evidence.fit_report_ref
        montage_ref = embedded_evidence.montage_ref
        reveal_ref = embedded_evidence.acquisition_reveal_ref
    else:
        if fit_report_path is None or montage_path is None:
            raise RecordingGeometryError(
                "External Palette evidence requires both fit report and review montage."
            )
        report_path = Path(fit_report_path).expanduser().resolve()
        review_montage = Path(montage_path).expanduser().resolve()
        if not report_path.is_file() or not review_montage.is_file():
            raise FileNotFoundError(
                "Palette fit report and review montage must both exist."
            )
        report_bytes = report_path.read_bytes()
        montage_bytes = review_montage.read_bytes()
        report_ref = str(report_path)
        montage_ref = str(review_montage)
        reveal_path = report_path.with_name("acquisition_reveal.json")
        reveal_bytes = reveal_path.read_bytes() if reveal_path.is_file() else None
        reveal_ref = str(reveal_path) if reveal_bytes is not None else None
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingGeometryError("Palette fit report is not valid JSON.") from exc
    if (
        report.get("schema_id") != "palette.diagnostics.recording_dish_rim_probe"
        or report.get("schema_version") != 1
        or report.get("status") != "provisional_visual_review_required"
        or report.get("fit_frozen_before_acquisition_reveal") is not True
    ):
        raise RecordingGeometryError("Palette fit report contract is unsupported.")
    source = _required_mapping(report.get("source"), label="fit_report.source")
    camera_serial = _required_text(
        source.get("camera_serial"), label="fit_report.source.camera_serial"
    )
    coordinate, arena, source_video_metadata = _source_camera_candidate_binding(
        zarr_path,
        expected_camera_serial=camera_serial,
        fit_source=source,
        arena_binding=arena_binding,
    )
    shape = _required_mapping(
        source.get("image_shape_px"), label="fit_report.source.image_shape_px"
    )
    if (
        int(shape.get("width", 0)) != coordinate["native_width_px"]
        or int(shape.get("height", 0)) != coordinate["native_height_px"]
    ):
        raise RecordingGeometryError(
            "Palette fit raster dimensions do not match the source-camera authority."
        )
    if source.get("pixel_contract") != "orange.camera.mono8.full_frame.v1":
        raise RecordingGeometryError(
            "Palette fit used an unsupported source pixel contract."
        )
    source_mode = str(source.get("mode") or "single_video")
    if source_mode == "single_video":
        if (
            int(source.get("frame_count", 0))
            != int(source_video_metadata.get("total_frames", 0))
            or int(source.get("video_size_bytes", 0))
            != int(
                _required_mapping(
                    source_video_metadata.get("file_fingerprint"),
                    label="root.source_video_metadata.file_fingerprint",
                ).get("size_bytes", 0)
            )
            or Path(str(source.get("video_path"))).name
            != str(source_video_metadata.get("source_video"))
        ):
            raise RecordingGeometryError(
                "Palette fit source video does not match the analysis Zarr source identity."
            )
    elif source_mode != "clipped_recording":
        raise RecordingGeometryError(
            f"Unsupported Palette fit source mode: {source_mode!r}."
        )

    consensus = _required_mapping(
        report.get("consensus_fit"), label="fit_report.consensus_fit"
    )
    observed_geometry, observed_values = _circle_mapping(
        consensus.get("geometry"), label="fit_report.consensus_fit.geometry"
    )
    if consensus.get("coordinate_space") != "camera_native_pixels":
        raise RecordingGeometryError(
            "Palette consensus is not in native camera pixels."
        )
    cx, cy, radius = observed_values
    width = int(coordinate["native_width_px"])
    height = int(coordinate["native_height_px"])
    if not (
        0.0 <= cx <= width and 0.0 <= cy <= height and radius <= max(width, height)
    ):
        raise RecordingGeometryError(
            "Palette consensus circle is outside its native raster."
        )

    windows = _required_mapping(report.get("windows"), label="fit_report.windows")
    if set(windows) != {"early", "middle", "late"}:
        raise RecordingGeometryError(
            "Palette fit must contain early, middle, and late windows."
        )
    normalized_windows: dict[str, Any] = {}
    window_values: list[tuple[float, float, float]] = []
    fit_method = str(report.get("fit_method") or "")
    for name in ("early", "middle", "late"):
        window = _required_mapping(windows[name], label=f"fit_report.windows.{name}")
        fit = _required_mapping(
            window.get("fit"), label=f"fit_report.windows.{name}.fit"
        )
        geometry, values = _circle_mapping(
            fit.get("geometry"), label=f"fit_report.windows.{name}.fit.geometry"
        )
        window_values.append(values)
        raw_candidates = fit.get("frozen_candidates")
        if raw_candidates is None:
            raw_candidates = []
        if not isinstance(raw_candidates, list) or not all(
            isinstance(item, Mapping) for item in raw_candidates
        ):
            raise RecordingGeometryError(
                f"Palette {name} frozen candidates must be a list of records."
            )
        if (
            fit_method.endswith("multicandidate_radial_edge_circle_v2")
            and not raw_candidates
        ):
            raise RecordingGeometryError(
                f"Palette {name} v2 fit did not preserve its frozen candidates."
            )
        normalized_candidates: list[dict[str, Any]] = []
        candidate_ids: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates):
            candidate_id = _required_text(
                raw_candidate.get("candidate_id"),
                label=f"fit_report.windows.{name}.candidate[{index}].candidate_id",
            )
            if candidate_id in candidate_ids:
                raise RecordingGeometryError(
                    f"Palette {name} frozen candidate IDs are not unique."
                )
            candidate_ids.add(candidate_id)
            candidate_geometry, _candidate_values = _circle_mapping(
                raw_candidate.get("geometry"),
                label=f"fit_report.windows.{name}.candidate[{index}].geometry",
            )
            if raw_candidate.get("coordinate_space") != "camera_native_pixels":
                raise RecordingGeometryError(
                    f"Palette {name} frozen candidate is not in native camera pixels."
                )
            normalized_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "geometry": candidate_geometry,
                    "coordinate_space": "camera_native_pixels",
                    "observed_feature_classification": _required_text(
                        raw_candidate.get("observed_feature_classification"),
                        label=(
                            f"fit_report.windows.{name}.candidate[{index}]."
                            "observed_feature_classification"
                        ),
                    ),
                    "angular_support_fraction": _required_finite(
                        raw_candidate.get("angular_support_fraction"),
                        label=f"fit_report.windows.{name}.candidate[{index}].support",
                    ),
                    "radial_residual_px": _required_finite(
                        raw_candidate.get("radial_residual_px"),
                        label=f"fit_report.windows.{name}.candidate[{index}].residual",
                    ),
                    "median_radial_gradient": _required_finite(
                        raw_candidate.get("median_radial_gradient"),
                        label=f"fit_report.windows.{name}.candidate[{index}].gradient",
                    ),
                    "evidence_score": _required_finite(
                        raw_candidate.get("evidence_score"),
                        label=f"fit_report.windows.{name}.candidate[{index}].score",
                    ),
                }
            )
        selected_candidate_id = fit.get("selected_candidate_id")
        if selected_candidate_id is not None:
            selected_candidate_id = _required_text(
                selected_candidate_id,
                label=f"fit_report.windows.{name}.fit.selected_candidate_id",
            )
            if selected_candidate_id not in candidate_ids:
                raise RecordingGeometryError(
                    f"Palette {name} selected candidate is not in frozen evidence."
                )
        raw_decoded_frames = window.get("decoded_frames") or []
        if not isinstance(raw_decoded_frames, list) or not all(
            isinstance(item, Mapping) for item in raw_decoded_frames
        ):
            raise RecordingGeometryError(
                f"Palette {name} decoded frame evidence is invalid."
            )
        if source_mode == "single_video":
            normalized_decoded_frames = [
                {
                    "frame_index": int(item.get("frame_index")),
                    "decoded_frame_sha256": _required_sha256(
                        item.get("decoded_frame_sha256"),
                        label=f"fit_report.windows.{name}.decoded_frame_sha256",
                    ),
                }
                for item in raw_decoded_frames
            ]
            frame_binding = {
                "center_frame": int(window.get("center_frame")),
                "frame_indices": [
                    int(value) for value in window.get("frame_indices", [])
                ],
            }
        else:
            normalized_decoded_frames = [
                {
                    "clip_id": _required_text(
                        item.get("clip_id"),
                        label=f"fit_report.windows.{name}.decoded_frame.clip_id",
                    ),
                    "clip_index": int(item.get("clip_index")),
                    "clip_local_frame_index": int(item.get("clip_local_frame_index")),
                    "recording_frame_id": int(item.get("recording_frame_id")),
                    "video_path": _required_text(
                        item.get("video_path"),
                        label=f"fit_report.windows.{name}.decoded_frame.video_path",
                    ),
                    "keyframe_path": _required_text(
                        item.get("keyframe_path"),
                        label=f"fit_report.windows.{name}.decoded_frame.keyframe_path",
                    ),
                    "decoded_frame_sha256": _required_sha256(
                        item.get("decoded_frame_sha256"),
                        label=f"fit_report.windows.{name}.decoded_frame_sha256",
                    ),
                }
                for item in raw_decoded_frames
            ]
            frame_binding = {
                "frame_coordinate": "one_based_recording_frame_id",
                "center_recording_frame_id": int(
                    window.get("center_recording_frame_id")
                ),
                "recording_frame_ids": [
                    int(value) for value in window.get("recording_frame_ids", [])
                ],
                "sampled_clip_ids": [
                    _required_text(value, label=f"fit_report.windows.{name}.clip_id")
                    for value in window.get("sampled_clip_ids", [])
                ],
            }
        normalized_windows[name] = {
            **frame_binding,
            "decoded_luma_sequence_sha256": _required_sha256(
                window.get("decoded_luma_sequence_sha256"),
                label=f"fit_report.windows.{name}.decoded_luma_sequence_sha256",
            ),
            "decoded_frames": normalized_decoded_frames,
            "composite_pixel_sha256": _required_sha256(
                window.get("composite_pixel_sha256"),
                label=f"fit_report.windows.{name}.composite_pixel_sha256",
            ),
            "geometry": geometry,
            "angular_support_fraction": _required_finite(
                fit.get("angular_support_fraction"),
                label=f"fit_report.windows.{name}.angular_support_fraction",
            ),
            "median_radial_gradient": _required_finite(
                fit.get("median_radial_gradient"),
                label=f"fit_report.windows.{name}.median_radial_gradient",
            ),
            "radial_residual_px": _required_finite(
                fit.get("radial_residual_px", 0.0),
                label=f"fit_report.windows.{name}.radial_residual_px",
            ),
            "selected_candidate_id": selected_candidate_id,
            "selection_reason": _required_text(
                fit.get("selection_reason", "legacy_unspecified_selection"),
                label=f"fit_report.windows.{name}.selection_reason",
            ),
            "frozen_candidates": normalized_candidates,
        }
        frame_values = (
            normalized_windows[name].get("frame_indices")
            if source_mode == "single_video"
            else normalized_windows[name].get("recording_frame_ids")
        )
        if not frame_values:
            raise RecordingGeometryError(
                f"Palette {name} fit has no source frame indices."
            )

    review_payload = _canonical_copy(review)
    if review_payload != {
        "status": "reviewer_accepted_for_offline_detection_gate_audit",
        "reviewer": review_payload.get("reviewer"),
        "reviewed_at_utc": review_payload.get("reviewed_at_utc"),
        "decision_source": "interactive_visual_review",
        "reviewed_feature": "visible_dish_top_rim_edge",
        "decision_scope": "candidate_and_detection_disagreement_audit_only",
    }:
        raise RecordingGeometryError("Palette fit review contract is invalid.")
    _required_text(review_payload.get("reviewer"), label="review.reviewer")
    _required_text(
        review_payload.get("reviewed_at_utc"), label="review.reviewed_at_utc"
    )

    xs = [value[0] for value in window_values]
    ys = [value[1] for value in window_values]
    radii = [value[2] for value in window_values]
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    acquisition_boundary_edge_support: dict[str, Any] = {
        "status": "not_measured",
        "reason": "acquisition_reveal_not_present_at_candidate_publication",
    }
    acquisition_reveal_binding: dict[str, Any] | None = None
    if reveal_bytes is not None:
        try:
            reveal = json.loads(reveal_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordingGeometryError(
                "Acquisition reveal is not valid JSON."
            ) from exc
        reveal_fit = _required_mapping(
            reveal.get("fit_report"), label="acquisition_reveal.fit_report"
        )
        if (
            reveal.get("schema_id")
            != "palette.diagnostics.recording_dish_rim_probe.acquisition_reveal"
            or reveal.get("schema_version") != 1
            or _required_sha256(
                reveal_fit.get("sha256"),
                label="acquisition_reveal.fit_report.sha256",
            )
            != report_sha256
        ):
            raise RecordingGeometryError(
                "Acquisition reveal does not bind the exact frozen fit report."
            )
        acquisition_boundary_edge_support = _canonical_copy(
            _required_mapping(
                reveal.get("acquisition_boundary_edge_support"),
                label="acquisition_reveal.acquisition_boundary_edge_support",
            )
        )
        if (
            acquisition_boundary_edge_support.get("status") != "measured"
            or acquisition_boundary_edge_support.get("fit_frozen_before_measurement")
            is not True
        ):
            raise RecordingGeometryError(
                "Acquisition reveal lacks post-freeze boundary support evidence."
            )
        acquisition_reveal_binding = {
            "path": _required_text(reveal_ref, label="acquisition_reveal_ref"),
            "sha256": hashlib.sha256(reveal_bytes).hexdigest(),
        }
    palette_fit_source: dict[str, Any] = {
        "fit_report_path": report_ref,
        "fit_report_sha256": report_sha256,
        "fit_report_schema_id": report.get("schema_id"),
        "fit_report_schema_version": report.get("schema_version"),
        "fit_method": fit_method,
        "blind_to_acquisition_geometry": (
            _required_mapping(
                report.get("parameters"), label="fit_report.parameters"
            ).get("acquisition_geometry_available_to_fitter")
            is False
        ),
        "probe_declared_target_feature": report.get("target_feature"),
        "reviewed_semantic_correction": {
            "status": "reviewer_corrected_probe_feature_label",
            "reviewed_feature": "visible_dish_top_rim_edge",
            "reason": "visual_review_found_fit_on_top_rim_not_inner_water_side",
        },
        "review_montage_path": montage_ref,
        "review_montage_sha256": hashlib.sha256(montage_bytes).hexdigest(),
        "windows": normalized_windows,
        "fit_evidence_contract": _canonical_copy(
            report.get("fit_evidence_contract")
            or {
                "all_window_candidates_frozen": False,
                "candidate_geometry_revealed_to_acquisition_fit": False,
                "candidate_feature_classification": "legacy_unspecified",
                "selection_scope": "legacy_review_only",
            }
        ),
        "temporal_stability_px": {
            "center_x_range": max(xs) - min(xs),
            "center_y_range": max(ys) - min(ys),
            "radius_range": max(radii) - min(radii),
        },
        "acquisition_boundary_edge_support": acquisition_boundary_edge_support,
        "acquisition_reveal_binding": acquisition_reveal_binding,
    }
    if source_mode == "single_video":
        palette_fit_source.update(
            {
                "source_video_path": source.get("video_path"),
                "source_video_size_bytes": int(source.get("video_size_bytes")),
                "source_summary_sha256": _required_sha256(
                    source.get("summary_sha256"),
                    label="fit_report.source.summary_sha256",
                ),
            }
        )
    else:
        source_collection = _canonical_copy(source)
        palette_fit_source.update(
            {
                "source_mode": source_mode,
                "source_collection": source_collection,
                "source_collection_sha256": _payload_sha256(source_collection),
                "validated_collection_binding": _canonical_copy(source_video_metadata),
            }
        )
    if embedded_evidence is not None:
        palette_fit_source.update(
            {
                "review_evidence_storage": "embedded_zarr_fit_review_run_v1",
                "fit_review_run": embedded_evidence.run_name,
                "fit_review_record_sha256": (embedded_evidence.review_record_sha256),
            }
        )
    record = {
        "schema_id": CANDIDATE_RECORD_SCHEMA_ID,
        "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "candidate_kind": PALETTE_CANDIDATE_KIND,
        "arena_binding": arena,
        "observed_boundary": {
            "coordinate_space": "camera_native_pixels",
            "target_plane": "dish_top_rim",
            "observed_feature": "visible_dish_top_rim_edge",
            "interpretation": (
                "recording_image_observation_not_acquisition_physical_inner_rim"
            ),
            "geometry": observed_geometry,
        },
        "valid_detection_region": {
            "coordinate_space": "camera_native_pixels",
            "purpose": "bounding_box_centroid_detection_gating",
            "geometry": observed_geometry,
            "derivation": "direct_from_reviewed_visible_dish_top_rim_edge",
            "boundary_inclusion": "inclusive",
            "additional_palette_tolerance_px": 0.0,
            "is_final_acquisition_tolerance": False,
        },
        "coordinate_binding": coordinate,
        "palette_fit_source": palette_fit_source,
        "review": review_payload,
        "candidate_policy": {
            "publication_role": "candidate_only",
            "operationally_selected": False,
            "legacy_dish_mask_projection_written": False,
            "detection_gate_applied": False,
            "eligible_for_detection_disagreement_audit": True,
            "requires_explicit_selection_before_gating": True,
        },
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    normalized = _canonical_copy(record)
    validate_palette_geometry_candidate_record(normalized)
    return normalized


def validate_palette_geometry_candidate_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("schema_id") != CANDIDATE_RECORD_SCHEMA_ID
        or record.get("schema_version") != CANDIDATE_RECORD_SCHEMA_VERSION
    ):
        raise RecordingGeometryError("Unsupported arena-geometry candidate schema.")
    if record.get("candidate_kind") != PALETTE_CANDIDATE_KIND:
        raise RecordingGeometryError("Unsupported Palette candidate kind.")
    arena = _required_mapping(record.get("arena_binding"), label="arena_binding")
    for name in ("rig_id", "canvas_name", "arena_id", "camera_serial"):
        _required_text(arena.get(name), label=f"arena_binding.{name}")
    observed = _required_mapping(
        record.get("observed_boundary"), label="observed_boundary"
    )
    gate = _required_mapping(
        record.get("valid_detection_region"), label="valid_detection_region"
    )
    if (
        observed.get("coordinate_space") != "camera_native_pixels"
        or observed.get("target_plane") != "dish_top_rim"
        or observed.get("observed_feature") != "visible_dish_top_rim_edge"
        or observed.get("interpretation")
        != "recording_image_observation_not_acquisition_physical_inner_rim"
    ):
        raise RecordingGeometryError("Palette observed-boundary semantics are invalid.")
    observed_geometry, _ = _circle_mapping(
        observed.get("geometry"), label="observed_boundary.geometry"
    )
    gate_geometry, _ = _circle_mapping(
        gate.get("geometry"), label="valid_detection_region.geometry"
    )
    if (
        gate.get("coordinate_space") != "camera_native_pixels"
        or gate.get("purpose") != "bounding_box_centroid_detection_gating"
        or gate.get("derivation") != "direct_from_reviewed_visible_dish_top_rim_edge"
        or gate.get("boundary_inclusion") != "inclusive"
        or gate.get("additional_palette_tolerance_px") != 0.0
        or gate.get("is_final_acquisition_tolerance") is not False
        or gate_geometry != observed_geometry
    ):
        raise RecordingGeometryError("Palette detection-gate derivation is invalid.")
    coordinate = _required_mapping(
        record.get("coordinate_binding"), label="coordinate_binding"
    )
    if (
        coordinate.get("space_id") != "source_camera_image_px"
        or coordinate.get("profile_id") != "source_camera_image_px.top_left_y_down.v1"
        or coordinate.get("pixel_convention") != "continuous"
        or coordinate.get("units") != "px"
        or coordinate.get("origin") != "top_left"
        or coordinate.get("positive_x") != "right"
        or coordinate.get("positive_y") != "down"
    ):
        raise RecordingGeometryError(
            "Palette source-camera coordinate binding is invalid."
        )
    for name in ("native_width_px", "native_height_px"):
        value = coordinate.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RecordingGeometryError(f"coordinate_binding.{name} must be positive.")
    _required_sha256(
        coordinate.get("pixel_frame_record_sha256"),
        label="coordinate_binding.pixel_frame_record_sha256",
    )
    authority_kind = coordinate.get("source_camera_frame_authority_kind")
    if authority_kind is None:
        expected_ref = (
            f"/analysis/coordinate_frames/source_camera/{arena['camera_serial']}"
            "/continuous@pixel_frame_authority"
        )
    elif authority_kind == CLIPPED_ACQUISITION_FRAME_AUTHORITY_KIND:
        expected_ref = (
            f"/analysis/acquisition_camera_frames/{arena['camera_serial']}"
            "@acquisition_camera_frame"
        )
    elif authority_kind == LEGACY_CLIPPED_SNAPSHOT_FRAME_AUTHORITY_KIND:
        expected_ref = (
            f"/recording_geometry_snapshot/camera_runtime/{arena['camera_serial']}"
            "/coordinate_frame@recording_snapshot_sha256"
        )
    else:
        raise RecordingGeometryError(
            "Palette candidate uses an unsupported source-camera frame authority."
        )
    if coordinate.get("pixel_frame_record_ref") != expected_ref:
        raise RecordingGeometryError(
            "Palette candidate binds the wrong source-camera frame."
        )
    source = _required_mapping(
        record.get("palette_fit_source"), label="palette_fit_source"
    )
    for name in ("fit_report_path", "fit_method", "review_montage_path"):
        _required_text(source.get(name), label=f"palette_fit_source.{name}")
    for name in ("fit_report_sha256", "review_montage_sha256"):
        _required_sha256(source.get(name), label=f"palette_fit_source.{name}")
    source_mode = str(source.get("source_mode") or "single_video")
    if source_mode == "single_video":
        _required_text(
            source.get("source_video_path"),
            label="palette_fit_source.source_video_path",
        )
        _required_sha256(
            source.get("source_summary_sha256"),
            label="palette_fit_source.source_summary_sha256",
        )
        if authority_kind is not None:
            raise RecordingGeometryError(
                "Single-video Palette candidates require persisted pixel-frame authority."
            )
    elif source_mode == "clipped_recording":
        if authority_kind not in {
            CLIPPED_ACQUISITION_FRAME_AUTHORITY_KIND,
            LEGACY_CLIPPED_SNAPSHOT_FRAME_AUTHORITY_KIND,
        }:
            raise RecordingGeometryError(
                "Clipped Palette candidates require persisted acquisition-camera "
                "authority or the legacy recording-snapshot authority."
            )
        collection = _required_mapping(
            source.get("source_collection"),
            label="palette_fit_source.source_collection",
        )
        if collection.get("mode") != "clipped_recording":
            raise RecordingGeometryError(
                "Palette clipped source collection has the wrong mode."
            )
        for name in (
            "recording_id",
            "camera_serial",
            "recording_clip_index_path",
            "recording_geometry_snapshot_path",
        ):
            _required_text(collection.get(name), label=f"source_collection.{name}")
        for name in (
            "recording_clip_index_sha256",
            "recording_geometry_snapshot_sha256",
        ):
            _required_sha256(collection.get(name), label=f"source_collection.{name}")
        if collection.get("camera_serial") != arena.get("camera_serial"):
            raise RecordingGeometryError(
                "Palette clipped source camera differs from its arena binding."
            )
        if _required_sha256(
            source.get("source_collection_sha256"),
            label="palette_fit_source.source_collection_sha256",
        ).removeprefix("sha256:") != _payload_sha256(collection):
            raise RecordingGeometryError(
                "Palette clipped source collection digest is stale."
            )
        validated = _required_mapping(
            source.get("validated_collection_binding"),
            label="palette_fit_source.validated_collection_binding",
        )
        for name in (
            "recording_clip_index_sha256",
            "recording_geometry_snapshot_sha256",
        ):
            if validated.get(name) != collection.get(name):
                raise RecordingGeometryError(
                    "Palette clipped source validation does not match fit evidence."
                )
    else:
        raise RecordingGeometryError(
            f"Unsupported Palette candidate source mode: {source_mode!r}."
        )
    review_storage = source.get("review_evidence_storage")
    if review_storage is not None:
        if review_storage != "embedded_zarr_fit_review_run_v1":
            raise RecordingGeometryError(
                "Palette fit review evidence storage contract is unsupported."
            )
        fit_review_run = _required_text(
            source.get("fit_review_run"),
            label="palette_fit_source.fit_review_run",
        )
        if Path(fit_review_run).name != fit_review_run:
            raise RecordingGeometryError("Palette fit-review run ID is unsafe.")
        _required_sha256(
            source.get("fit_review_record_sha256"),
            label="palette_fit_source.fit_review_record_sha256",
        )
        expected_prefix = f"analysis/{FIT_REVIEW_RUNS_PARENT}/{fit_review_run}/"
        if not source["fit_report_path"].startswith(expected_prefix) or not source[
            "review_montage_path"
        ].startswith(expected_prefix):
            raise RecordingGeometryError(
                "Palette candidate does not bind its embedded fit-review run."
            )
    if source.get("blind_to_acquisition_geometry") is not True:
        raise RecordingGeometryError(
            "Palette fit was not blind to acquisition geometry."
        )
    correction = _required_mapping(
        source.get("reviewed_semantic_correction"),
        label="palette_fit_source.reviewed_semantic_correction",
    )
    if correction != {
        "status": "reviewer_corrected_probe_feature_label",
        "reviewed_feature": "visible_dish_top_rim_edge",
        "reason": "visual_review_found_fit_on_top_rim_not_inner_water_side",
    }:
        raise RecordingGeometryError("Palette fit semantic correction is invalid.")
    windows = _required_mapping(
        source.get("windows"), label="palette_fit_source.windows"
    )
    if set(windows) != {"early", "middle", "late"}:
        raise RecordingGeometryError("Palette fit source windows are incomplete.")
    evidence_contract = _required_mapping(
        source.get("fit_evidence_contract"),
        label="palette_fit_source.fit_evidence_contract",
    )
    all_candidates_frozen = evidence_contract.get("all_window_candidates_frozen")
    if type(all_candidates_frozen) is not bool:
        raise RecordingGeometryError(
            "Palette fit evidence must declare whether every candidate was frozen."
        )
    if (
        evidence_contract.get("candidate_geometry_revealed_to_acquisition_fit")
        is not False
    ):
        raise RecordingGeometryError(
            "Palette fit evidence was not frozen blind to acquisition geometry."
        )
    for name, raw_window in windows.items():
        window = _required_mapping(
            raw_window, label=f"palette_fit_source.windows.{name}"
        )
        _circle_mapping(window.get("geometry"), label=f"{name}.geometry")
        for metric in (
            "angular_support_fraction",
            "median_radial_gradient",
            "radial_residual_px",
        ):
            value = _required_finite(window.get(metric), label=f"{name}.{metric}")
            if metric == "radial_residual_px" and value < 0:
                raise RecordingGeometryError(
                    f"Palette {name} radial residual cannot be negative."
                )
        raw_candidates = window.get("frozen_candidates")
        if not isinstance(raw_candidates, list):
            raise RecordingGeometryError(
                f"Palette {name} frozen candidates must be a list."
            )
        if all_candidates_frozen and not raw_candidates:
            raise RecordingGeometryError(
                f"Palette {name} claims frozen evidence without candidates."
            )
        ids: list[str] = []
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _required_mapping(
                raw_candidate,
                label=f"palette_fit_source.windows.{name}.candidate[{index}]",
            )
            ids.append(
                _required_text(
                    candidate.get("candidate_id"),
                    label=f"{name}.candidate[{index}].candidate_id",
                )
            )
            _circle_mapping(
                candidate.get("geometry"),
                label=f"{name}.candidate[{index}].geometry",
            )
            if (
                candidate.get("coordinate_space") != "camera_native_pixels"
                or candidate.get("observed_feature_classification")
                != "unclassified_concentric_rim_edge"
            ):
                raise RecordingGeometryError(
                    f"Palette {name} frozen candidate semantics are invalid."
                )
            for metric in (
                "angular_support_fraction",
                "radial_residual_px",
                "median_radial_gradient",
                "evidence_score",
            ):
                value = _required_finite(
                    candidate.get(metric),
                    label=f"{name}.candidate[{index}].{metric}",
                )
                if metric == "radial_residual_px" and value < 0:
                    raise RecordingGeometryError(
                        f"Palette {name} candidate residual cannot be negative."
                    )
        if len(ids) != len(set(ids)):
            raise RecordingGeometryError(
                f"Palette {name} frozen candidate IDs are not unique."
            )
        selected_id = window.get("selected_candidate_id")
        if selected_id is not None and selected_id not in ids:
            raise RecordingGeometryError(
                f"Palette {name} selected candidate is not frozen evidence."
            )
        _required_text(
            window.get("selection_reason"),
            label=f"palette_fit_source.windows.{name}.selection_reason",
        )
        decoded_frames = window.get("decoded_frames")
        if not isinstance(decoded_frames, list):
            raise RecordingGeometryError(
                f"Palette {name} decoded frame evidence must be a list."
            )
        frame_coordinate = str(
            window.get("frame_coordinate") or "zero_based_video_frame_index"
        )
        if frame_coordinate == "zero_based_video_frame_index":
            expected_frames = [int(value) for value in window.get("frame_indices", [])]
            observed_frames = [
                int(
                    _required_mapping(item, label=f"{name}.decoded_frame").get(
                        "frame_index"
                    )
                )
                for item in decoded_frames
            ]
        elif frame_coordinate == "one_based_recording_frame_id":
            expected_frames = [
                int(value) for value in window.get("recording_frame_ids", [])
            ]
            observed_frames = [
                int(
                    _required_mapping(item, label=f"{name}.decoded_frame").get(
                        "recording_frame_id"
                    )
                )
                for item in decoded_frames
            ]
            for item in decoded_frames:
                decoded = _required_mapping(item, label=f"{name}.decoded_frame")
                _required_text(decoded.get("clip_id"), label=f"{name}.clip_id")
                _required_text(decoded.get("video_path"), label=f"{name}.video_path")
                _required_text(
                    decoded.get("keyframe_path"), label=f"{name}.keyframe_path"
                )
        else:
            raise RecordingGeometryError(
                f"Palette {name} uses an unsupported frame coordinate."
            )
        if decoded_frames and observed_frames != expected_frames:
            raise RecordingGeometryError(
                f"Palette {name} decoded frame hashes do not cover source frames exactly."
            )
        for item in decoded_frames:
            _required_sha256(
                _required_mapping(item, label=f"{name}.decoded_frame").get(
                    "decoded_frame_sha256"
                ),
                label=f"{name}.decoded_frame_sha256",
            )
    stability = _required_mapping(
        source.get("temporal_stability_px"),
        label="palette_fit_source.temporal_stability_px",
    )
    for name in ("center_x_range", "center_y_range", "radius_range"):
        if (
            _required_finite(stability.get(name), label=f"temporal_stability_px.{name}")
            < 0
        ):
            raise RecordingGeometryError(
                "Palette temporal stability ranges cannot be negative."
            )
    boundary_support = _required_mapping(
        source.get("acquisition_boundary_edge_support"),
        label="palette_fit_source.acquisition_boundary_edge_support",
    )
    support_status = boundary_support.get("status")
    reveal_binding = source.get("acquisition_reveal_binding")
    if support_status == "measured":
        if (
            boundary_support.get("method") != "fixed_circle_radial_gradient_support_v1"
            or boundary_support.get("fit_frozen_before_measurement") is not True
            or boundary_support.get("coordinate_space") != "camera_native_pixels"
        ):
            raise RecordingGeometryError(
                "Acquisition boundary support evidence is not frozen native-camera evidence."
            )
        _circle_mapping(
            boundary_support.get("geometry"),
            label="acquisition_boundary_edge_support.geometry",
        )
        _required_sha256(
            boundary_support.get("source_observation_sha256"),
            label="acquisition_boundary_edge_support.source_observation_sha256",
        )
        support_windows = _required_mapping(
            boundary_support.get("windows"),
            label="acquisition_boundary_edge_support.windows",
        )
        if set(support_windows) != {"early", "middle", "late"}:
            raise RecordingGeometryError(
                "Acquisition boundary support windows are incomplete."
            )
        for window_name, raw_support in support_windows.items():
            window_support = _required_mapping(
                raw_support,
                label=f"acquisition_boundary_edge_support.windows.{window_name}",
            )
            if (
                window_support.get("status") != "measured"
                or window_support.get("geometry_frozen") is not True
                or window_support.get("method")
                != "fixed_circle_radial_gradient_support_v1"
            ):
                raise RecordingGeometryError(
                    f"Acquisition boundary support for {window_name} is invalid."
                )
            for metric in (
                "angular_edge_support_fraction",
                "median_radial_gradient",
                "median_absolute_radial_offset_px",
                "signed_median_radial_offset_px",
            ):
                _required_finite(
                    window_support.get(metric),
                    label=f"acquisition boundary support {window_name}.{metric}",
                )
        reveal = _required_mapping(
            reveal_binding,
            label="palette_fit_source.acquisition_reveal_binding",
        )
        reveal_path = _required_text(
            reveal.get("path"), label="acquisition_reveal_binding.path"
        )
        if review_storage is not None and not reveal_path.startswith(expected_prefix):
            raise RecordingGeometryError(
                "Acquisition reveal does not bind the embedded fit-review run."
            )
        _required_sha256(
            reveal.get("sha256"), label="acquisition_reveal_binding.sha256"
        )
    elif support_status == "not_measured":
        _required_text(boundary_support.get("reason"), label="boundary_support.reason")
        if reveal_binding is not None:
            raise RecordingGeometryError(
                "Unmeasured acquisition support cannot carry a reveal binding."
            )
    else:
        raise RecordingGeometryError(
            "Acquisition boundary support status must be measured or not_measured."
        )
    review = _required_mapping(record.get("review"), label="review")
    if (
        review.get("status") != "reviewer_accepted_for_offline_detection_gate_audit"
        or review.get("decision_source") != "interactive_visual_review"
        or review.get("reviewed_feature") != "visible_dish_top_rim_edge"
        or review.get("decision_scope")
        != "candidate_and_detection_disagreement_audit_only"
    ):
        raise RecordingGeometryError("Palette candidate review is invalid.")
    _required_text(review.get("reviewer"), label="review.reviewer")
    _required_text(review.get("reviewed_at_utc"), label="review.reviewed_at_utc")
    policy = _required_mapping(record.get("candidate_policy"), label="candidate_policy")
    if dict(policy) != {
        "publication_role": "candidate_only",
        "operationally_selected": False,
        "legacy_dish_mask_projection_written": False,
        "detection_gate_applied": False,
        "eligible_for_detection_disagreement_audit": True,
        "requires_explicit_selection_before_gating": True,
    }:
        raise RecordingGeometryError("Palette candidate policy is not fail-closed.")
    if record.get("canonicalization") != "canonical_json_sort_keys_v1":
        raise RecordingGeometryError("Candidate canonicalization is unsupported.")
    if _canonical_copy(record) != dict(record):
        raise RecordingGeometryError(
            "Candidate record is not strict canonical JSON data."
        )


def validate_arena_geometry_candidate_record(record: Mapping[str, Any]) -> None:
    kind = record.get("candidate_kind")
    if kind == ACQUISITION_CANDIDATE_KIND:
        validate_acquisition_geometry_candidate_record(record)
        return
    if kind == PALETTE_CANDIDATE_KIND:
        validate_palette_geometry_candidate_record(record)
        return
    raise RecordingGeometryError(
        f"Unsupported arena-geometry candidate kind: {kind!r}."
    )


def _recovery_binding(verified: VerifiedRecordingGeometryRecovery) -> dict[str, Any]:
    target = _required_mapping(verified.receipt.get("target"), label="receipt.target")
    return {
        "receipt_schema_id": verified.receipt.get("schema_id"),
        "receipt_id": verified.receipt.get("receipt_id"),
        "receipt_sha256": verified.receipt_sha256,
        "authority": verified.receipt.get("authority"),
        "reason": verified.receipt.get("recovery_reason"),
        "target_h5_sha256": target.get("h5_sha256"),
        "target_session_uuid": target.get("session_uuid"),
        "h5_geometry_capture_status": target.get("h5_geometry_capture_status"),
        "producer_artifacts_mutated": False,
    }


def _bound_recovered_mask(
    source_zarr: Path,
    verified: VerifiedRecordingGeometryRecovery,
) -> BoundRegisteredDishMask:
    receipt_recording_root = verified.receipt_path.parent.parent
    zarr_recording_root = source_zarr.parent.parent
    if receipt_recording_root != zarr_recording_root:
        raise RecordingGeometryError(
            "Recovery receipt and analysis Zarr must be siblings in one recording root."
        )
    mask: RegisteredDishMask = registered_dish_mask_from_verified_recovery(verified)
    return _bind_mask_to_zarr(source_zarr, mask)


def _bind_mask_to_zarr(
    source_zarr: Path,
    mask: RegisteredDishMask,
) -> BoundRegisteredDishMask:
    """Bind one already verified producer mask to Palette's persisted pixel frame."""

    root = open_zarr_root(source_zarr, mode="r")
    _ownership, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=mask.key.camera_serial,
    )
    frame_path = (
        f"analysis/coordinate_frames/source_camera/{mask.key.camera_serial}/continuous"
    )
    try:
        frame_node = root[frame_path]
    except KeyError as exc:
        raise RecordingGeometryError(
            f"Analysis Zarr lacks canonical continuous source-camera authority {frame_path}."
        ) from exc
    source_frame = load_source_camera_pixel_frame_authority(
        frame_node,
        acquisition_frame=acquisition,
    )
    return bind_registered_dish_mask_to_source_camera_frame(mask, source_frame)


def _producer_native_mask(
    *,
    source_path: Path,
    source_kind: str,
    camera_serial: str,
    arena_id: str,
) -> RegisteredDishMask:
    if source_kind == "orange_recording_folder":
        collection = load_registered_dish_masks_from_recording_folder(
            source_path,
            policy=GeometryLoadPolicy.REQUIRED,
        )
    elif source_kind == "citrus_h5":
        collection = load_registered_dish_masks_from_citrus_h5(
            source_path,
            policy=GeometryLoadPolicy.REQUIRED,
        )
    else:
        raise RecordingGeometryError(
            f"Unsupported producer-native geometry source kind: {source_kind!r}."
        )
    if collection.mask_geometry_status is not MaskGeometryStatus.VALID:
        details = "; ".join(issue.message for issue in collection.issues)
        raise RecordingGeometryError(
            "Producer-native recording geometry is not valid"
            + (f": {details}" if details else ".")
        )
    matches = [
        mask
        for key, mask in collection.masks.items()
        if key.camera_serial == camera_serial and key.arena_id == arena_id
    ]
    if len(matches) != 1:
        raise RecordingGeometryError(
            "Producer-native recording geometry must resolve exactly one requested "
            f"camera/arena pair; camera={camera_serial!r}, arena={arena_id!r}, "
            f"matches={len(matches)}."
        )
    mask = matches[0]
    if mask.source_kind != source_kind:
        raise RecordingGeometryError(
            "Producer-native mask source kind differs from the selected adapter."
        )
    return mask


def _record_from_producer_source_and_zarr(
    source_zarr: Path,
    *,
    source_path: Path,
    source_kind: str,
    camera_serial: str,
    arena_id: str,
) -> tuple[dict[str, Any], str]:
    mask = _producer_native_mask(
        source_path=source_path,
        source_kind=source_kind,
        camera_serial=camera_serial,
        arena_id=arena_id,
    )
    bound = _bind_mask_to_zarr(source_zarr, mask)
    record = build_acquisition_geometry_candidate_record(
        bound,
        recovery_binding=None,
    )
    return record, _payload_sha256(record)


def _recording_folder_input_artifacts(source_path: Path) -> tuple[dict[str, Any], ...]:
    snapshot = source_path / "recording_snapshot.json"
    if not snapshot.is_file():
        raise RecordingGeometryError(
            f"Producer-native recording snapshot is missing: {snapshot}"
        )
    try:

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant {value!r}")

        payload = json.loads(
            snapshot.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RecordingGeometryError(
            f"Producer-native recording snapshot is invalid: {exc}"
        ) from exc
    pointer = _required_mapping(
        (
            payload.get("recording_geometry_contract")
            if isinstance(payload, Mapping)
            else None
        ),
        label="recording_snapshot.recording_geometry_contract",
    )
    relative = Path(
        _required_text(pointer.get("relative_path"), label="geometry relative_path")
    )
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise RecordingGeometryError(
            "Recording geometry contract pointer must stay below the recording root."
        )
    contract = (source_path / relative).resolve()
    try:
        contract.relative_to(source_path)
    except ValueError as exc:
        raise RecordingGeometryError(
            "Recording geometry contract pointer escapes the recording root."
        ) from exc
    if not contract.is_file():
        raise RecordingGeometryError(
            f"Producer-native recording geometry contract is missing: {contract}"
        )
    expected = _required_sha256(
        pointer.get("sha256"),
        label="recording geometry contract pointer sha256",
    ).removeprefix("sha256:")
    observed = _file_sha256(contract)
    if observed != expected:
        raise RecordingGeometryError(
            "Recording geometry contract changed after adapter validation."
        )
    return (
        {
            "role": "recording_snapshot_geometry_pointer",
            "path": str(snapshot),
            "sha256": _file_sha256(snapshot),
        },
        {
            "role": "orange_recording_geometry_contract",
            "path": str(contract),
            "sha256": observed,
        },
    )


def _producer_geometry_folder(recording_root: Path) -> Path:
    """Resolve the one fixed producer bundle layout below a recording root."""

    direct_snapshot = recording_root / "recording_snapshot.json"
    archived_root = recording_root / "raw" / "recording_geometry_bundle"
    archived_present = archived_root.exists()
    if direct_snapshot.is_file() and archived_present:
        raise RecordingGeometryError(
            "Producer geometry is ambiguous: both the recording root and the fixed "
            "organized recording_geometry_bundle contain geometry inputs."
        )
    if not archived_present:
        return recording_root
    resolved = archived_root.resolve()
    try:
        resolved.relative_to(recording_root)
    except ValueError as exc:
        raise RecordingGeometryError(
            "Organized recording geometry bundle escapes the recording root."
        ) from exc
    return resolved


def _record_from_receipt_and_zarr(
    source_zarr: Path,
    receipt_path: Path,
) -> tuple[VerifiedRecordingGeometryRecovery, dict[str, Any], str]:
    verified = validate_recording_geometry_recovery_receipt(receipt_path)
    bound = _bound_recovered_mask(source_zarr, verified)
    record = build_acquisition_geometry_candidate_record(
        bound,
        recovery_binding=_recovery_binding(verified),
    )
    return verified, record, _payload_sha256(record)


def plan_recovered_acquisition_geometry_candidate(
    *,
    source_zarr: str | Path,
    receipt_path: str | Path,
) -> ArenaGeometryCandidatePlan:
    """Plan one deterministic candidate without writing the analysis Zarr."""

    zarr_path = Path(source_zarr).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    verified, record, digest = _record_from_receipt_and_zarr(zarr_path, receipt)
    candidate_id = f"arena-geometry-acquisition-{digest[:24]}"
    run_name = candidate_id
    params = {
        "algorithm_version": PUBLISH_ALGORITHM_VERSION,
        "candidate_id": candidate_id,
        "candidate_record_sha256": digest,
        "candidate_kind": CANDIDATE_KIND,
        "operational_selection": "not_performed",
    }
    provenance = build_writer_run_provenance(
        command="publish_acquisition_geometry_candidate",
        params=params,
        input_run_ids={},
        input_artifacts=(
            {
                "role": "recording_geometry_recovery_receipt",
                "path": str(receipt),
                "sha256": verified.receipt_sha256,
            },
            {
                "role": "orange_recording_geometry_contract",
                "path": str(
                    verified.evidence.bundle_root / "recording_geometry_contract.json"
                ),
                "sha256": verified.evidence.bundle_verification.contract_sha256,
            },
        ),
        include_system_context=False,
    )
    provenance_validation = validate_run_provenance(provenance)
    if not provenance_validation.valid:
        raise RuntimeError(
            "Acquisition candidate publication provenance is invalid: "
            f"{provenance_validation.errors}"
        )
    return ArenaGeometryCandidatePlan(
        source_zarr=zarr_path,
        receipt_path=receipt,
        receipt_sha256=verified.receipt_sha256,
        candidate_id=candidate_id,
        candidate_record_sha256=digest,
        candidate_record=record,
        run_name=run_name,
        target_run_path=zarr_path / "analysis" / CANDIDATE_RUNS_PARENT / run_name,
        run_provenance=provenance,
        candidate_kind=ACQUISITION_CANDIDATE_KIND,
    )


def plan_producer_native_acquisition_geometry_candidate(
    *,
    source_zarr: str | Path,
    camera_serial: str,
    arena_id: str,
    recording_folder: str | Path | None = None,
    citrus_h5: str | Path | None = None,
) -> ArenaGeometryCandidatePlan:
    """Plan one ordinary producer-linked candidate without a recovery receipt."""

    if (recording_folder is None) == (citrus_h5 is None):
        raise RecordingGeometryError(
            "Choose exactly one producer geometry source: recording_folder or citrus_h5."
        )
    zarr_path = Path(source_zarr).expanduser().resolve()
    recording_root = zarr_path.parent.parent
    serial = _required_text(camera_serial, label="camera_serial")
    arena = _required_text(arena_id, label="arena_id")
    if recording_folder is not None:
        supplied_recording_root = Path(recording_folder).expanduser().resolve()
        if supplied_recording_root != recording_root:
            raise RecordingGeometryError(
                "Producer recording folder and analysis Zarr must belong to the same "
                "recording root."
            )
        source = _producer_geometry_folder(supplied_recording_root)
        source_kind = "orange_recording_folder"
        input_artifacts = _recording_folder_input_artifacts(source)
    else:
        assert citrus_h5 is not None
        source = Path(citrus_h5).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Producer Citrus H5 is missing: {source}")
        try:
            source.relative_to(recording_root)
        except ValueError as exc:
            raise RecordingGeometryError(
                "Producer Citrus H5 and analysis Zarr must belong to the same "
                "recording root."
            ) from exc
        source_kind = "citrus_h5"
        input_artifacts = (
            {
                "role": "citrus_h5_recording_geometry_contract",
                "path": str(source),
                "sha256": _file_sha256(source),
            },
        )
    record, digest = _record_from_producer_source_and_zarr(
        zarr_path,
        source_path=source,
        source_kind=source_kind,
        camera_serial=serial,
        arena_id=arena,
    )
    source_record = _required_mapping(
        record.get("acquisition_source"), label="acquisition_source"
    )
    if source_record.get("recovery_binding") is not None:
        raise RecordingGeometryError(
            "Producer-native planning cannot carry a recovery receipt binding."
        )
    candidate_id = f"arena-geometry-acquisition-{digest[:24]}"
    provenance = build_writer_run_provenance(
        command="publish_producer_native_acquisition_geometry_candidate",
        params={
            "algorithm_version": PUBLISH_ALGORITHM_VERSION,
            "candidate_id": candidate_id,
            "candidate_record_sha256": digest,
            "candidate_kind": CANDIDATE_KIND,
            "source_kind": source_kind,
            "camera_serial": serial,
            "arena_id": arena,
            "operational_selection": "not_performed",
        },
        input_run_ids={},
        input_artifacts=input_artifacts,
        include_system_context=False,
    )
    provenance_validation = validate_run_provenance(provenance)
    if not provenance_validation.valid:
        raise RuntimeError(
            "Producer-native candidate publication provenance is invalid: "
            f"{provenance_validation.errors}"
        )
    return ArenaGeometryCandidatePlan(
        source_zarr=zarr_path,
        receipt_path=source,
        receipt_sha256=_required_sha256(
            source_record.get("source_contract_sha256"),
            label="acquisition_source.source_contract_sha256",
        ),
        candidate_id=candidate_id,
        candidate_record_sha256=digest,
        candidate_record=record,
        run_name=candidate_id,
        target_run_path=zarr_path / "analysis" / CANDIDATE_RUNS_PARENT / candidate_id,
        run_provenance=provenance,
        candidate_kind=ACQUISITION_CANDIDATE_KIND,
    )


def plan_reviewed_palette_geometry_candidate(
    *,
    source_zarr: str | Path,
    fit_report_path: str | Path | None = None,
    montage_path: str | Path | None = None,
    fit_review_run: str | None = None,
    reviewer: str,
    reviewed_at_utc: str,
    arena_binding: Mapping[str, Any] | None = None,
) -> ArenaGeometryCandidatePlan:
    """Plan a reviewed image-derived candidate without selecting or applying it."""

    zarr_path = Path(source_zarr).expanduser().resolve()
    review = {
        "status": "reviewer_accepted_for_offline_detection_gate_audit",
        "reviewer": _required_text(reviewer, label="reviewer"),
        "reviewed_at_utc": _required_text(reviewed_at_utc, label="reviewed_at_utc"),
        "decision_source": "interactive_visual_review",
        "reviewed_feature": "visible_dish_top_rim_edge",
        "decision_scope": "candidate_and_detection_disagreement_audit_only",
    }
    record = build_reviewed_palette_geometry_candidate_record(
        source_zarr=zarr_path,
        fit_report_path=fit_report_path,
        montage_path=montage_path,
        fit_review_run=fit_review_run,
        review=review,
        arena_binding=arena_binding,
    )
    digest = _payload_sha256(record)
    candidate_id = f"arena-geometry-palette-{digest[:24]}"
    source = _required_mapping(
        record.get("palette_fit_source"), label="palette_fit_source"
    )
    input_run_ids: dict[str, Any] = {}
    input_artifacts: list[dict[str, Any]] = [
        {
            "role": "palette_blind_dish_rim_fit_report",
            "path": source["fit_report_path"],
            "sha256": source["fit_report_sha256"],
        },
        {
            "role": "palette_dish_rim_review_montage",
            "path": source["review_montage_path"],
            "sha256": source["review_montage_sha256"],
        },
    ]
    if source.get("review_evidence_storage") is not None:
        run_name = _required_text(
            source.get("fit_review_run"), label="palette_fit_source.fit_review_run"
        )
        input_run_ids = {
            "arena_geometry_fit_review": run_name,
            "arena_geometry_fit_review_record_sha256": _required_sha256(
                source.get("fit_review_record_sha256"),
                label="palette_fit_source.fit_review_record_sha256",
            ),
        }
        receipt_path = zarr_path / "analysis" / FIT_REVIEW_RUNS_PARENT / run_name
        receipt_sha256 = input_run_ids["arena_geometry_fit_review_record_sha256"]
    else:
        if fit_report_path is None:
            raise RecordingGeometryError("External Palette fit report is missing.")
        receipt_path = Path(fit_report_path).expanduser().resolve()
        receipt_sha256 = _file_sha256(receipt_path)
    reveal_binding = source.get("acquisition_reveal_binding")
    if isinstance(reveal_binding, Mapping):
        input_artifacts.append(
            {
                "role": "post_freeze_acquisition_boundary_edge_support",
                "path": reveal_binding["path"],
                "sha256": reveal_binding["sha256"],
            }
        )
    collection = source.get("source_collection")
    if isinstance(collection, Mapping):
        input_artifacts.extend(
            [
                {
                    "role": "clipped_recording_clip_index",
                    "path": collection["recording_clip_index_path"],
                    "sha256": collection["recording_clip_index_sha256"],
                },
                {
                    "role": "clipped_recording_geometry_snapshot",
                    "path": collection["recording_geometry_snapshot_path"],
                    "sha256": collection["recording_geometry_snapshot_sha256"],
                },
            ]
        )
    provenance = build_writer_run_provenance(
        command="publish_reviewed_palette_geometry_candidate",
        params={
            "algorithm_version": PUBLISH_ALGORITHM_VERSION,
            "candidate_id": candidate_id,
            "candidate_record_sha256": digest,
            "candidate_kind": PALETTE_CANDIDATE_KIND,
            "reviewed_feature": "visible_dish_top_rim_edge",
            "gate_derivation": "direct_from_reviewed_visible_dish_top_rim_edge",
            "operational_selection": "not_performed",
        },
        input_run_ids=input_run_ids,
        input_artifacts=tuple(input_artifacts),
        include_system_context=False,
    )
    provenance_validation = validate_run_provenance(provenance)
    if not provenance_validation.valid:
        raise RuntimeError(
            "Palette candidate publication provenance is invalid: "
            f"{provenance_validation.errors}"
        )
    return ArenaGeometryCandidatePlan(
        source_zarr=zarr_path,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        candidate_id=candidate_id,
        candidate_record_sha256=digest,
        candidate_record=record,
        run_name=candidate_id,
        target_run_path=zarr_path / "analysis" / CANDIDATE_RUNS_PARENT / candidate_id,
        run_provenance=provenance,
        candidate_kind=PALETTE_CANDIDATE_KIND,
    )


def _candidate_attrs(plan: ArenaGeometryCandidatePlan) -> dict[str, Any]:
    return {
        "schema_id": CANDIDATE_RUN_SCHEMA_ID,
        "schema_version": CANDIDATE_RUN_SCHEMA_VERSION,
        "candidate_id": plan.candidate_id,
        "candidate_kind": plan.candidate_kind,
        "candidate_record": _canonical_copy(plan.candidate_record),
        "candidate_record_sha256": plan.candidate_record_sha256,
        "run_provenance": _canonical_copy(plan.run_provenance),
        "operational_selection_status": "not_selected",
        "legacy_dish_mask_projection_written": False,
        "detection_gate_applied": False,
    }


def validate_arena_geometry_candidate_run(
    run_path: str | Path,
    *,
    expected_plan: ArenaGeometryCandidatePlan,
    require_complete: bool = False,
    require_eligible: bool | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    path = Path(run_path).expanduser().resolve()
    try:
        group = open_zarr_root(path, mode="r")
        attrs = group.attrs
        expected = _candidate_attrs(expected_plan)
        for name, value in expected.items():
            if name == "run_provenance":
                continue
            if attrs.get(name) != value:
                errors.append(f"{name} mismatch")
        record = attrs.get("candidate_record")
        if isinstance(record, Mapping):
            try:
                validate_arena_geometry_candidate_record(record)
                if record.get("candidate_kind") != expected_plan.candidate_kind:
                    errors.append("candidate kind mismatch")
                if _payload_sha256(record) != attrs.get("candidate_record_sha256"):
                    errors.append("candidate record digest mismatch")
            except RecordingGeometryError as exc:
                errors.append(str(exc))
        else:
            errors.append("candidate_record missing")
        provenance = validate_run_provenance(attrs.get("run_provenance"))
        if not provenance.valid:
            errors.extend(f"run provenance: {item}" for item in provenance.errors)
        else:
            expected_provenance = expected_plan.run_provenance
            persisted_provenance = provenance.normalized or {}
            for name in (
                "command",
                "config_hash",
                "params",
                "input_run_ids",
                "input_artifacts",
            ):
                if persisted_provenance.get(name) != expected_provenance.get(name):
                    errors.append(f"run provenance {name} mismatch")
        if list(group.array_keys()) or list(group.group_keys()):
            errors.append("candidate run must be metadata-only")
        status = attrs.get("palette_run_completion_status")
        if require_complete and status != "complete":
            errors.append("candidate run is not complete")
        elif status not in {"running", "complete"}:
            errors.append("candidate run has invalid completion status")
        if require_eligible is not None and attrs.get(
            "stage_selector_eligible"
        ) is not (require_eligible):
            errors.append("candidate selector eligibility mismatch")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_id": expected_plan.candidate_id,
        "candidate_record_sha256": expected_plan.candidate_record_sha256,
        "run_path": str(path),
    }


def _materialize_local_run(plan: ArenaGeometryCandidatePlan, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing existing local candidate run: {path}")
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(json_attr_safe(_candidate_attrs(plan)))
    mark_run_started(
        group,
        run_name=plan.run_name,
        stage="arena_geometry_candidate",
    )
    validation = validate_arena_geometry_candidate_run(
        path,
        expected_plan=plan,
    )
    if not validation["valid"]:
        raise RuntimeError(
            f"Local arena-geometry candidate validation failed: {validation}"
        )


def _revalidate_candidate_sources(
    plan: ArenaGeometryCandidatePlan,
) -> dict[str, Any]:
    if plan.candidate_kind == ACQUISITION_CANDIDATE_KIND:
        acquisition_source = _required_mapping(
            plan.candidate_record.get("acquisition_source"),
            label="acquisition_source",
        )
        if acquisition_source.get("recovery_binding") is None:
            arena = _required_mapping(
                plan.candidate_record.get("arena_binding"),
                label="arena_binding",
            )
            source_kind = _required_text(
                acquisition_source.get("source_kind"),
                label="acquisition_source.source_kind",
            )
            current_record, current_digest = _record_from_producer_source_and_zarr(
                plan.source_zarr,
                source_path=plan.receipt_path,
                source_kind=source_kind,
                camera_serial=_required_text(
                    arena.get("camera_serial"), label="arena_binding.camera_serial"
                ),
                arena_id=_required_text(
                    arena.get("arena_id"), label="arena_binding.arena_id"
                ),
            )
            current_source = _required_mapping(
                current_record.get("acquisition_source"),
                label="current acquisition_source",
            )
            current_source_sha = _required_sha256(
                current_source.get("source_contract_sha256"),
                label="current source_contract_sha256",
            )
            if (
                current_source_sha != plan.receipt_sha256
                or current_digest != plan.candidate_record_sha256
                or current_record != plan.candidate_record
            ):
                raise RuntimeError(
                    "Producer-native acquisition geometry source changed during "
                    "publication."
                )
            return {
                "status": "current",
                "source_kind": source_kind,
                "source_contract_sha256": current_source_sha,
                "candidate_record_sha256": current_digest,
                "recovery_receipt_required": False,
            }
        verified, current_record, current_digest = _record_from_receipt_and_zarr(
            plan.source_zarr,
            plan.receipt_path,
        )
        if (
            verified.receipt_sha256 != plan.receipt_sha256
            or current_digest != plan.candidate_record_sha256
            or current_record != plan.candidate_record
        ):
            raise RuntimeError(
                "Acquisition geometry source changed during publication."
            )
        return {
            "status": "current",
            "source_kind": ACQUISITION_CANDIDATE_KIND,
            "receipt_sha256": verified.receipt_sha256,
            "candidate_record_sha256": current_digest,
        }
    if plan.candidate_kind == PALETTE_CANDIDATE_KIND:
        source = _required_mapping(
            plan.candidate_record.get("palette_fit_source"),
            label="palette_fit_source",
        )
        review = _required_mapping(plan.candidate_record.get("review"), label="review")
        if source.get("review_evidence_storage") is not None:
            fit_review_run = _required_text(
                source.get("fit_review_run"),
                label="palette_fit_source.fit_review_run",
            )
            current_record = build_reviewed_palette_geometry_candidate_record(
                source_zarr=plan.source_zarr,
                fit_review_run=fit_review_run,
                review=review,
                arena_binding=_required_mapping(
                    plan.candidate_record.get("arena_binding"),
                    label="arena_binding",
                ),
            )
            evidence = load_arena_geometry_fit_review_evidence(
                plan.source_zarr,
                run_name=fit_review_run,
            )
            current_report_sha256 = hashlib.sha256(
                evidence.fit_report_bytes
            ).hexdigest()
            current_receipt_sha256 = evidence.review_record_sha256
        else:
            current_record = build_reviewed_palette_geometry_candidate_record(
                source_zarr=plan.source_zarr,
                fit_report_path=plan.receipt_path,
                montage_path=_required_text(
                    source.get("review_montage_path"),
                    label="palette_fit_source.review_montage_path",
                ),
                review=review,
                arena_binding=_required_mapping(
                    plan.candidate_record.get("arena_binding"),
                    label="arena_binding",
                ),
            )
            current_report_sha256 = _file_sha256(plan.receipt_path)
            current_receipt_sha256 = current_report_sha256
        current_digest = _payload_sha256(current_record)
        if (
            current_receipt_sha256 != plan.receipt_sha256
            or current_digest != plan.candidate_record_sha256
            or current_record != plan.candidate_record
        ):
            raise RuntimeError("Palette geometry source changed during publication.")
        return {
            "status": "current",
            "source_kind": PALETTE_CANDIDATE_KIND,
            "fit_report_sha256": current_report_sha256,
            "review_montage_sha256": source.get("review_montage_sha256"),
            "fit_review_run": source.get("fit_review_run"),
            "candidate_record_sha256": current_digest,
        }
    raise RuntimeError(f"Unsupported candidate source kind: {plan.candidate_kind!r}.")


def publish_arena_geometry_candidate(
    plan: ArenaGeometryCandidatePlan,
    *,
    scratch_root: str | Path,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Publish one metadata-only candidate atomically, without selecting it."""

    if plan.target_run_path.exists():
        existing = validate_arena_geometry_candidate_run(
            plan.target_run_path,
            expected_plan=plan,
            require_complete=True,
            require_eligible=True,
        )
        if not existing["valid"]:
            raise FileExistsError(
                f"Existing candidate path is not the expected immutable run: {existing}"
            )
        return {"published": False, "status": "already_complete", **existing}

    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"palette-{plan.run_name}-",
        dir=scratch,
    ) as temporary:
        local_run = Path(temporary) / plan.run_name
        _materialize_local_run(plan, local_run)

        def validate(path: Path) -> dict[str, Any]:
            return validate_arena_geometry_candidate_run(
                path,
                expected_plan=plan,
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, CANDIDATE_RUNS_PARENT),)

        def after_rename(_root: zarr.Group, _run: zarr.Group) -> dict[str, Any]:
            return {"source_revision_audit": _revalidate_candidate_sources(plan)}

        def complete(
            _root: zarr.Group,
            parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            mark_run_complete(
                run_group,
                parent_group=parent,
                run_name=plan.run_name,
                run_provenance=plan.run_provenance,
            )

        def verify(root: zarr.Group) -> None:
            parent = root[f"analysis/{CANDIDATE_RUNS_PARENT}"]
            run_group = parent[plan.run_name]
            if (
                run_group.attrs.get("palette_run_completion_status") != "complete"
                or run_group.attrs.get("stage_selector_eligible") is not False
                or parent.attrs.get("latest") == plan.run_name
                or parent.attrs.get("latest_complete") == plan.run_name
            ):
                raise RuntimeError(
                    "Candidate must be complete and readable without becoming latest or selected."
                )

        def activate(
            _root: zarr.Group,
            _parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            if run_group.attrs.get("operational_selection_status") != "not_selected":
                raise RuntimeError(
                    "Candidate activation cannot perform operational selection."
                )
            run_group.attrs["stage_selector_eligible"] = True

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=local_run,
                target_run_path=plan.target_run_path,
                run_name=plan.run_name,
                lock_suffix="arena-geometry-candidate-publish",
                publish_schema_id=PUBLISH_SCHEMA_ID,
                policy="node_local_metadata_candidate_atomic_run_group_publish",
                rollback_policy=(
                    "retain_failed_public_tombstone_leave_parent_without_candidate_pointer"
                ),
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            activate_run=activate,
            after_rename=after_rename,
            payload_metadata={
                "algorithm_version": PUBLISH_ALGORITHM_VERSION,
                "candidate_id": plan.candidate_id,
                "candidate_record_sha256": plan.candidate_record_sha256,
                "selection_performed": False,
                "legacy_dish_mask_projection_written": False,
            },
        )

    final = validate_arena_geometry_candidate_run(
        plan.target_run_path,
        expected_plan=plan,
        require_complete=True,
        require_eligible=True,
    )
    if not final["valid"]:
        raise RuntimeError(
            f"Published arena-geometry candidate failed validation: {final}"
        )
    return {
        "published": True,
        "status": "complete_candidate_not_selected",
        "publication": publication,
        **final,
    }


__all__ = [
    "ACQUISITION_CANDIDATE_KIND",
    "ArenaGeometryCandidatePlan",
    "CANDIDATE_KIND",
    "CANDIDATE_RECORD_SCHEMA_ID",
    "CANDIDATE_RECORD_SCHEMA_VERSION",
    "CANDIDATE_RUNS_PARENT",
    "CANDIDATE_RUN_SCHEMA_ID",
    "CANDIDATE_RUN_SCHEMA_VERSION",
    "CLIPPED_ACQUISITION_FRAME_AUTHORITY_KIND",
    "LEGACY_CLIPPED_SNAPSHOT_FRAME_AUTHORITY_KIND",
    "PUBLISH_ALGORITHM_VERSION",
    "PUBLISH_SCHEMA_ID",
    "PALETTE_CANDIDATE_KIND",
    "build_acquisition_geometry_candidate_record",
    "build_reviewed_palette_geometry_candidate_record",
    "plan_producer_native_acquisition_geometry_candidate",
    "plan_recovered_acquisition_geometry_candidate",
    "plan_reviewed_palette_geometry_candidate",
    "publish_arena_geometry_candidate",
    "validate_acquisition_geometry_candidate_record",
    "validate_arena_geometry_candidate_record",
    "validate_arena_geometry_candidate_run",
    "validate_palette_geometry_candidate_record",
]
