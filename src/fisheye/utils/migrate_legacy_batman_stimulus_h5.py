#!/usr/bin/env python3
"""Create canonical, evidence-bound derivatives of legacy Batman Citrus H5s.

The transferred H5 and Orange sidecars are opened read-only.  ``--apply`` writes
a new H5 derivative and an external receipt below the recording's ``derived``
tree; it never edits or replaces a raw artifact.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from fisheye.shared.coordinate_descriptor import (
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    build_canonical_coordinate_descriptor,
    canonical_coordinate_descriptor_v2_attrs,
)
from fisheye.shared.coordinate_identity import (
    ROW_IDENTITY_CONTRACT_ATTR,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    build_row_identity_contract,
    identity_array_content_sha256,
    row_identity_contract_attrs,
    row_identity_key_attrs,
)
from fisheye.shared.pixel_frame_authority import (
    PIXEL_FRAME_AUTHORITY_ATTR,
    PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
)
from fisheye.shared.stimulus_coordinate_contract import (
    ARENA_GEOMETRY_RECORD_ATTR,
    ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
    CHASER_STATES_SCHEMA_ID,
    CHASER_STATES_SCHEMA_VERSION,
    COORDINATE_SURFACE_MANIFEST_ATTR,
    COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR,
    COORDINATE_SURFACE_MANIFEST_SCHEMA,
    COORDINATE_SURFACE_MANIFEST_VERSION,
    SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
    SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
    SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
    SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR,
    SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
    SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
    STIMULUS_RENDERER_SNAPSHOT_CAPTURE_PHASE,
    STIMULUS_RENDERER_SNAPSHOT_PATH,
    STIMULUS_RENDERER_SNAPSHOT_SCHEMA_ID,
    STIMULUS_RENDERER_SNAPSHOT_SCHEMA_VERSION,
    TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
    TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
    TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
    TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
    TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR,
    TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR,
    TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
    TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
    TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH,
    _classify_renderer_snapshot,
    _h5_dataset_content_digest,
    arena_geometry_record,
    canonical_mapping_digest,
    numpy_content_digest,
    preflight_stimulus_coordinate_contract,
    source_arena_pixel_frame_record,
)


MIGRATION_SCHEMA_ID = "palette.legacy_batman_stimulus_h5_migration"
MIGRATION_SCHEMA_VERSION = 1
MIGRATION_GROUP_PATH = "/palette_migration"
MIGRATION_RECEIPT_ATTR = "migration_receipt"
MIGRATION_RECEIPT_DIGEST_ATTR = f"{MIGRATION_RECEIPT_ATTR}_sha256"
EXTERNAL_RECEIPT_SCHEMA_ID = "palette.stimulus_h5_derivative_artifact"
DERIVATIVE_SUFFIX = ".canonical_stimulus_v1.h5"
SUPPORTED_CITRUS_VERSION = "v1.2.1-1491-g5ddcc39-dirty"

_CHASER_FIELDS = (
    "stimulus_frame_num",
    "timestamp_ns_session",
    "chaser_index",
    "is_chasing",
    "chaser_pos_x",
    "chaser_pos_y",
    "target_pos_x",
    "target_pos_y",
    "target_source_frame_id",
    "target_source_camera_id",
    "target_source_box_index_in_payload",
    "target_age_ms",
    "target_freshness_state",
    "target_area_state",
    "target_clamped_pos_x",
    "target_clamped_pos_y",
    "target_distance_outside_px",
    "chaser_radius_px",
    "chaser_radius_mm",
    "target_radius_px",
    "target_radius_mm",
    "distance_to_target_px",
    "distance_to_target_mm",
    "chase_speed_px_per_s",
    "chase_speed_mm_per_s",
    "behavior_program_active",
    "behavior_episode_active",
    "behavior_episode_id",
    "behavior_phase_index",
    "behavior_motion_type_id",
    "behavior_velocity_x_mm_per_s",
    "behavior_velocity_y_mm_per_s",
    "behavior_command_speed_mm_per_s",
    "behavior_retreat_plan_active",
    "behavior_retreat_requested_distance_mm",
    "behavior_retreat_actual_distance_mm",
    "behavior_retreat_endpoint_x_mm",
    "behavior_retreat_endpoint_y_mm",
    "behavior_retreat_target_x_mm",
    "behavior_retreat_target_y_mm",
    "behavior_retreat_angular_deviation_deg",
    "visual_angle_deg",
    "angular_velocity_deg_s",
    "tau_ms",
    "loom_mode",
    "loom_phase",
    "chaser_behavior_class_id",
    "l_over_v_ms",
    "initial_distance_mm",
    "max_angle_deg",
    "z_eff_mm",
    "pixels_per_mm",
    "trial_state",
    "chase_sequence_active",
    "chase_trial_id",
    "time_in_state_s",
)

_LEGACY_CHASER_ATTRS = {
    "angle_zero_direction",
    "behavior_retreat_distance_fields",
    "behavior_retreat_distance_units",
    "behavior_retreat_position_fields",
    "behavior_retreat_position_units",
    "camera_id",
    "coordinate_frame",
    "coordinate_origin",
    "physical_frame_status",
    "position_fields",
    "positive_rotation_direction",
    "runtime_behavior_plane_id",
    "runtime_correction_active",
    "runtime_render_plane_id",
    "scale_source",
    "schema_version",
    "target_mapping_semantics",
    "target_point_method",
    "units",
    "x_axis_direction",
    "y_axis_direction",
}

_ROW_IDENTITY_FIELDS = ("chaser_index", "stimulus_frame_num")
_COORDINATE_COMPONENT_FIELDS = {
    "chaser_pos_x",
    "chaser_pos_y",
    "target_pos_x",
    "target_pos_y",
    "target_clamped_pos_x",
    "target_clamped_pos_y",
}
_SURFACES = (
    {
        "array_name": "chaser_position_xy",
        "semantic_role": "chaser_position",
        "component_fields": ["chaser_pos_x", "chaser_pos_y"],
    },
    {
        "array_name": "target_position_xy",
        "semantic_role": "target_position",
        "component_fields": ["target_pos_x", "target_pos_y"],
    },
    {
        "array_name": "target_clamped_position_xy",
        "semantic_role": "target_clamped_position",
        "component_fields": [
            "target_clamped_pos_x",
            "target_clamped_pos_y",
        ],
    },
)


class BatmanMigrationError(ValueError):
    """Raised when historical evidence is incomplete or contradictory."""


@dataclass(frozen=True)
class MigrationPlan:
    recording_dir: Path
    source_h5: Path
    output_h5: Path
    external_receipt: Path
    manifest: Mapping[str, Any]
    camera_id: str
    camera_index: int
    recording_id: str
    orange_session_id: str
    total_frames: int
    jsonl_path: Path
    csv_path: Path
    recording_snapshot_path: Path
    recording_session_path: Path
    artifact_records: tuple[Mapping[str, Any], ...]
    source_chaser_sha256: str
    source_bounding_boxes_sha256: str | None
    renderer_snapshot: Mapping[str, Any]
    renderer_snapshot_sha256: str
    supported_row_count: int
    source_row_count: int
    terminal_row_key: tuple[int, int]
    row_identity: np.ndarray = field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = field(
        repr=False,
        compare=False,
    )
    target_source_acquisition_frame_index: np.ndarray = field(
        repr=False,
        compare=False,
    )
    target_source_acquisition_frame_valid: np.ndarray = field(
        repr=False,
        compare=False,
    )


def _json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatmanMigrationError(f"Unable to read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatmanMigrationError(f"JSON artifact {path} must contain an object.")
    return value


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatmanMigrationError(f"{label} must be non-empty text.")
    return value.strip()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BatmanMigrationError(f"{label} must be a positive integer.")
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise BatmanMigrationError(f"Unable to hash required artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact_record(path: Path, *, recording_dir: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise BatmanMigrationError(f"Required {role} artifact is missing: {path}")
    stat = path.stat()
    try:
        relative_path = str(path.relative_to(recording_dir))
    except ValueError:
        relative_path = str(path)
    return {
        "role": role,
        "relative_path": relative_path,
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _single_file(paths: Iterable[Path], *, label: str) -> Path:
    candidates = sorted(set(path.resolve() for path in paths))
    if len(candidates) != 1:
        raise BatmanMigrationError(
            f"{label} must resolve exactly one file; found {len(candidates)}."
        )
    return candidates[0]


def _manifest_paths(
    recording_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path, int, str, str, str]:
    if manifest.get("protocol_name") != "Batman":
        raise BatmanMigrationError("Historical migration is restricted to Batman recordings.")
    if manifest.get("software_version") != SUPPORTED_CITRUS_VERSION:
        raise BatmanMigrationError(
            "Historical migration is restricted to Citrus "
            f"{SUPPORTED_CITRUS_VERSION}."
        )
    camera_id = _required_text(manifest.get("camera_id"), label="manifest camera_id")
    recording_id = _required_text(
        manifest.get("recording_name"),
        label="manifest recording_name acquisition identity",
    )
    orange_session_id = _required_text(
        manifest.get("orange_session_id"),
        label="manifest orange_session_id",
    )
    streams = manifest.get("video_streams")
    if not isinstance(streams, Mapping) or streams.get("frame_clock") != "recording_frame_id":
        raise BatmanMigrationError(
            "Manifest must declare recording_frame_id as the authoritative frame clock."
        )
    stream_map = streams.get("streams")
    full = stream_map.get("full") if isinstance(stream_map, Mapping) else None
    if not isinstance(full, Mapping):
        raise BatmanMigrationError("Manifest lacks the authoritative full stream.")
    if (
        str(full.get("camera_id")) != camera_id
        or full.get("frame_clock") != "recording_frame_id"
        or full.get("role") != "ingest_authoritative_full_frame"
    ):
        raise BatmanMigrationError(
            "Manifest full-stream identity or frame-domain declaration is invalid."
        )
    total_frames = _positive_int(full.get("frame_count"), label="full frame_count")
    csv_relative = _required_text(
        full.get("frame_clock_metadata"),
        label="full frame_clock_metadata",
    )
    csv_path = (recording_dir / csv_relative).resolve()
    h5_candidates = [
        recording_dir / str(relative)
        for relative in manifest.get("files", {}).get("raw", [])
        if isinstance(relative, str) and relative.endswith(".h5")
    ]
    source_h5 = _single_file(h5_candidates, label="raw stimulus H5")
    jsonl_path = _single_file(
        recording_dir.glob(
            f"derived/external_crop_recorder/Cam{camera_id}_*_yolo_events.jsonl"
        ),
        label="Orange YOLO JSONL",
    )
    snapshot_path = (
        recording_dir
        / _required_text(
            manifest.get("recording_snapshot"),
            label="manifest recording_snapshot",
        )
    ).resolve()
    return (
        source_h5,
        jsonl_path,
        csv_path,
        snapshot_path,
        total_frames,
        camera_id,
        recording_id,
        orange_session_id,
    )


def _validate_session_and_snapshot(
    *,
    recording_session: Mapping[str, Any],
    recording_snapshot: Mapping[str, Any],
    camera_id: str,
    total_frames: int,
) -> None:
    camera_artifacts = recording_session.get("camera_artifacts")
    camera = camera_artifacts.get(camera_id) if isinstance(camera_artifacts, Mapping) else None
    if not isinstance(camera, Mapping) or any(
        camera.get(name) != expected
        for name, expected in (
            ("first_recording_frame_id", 1),
            ("last_recording_frame_id", total_frames),
            ("frame_count", total_frames),
            ("recording_frame_id_gaps", 0),
        )
    ):
        raise BatmanMigrationError(
            "recording_session.json does not declare a complete, gap-free camera clock."
        )
    runtime_map = recording_snapshot.get("camera_runtime")
    runtime = runtime_map.get(camera_id) if isinstance(runtime_map, Mapping) else None
    coordinate_frame = runtime.get("coordinate_frame") if isinstance(runtime, Mapping) else None
    extent = coordinate_frame.get("extent") if isinstance(coordinate_frame, Mapping) else None
    if (
        not isinstance(extent, Mapping)
        or extent.get("width_px") != 4512
        or extent.get("height_px") != 4512
        or coordinate_frame.get("coordinate_space") != "camera_native_pixels"
    ):
        raise BatmanMigrationError(
            "recording_snapshot.json lacks the expected 4512x4512 camera-native frame."
        )


def _load_orange_identity(
    *,
    jsonl_path: Path,
    csv_path: Path,
    camera_id: str,
    orange_session_id: str,
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    recording_by_ipc = np.full(total_frames + 1, -1, dtype=np.int64)
    camera_index_by_ipc = np.full(total_frames + 1, -1, dtype=np.int64)
    camera_frame_by_recording = np.full(total_frames + 1, -1, dtype=np.int64)
    jsonl_rows = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatmanMigrationError(
                    f"Invalid JSONL at {jsonl_path}:{line_number}: {exc.msg}."
                ) from exc
            frame = event.get("frame") if isinstance(event, Mapping) else None
            if (
                event.get("schema_id") != "orange.yolo_event"
                or event.get("event_kind") != "yolo_result"
                or str(event.get("camera_serial")) != camera_id
                or event.get("recording_id") != orange_session_id
                or not isinstance(frame, Mapping)
            ):
                raise BatmanMigrationError(
                    f"Orange JSONL identity is invalid at line {line_number}."
                )
            try:
                ipc_id = int(frame["ipc_frame_id"])
                recording_frame_id = int(frame["recording_frame_id"])
                camera_frame_id = int(frame["camera_frame_id"])
                camera_index = int(event["camera_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BatmanMigrationError(
                    f"Orange JSONL frame identity is invalid at line {line_number}."
                ) from exc
            if (
                ipc_id < 1
                or ipc_id > total_frames
                or recording_frame_id != ipc_id
                or recording_by_ipc[ipc_id] != -1
            ):
                raise BatmanMigrationError(
                    f"Orange JSONL IDs are duplicate, out of range, or disagree at line {line_number}."
                )
            recording_by_ipc[ipc_id] = recording_frame_id
            camera_index_by_ipc[ipc_id] = camera_index
            camera_frame_by_recording[recording_frame_id] = camera_frame_id
            jsonl_rows += 1
    if jsonl_rows != total_frames or np.any(recording_by_ipc[1:] < 1):
        raise BatmanMigrationError(
            "Orange JSONL does not provide one complete identity row per acquisition frame."
        )

    csv_camera_frames = np.full(total_frames + 1, -1, dtype=np.int64)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"recording_frame_id", "camera_frame_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise BatmanMigrationError(
                "Camera metadata CSV lacks recording_frame_id or camera_frame_id."
            )
        csv_rows = 0
        for row_offset, row in enumerate(reader):
            try:
                recording_frame_id = int(row["recording_frame_id"])
                camera_frame_id = int(row["camera_frame_id"])
            except (TypeError, ValueError) as exc:
                raise BatmanMigrationError(
                    f"Camera metadata CSV identity is invalid at row {row_offset + 2}."
                ) from exc
            if (
                recording_frame_id != row_offset + 1
                or recording_frame_id > total_frames
                or csv_camera_frames[recording_frame_id] != -1
            ):
                raise BatmanMigrationError(
                    "Camera metadata CSV is not the exact contiguous one-based full-stream clock."
                )
            csv_camera_frames[recording_frame_id] = camera_frame_id
            csv_rows += 1
    if (
        csv_rows != total_frames
        or np.any(csv_camera_frames[1:] < 0)
        or not np.array_equal(
            csv_camera_frames[1:],
            camera_frame_by_recording[1:],
        )
    ):
        raise BatmanMigrationError(
            "Camera metadata CSV and Orange JSONL frame identities disagree."
        )
    return recording_by_ipc, camera_index_by_ipc


def _normalize_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _validate_legacy_chaser_attrs(attrs: Mapping[str, Any], *, camera_id: str) -> None:
    normalized = {str(name): _normalize_attr(value) for name, value in attrs.items()}
    if set(normalized) != _LEGACY_CHASER_ATTRS:
        raise BatmanMigrationError("Legacy chaser_states attrs do not match the audited layout.")
    expected = {
        "camera_id": camera_id,
        "coordinate_frame": "arena_relative_canvas_px",
        "coordinate_origin": "top_left_of_active_arena",
        "physical_frame_status": "arena_relative_canvas_px_not_accepted_world_mm",
        "runtime_behavior_plane_id": "projected_surface",
        "runtime_render_plane_id": "projected_surface",
        "schema_version": 4,
        "units": "px",
        "x_axis_direction": "right",
        "y_axis_direction": "down",
    }
    if any(normalized.get(name) != value for name, value in expected.items()):
        raise BatmanMigrationError(
            "Legacy chaser coordinate semantics do not match the audited Batman contract."
        )


def _h5_evidence(
    *,
    source_h5: Path,
    recording_by_ipc: np.ndarray,
    camera_index_by_ipc: np.ndarray,
    camera_id: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
    str,
    str | None,
    dict[str, Any],
    str,
]:
    with h5py.File(source_h5, "r") as h5:
        renderer, renderer_path, renderer_digest = _classify_renderer_snapshot(h5)
        if renderer is None or renderer_path != "/stimulus_coordinates" or renderer_digest is None:
            raise BatmanMigrationError(
                "Legacy Batman H5 lacks the exact audited static renderer snapshot."
            )
        chaser_path = "/tracking_data/chaser_states"
        frame_path = "/video_metadata/frame_metadata"
        if chaser_path not in h5 or frame_path not in h5:
            raise BatmanMigrationError("Batman H5 lacks chaser_states or frame_metadata.")
        chaser = h5[chaser_path]
        frame_metadata = h5[frame_path]
        if (
            not isinstance(chaser, h5py.Dataset)
            or not isinstance(frame_metadata, h5py.Dataset)
            or tuple(chaser.dtype.names or ()) != _CHASER_FIELDS
        ):
            raise BatmanMigrationError("Batman chaser_states dtype is outside the audited layout.")
        _validate_legacy_chaser_attrs(chaser.attrs, camera_id=camera_id)
        required_frame_fields = {"stimulus_frame_num", "triggering_camera_frame_id"}
        if not required_frame_fields.issubset(frame_metadata.dtype.names or ()):
            raise BatmanMigrationError("frame_metadata lacks the audited identity fields.")
        source_rows = int(chaser.shape[0])
        supported_rows = int(frame_metadata.shape[0])
        if source_rows != supported_rows + 1:
            raise BatmanMigrationError(
                "Batman migration requires exactly one terminal chaser row without frame metadata."
            )
        chaser_stimulus = np.asarray(
            chaser["stimulus_frame_num"][:],
            dtype=np.int64,
        )
        frame_stimulus = np.asarray(
            frame_metadata["stimulus_frame_num"][:],
            dtype=np.int64,
        )
        if (
            not np.array_equal(chaser_stimulus[:-1], frame_stimulus)
            or np.unique(frame_stimulus).size != supported_rows
            or int(chaser_stimulus[-1]) in set(frame_stimulus.tolist())
        ):
            raise BatmanMigrationError(
                "Only the final shutdown chaser row may lack frame-metadata identity."
            )
        chaser_indices = np.asarray(chaser["chaser_index"][:], dtype=np.int64)
        row_identity = np.column_stack(
            (chaser_indices[:-1], chaser_stimulus[:-1])
        ).astype("<i8", copy=False)
        if np.unique(row_identity, axis=0).shape[0] != supported_rows:
            raise BatmanMigrationError("Batman stimulus-state identity is not unique.")

        triggering_ids = np.asarray(
            frame_metadata["triggering_camera_frame_id"][:],
            dtype=np.int64,
        )
        if (
            np.any(triggering_ids < 1)
            or np.any(triggering_ids >= recording_by_ipc.shape[0])
            or np.any(recording_by_ipc[triggering_ids] < 1)
        ):
            raise BatmanMigrationError(
                "A stimulus-state trigger ID has no exact Orange IPC identity."
            )
        state_acquisition = np.asarray(
            recording_by_ipc[triggering_ids] - 1,
            dtype="<i8",
        )

        target_ids = np.asarray(
            chaser["target_source_frame_id"][:-1],
            dtype=np.int64,
        )
        target_camera_indices = np.asarray(
            chaser["target_source_camera_id"][:-1],
            dtype=np.int64,
        )
        target_valid = target_ids > 0
        target_acquisition = np.full(supported_rows, -1, dtype="<i8")
        valid_ids = target_ids[target_valid]
        if (
            np.any(valid_ids >= recording_by_ipc.shape[0])
            or np.any(recording_by_ipc[valid_ids] < 1)
            or not np.array_equal(
                target_camera_indices[target_valid],
                camera_index_by_ipc[valid_ids],
            )
        ):
            raise BatmanMigrationError(
                "Target-source IDs do not map exactly to matching Orange camera evidence."
            )
        target_acquisition[target_valid] = recording_by_ipc[valid_ids] - 1
        terminal_key = (
            int(chaser_indices[-1]),
            int(chaser_stimulus[-1]),
        )
        chaser_digest = _h5_dataset_content_digest(chaser)
        bbox_digest = None
        if "/tracking_data/bounding_boxes" in h5:
            bbox = h5["/tracking_data/bounding_boxes"]
            if not isinstance(bbox, h5py.Dataset):
                raise BatmanMigrationError("Legacy bounding_boxes must be a dataset.")
            bbox_digest = _h5_dataset_content_digest(bbox)
        return (
            row_identity,
            state_acquisition,
            target_acquisition,
            np.asarray(target_valid, dtype=bool),
            terminal_key,
            chaser_digest,
            bbox_digest,
            dict(renderer),
            renderer_digest,
        )


def plan_migration(
    recording_dir: Path,
    *,
    output_h5: Path | None = None,
) -> MigrationPlan:
    recording_dir = recording_dir.expanduser().resolve()
    manifest_path = recording_dir / "recording_manifest.json"
    manifest = _json_load(manifest_path)
    (
        source_h5,
        jsonl_path,
        csv_path,
        snapshot_path,
        total_frames,
        camera_id,
        recording_id,
        orange_session_id,
    ) = _manifest_paths(recording_dir, manifest)
    session_path = (recording_dir / "raw/recording_session.json").resolve()
    _validate_session_and_snapshot(
        recording_session=_json_load(session_path),
        recording_snapshot=_json_load(snapshot_path),
        camera_id=camera_id,
        total_frames=total_frames,
    )
    recording_by_ipc, camera_index_by_ipc = _load_orange_identity(
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        camera_id=camera_id,
        orange_session_id=orange_session_id,
        total_frames=total_frames,
    )
    (
        row_identity,
        state_acquisition,
        target_acquisition,
        target_valid,
        terminal_key,
        chaser_digest,
        bbox_digest,
        renderer,
        renderer_digest,
    ) = _h5_evidence(
        source_h5=source_h5,
        recording_by_ipc=recording_by_ipc,
        camera_index_by_ipc=camera_index_by_ipc,
        camera_id=camera_id,
    )
    artifact_paths = (
        (source_h5, "source_h5"),
        (jsonl_path, "orange_yolo_jsonl"),
        (csv_path, "orange_full_camera_metadata_csv"),
        (snapshot_path, "recording_snapshot"),
        (session_path, "recording_session"),
        (manifest_path, "recording_manifest"),
    )
    artifact_records = tuple(
        _artifact_record(path, recording_dir=recording_dir, role=role)
        for path, role in artifact_paths
    )
    if output_h5 is None:
        output_h5 = (
            recording_dir
            / "derived/stimulus_coordinate_migration"
            / f"{source_h5.stem}{DERIVATIVE_SUFFIX}"
        )
    output_h5 = output_h5.expanduser().resolve()
    if output_h5 == source_h5:
        raise BatmanMigrationError("Derivative output must not replace the source H5.")
    return MigrationPlan(
        recording_dir=recording_dir,
        source_h5=source_h5,
        output_h5=output_h5,
        external_receipt=output_h5.with_suffix(output_h5.suffix + ".receipt.json"),
        manifest=manifest,
        camera_id=camera_id,
        camera_index=int(camera_index_by_ipc[1]),
        recording_id=recording_id,
        orange_session_id=orange_session_id,
        total_frames=total_frames,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        recording_snapshot_path=snapshot_path,
        recording_session_path=session_path,
        artifact_records=artifact_records,
        source_chaser_sha256=chaser_digest,
        source_bounding_boxes_sha256=bbox_digest,
        renderer_snapshot=renderer,
        renderer_snapshot_sha256=renderer_digest,
        supported_row_count=int(row_identity.shape[0]),
        source_row_count=int(row_identity.shape[0]) + 1,
        terminal_row_key=terminal_key,
        row_identity=row_identity,
        source_acquisition_frame_index=state_acquisition,
        target_source_acquisition_frame_index=target_acquisition,
        target_source_acquisition_frame_valid=target_valid,
    )


def _write_contract_attrs(
    node: h5py.Group | h5py.Dataset,
    attrs: Mapping[str, Any],
) -> None:
    for name, value in attrs.items():
        node.attrs[name] = (
            json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if isinstance(value, (dict, list))
            else value
        )


def _surface_manifest(dtype_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_id": COORDINATE_SURFACE_MANIFEST_SCHEMA,
        "schema_version": COORDINATE_SURFACE_MANIFEST_VERSION,
        "coordinate_fields_complete": True,
        "field_classifications": {
            name: (
                "row_identity"
                if name in _ROW_IDENTITY_FIELDS
                else "coordinate_component"
                if name in _COORDINATE_COMPONENT_FIELDS
                else "non_spatial"
            )
            for name in dtype_names
        },
        "row_identity_fields": list(_ROW_IDENTITY_FIELDS),
        "surfaces": [dict(surface) for surface in _SURFACES],
    }


def _mapping_records(
    plan: MigrationPlan,
    *,
    identity_contract: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_digest = identity_array_content_sha256(plan.row_identity)
    common = {
        "source_rowset_ref": "/tracking_data/chaser_states",
        "source_row_identity_ref": f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY_REF}",
        "source_row_identity_sha256": identity_digest,
        "source_row_identity_contract_sha256": identity_contract.digest(),
        "acquisition_recording_id": plan.recording_id,
        "acquisition_camera_id": plan.camera_id,
        "source_total_frames": plan.total_frames,
        "target_domain": "acquisition_frame_index",
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    state = {
        "schema_id": SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
        "schema_version": SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "mapping_method": "explicit_per_stimulus_state_v1",
        **common,
        "array_ref": SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
        "array_dtype": np.dtype("<i8").str,
        "array_shape": [plan.supported_row_count],
        "array_content_sha256": numpy_content_digest(
            plan.source_acquisition_frame_index
        ),
    }
    target = {
        "schema_id": TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
        "schema_version": TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "mapping_method": "explicit_per_stimulus_state_target_provenance_v1",
        **common,
        "source_target_frame_field": (
            "/tracking_data/chaser_states#target_source_frame_id"
        ),
        "source_target_camera_field": (
            "/tracking_data/chaser_states#target_source_camera_id"
        ),
        "array_ref": TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
        "array_dtype": np.dtype("<i8").str,
        "array_shape": [plan.supported_row_count],
        "array_content_sha256": numpy_content_digest(
            plan.target_source_acquisition_frame_index
        ),
        "validity_array_ref": TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH,
        "validity_array_dtype": np.dtype("bool").str,
        "validity_array_shape": [plan.supported_row_count],
        "validity_array_content_sha256": numpy_content_digest(
            plan.target_source_acquisition_frame_valid
        ),
        "invalid_index_sentinel": -1,
    }
    return state, target


def _migration_receipt(
    plan: MigrationPlan,
    *,
    identity_contract: Any,
    state_mapping: Mapping[str, Any],
    target_mapping: Mapping[str, Any],
    arena_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": MIGRATION_SCHEMA_ID,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_method": "legacy_batman_orange_identity_join_v1",
        "source_artifacts": list(plan.artifact_records),
        "source_chaser_states_sha256": plan.source_chaser_sha256,
        "source_bounding_boxes_sha256": plan.source_bounding_boxes_sha256,
        "source_renderer_snapshot": dict(plan.renderer_snapshot),
        "source_renderer_snapshot_sha256": plan.renderer_snapshot_sha256,
        "source_row_count": plan.source_row_count,
        "supported_row_count": plan.supported_row_count,
        "omitted_rows": [
            {
                "source_row_index": plan.source_row_count - 1,
                "stimulus_state_key": list(plan.terminal_row_key),
                "reason_code": "citrus_shutdown_state_without_frame_metadata",
            }
        ],
        "omitted_source_paths": [
            "/stimulus_coordinates",
            "/tracking_data/bounding_boxes",
        ],
        "preserved_renderer_snapshot_path": STIMULUS_RENDERER_SNAPSHOT_PATH,
        "row_identity_contract": identity_contract.to_dict(),
        "row_identity_contract_sha256": identity_contract.digest(),
        "stimulus_state_key_sha256": identity_array_content_sha256(
            plan.row_identity
        ),
        "state_acquisition_mapping": dict(state_mapping),
        "state_acquisition_mapping_sha256": canonical_mapping_digest(state_mapping),
        "target_source_acquisition_mapping": dict(target_mapping),
        "target_source_acquisition_mapping_sha256": canonical_mapping_digest(
            target_mapping
        ),
        "arena_geometry_record": dict(arena_record),
        "arena_geometry_record_sha256": canonical_mapping_digest(arena_record),
        "coordinate_semantics": {
            "space_id": "arena_relative_canvas_px",
            "units": "px",
            "origin": "arena_top_left",
            "positive_x": "right",
            "positive_y": "down",
            "pixel_convention": "continuous",
            "render_plane": "projected_surface",
            "behavior_plane": "projected_surface",
            "accepted_world_mm": False,
        },
        "orange_join_checks": {
            "ipc_frame_id_equals_recording_frame_id": True,
            "csv_row_offset_equals_recording_frame_id_minus_one": True,
            "jsonl_csv_camera_frame_id_equal": True,
            "duplicate_ids": 0,
            "missing_ids": 0,
            "camera_mismatches": 0,
        },
        "bounding_boxes_status": "omitted_pending_separate_camera_native_import",
        "canonicalization": "canonical_json_sort_keys_v1",
    }


def _mutate_derivative(path: Path, plan: MigrationPlan) -> dict[str, Any]:
    with h5py.File(path, "r+") as h5:
        chaser = h5["/tracking_data/chaser_states"]
        if chaser.maxshape is None or chaser.maxshape[0] is not None:
            raise BatmanMigrationError(
                "Legacy chaser_states is not resizeable; refusing a lossy recreation."
            )
        chaser.resize((plan.supported_row_count,))
        for name in tuple(chaser.attrs.keys()):
            del chaser.attrs[name]

        tracking = h5["/tracking_data"]
        for name in (
            STIMULUS_STATE_KEY_ARRAY_REF,
            SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
        ):
            if name in tracking:
                raise BatmanMigrationError(
                    f"Legacy source unexpectedly contains canonical dataset {name}."
                )

        identity_contract = build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=plan.row_identity,
            components=_ROW_IDENTITY_FIELDS,
        )
        state_mapping, target_mapping = _mapping_records(
            plan,
            identity_contract=identity_contract,
        )
        state_key = tracking.create_dataset(
            STIMULUS_STATE_KEY_ARRAY_REF,
            data=plan.row_identity,
            dtype="<i8",
        )
        state_time = tracking.create_dataset(
            SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=plan.source_acquisition_frame_index,
            dtype="<i8",
        )
        target_time = tracking.create_dataset(
            TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=plan.target_source_acquisition_frame_index,
            dtype="<i8",
        )
        target_valid = tracking.create_dataset(
            TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
            data=plan.target_source_acquisition_frame_valid,
            dtype="bool",
        )
        _write_contract_attrs(chaser, row_identity_contract_attrs(identity_contract))
        _write_contract_attrs(state_key, row_identity_key_attrs(identity_contract))
        _write_contract_attrs(
            state_time,
            {
                SOURCE_ACQUISITION_MAPPING_RECORD_ATTR: state_mapping,
                SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: (
                    canonical_mapping_digest(state_mapping)
                ),
            },
        )
        target_mapping_digest = canonical_mapping_digest(target_mapping)
        _write_contract_attrs(
            target_time,
            {
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR: target_mapping,
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: (
                    target_mapping_digest
                ),
            },
        )
        _write_contract_attrs(
            target_valid,
            {
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR: (
                    f"{TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH}@"
                    f"{TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}"
                ),
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: (
                    target_mapping_digest
                ),
            },
        )

        arena = h5["/calibration_snapshot/arena_geometry"]
        arena_record = arena_geometry_record(dict(arena.attrs))
        arena_digest = canonical_mapping_digest(arena_record)
        arena_frame = source_arena_pixel_frame_record(arena_record)
        arena_frame_digest = canonical_mapping_digest(arena_frame)
        for name in (
            "arena_region_width_px",
            "arena_region_height_px",
            "arena_origin_in_canvas_x_px",
            "arena_origin_in_canvas_y_px",
        ):
            del arena.attrs[name]
            arena.attrs[name] = int(arena_record[name])
        _write_contract_attrs(
            arena,
            {
                ARENA_GEOMETRY_RECORD_ATTR: arena_record,
                ARENA_GEOMETRY_RECORD_DIGEST_ATTR: arena_digest,
                PIXEL_FRAME_AUTHORITY_ATTR: arena_frame,
                PIXEL_FRAME_AUTHORITY_DIGEST_ATTR: arena_frame_digest,
            },
        )

        frame_ref = (
            f"/calibration_snapshot/arena_geometry@{PIXEL_FRAME_AUTHORITY_ATTR}"
        )
        descriptor = build_canonical_coordinate_descriptor(
            profile_id="arena_relative_canvas_px.top_left_y_down.v1",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            reference_width=int(arena_record["arena_region_width_px"]),
            reference_height=int(arena_record["arena_region_height_px"]),
            reference_authority=DigestBoundCoordinateRecordRef(
                record_ref=frame_ref,
                record_sha256=arena_frame_digest,
            ),
            reference_selector="record",
            pixel_convention="continuous",
            row_identity_contract=identity_contract,
            row_identity_record_ref=(
                f"/tracking_data/chaser_states@{ROW_IDENTITY_CONTRACT_ATTR}"
            ),
            source_camera_overlay_status="not_suitable",
            frame_record=CanonicalFrameRecord(
                kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
                record_ref=frame_ref,
                record_sha256=arena_frame_digest,
            ),
        )
        manifest = _surface_manifest(tuple(chaser.dtype.names or ()))
        _write_contract_attrs(
            chaser,
            {
                "schema_id": CHASER_STATES_SCHEMA_ID,
                "schema_version": CHASER_STATES_SCHEMA_VERSION,
                **canonical_coordinate_descriptor_v2_attrs(descriptor),
                COORDINATE_SURFACE_MANIFEST_ATTR: manifest,
                COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR: (
                    canonical_mapping_digest(manifest)
                ),
            },
        )

        if STIMULUS_RENDERER_SNAPSHOT_PATH in h5:
            raise BatmanMigrationError(
                "Legacy source unexpectedly contains stimulus_renderer_snapshot."
            )
        h5.move("/stimulus_coordinates", STIMULUS_RENDERER_SNAPSHOT_PATH)
        renderer = h5[STIMULUS_RENDERER_SNAPSHOT_PATH]
        for name in tuple(renderer.attrs.keys()):
            del renderer.attrs[name]
        renderer.attrs.update(
            {
                "schema_id": STIMULUS_RENDERER_SNAPSHOT_SCHEMA_ID,
                "schema_version": STIMULUS_RENDERER_SNAPSHOT_SCHEMA_VERSION,
                "capture_phase": STIMULUS_RENDERER_SNAPSHOT_CAPTURE_PHASE,
            }
        )
        if "/tracking_data/bounding_boxes" in h5:
            del h5["/tracking_data/bounding_boxes"]

        receipt = _migration_receipt(
            plan,
            identity_contract=identity_contract,
            state_mapping=state_mapping,
            target_mapping=target_mapping,
            arena_record=arena_record,
        )
        migration = h5.create_group(MIGRATION_GROUP_PATH)
        _write_contract_attrs(
            migration,
            {
                MIGRATION_RECEIPT_ATTR: receipt,
                MIGRATION_RECEIPT_DIGEST_ATTR: canonical_mapping_digest(receipt),
            },
        )
        h5.flush()
    return receipt


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def migrate_recording(
    recording_dir: Path,
    *,
    apply: bool = False,
    output_h5: Path | None = None,
) -> dict[str, Any]:
    plan = plan_migration(recording_dir, output_h5=output_h5)
    summary = {
        "recording_dir": str(plan.recording_dir),
        "source_h5": str(plan.source_h5),
        "output_h5": str(plan.output_h5),
        "external_receipt": str(plan.external_receipt),
        "recording_id": plan.recording_id,
        "camera_id": plan.camera_id,
        "source_total_frames": plan.total_frames,
        "source_row_count": plan.source_row_count,
        "supported_row_count": plan.supported_row_count,
        "omitted_terminal_row_key": list(plan.terminal_row_key),
        "status": "would_migrate",
    }
    if not apply:
        return summary
    if plan.output_h5.exists() or plan.external_receipt.exists():
        raise BatmanMigrationError(
            "Derivative output or receipt already exists; immutable migration "
            "artifacts are never overwritten."
        )
    plan.output_h5.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{plan.output_h5.name}.",
        suffix=".tmp",
        dir=plan.output_h5.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(plan.source_h5, temp_path)
        receipt = _mutate_derivative(temp_path, plan)
        with h5py.File(temp_path, "r") as derivative:
            preflight = preflight_stimulus_coordinate_contract(
                derivative,
                source_h5=temp_path,
            )
            if (
                not preflight.has_chaser_states
                or preflight.renderer_snapshot_source_path
                != STIMULUS_RENDERER_SNAPSHOT_PATH
            ):
                raise BatmanMigrationError(
                    "Canonical derivative failed post-migration coordinate preflight."
                )
        os.replace(temp_path, plan.output_h5)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    derivative_digest = _sha256_file(plan.output_h5)
    external = {
        "schema_id": EXTERNAL_RECEIPT_SCHEMA_ID,
        "schema_version": 1,
        "derivative_h5": str(plan.output_h5),
        "derivative_h5_sha256": derivative_digest,
        "migration_receipt_ref": f"{MIGRATION_GROUP_PATH}@{MIGRATION_RECEIPT_ATTR}",
        "migration_receipt_sha256": canonical_mapping_digest(receipt),
        "source_h5": str(plan.source_h5),
        "source_h5_sha256": next(
            record["sha256"]
            for record in plan.artifact_records
            if record["role"] == "source_h5"
        ),
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    _write_json_atomic(plan.external_receipt, external)
    summary.update(
        {
            "status": "migrated",
            "derivative_h5_sha256": derivative_digest,
            "migration_receipt_sha256": canonical_mapping_digest(receipt),
        }
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording_dirs",
        nargs="+",
        type=Path,
        help="Organized Batman recording directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write immutable derivative H5s; default is read-only dry-run.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    results = [
        migrate_recording(recording_dir, apply=bool(args.apply))
        for recording_dir in args.recording_dirs
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
