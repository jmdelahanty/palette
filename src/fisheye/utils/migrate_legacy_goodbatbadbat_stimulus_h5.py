#!/usr/bin/env python3
"""Create canonical v5 derivatives of the audited legacy GoodBatBadBat H5s.

The transferred H5 and Orange sidecars are opened read-only. ``--apply`` writes
an immutable derivative and receipt below the recording's ``derived`` tree; it
never edits or replaces a raw artifact.
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
    STIMULUS_RENDERER_SNAPSHOT_PATH,
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


SUPPORTED_PROTOCOL = "goodbatbadbat"
SUPPORTED_CITRUS_VERSION = "v1.2.1-1529-g6827d7c"
SUPPORTED_CITRUS_VERSIONS = frozenset(
    {
        SUPPORTED_CITRUS_VERSION,
        f"{SUPPORTED_CITRUS_VERSION}-dirty",
    }
)
EXPECTED_CHASER_COUNT = 2
DERIVATIVE_SUFFIX = ".canonical_stimulus_v1.h5"

MIGRATION_SCHEMA_ID = "palette.legacy_goodbatbadbat_stimulus_h5_migration"
MIGRATION_SCHEMA_VERSION = 2
MIGRATION_GROUP_PATH = "/palette_migration"
MIGRATION_RECEIPT_ATTR = "migration_receipt"
MIGRATION_RECEIPT_DIGEST_ATTR = f"{MIGRATION_RECEIPT_ATTR}_sha256"
EXTERNAL_RECEIPT_SCHEMA_ID = "palette.stimulus_h5_derivative_artifact"

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
        "component_fields": ["target_clamped_pos_x", "target_clamped_pos_y"],
    },
)


class GoodBatBadBatMigrationError(ValueError):
    """Raised when recovery evidence is incomplete or contradictory."""


@dataclass(frozen=True)
class H5Evidence:
    row_identity: np.ndarray = field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = field(repr=False, compare=False)
    target_source_acquisition_frame_index: np.ndarray = field(
        repr=False,
        compare=False,
    )
    target_source_acquisition_frame_valid: np.ndarray = field(
        repr=False,
        compare=False,
    )
    chaser_indices: tuple[int, ...]
    stimulus_frame_count: int
    frame_metadata_row_count: int
    frame_metadata_coverage_mode: str
    frame_metadata_prefix_row_count: int
    frame_metadata_prefix_stimulus_frame_start: int | None
    frame_metadata_prefix_stimulus_frame_end: int | None
    chaser_stimulus_frame_start: int
    chaser_stimulus_frame_end: int
    source_chaser_sha256: str
    source_frame_metadata_sha256: str
    source_bounding_boxes_sha256: str | None
    renderer_snapshot: Mapping[str, Any]
    renderer_snapshot_sha256: str


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
    source_citrus_version: str
    total_frames: int
    jsonl_path: Path
    csv_path: Path
    recording_snapshot_path: Path
    recording_session_path: Path
    artifact_records: tuple[Mapping[str, Any], ...]
    evidence: H5Evidence

    @property
    def supported_row_count(self) -> int:
        return int(self.evidence.row_identity.shape[0])


def _json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoodBatBadBatMigrationError(
            f"Unable to read JSON artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GoodBatBadBatMigrationError(
            f"JSON artifact {path} must contain an object."
        )
    return value


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoodBatBadBatMigrationError(f"{label} must be non-empty text.")
    return value.strip()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GoodBatBadBatMigrationError(f"{label} must be a positive integer.")
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise GoodBatBadBatMigrationError(
            f"Unable to hash required artifact {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _artifact_record(
    path: Path,
    *,
    recording_dir: Path,
    role: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise GoodBatBadBatMigrationError(
            f"Required {role} artifact is missing: {path}"
        )
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
        raise GoodBatBadBatMigrationError(
            f"{label} must resolve exactly one file; found {len(candidates)}."
        )
    return candidates[0]


def _manifest_paths(
    recording_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path, int, str, str, str, str]:
    if manifest.get("protocol_name") != SUPPORTED_PROTOCOL:
        raise GoodBatBadBatMigrationError(
            f"Migration is restricted to {SUPPORTED_PROTOCOL!r} recordings."
        )
    source_citrus_version = _required_text(
        manifest.get("software_version"),
        label="manifest software_version",
    )
    if source_citrus_version not in SUPPORTED_CITRUS_VERSIONS:
        raise GoodBatBadBatMigrationError(
            "Migration is restricted to audited Citrus versions: "
            f"{sorted(SUPPORTED_CITRUS_VERSIONS)}."
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
        raise GoodBatBadBatMigrationError(
            "Manifest must declare recording_frame_id as the authoritative frame clock."
        )
    stream_map = streams.get("streams")
    full = stream_map.get("full") if isinstance(stream_map, Mapping) else None
    if not isinstance(full, Mapping):
        raise GoodBatBadBatMigrationError("Manifest lacks the authoritative full stream.")
    if (
        str(full.get("camera_id")) != camera_id
        or full.get("frame_clock") != "recording_frame_id"
        or full.get("role") != "ingest_authoritative_full_frame"
    ):
        raise GoodBatBadBatMigrationError(
            "Manifest full-stream identity or frame-domain declaration is invalid."
        )
    total_frames = _positive_int(full.get("frame_count"), label="full frame_count")
    csv_path = (
        recording_dir
        / _required_text(
            full.get("frame_clock_metadata"),
            label="full frame_clock_metadata",
        )
    ).resolve()
    raw_files = manifest.get("files", {}).get("raw", [])
    h5_candidates = [
        recording_dir / relative
        for relative in raw_files
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
        source_citrus_version,
    )


def _validate_session_and_snapshot(
    *,
    recording_session: Mapping[str, Any],
    recording_snapshot: Mapping[str, Any],
    camera_id: str,
    total_frames: int,
) -> None:
    artifacts = recording_session.get("camera_artifacts")
    camera = artifacts.get(camera_id) if isinstance(artifacts, Mapping) else None
    if not isinstance(camera, Mapping) or any(
        camera.get(name) != expected
        for name, expected in (
            ("first_recording_frame_id", 1),
            ("last_recording_frame_id", total_frames),
            ("frame_count", total_frames),
            ("recording_frame_id_gaps", 0),
        )
    ):
        raise GoodBatBadBatMigrationError(
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
        raise GoodBatBadBatMigrationError(
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
                raise GoodBatBadBatMigrationError(
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
                raise GoodBatBadBatMigrationError(
                    f"Orange JSONL identity is invalid at line {line_number}."
                )
            try:
                ipc_id = int(frame["ipc_frame_id"])
                recording_frame_id = int(frame["recording_frame_id"])
                camera_frame_id = int(frame["camera_frame_id"])
                camera_index = int(event["camera_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GoodBatBadBatMigrationError(
                    f"Orange JSONL frame identity is invalid at line {line_number}."
                ) from exc
            if (
                ipc_id < 1
                or ipc_id > total_frames
                or recording_frame_id != ipc_id
                or recording_by_ipc[ipc_id] != -1
            ):
                raise GoodBatBadBatMigrationError(
                    "Orange JSONL IDs are duplicate, out of range, or disagree "
                    f"at line {line_number}."
                )
            recording_by_ipc[ipc_id] = recording_frame_id
            camera_index_by_ipc[ipc_id] = camera_index
            camera_frame_by_recording[recording_frame_id] = camera_frame_id
            jsonl_rows += 1
    if jsonl_rows != total_frames or np.any(recording_by_ipc[1:] < 1):
        raise GoodBatBadBatMigrationError(
            "Orange JSONL does not provide one complete identity row per acquisition frame."
        )
    if np.unique(camera_index_by_ipc[1:]).size != 1:
        raise GoodBatBadBatMigrationError(
            "Orange JSONL does not bind the recording to one camera index."
        )

    csv_camera_frames = np.full(total_frames + 1, -1, dtype=np.int64)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"recording_frame_id", "camera_frame_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise GoodBatBadBatMigrationError(
                "Camera metadata CSV lacks recording_frame_id or camera_frame_id."
            )
        csv_rows = 0
        for row_offset, row in enumerate(reader):
            try:
                recording_frame_id = int(row["recording_frame_id"])
                camera_frame_id = int(row["camera_frame_id"])
            except (TypeError, ValueError) as exc:
                raise GoodBatBadBatMigrationError(
                    f"Camera metadata CSV identity is invalid at row {row_offset + 2}."
                ) from exc
            if (
                recording_frame_id != row_offset + 1
                or recording_frame_id > total_frames
                or csv_camera_frames[recording_frame_id] != -1
            ):
                raise GoodBatBadBatMigrationError(
                    "Camera metadata CSV is not the exact contiguous one-based full-stream clock."
                )
            csv_camera_frames[recording_frame_id] = camera_frame_id
            csv_rows += 1
    if (
        csv_rows != total_frames
        or np.any(csv_camera_frames[1:] < 0)
        or not np.array_equal(csv_camera_frames[1:], camera_frame_by_recording[1:])
    ):
        raise GoodBatBadBatMigrationError(
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
        raise GoodBatBadBatMigrationError(
            "Legacy chaser_states attrs do not match the audited layout."
        )
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
        raise GoodBatBadBatMigrationError(
            "Legacy chaser coordinate semantics do not match the audited contract."
        )


def _map_frame_rows(
    *,
    frame_stimulus: np.ndarray,
    frame_values: np.ndarray,
    chaser_stimulus: np.ndarray,
) -> np.ndarray:
    order = np.argsort(frame_stimulus, kind="stable")
    sorted_stimulus = frame_stimulus[order]
    positions = np.searchsorted(sorted_stimulus, chaser_stimulus)
    if (
        np.any(positions >= sorted_stimulus.size)
        or not np.array_equal(sorted_stimulus[positions], chaser_stimulus)
    ):
        raise GoodBatBadBatMigrationError(
            "A chaser row has no exact stimulus-frame metadata identity."
        )
    return frame_values[order[positions]]


def _h5_evidence(
    *,
    source_h5: Path,
    recording_by_ipc: np.ndarray,
    camera_index_by_ipc: np.ndarray,
    camera_id: str,
) -> H5Evidence:
    with h5py.File(source_h5, "r") as h5:
        renderer, renderer_path, renderer_digest = _classify_renderer_snapshot(h5)
        if (
            renderer is None
            or renderer_path != STIMULUS_RENDERER_SNAPSHOT_PATH
            or renderer_digest is None
            or "/stimulus_coordinates" in h5
        ):
            raise GoodBatBadBatMigrationError(
                "GoodBatBadBat H5 lacks the exact canonical renderer snapshot."
            )
        chaser_path = "/tracking_data/chaser_states"
        frame_path = "/video_metadata/frame_metadata"
        if chaser_path not in h5 or frame_path not in h5:
            raise GoodBatBadBatMigrationError(
                "GoodBatBadBat H5 lacks chaser_states or frame_metadata."
            )
        chaser = h5[chaser_path]
        frames = h5[frame_path]
        if (
            not isinstance(chaser, h5py.Dataset)
            or not isinstance(frames, h5py.Dataset)
            or tuple(chaser.dtype.names or ()) != _CHASER_FIELDS
        ):
            raise GoodBatBadBatMigrationError(
                "GoodBatBadBat chaser_states dtype is outside the audited layout."
            )
        _validate_legacy_chaser_attrs(chaser.attrs, camera_id=camera_id)
        required_frame_fields = {"stimulus_frame_num", "triggering_camera_frame_id"}
        if not required_frame_fields.issubset(frames.dtype.names or ()):
            raise GoodBatBadBatMigrationError(
                "frame_metadata lacks the audited identity fields."
            )
        if any(
            f"/tracking_data/{name}" in h5
            for name in (
                STIMULUS_STATE_KEY_ARRAY_REF,
                SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
                TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
                TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
            )
        ):
            raise GoodBatBadBatMigrationError(
                "Legacy source unexpectedly contains canonical tracking arrays."
            )
        if MIGRATION_GROUP_PATH in h5:
            raise GoodBatBadBatMigrationError(
                "Legacy source unexpectedly contains a Palette migration receipt."
            )

        chaser_indices = np.asarray(chaser["chaser_index"][:], dtype=np.int64)
        chaser_stimulus = np.asarray(chaser["stimulus_frame_num"][:], dtype=np.int64)
        if np.any(chaser_indices < 0) or np.any(chaser_stimulus < 0):
            raise GoodBatBadBatMigrationError("Chaser row identity contains negative values.")
        row_identity = np.column_stack((chaser_indices, chaser_stimulus)).astype(
            "<i8",
            copy=False,
        )
        if np.unique(row_identity, axis=0).shape[0] != row_identity.shape[0]:
            raise GoodBatBadBatMigrationError(
                "GoodBatBadBat composite stimulus-state identity is not unique."
            )
        unique_chasers = tuple(int(value) for value in np.unique(chaser_indices))
        unique_stimulus, rows_per_stimulus = np.unique(
            chaser_stimulus,
            return_counts=True,
        )
        if (
            len(unique_chasers) != EXPECTED_CHASER_COUNT
            or np.any(rows_per_stimulus != EXPECTED_CHASER_COUNT)
        ):
            raise GoodBatBadBatMigrationError(
                "Migration requires exactly two distinct chaser rows per stimulus frame."
            )

        frame_stimulus = np.asarray(frames["stimulus_frame_num"][:], dtype=np.int64)
        triggering_ids = np.asarray(
            frames["triggering_camera_frame_id"][:],
            dtype=np.int64,
        )
        if np.any(frame_stimulus < 0):
            raise GoodBatBadBatMigrationError(
                "frame_metadata stimulus-frame identity contains negative values."
            )
        frame_metadata_stimulus, frame_metadata_counts = np.unique(
            frame_stimulus,
            return_counts=True,
        )
        if np.any(frame_metadata_counts != 1):
            raise GoodBatBadBatMigrationError(
                "frame_metadata stimulus-frame identity must be unique."
            )
        missing_chaser_stimulus = np.setdiff1d(
            unique_stimulus,
            frame_metadata_stimulus,
            assume_unique=True,
        )
        if missing_chaser_stimulus.size:
            raise GoodBatBadBatMigrationError(
                "frame_metadata is missing one or more chaser stimulus frames."
            )
        extra_frame_metadata_stimulus = np.setdiff1d(
            frame_metadata_stimulus,
            unique_stimulus,
            assume_unique=True,
        )
        if extra_frame_metadata_stimulus.size:
            expected_prefix = np.arange(
                int(extra_frame_metadata_stimulus[0]),
                int(unique_stimulus[0]),
                dtype=np.int64,
            )
            if not np.array_equal(extra_frame_metadata_stimulus, expected_prefix):
                raise GoodBatBadBatMigrationError(
                    "frame_metadata rows outside chaser coverage must form one "
                    "contiguous pre-chaser prefix."
                )
            frame_metadata_coverage_mode = "contiguous_pre_chaser_prefix_v1"
            frame_metadata_prefix_start = int(extra_frame_metadata_stimulus[0])
            frame_metadata_prefix_end = int(extra_frame_metadata_stimulus[-1])
        else:
            frame_metadata_coverage_mode = "exact_chaser_frame_set_v1"
            frame_metadata_prefix_start = None
            frame_metadata_prefix_end = None
        chaser_trigger_ids = _map_frame_rows(
            frame_stimulus=frame_stimulus,
            frame_values=triggering_ids,
            chaser_stimulus=chaser_stimulus,
        )
        if (
            np.any(chaser_trigger_ids < 1)
            or np.any(chaser_trigger_ids >= recording_by_ipc.shape[0])
            or np.any(recording_by_ipc[chaser_trigger_ids] < 1)
        ):
            raise GoodBatBadBatMigrationError(
                "A stimulus trigger ID has no exact Orange IPC identity."
            )
        state_acquisition = np.asarray(
            recording_by_ipc[chaser_trigger_ids] - 1,
            dtype="<i8",
        )

        target_ids = np.asarray(chaser["target_source_frame_id"][:], dtype=np.int64)
        target_camera_indices = np.asarray(
            chaser["target_source_camera_id"][:],
            dtype=np.int64,
        )
        target_valid = target_ids > 0
        target_acquisition = np.full(chaser.shape[0], -1, dtype="<i8")
        valid_ids = target_ids[target_valid]
        if (
            np.any(valid_ids >= recording_by_ipc.shape[0])
            or np.any(recording_by_ipc[valid_ids] < 1)
            or not np.array_equal(
                target_camera_indices[target_valid],
                camera_index_by_ipc[valid_ids],
            )
        ):
            raise GoodBatBadBatMigrationError(
                "Target-source IDs do not map exactly to matching Orange camera evidence."
            )
        target_acquisition[target_valid] = recording_by_ipc[valid_ids] - 1

        bbox_digest = None
        if "/tracking_data/bounding_boxes" in h5:
            bbox = h5["/tracking_data/bounding_boxes"]
            if not isinstance(bbox, h5py.Dataset):
                raise GoodBatBadBatMigrationError(
                    "Legacy bounding_boxes must be a dataset."
                )
            bbox_digest = _h5_dataset_content_digest(bbox)
        return H5Evidence(
            row_identity=row_identity,
            source_acquisition_frame_index=state_acquisition,
            target_source_acquisition_frame_index=target_acquisition,
            target_source_acquisition_frame_valid=np.asarray(target_valid, dtype=bool),
            chaser_indices=unique_chasers,
            stimulus_frame_count=int(unique_stimulus.size),
            frame_metadata_row_count=int(frames.shape[0]),
            frame_metadata_coverage_mode=frame_metadata_coverage_mode,
            frame_metadata_prefix_row_count=int(extra_frame_metadata_stimulus.size),
            frame_metadata_prefix_stimulus_frame_start=frame_metadata_prefix_start,
            frame_metadata_prefix_stimulus_frame_end=frame_metadata_prefix_end,
            chaser_stimulus_frame_start=int(unique_stimulus[0]),
            chaser_stimulus_frame_end=int(unique_stimulus[-1]),
            source_chaser_sha256=_h5_dataset_content_digest(chaser),
            source_frame_metadata_sha256=_h5_dataset_content_digest(frames),
            source_bounding_boxes_sha256=bbox_digest,
            renderer_snapshot=dict(renderer),
            renderer_snapshot_sha256=renderer_digest,
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
        source_citrus_version,
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
    evidence = _h5_evidence(
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
        raise GoodBatBadBatMigrationError(
            "Derivative output must not replace the source H5."
        )
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
        source_citrus_version=source_citrus_version,
        total_frames=total_frames,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        recording_snapshot_path=snapshot_path,
        recording_session_path=session_path,
        artifact_records=artifact_records,
        evidence=evidence,
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
    evidence = plan.evidence
    identity_digest = identity_array_content_sha256(evidence.row_identity)
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
            evidence.source_acquisition_frame_index
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
            evidence.target_source_acquisition_frame_index
        ),
        "validity_array_ref": TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH,
        "validity_array_dtype": np.dtype("bool").str,
        "validity_array_shape": [plan.supported_row_count],
        "validity_array_content_sha256": numpy_content_digest(
            evidence.target_source_acquisition_frame_valid
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
    evidence = plan.evidence
    omitted_paths = (
        ["/tracking_data/bounding_boxes"]
        if evidence.source_bounding_boxes_sha256 is not None
        else []
    )
    return {
        "schema_id": MIGRATION_SCHEMA_ID,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_method": "legacy_goodbatbadbat_orange_identity_join_v2",
        "source_protocol": SUPPORTED_PROTOCOL,
        "source_citrus_version": plan.source_citrus_version,
        "source_artifacts": list(plan.artifact_records),
        "source_chaser_states_sha256": evidence.source_chaser_sha256,
        "source_frame_metadata_sha256": evidence.source_frame_metadata_sha256,
        "source_bounding_boxes_sha256": evidence.source_bounding_boxes_sha256,
        "source_renderer_snapshot": dict(evidence.renderer_snapshot),
        "source_renderer_snapshot_sha256": evidence.renderer_snapshot_sha256,
        "source_row_count": plan.supported_row_count,
        "supported_row_count": plan.supported_row_count,
        "stimulus_frame_count": evidence.stimulus_frame_count,
        "frame_metadata_row_count": evidence.frame_metadata_row_count,
        "frame_metadata_coverage": {
            "mode": evidence.frame_metadata_coverage_mode,
            "chaser_stimulus_frame_start": evidence.chaser_stimulus_frame_start,
            "chaser_stimulus_frame_end": evidence.chaser_stimulus_frame_end,
            "pre_chaser_prefix_row_count": evidence.frame_metadata_prefix_row_count,
            "pre_chaser_prefix_stimulus_frame_start": (
                evidence.frame_metadata_prefix_stimulus_frame_start
            ),
            "pre_chaser_prefix_stimulus_frame_end": (
                evidence.frame_metadata_prefix_stimulus_frame_end
            ),
            "source_frame_metadata_preserved_in_derivative": True,
        },
        "chaser_indices": list(evidence.chaser_indices),
        "rows_per_stimulus_frame": EXPECTED_CHASER_COUNT,
        "omitted_rows": [],
        "omitted_source_paths": omitted_paths,
        "preserved_renderer_snapshot_path": STIMULUS_RENDERER_SNAPSHOT_PATH,
        "row_identity_contract": identity_contract.to_dict(),
        "row_identity_contract_sha256": identity_contract.digest(),
        "stimulus_state_key_sha256": identity_array_content_sha256(
            evidence.row_identity
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
            "legacy_trigger_id_joined_through_orange_ipc_identity": True,
            "ipc_frame_id_equals_recording_frame_id": True,
            "csv_row_offset_equals_recording_frame_id_minus_one": True,
            "jsonl_csv_camera_frame_id_equal": True,
            "frame_metadata_unique_by_stimulus_frame": True,
            "chaser_rows_resolve_exactly_one_frame_metadata_row": True,
            "non_chaser_frame_metadata_is_contiguous_pre_chaser_prefix": True,
            "non_chaser_frame_metadata_row_count": (
                evidence.frame_metadata_prefix_row_count
            ),
            "duplicate_composite_keys": 0,
            "missing_ids": 0,
            "camera_mismatches": 0,
        },
        "bounding_boxes_status": (
            "omitted_pending_separate_camera_native_import"
            if evidence.source_bounding_boxes_sha256 is not None
            else "not_present"
        ),
        "canonicalization": "canonical_json_sort_keys_v1",
    }


def _mutate_derivative(path: Path, plan: MigrationPlan) -> dict[str, Any]:
    evidence = plan.evidence
    with h5py.File(path, "r+") as h5:
        chaser = h5["/tracking_data/chaser_states"]
        if int(chaser.shape[0]) != plan.supported_row_count:
            raise GoodBatBadBatMigrationError(
                "Derivative chaser row count changed before canonicalization."
            )
        if _h5_dataset_content_digest(chaser) != evidence.source_chaser_sha256:
            raise GoodBatBadBatMigrationError(
                "Derivative chaser payload differs from the audited raw source."
            )
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
                raise GoodBatBadBatMigrationError(
                    f"Legacy source unexpectedly contains canonical dataset {name}."
                )

        identity_contract = build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=evidence.row_identity,
            components=_ROW_IDENTITY_FIELDS,
        )
        state_mapping, target_mapping = _mapping_records(
            plan,
            identity_contract=identity_contract,
        )
        state_key = tracking.create_dataset(
            STIMULUS_STATE_KEY_ARRAY_REF,
            data=evidence.row_identity,
            dtype="<i8",
        )
        state_time = tracking.create_dataset(
            SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=evidence.source_acquisition_frame_index,
            dtype="<i8",
        )
        target_time = tracking.create_dataset(
            TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=evidence.target_source_acquisition_frame_index,
            dtype="<i8",
        )
        target_valid = tracking.create_dataset(
            TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
            data=evidence.target_source_acquisition_frame_valid,
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

        if "/tracking_data/bounding_boxes" in h5:
            del h5["/tracking_data/bounding_boxes"]
        if MIGRATION_GROUP_PATH in h5:
            raise GoodBatBadBatMigrationError(
                "Legacy source unexpectedly contains a migration group."
            )
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
    evidence = plan.evidence
    summary = {
        "recording_dir": str(plan.recording_dir),
        "source_h5": str(plan.source_h5),
        "output_h5": str(plan.output_h5),
        "external_receipt": str(plan.external_receipt),
        "recording_id": plan.recording_id,
        "camera_id": plan.camera_id,
        "source_citrus_version": plan.source_citrus_version,
        "source_total_frames": plan.total_frames,
        "source_row_count": plan.supported_row_count,
        "supported_row_count": plan.supported_row_count,
        "stimulus_frame_count": evidence.stimulus_frame_count,
        "frame_metadata_row_count": evidence.frame_metadata_row_count,
        "frame_metadata_coverage_mode": evidence.frame_metadata_coverage_mode,
        "frame_metadata_prefix_row_count": evidence.frame_metadata_prefix_row_count,
        "frame_metadata_prefix_stimulus_frame_start": (
            evidence.frame_metadata_prefix_stimulus_frame_start
        ),
        "frame_metadata_prefix_stimulus_frame_end": (
            evidence.frame_metadata_prefix_stimulus_frame_end
        ),
        "chaser_indices": list(evidence.chaser_indices),
        "source_acquisition_frame_index_sha256": numpy_content_digest(
            evidence.source_acquisition_frame_index
        ),
        "target_source_acquisition_frame_index_sha256": numpy_content_digest(
            evidence.target_source_acquisition_frame_index
        ),
        "target_source_acquisition_frame_valid_sha256": numpy_content_digest(
            evidence.target_source_acquisition_frame_valid
        ),
        "status": "would_migrate",
    }
    if not apply:
        return summary
    if plan.output_h5.exists() or plan.external_receipt.exists():
        raise GoodBatBadBatMigrationError(
            "Derivative output or receipt already exists; immutable migration "
            "artifacts are never overwritten."
        )
    source_digest_before = next(
        record["sha256"]
        for record in plan.artifact_records
        if record["role"] == "source_h5"
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
                or preflight.row_identity_values is None
                or not np.array_equal(
                    preflight.row_identity_values,
                    evidence.row_identity,
                )
                or _h5_dataset_content_digest(
                    derivative["/tracking_data/chaser_states"]
                )
                != evidence.source_chaser_sha256
                or "/tracking_data/bounding_boxes" in derivative
            ):
                raise GoodBatBadBatMigrationError(
                    "Canonical derivative failed post-migration validation."
                )
        if _sha256_file(plan.source_h5) != source_digest_before:
            raise GoodBatBadBatMigrationError(
                "Raw source H5 changed during derivative construction."
            )
        derivative_digest = _sha256_file(temp_path)
        os.replace(temp_path, plan.output_h5)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    external = {
        "schema_id": EXTERNAL_RECEIPT_SCHEMA_ID,
        "schema_version": 1,
        "derivative_h5": str(plan.output_h5),
        "derivative_h5_sha256": derivative_digest,
        "migration_receipt_ref": f"{MIGRATION_GROUP_PATH}@{MIGRATION_RECEIPT_ATTR}",
        "migration_receipt_sha256": canonical_mapping_digest(receipt),
        "migration_schema_id": MIGRATION_SCHEMA_ID,
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "source_citrus_version": plan.source_citrus_version,
        "source_h5": str(plan.source_h5),
        "source_h5_sha256": source_digest_before,
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    _write_json_atomic(plan.external_receipt, external)
    summary.update(
        {
            "status": "migrated",
            "derivative_h5_sha256": derivative_digest,
            "migration_receipt_sha256": canonical_mapping_digest(receipt),
            "raw_source_unchanged": True,
        }
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording_dirs",
        nargs="+",
        type=Path,
        help="Organized legacy GoodBatBadBat recording directories.",
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
