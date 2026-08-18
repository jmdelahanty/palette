"""Selector-ineligible fish-position-provider chaser-distance candidates.

This module is the canary boundary between an explicit subject-position
provider and the established chaser-distance science.  It deliberately does
not publish ``analysis/chaser_distance_runs`` or mutate any selector.  A later
reviewed promotion can use the candidate evidence to define a sealed generic
position-source contract without relabelling provider rows as detections.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Optional, Sequence
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_distance_coordinate_publication import (
    CHASER_POSITION_ARRAY,
    _apply_homography,
    _frame_pointer,
    _identity_component,
    _positive_fps,
)
from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceWindow,
)
from fisheye.analysis.stimulus_epoch_consumer import (
    StimulusEpochCompatibilityPolicy,
    read_stimulus_epoch_snapshot,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_EPOCH_RUN_MANIFEST_ATTR,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
    load_subject_position_source_handle,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.directed_transform_v2 import (
    apply_bound_directed_transform_v2,
    require_bound_directed_transform_v2,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.stimulus_coordinate_contract import (
    BoundStimulusCoordinateEvidence,
    load_bound_stimulus_coordinate_evidence,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)


SCHEMA_ID = "palette.provider_chaser_distance_candidate"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_ID = "palette.provider_chaser_distance_candidate_manifest"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_ATTR = "provider_chaser_distance_candidate_manifest"
MANIFEST_DIGEST_ATTR = "provider_chaser_distance_candidate_manifest_sha256"
PUBLISH_SCHEMA_ID = "palette.provider_chaser_distance_candidate_publish.v1"
PARENT_PATH = "analysis/provider_chaser_distance_candidate_runs"
METHOD = "explicit_subject_position_to_chaser_distance_candidate"
METHOD_VERSION = 1
PUBLICATION_POLICY = "atomic_complete_selector_ineligible_no_selector_mutation_v1"
MISSING_SOURCE_ROW = -1
MISSING_FAILURE_REASON = np.iinfo(np.uint16).max
_SELECTOR_ATTRS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "selected",
        "default",
    }
)


class ProviderChaserDistanceCandidateError(ValueError):
    """Raised when a provider candidate cannot remain exact and non-promoting."""


def _fail(message: str) -> None:
    raise ProviderChaserDistanceCandidateError(message)


def _controlled_run_path(value: object, *, parent: str, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be one exact string path.")
    path = value.strip().strip("/")
    prefix = f"{parent}/"
    if (
        value != path
        or not path.startswith(prefix)
        or path.count("/") != parent.count("/") + 1
        or path.rsplit("/", 1)[-1] in {"", ".", "..", "latest", "default"}
    ):
        _fail(f"{label} must name one exact child below {parent!r}.")
    return path


def _controlled_name(value: object, *, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be one exact string.")
    name = value.strip()
    if not name or name != value or "/" in name or name in {".", "..", "latest"}:
        _fail(f"{label} must be one exact child name.")
    return name


def _read(node: Any, *, label: str) -> np.ndarray:
    try:
        return np.asarray(node[:])
    except Exception as exc:
        _fail(f"Unable to read {label}: {exc}.")


def _array_record(values: np.ndarray, *, path: str) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "path": path,
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
        "sha256": sha256_array(array),
    }


def _write_array(root: zarr.Group, path: str, values: np.ndarray) -> zarr.Array:
    parts = path.split("/")
    group = root
    for component in parts[:-1]:
        group = group.require_group(component)
    name = parts[-1]
    if name in group:
        raise RuntimeError(f"Candidate local materialization occupied {path!r}.")
    array = np.asarray(values)
    if array.ndim == 0:
        chunks = ()
    elif array.ndim == 1:
        chunks = (max(1, min(int(array.shape[0]), 8192)),)
    else:
        chunks = (max(1, min(int(array.shape[0]), 2048)), *array.shape[1:])
    return group.create_array(name, data=array, chunks=chunks, overwrite=False)


def _source_camera_to_arena_xy(
    points_xy: np.ndarray,
    stimulus: BoundStimulusCoordinateEvidence,
) -> np.ndarray:
    transform = stimulus.frame_transform
    camera_to_canvas = transform.canvas_to_source_camera.inverse_of
    if camera_to_canvas is None:
        _fail("Stimulus transform lacks its explicit camera-to-canvas direction.")
    canvas = apply_bound_directed_transform_v2(
        np.asarray(points_xy, dtype=np.float64),
        require_bound_directed_transform_v2(camera_to_canvas),
    )
    arena_to_canvas = require_bound_directed_transform_v2(transform.arena_to_canvas)
    return _apply_homography(canvas, np.linalg.inv(arena_to_canvas.matrix))


def _dense_fish_positions(
    *,
    total_frames: int,
    frames: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    source_rows: np.ndarray,
    instance_keys: np.ndarray,
    failure_reasons: np.ndarray,
) -> dict[str, np.ndarray]:
    if (
        frames.dtype != np.dtype("<i8")
        or frames.ndim != 1
        or positions.shape != (frames.size, 2)
        or positions.dtype.kind != "f"
        or valid.dtype != np.dtype(bool)
        or valid.shape != frames.shape
        or source_rows.dtype != np.dtype("<i8")
        or source_rows.shape != frames.shape
        or instance_keys.shape != frames.shape
        or instance_keys.dtype.kind not in "iu"
        or failure_reasons.dtype != np.dtype("<u2")
        or failure_reasons.shape != frames.shape
    ):
        _fail("Subject-position arrays have an unsupported dtype or aligned shape.")
    if frames.size and (
        int(np.min(frames)) < 0 or int(np.max(frames)) >= int(total_frames)
    ):
        _fail("Subject-position frame lineage lies outside the acquisition domain.")
    valid_rows = valid & np.isfinite(positions).all(axis=1)
    valid_frames = frames[valid_rows]
    if np.unique(valid_frames).size != valid_frames.size:
        _fail(
            "Single-subject provider candidate found duplicate valid position rows "
            "for one acquisition frame."
        )
    dense_position = np.full((total_frames, 2), np.nan, dtype=np.float32)
    dense_valid = np.zeros(total_frames, dtype=bool)
    dense_run_row = np.full(total_frames, MISSING_SOURCE_ROW, dtype=np.int64)
    dense_source_row = np.full(total_frames, MISSING_SOURCE_ROW, dtype=np.int64)
    dense_instance_key = np.zeros(total_frames, dtype=np.uint64)
    dense_failure = np.full(
        total_frames,
        MISSING_FAILURE_REASON,
        dtype=np.uint16,
    )
    seen_rows: set[int] = set()
    for run_row, frame in enumerate(frames.tolist()):
        frame_i = int(frame)
        if frame_i in seen_rows:
            # Multiple invalid observations at one frame are also ambiguous
            # lineage even though only duplicate valid rows are scientifically
            # selectable. Preserve the stricter single-subject boundary.
            _fail("Subject-position run has multiple rows for one acquisition frame.")
        seen_rows.add(frame_i)
        dense_run_row[frame_i] = int(run_row)
        dense_source_row[frame_i] = int(source_rows[run_row])
        dense_instance_key[frame_i] = np.uint64(instance_keys[run_row])
        dense_failure[frame_i] = np.uint16(failure_reasons[run_row])
        if valid_rows[run_row]:
            dense_position[frame_i] = positions[run_row].astype(np.float32)
            dense_valid[frame_i] = True
    return {
        "source_position_run_row_index": dense_run_row,
        "source_position_source_row_index": dense_source_row,
        "source_position_instance_key": dense_instance_key,
        "source_position_failure_reason_code": dense_failure,
        "fish_position_source_camera_xy": dense_position,
        "fish_valid": dense_valid,
    }


def _stimulus_sample_positions(
    stimulus: BoundStimulusCoordinateEvidence,
    chaser_group: Any,
    *,
    total_frames: int,
) -> dict[str, np.ndarray]:
    """Preserve every stimulus state while binding its acquisition source.

    Stimulus-state rows are keyed by ``(chaser_index, stimulus_frame_num)``.
    Several consecutive stimulus frames may legitimately bind the same source
    acquisition frame, so acquisition-frame deduplication is forbidden here.
    """

    chaser_xy = _read(chaser_group[CHASER_POSITION_ARRAY], label=CHASER_POSITION_ARRAY)
    acquisition_frames = np.asarray(
        stimulus.source_acquisition_frame_index,
        dtype=np.int64,
    )
    stimulus_frames = _identity_component(stimulus, "stimulus_frame_num")
    source_indices = _identity_component(stimulus, "chaser_index")
    if stimulus_frames is None or source_indices is None:
        _fail("Canonical stimulus rows require chaser_index and stimulus_frame_num.")
    stimulus_frames = np.asarray(stimulus_frames, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if "timestamp_ns_session" not in chaser_group:
        _fail("Canonical stimulus rows lack timestamp_ns_session.")
    timestamps = _read(
        chaser_group["timestamp_ns_session"],
        label="stimulus timestamp_ns_session",
    )
    source_rows = (
        _read(chaser_group["source_row_indices"], label="stimulus source_row_indices")
        if "source_row_indices" in chaser_group
        else np.arange(chaser_xy.shape[0], dtype=np.int64)
    )
    if (
        chaser_xy.ndim != 2
        or chaser_xy.shape[1] != 2
        or chaser_xy.dtype.kind != "f"
        or acquisition_frames.shape != (chaser_xy.shape[0],)
        or acquisition_frames.dtype != np.dtype("<i8")
        or stimulus_frames.shape != acquisition_frames.shape
        or source_indices.shape != acquisition_frames.shape
        or timestamps.shape != acquisition_frames.shape
        or timestamps.dtype != np.dtype("<i8")
        or source_rows.shape != acquisition_frames.shape
        or source_rows.dtype != np.dtype("<i8")
    ):
        _fail("Canonical stimulus chaser positions have an invalid exact layout.")
    indices = np.asarray(sorted(np.unique(source_indices).tolist()), dtype=np.int16)
    if indices.size == 0:
        _fail("Canonical stimulus run has no chaser rows.")
    sample_frames = np.asarray(sorted(np.unique(stimulus_frames).tolist()), dtype=np.int64)
    if sample_frames.size == 0:
        _fail("Canonical stimulus run has no stimulus samples.")
    sample_rows = np.searchsorted(sample_frames, stimulus_frames)
    columns = {int(value): index for index, value in enumerate(indices.tolist())}
    positions = np.full((sample_frames.size, indices.size, 2), np.nan, dtype=np.float32)
    valid = np.zeros((sample_frames.size, indices.size), dtype=bool)
    occupied = np.zeros((sample_frames.size, indices.size), dtype=bool)
    run_rows = np.full((sample_frames.size, indices.size), -1, dtype=np.int64)
    original_rows = np.full((sample_frames.size, indices.size), -1, dtype=np.int64)
    sample_acquisition = np.full(sample_frames.size, -1, dtype=np.int64)
    sample_timestamps = np.full(sample_frames.size, -1, dtype=np.int64)
    for run_row, (sample_row, acquisition_frame, source_index, timestamp, point) in enumerate(zip(
        sample_rows,
        acquisition_frames,
        source_indices,
        timestamps,
        chaser_xy,
        strict=True,
    )):
        sample_i = int(sample_row)
        acquisition_i = int(acquisition_frame)
        if acquisition_i < 0 or acquisition_i >= total_frames:
            _fail("Stimulus chaser frame lies outside the acquisition domain.")
        column = columns[int(source_index)]
        if occupied[sample_i, column]:
            _fail("Stimulus has duplicate stimulus-frame/chaser identity rows.")
        if sample_acquisition[sample_i] == -1:
            sample_acquisition[sample_i] = acquisition_i
            sample_timestamps[sample_i] = int(timestamp)
        elif (
            sample_acquisition[sample_i] != acquisition_i
            or sample_timestamps[sample_i] != int(timestamp)
        ):
            _fail("Chasers in one stimulus sample disagree on time lineage.")
        occupied[sample_i, column] = True
        run_rows[sample_i, column] = int(run_row)
        original_rows[sample_i, column] = int(source_rows[run_row])
        if np.isfinite(point).all():
            positions[sample_i, column] = point.astype(np.float32)
            valid[sample_i, column] = True
    if not np.all(occupied):
        _fail("Stimulus samples do not contain every declared chaser identity.")
    return {
        "stimulus_frame_num": sample_frames,
        "source_acquisition_frame_index": sample_acquisition,
        "timestamp_ns": sample_timestamps,
        "chaser_index": indices,
        "chaser_position_arena_xy": positions,
        "chaser_valid": valid,
        "source_stimulus_run_row_index": run_rows,
        "source_stimulus_source_row_index": original_rows,
    }


def _sample_epoch_summaries(
    distance_mm: np.ndarray,
    fish_valid: np.ndarray,
    chaser_valid: np.ndarray,
    epoch_ids: np.ndarray,
    *,
    windows: Sequence[ChaserDistanceWindow],
    threshold_mm: float,
) -> tuple[np.ndarray, ...]:
    n_windows = len(windows)
    n_chasers = distance_mm.shape[1]
    counts = np.zeros((n_windows, n_chasers), dtype=np.int64)
    metrics = [
        np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
        for _ in range(6)
    ]
    for window_index, window in enumerate(windows):
        for chaser_column in range(n_chasers):
            mask = (
                (epoch_ids == int(window.window_id))
                & fish_valid
                & chaser_valid[:, chaser_column]
                & np.isfinite(distance_mm[:, chaser_column])
            )
            values = distance_mm[:, chaser_column][mask]
            if values.size == 0:
                continue
            counts[window_index, chaser_column] = values.size
            metrics[0][window_index, chaser_column] = np.mean(values)
            metrics[1][window_index, chaser_column] = np.min(values)
            metrics[2][window_index, chaser_column] = np.percentile(values, 5)
            metrics[3][window_index, chaser_column] = np.percentile(values, 50)
            metrics[4][window_index, chaser_column] = np.percentile(values, 95)
            metrics[5][window_index, chaser_column] = np.mean(values <= threshold_mm)
    return (counts, *metrics)


def _sample_epoch_distributions(
    distance_mm: np.ndarray,
    fish_valid: np.ndarray,
    chaser_valid: np.ndarray,
    epoch_ids: np.ndarray,
    *,
    windows: Sequence[ChaserDistanceWindow],
    bin_width_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = distance_mm[np.isfinite(distance_mm)]
    max_distance = float(np.max(finite)) if finite.size else float(bin_width_mm)
    max_edge = max(
        float(bin_width_mm),
        float(np.ceil(max_distance / bin_width_mm) * bin_width_mm),
    )
    edges = np.arange(
        0.0,
        max_edge + bin_width_mm * 0.5,
        bin_width_mm,
        dtype=np.float32,
    )
    if edges.size < 2:
        edges = np.asarray([0.0, bin_width_mm], dtype=np.float32)
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    counts = np.zeros(
        (len(windows), distance_mm.shape[1], centers.size),
        dtype=np.uint32,
    )
    density = np.zeros_like(counts, dtype=np.float32)
    for window_index, window in enumerate(windows):
        for chaser_column in range(distance_mm.shape[1]):
            mask = (
                (epoch_ids == int(window.window_id))
                & fish_valid
                & chaser_valid[:, chaser_column]
                & np.isfinite(distance_mm[:, chaser_column])
            )
            values = distance_mm[:, chaser_column][mask]
            if values.size == 0:
                continue
            histogram, _ = np.histogram(values, bins=edges)
            counts[window_index, chaser_column] = histogram.astype(np.uint32)
            density[window_index, chaser_column] = (
                histogram.astype(np.float32) / (values.size * bin_width_mm)
            )
    return edges, centers, counts, density


@dataclass(frozen=True)
class ProviderChaserDistanceCandidate:
    source_zarr: Path
    run_name: str
    recording_id: str
    position_run_path: str
    position_manifest_sha256: str
    position_estimator_id: str
    stimulus_run_path: str
    stimulus_epoch_run_path: str
    total_frames: int
    fps: float
    pixels_per_mm_projector: float
    threshold_mm: float
    distribution_bin_width_mm: float
    windows: tuple[ChaserDistanceWindow, ...]
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    source_authority: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"


def build_provider_chaser_distance_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    position_run_path: str,
    expected_position_manifest_sha256: str,
    stimulus_run_path: str,
    stimulus_epoch_run_path: str,
    threshold_mm: float = 20.0,
    distribution_bin_width_mm: float = 2.0,
) -> ProviderChaserDistanceCandidate:
    """Build one exact in-memory candidate without mutating the archive."""

    source = Path(source_zarr).expanduser().resolve()
    name = _controlled_name(run_name, label="run_name")
    position_path = _controlled_run_path(
        position_run_path,
        parent="analysis/subject_position_runs/observation",
        label="position_run_path",
    )
    stimulus_path = _controlled_run_path(
        stimulus_run_path,
        parent="analysis/stimulus_runs",
        label="stimulus_run_path",
    )
    epoch_path = _controlled_run_path(
        stimulus_epoch_run_path,
        parent="analysis/stimulus_epoch_runs",
        label="stimulus_epoch_run_path",
    )
    if (
        not np.isfinite(float(threshold_mm))
        or float(threshold_mm) <= 0
        or not np.isfinite(float(distribution_bin_width_mm))
        or float(distribution_bin_width_mm) <= 0
    ):
        _fail("Distance threshold and histogram bin width must be positive.")

    position: SubjectPositionSourceHandle = load_subject_position_source_handle(
        source,
        position_path,
        expected_selector_eligible=False,
        use_consolidated=True,
        expected_manifest_sha256=expected_position_manifest_sha256,
    )
    root = open_zarr_root(source, mode="r", use_consolidated=True)
    try:
        stimulus_group = root[stimulus_path]
        chaser_group = stimulus_group["tracking_data/chaser_states"]
        stimulus = load_bound_stimulus_coordinate_evidence(
            stimulus_group,
            chaser_group,
            root_node=root,
        )
        epoch_group = root[epoch_path]
        _ = epoch_group["windows"]
        _, acquisition = load_persisted_acquisition_camera_authority(root)
    except Exception as exc:
        _fail(f"Provider chaser candidate source preflight failed: {exc}.")

    position_camera = position.source_record.get("source_camera_frame")
    stimulus_camera = _frame_pointer(stimulus.frame_transform.source_camera_frame)
    if position_camera != stimulus_camera:
        _fail("Subject-position and stimulus sources bind different camera frames.")
    acquisition_pointer = _frame_pointer(acquisition)
    if _frame_pointer(stimulus.source_temporal_authority.acquisition_frame) != acquisition_pointer:
        _fail("Stimulus and archive bind different acquisition-frame domains.")
    total_frames = int(acquisition.record.source_total_frames)
    if total_frames <= 0:
        _fail("Acquisition frame authority has no positive frame count.")

    frames = _read(
        position.source_acquisition_frame_index_node,
        label="subject-position source_acquisition_frame_index",
    )
    positions = _read(position.position_xy_node, label="subject-position position_xy")
    position_valid = _read(position.valid_node, label="subject-position valid")
    source_rows = _read(position.source_row_index_node, label="subject-position source_row_index")
    instance_keys = _read(position.instance_key_node, label="subject-position instance_key")
    failure_reasons = _read(
        position.failure_reason_codes_node,
        label="subject-position failure_reason_codes",
    )
    fish = _dense_fish_positions(
        total_frames=total_frames,
        frames=frames,
        positions=positions,
        valid=position_valid,
        source_rows=source_rows,
        instance_keys=instance_keys,
        failure_reasons=failure_reasons,
    )
    samples = _stimulus_sample_positions(
        stimulus,
        chaser_group,
        total_frames=total_frames,
    )
    sample_acquisition = samples["source_acquisition_frame_index"]
    sample_fish = {
        path: values[sample_acquisition]
        for path, values in fish.items()
    }
    fish_camera = sample_fish["fish_position_source_camera_xy"]
    fish_valid = sample_fish["fish_valid"]
    fish_arena = np.full_like(fish_camera, np.nan, dtype=np.float32)
    if np.any(fish_valid):
        fish_arena[fish_valid] = _source_camera_to_arena_xy(
            fish_camera[fish_valid],
            stimulus,
        ).astype(np.float32)
    chaser_indices = samples["chaser_index"]
    chaser_arena = samples["chaser_position_arena_xy"]
    chaser_valid = samples["chaser_valid"]
    ppm = stimulus.frame_transform.selected_calibration.pixels_per_mm_projector
    if (
        isinstance(ppm, bool)
        or not isinstance(ppm, (int, float))
        or not np.isfinite(float(ppm))
        or float(ppm) <= 0
    ):
        _fail("Selected stimulus calibration lacks positive projector pixels/mm.")
    fps, fps_authority = _positive_fps(root, acquisition)

    sample_count = sample_acquisition.size
    distance_px = np.full((sample_count, chaser_indices.size), np.nan, dtype=np.float32)
    for column in range(chaser_indices.size):
        valid = fish_valid & chaser_valid[:, column] & np.isfinite(fish_arena).all(axis=1)
        delta = chaser_arena[:, column, :] - fish_arena
        distance_px[valid, column] = np.linalg.norm(delta[valid], axis=1).astype(np.float32)
    distance_mm = (distance_px / np.float32(ppm)).astype(np.float32)
    nearest_index = np.full(sample_count, -1, dtype=np.int16)
    nearest_mm = np.full(sample_count, np.nan, dtype=np.float32)
    any_finite = np.isfinite(distance_mm).any(axis=1)
    if np.any(any_finite):
        filled = np.where(np.isfinite(distance_mm), distance_mm, np.inf)
        nearest_columns = np.argmin(filled[any_finite], axis=1)
        nearest_index[any_finite] = chaser_indices[nearest_columns]
        nearest_mm[any_finite] = filled[any_finite, nearest_columns].astype(np.float32)

    epoch_snapshot = read_stimulus_epoch_snapshot(
        source,
        run_name=epoch_path.rsplit("/", 1)[-1],
        compatibility_policy=StimulusEpochCompatibilityPolicy.EXACT_V2_ONLY,
    )
    if epoch_snapshot.run_path != epoch_path:
        _fail("Exact stimulus epoch run did not resolve to the requested path.")
    windows = tuple(
        ChaserDistanceWindow(
            window_id=segment.segment_id,
            label=segment.label,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            start_time_s=segment.start_time_s,
            end_time_s=segment.end_time_s,
            duration_s=segment.duration_s,
        )
        for segment in epoch_snapshot.segments
    )
    epoch_manifest = epoch_group.attrs.get(STIMULUS_EPOCH_RUN_MANIFEST_ATTR)
    if not isinstance(epoch_manifest, Mapping):
        _fail("Exact stimulus epoch run lacks its validated run manifest.")
    acquisition_epoch_assignment = np.full(total_frames, -1, dtype=np.int32)
    for window in windows:
        start = max(0, int(window.start_frame))
        end = min(total_frames - 1, int(window.end_frame))
        if end >= start:
            acquisition_epoch_assignment[start : end + 1] = int(window.window_id)
    epoch_assignment = acquisition_epoch_assignment[sample_acquisition]
    summaries = _sample_epoch_summaries(
        distance_mm,
        fish_valid,
        chaser_valid,
        epoch_assignment,
        windows=windows,
        threshold_mm=float(threshold_mm),
    )
    distributions = _sample_epoch_distributions(
        distance_mm,
        fish_valid,
        chaser_valid,
        epoch_assignment,
        windows=windows,
        bin_width_mm=float(distribution_bin_width_mm),
    )

    arrays: dict[str, np.ndarray] = {
        "samples/stimulus_frame_num": samples["stimulus_frame_num"],
        "samples/source_acquisition_frame_index": sample_acquisition,
        "samples/timestamp_ns": samples["timestamp_ns"],
        "samples/stimulus_epoch_window_id": epoch_assignment,
        "samples/source_stimulus_run_row_index": samples[
            "source_stimulus_run_row_index"
        ],
        "samples/source_stimulus_source_row_index": samples[
            "source_stimulus_source_row_index"
        ],
        **{f"positions/{path}": values for path, values in sample_fish.items()},
        "positions/fish_position_arena_xy": fish_arena,
        "positions/chaser_position_arena_xy": chaser_arena,
        "positions/chaser_valid": chaser_valid,
        "chasers/chaser_index": chaser_indices,
        "distances/distance_px": distance_px,
        "distances/distance_mm": distance_mm,
        "distances/nearest_chaser_index": nearest_index,
        "distances/nearest_distance_mm": nearest_mm,
        "epoch_summary/window_id": np.asarray([window.window_id for window in windows], dtype=np.int32),
        "epoch_summary/label_bytes": _text_rows([window.label for window in windows]),
        "epoch_summary/start_frame": np.asarray([window.start_frame for window in windows], dtype=np.int64),
        "epoch_summary/end_frame": np.asarray([window.end_frame for window in windows], dtype=np.int64),
        "epoch_summary/valid_frame_count": summaries[0],
        "epoch_summary/mean_distance_mm": summaries[1],
        "epoch_summary/min_distance_mm": summaries[2],
        "epoch_summary/p05_distance_mm": summaries[3],
        "epoch_summary/p50_distance_mm": summaries[4],
        "epoch_summary/p95_distance_mm": summaries[5],
        "epoch_summary/fraction_within_threshold": summaries[6],
        "epoch_distributions/window_id": np.asarray([window.window_id for window in windows], dtype=np.int32),
        "epoch_distributions/chaser_index": chaser_indices,
        "epoch_distributions/bin_edges_mm": distributions[0],
        "epoch_distributions/bin_centers_mm": distributions[1],
        "epoch_distributions/hist_counts": distributions[2],
        "epoch_distributions/hist_density": distributions[3],
        "epoch_distributions/valid_sample_count": summaries[0],
    }
    estimator_id = str(position.estimator_record.get("estimator_id") or "unknown")
    source_authority = {
        "schema_id": "palette.provider_chaser_distance_candidate_source_authority",
        "schema_version": 1,
        "recording_id": acquisition.record.recording_id,
        "position": {
            "run_path": position.run_path,
            "manifest_sha256": position.manifest_sha256,
            "decoded_content_sha256": position.decoded_content_sha256,
            "estimator_id": estimator_id,
            "estimator_sha256": position.estimator_sha256,
            "policy_sha256": position.policy_sha256,
            "source_sha256": position.source_sha256,
            "anatomy_sha256": position.anatomy_sha256,
            "coordinate_sha256": position.coordinate_sha256,
            "source_camera_frame": position_camera,
        },
        "stimulus": {
            "run_path": stimulus_path,
            "row_identity": {
                "record_ref": stimulus.row_identity.record_ref,
                "record_sha256": stimulus.row_identity.record_sha256,
            },
            "temporal_authority": {
                "record_ref": stimulus.source_temporal_authority.record_ref,
                "record_sha256": stimulus.source_temporal_authority.record_sha256,
            },
            "surface_manifest": {
                "record_ref": stimulus.surface_manifest.record_ref,
                "record_sha256": stimulus.surface_manifest.record_sha256,
            },
            "output_manifest": {
                "record_ref": stimulus.output_manifest.record_ref,
                "record_sha256": stimulus.output_manifest.record_sha256,
            },
            "transform_manifest": {
                "record_ref": stimulus.frame_transform.manifest.record_ref,
                "record_sha256": stimulus.frame_transform.manifest.record_sha256,
            },
            "source_camera_frame": stimulus_camera,
        },
        "stimulus_epoch": {
            "run_path": epoch_path,
            "schema_id": epoch_snapshot.schema_id,
            "schema_version": epoch_snapshot.schema_version,
            "manifest_sha256": epoch_manifest.get("payload_digest"),
            "metadata_equivalence": (
                epoch_snapshot.metadata_equivalence.to_json()
                if epoch_snapshot.metadata_equivalence is not None
                else None
            ),
        },
        "acquisition_frame_authority": acquisition_pointer,
        "total_frames": total_frames,
        "stimulus_sample_count": sample_count,
        "fps": float(fps),
        "fps_authority": fps_authority,
        "pixels_per_mm_projector": float(ppm),
        "temporal_join_policy": (
            "preserve_unique_stimulus_frame_num_then_join_exact_"
            "source_acquisition_frame_index_v1"
        ),
        "numeric_transform": (
            "source_camera_to_selected_canvas_then_inverse_arena_to_canvas_v1"
        ),
    }
    return ProviderChaserDistanceCandidate(
        source_zarr=source,
        run_name=name,
        recording_id=acquisition.record.recording_id,
        position_run_path=position.run_path,
        position_manifest_sha256=position.manifest_sha256,
        position_estimator_id=estimator_id,
        stimulus_run_path=stimulus_path,
        stimulus_epoch_run_path=epoch_path,
        total_frames=total_frames,
        fps=float(fps),
        pixels_per_mm_projector=float(ppm),
        threshold_mm=float(threshold_mm),
        distribution_bin_width_mm=float(distribution_bin_width_mm),
        windows=windows,
        arrays=arrays,
        source_authority=source_authority,
    )


def _text_rows(values: Sequence[str], *, width: int = 96) -> np.ndarray:
    rows = np.zeros((len(values), width), dtype=np.uint8)
    for index, value in enumerate(values):
        encoded = str(value).encode("utf-8")[: width - 1]
        if encoded:
            rows[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return rows


def _render_distance_histogram(candidate: ProviderChaserDistanceCandidate) -> bytes:
    edges = candidate.arrays["epoch_distributions/bin_edges_mm"]
    counts = candidate.arrays["epoch_distributions/hist_counts"]
    labels = [window.label.replace("_", " ") for window in candidate.windows]
    figure, axes = plt.subplots(
        1,
        max(1, len(labels)),
        figsize=(4.8 * max(1, len(labels)), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(candidate.distribution_bin_width_mm) * 0.9
    for window_index, (axis, label) in enumerate(zip(axes[0], labels, strict=True)):
        for chaser_column, chaser_index in enumerate(
            candidate.arrays["chasers/chaser_index"].tolist()
        ):
            values = counts[window_index, chaser_column].astype(np.float64)
            denominator = int(values.sum())
            fraction = values / denominator if denominator else values
            axis.bar(
                centers,
                fraction,
                width=width,
                alpha=0.55,
                label=f"chaser {int(chaser_index)} (n={denominator:,})",
            )
        axis.set_title(label)
        axis.set_xlabel("fish-to-chaser distance (mm)")
        axis.set_ylabel("fraction of valid stimulus samples per bin")
        axis.set_xlim(float(edges[0]), float(edges[-1]))
        axis.set_ylim(bottom=0)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Provider-position fish-to-chaser distance\n{candidate.recording_id}")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    return buffer.getvalue()


def _render_distance_trace(candidate: ProviderChaserDistanceCandidate) -> bytes:
    timestamp = candidate.arrays["samples/timestamp_ns"]
    distance = candidate.arrays["distances/distance_mm"]
    sample_count = timestamp.size
    step = max(1, int(np.ceil(sample_count / 5000)))
    selected = np.arange(0, sample_count, step, dtype=np.int64)
    time_minutes = (timestamp - timestamp[0]).astype(np.float64) / 60e9
    figure, axis = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    for column, chaser_index in enumerate(candidate.arrays["chasers/chaser_index"]):
        axis.plot(
            time_minutes[selected],
            distance[selected, column],
            linewidth=0.8,
            label=f"chaser {int(chaser_index)}",
        )
    axis.set_xlabel("time (min)")
    axis.set_ylabel("fish-to-chaser distance (mm)")
    axis.set_title(candidate.recording_id)
    axis.legend(frameon=False, fontsize=8)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    return buffer.getvalue()


def _manifest(
    candidate: ProviderChaserDistanceCandidate,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    payload = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_name": candidate.run_name,
        "run_path": candidate.run_path,
        "status": RUN_STATUS_COMPLETE,
        "stage_selector_eligible": False,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "source_authority": json_attr_safe(candidate.source_authority),
        "parameters": {
            "threshold_mm": candidate.threshold_mm,
            "distribution_bin_width_mm": candidate.distribution_bin_width_mm,
            "single_subject_frame_policy": "exactly_zero_or_one_position_row_per_frame_v1",
            "invalid_position_policy": "preserve_invalid_no_fallback_no_interpolation_v1",
            "sample_axis_policy": (
                "preserve_all_stimulus_frames_with_exact_many_to_one_"
                "acquisition_join_v1"
            ),
        },
        "arrays": [
            _array_record(values, path=path)
            for path, values in sorted(arrays.items())
        ],
        "publication": {
            "policy": PUBLICATION_POLICY,
            "selector_mutation": "forbidden",
            "retry_policy": "new_immutable_run_name_required",
        },
    }
    return {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _materialize_local(
    candidate: ProviderChaserDistanceCandidate,
    *,
    local_zarr: Path,
) -> tuple[Path, str]:
    if local_zarr.exists():
        raise FileExistsError(f"Local candidate Zarr already exists: {local_zarr}")
    root = open_zarr_root(local_zarr, mode="w", use_consolidated=False)
    run = root.require_group(candidate.run_path)
    mark_run_started(run, run_name=candidate.run_name, stage="provider_chaser_distance_candidate")
    run.attrs.update(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "run_name": candidate.run_name,
            "recording_id": candidate.recording_id,
            "stage_selector_eligible": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_position_run_path": candidate.position_run_path,
            "source_position_manifest_sha256": candidate.position_manifest_sha256,
            "source_position_estimator_id": candidate.position_estimator_id,
            "source_stimulus_run_path": candidate.stimulus_run_path,
            "source_stimulus_epoch_run_path": candidate.stimulus_epoch_run_path,
            "coordinate_publication_status": "candidate_unsealed_not_selector_eligible",
            "row_axis": "stimulus_samples",
        }
    )
    arrays = {path: np.asarray(values) for path, values in candidate.arrays.items()}
    arrays["visualizations/distance_histogram_png"] = np.frombuffer(
        _render_distance_histogram(candidate),
        dtype=np.uint8,
    ).copy()
    arrays["visualizations/distance_trace_png"] = np.frombuffer(
        _render_distance_trace(candidate),
        dtype=np.uint8,
    ).copy()
    for path, values in arrays.items():
        _write_array(run, path, values)
    run["epoch_distributions"].attrs.update(
        {
            "bin_width_mm": candidate.distribution_bin_width_mm,
            "normalization": "sum(hist_density*bin_width_mm)==1_for_nonempty",
            "shared_bin_edges_across_epochs_and_chasers": True,
        }
    )
    run["visualizations"].attrs.update(
        {
            "distance_histogram_png_media_type": "image/png",
            "distance_histogram_semantics": (
                "fraction_of_valid_stimulus_samples_per_shared_linear_distance_bin"
            ),
            "distance_trace_png_media_type": "image/png",
        }
    )
    manifest = _manifest(candidate, arrays)
    run.attrs[MANIFEST_ATTR] = json_attr_safe(manifest)
    run.attrs[MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]
    mark_run_complete(
        run,
        parent_group=None,
        run_name=candidate.run_name,
        run_provenance=build_writer_run_provenance(
            command="fisheye.analysis.provider_chaser_distance_candidates",
            params=manifest["payload"]["parameters"],
            input_run_ids={
                "position": candidate.position_run_path,
                "stimulus": candidate.stimulus_run_path,
                "stimulus_epoch": candidate.stimulus_epoch_run_path,
            },
        ),
    )
    consolidate_metadata_capture_expected_warnings(local_zarr)
    return local_zarr / candidate.run_path, manifest["payload_digest"]


def validate_provider_chaser_distance_candidate(
    run_path: str | Path,
    *,
    use_consolidated: bool = False,
    archive_path: str | Path | None = None,
    archive_run_path: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one candidate run root copied locally or into an archive."""

    path = Path(run_path)
    if use_consolidated:
        if archive_path is None or archive_run_path is None:
            raise ValueError(
                "Consolidated candidate validation requires the archive root "
                "and exact archive run path."
            )
        archive = Path(archive_path)
        canonical_run_path = _controlled_run_path(
            archive_run_path,
            parent=PARENT_PATH,
            label="archive_run_path",
        )
        root = open_zarr_root(archive, mode="r", use_consolidated=True)
        run = root[canonical_run_path]
    else:
        run = open_zarr_root(path, mode="r", use_consolidated=False)
    attrs = run.attrs
    errors: list[str] = []
    if attrs.get("schema_id") != SCHEMA_ID or attrs.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema identity")
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("completion status")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append("selector eligibility")
    manifest = attrs.get(MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        errors.append("manifest missing")
        manifest = {}
    payload = manifest.get("payload") if isinstance(manifest, Mapping) else None
    digest = manifest.get("payload_digest") if isinstance(manifest, Mapping) else None
    if not isinstance(payload, Mapping) or canonical_json_sha256(payload) != digest:
        errors.append("manifest digest")
        payload = {}
    if attrs.get(MANIFEST_DIGEST_ATTR) != digest:
        errors.append("manifest digest attr")
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        errors.append("expected manifest digest")
    declared = payload.get("arrays", []) if isinstance(payload, Mapping) else []
    if not isinstance(declared, list):
        errors.append("manifest arrays")
        declared = []
    for record in declared:
        try:
            node = run[str(record["path"])]
            values = np.asarray(node[:])
            if (
                values.dtype.str != record["dtype"]
                or list(values.shape) != record["shape"]
                or sha256_array(values) != record["sha256"]
            ):
                errors.append(f"array mismatch:{record['path']}")
        except Exception:
            errors.append(f"array missing:{record.get('path')}")
    return {
        "valid": not errors,
        "errors": errors,
        "run_path": str(path),
        "manifest_sha256": digest,
        "array_count": len(declared),
    }


def publish_provider_chaser_distance_candidate(
    candidate: ProviderChaserDistanceCandidate,
    *,
    scratch_root: str | Path,
    copy_backend: str = "python",
    keep_scratch: bool = True,
) -> dict[str, Any]:
    """Materialize locally and atomically publish without selector mutation."""

    scratch = Path(scratch_root).expanduser().resolve()
    local_zarr = scratch / f"{candidate.run_name}.{uuid.uuid4().hex}.zarr"
    local_run_path, manifest_sha = _materialize_local(candidate, local_zarr=local_zarr)
    target_run_path = candidate.source_zarr / candidate.run_path
    parent_attrs_before: dict[str, Any] = {}

    def validate(path: Path) -> Mapping[str, Any]:
        return validate_provider_chaser_distance_candidate(
            path,
            use_consolidated=False,
            expected_manifest_sha256=manifest_sha,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        analysis = root.require_group("analysis")
        parent = require_runs_parent(
            analysis,
            "provider_chaser_distance_candidate_runs",
        )
        occupied = _SELECTOR_ATTRS.intersection(parent.attrs)
        if occupied:
            raise RuntimeError(
                "Provider candidate parent contains forbidden selector attrs: "
                f"{sorted(occupied)!r}."
            )
        if not parent_attrs_before:
            parent_attrs_before.update(dict(parent.attrs))
        return (parent,)

    def complete(_root: zarr.Group, _parent: zarr.Group, run: zarr.Group) -> None:
        if (
            run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError("Published provider candidate lost complete ineligible state.")

    def verify(root: zarr.Group) -> None:
        parent = root[PARENT_PATH]
        if dict(parent.attrs) != parent_attrs_before:
            raise RuntimeError("Provider candidate publication changed parent attributes.")
        if _SELECTOR_ATTRS.intersection(parent.attrs):
            raise RuntimeError("Provider candidate publication introduced a selector.")
        validation = validate_provider_chaser_distance_candidate(
            candidate.source_zarr / candidate.run_path,
            use_consolidated=False,
            expected_manifest_sha256=manifest_sha,
        )
        if not validation["valid"]:
            raise RuntimeError(f"Published provider candidate is invalid: {validation}.")

    acceptance: dict[str, Any] = {}

    def finalize(_root: zarr.Group, _parent: zarr.Group, _run: zarr.Group) -> None:
        consolidate_metadata_capture_expected_warnings(candidate.source_zarr)
        metadata = validate_direct_consolidated_subtree(
            candidate.source_zarr,
            subtree_path=candidate.run_path,
        )
        consolidated = validate_provider_chaser_distance_candidate(
            candidate.source_zarr / candidate.run_path,
            use_consolidated=True,
            archive_path=candidate.source_zarr,
            archive_run_path=candidate.run_path,
            expected_manifest_sha256=manifest_sha,
        )
        if not consolidated["valid"]:
            raise RuntimeError(f"Consolidated provider candidate is invalid: {consolidated}.")
        acceptance.update(
            direct_consolidated=metadata.to_json(),
            consolidated_validation=consolidated,
        )

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=candidate.source_zarr,
            local_run_path=local_run_path,
            target_run_path=target_run_path,
            run_name=candidate.run_name,
            lock_suffix="provider-chaser-distance-candidate",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy=PUBLICATION_POLICY,
            rollback_policy="retain_failed_selector_ineligible_tombstone_no_selector_mutation",
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=finalize,
        repair_failed_publication_visibility=(
            lambda _path: consolidate_metadata_capture_expected_warnings(candidate.source_zarr)
        ),
        accept_persisted_activation_on_callback_error=False,
        payload_metadata={
            "manifest_sha256": manifest_sha,
            "selector_ineligible": True,
            "source_position_run_path": candidate.position_run_path,
            "source_position_manifest_sha256": candidate.position_manifest_sha256,
        },
    )
    result = {
        "schema_id": PUBLISH_SCHEMA_ID,
        "candidate_run_path": candidate.run_path,
        "manifest_sha256": manifest_sha,
        "publication": publication,
        "acceptance": acceptance,
    }
    if not keep_scratch and local_zarr.exists():
        shutil.rmtree(local_zarr)
    return result


def _summary(candidate: ProviderChaserDistanceCandidate) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "recording_id": candidate.recording_id,
        "run_name": candidate.run_name,
        "run_path": candidate.run_path,
        "source_position_run_path": candidate.position_run_path,
        "source_position_manifest_sha256": candidate.position_manifest_sha256,
        "source_position_estimator_id": candidate.position_estimator_id,
        "source_stimulus_run_path": candidate.stimulus_run_path,
        "source_stimulus_epoch_run_path": candidate.stimulus_epoch_run_path,
        "total_frames": candidate.total_frames,
        "stimulus_sample_count": int(
            candidate.arrays["samples/stimulus_frame_num"].size
        ),
        "fish_valid_sample_count": int(
            candidate.arrays["positions/fish_valid"].sum()
        ),
        "distance_valid_sample_count": np.sum(
            np.isfinite(candidate.arrays["distances/distance_mm"]),
            axis=0,
        ).astype(int).tolist(),
        "chaser_indices": candidate.arrays["chasers/chaser_index"].astype(int).tolist(),
        "epoch_labels": [window.label for window in candidate.windows],
        "selector_eligible": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--position-run-path", required=True)
    parser.add_argument("--expected-position-manifest-sha256", required=True)
    parser.add_argument("--stimulus-run-path", required=True)
    parser.add_argument("--stimulus-epoch-run-path", required=True)
    parser.add_argument("--threshold-mm", type=float, default=20.0)
    parser.add_argument("--distribution-bin-width-mm", type=float, default=2.0)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--discard-scratch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    candidate = build_provider_chaser_distance_candidate(
        args.analysis_zarr,
        run_name=args.run_name,
        position_run_path=args.position_run_path,
        expected_position_manifest_sha256=args.expected_position_manifest_sha256,
        stimulus_run_path=args.stimulus_run_path,
        stimulus_epoch_run_path=args.stimulus_epoch_run_path,
        threshold_mm=args.threshold_mm,
        distribution_bin_width_mm=args.distribution_bin_width_mm,
    )
    payload = _summary(candidate)
    if args.apply:
        if args.scratch_root is None:
            raise SystemExit("--apply requires --scratch-root")
        payload["publication"] = publish_provider_chaser_distance_candidate(
            candidate,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
            keep_scratch=not args.discard_scratch,
        )
    if args.json:
        print(json.dumps(json_attr_safe(payload), sort_keys=True))
    else:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
