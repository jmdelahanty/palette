"""Publish and read a selector-ineligible keypoint-quality integration fixture.

The deterministic source exercises empty frames, multi-observation frames,
invalid landmarks, failed poses, and row-specific crop geometry.  It is an
integration source, not representative-data or profile-promotion evidence.
The measurement path accepts any already prepared quality snapshot so the same
workload can later run against a real raw-keypoint-v2 canary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_filesystem import describe_filesystem
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import (
    local_environment_manifest,
    peak_rss_bytes,
    sha256_array,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.keypoint_quality_producer import (
    ObservationLocalKeypointQualityPolicy,
    PreparedKeypointQualitySnapshot,
    prepare_observation_local_keypoint_quality,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    DEFAULT_KEYPOINT_QUALITY_SHADOW_ROOT,
    publish_selector_ineligible_keypoint_quality_snapshot,
)
from fisheye.shared.zarr.keypoint_quality_schema import (
    KEYPOINT_QUALITY_SCHEMA_V1,
    KeypointQualitySourceReference,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_frame_row_offsets,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


KEYPOINT_QUALITY_BENCHMARK_SCHEMA_ID = (
    "palette.keypoint_quality.publication_read_benchmark"
)
KEYPOINT_QUALITY_BENCHMARK_SCHEMA_VERSION = 1
DETERMINISTIC_SOURCE_SCHEMA_ID = "palette.keypoint.benchmark_source"
DETERMINISTIC_SOURCE_SCHEMA_VERSION = 1

_HOT_FRAME_ARRAYS = (
    "keypoint_metric_values",
    "keypoint_metric_valid",
    "pose_metric_values",
    "pose_metric_valid",
    "keypoint_quality_flags",
    "pose_quality_flags",
    "proposed_keypoint_valid",
    "proposed_pose_usable",
)


def _require_safe_report_path(
    output_json: Path,
    *,
    shadow_root: Path,
    destination: Path,
) -> Path:
    report = output_json.expanduser().resolve()
    root = shadow_root.expanduser().resolve()
    archive = destination.expanduser().resolve()
    try:
        report.relative_to(root)
    except ValueError as exc:
        raise ValueError("Benchmark evidence must be below shadow_root.") from exc
    try:
        report.relative_to(archive)
    except ValueError:
        pass
    else:
        raise ValueError("Benchmark evidence must remain outside the Zarr archive.")
    if report.exists():
        raise FileExistsError(f"Benchmark evidence already exists: {report}")
    return report


def _git_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]

    def command(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "repository": str(repository),
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "worktree_clean": command("status", "--short") == "",
        "driver": str(Path(__file__).resolve().relative_to(repository)),
    }


def _array_declarations(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        path: {
            "shape": list(np.asarray(values).shape),
            "dtype": str(np.asarray(values).dtype),
            "sha256": sha256_array(np.asarray(values)),
        }
        for path, values in sorted(arrays.items())
    }


def _frame_indices(
    *,
    n_frames: int,
    n_instances: int,
    empty_frames: int,
    seed: int,
) -> np.ndarray:
    if not (0 <= empty_frames < n_frames):
        raise ValueError("empty_frames must be in [0, n_frames).")
    populated_count = n_frames - empty_frames
    if n_instances < populated_count:
        raise ValueError(
            "n_instances must cover every nonempty frame; increase empty_frames."
        )
    rng = np.random.default_rng(seed)
    empty = np.sort(
        rng.choice(n_frames, size=empty_frames, replace=False).astype(np.int64)
    )
    populated = np.setdiff1d(
        np.arange(n_frames, dtype=np.int64), empty, assume_unique=True
    )
    extra_count = n_instances - populated_count
    extras = (
        rng.choice(populated, size=extra_count, replace=True).astype(np.int64)
        if extra_count
        else np.empty(0, dtype=np.int64)
    )
    return np.sort(np.concatenate((populated, extras)), kind="stable")


def build_deterministic_keypoint_v2_source(
    *,
    n_frames: int = 23_287,
    n_instances: int = 22_926,
    n_keypoints: int = 5,
    source_width: int = 4_512,
    source_height: int = 4_512,
    empty_frames: int = 365,
    seed: int = 20_260_729,
) -> tuple[
    KeypointDimensions,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
    KeypointQualitySourceReference,
]:
    """Build a deterministic, contract-valid raw-keypoint-v2 source."""

    dimensions = KeypointDimensions(
        n_frames=int(n_frames),
        n_instances=int(n_instances),
        n_keypoints=int(n_keypoints),
        source_width=int(source_width),
        source_height=int(source_height),
    )
    frames = _frame_indices(
        n_frames=dimensions.n_frames,
        n_instances=dimensions.n_instances,
        empty_frames=int(empty_frames),
        seed=int(seed),
    )
    rng = np.random.default_rng(int(seed))
    rows = np.arange(dimensions.n_instances, dtype=np.int64)
    keys = (rows.astype(np.uint64) + np.uint64(1_000_000_000)).astype(
        np.uint64, copy=False
    )

    size_choices = np.asarray((256, 384, 512), dtype=np.int32)
    widths = rng.choice(size_choices, size=dimensions.n_instances)
    heights = rng.choice(size_choices, size=dimensions.n_instances)
    max_x = np.maximum(1, dimensions.source_width - widths + 1)
    max_y = np.maximum(1, dimensions.source_height - heights + 1)
    origins = np.column_stack(
        (
            (rng.random(dimensions.n_instances) * max_x).astype(np.int32),
            (rng.random(dimensions.n_instances) * max_y).astype(np.int32),
        )
    )
    sizes = np.column_stack((widths, heights)).astype(np.int32, copy=False)
    signature_columns = np.arange(32, dtype=np.uint64)[None, :]
    crop_signatures = (
        (rows.astype(np.uint64)[:, None] * np.uint64(17))
        + (signature_columns * np.uint64(29))
        + np.uint64(int(seed) & 0xFF)
    ).astype(np.uint8)
    crop_arrays = {
        "instance_key": keys.copy(),
        "frame_indices": frames.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "source_row_signature": crop_signatures,
        "roi_coordinates_full": origins,
        "roi_sizes_full": sizes,
    }

    point_scale = sizes.astype(np.float32)[:, None, :]
    keypoints_roi = (
        rng.random(
            (dimensions.n_instances, dimensions.n_keypoints, 2),
            dtype=np.float32,
        )
        * (point_scale - np.float32(1.0))
    ).astype(np.float32, copy=False)
    pose_success = rng.random(dimensions.n_instances) >= 0.015
    keypoint_valid = (
        rng.random((dimensions.n_instances, dimensions.n_keypoints)) >= 0.04
    ) & pose_success[:, None]
    keypoints_roi[~keypoint_valid] = np.float32(np.nan)
    keypoints_img = keypoints_roi + origins.astype(np.float32)[:, None, :]
    confidences = rng.random(
        (dimensions.n_instances, dimensions.n_keypoints), dtype=np.float32
    )
    confidences[~keypoint_valid] = np.float32(np.nan)
    pose_confidence = rng.random(dimensions.n_instances, dtype=np.float32)
    pose_confidence[~pose_success] = np.float32(np.nan)
    bbox_roi = np.column_stack(
        (
            np.full(dimensions.n_instances, 1.0, dtype=np.float32),
            np.full(dimensions.n_instances, 1.0, dtype=np.float32),
            widths.astype(np.float32) - np.float32(1.0),
            heights.astype(np.float32) - np.float32(1.0),
        )
    )
    bbox_roi[~pose_success] = np.float32(np.nan)
    bbox_img = bbox_roi + np.column_stack((origins, origins)).astype(np.float32)

    skeleton_document = {
        "schema_id": "palette.keypoint.benchmark_skeleton",
        "schema_version": 1,
        "skeleton_id": "deterministic_keypoint_quality_fixture",
        "landmark_labels": [
            f"landmark_{index:02d}" for index in range(dimensions.n_keypoints)
        ],
        "heading_computation": "excluded_from_keypoint_quality_v1",
    }
    skeleton_digest = canonical_json_sha256(skeleton_document)
    row_signatures = derive_keypoint_row_signatures(
        instance_key=keys,
        source_crop_row_signature=crop_signatures,
        keypoints_roi=keypoints_roi,
        keypoint_valid=keypoint_valid,
        skeleton_digest=skeleton_digest,
    )
    arrays = {
        "instance_key": keys,
        "source_crop_row_ids": rows.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(
            frames, n_frames=dimensions.n_frames
        ),
        "source_crop_row_signature": crop_signatures.copy(),
        "keypoint_row_signature": row_signatures,
        "keypoints_roi": keypoints_roi,
        "keypoints_img": keypoints_img,
        "keypoint_confidences": confidences,
        "keypoint_valid": keypoint_valid,
        "pose_confidence": pose_confidence,
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": pose_success,
    }
    KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop_arrays,
        skeleton_digest=skeleton_digest,
    )
    source_manifest: dict[str, object] = {
        "schema_id": DETERMINISTIC_SOURCE_SCHEMA_ID,
        "schema_version": DETERMINISTIC_SOURCE_SCHEMA_VERSION,
        "artifact_class": "deterministic_integration_source_not_promotion_evidence",
        "selector_eligible": False,
        "seed": int(seed),
        "empty_frame_count": int(empty_frames),
        "multi_observation_frame_count": int(
            np.count_nonzero(np.diff(arrays["frame_row_offsets"]) > 1)
        ),
        "logical_schema": KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions),
        "skeleton": skeleton_document,
        "skeleton_digest": skeleton_digest,
        "arrays": _array_declarations(arrays),
        "crop_evidence_arrays": _array_declarations(crop_arrays),
    }
    source = KeypointQualitySourceReference(
        run_name="deterministic_raw_keypoints_v2",
        manifest_digest=canonical_json_sha256(source_manifest),
        skeleton_id=str(skeleton_document["skeleton_id"]),
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=sha256_array(row_signatures),
    )
    return dimensions, arrays, crop_arrays, source_manifest, source


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty latency sample.")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "minimum_seconds": min(samples),
        "median_seconds": float(statistics.median(samples)),
        "p95_seconds": _percentile(samples, 95.0),
        "maximum_seconds": max(samples),
    }


def _update_digest(digest: Any, path: str, values: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(values)
    digest.update(path.encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.view(np.uint8))
    return int(contiguous.nbytes)


def _frame_workload_digest(
    arrays: Mapping[str, Any],
    *,
    offsets: np.ndarray,
    frames: Sequence[int],
) -> tuple[str, int, list[float]]:
    digest = hashlib.sha256()
    logical_bytes = 0
    latencies: list[float] = []
    for frame in frames:
        started = time.perf_counter()
        start = int(offsets[int(frame)])
        stop = int(offsets[int(frame) + 1])
        digest.update(np.asarray((int(frame), start, stop), dtype="<i8").tobytes())
        for path in _HOT_FRAME_ARRAYS:
            logical_bytes += _update_digest(
                digest, path, np.asarray(arrays[path][start:stop])
            )
        latencies.append(time.perf_counter() - started)
    return digest.hexdigest(), logical_bytes, latencies


def _window_workload_digest(
    arrays: Mapping[str, Any],
    *,
    offsets: np.ndarray,
    starts: Sequence[int],
    window_frames: int,
) -> tuple[str, int, list[float]]:
    digest = hashlib.sha256()
    logical_bytes = 0
    latencies: list[float] = []
    for frame_start in starts:
        started = time.perf_counter()
        frame_stop = min(len(offsets) - 1, int(frame_start) + window_frames)
        row_start = int(offsets[int(frame_start)])
        row_stop = int(offsets[frame_stop])
        digest.update(
            np.asarray(
                (int(frame_start), frame_stop, row_start, row_stop), dtype="<i8"
            ).tobytes()
        )
        for path in _HOT_FRAME_ARRAYS:
            logical_bytes += _update_digest(
                digest, path, np.asarray(arrays[path][row_start:row_stop])
            )
        latencies.append(time.perf_counter() - started)
    return digest.hexdigest(), logical_bytes, latencies


def _full_scan(
    arrays: Mapping[str, Any],
    *,
    expected_declarations: Mapping[str, Mapping[str, Any]],
    batch_rows: int,
) -> dict[str, object]:
    started = time.perf_counter()
    logical_bytes = 0
    observed: dict[str, str] = {}
    for path in KEYPOINT_QUALITY_SCHEMA_V1.binding_paths:
        array = arrays[path]
        digest = hashlib.sha256()
        for start in range(0, int(array.shape[0]), int(batch_rows)):
            stop = min(int(array.shape[0]), start + int(batch_rows))
            values = np.ascontiguousarray(array[start:stop])
            logical_bytes += int(values.nbytes)
            digest.update(values.view(np.uint8))
        observed[path] = digest.hexdigest()
        if observed[path] != expected_declarations[path]["sha256"]:
            raise RuntimeError(f"Full-scan digest mismatch for {path!r}.")
    seconds = time.perf_counter() - started
    return {
        "batch_rows": int(batch_rows),
        "logical_bytes": int(logical_bytes),
        "seconds": float(seconds),
        "mib_per_second": (
            float(logical_bytes / 1024**2 / seconds) if seconds > 0 else None
        ),
        "array_sha256": observed,
        "all_digests_match": True,
    }


def benchmark_prepared_keypoint_quality_publication(
    prepared: PreparedKeypointQualitySnapshot,
    *,
    source_manifest: Mapping[str, Any],
    destination: Path,
    run_id: str,
    shadow_root: Path,
    source_evidence_class: str,
    preparation_seconds: Mapping[str, float] | None = None,
    random_frame_count: int = 128,
    window_count: int = 32,
    window_frames: int = 70,
    full_scan_batch_rows: int = 131_072,
    seed: int = 20_260_729,
    output_json: Path | None = None,
) -> dict[str, object]:
    """Run the fixed publication/read workload for one prepared snapshot."""

    report_path = (
        _require_safe_report_path(
            Path(output_json),
            shadow_root=Path(shadow_root),
            destination=Path(destination),
        )
        if output_json is not None
        else None
    )
    for name, value in (
        ("random_frame_count", random_frame_count),
        ("window_count", window_count),
        ("window_frames", window_frames),
        ("full_scan_batch_rows", full_scan_batch_rows),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    rss_before = peak_rss_bytes()
    publication = publish_selector_ineligible_keypoint_quality_snapshot(
        prepared,
        source_manifest=source_manifest,
        destination=Path(destination),
        run_id=str(run_id),
        shadow_root=Path(shadow_root),
        created_by="benchmark_keypoint_quality_publication",
    )
    direct_started = time.perf_counter()
    direct_root = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=False
    )
    direct_run = direct_root["keypoint_quality_runs"][str(run_id)]
    direct_open_seconds = time.perf_counter() - direct_started
    consolidated_started = time.perf_counter()
    consolidated_root = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=True
    )
    run = consolidated_root["keypoint_quality_runs"][str(run_id)]
    consolidated_open_seconds = time.perf_counter() - consolidated_started
    if dict(direct_run.attrs["run_manifest"]) != dict(run.attrs["run_manifest"]):
        raise RuntimeError("Direct and consolidated run manifests differ.")

    offset_started = time.perf_counter()
    offsets = np.asarray(run["frame_row_offsets"][:])
    offset_seconds = time.perf_counter() - offset_started
    expected_offsets = np.asarray(prepared.arrays["frame_row_offsets"])
    if not np.array_equal(offsets, expected_offsets):
        raise RuntimeError("Retained frame_row_offsets differs from prepared values.")

    rng = np.random.default_rng(int(seed))
    n_frames = prepared.dimensions.n_frames
    random_frames = rng.choice(
        n_frames,
        size=min(int(random_frame_count), n_frames),
        replace=False,
    ).astype(np.int64)
    maximum_start = max(0, n_frames - int(window_frames))
    window_starts = rng.choice(
        maximum_start + 1,
        size=min(int(window_count), maximum_start + 1),
        replace=False,
    ).astype(np.int64)
    stored_arrays = {path: run[path] for path in KEYPOINT_QUALITY_SCHEMA_V1.binding_paths}

    random_digest, random_bytes, random_latencies = _frame_workload_digest(
        stored_arrays, offsets=offsets, frames=random_frames
    )
    expected_random_digest, _, _ = _frame_workload_digest(
        prepared.arrays, offsets=expected_offsets, frames=random_frames
    )
    window_digest, window_bytes, window_latencies = _window_workload_digest(
        stored_arrays,
        offsets=offsets,
        starts=window_starts,
        window_frames=int(window_frames),
    )
    expected_window_digest, _, _ = _window_workload_digest(
        prepared.arrays,
        offsets=expected_offsets,
        starts=window_starts,
        window_frames=int(window_frames),
    )
    if random_digest != expected_random_digest:
        raise RuntimeError("Random-frame workload digest mismatch.")
    if window_digest != expected_window_digest:
        raise RuntimeError("Window workload digest mismatch.")

    declarations = publication.manifest["payload"]["logical_content"]["document"][
        "arrays"
    ]
    full_scan = _full_scan(
        stored_arrays,
        expected_declarations=declarations,
        batch_rows=int(full_scan_batch_rows),
    )
    plans = publication.plans.as_manifest()
    environment = local_environment_manifest()
    destination_filesystem = describe_filesystem(publication.output_path)
    environment["storage_tier"] = destination_filesystem["storage_tier"]
    environment["destination_filesystem"] = destination_filesystem
    evidence: dict[str, object] = {
        "schema_id": KEYPOINT_QUALITY_BENCHMARK_SCHEMA_ID,
        "schema_version": KEYPOINT_QUALITY_BENCHMARK_SCHEMA_VERSION,
        "status": "passed",
        "created_at_utc": utc_now(),
        "evidence_class": str(source_evidence_class),
        "promotion_eligible": False,
        "profile_promoted": False,
        "archive": str(publication.output_path),
        "run_id": str(run_id),
        "source_manifest_digest": prepared.source.manifest_digest,
        "quality_manifest_digest": publication.manifest["payload_digest"],
        "quality_profile_digest": prepared.profile.profile_digest,
        "policy_digest": prepared.policy.policy_digest,
        "dimensions": prepared.dimensions.as_manifest(),
        "source_characteristics": {
            "empty_frame_count": int(np.count_nonzero(np.diff(offsets) == 0)),
            "multi_observation_frame_count": int(
                np.count_nonzero(np.diff(offsets) > 1)
            ),
            "maximum_observations_per_frame": int(np.max(np.diff(offsets))),
        },
        "publication": {
            "phase_seconds": dict(publication.phase_seconds),
            "elapsed_seconds": float(publication.elapsed_seconds),
            "storage_plan": plans,
            "planned_payload_objects": int(
                plans["object_estimate"]["payload_objects"]
            ),
            "planned_metadata_objects": int(
                plans["object_estimate"]["array_metadata_objects"]
                + plans["object_estimate"]["group_metadata_objects"]
            ),
            "observed_storage": storage_stats(publication.output_path),
            "selector_eligible": False,
            "registry_registered": False,
        },
        "reads": {
            "direct_open_seconds": float(direct_open_seconds),
            "consolidated_open_seconds": float(consolidated_open_seconds),
            "offset_index": {
                "reads": 1,
                "retained": True,
                "logical_bytes": int(offsets.nbytes),
                "seconds": float(offset_seconds),
                "later_workload_offset_reads": 0,
            },
            "random_frames": {
                "frame_count": int(random_frames.size),
                "logical_bytes": int(random_bytes),
                "digest": random_digest,
                "expected_digest": expected_random_digest,
                "latency": _latency_summary(random_latencies),
            },
            "windows": {
                "window_count": int(window_starts.size),
                "window_frames": int(window_frames),
                "logical_bytes": int(window_bytes),
                "digest": window_digest,
                "expected_digest": expected_window_digest,
                "latency": _latency_summary(window_latencies),
            },
            "full_scan": full_scan,
        },
        "correctness": {
            "publication_gate": "passed",
            "direct_consolidated_manifest_equal": True,
            "offset_index_equal": True,
            "random_frame_digest_equal": True,
            "window_digest_equal": True,
            "full_scan_digests_equal": True,
        },
        "preparation_seconds": dict(preparation_seconds or {}),
        "peak_rss_bytes": int(peak_rss_bytes()),
        "peak_rss_delta_from_publication_entry_bytes": int(
            peak_rss_bytes() - rss_before
        ),
        "environment": environment,
        "palette": _git_provenance(),
        "workload": {
            "hot_frame_arrays": list(_HOT_FRAME_ARRAYS),
            "random_frame_count": int(random_frames.size),
            "window_count": int(window_starts.size),
            "window_frames": int(window_frames),
            "full_scan_batch_rows": int(full_scan_batch_rows),
            "seed": int(seed),
        },
    }
    if report_path is not None:
        write_json_atomic(report_path, evidence, overwrite=False)
    return evidence


def run_deterministic_keypoint_quality_benchmark(
    *,
    destination: Path,
    run_id: str,
    shadow_root: Path,
    n_frames: int = 23_287,
    n_instances: int = 22_926,
    n_keypoints: int = 5,
    source_width: int = 4_512,
    source_height: int = 4_512,
    empty_frames: int = 365,
    seed: int = 20_260_729,
    confidence_threshold: float = 0.5,
    minimum_valid_keypoints: int = 1,
    random_frame_count: int = 128,
    window_count: int = 32,
    window_frames: int = 70,
    full_scan_batch_rows: int = 131_072,
    output_json: Path | None = None,
) -> dict[str, object]:
    """Build the deterministic source, compute quality, publish, and read it."""

    report_path = (
        _require_safe_report_path(
            Path(output_json),
            shadow_root=Path(shadow_root),
            destination=Path(destination),
        )
        if output_json is not None
        else None
    )
    driver_rss_entry = peak_rss_bytes()
    fixture_started = time.perf_counter()
    dimensions, arrays, crop, source_manifest, source = (
        build_deterministic_keypoint_v2_source(
            n_frames=int(n_frames),
            n_instances=int(n_instances),
            n_keypoints=int(n_keypoints),
            source_width=int(source_width),
            source_height=int(source_height),
            empty_frames=int(empty_frames),
            seed=int(seed),
        )
    )
    fixture_seconds = time.perf_counter() - fixture_started
    compute_started = time.perf_counter()
    prepared = prepare_observation_local_keypoint_quality(
        arrays,
        source_dimensions=dimensions,
        source_crop_arrays=crop,
        source=source,
        skeleton_digest=source.skeleton_digest,
        policy=ObservationLocalKeypointQualityPolicy(
            confidence_threshold=float(confidence_threshold),
            minimum_valid_keypoints=int(minimum_valid_keypoints),
        ),
    )
    compute_seconds = time.perf_counter() - compute_started
    evidence = benchmark_prepared_keypoint_quality_publication(
        prepared,
        source_manifest=source_manifest,
        destination=Path(destination),
        run_id=str(run_id),
        shadow_root=Path(shadow_root),
        source_evidence_class=(
            "deterministic_integration_fixture_not_representative_or_promotion_evidence"
        ),
        preparation_seconds={
            "source_fixture_build": float(fixture_seconds),
            "source_validation_and_quality_compute": float(compute_seconds),
        },
        random_frame_count=int(random_frame_count),
        window_count=int(window_count),
        window_frames=int(window_frames),
        full_scan_batch_rows=int(full_scan_batch_rows),
        seed=int(seed),
        output_json=None,
    )
    evidence["source_manifest"] = source_manifest
    evidence["peak_rss_at_driver_entry_bytes"] = int(driver_rss_entry)
    evidence["peak_rss_delta_from_driver_entry_bytes"] = int(
        peak_rss_bytes() - driver_rss_entry
    )
    if report_path is not None:
        write_json_atomic(report_path, evidence, overwrite=False)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--shadow-root", type=Path, default=DEFAULT_KEYPOINT_QUALITY_SHADOW_ROOT
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=23_287)
    parser.add_argument("--n-instances", type=int, default=22_926)
    parser.add_argument("--n-keypoints", type=int, default=5)
    parser.add_argument("--source-width", type=int, default=4_512)
    parser.add_argument("--source-height", type=int, default=4_512)
    parser.add_argument("--empty-frames", type=int, default=365)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-valid-keypoints", type=int, default=1)
    parser.add_argument("--random-frame-count", type=int, default=128)
    parser.add_argument("--window-count", type=int, default=32)
    parser.add_argument("--window-frames", type=int, default=70)
    parser.add_argument("--full-scan-batch-rows", type=int, default=131_072)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Required to create the selector-ineligible fixture and evidence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "writes_performed": False,
                    "destination": str(args.destination.expanduser().resolve()),
                    "shadow_root": str(args.shadow_root.expanduser().resolve()),
                    "run_id": args.run_id,
                    "dimensions": {
                        "n_frames": args.n_frames,
                        "n_instances": args.n_instances,
                        "n_keypoints": args.n_keypoints,
                        "source_width": args.source_width,
                        "source_height": args.source_height,
                    },
                    "evidence_class": (
                        "deterministic_integration_fixture_not_representative_or_"
                        "promotion_evidence"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_deterministic_keypoint_quality_benchmark(
        destination=args.destination,
        run_id=args.run_id,
        shadow_root=args.shadow_root,
        n_frames=args.n_frames,
        n_instances=args.n_instances,
        n_keypoints=args.n_keypoints,
        source_width=args.source_width,
        source_height=args.source_height,
        empty_frames=args.empty_frames,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
        minimum_valid_keypoints=args.minimum_valid_keypoints,
        random_frame_count=args.random_frame_count,
        window_count=args.window_count,
        window_frames=args.window_frames,
        full_scan_batch_rows=args.full_scan_batch_rows,
        output_json=args.output_json,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
