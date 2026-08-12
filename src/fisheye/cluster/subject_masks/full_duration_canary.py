"""Plan and run an isolated full-duration subject-mask publication canary.

The workflow is intentionally benchmark-only.  It stages immutable crop and
refined-keypoint references plus one real acquisition-video window to node-local
scratch, writes one raw shard, refines that shard in a separate CPU stage, and
finally publishes one inactive recording-level raw/refined/quality bundle.

One window may cover the complete recording (the whole-video topology), or a
strict clip index may define the real recording windows.  Window boundaries are
never invented for scheduler convenience and every nonempty crop-row interval
is owned by exactly one worker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4, uuid5

import numpy as np
import zarr

from fisheye.cluster.clipped_lsf import (
    build_execution_task,
    build_job,
    build_task_group_job,
)
from fisheye.cluster.lsf import (
    LsfExecutionMode,
    LsfResources,
    LsfWorkflow,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.backend import build_ssh_bsub_runner
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
    RUNTIME_USER_TOKEN,
)
from fisheye.cluster.subject_masks.publish_recording_bundle import (
    _worker_evidence,
    publish_recording_subject_mask_bundle,
)
from fisheye.refinement.finalize_subject_masks import finalize_subject_mask_run
from fisheye.segmentation import infer_unet_subject_masks
from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    build_crop_pixel_work_package_from_video_window,
)
from fisheye.shared.gpu_runtime_telemetry import (
    GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
    GPU_RUNTIME_TELEMETRY_SCHEMA_ID,
    GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION,
    GpuRuntimeTelemetrySampler,
    require_gpu_runtime_telemetry,
)
from fisheye.shared.zarr.crop_manifest import validate_crop_run_manifest
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    derive_subject_mask_frame_row_offsets,
)
from fisheye.shared.zarr.refined_subject_mask_extensions import (
    default_subject_mask_sampled_contour_profile,
)
from fisheye.shared.zarr.subject_mask_cache_storage import (
    plan_subject_mask_sampled_contour_storage,
)
from fisheye.shared.zarr.subject_mask_final_layout_units import (
    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS,
    SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION,
    build_subject_mask_final_layout_unit_package,
    subject_mask_final_layout_payload_plan,
    validate_subject_mask_final_layout_unit_package,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SubjectMaskCoreValidationMode,
)
from fisheye.shared.zarr.subject_mask_sampled_contour_worker_receipt import (
    load_subject_mask_sampled_contour_worker_receipt,
    validate_subject_mask_sampled_contour_worker_receipt,
    write_subject_mask_sampled_contour_worker_receipt,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    require_runs_parent,
)

PLAN_SCHEMA_ID = "palette.subject_mask.full_duration_canary_plan"
PLAN_SCHEMA_LEGACY_VERSION = 4
PLAN_SCHEMA_VERSION = 8
INFERENCE_REUSE_SCHEMA_ID = "palette.subject_mask.inference_reuse"
INFERENCE_REUSE_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.subject_mask.full_duration_canary_result"
RESULT_SCHEMA_VERSION = 5
WORKER_RESULT_SCHEMA_ID = "palette.subject_mask.full_duration_canary_worker"
WORKER_RESULT_SCHEMA_LEGACY_VERSION = 3
WORKER_RESULT_SCHEMA_VERSION = 4
PROBABILITY_DESTINATION_VALIDATION_HANDOFF_SCHEMA_ID = (
    "palette.subject_mask.probability_destination_validation_handoff"
)
PROBABILITY_DESTINATION_VALIDATION_HANDOFF_SCHEMA_VERSION = 1
FAMILY = "subject_mask_full_duration_canary"
BENCHMARK_CLASSIFICATION = "selector_ineligible_full_duration_canary"
DEFAULT_GPU_CONCURRENCY = 4
DEFAULT_CPU_CONCURRENCY = 4
DEFAULT_GPU_TELEMETRY_INTERVAL_SECONDS = 1
_ATTEMPT_NAMESPACE = UUID("79676a9f-24f1-4be9-ac50-c374b0fdccae")
_RAW_MASK_LABELS = ("subject_body", "eyes_union", "swim_bladder")
_REFINED_MASK_LABELS = (
    "subject_body",
    "eye_left",
    "eye_right",
    "swim_bladder",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r}.")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _same_file_identity(path: Path, expected: Mapping[str, Any]) -> bool:
    observed = _file_identity(path)
    return all(
        observed[name] == expected.get(name)
        for name in ("path", "size_bytes", "mtime_ns", "device", "inode")
    )


def _same_cluster_file_identity(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare file evidence that remains stable across cluster mount clients."""

    return all(
        observed.get(name) == expected.get(name)
        for name in ("path", "size_bytes", "mtime_ns")
    )


def _bind_scratch_window_video(
    root: Any, *, crop_run: str, window: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a proof-frozen clip to a node-local geometry reference copy."""

    source_path = str(Path(str(window["source_video_path"])).expanduser().resolve())
    if source_path != window["source_file"].get("path"):
        raise ValueError("Window source-video path differs from its frozen identity.")
    crop_group = root[f"crop_runs/{crop_run}"]
    previous = crop_group.attrs.get("source_video_path")
    crop_group.attrs["source_video_path"] = source_path
    return {
        "schema_id": "palette.subject_mask.scratch_window_video_binding",
        "schema_version": 1,
        "scope": "node_local_inference_reference",
        "declared_source_video_path": source_path,
        "previous_declared_source_video_path": (
            None if previous is None else str(previous)
        ),
        "source_file": dict(window["source_file"]),
    }


def _copy_file_with_digest(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    before = _file_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    started = time.perf_counter()
    total = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(16 * 1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
            total += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if not _same_file_identity(source, before):
        raise RuntimeError(f"Source changed while it was staged: {source}")
    if total != int(before["size_bytes"]):
        raise RuntimeError(f"Staged byte count differs for {source}.")
    return {
        "source": before,
        "destination": str(destination.resolve()),
        "size_bytes": total,
        "sha256": digest.hexdigest(),
        "duration_seconds": float(time.perf_counter() - started),
    }


def _tree_stats(path: Path) -> dict[str, int]:
    files = 0
    bytes_total = 0
    for root, _directories, names in os.walk(path):
        for name in names:
            candidate = Path(root) / name
            if candidate.is_symlink():
                continue
            stat = candidate.stat()
            files += 1
            bytes_total += int(stat.st_size)
    return {"file_count": files, "apparent_bytes": bytes_total}


def _tree_content_digest(path: Path) -> str:
    root = path.expanduser().resolve()
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    )
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "little"))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _resource_usage() -> dict[str, Any]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "self_maxrss_kib": int(own.ru_maxrss),
        "children_maxrss_kib": int(children.ru_maxrss),
        "self_user_seconds": float(own.ru_utime),
        "self_system_seconds": float(own.ru_stime),
        "children_user_seconds": float(children.ru_utime),
        "children_system_seconds": float(children.ru_stime),
    }


def _repo_identity(repo: Path, *, require_clean: bool = True) -> dict[str, Any]:
    resolved = repo.expanduser().resolve()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(resolved), *arguments],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    if require_clean and status:
        raise RuntimeError("Full-duration canaries require a clean committed worktree.")
    return {
        "path": str(resolved),
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _require_benchmark_root(run_root: Path) -> Path:
    resolved = run_root.expanduser().resolve()
    if ".palette_benchmarks" not in resolved.parts:
        raise ValueError(
            "Full-duration canary output must live below a .palette_benchmarks namespace."
        )
    return resolved


def _manifest_reference(run: Any, *, kind: str) -> dict[str, Any]:
    value = run.attrs.get("run_manifest")
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind} run has no exact run_manifest.")
    errors = (
        validate_crop_run_manifest(value)
        if kind == "crop"
        else validate_refined_keypoint_run_manifest(value)
    )
    if errors:
        raise ValueError(f"Invalid {kind} run_manifest: {list(errors)!r}")
    return {
        "schema_id": value["schema_id"],
        "schema_version": int(value["schema_version"]),
        "payload_digest": value["payload_digest"],
    }


def _copy_reference_run(
    *,
    source_archive: Path,
    parent: str,
    run_name: str,
    target_archive: Path,
) -> dict[str, Any]:
    source = source_archive.expanduser().resolve() / parent / run_name
    destination = target_archive.expanduser().resolve() / parent / run_name
    if not (source / "zarr.json").is_file():
        raise FileNotFoundError(source / "zarr.json")
    if destination.exists():
        raise FileExistsError(destination)
    target_root = zarr.open_group(
        str(target_archive), mode="a", zarr_format=3, use_consolidated=False
    )
    if parent not in target_root:
        target_root.create_group(parent)
    started = time.perf_counter()
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    source_stats = _tree_stats(source)
    destination_stats = _tree_stats(destination)
    if source_stats != destination_stats:
        raise RuntimeError(f"Reference-copy inventory differs for {parent}/{run_name}.")
    source_digest = _tree_content_digest(source)
    destination_digest = _tree_content_digest(destination)
    if source_digest != destination_digest:
        raise RuntimeError(f"Reference-copy bytes differ for {parent}/{run_name}.")
    return {
        "source_path": str(source),
        "destination_path": str(destination),
        "tree": destination_stats,
        "tree_sha256": destination_digest,
        "duration_seconds": float(time.perf_counter() - started),
    }


def _window_rows(
    *,
    frame_offsets: np.ndarray,
    frame_indices: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> tuple[int, int]:
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("Frame windows must be positive half-open intervals.")
    if end_frame >= frame_offsets.shape[0]:
        raise ValueError("Frame window exceeds the crop frame domain.")
    start_row = int(frame_offsets[start_frame])
    stop_row = int(frame_offsets[end_frame])
    selected = frame_indices[start_row:stop_row]
    if selected.size and (
        int(selected[0]) < start_frame or int(selected[-1]) >= end_frame
    ):
        raise ValueError("Frame offsets do not isolate their declared window rows.")
    return start_row, stop_row


def _resolve_windows(
    *,
    n_frames: int,
    frame_offsets: np.ndarray,
    frame_indices: np.ndarray,
    recording_dir: Path | None,
    clip_index: Path | None,
    whole_video: Path | None,
    recording_id: str,
    camera_identity: str | None,
    workflow_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if bool(clip_index) == bool(whole_video):
        raise ValueError("Pass exactly one of clip_index or whole_video.")
    source_document: dict[str, Any]
    raw_windows: list[dict[str, Any]]
    if clip_index is not None:
        index_path = clip_index.expanduser().resolve()
        payload = _strict_json(index_path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("status") != "ok"
            or payload.get("mode") != "materialized_stream_copy"
        ):
            raise ValueError(
                "Clip index must be an ok materialized-stream-copy document."
            )
        checks = payload.get("checks")
        if (
            not isinstance(checks, list)
            or not checks
            or any(
                not isinstance(check, Mapping) or check.get("status") != "ok"
                for check in checks
            )
        ):
            raise ValueError("Clip index contains missing or failed checks.")
        clips = payload.get("clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError("Clip index has no clips.")
        if type(payload.get("clip_count")) is not int or payload.get(
            "clip_count"
        ) != len(clips):
            raise ValueError("Clip-index clip_count differs from its clip table.")
        base = (
            recording_dir.expanduser().resolve()
            if recording_dir is not None
            else index_path.parent
        )
        raw_windows = []
        for ordinal, item in enumerate(clips):
            if not isinstance(item, Mapping):
                raise ValueError(f"Clip row {ordinal} is not an object.")
            if (
                item.get("status") != "materialized"
                or item.get("clip_index") != ordinal
            ):
                raise ValueError(
                    "Clip rows must be materialized and contiguously indexed."
                )
            raw_video = Path(str(item.get("video_path") or ""))
            video = raw_video if raw_video.is_absolute() else base / raw_video
            raw_manifest = Path(str(item.get("clip_manifest_path") or ""))
            clip_manifest = (
                raw_manifest if raw_manifest.is_absolute() else base / raw_manifest
            )
            if not clip_manifest.is_file():
                raise FileNotFoundError(clip_manifest)
            raw_windows.append(
                {
                    "window_id": str(item.get("clip_id") or f"clip_{ordinal:06d}"),
                    "window_index": ordinal,
                    "source_video_path": str(video.resolve()),
                    "start_frame": int(item.get("actual_start_frame", -1)),
                    "end_frame": int(item.get("end_frame_exclusive", -1)),
                    "frame_count": int(item.get("frame_count", -1)),
                    "camera_identity": str(
                        item.get("camera_serial") or camera_identity or ""
                    ),
                    "source_kind": "recording_clip_index",
                    "clip_manifest_path": str(clip_manifest.resolve()),
                    "clip_manifest_sha256": _sha256_file(clip_manifest),
                }
            )
        source_document = {
            "mode": "clipped_recording",
            "clip_index_path": str(index_path),
            "clip_index_sha256": _sha256_file(index_path),
            "clip_count": len(raw_windows),
        }
        source_document["window_index_sha256"] = source_document["clip_index_sha256"]
        index_recording = str(payload.get("recording_id") or "")
        if index_recording and index_recording != recording_id:
            raise ValueError("Clip-index recording identity differs from the canary.")
    else:
        assert whole_video is not None
        video = whole_video.expanduser().resolve()
        raw_windows = [
            {
                "window_id": "whole_recording",
                "window_index": 0,
                "source_video_path": str(video),
                "start_frame": 0,
                "end_frame": n_frames,
                "frame_count": n_frames,
                "camera_identity": str(camera_identity or "unknown_camera"),
                "source_kind": "whole_recording_video",
            }
        ]
        source_document = {
            "mode": "whole_recording",
            "clip_count": 1,
            "window_index_sha256": canonical_json_sha256(
                {
                    "schema_id": "palette.whole_recording_frame_window",
                    "schema_version": 1,
                    "recording_id": recording_id,
                    "camera_identity": str(camera_identity or "unknown_camera"),
                    "video_path": str(video),
                    "start_frame": 0,
                    "end_frame_exclusive": n_frames,
                }
            ),
        }

    expected_start = 0
    windows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(raw_windows):
        if item["window_index"] != ordinal:
            raise ValueError("Window ordinals must be contiguous and zero-based.")
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        if start != expected_start or end - start != int(item["frame_count"]):
            raise ValueError(
                "Video windows must exactly and contiguously cover frames."
            )
        video = Path(str(item["source_video_path"]))
        if not video.is_file():
            raise FileNotFoundError(video)
        start_row, stop_row = _window_rows(
            frame_offsets=frame_offsets,
            frame_indices=frame_indices,
            start_frame=start,
            end_frame=end,
        )
        window_id = str(item["window_id"])
        item.update(
            {
                "source_file": _file_identity(video),
                "row_start": start_row,
                "row_stop": stop_row,
                "row_count": stop_row - start_row,
                "raw_run": f"subject_mask_canary_raw_{window_id}",
                "refined_run": f"subject_mask_canary_refined_{window_id}",
                "raw_attempt_id": str(
                    uuid5(
                        _ATTEMPT_NAMESPACE,
                        f"{recording_id}:{workflow_id}:raw:{window_id}",
                    )
                ),
                "refined_attempt_id": str(
                    uuid5(
                        _ATTEMPT_NAMESPACE,
                        f"{recording_id}:{workflow_id}:refined:{window_id}",
                    )
                ),
            }
        )
        windows.append(item)
        expected_start = end
    if expected_start != n_frames:
        raise ValueError("Video windows do not cover the complete frame domain.")
    nonempty = [item for item in windows if int(item["row_count"]) > 0]
    if not nonempty:
        raise ValueError("Full-duration mask canary has no crop rows.")
    observed_rows = np.concatenate(
        [
            np.arange(item["row_start"], item["row_stop"], dtype=np.int64)
            for item in nonempty
        ]
    )
    if not np.array_equal(observed_rows, np.arange(frame_indices.shape[0])):
        raise ValueError(
            "Nonempty video windows do not own every crop row exactly once."
        )
    return windows, source_document


def prepare_canary(
    *,
    run_root: Path,
    repo: Path,
    source_crop_zarr: Path,
    crop_run: str,
    source_refined_keypoint_zarr: Path,
    refined_keypoint_run: str,
    model_path: Path,
    model_sha256: str,
    recording_id: str,
    recording_dir: Path | None = None,
    clip_index: Path | None = None,
    whole_video: Path | None = None,
    camera_identity: str | None = None,
    run_label: str,
    reuse_inference_plan: Path | None = None,
    require_clean_repo: bool = True,
    core_physical_unit_workers: int = 4,
    quality_compute_workers: int = 4,
    gpu_telemetry_interval_seconds: int = DEFAULT_GPU_TELEMETRY_INTERVAL_SECONDS,
    synchronized_stage_profiling: bool = False,
) -> dict[str, Any]:
    """Freeze inputs and copy exact maintained references into an isolated store."""

    if (
        type(gpu_telemetry_interval_seconds) is not int
        or gpu_telemetry_interval_seconds <= 0
    ):
        raise ValueError("GPU telemetry interval must be one positive integer.")
    if type(synchronized_stage_profiling) is not bool:
        raise ValueError("Synchronized stage profiling must be one exact boolean.")
    if type(core_physical_unit_workers) is not int or core_physical_unit_workers <= 0:
        raise ValueError("core_physical_unit_workers must be a positive integer.")
    if type(quality_compute_workers) is not int or quality_compute_workers <= 0:
        raise ValueError("quality_compute_workers must be a positive integer.")
    output = _require_benchmark_root(run_root)
    if output.exists():
        raise FileExistsError(f"Immutable canary run root already exists: {output}")
    repo_identity = _repo_identity(repo, require_clean=require_clean_repo)
    model = model_path.expanduser().resolve()
    if _sha256_file(model) != str(model_sha256):
        raise ValueError("Subject-mask model SHA-256 differs from the requested model.")
    crop_archive = source_crop_zarr.expanduser().resolve()
    keypoint_archive = source_refined_keypoint_zarr.expanduser().resolve()
    crop_root = open_zarr_root(crop_archive, mode="r")
    crop = crop_root[f"crop_runs/{crop_run}"]
    crop_manifest = _manifest_reference(crop, kind="crop")
    required_crop = {
        "frame_row_offsets",
        "source_acquisition_frame_index",
        "frame_indices",
        "instance_key",
        "source_crop_xywh",
        "roi_coordinates_full",
        "source_row_signature",
    }
    missing_crop = sorted(required_crop - set(crop.keys()))
    if missing_crop:
        raise ValueError(f"Crop-v2 input lacks required arrays: {missing_crop!r}")
    offsets = np.asarray(crop["frame_row_offsets"][:], dtype=np.int64)
    frames = np.asarray(crop["source_acquisition_frame_index"][:], dtype=np.int64)
    if offsets.ndim != 1 or offsets.shape[0] < 2:
        raise ValueError("Crop frame_row_offsets must have shape [F+1].")
    n_frames = int(offsets.shape[0] - 1)
    n_rows = int(frames.shape[0])
    crop_xywh = np.asarray(crop["source_crop_xywh"][:])
    if crop_xywh.shape != (n_rows, 4):
        raise ValueError("Crop source_crop_xywh must have shape [R,4].")
    roi_sizes = np.unique(crop_xywh[:, 2:4].astype(np.int64), axis=0)
    if roi_sizes.shape[0] != 1 or np.any(roi_sizes[0] <= 0):
        raise ValueError(
            "Subject-mask final-layout planning requires one positive ROI extent."
        )
    roi_width, roi_height = (int(value) for value in roi_sizes[0])
    expected_offsets = derive_subject_mask_frame_row_offsets(frames, n_frames=n_frames)
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError(
            "Crop frame_row_offsets do not exactly index acquisition frames."
        )
    if not np.array_equal(np.asarray(crop["frame_indices"][:], dtype=np.int64), frames):
        raise ValueError("Crop frame_indices differ from acquisition-frame identity.")

    keypoint_root = open_zarr_root(keypoint_archive, mode="r")
    keypoints = keypoint_root[f"refined_keypoints_runs/{refined_keypoint_run}"]
    keypoint_manifest = _manifest_reference(keypoints, kind="refined_keypoint")
    for name in ("source_crop_row_ids", "instance_key"):
        if name not in keypoints:
            raise ValueError(f"Refined keypoints lack {name}.")
    if not np.array_equal(
        np.asarray(keypoints["source_crop_row_ids"][:], dtype=np.int64),
        np.arange(n_rows, dtype=np.int64),
    ):
        raise ValueError("Refined keypoints do not cover canonical crop rows in order.")
    if not np.array_equal(
        np.asarray(keypoints["instance_key"][:], dtype=np.uint64),
        np.asarray(crop["instance_key"][:], dtype=np.uint64),
    ):
        raise ValueError("Refined keypoint and crop instance keys differ.")

    label = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(run_label)
    ).strip("_")
    if not label:
        raise ValueError("run_label does not produce a safe identifier.")
    windows, video_source = _resolve_windows(
        n_frames=n_frames,
        frame_offsets=offsets,
        frame_indices=frames,
        recording_dir=recording_dir,
        clip_index=clip_index,
        whole_video=whole_video,
        recording_id=recording_id,
        camera_identity=camera_identity,
        workflow_id=label,
    )
    raw_final_dimensions = SubjectMaskDimensions(
        n_frames=n_frames,
        n_rois=n_rows,
        n_channels=len(_RAW_MASK_LABELS),
        roi_height=roi_height,
        roi_width=roi_width,
    )
    refined_final_dimensions = SubjectMaskDimensions(
        n_frames=n_frames,
        n_rois=n_rows,
        n_channels=len(_REFINED_MASK_LABELS),
        roi_height=roi_height,
        roi_width=roi_width,
    )
    refined_components = SubjectMaskComponentRegistry(_REFINED_MASK_LABELS)
    sampled_contour_profile = default_subject_mask_sampled_contour_profile(
        refined_components
    )
    output.mkdir(parents=True)
    for child in ("logs", "status", "workers", "bundles", "publish"):
        (output / child).mkdir()
    target = output / "analysis.zarr"
    target_root = zarr.open_group(str(target), mode="w", zarr_format=3)
    target_root.attrs.update(
        {
            "recording_id": str(recording_id),
            "benchmark_classification": BENCHMARK_CLASSIFICATION,
            "stage_selector_eligible": False,
            "source_frame_count": n_frames,
        }
    )
    crop_copy = _copy_reference_run(
        source_archive=crop_archive,
        parent="crop_runs",
        run_name=crop_run,
        target_archive=target,
    )
    keypoint_copy = _copy_reference_run(
        source_archive=keypoint_archive,
        parent="refined_keypoints_runs",
        run_name=refined_keypoint_run,
        target_archive=target,
    )
    payload: dict[str, Any] = {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "planned",
        "classification": BENCHMARK_CLASSIFICATION,
        "created_at_utc": _utc_now(),
        "workflow_id": label,
        "run_root": str(output),
        "repo": repo_identity,
        "recording": {
            "recording_id": str(recording_id),
            "camera_identity": str(
                next(
                    (
                        item["camera_identity"]
                        for item in windows
                        if item["camera_identity"]
                    ),
                    camera_identity or "unknown_camera",
                )
            ),
            "n_frames": n_frames,
            "n_rows": n_rows,
            "video_source": video_source,
        },
        "references": {
            "analysis_zarr": str(target),
            "crop": {
                "source_archive": str(crop_archive),
                "parent": "crop_runs",
                "run": str(crop_run),
                "manifest": crop_manifest,
                "copy": crop_copy,
            },
            "refined_keypoints": {
                "source_archive": str(keypoint_archive),
                "parent": "refined_keypoints_runs",
                "run": str(refined_keypoint_run),
                "manifest": keypoint_manifest,
                "copy": keypoint_copy,
            },
        },
        "model": {
            **_file_identity(model),
            "sha256": str(model_sha256),
        },
        "final_layout": {
            "schema_id": "palette.subject_mask.final_layout_work_plan",
            "schema_version": 1,
            "ownership_policy": (
                "complete_final_outer_row_units_per_worker_with_"
                "deterministic_boundary_rebuild_v1"
            ),
            "raw": subject_mask_final_layout_payload_plan(
                kind="raw_probability_uint8",
                dimensions=raw_final_dimensions,
            ),
            "refined": subject_mask_final_layout_payload_plan(
                kind="refined_dense_core",
                dimensions=refined_final_dimensions,
            ),
            "sampled_contours": plan_subject_mask_sampled_contour_storage(
                refined_final_dimensions,
                components=refined_components,
                contour_profile=sampled_contour_profile,
            ).as_manifest(),
        },
        "windows": windows,
        "outputs": {
            "raw_run": f"subject_masks_{label}",
            "refined_run": f"refined_subject_masks_{label}",
            "quality_run": f"subject_mask_quality_{label}",
            "cache_run": f"subject_mask_sampled_contours_{label}",
            "bundle_id": f"subject_mask_bundle_{label}",
            "result_path": str(output / "result.json"),
        },
        "execution": {
            "inference": {
                "device": "0",
                "batch_size": 128,
                "probability_dtype": "uint8",
                "inner_chunk_rows": 32,
                "outer_shard_rows": 2048,
                "destination_validation_mode": (
                    infer_unet_subject_masks.MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT
                ),
                "synchronized_stage_profiling": synchronized_stage_profiling,
                "gpu_runtime_telemetry": {
                    "enabled": True,
                    "sample_interval_seconds": gpu_telemetry_interval_seconds,
                    "schema_id": GPU_RUNTIME_TELEMETRY_SCHEMA_ID,
                    "schema_version": GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION,
                    "identity_policy": GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
                },
            },
            "refinement": {
                "chunk_rows": 256,
                "dense_mask_chunk_rows": 256,
                "workers": 16,
                "metric_level": "cheap",
            },
            "publication": {
                "core_physical_unit_workers": int(core_physical_unit_workers),
                "quality_compute_workers": int(quality_compute_workers),
                "core_validation_mode": (
                    SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE.value
                ),
                "logical_identity_unit_rows": (
                    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
                ),
                "ownership_policy": (
                    "single_writer_v1_future_workers_require_disjoint_whole_shards"
                    if int(core_physical_unit_workers) == 1
                    else "bounded_threaded_disjoint_whole_physical_row_bands_v1"
                ),
            },
        },
        "safety": {
            "production_registry_used": False,
            "production_selector_mutation_allowed": False,
            "bundle_activation_allowed": False,
            "all_outputs_below_run_root": True,
            "worker_writes_are_node_local_until_atomic_bundle_publish": True,
            "window_rows_are_exact_nonoverlapping_complete": True,
            "final_layout_units_are_selector_ineligible_transport": True,
            "receipt_bound_composable_dense_identity_required": True,
            "finalizer_full_dense_decode_hash_allowed": False,
            "worker_sampled_contours_required": True,
            "full_ragged_contours_allowed": False,
        },
    }
    payload["inference_reuse"] = (
        _build_inference_reuse_contract(
            source_plan_path=reuse_inference_plan,
            candidate_plan=payload,
        )
        if reuse_inference_plan is not None
        else None
    )
    payload["plan_digest"] = canonical_json_sha256(payload)
    _write_json_atomic(output / "plan.json", payload)
    return payload


def load_plan(path: Path) -> dict[str, Any]:
    plan_path = path.expanduser().resolve()
    payload = _strict_json(plan_path)
    if not isinstance(payload, dict):
        raise ValueError("Canary plan root must be an object.")
    if (
        payload.get("schema_id") != PLAN_SCHEMA_ID
        or payload.get("schema_version")
        not in {PLAN_SCHEMA_LEGACY_VERSION, 5, 6, 7, PLAN_SCHEMA_VERSION}
        or payload.get("status") != "planned"
        or payload.get("classification") != BENCHMARK_CLASSIFICATION
    ):
        raise ValueError("Unsupported or non-planned full-duration canary document.")
    digest = payload.pop("plan_digest", None)
    observed = canonical_json_sha256(payload)
    payload["plan_digest"] = digest
    if digest != observed:
        raise ValueError("Full-duration canary plan digest differs.")
    run_root = _require_benchmark_root(Path(str(payload.get("run_root") or "")))
    if plan_path != run_root / "plan.json":
        raise ValueError(
            "Canary plan is not stored at its declared immutable run root."
        )
    outputs = payload.get("outputs")
    references = payload.get("references")
    if not isinstance(outputs, Mapping) or not isinstance(references, Mapping):
        raise ValueError("Canary plan lacks outputs or references.")
    analysis = Path(str(references.get("analysis_zarr") or "")).resolve()
    if run_root not in analysis.parents:
        raise ValueError("Canary analysis target escapes its benchmark run root.")
    if payload.get("safety", {}).get("bundle_activation_allowed") is not False:
        raise ValueError("Canary plan does not fail closed on activation.")
    reuse = payload.get("inference_reuse")
    if payload.get("schema_version") in {5, 6, 7, PLAN_SCHEMA_VERSION}:
        _validate_inference_reuse_contract(reuse)
    elif reuse is not None:
        raise ValueError("Legacy canary plans cannot declare inference reuse.")
    if (
        payload.get("safety", {}).get("worker_sampled_contours_required") is not True
        or payload.get("safety", {}).get("full_ragged_contours_allowed") is not False
    ):
        raise ValueError("Canary plan does not enforce the sampled-contour profile.")
    publication = payload.get("execution", {}).get("publication", {})
    if (
        payload.get("safety", {}).get(
            "receipt_bound_composable_dense_identity_required"
        )
        is not True
        or payload.get("safety", {}).get("finalizer_full_dense_decode_hash_allowed")
        is not False
        or publication.get("core_validation_mode")
        != SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE.value
        or publication.get("logical_identity_unit_rows")
        != SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
    ):
        raise ValueError("Canary plan does not enforce composable dense identity.")
    if payload["schema_version"] in {6, 7, PLAN_SCHEMA_VERSION}:
        inference = payload.get("execution", {}).get("inference", {})
        telemetry = inference.get("gpu_runtime_telemetry")
        if (
            type(inference.get("synchronized_stage_profiling")) is not bool
            or not isinstance(telemetry, Mapping)
            or set(telemetry)
            != {
                "enabled",
                "sample_interval_seconds",
                "schema_id",
                "schema_version",
                "identity_policy",
            }
            or telemetry["enabled"] is not True
            or type(telemetry["sample_interval_seconds"]) is not int
            or telemetry["sample_interval_seconds"] <= 0
            or telemetry["schema_id"] != GPU_RUNTIME_TELEMETRY_SCHEMA_ID
            or telemetry["schema_version"] != GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION
            or telemetry["identity_policy"] != GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY
        ):
            raise ValueError("Canary GPU runtime telemetry plan differs.")
    if payload["schema_version"] in {7, PLAN_SCHEMA_VERSION}:
        inference = payload.get("execution", {}).get("inference", {})
        if (
            inference.get("destination_validation_mode")
            != infer_unet_subject_masks.MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT
        ):
            raise ValueError(
                "Canary probability destination validation policy differs."
            )
    if payload["schema_version"] == PLAN_SCHEMA_VERSION:
        quality_workers = publication.get("quality_compute_workers")
        if type(quality_workers) is not int or quality_workers <= 0:
            raise ValueError("Canary quality compute-worker policy differs.")
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Canary plan has no windows.")
    if reuse is not None:
        if Path(str(reuse["source_plan_path"])).resolve() == plan_path:
            raise ValueError("Inference-reuse source plan cannot be the retry plan.")
        expected_reused_windows = {
            str(window["window_id"])
            for window in windows
            if isinstance(window, Mapping) and int(window.get("row_count", 0)) > 0
        }
        if set(reuse["window_results"]) != expected_reused_windows:
            raise ValueError("Inference-reuse window coverage differs from the plan.")
    for stage, kind, role in (
        ("inference", "raw_probability_uint8", "raw"),
        ("refinement", "refined_dense_core", "refined"),
    ):
        dimensions = _final_layout_dimensions(payload, stage=stage)
        expected = subject_mask_final_layout_payload_plan(
            kind=kind,
            dimensions=dimensions,
        )
        final_layout = payload.get("final_layout")
        observed = final_layout.get(role) if isinstance(final_layout, Mapping) else None
        if observed != expected:
            raise ValueError(f"Canary {role} final-layout plan differs from policy.")
    refined_dimensions = _final_layout_dimensions(payload, stage="refinement")
    refined_components = SubjectMaskComponentRegistry(_REFINED_MASK_LABELS)
    expected_sampled = plan_subject_mask_sampled_contour_storage(
        refined_dimensions,
        components=refined_components,
        contour_profile=default_subject_mask_sampled_contour_profile(
            refined_components
        ),
    ).as_manifest()
    final_layout = payload.get("final_layout")
    if (
        not isinstance(final_layout, Mapping)
        or final_layout.get("sampled_contours") != expected_sampled
    ):
        raise ValueError("Canary sampled-contour final layout differs from policy.")
    return payload


def _window(plan: Mapping[str, Any], index: int) -> dict[str, Any]:
    windows = plan["windows"]
    matches = [item for item in windows if int(item["window_index"]) == int(index)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one canary window index {index}; found {len(matches)}."
        )
    result = dict(matches[0])
    if int(result["row_count"]) <= 0:
        raise ValueError("Zero-row windows do not receive inference/refinement tasks.")
    return result


def _final_layout_dimensions(
    plan: Mapping[str, Any], *, stage: str
) -> SubjectMaskDimensions:
    final_layout = plan.get("final_layout")
    if not isinstance(final_layout, Mapping):
        raise ValueError("Canary plan lacks the final-layout work plan.")
    role = "raw" if stage == "inference" else "refined"
    role_plan = final_layout.get(role)
    dimensions = role_plan.get("dimensions") if isinstance(role_plan, Mapping) else None
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "n_frames",
        "n_rois",
        "n_channels",
        "roi_height",
        "roi_width",
    }:
        raise ValueError("Canary final-layout dimensions are not exact.")
    return SubjectMaskDimensions(
        n_frames=int(dimensions["n_frames"]),
        n_rois=int(dimensions["n_rois"]),
        n_channels=int(dimensions["n_channels"]),
        roi_height=int(dimensions["roi_height"]),
        roi_width=int(dimensions["roi_width"]),
    )


def _stage_reference_archive(plan: Mapping[str, Any], destination: Path) -> Path:
    target = destination.expanduser().resolve()
    root = zarr.open_group(str(target), mode="w", zarr_format=3)
    recording = plan["recording"]
    root.attrs.update(
        {
            "recording_id": recording["recording_id"],
            "source_frame_count": int(recording["n_frames"]),
            "benchmark_classification": BENCHMARK_CLASSIFICATION,
        }
    )
    reference_archive = Path(plan["references"]["analysis_zarr"])
    for name in ("crop", "refined_keypoints"):
        reference = plan["references"][name]
        parent = str(reference["parent"])
        run_name = str(reference["run"])
        root.create_group(parent)
        shutil.copytree(
            reference_archive / parent / run_name,
            target / parent / run_name,
            copy_function=shutil.copy2,
        )
    return target


def _existing_worker_result(
    *,
    bundle: Path,
    plan: Mapping[str, Any],
    window: Mapping[str, Any],
    stage: str,
) -> dict[str, Any] | None:
    result_path = bundle / "result.json"
    if not bundle.exists():
        return None
    if not result_path.is_file():
        raise RuntimeError(
            f"Incomplete immutable worker bundle already exists: {bundle}"
        )
    result = _strict_json(result_path)
    if (
        not isinstance(result, dict)
        or result.get("schema_id") != WORKER_RESULT_SCHEMA_ID
        or result.get("schema_version")
        not in {WORKER_RESULT_SCHEMA_LEGACY_VERSION, WORKER_RESULT_SCHEMA_VERSION}
        or result.get("status") != "complete"
        or result.get("stage") != stage
        or result.get("plan_digest") != plan["plan_digest"]
        or result.get("window_id") != window["window_id"]
    ):
        raise RuntimeError(f"Existing worker bundle identity differs: {bundle}")
    binding = result.get("final_layout_unit_package")
    if not isinstance(binding, Mapping) or set(binding) != {
        "relative_path",
        "schema_version",
        "payload_digest",
        "kind",
        "array_path",
        "storage_plan_digest",
        "complete_unit_count",
        "encoded_object_count",
        "encoded_bytes",
        "worker_receipt_payload_digest",
        "source_array_validation_digest",
    }:
        raise RuntimeError(f"Worker bundle lacks exact final-layout evidence: {bundle}")
    if binding.get("relative_path") != "final_layout_unit":
        raise RuntimeError(f"Worker final-layout package path is unsafe: {bundle}")
    package_receipt = validate_subject_mask_final_layout_unit_package(
        bundle / "final_layout_unit",
        verify_object_digests=False,
    )
    expected_role = "raw" if stage == "inference" else "refined"
    expected = plan["final_layout"][expected_role]
    if (
        binding.get("payload_digest") != package_receipt["payload_digest"]
        or binding.get("schema_version")
        != SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION
        or package_receipt.get("schema_version")
        != SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION
        or binding.get("kind") != package_receipt["payload"]["kind"]
        or binding.get("array_path") != package_receipt["payload"]["array_path"]
        or binding.get("storage_plan_digest")
        != package_receipt["payload"]["storage_plan_digest"]
        or package_receipt["payload"]["storage_plan_digest"]
        != expected["storage_plan_digest"]
        or package_receipt["payload"]["producer_commit"] != plan["repo"]["commit"]
        or binding.get("worker_receipt_payload_digest")
        != package_receipt["payload"]["worker_receipt_payload_digest"]
        or binding.get("worker_receipt_payload_digest")
        != result.get("proof", {}).get("receipt_payload_digest")
        or binding.get("source_array_validation_digest")
        != canonical_json_sha256(package_receipt["payload"]["source_array_validation"])
    ):
        raise RuntimeError(f"Worker final-layout package binding differs: {bundle}")
    if stage == "inference" and plan.get("schema_version") in {
        7,
        PLAN_SCHEMA_VERSION,
    }:
        _validate_probability_destination_validation_handoff(
            result.get("probability_destination_validation_handoff"),
            worker_receipt_payload_digest=str(
                package_receipt["payload"]["worker_receipt_payload_digest"]
            ),
            final_layout_receipt=package_receipt,
        )
    sampled_binding = result.get("sampled_contour_worker_receipt")
    if stage == "inference":
        if sampled_binding is not None:
            raise RuntimeError(
                f"Inference worker unexpectedly carries sampled contours: {bundle}"
            )
    else:
        expected_fields = {
            "relative_path",
            "payload_digest",
            "source_run_path",
            "global_row_interval",
            "array_document_digest",
            "source_worker_receipt_payload_digest",
            "producer_commit",
        }
        if not isinstance(sampled_binding, Mapping) or set(sampled_binding) != (
            expected_fields
        ):
            raise RuntimeError(
                f"Refinement worker lacks exact sampled-contour evidence: {bundle}"
            )
        relative = str(sampled_binding.get("relative_path") or "")
        if relative != "sampled_contour_receipt.json":
            raise RuntimeError(
                f"Worker sampled-contour receipt path is unsafe: {bundle}"
            )
        sampled_receipt = load_subject_mask_sampled_contour_worker_receipt(
            bundle / relative
        )
        run = open_zarr_root(bundle / "archive.zarr", mode="r")[
            f"refined_subject_masks_runs/{window['refined_run']}"
        ]
        proof = _worker_evidence(bundle / "archive.zarr", run)
        validated = validate_subject_mask_sampled_contour_worker_receipt(
            sampled_receipt,
            run=run,
            worker_receipt=proof["receipt"],
            verify_values=False,
        )
        payload = validated["payload"]
        if (
            sampled_binding.get("payload_digest") != validated["payload_digest"]
            or sampled_binding.get("source_run_path") != payload["source_run_path"]
            or sampled_binding.get("global_row_interval")
            != payload["global_row_interval"]
            or sampled_binding.get("array_document_digest")
            != payload["array_document_digest"]
            or sampled_binding.get("source_worker_receipt_payload_digest")
            != payload["source_dense_worker"]["worker_receipt_payload_digest"]
            or sampled_binding.get("producer_commit") != plan["repo"]["commit"]
        ):
            raise RuntimeError(
                f"Worker sampled-contour receipt binding differs: {bundle}"
            )
    return result


def _validate_probability_destination_validation_handoff(
    value: object,
    *,
    worker_receipt_payload_digest: str,
    final_layout_receipt: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema_id",
        "schema_version",
        "status",
        "writer_mode",
        "writer_status",
        "worker_receipt_payload_digest",
        "final_layout_payload_digest",
        "final_layout_source_array_validation_digest",
        "publication_requirement",
    }
    payload = final_layout_receipt.get("payload")
    source_validation = (
        payload.get("source_array_validation")
        if isinstance(payload, Mapping)
        else None
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema_id")
        != PROBABILITY_DESTINATION_VALIDATION_HANDOFF_SCHEMA_ID
        or value.get("schema_version")
        != PROBABILITY_DESTINATION_VALIDATION_HANDOFF_SCHEMA_VERSION
        or value.get("status") != "complete"
        or value.get("writer_mode")
        != infer_unet_subject_masks.MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT
        or value.get("writer_status")
        != "deferred_to_mandatory_final_layout_unit"
        or value.get("worker_receipt_payload_digest")
        != worker_receipt_payload_digest
        or value.get("final_layout_payload_digest")
        != final_layout_receipt.get("payload_digest")
        or not isinstance(source_validation, Mapping)
        or value.get("final_layout_source_array_validation_digest")
        != canonical_json_sha256(source_validation)
        or value.get("publication_requirement")
        != "immutable_worker_bundle_requires_verified_final_layout_unit_v1"
    ):
        raise RuntimeError(
            "Probability destination validation handoff is absent or differs."
        )


def _is_sha256_text(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit_text(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_inference_reuse_contract(value: object) -> None:
    if value is None:
        return
    fields = {
        "schema_id",
        "schema_version",
        "source_plan_path",
        "source_plan_digest",
        "source_run_root",
        "source_palette_commit",
        "window_results",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Inference-reuse contract fields are not exact.")
    if (
        value.get("schema_id") != INFERENCE_REUSE_SCHEMA_ID
        or value.get("schema_version") != INFERENCE_REUSE_SCHEMA_VERSION
        or not _is_sha256_text(value.get("source_plan_digest"))
        or not _is_git_commit_text(value.get("source_palette_commit"))
    ):
        raise ValueError("Inference-reuse contract envelope is invalid.")
    for name in ("source_plan_path", "source_run_root"):
        raw = value.get(name)
        if type(raw) is not str or not raw.strip() or not Path(raw).is_absolute():
            raise ValueError(f"Inference-reuse {name} is invalid.")
    windows = value.get("window_results")
    if not isinstance(windows, Mapping) or not windows:
        raise ValueError("Inference-reuse window results are absent.")
    expected_window_fields = {
        "window_index",
        "raw_run",
        "bundle_path",
        "result_sha256",
        "final_layout_payload_digest",
    }
    for window_id, record in windows.items():
        if type(window_id) is not str or not window_id:
            raise ValueError("Inference-reuse window ID is invalid.")
        if not isinstance(record, Mapping) or set(record) != expected_window_fields:
            raise ValueError(f"Inference-reuse record fields differ for {window_id!r}.")
        if type(record.get("window_index")) is not int or record["window_index"] < 0:
            raise ValueError(
                f"Inference-reuse window index is invalid for {window_id!r}."
            )
        if type(record.get("raw_run")) is not str or not record["raw_run"]:
            raise ValueError(f"Inference-reuse raw run is invalid for {window_id!r}.")
        bundle = record.get("bundle_path")
        if type(bundle) is not str or not bundle or not Path(bundle).is_absolute():
            raise ValueError(
                f"Inference-reuse bundle path is invalid for {window_id!r}."
            )
        for digest_name in ("result_sha256", "final_layout_payload_digest"):
            if not _is_sha256_text(record.get(digest_name)):
                raise ValueError(
                    f"Inference-reuse {digest_name} is invalid for {window_id!r}."
                )


def _reference_contract_identity(plan: Mapping[str, Any], name: str) -> dict[str, Any]:
    reference = plan["references"][name]
    return {
        "parent": reference["parent"],
        "run": reference["run"],
        "manifest": reference["manifest"],
    }


def _build_inference_reuse_contract(
    *,
    source_plan_path: Path,
    candidate_plan: Mapping[str, Any],
) -> dict[str, Any]:
    source_path = source_plan_path.expanduser().resolve()
    source_plan = load_plan(source_path)
    if source_plan.get("inference_reuse") is not None:
        raise ValueError(
            "Inference reuse cannot currently chain through another retry."
        )
    if source_plan.get("workflow_id") != candidate_plan.get("workflow_id"):
        raise ValueError("Inference reuse requires the same workflow/run label.")
    for name in ("recording", "windows"):
        if source_plan.get(name) != candidate_plan.get(name):
            raise ValueError(f"Inference reuse {name} identity differs.")
    if source_plan.get("model", {}).get("sha256") != candidate_plan.get(
        "model", {}
    ).get("sha256"):
        raise ValueError("Inference reuse model digest differs.")
    if source_plan.get("final_layout", {}).get("raw") != candidate_plan.get(
        "final_layout", {}
    ).get("raw"):
        raise ValueError("Inference reuse raw final-layout plan differs.")
    for name in ("crop", "refined_keypoints"):
        if _reference_contract_identity(source_plan, name) != (
            _reference_contract_identity(candidate_plan, name)
        ):
            raise ValueError(f"Inference reuse {name} reference differs.")

    source_root = Path(source_plan["run_root"])
    window_results: dict[str, dict[str, Any]] = {}
    for window in source_plan["windows"]:
        if int(window["row_count"]) <= 0:
            continue
        window_id = str(window["window_id"])
        bundle = source_root / "bundles" / "inference" / window_id
        result = _existing_worker_result(
            bundle=bundle,
            plan=source_plan,
            window=window,
            stage="inference",
        )
        if result is None:
            raise ValueError(f"Inference reuse bundle is absent for {window_id!r}.")
        window_results[window_id] = {
            "window_index": int(window["window_index"]),
            "raw_run": str(window["raw_run"]),
            "bundle_path": str(bundle.resolve()),
            "result_sha256": _sha256_file(bundle / "result.json"),
            "final_layout_payload_digest": result["final_layout_unit_package"][
                "payload_digest"
            ],
        }
    contract = {
        "schema_id": INFERENCE_REUSE_SCHEMA_ID,
        "schema_version": INFERENCE_REUSE_SCHEMA_VERSION,
        "source_plan_path": str(source_path),
        "source_plan_digest": source_plan["plan_digest"],
        "source_run_root": str(source_root.resolve()),
        "source_palette_commit": source_plan["repo"]["commit"],
        "window_results": window_results,
    }
    _validate_inference_reuse_contract(contract)
    return contract


def _resolve_inference_bundle(
    *,
    plan: Mapping[str, Any],
    window: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    reuse = plan.get("inference_reuse")
    if reuse is None:
        bundle = (
            Path(plan["run_root"]) / "bundles" / "inference" / str(window["window_id"])
        )
        result = _existing_worker_result(
            bundle=bundle,
            plan=plan,
            window=window,
            stage="inference",
        )
        if result is None:
            raise RuntimeError("Inference bundle is missing.")
        return bundle, result

    _validate_inference_reuse_contract(reuse)
    source_plan = load_plan(Path(str(reuse["source_plan_path"])))
    if (
        source_plan["plan_digest"] != reuse["source_plan_digest"]
        or source_plan["repo"]["commit"] != reuse["source_palette_commit"]
        or str(Path(source_plan["run_root"]).resolve()) != reuse["source_run_root"]
    ):
        raise RuntimeError("Inference-reuse source plan identity differs.")
    source_window = _window(source_plan, int(window["window_index"]))
    if source_window != window:
        raise RuntimeError("Inference-reuse window identity differs.")
    record = reuse["window_results"].get(str(window["window_id"]))
    if not isinstance(record, Mapping):
        raise RuntimeError("Inference-reuse window receipt is absent.")
    bundle = Path(str(record["bundle_path"])).resolve()
    expected_bundle = (
        Path(source_plan["run_root"])
        / "bundles"
        / "inference"
        / str(source_window["window_id"])
    ).resolve()
    if bundle != expected_bundle:
        raise RuntimeError("Inference-reuse bundle path differs from its source plan.")
    if _sha256_file(bundle / "result.json") != record["result_sha256"]:
        raise RuntimeError("Inference-reuse result digest differs.")
    result = _existing_worker_result(
        bundle=bundle,
        plan=source_plan,
        window=source_window,
        stage="inference",
    )
    if (
        result is None
        or result["final_layout_unit_package"]["payload_digest"]
        != record["final_layout_payload_digest"]
    ):
        raise RuntimeError("Inference-reuse final-layout receipt differs.")
    return bundle, result


def _publish_worker_bundle(
    *,
    local_archive: Path,
    parent: str,
    run_name: str,
    bundle: Path,
    result: dict[str, Any],
    final_layout_unit_package: Path | None = None,
    sampled_contour_receipt: Path | None = None,
    gpu_runtime_telemetry_path: Path | None = None,
) -> dict[str, Any]:
    destination = bundle.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        archive = temporary / "archive.zarr"
        root = zarr.open_group(str(archive), mode="w", zarr_format=3)
        root.attrs.update(
            {
                "benchmark_classification": BENCHMARK_CLASSIFICATION,
                "stage_selector_eligible": False,
            }
        )
        root.create_group(parent)
        started = time.perf_counter()
        shutil.copytree(
            local_archive / parent / run_name,
            archive / parent / run_name,
            copy_function=shutil.copy2,
        )
        copied = open_zarr_root(archive, mode="r")[f"{parent}/{run_name}"]
        if copied.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            raise RuntimeError("Copied worker run is not complete.")
        if copied.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError("Copied worker run is unexpectedly selector eligible.")
        proof = _worker_evidence(archive, copied)
        final_layout_binding = None
        if final_layout_unit_package is not None:
            package_destination = temporary / "final_layout_unit"
            shutil.copytree(
                final_layout_unit_package.expanduser().resolve(),
                package_destination,
                copy_function=shutil.copy2,
            )
            package_receipt = validate_subject_mask_final_layout_unit_package(
                package_destination
            )
            if package_receipt.get("schema_version") != (
                SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION
            ):
                raise RuntimeError(
                    "Canary worker requires receipt-bound composable final-layout "
                    "evidence."
                )
            final_layout_binding = {
                "relative_path": "final_layout_unit",
                "schema_version": package_receipt["schema_version"],
                "payload_digest": package_receipt["payload_digest"],
                "kind": package_receipt["payload"]["kind"],
                "array_path": package_receipt["payload"]["array_path"],
                "storage_plan_digest": package_receipt["payload"][
                    "storage_plan_digest"
                ],
                "complete_unit_count": package_receipt["payload"][
                    "complete_unit_count"
                ],
                "encoded_object_count": package_receipt["payload"][
                    "encoded_object_count"
                ],
                "encoded_bytes": package_receipt["payload"]["encoded_bytes"],
                "worker_receipt_payload_digest": package_receipt["payload"][
                    "worker_receipt_payload_digest"
                ],
                "source_array_validation_digest": canonical_json_sha256(
                    package_receipt["payload"]["source_array_validation"]
                ),
            }
            if result.get("stage") == "inference" and (
                result.get("probability_destination_validation_handoff") is not None
            ):
                _validate_probability_destination_validation_handoff(
                    result["probability_destination_validation_handoff"],
                    worker_receipt_payload_digest=str(
                        proof["receipt"]["payload_digest"]
                    ),
                    final_layout_receipt=package_receipt,
                )
        sampled_contour_binding = None
        if sampled_contour_receipt is not None:
            receipt_destination = temporary / "sampled_contour_receipt.json"
            shutil.copy2(
                sampled_contour_receipt.expanduser().resolve(), receipt_destination
            )
            sampled_receipt = load_subject_mask_sampled_contour_worker_receipt(
                receipt_destination
            )
            sampled_receipt = validate_subject_mask_sampled_contour_worker_receipt(
                sampled_receipt,
                run=copied,
                worker_receipt=proof["receipt"],
            )
            sampled_payload = sampled_receipt["payload"]
            sampled_contour_binding = {
                "relative_path": "sampled_contour_receipt.json",
                "payload_digest": sampled_receipt["payload_digest"],
                "source_run_path": sampled_payload["source_run_path"],
                "global_row_interval": sampled_payload["global_row_interval"],
                "array_document_digest": sampled_payload["array_document_digest"],
                "source_worker_receipt_payload_digest": sampled_payload[
                    "source_dense_worker"
                ]["worker_receipt_payload_digest"],
                "producer_commit": sampled_payload["producer_commit"],
            }
        if gpu_runtime_telemetry_path is not None:
            telemetry_directory = temporary / "performance"
            try:
                telemetry_source = gpu_runtime_telemetry_path.expanduser().resolve()
                telemetry_document = _strict_json(telemetry_source)
                if not isinstance(telemetry_document, Mapping):
                    raise ValueError("GPU runtime telemetry sidecar is not an object.")
                require_gpu_runtime_telemetry(telemetry_document)
                telemetry_destination = telemetry_directory / "gpu_runtime.json"
                telemetry_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(telemetry_source, telemetry_destination)
                copied_telemetry = _strict_json(telemetry_destination)
                if not isinstance(copied_telemetry, Mapping):
                    raise ValueError("Copied GPU runtime telemetry is not an object.")
                require_gpu_runtime_telemetry(copied_telemetry)
                source_sha256 = _sha256_file(telemetry_source)
                copied_sha256 = _sha256_file(telemetry_destination)
                if copied_sha256 != source_sha256:
                    raise RuntimeError(
                        "GPU runtime telemetry changed during bundle copy."
                    )
                result["performance_telemetry"] = {
                    "identity_policy": GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
                    "scientific_identity_included": False,
                    "gpu_runtime": {
                        "status": "captured",
                        "capture_status": copied_telemetry["status"],
                        "relative_path": "performance/gpu_runtime.json",
                        "schema_id": GPU_RUNTIME_TELEMETRY_SCHEMA_ID,
                        "schema_version": GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION,
                        "payload_digest": copied_telemetry["payload_digest"],
                        "file_sha256": copied_sha256,
                        "size_bytes": telemetry_destination.stat().st_size,
                    },
                }
            except Exception as exc:
                shutil.rmtree(telemetry_directory, ignore_errors=True)
                result["performance_telemetry"] = {
                    "identity_policy": GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
                    "scientific_identity_included": False,
                    "gpu_runtime": {
                        "status": "missing",
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                }
        result.update(
            {
                "bundle_path": str(destination),
                "run_path": f"{parent}/{run_name}",
                "copy_duration_seconds": float(time.perf_counter() - started),
                "bundle_tree": _tree_stats(temporary),
                "proof": {
                    "scientific_identity_digest": proof["scientific_identity"][
                        "digest"
                    ],
                    "attempt_payload_digest": proof["attempt"]["payload_digest"],
                    "receipt_payload_digest": proof["receipt"]["payload_digest"],
                },
                "final_layout_unit_package": final_layout_binding,
                "sampled_contour_worker_receipt": sampled_contour_binding,
            }
        )
        _write_json_atomic(temporary / "result.json", result)
        os.replace(temporary, destination)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_inference_window(
    *,
    plan_path: Path,
    window_index: int,
    scratch_root: Path,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    window = _window(plan, window_index)
    run_root = Path(plan["run_root"])
    if plan.get("inference_reuse") is not None:
        raise RuntimeError("Inference-reuse plans do not authorize inference workers.")
    bundle = run_root / "bundles" / "inference" / str(window["window_id"])
    existing = _existing_worker_result(
        bundle=bundle, plan=plan, window=window, stage="inference"
    )
    if existing is not None:
        return existing
    started = time.perf_counter()
    work = scratch_root.expanduser().resolve() / f"inference_{window['window_id']}"
    work.mkdir(parents=True, exist_ok=False)
    telemetry_sampler: GpuRuntimeTelemetrySampler | None = None
    telemetry_path: Path | None = None
    telemetry_stopped = False
    telemetry_start_error: str | None = None
    telemetry_config = (
        plan.get("execution", {}).get("inference", {}).get("gpu_runtime_telemetry")
    )
    if (
        isinstance(telemetry_config, Mapping)
        and telemetry_config.get("enabled") is True
    ):
        telemetry_path = work / "performance" / "gpu_runtime.json"
        try:
            telemetry_sampler = GpuRuntimeTelemetrySampler(
                output_path=telemetry_path,
                sample_interval_seconds=int(
                    telemetry_config["sample_interval_seconds"]
                ),
                execution_context={
                    "workflow_id": plan["workflow_id"],
                    "plan_digest": plan["plan_digest"],
                    "palette_commit": plan["repo"]["commit"],
                    "stage": "subject_mask_inference",
                    "window_id": window["window_id"],
                    "window_index": int(window["window_index"]),
                    "row_start": int(window["row_start"]),
                    "row_stop": int(window["row_stop"]),
                },
            ).start()
        except Exception as exc:
            telemetry_start_error = f"{type(exc).__name__}: {exc}"
            telemetry_sampler = None
            telemetry_path = None
    try:
        performance_phase_durations: dict[str, float] = {}
        phase_started = time.perf_counter()
        local_archive = _stage_reference_archive(plan, work / "draft.zarr")
        performance_phase_durations["reference_archive_stage"] = float(
            time.perf_counter() - phase_started
        )
        staged_video = work / Path(window["source_video_path"]).name
        phase_started = time.perf_counter()
        video_copy = _copy_file_with_digest(
            Path(window["source_video_path"]), staged_video
        )
        performance_phase_durations["video_copy"] = float(
            time.perf_counter() - phase_started
        )
        if not _same_cluster_file_identity(video_copy["source"], window["source_file"]):
            raise RuntimeError("Window source-video identity changed after planning.")
        staged_model = work / Path(plan["model"]["path"]).name
        phase_started = time.perf_counter()
        model_copy = _copy_file_with_digest(Path(plan["model"]["path"]), staged_model)
        performance_phase_durations["model_copy"] = float(
            time.perf_counter() - phase_started
        )
        if model_copy["sha256"] != plan["model"]["sha256"]:
            raise RuntimeError("Staged model digest differs from the frozen plan.")

        root = open_zarr_root(local_archive, mode="r+")
        crop_run = str(plan["references"]["crop"]["run"])
        scratch_video_binding = _bind_scratch_window_video(
            root, crop_run=crop_run, window=window
        )
        source = CropImageSource.open(
            root,
            crop_run=crop_run,
            zarr_path=local_archive,
            roi_cache_policy="never",
            roi_live_acceleration="gpu",
            source_video_path_override=staged_video,
            source_video_frame_offset=int(window["start_frame"]),
            source_video_frame_count=int(window["frame_count"]),
        )
        package_manifest = work / "crop_pixel_work_package.json"
        phase_started = time.perf_counter()
        try:
            package = build_crop_pixel_work_package_from_video_window(
                source,
                target_crop_rows=np.arange(
                    int(window["row_start"]),
                    int(window["row_stop"]),
                    dtype=np.int64,
                ),
                video_path=staged_video,
                source_video_frame_offset=int(window["start_frame"]),
                source_video_frame_count=int(window["frame_count"]),
                frame_window_binding={
                    "schema_id": "palette.acquisition_video_frame_window",
                    "schema_version": 1,
                    "recording_identity": plan["recording"]["recording_id"],
                    "camera_identity": window["camera_identity"],
                    "clip_id": window["window_id"],
                    "actual_start_frame": int(window["start_frame"]),
                    "end_frame_exclusive": int(window["end_frame"]),
                    "frame_count": int(window["frame_count"]),
                    "clip_index_document_sha256": plan["recording"]["video_source"][
                        "window_index_sha256"
                    ],
                    "clip_video_sha256": video_copy["sha256"],
                },
                manifest_path=package_manifest,
                archive_path=local_archive,
                batch_rows=1024,
            )
        finally:
            source.close()
        performance_phase_durations["crop_pixel_materialization"] = float(
            time.perf_counter() - phase_started
        )
        execution = plan["execution"]["inference"]
        arguments = [
            str(local_archive),
            "--checkpoint",
            str(staged_model),
            "--run-name",
            str(window["raw_run"]),
            "--attempt-id",
            str(window["raw_attempt_id"]),
            "--output-parent",
            "subject_mask_shard_runs",
            "--crop-run",
            crop_run,
            "--source-collection-id",
            str(plan["workflow_id"]),
            "--source-collection-path",
            str(plan_path.expanduser().resolve()),
            "--source-clip-id",
            str(window["window_id"]),
            "--source-clip-index",
            str(window["window_index"]),
            "--source-work-unit-id",
            f"{plan['workflow_id']}:{window['window_id']}",
            "--source-shard-id",
            str(window["window_id"]),
            "--roi-work-package-manifest",
            str(package_manifest),
            "--roi-work-package-role",
            "complete_collection_partition",
            "--roi-cache-policy",
            "never",
            "--batch-size",
            str(execution["batch_size"]),
            "--device",
            str(execution["device"]),
            "--mask-probs-dtype",
            str(execution["probability_dtype"]),
            "--mask-probs-chunk-rois",
            str(execution["inner_chunk_rows"]),
            "--mask-probs-shard-rois",
            str(execution["outer_shard_rows"]),
            "--mask-probs-destination-validation",
            str(
                execution.get(
                    "destination_validation_mode",
                    infer_unet_subject_masks.MASK_PROBS_DESTINATION_VALIDATION_FULL,
                )
            ),
            "--no-write-masks-roi",
            "--async-output",
            "--output-queue-size",
            "2",
            "--no-progress",
            "--defer-registry-status",
        ]
        if bool(execution.get("synchronized_stage_profiling", False)):
            arguments.append("--profile-timings")
        phase_started = time.perf_counter()
        infer_unet_subject_masks.main(arguments)
        performance_phase_durations["inference_cli"] = float(
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        run = open_zarr_root(local_archive, mode="r")[
            f"subject_mask_shard_runs/{window['raw_run']}"
        ]
        shard_write = run.attrs.get("mask_probs_shard_write")
        if (
            execution.get("destination_validation_mode")
            == infer_unet_subject_masks.MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT
            and (
                not isinstance(shard_write, Mapping)
                or shard_write.get("destination_validation_status")
                != "deferred_to_mandatory_final_layout_unit"
            )
        ):
            raise RuntimeError(
                "Inference did not preserve the planned final-layout validation handoff."
            )
        proof = _worker_evidence(local_archive, run)
        performance_phase_durations["local_proof"] = float(
            time.perf_counter() - phase_started
        )
        final_layout_package_path = work / "raw_final_layout_unit"
        final_layout_started = time.perf_counter()
        final_layout_package = build_subject_mask_final_layout_unit_package(
            source_array=run["mask_probs_roi"],
            source_crop_row_ids=run["source_crop_row_ids"],
            destination=final_layout_package_path,
            kind="raw_probability_uint8",
            dimensions=_final_layout_dimensions(plan, stage="inference"),
            global_start_row=int(window["row_start"]),
            source_run_path=str(proof["receipt"]["payload"]["run_path"]),
            worker_receipt_payload_digest=str(proof["receipt"]["payload_digest"]),
            producer_commit=str(plan["repo"]["commit"]),
            worker_array_validation_record=proof["receipt"]["payload"]["arrays"][
                "mask_probs_roi"
            ],
        )
        performance_phase_durations["final_layout_unit"] = float(
            time.perf_counter() - final_layout_started
        )
        worker_pre_bundle_seconds = float(time.perf_counter() - started)
        result = {
            "schema_id": WORKER_RESULT_SCHEMA_ID,
            "schema_version": WORKER_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "stage": "inference",
            "finished_at_utc": _utc_now(),
            "plan_digest": plan["plan_digest"],
            "palette_commit": plan["repo"]["commit"],
            "window_id": window["window_id"],
            "window_index": int(window["window_index"]),
            "row_start": int(window["row_start"]),
            "row_stop": int(window["row_stop"]),
            "row_count": int(window["row_count"]),
            "video_copy": video_copy,
            "scratch_video_binding": scratch_video_binding,
            "model_copy": model_copy,
            "work_package": package,
            "local_proof": {
                "scientific_identity_digest": proof["scientific_identity"]["digest"],
                "attempt_payload_digest": proof["attempt"]["payload_digest"],
                "receipt_payload_digest": proof["receipt"]["payload_digest"],
            },
            "compute_duration_seconds": worker_pre_bundle_seconds,
            "performance_phase_durations_seconds": {
                **performance_phase_durations,
                "worker_pre_bundle_total": worker_pre_bundle_seconds,
            },
            "resource_usage": _resource_usage(),
            "local_final_layout_unit_payload_digest": final_layout_package[
                "payload_digest"
            ],
            "final_layout_unit_duration_seconds": performance_phase_durations[
                "final_layout_unit"
            ],
        }
        if (
            execution.get("destination_validation_mode")
            == infer_unet_subject_masks.MASK_PROBS_DESTINATION_VALIDATION_FINAL_LAYOUT
        ):
            assert isinstance(shard_write, Mapping)
            result["probability_destination_validation_handoff"] = {
                "schema_id": (
                    PROBABILITY_DESTINATION_VALIDATION_HANDOFF_SCHEMA_ID
                ),
                "schema_version": (
                    PROBABILITY_DESTINATION_VALIDATION_HANDOFF_SCHEMA_VERSION
                ),
                "status": "complete",
                "writer_mode": shard_write.get("destination_validation_mode"),
                "writer_status": shard_write.get("destination_validation_status"),
                "worker_receipt_payload_digest": proof["receipt"]["payload_digest"],
                "final_layout_payload_digest": final_layout_package[
                    "payload_digest"
                ],
                "final_layout_source_array_validation_digest": (
                    canonical_json_sha256(
                        final_layout_package["payload"]["source_array_validation"]
                    )
                ),
                "publication_requirement": (
                    "immutable_worker_bundle_requires_verified_final_layout_unit_v1"
                ),
            }
        telemetry_error = telemetry_start_error
        if telemetry_sampler is not None:
            try:
                telemetry_sampler.stop(workload_outcome="success")
            except Exception as exc:  # telemetry cannot invalidate scientific output
                telemetry_error = f"{type(exc).__name__}: {exc}"
                telemetry_path = None
            finally:
                telemetry_stopped = True
        if telemetry_error is not None:
            result["performance_telemetry"] = {
                "identity_policy": GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
                "scientific_identity_included": False,
                "gpu_runtime": {
                    "status": "missing",
                    "reason": telemetry_error,
                },
            }
        return _publish_worker_bundle(
            local_archive=local_archive,
            parent="subject_mask_shard_runs",
            run_name=str(window["raw_run"]),
            bundle=bundle,
            result=result,
            final_layout_unit_package=final_layout_package_path,
            gpu_runtime_telemetry_path=telemetry_path,
        )
    except Exception as exc:
        if telemetry_sampler is not None and not telemetry_stopped:
            try:
                telemetry_sampler.stop(
                    workload_outcome="error",
                    workload_error_type=type(exc).__name__,
                )
            except Exception:
                pass
            telemetry_stopped = True
        raise
    finally:
        if not keep_scratch:
            shutil.rmtree(work, ignore_errors=True)


def run_refinement_window(
    *,
    plan_path: Path,
    window_index: int,
    scratch_root: Path,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    window = _window(plan, window_index)
    run_root = Path(plan["run_root"])
    inference_bundle, _ = _resolve_inference_bundle(
        plan=plan,
        window=window,
    )
    bundle = run_root / "bundles" / "refinement" / str(window["window_id"])
    existing = _existing_worker_result(
        bundle=bundle, plan=plan, window=window, stage="refinement"
    )
    if existing is not None:
        return existing
    started = time.perf_counter()
    work = scratch_root.expanduser().resolve() / f"refinement_{window['window_id']}"
    work.mkdir(parents=True, exist_ok=False)
    try:
        local_archive = _stage_reference_archive(plan, work / "draft.zarr")
        local_root = open_zarr_root(local_archive, mode="r+")
        require_runs_parent(
            local_root,
            "subject_mask_shard_runs",
            completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
        )
        shutil.copytree(
            inference_bundle
            / "archive.zarr"
            / "subject_mask_shard_runs"
            / str(window["raw_run"]),
            local_archive / "subject_mask_shard_runs" / str(window["raw_run"]),
            copy_function=shutil.copy2,
        )
        execution = plan["execution"]["refinement"]
        summary = finalize_subject_mask_run(
            local_root,
            zarr_path=local_archive,
            subject_shard_runs=[str(window["raw_run"])],
            target_crop_run=str(plan["references"]["crop"]["run"]),
            refined_run=str(window["refined_run"]),
            components=["subject_body", "eyes_union", "swim_bladder"],
            chunk_size=int(execution["chunk_rows"]),
            metric_level=str(execution["metric_level"]),
            write_eye_geometry=True,
            write_component_contours=False,
            write_sampled_component_contours=True,
            retain_source_seeds=False,
            mask_storage="dense_uint8",
            mask_rle_validation_mode="invariants",
            dense_mask_row_chunk=int(execution["dense_mask_chunk_rows"]),
            postcompute_backend="process_shards",
            postcompute_chunk_size=int(execution["chunk_rows"]),
            postcompute_num_workers=int(execution["workers"]),
            execution_backend="process_shards",
            num_workers=int(execution["workers"]),
            assignment_keypoint_group="refined_keypoints_runs",
            assignment_keypoints_run=str(
                plan["references"]["refined_keypoints"]["run"]
            ),
            attempt_id=str(window["refined_attempt_id"]),
            require_production_proof=True,
        )
        run = open_zarr_root(local_archive, mode="r")[
            f"refined_subject_masks_runs/{window['refined_run']}"
        ]
        proof = _worker_evidence(local_archive, run)
        final_layout_package_path = work / "refined_final_layout_unit"
        final_layout_started = time.perf_counter()
        final_layout_package = build_subject_mask_final_layout_unit_package(
            source_array=run["masks_roi"],
            source_crop_row_ids=run["source_crop_row_ids"],
            destination=final_layout_package_path,
            kind="refined_dense_core",
            dimensions=_final_layout_dimensions(plan, stage="refinement"),
            global_start_row=int(window["row_start"]),
            source_run_path=str(proof["receipt"]["payload"]["run_path"]),
            worker_receipt_payload_digest=str(proof["receipt"]["payload_digest"]),
            producer_commit=str(plan["repo"]["commit"]),
            worker_array_validation_record=proof["receipt"]["payload"]["arrays"][
                "masks_roi"
            ],
        )
        sampled_contour_receipt_path = work / "sampled_contour_receipt.json"
        sampled_contour_started = time.perf_counter()
        sampled_contour_receipt = write_subject_mask_sampled_contour_worker_receipt(
            run,
            destination=sampled_contour_receipt_path,
            global_start_row=int(window["row_start"]),
            worker_receipt=proof["receipt"],
            producer_commit=str(plan["repo"]["commit"]),
        )
        result = {
            "schema_id": WORKER_RESULT_SCHEMA_ID,
            "schema_version": WORKER_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "stage": "refinement",
            "finished_at_utc": _utc_now(),
            "plan_digest": plan["plan_digest"],
            "palette_commit": plan["repo"]["commit"],
            "window_id": window["window_id"],
            "window_index": int(window["window_index"]),
            "row_start": int(window["row_start"]),
            "row_stop": int(window["row_stop"]),
            "row_count": int(window["row_count"]),
            "summary": summary,
            "local_proof": {
                "scientific_identity_digest": proof["scientific_identity"]["digest"],
                "attempt_payload_digest": proof["attempt"]["payload_digest"],
                "receipt_payload_digest": proof["receipt"]["payload_digest"],
            },
            "compute_duration_seconds": float(time.perf_counter() - started),
            "resource_usage": _resource_usage(),
            "local_final_layout_unit_payload_digest": final_layout_package[
                "payload_digest"
            ],
            "final_layout_unit_duration_seconds": float(
                time.perf_counter() - final_layout_started
            ),
            "local_sampled_contour_payload_digest": sampled_contour_receipt[
                "payload_digest"
            ],
            "sampled_contour_receipt_duration_seconds": float(
                time.perf_counter() - sampled_contour_started
            ),
        }
        return _publish_worker_bundle(
            local_archive=local_archive,
            parent="refined_subject_masks_runs",
            run_name=str(window["refined_run"]),
            bundle=bundle,
            result=result,
            final_layout_unit_package=final_layout_package_path,
            sampled_contour_receipt=sampled_contour_receipt_path,
        )
    finally:
        if not keep_scratch:
            shutil.rmtree(work, ignore_errors=True)


def _symlink_run(*, source: Path, archive: Path, parent: str, run_name: str) -> None:
    root = open_zarr_root(archive, mode="r+")
    if parent not in root:
        root.create_group(parent)
    destination = archive / parent / run_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.symlink(source.resolve(), destination, target_is_directory=True)


def _assemble_draft(
    plan: Mapping[str, Any], destination: Path
) -> tuple[list[str], list[str]]:
    archive = destination.expanduser().resolve()
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": plan["recording"]["recording_id"],
            "benchmark_classification": BENCHMARK_CLASSIFICATION,
        }
    )
    reference_archive = Path(plan["references"]["analysis_zarr"])
    for name in ("crop", "refined_keypoints"):
        reference = plan["references"][name]
        _symlink_run(
            source=reference_archive / reference["parent"] / reference["run"],
            archive=archive,
            parent=str(reference["parent"]),
            run_name=str(reference["run"]),
        )
    raw_runs: list[str] = []
    refined_runs: list[str] = []
    for window in plan["windows"]:
        if int(window["row_count"]) == 0:
            continue
        window_id = str(window["window_id"])
        inference_bundle, _ = _resolve_inference_bundle(plan=plan, window=window)
        refinement_bundle = (
            Path(plan["run_root"]) / "bundles" / "refinement" / window_id
        )
        if (
            _existing_worker_result(
                bundle=refinement_bundle,
                plan=plan,
                window=window,
                stage="refinement",
            )
            is None
        ):
            raise RuntimeError(f"Window {window_id} lacks terminal worker evidence.")
        raw_name = str(window["raw_run"])
        refined_name = str(window["refined_run"])
        _symlink_run(
            source=inference_bundle
            / "archive.zarr"
            / "subject_mask_shard_runs"
            / raw_name,
            archive=archive,
            parent="subject_mask_shard_runs",
            run_name=raw_name,
        )
        _symlink_run(
            source=refinement_bundle
            / "archive.zarr"
            / "refined_subject_masks_runs"
            / refined_name,
            archive=archive,
            parent="refined_subject_masks_runs",
            run_name=refined_name,
        )
        raw_runs.append(raw_name)
        refined_runs.append(refined_name)
    return raw_runs, refined_runs


def finalize_canary(
    *,
    plan_path: Path,
    scratch_root: Path,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    publication_repo = _repo_identity(Path(__file__).resolve().parents[4])
    result_path = Path(plan["outputs"]["result_path"])
    if result_path.exists():
        result = _strict_json(result_path)
        if (
            isinstance(result, dict)
            and result.get("schema_id") == RESULT_SCHEMA_ID
            and result.get("status") == "complete"
            and result.get("plan_digest") == plan["plan_digest"]
        ):
            return result
        raise RuntimeError(
            "Existing canary result is malformed or belongs to another plan."
        )
    started = time.perf_counter()
    work = scratch_root.expanduser().resolve() / "publication"
    work.mkdir(parents=True, exist_ok=False)
    try:
        assembly = work / "assembly.zarr"
        raw_runs, refined_runs = _assemble_draft(plan, assembly)
        nonempty_windows = [
            window for window in plan["windows"] if int(window["row_count"]) > 0
        ]
        nonempty_window_ids = [
            str(window["window_id"]) for window in nonempty_windows
        ]
        inference_bundles = {
            str(window["window_id"]): _resolve_inference_bundle(
                plan=plan,
                window=window,
            )[0]
            for window in nonempty_windows
        }
        raw_final_layout_packages = [
            inference_bundles[window_id] / "final_layout_unit"
            for window_id in nonempty_window_ids
        ]
        refined_final_layout_packages = [
            Path(plan["run_root"])
            / "bundles"
            / "refinement"
            / window_id
            / "final_layout_unit"
            for window_id in nonempty_window_ids
        ]
        sampled_contour_worker_receipts = [
            Path(plan["run_root"])
            / "bundles"
            / "refinement"
            / window_id
            / "sampled_contour_receipt.json"
            for window_id in nonempty_window_ids
        ]
        output = plan["outputs"]
        publication = publish_recording_subject_mask_bundle(
            analysis_zarr=Path(plan["references"]["analysis_zarr"]),
            draft_zarr=assembly,
            crop_run=str(plan["references"]["crop"]["run"]),
            raw_draft_parent="subject_mask_shard_runs",
            raw_draft_run=raw_runs[0],
            raw_draft_runs=raw_runs,
            refined_draft_run=refined_runs[0],
            refined_draft_runs=refined_runs,
            raw_run=str(output["raw_run"]),
            refined_run=str(output["refined_run"]),
            quality_run=str(output["quality_run"]),
            cache_run=str(output["cache_run"]),
            bundle_id=str(output["bundle_id"]),
            local_output_root=work / "snapshots",
            quality_scratch_root=work / "quality_scratch",
            quality_compute_workers=int(
                plan["execution"]["publication"].get(
                    "quality_compute_workers", 1
                )
            ),
            activate=False,
            expected_work_units=[
                {
                    "work_unit_id": (f"{plan['workflow_id']}:{window['window_id']}"),
                    "work_unit_index": int(window["window_index"]),
                    "source_clip_id": str(window["window_id"]),
                    "source_clip_index": int(window["window_index"]),
                    "frame_start": int(window["start_frame"]),
                    "frame_stop": int(window["end_frame"]),
                    "row_start": int(window["row_start"]),
                    "row_stop": int(window["row_stop"]),
                }
                for window in plan["windows"]
            ],
            core_physical_unit_workers=int(
                plan["execution"]["publication"]["core_physical_unit_workers"]
            ),
            core_validation_mode=str(
                plan["execution"]["publication"]["core_validation_mode"]
            ),
            raw_final_layout_unit_packages=raw_final_layout_packages,
            refined_final_layout_unit_packages=refined_final_layout_packages,
            require_complete_final_layout_units=True,
            sampled_contour_worker_receipts=sampled_contour_worker_receipts,
            require_worker_sampled_contours=True,
            sampled_contour_producer_commit=str(plan["repo"]["commit"]),
        )
        target = open_zarr_root(Path(plan["references"]["analysis_zarr"]), mode="r")
        if "subject_mask_authority" in target.attrs:
            raise RuntimeError(
                "Inactive canary unexpectedly changed subject-mask authority."
            )
        expected_paths = {
            "subject_mask_runs": output["raw_run"],
            "refined_subject_masks_runs": output["refined_run"],
            "subject_mask_quality_runs": output["quality_run"],
            "subject_mask_cache_runs": output["cache_run"],
            "subject_mask_bundle_runs": output["bundle_id"],
        }
        for parent, run_name in expected_paths.items():
            run = target[f"{parent}/{run_name}"]
            if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                raise RuntimeError(
                    f"Published canary member is incomplete: {parent}/{run_name}"
                )
            if run.attrs.get("stage_selector_eligible") is not False:
                raise RuntimeError(
                    f"Published canary member is selector eligible: {parent}/{run_name}"
                )
        worker_results = {
            "inference": [
                _resolve_inference_bundle(plan=plan, window=window)[1]
                for window in nonempty_windows
            ],
            "refinement": [
                _strict_json(
                    Path(plan["run_root"])
                    / "bundles"
                    / "refinement"
                    / str(window["window_id"])
                    / "result.json"
                )
                for window in nonempty_windows
            ],
        }
        result = {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "classification": BENCHMARK_CLASSIFICATION,
            "finished_at_utc": _utc_now(),
            "plan_digest": plan["plan_digest"],
            "palette_commit": publication_repo["commit"],
            "worker_palette_commit": plan["repo"]["commit"],
            "inference_reuse": plan.get("inference_reuse"),
            "publication_repo": publication_repo,
            "recording": plan["recording"],
            "publication": publication,
            "worker_result_digests": {
                stage: [canonical_json_sha256(value) for value in values]
                for stage, values in worker_results.items()
            },
            "analysis_tree": _tree_stats(Path(plan["references"]["analysis_zarr"])),
            "duration_seconds": float(time.perf_counter() - started),
            "resource_usage": _resource_usage(),
            "safety": {
                "subject_mask_authority_present": False,
                "registry_mutated": False,
                "production_paths_written": False,
                "all_bundle_members_selector_ineligible": True,
                "exact_window_row_coverage": True,
                "worker_sampled_contours_assembled": (
                    publication["publication_execution"]["sampled_contour_source_mode"]
                    == "receipt_bound_worker_arrays"
                ),
                "full_ragged_contours_published": False,
            },
        }
        result["payload_digest"] = canonical_json_sha256(result)
        _write_json_atomic(result_path, result)
        return result
    finally:
        if not keep_scratch:
            shutil.rmtree(work, ignore_errors=True)


def build_lsf_workflow(
    *,
    plan_path: Path,
    gpu_concurrency: int = DEFAULT_GPU_CONCURRENCY,
    cpu_concurrency: int = DEFAULT_CPU_CONCURRENCY,
) -> LsfWorkflow:
    plan = load_plan(plan_path)
    repo = Path(plan["repo"]["path"])
    run_root = Path(plan["run_root"])
    inference_reused = plan.get("inference_reuse") is not None
    nonempty = [window for window in plan["windows"] if int(window["row_count"]) > 0]
    scratch_template = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}_"
        f"{RUNTIME_JOB_INDEX_TOKEN}/subject_mask_full_duration"
    )
    inference_tasks = []
    refinement_tasks = []
    for window in nonempty:
        index = int(window["window_index"])
        window_id = str(window["window_id"])
        inference_tasks.append(
            build_execution_task(
                run_root=run_root,
                task_key=f"inference:{window_id}",
                stage="subject_mask_inference",
                command=(
                    "scripts/py",
                    "-m",
                    "fisheye.cluster.subject_masks.full_duration_canary",
                    "inference-worker",
                    "--plan",
                    str(plan_path),
                    "--window-index",
                    str(index),
                    "--scratch-root",
                    scratch_template,
                ),
                expected_outputs=(
                    run_root / "bundles" / "inference" / window_id / "result.json",
                ),
                cleanup_paths=(scratch_template,),
                array_indexed=True,
            )
        )
        refinement_tasks.append(
            build_execution_task(
                run_root=run_root,
                task_key=f"refinement:{window_id}",
                stage="subject_mask_refinement",
                command=(
                    "scripts/py",
                    "-m",
                    "fisheye.cluster.subject_masks.full_duration_canary",
                    "refinement-worker",
                    "--plan",
                    str(plan_path),
                    "--window-index",
                    str(index),
                    "--scratch-root",
                    scratch_template,
                ),
                expected_outputs=(
                    run_root / "bundles" / "refinement" / window_id / "result.json",
                ),
                cleanup_paths=(scratch_template,),
                array_indexed=True,
            )
        )
    inference = (
        None
        if inference_reused
        else build_task_group_job(
            workflow_id=str(plan["workflow_id"]),
            family=FAMILY,
            repo=repo,
            run_root=run_root,
            job_key="subject_mask_inference_array",
            stage="subject_mask_inference",
            tasks=inference_tasks,
            mode=LsfExecutionMode.ARRAY,
            max_concurrent=int(gpu_concurrency),
            resources=LsfResources(
                queue="gpu_l4", ncores=8, mem_gb=64, gpus=1, walltime="8:00"
            ),
        )
    )
    refinement = build_task_group_job(
        workflow_id=str(plan["workflow_id"]),
        family=FAMILY,
        repo=repo,
        run_root=run_root,
        job_key="subject_mask_refinement_array",
        stage="subject_mask_refinement",
        tasks=refinement_tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=int(cpu_concurrency),
        resources=LsfResources(queue="short", ncores=16, mem_gb=64, walltime="1:00"),
        upstream=() if inference_reused else ("subject_mask_inference_array",),
    )
    final_scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "subject_mask_full_duration"
    )
    publication = build_job(
        workflow_id=str(plan["workflow_id"]),
        family=FAMILY,
        repo=repo,
        run_root=run_root,
        job_key="subject_mask_recording_publication",
        stage="subject_mask_recording_publication",
        command=(
            "scripts/py",
            "-m",
            "fisheye.cluster.subject_masks.full_duration_canary",
            "finalize",
            "--plan",
            str(plan_path),
            "--scratch-root",
            final_scratch,
        ),
        resources=LsfResources(queue="local", ncores=16, mem_gb=128, walltime="8:00"),
        upstream=("subject_mask_refinement_array",),
        expected_outputs=(Path(plan["outputs"]["result_path"]),),
        cleanup_paths=(final_scratch,),
    )
    jobs = (
        (refinement, publication)
        if inference is None
        else (inference, refinement, publication)
    )
    return LsfWorkflow(
        workflow_id=str(plan["workflow_id"]),
        family=FAMILY,
        jobs=jobs,
        metadata={
            "classification": BENCHMARK_CLASSIFICATION,
            "plan_digest": plan["plan_digest"],
            "recording_layout": plan["recording"]["video_source"]["mode"],
            "window_count": len(plan["windows"]),
            "nonempty_window_count": len(nonempty),
            "gpu_concurrency": int(gpu_concurrency),
            "cpu_concurrency": int(cpu_concurrency),
            "inference_reused": bool(inference_reused),
            "production_state_changes": False,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-root", required=True, type=Path)
    prepare.add_argument("--repo", required=True, type=Path)
    prepare.add_argument("--source-crop-zarr", required=True, type=Path)
    prepare.add_argument("--crop-run", required=True)
    prepare.add_argument("--source-refined-keypoint-zarr", required=True, type=Path)
    prepare.add_argument("--refined-keypoint-run", required=True)
    prepare.add_argument("--model", required=True, type=Path)
    prepare.add_argument("--model-sha256", required=True)
    prepare.add_argument("--recording-id", required=True)
    prepare.add_argument("--recording-dir", type=Path)
    video = prepare.add_mutually_exclusive_group(required=True)
    video.add_argument("--clip-index", type=Path)
    video.add_argument("--whole-video", type=Path)
    prepare.add_argument("--camera-identity")
    prepare.add_argument("--run-label", required=True)
    prepare.add_argument(
        "--reuse-inference-plan",
        type=Path,
        help=(
            "Bind and reuse terminal inference bundles from an exact prior canary "
            "plan; the new workflow submits refinement and publication only."
        ),
    )
    prepare.add_argument(
        "--core-physical-unit-workers",
        type=int,
        default=4,
        help=(
            "Bounded final raw/refined physical row-band writers recorded in "
            "the immutable canary plan (default: 4)."
        ),
    )
    prepare.add_argument(
        "--quality-compute-workers",
        type=int,
        default=4,
        help=(
            "Bounded row-local subject-mask QC compute threads; source reads, "
            "hashing, and output writes remain ordered (default: 4)."
        ),
    )
    prepare.add_argument(
        "--gpu-telemetry-interval-seconds",
        type=int,
        default=DEFAULT_GPU_TELEMETRY_INTERVAL_SECONDS,
        help=(
            "Continuous nvidia-smi sampling interval for report-only inference "
            "sidecars (default: 1 second)."
        ),
    )
    prepare.add_argument(
        "--synchronized-stage-profiling",
        action="store_true",
        help=(
            "Opt into perturbative per-batch CUDA synchronization for exact phase "
            "attribution. Disabled for representative throughput by default."
        ),
    )
    prepare.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow explicitly non-reproducible development preflight only.",
    )

    for name in ("inference-worker", "refinement-worker"):
        worker = subparsers.add_parser(name)
        worker.add_argument("--plan", required=True, type=Path)
        worker.add_argument("--window-index", required=True, type=int)
        worker.add_argument("--scratch-root", required=True, type=Path)
        worker.add_argument("--keep-scratch", action="store_true")

    final = subparsers.add_parser("finalize")
    final.add_argument("--plan", required=True, type=Path)
    final.add_argument("--scratch-root", required=True, type=Path)
    final.add_argument("--keep-scratch", action="store_true")

    submit = subparsers.add_parser("submit")
    submit.add_argument("--plan", required=True, type=Path)
    submit.add_argument("--gpu-concurrency", type=int, default=DEFAULT_GPU_CONCURRENCY)
    submit.add_argument("--cpu-concurrency", type=int, default=DEFAULT_CPU_CONCURRENCY)
    submit.add_argument("--submit-host", default="login1-citrus-poller")
    mode = submit.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_canary(
            run_root=args.run_root,
            repo=args.repo,
            source_crop_zarr=args.source_crop_zarr,
            crop_run=args.crop_run,
            source_refined_keypoint_zarr=args.source_refined_keypoint_zarr,
            refined_keypoint_run=args.refined_keypoint_run,
            model_path=args.model,
            model_sha256=args.model_sha256,
            recording_id=args.recording_id,
            recording_dir=args.recording_dir,
            clip_index=args.clip_index,
            whole_video=args.whole_video,
            camera_identity=args.camera_identity,
            run_label=args.run_label,
            reuse_inference_plan=args.reuse_inference_plan,
            require_clean_repo=not bool(args.allow_dirty),
            core_physical_unit_workers=args.core_physical_unit_workers,
            quality_compute_workers=args.quality_compute_workers,
            gpu_telemetry_interval_seconds=args.gpu_telemetry_interval_seconds,
            synchronized_stage_profiling=bool(args.synchronized_stage_profiling),
        )
    elif args.command == "inference-worker":
        result = run_inference_window(
            plan_path=args.plan,
            window_index=args.window_index,
            scratch_root=args.scratch_root,
            keep_scratch=bool(args.keep_scratch),
        )
    elif args.command == "refinement-worker":
        result = run_refinement_window(
            plan_path=args.plan,
            window_index=args.window_index,
            scratch_root=args.scratch_root,
            keep_scratch=bool(args.keep_scratch),
        )
    elif args.command == "finalize":
        result = finalize_canary(
            plan_path=args.plan,
            scratch_root=args.scratch_root,
            keep_scratch=bool(args.keep_scratch),
        )
    else:
        workflow = build_lsf_workflow(
            plan_path=args.plan,
            gpu_concurrency=args.gpu_concurrency,
            cpu_concurrency=args.cpu_concurrency,
        )
        plan = load_plan(args.plan)
        run_root = Path(plan["run_root"])
        lsf_plan = run_root / "lsf_plan.json"
        if args.dry_run:
            result = write_json_snapshot(lsf_plan, workflow.to_json())
        else:
            result = submit_lsf_workflow(
                workflow,
                cwd=Path(plan["repo"]["path"]),
                plan_path=lsf_plan,
                submission_path=run_root / "lsf_submission.json",
                runner=build_ssh_bsub_runner(args.submit_host),
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_CLASSIFICATION",
    "FAMILY",
    "PLAN_SCHEMA_ID",
    "PLAN_SCHEMA_VERSION",
    "RESULT_SCHEMA_ID",
    "RESULT_SCHEMA_VERSION",
    "build_lsf_workflow",
    "finalize_canary",
    "load_plan",
    "main",
    "prepare_canary",
    "run_inference_window",
    "run_refinement_window",
]
