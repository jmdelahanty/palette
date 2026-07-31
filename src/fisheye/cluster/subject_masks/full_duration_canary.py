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
from fisheye.shared.zarr.crop_manifest import validate_crop_run_manifest
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr.subject_mask_schema import (
    derive_subject_mask_frame_row_offsets,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    require_runs_parent,
)

PLAN_SCHEMA_ID = "palette.subject_mask.full_duration_canary_plan"
PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.subject_mask.full_duration_canary_result"
RESULT_SCHEMA_VERSION = 1
WORKER_RESULT_SCHEMA_ID = "palette.subject_mask.full_duration_canary_worker"
WORKER_RESULT_SCHEMA_VERSION = 1
FAMILY = "subject_mask_full_duration_canary"
BENCHMARK_CLASSIFICATION = "selector_ineligible_full_duration_canary"
DEFAULT_GPU_CONCURRENCY = 4
DEFAULT_CPU_CONCURRENCY = 4
_ATTEMPT_NAMESPACE = UUID("79676a9f-24f1-4be9-ac50-c374b0fdccae")


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
    require_clean_repo: bool = True,
) -> dict[str, Any]:
    """Freeze inputs and copy exact maintained references into an isolated store."""

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
        "windows": windows,
        "outputs": {
            "raw_run": f"subject_masks_{label}",
            "refined_run": f"refined_subject_masks_{label}",
            "quality_run": f"subject_mask_quality_{label}",
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
            },
            "refinement": {
                "chunk_rows": 256,
                "dense_mask_chunk_rows": 256,
                "workers": 16,
                "metric_level": "cheap",
            },
        },
        "safety": {
            "production_registry_used": False,
            "production_selector_mutation_allowed": False,
            "bundle_activation_allowed": False,
            "all_outputs_below_run_root": True,
            "worker_writes_are_node_local_until_atomic_bundle_publish": True,
            "window_rows_are_exact_nonoverlapping_complete": True,
        },
    }
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
        or payload.get("schema_version") != PLAN_SCHEMA_VERSION
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
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Canary plan has no windows.")
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
        or result.get("status") != "complete"
        or result.get("stage") != stage
        or result.get("plan_digest") != plan["plan_digest"]
        or result.get("window_id") != window["window_id"]
    ):
        raise RuntimeError(f"Existing worker bundle identity differs: {bundle}")
    return result


def _publish_worker_bundle(
    *,
    local_archive: Path,
    parent: str,
    run_name: str,
    bundle: Path,
    result: dict[str, Any],
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
    bundle = run_root / "bundles" / "inference" / str(window["window_id"])
    existing = _existing_worker_result(
        bundle=bundle, plan=plan, window=window, stage="inference"
    )
    if existing is not None:
        return existing
    started = time.perf_counter()
    work = scratch_root.expanduser().resolve() / f"inference_{window['window_id']}"
    work.mkdir(parents=True, exist_ok=False)
    try:
        local_archive = _stage_reference_archive(plan, work / "draft.zarr")
        staged_video = work / Path(window["source_video_path"]).name
        video_copy = _copy_file_with_digest(
            Path(window["source_video_path"]), staged_video
        )
        if not _same_cluster_file_identity(video_copy["source"], window["source_file"]):
            raise RuntimeError("Window source-video identity changed after planning.")
        staged_model = work / Path(plan["model"]["path"]).name
        model_copy = _copy_file_with_digest(Path(plan["model"]["path"]), staged_model)
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
            "--no-write-masks-roi",
            "--async-output",
            "--output-queue-size",
            "2",
            "--no-progress",
            "--profile-timings",
            "--defer-registry-status",
        ]
        infer_unet_subject_masks.main(arguments)
        run = open_zarr_root(local_archive, mode="r")[
            f"subject_mask_shard_runs/{window['raw_run']}"
        ]
        proof = _worker_evidence(local_archive, run)
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
            "compute_duration_seconds": float(time.perf_counter() - started),
            "resource_usage": _resource_usage(),
        }
        return _publish_worker_bundle(
            local_archive=local_archive,
            parent="subject_mask_shard_runs",
            run_name=str(window["raw_run"]),
            bundle=bundle,
            result=result,
        )
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
    inference_bundle = run_root / "bundles" / "inference" / str(window["window_id"])
    inference_result = _existing_worker_result(
        bundle=inference_bundle, plan=plan, window=window, stage="inference"
    )
    if inference_result is None:
        raise RuntimeError("Inference bundle is missing.")
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
        }
        return _publish_worker_bundle(
            local_archive=local_archive,
            parent="refined_subject_masks_runs",
            run_name=str(window["refined_run"]),
            bundle=bundle,
            result=result,
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
        inference_bundle = Path(plan["run_root"]) / "bundles" / "inference" / window_id
        refinement_bundle = (
            Path(plan["run_root"]) / "bundles" / "refinement" / window_id
        )
        if (
            _existing_worker_result(
                bundle=inference_bundle,
                plan=plan,
                window=window,
                stage="inference",
            )
            is None
            or _existing_worker_result(
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
            bundle_id=str(output["bundle_id"]),
            local_output_root=work / "snapshots",
            quality_scratch_root=work / "quality_scratch",
            activate=False,
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
            stage: [
                _strict_json(
                    Path(plan["run_root"])
                    / "bundles"
                    / stage
                    / str(window["window_id"])
                    / "result.json"
                )
                for window in plan["windows"]
                if int(window["row_count"]) > 0
            ]
            for stage in ("inference", "refinement")
        }
        result = {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "classification": BENCHMARK_CLASSIFICATION,
            "finished_at_utc": _utc_now(),
            "plan_digest": plan["plan_digest"],
            "palette_commit": plan["repo"]["commit"],
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
    inference = build_task_group_job(
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
        upstream=("subject_mask_inference_array",),
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
    return LsfWorkflow(
        workflow_id=str(plan["workflow_id"]),
        family=FAMILY,
        jobs=(inference, refinement, publication),
        metadata={
            "classification": BENCHMARK_CLASSIFICATION,
            "plan_digest": plan["plan_digest"],
            "recording_layout": plan["recording"]["video_source"]["mode"],
            "window_count": len(plan["windows"]),
            "nonempty_window_count": len(nonempty),
            "gpu_concurrency": int(gpu_concurrency),
            "cpu_concurrency": int(cpu_concurrency),
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
            require_clean_repo=not bool(args.allow_dirty),
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
