"""Keypoint-family bindings and deterministic LSF job builders."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from fisheye.cluster.lsf import LsfDependency, LsfJob, LsfResources
from fisheye.cluster.lsf.runtime import build_runtime_command
from fisheye.registry.db import Registry
from fisheye.registry.model_resolution import (
    load_candidates,
    load_target_profile,
    resolve_recording_id,
)
from fisheye.shared.flat_roi_cache import (
    crop_run_name_from_manifest,
    load_flat_roi_cache_manifest,
)
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent


DEFAULT_ZEBRAFISH_MIN_ROI_SIZE = 348
DEFAULT_KEYPOINT_ROI_SHARD_ROWS = 65_536
DEFAULT_KEYPOINT_FRAME_SHARD_ROWS = 262_144


def resolve_keypoint_storage(
    *,
    roi_shard_rows: int | None,
    frame_shard_rows: int,
) -> dict[str, Any]:
    """Return requested/effective immutable keypoint storage provenance."""

    if roi_shard_rows is not None and int(roi_shard_rows) <= 0:
        raise ValueError("Keypoint ROI shard rows must be positive or None.")
    if int(frame_shard_rows) <= 0:
        raise ValueError("Keypoint frame shard rows must be positive.")
    enabled = roi_shard_rows is not None
    return {
        "requested": {
            "keypoint_roi_shard_rows": (
                int(roi_shard_rows) if roi_shard_rows is not None else None
            ),
            "keypoint_frame_shard_rows": int(frame_shard_rows),
            "no_keypoint_sharding": not enabled,
        },
        "effective": {
            "keypoint_storage_layout": (
                "indexed_sharding_v1" if enabled else "regular_chunks_v1"
            ),
            "keypoint_storage_policy": (
                "default_indexed_sharding_v1"
                if enabled
                else "explicit_regular_chunks_override"
            ),
            "keypoint_roi_shard_rows": (
                int(roi_shard_rows) if roi_shard_rows is not None else None
            ),
            "keypoint_frame_shard_rows": int(frame_shard_rows) if enabled else None,
        },
    }


@dataclass(frozen=True)
class KeypointRunNames:
    keypoint_run: str
    refined_keypoint_run: str

    def to_json(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PoseModelBinding:
    recording_id: str
    set_id: str
    run_id: str
    model_path: Path
    model_sha256: str
    score: float
    created_utc: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "model_path": str(self.model_path),
        }


@dataclass(frozen=True)
class FlatRoiCacheBinding:
    manifest_path: Path
    manifest_sha256: str
    payload_path: Path
    crop_run: str
    cache_key: str
    crop_signature: Any
    crop_revision: Any
    shape: tuple[int, int, int]
    total_bytes: int
    payload_sha256: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "manifest_path": str(self.manifest_path),
            "payload_path": str(self.payload_path),
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class KeypointInputCapability:
    selected_source: str
    min_roi_size: int
    crop_run: str
    crop_storage_mode: str
    crop_shape: tuple[int, int, int]
    persisted_roi_images_available: bool
    persisted_roi_images_eligible: bool
    acquisition_crop_video_available: bool
    acquisition_crop_video_eligible: bool
    acquisition_crop_video_path: Path | None
    acquisition_crop_video_shape: tuple[int, int] | None
    acquisition_crop_video_probe_status: str
    flat_roi_cache_eligible: bool
    rejected_sources: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "crop_shape": list(self.crop_shape),
            "acquisition_crop_video_path": (
                str(self.acquisition_crop_video_path)
                if self.acquisition_crop_video_path is not None
                else None
            ),
            "acquisition_crop_video_shape": (
                list(self.acquisition_crop_video_shape)
                if self.acquisition_crop_video_shape is not None
                else None
            ),
            "rejected_sources": dict(self.rejected_sources),
        }


def safe_component(value: object, *, default: str, max_length: int = 72) -> str:
    text = str(value or "").strip()
    normalized = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in text
    ).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized or default
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[: max_length - len(digest) - 1]}_{digest}"


def build_keypoint_run_names(run_label: str) -> KeypointRunNames:
    safe_label = safe_component(run_label, default="keypoints")
    return KeypointRunNames(
        keypoint_run=f"keypoints_{safe_label}",
        refined_keypoint_run=f"refined_keypoints_{safe_label}",
    )


def resolve_pose_model_binding(
    *,
    registry_path: Path,
    recording_id: str,
    recording_dir: Path,
    set_id: str,
    run_id: str,
) -> PoseModelBinding:
    """Resolve and validate one exact registered pose-model deployment."""

    resolved_registry = registry_path.expanduser().resolve()
    resolved_recording_dir = recording_dir.expanduser().resolve()
    registry = Registry(resolved_registry)
    try:
        recording_row = registry.conn.execute(
            "SELECT recording_path FROM recordings WHERE recording_id = ?;",
            (str(recording_id),),
        ).fetchone()
        if recording_row is None:
            raise ValueError(f"Recording id is not registered: {recording_id}")
        raw_recording_path = str(recording_row["recording_path"] or "").strip()
        if not raw_recording_path:
            raise ValueError(
                f"Recording id {recording_id!r} has no registered recording_path."
            )
        registered_path = Path(raw_recording_path).expanduser().resolve()
        if registered_path != resolved_recording_dir:
            raise ValueError(
                f"Recording path mismatch for {recording_id}: registry has "
                f"{registered_path}, manifest has {resolved_recording_dir}."
            )
        resolved_recording_id = resolve_recording_id(
            registry,
            recording_id=str(recording_id),
            recording_dir=resolved_recording_dir,
        )
        target = load_target_profile(registry, resolved_recording_id)
        candidates = load_candidates(
            registry,
            target=target,
            task="pose",
            set_id_filter=str(set_id),
            include_non_success=False,
        )
        exact = [candidate for candidate in candidates if candidate.run_id == str(run_id)]
    finally:
        registry.close()

    if len(exact) != 1:
        raise ValueError(
            f"Expected one successful pose model for set {set_id!r} and run "
            f"{run_id!r}; found {len(exact)} for recording {recording_id!r}."
        )
    selected = exact[0]
    model_path = Path(selected.model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Registered pose model artifact not found: {model_path}")
    model_sha256 = str(selected.model_sha256 or "").strip()
    if not model_sha256:
        raise ValueError(
            f"Registered pose model run {selected.run_id!r} has no model_sha256; "
            "refusing an unpinned deployment."
        )
    return PoseModelBinding(
        recording_id=resolved_recording_id,
        set_id=selected.set_id,
        run_id=selected.run_id,
        model_path=model_path,
        model_sha256=model_sha256,
        score=float(selected.weighted_score),
        created_utc=selected.created_utc,
    )


def validate_registered_analysis_zarr(
    *,
    registry_path: Path,
    recording_id: str,
    analysis_zarr: Path,
) -> None:
    resolved_zarr = analysis_zarr.expanduser().resolve()
    registry = Registry(registry_path.expanduser().resolve())
    try:
        rows = registry.conn.execute(
            """
            SELECT dataset_id
            FROM datasets
            WHERE recording_id = ?
              AND zarr_path = ?
              AND LOWER(COALESCE(zarr_use, '')) = 'analysis';
            """,
            (str(recording_id), str(resolved_zarr)),
        ).fetchall()
    finally:
        registry.close()
    if not rows:
        raise ValueError(
            f"Analysis Zarr is not registered to recording {recording_id!r}: "
            f"{resolved_zarr}"
        )


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Flat ROI cache manifest field {field!r} must be an object.")
    return value


def validate_flat_roi_cache_binding(
    *,
    manifest_path: Path,
    analysis_zarr: Path,
    crop_run: str | None,
    min_roi_size: int = DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
) -> FlatRoiCacheBinding:
    """Validate manifest identity and payload size without opening the Zarr."""

    resolved_manifest = manifest_path.expanduser().resolve()
    resolved_zarr = analysis_zarr.expanduser().resolve()
    manifest = load_flat_roi_cache_manifest(resolved_manifest)
    manifest_sha256 = hashlib.sha256(resolved_manifest.read_bytes()).hexdigest()
    manifest_crop_run = crop_run_name_from_manifest(resolved_manifest)
    requested_crop_run = str(crop_run).strip() if crop_run is not None else manifest_crop_run
    if requested_crop_run != manifest_crop_run:
        raise ValueError(
            f"Flat ROI cache crop run mismatch: manifest has {manifest_crop_run!r}, "
            f"target requests {requested_crop_run!r}."
        )
    source = _mapping(manifest.get("source"), field="source")
    source_archive = str(source.get("archive_path") or "").strip()
    if not source_archive:
        raise ValueError("Flat ROI cache manifest is missing source.archive_path.")
    if Path(source_archive).expanduser().resolve() != resolved_zarr:
        raise ValueError(
            f"Flat ROI cache archive mismatch: manifest has {source_archive!r}, "
            f"target is {resolved_zarr}."
        )

    array = _mapping(manifest.get("array"), field="array")
    raw_shape = array.get("shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != 3:
        raise ValueError("Flat ROI cache manifest array.shape must be [N, H, W].")
    try:
        shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Flat ROI cache manifest array.shape must contain integers."
        ) from exc
    if any(value <= 0 for value in shape):
        raise ValueError("Flat ROI cache manifest array.shape values must be positive.")
    dtype_name = str(array.get("dtype") or "")
    if dtype_name != "uint8":
        raise ValueError(
            f"Unsupported flat ROI cache dtype {dtype_name!r}. Expected uint8."
        )
    order = str(array.get("order") or "C")
    if order != "C":
        raise ValueError(
            f"Unsupported flat ROI cache order {order!r}. Expected C."
        )
    raw_payload_path = str(array.get("bin_path") or "").strip()
    if not raw_payload_path:
        raise ValueError("Flat ROI cache manifest is missing array.bin_path.")
    payload_path = Path(raw_payload_path).expanduser()
    if not payload_path.is_absolute():
        payload_path = resolved_manifest.parent / payload_path
    payload_path = payload_path.resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(f"Flat ROI cache payload not found: {payload_path}")
    expected_bytes = int(shape[0]) * int(shape[1]) * int(shape[2])
    actual_bytes = payload_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            "Flat ROI cache payload size mismatch: "
            f"expected {expected_bytes} bytes, got {actual_bytes}."
        )
    cache_key = str(manifest.get("cache_key") or "").strip()
    crop_signature = source.get("crop_signature")
    crop_revision = source.get("crop_revision")
    if not cache_key:
        raise ValueError("Flat ROI cache manifest is missing cache_key.")
    if (
        crop_signature is None
        or crop_signature == ""
        or crop_revision is None
        or crop_revision == ""
    ):
        raise ValueError(
            "Flat ROI cache manifest must pin source.crop_signature and "
            "source.crop_revision."
        )
    if int(shape[1]) < int(min_roi_size) or int(shape[2]) < int(min_roi_size):
        raise ValueError(
            f"Flat ROI cache is {shape[2]}x{shape[1]}, but zebrafish keypoint "
            f"inference requires at least {int(min_roi_size)}x{int(min_roi_size)}."
        )
    total_bytes = int(array.get("total_bytes") or actual_bytes)
    if actual_bytes != total_bytes:
        raise ValueError(
            f"Flat ROI cache total_bytes mismatch: manifest has {total_bytes}, "
            f"payload has {actual_bytes}."
        )
    return FlatRoiCacheBinding(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha256,
        payload_path=payload_path,
        crop_run=requested_crop_run,
        cache_key=cache_key,
        crop_signature=crop_signature,
        crop_revision=crop_revision,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        total_bytes=total_bytes,
        payload_sha256=(str(array.get("sha256")) if array.get("sha256") else None),
    )


def _canonical_identity(value: object) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def _crop_storage_mode(crop_group: Any) -> str:
    explicit = str(crop_group.attrs.get("crop_storage_mode") or "").strip()
    if explicit in {"materialized", "geometry_only"}:
        return explicit
    return "materialized" if "roi_images" in crop_group else "geometry_only"


def _crop_roi_shape(crop_group: Any) -> tuple[int, int]:
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    roi_images = crop_group.get("roi_images")
    shape = getattr(roi_images, "shape", ())
    if len(shape) >= 3:
        return int(shape[1]), int(shape[2])
    raise ValueError("Crop run is missing a fixed roi_size and persisted roi_images shape.")


def _probe_video_shape(path: Path) -> tuple[tuple[int, int] | None, str]:
    try:
        completed = subprocess.run(
            (
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, "ffprobe_missing"
    if completed.returncode != 0:
        return None, "ffprobe_failed"
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        height = int(stream["height"])
        width = int(stream["width"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None, "ffprobe_invalid_output"
    return (height, width), "ok"


def validate_keypoint_input_dag(
    *,
    analysis_zarr: Path,
    cache: FlatRoiCacheBinding,
    min_roi_size: int = DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
) -> KeypointInputCapability:
    """Prove that a cache-backed keypoint job is runnable from the live crop DAG."""

    resolved_zarr = analysis_zarr.expanduser().resolve()
    root = open_zarr_group_direct(resolved_zarr, mode="r")
    crop_parent = root.get("crop_runs")
    if crop_parent is None or cache.crop_run not in crop_parent:
        raise ValueError(
            f"Live crop DAG is missing crop_runs/{cache.crop_run} required by the cache."
        )
    crop_group = crop_parent[cache.crop_run]
    if not is_run_complete_in_parent(crop_parent, crop_group):
        raise ValueError(f"Live crop DAG node crop_runs/{cache.crop_run} is not complete.")

    live_signature = crop_group.attrs.get("crop_signature")
    live_revision = crop_group.attrs.get("crop_revision")
    if live_signature is None or live_revision is None:
        raise ValueError(
            f"Live crop DAG node crop_runs/{cache.crop_run} must declare "
            "crop_signature and crop_revision."
        )
    if _canonical_identity(live_signature) != _canonical_identity(cache.crop_signature):
        raise ValueError(
            f"Flat cache crop_signature does not match crop_runs/{cache.crop_run}."
        )
    if _canonical_identity(live_revision) != _canonical_identity(cache.crop_revision):
        raise ValueError(
            f"Flat cache crop_revision does not match crop_runs/{cache.crop_run}."
        )

    if "roi_coordinates_full" not in crop_group:
        raise ValueError(f"crop_runs/{cache.crop_run} is missing roi_coordinates_full.")
    if "frame_indices" not in crop_group:
        raise ValueError(f"crop_runs/{cache.crop_run} is missing frame_indices lineage.")
    total_rows = int(crop_group["roi_coordinates_full"].shape[0])
    frame_rows = int(crop_group["frame_indices"].shape[0])
    roi_height, roi_width = _crop_roi_shape(crop_group)
    live_shape = (total_rows, roi_height, roi_width)
    if frame_rows != total_rows:
        raise ValueError(
            f"crop_runs/{cache.crop_run} frame_indices has {frame_rows} rows, "
            f"but roi_coordinates_full has {total_rows}."
        )
    if tuple(cache.shape) != live_shape:
        raise ValueError(
            f"Flat cache shape {cache.shape} does not match live crop DAG shape "
            f"{live_shape}."
        )
    if roi_height < int(min_roi_size) or roi_width < int(min_roi_size):
        raise ValueError(
            f"Live crop DAG node is {roi_width}x{roi_height}; zebrafish keypoint "
            f"inference requires at least {int(min_roi_size)}x{int(min_roi_size)}."
        )

    storage_mode = _crop_storage_mode(crop_group)
    roi_images = crop_group.get("roi_images")
    roi_images_shape = getattr(roi_images, "shape", ())
    persisted_available = len(roi_images_shape) >= 3
    persisted_eligible = bool(
        persisted_available
        and int(roi_images_shape[1]) >= int(min_roi_size)
        and int(roi_images_shape[2]) >= int(min_roi_size)
        and int(roi_images_shape[0]) == total_rows
    )

    source_kind = str(
        crop_group.attrs.get("source_pixels")
        or crop_group.attrs.get("roi_pixel_provider")
        or crop_group.attrs.get("source_type")
        or ""
    ).strip()
    raw_crop_video_path = str(
        crop_group.attrs.get("source_crop_video_path")
        or crop_group.attrs.get("source_video_path")
        or ""
    ).strip()
    crop_video_path: Path | None = None
    if raw_crop_video_path:
        candidate = Path(raw_crop_video_path).expanduser()
        if not candidate.is_absolute():
            recording_dir = (
                resolved_zarr.parent.parent
                if resolved_zarr.parent.name == "zarr"
                else resolved_zarr.parent
            )
            candidate = recording_dir / candidate
        crop_video_path = candidate.resolve()
    crop_video_available = bool(
        source_kind == "acquisition_crop_video"
        and crop_video_path is not None
        and crop_video_path.is_file()
    )
    crop_video_shape: tuple[int, int] | None = None
    crop_video_probe_status = "not_available"
    if crop_video_available and crop_video_path is not None:
        crop_video_shape, crop_video_probe_status = _probe_video_shape(crop_video_path)
    crop_video_eligible = bool(
        crop_video_available
        and crop_video_shape is not None
        and crop_video_shape == (roi_height, roi_width)
        and crop_video_shape[0] >= int(min_roi_size)
        and crop_video_shape[1] >= int(min_roi_size)
    )
    rejected_sources: dict[str, str] = {}
    if not persisted_eligible:
        rejected_sources["persisted_roi_images"] = (
            "missing"
            if not persisted_available
            else f"too_small_or_row_mismatch:{tuple(int(v) for v in roi_images_shape)}"
        )
    if not crop_video_eligible:
        if source_kind != "acquisition_crop_video":
            rejected_sources["acquisition_crop_video"] = "not_declared_by_crop_run"
        elif crop_video_path is None or not crop_video_path.is_file():
            rejected_sources["acquisition_crop_video"] = "missing_video_file"
        elif crop_video_probe_status != "ok" or crop_video_shape is None:
            rejected_sources["acquisition_crop_video"] = crop_video_probe_status
        elif crop_video_shape != (roi_height, roi_width):
            rejected_sources["acquisition_crop_video"] = (
                f"shape_mismatch:video={crop_video_shape},crop={(roi_height, roi_width)}"
            )
        else:
            rejected_sources["acquisition_crop_video"] = (
                f"too_small:{crop_video_shape[1]}x{crop_video_shape[0]}"
            )

    return KeypointInputCapability(
        selected_source="flat_roi_cache",
        min_roi_size=int(min_roi_size),
        crop_run=cache.crop_run,
        crop_storage_mode=storage_mode,
        crop_shape=live_shape,
        persisted_roi_images_available=persisted_available,
        persisted_roi_images_eligible=persisted_eligible,
        acquisition_crop_video_available=crop_video_available,
        acquisition_crop_video_eligible=crop_video_eligible,
        acquisition_crop_video_path=crop_video_path,
        acquisition_crop_video_shape=crop_video_shape,
        acquisition_crop_video_probe_status=crop_video_probe_status,
        flat_roi_cache_eligible=True,
        rejected_sources=rejected_sources,
    )


def build_prediction_job(
    *,
    workflow_id: str,
    target_id: str,
    recording_dir: Path,
    analysis_zarr: Path,
    registry_path: Path,
    repo: Path,
    run_root: Path,
    run_names: KeypointRunNames,
    model: PoseModelBinding,
    cache: FlatRoiCacheBinding,
    pose_schema: str,
    batch_size: int,
    device: str,
    input_mode: str,
    progress_every_batches: int,
    keypoint_roi_shard_rows: int | None = DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    keypoint_frame_shard_rows: int = DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    resources: LsfResources,
) -> LsfJob:
    safe_target = safe_component(target_id, default="target", max_length=56)
    job_key = f"predict:{target_id}"
    job_name = safe_component(
        f"kp_{workflow_id}_{safe_target}",
        default="keypoint_prediction",
        max_length=120,
    )
    scratch_stage = "/scratch/<user>/<jobid>/palette_keypoint_roi_cache_stage"
    storage = resolve_keypoint_storage(
        roi_shard_rows=keypoint_roi_shard_rows,
        frame_shard_rows=keypoint_frame_shard_rows,
    )
    worker: list[str] = [
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.utils.run_keypoints_with_registry_model",
        "--recording-dir",
        str(recording_dir),
        "--output",
        str(analysis_zarr),
        "--registry",
        str(registry_path),
        "--set-id",
        model.set_id,
        "--model-run-id",
        model.run_id,
        "--require-unique",
        "--run-name",
        run_names.keypoint_run,
        "--crop-run",
        cache.crop_run,
        "--pose-schema",
        str(pose_schema),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
        "--roi-cache-policy",
        "always",
        "--roi-cache-manifest",
        str(cache.manifest_path),
        "--stage-roi-cache-to-scratch",
        "--roi-cache-staging-dir",
        scratch_stage,
        "--profile-timings",
        "--input-mode",
        str(input_mode),
        "--progress-jsonl",
        str(run_root / "progress" / f"{safe_target}.prediction.<jobid>.jsonl"),
        "--progress-every-batches",
        str(int(progress_every_batches)),
        "--json",
    ]
    if keypoint_roi_shard_rows is None:
        worker.append("--no-keypoint-sharding")
    else:
        worker.extend(
            [
                "--keypoint-roi-shard-rows",
                str(int(keypoint_roi_shard_rows)),
                "--keypoint-frame-shard-rows",
                str(int(keypoint_frame_shard_rows)),
            ]
        )
    command = build_runtime_command(
        worker,
        status_path_template=run_root / "status" / f"{safe_target}.prediction.<jobid>.json",
        workflow_id=workflow_id,
        family="keypoints.whole_recording",
        job_key=job_key,
        stage="keypoint_prediction",
        cwd=repo,
        environment_overrides={"PALETTE_DISABLE_REGISTRY_WRITES": "1"},
        cleanup_path_templates=(scratch_stage,),
        expected_output_templates=(
            str(analysis_zarr / "keypoints_runs" / run_names.keypoint_run),
        ),
        python_launcher=(str(repo / "scripts" / "py"),),
    )
    return LsfJob(
        job_key=job_key,
        job_name=job_name,
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{job_name}.%J.out",
        stderr_path=run_root / "logs" / f"{job_name}.%J.err",
        metadata={
            "target_id": target_id,
            "recording_id": model.recording_id,
            "analysis_zarr": str(analysis_zarr),
            "keypoint_run": run_names.keypoint_run,
            "model": model.to_json(),
            "cache": cache.to_json(),
            "keypoint_storage": storage,
        },
    )


def build_refinement_job(
    *,
    workflow_id: str,
    target_id: str,
    analysis_zarr: Path,
    repo: Path,
    run_root: Path,
    run_names: KeypointRunNames,
    chunk_size: int,
    scheduler: str,
    num_workers: int,
    memory_limit: str | None,
    resources: LsfResources,
    prediction_job: LsfJob,
) -> LsfJob:
    safe_target = safe_component(target_id, default="target", max_length=56)
    job_key = f"refine:{target_id}"
    job_name = safe_component(
        f"kp_refine_{workflow_id}_{safe_target}",
        default="keypoint_refinement",
        max_length=120,
    )
    worker: list[str] = [
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.refinement.refine_keypoints",
        str(analysis_zarr),
        "--keypoint-run",
        run_names.keypoint_run,
        "--run-name",
        run_names.refined_keypoint_run,
        "--config",
        str(repo / "configs" / "fisheye" / "default.yaml"),
        "--chunk-size",
        str(int(chunk_size)),
        "--scheduler",
        str(scheduler),
        "--num-workers",
        str(int(num_workers)),
        "--no-post-audit",
    ]
    if memory_limit:
        worker.extend(["--memory-limit", str(memory_limit)])
    command = build_runtime_command(
        worker,
        status_path_template=run_root / "status" / f"{safe_target}.refinement.<jobid>.json",
        workflow_id=workflow_id,
        family="keypoints.whole_recording",
        job_key=job_key,
        stage="keypoint_refinement",
        cwd=repo,
        environment_overrides={"PALETTE_DISABLE_REGISTRY_WRITES": "1"},
        expected_output_templates=(
            str(
                analysis_zarr
                / "refined_keypoints_runs"
                / run_names.refined_keypoint_run
            ),
        ),
        python_launcher=(str(repo / "scripts" / "py"),),
    )
    return LsfJob(
        job_key=job_key,
        job_name=job_name,
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{job_name}.%J.out",
        stderr_path=run_root / "logs" / f"{job_name}.%J.err",
        dependency=LsfDependency((prediction_job.job_key,)),
        metadata={
            "target_id": target_id,
            "analysis_zarr": str(analysis_zarr),
            "source_keypoint_run": run_names.keypoint_run,
            "refined_keypoint_run": run_names.refined_keypoint_run,
        },
    )


__all__ = [
    "DEFAULT_ZEBRAFISH_MIN_ROI_SIZE",
    "DEFAULT_KEYPOINT_FRAME_SHARD_ROWS",
    "DEFAULT_KEYPOINT_ROI_SHARD_ROWS",
    "FlatRoiCacheBinding",
    "KeypointInputCapability",
    "KeypointRunNames",
    "PoseModelBinding",
    "build_keypoint_run_names",
    "build_prediction_job",
    "build_refinement_job",
    "resolve_pose_model_binding",
    "resolve_keypoint_storage",
    "safe_component",
    "validate_flat_roi_cache_binding",
    "validate_keypoint_input_dag",
    "validate_registered_analysis_zarr",
]
