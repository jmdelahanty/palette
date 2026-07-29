"""Run the representative crop-v2 -> keypoint-v2 integration canary.

The workflow keeps the source archive read-only, materializes or reuses one
durable NRS flat ROI cache, runs YOLO from node-local scratch, derives the
strict raw-keypoint, quality, and body-frame snapshots locally, and publishes
the complete selector-ineligible workflow directory atomically.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.cluster.flat_roi_cache import publish_flat_roi_cache
from fisheye.cluster.keypoints.common import validate_flat_roi_cache_binding
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.zarr.benchmark_runtime import peak_rss_bytes, sha256_array, utc_now
from fisheye.shared.zarr.body_frame_producer import (
    BodyFrameSourceReference,
    build_keypoint_body_frame_recipe,
    prepare_keypoint_body_frame,
)
from fisheye.shared.zarr.body_frame_publication import (
    publish_selector_ineligible_body_frame_snapshot,
    validate_body_frame_shadow_publication,
)
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1
from fisheye.shared.zarr.keypoint_manifest import (
    KeypointPreprocessingReference,
    keypoint_crop_source_from_manifest,
    keypoint_skeleton_digest,
)
from fisheye.shared.zarr.keypoint_publication import (
    prepare_raw_keypoint_v2_from_yolo_arrays,
    publish_selector_ineligible_keypoint_snapshot,
    validate_keypoint_shadow_publication,
)
from fisheye.shared.zarr.keypoint_quality_producer import (
    ObservationLocalKeypointQualityPolicy,
    prepare_observation_local_keypoint_quality,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    publish_selector_ineligible_keypoint_quality_snapshot,
    validate_keypoint_quality_shadow_publication,
)
from fisheye.shared.zarr.keypoint_quality_schema import (
    KeypointQualitySourceReference,
)
from fisheye.shared.zarr.keypoint_schema import KeypointDimensions
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


CANARY_SCHEMA_ID = "palette.keypoint.crop_v2_integration_canary"
CANARY_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path(
    "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/"
    "keypoint_storage/integration"
)
DEFAULT_CACHE_ROOT = Path("/nrs/johnson/palette_staging/flat_roi_cache")


def _sha256_file(path: Path, *, block_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _metadata_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("zarr.json")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _git_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]

    def command(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "repository": str(repository),
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "worktree_clean": command("status", "--short") == "",
        "driver": str(Path(__file__).resolve().relative_to(repository)),
    }


def _require_below(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    safe_root = root.expanduser().resolve()
    if resolved == safe_root:
        raise ValueError(f"{label} cannot equal its namespace root.")
    try:
        resolved.relative_to(safe_root)
    except ValueError as exc:
        raise ValueError(f"{label} must be below {safe_root}.") from exc
    return resolved


def _copy_file(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    shutil.copyfile(source, destination)
    seconds = float(time.perf_counter() - started)
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"Staged file size mismatch for {source}.")
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": int(destination.stat().st_size),
        "seconds": seconds,
        "mib_per_second": (
            float(destination.stat().st_size) / 1024**2 / seconds
            if seconds > 0
            else None
        ),
    }


def _stage_cache(
    manifest_path: Path,
    *,
    destination_dir: Path,
) -> tuple[Path, dict[str, object]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    array = payload.get("array")
    if not isinstance(array, Mapping):
        raise ValueError("Flat-cache manifest is missing array metadata.")
    raw_payload_path = Path(str(array.get("bin_path") or ""))
    source_payload = (
        raw_payload_path
        if raw_payload_path.is_absolute()
        else manifest_path.parent / raw_payload_path
    )
    local_manifest = destination_dir / manifest_path.name
    local_payload = destination_dir / source_payload.name
    payload_receipt = _copy_file(source_payload, local_payload)
    manifest_receipt = _copy_file(manifest_path, local_manifest)
    return local_manifest, {
        "manifest": manifest_receipt,
        "payload": payload_receipt,
    }


def _publish_workflow_directory(local: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"Canary destination already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f".{final.name}.tmp.{os.getpid()}.{socket.gethostname()}")
    if temporary.exists():
        raise FileExistsError(f"Canary temporary destination exists: {temporary}")
    try:
        shutil.copytree(local, temporary, copy_function=shutil.copy2)
        os.replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _crop_arrays(group: Any) -> dict[str, np.ndarray]:
    return {
        path: np.asarray(group[path][...])
        for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths
    }


def _pose_binding(model_sha256: str) -> dict[str, Any]:
    return build_explicit_pose_model_schema_binding(
        model_sha256=model_sha256,
        assertion_id="selector_ineligible_crop_v2_keypoint_canary_20260729",
        skeleton_id="pose_schema:traditional_v1",
        model_kpt_shape=[3, 3],
        keypoint_labels=["swim_bladder", "eye_left", "eye_right"],
        edges=[[0, 1], [0, 2], [1, 2]],
    )


def run_canary(
    *,
    source_analysis_zarr: Path,
    crop_run: str,
    source_video_path: Path,
    model_path: Path,
    cache_manifest_path: Path,
    destination: Path,
    scratch_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    device: str = "cuda:0",
    batch_size: int = 256,
    cache_batch_size: int = 512,
) -> dict[str, object]:
    """Execute and atomically publish one representative selector-ineligible chain."""

    source = source_analysis_zarr.expanduser().resolve()
    source_video = source_video_path.expanduser().resolve()
    model = model_path.expanduser().resolve()
    final = _require_below(destination, output_root, label="Canary destination")
    cache_manifest = _require_below(
        cache_manifest_path, cache_root, label="Flat-cache manifest"
    )
    scratch = scratch_root.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr is missing: {source}")
    for label, path in (("source video", source_video), ("model", model)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if final.exists():
        raise FileExistsError(f"Canary destination already exists: {final}")
    if scratch.exists():
        raise FileExistsError(f"Canary scratch root already exists: {scratch}")
    scratch.mkdir(parents=True)
    started = time.perf_counter()
    phases: dict[str, float] = {}
    source_metadata_before = _metadata_fingerprint(source)

    cache_payload = cache_manifest.with_suffix(".bin")
    if cache_manifest.exists() != cache_payload.exists():
        raise RuntimeError("Flat-cache manifest/payload publication is incomplete.")
    cache_created = not cache_manifest.exists()
    cache_publish_report: dict[str, object] | None = None
    if cache_created:
        staged_video = scratch / "source" / source_video.name
        stage_started = time.perf_counter()
        video_stage = _copy_file(source_video, staged_video)
        phases["stage_source_video"] = time.perf_counter() - stage_started
        cache_started = time.perf_counter()
        cache_publish_report = publish_flat_roi_cache(
            analysis_zarr=source,
            crop_run=crop_run,
            manifest_path=cache_manifest,
            scratch_dir=scratch / "cache_build",
            batch_size=int(cache_batch_size),
            decode_backend="pynvvc_luma",
            roi_live_acceleration="cpu",
            roi_live_gpu_chunk_frames=32,
            source_video_path_override=staged_video,
            compute_sha256=True,
            progress_jsonl=cache_manifest.parent / "cache_build_progress.jsonl",
        )
        phases["build_and_publish_cache"] = time.perf_counter() - cache_started
        staged_video.unlink()
        local_cache_manifest = scratch / "cache_build" / cache_manifest.name
        cache_stage = {
            "reuse_local_builder_output": True,
            "video_stage": video_stage,
        }
    else:
        cache_started = time.perf_counter()
        local_cache_manifest, cache_stage = _stage_cache(
            cache_manifest,
            destination_dir=scratch / "cache_stage",
        )
        phases["stage_existing_cache"] = time.perf_counter() - cache_started

    cache_binding = validate_flat_roi_cache_binding(
        manifest_path=local_cache_manifest,
        analysis_zarr=source,
        crop_run=crop_run,
        min_roi_size=1,
    )
    cache_document = json.loads(cache_manifest.read_text(encoding="utf-8"))
    cache_array = cache_document.get("array")
    if not isinstance(cache_array, Mapping) or not cache_array.get("sha256"):
        raise ValueError("Representative persistent cache requires payload SHA-256.")

    local_archive = scratch / "compute" / "analysis.zarr"
    copy_started = time.perf_counter()
    shutil.copytree(source, local_archive, copy_function=shutil.copy2)
    phases["stage_crop_archive"] = time.perf_counter() - copy_started
    staged_model = scratch / "compute" / "model" / model.name
    model_stage = _copy_file(model, staged_model)
    model_sha256 = _sha256_file(model)
    if _sha256_file(staged_model) != model_sha256:
        raise RuntimeError("Staged keypoint model digest mismatch.")
    binding = _pose_binding(model_sha256)

    yolo_run_id = "source_yolo_crop_v2_legacy_boundary"
    inference_started = time.perf_counter()
    # Import the GPU/model stack only inside the execution phase. Planning and
    # manifest tests must not initialize Ultralytics or CUDA.
    from fisheye.detection.detect_keypoints_yolo import detect_keypoints_yolo

    resolved_yolo_run = detect_keypoints_yolo(
        zarr_path=local_archive,
        model_path=staged_model,
        run_name=yolo_run_id,
        output_parent="keypoint_shard_runs",
        crop_run=crop_run,
        pose_schema="traditional_v1",
        batch_size=int(batch_size),
        device=device,
        imgsz=512,
        conf=0.25,
        iou=0.5,
        max_det=1,
        roi_cache_policy="never",
        roi_cache_manifest=local_cache_manifest,
        roi_cache_expected_archive_path=source,
        roi_cache_source_tier="node_scratch",
        roi_cache_staged_to_node_scratch=True,
        input_mode="tensor",
        model_input_transform_mode="auto",
        coordinate_contract_mode="legacy_noncanonical",
        profile_timings=True,
        keypoint_roi_shard_rows=None,
        registry=None,
    )
    phases["yolo_inference"] = time.perf_counter() - inference_started
    if resolved_yolo_run != yolo_run_id:
        raise RuntimeError("YOLO returned an unexpected run identity.")

    local_root = zarr.open_group(
        str(local_archive), mode="r", use_consolidated=False
    )
    crop_group = local_root[f"crop_runs/{crop_run}"]
    crop_manifest_document = crop_group.attrs.get("run_manifest")
    if not isinstance(crop_manifest_document, Mapping):
        raise ValueError("Crop-v2 source is missing its exact run manifest.")
    crop_manifest_copy = json.loads(
        json.dumps(dict(crop_manifest_document), sort_keys=True)
    )
    crop_values = _crop_arrays(crop_group)
    crop_source = keypoint_crop_source_from_manifest(crop_manifest_copy)
    dimensions = KeypointDimensions(
        n_frames=crop_source.n_frames,
        n_instances=crop_source.n_instances,
        n_keypoints=3,
        source_width=crop_source.source_width,
        source_height=crop_source.source_height,
    )
    yolo_group = local_root[f"keypoint_shard_runs/{yolo_run_id}"]
    yolo_arrays = {name: yolo_group[name] for name in yolo_group.array_keys()}
    preprocessing = KeypointPreprocessingReference(
        profile_id="yolo_pose_flat_cache_v1",
        profile_version=1,
        input_mode="flat_bin_node_scratch",
        document={
            "source_pixel_dtype": "uint8",
            "source_pixel_channels": "grayscale",
            "source_pixel_contract": cache_document.get("builder", {}).get(
                "pixel_contract"
            ),
            "cache_manifest_sha256": _sha256_file(cache_manifest),
            "cache_payload_sha256": cache_array["sha256"],
            "model_input_mode": "tensor",
            "model_input_transform": "auto_identity_for_512x512",
            "imgsz": 512,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "max_detections_per_roi": 1,
        },
    )
    prepare_started = time.perf_counter()
    conversion = prepare_raw_keypoint_v2_from_yolo_arrays(
        yolo_arrays,
        dimensions=dimensions,
        source_crop_arrays=crop_values,
        source_crop_manifest=crop_manifest_copy,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
    )
    phases["prepare_raw_keypoint_v2"] = time.perf_counter() - prepare_started

    local_workflow = scratch / "publication" / "workflow"
    raw_run_id = "raw_keypoints_crop_v2_yolo_v2"
    raw_started = time.perf_counter()
    raw_publication = publish_selector_ineligible_keypoint_snapshot(
        conversion.prepared,
        destination=local_workflow / "raw.zarr",
        run_id=raw_run_id,
        shadow_root=local_workflow,
        created_by="keypoint_crop_v2_integration_canary",
    )
    phases["publish_raw_keypoint_v2"] = time.perf_counter() - raw_started
    skeleton_digest = keypoint_skeleton_digest(binding)
    skeleton_id = str(binding["pose_schema"]["skeleton_id"])
    raw_manifest_digest = canonical_json_sha256(raw_publication.manifest)
    row_signature_digest = sha256_array(
        conversion.prepared.arrays["keypoint_row_signature"]
    )

    quality_source = KeypointQualitySourceReference(
        run_name=raw_run_id,
        manifest_digest=raw_manifest_digest,
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=row_signature_digest,
    )
    quality_prepared = prepare_observation_local_keypoint_quality(
        conversion.prepared.arrays,
        source_dimensions=dimensions,
        source_crop_arrays=crop_values,
        source=quality_source,
        skeleton_digest=skeleton_digest,
        policy=ObservationLocalKeypointQualityPolicy(
            confidence_threshold=0.5,
            minimum_valid_keypoints=1,
        ),
    )
    quality_started = time.perf_counter()
    quality_publication = publish_selector_ineligible_keypoint_quality_snapshot(
        quality_prepared,
        source_manifest=raw_publication.manifest,
        destination=local_workflow / "quality.zarr",
        run_id="keypoint_quality_crop_v2_v1",
        shadow_root=local_workflow,
        created_by="keypoint_crop_v2_integration_canary",
    )
    phases["publish_keypoint_quality_v1"] = time.perf_counter() - quality_started

    body_source = BodyFrameSourceReference(
        stage="keypoints",
        run_name=raw_run_id,
        manifest_digest=raw_manifest_digest,
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=row_signature_digest,
    )
    body_recipe = build_keypoint_body_frame_recipe(
        pose_schema=binding["pose_schema"],
        skeleton_digest=skeleton_digest,
        keypoint_count=dimensions.n_keypoints,
    )
    body_prepared = prepare_keypoint_body_frame(
        conversion.prepared.arrays,
        source_dimensions=dimensions,
        source_crop_arrays=crop_values,
        source=body_source,
        source_manifest=raw_publication.manifest,
        recipe=body_recipe,
    )
    body_started = time.perf_counter()
    body_publication = publish_selector_ineligible_body_frame_snapshot(
        body_prepared,
        source_manifest=raw_publication.manifest,
        destination=local_workflow / "body_frame.zarr",
        run_id="body_frame_crop_v2_keypoints_v1",
        shadow_root=local_workflow,
        created_by="keypoint_crop_v2_integration_canary",
    )
    phases["publish_body_frame_v1"] = time.perf_counter() - body_started

    source_metadata_after = _metadata_fingerprint(source)
    if source_metadata_after != source_metadata_before:
        raise RuntimeError("Read-only source archive metadata changed during canary.")
    handoff: dict[str, object] = {
        "schema_id": CANARY_SCHEMA_ID,
        "schema_version": CANARY_SCHEMA_VERSION,
        "status": "complete",
        "benchmark_only": True,
        "selector_eligible": False,
        "created_at_utc": utc_now(),
        "source": {
            "analysis_zarr": str(source),
            "crop_run": crop_run,
            "crop_manifest_digest": canonical_json_sha256(crop_manifest_copy),
            "metadata_fingerprint_before": source_metadata_before,
            "metadata_fingerprint_after": source_metadata_after,
            "source_archive_unchanged": True,
        },
        "cache": {
            "created_in_this_run": cache_created,
            "manifest_path": str(cache_manifest),
            "manifest_sha256": _sha256_file(cache_manifest),
            "payload_path": str(cache_payload),
            "payload_bytes": int(cache_payload.stat().st_size),
            "payload_sha256": cache_array["sha256"],
            "shape": list(cache_binding.shape),
            "stage_to_node_scratch": cache_stage,
            "publish_report": cache_publish_report,
        },
        "model": {
            "source_path": str(model),
            "sha256": model_sha256,
            "stage": model_stage,
            "pose_model_schema_binding": binding,
        },
        "dimensions": {
            "n_frames": dimensions.n_frames,
            "n_instances": dimensions.n_instances,
            "n_keypoints": dimensions.n_keypoints,
            "source_width": dimensions.source_width,
            "source_height": dimensions.source_height,
        },
        "conversion_receipt": dict(conversion.conversion_receipt),
        "artifacts": {
            "raw_keypoints": {
                "path": str(final / "raw.zarr"),
                "run_id": raw_publication.run_id,
                "manifest_digest": raw_manifest_digest,
            },
            "keypoint_quality": {
                "path": str(final / "quality.zarr"),
                "run_id": quality_publication.run_id,
                "manifest_digest": canonical_json_sha256(
                    quality_publication.manifest
                ),
            },
            "body_frame": {
                "path": str(final / "body_frame.zarr"),
                "run_id": body_publication.run_id,
                "manifest_digest": canonical_json_sha256(body_publication.manifest),
            },
        },
        "timing_seconds": phases,
        "peak_rss_bytes": peak_rss_bytes(),
        "environment": {
            "host": socket.gethostname(),
            "lsb_job_id": os.environ.get("LSB_JOBID"),
            "device": device,
        },
        "palette": _git_provenance(),
        "production_state": {
            "source_archive_mutated": False,
            "registry_written": False,
            "selectors_written": False,
            "training_artifacts_written": False,
        },
    }
    handoff["elapsed_seconds"] = float(time.perf_counter() - started)
    write_json_atomic(local_workflow / "handoff_manifest.json", handoff)

    publish_started = time.perf_counter()
    _publish_workflow_directory(local_workflow, final)
    phases["publish_workflow_to_shared"] = time.perf_counter() - publish_started
    raw_final = replace(raw_publication, output_path=final / "raw.zarr")
    quality_final = replace(quality_publication, output_path=final / "quality.zarr")
    body_final = replace(body_publication, output_path=final / "body_frame.zarr")
    final_errors = (
        *validate_keypoint_shadow_publication(raw_final),
        *validate_keypoint_quality_shadow_publication(quality_final),
        *validate_body_frame_shadow_publication(body_final),
    )
    if final_errors:
        raise RuntimeError(
            "Published keypoint-v2 canary validation failed: "
            + "; ".join(final_errors)
        )
    return handoff


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-analysis-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--source-video-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-manifest-path", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cache-batch-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_canary(
        source_analysis_zarr=args.source_analysis_zarr,
        crop_run=args.crop_run,
        source_video_path=args.source_video_path,
        model_path=args.model_path,
        cache_manifest_path=args.cache_manifest_path,
        destination=args.destination,
        scratch_root=args.scratch_root,
        device=args.device,
        batch_size=int(args.batch_size),
        cache_batch_size=int(args.cache_batch_size),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
