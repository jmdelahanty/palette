"""Plan and publish reusable flat ROI cache workflow fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from fisheye.cluster.keypoints.common import (
    DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
    FlatRoiCacheBinding,
    safe_component,
    validate_flat_roi_cache_binding,
)
from fisheye.cluster.lsf import LsfJob, LsfResources
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_USER_TOKEN,
    build_runtime_command,
)
from fisheye.shared.flat_roi_cache import build_flat_roi_cache
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent


PUBLISH_REPORT_SCHEMA = "palette.flat_roi_cache_publish.v1"
DEFAULT_SHARED_CACHE_ROOT = Path(
    "/nrs/johnson/palette_staging/flat_roi_cache"
)


def _cache_contract(cache: FlatRoiCacheBinding) -> dict[str, Any]:
    return {
        "crop_signature": json_ready(cache.crop_signature),
        "crop_revision": json_ready(cache.crop_revision),
        "shape": list(cache.shape),
        "total_bytes": cache.total_bytes,
    }


def _validate_binding_contract(
    binding: FlatRoiCacheBinding,
    expected: dict[str, Any],
) -> None:
    if list(binding.shape) != [int(value) for value in expected.get("shape") or []]:
        raise ValueError("Published cache shape does not match the planned contract.")
    if binding.total_bytes != int(expected.get("total_bytes") or -1):
        raise ValueError("Published cache size does not match the planned contract.")
    for field in ("crop_signature", "crop_revision"):
        actual = json_ready(getattr(binding, field))
        if json.dumps(actual, sort_keys=True, separators=(",", ":")) != json.dumps(
            expected.get(field), sort_keys=True, separators=(",", ":")
        ):
            raise ValueError(f"Published cache {field} drifted from the planned contract.")


def _crop_shape_and_identity(
    analysis_zarr: Path,
    *,
    crop_run: str,
) -> tuple[tuple[int, int, int], Any, Any]:
    root = open_zarr_group_direct(analysis_zarr, mode="r")
    crop_parent = root.get("crop_runs")
    if crop_parent is None or crop_run not in crop_parent:
        raise ValueError(f"Analysis Zarr is missing crop_runs/{crop_run}.")
    crop_group = crop_parent[crop_run]
    if not is_run_complete_in_parent(crop_parent, crop_group):
        raise ValueError(f"crop_runs/{crop_run} is not complete.")
    signature = crop_group.attrs.get("crop_signature")
    revision = crop_group.attrs.get("crop_revision")
    if signature is None or revision is None:
        raise ValueError(
            f"crop_runs/{crop_run} must declare crop_signature and crop_revision."
        )
    coordinates = crop_group.get("roi_coordinates_full")
    frame_indices = crop_group.get("frame_indices")
    if coordinates is None or frame_indices is None:
        raise ValueError(
            f"crop_runs/{crop_run} requires roi_coordinates_full and frame_indices."
        )
    rows = int(coordinates.shape[0])
    if int(frame_indices.shape[0]) != rows:
        raise ValueError(f"crop_runs/{crop_run} has inconsistent row lineage.")
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        height, width = int(roi_size[0]), int(roi_size[1])
    else:
        roi_images = crop_group.get("roi_images")
        roi_shape = getattr(roi_images, "shape", ())
        if len(roi_shape) < 3:
            raise ValueError(f"crop_runs/{crop_run} does not declare a fixed ROI size.")
        height, width = int(roi_shape[1]), int(roi_shape[2])
    return (rows, height, width), signature, revision


def plan_flat_roi_cache_binding(
    *,
    analysis_zarr: Path,
    crop_run: str,
    manifest_path: Path,
    producer_job_key: str,
    min_roi_size: int = DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
) -> FlatRoiCacheBinding:
    """Describe a cache that will be produced later in the same workflow."""

    resolved_zarr = analysis_zarr.expanduser().resolve()
    resolved_manifest = manifest_path.expanduser().resolve()
    if resolved_manifest.exists() or resolved_manifest.with_suffix(".bin").exists():
        raise FileExistsError(
            "Planned cache output already exists; select the existing-cache policy or "
            f"choose a new path: {resolved_manifest}"
        )
    shape, signature, revision = _crop_shape_and_identity(
        resolved_zarr,
        crop_run=str(crop_run),
    )
    if shape[1] < int(min_roi_size) or shape[2] < int(min_roi_size):
        raise ValueError(
            f"crop_runs/{crop_run} is {shape[2]}x{shape[1]}, but inference "
            f"requires at least {int(min_roi_size)}x{int(min_roi_size)}."
        )
    return FlatRoiCacheBinding(
        manifest_path=resolved_manifest,
        manifest_sha256=None,
        payload_path=resolved_manifest.with_suffix(".bin"),
        crop_run=str(crop_run),
        cache_key=None,
        crop_signature=signature,
        crop_revision=revision,
        shape=shape,
        total_bytes=int(shape[0]) * int(shape[1]) * int(shape[2]),
        payload_sha256=None,
        availability="planned",
        producer_job_key=str(producer_job_key),
    )


def build_flat_roi_cache_job(
    *,
    workflow_id: str,
    target_id: str,
    analysis_zarr: Path,
    cache: FlatRoiCacheBinding,
    repo: Path,
    run_root: Path,
    resources: LsfResources,
    batch_size: int = 1024,
    decode_backend: str = "pynvvc_luma",
    roi_live_acceleration: str = "cpu",
    roi_live_gpu_chunk_frames: int = 32,
) -> LsfJob:
    if cache.availability != "planned" or not cache.producer_job_key:
        raise ValueError("A cache build job requires a planned cache binding.")
    safe_target = safe_component(target_id, default="target", max_length=56)
    job_name = safe_component(
        f"cache_{workflow_id}_{safe_target}",
        default="flat_roi_cache",
        max_length=120,
    )
    scratch_dir = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        f"palette_flat_roi_cache_{safe_target}"
    )
    report_path = (
        run_root
        / "cache"
        / f"{safe_target}.publish.{RUNTIME_JOB_ID_TOKEN}.json"
    )
    progress_path = (
        run_root
        / "progress"
        / f"{safe_target}.cache.{RUNTIME_JOB_ID_TOKEN}.jsonl"
    )
    worker = (
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.cluster.flat_roi_cache",
        "--analysis-zarr",
        str(analysis_zarr),
        "--crop-run",
        cache.crop_run,
        "--manifest-path",
        str(cache.manifest_path),
        "--scratch-dir",
        scratch_dir,
        "--batch-size",
        str(int(batch_size)),
        "--decode-backend",
        str(decode_backend),
        "--roi-live-acceleration",
        str(roi_live_acceleration),
        "--roi-live-gpu-chunk-frames",
        str(int(roi_live_gpu_chunk_frames)),
        "--expected-contract-json",
        json.dumps(_cache_contract(cache), sort_keys=True, separators=(",", ":")),
        "--progress-jsonl",
        str(progress_path),
        "--output-json",
        str(report_path),
    )
    command = build_runtime_command(
        worker,
        status_path_template=(
            run_root
            / "status"
            / f"{safe_target}.cache.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
        workflow_id=workflow_id,
        family="analysis.whole_recording",
        job_key=cache.producer_job_key,
        stage="flat_roi_cache_publish",
        cwd=repo,
        cleanup_path_templates=(scratch_dir,),
        expected_output_templates=(
            str(cache.manifest_path),
            str(cache.payload_path),
            str(report_path),
        ),
        python_launcher=(str(repo / "scripts" / "py"),),
    )
    return LsfJob(
        job_key=cache.producer_job_key,
        job_name=job_name,
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{job_name}.%J.out",
        stderr_path=run_root / "logs" / f"{job_name}.%J.err",
        metadata={
            "target_id": target_id,
            "analysis_zarr": str(analysis_zarr),
            "crop_run": cache.crop_run,
            "cache": cache.to_json(),
            "publish_policy": "payload_first_manifest_last",
        },
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def publish_flat_roi_cache(
    *,
    analysis_zarr: Path,
    crop_run: str,
    manifest_path: Path,
    scratch_dir: Path,
    batch_size: int,
    decode_backend: str,
    roi_live_acceleration: str,
    roi_live_gpu_chunk_frames: int,
    progress_jsonl: Path | None = None,
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build on node-local scratch and atomically publish payload then manifest."""

    resolved_zarr = analysis_zarr.expanduser().resolve()
    final_manifest = manifest_path.expanduser().resolve()
    final_payload = final_manifest.with_suffix(".bin")
    if final_manifest.exists() or final_payload.exists():
        raise FileExistsError(
            f"Refusing to replace existing cache artifacts: {final_manifest}"
        )
    scratch = scratch_dir.expanduser()
    scratch.mkdir(parents=True, exist_ok=True)
    local_manifest = scratch / final_manifest.name
    local_payload = local_manifest.with_suffix(".bin")
    final_manifest.parent.mkdir(parents=True, exist_ok=True)

    progress_handle = None
    if progress_jsonl is not None:
        progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        progress_handle = progress_jsonl.open("a", encoding="utf-8", buffering=1)

    def emit_progress(event: dict[str, Any]) -> None:
        if progress_handle is not None:
            progress_handle.write(json.dumps(event, sort_keys=True) + "\n")
            progress_handle.flush()

    started = time.perf_counter()
    try:
        build_flat_roi_cache(
            zarr_path=resolved_zarr,
            crop_run=crop_run,
            manifest_path=local_manifest,
            batch_size=int(batch_size),
            overwrite=False,
            compute_sha256=False,
            roi_live_acceleration=roi_live_acceleration,
            roi_live_gpu_chunk_frames=int(roi_live_gpu_chunk_frames),
            roi_decode_backend=decode_backend,
            progress_callback=emit_progress if progress_handle is not None else None,
            progress_interval_seconds=30.0,
        )
    finally:
        if progress_handle is not None:
            progress_handle.close()

    if not local_manifest.is_file() or not local_payload.is_file():
        raise RuntimeError("Flat ROI cache builder did not create both local artifacts.")
    local_binding = validate_flat_roi_cache_binding(
        manifest_path=local_manifest,
        analysis_zarr=resolved_zarr,
        crop_run=crop_run,
        min_roi_size=1,
    )
    if expected_contract is not None:
        _validate_binding_contract(local_binding, expected_contract)
    host = socket.gethostname()
    job_id = os.environ.get("LSB_JOBID", "manual")
    temp_payload = final_payload.with_name(
        f"{final_payload.name}.tmp.{job_id}.{host}"
    )
    temp_manifest = final_manifest.with_name(
        f"{final_manifest.name}.tmp.{job_id}.{host}"
    )
    copy_started = time.perf_counter()
    try:
        shutil.copyfile(local_payload, temp_payload)
        os.replace(temp_payload, final_payload)
        copy_seconds = float(time.perf_counter() - copy_started)
        published = json.loads(local_manifest.read_text(encoding="utf-8"))
        published["manifest_path"] = str(final_manifest)
        published["publisher"] = {
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "publish_host": host,
            "lsb_jobid": os.environ.get("LSB_JOBID"),
            "source_manifest_path": str(local_manifest),
            "published_manifest_path": str(final_manifest),
            "published_bin_path": str(final_payload),
            "published_bin_size_bytes": final_payload.stat().st_size,
            "payload_copy_seconds": copy_seconds,
            "publish_policy": "payload_first_manifest_last",
        }
        _write_json_atomic(temp_manifest, published)
        os.replace(temp_manifest, final_manifest)
    finally:
        temp_payload.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)

    binding = validate_flat_roi_cache_binding(
        manifest_path=final_manifest,
        analysis_zarr=resolved_zarr,
        crop_run=crop_run,
        min_roi_size=1,
    )
    return {
        "schema": PUBLISH_REPORT_SCHEMA,
        "status": "ok",
        "analysis_zarr": str(resolved_zarr),
        "crop_run": crop_run,
        "manifest_path": str(final_manifest),
        "manifest_sha256": hashlib.sha256(final_manifest.read_bytes()).hexdigest(),
        "payload_path": str(final_payload),
        "payload_bytes": final_payload.stat().st_size,
        "shape": list(binding.shape),
        "duration_seconds": float(time.perf_counter() - started),
        "publisher": published["publisher"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--decode-backend",
        choices=("auto", "pynvvc_luma", "read_slice"),
        default="pynvvc_luma",
    )
    parser.add_argument(
        "--roi-live-acceleration", choices=("auto", "cpu", "gpu"), default="cpu"
    )
    parser.add_argument("--roi-live-gpu-chunk-frames", type=int, default=32)
    parser.add_argument("--expected-contract-json")
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = publish_flat_roi_cache(
        analysis_zarr=args.analysis_zarr,
        crop_run=args.crop_run,
        manifest_path=args.manifest_path,
        scratch_dir=args.scratch_dir,
        batch_size=int(args.batch_size),
        decode_backend=args.decode_backend,
        roi_live_acceleration=args.roi_live_acceleration,
        roi_live_gpu_chunk_frames=int(args.roi_live_gpu_chunk_frames),
        progress_jsonl=args.progress_jsonl,
        expected_contract=(
            json.loads(args.expected_contract_json)
            if args.expected_contract_json is not None
            else None
        ),
    )
    if args.output_json is not None:
        _write_json_atomic(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SHARED_CACHE_ROOT",
    "PUBLISH_REPORT_SCHEMA",
    "build_flat_roi_cache_job",
    "main",
    "plan_flat_roi_cache_binding",
    "publish_flat_roi_cache",
]
