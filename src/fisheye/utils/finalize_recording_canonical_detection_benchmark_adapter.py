#!/usr/bin/env python3
"""Normalize clipped legacy detections into one current canonical candidate.

This adapter is benchmark-only.  It reads the maintained per-clip detection
outputs, rebuilds the complete recording table through the current canonical
binder, writes the approved access-aware profile on node-local scratch, and
atomically places the
validated standalone store below ``.palette_benchmarks``.  It never mutates an
analysis archive, selector, or registry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.detection.clipped_native_artifact_io import (
    load_parent_frame_mapping,
)
from fisheye.detection.clipped_native_binding import (
    ClippedDetectionArtifactMember,
    bind_clipped_detection_artifacts,
)
from fisheye.detection.native_canonical_candidate import (
    write_native_clipped_detection_candidate,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    sha256_array,
    sha256_file,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.native_canonical_detection_publication import (
    load_native_canonical_detection_candidate,
)
from fisheye.shared.zarr_run_completion import is_run_complete


ADAPTER_SCHEMA_ID = "palette.canonical_detection.recording_benchmark_adapter"
ADAPTER_SCHEMA_VERSION = 3
ADAPTER_RECEIPT_NAME = "recording_canonical_adapter_receipt.json"
_ARRAY_NAMES = (
    "frame_indices",
    "bbox_norm_coords",
    "scores",
    "class_ids",
    "frame_counts",
    "n_detections",
    "instance_key",
)
_HEX = frozenset("0123456789abcdef")


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return text


def _safe_group(value: object, *, name: str) -> str:
    group = str(value).strip().strip("/")
    if not group or any(part in {"", ".", ".."} for part in group.split("/")):
        raise ValueError(f"{name} must be one safe archive-relative group path.")
    return group


def _array(group: Any, name: str) -> np.ndarray:
    values = np.asarray(group[name][...])
    return np.ascontiguousarray(values)


def _require_node_local_scratch(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    }:
        raise ValueError("Scratch must be one bounded child directory, not a root.")
    if str(root).startswith(("/groups/", "/nrs/")):
        raise ValueError("Scratch must be node-local, not shared storage.")
    if not root.is_dir():
        raise FileNotFoundError(f"Node-local scratch parent not found: {root}")
    return root


def _require_benchmark_destination(destination: Path, *, benchmark_root: Path) -> Path:
    path = destination.expanduser().resolve()
    root = benchmark_root.expanduser().resolve()
    if ".palette_benchmarks" not in root.parts:
        raise ValueError("Canonical adapter root must be below .palette_benchmarks.")
    if path == root or not path.is_relative_to(root):
        raise ValueError("Canonical adapter destination must be a child of its root.")
    if path.suffix != ".zarr":
        raise ValueError("Canonical adapter destination must use a .zarr suffix.")
    if path.exists():
        raise FileExistsError(f"Canonical benchmark destination exists: {path}")
    return path


def _publish_directory(local: Path, destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{os.uname().nodename}"
    )
    if temporary.exists():
        raise FileExistsError(f"Temporary publication exists: {temporary}")
    started = time.perf_counter()
    try:
        shutil.copytree(local, temporary, copy_function=shutil.copy2)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return float(time.perf_counter() - started)


def _source_member(
    *,
    archive: Any,
    analysis_zarr: Path,
    recording_frame_index: Path,
    recording_identity: str,
    unit: Mapping[str, Any],
    expected_index: int,
    expected_model_sha256: str,
) -> tuple[ClippedDetectionArtifactMember, dict[str, object], np.ndarray]:
    if unit.get("clip_index") != expected_index:
        raise ValueError("Detection plan clips must form ordered [0, clip_count).")
    clip_id = str(unit.get("clip_id") or "").strip()
    camera_serial = str(unit.get("camera_serial") or "").strip()
    paths = unit.get("zarr_paths")
    if not clip_id or not camera_serial or not isinstance(paths, Mapping):
        raise ValueError(f"Detection plan unit {expected_index} is incomplete.")
    group_path = _safe_group(
        paths.get("detect_target_group_path"),
        name=f"work_units[{expected_index}].detect_target_group_path",
    )
    run = archive[group_path]
    if not is_run_complete(run):
        raise ValueError(f"Legacy clip detection is not complete: {group_path}")
    if run.attrs.get("instance_key_backfill_status") != "complete":
        raise ValueError(f"Legacy clip lacks stable instance keys: {group_path}")
    if run.attrs.get("instance_key_backfill_recording_identity") != (
        recording_identity
    ):
        raise ValueError(f"Legacy clip binds a different recording: {group_path}")
    source_width = int(run.attrs.get("source_video_width") or 0)
    source_height = int(run.attrs.get("source_video_height") or 0)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Legacy clip lacks source dimensions: {group_path}")
    model_path = Path(str(run.attrs.get("model_path") or "")).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Legacy clip model is missing: {model_path}")
    if sha256_file(model_path) != expected_model_sha256:
        raise ValueError(f"Legacy clip model differs from the pin: {group_path}")

    arrays = {name: _array(run, name) for name in _ARRAY_NAMES}
    n_rows = int(arrays["frame_indices"].shape[0])
    parent_frames = load_parent_frame_mapping(
        recording_frame_index,
        camera_serial=camera_serial,
        clip_id=clip_id,
    )
    if int(unit.get("frame_count") or -1) != parent_frames.shape[0]:
        raise ValueError(f"Detection plan frame count differs for {clip_id}.")
    metadata_path = analysis_zarr / group_path / "zarr.json"
    metadata_sha = sha256_file(metadata_path)
    logical_hashes = {name: sha256_array(values) for name, values in arrays.items()}
    tree_sha = canonical_json_sha256(
        {
            "group_path": group_path,
            "metadata_sha256": metadata_sha,
            "logical_array_sha256": logical_hashes,
        }
    )
    member = ClippedDetectionArtifactMember(
        work_unit_id=f"{clip_id}:camera_{camera_serial}",
        artifact_run_id=group_path.rsplit("/", 1)[-1],
        clip_id=clip_id,
        clip_index=expected_index,
        camera_serial=camera_serial,
        source_width=source_width,
        source_height=source_height,
        artifact_manifest_sha256=metadata_sha,
        run_group_tree_sha256=tree_sha,
        parent_frame_indices=parent_frames,
        frame_indices=arrays["frame_indices"],
        bbox_norm_coords=arrays["bbox_norm_coords"],
        scores=arrays["scores"],
        class_ids=arrays["class_ids"],
        artifact_row_id=np.arange(n_rows, dtype=np.uint64),
        frame_counts=arrays["frame_counts"],
        n_detections=arrays["n_detections"],
    )
    evidence: dict[str, object] = {
        "clip_id": clip_id,
        "clip_index": expected_index,
        "camera_serial": camera_serial,
        "source_group_path": group_path,
        "source_group_metadata_sha256": metadata_sha,
        "source_group_logical_digest": tree_sha,
        "source_rows": n_rows,
        "source_frames": int(parent_frames.shape[0]),
    }
    return member, evidence, arrays["instance_key"]


def finalize_recording_canonical_detection_benchmark_adapter(
    *,
    analysis_zarr: Path,
    detection_plan_path: Path,
    recording_frame_index: Path,
    recording_identity: str,
    expected_model_sha256: str,
    expected_n_frames: int,
    expected_n_instances: int,
    destination: Path,
    benchmark_root: Path,
    run_id: str,
    scratch_parent: Path,
    coordinate_catalog: bool = False,
) -> dict[str, object]:
    """Rebuild and publish one standalone current canonical detection store."""

    started = time.perf_counter()
    phases: dict[str, float] = {}
    archive_path = analysis_zarr.expanduser().resolve()
    plan_path = detection_plan_path.expanduser().resolve()
    frame_index = recording_frame_index.expanduser().resolve()
    output = _require_benchmark_destination(
        destination,
        benchmark_root=benchmark_root,
    )
    model_sha = _require_sha256(expected_model_sha256, name="detection model")
    if type(coordinate_catalog) is not bool:
        raise TypeError("coordinate_catalog must be an exact bool.")
    if type(expected_n_frames) is not int or expected_n_frames <= 0:
        raise ValueError("expected_n_frames must be a positive exact integer.")
    if type(expected_n_instances) is not int or expected_n_instances < 0:
        raise ValueError("expected_n_instances must be a nonnegative exact integer.")
    if not archive_path.is_dir():
        raise FileNotFoundError("Analysis Zarr is missing.")
    plan = _read_json(plan_path)
    if plan.get("recording_id") != recording_identity:
        raise ValueError("Detection plan recording identity differs from the request.")
    if Path(str(plan.get("analysis_zarr") or "")).resolve() != archive_path:
        raise ValueError("Detection plan targets a different analysis archive.")
    units = plan.get("work_units")
    if not isinstance(units, list) or not units:
        raise ValueError("Detection plan lacks work_units.")
    if plan.get("work_unit_count") != len(units):
        raise ValueError("Detection plan work_unit_count is inconsistent.")

    load_started = time.perf_counter()
    archive = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    loaded = [
        _source_member(
            archive=archive,
            analysis_zarr=archive_path,
            recording_frame_index=frame_index,
            recording_identity=recording_identity,
            unit=unit,
            expected_index=index,
            expected_model_sha256=model_sha,
        )
        for index, unit in enumerate(units)
        if isinstance(unit, Mapping)
    ]
    if len(loaded) != len(units):
        raise ValueError("Every detection plan work unit must be an object.")
    members = tuple(item[0] for item in loaded)
    source_evidence = tuple(item[1] for item in loaded)
    source_keys = np.concatenate([item[2] for item in loaded])
    dimensions = {(item.source_width, item.source_height) for item in members}
    if len(dimensions) != 1:
        raise ValueError("Legacy detection clips have inconsistent source dimensions.")
    source_width, source_height = next(iter(dimensions))
    bound = bind_clipped_detection_artifacts(
        members,
        recording_identity=recording_identity,
        n_frames=expected_n_frames,
        source_width=source_width,
        source_height=source_height,
    )
    if bound.dimensions.n_instances != expected_n_instances:
        raise ValueError("Bound clip row count differs from expected_n_instances.")
    if not np.array_equal(
        source_keys,
        np.asarray(bound.arrays["instances/instance_key"]),
    ):
        raise ValueError("Rebuilt instance keys differ from persisted clip identities.")
    phases["load_and_bind_clip_sources"] = time.perf_counter() - load_started

    source_pixel_document = {
        "recording_identity": recording_identity,
        "source_width": source_width,
        "source_height": source_height,
        "clip_sources": list(source_evidence),
    }
    provenance: dict[str, object] = {
        "schema_id": ADAPTER_SCHEMA_ID,
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "benchmark_only": True,
        "recording_identity": recording_identity,
        "detection_plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "source_clips": list(source_evidence),
        "source_clips_digest": canonical_json_sha256(list(source_evidence)),
        "input_artifacts": [{"role": "detect_model", "sha256": model_sha}],
        "conversion": ("legacy_clip_detection_groups_to_native_canonical_v2_benchmark"),
        "inference_reexecuted": False,
        "production_state_changes": [],
    }
    frame_authority_sha = sha256_file(frame_index)
    scratch = _require_node_local_scratch(scratch_parent)
    with tempfile.TemporaryDirectory(
        prefix="palette_crimson_canonical_", dir=scratch
    ) as temporary_directory:
        local = Path(temporary_directory) / "canonical.zarr"
        write_started = time.perf_counter()
        candidate = write_native_clipped_detection_candidate(
            bound,
            destination=local,
            run_id=run_id,
            recording_identity=recording_identity,
            producer_id=(
                "fisheye.utils.finalize_recording_canonical_detection_benchmark_adapter"
            ),
            producer_version="v3",
            source_frame_authority={
                "record_ref": str(frame_index),
                "record_sha256": frame_authority_sha,
            },
            source_pixel_authority={
                "record_ref": "legacy_clip_detect_groups@source_pixel_dimensions",
                "record_sha256": canonical_json_sha256(source_pixel_document),
            },
            model_artifact_sha256=model_sha,
            run_provenance=provenance,
            coordinate_catalog=coordinate_catalog,
        )
        phases["node_local_publication"] = time.perf_counter() - write_started
        native_receipt_path = local / "native_detection_candidate_receipt.json"
        native_receipt = _read_json(native_receipt_path)
        native_receipt["output_path"] = str(output)
        write_json_atomic(native_receipt_path, native_receipt)
        adapter_payload: dict[str, object] = {
            "status": "complete",
            "recording_identity": recording_identity,
            "output_archive": str(output),
            "output_run_id": run_id,
            "n_frames": bound.dimensions.n_frames,
            "n_instances": bound.dimensions.n_instances,
            "source_width": bound.dimensions.source_width,
            "source_height": bound.dimensions.source_height,
            "run_manifest_digest": candidate.manifest["payload_digest"],
            "run_manifest_schema_version": candidate.manifest["schema_version"],
            "coordinate_catalog": coordinate_catalog,
            "storage_profile_id": candidate.plans.profile.profile_id,
            "detection_plan_sha256": sha256_file(plan_path),
            "recording_frame_index_sha256": frame_authority_sha,
            "detection_model_sha256": model_sha,
            "clip_source_digest": canonical_json_sha256(list(source_evidence)),
            "selector_eligible": False,
            "registry_registered": False,
            "production_state_changes": [],
            "inference_reexecuted": False,
        }
        adapter_receipt = {
            "schema_id": ADAPTER_SCHEMA_ID,
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "payload_digest": canonical_json_sha256(adapter_payload),
            "payload": adapter_payload,
        }
        write_json_atomic(local / ADAPTER_RECEIPT_NAME, adapter_receipt)
        local_stats = storage_stats(local)
        phases["publish_to_shared_storage"] = _publish_directory(local, output)

    reopened = load_native_canonical_detection_candidate(
        output,
        run_id=run_id,
        expected_manifest_schema_version=candidate.manifest["schema_version"],
    )
    if reopened.manifest["payload_digest"] != candidate.manifest["payload_digest"]:
        raise RuntimeError(
            "Published canonical manifest differs from node-local output."
        )
    return {
        "schema_id": ADAPTER_SCHEMA_ID,
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "output_archive": str(output),
        "output_run_id": run_id,
        "run_manifest_digest": reopened.manifest["payload_digest"],
        "run_manifest_schema_version": reopened.manifest["schema_version"],
        "coordinate_catalog": coordinate_catalog,
        "adapter_receipt_path": str(output / ADAPTER_RECEIPT_NAME),
        "adapter_receipt_digest": adapter_receipt["payload_digest"],
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "inference_reexecuted": False,
        "node_local_materialization": True,
        "native_clip_binding_validated": True,
        "local_store_stats": local_stats,
        "timing_seconds": phases,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes": peak_rss_bytes(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--detection-plan", type=Path, required=True)
    parser.add_argument("--recording-frame-index", type=Path, required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-n-frames", type=int, required=True)
    parser.add_argument("--expected-n-instances", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument(
        "--coordinate-catalog",
        action="store_true",
        help="Publish canonical run-manifest v3 with the exact coordinate catalog.",
    )
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scratch_parent = args.scratch_parent
    if scratch_parent is None:
        base = Path(os.environ.get("TMPDIR", "/tmp")).expanduser().resolve()
        scratch_parent = base / f"palette_{os.getuid()}"
        scratch_parent.mkdir(parents=True, exist_ok=True)
    try:
        result = finalize_recording_canonical_detection_benchmark_adapter(
            analysis_zarr=args.analysis_zarr,
            detection_plan_path=args.detection_plan,
            recording_frame_index=args.recording_frame_index,
            recording_identity=args.recording_identity,
            expected_model_sha256=args.expected_model_sha256,
            expected_n_frames=args.expected_n_frames,
            expected_n_instances=args.expected_n_instances,
            destination=args.destination,
            benchmark_root=args.benchmark_root,
            run_id=args.run_id,
            scratch_parent=scratch_parent,
            coordinate_catalog=args.coordinate_catalog,
        )
    except Exception as exc:
        result = {
            "schema_id": ADAPTER_SCHEMA_ID,
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "status": "failed",
            "output_archive": str(args.destination),
            "output_run_id": args.run_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_RECEIPT_NAME",
    "ADAPTER_SCHEMA_ID",
    "ADAPTER_SCHEMA_VERSION",
    "finalize_recording_canonical_detection_benchmark_adapter",
]
