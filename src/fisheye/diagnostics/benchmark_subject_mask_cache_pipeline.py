"""Run the bounded crop-cache -> masks -> quality integration pipeline.

All large computation and intermediate writes occur below a node-local scratch
root.  Only completed selector-ineligible stores and a terminal handoff are
copied to the benchmark namespace on shared storage.
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
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.flat_roi_cache import load_flat_roi_cache_manifest
from fisheye.shared.zarr.benchmark_runtime import storage_stats
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    publish_selector_ineligible_subject_mask_core_snapshot,
)
from fisheye.shared.zarr.subject_mask_quality_publication import (
    publish_selector_ineligible_subject_mask_quality_snapshot,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
)

SCHEMA_ID = "palette.subject_mask_cache_pipeline_benchmark"
SCHEMA_VERSION = 1
DEFAULT_COMPONENTS = (
    "subject_body",
    "eye_left",
    "eye_right",
    "swim_bladder",
)
DEFAULT_RAW_LABELS = ("subject_body", "eyes_union", "swim_bladder")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _peak_rss_bytes() -> int:
    own = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    children = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    factor = 1 if sys.platform == "darwin" else 1024
    return max(own, children) * factor


def _require_node_local(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = (Path("/groups"), Path("/nrs"), Path("/Volumes"))
    if any(resolved == prefix or prefix in resolved.parents for prefix in forbidden):
        raise ValueError(f"Scratch must be node-local, got {resolved}.")
    if resolved.exists():
        raise FileExistsError(f"Scratch path already exists: {resolved}")
    resolved.mkdir(parents=True)
    return resolved


def _require_existing_node_local(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = (Path("/groups"), Path("/nrs"), Path("/Volumes"))
    if any(resolved == prefix or prefix in resolved.parents for prefix in forbidden):
        raise ValueError(f"Resume scratch must be node-local, got {resolved}.")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Resume scratch does not exist: {resolved}")
    return resolved


def _require_destination(path: Path, *, benchmark_root: Path) -> Path:
    destination = path.expanduser().resolve()
    root = benchmark_root.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Destination must be below the benchmark root.") from exc
    if destination == root or destination.exists():
        raise FileExistsError(
            f"Benchmark destination is invalid or exists: {destination}"
        )
    return destination


def _copytree(source: Path, destination: Path) -> float:
    started = time.perf_counter()
    shutil.copytree(source, destination, symlinks=False)
    return time.perf_counter() - started


def _stage_cache(
    source_manifest: Path, destination: Path
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    payload = load_flat_roi_cache_manifest(source_manifest)
    array = payload.get("array")
    if not isinstance(array, Mapping):
        raise ValueError("Flat ROI cache lacks array metadata.")
    raw_path = Path(str(array.get("bin_path") or ""))
    if not raw_path.is_absolute():
        raw_path = source_manifest.parent / raw_path
    raw_path = raw_path.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    destination.mkdir(parents=True)
    local_bin = destination / raw_path.name
    local_manifest = destination / source_manifest.name
    shutil.copy2(raw_path, local_bin)
    declared_payload_sha256 = str(array.get("sha256") or "")
    if len(declared_payload_sha256) != 64:
        raise ValueError("Flat ROI cache lacks an exact payload SHA-256.")
    local_payload_sha256 = _sha256_file(local_bin)
    if local_payload_sha256 != declared_payload_sha256:
        raise RuntimeError("Staged flat ROI cache payload hash mismatch.")
    staged = dict(payload)
    staged_array = dict(array)
    staged_array["bin_path"] = local_bin.name
    staged["array"] = staged_array
    staged["manifest_path"] = str(local_manifest)
    staged["benchmark_staging"] = {
        "policy": "copy_complete_cache_to_node_local_before_compute",
        "source_manifest": str(source_manifest),
        "source_payload": str(raw_path),
        "source_manifest_sha256": _sha256_file(source_manifest),
        "source_payload_sha256": declared_payload_sha256,
        "effective_manifest": str(local_manifest),
        "effective_payload": str(local_bin),
    }
    _write_json(local_manifest, staged)
    elapsed = time.perf_counter() - started
    return local_manifest, {
        "seconds": elapsed,
        "bytes": int(local_bin.stat().st_size),
        "mib_per_second": (
            (int(local_bin.stat().st_size) / (1024 * 1024)) / elapsed
            if elapsed > 0
            else None
        ),
        "source_manifest_sha256": _sha256_file(source_manifest),
        "payload_sha256": local_payload_sha256,
    }


def _resume_cache(local_manifest: Path) -> tuple[Path, dict[str, Any]]:
    payload = load_flat_roi_cache_manifest(local_manifest)
    array = payload.get("array")
    if not isinstance(array, Mapping):
        raise ValueError("Resumed flat ROI cache lacks array metadata.")
    local_bin = Path(str(array.get("bin_path") or ""))
    if not local_bin.is_absolute():
        local_bin = local_manifest.parent / local_bin
    local_bin = local_bin.resolve()
    declared = str(array.get("sha256") or "")
    if len(declared) != 64 or not local_bin.is_file():
        raise ValueError("Resumed flat ROI cache is incomplete.")
    started = time.perf_counter()
    observed = _sha256_file(local_bin)
    if observed != declared:
        raise RuntimeError("Resumed flat ROI cache payload hash mismatch.")
    return local_manifest, {
        "resumed": True,
        "verification_seconds": time.perf_counter() - started,
        "bytes": int(local_bin.stat().st_size),
        "payload_sha256": observed,
    }


def _run_logged(command: Sequence[str], *, log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result = {
        "command": list(command),
        "log_path": str(log_path),
        "returncode": int(completed.returncode),
        "seconds": time.perf_counter() - started,
    }
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            + "\n".join(tail)
        )
    return result


def _copy_keypoint_family(source_zarr: Path, local_zarr: Path) -> float:
    source = source_zarr / "refined_keypoints_runs"
    destination = local_zarr / "refined_keypoints_runs"
    if not source.is_dir():
        raise ValueError(f"Missing refined_keypoints_runs in {source_zarr}.")
    if destination.exists():
        raise ValueError("Local crop archive already has refined_keypoints_runs.")
    return _copytree(source, destination)


def _paths(group: Any, names: Sequence[str]) -> dict[str, Any]:
    missing = [name for name in names if name not in group]
    if missing:
        raise ValueError(f"Group {group.path!r} lacks exact arrays: {missing!r}.")
    return {name: group[name] for name in names}


def _source_crop_arrays(crop: Any) -> dict[str, Any]:
    return _paths(
        crop,
        ("instance_key", "source_acquisition_frame_index", "source_crop_xywh"),
    )


def _raw_arrays(shard: Any, crop: Any, *, n_frames: int) -> dict[str, Any]:
    rows = np.asarray(shard["source_crop_row_ids"][...], dtype=np.int64)
    expected = np.arange(int(crop["instance_key"].shape[0]), dtype=np.int64)
    if not np.array_equal(rows, expected):
        raise ValueError("Raw shard does not cover the complete crop rowset in order.")
    frames = np.asarray(shard["source_acquisition_frame_index"][...], dtype=np.int64)
    return {
        **_paths(
            shard,
            (
                "source_crop_row_ids",
                "instance_key",
                "source_acquisition_frame_index",
                "mask_probs_roi",
                "available_channels",
                "metrics/prob_max",
                "metrics/mask_present",
                "metrics/area_px",
                "metrics/centroid_xy",
                "metrics/centroid_valid",
                "metrics/bbox_xyxy",
                "metrics/bbox_valid",
            ),
        ),
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(
            frames, n_frames=n_frames
        ),
        "source_crop_xywh": crop["source_crop_xywh"],
    }


def _refined_arrays(run: Any, crop: Any, *, n_frames: int) -> dict[str, Any]:
    payload_names = (
        "source_crop_row_ids",
        "masks_roi",
        "available_channels",
        "metrics/mask_present",
        "metrics/area_px",
        "metrics/centroid_xy",
        "metrics/centroid_valid",
        "metrics/bbox_xyxy",
        "metrics/bbox_valid",
    )
    payload = _paths(run, payload_names)
    rows = np.asarray(payload["source_crop_row_ids"][...], dtype=np.int64)
    expected = np.arange(int(crop["instance_key"].shape[0]), dtype=np.int64)
    if not np.array_equal(rows, expected):
        raise ValueError(
            "Refined draft does not cover the complete crop rowset in order."
        )
    frames = np.asarray(crop["source_acquisition_frame_index"][rows], dtype=np.int64)
    return {
        **payload,
        "instance_key": crop["instance_key"],
        "source_acquisition_frame_index": frames,
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(
            frames,
            n_frames=n_frames,
        ),
        "source_crop_xywh": crop["source_crop_xywh"],
    }


def _strict_attribute_subset(
    attrs: Mapping[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name in names:
        if name not in attrs:
            continue
        value = attrs[name]
        json.dumps(value, allow_nan=False)
        selected[name] = value
    return selected


def _bound_run_manifest(group: Any, *, label: str) -> dict[str, Any]:
    manifest = group.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{label} lacks its required persisted run_manifest.")
    document = dict(manifest)
    declared = str(document.get("payload_digest") or "")
    payload = document.get("payload")
    if len(declared) != 64 or not isinstance(payload, Mapping):
        raise ValueError(f"{label} has a malformed persisted run_manifest.")
    if canonical_json_sha256(payload) != declared:
        raise ValueError(f"{label} run_manifest payload digest is invalid.")
    return document


def _source_preflight(
    *,
    crop_source: Path,
    crop_run: str,
    keypoint_source: Path,
    refined_keypoint_run: str,
    cache_source: Path,
) -> dict[str, Any]:
    cache = load_flat_roi_cache_manifest(cache_source)
    cache_array = cache.get("array")
    cache_binding = cache.get("source")
    if not isinstance(cache_array, Mapping) or not isinstance(cache_binding, Mapping):
        raise ValueError("Flat ROI cache lacks its array/source binding.")
    if Path(str(cache_binding.get("archive_path") or "")).resolve() != crop_source:
        raise ValueError("Flat ROI cache is bound to a different crop archive.")
    if str(cache_binding.get("crop_run_name") or "") != crop_run:
        raise ValueError("Flat ROI cache is bound to a different crop run.")

    crop_root = zarr.open_group(str(crop_source), mode="r", use_consolidated=True)
    keypoint_root = zarr.open_group(
        str(keypoint_source), mode="r", use_consolidated=True
    )
    crop = crop_root[f"crop_runs/{crop_run}"]
    keypoints = keypoint_root[f"refined_keypoints_runs/{refined_keypoint_run}"]
    crop_manifest = _bound_run_manifest(crop, label=f"crop_runs/{crop_run}")
    keypoint_manifest = _bound_run_manifest(
        keypoints,
        label=f"refined_keypoints_runs/{refined_keypoint_run}",
    )
    n_rows = int(crop["instance_key"].shape[0])
    n_frames = int(crop["frame_row_offsets"].shape[0]) - 1
    shape = tuple(int(value) for value in cache_array.get("shape", []))
    if len(shape) != 3 or shape[0] != n_rows:
        raise ValueError("Flat ROI cache shape differs from the crop-v2 rowset.")
    if np.dtype(crop["source_crop_xywh"].dtype) != np.dtype(np.float32):
        raise ValueError("Crop-v2 placement must use exact float32.")
    if int(keypoints["instance_key"].shape[0]) != n_rows:
        raise ValueError("Refined keypoint and crop row counts differ.")
    exact_pairs = (
        ("instance_key", "instance_key"),
        ("source_acquisition_frame_index", "source_acquisition_frame_index"),
        ("frame_row_offsets", "frame_row_offsets"),
    )
    for crop_name, keypoint_name in exact_pairs:
        crop_values = np.asarray(crop[crop_name][...])
        keypoint_values = np.asarray(keypoints[keypoint_name][...])
        if crop_values.dtype != keypoint_values.dtype or not np.array_equal(
            crop_values, keypoint_values
        ):
            raise ValueError(
                f"Crop and refined-keypoint {crop_name} identities differ."
            )
    crop_rows = np.asarray(keypoints["source_crop_row_ids"][...], dtype=np.int64)
    if not np.array_equal(crop_rows, np.arange(n_rows, dtype=np.int64)):
        raise ValueError("Refined keypoints do not cover the crop rowset in order.")
    return {
        "n_frames": n_frames,
        "n_rows": n_rows,
        "roi_shape": list(shape[1:]),
        "crop_run_manifest_digest": canonical_json_sha256(crop_manifest),
        "refined_keypoint_run_manifest_digest": canonical_json_sha256(
            keypoint_manifest
        ),
        "identity_checks": [
            "instance_key",
            "source_acquisition_frame_index",
            "frame_row_offsets",
            "source_crop_row_ids",
        ],
    }


def _percentile_ms(values: Sequence[float], percentile: float) -> float:
    return float(
        np.percentile(np.asarray(values, dtype=np.float64), percentile) * 1000.0
    )


def _read_benchmark(
    *,
    raw_store: Path,
    raw_run: str,
    refined_store: Path,
    refined_run: str,
    quality_store: Path,
    quality_run: str,
) -> dict[str, Any]:
    stores = {
        "raw": (
            raw_store,
            f"subject_mask_runs/{raw_run}",
            "mask_probs_roi",
        ),
        "refined": (
            refined_store,
            f"refined_subject_masks_runs/{refined_run}",
            "masks_roi",
        ),
        "quality": (
            quality_store,
            f"subject_mask_quality_runs/{quality_run}",
            "observation_quality_flags",
        ),
    }
    result: dict[str, Any] = {}
    rng = np.random.default_rng(20260730)
    for name, (path, run_path, payload_path) in stores.items():
        opened = time.perf_counter()
        # These stores were created earlier in this process.  Their publishers
        # have already gated direct/consolidated declaration equivalence, while
        # this phase is intended to measure payload reads rather than metadata
        # traversal.  Open the exact mutable-local run through direct metadata:
        # nested Zarr v3 group envelopes may otherwise retain an empty inline
        # consolidated map and hide arrays that are present in direct metadata.
        run = zarr.open_group(
            str(path / run_path),
            mode="r",
            use_consolidated=False,
        )
        open_seconds = time.perf_counter() - opened
        started = time.perf_counter()
        offsets = np.asarray(run["frame_row_offsets"][...], dtype=np.int64)
        offset_seconds = time.perf_counter() - started
        frames = rng.integers(0, offsets.shape[0] - 1, size=64, endpoint=False)
        latencies: list[float] = []
        logical_bytes = 0
        payload = run[payload_path]
        for frame in frames.tolist():
            start = int(offsets[frame])
            stop = int(offsets[frame + 1])
            phase = time.perf_counter()
            values = np.asarray(payload[start:stop])
            latencies.append(time.perf_counter() - phase)
            logical_bytes += int(values.nbytes)
        result[name] = {
            "open_seconds": open_seconds,
            "offset_read_seconds": offset_seconds,
            "offset_reads": 1,
            "random_frame_count": len(latencies),
            "random_frame_median_ms": _percentile_ms(latencies, 50),
            "random_frame_p95_ms": _percentile_ms(latencies, 95),
            "random_frame_logical_bytes": logical_bytes,
        }
    return result


def _validate_published_metadata_equivalence(
    store: Path, *, run_path: str
) -> dict[str, Any]:
    """Revalidate every direct declaration against root consolidation."""

    root_document = _strict_json(store / "zarr.json")
    envelope = root_document.get("consolidated_metadata")
    if not isinstance(envelope, Mapping) or envelope.get("kind") != "inline":
        raise ValueError(f"{store} lacks inline consolidated metadata.")
    consolidated = envelope.get("metadata")
    if not isinstance(consolidated, Mapping):
        raise ValueError(f"{store} has malformed consolidated metadata.")

    direct: dict[str, dict[str, Any]] = {}
    run_root = store / run_path
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    for metadata_path in sorted(run_root.rglob("zarr.json")):
        relative = metadata_path.parent.relative_to(store).as_posix()
        direct[relative] = _strict_json(metadata_path)
    consolidated_scope = {
        str(path): dict(value)
        for path, value in consolidated.items()
        if str(path) == run_path or str(path).startswith(f"{run_path}/")
        if isinstance(value, Mapping)
    }
    if set(direct) != set(consolidated_scope):
        missing = sorted(set(direct) - set(consolidated_scope))
        unexpected = sorted(set(consolidated_scope) - set(direct))
        raise ValueError(
            f"{run_path} direct/consolidated path inventory differs: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for path, document in direct.items():
        direct_document = metadata_without_empty_group_consolidation(
            document,
            path=path,
        )
        consolidated_document = metadata_without_empty_group_consolidation(
            consolidated_scope[path],
            path=path,
        )
        if direct_document != consolidated_document:
            raise ValueError(
                f"{path} direct and consolidated metadata declarations differ."
            )
        normalized[path] = direct_document
    return {
        "declaration_count": len(normalized),
        "normalized_digest": canonical_json_sha256(normalized),
    }


def _resume_published_pipeline(
    args: argparse.Namespace, scratch: Path, local_output: Path
) -> dict[str, Any]:
    """Finish a failed diagnostic handoff from already gated local stores."""

    crop_source = args.source_crop_zarr.expanduser().resolve()
    keypoint_source = args.source_refined_keypoint_zarr.expanduser().resolve()
    cache_source = args.roi_cache_manifest.expanduser().resolve()
    model_source = args.subject_mask_model.expanduser().resolve()
    source_preflight = _source_preflight(
        crop_source=crop_source,
        crop_run=args.crop_run,
        keypoint_source=keypoint_source,
        refined_keypoint_run=args.refined_keypoint_run,
        cache_source=cache_source,
    )
    source_hashes = {
        "crop_root_metadata": _sha256_file(crop_source / "zarr.json"),
        "keypoint_root_metadata": _sha256_file(keypoint_source / "zarr.json"),
        "cache_manifest": _sha256_file(cache_source),
        "model": _sha256_file(model_source),
    }
    specs = {
        "raw": (
            local_output / "raw.zarr",
            f"subject_mask_runs/{args.raw_run}",
            "palette.subject_mask_core.run_manifest",
            2,
        ),
        "refined": (
            local_output / "refined.zarr",
            f"refined_subject_masks_runs/{args.refined_run}",
            "palette.subject_mask_core.run_manifest",
            2,
        ),
        "quality": (
            local_output / "quality.zarr",
            f"subject_mask_quality_runs/{args.quality_run}",
            "palette.subject_mask_quality.run_manifest",
            1,
        ),
    }
    manifests: dict[str, dict[str, Any]] = {}
    groups: dict[str, Any] = {}
    metadata_receipts: dict[str, dict[str, Any]] = {}
    for name, (store, run_path, schema_id, schema_version) in specs.items():
        metadata_receipts[name] = _validate_published_metadata_equivalence(
            store,
            run_path=run_path,
        )
        run = zarr.open_group(
            str(store / run_path),
            mode="r",
            use_consolidated=False,
        )
        manifest = _bound_run_manifest(run, label=run_path)
        if manifest.get("schema_id") != schema_id or manifest.get(
            "schema_version"
        ) != schema_version:
            raise ValueError(f"{run_path} has the wrong run-manifest contract.")
        payload = manifest["payload"]
        publication = payload.get("publication")
        logical_schema = payload.get("logical_schema")
        if not isinstance(publication, Mapping) or not isinstance(
            logical_schema, Mapping
        ):
            raise ValueError(f"{run_path} has an incomplete persisted manifest.")
        if payload.get("run_id") != run_path.rsplit("/", 1)[-1]:
            raise ValueError(f"{run_path} run identity differs from its path.")
        if (
            publication.get("completion_status") != "complete"
            or publication.get("stage_selector_eligible") is not False
            or publication.get("metadata_state")
            != "direct_and_consolidated_validated"
            or run.attrs.get("palette_run_completion_status") != "complete"
            or run.attrs.get("stage_selector_eligible") is not False
            or run.attrs.get("shadow_only") is not True
        ):
            raise ValueError(f"{run_path} is not a completed selector-ineligible run.")
        dimensions = logical_schema.get("dimensions")
        if not isinstance(dimensions, Mapping) or (
            int(dimensions.get("n_frames", -1)) != source_preflight["n_frames"]
            or int(dimensions.get("n_rois", -1)) != source_preflight["n_rows"]
        ):
            raise ValueError(f"{run_path} dimensions differ from source preflight.")
        bindings = logical_schema.get("bindings")
        if not isinstance(bindings, list):
            raise ValueError(f"{run_path} lacks its exact logical bindings.")
        missing = [
            str(binding.get("path"))
            for binding in bindings
            if isinstance(binding, Mapping)
            and binding.get("required") is True
            and str(binding.get("path")) not in run
        ]
        if missing:
            raise ValueError(f"{run_path} lacks required arrays: {missing!r}.")
        groups[name] = run
        manifests[name] = manifest

    expected_keys = np.asarray(groups["refined"]["instance_key"][...])
    expected_frames = np.asarray(
        groups["refined"]["source_acquisition_frame_index"][...],
        dtype=np.int64,
    )
    expected_offsets = derive_subject_mask_frame_row_offsets(
        expected_frames,
        n_frames=int(source_preflight["n_frames"]),
    )
    for name, run in groups.items():
        if not np.array_equal(np.asarray(run["instance_key"][...]), expected_keys):
            raise ValueError(f"{name} instance keys differ during resume.")
        if not np.array_equal(
            np.asarray(run["source_acquisition_frame_index"][...], dtype=np.int64),
            expected_frames,
        ):
            raise ValueError(f"{name} frame identities differ during resume.")
        if not np.array_equal(
            np.asarray(run["frame_row_offsets"][...], dtype=np.int64),
            expected_offsets,
        ):
            raise ValueError(f"{name} frame offsets differ during resume.")

    quality_source = manifests["quality"]["payload"].get(
        "source_refined_subject_mask_snapshot"
    )
    if not isinstance(quality_source, Mapping) or quality_source.get(
        "manifest_digest"
    ) != canonical_json_sha256(manifests["refined"]):
        raise ValueError("Quality snapshot is not bound to the refined manifest.")

    phases: dict[str, Any] = {
        "source_preflight": source_preflight,
        "resume_after_publication": {
            "source_job_id": args.resume_source_job_id,
            "source_palette_commit": args.resume_source_palette_commit,
            "policy": "reuse_only_completed_gated_node_local_stores",
            "metadata": metadata_receipts,
        },
        "node_local_reads": _read_benchmark(
            raw_store=specs["raw"][0],
            raw_run=args.raw_run,
            refined_store=specs["refined"][0],
            refined_run=args.refined_run,
            quality_store=specs["quality"][0],
            quality_run=args.quality_run,
        ),
    }
    evidence_dir = local_output / "evidence"
    if evidence_dir.exists():
        raise FileExistsError(
            "Post-publication resume requires no pre-existing evidence directory."
        )
    evidence_dir.mkdir()
    evidence_paths = {
        "raw_inference_log": scratch / "raw_inference.log",
        "refinement_log": scratch / "refinement.log",
        "refinement_progress": scratch / "refinement_progress.jsonl",
    }
    published_evidence: dict[str, str] = {}
    for name, source_path in evidence_paths.items():
        if not source_path.is_file():
            continue
        target = evidence_dir / source_path.name
        shutil.copy2(source_path, target)
        published_evidence[name] = str(target.relative_to(local_output))

    def summary(name: str) -> dict[str, Any]:
        payload = manifests[name]["payload"]
        return {
            "run_id": payload["run_id"],
            "manifest_digest": manifests[name]["payload_digest"],
            "logical_content_digest": payload["logical_content"]["digest"],
            "storage": payload["storage_plan"]["object_estimate"],
        }

    return {
        "source_hashes": source_hashes,
        "n_frames": int(source_preflight["n_frames"]),
        "n_rows": int(source_preflight["n_rows"]),
        "raw": summary("raw"),
        "refined": summary("refined"),
        "quality": summary("quality"),
        "phases": phases,
        "evidence": published_evidence,
        "stores": {
            name: storage_stats(local_output / f"{name}.zarr")
            for name in ("raw", "refined", "quality")
        },
    }


def _run_pipeline(
    args: argparse.Namespace, scratch: Path, local_output: Path
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    crop_source = args.source_crop_zarr.expanduser().resolve()
    keypoint_source = args.source_refined_keypoint_zarr.expanduser().resolve()
    cache_source = args.roi_cache_manifest.expanduser().resolve()
    model_source = args.subject_mask_model.expanduser().resolve()
    for path in (
        crop_source / "zarr.json",
        keypoint_source / "zarr.json",
        cache_source,
        model_source,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    phases["source_preflight"] = _source_preflight(
        crop_source=crop_source,
        crop_run=args.crop_run,
        keypoint_source=keypoint_source,
        refined_keypoint_run=args.refined_keypoint_run,
        cache_source=cache_source,
    )
    source_hashes_before = {
        "crop_root_metadata": _sha256_file(crop_source / "zarr.json"),
        "keypoint_root_metadata": _sha256_file(keypoint_source / "zarr.json"),
        "cache_manifest": _sha256_file(cache_source),
        "model": _sha256_file(model_source),
    }

    local_archive = scratch / "working_analysis.zarr"
    local_cache = scratch / "roi_cache" / cache_source.name
    local_model = scratch / "model" / model_source.name
    raw_shard_run = args.raw_shard_run
    inference_log = scratch / "raw_inference.log"
    if args.resume_after_raw_inference:
        for path in (
            local_archive / "zarr.json",
            local_cache,
            local_model,
            inference_log,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Resume input is absent: {path}")
        local_cache, cache_staging = _resume_cache(local_cache)
        phases["stage_crop_archive"] = {"resumed": True}
        phases["stage_refined_keypoints"] = {"resumed": True}
        phases["stage_roi_cache"] = cache_staging
        local_model_hash = _sha256_file(local_model)
        if local_model_hash != source_hashes_before["model"]:
            raise RuntimeError("Resumed subject-mask model hash mismatch.")
        phases["stage_model"] = {
            "resumed": True,
            "sha256": local_model_hash,
        }
        phases["raw_inference"] = {
            "resumed": True,
            "source_job_id": args.resume_source_job_id,
            "source_palette_commit": args.resume_source_palette_commit,
            "log_path": str(inference_log),
            "log_sha256": _sha256_file(inference_log),
        }
    else:
        phases["stage_crop_archive_seconds"] = _copytree(crop_source, local_archive)
        phases["stage_refined_keypoints_seconds"] = _copy_keypoint_family(
            keypoint_source, local_archive
        )
        local_cache, cache_staging = _stage_cache(cache_source, scratch / "roi_cache")
        phases["stage_roi_cache"] = cache_staging
        local_model.parent.mkdir()
        started = time.perf_counter()
        shutil.copy2(model_source, local_model)
        phases["stage_model_seconds"] = time.perf_counter() - started
        if _sha256_file(local_model) != source_hashes_before["model"]:
            raise RuntimeError("Staged subject-mask model hash mismatch.")

        inference_command = [
            sys.executable,
            "-m",
            "fisheye.segmentation.infer_unet_subject_masks",
            str(local_archive),
            str(local_model),
            "--run-name",
            raw_shard_run,
            "--output-parent",
            "subject_mask_shard_runs",
            "--crop-run",
            args.crop_run,
            "--roi-cache-manifest",
            str(local_cache),
            "--roi-cache-expected-archive-path",
            str(crop_source),
            "--roi-cache-policy",
            "never",
            "--source-shard-id",
            "complete_recording_rowset",
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--model-input-size",
            "512",
            "--model-input-transform",
            "auto",
            "--mask-probs-dtype",
            "uint8",
            "--mask-probs-chunk-rois",
            "32",
            "--mask-probs-shard-rois",
            "2048",
            "--no-write-masks-roi",
            "--async-output",
            "--output-queue-size",
            "2",
            "--profile-timings",
            "--defer-registry-status",
            "--no-progress",
        ]
        phases["raw_inference"] = _run_logged(inference_command, log_path=inference_log)

    local_root = zarr.open_group(str(local_archive), mode="r", use_consolidated=False)
    crop = local_root[f"crop_runs/{args.crop_run}"]
    crop_manifest = _bound_run_manifest(
        crop,
        label=f"crop_runs/{args.crop_run}",
    )
    refined_keypoint = local_root[f"refined_keypoints_runs/{args.refined_keypoint_run}"]
    refined_keypoint_manifest = _bound_run_manifest(
        refined_keypoint,
        label=f"refined_keypoints_runs/{args.refined_keypoint_run}",
    )
    n_frames = int(crop["frame_row_offsets"].shape[0]) - 1
    shard = local_root[f"subject_mask_shard_runs/{raw_shard_run}"]
    if shard.attrs.get("palette_run_completion_status") != "complete":
        raise ValueError("Raw subject-mask shard is not complete.")
    if (
        tuple(str(value) for value in shard.attrs.get("mask_labels", []))
        != DEFAULT_RAW_LABELS
    ):
        raise ValueError("Raw U-Net output labels differ from subject_v1_union.")
    raw_source_manifest = {
        "schema_id": "palette.subject_mask.raw_inference_receipt",
        "schema_version": 1,
        "run_path": f"subject_mask_shard_runs/{raw_shard_run}",
        "completion_status": str(
            shard.attrs.get("palette_run_completion_status") or ""
        ),
        "cache_manifest_sha256": source_hashes_before["cache_manifest"],
        "cache_payload_sha256": cache_staging["payload_sha256"],
        "model_sha256": source_hashes_before["model"],
        "source_crop": {
            "run_path": f"crop_runs/{args.crop_run}",
            "run_manifest_digest": canonical_json_sha256(crop_manifest),
            "run_manifest_payload_digest": crop_manifest["payload_digest"],
        },
        "row_count": int(shard["mask_probs_roi"].shape[0]),
        "mask_labels": list(DEFAULT_RAW_LABELS),
        "inference_stage_provenance": dict(shard.attrs.get("provenance") or {}),
        "inference_run_provenance": dict(
            shard.attrs.get("run_provenance") or {}
        ),
    }
    raw_attributes = _strict_attribute_subset(
        shard.attrs,
        (
            "mask_labels",
            "label_schema_id",
            "mask_probability_threshold",
            "source_crop_run",
            "source_crop_storage_mode",
            "source_crop_signature",
            "source_crop_revision",
            "source_roi_read_mode",
            "source_roi_cache_used",
            "source_roi_cache_backend",
            "source_roi_cache_key",
            "source_checkpoint",
            "subject_mask_model_artifact",
            "model_input_transform",
            "model_input_transform_name",
            "model_input_shape_hw",
            "native_roi_shape_hw",
            "probability_semantics",
            "probabilities_encoding",
            "output_semantics",
            "overlap_policy",
        ),
    )
    raw_store = local_output / "raw.zarr"
    started = time.perf_counter()
    raw_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        _raw_arrays(shard, crop, n_frames=n_frames),
        source_crop_arrays=_source_crop_arrays(crop),
        source_manifest=raw_source_manifest,
        n_frames=n_frames,
        components=SubjectMaskComponentRegistry(DEFAULT_RAW_LABELS),
        destination=raw_store,
        run_id=args.raw_run,
        kind="raw_probability_uint8",
        source_run_path=f"subject_mask_shard_runs/{raw_shard_run}",
        source_attributes=raw_attributes,
        threshold=0.5,
        created_by="benchmark_subject_mask_cache_pipeline",
    )
    phases["raw_core_publication"] = {
        "seconds": time.perf_counter() - started,
        "internal": dict(raw_publication.phase_seconds),
    }

    del shard
    del crop
    del refined_keypoint
    del local_root
    refined_draft_run = args.refined_draft_run
    finalizer_log = scratch / "refinement.log"
    finalizer_command = [
        sys.executable,
        "-m",
        "fisheye.refinement.finalize_subject_masks",
        str(local_archive),
        "--subject-shard-run",
        raw_shard_run,
        "--target-crop-run",
        args.crop_run,
        "--refined-run",
        refined_draft_run,
        "--component",
        "subject_body",
        "--component",
        "eyes_union",
        "--component",
        "swim_bladder",
        "--chunk-size",
        "256",
        "--metric-level",
        "cheap",
        "--mask-storage",
        "dense_uint8",
        "--dense-mask-row-chunk",
        "128",
        "--execution-backend",
        "process_shards",
        "--num-workers",
        str(args.finalize_workers),
        "--postcompute-backend",
        "process_shards",
        "--postcompute-num-workers",
        str(args.finalize_workers),
        "--postcompute-chunk-size",
        "256",
        "--assignment-keypoint-group",
        "refined_keypoints_runs",
        "--assignment-keypoints-run",
        args.refined_keypoint_run,
        "--write-eye-geometry",
        "--no-write-component-contours",
        "--write-sampled-component-contours",
        "--defer-registry-status",
        "--progress-jsonl",
        str(scratch / "refinement_progress.jsonl"),
        "--json",
    ]
    phases["dense_refinement"] = _run_logged(finalizer_command, log_path=finalizer_log)

    local_root = zarr.open_group(str(local_archive), mode="r", use_consolidated=False)
    crop = local_root[f"crop_runs/{args.crop_run}"]
    refined_draft = local_root[f"refined_subject_masks_runs/{refined_draft_run}"]
    refined_source_manifest = {
        "schema_id": "palette.subject_mask.refinement_receipt",
        "schema_version": 1,
        "run_path": f"refined_subject_masks_runs/{refined_draft_run}",
        "source_raw_manifest_digest": raw_publication.manifest["payload_digest"],
        "source_raw_manifest_document_digest": canonical_json_sha256(
            raw_publication.manifest
        ),
        "source_crop": {
            "run_path": f"crop_runs/{args.crop_run}",
            "run_manifest_digest": canonical_json_sha256(crop_manifest),
            "run_manifest_payload_digest": crop_manifest["payload_digest"],
        },
        "source_refined_keypoints": {
            "run_path": f"refined_keypoints_runs/{args.refined_keypoint_run}",
            "run_manifest_digest": canonical_json_sha256(refined_keypoint_manifest),
            "run_manifest_payload_digest": refined_keypoint_manifest["payload_digest"],
            "source_root_metadata_sha256": source_hashes_before[
                "keypoint_root_metadata"
            ],
        },
        "completion_status": str(
            refined_draft.attrs.get("palette_run_completion_status") or ""
        ),
        "row_count": int(refined_draft["masks_roi"].shape[0]),
    }
    refined_attributes = _strict_attribute_subset(
        refined_draft.attrs,
        (
            "mask_labels",
            "source_crop_run",
            "source_subject_mask_run",
            "source_refined_keypoints_run",
            "mask_probability_threshold",
            "mask_storage_encoding",
            "source_roi_pixel_contract_name",
            "source_roi_pixel_contract",
        ),
    )
    refined_store = local_output / "refined.zarr"
    started = time.perf_counter()
    refined_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        _refined_arrays(refined_draft, crop, n_frames=n_frames),
        source_crop_arrays=_source_crop_arrays(crop),
        source_manifest=refined_source_manifest,
        n_frames=n_frames,
        components=SubjectMaskComponentRegistry(DEFAULT_COMPONENTS),
        destination=refined_store,
        run_id=args.refined_run,
        kind="refined_dense_core",
        source_run_path=f"refined_subject_masks_runs/{refined_draft_run}",
        source_attributes=refined_attributes,
        created_by="benchmark_subject_mask_cache_pipeline",
    )
    phases["refined_core_publication"] = {
        "seconds": time.perf_counter() - started,
        "internal": dict(refined_publication.phase_seconds),
    }

    refined_root = zarr.open_group(str(refined_store), mode="r", use_consolidated=False)
    refined_run = refined_root[f"refined_subject_masks_runs/{args.refined_run}"]
    dense_sha = str(
        refined_publication.manifest["payload"]["logical_content"]["document"][
            "arrays"
        ]["masks_roi"]["sha256"]
    )
    components = SubjectMaskComponentRegistry(DEFAULT_COMPONENTS)
    quality_source = SubjectMaskQualitySourceReference(
        run_name=args.refined_run,
        manifest_digest=canonical_json_sha256(refined_publication.manifest),
        dense_array_values_sha256=dense_sha,
        component_registry_digest=canonical_json_sha256(components.as_manifest()),
    )
    quality_store = local_output / "quality.zarr"
    started = time.perf_counter()
    quality_publication = publish_selector_ineligible_subject_mask_quality_snapshot(
        {
            "masks_roi": refined_run["masks_roi"],
            "instance_key": refined_run["instance_key"],
            "source_acquisition_frame_index": refined_run[
                "source_acquisition_frame_index"
            ],
            "available_channels": refined_run["available_channels"],
        },
        n_frames=n_frames,
        components=components,
        source=quality_source,
        source_manifest=refined_publication.manifest,
        destination=quality_store,
        run_id=args.quality_run,
        shadow_root=local_output,
        scratch_root=scratch / "quality_scratch",
        created_by="benchmark_subject_mask_cache_pipeline",
    )
    phases["quality_publication"] = {
        "seconds": time.perf_counter() - started,
        "internal": dict(quality_publication.phase_seconds),
    }

    phases["node_local_reads"] = _read_benchmark(
        raw_store=raw_store,
        raw_run=args.raw_run,
        refined_store=refined_store,
        refined_run=args.refined_run,
        quality_store=quality_store,
        quality_run=args.quality_run,
    )
    source_hashes_after = {
        "crop_root_metadata": _sha256_file(crop_source / "zarr.json"),
        "keypoint_root_metadata": _sha256_file(keypoint_source / "zarr.json"),
        "cache_manifest": _sha256_file(cache_source),
        "model": _sha256_file(model_source),
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A benchmark source changed during execution.")
    evidence_dir = local_output / "evidence"
    evidence_dir.mkdir()
    evidence_paths = {
        "raw_inference_log": inference_log,
        "refinement_log": finalizer_log,
        "refinement_progress": scratch / "refinement_progress.jsonl",
    }
    published_evidence: dict[str, str] = {}
    for name, source_path in evidence_paths.items():
        if not source_path.is_file():
            continue
        target = evidence_dir / source_path.name
        shutil.copy2(source_path, target)
        published_evidence[name] = str(target.relative_to(local_output))
    phases["raw_inference"]["log_path"] = published_evidence.get("raw_inference_log")
    phases["dense_refinement"]["log_path"] = published_evidence.get("refinement_log")
    return {
        "source_hashes": source_hashes_before,
        "n_frames": n_frames,
        "n_rows": int(refined_publication.dimensions.n_rois),
        "raw": {
            "run_id": args.raw_run,
            "manifest_digest": raw_publication.manifest["payload_digest"],
            "logical_content_digest": raw_publication.manifest["payload"][
                "logical_content"
            ]["digest"],
            "storage": raw_publication.plans.as_manifest()["object_estimate"],
        },
        "refined": {
            "run_id": args.refined_run,
            "manifest_digest": refined_publication.manifest["payload_digest"],
            "logical_content_digest": refined_publication.manifest["payload"][
                "logical_content"
            ]["digest"],
            "storage": refined_publication.plans.as_manifest()["object_estimate"],
        },
        "quality": {
            "run_id": args.quality_run,
            "manifest_digest": quality_publication.manifest["payload_digest"],
            "logical_content_digest": quality_publication.manifest["payload"][
                "logical_content"
            ]["digest"],
            "storage": quality_publication.plans.as_manifest()["object_estimate"],
        },
        "phases": phases,
        "evidence": published_evidence,
        "stores": {
            name: storage_stats(local_output / f"{name}.zarr")
            for name in ("raw", "refined", "quality")
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    destination = _require_destination(
        args.destination, benchmark_root=args.benchmark_root
    )
    scratch = (
        _require_existing_node_local(args.scratch_root)
        if args.resume_after_raw_inference or args.resume_after_publication
        else _require_node_local(args.scratch_root)
    )
    local_output = scratch / "completed"
    if args.resume_after_raw_inference:
        if not local_output.is_dir() or any(local_output.iterdir()):
            raise ValueError(
                "Resume requires an existing empty completed output directory."
            )
    elif args.resume_after_publication:
        expected = {
            local_output / "raw.zarr",
            local_output / "refined.zarr",
            local_output / "quality.zarr",
        }
        if not local_output.is_dir() or not all(path.is_dir() for path in expected):
            raise ValueError(
                "Post-publication resume requires existing raw, refined, and "
                "quality stores."
            )
    else:
        local_output.mkdir()
    hidden: Path | None = None
    published = False
    try:
        result = (
            _resume_published_pipeline(args, scratch, local_output)
            if args.resume_after_publication
            else _run_pipeline(args, scratch, local_output)
        )
        hidden = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
        if hidden.exists():
            raise FileExistsError(hidden)
        hidden.parent.mkdir(parents=True, exist_ok=True)
        copy_started = time.perf_counter()
        shutil.copytree(local_output, hidden)
        shared_copy_seconds = time.perf_counter() - copy_started
        payload: dict[str, Any] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "production_state_changes": [],
            "destination": str(destination),
            "inputs": {
                "source_crop_zarr": str(args.source_crop_zarr.expanduser().resolve()),
                "crop_run": args.crop_run,
                "source_refined_keypoint_zarr": str(
                    args.source_refined_keypoint_zarr.expanduser().resolve()
                ),
                "refined_keypoint_run": args.refined_keypoint_run,
                "roi_cache_manifest": str(
                    args.roi_cache_manifest.expanduser().resolve()
                ),
                "subject_mask_model": str(
                    args.subject_mask_model.expanduser().resolve()
                ),
                "resume_after_raw_inference": bool(args.resume_after_raw_inference),
                "resume_after_publication": bool(args.resume_after_publication),
                "resume_source_job_id": args.resume_source_job_id,
                "resume_source_palette_commit": (args.resume_source_palette_commit),
            },
            "outputs": result,
            "publication": {
                "policy": "node_local_complete_then_hidden_shared_copy_then_atomic_rename",
                "shared_copy_seconds": shared_copy_seconds,
                "intermediate_subject_mask_shards_published": False,
                "visible_stores": ["raw.zarr", "refined.zarr", "quality.zarr"],
            },
            "runtime": {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "lsb_jobid": os.environ.get("LSB_JOBID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python_executable": sys.executable,
                "peak_rss_bytes": _peak_rss_bytes(),
                "elapsed_seconds_before_handoff": time.perf_counter() - started,
            },
        }
        handoff = {
            "payload": payload,
            "payload_digest": canonical_json_sha256(payload),
        }
        _write_json(hidden / "handoff_manifest.json", handoff)
        os.replace(hidden, destination)
        reopened = _strict_json(destination / "handoff_manifest.json")
        if reopened != handoff:
            raise RuntimeError("Published handoff differs after atomic rename.")
        published = True
        return handoff
    finally:
        if not published and hidden is not None and hidden.exists():
            shutil.rmtree(hidden, ignore_errors=True)
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-crop-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--source-refined-keypoint-zarr", type=Path, required=True)
    parser.add_argument("--refined-keypoint-run", required=True)
    parser.add_argument("--roi-cache-manifest", type=Path, required=True)
    parser.add_argument("--subject-mask-model", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(
            "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/"
            "subject_mask_storage"
        ),
    )
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument(
        "--raw-shard-run", default="subject_masks_cache_canary_shard_v1"
    )
    parser.add_argument("--raw-run", default="subject_masks_cache_canary_raw_v1")
    parser.add_argument(
        "--refined-draft-run", default="refined_subject_masks_cache_canary_draft_v1"
    )
    parser.add_argument(
        "--refined-run", default="refined_subject_masks_cache_canary_v1"
    )
    parser.add_argument("--quality-run", default="subject_mask_quality_cache_canary_v1")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--finalize-workers", type=int, default=8)
    parser.add_argument("--resume-after-raw-inference", action="store_true")
    parser.add_argument("--resume-after-publication", action="store_true")
    parser.add_argument("--resume-source-job-id")
    parser.add_argument("--resume-source-palette-commit")
    parser.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.finalize_workers <= 0:
        raise SystemExit("--batch-size and --finalize-workers must be positive")
    if args.resume_after_raw_inference and args.resume_after_publication:
        raise SystemExit("Choose only one resume boundary.")
    if (args.resume_after_raw_inference or args.resume_after_publication) and (
        not args.resume_source_job_id or not args.resume_source_palette_commit
    ):
        raise SystemExit(
            "Resume requires --resume-source-job-id and "
            "--resume-source-palette-commit."
        )
    try:
        result = run(args)
    except Exception as exc:
        failure = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "production_state_changes": [],
        }
        failure_path = args.destination.with_name(
            f".{args.destination.name}.failed.{os.getpid()}.json"
        )
        _write_json(failure_path, failure)
        raise
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
