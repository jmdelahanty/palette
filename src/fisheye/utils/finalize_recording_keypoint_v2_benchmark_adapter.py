#!/usr/bin/env python3
"""Republish one historical recording keypoint aggregate for Crimson testing.

This is an intentionally narrow benchmark adapter.  It never runs inference,
never writes the source archive, and never activates a selector.  The adapter
proves exact observation-key and crop-geometry equivalence, normalizes the
legacy float64 payload at the existing keypoint-v2 boundary, materializes all
new stores on node-local scratch, and only then copies the complete bundle to
the benchmark namespace.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    sha256_array,
    sha256_file,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    CLIPPED_KEYPOINT_FINALIZATION_RECEIPT_NAME,
    ClipTerminalKeypointResult,
    clipped_keypoint_binding_digests,
    publish_selector_ineligible_clipped_keypoint_chain,
    validate_clipped_keypoint_finalization_receipt,
)
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr.keypoint_manifest import KeypointPreprocessingReference
from fisheye.shared.zarr.keypoint_publication import (
    prepare_raw_keypoint_v2_from_yolo_arrays,
)
from fisheye.shared.zarr.keypoint_schema import KeypointDimensions
from fisheye.shared.zarr.keypoint_successor import TerminalKeypointInferenceBatch
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    initial_refined_keypoint_snapshot_identity,
)
from fisheye.shared.zarr_run_completion import is_run_complete


ADAPTER_SCHEMA_ID = "palette.keypoint.recording_aggregate_benchmark_adapter"
ADAPTER_SCHEMA_VERSION = 1
ADAPTER_RECEIPT_NAME = "historical_aggregate_adapter_receipt.json"
_SOURCE_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "source_frame_indices",
    "frame_indices",
    "keypoints_roi",
    "keypoint_confidences",
    "confidence",
    "detection_success",
    "pose_bbox_xyxy_roi",
)
_HEX = frozenset("0123456789abcdef")
_SELECTOR_ATTRIBUTE_NAMES = ("latest", "latest_complete", "latest_pending")


def _safe_group(value: str) -> str:
    group = str(value).strip().strip("/")
    if not group or any(part in {"", ".", ".."} for part in group.split("/")):
        raise ValueError("source_group must be one safe relative Zarr path.")
    return group


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return text


def _array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _row_lookup(*, requested: np.ndarray, available: np.ndarray) -> np.ndarray:
    requested_keys = np.asarray(requested)
    available_keys = np.asarray(available)
    if (
        requested_keys.dtype != np.dtype(np.uint64)
        or available_keys.dtype != np.dtype(np.uint64)
        or requested_keys.ndim != 1
        or available_keys.ndim != 1
    ):
        raise ValueError("Instance keys must have exact uint64 shape [N].")
    if np.unique(requested_keys).size != requested_keys.size:
        raise ValueError("Requested crop-v2 instance keys are not unique.")
    if np.unique(available_keys).size != available_keys.size:
        raise ValueError("Historical keypoint instance keys are not unique.")
    order = np.argsort(available_keys, kind="stable")
    sorted_keys = available_keys[order]
    positions = np.searchsorted(sorted_keys, requested_keys)
    if np.any(positions >= sorted_keys.shape[0]):
        raise ValueError("Historical keypoints do not cover the crop-v2 key set.")
    source_rows = order[positions]
    if not np.array_equal(available_keys[source_rows], requested_keys):
        raise ValueError("Historical keypoints do not cover the crop-v2 key set.")
    if requested_keys.size != available_keys.size:
        raise ValueError("Historical keypoints contain rows absent from crop-v2.")
    return source_rows.astype(np.int64, copy=False)


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


def _metadata_sha256(group_path: Path) -> str:
    metadata = group_path / "zarr.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Source group metadata not found: {metadata}")
    return sha256_file(metadata)


def _source_selector_evidence(
    parent_attrs: Mapping[str, Any],
    *,
    run_id: str,
    stage_selector_eligible: object,
) -> dict[str, object]:
    """Prove that one explicitly pinned historical source is not selected."""

    if type(stage_selector_eligible) is not bool:
        raise ValueError(
            "Historical aggregate stage_selector_eligible must be an exact bool."
        )
    selectors: dict[str, str | None] = {}
    for attribute_name in _SELECTOR_ATTRIBUTE_NAMES:
        value = parent_attrs.get(attribute_name)
        if value is None:
            selectors[attribute_name] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Keypoint selector {attribute_name} must be a non-empty string or null."
            )
        selectors[attribute_name] = value.strip()
    selected_by = [
        attribute_name
        for attribute_name, selected_run_id in selectors.items()
        if selected_run_id == run_id
    ]
    if selected_by:
        raise ValueError(
            "Historical benchmark source is currently selected by "
            + ", ".join(selected_by)
            + "."
        )
    return {
        "run_id": run_id,
        "stage_selector_eligible": stage_selector_eligible,
        "selectors": selectors,
        "selected_by": [],
        "explicit_metadata_pin_required": True,
    }


def _pose_binding(attrs: Mapping[str, Any], *, model_sha256: str) -> dict[str, Any]:
    pose_schema = attrs.get("pose_schema")
    if not isinstance(pose_schema, Mapping):
        raise ValueError("Historical aggregate lacks its ordered pose schema.")
    labels = attrs.get("keypoint_labels")
    edges = pose_schema.get("edges")
    model_shape = attrs.get("model_kpt_shape")
    skeleton_id = attrs.get("skeleton_id")
    if not isinstance(labels, list) or not isinstance(edges, list):
        raise ValueError("Historical aggregate pose labels or edges are missing.")
    if not isinstance(model_shape, list) or not isinstance(skeleton_id, str):
        raise ValueError("Historical aggregate model shape or skeleton id is missing.")
    if pose_schema.get("keypoint_labels") != labels:
        raise ValueError("Historical aggregate ordered pose labels disagree.")
    if pose_schema.get("skeleton_id") != skeleton_id:
        raise ValueError("Historical aggregate skeleton identities disagree.")
    return build_explicit_pose_model_schema_binding(
        model_sha256=model_sha256,
        assertion_id="historical_recording_aggregate_benchmark_adapter_v1",
        skeleton_id=skeleton_id,
        model_kpt_shape=model_shape,
        keypoint_labels=labels,
        edges=edges,
    )


def _preprocessing(
    attrs: Mapping[str, Any],
    *,
    source_group: str,
    source_metadata_sha256: str,
) -> KeypointPreprocessingReference:
    model_transform = attrs.get("model_input_transform")
    if not isinstance(model_transform, Mapping):
        raise ValueError("Historical aggregate lacks model_input_transform.")
    input_mode = str(attrs.get("input_mode_effective") or "").strip()
    source_crop_run = str(attrs.get("source_crop_run") or "").strip()
    if not input_mode or not source_crop_run or "/" in source_crop_run:
        raise ValueError("Historical aggregate input/crop provenance is incomplete.")
    return KeypointPreprocessingReference(
        profile_id="historical_recording_aggregate_adapter_v1",
        profile_version=1,
        input_mode="legacy_recording_aggregate_republication",
        document={
            "benchmark_only_adapter": True,
            "source_group": source_group,
            "source_group_metadata_sha256": source_metadata_sha256,
            "source_crop_run": source_crop_run,
            "source_input_mode_effective": input_mode,
            "source_model_input_transform": dict(model_transform),
            "source_roi_cache": {
                "backend": attrs.get("source_roi_cache_backend"),
                "policy": attrs.get("roi_cache_policy"),
                "read_mode": attrs.get("source_roi_read_mode"),
                "source_tier": attrs.get("source_roi_cache_source_tier"),
                "staged_to_node_scratch": attrs.get(
                    "source_roi_cache_staged_to_node_scratch"
                ),
                "used": attrs.get("source_roi_cache_used"),
            },
            "canonicalization_boundary": (
                "float64_legacy_payload_to_keypoint_v2_float32"
            ),
            "inference_reexecuted": False,
        },
    )


def _rebind_finalization_receipt(
    receipt: Mapping[str, Any],
    *,
    destination: Path,
) -> dict[str, Any]:
    rebound = copy.deepcopy(dict(receipt))
    payload = rebound["payload"]
    for binding in payload["outputs"].values():
        binding["path"] = str(destination / Path(binding["path"]).name)
    rebound["payload_digest"] = canonical_json_sha256(payload)
    errors = validate_clipped_keypoint_finalization_receipt(rebound)
    if errors:
        raise ValueError(
            "Rebound finalization receipt is invalid: " + "; ".join(errors)
        )
    return rebound


def _publish_directory(local: Path, destination: Path) -> float:
    if destination.exists():
        raise FileExistsError(f"Benchmark bundle already exists: {destination}")
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


def finalize_recording_keypoint_v2_benchmark_adapter(
    *,
    analysis_zarr: Path,
    source_group_path: str,
    source_group_metadata_sha256: str,
    expected_model_sha256: str,
    expected_n_frames: int,
    expected_n_instances: int,
    crop_archive: Path,
    refined_archive: Path,
    crop_run_id: str,
    bundle_root: Path,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    recording_identity: str,
    refined_lineage_id: str,
    refined_snapshot_id: str,
    scratch_parent: Path,
) -> dict[str, object]:
    """Validate, normalize, locally materialize, and publish one full bundle."""

    started = time.perf_counter()
    phases: dict[str, float] = {}
    archive = analysis_zarr.expanduser().resolve()
    source_group_name = _safe_group(source_group_path)
    destination = bundle_root.expanduser().resolve()
    if ".palette_benchmarks" not in destination.parts:
        raise ValueError(
            "Benchmark adapter destination must be in .palette_benchmarks."
        )
    if destination.exists():
        raise FileExistsError(f"Benchmark bundle already exists: {destination}")
    if type(expected_n_frames) is not int or expected_n_frames <= 0:
        raise ValueError("expected_n_frames must be a positive exact integer.")
    if type(expected_n_instances) is not int or expected_n_instances <= 0:
        raise ValueError("expected_n_instances must be a positive exact integer.")
    expected_metadata = _require_sha256(
        source_group_metadata_sha256, name="source group metadata"
    )
    expected_model = _require_sha256(expected_model_sha256, name="model")
    source_metadata_path = archive / source_group_name
    source_parent_name, separator, source_run_id = source_group_name.rpartition("/")
    if not separator or not source_parent_name or not source_run_id:
        raise ValueError("source_group must include its run-family parent and run id.")
    source_parent_metadata_path = archive / source_parent_name
    observed_metadata_before = _metadata_sha256(source_metadata_path)
    observed_parent_metadata_before = _metadata_sha256(source_parent_metadata_path)
    if observed_metadata_before != expected_metadata:
        raise ValueError("Historical aggregate metadata differs from the pinned input.")

    open_started = time.perf_counter()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = root[source_group_name]
    if not is_run_complete(source):
        raise ValueError("Historical recording keypoint aggregate is not complete.")
    attrs = dict(source.attrs)
    if attrs.get("source_kind") != "keypoint_shard_collection_finalizer":
        raise ValueError("Source is not the frozen recording keypoint aggregate kind.")
    selector_evidence = _source_selector_evidence(
        dict(root[source_parent_name].attrs),
        run_id=source_run_id,
        stage_selector_eligible=attrs.get("stage_selector_eligible"),
    )
    if attrs.get("instance_key_backfill_status") != "complete":
        raise ValueError("Historical aggregate lacks completed stable instance keys.")
    if attrs.get("instance_key_backfill_recording_identity") != recording_identity:
        raise ValueError("Historical aggregate binds a different recording identity.")
    model_path = Path(str(attrs.get("model_path") or "")).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Historical keypoint model not found: {model_path}")
    observed_model = sha256_file(model_path)
    if observed_model != expected_model:
        raise ValueError("Historical keypoint model digest differs from the pin.")
    binding = _pose_binding(attrs, model_sha256=observed_model)
    preprocessing = _preprocessing(
        attrs,
        source_group=source_group_name,
        source_metadata_sha256=observed_metadata_before,
    )
    crop = open_persisted_crop_geometry_publication(
        crop_archive,
        run_id=crop_run_id,
        source_refined_archive=refined_archive,
    )
    if crop.dimensions.n_frames != expected_n_frames:
        raise ValueError("Crop-v2 frame count differs from the full-duration pin.")
    if crop.dimensions.n_instances != expected_n_instances:
        raise ValueError("Crop-v2 row count differs from the full-duration pin.")
    phases["open_and_bind_sources"] = time.perf_counter() - open_started

    load_started = time.perf_counter()
    crop_arrays = {
        path: _array(crop.arrays[path])
        for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths
    }
    target_keys = crop_arrays["instance_key"]
    source_keys = _array(source["instance_key"])
    source_rows = _row_lookup(requested=target_keys, available=source_keys)
    source_values = {path: _array(source[path]) for path in _SOURCE_ARRAY_PATHS}
    source_hashes = {
        path: sha256_array(values) for path, values in source_values.items()
    }
    if not np.array_equal(source_values["instance_key"][source_rows], target_keys):
        raise ValueError("Historical keypoint rows do not align by instance_key.")

    old_crop_run = str(attrs["source_crop_run"])
    old_crop = root[f"crop_runs/{old_crop_run}"]
    old_crop_keys = _array(old_crop["instance_key"])
    old_rows = source_values["source_crop_row_ids"]
    if old_rows.dtype != np.dtype(np.int64) or old_rows.shape != source_keys.shape:
        raise ValueError("Historical source_crop_row_ids has the wrong contract.")
    if np.any(old_rows < 0) or np.any(old_rows >= old_crop_keys.size):
        raise ValueError("Historical source_crop_row_ids exceeds its source crop.")
    if not np.array_equal(old_crop_keys[old_rows], source_keys):
        raise ValueError(
            "Historical keypoints and source crop disagree on instance keys."
        )
    mapped_old_rows = old_rows[source_rows]
    old_origins = _array(old_crop["roi_coordinates_full"])[mapped_old_rows]
    if not np.array_equal(old_origins, crop_arrays["roi_coordinates_full"]):
        raise ValueError("Historical and crop-v2 ROI origins differ.")
    old_roi_shape = old_crop.attrs.get("roi_shape") or old_crop.attrs.get("roi_size")
    if not isinstance(old_roi_shape, (list, tuple)) or len(old_roi_shape) != 2:
        raise ValueError("Historical source crop lacks a fixed ROI shape.")
    old_size_wh = np.asarray([old_roi_shape[1], old_roi_shape[0]], dtype=np.int32)
    if not np.all(crop_arrays["roi_sizes_full"] == old_size_wh):
        raise ValueError("Historical and crop-v2 ROI sizes differ.")
    if not np.array_equal(
        source_values["frame_indices"][source_rows], crop_arrays["frame_indices"]
    ):
        raise ValueError("Historical and crop-v2 frame indices differ.")
    if not np.array_equal(
        source_values["source_frame_indices"][source_rows],
        crop_arrays["source_acquisition_frame_index"],
    ):
        raise ValueError("Historical and crop-v2 acquisition frames differ.")
    phases["load_hash_and_align_legacy_rows"] = time.perf_counter() - load_started

    prepare_started = time.perf_counter()
    yolo_arrays = {
        "instance_key": np.array(target_keys, copy=True),
        "source_crop_row_ids": np.arange(expected_n_instances, dtype=np.int64),
        "source_acquisition_frame_index": np.array(
            crop_arrays["source_acquisition_frame_index"], copy=True
        ),
        "frame_indices": np.array(crop_arrays["frame_indices"], copy=True),
        "keypoints_roi": source_values["keypoints_roi"][source_rows],
        "keypoint_confidences": source_values["keypoint_confidences"][source_rows],
        "confidence": source_values["confidence"][source_rows],
        "detection_success": source_values["detection_success"][source_rows],
        "pose_bbox_xyxy_roi": source_values["pose_bbox_xyxy_roi"][source_rows],
    }
    dimensions = KeypointDimensions(
        n_frames=expected_n_frames,
        n_instances=expected_n_instances,
        n_keypoints=int(yolo_arrays["keypoints_roi"].shape[1]),
        source_width=crop.dimensions.source_width,
        source_height=crop.dimensions.source_height,
    )
    conversion = prepare_raw_keypoint_v2_from_yolo_arrays(
        yolo_arrays,
        dimensions=dimensions,
        source_crop_arrays=crop_arrays,
        source_crop_manifest=crop.manifest,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
    )
    evidence_payload = {
        "source_archive": str(archive),
        "source_group": source_group_name,
        "source_group_metadata_sha256": observed_metadata_before,
        "source_parent_group": source_parent_name,
        "source_parent_metadata_sha256": observed_parent_metadata_before,
        "source_selector_evidence": selector_evidence,
        "source_array_hashes": source_hashes,
        "source_crop_run": old_crop_run,
        "source_crop_geometry_equal": True,
        "instance_key_set_equal": True,
        "model_path": str(model_path),
        "model_sha256": observed_model,
        "pose_binding_digest": canonical_json_sha256(binding),
        "preprocessing_digest": canonical_json_sha256(preprocessing.as_manifest()),
        "conversion_receipt": dict(conversion.conversion_receipt),
        "inference_reexecuted": False,
        "selector_eligible": False,
        "production_state_changes": [],
    }
    evidence_digest = canonical_json_sha256(evidence_payload)
    digests = clipped_keypoint_binding_digests(
        crop=crop,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
    )
    normalized = conversion.prepared.arrays
    clip = ClipTerminalKeypointResult(
        clip_id="historical_recording_aggregate",
        clip_index=0,
        terminal_status="complete",
        inference=TerminalKeypointInferenceBatch(
            instance_key=np.asarray(normalized["instance_key"]),
            keypoints_roi=np.asarray(normalized["keypoints_roi"]),
            keypoint_confidences=np.asarray(normalized["keypoint_confidences"]),
            pose_confidence=np.asarray(normalized["pose_confidence"]),
            pose_bbox_xyxy_roi=np.asarray(normalized["pose_bbox_xyxy_roi"]),
            pose_success=np.asarray(normalized["pose_success"]),
        ),
        source_crop_row_signature=np.asarray(normalized["source_crop_row_signature"]),
        crop_run_id=crop.run_id,
        input_package_manifest_digest=evidence_digest,
        **digests,
    )
    identity = initial_refined_keypoint_snapshot_identity(
        recording_identity=recording_identity,
        lineage_id=refined_lineage_id,
        snapshot_id=refined_snapshot_id,
    )
    phases["normalize_and_validate_keypoint_v2"] = time.perf_counter() - prepare_started

    scratch = _require_node_local_scratch(scratch_parent)
    with tempfile.TemporaryDirectory(
        prefix="palette_crimson_keypoint_v2_", dir=scratch
    ) as temporary_directory:
        local_bundle = Path(temporary_directory) / "bundle"
        publication_started = time.perf_counter()
        chain = publish_selector_ineligible_clipped_keypoint_chain(
            crop,
            (clip,),
            pose_model_schema_binding=binding,
            preprocessing=preprocessing,
            bundle_root=local_bundle,
            raw_run_id=raw_run_id,
            quality_run_id=quality_run_id,
            refined_run_id=refined_run_id,
            body_frame_run_id=body_frame_run_id,
            refined_identity=identity,
            created_by="recording_keypoint_v2_benchmark_adapter",
        )
        phases["node_local_publication"] = time.perf_counter() - publication_started
        evidence = {
            "schema_id": ADAPTER_SCHEMA_ID,
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": utc_now(),
            "payload_digest": evidence_digest,
            "payload": evidence_payload,
        }
        write_json_atomic(local_bundle / ADAPTER_RECEIPT_NAME, evidence)
        rebound = _rebind_finalization_receipt(
            chain.receipt,
            destination=destination,
        )
        write_json_atomic(
            local_bundle / CLIPPED_KEYPOINT_FINALIZATION_RECEIPT_NAME,
            rebound,
        )
        local_stats = storage_stats(local_bundle)
        phases["publish_to_shared_storage"] = _publish_directory(
            local_bundle, destination
        )

    observed_metadata_after = _metadata_sha256(source_metadata_path)
    if observed_metadata_after != observed_metadata_before:
        raise RuntimeError("Historical source metadata changed during republishing.")
    observed_parent_metadata_after = _metadata_sha256(source_parent_metadata_path)
    if observed_parent_metadata_after != observed_parent_metadata_before:
        raise RuntimeError("Historical source selector metadata changed during republishing.")
    return {
        "schema_id": ADAPTER_SCHEMA_ID,
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "complete",
        "analysis_zarr": str(archive),
        "crop_archive": str(crop_archive.expanduser().resolve()),
        "refined_archive": str(refined_archive.expanduser().resolve()),
        "crop_run_id": crop.run_id,
        "source_group_path": source_group_name,
        "source_group_metadata_sha256": observed_metadata_before,
        "source_parent_group": source_parent_name,
        "source_parent_metadata_sha256": observed_parent_metadata_before,
        "source_selector_evidence": selector_evidence,
        "bundle_root": str(destination),
        "finalization_receipt_path": str(
            destination / CLIPPED_KEYPOINT_FINALIZATION_RECEIPT_NAME
        ),
        "finalization_receipt_digest": rebound["payload_digest"],
        "adapter_receipt_path": str(destination / ADAPTER_RECEIPT_NAME),
        "adapter_receipt_digest": evidence_digest,
        "outputs": rebound["payload"]["outputs"],
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "inference_reexecuted": False,
        "node_local_materialization": True,
        "local_bundle_stats": local_stats,
        "timing_seconds": phases,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes": peak_rss_bytes(),
        "source_unchanged": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--source-group-metadata-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-n-frames", type=int, required=True)
    parser.add_argument("--expected-n-instances", type=int, required=True)
    parser.add_argument("--crop-archive", type=Path, required=True)
    parser.add_argument("--refined-archive", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--body-frame-run", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--refined-lineage-id", required=True)
    parser.add_argument("--refined-snapshot-id", required=True)
    parser.add_argument("--scratch-parent", type=Path)
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
        result = finalize_recording_keypoint_v2_benchmark_adapter(
            analysis_zarr=args.analysis_zarr,
            source_group_path=args.source_group,
            source_group_metadata_sha256=args.source_group_metadata_sha256,
            expected_model_sha256=args.expected_model_sha256,
            expected_n_frames=args.expected_n_frames,
            expected_n_instances=args.expected_n_instances,
            crop_archive=args.crop_archive,
            refined_archive=args.refined_archive,
            crop_run_id=args.crop_run,
            bundle_root=args.bundle_root,
            raw_run_id=args.raw_run,
            quality_run_id=args.quality_run,
            refined_run_id=args.refined_run,
            body_frame_run_id=args.body_frame_run,
            recording_identity=args.recording_identity,
            refined_lineage_id=args.refined_lineage_id,
            refined_snapshot_id=args.refined_snapshot_id,
            scratch_parent=scratch_parent,
        )
    except Exception as exc:
        result = {
            "schema_id": ADAPTER_SCHEMA_ID,
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "status": "failed",
            "bundle_root": str(args.bundle_root),
            "source_group_path": args.source_group,
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
    "finalize_recording_keypoint_v2_benchmark_adapter",
]
