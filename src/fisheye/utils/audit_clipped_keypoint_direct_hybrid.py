#!/usr/bin/env python3
"""Prove whether legacy clipped keypoint shards can enter strict-v2 safely.

This command is read-only.  It validates a signed hybrid pixel provider, an
independently sealed crop-v2 geometry publication, the exact provider/crop row
equivalence, the registered pose-model binding, every source shard's immutable
scientific payload hashes, and complete nonoverlapping recording-row coverage.
It never creates a receipt, publication, selector, or scratch Zarr.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.registry.db import Registry
from fisheye.pose.schema import undirected_edge_topology
from fisheye.shared.hybrid_crop_provider import (
    validate_hybrid_provider_strict_crop_geometry,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.keypoint_terminal_pixel_evidence import (
    DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
    build_direct_hybrid_terminal_pixel_evidence,
)
from fisheye.shared.pose_inference_failure import (
    validate_pose_inference_failure_codes,
)
from fisheye.shared.pose_model_schema_binding import (
    resolve_registered_pose_model_schema_binding,
)
from fisheye.shared.provenance_attrs import build_source_crop_snapshot_attrs
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    clip_terminal_result_from_yolo_arrays,
)
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    build_crop_run_reference,
    validate_crop_run_reference,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr.keypoint_manifest import KeypointPreprocessingReference
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import is_run_complete


REPORT_SCHEMA_ID = "palette.keypoint.clipped_direct_hybrid_proof_audit"
REPORT_SCHEMA_VERSION = 1
DIRECT_HYBRID_EVIDENCE_PROFILE = DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE

_SOURCE_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "frame_indices",
    "keypoints_roi",
    "keypoints_img",
    "keypoints_norm",
    "keypoint_confidences",
    "confidence",
    "pose_bbox_xyxy_roi",
    "pose_bbox_xyxy_img",
    "pose_bbox_xyxy_norm",
    "detection_success",
    "pose_failure_codes",
    "detection_source",
    "effective_se2_radius",
    "effective_threshold",
    "heading",
    "heading_finite",
    "heading_usable",
)
_TERMINAL_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "frame_indices",
    "keypoints_roi",
    "keypoint_confidences",
    "confidence",
    "pose_bbox_xyxy_roi",
    "detection_success",
    "pose_failure_codes",
)
_OBSERVED_RUNTIME_ATTRS = (
    "input_mode_effective",
    "model_input_transform",
    "model_input_stride",
    "model_input_shape_hw",
    "model_network_input_shape_hw",
    "native_roi_shape_hw",
    "ultralytics_version",
    "pose_schema",
    "skeleton_id",
    "model_kpt_shape",
    "kpt_shape",
)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _model_artifact(attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = attrs.get("run_provenance")
    artifacts = provenance.get("input_artifacts") if isinstance(provenance, Mapping) else None
    matches = [
        item
        for item in artifacts or ()
        if isinstance(item, Mapping) and item.get("role") == "keypoint_model"
    ]
    if len(matches) != 1:
        raise ValueError("Shard must bind exactly one content-addressed keypoint model.")
    digest = str(matches[0].get("sha256") or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Shard keypoint-model SHA-256 is invalid.")
    return matches[0]


def _persisted_source_hashes(attrs: Mapping[str, Any]) -> Mapping[str, str]:
    provenance = attrs.get("provenance")
    artifacts = provenance.get("artifacts") if isinstance(provenance, Mapping) else None
    write = artifacts.get("keypoint_shard_write") if isinstance(artifacts, Mapping) else None
    hashes = write.get("source_sha256_by_array") if isinstance(write, Mapping) else None
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("Shard lacks its immutable source-array hash receipt.")
    normalized = {str(name): str(digest) for name, digest in hashes.items()}
    for name, digest in normalized.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Shard source hash for {name!r} is invalid.")
    return normalized


def _observed_runtime(attrs: Mapping[str, Any]) -> dict[str, Any]:
    runtime = {name: attrs.get(name) for name in _OBSERVED_RUNTIME_ATTRS}
    provenance = attrs.get("provenance")
    parameters = provenance.get("parameters") if isinstance(provenance, Mapping) else None
    runtime["thresholds"] = {
        name: parameters.get(name) if isinstance(parameters, Mapping) else None
        for name in ("confidence_threshold", "iou_threshold", "max_det", "imgsz", "model_input_size")
    }
    if any(runtime[name] is None for name in _OBSERVED_RUNTIME_ATTRS):
        missing = [name for name in _OBSERVED_RUNTIME_ATTRS if runtime[name] is None]
        raise ValueError("Shard lacks observed runtime attrs: " + ", ".join(missing))
    canonical_json_sha256(runtime)
    return runtime


def _resolve_pose_binding(
    registry: Registry,
    attrs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _model_artifact(attrs)
    model_path = str(attrs.get("model_resolution_selected_model_path") or attrs.get("model_path") or "")
    run_id = str(attrs.get("model_resolution_selected_run_id") or "")
    set_id = str(attrs.get("model_resolution_selected_set_id") or "")
    if not model_path or not run_id or not set_id:
        raise ValueError("Shard lacks exact registry model selection identity.")
    if Path(model_path).expanduser().resolve() != Path(str(artifact.get("path"))).expanduser().resolve():
        raise ValueError("Shard model path differs from its content receipt.")
    binding = resolve_registered_pose_model_schema_binding(
        registry,
        run_id=run_id,
        expected_set_id=set_id,
        expected_model_path=model_path,
        expected_model_sha256=str(artifact["sha256"]),
    )
    stored_schema = attrs.get("pose_schema")
    bound_schema = binding.get("pose_schema")
    if not isinstance(stored_schema, Mapping) or not isinstance(bound_schema, Mapping):
        raise ValueError("Shard or model binding lacks an ordered pose schema.")
    comparisons = {
        "skeleton_id": (stored_schema.get("skeleton_id"), bound_schema.get("skeleton_id")),
        "keypoint_labels": (stored_schema.get("keypoint_labels"), bound_schema.get("keypoint_labels")),
        "kpt_shape": (stored_schema.get("kpt_shape"), bound_schema.get("kpt_shape")),
    }
    mismatched = [name for name, (left, right) in comparisons.items() if left != right]
    if undirected_edge_topology(stored_schema.get("edges")) != undirected_edge_topology(
        bound_schema.get("edges")
    ):
        mismatched.append("edges")
    if mismatched:
        raise ValueError("Shard pose schema differs from its model binding at: " + ", ".join(mismatched))
    return binding, {
        "set_id": set_id,
        "run_id": run_id,
        "path": str(Path(model_path).expanduser().resolve()),
        "sha256": str(artifact["sha256"]),
        "pose_model_schema_binding_digest": canonical_json_sha256(binding),
    }


def _validate_shard_model_claim(
    attrs: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    expected_model: Mapping[str, Any],
) -> None:
    artifact = _model_artifact(attrs)
    observed_model = {
        "set_id": str(attrs.get("model_resolution_selected_set_id") or ""),
        "run_id": str(attrs.get("model_resolution_selected_run_id") or ""),
        "path": str(
            Path(
                str(
                    attrs.get("model_resolution_selected_model_path")
                    or attrs.get("model_path")
                    or ""
                )
            )
            .expanduser()
            .resolve()
        ),
        "sha256": str(artifact.get("sha256") or ""),
        "pose_model_schema_binding_digest": canonical_json_sha256(binding),
    }
    expected_claim = {
        key: value
        for key, value in expected_model.items()
        if key != "pose_model_schema_binding"
    }
    if observed_model != expected_claim:
        raise ValueError("Shard model binding differs within the recording.")
    stored_schema = attrs.get("pose_schema")
    bound_schema = binding.get("pose_schema")
    if not isinstance(stored_schema, Mapping) or not isinstance(bound_schema, Mapping):
        raise ValueError("Shard or model binding lacks an ordered pose schema.")
    if (
        stored_schema.get("skeleton_id") != bound_schema.get("skeleton_id")
        or stored_schema.get("keypoint_labels") != bound_schema.get("keypoint_labels")
        or stored_schema.get("kpt_shape") != bound_schema.get("kpt_shape")
        or undirected_edge_topology(stored_schema.get("edges"))
        != undirected_edge_topology(bound_schema.get("edges"))
    ):
        raise ValueError("Shard pose schema differs from its model binding.")


def _source_arrays(group: Any) -> dict[str, np.ndarray]:
    missing = [name for name in _SOURCE_ARRAY_PATHS if name not in group]
    if missing:
        raise ValueError("Shard lacks scientific arrays: " + ", ".join(missing))
    return {name: np.asarray(group[name][...]) for name in _SOURCE_ARRAY_PATHS}


def _verify_scientific_hashes(
    arrays: Mapping[str, np.ndarray],
    expected: Mapping[str, str],
) -> dict[str, str]:
    missing = sorted(set(expected) - set(arrays))
    if missing:
        raise ValueError("Source hash receipt names absent arrays: " + ", ".join(missing))
    observed = {name: sha256_array(arrays[name]) for name in sorted(expected)}
    mismatched = [name for name in sorted(expected) if observed[name] != expected[name]]
    if mismatched:
        raise ValueError("Scientific source arrays changed after inference: " + ", ".join(mismatched))
    return observed


def _validate_derived_coordinates(
    arrays: Mapping[str, np.ndarray],
    *,
    crop_origins: np.ndarray,
) -> None:
    origins = np.asarray(crop_origins, dtype=np.float64)
    keypoints_roi = np.asarray(arrays["keypoints_roi"], dtype=np.float64)
    keypoints_img = np.asarray(arrays["keypoints_img"], dtype=np.float64)
    expected_keypoints_img = keypoints_roi + origins[:, None, :]
    if not np.array_equal(keypoints_img, expected_keypoints_img, equal_nan=True):
        raise ValueError("keypoints_img is not the exact ROI-to-image translation.")
    bbox_roi = np.asarray(arrays["pose_bbox_xyxy_roi"], dtype=np.float32)
    bbox_img = np.asarray(arrays["pose_bbox_xyxy_img"], dtype=np.float32)
    offset = np.column_stack((origins, origins)).astype(np.float32)
    if not np.array_equal(bbox_img, bbox_roi + offset, equal_nan=True):
        raise ValueError("pose_bbox_xyxy_img is not the exact ROI-to-image translation.")


def _audit_target(
    target: Mapping[str, Any],
    *,
    registry: Registry,
) -> dict[str, Any]:
    target_id = str(target.get("target_id") or "")
    archive = Path(str(target.get("analysis_zarr") or "")).expanduser().resolve()
    crop_run = str(target.get("target_geometry_crop_run") or "")
    shard_names = target.get("source_keypoint_shards")
    if not target_id or not crop_run or not isinstance(shard_names, list) or not shard_names:
        raise ValueError("Recovery target lacks its exact id, crop, or shard roster.")

    crop = open_persisted_crop_geometry_publication(archive, run_id=crop_run)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    shard_parent = root["keypoint_shard_runs"]
    first_name = str(shard_names[0])
    if first_name not in shard_parent:
        raise FileNotFoundError(f"Missing keypoint shard {first_name!r}.")
    first = shard_parent[first_name]
    provider_run = str(first.attrs.get("source_crop_run") or "")
    if not provider_run or "/" in provider_run:
        raise ValueError("First shard lacks a safe hybrid provider run id.")
    provider = root[f"crop_runs/{provider_run}"]
    provider_reference = validate_crop_run_reference(
        build_crop_run_reference(provider, run_id=provider_run)
    )
    if provider_reference.get("profile") != CROP_RUN_REFERENCE_SIGNED_PROFILE:
        raise ValueError("Source crop provider is not a signed-current profile.")
    provider_record_sha256 = str(provider.attrs.get("provider_record_sha256") or "")
    provider_binding = validate_hybrid_provider_strict_crop_geometry(
        provider,
        crop,
        expected_provider_record_sha256=provider_record_sha256,
    )
    pose_binding, model = _resolve_pose_binding(registry, first.attrs)
    model = {**model, "pose_model_schema_binding": pose_binding}
    runtime = _observed_runtime(first.attrs)
    crop_snapshot = build_source_crop_snapshot_attrs(
        provider.attrs,
        source_crop_storage_mode=provider.attrs.get("crop_storage_mode"),
    )
    preprocessing = KeypointPreprocessingReference(
        profile_id=DIRECT_HYBRID_EVIDENCE_PROFILE,
        profile_version=1,
        input_mode="numpy_list",
        document={
            "evidence_semantics": "observed_completed_inference_runtime_v1",
            "observed_input_mode_effective": first.attrs.get(
                "input_mode_effective"
            ),
            "roi_provider": {
                "crop_run": provider_run,
                "record_sha256": provider_binding["provider_record_sha256"],
                "source_pixel_fingerprint": provider_binding["source_pixel_fingerprint"],
                "source_rowset_fingerprint": provider_binding["source_rowset_fingerprint"],
                "source_row_signature_spec_digest": provider_binding[
                    "source_row_signature_spec_digest"
                ],
            },
            "provider_reference": provider_reference,
            "source_crop_snapshot": crop_snapshot,
            "observed_runtime": runtime,
            "coordinate_contract_mode": "legacy_noncanonical",
        },
    )
    proof_basis = {
        "schema_id": "palette.keypoint.direct_hybrid_terminal_evidence",
        "schema_version": 1,
        "profile": DIRECT_HYBRID_EVIDENCE_PROFILE,
        "provider_run": provider_run,
        "provider_reference": provider_reference,
        "provider_binding": provider_binding,
        "geometry_crop_run": crop.run_id,
        "geometry_crop_manifest_digest": canonical_json_sha256(crop.manifest),
        "pose_model_schema_binding_digest": canonical_json_sha256(pose_binding),
        "preprocessing_digest": canonical_json_sha256(preprocessing.as_manifest()),
    }
    proof_basis_digest = canonical_json_sha256(proof_basis)

    crop_rows = int(np.asarray(crop.arrays["instance_key"]).shape[0])
    virtual_source_rows = np.arange(crop_rows, dtype=np.int64)
    source_crop_arrays = {
        "instance_key": provider["instance_key"],
        "source_crop_row_ids": virtual_source_rows,
        "frame_indices": provider["frame_indices"],
        "roi_coordinates_full": provider["roi_coordinates_full"],
        "roi_sizes_full": provider["roi_sizes_full"],
    }
    crop_origins = np.asarray(crop.arrays["roi_coordinates_full"], dtype=np.int32)
    crop_acquisition_frames = np.asarray(
        crop.arrays["source_acquisition_frame_index"], dtype=np.int64
    )
    covered = np.zeros(crop_rows, dtype=bool)
    reports: list[dict[str, Any]] = []
    runtime_digest = canonical_json_sha256(runtime)
    for clip_index, raw_name in enumerate(shard_names):
        name = str(raw_name)
        try:
            if name not in shard_parent:
                raise FileNotFoundError(f"Missing keypoint shard {name!r}.")
            group = shard_parent[name]
            if not is_run_complete(group):
                raise ValueError("Shard is not complete.")
            if group.attrs.get("coordinate_contract_mode") != "legacy_noncanonical":
                raise ValueError("Shard does not retain its legacy_noncanonical source label.")
            if group.attrs.get("source_crop_run") != provider_run:
                raise ValueError("Shard binds a different hybrid provider.")
            if group.attrs.get("source_crop_pixel_work_package_manifest") is not None:
                raise ValueError("Direct-hybrid shard unexpectedly claims a work package.")
            expected_crop_attrs = {
                "source_crop_signature": crop_snapshot.get("source_crop_signature"),
                "source_crop_revision": crop_snapshot.get("source_crop_revision"),
            }
            for attr_name, expected in expected_crop_attrs.items():
                if group.attrs.get(attr_name) != expected:
                    raise ValueError(f"Shard {attr_name} differs from the live signed provider.")
            observed_runtime = _observed_runtime(group.attrs)
            if canonical_json_sha256(observed_runtime) != runtime_digest:
                raise ValueError("Shard runtime/preprocessing evidence differs within the recording.")
            _validate_shard_model_claim(
                group.attrs,
                binding=pose_binding,
                expected_model=model,
            )

            arrays = _source_arrays(group)
            source_hashes = _verify_scientific_hashes(
                arrays,
                _persisted_source_hashes(group.attrs),
            )
            rows = np.asarray(arrays["source_crop_row_ids"])
            if rows.dtype != np.dtype(np.int64) or rows.ndim != 1 or rows.size == 0:
                raise ValueError("Shard source_crop_row_ids must be nonempty exact int64 [M].")
            if np.any(rows < 0) or np.any(rows >= crop_rows):
                raise ValueError("Shard source_crop_row_ids are outside the crop-v2 rowset.")
            if np.unique(rows).size != rows.size:
                raise ValueError("Shard contains duplicate crop rows.")
            if np.any(covered[rows]):
                raise ValueError("Shard overlaps crop rows already claimed by another shard.")
            validate_pose_inference_failure_codes(
                np.asarray(arrays["pose_failure_codes"], dtype=np.uint8),
                pose_success=np.asarray(arrays["detection_success"], dtype=bool),
            )
            if not np.array_equal(
                np.asarray(arrays["source_acquisition_frame_index"], dtype=np.int64),
                crop_acquisition_frames[rows],
            ):
                raise ValueError("Shard acquisition-frame lineage differs from crop-v2.")
            _validate_derived_coordinates(arrays, crop_origins=crop_origins[rows])
            terminal = clip_terminal_result_from_yolo_arrays(
                crop,
                {name: arrays[name] for name in _TERMINAL_ARRAY_PATHS},
                source_crop_arrays=source_crop_arrays,
                clip_id=f"clip_{clip_index:06d}",
                clip_index=clip_index,
                pose_model_schema_binding=pose_binding,
                preprocessing=preprocessing,
                input_package_manifest_digest=proof_basis_digest,
            )
            covered[rows] = True
            reports.append(
                {
                    "shard_run": name,
                    "clip_index": clip_index,
                    "status": "migratable",
                    "row_count": int(rows.size),
                    "source_crop_row_min": int(rows.min()),
                    "source_crop_row_max": int(rows.max()),
                    "terminal_result_digest": canonical_json_sha256(terminal.as_manifest()),
                    "scientific_array_hashes": source_hashes,
                }
            )
        except Exception as exc:
            reports.append(
                {
                    "shard_run": name,
                    "clip_index": clip_index,
                    "status": "unmigratable",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    failures = [item for item in reports if item["status"] != "migratable"]
    missing_rows = np.flatnonzero(~covered)
    if missing_rows.size:
        failures.append(
            {
                "status": "unmigratable",
                "error_type": "IncompleteCropCoverage",
                "error": f"Shard roster leaves {int(missing_rows.size)} crop-v2 rows uncovered.",
                "first_missing_rows": [int(value) for value in missing_rows[:16]],
            }
        )
    status = "migratable" if not failures else "unmigratable"
    source_shard_evidence_digest = canonical_json_sha256(
        [
            {
                "shard_run": item.get("shard_run"),
                "clip_index": item.get("clip_index"),
                "status": item.get("status"),
                "row_count": item.get("row_count"),
                "source_crop_row_min": item.get("source_crop_row_min"),
                "source_crop_row_max": item.get("source_crop_row_max"),
                "terminal_result_digest": item.get("terminal_result_digest"),
                "scientific_array_hashes": item.get("scientific_array_hashes"),
            }
            for item in reports
        ]
    )
    terminal_pixel_evidence = (
        build_direct_hybrid_terminal_pixel_evidence(
            provider_run=provider_run,
            provider_reference=provider_reference,
            provider_binding=provider_binding,
            geometry_crop_run=crop.run_id,
            geometry_crop_manifest_digest=canonical_json_sha256(crop.manifest),
            source_shard_runs=[str(value) for value in shard_names],
            source_shard_evidence_digest=source_shard_evidence_digest,
        )
        if status == "migratable"
        else None
    )
    return {
        "target_id": target_id,
        "analysis_zarr": str(archive),
        "status": status,
        "metadata_read_mode": "unconsolidated_explicit_read_only",
        "geometry_crop_run": crop.run_id,
        "geometry_crop_manifest_digest": canonical_json_sha256(crop.manifest),
        "hybrid_provider_run": provider_run,
        "hybrid_provider_reference": provider_reference,
        "hybrid_provider_binding": provider_binding,
        "proof_basis": proof_basis,
        "proof_basis_digest": proof_basis_digest,
        "source_shard_evidence_digest": source_shard_evidence_digest,
        "terminal_pixel_evidence": terminal_pixel_evidence,
        "model": model,
        "preprocessing": preprocessing.as_manifest(),
        "observed_runtime": runtime,
        "crop_row_count": crop_rows,
        "covered_crop_row_count": int(np.count_nonzero(covered)),
        "shard_count": len(reports),
        "migratable_shard_count": sum(item["status"] == "migratable" for item in reports),
        "unmigratable_shard_count": sum(item["status"] != "migratable" for item in reports),
        "shards": reports,
        "coverage_failures": [item for item in failures if "shard_run" not in item],
    }


def audit_recovery_manifest(
    manifest_path: Path,
    *,
    registry_path: Path | None = None,
    target_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    manifest = manifest_path.expanduser().resolve()
    payload = _read_json(manifest)
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Recovery manifest has no targets.")
    requested_targets = {str(value) for value in target_ids or ()}
    if requested_targets:
        targets = [
            target
            for target in targets
            if isinstance(target, Mapping)
            and str(target.get("target_id")) in requested_targets
        ]
        observed_targets = {
            str(target.get("target_id"))
            for target in targets
            if isinstance(target, Mapping)
        }
        missing_targets = sorted(requested_targets - observed_targets)
        if missing_targets:
            raise ValueError(
                "Requested targets are absent from the recovery manifest: "
                + ", ".join(missing_targets)
            )
    resolved_registry = (
        registry_path.expanduser().resolve()
        if registry_path is not None
        else Path(str(payload.get("registry") or "")).expanduser().resolve()
    )
    if not resolved_registry.is_file():
        raise FileNotFoundError(f"Registry not found: {resolved_registry}")
    registry = Registry(resolved_registry)
    results: list[dict[str, Any]] = []
    try:
        for target in targets:
            if not isinstance(target, Mapping):
                results.append(
                    {
                        "target_id": None,
                        "status": "unmigratable",
                        "error_type": "InvalidTarget",
                        "error": "Recovery target is not an object.",
                    }
                )
                continue
            try:
                results.append(_audit_target(target, registry=registry))
            except Exception as exc:
                results.append(
                    {
                        "target_id": target.get("target_id"),
                        "analysis_zarr": target.get("analysis_zarr"),
                        "status": "unmigratable",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    finally:
        registry.close()
    status = "migratable" if all(item.get("status") == "migratable" for item in results) else "unmigratable"
    report_payload = {
        "status": status,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_read_mode": "unconsolidated_explicit_read_only",
        "recovery_manifest": str(manifest),
        "recovery_manifest_sha256": _sha256_file(manifest),
        "registry": str(resolved_registry),
        "target_count": len(results),
        "migratable_target_count": sum(item.get("status") == "migratable" for item in results),
        "shard_count": sum(int(item.get("shard_count") or 0) for item in results),
        "migratable_shard_count": sum(int(item.get("migratable_shard_count") or 0) for item in results),
        "unmigratable_shard_count": sum(int(item.get("unmigratable_shard_count") or 0) for item in results),
        "targets": results,
        "production_state_changes": [],
    }
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(report_payload),
        "payload": report_payload,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--target-id", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    report = audit_recovery_manifest(
        args.recovery_manifest,
        registry_path=args.registry,
        target_ids=args.target_id,
    )
    if args.output_json is not None:
        write_json_atomic(args.output_json.expanduser().resolve(), report)
        printable: Mapping[str, Any] = {
            key: report["payload"][key]
            for key in (
                "status",
                "target_count",
                "migratable_target_count",
                "shard_count",
                "migratable_shard_count",
                "unmigratable_shard_count",
            )
        }
        printable = {
            **printable,
            "output_json": str(args.output_json.expanduser().resolve()),
            "payload_digest": report["payload_digest"],
        }
    else:
        printable = report
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if report["payload"]["status"] == "migratable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
