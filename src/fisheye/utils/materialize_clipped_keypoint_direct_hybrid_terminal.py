#!/usr/bin/env python3
"""Materialize admitted clipped shards as one direct-hybrid terminal artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.hybrid_crop_provider import (
    validate_hybrid_provider_strict_crop_geometry,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.keypoint_terminal_pixel_evidence import (
    validate_direct_hybrid_terminal_pixel_evidence,
)
from fisheye.shared.pose_inference_failure import (
    POSE_INFERENCE_FAILURE_SCHEMA_ID,
    POSE_INFERENCE_FAILURE_SCHEMA_VERSION,
    pose_inference_failure_code_map_json,
    pose_inference_failure_histogram,
    validate_pose_inference_failure_codes,
)
from fisheye.shared.pose_model_schema_binding import (
    validate_pose_model_schema_binding,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_consumer import (
    build_crop_run_reference,
    validate_crop_run_reference,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr.keypoint_manifest import keypoint_preprocessing_from_manifest
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    RUN_PROVENANCE_ATTR,
    is_run_complete,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.utils.audit_clipped_keypoint_direct_hybrid import (
    REPORT_SCHEMA_ID,
    REPORT_SCHEMA_VERSION,
)
from fisheye.utils.finalize_whole_recording_keypoint_v2 import (
    DIRECT_HYBRID_TERMINAL_RECEIPT_SCHEMA_VERSION,
)
from fisheye.utils.run_whole_recording_keypoint_terminal import (
    TERMINAL_RECEIPT_NAME,
    WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID,
)


MATERIALIZATION_SCHEMA_ID = "palette.keypoint.direct_hybrid_terminal_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1
_TERMINAL_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "frame_indices",
    "keypoints_roi",
    "keypoints_img",
    "keypoint_confidences",
    "confidence",
    "pose_bbox_xyxy_roi",
    "pose_bbox_xyxy_img",
    "detection_success",
    "pose_failure_codes",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _require_proof_report(path: Path, *, target_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = path.expanduser().resolve()
    report = _read_json(report_path)
    if (
        report.get("schema_id") != REPORT_SCHEMA_ID
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
    ):
        raise ValueError("Direct-hybrid proof report schema mismatch.")
    payload = report.get("payload")
    if not isinstance(payload, dict) or report.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Direct-hybrid proof report digest mismatch.")
    if payload.get("status") != "migratable" or payload.get("production_state_changes") != []:
        raise ValueError("Direct-hybrid proof report did not conclude migratable read-only evidence.")
    manifest = Path(str(payload.get("recovery_manifest") or "")).expanduser().resolve()
    if not manifest.is_file() or _sha256_file(manifest) != payload.get("recovery_manifest_sha256"):
        raise ValueError("Recovery manifest changed after the proof report was sealed.")
    matches = [
        item
        for item in payload.get("targets") or ()
        if isinstance(item, dict) and item.get("target_id") == target_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "migratable":
        raise ValueError("Proof report lacks exactly one migratable requested target.")
    return report, matches[0]


def _target_manifest_entry(report: Mapping[str, Any], *, target_id: str) -> dict[str, Any]:
    payload = report["payload"]
    manifest = _read_json(Path(payload["recovery_manifest"]))
    matches = [
        item
        for item in manifest.get("targets") or ()
        if isinstance(item, dict) and item.get("target_id") == target_id
    ]
    if len(matches) != 1:
        raise ValueError("Recovery manifest lacks exactly one requested target.")
    return matches[0]


def _create_terminal_arrays(group: Any, first: Any, *, row_count: int) -> None:
    for name in _TERMINAL_ARRAY_PATHS:
        if name not in first:
            raise ValueError(f"First source shard lacks terminal array {name!r}.")
        source = first[name]
        trailing = tuple(int(value) for value in source.shape[1:])
        chunks = (min(131_072, row_count), *trailing)
        group.create_array(
            name,
            shape=(row_count, *trailing),
            dtype=source.dtype,
            chunks=chunks,
        )


def _hash_update(hasher: Any, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    hasher.update(contiguous.view(np.uint8))


def materialize_direct_hybrid_terminal(
    *,
    proof_report: Path,
    target_id: str,
    terminal_run_id: str,
    terminal_output: Path,
) -> dict[str, Any]:
    """Revalidate proven source bytes and atomically publish a terminal artifact."""

    if not target_id or "/" in target_id or not terminal_run_id or "/" in terminal_run_id:
        raise ValueError("target_id and terminal_run_id must be path-safe components.")
    report, target_proof = _require_proof_report(proof_report, target_id=target_id)
    target = _target_manifest_entry(report, target_id=target_id)
    archive = Path(str(target["analysis_zarr"])).expanduser().resolve()
    crop_run = str(target["target_geometry_crop_run"])
    shard_runs = [str(value) for value in target["source_keypoint_shards"]]
    pixel_evidence = validate_direct_hybrid_terminal_pixel_evidence(
        target_proof.get("terminal_pixel_evidence")
    )
    if (
        pixel_evidence["geometry_crop_run"] != crop_run
        or pixel_evidence["source_shard_roster"]["shard_runs"] != shard_runs
        or pixel_evidence["source_shard_roster"]["evidence_digest"]
        != target_proof.get("source_shard_evidence_digest")
    ):
        raise ValueError("Proof report terminal pixel evidence differs from its target roster.")

    crop = open_persisted_crop_geometry_publication(archive, run_id=crop_run)
    if canonical_json_sha256(crop.manifest) != pixel_evidence["geometry_crop_manifest_digest"]:
        raise ValueError("Crop-v2 manifest changed after proof admission.")
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    provider_run = str(pixel_evidence["provider_run"])
    provider = root[f"crop_runs/{provider_run}"]
    provider_reference = validate_crop_run_reference(
        build_crop_run_reference(provider, run_id=provider_run)
    )
    if provider_reference != pixel_evidence["provider_reference"]:
        raise ValueError("Signed hybrid provider reference changed after proof admission.")
    live_provider = validate_hybrid_provider_strict_crop_geometry(
        provider,
        crop,
        expected_provider_record_sha256=str(
            pixel_evidence["provider_binding"]["provider_record_sha256"]
        ),
    )
    live_binding = {
        name: live_provider[name]
        for name in pixel_evidence["provider_binding"]
    }
    if live_binding != pixel_evidence["provider_binding"]:
        raise ValueError("Signed hybrid provider identity changed after proof admission.")

    model = target_proof.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Proof report lacks model evidence.")
    pose_binding = validate_pose_model_schema_binding(
        model.get("pose_model_schema_binding"),
        expected_model_sha256=str(model.get("sha256") or ""),
    )
    preprocessing = keypoint_preprocessing_from_manifest(
        target_proof.get("preprocessing")
    )
    source_reports = target_proof.get("shards")
    if not isinstance(source_reports, list) or len(source_reports) != len(shard_runs):
        raise ValueError("Proof report shard evidence roster is incomplete.")
    reports_by_run = {
        str(item.get("shard_run")): item
        for item in source_reports
        if isinstance(item, Mapping)
    }
    if set(reports_by_run) != set(shard_runs):
        raise ValueError("Proof report shard names differ from the recovery manifest.")

    row_count = int(np.asarray(crop.arrays["instance_key"]).shape[0])
    output = terminal_output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Terminal output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if temporary.exists():
        raise FileExistsError(f"Temporary terminal output already exists: {temporary}")
    temporary.mkdir(parents=False)
    try:
        terminal_root = zarr.open_group(str(temporary), mode="w", zarr_format=3)
        parent = require_runs_parent(terminal_root, "keypoint_terminal_runs")
        terminal = parent.create_group(terminal_run_id)
        mark_run_started(terminal, run_name=terminal_run_id, stage="keypoint_terminal")
        terminal.attrs.update(
            {
                "stage_selector_eligible": False,
                "registry_registered": False,
                "artifact_profile": pixel_evidence["profile"],
                "source_analysis_zarr": str(archive),
                "source_crop_run": provider_run,
                "geometry_crop_run": crop_run,
                "source_keypoint_shard_runs": shard_runs,
            }
        )
        first = root[f"keypoint_shard_runs/{shard_runs[0]}"]
        _create_terminal_arrays(terminal, first, row_count=row_count)
        output_hashers = {name: hashlib.sha256() for name in _TERMINAL_ARRAY_PATHS}
        expected_start = 0
        success_count = 0
        failure_histogram_arrays: list[np.ndarray] = []
        for shard_run in shard_runs:
            group = root[f"keypoint_shard_runs/{shard_run}"]
            if not is_run_complete(group):
                raise ValueError(f"Source shard became incomplete: {shard_run}")
            report_item = reports_by_run[shard_run]
            expected_hashes = report_item.get("scientific_array_hashes")
            if not isinstance(expected_hashes, Mapping) or not expected_hashes:
                raise ValueError(f"Proof report lacks source hashes for {shard_run}.")
            arrays: dict[str, np.ndarray] = {}
            for name, expected_digest in expected_hashes.items():
                if name not in group:
                    raise ValueError(f"Source shard lost array {name!r}: {shard_run}")
                values = np.asarray(group[name][...])
                arrays[str(name)] = values
                if sha256_array(values) != expected_digest:
                    raise ValueError(f"Source shard array changed after proof: {shard_run}/{name}")
            for name in _TERMINAL_ARRAY_PATHS:
                if name not in arrays:
                    arrays[name] = np.asarray(group[name][...])
            rows = np.asarray(arrays["source_crop_row_ids"])
            if (
                rows.dtype != np.dtype(np.int64)
                or rows.ndim != 1
                or not np.array_equal(
                    rows,
                    np.arange(expected_start, expected_start + rows.size, dtype=np.int64),
                )
            ):
                raise ValueError(
                    f"Source shard roster is not one ordered contiguous crop partition: {shard_run}"
                )
            stop = expected_start + int(rows.size)
            for name in _TERMINAL_ARRAY_PATHS:
                values = arrays[name]
                terminal[name][expected_start:stop] = values
                _hash_update(output_hashers[name], values)
            success = np.asarray(arrays["detection_success"], dtype=bool)
            failures = np.asarray(arrays["pose_failure_codes"], dtype=np.uint8)
            validate_pose_inference_failure_codes(failures, pose_success=success)
            success_count += int(np.count_nonzero(success))
            failure_histogram_arrays.append(failures)
            expected_start = stop
        if expected_start != row_count:
            raise ValueError(
                f"Source shard roster materialized {expected_start} rows; expected {row_count}."
            )

        run_provenance = build_writer_run_provenance(
            command="fisheye.utils.materialize_clipped_keypoint_direct_hybrid_terminal",
            params={
                "target_id": target_id,
                "terminal_run_id": terminal_run_id,
                "artifact_profile": pixel_evidence["profile"],
                "source_shard_count": len(shard_runs),
            },
            input_run_ids={
                "geometry_crop": crop_run,
                "hybrid_provider": provider_run,
                "keypoint_shards": shard_runs,
                "proof_report_digest": report["payload_digest"],
            },
            cwd=Path.cwd(),
        )
        terminal.attrs[RUN_PROVENANCE_ATTR] = run_provenance
        mark_run_complete(
            terminal,
            parent_group=parent,
            run_name=terminal_run_id,
            run_provenance=run_provenance,
        )
        consolidate_metadata_capture_expected_warnings(temporary)
        source_array_hashes = {
            name: hasher.hexdigest() for name, hasher in output_hashers.items()
        }
        all_failure_codes = np.concatenate(failure_histogram_arrays)
        payload = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "recording_id": target_id,
            "analysis_zarr": str(archive),
            "crop_run": provider_run,
            "terminal_run_id": terminal_run_id,
            "terminal_group_path": f"keypoint_terminal_runs/{terminal_run_id}",
            "row_count": row_count,
            "terminal_success_count": success_count,
            "terminal_failure_count": row_count - success_count,
            "pose_failure_codes": {
                "schema_id": POSE_INFERENCE_FAILURE_SCHEMA_ID,
                "schema_version": POSE_INFERENCE_FAILURE_SCHEMA_VERSION,
                "array_path": "pose_failure_codes",
                "dtype": "uint8",
                "code_map": pose_inference_failure_code_map_json(),
                "histogram": pose_inference_failure_histogram(all_failure_codes),
                "success_alignment": "code_zero_iff_detection_success_true",
                "public_raw_v2_array": False,
            },
            "source_array_hashes": source_array_hashes,
            "pixel_evidence": pixel_evidence,
            "model": {
                **{key: value for key, value in model.items() if key != "pose_model_schema_binding"},
                "pose_model_schema_binding": pose_binding,
                "pose_model_schema_binding_digest": canonical_json_sha256(pose_binding),
            },
            "preprocessing": preprocessing.as_manifest(),
            "row_terminal_semantics": "every_crop_row_present_with_exact_pose_failure_code_v2",
            "selector_eligible": False,
            "registry_registered": False,
            "production_state_changes": [],
        }
        receipt = {
            "schema_id": WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID,
            "schema_version": DIRECT_HYBRID_TERMINAL_RECEIPT_SCHEMA_VERSION,
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "payload_digest": canonical_json_sha256(payload),
            "payload": payload,
        }
        write_json_atomic(temporary / TERMINAL_RECEIPT_NAME, receipt)
        os.replace(temporary, output)
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": "complete",
            "target_id": target_id,
            "terminal_output": str(output),
            "terminal_run_id": terminal_run_id,
            "row_count": row_count,
            "source_shard_count": len(shard_runs),
            "terminal_receipt_digest": receipt["payload_digest"],
            "pixel_evidence_profile": pixel_evidence["profile"],
            "production_state_changes": [],
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-report", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--terminal-run-id", required=True)
    parser.add_argument("--terminal-output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args(argv)
    result = materialize_direct_hybrid_terminal(
        proof_report=args.proof_report,
        target_id=args.target_id,
        terminal_run_id=args.terminal_run_id,
        terminal_output=args.terminal_output,
    )
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
