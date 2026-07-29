"""Run one isolated local refined-detection compaction benchmark.

The driver copies an existing selector-ineligible refined-v1 standalone store
to fresh local scratch, proves logical equality, appends one manual detection
to an empty frame in a separate delta store, rolls the generation, and invokes
the immutable compactor.  It never writes the source, copies back to shared
storage, or changes a selector/registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.benchmark_runtime import storage_stats, utc_now
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_compaction import (
    compact_frozen_refined_detection_delta_generation,
    validate_refined_detection_compaction_receipt,
)
from fisheye.shared.zarr.refined_detection_delta import (
    REFINED_DETECTION_DELTA_ARRAYS,
    REFINED_DETECTION_DELTA_OPERATION_CODE_MAP,
    RefinedDetectionDeltaBatch,
)
from fisheye.shared.zarr.refined_detection_delta_storage import (
    RefinedDetectionDeltaLineageBinding,
    create_refined_detection_delta_lineage,
    rollover_refined_detection_delta_generation,
    write_refined_detection_delta_partition,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    refined_detection_logical_content_digest,
    validate_refined_detection_publication,
    validate_refined_detection_run_manifest,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    refined_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.refined_detection_storage import (
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest


BENCHMARK_SCHEMA_ID = "palette.refined_detection.local_compaction_benchmark"
BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_RECEIPT_NAME = "benchmark_receipt.json"
_BENCHMARK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_UUID_NAMESPACE = uuid.UUID("8d6ac195-1913-4424-96bf-ae4156a8731a")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _palette_provenance() -> Mapping[str, Any]:
    repository = Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    relative_driver = Path(__file__).resolve().relative_to(repository)
    return {
        "repository": str(repository),
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "worktree_clean": git("status", "--short") == "",
        "benchmark_driver": str(relative_driver),
        "benchmark_driver_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _require_safe_work_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    temporary_root = Path("/tmp").resolve()
    safe = resolved.is_relative_to(temporary_root) or ".palette_scratch" in (
        resolved.parts
    )
    if not safe:
        raise ValueError("Work root must be below /tmp or a .palette_scratch path.")
    if resolved.exists():
        raise FileExistsError(f"Work root must be fresh: {resolved}")
    if any(part.endswith("_analysis.zarr") for part in resolved.parts):
        raise ValueError("Work root cannot be inside a recording archive.")
    return resolved


def _dimensions(payload: Mapping[str, Any]) -> RefinedDetectionDimensions:
    logical = payload.get("logical_schema")
    raw = logical.get("dimensions") if isinstance(logical, Mapping) else None
    if not isinstance(raw, Mapping):
        raise ValueError("Refined manifest lacks logical dimensions.")
    dimensions = RefinedDetectionDimensions(
        n_frames=raw["n_frames"],
        n_instances=raw["n_instances"],
        n_source_detections=raw["n_source_detections"],
        source_width=raw["source_width"],
        source_height=raw["source_height"],
        lineage_profile=raw["lineage_profile"],
    )
    if (
        dimensions.lineage_profile
        is not RefinedDetectionLineageProfile.FULL_ACQUISITION
    ):
        raise ValueError("The local benchmark requires a full-acquisition snapshot.")
    return dimensions


def _open_refined_snapshot(
    path: Path,
    *,
    run_id: str,
) -> tuple[Mapping[str, Any], RefinedDetectionDimensions, dict[str, Any]]:
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    run = root[f"refined_detect_runs/{run_id}"]
    manifest = dict(run.attrs["run_manifest"])
    errors = validate_refined_detection_run_manifest(manifest)
    if errors:
        raise ValueError("Refined base manifest is invalid: " + "; ".join(errors))
    payload = manifest["payload"]
    dimensions = _dimensions(payload)
    arrays = {
        array_path: run[array_path]
        for array_path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
    }
    REFINED_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)
    if manifest["payload"]["publication"]["stage_selector_eligible"] is not False:
        raise ValueError("The benchmark source must be selector-ineligible.")
    return manifest, dimensions, arrays


def _manual_add_batch(
    *,
    delta_lineage_id: str,
    base_snapshot_id: str,
    base_manifest_digest: str,
    recording_identity: str,
    refined_row_id: int,
    frame_index: int,
    timestamp_ns: int,
) -> RefinedDetectionDeltaBatch:
    bbox = np.asarray([0.5, 0.5, 0.1, 0.1], dtype=np.float32)
    class_id = 0
    instance_key = int(
        mint_manual_curation_instance_keys(
            recording_identity=recording_identity,
            refined_row_ids=np.asarray([refined_row_id], dtype=np.int64),
            frame_indices=np.asarray([frame_index], dtype=np.int32),
            bbox_norm_coords=bbox.reshape(1, 4),
            class_ids=np.asarray([class_id], dtype=np.int32),
        )[0]
    )
    event = {
        "event_sequence": 1,
        "expected_previous_event_sequence": 0,
        "operation_codes": REFINED_DETECTION_DELTA_OPERATION_CODE_MAP["add_instance"],
        "instance_key": instance_key,
        "refined_row_ids": refined_row_id,
        "row_index_hint": -1,
        "timestamp_ns": timestamp_ns,
        "reason_codes": 1,
        "payload_valid": True,
        "frame_indices": frame_index,
        "source_acquisition_frame_index": frame_index,
        "bbox_norm_coords": bbox,
        "scores": 0.0,
        "score_valid": False,
        "class_ids": class_id,
        "source_kind_codes": SOURCE_KIND_CODE_MAP["manual"],
        "manual_edit_flags": True,
        "source_detect_row_index": -1,
    }
    arrays = {
        declaration.name: np.asarray(
            [event[declaration.name]],
            dtype=np.dtype(declaration.dtype),
        ).reshape(1, *declaration.trailing_shape)
        for declaration in REFINED_DETECTION_DELTA_ARRAYS
    }
    return RefinedDetectionDeltaBatch(
        delta_lineage_id=delta_lineage_id,
        base_snapshot_id=base_snapshot_id,
        base_manifest_digest=base_manifest_digest,
        generation_ordinal=0,
        partition_id="manual_add_empty_frame",
        actor_id="local_compaction_benchmark",
        reason_code_map={0: "none", 1: "benchmark_manual_add"},
        arrays=arrays,
    )


def run_local_compaction_benchmark(
    *,
    source_zarr: Path,
    source_run_id: str,
    work_root: Path,
    benchmark_id: str,
) -> Mapping[str, Any]:
    """Copy, verify, edit, compact, and return one strict benchmark receipt."""

    resolved_id = str(benchmark_id).strip()
    if not _BENCHMARK_ID_RE.fullmatch(resolved_id):
        raise ValueError("benchmark_id must be lowercase and path-safe.")
    source_path = source_zarr.expanduser().resolve()
    if not source_path.is_dir() or source_path.suffix != ".zarr":
        raise ValueError(f"source_zarr is not a Zarr directory: {source_path}")
    destination_root = _require_safe_work_root(work_root)
    destination_root.mkdir(parents=True)
    input_path = destination_root / "input" / "base.zarr"
    input_path.parent.mkdir()

    total_started = time.perf_counter()
    source_manifest, source_dimensions, source_arrays = _open_refined_snapshot(
        source_path,
        run_id=source_run_id,
    )
    phase_started = time.perf_counter()
    source_content_digest = refined_detection_logical_content_digest(
        source_arrays,
        dimensions=source_dimensions,
    )
    source_hash_seconds = time.perf_counter() - phase_started
    source_stats = storage_stats(source_path)

    phase_started = time.perf_counter()
    shutil.copytree(source_path, input_path, copy_function=shutil.copy2)
    copy_to_local_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    local_manifest, local_dimensions, local_arrays = _open_refined_snapshot(
        input_path,
        run_id=source_run_id,
    )
    local_content_digest = refined_detection_logical_content_digest(
        local_arrays,
        dimensions=local_dimensions,
    )
    local_validation_seconds = time.perf_counter() - phase_started
    if local_manifest != source_manifest or local_dimensions != source_dimensions:
        raise RuntimeError("Local copied manifest or dimensions differ from source.")
    if local_content_digest != source_content_digest:
        raise RuntimeError("Local copied arrays differ logically from source.")

    phase_started = time.perf_counter()
    source_content_digest_after_copy = refined_detection_logical_content_digest(
        source_arrays,
        dimensions=source_dimensions,
    )
    source_recheck_seconds = time.perf_counter() - phase_started
    if source_content_digest_after_copy != source_content_digest:
        raise RuntimeError("Source arrays changed while the local copy was built.")

    payload = local_manifest["payload"]
    lineage = payload["snapshot_lineage"]
    allocator = lineage["refined_row_id_allocator"]
    key_allocator = lineage["manual_instance_key_allocator"]
    offsets = np.asarray(
        local_arrays["instances/frame_row_offsets"][:],
        dtype=np.int64,
    )
    empty_frames = np.flatnonzero(np.diff(offsets) == 0)
    if empty_frames.size == 0:
        raise RuntimeError("Benchmark source has no empty frame for a manual add.")
    target_frame = int(empty_frames[0])
    delta_lineage_id = str(
        uuid.uuid5(
            _UUID_NAMESPACE,
            f"{resolved_id}:{local_manifest['payload_digest']}:delta",
        )
    )
    successor_snapshot_id = str(
        uuid.uuid5(
            _UUID_NAMESPACE,
            f"{resolved_id}:{local_manifest['payload_digest']}:snapshot",
        )
    )
    created_at = utc_now()
    phase_started = time.perf_counter()
    delta_path = destination_root / "delta.zarr"
    delta_root = zarr.open_group(str(delta_path), mode="w", zarr_format=3)
    create_refined_detection_delta_lineage(
        delta_root,
        binding=RefinedDetectionDeltaLineageBinding(
            delta_lineage_id=delta_lineage_id,
            base_run_path=f"refined_detect_runs/{source_run_id}",
            base_snapshot_id=str(lineage["snapshot_id"]),
            base_manifest_digest=str(local_manifest["payload_digest"]),
            base_logical_content_digest=local_content_digest,
            recording_identity=str(key_allocator["recording_identity"]),
            base_next_refined_row_id=int(allocator["next_id"]),
        ),
        created_by="local_compaction_benchmark",
        created_at_utc=created_at,
    )
    write_refined_detection_delta_partition(
        delta_root,
        batch=_manual_add_batch(
            delta_lineage_id=delta_lineage_id,
            base_snapshot_id=str(lineage["snapshot_id"]),
            base_manifest_digest=str(local_manifest["payload_digest"]),
            recording_identity=str(key_allocator["recording_identity"]),
            refined_row_id=int(allocator["next_id"]),
            frame_index=target_frame,
            timestamp_ns=time.time_ns(),
        ),
        created_at_utc=created_at,
    )
    delta_authoring_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    rollover_receipt = rollover_refined_detection_delta_generation(
        delta_root,
        delta_lineage_id=delta_lineage_id,
        generation_ordinal=0,
        next_generation_ordinal=1,
        actor_id="local_compaction_benchmark",
        frozen_at_utc=created_at,
        next_created_at_utc=created_at,
    )
    rollover_seconds = time.perf_counter() - phase_started

    output_root = destination_root / "output"
    compacted = compact_frozen_refined_detection_delta_generation(
        delta_root=delta_root,
        delta_lineage_id=delta_lineage_id,
        generation_ordinal=0,
        base_manifest=local_manifest,
        base_arrays=local_arrays,
        destination=output_root / "compacted.zarr",
        run_id=f"refined_compacted_{resolved_id}",
        snapshot_id=successor_snapshot_id,
        created_by="local_compaction_benchmark",
        safe_root=output_root,
    )
    receipt_payload: dict[str, object] = {
        "benchmark_id": resolved_id,
        "status": "complete",
        "palette_provenance": _palette_provenance(),
        "source": {
            "path": str(source_path),
            "run_id": source_run_id,
            "manifest_digest": source_manifest["payload_digest"],
            "logical_content_digest": source_content_digest,
            "dimensions": source_dimensions.as_manifest(),
            "storage_stats": source_stats,
        },
        "local_input": {
            "path": str(input_path),
            "manifest_digest": local_manifest["payload_digest"],
            "logical_content_digest": local_content_digest,
            "storage_stats": storage_stats(input_path),
            "exact_source_equality": True,
        },
        "edit": {
            "operation": "add_instance",
            "target_was_empty_frame": True,
            "frame_index": target_frame,
            "refined_row_id": int(allocator["next_id"]),
            "delta_lineage_id": delta_lineage_id,
            "rollover": dict(rollover_receipt),
        },
        "output": {
            "path": str(compacted.publication.output_path),
            "run_id": compacted.publication.run_id,
            "manifest_digest": compacted.publication.manifest["payload_digest"],
            "logical_content_digest": compacted.publication.receipt[
                "logical_content_digest"
            ],
            "compaction_receipt_digest": compacted.receipt["payload_digest"],
            "storage_stats": storage_stats(compacted.publication.output_path),
        },
        "phase_seconds": {
            "source_logical_hash": source_hash_seconds,
            "copy_source_to_local": copy_to_local_seconds,
            "open_hash_validate_local_copy": local_validation_seconds,
            "source_postcopy_immutability_recheck": source_recheck_seconds,
            "author_delta_partition": delta_authoring_seconds,
            "generation_freeze_and_successor_open": rollover_seconds,
            "compactor_total_before_receipt": compacted.receipt["payload"][
                "phase_seconds"
            ]["total_before_receipt"],
            "compactor": compacted.receipt["payload"]["phase_seconds"],
            "end_to_end_before_driver_receipt": time.perf_counter() - total_started,
        },
        "copy_back": {
            "performed": False,
            "seconds": None,
            "destination": None,
        },
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": BENCHMARK_SCHEMA_ID,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(receipt_payload),
        "payload": receipt_payload,
    }
    with (destination_root / BENCHMARK_RECEIPT_NAME).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            receipt,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    return receipt


def _read_strict_json(path: Path) -> Mapping[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def validate_local_compaction_benchmark_receipt(
    receipt_path: Path,
) -> tuple[str, ...]:
    """Reopen every local artifact and independently verify one benchmark."""

    errors: list[str] = []
    try:
        receipt = _read_strict_json(receipt_path.expanduser().resolve())
    except (OSError, TypeError, ValueError) as exc:
        return (f"cannot read strict benchmark receipt: {exc}",)
    if set(receipt) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("benchmark receipt envelope has an unexpected field set")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "benchmark receipt payload must be an object")
    if (
        receipt.get("schema_id") != BENCHMARK_SCHEMA_ID
        or receipt.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("benchmark receipt schema or digest algorithm mismatch")
    if receipt.get("payload_digest") != canonical_json_sha256(payload):
        errors.append("benchmark receipt payload digest mismatch")
    if (
        payload.get("status") != "complete"
        or payload.get("selector_eligible") is not False
        or payload.get("registry_registered") is not False
        or payload.get("production_state_changes") != []
    ):
        errors.append("benchmark receipt safety status is invalid")
    provenance = payload.get("palette_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("benchmark receipt lacks Palette provenance")
    elif provenance.get("worktree_clean") is not True:
        errors.append("benchmark was not executed from a clean Palette worktree")
    copy_back = payload.get("copy_back")
    if not isinstance(copy_back, Mapping) or copy_back.get("performed") is not False:
        errors.append("benchmark receipt must prove copy-back was not performed")

    try:
        source = payload["source"]
        local = payload["local_input"]
        output = payload["output"]
        edit = payload["edit"]
        source_manifest, source_dimensions, source_arrays = _open_refined_snapshot(
            Path(str(source["path"])),
            run_id=str(source["run_id"]),
        )
        local_manifest, local_dimensions, local_arrays = _open_refined_snapshot(
            Path(str(local["path"])),
            run_id=str(source["run_id"]),
        )
        output_manifest, output_dimensions, output_arrays = _open_refined_snapshot(
            Path(str(output["path"])),
            run_id=str(output["run_id"]),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return (*errors, f"cannot reopen benchmark snapshots: {exc}")

    if source_manifest != local_manifest or source_dimensions != local_dimensions:
        errors.append("source and local-copy manifests or dimensions differ")
    source_digest = refined_detection_logical_content_digest(
        source_arrays,
        dimensions=source_dimensions,
    )
    local_digest = refined_detection_logical_content_digest(
        local_arrays,
        dimensions=local_dimensions,
    )
    output_digest = refined_detection_logical_content_digest(
        output_arrays,
        dimensions=output_dimensions,
    )
    if source_digest != local_digest or source_digest != source.get(
        "logical_content_digest"
    ):
        errors.append("source and local-copy logical content differ")
    if output_digest != output.get("logical_content_digest"):
        errors.append("output logical content digest differs from receipt")
    if (
        output_dimensions.n_frames != source_dimensions.n_frames
        or output_dimensions.n_source_detections
        != source_dimensions.n_source_detections
        or output_dimensions.n_instances != source_dimensions.n_instances + 1
    ):
        errors.append("output dimensions do not represent exactly one manual add")

    source_counts = np.diff(
        np.asarray(source_arrays["instances/frame_row_offsets"][:], dtype=np.int64)
    )
    output_offsets = np.asarray(
        output_arrays["instances/frame_row_offsets"][:],
        dtype=np.int64,
    )
    output_counts = np.diff(output_offsets)
    target_frame = int(edit.get("frame_index", -1))
    expected_counts = source_counts.copy()
    if 0 <= target_frame < expected_counts.size:
        if expected_counts[target_frame] != 0:
            errors.append("manual-add target was not empty in the source")
        expected_counts[target_frame] += 1
    else:
        errors.append("manual-add target frame is out of bounds")
    if not np.array_equal(output_counts, expected_counts):
        errors.append("output offsets differ by more than the declared manual add")
    elif 0 <= target_frame < output_counts.size:
        row = int(output_offsets[target_frame])
        expected_row_id = int(edit.get("refined_row_id", -1))
        checks = {
            "refined_row_id": int(output_arrays["instances/refined_row_ids"][row])
            == expected_row_id,
            "manual_edit": bool(output_arrays["instances/manual_edit_flags"][row]),
            "manual_kind": int(output_arrays["instances/source_kind_codes"][row])
            == SOURCE_KIND_CODE_MAP["manual"],
            "score_invalid": not bool(output_arrays["instances/score_valid"][row]),
            "no_raw_source": int(
                output_arrays["instances/source_detect_row_index"][row]
            )
            == -1,
        }
        errors.extend(
            f"manual output row failed {name}"
            for name, passed in checks.items()
            if not passed
        )

    for path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(source_dimensions):
        if not path.startswith("source_detections/"):
            continue
        if not np.array_equal(source_arrays[path][:], output_arrays[path][:]):
            errors.append(f"manual add changed source-audit array {path!r}")

    try:
        profile = storage_profile_from_manifest(
            output_manifest["payload"]["storage_plan"]["storage_profile"]
        )
        plans = plan_refined_detection_storage(
            output_dimensions,
            profile=profile,
        )
        direct, consolidated = refined_detection_metadata_declaration_maps(
            Path(str(output["path"])),
            run_id=str(output["run_id"]),
            plans=plans,
        )
        publication_errors = validate_refined_detection_publication(
            output_manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=output_arrays,
            parent_manifest=local_manifest,
            parent_arrays=local_arrays,
        )
        errors.extend(
            f"output publication validation: {error}" for error in publication_errors
        )
        compaction_receipt = _read_strict_json(
            Path(str(output["path"])) / "compaction_benchmark_receipt.json"
        )
        errors.extend(
            f"compaction receipt: {error}"
            for error in validate_refined_detection_compaction_receipt(
                compaction_receipt
            )
        )
        if compaction_receipt.get("payload_digest") != output.get(
            "compaction_receipt_digest"
        ):
            errors.append(
                "nested compaction receipt digest differs from driver receipt"
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"cannot validate output publication evidence: {exc}")
    return tuple(dict.fromkeys(errors))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path)
    parser.add_argument("--source-run-id")
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--benchmark-id")
    parser.add_argument(
        "--verify-receipt",
        type=Path,
        help="Reopen and independently validate an existing benchmark receipt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.verify_receipt is not None:
        errors = validate_local_compaction_benchmark_receipt(args.verify_receipt)
        print(
            json.dumps(
                {"status": "pass" if not errors else "fail", "errors": errors},
                indent=2,
            )
        )
        return int(bool(errors))
    missing = [
        name
        for name in ("source_zarr", "source_run_id", "work_root", "benchmark_id")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "a benchmark run requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    receipt = run_local_compaction_benchmark(
        source_zarr=args.source_zarr,
        source_run_id=args.source_run_id,
        work_root=args.work_root,
        benchmark_id=args.benchmark_id,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
