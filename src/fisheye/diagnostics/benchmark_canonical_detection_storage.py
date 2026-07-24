#!/usr/bin/env python3
"""Build and benchmark disposable canonical detection Zarr candidates.

The source group is opened read-only. Destinations must be fresh paths below
``/tmp/palette-zarr-benchmarks``; this utility cannot write into a Palette
recording archive or update selectors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import zarr

from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_contracts import (
    BenchmarkPhase,
    EAGER_FULL_READ_V1,
    StorageBenchmarkCase,
    WINDOWED_ROWS_READ_V1,
    WRITE_MATERIALIZATION_V1,
    benchmark_result_envelope,
)
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.detection_storage import (
    CanonicalDetectionStoragePlanSet,
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import (
    KIB,
    MIB,
    make_benchmark_storage_profile,
)


BENCHMARK_OUTPUT_ROOT = Path("/tmp/palette-zarr-benchmarks")
REPORT_SCHEMA_ID = "palette.canonical_detection_storage_benchmark"
REPORT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_stats(path: Path) -> dict[str, int]:
    stats = {
        "file_count": 0,
        "metadata_file_count": 0,
        "payload_file_count": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
    }
    for root, _directories, filenames in os.walk(path):
        for filename in filenames:
            item = Path(root) / filename
            result = item.stat()
            stats["file_count"] += 1
            if filename == "zarr.json":
                stats["metadata_file_count"] += 1
            else:
                stats["payload_file_count"] += 1
            stats["apparent_bytes"] += int(result.st_size)
            stats["allocated_bytes"] += int(result.st_blocks * 512)
    return stats


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if platform.system() != "Darwin" else value


@dataclass(frozen=True)
class CanonicalDetectionBenchmarkInput:
    """Validated canonical arrays held in memory before timed writes."""

    dimensions: CanonicalDetectionDimensions
    arrays: Mapping[str, np.ndarray]
    source_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        expected = CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        if tuple(self.arrays) != expected:
            raise ValueError(
                "Benchmark input arrays must match canonical binding order exactly."
            )
        CANONICAL_DETECTION_SCHEMA_V1.require(
            self.arrays,
            dimensions=self.dimensions,
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.canonical_detection_benchmark_input",
            "schema_version": 1,
            "dimensions": self.dimensions.as_manifest(),
            "source_identity": dict(self.source_identity),
            "canonical_arrays": {
                path: {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": _sha256_array(values),
                }
                for path, values in self.arrays.items()
            },
        }


def build_canonical_detection_benchmark_input(
    source_arrays: Mapping[str, Any],
    *,
    recording_identity: str,
    frame_count: int,
    source_width: int,
    source_height: int,
    frame_limit: int | None = None,
    source_identity: Mapping[str, object] | None = None,
) -> CanonicalDetectionBenchmarkInput:
    """Convert one legacy/current detect table to exact canonical v1 arrays."""

    total_frames = int(frame_count)
    if total_frames < 0:
        raise ValueError("frame_count cannot be negative.")
    selected_frames = total_frames if frame_limit is None else int(frame_limit)
    if selected_frames < 0 or selected_frames > total_frames:
        raise ValueError("frame_limit must be within the source frame domain.")

    source_frame_indices = np.asarray(source_arrays["frame_indices"][:])
    source_bbox = np.asarray(source_arrays["bbox_norm_coords"][:])
    source_scores = np.asarray(source_arrays["scores"][:])
    source_class_ids = np.asarray(source_arrays["class_ids"][:])
    row_count = int(source_frame_indices.shape[0])
    if not (
        source_bbox.shape == (row_count, 4)
        and source_scores.shape == (row_count,)
        and source_class_ids.shape == (row_count,)
    ):
        raise ValueError("Source detection arrays do not share one row cardinality.")

    source_frames_i64 = np.asarray(source_frame_indices, dtype=np.int64)
    if source_frames_i64.size > 1 and np.any(np.diff(source_frames_i64) < 0):
        raise ValueError("Source detection rows must already be frame sorted.")
    stop = int(np.searchsorted(source_frames_i64, selected_frames, side="left"))
    frame_indices = np.asarray(source_frame_indices[:stop], dtype=np.int32)
    bbox_norm = np.asarray(source_bbox[:stop], dtype=np.float32)
    scores = np.asarray(source_scores[:stop], dtype=np.float32)
    class_ids = np.asarray(source_class_ids[:stop], dtype=np.int32)
    source_acquisition_frames = frame_indices.astype(np.int64)
    instance_keys = mint_detection_instance_keys(
        recording_identity=str(recording_identity),
        frame_indices=frame_indices,
        bbox_norm_coords=bbox_norm,
        class_ids=class_ids,
    )
    bbox_img, centers_img = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=int(source_width),
        source_height=int(source_height),
    )
    counts = np.bincount(
        frame_indices.astype(np.int64, copy=False),
        minlength=selected_frames,
    )
    offsets = np.zeros(selected_frames + 1, dtype=np.int64)
    if selected_frames:
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

    dimensions = CanonicalDetectionDimensions(
        n_frames=selected_frames,
        n_instances=stop,
        source_width=int(source_width),
        source_height=int(source_height),
    )
    arrays = {
        "instances/frame_indices": frame_indices,
        "instances/source_acquisition_frame_index": source_acquisition_frames,
        "instances/instance_key": instance_keys,
        "instances/bbox_norm_coords": bbox_norm,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers_img,
        "instances/scores": scores,
        "instances/class_ids": class_ids,
        "instances/frame_row_offsets": offsets,
    }
    identity = {
        **dict(source_identity or {}),
        "recording_identity": str(recording_identity),
        "source_frame_count": total_frames,
        "selected_frame_count": selected_frames,
        "source_detection_rows": row_count,
        "selected_detection_rows": stop,
        "conversion": {
            "bbox_norm_coords": f"{source_bbox.dtype}->float32",
            "geometry_projection": "canonical_float32_exact",
            "source_acquisition_frame_index": "widened_frame_indices_identity",
            "frame_row_offsets": "cumsum_bincount_frame_indices",
            "instance_key": "minted_from_canonical_float32_bbox",
        },
    }
    return CanonicalDetectionBenchmarkInput(
        dimensions=dimensions,
        arrays=arrays,
        source_identity=identity,
    )


def load_detection_benchmark_input(
    source_group_path: Path,
    *,
    recording_identity: str,
    frame_limit: int | None,
) -> CanonicalDetectionBenchmarkInput:
    """Read one disposable legacy detection source group without mutation."""

    source_path = source_group_path.expanduser().resolve()
    metadata_path = source_path / "zarr.json"
    if not metadata_path.is_file():
        raise ValueError(f"Source is not a Zarr v3 group: {source_path}")
    source = zarr.open_group(
        str(source_path),
        mode="r",
        use_consolidated=False,
    )
    count_name = "frame_counts" if "frame_counts" in source else "n_detections"
    frame_count = int(source[count_name].shape[0])
    source_width = int(
        source.attrs.get("source_video_width") or source.attrs.get("source_full_width")
    )
    source_height = int(
        source.attrs.get("source_video_height")
        or source.attrs.get("source_full_height")
    )
    return build_canonical_detection_benchmark_input(
        source,
        recording_identity=recording_identity,
        frame_count=frame_count,
        source_width=source_width,
        source_height=source_height,
        frame_limit=frame_limit,
        source_identity={
            "source_group": str(source_path),
            "source_group_metadata_sha256": _sha256_file(metadata_path),
            "source_open_mode": "read_only_direct_metadata",
        },
    )


def _require_safe_destination(destination: Path) -> Path:
    path = destination.expanduser().resolve()
    root = BENCHMARK_OUTPUT_ROOT.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Benchmark destination must be below {root}.")
    if path.exists():
        raise FileExistsError(f"Benchmark destination already exists: {path}")
    return path


def _write_array_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> None:
    unit_rows = int(
        plan.shard_shape[0] if plan.shard_shape is not None else plan.chunk_shape[0]
    )
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def _environment_manifest() -> dict[str, object]:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "storage_tier": "local_tmp",
        "cache_state": "uncontrolled_exploratory_smoke",
        "request_counting": "unavailable_local_filesystem",
    }


def write_detection_benchmark_candidate(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    destination: Path,
    plans: CanonicalDetectionStoragePlanSet,
) -> dict[str, object]:
    """Write, validate, consolidate, and smoke-read one fresh candidate."""

    output_path = _require_safe_destination(destination)
    if plans.dimensions != benchmark_input.dimensions:
        raise ValueError("Storage plan dimensions do not match benchmark input.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "schema_id": REPORT_SCHEMA_ID,
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "logical_schema": CANONICAL_DETECTION_SCHEMA_V1.as_manifest(
                dimensions=benchmark_input.dimensions
            ),
            "storage_plan": plans.as_manifest(),
        }
    )
    instances = root.create_group("instances")
    write_results: list[dict[str, object]] = []
    destination_arrays: dict[str, Any] = {}
    for entry in plans.entries:
        path = entry.rule.path
        leaf = path.rsplit("/", 1)[-1]
        values = benchmark_input.arrays[path]
        binding = next(
            item for item in CANONICAL_DETECTION_SCHEMA_V1.bindings if item.path == path
        )
        contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        array_started = time.perf_counter()
        destination_array = create_array_from_plan(
            instances,
            name=leaf,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
        )
        _write_array_by_physical_units(
            destination_array,
            values,
            plan=entry.plan,
        )
        write_seconds = float(time.perf_counter() - array_started)
        destination_arrays[path] = destination_array
        write_results.append(
            {
                "path": path,
                "write_seconds": write_seconds,
                "logical_bytes": int(values.nbytes),
                "peak_rss_bytes": _peak_rss_bytes(),
                "plan": entry.plan.as_dict(),
            }
        )

    validation_started = time.perf_counter()
    CANONICAL_DETECTION_SCHEMA_V1.require(
        destination_arrays,
        dimensions=benchmark_input.dimensions,
    )
    digest_validation: dict[str, dict[str, object]] = {}
    for path, source_values in benchmark_input.arrays.items():
        destination_values = np.asarray(destination_arrays[path][:])
        source_digest = _sha256_array(source_values)
        destination_digest = _sha256_array(destination_values)
        digest_validation[path] = {
            "source_sha256": source_digest,
            "destination_sha256": destination_digest,
            "exact": source_digest == destination_digest,
        }
    validation_seconds = float(time.perf_counter() - validation_started)
    if not all(bool(item["exact"]) for item in digest_validation.values()):
        raise RuntimeError("Canonical detection candidate digest mismatch.")

    consolidation_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as consolidation_warning_records:
        warnings.simplefilter("always")
        zarr.consolidate_metadata(str(output_path))
    consolidation_seconds = float(time.perf_counter() - consolidation_started)
    consolidation_warnings = [
        str(item.message) for item in consolidation_warning_records
    ]
    direct_started = time.perf_counter()
    zarr.open_group(str(output_path), mode="r", use_consolidated=False)
    direct_open_seconds = float(time.perf_counter() - direct_started)
    consolidated_started = time.perf_counter()
    consolidated = zarr.open_group(
        str(output_path),
        mode="r",
        use_consolidated=True,
    )
    consolidated_open_seconds = float(time.perf_counter() - consolidated_started)

    read_results: list[dict[str, object]] = []
    for entry in plans.entries:
        path = entry.rule.path
        array = consolidated[path]
        rows = int(array.shape[0])
        window_rows = min(1024, rows)
        window_start = max(0, (rows - window_rows) // 2)
        read_started = time.perf_counter()
        window = np.asarray(array[window_start : window_start + window_rows, ...])
        window_seconds = float(time.perf_counter() - read_started)
        read_started = time.perf_counter()
        full = np.asarray(array[:])
        full_seconds = float(time.perf_counter() - read_started)
        read_results.append(
            {
                "path": path,
                "window_rows": window_rows,
                "window_seconds": window_seconds,
                "window_sha256": _sha256_array(window),
                "full_seconds": full_seconds,
                "full_sha256": _sha256_array(full),
            }
        )

    total_seconds = float(time.perf_counter() - started)
    physical = _storage_stats(output_path)
    source_manifest = benchmark_input.as_manifest()
    environment = _environment_manifest()
    common_envelopes: list[dict[str, object]] = []
    write_by_path = {str(item["path"]): item for item in write_results}
    read_by_path = {str(item["path"]): item for item in read_results}
    for entry in plans.entries:
        path = entry.rule.path
        binding = next(
            item for item in CANONICAL_DETECTION_SCHEMA_V1.bindings if item.path == path
        )
        contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        write_case = StorageBenchmarkCase(
            case_id=f"{path}__{plans.profile.profile_id}__write",
            phase=BenchmarkPhase.WRITE,
            array_contract=contract,
            storage_plan=entry.plan,
            workload=WRITE_MATERIALIZATION_V1,
        )
        array_stats = _storage_stats(output_path / path)
        write_trial = {
            **write_by_path[path],
            "physical_bytes": array_stats["apparent_bytes"],
            "payload_object_count": array_stats["payload_file_count"],
        }
        common_envelopes.append(
            benchmark_result_envelope(
                write_case,
                source_identity=source_manifest["source_identity"],
                environment=environment,
                trials=[write_trial],
                summary=write_trial,
                validation=digest_validation[path],
            )
        )
        read_workload = (
            EAGER_FULL_READ_V1
            if entry.plan.access_pattern == "eager"
            else WINDOWED_ROWS_READ_V1
        )
        read_case = StorageBenchmarkCase(
            case_id=f"{path}__{plans.profile.profile_id}__read",
            phase=BenchmarkPhase.READ,
            array_contract=contract,
            storage_plan=entry.plan,
            workload=read_workload,
        )
        read_result = read_by_path[path]
        eager_read = entry.plan.access_pattern == "eager"
        read_trial = {
            **read_result,
            "logical_bytes": int(benchmark_input.arrays[path].nbytes),
            "requested_rows": (
                int(benchmark_input.arrays[path].shape[0])
                if eager_read
                else int(read_result["window_rows"])
            ),
            "decoded_bytes": None,
            "transferred_bytes": None,
            "request_count": None,
            "read_seconds": (
                read_result["full_seconds"]
                if eager_read
                else read_result["window_seconds"]
            ),
        }
        common_envelopes.append(
            benchmark_result_envelope(
                read_case,
                source_identity=source_manifest["source_identity"],
                environment=environment,
                trials=[read_trial],
                summary=read_trial,
                validation=digest_validation[path],
            )
        )

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete_exploratory_smoke",
        "destination": str(output_path),
        "source": source_manifest,
        "storage_plan": plans.as_manifest(),
        "timing": {
            "total_seconds": total_seconds,
            "validation_seconds": validation_seconds,
            "consolidation_seconds": consolidation_seconds,
            "consolidation_warnings": consolidation_warnings,
            "direct_open_seconds": direct_open_seconds,
            "consolidated_open_seconds": consolidated_open_seconds,
        },
        "physical": physical,
        "digest_validation": digest_validation,
        "common_benchmark_envelopes": common_envelopes,
        "environment": environment,
    }


def _candidate_profile(args: argparse.Namespace):
    return make_benchmark_storage_profile(
        target_chunk_bytes=int(args.chunk_kib) * KIB,
        target_shard_bytes=int(args.shard_mib) * MIB,
        shard_immutable=args.layout == "sharded",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_group", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--chunk-kib", type=int, default=1024)
    parser.add_argument("--shard-mib", type=int, default=32)
    parser.add_argument("--layout", choices=("regular", "sharded"), default="sharded")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    benchmark_input = load_detection_benchmark_input(
        args.source_group,
        recording_identity=args.recording_identity,
        frame_limit=args.frame_limit,
    )
    profile = _candidate_profile(args)
    plans = plan_canonical_detection_storage(
        benchmark_input.dimensions,
        profile=profile,
    )
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "source": benchmark_input.as_manifest(),
                    "storage_plan": plans.as_manifest(),
                    "destination": str(args.destination.expanduser().resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = write_detection_benchmark_candidate(
        benchmark_input,
        destination=args.destination,
        plans=plans,
    )
    report_path = args.destination.expanduser().resolve().with_suffix(".benchmark.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "destination": report["destination"],
                "report": str(report_path),
                "profile_id": plans.profile.profile_id,
                "dimensions": benchmark_input.dimensions.as_manifest(),
                "timing": report["timing"],
                "physical": report["physical"],
                "all_digests_exact": all(
                    bool(item["exact"]) for item in report["digest_validation"].values()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
