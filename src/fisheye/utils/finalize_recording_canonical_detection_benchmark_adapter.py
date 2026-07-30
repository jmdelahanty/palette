#!/usr/bin/env python3
"""Normalize one legacy recording detection run into a canonical candidate.

This adapter is benchmark-only.  It reads one explicitly pinned recording-level
legacy detection run and an earlier logically equivalent recording fixture,
converts the complete table to the current coordinate-catalog contract, writes
the approved access-aware profile on node-local scratch, and atomically places
the validated standalone store below ``.palette_benchmarks``.  It never mutates
an analysis archive, selector, or registry.
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

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    sha256_array,
    sha256_file,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    CanonicalDetectionBenchmarkInput,
    load_detection_benchmark_input,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    publish_legacy_canonical_detection_shadow,
)
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


ADAPTER_SCHEMA_ID = "palette.canonical_detection.recording_benchmark_adapter"
ADAPTER_SCHEMA_VERSION = 2
ADAPTER_RECEIPT_NAME = "recording_canonical_adapter_receipt.json"
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


def _require_recorded_model(
    source: Any,
    *,
    model_artifact: Path,
    expected_sha256: str,
) -> dict[str, str]:
    artifact = model_artifact.expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Canonical source model artifact is missing: {artifact}")
    observed_sha = sha256_file(artifact)
    if observed_sha != expected_sha256:
        raise ValueError("Canonical source model artifact differs from its pin.")
    recorded = Path(str(source.attrs.get("model_path") or ""))
    if len(recorded.parts) < 3 or tuple(recorded.parts[-3:]) != tuple(
        artifact.parts[-3:]
    ):
        raise ValueError(
            "Canonical source recorded model identity differs from the pinned artifact."
        )
    selected_run = str(
        source.attrs.get("model_resolution_selected_run_id") or ""
    ).strip()
    if selected_run and selected_run != artifact.parts[-3]:
        raise ValueError("Canonical source model-resolution run differs from the pin.")
    return {
        "recorded_path": str(recorded),
        "artifact_path": str(artifact),
        "sha256": observed_sha,
    }


def _validate_legacy_counts(
    source: Any,
    *,
    benchmark_input: CanonicalDetectionBenchmarkInput,
) -> dict[str, object]:
    frame_counts = np.asarray(source["frame_counts"][...], dtype=np.int64)
    n_detections = np.asarray(source["n_detections"][...], dtype=np.int64)
    if not np.array_equal(frame_counts, n_detections):
        raise ValueError("Canonical source count aliases differ.")
    if frame_counts.shape != (benchmark_input.dimensions.n_frames,):
        raise ValueError("Canonical source frame-count length is inconsistent.")
    frame_indices = np.asarray(
        benchmark_input.arrays["instances/frame_indices"], dtype=np.int64
    )
    observed = np.bincount(
        frame_indices,
        minlength=benchmark_input.dimensions.n_frames,
    )
    if not np.array_equal(frame_counts, observed):
        raise ValueError("Canonical source counts disagree with frame_indices.")
    summary = source.attrs.get("summary_statistics")
    if isinstance(summary, Mapping) and summary.get("total_detections") != (
        benchmark_input.dimensions.n_instances
    ):
        raise ValueError("Canonical source summary row count is inconsistent.")
    return {
        "n_frames": benchmark_input.dimensions.n_frames,
        "n_instances": benchmark_input.dimensions.n_instances,
        "empty_frames": int(np.count_nonzero(frame_counts == 0)),
        "multiple_detection_frames": int(np.count_nonzero(frame_counts > 1)),
        "frame_counts_sha256": sha256_array(frame_counts),
    }


def _validate_anchor(
    *,
    anchor_group_path: Path,
    canonical_input: CanonicalDetectionBenchmarkInput,
) -> dict[str, object]:
    anchor = zarr.open_group(
        str(anchor_group_path),
        mode="r",
        use_consolidated=False,
    )
    logical = anchor.attrs.get("logical_schema")
    raw_dimensions = logical.get("dimensions") if isinstance(logical, Mapping) else None
    if not isinstance(raw_dimensions, Mapping):
        raise ValueError("Canonical anchor lacks its frozen logical dimensions.")
    dimensions = CanonicalDetectionDimensions(
        n_frames=raw_dimensions.get("n_frames"),
        n_instances=raw_dimensions.get("n_instances"),
        source_width=raw_dimensions.get("source_width"),
        source_height=raw_dimensions.get("source_height"),
    )
    if dimensions != canonical_input.dimensions:
        raise ValueError("Canonical anchor dimensions differ from rebuilt detections.")
    arrays = {
        path: anchor[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    }
    CANONICAL_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)
    if anchor.attrs.get("selector_eligible") is not False:
        raise ValueError("Canonical anchor must remain selector-ineligible.")
    hashes: dict[str, str] = {}
    for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        expected = np.asarray(canonical_input.arrays[path])
        observed = np.asarray(arrays[path][...])
        if not np.array_equal(observed, expected):
            raise ValueError(f"Canonical anchor differs at {path!r}.")
        hashes[path] = sha256_array(expected)
    return {
        "source_group_path": str(anchor_group_path),
        "source_group_metadata_sha256": sha256_file(anchor_group_path / "zarr.json"),
        "logical_array_sha256": hashes,
        "decoded_values_equal": True,
    }


def finalize_recording_canonical_detection_benchmark_adapter(
    *,
    source_detection_group_path: Path,
    recording_identity: str,
    source_model_artifact: Path,
    canonical_anchor_archive: Path,
    canonical_anchor_run_id: str,
    expected_model_sha256: str,
    expected_n_frames: int,
    destination: Path,
    benchmark_root: Path,
    run_id: str,
    scratch_parent: Path,
) -> dict[str, object]:
    """Rebuild and publish one standalone current canonical detection store."""

    started = time.perf_counter()
    phases: dict[str, float] = {}
    source_path = source_detection_group_path.expanduser().resolve()
    anchor_archive = canonical_anchor_archive.expanduser().resolve()
    anchor_group = anchor_archive / "detect_runs" / canonical_anchor_run_id
    output = _require_benchmark_destination(
        destination,
        benchmark_root=benchmark_root,
    )
    model_sha = _require_sha256(expected_model_sha256, name="detection model")
    if type(expected_n_frames) is not int or expected_n_frames <= 0:
        raise ValueError("expected_n_frames must be a positive exact integer.")
    if not source_path.is_dir() or not anchor_group.is_dir():
        raise FileNotFoundError("Canonical source run or canonical anchor is missing.")

    load_started = time.perf_counter()
    source_metadata_path = source_path / "zarr.json"
    source_metadata_sha_before = sha256_file(source_metadata_path)
    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    canonical_input = load_detection_benchmark_input(
        source_path,
        recording_identity=recording_identity,
        frame_limit=None,
    )
    if canonical_input.dimensions.n_frames != expected_n_frames:
        raise ValueError("Canonical source frame count differs from the request.")
    model_evidence = _require_recorded_model(
        source,
        model_artifact=source_model_artifact,
        expected_sha256=model_sha,
    )
    count_evidence = _validate_legacy_counts(
        source,
        benchmark_input=canonical_input,
    )
    phases["load_and_validate_recording_source"] = (
        time.perf_counter() - load_started
    )

    anchor_started = time.perf_counter()
    anchor_evidence = _validate_anchor(
        anchor_group_path=anchor_group,
        canonical_input=canonical_input,
    )
    phases["validate_canonical_anchor"] = time.perf_counter() - anchor_started

    scratch = _require_node_local_scratch(scratch_parent)
    with tempfile.TemporaryDirectory(
        prefix="palette_crimson_canonical_", dir=scratch
    ) as temporary_directory:
        local_root = Path(temporary_directory) / ".palette_benchmarks"
        local_root.mkdir()
        local = local_root / "canonical.zarr"
        write_started = time.perf_counter()
        candidate = publish_legacy_canonical_detection_shadow(
            source_group_path=source_path,
            destination=local,
            run_id=run_id,
            recording_identity=recording_identity,
            source_run_id=source_path.name,
            shadow_root=local_root,
            coordinate_catalog=True,
        )
        phases["node_local_publication"] = time.perf_counter() - write_started
        shadow_receipt_path = local / "shadow_publication_receipt.json"
        shadow_receipt = _read_json(shadow_receipt_path)
        shadow_receipt["output_path"] = str(output)
        write_json_atomic(shadow_receipt_path, shadow_receipt)
        adapter_payload: dict[str, object] = {
            "status": "complete",
            "recording_identity": recording_identity,
            "output_archive": str(output),
            "output_run_id": run_id,
            "n_frames": canonical_input.dimensions.n_frames,
            "n_instances": canonical_input.dimensions.n_instances,
            "source_width": canonical_input.dimensions.source_width,
            "source_height": canonical_input.dimensions.source_height,
            "run_manifest_digest": candidate.manifest["payload_digest"],
            "storage_profile_id": candidate.plans.profile.profile_id,
            "source_detection_group_path": str(source_path),
            "source_detection_metadata_sha256": source_metadata_sha_before,
            "source_model_artifact": model_evidence,
            "source_count_validation": count_evidence,
            "detection_model_sha256": model_sha,
            "canonical_anchor": anchor_evidence,
            "canonical_manifest_schema_version": candidate.manifest["schema_version"],
            "coordinate_catalog": True,
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

    reopened = zarr.open_group(
        str(output / "detect_runs" / run_id),
        mode="r",
        use_consolidated=False,
    )
    reopened_manifest = reopened.attrs.get("run_manifest")
    if not isinstance(reopened_manifest, Mapping) or reopened_manifest.get(
        "payload_digest"
    ) != candidate.manifest["payload_digest"]:
        raise RuntimeError("Published canonical manifest differs from node-local output.")
    if sha256_file(source_metadata_path) != source_metadata_sha_before:
        raise RuntimeError("Canonical source metadata changed during publication.")
    return {
        "schema_id": ADAPTER_SCHEMA_ID,
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "output_archive": str(output),
        "output_run_id": run_id,
        "run_manifest_digest": reopened_manifest["payload_digest"],
        "adapter_receipt_path": str(output / ADAPTER_RECEIPT_NAME),
        "adapter_receipt_digest": adapter_receipt["payload_digest"],
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "inference_reexecuted": False,
        "node_local_materialization": True,
        "canonical_anchor_equal": True,
        "canonical_manifest_schema_version": candidate.manifest["schema_version"],
        "coordinate_catalog": True,
        "local_store_stats": local_stats,
        "timing_seconds": phases,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes": peak_rss_bytes(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-detection-group", type=Path, required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--source-model-artifact", type=Path, required=True)
    parser.add_argument("--canonical-anchor-archive", type=Path, required=True)
    parser.add_argument("--canonical-anchor-run", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-n-frames", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
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
        result = finalize_recording_canonical_detection_benchmark_adapter(
            source_detection_group_path=args.source_detection_group,
            recording_identity=args.recording_identity,
            source_model_artifact=args.source_model_artifact,
            canonical_anchor_archive=args.canonical_anchor_archive,
            canonical_anchor_run_id=args.canonical_anchor_run,
            expected_model_sha256=args.expected_model_sha256,
            expected_n_frames=args.expected_n_frames,
            destination=args.destination,
            benchmark_root=args.benchmark_root,
            run_id=args.run_id,
            scratch_parent=scratch_parent,
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
