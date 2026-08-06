#!/usr/bin/env python3
"""Infer sampled training detections locally and publish strict review seeds.

The detector first writes an unbound, selector-free artifact into a complete
node-local copy of the training Zarr.  Palette then atomically imports only
that artifact and materializes a separate acquisition-bound ``detect_runs``
snapshot through the sampled-training detection contract.  Neither step
changes a selector, consolidates the still-mutable archive, or registers it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Optional, Sequence

import zarr

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    tree_inventory,
)
from fisheye.registry.db import RegistryPaths
from fisheye.shared.zarr.sampled_training_detection_publication import (
    build_and_publish_sampled_training_detection,
    publish_detection_artifact_run,
)
from fisheye.shared.zarr_helpers import archive_metadata_publication_lock
from fisheye.utils.predict_training_detections import (
    resolve_model_input_spec,
    run_training_zarr_prediction,
    select_frame_source,
)

CANARY_SCHEMA_ID = "palette.sampled_training_detection_canary"
CANARY_SCHEMA_VERSION = 1


def _safe_run_id(value: str, *, option: str) -> str:
    normalized = str(value).strip()
    if not normalized or "/" in normalized or normalized in {".", ".."}:
        raise ValueError(f"{option} must be one path-safe name.")
    return normalized


def _node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Node-local scratch root does not exist: {resolved}")
    if resolved in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    } or str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError(
            "Scratch must be a bounded node-local directory, not shared storage."
        )
    return resolved


def _copy_training_archive_to_scratch(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    """Copy and content-authenticate the complete mutable training base."""

    started = time.perf_counter()
    with archive_metadata_publication_lock(source):
        source_inventory = tree_inventory(source, hash_content=True)
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        local_inventory = tree_inventory(destination, hash_content=True)
    if local_inventory != source_inventory:
        raise RuntimeError(
            "Node-local training copy differs from the source physical inventory."
        )
    return {
        "seconds": time.perf_counter() - started,
        "source": str(source),
        "destination": str(destination),
        "physical_inventory": source_inventory.to_json(),
        "content_authenticated": True,
    }


def run_sampled_training_detection_canary(
    *,
    archive: str | Path,
    scratch_root: str | Path,
    registry_path: str | Path,
    model_run_id: Optional[str],
    model_path: Optional[str | Path],
    model_set_id: Optional[str],
    artifact_kind: str,
    artifact_run_id: str,
    detect_run_id: str,
    batch_size: int,
    conf: float,
    iou: float,
    max_det: int,
    cpu: bool,
    copy_backend: str,
    argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Execute the complete node-local inference and strict publication chain."""

    target = Path(archive).expanduser().resolve()
    scratch = _node_local_scratch(Path(scratch_root))
    registry = Path(registry_path).expanduser().resolve()
    artifact_name = _safe_run_id(artifact_run_id, option="artifact_run_id")
    detect_name = _safe_run_id(detect_run_id, option="detect_run_id")
    if not target.is_dir() or target.suffix != ".zarr":
        raise FileNotFoundError(f"Training Zarr does not exist: {target}")
    if not registry.is_file():
        raise FileNotFoundError(f"Registry does not exist: {registry}")
    if batch_size <= 0 or max_det <= 0:
        raise ValueError("batch_size and max_det must be positive.")
    if (target / "detection_artifact_runs" / artifact_name).exists():
        raise FileExistsError(
            f"Target artifact run already exists: detection_artifact_runs/{artifact_name}"
        )
    if (target / "detect_runs" / detect_name).exists():
        raise FileExistsError(
            f"Target detect run already exists: detect_runs/{detect_name}"
        )

    spec = resolve_model_input_spec(
        registry,
        model_run_id=model_run_id,
        model_path=(Path(model_path) if model_path is not None else None),
        set_id=model_set_id,
        artifact_kind=str(artifact_kind),
    )
    phases: dict[str, float] = {}
    with tempfile.TemporaryDirectory(
        prefix="palette-sampled-training-detect-",
        dir=str(scratch),
    ) as temporary:
        local_archive = Path(temporary) / target.name
        copy_receipt = _copy_training_archive_to_scratch(target, local_archive)
        phases["source_to_node_local_copy_and_authentication"] = float(
            copy_receipt["seconds"]
        )

        inference_started = time.perf_counter()
        inference = run_training_zarr_prediction(
            zarr_path=local_archive,
            spec=spec,
            run_name=artifact_name,
            batch_size=int(batch_size),
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_det),
            cpu=bool(cpu),
            overwrite=False,
            argv=list(argv) if argv is not None else None,
        )
        phases["node_local_inference"] = time.perf_counter() - inference_started

        artifact_started = time.perf_counter()
        artifact_publication = publish_detection_artifact_run(
            local_archive=local_archive,
            target_archive=target,
            artifact_run_id=artifact_name,
            copy_backend=copy_backend,
        )
        phases["artifact_atomic_publication"] = time.perf_counter() - artifact_started

    binding_started = time.perf_counter()
    detection_publication = build_and_publish_sampled_training_detection(
        archive=target,
        artifact_run_id=artifact_name,
        scratch_root=scratch,
        run_id=detect_name,
        copy_backend=copy_backend,
    )
    phases["bound_detection_materialization_and_publication"] = (
        time.perf_counter() - binding_started
    )
    return {
        "schema_id": CANARY_SCHEMA_ID,
        "schema_version": CANARY_SCHEMA_VERSION,
        "status": "complete",
        "archive": str(target),
        "registry": str(registry),
        "model": asdict(spec),
        "artifact_run_id": artifact_name,
        "detect_run_id": detect_name,
        "parameters": {
            "batch_size": int(batch_size),
            "conf": float(conf),
            "iou": float(iou),
            "max_det": int(max_det),
            "cpu": bool(cpu),
            "copy_backend": str(copy_backend),
        },
        "source_copy": copy_receipt,
        "inference": inference,
        "artifact_publication": artifact_publication,
        "detection_publication": detection_publication,
        "phase_seconds": phases,
        "stage_selector_eligible": False,
        "registry_activation": "deferred",
        "root_consolidation": "deferred_while_training_archive_is_mutable",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--model-run-id")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-set-id")
    parser.add_argument(
        "--artifact-kind",
        choices=("training", "onnx", "tensorrt"),
        default="training",
    )
    parser.add_argument("--artifact-run-id", required=True)
    parser.add_argument("--detect-run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--copy-backend", default="python")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    archive = args.archive.expanduser().resolve()
    scratch = _node_local_scratch(args.scratch_root)
    registry = (
        (args.registry or RegistryPaths.from_env(Path.cwd()).path)
        .expanduser()
        .resolve()
    )
    spec = resolve_model_input_spec(
        registry,
        model_run_id=args.model_run_id,
        model_path=args.model_path,
        set_id=args.model_set_id,
        artifact_kind=args.artifact_kind,
    )
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    selection = select_frame_source(root, spec)
    plan = {
        "schema_id": CANARY_SCHEMA_ID,
        "schema_version": CANARY_SCHEMA_VERSION,
        "mode": "apply" if args.apply else "dry_run",
        "archive": str(archive),
        "scratch_root": str(scratch),
        "registry": str(registry),
        "model": asdict(spec),
        "frame_source": asdict(selection),
        "artifact_run_id": _safe_run_id(
            args.artifact_run_id, option="--artifact-run-id"
        ),
        "detect_run_id": _safe_run_id(args.detect_run_id, option="--detect-run-id"),
        "parameters": {
            "batch_size": int(args.batch_size),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "cpu": bool(args.cpu),
            "copy_backend": str(args.copy_backend),
        },
        "invariants": {
            "node_local_inference": True,
            "atomic_run_publication": True,
            "stage_selector_eligible": False,
            "registry_activation": "deferred",
            "root_consolidation": "deferred_while_mutable",
            "instances_per_frame": "zero_one_or_many_reported_not_forced",
        },
    }
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    result = run_sampled_training_detection_canary(
        archive=archive,
        scratch_root=args.scratch_root,
        registry_path=registry,
        model_run_id=args.model_run_id,
        model_path=args.model_path,
        model_set_id=args.model_set_id,
        artifact_kind=args.artifact_kind,
        artifact_run_id=args.artifact_run_id,
        detect_run_id=args.detect_run_id,
        batch_size=int(args.batch_size),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        cpu=bool(args.cpu),
        copy_backend=str(args.copy_backend),
        argv=list(argv) if argv is not None else sys.argv,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        stats = result["detection_publication"]["cardinality_statistics"]
        print(
            f"Published selector-ineligible detect_runs/{args.detect_run_id}: "
            f"{stats['detection_rows']} rows across {stats['sampled_frames']} frames; "
            f"zero={stats['frames_with_zero_detections']}, "
            f"one={stats['frames_with_one_detection']}, "
            f"multiple={stats['frames_with_multiple_detections']}."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
