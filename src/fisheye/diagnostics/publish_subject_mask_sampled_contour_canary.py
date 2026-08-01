"""Publish one full-duration sampled-contour cache canary safely.

The immutable refined dense source is copied from PRFS to node-local scratch,
all computation and Zarr writes occur there, and only a validated
selector-ineligible cache artifact is atomically renamed into the benchmark
namespace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import zarr

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_cache_publication import (
    DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    SUBJECT_MASK_CACHE_FAMILY,
    publish_selector_ineligible_subject_mask_sampled_contours,
    validate_persisted_subject_mask_cache_publication,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    validate_persisted_subject_mask_core_publication,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)

CANARY_SCHEMA_ID = "palette.subject_mask.sampled_contour_full_duration_canary"
CANARY_SCHEMA_VERSION = 1


def _safe_id(value: object, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved or "/" in resolved or resolved in {".", ".."}:
        raise ValueError(f"{name} must be one safe nonempty identifier.")
    return resolved


def _require_benchmark_destination(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if ".palette_benchmarks" not in resolved.parts:
        raise ValueError("Canary destination must be below .palette_benchmarks.")
    if resolved.exists():
        raise FileExistsError(f"Canary destination exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == Path("/") or str(resolved).startswith("/groups/"):
        raise ValueError("Canary scratch must be node-local, never /groups.")
    if resolved.exists():
        raise FileExistsError(f"Canary scratch exists: {resolved}")
    resolved.mkdir(parents=True)
    return resolved


def _tree_inventory(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "physical_bytes": sum(int(item.stat().st_size) for item in files),
    }


def _mac_path(path: Path) -> str | None:
    parts = path.resolve().parts
    prefix = ("/", "groups", "johnson", "johnsonlab")
    if parts[:4] != prefix:
        return None
    return str(Path("/Volumes/johnsonlab", *parts[4:]))


def _stage_refined_source(
    source: Path,
    *,
    refined_run_id: str,
    destination: Path,
) -> Mapping[str, Any]:
    errors = validate_persisted_subject_mask_core_publication(
        source,
        family="refined_subject_masks_runs",
        run_id=refined_run_id,
    )
    if errors:
        raise ValueError("Invalid source refined run: " + "; ".join(errors))
    source_root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    source_run = source_root[f"refined_subject_masks_runs/{refined_run_id}"]
    manifest = source_run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError("Source refined run_manifest is absent.")

    local_root = zarr.open_group(str(destination), mode="w-", zarr_format=3)
    local_root.attrs.update(dict(source_root.attrs))
    local_family = local_root.create_group("refined_subject_masks_runs")
    source_family = source_root["refined_subject_masks_runs"]
    local_family.attrs.update(dict(source_family.attrs))
    shutil.copytree(
        source / "refined_subject_masks_runs" / refined_run_id,
        destination / "refined_subject_masks_runs" / refined_run_id,
    )
    consolidate_metadata_capture_expected_warnings(destination)
    staged_errors = validate_persisted_subject_mask_core_publication(
        destination,
        family="refined_subject_masks_runs",
        run_id=refined_run_id,
    )
    if staged_errors:
        raise RuntimeError(
            "Staged refined source is invalid: " + "; ".join(staged_errors)
        )
    return manifest


def publish_canary(
    *,
    source_analysis_zarr: Path,
    source_refined_run: str,
    destination: Path,
    scratch_root: Path,
    canary_id: str,
    cache_run_id: str,
    source_compute_block_bytes: int = DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    compute_workers: int = 1,
    palette_commit: str,
) -> dict[str, object]:
    source = source_analysis_zarr.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    resolved_destination = _require_benchmark_destination(destination)
    scratch = _require_local_scratch(scratch_root)
    resolved_canary = _safe_id(canary_id, name="canary_id")
    refined_run = _safe_id(source_refined_run, name="source_refined_run")
    cache_run = _safe_id(cache_run_id, name="cache_run_id")
    if int(source_compute_block_bytes) <= 0:
        raise ValueError("source_compute_block_bytes must be positive.")
    if int(compute_workers) <= 0:
        raise ValueError("compute_workers must be positive.")

    started = time.perf_counter()
    phases: dict[str, float] = {}
    staged_source = scratch / "source_refined.zarr"
    local_cache = scratch / "cache.zarr"
    local_artifact = scratch / "artifact"
    publication_temp = resolved_destination.parent / (
        f".{resolved_destination.name}.publish_tmp.{uuid4().hex}"
    )
    try:
        phase = time.perf_counter()
        source_manifest = _stage_refined_source(
            source,
            refined_run_id=refined_run,
            destination=staged_source,
        )
        phases["prfs_to_node_local_source_stage"] = time.perf_counter() - phase

        phase = time.perf_counter()
        cache = publish_selector_ineligible_subject_mask_sampled_contours(
            refined_snapshot_root=staged_source,
            refined_run_id=refined_run,
            destination=local_cache,
            cache_run_id=cache_run,
            source_compute_block_bytes=int(source_compute_block_bytes),
            compute_workers=int(compute_workers),
            created_by="publish_subject_mask_sampled_contour_canary",
        )
        phases["node_local_cache_publication"] = time.perf_counter() - phase

        local_errors = validate_persisted_subject_mask_cache_publication(
            local_cache,
            run_id=cache_run,
            source_manifest=source_manifest,
        )
        if local_errors:
            raise RuntimeError(
                "Local cache validation failed: " + "; ".join(local_errors)
            )

        local_artifact.mkdir()
        shutil.move(str(local_cache), str(local_artifact / "cache.zarr"))
        cache_manifest = cache.manifest
        result: dict[str, object] = {
            "schema_id": CANARY_SCHEMA_ID,
            "schema_version": CANARY_SCHEMA_VERSION,
            "status": "complete",
            "classification": "selector_ineligible_full_duration_cache_canary",
            "canary_id": resolved_canary,
            "palette_commit": str(palette_commit),
            "host": socket.gethostname(),
            "destination": str(resolved_destination),
            "destination_macos": _mac_path(resolved_destination),
            "source": {
                "analysis_zarr": str(source),
                "analysis_zarr_macos": _mac_path(source),
                "refined_run_id": refined_run,
                "refined_run_path": (f"refined_subject_masks_runs/{refined_run}"),
                "manifest_payload_digest": source_manifest["payload_digest"],
                "manifest_document_digest": canonical_json_sha256(source_manifest),
                "physical_inventory": _tree_inventory(
                    source / "refined_subject_masks_runs" / refined_run
                ),
            },
            "cache": {
                "relative_zarr": "cache.zarr",
                "run_id": cache_run,
                "run_path": f"{SUBJECT_MASK_CACHE_FAMILY}/{cache_run}",
                "manifest_payload_digest": cache_manifest["payload_digest"],
                "manifest_document_digest": canonical_json_sha256(cache_manifest),
                "logical_content_digest": cache_manifest["payload"]["logical_content"][
                    "digest"
                ],
                "storage_profile_id": cache.plans.profile.profile_id,
                "storage_plan": cache.plans.as_manifest(),
                "physical_inventory": _tree_inventory(local_artifact / "cache.zarr"),
            },
            "execution": {
                "source_stage": "prfs_to_node_local_complete_refined_run_copy",
                "compute_and_zarr_write": "node_local_only",
                "publication": "validated_tree_copy_then_atomic_sibling_rename",
                "production_selectors_modified": False,
                "production_registries_modified": False,
                "source_archive_modified": False,
                "source_compute_block_bytes": int(source_compute_block_bytes),
                "compute_workers": int(compute_workers),
                "cache_phase_seconds": dict(cache.phase_seconds),
                "phase_seconds": dict(phases),
            },
        }
        result["result_digest"] = canonical_json_sha256(result)
        (local_artifact / "canary_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        phase = time.perf_counter()
        shutil.copytree(local_artifact, publication_temp)
        published_errors = validate_persisted_subject_mask_cache_publication(
            publication_temp / "cache.zarr",
            run_id=cache_run,
            source_manifest=source_manifest,
        )
        if published_errors:
            raise RuntimeError(
                "Published temporary cache validation failed: "
                + "; ".join(published_errors)
            )
        phases["node_local_to_prfs_validated_tree_copy"] = time.perf_counter() - phase
        publication_receipt = {
            "schema_id": (
                "palette.subject_mask.sampled_contour_canary_publication_receipt"
            ),
            "schema_version": 1,
            "status": "complete",
            "canary_id": resolved_canary,
            "destination": str(resolved_destination),
            "canary_manifest_digest": canonical_json_sha256(result),
            "cache_manifest_document_digest": canonical_json_sha256(cache_manifest),
            "phase_seconds": dict(phases),
            "elapsed_seconds_before_atomic_rename": float(
                time.perf_counter() - started
            ),
            "production_selectors_modified": False,
            "production_registries_modified": False,
            "source_archive_modified": False,
        }
        publication_receipt["receipt_digest"] = canonical_json_sha256(
            publication_receipt
        )
        (publication_temp / "publication_receipt.json").write_text(
            json.dumps(
                publication_receipt,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(publication_temp, resolved_destination)
        return {
            "canary_manifest": result,
            "publication_receipt": publication_receipt,
        }
    except BaseException:
        if publication_temp.exists():
            shutil.rmtree(publication_temp)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-analysis-zarr", required=True, type=Path)
    parser.add_argument("--source-refined-run", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--canary-id", required=True)
    parser.add_argument("--cache-run-id", required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument(
        "--source-compute-block-bytes",
        type=int,
        default=DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    )
    parser.add_argument("--compute-workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = publish_canary(
        source_analysis_zarr=args.source_analysis_zarr,
        source_refined_run=args.source_refined_run,
        destination=args.destination,
        scratch_root=args.scratch_root,
        canary_id=args.canary_id,
        cache_run_id=args.cache_run_id,
        source_compute_block_bytes=args.source_compute_block_bytes,
        compute_workers=args.compute_workers,
        palette_commit=args.palette_commit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "publish_canary"]
