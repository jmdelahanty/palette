"""Publish a receipt-composed subject-mask bundle from immutable clip packages.

The wrapper assembles only metadata and worker run views on node-local scratch.
Raw worker runs remain read-only in the analysis archive; refined workers and
their sealed publication evidence are extracted from immutable handoff packages.
The underlying bundle publisher remains the sole owner of destination writes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

import zarr

from fisheye.cluster.subject_masks.publish_recording_bundle import (
    publish_recording_subject_mask_bundle,
)
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.runtime_telemetry import PhaseTelemetry
from fisheye.shared.zarr.subject_mask_core_publication import (
    SubjectMaskCoreValidationMode,
)
from fisheye.shared.refined_subject_mask_encoded_chunks import (
    ENCODED_PACKAGE_SCHEMA_ID,
)
from fisheye.utils.finalize_subject_mask_clip_package import PACKAGE_SCHEMA_ID


DEFAULT_PACKAGE_EXTRACT_WORKERS = 4


@dataclass(frozen=True)
class _ExtractedClipPackage:
    index: int
    package: Path
    extracted: Path
    run_path: PurePosixPath
    publication_evidence: Mapping[str, Any]
    required_evidence: Mapping[str, Path]
    compressed_bytes: int


@dataclass(frozen=True)
class _ClipPackageAssembly:
    refined_runs: tuple[str, ...]
    raw_units: tuple[Path, ...]
    refined_units: tuple[Path, ...]
    contours: tuple[Path, ...]
    quality: tuple[Path, ...]
    work_units: tuple[dict[str, Any], ...]
    package_count: int
    compressed_bytes: int
    extract_workers_requested: int
    extract_workers_effective: int


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    members = tar.getmembers()
    for member in members:
        resolved = (root / member.name).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Package member escapes extraction root: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Package links are forbidden: {member.name}")
        if not member.isfile() and not member.isdir():
            raise ValueError(f"Package special members are forbidden: {member.name}")
    tar.extractall(destination, members=members)


def _copy_root_attributes(source: Path, destination: Path) -> None:
    source_root = zarr.open_group(
        str(source), mode="r", zarr_format=3, use_consolidated=False
    )
    destination_root = zarr.open_group(str(destination), mode="w", zarr_format=3)
    destination_root.attrs.update(dict(source_root.attrs))


def _extract_one_clip_package(
    index: int,
    raw_path: Path,
    *,
    extract_root: Path,
    refined_parent_name: str,
) -> _ExtractedClipPackage:
    """Extract and validate one immutable package without shared-tree mutation."""

    package = raw_path.expanduser().resolve()
    if not package.is_file():
        raise FileNotFoundError(package)
    extracted = extract_root / f"package_{int(index):06d}"
    extracted.mkdir()
    with tarfile.open(package, "r:gz") as handle:
        _safe_extract(handle, extracted)
    manifest = json.loads((extracted / "package.json").read_text(encoding="utf-8"))
    publication_evidence = manifest.get("publication_evidence")
    if (
        manifest.get("schema_id") not in {PACKAGE_SCHEMA_ID, ENCODED_PACKAGE_SCHEMA_ID}
        or manifest.get("package_completion_status") != "complete"
        or not isinstance(publication_evidence, Mapping)
    ):
        raise ValueError(f"Clip package lacks current publication evidence: {package}")
    run_path = PurePosixPath(str(manifest.get("run_group_path") or ""))
    if len(run_path.parts) != 2 or run_path.parts[0] != refined_parent_name:
        raise ValueError(f"Clip package run path is invalid: {package}")
    source_run = extracted.joinpath(*run_path.parts)
    if not source_run.is_dir():
        raise ValueError(f"Clip package refined run is absent: {package}")
    evidence = extracted / "publication_evidence"
    required = {
        "raw": evidence / "raw_final_layout_unit",
        "refined": evidence / "refined_final_layout_unit",
        "contour": evidence / "sampled_contour_receipt.json",
        "quality": evidence / "quality_partition",
    }
    missing = [name for name, value in required.items() if not value.exists()]
    if missing:
        raise ValueError(
            f"Clip package publication evidence is incomplete ({package}): {missing}"
        )
    frame_interval = publication_evidence.get("global_frame_interval")
    row_interval = publication_evidence.get("global_row_interval")
    if not isinstance(frame_interval, Mapping) or not isinstance(row_interval, Mapping):
        raise ValueError(f"Clip package work-unit intervals are absent: {package}")
    return _ExtractedClipPackage(
        index=int(index),
        package=package,
        extracted=extracted,
        run_path=run_path,
        publication_evidence=dict(publication_evidence),
        required_evidence=required,
        compressed_bytes=int(package.stat().st_size),
    )


def _extract_clip_packages(
    package_paths: Sequence[Path],
    *,
    assembly: Path,
    extract_workers: int,
) -> _ClipPackageAssembly:
    if type(extract_workers) is not int or extract_workers <= 0:
        raise ValueError("extract_workers must be a positive integer.")
    if not package_paths:
        raise ValueError("At least one refined clip package is required.")
    refined_parent = assembly / "refined_subject_masks_runs"
    zarr.open_group(str(refined_parent), mode="w", zarr_format=3)
    extract_root = assembly.parent / "package_extracts"
    extract_root.mkdir()
    effective_workers = min(int(extract_workers), len(package_paths))
    if effective_workers == 1:
        extracted_packages = [
            _extract_one_clip_package(
                index,
                raw_path,
                extract_root=extract_root,
                refined_parent_name=refined_parent.name,
            )
            for index, raw_path in enumerate(package_paths)
        ]
    else:
        futures: list[Future[_ExtractedClipPackage]] = []
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="subject-mask-package-extract",
        ) as executor:
            for index, raw_path in enumerate(package_paths):
                futures.append(
                    executor.submit(
                        _extract_one_clip_package,
                        index,
                        raw_path,
                        extract_root=extract_root,
                        refined_parent_name=refined_parent.name,
                    )
                )
            try:
                # Input-order collection keeps error reporting and assembly
                # deterministic while decompression runs concurrently.
                extracted_packages = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    refined_runs: list[str] = []
    raw_units: list[Path] = []
    refined_units: list[Path] = []
    contours: list[Path] = []
    quality: list[Path] = []
    work_units: list[dict[str, Any]] = []
    for item in extracted_packages:
        run_name = item.run_path.parts[1]
        if run_name in refined_runs:
            raise ValueError(f"Duplicate refined worker run: {run_name}")
        source_run = item.extracted.joinpath(*item.run_path.parts)
        os.replace(source_run, refined_parent / run_name)
        refined_runs.append(run_name)
        raw_units.append(item.required_evidence["raw"])
        refined_units.append(item.required_evidence["refined"])
        contours.append(item.required_evidence["contour"])
        quality.append(item.required_evidence["quality"])
        publication_evidence = item.publication_evidence
        frame_interval = publication_evidence["global_frame_interval"]
        row_interval = publication_evidence["global_row_interval"]
        work_units.append(
            {
                "work_unit_id": publication_evidence.get("work_unit_id"),
                "work_unit_index": publication_evidence.get("work_unit_index"),
                "source_clip_id": publication_evidence.get("source_clip_id"),
                "source_clip_index": publication_evidence.get("source_clip_index"),
                "frame_start": frame_interval.get("start_frame"),
                "frame_stop": frame_interval.get("stop_frame"),
                "row_start": row_interval.get("start_row"),
                "row_stop": row_interval.get("stop_row"),
            }
        )
    ordered_units = tuple(
        sorted(work_units, key=lambda item: int(item["work_unit_index"]))
    )
    return _ClipPackageAssembly(
        refined_runs=tuple(refined_runs),
        raw_units=tuple(raw_units),
        refined_units=tuple(refined_units),
        contours=tuple(contours),
        quality=tuple(quality),
        work_units=ordered_units,
        package_count=len(extracted_packages),
        compressed_bytes=sum(item.compressed_bytes for item in extracted_packages),
        extract_workers_requested=int(extract_workers),
        extract_workers_effective=effective_workers,
    )


def publish_receipt_composed_bundle(
    *,
    analysis_zarr: Path,
    crop_run: str,
    raw_draft_runs: Sequence[str],
    refined_package_paths: Sequence[Path],
    raw_run: str,
    refined_run: str,
    quality_run: str,
    cache_run: str,
    bundle_id: str,
    producer_commit: str,
    local_output_root: Path,
    quality_scratch_root: Path,
    core_physical_unit_workers: int = 4,
    package_extract_workers: int = DEFAULT_PACKAGE_EXTRACT_WORKERS,
    copy_backend: str = "python",
    allow_signed_hybrid_crop_rebase: bool = False,
) -> dict[str, object]:
    """Stage exact worker views and invoke the fail-closed composable publisher."""

    if not raw_draft_runs or len(raw_draft_runs) != len(refined_package_paths):
        raise ValueError("Raw runs and refined clip packages must have 1:1 coverage.")
    if type(package_extract_workers) is not int or package_extract_workers <= 0:
        raise ValueError("package_extract_workers must be a positive integer.")
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    telemetry = PhaseTelemetry(
        materializer="publish_receipt_composed_subject_mask_bundle",
        context={
            "package_count": len(refined_package_paths),
            "package_extract_workers_requested": int(package_extract_workers),
            "core_physical_unit_workers_requested": int(core_physical_unit_workers),
        },
    )
    output = local_output_root.expanduser().resolve()
    with telemetry.phase("output_and_staging_setup"):
        output.mkdir(parents=True, exist_ok=False)
    temporary = output / f".receipt_composed_inputs.{uuid4().hex}"
    assembly = temporary / "assembly.zarr"
    with telemetry.phase("assembly_root_setup"):
        temporary.mkdir()
    result: dict[str, object]
    extracted: _ClipPackageAssembly
    try:
        source = analysis_zarr.expanduser().resolve()
        with telemetry.phase("assembly_metadata_and_links"):
            _copy_root_attributes(source, assembly)
            for parent in ("crop_runs", "subject_mask_shard_runs"):
                source_parent = source / parent
                if not source_parent.is_dir():
                    raise FileNotFoundError(source_parent)
                os.symlink(source_parent, assembly / parent, target_is_directory=True)
        with telemetry.phase("package_extraction_and_validation"):
            extracted = _extract_clip_packages(
                refined_package_paths,
                assembly=assembly,
                extract_workers=int(package_extract_workers),
            )
        with telemetry.phase("recording_bundle_publication"):
            result = publish_recording_subject_mask_bundle(
                analysis_zarr=source,
                draft_zarr=assembly,
                crop_run=crop_run,
                raw_draft_parent="subject_mask_shard_runs",
                raw_draft_run=str(raw_draft_runs[0]),
                raw_draft_runs=tuple(raw_draft_runs),
                refined_draft_run=extracted.refined_runs[0],
                refined_draft_runs=extracted.refined_runs,
                raw_run=raw_run,
                refined_run=refined_run,
                quality_run=quality_run,
                cache_run=cache_run,
                bundle_id=bundle_id,
                local_output_root=output,
                quality_scratch_root=quality_scratch_root,
                quality_partition_roots=extracted.quality,
                require_worker_quality=True,
                quality_partition_producer_commit=producer_commit,
                core_physical_unit_workers=int(core_physical_unit_workers),
                core_validation_mode=(
                    SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
                ),
                raw_final_layout_unit_packages=extracted.raw_units,
                refined_final_layout_unit_packages=extracted.refined_units,
                require_complete_final_layout_units=True,
                sampled_contour_worker_receipts=extracted.contours,
                require_worker_sampled_contours=True,
                sampled_contour_producer_commit=producer_commit,
                allow_signed_hybrid_crop_rebase=allow_signed_hybrid_crop_rebase,
                expected_work_units=list(extracted.work_units),
                activate=False,
                copy_backend=copy_backend,
            )
    finally:
        with telemetry.phase("staging_cleanup"):
            shutil.rmtree(temporary, ignore_errors=True)
    return {
        **result,
        "publication_profile": "receipt_composed_clip_workers_v1",
        "worker_count": len(extracted.refined_runs),
        "producer_commit": producer_commit,
        "package_extraction": {
            "transport": "gzip_tar_independent_packages_v1",
            "package_count": extracted.package_count,
            "compressed_bytes": extracted.compressed_bytes,
            "workers_requested": extracted.extract_workers_requested,
            "workers_effective": extracted.extract_workers_effective,
            "ordering": "input_index_results_with_work_unit_index_plan_sort_v1",
            "shared_tree_mutation": "serialized_after_parallel_validation_v1",
        },
        "receipt_composed_runtime_telemetry": telemetry.to_json(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--raw-draft-run", action="append", required=True)
    parser.add_argument("--refined-package", action="append", required=True, type=Path)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--cache-run", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--local-output-root", required=True, type=Path)
    parser.add_argument("--quality-scratch-root", required=True, type=Path)
    parser.add_argument("--core-physical-unit-workers", type=int, default=4)
    parser.add_argument(
        "--package-extract-workers",
        type=int,
        default=DEFAULT_PACKAGE_EXTRACT_WORKERS,
        help=(
            "Bounded workers for independent gzip package extraction. Shared "
            "assembly mutation remains serialized."
        ),
    )
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
        help="Atomic bundle-member transfer backend (default: python).",
    )
    parser.add_argument(
        "--allow-signed-hybrid-crop-rebase",
        action="store_true",
        help=(
            "Recovery-only: bind sealed signed-hybrid workers to the exact crop-v2 "
            "candidate published from that provider."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = publish_receipt_composed_bundle(
        analysis_zarr=args.analysis_zarr,
        crop_run=args.crop_run,
        raw_draft_runs=args.raw_draft_run,
        refined_package_paths=args.refined_package,
        raw_run=args.raw_run,
        refined_run=args.refined_run,
        quality_run=args.quality_run,
        cache_run=args.cache_run,
        bundle_id=args.bundle_id,
        producer_commit=args.producer_commit,
        local_output_root=args.local_output_root,
        quality_scratch_root=args.quality_scratch_root,
        core_physical_unit_workers=int(args.core_physical_unit_workers),
        package_extract_workers=int(args.package_extract_workers),
        copy_backend=args.copy_backend,
        allow_signed_hybrid_crop_rebase=bool(args.allow_signed_hybrid_crop_rebase),
    )
    print(
        json.dumps(json_ready(result), indent=None if args.json else 2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PACKAGE_EXTRACT_WORKERS",
    "main",
    "publish_receipt_composed_bundle",
]
