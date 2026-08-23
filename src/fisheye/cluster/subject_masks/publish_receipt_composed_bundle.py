"""Publish a receipt-composed subject-mask bundle from immutable clip packages.

The wrapper assembles only metadata and worker run views on node-local scratch.
Raw worker runs remain read-only in the analysis archive; refined workers and
their sealed publication evidence are extracted from immutable handoff packages.
The underlying bundle publisher remains the sole owner of destination writes.
"""

from __future__ import annotations

import argparse
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
from fisheye.shared.zarr.subject_mask_core_publication import (
    SubjectMaskCoreValidationMode,
)
from fisheye.shared.refined_subject_mask_encoded_chunks import (
    ENCODED_PACKAGE_SCHEMA_ID,
)
from fisheye.utils.finalize_subject_mask_clip_package import PACKAGE_SCHEMA_ID


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


def _extract_clip_packages(
    package_paths: Sequence[Path],
    *,
    assembly: Path,
) -> tuple[
    list[str],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[dict[str, Any]],
]:
    refined_parent = assembly / "refined_subject_masks_runs"
    zarr.open_group(str(refined_parent), mode="w", zarr_format=3)
    refined_runs: list[str] = []
    raw_units: list[Path] = []
    refined_units: list[Path] = []
    contours: list[Path] = []
    quality: list[Path] = []
    work_units: list[dict[str, Any]] = []
    extract_root = assembly.parent / "package_extracts"
    extract_root.mkdir()
    for index, raw_path in enumerate(package_paths):
        package = raw_path.expanduser().resolve()
        if not package.is_file():
            raise FileNotFoundError(package)
        extracted = extract_root / f"package_{index:06d}"
        extracted.mkdir()
        with tarfile.open(package, "r:gz") as handle:
            _safe_extract(handle, extracted)
        manifest = json.loads((extracted / "package.json").read_text(encoding="utf-8"))
        publication_evidence = manifest.get("publication_evidence")
        if (
            manifest.get("schema_id")
            not in {PACKAGE_SCHEMA_ID, ENCODED_PACKAGE_SCHEMA_ID}
            or manifest.get("package_completion_status") != "complete"
            or not isinstance(publication_evidence, Mapping)
        ):
            raise ValueError(
                f"Clip package lacks current publication evidence: {package}"
            )
        run_path = PurePosixPath(str(manifest.get("run_group_path") or ""))
        if len(run_path.parts) != 2 or run_path.parts[0] != refined_parent.name:
            raise ValueError(f"Clip package run path is invalid: {package}")
        run_name = run_path.parts[1]
        if run_name in refined_runs:
            raise ValueError(f"Duplicate refined worker run: {run_name}")
        source_run = extracted.joinpath(*run_path.parts)
        target_run = refined_parent / run_name
        if not source_run.is_dir():
            raise ValueError(f"Clip package refined run is absent: {package}")
        os.replace(source_run, target_run)
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
        refined_runs.append(run_name)
        raw_units.append(required["raw"])
        refined_units.append(required["refined"])
        contours.append(required["contour"])
        quality.append(required["quality"])
        frame_interval = publication_evidence.get("global_frame_interval")
        row_interval = publication_evidence.get("global_row_interval")
        if not isinstance(frame_interval, Mapping) or not isinstance(
            row_interval, Mapping
        ):
            raise ValueError(f"Clip package work-unit intervals are absent: {package}")
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
    ordered_units = sorted(work_units, key=lambda item: int(item["work_unit_index"]))
    return refined_runs, raw_units, refined_units, contours, quality, ordered_units


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
    allow_signed_hybrid_crop_rebase: bool = False,
) -> dict[str, object]:
    """Stage exact worker views and invoke the fail-closed composable publisher."""

    if not raw_draft_runs or len(raw_draft_runs) != len(refined_package_paths):
        raise ValueError("Raw runs and refined clip packages must have 1:1 coverage.")
    output = local_output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / f".receipt_composed_inputs.{uuid4().hex}"
    assembly = temporary / "assembly.zarr"
    temporary.mkdir()
    try:
        source = analysis_zarr.expanduser().resolve()
        _copy_root_attributes(source, assembly)
        for parent in ("crop_runs", "subject_mask_shard_runs"):
            source_parent = source / parent
            if not source_parent.is_dir():
                raise FileNotFoundError(source_parent)
            os.symlink(source_parent, assembly / parent, target_is_directory=True)
        refined_runs, raw_units, refined_units, contours, quality, work_units = (
            _extract_clip_packages(refined_package_paths, assembly=assembly)
        )
        result = publish_recording_subject_mask_bundle(
            analysis_zarr=source,
            draft_zarr=assembly,
            crop_run=crop_run,
            raw_draft_parent="subject_mask_shard_runs",
            raw_draft_run=str(raw_draft_runs[0]),
            raw_draft_runs=tuple(raw_draft_runs),
            refined_draft_run=refined_runs[0],
            refined_draft_runs=tuple(refined_runs),
            raw_run=raw_run,
            refined_run=refined_run,
            quality_run=quality_run,
            cache_run=cache_run,
            bundle_id=bundle_id,
            local_output_root=output,
            quality_scratch_root=quality_scratch_root,
            quality_partition_roots=quality,
            require_worker_quality=True,
            quality_partition_producer_commit=producer_commit,
            core_physical_unit_workers=int(core_physical_unit_workers),
            core_validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE,
            raw_final_layout_unit_packages=raw_units,
            refined_final_layout_unit_packages=refined_units,
            require_complete_final_layout_units=True,
            sampled_contour_worker_receipts=contours,
            require_worker_sampled_contours=True,
            sampled_contour_producer_commit=producer_commit,
            allow_signed_hybrid_crop_rebase=allow_signed_hybrid_crop_rebase,
            expected_work_units=work_units,
            activate=False,
        )
        return {
            **result,
            "publication_profile": "receipt_composed_clip_workers_v1",
            "worker_count": len(refined_runs),
            "producer_commit": producer_commit,
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


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
        allow_signed_hybrid_crop_rebase=bool(args.allow_signed_hybrid_crop_rebase),
    )
    print(
        json.dumps(json_ready(result), indent=None if args.json else 2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "publish_receipt_composed_bundle"]
