#!/usr/bin/env python3
"""Publish one strict selector-ineligible clipped refined-detection snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.canonical_detection_manifest import (
    canonical_detection_dimensions_from_manifest,
    refined_source_identity_from_canonical_manifest,
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    canonical_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.clipped_refined_detection_finalization import (
    CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_ID,
    prepare_clipped_refined_detection_snapshot,
    publish_selector_ineligible_clipped_refined_detection_snapshot,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import plan_canonical_detection_storage
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionBoundClipEvidence,
    parse_refined_detection_clipped_binding,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _load_canonical(archive: Path, run_id: str):
    path = archive.expanduser().resolve()
    root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    run = root["detect_runs"][run_id]
    manifest = run.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Canonical recording run lacks its run_manifest.")
    dimensions = canonical_detection_dimensions_from_manifest(manifest)
    arrays = {
        array_path: run[array_path]
        for array_path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    }
    profile = storage_profile_from_manifest(
        manifest["payload"]["storage_plan"]["storage_profile"]
    )
    plans = plan_canonical_detection_storage(dimensions, profile=profile)
    direct, consolidated = canonical_detection_metadata_declaration_maps(
        path,
        run_id=run_id,
        plans=plans,
    )
    errors = validate_canonical_detection_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
    )
    if errors:
        raise ValueError("Canonical recording run is invalid: " + "; ".join(errors))
    return dimensions, arrays, refined_source_identity_from_canonical_manifest(manifest)


def finalize_clipped_refined_detection_v1(
    *,
    canonical_archive: Path,
    canonical_run_id: str,
    clip_archives: Sequence[Path],
    clip_run_ids: Sequence[str],
    clipped_binding_path: Path,
    output_archive: Path,
    safe_root: Path,
    output_run_id: str,
    recording_identity: str,
    lineage_id: str,
    snapshot_id: str,
) -> dict[str, object]:
    """Reopen every authority, finalize, and publish one recording snapshot."""

    if not clip_archives or len(clip_archives) != len(clip_run_ids):
        raise ValueError("--clip-archive and --clip-run require equal nonzero counts.")
    binding = parse_refined_detection_clipped_binding(
        _read_json(clipped_binding_path.expanduser().resolve())
    )
    if len(binding.clips) != len(clip_archives):
        raise ValueError("Clipped binding and clip artifact counts differ.")
    canonical_dimensions, canonical_arrays, canonical_source = _load_canonical(
        canonical_archive,
        canonical_run_id,
    )
    evidence: list[RefinedDetectionBoundClipEvidence] = []
    for index, (archive, run_id) in enumerate(
        zip(clip_archives, clip_run_ids, strict=True)
    ):
        source = bind_refined_detection_crop_source(
            archive.expanduser().resolve(),
            run_id=run_id,
            allow_selector_ineligible_benchmark=True,
        )
        evidence.append(
            RefinedDetectionBoundClipEvidence(
                clip_index=index,
                manifest=source.manifest,
                arrays=source.arrays,
                parent_manifest=source.parent_manifest,
                parent_arrays=source.parent_arrays,
            )
        )
    prepared = prepare_clipped_refined_detection_snapshot(
        evidence,
        clipped_binding=binding,
        canonical_arrays=canonical_arrays,
        canonical_dimensions=canonical_dimensions,
        canonical_source=canonical_source,
        recording_identity=recording_identity,
    )
    publication = publish_selector_ineligible_clipped_refined_detection_snapshot(
        prepared,
        destination=output_archive,
        run_id=output_run_id,
        safe_root=safe_root,
        lineage_id=lineage_id,
        snapshot_id=snapshot_id,
        recording_identity=recording_identity,
    )
    return {
        "schema_id": CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_ID,
        "schema_version": 1,
        "status": "complete",
        "canonical_archive": str(canonical_archive.expanduser().resolve()),
        "canonical_run_id": canonical_run_id,
        "clip_archives": [str(path.expanduser().resolve()) for path in clip_archives],
        "clip_run_ids": list(clip_run_ids),
        "output_archive": str(publication.snapshot.output_path),
        "output_run_id": publication.snapshot.run_id,
        "finalization_receipt_path": str(publication.receipt_path),
        "finalization_receipt_digest": publication.receipt["payload_digest"],
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-archive", type=Path, required=True)
    parser.add_argument("--canonical-run", required=True)
    parser.add_argument("--clip-archive", type=Path, action="append", required=True)
    parser.add_argument("--clip-run", action="append", required=True)
    parser.add_argument("--clipped-binding", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = finalize_clipped_refined_detection_v1(
            canonical_archive=args.canonical_archive,
            canonical_run_id=args.canonical_run,
            clip_archives=args.clip_archive,
            clip_run_ids=args.clip_run,
            clipped_binding_path=args.clipped_binding,
            output_archive=args.output_archive,
            safe_root=args.safe_root,
            output_run_id=args.output_run,
            recording_identity=args.recording_identity,
            lineage_id=args.lineage_id,
            snapshot_id=args.snapshot_id,
        )
    except Exception as exc:
        result = {
            "schema_id": CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_ID,
            "schema_version": 1,
            "status": "failed",
            "output_archive": str(args.output_archive),
            "output_run_id": args.output_run,
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
