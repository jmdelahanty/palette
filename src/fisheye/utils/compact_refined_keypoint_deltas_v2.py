"""Publish a selector-ineligible refined-keypoint successor from frozen deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.keypoint_manual_review_qc import (
    build_manual_keypoint_review_derivation,
)
from fisheye.shared.tabular_deltas import resolve_keypoint_delta_overlay
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.keypoint_manifest import KEYPOINT_RUN_MANIFEST_ATTRIBUTE
from fisheye.shared.zarr.keypoint_quality_manifest import (
    KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE,
    quality_profile_from_manifest,
)
from fisheye.shared.zarr.keypoint_quality_schema import KeypointQualityDimensions
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_compaction import (
    prepare_refined_keypoint_compaction,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    refined_keypoint_code_maps_from_manifest,
    refined_keypoint_snapshot_identity_from_manifest,
    refined_keypoint_source_bindings_from_manifest,
    successor_refined_keypoint_snapshot_identity,
)
from fisheye.shared.zarr.refined_keypoint_publication import (
    publish_selector_ineligible_refined_keypoint_snapshot,
)
from fisheye.shared.zarr_io import open_zarr_root

COMPACTION_RECEIPT_SCHEMA_ID = "palette.refined_keypoint.delta_compaction"
COMPACTION_RECEIPT_SCHEMA_VERSION = 2


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    return value


def compact_refined_keypoint_deltas_v2(
    *,
    archive: Path,
    delta_run: str,
    generation: str,
    output: Path,
    shadow_root: Path,
    output_run_id: str,
    output_snapshot_id: str,
    created_by: str,
) -> dict[str, object]:
    """Read one frozen generation and publish a complete immutable successor."""

    source_path = archive.expanduser().resolve()
    root = open_zarr_root(source_path, mode="r")
    delta = root[f"edit_delta_runs/{delta_run}"]
    base_run_path = str(delta.attrs.get("base_run_path") or "")
    if not base_run_path.startswith("refined_keypoints_runs/"):
        raise ValueError("The selected delta run does not bind refined keypoints.")
    parent_run_id = base_run_path.split("/", 1)[1]
    parent = root[base_run_path]
    parent_manifest = _mapping(
        parent.attrs.get(REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE),
        name="parent refined-keypoint manifest",
    )
    parent_identity = refined_keypoint_snapshot_identity_from_manifest(parent_manifest)
    if parent_identity.retired_instance_key_count != 0:
        raise ValueError(
            "This compactor requires explicit retired-key payload support when "
            "the parent retired-key set is nonempty."
        )
    parent_payload = _mapping(parent_manifest.get("payload"), name="parent payload")
    source = refined_keypoint_source_bindings_from_manifest(
        _mapping(parent_payload.get("source_bindings"), name="source bindings")
    )
    review_state_map, reason_code_map = refined_keypoint_code_maps_from_manifest(
        parent_manifest
    )
    overlay = resolve_keypoint_delta_overlay(
        root,
        delta_run=delta_run,
        generation=generation,
        n_keypoints=source.dimensions.n_keypoints,
    )
    if overlay.generation_status != "frozen":
        raise ValueError("Compaction input generation must have status=frozen.")

    raw = root[f"keypoints_runs/{source.raw_run_id}"]
    quality = root[f"keypoint_quality_runs/{source.quality_run_id}"]
    crop = root[f"crop_runs/{source.crop_run_id}"]
    raw_manifest = _mapping(
        raw.attrs.get(KEYPOINT_RUN_MANIFEST_ATTRIBUTE), name="raw keypoint manifest"
    )
    quality_manifest = _mapping(
        quality.attrs.get(KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE),
        name="keypoint-quality manifest",
    )
    crop_manifest = _mapping(crop.attrs.get("run_manifest"), name="crop manifest")
    quality_payload = _mapping(
        quality_manifest.get("payload"), name="keypoint-quality payload"
    )
    quality_logical = _mapping(
        quality_payload.get("logical_schema"), name="keypoint-quality logical schema"
    )
    quality_profile = quality_profile_from_manifest(
        _mapping(quality_logical.get("profile"), name="keypoint-quality profile")
    )
    quality_dimensions = KeypointQualityDimensions(
        n_frames=source.dimensions.n_frames,
        n_instances=source.dimensions.n_instances,
        n_keypoints=source.dimensions.n_keypoints,
        n_keypoint_metrics=len(quality_profile.keypoint_metrics),
        n_pose_metrics=len(quality_profile.pose_metrics),
    )
    compacted = prepare_refined_keypoint_compaction(
        parent,
        raw_arrays=raw,
        dimensions=source.dimensions,
        source_crop_arrays=crop,
        skeleton_digest=source.skeleton_digest,
        quality_dimensions=quality_dimensions,
        quality_profile=quality_profile,
        parent_review_state_map=review_state_map,
        parent_reason_code_map=reason_code_map,
        overlay=overlay,
    )
    identity = successor_refined_keypoint_snapshot_identity(
        parent_manifest=parent_manifest,
        snapshot_id=output_snapshot_id,
    )
    generation_group = root[f"edit_delta_runs/{delta_run}/generations/{generation}"]
    review_derivation = build_manual_keypoint_review_derivation(
        base_run_path=base_run_path,
        delta_run=delta_run,
        generation=generation,
        generation_sha256=str(generation_group.attrs.get("generation_sha256") or ""),
        overlay_sha256=overlay.overlay_sha256,
        partition_count=overlay.partition_count,
        event_count=overlay.event_count,
        policy=overlay.review_qc_policy,
    )
    publication = publish_selector_ineligible_refined_keypoint_snapshot(
        compacted.prepared,
        source=source,
        raw_manifest=raw_manifest,
        quality_manifest=quality_manifest,
        crop_manifest=crop_manifest,
        raw_arrays=raw,
        quality_arrays=quality,
        source_crop_arrays=crop,
        identity=identity,
        review_state_map=compacted.review_state_map,
        reason_code_map=compacted.reason_code_map,
        destination=output,
        run_id=output_run_id,
        shadow_root=shadow_root,
        created_by=created_by,
        parent_manifest=parent_manifest,
        parent_arrays=parent,
        parent_retired_instance_keys=(),
        review_derivation=review_derivation,
    )
    payload = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "created_by": str(created_by),
        "source_archive": str(source_path),
        "base": {
            "run_path": base_run_path,
            "run_id": parent_run_id,
            "manifest_digest": canonical_json_sha256(parent_manifest),
            "snapshot_id": parent_identity.snapshot_id,
        },
        "delta": {
            "delta_run": str(delta_run),
            "generation": str(generation),
            "generation_sha256": str(
                generation_group.attrs.get("generation_sha256") or ""
            ),
            "overlay_sha256": overlay.overlay_sha256,
            "partition_count": overlay.partition_count,
            "event_count": overlay.event_count,
            "review_qc_policy_digest": overlay.review_qc_policy_digest,
            "review_derivation": review_derivation,
        },
        "output": {
            "path": str(publication.output_path),
            "run_id": publication.run_id,
            "manifest_digest": canonical_json_sha256(publication.manifest),
            "snapshot_id": publication.identity.snapshot_id,
            "edited_instance_keys": list(compacted.edited_instance_keys),
            "stage_selector_eligible": False,
        },
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": COMPACTION_RECEIPT_SCHEMA_ID,
        "schema_version": COMPACTION_RECEIPT_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    receipt_path = publication.output_path.with_name(
        publication.output_path.name + ".compaction_receipt.json"
    )
    write_json_atomic(receipt_path, receipt, overwrite=False)
    return {
        "status": "complete",
        "output": str(publication.output_path),
        "run_id": publication.run_id,
        "receipt": str(receipt_path),
        "receipt_digest": receipt["payload_digest"],
        "edited_instance_count": len(compacted.edited_instance_keys),
        "stage_selector_eligible": False,
        "production_state_changes": [],
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--delta-run", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shadow-root", required=True, type=Path)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--output-snapshot-id", required=True)
    parser.add_argument("--created-by", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = compact_refined_keypoint_deltas_v2(
        archive=args.archive,
        delta_run=args.delta_run,
        generation=args.generation,
        output=args.output,
        shadow_root=args.shadow_root,
        output_run_id=args.output_run_id,
        output_snapshot_id=args.output_snapshot_id,
        created_by=args.created_by,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPACTION_RECEIPT_SCHEMA_ID",
    "COMPACTION_RECEIPT_SCHEMA_VERSION",
    "compact_refined_keypoint_deltas_v2",
    "main",
]
