"""Publish one immutable raw/quality-bound refined-keypoint/body-frame canary."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.body_frame_producer import (
    BodyFrameSourceReference,
    build_keypoint_body_frame_recipe,
    prepare_keypoint_body_frame,
)
from fisheye.shared.zarr.body_frame_publication import (
    publish_selector_ineligible_body_frame_snapshot,
)
from fisheye.shared.zarr.keypoint_manifest import validate_keypoint_publication
from fisheye.shared.zarr.keypoint_publication import (
    keypoint_metadata_declaration_maps,
)
from fisheye.shared.zarr.keypoint_quality_manifest import (
    quality_profile_from_manifest,
    validate_keypoint_quality_publication,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    keypoint_quality_metadata_declaration_maps,
)
from fisheye.shared.zarr.keypoint_quality_schema import KeypointQualityDimensions
from fisheye.shared.zarr.keypoint_quality_storage import (
    plan_keypoint_quality_storage,
)
from fisheye.shared.zarr.keypoint_schema import KEYPOINT_SCHEMA_V2
from fisheye.shared.zarr.keypoint_storage import plan_keypoint_storage
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    build_refined_keypoint_source_bindings,
    initial_refined_keypoint_snapshot_identity,
)
from fisheye.shared.zarr.refined_keypoint_producer import (
    LandmarkCoordinateEdit,
    RefinedKeypointDecision,
    prepare_refined_keypoint_snapshot,
)
from fisheye.shared.zarr.refined_keypoint_publication import (
    publish_selector_ineligible_refined_keypoint_snapshot,
)


SCHEMA_ID = "palette.refined_keypoint_v2.integration_canary"
SCHEMA_VERSION = 1
REVIEW_STATE_MAP = {0: "unreviewed", 1: "accepted", 2: "rejected"}
REASON_CODE_MAP = {
    0: "none",
    1: "synthetic_canary_correction",
    2: "synthetic_canary_rejection",
    3: "synthetic_canary_recovery",
}


def _strict_manifest(group: Any) -> dict[str, Any]:
    value = group.attrs.get("run_manifest")
    if not isinstance(value, Mapping):
        raise ValueError("Source run does not contain an object run_manifest.")
    canonical_json_bytes(value)
    return dict(value)


def _metadata_fingerprint(path: Path) -> str:
    digest = sha256()
    for metadata in sorted(path.rglob("zarr.json")):
        relative = metadata.relative_to(path).as_posix().encode("utf-8")
        payload = metadata.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _artifact_stats(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "apparent_bytes": sum(item.stat().st_size for item in files),
    }


def _source_arrays(group: Any, paths: tuple[str, ...]) -> dict[str, Any]:
    return {path: group[path] for path in paths}


def _synthetic_decisions(
    raw_arrays: Mapping[str, Any],
    crop_arrays: Mapping[str, Any],
) -> tuple[RefinedKeypointDecision, ...]:
    keys = np.asarray(raw_arrays["instance_key"][:], dtype=np.uint64)
    success = np.asarray(raw_arrays["pose_success"][:], dtype=bool)
    valid = np.asarray(raw_arrays["keypoint_valid"][:], dtype=bool)
    points = np.asarray(raw_arrays["keypoints_roi"][:], dtype=np.float32)
    crop_rows = np.asarray(raw_arrays["source_crop_row_ids"][:], dtype=np.int64)
    sizes = np.asarray(crop_arrays["roi_sizes_full"][:], dtype=np.int32)[crop_rows]
    successful = np.flatnonzero(success & np.all(valid, axis=1))
    failed = np.flatnonzero(~success)
    if successful.size < 2 or failed.size < 1 or points.shape[1] != 3:
        raise ValueError(
            "The deterministic canary requires two complete poses, one failure, "
            "and the three-landmark skeleton."
        )

    correction_row = int(successful[0])
    rejection_row = int(successful[1])
    recovery_row = int(failed[0])
    original = points[correction_row, 0].astype(np.float64)
    width = float(sizes[correction_row, 0])
    delta = 0.25 if original[0] + 0.25 < width else -0.25
    corrected = (float(original[0] + delta), float(original[1]))

    recovery_width = float(sizes[recovery_row, 0])
    recovery_height = float(sizes[recovery_row, 1])
    recovery_points = (
        (0.40 * recovery_width, 0.50 * recovery_height),
        (0.60 * recovery_width, 0.40 * recovery_height),
        (0.60 * recovery_width, 0.60 * recovery_height),
    )
    return (
        RefinedKeypointDecision(
            instance_key=int(keys[correction_row]),
            accepted=True,
            review_state_code=1,
            reason_code=1,
            coordinate_edits=(LandmarkCoordinateEdit(0, corrected),),
            confidence_valid=True,
            geometry_valid=True,
        ),
        RefinedKeypointDecision(
            instance_key=int(keys[rejection_row]),
            accepted=False,
            review_state_code=2,
            reason_code=2,
        ),
        RefinedKeypointDecision(
            instance_key=int(keys[recovery_row]),
            accepted=True,
            review_state_code=1,
            reason_code=3,
            coordinate_edits=tuple(
                LandmarkCoordinateEdit(index, point)
                for index, point in enumerate(recovery_points)
            ),
            confidence_valid=True,
            geometry_valid=True,
        ),
    )


def _decision_receipt(
    decisions: tuple[RefinedKeypointDecision, ...],
) -> list[dict[str, object]]:
    return [
        {
            "instance_key": decision.instance_key,
            "accepted": decision.accepted,
            "review_state_code": decision.review_state_code,
            "reason_code": decision.reason_code,
            "coordinate_edits": [
                {
                    "keypoint_index": edit.keypoint_index,
                    "xy_roi": list(edit.xy_roi),
                }
                for edit in decision.coordinate_edits
            ],
        }
        for decision in decisions
    ]


def publish(args: argparse.Namespace) -> dict[str, object]:
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    hidden = destination.parent / f".{destination.name}.partial.{uuid4().hex}"
    if hidden.exists():
        raise FileExistsError(f"Hidden work package already exists: {hidden}")
    hidden.mkdir()

    source_paths = (
        args.raw_zarr.expanduser().resolve(),
        args.quality_zarr.expanduser().resolve(),
        args.crop_zarr.expanduser().resolve(),
    )
    fingerprints_before = {
        str(path): _metadata_fingerprint(path) for path in source_paths
    }
    started = time.perf_counter()
    try:
        raw_root = zarr.open_group(
            str(source_paths[0]), mode="r", use_consolidated=False
        )
        quality_root = zarr.open_group(
            str(source_paths[1]), mode="r", use_consolidated=False
        )
        crop_root = zarr.open_group(
            str(source_paths[2]), mode="r", use_consolidated=False
        )
        raw_group = raw_root["keypoints_runs"][args.raw_run]
        quality_group = quality_root["keypoint_quality_runs"][args.quality_run]
        crop_group = crop_root["crop_runs"][args.crop_run]
        raw_manifest = _strict_manifest(raw_group)
        quality_manifest = _strict_manifest(quality_group)
        crop_manifest = _strict_manifest(crop_group)
        source = build_refined_keypoint_source_bindings(
            raw_manifest=raw_manifest,
            quality_manifest=quality_manifest,
            crop_manifest=crop_manifest,
        )
        dimensions = source.dimensions
        raw_arrays = _source_arrays(raw_group, KEYPOINT_SCHEMA_V2.binding_paths)
        crop_arrays = {path: crop_group[path] for path in crop_group.array_keys()}
        quality_profile_raw = quality_manifest["payload"]["logical_schema"][
            "profile"
        ]
        quality_profile = quality_profile_from_manifest(quality_profile_raw)
        quality_dimensions = KeypointQualityDimensions(
            n_frames=dimensions.n_frames,
            n_instances=dimensions.n_instances,
            n_keypoints=dimensions.n_keypoints,
            n_keypoint_metrics=len(quality_profile.keypoint_metrics),
            n_pose_metrics=len(quality_profile.pose_metrics),
        )
        quality_arrays = {
            path: quality_group[path]
            for path in quality_group.array_keys()
        }

        raw_plans = plan_keypoint_storage(dimensions)
        raw_direct, raw_consolidated = keypoint_metadata_declaration_maps(
            source_paths[0], run_id=args.raw_run, plans=raw_plans
        )
        raw_errors = validate_keypoint_publication(
            raw_manifest,
            direct_metadata_declarations=raw_direct,
            consolidated_metadata_declarations=raw_consolidated,
            arrays=raw_arrays,
            source_crop_arrays=crop_arrays,
            source_crop_manifest=crop_manifest,
        )
        if raw_errors:
            raise ValueError("Raw source gate failed: " + "; ".join(raw_errors))
        quality_plans = plan_keypoint_quality_storage(quality_dimensions)
        quality_direct, quality_consolidated = (
            keypoint_quality_metadata_declaration_maps(
                source_paths[1], run_id=args.quality_run, plans=quality_plans
            )
        )
        quality_errors = validate_keypoint_quality_publication(
            quality_manifest,
            direct_metadata_declarations=quality_direct,
            consolidated_metadata_declarations=quality_consolidated,
            arrays=quality_arrays,
            source_arrays=raw_arrays,
            source_manifest=raw_manifest,
        )
        if quality_errors:
            raise ValueError(
                "Quality source gate failed: " + "; ".join(quality_errors)
            )

        decisions = _synthetic_decisions(raw_arrays, crop_arrays)
        prepared = prepare_refined_keypoint_snapshot(
            raw_arrays,
            dimensions=dimensions,
            source_crop_arrays=crop_arrays,
            skeleton_digest=source.skeleton_digest,
            keypoint_quality_arrays=quality_arrays,
            quality_dimensions=quality_dimensions,
            quality_profile=quality_profile,
            decisions=decisions,
            review_state_map=REVIEW_STATE_MAP,
            reason_code_map=REASON_CODE_MAP,
        )
        lineage_id = str(
            uuid5(
                NAMESPACE_URL,
                f"palette:{source.recording_identity}:refined_keypoints_v2",
            )
        )
        snapshot_id = str(
            uuid5(
                NAMESPACE_URL,
                f"palette:{source.raw_manifest_digest}:{source.quality_manifest_digest}:"
                f"{args.refined_run}:synthetic_canary_v1",
            )
        )
        identity = initial_refined_keypoint_snapshot_identity(
            recording_identity=source.recording_identity,
            lineage_id=lineage_id,
            snapshot_id=snapshot_id,
        )
        refined = publish_selector_ineligible_refined_keypoint_snapshot(
            prepared,
            source=source,
            raw_manifest=raw_manifest,
            quality_manifest=quality_manifest,
            crop_manifest=crop_manifest,
            raw_arrays=raw_arrays,
            quality_arrays=quality_arrays,
            source_crop_arrays=crop_arrays,
            identity=identity,
            review_state_map=REVIEW_STATE_MAP,
            reason_code_map=REASON_CODE_MAP,
            destination=hidden / "refined.zarr",
            run_id=args.refined_run,
            shadow_root=hidden,
            created_by="publish_refined_keypoint_v2_canary",
        )

        pose_binding = raw_manifest["payload"]["pose_model_schema_binding"]
        pose_schema = pose_binding["pose_schema"]
        recipe = build_keypoint_body_frame_recipe(
            pose_schema=pose_schema,
            skeleton_digest=source.skeleton_digest,
            keypoint_count=dimensions.n_keypoints,
        )
        body_source = BodyFrameSourceReference(
            stage="refined_keypoints",
            run_name=args.refined_run,
            manifest_digest=canonical_json_sha256(refined.manifest),
            skeleton_id=source.skeleton_id,
            skeleton_digest=source.skeleton_digest,
            keypoint_row_signatures_digest=sha256_array(
                prepared.arrays["keypoint_row_signature"]
            ),
        )
        body_prepared = prepare_keypoint_body_frame(
            prepared.arrays,
            source_dimensions=dimensions,
            source_crop_arrays=crop_arrays,
            source=body_source,
            source_manifest=refined.manifest,
            recipe=recipe,
            review_state_map=REVIEW_STATE_MAP,
            reason_code_map=REASON_CODE_MAP,
        )
        body = publish_selector_ineligible_body_frame_snapshot(
            body_prepared,
            source_manifest=refined.manifest,
            destination=hidden / "body_frame.zarr",
            run_id=args.body_frame_run,
            shadow_root=hidden,
            created_by="publish_refined_keypoint_v2_canary",
        )

        fingerprints_after = {
            str(path): _metadata_fingerprint(path) for path in source_paths
        }
        if fingerprints_after != fingerprints_before:
            raise RuntimeError("A source metadata tree changed during publication.")
        result: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": utc_now(),
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "synthetic_review_decisions": True,
            "source": {
                "raw_zarr": str(source_paths[0]),
                "raw_run": args.raw_run,
                "raw_manifest_digest": source.raw_manifest_digest,
                "quality_zarr": str(source_paths[1]),
                "quality_run": args.quality_run,
                "quality_manifest_digest": source.quality_manifest_digest,
                "crop_zarr": str(source_paths[2]),
                "crop_run": args.crop_run,
                "crop_manifest_digest": source.crop_manifest_digest,
                "metadata_fingerprints_before": fingerprints_before,
                "metadata_fingerprints_after": fingerprints_after,
                "unchanged": True,
            },
            "dimensions": dimensions.as_manifest(),
            "decisions": _decision_receipt(decisions),
            "artifacts": {
                "refined_keypoints": {
                    "path": "refined.zarr",
                    "run_id": refined.run_id,
                    "manifest_digest": canonical_json_sha256(refined.manifest),
                    "logical_content_digest": refined.manifest["payload"][
                        "logical_content"
                    ]["digest"],
                    "storage": refined.plans.as_manifest()["object_estimate"],
                    "timing_seconds": dict(refined.phase_seconds),
                    **_artifact_stats(refined.output_path),
                },
                "body_frame": {
                    "path": "body_frame.zarr",
                    "run_id": body.run_id,
                    "manifest_digest": canonical_json_sha256(body.manifest),
                    "logical_content_digest": body.manifest["payload"][
                        "logical_content"
                    ]["digest"],
                    "storage": body.plans.as_manifest()["object_estimate"],
                    "timing_seconds": dict(body.phase_seconds),
                    **_artifact_stats(body.output_path),
                },
            },
            "elapsed_seconds": time.perf_counter() - started,
            "production_state": {
                "selectors_written": False,
                "registry_written": False,
                "source_archives_mutated": False,
                "training_artifacts_written": False,
            },
        }
        (hidden / "handoff_manifest.json").write_bytes(
            canonical_json_bytes(result) + b"\n"
        )
        os.replace(hidden, destination)
        return result
    except Exception:
        if args.remove_failed_partial and hidden.exists():
            shutil.rmtree(hidden)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zarr", type=Path, required=True)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--quality-zarr", type=Path, required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--crop-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--body-frame-run", required=True)
    parser.add_argument("--remove-failed-partial", action="store_true")
    return parser


def main() -> None:
    result = publish(_parser().parse_args())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
