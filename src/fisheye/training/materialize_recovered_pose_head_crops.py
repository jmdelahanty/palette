"""Publish versioned 192-pixel head crops in recovered training Zarrs.

These crops use exact pixels from the surviving 512-pixel pose ROIs. Their
origin is expressed in that ROI, because the deleted full-frame archives' exact
sensor-pixel crop origins are unavailable. No root or crop selector is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.crop_roi_layout import (
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import (
    is_run_complete,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.training.pose_head_crop_geometry import (
    POSE_HEAD_CROP_SIZE_PX,
    POSE_HEAD_KEYPOINT_LABELS,
    project_pose_head_keypoints,
)
from fisheye.training.recover_merged_training_recording import (
    HEAD_LABELS,
    SOURCE_ONLY_SCHEMA_ID,
    _sha256_array,
    validate_recovered_recording,
)


RUN_ID = "pose_head_center_192_recovered_v001"
RECIPE_ID = "recovered_pose_roi_center_192_visibility_v1"
SCHEMA_ID = "palette.training.recovered_pose_head_crop_visibility.v1"
SOURCE_IMAGE_SHAPE = (512, 512)
ARRAY_NAMES = (
    "roi_images",
    "keypoints_roi",
    "keypoint_visibility",
    "roi_origin_xy_in_pose_512",
    "source_pose_local_row",
    "source_detect_local_row",
    "source_frame_idx",
    "source_pose_merged_row",
    "source_detect_merged_row",
    "source_bbox_norm_coords",
)
SELECTOR_NAMES = ("latest", "latest_complete", "authoritative_run")


def _safe_run_id(value: str) -> str:
    name = str(value).strip()
    if not name or name.startswith(".") or "/" in name or name in {".", ".."}:
        raise ValueError("run ID must be one non-hidden Zarr path component")
    return name


def _digest_index_sha256(index: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RecoveredCropPlan:
    recording_id: str
    source_digest_index_sha256: str
    source_pose: dict[str, Any]
    source_detect: dict[str, Any]
    review_snapshot: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def row_count(self) -> int:
        return int(self.arrays["roi_images"].shape[0])

    @property
    def invisible_point_count(self) -> int:
        return int(np.count_nonzero(self.arrays["keypoint_visibility"] == 0))

    @property
    def rows_with_invisible_points(self) -> int:
        return int(
            np.count_nonzero(np.any(self.arrays["keypoint_visibility"] == 0, axis=1))
        )

    def summary(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "row_count": self.row_count,
            "invisible_point_count": self.invisible_point_count,
            "rows_with_invisible_points": self.rows_with_invisible_points,
            "source_digest_index_sha256": self.source_digest_index_sha256,
            "array_sha256": {
                name: _sha256_array(values) for name, values in self.arrays.items()
            },
        }


def build_recovered_crop_plan(
    root: zarr.Group, source_attrs: dict[str, Any]
) -> RecoveredCropPlan:
    """Bind all output rows to validated pose and detector source axes."""
    if (
        source_attrs.get("schema_id") != SOURCE_ONLY_SCHEMA_ID
        or source_attrs.get("recovery_mode") != "source_only"
        or source_attrs.get("stage_selector_eligible") is not False
        or source_attrs.get("zarr_purpose") != "training"
        or not isinstance(source_attrs.get("source_pose"), dict)
    ):
        raise ValueError("Source is not a pose-bearing recovered training archive")
    review = source_attrs.get("review_snapshot")
    if not isinstance(review, dict) or any(
        review.get(key) != expected
        for key, expected in (
            ("species", "Danio rerio"),
            ("pose_review_state", "approved"),
            ("pose_review_intended_use", "training"),
            ("pose_review_method", "manual"),
        )
    ):
        raise ValueError("Recovered pose source lacks the approved review snapshot")
    pose = root["recovered_sources/pose"]
    detect = root["recovered_sources/detect"]
    source_pixels = np.asarray(pose["roi_images"][:])
    source_points = np.asarray(pose["keypoints_roi"][:])
    n = len(source_pixels)
    if (
        n == 0
        or source_pixels.shape != (n, *SOURCE_IMAGE_SHAPE)
        or source_pixels.dtype != np.uint8
        or source_points.shape != (n, 3, 2)
        or not np.isfinite(source_points).all()
        or int(source_attrs["pose_source_row_count"]) != n
    ):
        raise ValueError("Recovered pose images or three-point labels are invalid")
    left = (SOURCE_IMAGE_SHAPE[1] - POSE_HEAD_CROP_SIZE_PX) // 2
    top = (SOURCE_IMAGE_SHAPE[0] - POSE_HEAD_CROP_SIZE_PX) // 2
    origins = np.tile([left, top], (n, 1)).astype(np.int32)
    projected, visibility = project_pose_head_keypoints(
        source_points,
        source_labels=HEAD_LABELS,
        origins_xy=origins,
        crop_size_px=POSE_HEAD_CROP_SIZE_PX,
    )
    if not np.isfinite(projected).all():
        raise ValueError("Projected recovered keypoints are nonfinite")
    source_detect_local = np.asarray(pose["detect_local_row"][:], dtype=np.int32)
    detect_merged = np.asarray(detect["source_merged_row"][:], dtype=np.int64)
    if (
        source_detect_local.shape != (n,)
        or np.any(source_detect_local < 0)
        or np.any(source_detect_local >= len(detect_merged))
    ):
        raise ValueError("Recovered pose-to-detector row binding is invalid")
    arrays = {
        "roi_images": np.ascontiguousarray(
            source_pixels[
                :,
                top : top + POSE_HEAD_CROP_SIZE_PX,
                left : left + POSE_HEAD_CROP_SIZE_PX,
            ]
        ),
        "keypoints_roi": projected.astype(np.float32),
        "keypoint_visibility": visibility,
        "roi_origin_xy_in_pose_512": origins,
        "source_pose_local_row": np.arange(n, dtype=np.int64),
        "source_detect_local_row": source_detect_local,
        "source_frame_idx": np.asarray(pose["source_frame_idx"][:], dtype=np.int64),
        "source_pose_merged_row": np.asarray(
            pose["source_merged_row"][:], dtype=np.int64
        ),
        "source_detect_merged_row": detect_merged[source_detect_local],
        "source_bbox_norm_coords": np.asarray(pose["crop_bbox_norm_coords"][:]),
    }
    if (
        arrays["roi_images"].shape != (n, 192, 192)
        or any(len(value) != n for value in arrays.values())
        or not np.array_equal(
            arrays["source_frame_idx"],
            np.asarray(detect["source_frame_idx"][:])[source_detect_local],
        )
        or not np.array_equal(
            arrays["source_bbox_norm_coords"],
            np.asarray(detect["bbox_norm_coords"][:])[source_detect_local],
        )
    ):
        raise ValueError("Recovered crop output axes disagree with source identities")
    return RecoveredCropPlan(
        recording_id=str(source_attrs["recording_id"]),
        source_digest_index_sha256=_digest_index_sha256(source_attrs["array_sha256"]),
        source_pose=dict(source_attrs["source_pose"]),
        source_detect=dict(source_attrs["source_detect"]),
        review_snapshot=dict(review),
        arrays=arrays,
    )


def _write_local_run(root: zarr.Group, *, run_id: str, plan: RecoveredCropPlan) -> None:
    parent = require_runs_parent(root, "crop_runs")
    run = parent.create_group(run_id)
    mark_run_started(run, run_name=run_id, stage="crop")
    layout = build_canonical_crop_roi_layout(
        total_rois=plan.row_count, preferred_chunk_len=32
    )
    for name, values in plan.arrays.items():
        if name == "roi_images":
            array = run.create_array(
                name,
                **build_crop_roi_create_kwargs(
                    total_rois=plan.row_count,
                    roi_sz=(POSE_HEAD_CROP_SIZE_PX, POSE_HEAD_CROP_SIZE_PX),
                    layout=layout,
                    overwrite=False,
                ),
            )
            array[:] = values
        else:
            run.create_array(
                name,
                data=np.ascontiguousarray(values),
                chunks=(min(1024, plan.row_count), *values.shape[1:]),
            )
    run.attrs.update(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": 1,
            "stage_selector_eligible": False,
            "crop_storage_mode": "materialized",
            "source_pixels": "recovered_sources/pose/roi_images",
            "source_coordinate_system": "recovered_pose_roi_512_xy",
            "sensor_pixel_origin_available": False,
            "roi_size": [192, 192],
            "source_roi_size": [512, 512],
            "model_input_preprocessing": "replicate_mono_to_rgb_then_scale_1_over_255_no_mean_std_v1",
            "crop_recipe": {
                "id": RECIPE_ID,
                "size_px": 192,
                "origin_xy_in_pose_roi": [160, 160],
                "resize": "none",
                "rotation": "none",
                "padding": "none",
                "outside_point_policy": "retain_row_mark_point_invisible",
                "sensor_crop_equivalence_claim": False,
            },
            "pose_schema": {
                "name": "traditional_v1",
                "skeleton_id": "pose_schema:traditional_v1",
                "keypoint_labels": list(POSE_HEAD_KEYPOINT_LABELS),
                "kpt_shape": [3, 3],
            },
            "source_bindings": {
                "recording_id": plan.recording_id,
                "source_recovery_schema_id": SOURCE_ONLY_SCHEMA_ID,
                "source_array_digest_index_sha256": plan.source_digest_index_sha256,
                "source_pose": plan.source_pose,
                "source_detect": plan.source_detect,
                "source_review_snapshot": plan.review_snapshot,
                "source_pose_row_axis": "recovered_sources/pose",
                "source_detect_row_axis": "recovered_sources/detect",
            },
            "row_count": plan.row_count,
            "invisible_point_count": plan.invisible_point_count,
            "rows_with_invisible_points": plan.rows_with_invisible_points,
            "array_sha256": {
                name: _sha256_array(values) for name, values in plan.arrays.items()
            },
            **crop_roi_layout_attrs(layout),
        }
    )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_id,
        run_provenance=build_writer_run_provenance(
            command="fisheye.training.materialize_recovered_pose_head_crops",
            params={"crop_recipe": RECIPE_ID, "size_px": 192},
            input_run_ids={
                "source_pose": str(plan.source_pose["run_id"]),
                "source_detect": str(plan.source_detect["run_id"]),
            },
        ),
    )


def _validate_run(path: Path) -> dict[str, Any]:
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        attrs = dict(run.attrs)
        n = int(attrs["row_count"])
        hashes = attrs["array_sha256"]
        valid = (
            attrs.get("schema_id") == SCHEMA_ID
            and attrs.get("schema_version") == 1
            and attrs.get("stage_selector_eligible") is False
            and attrs.get("crop_recipe", {}).get("id") == RECIPE_ID
            and attrs.get("source_coordinate_system") == "recovered_pose_roi_512_xy"
            and attrs.get("sensor_pixel_origin_available") is False
            and is_run_complete(run, legacy_default=False)
            and set(hashes) == set(ARRAY_NAMES)
            and tuple(run["roi_images"].shape) == (n, 192, 192)
            and tuple(run["keypoints_roi"].shape) == (n, 3, 2)
            and tuple(run["keypoint_visibility"].shape) == (n, 3)
            and all(int(run[name].shape[0]) == n for name in ARRAY_NAMES)
            and all(
                _sha256_array(np.asarray(run[name][:])) == hashes[name]
                for name in ARRAY_NAMES
            )
        )
        if valid:
            points = np.asarray(run["keypoints_roi"][:])
            visibility = np.asarray(run["keypoint_visibility"][:])
            expected = np.where(
                np.logical_and(points >= 0, points < 192).all(axis=2), 2, 0
            ).astype(np.uint8)
            valid = bool(
                np.array_equal(visibility, expected)
                and int(np.count_nonzero(visibility == 0))
                == int(attrs["invisible_point_count"])
                and int(np.count_nonzero(np.any(visibility == 0, axis=1)))
                == int(attrs["rows_with_invisible_points"])
            )
        return {"valid": bool(valid), "row_count": n}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def validate_published_recovered_crop(
    archive: Path, *, run_id: str = RUN_ID, expected_digest_index: str | None = None
) -> dict[str, Any]:
    """Read the consolidated publication and compare every row to its source."""
    source_attrs = validate_recovered_recording(archive, require_source_only=True)
    source_index = _digest_index_sha256(source_attrs["array_sha256"])
    if expected_digest_index is not None and source_index != expected_digest_index:
        raise ValueError("Recovered source digest index differs from cohort receipt")
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    if "crop_runs" not in root or run_id not in root["crop_runs"]:
        raise ValueError("Consolidated metadata omits the requested crop run")
    run = root[f"crop_runs/{run_id}"]
    if any(root["crop_runs"].attrs.get(key) == run_id for key in SELECTOR_NAMES):
        raise ValueError("Recovered crop run was selected unexpectedly")
    if (
        run.attrs.get("source_bindings", {}).get("source_array_digest_index_sha256")
        != source_index
    ):
        raise ValueError("Crop run binds a different recovered source generation")
    local = _validate_run(archive / "crop_runs" / run_id)
    if not local.get("valid"):
        raise ValueError(f"Published crop run failed content validation: {local}")
    plan = build_recovered_crop_plan(root, source_attrs)
    for name, expected in plan.arrays.items():
        if not np.array_equal(np.asarray(run[name][:]), expected):
            raise ValueError(f"Published crop {name} differs from recovered source")
    return {"run_id": run_id, **plan.summary()}


def materialize_recovered_pose_head_crop(
    *,
    archive: Path,
    run_id: str = RUN_ID,
    scratch_root: Path = Path("/tmp"),
    expected_digest_index: str | None = None,
    apply: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    candidate = _safe_run_id(run_id)
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    target = archive / "crop_runs" / candidate
    if target.exists():
        if not resume:
            raise FileExistsError(target)
        result = validate_published_recovered_crop(
            archive, run_id=candidate, expected_digest_index=expected_digest_index
        )
        return {"status": "already_materialized", "archive": str(archive), **result}
    source_attrs = validate_recovered_recording(archive, require_source_only=True)
    digest_index = _digest_index_sha256(source_attrs["array_sha256"])
    if expected_digest_index is not None and digest_index != expected_digest_index:
        raise ValueError("Recovered source digest index differs from cohort receipt")
    source = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    plan = build_recovered_crop_plan(source, source_attrs)
    result = {"archive": str(archive), "run_id": candidate, **plan.summary()}
    if not apply:
        return {"status": "planned", **result}
    if not scratch_root.is_dir():
        raise FileNotFoundError(scratch_root)
    selectors_before = (
        {name: source["crop_runs"].attrs.get(name) for name in SELECTOR_NAMES}
        if "crop_runs" in source
        else {name: None for name in SELECTOR_NAMES}
    )
    with tempfile.TemporaryDirectory(
        prefix="palette-recovered-head-", dir=scratch_root
    ) as temporary:
        local_archive = Path(temporary) / "training.zarr"
        local_root = zarr.open_group(
            str(local_archive), mode="w", zarr_format=3, use_consolidated=False
        )
        local_root.attrs["zarr_purpose"] = "training"
        _write_local_run(local_root, run_id=candidate, plan=plan)

        def prepare_parents(current_root: zarr.Group) -> tuple[zarr.Group, ...]:
            return (current_root, require_runs_parent(current_root, "crop_runs"))

        def complete_run(
            _root: zarr.Group, parent: zarr.Group, run: zarr.Group
        ) -> None:
            run.attrs["stage_selector_eligible"] = False
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=candidate,
                run_provenance=run.attrs.get("run_provenance"),
            )

        def verify_unselected(current_root: zarr.Group) -> None:
            parent = current_root["crop_runs"]
            if any(
                parent.attrs.get(name) != selectors_before[name]
                for name in SELECTOR_NAMES
            ):
                raise RuntimeError("Crop selector changed during recovered publication")

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_archive / "crop_runs" / candidate,
                target_run_path=target,
                run_name=candidate,
                lock_suffix="recovered_pose_head_crop_materialization",
                publish_schema_id=SCHEMA_ID,
                policy="local_materialize_then_atomic_selector_ineligible_import_v1",
                rollback_policy="retain_failed_selector_ineligible_child_v1",
                content_checksum=True,
            ),
            copy_backend="python",
            validate_run=_validate_run,
            prepare_parents=prepare_parents,
            complete_run=complete_run,
            verify_pointers=verify_unselected,
            payload_metadata={
                "source_recovery_digest_index_sha256": digest_index,
                "crop_recipe": RECIPE_ID,
            },
        )
    with archive_metadata_publication_lock(archive):
        consolidate_metadata_capture_expected_warnings(archive)
    final = validate_published_recovered_crop(
        archive, run_id=candidate, expected_digest_index=expected_digest_index
    )
    return {
        "status": "materialized",
        **result,
        "publication": publication,
        "validated": final,
    }


def materialize_recovered_crop_cohort(
    *,
    relocation_receipt_path: Path,
    run_id: str = RUN_ID,
    scratch_root: Path = Path("/tmp"),
    apply: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Apply one version to every reviewed pose source, retaining detect-only."""
    receipt = json.loads(relocation_receipt_path.read_text())
    if (
        receipt.get("schema_id") != "palette.training.merged_recovery_relocation.v1"
        or receipt.get("stage_selector_eligible") is not False
        or receipt.get("recording_count") != len(receipt.get("recordings", ()))
    ):
        raise ValueError("Invalid recovery relocation receipt")
    collection_path = Path(receipt["source_collection_path"])
    if _sha256_file(collection_path) != receipt["source_collection_sha256"]:
        raise ValueError("Recovery collection receipt content changed")
    candidate = _safe_run_id(run_id)
    output_receipt = relocation_receipt_path.parent / f"{candidate}.collection.json"
    if apply and output_receipt.exists():
        raise FileExistsError(output_receipt)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, source_row in enumerate(receipt["recordings"], start=1):
        recording_id = str(source_row["recording_id"])
        if recording_id in seen:
            raise ValueError("Duplicate recording in relocation receipt")
        seen.add(recording_id)
        archive = Path(source_row["destination_path"])
        if int(source_row["pose_rows"]) == 0:
            attrs = validate_recovered_recording(
                archive, expected_recording_id=recording_id, require_source_only=True
            )
            if (
                _digest_index_sha256(attrs["array_sha256"])
                != source_row["array_digest_index_sha256"]
                or attrs.get("source_pose") is not None
                or (archive / "crop_runs" / candidate).exists()
            ):
                raise ValueError(
                    "Detect-only archive disagrees with its source receipt"
                )
            item = {
                "recording_id": recording_id,
                "archive": str(archive),
                "status": "detect_only_no_pose_crop",
                "row_count": 0,
                "invisible_point_count": 0,
                "rows_with_invisible_points": 0,
            }
        else:
            result = materialize_recovered_pose_head_crop(
                archive=archive,
                run_id=candidate,
                scratch_root=scratch_root,
                expected_digest_index=source_row["array_digest_index_sha256"],
                apply=apply,
                resume=resume,
            )
            if result["recording_id"] != recording_id or int(
                result["row_count"]
            ) != int(source_row["pose_rows"]):
                raise ValueError("Crop row identity differs from relocation receipt")
            item = {
                "recording_id": recording_id,
                "archive": str(archive),
                "status": result["status"],
                "run_path": str(archive / "crop_runs" / candidate),
                "row_count": result["row_count"],
                "invisible_point_count": result["invisible_point_count"],
                "rows_with_invisible_points": result["rows_with_invisible_points"],
                "source_digest_index_sha256": result["source_digest_index_sha256"],
                "array_sha256": result["array_sha256"],
            }
        rows.append(item)
        print(
            f"[{ordinal}/{len(receipt['recordings'])}] {recording_id}: "
            f"{item['status']} rows={item['row_count']} "
            f"invisible_points={item['invisible_point_count']}",
            flush=True,
        )
    if sum(row["row_count"] for row in rows) != int(receipt["pose_rows"]) or sum(
        row["status"] == "detect_only_no_pose_crop" for row in rows
    ) != int(receipt["recording_count"]) - int(receipt["recordings_with_pose"]):
        raise ValueError("Crop cohort totals differ from recovered pose sources")
    result = {
        "schema_id": "palette.training.recovered_pose_head_crop_collection.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": candidate,
        "recipe_id": RECIPE_ID,
        "source_relocation_receipt_path": str(relocation_receipt_path),
        "source_relocation_receipt_sha256": _sha256_file(relocation_receipt_path),
        "recording_count": len(rows),
        "pose_recording_count": sum(row["row_count"] > 0 for row in rows),
        "detect_only_recording_count": sum(
            row["status"] == "detect_only_no_pose_crop" for row in rows
        ),
        "pose_row_count": sum(row["row_count"] for row in rows),
        "invisible_point_count": sum(row["invisible_point_count"] for row in rows),
        "rows_with_invisible_points": sum(
            row["rows_with_invisible_points"] for row in rows
        ),
        "stage_selector_eligible": False,
        "recordings": rows,
    }
    if apply:
        temporary = output_receipt.with_name(f".{output_receipt.name}.partial")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_receipt)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--relocation-receipt", type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.archive) == bool(args.relocation_receipt):
        parser.error("choose exactly one of --archive or --relocation-receipt")
    if args.archive is not None:
        result = materialize_recovered_pose_head_crop(
            archive=args.archive,
            run_id=args.run_id,
            scratch_root=args.scratch_root,
            apply=args.apply,
            resume=args.resume,
        )
    else:
        result = materialize_recovered_crop_cohort(
            relocation_receipt_path=args.relocation_receipt,
            run_id=args.run_id,
            scratch_root=args.scratch_root,
            apply=args.apply,
            resume=args.resume,
        )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"recordings", "publication", "validated", "array_sha256"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
