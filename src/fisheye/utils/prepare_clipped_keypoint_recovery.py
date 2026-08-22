"""Prepare a failed clipped DAG for recovery from its keypoint stage.

Dry-run is the default. Apply mode repairs historical proxy crop dimensions
when needed, preserves a signed recording-level hybrid provider when present,
and removes only the exact incomplete keypoint shard groups created by the
failed campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.cluster.clipped_inference import SUPPORTED_PLAN_SCHEMAS
from fisheye.shared.hybrid_crop_provider import HYBRID_CROP_RUN_SCHEMA_ID
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import require_runs_parent
from fisheye.utils.repair_clipped_proxy_crop_contract import (
    repair_clipped_proxy_crop_contract,
)


REPORT_SCHEMA = "palette.clipped_keypoint_recovery_preparation.v1"


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _positive_int(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _manifest_row_count(path: Path) -> int:
    payload = _read_json(path)
    array = payload.get("array") if isinstance(payload, Mapping) else None
    shape = array.get("shape") if isinstance(array, Mapping) else None
    if not isinstance(shape, list) or len(shape) != 3 or _positive_int(shape[0]) is None:
        raise ValueError(f"Invalid cache shape in {path}: {shape!r}")
    if not bool(payload.get("cache_complete")):
        raise ValueError(f"Cache is not complete: {path}")
    return int(shape[0])


def _attrs_with_repairs(
    group: zarr.Group,
    repair_report: Mapping[str, Any],
) -> dict[str, Any]:
    attrs = dict(group.attrs)
    updates = repair_report.get("attr_updates")
    if isinstance(updates, Mapping):
        attrs.update(updates)
    return attrs


def _matching_model_artifact(
    provenance: Any,
    *,
    expected_path: str,
    expected_sha256: str,
) -> bool:
    if not isinstance(provenance, Mapping):
        return False
    artifacts = provenance.get("input_artifacts")
    if not isinstance(artifacts, list):
        return False
    return any(
        isinstance(artifact, Mapping)
        and artifact.get("role") == "subject_mask_unet_checkpoint"
        and artifact.get("path") == expected_path
        and artifact.get("sha256") == expected_sha256
        for artifact in artifacts
    )


def _signed_hybrid_crop_report(
    crop_parent: zarr.Group,
    *,
    target: Mapping[str, Any],
    clips: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    run_name = str(target.get("hybrid_crop_run") or "").strip()
    if not run_name or run_name not in crop_parent:
        raise RuntimeError(f"Signed hybrid crop provider is missing: {run_name!r}")
    group = crop_parent[run_name]
    attrs = dict(group.attrs)
    if attrs.get("palette_run_completion_status") != "complete":
        raise RuntimeError(f"Hybrid crop provider is not complete: {run_name}")
    if attrs.get("palette_run_name") != run_name:
        raise RuntimeError(f"Hybrid crop provider run identity mismatch: {run_name}")
    if attrs.get("schema_id") != HYBRID_CROP_RUN_SCHEMA_ID:
        raise RuntimeError(f"Hybrid crop provider schema mismatch: {run_name}")
    if attrs.get("source_pixels") != SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME:
        raise RuntimeError(f"Hybrid crop source-pixel identity mismatch: {run_name}")

    expected_refined = str(target.get("finalized_refined_detect_run") or "").strip()
    if not expected_refined or attrs.get("source_refined_detect_run") != expected_refined:
        raise RuntimeError(f"Hybrid crop refined-detection identity mismatch: {run_name}")
    record = attrs.get("provider_record")
    observed_digest = str(attrs.get("provider_record_sha256") or "").strip()
    if not isinstance(record, Mapping):
        raise RuntimeError(f"Hybrid crop provider record is missing: {run_name}")
    if canonical_json_sha256(record) != observed_digest:
        raise RuntimeError(f"Hybrid crop provider record digest is invalid: {run_name}")
    if record.get("schema_id") != "palette.roi_pixel_provider_record.v1":
        raise RuntimeError(f"Hybrid crop provider record schema mismatch: {run_name}")
    if record.get("crop_run") != run_name:
        raise RuntimeError(f"Hybrid crop provider record binds another run: {run_name}")
    if record.get("refined_detect_run") != expected_refined:
        raise RuntimeError(f"Hybrid crop provider record binds another detection: {run_name}")
    crop_signature = attrs.get("crop_signature")
    if not isinstance(crop_signature, Mapping) or crop_signature.get(
        "provider_record_sha256"
    ) != observed_digest:
        raise RuntimeError(f"Hybrid crop signed identity is stale: {run_name}")

    if "instance_key" not in group:
        raise RuntimeError(f"Hybrid crop instance identity is missing: {run_name}")
    row_count = int(group["instance_key"].shape[0])
    if int(record.get("row_count") or -1) != row_count:
        raise RuntimeError(f"Hybrid crop provider row count is stale: {run_name}")
    cursor = 0
    partitions: list[dict[str, Any]] = []
    for clip in clips:
        start = int(clip.get("crop_row_start") or 0)
        stop = int(clip.get("crop_row_stop") or 0)
        if start != cursor or stop <= start:
            raise RuntimeError(
                "Hybrid crop clip row partitions are not contiguous at "
                f"{clip.get('clip_id')!r}: {start}:{stop}, expected start {cursor}."
            )
        partitions.append(
            {
                "clip_id": str(clip["clip_id"]),
                "crop_row_start": start,
                "crop_row_stop": stop,
                "crop_row_count": stop - start,
            }
        )
        cursor = stop
    if cursor != row_count:
        raise RuntimeError(
            f"Hybrid crop clip partitions cover {cursor} rows, expected {row_count}."
        )
    return {
        "mode": "signed_recording_level_hybrid_provider",
        "crop_run": run_name,
        "provider_record_sha256": observed_digest,
        "source_refined_detect_run": expected_refined,
        "row_count": row_count,
        "source_pixels": attrs["source_pixels"],
        "clip_partitions": partitions,
    }


def _inspect_plan(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(plan_path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") not in SUPPORTED_PLAN_SCHEMAS
    ):
        raise ValueError(f"Unsupported clipped inference plan: {plan_path}")
    targets = payload.get("targets")
    models = payload.get("models")
    mask_model = models.get("subject_masks") if isinstance(models, Mapping) else None
    if not isinstance(targets, list) or not targets or not isinstance(mask_model, Mapping):
        raise ValueError("Plan requires non-empty targets and a subject-mask model binding.")
    expected_model_path = str(mask_model.get("path") or "")
    expected_model_sha = str(mask_model.get("sha256") or "")
    if not expected_model_path or not expected_model_sha:
        raise ValueError("Plan subject-mask model binding is incomplete.")

    reports: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("Plan target must be an object.")
        zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
        clips = target.get("clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError(f"Target {target.get('target_id')!r} has no clips.")
        hybrid_mode = str(payload.get("workflow_scope") or "") == "downstream"
        root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        crop_parent = root["crop_runs"]
        crop_authority: dict[str, Any]
        repair: Mapping[str, Any] | None
        repairs_by_run: dict[str, Mapping[str, Any]]
        if hybrid_mode:
            crop_authority = _signed_hybrid_crop_report(
                crop_parent,
                target=target,
                clips=clips,
            )
            repair = None
            repairs_by_run = {}
        else:
            # The merged proxy is intentionally created by keypoint
            # finalization after every per-clip keypoint shard succeeds.
            proxy_runs = [str(clip["proxy_crop_run"]) for clip in clips]
            repair = repair_clipped_proxy_crop_contract(
                zarr_path,
                crop_runs=proxy_runs,
                apply=False,
            )
            if repair.get("status") != "ok" or int(
                repair.get("blocked_crop_run_count") or 0
            ):
                raise RuntimeError(
                    "Proxy crop repair preflight failed: "
                    + json.dumps(repair, sort_keys=True, default=str)
                )
            repairs_by_run = {
                str(item["crop_run"]): item
                for item in repair.get("crop_runs", [])
                if isinstance(item, Mapping) and item.get("crop_run")
            }
            crop_authority = {
                "mode": "per_clip_proxy_crop_runs",
                "crop_runs": proxy_runs,
            }
        keypoint_parent = root.get("keypoint_shard_runs")
        latest = dict(keypoint_parent.attrs) if keypoint_parent is not None else {}
        planned_keypoint_runs = {str(clip["keypoint_shard_run"]) for clip in clips}
        for pointer in ("latest", "latest_complete"):
            if latest.get(pointer) in planned_keypoint_runs:
                raise RuntimeError(
                    f"Refusing to remove keypoint shard selected by {pointer}: {latest[pointer]}"
                )
        mask_parent = root.get("subject_mask_shard_runs")
        mask_latest = dict(mask_parent.attrs) if mask_parent is not None else {}
        planned_mask_runs = {str(clip["subject_mask_shard_run"]) for clip in clips}
        for pointer in ("latest", "latest_complete"):
            if mask_latest.get(pointer) in planned_mask_runs:
                raise RuntimeError(
                    "Refusing recovery because a planned raw mask shard is "
                    f"selected by {pointer}: {mask_latest[pointer]}"
                )

        clip_reports: list[dict[str, Any]] = []
        for clip in clips:
            clip_id = str(clip["clip_id"])
            if hybrid_mode:
                crop_run = str(target["hybrid_crop_run"])
                row_start = int(clip["crop_row_start"])
                row_stop = int(clip["crop_row_stop"])
                source_rows = row_stop - row_start
                alias_manifest = None
                expected_collection = str(target["finalized_refined_detect_run"])
            else:
                crop_run = str(clip["proxy_crop_run"])
                proxy = crop_parent[crop_run]
                proxy_attrs = _attrs_with_repairs(
                    proxy, repairs_by_run.get(crop_run, {})
                )
                width = _positive_int(
                    proxy_attrs.get("source_video_width") or proxy_attrs.get("width")
                )
                height = _positive_int(
                    proxy_attrs.get("source_video_height") or proxy_attrs.get("height")
                )
                if width is None or height is None:
                    raise RuntimeError(
                        f"Proxy dimensions remain unresolved for {crop_run}."
                    )
                cache_manifest = (
                    Path(str(clip["cache_manifest"])).expanduser().resolve()
                )
                alias_manifest = (
                    Path(str(clip["alias_manifest"])).expanduser().resolve()
                )
                if not cache_manifest.is_file() or not alias_manifest.is_file():
                    raise FileNotFoundError(
                        f"Cache manifest or alias is missing for {clip_id}."
                    )
                source_rows = _manifest_row_count(cache_manifest)
                expected_collection = str(target["collection_id"])

            mask_run = str(clip["subject_mask_shard_run"])
            raw_mask_action = "rerun_absent"
            mask_rows: int | None = None
            if mask_parent is not None and mask_run in mask_parent:
                raw_mask = mask_parent[mask_run]
                completion = str(
                    raw_mask.attrs.get("palette_run_completion_status") or ""
                )
                if completion == "failed":
                    if raw_mask.attrs.get("palette_run_name") != mask_run:
                        raise RuntimeError(
                            f"Failed raw subject-mask identity mismatch: {mask_run}"
                        )
                    raw_mask_action = "remove_failed_and_rerun"
                elif completion != "complete":
                    raise RuntimeError(
                        "Raw subject-mask shard is neither complete nor failed: "
                        f"{mask_run}"
                    )
                else:
                    raw_mask_action = "reuse_complete"
                    if raw_mask.attrs.get("source_crop_run") != crop_run:
                        raise RuntimeError(
                            f"Raw subject-mask crop identity mismatch: {mask_run}"
                        )
                    if raw_mask.attrs.get("source_collection_id") != (
                        expected_collection
                    ):
                        raise RuntimeError(
                            "Raw subject-mask collection identity mismatch: "
                            f"{mask_run}"
                        )
                    if hybrid_mode:
                        if raw_mask.attrs.get("source_collection_path") != (
                            f"refined_detect_runs/{expected_collection}"
                        ):
                            raise RuntimeError(
                                "Raw subject-mask collection path mismatch: "
                                f"{mask_run}"
                            )
                        for attr, expected in (
                            ("source_clip_id", clip_id),
                            ("source_clip_index", int(clip["clip_index"])),
                            ("source_work_unit_id", str(clip["work_unit_id"])),
                        ):
                            if raw_mask.attrs.get(attr) != expected:
                                raise RuntimeError(
                                    f"Raw subject-mask {attr} mismatch: {mask_run}"
                                )
                        if "source_crop_row_ids" not in raw_mask:
                            raise RuntimeError(
                                "Raw subject-mask crop-row identity is missing: "
                                f"{mask_run}"
                            )
                        observed_rows = np.asarray(
                            raw_mask["source_crop_row_ids"][:], dtype=np.int64
                        )
                        expected_rows = np.arange(
                            row_start, row_stop, dtype=np.int64
                        )
                        if not np.array_equal(observed_rows, expected_rows):
                            raise RuntimeError(
                                "Raw subject-mask crop-row identity mismatch: "
                                f"{mask_run}"
                            )
                    elif raw_mask.attrs.get(
                        "source_roi_cache_alias_manifest"
                    ) != str(alias_manifest):
                        raise RuntimeError(
                            f"Raw subject-mask cache identity mismatch: {mask_run}"
                        )
                    if "mask_probs_roi" not in raw_mask:
                        raise RuntimeError(
                            f"Raw subject-mask probabilities are missing: {mask_run}"
                        )
                    mask_rows = int(raw_mask["mask_probs_roi"].shape[0])
                    if mask_rows != source_rows:
                        raise RuntimeError(
                            "Raw subject-mask row count mismatch for "
                            f"{clip_id}: {mask_rows} != {source_rows}"
                        )
                    if not _matching_model_artifact(
                        raw_mask.attrs.get("run_provenance"),
                        expected_path=expected_model_path,
                        expected_sha256=expected_model_sha,
                    ):
                        raise RuntimeError(
                            f"Raw subject-mask model identity mismatch: {mask_run}"
                        )

            keypoint_run = str(clip["keypoint_shard_run"])
            keypoint_action = "absent"
            if keypoint_parent is not None and keypoint_run in keypoint_parent:
                shard = keypoint_parent[keypoint_run]
                status = str(shard.attrs.get("palette_run_completion_status") or "")
                if status == "complete":
                    raise RuntimeError(
                        f"Refusing recovery because keypoint shard is already complete: {keypoint_run}"
                    )
                if shard.attrs.get("palette_run_name") != keypoint_run:
                    raise RuntimeError(f"Incomplete keypoint shard identity mismatch: {keypoint_run}")
                if shard.attrs.get("output_parent") != "keypoint_shard_runs":
                    raise RuntimeError(f"Incomplete keypoint shard parent mismatch: {keypoint_run}")
                keypoint_action = "remove_incomplete"

            package = Path(str(clip["package_path"])).expanduser().resolve()
            if package.exists():
                raise FileExistsError(f"Refined-mask package already exists: {package}")
            clip_report = {
                "clip_id": clip_id,
                "crop_run": crop_run,
                "source_rows": source_rows,
                "raw_subject_mask_run": mask_run,
                "raw_subject_mask_rows": mask_rows,
                "raw_subject_mask_action": raw_mask_action,
                "keypoint_shard_run": keypoint_run,
                "keypoint_action": keypoint_action,
            }
            if hybrid_mode:
                clip_report.update(
                    {
                        "crop_row_start": row_start,
                        "crop_row_stop": row_stop,
                    }
                )
            else:
                clip_report.update(
                    {
                        "proxy_crop_run": crop_run,
                        "source_video_width": width,
                        "source_video_height": height,
                        "cache_rows": source_rows,
                    }
                )
            clip_reports.append(clip_report)

        downstream = (
            ("keypoints_runs", str(target["keypoint_run"])),
            ("refined_keypoints_runs", str(target["refined_keypoint_run"])),
            ("refined_subject_masks_runs", str(target["refined_subject_mask_run"])),
        )
        collisions = [f"{parent}/{name}" for parent, name in downstream if parent in root and name in root[parent]]
        if collisions:
            raise FileExistsError("Downstream recovery outputs already exist: " + ", ".join(collisions))
        reports.append(
            {
                "target_id": str(target["target_id"]),
                "analysis_zarr": str(zarr_path),
                "proxy_repair": repair,
                "crop_authority": crop_authority,
                "merged_proxy_crop_run": str(target["merged_proxy_crop_run"]),
                "merged_proxy_status": (
                    "reused_signed_hybrid_provider"
                    if hybrid_mode
                    else "present"
                    if str(target["merged_proxy_crop_run"]) in crop_parent
                    else "will_create_during_keypoint_finalize"
                ),
                "latest_pending": latest.get("latest_pending"),
                "clips": clip_reports,
            }
        )
    return payload, reports


def prepare_keypoint_recovery(
    plan_path: Path,
    *,
    apply: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    payload, reports = _inspect_plan(plan_path)
    removed_keypoints: list[str] = []
    removed_masks: list[str] = []
    if apply:
        targets_by_id = {
            str(target["target_id"]): target
            for target in payload["targets"]
        }
        for report in reports:
            target = targets_by_id[str(report["target_id"])]
            zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
            hybrid_mode = str(payload.get("workflow_scope") or "") == "downstream"
            if not hybrid_mode:
                proxy_runs = [
                    str(clip["proxy_crop_run"]) for clip in target["clips"]
                ]
                applied_repair = repair_clipped_proxy_crop_contract(
                    zarr_path,
                    crop_runs=proxy_runs,
                    apply=True,
                )
                if applied_repair.get("status") != "ok":
                    raise RuntimeError(
                        f"Proxy repair apply failed for {report['target_id']}"
                    )
            root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
            mask_parent = require_runs_parent(root, "subject_mask_shard_runs")
            selected_masks = {
                str(clip["subject_mask_shard_run"])
                for clip in target["clips"]
                if clip.get("subject_mask_shard_run")
            }
            for run_name in sorted(selected_masks):
                if run_name not in mask_parent:
                    continue
                run = mask_parent[run_name]
                completion = str(
                    run.attrs.get("palette_run_completion_status") or ""
                )
                if completion == "complete":
                    continue
                if completion != "failed" or run.attrs.get(
                    "palette_run_name"
                ) != run_name:
                    raise RuntimeError(
                        f"Raw subject-mask shard became unsafe to remove: {run_name}"
                    )
                del mask_parent[run_name]
                removed_masks.append(
                    f"{zarr_path}/subject_mask_shard_runs/{run_name}"
                )
            for pointer in ("latest_pending",):
                if mask_parent.attrs.get(pointer) in selected_masks:
                    del mask_parent.attrs[pointer]

            parent = require_runs_parent(root, "keypoint_shard_runs")
            selected = {
                str(clip["keypoint_shard_run"])
                for clip in target["clips"]
                if clip.get("keypoint_shard_run")
            }
            for run_name in sorted(selected):
                if run_name not in parent:
                    continue
                if parent[run_name].attrs.get("palette_run_completion_status") == "complete":
                    raise RuntimeError(f"Keypoint shard became complete during recovery: {run_name}")
                del parent[run_name]
                removed_keypoints.append(
                    f"{zarr_path}/keypoint_shard_runs/{run_name}"
                )
            if parent.attrs.get("latest_pending") in selected:
                del parent.attrs["latest_pending"]

        _payload_after, reports = _inspect_plan(plan_path)

    result = {
        "schema": REPORT_SCHEMA,
        "status": "ok",
        "plan_path": str(plan_path),
        "apply": bool(apply),
        "target_count": len(reports),
        "clip_count": sum(len(report["clips"]) for report in reports),
        "removed_incomplete_keypoint_group_count": len(removed_keypoints),
        "removed_incomplete_keypoint_groups": removed_keypoints,
        "removed_failed_subject_mask_group_count": len(removed_masks),
        "removed_failed_subject_mask_groups": removed_masks,
        "subject_mask_rerun_count": sum(
            clip["raw_subject_mask_action"] != "reuse_complete"
            for report in reports
            for clip in report["clips"]
        ),
        "targets": reports,
    }
    if output_path is not None:
        write_json_atomic(output_path.expanduser().resolve(), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    result = prepare_keypoint_recovery(
        args.plan,
        apply=bool(args.apply),
        output_path=args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
