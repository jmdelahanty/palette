"""Prepare a failed clipped DAG for recovery from its keypoint stage.

Dry-run is the default. Apply mode repairs source-video dimensions on the
planned proxy crop runs and removes only the exact incomplete keypoint shard
groups created by the failed campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import zarr

from fisheye.cluster.clipped_inference import SUPPORTED_PLAN_SCHEMAS
from fisheye.shared.json_safety import write_json_atomic
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
        # The merged proxy is intentionally created by keypoint finalization,
        # after every per-clip keypoint shard succeeds.  A failed keypoint
        # branch therefore normally has only the per-clip proxies available.
        proxy_runs = [str(clip["proxy_crop_run"]) for clip in clips]
        repair = repair_clipped_proxy_crop_contract(
            zarr_path,
            crop_runs=proxy_runs,
            apply=False,
        )
        if repair.get("status") != "ok" or int(repair.get("blocked_crop_run_count") or 0):
            raise RuntimeError(
                "Proxy crop repair preflight failed: "
                + json.dumps(repair, sort_keys=True, default=str)
            )
        repairs_by_run = {
            str(item["crop_run"]): item
            for item in repair.get("crop_runs", [])
            if isinstance(item, Mapping) and item.get("crop_run")
        }
        root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        crop_parent = root["crop_runs"]
        keypoint_parent = root.get("keypoint_shard_runs")
        latest = dict(keypoint_parent.attrs) if keypoint_parent is not None else {}
        planned_keypoint_runs = {str(clip["keypoint_shard_run"]) for clip in clips}
        for pointer in ("latest", "latest_complete"):
            if latest.get(pointer) in planned_keypoint_runs:
                raise RuntimeError(
                    f"Refusing to remove keypoint shard selected by {pointer}: {latest[pointer]}"
                )

        clip_reports: list[dict[str, Any]] = []
        for clip in clips:
            clip_id = str(clip["clip_id"])
            proxy_run = str(clip["proxy_crop_run"])
            proxy = crop_parent[proxy_run]
            proxy_attrs = _attrs_with_repairs(proxy, repairs_by_run.get(proxy_run, {}))
            width = _positive_int(
                proxy_attrs.get("source_video_width") or proxy_attrs.get("width")
            )
            height = _positive_int(
                proxy_attrs.get("source_video_height") or proxy_attrs.get("height")
            )
            if width is None or height is None:
                raise RuntimeError(f"Proxy dimensions remain unresolved for {proxy_run}.")

            cache_manifest = Path(str(clip["cache_manifest"])).expanduser().resolve()
            alias_manifest = Path(str(clip["alias_manifest"])).expanduser().resolve()
            if not cache_manifest.is_file() or not alias_manifest.is_file():
                raise FileNotFoundError(f"Cache manifest or alias is missing for {clip_id}.")
            cache_rows = _manifest_row_count(cache_manifest)

            mask_run = str(clip["subject_mask_shard_run"])
            raw_mask = root["subject_mask_shard_runs"][mask_run]
            if raw_mask.attrs.get("palette_run_completion_status") != "complete":
                raise RuntimeError(f"Raw subject-mask shard is not complete: {mask_run}")
            if raw_mask.attrs.get("source_crop_run") != proxy_run:
                raise RuntimeError(f"Raw subject-mask crop identity mismatch: {mask_run}")
            if raw_mask.attrs.get("source_collection_id") != target["collection_id"]:
                raise RuntimeError(f"Raw subject-mask collection identity mismatch: {mask_run}")
            if raw_mask.attrs.get("source_roi_cache_alias_manifest") != str(alias_manifest):
                raise RuntimeError(f"Raw subject-mask cache identity mismatch: {mask_run}")
            if "mask_probs_roi" not in raw_mask:
                raise RuntimeError(f"Raw subject-mask probabilities are missing: {mask_run}")
            mask_rows = int(raw_mask["mask_probs_roi"].shape[0])
            if mask_rows != cache_rows:
                raise RuntimeError(
                    f"Raw subject-mask row count mismatch for {clip_id}: {mask_rows} != {cache_rows}"
                )
            if not _matching_model_artifact(
                raw_mask.attrs.get("run_provenance"),
                expected_path=expected_model_path,
                expected_sha256=expected_model_sha,
            ):
                raise RuntimeError(f"Raw subject-mask model identity mismatch: {mask_run}")

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
            clip_reports.append(
                {
                    "clip_id": clip_id,
                    "proxy_crop_run": proxy_run,
                    "source_video_width": width,
                    "source_video_height": height,
                    "cache_rows": cache_rows,
                    "raw_subject_mask_run": mask_run,
                    "raw_subject_mask_rows": mask_rows,
                    "keypoint_shard_run": keypoint_run,
                    "keypoint_action": keypoint_action,
                }
            )

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
                "merged_proxy_crop_run": str(target["merged_proxy_crop_run"]),
                "merged_proxy_status": (
                    "present"
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
    removed: list[str] = []
    if apply:
        targets_by_id = {
            str(target["target_id"]): target
            for target in payload["targets"]
        }
        for report in reports:
            target = targets_by_id[str(report["target_id"])]
            zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
            proxy_runs = [str(clip["proxy_crop_run"]) for clip in target["clips"]]
            applied_repair = repair_clipped_proxy_crop_contract(
                zarr_path,
                crop_runs=proxy_runs,
                apply=True,
            )
            if applied_repair.get("status") != "ok":
                raise RuntimeError(f"Proxy repair apply failed for {report['target_id']}")
            root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
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
                removed.append(f"{zarr_path}/keypoint_shard_runs/{run_name}")
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
        "removed_incomplete_keypoint_group_count": len(removed),
        "removed_incomplete_keypoint_groups": removed,
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
