#!/usr/bin/env python3
"""Promote legacy reviewed detect labels into canonical sparse instances.

This is a repair utility for older training archives where the legacy
``manual`` subgroup contains the reviewed labels, but the modern canonical
``instances`` surface was later recomputed from raw detect/quality outputs and
therefore still contains ``missing`` frames.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
    extract_source_detection_rows,
    has_curated_refined_source_detections_projection,
    write_curated_refined_detect_surfaces,
)
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.utils.detection_profile import write_detection_profile
from fisheye.registry.inline_refresh import sync_latest_detection_profile_for_zarr


_REFINED_PARENTS = ("refined_detect_runs", "refined_runs")
_RUN_NAME_NULLS = {"", "n/a", "na", "none", "null", "unknown"}


@dataclass(frozen=True)
class LegacyLabelPlan:
    zarr_path: Path
    source_refined_run: str
    output_refined_run: str
    source_group: str
    total_frames: int
    source_rows: int
    source_unique_frames: int
    canonical_rows_before: int
    canonical_missing_before: int
    manual_rows: int
    complete: bool
    reason: str


@dataclass(frozen=True)
class LegacyLabelApplyResult:
    plan: LegacyLabelPlan
    rows_written: int
    rows_manual: int
    rows_raw_detect: int
    review_state: str
    detection_profile_run: str | None
    registry_sync_status: str


def _normalize_text(value: Any) -> str | None:
    text = normalize_attr(value)
    if text is None:
        return None
    if text.strip().lower() in _RUN_NAME_NULLS:
        return None
    return text


def _open_child_group(parent: zarr.Group, name: str, *, mode: str) -> zarr.Group:
    if not (hasattr(parent, "store_path") and hasattr(parent, "path")):
        if name in parent:
            return parent[name]
        if mode == "a":
            return parent.create_group(name)
        raise KeyError(name)
    child_path = f"{parent.path}/{name}" if getattr(parent, "path", "") else name
    try:
        return zarr.open_group(
            store=parent.store_path.store,
            path=child_path,
            mode=mode,
            use_consolidated=False,
        )
    except TypeError:
        return zarr.open_group(
            store=parent.store_path.store,
            path=child_path,
            mode=mode,
        )


def _select_refined_parent(root: zarr.Group) -> tuple[zarr.Group, str]:
    for name in _REFINED_PARENTS:
        if name in root:
            return root[name], name
    raise RuntimeError("No refined_detect_runs group found.")


def _group_names(group: zarr.Group) -> list[str]:
    if hasattr(group, "group_keys"):
        return sorted(str(name) for name in group.group_keys())
    return sorted(str(name) for name, value in group.items() if hasattr(value, "attrs"))


def _select_refined_run(parent: zarr.Group, requested: str | None) -> str:
    requested_name = _normalize_text(requested)
    if requested_name:
        if requested_name in parent:
            return requested_name
        raise RuntimeError(f"Refined run '{requested_name}' not found.")
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    names = _group_names(parent)
    if not names:
        raise RuntimeError("No refined detect runs are available.")
    return names[-1]


def _default_output_run_name(parent: zarr.Group, *, source_run: str) -> str:
    base = f"{source_run}_legacy_labels_canonical"
    candidate = base
    suffix = 2
    while candidate in parent:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _array_or_none(group: zarr.Group | None, name: str) -> np.ndarray | None:
    if group is None or name not in group:
        return None
    return np.asarray(group[name][:])


def _row_count(group: zarr.Group | None) -> int:
    if group is None or "frame_indices" not in group:
        return 0
    return int(group["frame_indices"].shape[0])


def _unique_frame_count(group: zarr.Group | None) -> int:
    frames = _array_or_none(group, "frame_indices")
    if frames is None or frames.size == 0:
        return 0
    return int(np.unique(np.asarray(frames, dtype=np.int32).reshape(-1)).shape[0])


def _total_frames(root: zarr.Group, refined_run: zarr.Group, groups: Sequence[zarr.Group | None]) -> int:
    for value in (
        root.attrs.get("total_frames"),
        root.attrs.get("n_frames"),
        refined_run.attrs.get("coverage_frames_total"),
        refined_run.attrs.get("coverage_frames_full"),
    ):
        parsed = _as_nonnegative_int(value)
        if parsed is not None:
            return parsed
    raw = root.get("raw_video")
    if raw is not None:
        for value in (raw.attrs.get("total_frames"), raw.attrs.get("n_frames")):
            parsed = _as_nonnegative_int(value)
            if parsed is not None:
                return parsed
        for name in ("images_full", "images_ds_rgb", "images_ds"):
            if name in raw:
                return int(raw[name].shape[0])
    for group in groups:
        if group is not None and "frame_counts" in group:
            return int(group["frame_counts"].shape[0])
    max_frame = -1
    for group in groups:
        frames = _array_or_none(group, "frame_indices")
        if frames is not None and frames.size:
            max_frame = max(max_frame, int(np.max(frames)))
    if max_frame >= 0:
        return max_frame + 1
    raise RuntimeError("Could not resolve total frame count.")


def _as_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _manual_group_name(refined_run: zarr.Group) -> str | None:
    manual_latest = _normalize_text(refined_run.attrs.get("manual_review_latest"))
    if manual_latest and manual_latest in refined_run:
        return manual_latest
    return "manual" if "manual" in refined_run else None


def _canonical_summary(refined_run: zarr.Group) -> Mapping[str, Any]:
    value = refined_run.attrs.get("summary_statistics")
    return value if isinstance(value, Mapping) else {}


def build_legacy_detect_label_plan(
    root: zarr.Group,
    *,
    zarr_path: Path,
    refined_run_name: str | None = None,
    output_run_name: str | None = None,
) -> LegacyLabelPlan:
    parent, _parent_name = _select_refined_parent(root)
    source_run_name = _select_refined_run(parent, refined_run_name)
    source_run = _open_child_group(parent, source_run_name, mode="r")
    manual_name = _manual_group_name(source_run)
    manual_group = source_run[manual_name] if manual_name else None
    instances_group = source_run["instances"] if "instances" in source_run else None
    total = _total_frames(root, source_run, [manual_group, instances_group])
    manual_rows = _row_count(manual_group)
    instances_rows = _row_count(instances_group)
    manual_unique = _unique_frame_count(manual_group)
    instances_unique = _unique_frame_count(instances_group)

    if manual_group is not None and manual_rows == total and manual_unique == total:
        source_group = str(manual_name)
        source_rows = manual_rows
        source_unique = manual_unique
        reason = "legacy_manual_complete"
    elif instances_group is not None and instances_rows == total and instances_unique == total:
        source_group = "instances"
        source_rows = instances_rows
        source_unique = instances_unique
        reason = "canonical_instances_already_complete"
    else:
        source_group = str(manual_name or "instances")
        source_rows = max(manual_rows, instances_rows)
        source_unique = max(manual_unique, instances_unique)
        reason = "no_complete_source_surface"

    requested_output = _normalize_text(output_run_name)
    if requested_output is None:
        requested_output = _default_output_run_name(parent, source_run=source_run_name)
    elif requested_output == source_run_name:
        raise RuntimeError("Output refined run must differ from source refined run.")

    summary = _canonical_summary(source_run)
    return LegacyLabelPlan(
        zarr_path=zarr_path,
        source_refined_run=source_run_name,
        output_refined_run=requested_output,
        source_group=source_group,
        total_frames=total,
        source_rows=source_rows,
        source_unique_frames=source_unique,
        canonical_rows_before=instances_rows,
        canonical_missing_before=int(summary.get("rows_missing") or max(total - instances_unique, 0)),
        manual_rows=manual_rows,
        complete=source_rows == total and source_unique == total,
        reason=reason,
    )


def _read_group_payload(
    source_group: zarr.Group,
    *,
    total_frames: int,
    source_kind_by_frame: Mapping[int, str] | None = None,
    source_row_by_frame: Mapping[int, int] | None = None,
) -> dict[str, np.ndarray]:
    frames = np.asarray(source_group["frame_indices"][:], dtype=np.int32).reshape(-1)
    bbox_norm = np.asarray(source_group["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    if frames.shape[0] != bbox_norm.shape[0]:
        raise RuntimeError("Legacy label source has mismatched frame/bbox row counts.")
    if frames.shape[0] != total_frames or int(np.unique(frames).shape[0]) != total_frames:
        raise RuntimeError("Legacy label source is not complete over sampled frames.")
    if np.any(frames < 0) or np.any(frames >= total_frames):
        raise RuntimeError("Legacy label source contains out-of-range frame indices.")

    scores = _array_or_none(source_group, "scores")
    if scores is None:
        scores = _array_or_none(source_group, "confidence_scores")
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = _array_or_none(source_group, "class_ids")
    if class_ids is not None:
        class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    reason_labels = read_reason_labels(source_group)
    if reason_labels is None:
        reason_labels = np.full(frames.shape[0], "legacy_label", dtype=object)
    else:
        reason_labels = np.asarray(reason_labels, dtype=object).reshape(-1)

    source_row_index = np.full(frames.shape[0], -1, dtype=np.int32)
    if source_row_by_frame:
        for idx, frame in enumerate(frames.tolist()):
            source_row_index[idx] = int(source_row_by_frame.get(int(frame), -1))

    retune_id = _array_or_none(source_group, "retune_id")
    reason_lower = np.asarray([str(label).strip().lower() for label in reason_labels.tolist()], dtype=object)
    manual_mask = np.zeros(frames.shape[0], dtype=bool)
    if retune_id is not None:
        retune_arr = np.asarray(retune_id, dtype=np.int32).reshape(-1)
        if retune_arr.shape[0] == frames.shape[0]:
            manual_mask |= retune_arr >= 0
    manual_mask |= np.isin(reason_lower.astype(str), ["retune", "manual", "manual_edit"])

    source_kind_labels = np.full(frames.shape[0], "raw_detect", dtype=object)
    if source_kind_by_frame:
        for idx, frame in enumerate(frames.tolist()):
            source_kind_labels[idx] = str(source_kind_by_frame.get(int(frame), "raw_detect"))
    source_kind_labels[manual_mask] = "manual"
    return {
        "frame_indices": frames,
        "bbox_norm_coords": bbox_norm,
        "confidence_scores": scores,
        "class_ids": class_ids,
        "reason_labels": reason_labels,
        "source_kind_labels": source_kind_labels,
        "manual_edit_flags": manual_mask,
        "source_detect_row_index": source_row_index,
        "refined_row_ids": frames.astype(np.int64, copy=False),
    }


def _source_maps_from_instances(refined_run: zarr.Group) -> tuple[dict[int, str], dict[int, int]]:
    if "instances" not in refined_run:
        return {}, {}
    instances = refined_run["instances"]
    if "frame_indices" not in instances:
        return {}, {}
    frames = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
    kind_codes = (
        np.asarray(instances["source_kind_codes"][:], dtype=np.int8).reshape(-1)
        if "source_kind_codes" in instances
        else np.full(frames.shape[0], REFINED_SOURCE_KIND_CODE_MAP["raw_detect"], dtype=np.int8)
    )
    source_rows = (
        np.asarray(instances["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
        if "source_detect_row_index" in instances
        else np.full(frames.shape[0], -1, dtype=np.int32)
    )
    reverse_kind = {int(v): str(k) for k, v in REFINED_SOURCE_KIND_CODE_MAP.items()}
    kind_by_frame: dict[int, str] = {}
    source_row_by_frame: dict[int, int] = {}
    for frame, kind_code, source_row in zip(frames.tolist(), kind_codes.tolist(), source_rows.tolist()):
        frame_int = int(frame)
        kind_by_frame.setdefault(frame_int, reverse_kind.get(int(kind_code), "raw_detect"))
        source_row_by_frame.setdefault(frame_int, int(source_row))
    return kind_by_frame, source_row_by_frame


def _source_detection_payload(refined_run: zarr.Group) -> dict[str, np.ndarray] | None:
    if not has_curated_refined_source_detections_projection(refined_run):
        return None
    rows = extract_source_detection_rows(refined_run)
    reverse_decision = {
        int(value): str(key)
        for key, value in REFINED_SOURCE_DETECTION_DECISION_CODE_MAP.items()
    }
    decision_labels = np.asarray(
        [
            reverse_decision.get(int(code), "filtered")
            for code in np.asarray(rows["decision_codes"], dtype=np.int8).reshape(-1).tolist()
        ],
        dtype=object,
    )
    return {
        "source_detect_row_index": np.asarray(rows["source_detect_row_index"], dtype=np.int32).reshape(-1),
        "frame_indices": np.asarray(rows["frame_indices"], dtype=np.int32).reshape(-1),
        "bbox_norm_coords": np.asarray(rows["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4),
        "decision_labels": decision_labels,
        "reason_labels": np.asarray(rows.get("reason", decision_labels), dtype=object).reshape(-1),
        "confidence_scores": np.asarray(rows["confidence_scores"], dtype=np.float32).reshape(-1)
        if "confidence_scores" in rows
        else None,
        "class_ids": np.asarray(rows["class_ids"], dtype=np.int32).reshape(-1)
        if "class_ids" in rows
        else None,
    }


def _copy_attrs(source: zarr.Group, target: zarr.Group) -> None:
    for key, value in dict(source.attrs).items():
        target.attrs[str(key)] = json_attr_safe(value)


def _review_payload(
    *,
    reviewer: str | None,
    notes: str | None,
    source_refined_run: str,
    source_group: str,
    previous_review_status: Any,
) -> dict[str, Any]:
    timestamp = _utc_now()
    payload: dict[str, Any] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "timestamp_utc": timestamp,
        "timestamp": timestamp,
        "resolved_group": "refined",
        "target_group": "refined",
        "preference_chain": ["refined", "manual", "interpolated", "filtered", "raw"],
        "migration_method": "legacy_detect_labels_to_sparse_instances_v1",
        "migration_source_refined_run": source_refined_run,
        "migration_source_group": source_group,
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes
    if previous_review_status is not None:
        payload["previous_review_status"] = json_attr_safe(previous_review_status)
    return payload


def migrate_legacy_detect_labels_for_zarr(
    zarr_path: Path,
    *,
    refined_run_name: str | None = None,
    output_run_name: str | None = None,
    reviewer: str | None = None,
    notes: str | None = None,
    apply: bool = False,
    overwrite: bool = False,
    promote_latest: bool = True,
    write_profile: bool = True,
    registry_path: Path | None = None,
    sync_registry: bool = True,
) -> LegacyLabelPlan | LegacyLabelApplyResult:
    mode = "a" if apply else "r"
    root = open_zarr_group_direct(zarr_path, mode=mode)
    parent, parent_name = _select_refined_parent(root)
    plan = build_legacy_detect_label_plan(
        root,
        zarr_path=zarr_path,
        refined_run_name=refined_run_name,
        output_run_name=output_run_name,
    )
    if not plan.complete:
        raise RuntimeError(
            f"{zarr_path}: no complete legacy/manual or canonical source surface "
            f"({plan.source_rows}/{plan.total_frames} rows, {plan.source_unique_frames} unique frames)."
        )
    if not apply:
        return plan
    if plan.output_refined_run in parent:
        if not overwrite:
            raise RuntimeError(f"Output refined run '{plan.output_refined_run}' already exists.")
        del parent[plan.output_refined_run]

    source_run = _open_child_group(parent, plan.source_refined_run, mode="r")
    output_run = parent.create_group(plan.output_refined_run)
    _copy_attrs(source_run, output_run)
    output_run.attrs["legacy_label_migration_source_run"] = plan.source_refined_run
    output_run.attrs["legacy_label_migration_source_group"] = plan.source_group
    output_run.attrs["legacy_label_migration_created_at_utc"] = _utc_now()

    source_group = source_run[plan.source_group]
    kind_by_frame, source_row_by_frame = _source_maps_from_instances(source_run)
    instance_payload = _read_group_payload(
        source_group,
        total_frames=plan.total_frames,
        source_kind_by_frame=kind_by_frame,
        source_row_by_frame=source_row_by_frame,
    )
    source_payload = _source_detection_payload(source_run)
    source_context = {
        "selection_policy": "legacy_reviewed_labels_to_sparse_instances",
        "migration_mode": "legacy_detect_labels_to_sparse_instances_v1",
        "source_refined_run": plan.source_refined_run,
        "source_group": plan.source_group,
        "canonical_rows_before": plan.canonical_rows_before,
        "canonical_missing_before": plan.canonical_missing_before,
        "legacy_manual_rows": plan.manual_rows,
        "total_frames": plan.total_frames,
    }
    write_curated_refined_detect_surfaces(
        root,
        zarr_path=zarr_path,
        refined_run_name=plan.output_refined_run,
        instance_frame_indices=instance_payload["frame_indices"],
        instance_bbox_norm_coords=instance_payload["bbox_norm_coords"],
        instance_source_kind_labels=instance_payload["source_kind_labels"],
        instance_reason_labels=instance_payload["reason_labels"],
        instance_source_detect_row_index=instance_payload["source_detect_row_index"],
        instance_manual_edit_flags=instance_payload["manual_edit_flags"],
        instance_confidence_scores=instance_payload["confidence_scores"],
        instance_class_ids=instance_payload["class_ids"],
        instance_refined_row_ids=instance_payload["refined_row_ids"],
        source_detection_source_detect_row_index=source_payload["source_detect_row_index"] if source_payload else None,
        source_detection_frame_indices=source_payload["frame_indices"] if source_payload else None,
        source_detection_bbox_norm_coords=source_payload["bbox_norm_coords"] if source_payload else None,
        source_detection_decision_labels=source_payload["decision_labels"] if source_payload else None,
        source_detection_reason_labels=source_payload["reason_labels"] if source_payload else None,
        source_detection_confidence_scores=source_payload["confidence_scores"] if source_payload else None,
        source_detection_class_ids=source_payload["class_ids"] if source_payload else None,
        command=" ".join(sys.argv),
        source_context=source_context,
    )

    output_run = _open_child_group(parent, plan.output_refined_run, mode="a")
    review = _review_payload(
        reviewer=reviewer,
        notes=notes,
        source_refined_run=plan.source_refined_run,
        source_group=plan.source_group,
        previous_review_status=source_run.attrs.get("detect_review_status"),
    )
    output_run.attrs["detect_review_status"] = json_attr_safe(review)
    if promote_latest:
        parent.attrs["latest"] = plan.output_refined_run
        # Legacy-pointer write retained deliberately: archive-repair migrations keep
        # minting detect_review_status_latest for old stores (2026-07 writer retirement).
        parent.attrs["detect_review_status_latest"] = plan.output_refined_run

    profile_run: str | None = None
    registry_sync_status = "skipped"
    if write_profile:
        profile_result = write_detection_profile(
            root,
            zarr_path=zarr_path,
            source_detection_path=f"{parent_name}/{plan.output_refined_run}/instances",
            refined_run=None,
        )
        profile_run = profile_result.run_name
        if sync_registry:
            resolved_registry = registry_path or RegistryPaths.from_env(Path.cwd()).path
            if resolved_registry.exists():
                registry = Registry(resolved_registry)
                try:
                    sync_result = sync_latest_detection_profile_for_zarr(
                        registry,
                        zarr_path,
                        root=root,
                        profile_run=profile_run,
                        apply=True,
                    )
                finally:
                    registry.close()
                registry_sync_status = str(sync_result.get("status", "unknown"))
            else:
                registry_sync_status = "skipped_no_registry"

    source_kind = np.asarray(instance_payload["source_kind_labels"], dtype=object)
    return LegacyLabelApplyResult(
        plan=plan,
        rows_written=int(instance_payload["frame_indices"].shape[0]),
        rows_manual=int(np.sum(source_kind == "manual")),
        rows_raw_detect=int(np.sum(source_kind == "raw_detect")),
        review_state="approved",
        detection_profile_run=profile_run,
        registry_sync_status=registry_sync_status,
    )


def _paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value).expanduser() for value in (args.zarr_path or [])]
    if args.zarr_list:
        paths.extend(
            Path(line.strip()).expanduser()
            for line in args.zarr_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _json_result(result: LegacyLabelPlan | LegacyLabelApplyResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, LegacyLabelApplyResult):
        plan = result.plan
        return {
            "status": "applied",
            "zarr_path": str(plan.zarr_path),
            "source_refined_run": plan.source_refined_run,
            "output_refined_run": plan.output_refined_run,
            "source_group": plan.source_group,
            "total_frames": plan.total_frames,
            "rows_written": result.rows_written,
            "rows_manual": result.rows_manual,
            "rows_raw_detect": result.rows_raw_detect,
            "review_state": result.review_state,
            "detection_profile_run": result.detection_profile_run,
            "registry_sync_status": result.registry_sync_status,
        }
    if isinstance(result, LegacyLabelPlan):
        return {
            "status": "planned",
            "zarr_path": str(result.zarr_path),
            "source_refined_run": result.source_refined_run,
            "output_refined_run": result.output_refined_run,
            "source_group": result.source_group,
            "total_frames": result.total_frames,
            "source_rows": result.source_rows,
            "source_unique_frames": result.source_unique_frames,
            "canonical_rows_before": result.canonical_rows_before,
            "canonical_missing_before": result.canonical_missing_before,
            "manual_rows": result.manual_rows,
            "complete": result.complete,
            "reason": result.reason,
        }
    return result


def _emit(results: Sequence[dict[str, Any]], *, as_json: bool) -> None:
    summary = {
        "total": len(results),
        "applied": sum(1 for row in results if row.get("status") == "applied"),
        "planned": sum(1 for row in results if row.get("status") == "planned"),
        "errors": sum(1 for row in results if row.get("status") == "error"),
        "rows": sum(int(row.get("rows_written") or row.get("source_rows") or 0) for row in results),
    }
    if as_json:
        print(json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True))
        return
    print(json.dumps(summary, sort_keys=True))
    for row in results:
        status = row.get("status")
        path = row.get("zarr_path")
        output = row.get("output_refined_run")
        count = row.get("rows_written") or row.get("source_rows")
        detail = row.get("error") or row.get("reason") or ""
        print(f"{status}: rows={count} output={output} path={path} {detail}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", nargs="*", help="Training Zarr path(s).")
    parser.add_argument("--zarr-list", type=Path, help="Text file with one Zarr path per line.")
    parser.add_argument("--refined-run", help="Source refined detect run (default: parent latest).")
    parser.add_argument("--output-run", help="Output run name. Default: source + _legacy_labels_canonical.")
    parser.add_argument("--apply", action="store_true", help="Write the migrated canonical run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output run.")
    parser.add_argument("--no-promote-latest", action="store_true", help="Do not promote output run to parent latest.")
    parser.add_argument("--reviewer", help="Reviewer/operator name for approved review payload.")
    parser.add_argument("--notes", help="Review/migration notes.")
    parser.add_argument("--skip-detection-profile", action="store_true", help="Do not write detection profile artifacts.")
    parser.add_argument("--no-registry-sync", action="store_true", help="Do not sync detection profile rows into the registry.")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path for profile sync.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after per-archive errors.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    paths = _paths_from_args(args)
    if not paths:
        raise SystemExit("No Zarr paths provided.")
    results: list[dict[str, Any]] = []
    exit_code = 0
    for zarr_path in paths:
        try:
            result = migrate_legacy_detect_labels_for_zarr(
                zarr_path,
                refined_run_name=args.refined_run,
                output_run_name=args.output_run,
                reviewer=args.reviewer,
                notes=args.notes,
                apply=bool(args.apply),
                overwrite=bool(args.overwrite),
                promote_latest=not bool(args.no_promote_latest),
                write_profile=not bool(args.skip_detection_profile),
                registry_path=args.registry,
                sync_registry=not bool(args.no_registry_sync),
            )
            results.append(_json_result(result))
        except Exception as exc:
            exit_code = 1
            results.append({"status": "error", "zarr_path": str(zarr_path), "error": str(exc)})
            if not args.keep_going:
                break
    _emit(results, as_json=bool(args.json))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
