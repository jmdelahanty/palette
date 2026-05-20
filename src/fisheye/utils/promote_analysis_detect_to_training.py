"""Promote reviewed analysis-Zarr detection frames into a recording training Zarr."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fisheye.registry.db import RegistryPaths
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.tune import detect_review as detect_review_mod
from fisheye.tune.detect_training_promotion_backend import (
    ClippedPromotionFrame,
    DEFAULT_PROMOTED_CROP_RUN,
    PromotionOptions,
    promote_clipped_detection_frames,
    promote_detection_frames,
    write_promotion_result,
)
from fisheye.utils.resolve_clipped_refined_detect_collection import build_collection_frame_map


def _parse_frames(value: str) -> list[int]:
    text = str(value).strip()
    if not text:
        raise ValueError("--frames cannot be empty.")
    frames: set[int] = set()
    for token in re.split(r"[,\s]+", text):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, stop_raw = token.split("-", 1)
            start = int(start_raw)
            stop = int(stop_raw)
            if stop < start:
                raise ValueError(f"Invalid frame range {token!r}: stop < start.")
            frames.update(range(start, stop + 1))
        else:
            frames.add(int(token))
    if not frames:
        raise ValueError("--frames did not contain any valid frame indices.")
    return sorted(frames)


def _parse_clip_index(clip_id: object) -> int:
    match = re.search(r"(\d+)$", str(clip_id))
    return int(match.group(1)) if match else -1


def _infer_training_zarr_from_analysis(analysis_zarr: Path) -> Path:
    path = analysis_zarr.expanduser()
    name = path.name
    suffix = "_analysis.zarr"
    if not name.endswith(suffix):
        raise ValueError(
            "Could not infer training Zarr path: analysis Zarr name must end with "
            f"{suffix!r}. Pass the target as the second positional argument or with --training-zarr."
        )
    return path.with_name(f"{name[: -len(suffix)]}_training.zarr")


def _resolve_registry_training_zarr(analysis_zarr: Path, registry_path: Path | None) -> tuple[Path, dict[str, Any]]:
    resolved_registry_path = (registry_path or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    if not resolved_registry_path.exists():
        raise ValueError(f"Registry database does not exist: {resolved_registry_path}")

    analysis_path = analysis_zarr.expanduser().resolve()
    conn = sqlite3.connect(str(resolved_registry_path))
    conn.row_factory = sqlite3.Row
    try:
        analysis_rows = conn.execute(
            """
            SELECT dataset_id, recording_id, zarr_path, status
            FROM datasets
            WHERE zarr_path IN (?, ?)
            ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, dataset_id
            """,
            (str(analysis_path), str(analysis_zarr.expanduser())),
        ).fetchall()
        if not analysis_rows:
            raise ValueError(f"Analysis Zarr is not registered in datasets: {analysis_path}")
        analysis_row = analysis_rows[0]
        recording_id = str(analysis_row["recording_id"] or "").strip()
        if not recording_id:
            raise ValueError(f"Registered analysis dataset has no recording_id: {analysis_row['dataset_id']}")

        training_rows = conn.execute(
            """
            SELECT dataset_id, zarr_path, status
            FROM datasets
            WHERE recording_id = ?
              AND zarr_use = 'training'
              AND status = 'active'
            ORDER BY dataset_id
            """,
            (recording_id,),
        ).fetchall()
    finally:
        conn.close()

    if not training_rows:
        raise ValueError(f"No active training dataset registered for recording_id={recording_id!r}")
    if len(training_rows) > 1:
        choices = [f"{row['dataset_id']} -> {row['zarr_path']}" for row in training_rows]
        raise ValueError(
            "Multiple active training datasets are registered for "
            f"recording_id={recording_id!r}; use --training-zarr. Choices: {choices}"
        )

    training_row = training_rows[0]
    target = Path(str(training_row["zarr_path"])).expanduser()
    if not target.exists():
        raise ValueError(
            f"Registry target training Zarr does not exist: {target}. "
            "Use --training-zarr to override or repair the registry row."
        )
    return target, {
        "registry_path": str(resolved_registry_path),
        "analysis_dataset_id": str(analysis_row["dataset_id"]),
        "training_dataset_id": str(training_row["dataset_id"]),
        "recording_id": recording_id,
    }


def _resolve_training_zarr_arg(
    analysis_zarr: Path,
    positional_training_zarr: Path | None,
    override_training_zarr: Path | None,
    *,
    use_registry_target: bool,
    registry_path: Path | None,
) -> tuple[Path, str, dict[str, Any] | None]:
    if positional_training_zarr is not None and override_training_zarr is not None:
        raise ValueError("Provide training Zarr either as the second positional argument or --training-zarr, not both.")
    if use_registry_target and (positional_training_zarr is not None or override_training_zarr is not None):
        raise ValueError("Do not combine --use-registry-target with an explicit training Zarr path.")
    if use_registry_target:
        target, registry_info = _resolve_registry_training_zarr(analysis_zarr, registry_path)
        return target, "registry", registry_info
    if override_training_zarr is not None:
        return override_training_zarr, "explicit", None
    if positional_training_zarr is not None:
        return positional_training_zarr, "explicit", None
    return _infer_training_zarr_from_analysis(analysis_zarr), "inferred", None


def _table_rows(table: Any) -> list[dict[str, Any]]:
    """Return pyarrow table rows as plain dictionaries."""
    if hasattr(table, "to_pylist"):
        return list(table.to_pylist())
    return [dict(row) for row in table]


def _first_present_int(row: Mapping[str, Any], keys: Sequence[str], default: int) -> int:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return int(value)
    return int(default)


def _selected_run_video_path(selected_run: Mapping[str, Any], frame_row: Mapping[str, Any]) -> str:
    source = selected_run.get("source")
    if isinstance(source, Mapping) and source.get("video_path"):
        return str(source["video_path"])
    for key in ("video_path", "source_video_path"):
        if selected_run.get(key):
            return str(selected_run[key])
    if frame_row.get("video_path"):
        return str(frame_row["video_path"])
    raise RuntimeError(
        f"No source video path found for camera={selected_run.get('camera_serial')!r} clip={selected_run.get('clip_id')!r}"
    )


def _manual_frame_indices_for_refined_group(
    analysis_root: Any,
    refined_group_path: str,
    *,
    total_frames: int,
) -> set[int]:
    refined_group = analysis_root[str(refined_group_path).strip("/")]
    payload = detect_review_mod._load_dense_curated_edit_payload(  # type: ignore[attr-defined]
        refined_group,
        total_frames=int(total_frames),
    )
    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64).reshape(-1)
    manual_flags = np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)
    return {int(frame) for frame, is_manual in zip(frame_indices.tolist(), manual_flags.tolist()) if bool(is_manual)}


def _discover_clipped_promotion_frames(
    analysis_zarr: Path,
    *,
    collection_id: str | None,
    recording_frame_index: Path | None,
    frames: Sequence[int] | None,
    manual_only: bool,
    limit: int | None,
) -> tuple[list[ClippedPromotionFrame], dict[str, Any]]:
    summary, table = build_collection_frame_map(
        analysis_zarr,
        collection_id=collection_id,
        recording_frame_index=recording_frame_index,
    )
    rows = _table_rows(table)
    if not rows:
        raise RuntimeError("Resolved clipped collection has no mapped frames.")

    selected_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary.get("selected_runs", []):
        if not isinstance(row, Mapping):
            continue
        key = (str(row.get("camera_serial") or ""), str(row.get("clip_id") or ""))
        if all(key):
            selected_by_pair[key] = dict(row)

    requested_parent_frames = set(int(frame) for frame in frames) if frames is not None else None
    analysis_root = open_zarr_group_direct(analysis_zarr, mode="r")

    rows_by_refined_group: dict[str, list[dict[str, Any]]] = {}
    for row_number, row in enumerate(rows):
        parent_frame = _first_present_int(row, ("parent_frame_index",), row_number)
        if requested_parent_frames is not None and parent_frame not in requested_parent_frames:
            continue
        refined_group_path = str(row.get("refined_group_path") or "").strip("/")
        if not refined_group_path:
            continue
        rows_by_refined_group.setdefault(refined_group_path, []).append(dict(row))

    manual_local_frames_by_group: dict[str, set[int]] = {}
    if manual_only:
        for refined_group_path, group_rows in rows_by_refined_group.items():
            max_local = max(int(row["clip_local_frame_index"]) for row in group_rows)
            manual_local_frames_by_group[refined_group_path] = _manual_frame_indices_for_refined_group(
                analysis_root,
                refined_group_path,
                total_frames=max_local + 1,
            )

    discovered: list[ClippedPromotionFrame] = []
    skipped_non_manual = 0
    for refined_group_path in sorted(rows_by_refined_group):
        for row in sorted(
            rows_by_refined_group[refined_group_path],
            key=lambda item: int(item.get("parent_frame_index", 0)),
        ):
            local_frame = int(row["clip_local_frame_index"])
            if manual_only and local_frame not in manual_local_frames_by_group.get(refined_group_path, set()):
                skipped_non_manual += 1
                continue
            camera_serial = str(row["camera_serial"])
            clip_id = str(row["clip_id"])
            selected = selected_by_pair.get((camera_serial, clip_id), {})
            raw_clip_index = selected.get("clip_index")
            clip_index = int(raw_clip_index) if raw_clip_index is not None else _parse_clip_index(clip_id)
            parent_frame = _first_present_int(row, ("parent_frame_index", "recording_frame_id"), local_frame)
            discovered.append(
                ClippedPromotionFrame(
                    parent_frame_index=parent_frame,
                    clip_local_frame_index=local_frame,
                    refined_group_path=refined_group_path,
                    refined_run=str(row.get("refined_detect_run") or selected.get("refined_detect_run") or Path(refined_group_path).name),
                    collection_id=str(summary["collection_id"]),
                    clip_id=clip_id,
                    clip_index=clip_index,
                    camera_serial=camera_serial,
                    source_video_path=_selected_run_video_path(selected, row),
                    recording_frame_id=int(row["recording_frame_id"]) if row.get("recording_frame_id") is not None else None,
                )
            )
            if limit is not None and len(discovered) >= int(limit):
                break
        if limit is not None and len(discovered) >= int(limit):
            break

    discovery = {
        "mode": "clipped",
        "collection_id": str(summary["collection_id"]),
        "recording_frame_index": summary.get("recording_frame_index"),
        "selected_run_count": int(summary.get("selected_run_count", 0)),
        "mapped_frame_count": int(summary.get("mapped_frame_count", len(rows))),
        "requested_parent_frame_count": None if requested_parent_frames is None else int(len(requested_parent_frames)),
        "manual_only": bool(manual_only),
        "skipped_non_manual_frame_count": int(skipped_non_manual),
        "discovered_frame_count": int(len(discovered)),
        "limit": None if limit is None else int(limit),
    }
    return discovered, discovery


def _empty_clipped_result(
    *,
    analysis_zarr: Path,
    training_zarr: Path,
    options: PromotionOptions,
    apply: bool,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "palette.analysis_detect_to_training_promotion.v1",
        "status": "ok",
        "mode": "clipped",
        "apply": bool(apply),
        "analysis_zarr": str(analysis_zarr.expanduser().resolve()),
        "training_zarr": str(training_zarr.expanduser().resolve()),
        "training_zarr_exists": bool(training_zarr.exists()),
        "collection_ids": [str(discovery.get("collection_id", ""))],
        "target_crop_run": str(options.target_crop_run or DEFAULT_PROMOTED_CROP_RUN),
        "parent_frames_requested": [],
        "action_counts": {},
        "items": [],
        "image_shape": [int(options.target_size[0]), int(options.target_size[1])] if options.target_size else [640, 640],
        "images_appended": 0,
        "images_updated": 0,
        "image_sources": [],
        "label_origin": str(options.label_origin),
        "include_negative": bool(options.include_negative),
        "allow_unreviewed_negative": bool(options.allow_unreviewed_negative),
        "discovery": dict(discovery),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path, help="Source analysis Zarr.")
    parser.add_argument(
        "training_zarr",
        type=Path,
        nargs="?",
        help="Target per-recording training Zarr. Optional for standard <recording>_analysis.zarr layout.",
    )
    parser.add_argument(
        "--training-zarr",
        type=Path,
        dest="training_zarr_override",
        help="Explicit target per-recording training Zarr override.",
    )
    parser.add_argument(
        "--use-registry-target",
        action="store_true",
        help="Resolve the active training Zarr for the analysis recording from the registry.",
    )
    parser.add_argument("--registry", type=Path, help="Registry SQLite path for --use-registry-target.")
    parser.add_argument(
        "--frames",
        help=(
            "Comma/space-separated parent frame list or inclusive ranges, e.g. 10,15,20-25. "
            "Required for traditional runs. Optional for clipped backfill, where omitted means all manual frames."
        ),
    )
    parser.add_argument("--refined-run", help="Source refined detect run. Defaults to refined_detect_runs.attrs['latest'].")
    parser.add_argument("--collection-id", help="Finalized clipped refined-detect collection id.")
    parser.add_argument("--recording-frame-index", type=Path, help="Override recording_frame_index.parquet for clipped collections.")
    parser.add_argument("--target-crop-run", help="Target crop_runs/<run>. Defaults to crop_runs.attrs['latest'] or promoted_detect_manual.")
    parser.add_argument(
        "--target-refined-run",
        help=(
            "Target refined_detect_runs/<run> for canonical promoted instances. "
            "Defaults to refined_detect_runs.attrs['latest'] or refined_detect_promoted_manual."
        ),
    )
    parser.add_argument(
        "--no-sync-refined-instances",
        action="store_true",
        help="Do not mirror promoted labels into refined_detect_runs/<run>/instances.",
    )
    parser.add_argument("--label-origin", default="manual_review", help="label_origin value for promoted rows.")
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="For clipped mode, include requested frames even if manual_edit_flags is false. Default promotes manual frames only.",
    )
    parser.add_argument("--limit", type=int, help="Limit discovered clipped promotion frames, useful for smoke dry-runs.")
    parser.add_argument("--no-negative", action="store_true", help="Skip negative promotion rows.")
    parser.add_argument(
        "--allow-unreviewed-negative",
        action="store_true",
        help="Allow non-present rows without manual_edit=True to become negative examples.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Target images_ds shape when the training Zarr does not already define one.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply the upsert. Default is dry-run.")
    parser.add_argument("--output-json", type=Path, help="Write the plan/result JSON.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        training_zarr, training_zarr_source, registry_target = _resolve_training_zarr_arg(
            args.analysis_zarr,
            args.training_zarr,
            args.training_zarr_override,
            use_registry_target=bool(args.use_registry_target),
            registry_path=args.registry,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    frames = _parse_frames(args.frames) if args.frames else None
    options = PromotionOptions(
        refined_run=args.refined_run,
        target_crop_run=args.target_crop_run,
        target_refined_run=args.target_refined_run,
        sync_refined_instances=not bool(args.no_sync_refined_instances),
        label_origin=args.label_origin,
        include_negative=not bool(args.no_negative),
        allow_unreviewed_negative=bool(args.allow_unreviewed_negative),
        target_size=tuple(args.target_size) if args.target_size else None,
    )
    discovery: dict[str, Any] | None = None
    if args.collection_id:
        clipped_frames, discovery = _discover_clipped_promotion_frames(
            args.analysis_zarr,
            collection_id=args.collection_id,
            recording_frame_index=args.recording_frame_index,
            frames=frames,
            manual_only=not bool(args.include_unreviewed),
            limit=args.limit,
        )
        if clipped_frames:
            result = promote_clipped_detection_frames(
                args.analysis_zarr,
                training_zarr,
                clipped_frames,
                options=options,
                apply=bool(args.apply),
            )
            result["discovery"] = discovery
        else:
            result = _empty_clipped_result(
                analysis_zarr=args.analysis_zarr,
                training_zarr=training_zarr,
                options=options,
                apply=bool(args.apply),
                discovery=discovery,
            )
    else:
        if frames is None:
            raise SystemExit("--frames is required unless --collection-id is provided for clipped backfill.")
        result = promote_detection_frames(
            args.analysis_zarr,
            training_zarr,
            frames,
            options=options,
            apply=bool(args.apply),
        )
    result["training_zarr_source"] = str(training_zarr_source)
    result["training_zarr_inferred"] = training_zarr_source == "inferred"
    if registry_target is not None:
        result["registry_target"] = registry_target
    if args.output_json:
        write_promotion_result(args.output_json, result)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"status: {result['status']}")
        print(f"apply: {result['apply']}")
        print(f"analysis_zarr: {result['analysis_zarr']}")
        print(f"training_zarr: {result['training_zarr']}")
        print(f"training_zarr_source: {result['training_zarr_source']}")
        print(f"training_zarr_inferred: {result['training_zarr_inferred']}")
        if registry_target is not None:
            print(f"registry_target: {registry_target}")
        if result.get("mode") == "clipped":
            print(f"mode: clipped")
            if discovery:
                print(f"collection_id: {discovery['collection_id']}")
                print(f"manual_only: {discovery['manual_only']}")
                print(f"discovered_frame_count: {discovery['discovered_frame_count']}")
                print(f"mapped_frame_count: {discovery['mapped_frame_count']}")
        else:
            print(f"refined_detect_run: {result['refined_detect_run']}")
        print(f"target_crop_run: {result['target_crop_run']}")
        if result.get("refined_instances_sync"):
            sync = result["refined_instances_sync"]
            print(f"refined_instances_sync: {sync['refined_instances_path']} rows={sync['rows_instances']}")
        print(f"action_counts: {result['action_counts']}")
        print(f"images_appended: {result['images_appended']}")
        print(f"images_updated: {result['images_updated']}")
        if args.output_json:
            print(f"output_json: {args.output_json}")
        if not args.apply:
            print("dry_run: pass --apply to write the target training Zarr")
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
