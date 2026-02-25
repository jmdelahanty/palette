#!/usr/bin/env python3
"""Sync latest eye-mask profile summaries from Zarr into registry rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import zarr

from fisheye.registry.db import Registry, RegistryPaths


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        raw = value.decode("utf-8", "ignore")
    elif isinstance(value, str):
        raw = value
    else:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _to_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _first_non_null(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if value is not None:
                return value
    return None


def _metric_percentile(
    geometry: Mapping[str, Any],
    *,
    metric_name: str,
    percentile_key: str,
) -> Optional[float]:
    metric_payload = geometry.get(metric_name)
    if not isinstance(metric_payload, Mapping):
        return None
    stats_payload = metric_payload.get("stats")
    if isinstance(stats_payload, Mapping):
        return _as_float(stats_payload.get(percentile_key))
    return _as_float(metric_payload.get(percentile_key))


def _metric_percentile_aliases(
    geometry: Mapping[str, Any],
    *,
    metric_names: tuple[str, ...],
    percentile_key: str,
) -> Optional[float]:
    for metric_name in metric_names:
        value = _metric_percentile(
            geometry,
            metric_name=metric_name,
            percentile_key=percentile_key,
        )
        if value is not None:
            return value
    return None


def _select_latest_profile_run(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = sorted(str(name) for name in parent.group_keys())
    except Exception:
        names = sorted(str(name) for name in parent.keys())
    return names[-1] if names else None


def _open_root(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="r", consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _open_child_group(parent: zarr.Group, key: str) -> Optional[zarr.Group]:
    store = getattr(parent, "store", None)
    if store is None:
        return None
    parent_path = _normalize_text(getattr(parent, "path", None))
    child_path = f"{parent_path}/{key}" if parent_path else key
    try:
        return zarr.open_group(store=store, path=child_path, mode="r")
    except TypeError:
        try:
            return zarr.open_group(store, mode="r", path=child_path)
        except Exception:
            return None
    except Exception:
        return None


def _get_group(parent: zarr.Group, key: str) -> Optional[zarr.Group]:
    child = parent.get(key)
    if child is not None:
        return child
    try:
        child = parent[key]
    except Exception:
        child = None
    if child is None:
        child = _open_child_group(parent, key)
    return child


def _latest_profile_summary(root: zarr.Group) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[str]]:
    analysis = _get_group(root, "analysis")
    if analysis is None:
        return None, None, "analysis group missing"
    runs_parent = _get_group(analysis, "eye_mask_profile_runs")
    if runs_parent is None:
        return None, None, "analysis/eye_mask_profile_runs missing"
    run_name = _select_latest_profile_run(runs_parent)
    if run_name is None:
        return None, None, "analysis/eye_mask_profile_runs has no runs"
    run_group = _get_group(runs_parent, run_name)
    if run_group is None:
        return None, None, f"profile run missing: {run_name}"
    summary = _coerce_mapping(run_group.attrs.get("profile_summary"))
    if summary is None:
        return run_name, None, f"profile_summary missing or invalid on run: {run_name}"
    return run_name, summary, None


def _build_profile_payload(
    *,
    dataset_id: str,
    fallback_recording_id: Optional[str],
    fallback_zarr_use: Optional[str],
    fallback_genotype: Optional[str],
    fallback_dpf_at_acquisition: Optional[int],
    profile_run: str,
    summary: Mapping[str, Any],
    zarr_path: Path,
) -> dict[str, Any]:
    dataset_map = _coerce_mapping(summary.get("dataset")) or {}
    source_map = _coerce_mapping(summary.get("source")) or {}
    quality_map = _coerce_mapping(summary.get("quality")) or {}
    geometry_map = _coerce_mapping(summary.get("geometry")) or {}
    spatial_map = _coerce_mapping(summary.get("spatial")) or {}
    composition_map = _coerce_mapping(summary.get("composition")) or {}
    freshness_map = _coerce_mapping(summary.get("freshness")) or {}
    review_map = _coerce_mapping(source_map.get("review")) or {}
    stale_map = (
        _coerce_mapping(source_map.get("source_keypoint_stale"))
        or _coerce_mapping(summary.get("source_keypoint_stale"))
        or _coerce_mapping(freshness_map.get("source_keypoint_stale"))
        or {}
    )
    exclusion_reasons = (
        _coerce_mapping(quality_map.get("exclusion_reasons"))
        or _coerce_mapping(quality_map.get("excluded_reasons"))
    )

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except OSError:
        zarr_mtime_ns = None

    dpf_at_acquisition = _as_int(composition_map.get("dpf_at_acquisition"))
    if dpf_at_acquisition is None:
        dpf_at_acquisition = fallback_dpf_at_acquisition

    exclusion_reasons_json = None
    if exclusion_reasons:
        exclusion_reasons_json = _to_json_text(exclusion_reasons)
    else:
        exclusion_reasons_json = _normalize_text(quality_map.get("exclusion_reasons_json"))

    review_timestamp_utc = (
        _normalize_text(source_map.get("review_timestamp_utc"))
        or _normalize_text(source_map.get("reviewed_at_utc"))
        or _normalize_text(source_map.get("review_timestamp"))
        or _normalize_text(review_map.get("timestamp_utc"))
        or _normalize_text(review_map.get("timestamp"))
        or _normalize_text(review_map.get("reviewed_at_utc"))
        or _normalize_text(review_map.get("reviewed_at"))
    )
    source_keypoint_stale_timestamp_utc = (
        _normalize_text(stale_map.get("timestamp_utc"))
        or _normalize_text(stale_map.get("timestamp"))
        or _normalize_text(stale_map.get("stale_at_utc"))
        or _normalize_text(stale_map.get("stale_at"))
    )

    return {
        "dataset_id": dataset_id,
        "profile_run": profile_run,
        "recording_id": _normalize_text(dataset_map.get("recording_id")) or fallback_recording_id,
        "zarr_use": _normalize_text(dataset_map.get("zarr_use")) or fallback_zarr_use,
        "stage_group": (
            _normalize_text(source_map.get("stage_group"))
            or _normalize_text(source_map.get("eye_stage_group"))
            or _normalize_text(source_map.get("source_eye_stage"))
        ),
        "eye_mask_method": (
            _normalize_text(source_map.get("eye_mask_method"))
            or _normalize_text(source_map.get("method"))
        ),
        "source_eye_mask_path": (
            _normalize_text(source_map.get("eye_mask_path"))
            or _normalize_text(source_map.get("source_eye_mask_path"))
        ),
        "source_eye_mask_run": (
            _normalize_text(source_map.get("eye_mask_run"))
            or _normalize_text(source_map.get("source_eye_mask_run"))
        ),
        "source_keypoint_path": (
            _normalize_text(source_map.get("keypoint_path"))
            or _normalize_text(source_map.get("source_keypoint_path"))
        ),
        "source_keypoint_run": (
            _normalize_text(source_map.get("keypoint_run"))
            or _normalize_text(source_map.get("source_keypoint_run"))
        ),
        "source_crop_run": _normalize_text(source_map.get("source_crop_run")),
        "profile_created_utc": _normalize_text(summary.get("created_at_utc")),
        "rows_total": _as_int(_first_non_null(quality_map, "rows_total", "total_rows", "total_rois")),
        "rows_usable": _as_int(
            _first_non_null(
                quality_map,
                "rows_usable",
                "usable_rows",
                "usable_rois",
                "successful_roi_pairs",
            )
        ),
        "usable_rate": _as_float(
            _first_non_null(
                quality_map,
                "usable_rate",
                "usable_rows_rate",
                "usable_roi_pair_rate",
                "successful_roi_pair_rate",
            )
        ),
        "reviewed_rate": _as_float(_first_non_null(quality_map, "reviewed_rate", "review_rate")),
        "excluded_rate": _as_float(_first_non_null(quality_map, "excluded_rate", "exclude_rate")),
        "exclusion_reasons_json": exclusion_reasons_json,
        "ellipse_success_rate": _as_float(
            _first_non_null(quality_map, "ellipse_success_rate", "successful_ellipse_rate")
        ),
        "pair_success_rate": _as_float(
            _first_non_null(
                quality_map,
                "pair_success_rate",
                "successful_roi_pair_rate",
                "usable_rate",
            )
        ),
        "area_p10": _metric_percentile(geometry_map, metric_name="area", percentile_key="p10"),
        "area_p50": _metric_percentile(geometry_map, metric_name="area", percentile_key="p50"),
        "area_p90": _metric_percentile(geometry_map, metric_name="area", percentile_key="p90"),
        "left_area_p10": _metric_percentile_aliases(
            geometry_map,
            metric_names=("left_area", "area_left", "left_eye_area"),
            percentile_key="p10",
        ),
        "left_area_p50": _metric_percentile_aliases(
            geometry_map,
            metric_names=("left_area", "area_left", "left_eye_area"),
            percentile_key="p50",
        ),
        "left_area_p90": _metric_percentile_aliases(
            geometry_map,
            metric_names=("left_area", "area_left", "left_eye_area"),
            percentile_key="p90",
        ),
        "right_area_p10": _metric_percentile_aliases(
            geometry_map,
            metric_names=("right_area", "area_right", "right_eye_area"),
            percentile_key="p10",
        ),
        "right_area_p50": _metric_percentile_aliases(
            geometry_map,
            metric_names=("right_area", "area_right", "right_eye_area"),
            percentile_key="p50",
        ),
        "right_area_p90": _metric_percentile_aliases(
            geometry_map,
            metric_names=("right_area", "area_right", "right_eye_area"),
            percentile_key="p90",
        ),
        "union_area_p10": _metric_percentile_aliases(
            geometry_map,
            metric_names=("union_area", "area_union", "combined_area", "area"),
            percentile_key="p10",
        ),
        "union_area_p50": _metric_percentile_aliases(
            geometry_map,
            metric_names=("union_area", "area_union", "combined_area", "area"),
            percentile_key="p50",
        ),
        "union_area_p90": _metric_percentile_aliases(
            geometry_map,
            metric_names=("union_area", "area_union", "combined_area", "area"),
            percentile_key="p90",
        ),
        "area_lr_ratio_p10": _metric_percentile_aliases(
            geometry_map,
            metric_names=(
                "area_lr_ratio",
                "left_right_area_ratio",
                "area_ratio_left_right",
                "lr_area_ratio",
            ),
            percentile_key="p10",
        ),
        "area_lr_ratio_p50": _metric_percentile_aliases(
            geometry_map,
            metric_names=(
                "area_lr_ratio",
                "left_right_area_ratio",
                "area_ratio_left_right",
                "lr_area_ratio",
            ),
            percentile_key="p50",
        ),
        "area_lr_ratio_p90": _metric_percentile_aliases(
            geometry_map,
            metric_names=(
                "area_lr_ratio",
                "left_right_area_ratio",
                "area_ratio_left_right",
                "lr_area_ratio",
            ),
            percentile_key="p90",
        ),
        "major_axis_p10": _metric_percentile(geometry_map, metric_name="major_axis", percentile_key="p10"),
        "major_axis_p50": _metric_percentile(geometry_map, metric_name="major_axis", percentile_key="p50"),
        "major_axis_p90": _metric_percentile(geometry_map, metric_name="major_axis", percentile_key="p90"),
        "minor_axis_p10": _metric_percentile(geometry_map, metric_name="minor_axis", percentile_key="p10"),
        "minor_axis_p50": _metric_percentile(geometry_map, metric_name="minor_axis", percentile_key="p50"),
        "minor_axis_p90": _metric_percentile(geometry_map, metric_name="minor_axis", percentile_key="p90"),
        "aspect_ratio_p10": _metric_percentile(geometry_map, metric_name="aspect_ratio", percentile_key="p10"),
        "aspect_ratio_p50": _metric_percentile(geometry_map, metric_name="aspect_ratio", percentile_key="p50"),
        "aspect_ratio_p90": _metric_percentile(geometry_map, metric_name="aspect_ratio", percentile_key="p90"),
        "eye_separation_p10": _metric_percentile(geometry_map, metric_name="eye_separation", percentile_key="p10"),
        "eye_separation_p50": _metric_percentile(geometry_map, metric_name="eye_separation", percentile_key="p50"),
        "eye_separation_p90": _metric_percentile(geometry_map, metric_name="eye_separation", percentile_key="p90"),
        "edge_proximity_rate": _as_float(
            _first_non_null(spatial_map, "edge_proximity_rate", "edge_touch_rate")
        ),
        "review_state": _normalize_text(source_map.get("review_state")) or _normalize_text(review_map.get("state")),
        "review_method": _normalize_text(source_map.get("review_method")) or _normalize_text(review_map.get("method")),
        "review_intended_use": (
            _normalize_text(source_map.get("review_intended_use"))
            or _normalize_text(review_map.get("intended_use"))
        ),
        "review_timestamp_utc": review_timestamp_utc,
        "source_keypoint_stale_state": _normalize_text(stale_map.get("state")),
        "source_keypoint_stale_reason": _normalize_text(stale_map.get("reason")),
        "source_keypoint_stale_timestamp_utc": source_keypoint_stale_timestamp_utc,
        "source_keypoint_stale_json": _to_json_text(stale_map) if stale_map else None,
        "rig_id": _normalize_text(composition_map.get("rig_id")),
        "camera_id": _normalize_text(composition_map.get("camera_id")),
        "arena_id": _normalize_text(composition_map.get("arena_id")),
        "dish_design": _normalize_text(composition_map.get("dish_design")),
        "canvas_name": _normalize_text(composition_map.get("canvas_name")),
        "protocol_name": _normalize_text(composition_map.get("protocol_name")),
        "genotype": _normalize_text(composition_map.get("genotype")) or fallback_genotype,
        "dpf_at_acquisition": dpf_at_acquisition,
        "profile_json": _to_json_text(summary),
        "zarr_mtime_ns": zarr_mtime_ns,
    }


def _profile_signature(payload: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        payload.get("recording_id"),
        payload.get("zarr_use"),
        payload.get("stage_group"),
        payload.get("eye_mask_method"),
        payload.get("source_eye_mask_path"),
        payload.get("source_eye_mask_run"),
        payload.get("source_keypoint_path"),
        payload.get("source_keypoint_run"),
        payload.get("source_crop_run"),
        payload.get("profile_created_utc"),
        payload.get("zarr_mtime_ns"),
        payload.get("rows_total"),
        payload.get("rows_usable"),
        payload.get("usable_rate"),
        payload.get("reviewed_rate"),
        payload.get("excluded_rate"),
        payload.get("exclusion_reasons_json"),
        payload.get("ellipse_success_rate"),
        payload.get("pair_success_rate"),
        payload.get("area_p10"),
        payload.get("area_p50"),
        payload.get("area_p90"),
        payload.get("left_area_p10"),
        payload.get("left_area_p50"),
        payload.get("left_area_p90"),
        payload.get("right_area_p10"),
        payload.get("right_area_p50"),
        payload.get("right_area_p90"),
        payload.get("union_area_p10"),
        payload.get("union_area_p50"),
        payload.get("union_area_p90"),
        payload.get("area_lr_ratio_p10"),
        payload.get("area_lr_ratio_p50"),
        payload.get("area_lr_ratio_p90"),
        payload.get("major_axis_p10"),
        payload.get("major_axis_p50"),
        payload.get("major_axis_p90"),
        payload.get("minor_axis_p10"),
        payload.get("minor_axis_p50"),
        payload.get("minor_axis_p90"),
        payload.get("aspect_ratio_p10"),
        payload.get("aspect_ratio_p50"),
        payload.get("aspect_ratio_p90"),
        payload.get("eye_separation_p10"),
        payload.get("eye_separation_p50"),
        payload.get("eye_separation_p90"),
        payload.get("edge_proximity_rate"),
        payload.get("review_state"),
        payload.get("review_method"),
        payload.get("review_intended_use"),
        payload.get("review_timestamp_utc"),
        payload.get("source_keypoint_stale_state"),
        payload.get("source_keypoint_stale_reason"),
        payload.get("source_keypoint_stale_timestamp_utc"),
        payload.get("source_keypoint_stale_json"),
        payload.get("rig_id"),
        payload.get("camera_id"),
        payload.get("arena_id"),
        payload.get("dish_design"),
        payload.get("canvas_name"),
        payload.get("protocol_name"),
        payload.get("genotype"),
        payload.get("dpf_at_acquisition"),
        payload.get("profile_json"),
    )


def _format_status(
    *,
    status: str,
    dataset_id: str,
    profile_run: Optional[str],
    zarr_path: Path,
    reason: Optional[str] = None,
) -> str:
    return (
        f"{status}\t"
        f"{dataset_id}\t"
        f"{profile_run or '-'}\t"
        f"{zarr_path}\t"
        f"{reason or '-'}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync latest eye-mask profile summaries from Zarr into eye_mask_data_profile registry rows."
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="any",
        help="Scope datasets by zarr_use before sync (default: any).",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        help="Optional dataset_id filter. Repeat to pass multiple IDs.",
    )
    parser.add_argument("--path-contains", type=str, help="Optional substring filter on dataset zarr_path.")
    parser.add_argument("--limit", type=int, help="Optional limit on candidate datasets.")
    parser.add_argument("--apply", action="store_true", help="Write registry rows (default: dry-run).")
    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        dataset_rows = [
            dict(row)
            for row in registry.query_datasets(
                zarr_use=(None if str(args.zarr_use) == "any" else str(args.zarr_use)),
                path_contains=args.path_contains,
                limit=args.limit,
            )
        ]
        dataset_id_filters = {
            str(value).strip()
            for value in (args.dataset_id or [])
            if str(value).strip()
        }
        if dataset_id_filters:
            dataset_rows = [
                row
                for row in dataset_rows
                if str(row.get("dataset_id") or "") in dataset_id_filters
            ]

        if not dataset_rows:
            print("No candidate datasets matched filters.")
            return 1

        counts = {
            "datasets": len(dataset_rows),
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "would_insert": 0,
            "would_update": 0,
            "would_unchanged": 0,
            "missing_zarr": 0,
            "missing_profile": 0,
            "error": 0,
        }

        for row in dataset_rows:
            dataset_id = str(row.get("dataset_id") or "")
            zarr_path_raw = row.get("zarr_path")
            zarr_path = Path(str(zarr_path_raw)) if zarr_path_raw else Path("")
            if not dataset_id or not str(zarr_path):
                counts["error"] += 1
                print(
                    _format_status(
                        status="error",
                        dataset_id=dataset_id or "-",
                        profile_run=None,
                        zarr_path=zarr_path,
                        reason="dataset_id or zarr_path missing in registry",
                    )
                )
                continue

            if not zarr_path.exists():
                counts["missing_zarr"] += 1
                print(
                    _format_status(
                        status="missing_zarr",
                        dataset_id=dataset_id,
                        profile_run=None,
                        zarr_path=zarr_path,
                        reason="zarr path not found",
                    )
                )
                continue

            try:
                root = _open_root(zarr_path)
                profile_run, summary, summary_error = _latest_profile_summary(root)
                if summary is None:
                    counts["missing_profile"] += 1
                    print(
                        _format_status(
                            status="missing_profile",
                            dataset_id=dataset_id,
                            profile_run=profile_run,
                            zarr_path=zarr_path,
                            reason=summary_error,
                        )
                    )
                    continue

                if profile_run is None:
                    counts["error"] += 1
                    print(
                        _format_status(
                            status="error",
                            dataset_id=dataset_id,
                            profile_run=None,
                            zarr_path=zarr_path,
                            reason="latest profile run could not be resolved",
                        )
                    )
                    continue

                recording_row = registry.conn.execute(
                    "SELECT recording_id FROM datasets WHERE dataset_id = ?;",
                    (dataset_id,),
                ).fetchone()
                dataset_recording_id = None
                if recording_row is not None:
                    dataset_recording_id = _normalize_text(recording_row["recording_id"])

                payload = _build_profile_payload(
                    dataset_id=dataset_id,
                    fallback_recording_id=dataset_recording_id,
                    fallback_zarr_use=_normalize_text(row.get("zarr_use")),
                    fallback_genotype=_normalize_text(row.get("genotype")),
                    fallback_dpf_at_acquisition=_as_int(row.get("dpf_at_acquisition")),
                    profile_run=str(profile_run),
                    summary=summary,
                    zarr_path=zarr_path,
                )

                existing_row = registry.conn.execute(
                    """
                    SELECT *
                    FROM eye_mask_data_profile
                    WHERE dataset_id = ? AND profile_run = ?;
                    """,
                    (dataset_id, str(profile_run)),
                ).fetchone()

                next_status = "insert"
                if existing_row is not None:
                    existing_payload = {key: existing_row[key] for key in existing_row.keys()}
                    if _profile_signature(existing_payload) == _profile_signature(payload):
                        next_status = "unchanged"
                    else:
                        next_status = "update"

                if args.apply:
                    if next_status == "insert":
                        counts["inserted"] += 1
                        row_status = "inserted"
                    elif next_status == "update":
                        counts["updated"] += 1
                        row_status = "updated"
                    else:
                        counts["unchanged"] += 1
                        row_status = "unchanged"

                    if next_status != "unchanged":
                        registry.upsert_eye_mask_data_profile(
                            dataset_id=str(payload["dataset_id"]),
                            profile_run=str(payload["profile_run"]),
                            recording_id=_normalize_text(payload.get("recording_id")),
                            zarr_use=_normalize_text(payload.get("zarr_use")),
                            stage_group=_normalize_text(payload.get("stage_group")),
                            eye_mask_method=_normalize_text(payload.get("eye_mask_method")),
                            source_eye_mask_path=_normalize_text(payload.get("source_eye_mask_path")),
                            source_eye_mask_run=_normalize_text(payload.get("source_eye_mask_run")),
                            source_keypoint_path=_normalize_text(payload.get("source_keypoint_path")),
                            source_keypoint_run=_normalize_text(payload.get("source_keypoint_run")),
                            source_crop_run=_normalize_text(payload.get("source_crop_run")),
                            profile_created_utc=_normalize_text(payload.get("profile_created_utc")),
                            rows_total=_as_int(payload.get("rows_total")),
                            rows_usable=_as_int(payload.get("rows_usable")),
                            usable_rate=_as_float(payload.get("usable_rate")),
                            reviewed_rate=_as_float(payload.get("reviewed_rate")),
                            excluded_rate=_as_float(payload.get("excluded_rate")),
                            exclusion_reasons_json=_normalize_text(payload.get("exclusion_reasons_json")),
                            ellipse_success_rate=_as_float(payload.get("ellipse_success_rate")),
                            pair_success_rate=_as_float(payload.get("pair_success_rate")),
                            area_p10=_as_float(payload.get("area_p10")),
                            area_p50=_as_float(payload.get("area_p50")),
                            area_p90=_as_float(payload.get("area_p90")),
                            left_area_p10=_as_float(payload.get("left_area_p10")),
                            left_area_p50=_as_float(payload.get("left_area_p50")),
                            left_area_p90=_as_float(payload.get("left_area_p90")),
                            right_area_p10=_as_float(payload.get("right_area_p10")),
                            right_area_p50=_as_float(payload.get("right_area_p50")),
                            right_area_p90=_as_float(payload.get("right_area_p90")),
                            union_area_p10=_as_float(payload.get("union_area_p10")),
                            union_area_p50=_as_float(payload.get("union_area_p50")),
                            union_area_p90=_as_float(payload.get("union_area_p90")),
                            area_lr_ratio_p10=_as_float(payload.get("area_lr_ratio_p10")),
                            area_lr_ratio_p50=_as_float(payload.get("area_lr_ratio_p50")),
                            area_lr_ratio_p90=_as_float(payload.get("area_lr_ratio_p90")),
                            major_axis_p10=_as_float(payload.get("major_axis_p10")),
                            major_axis_p50=_as_float(payload.get("major_axis_p50")),
                            major_axis_p90=_as_float(payload.get("major_axis_p90")),
                            minor_axis_p10=_as_float(payload.get("minor_axis_p10")),
                            minor_axis_p50=_as_float(payload.get("minor_axis_p50")),
                            minor_axis_p90=_as_float(payload.get("minor_axis_p90")),
                            aspect_ratio_p10=_as_float(payload.get("aspect_ratio_p10")),
                            aspect_ratio_p50=_as_float(payload.get("aspect_ratio_p50")),
                            aspect_ratio_p90=_as_float(payload.get("aspect_ratio_p90")),
                            eye_separation_p10=_as_float(payload.get("eye_separation_p10")),
                            eye_separation_p50=_as_float(payload.get("eye_separation_p50")),
                            eye_separation_p90=_as_float(payload.get("eye_separation_p90")),
                            edge_proximity_rate=_as_float(payload.get("edge_proximity_rate")),
                            review_state=_normalize_text(payload.get("review_state")),
                            review_method=_normalize_text(payload.get("review_method")),
                            review_intended_use=_normalize_text(payload.get("review_intended_use")),
                            review_timestamp_utc=_normalize_text(payload.get("review_timestamp_utc")),
                            source_keypoint_stale_state=_normalize_text(payload.get("source_keypoint_stale_state")),
                            source_keypoint_stale_reason=_normalize_text(payload.get("source_keypoint_stale_reason")),
                            source_keypoint_stale_timestamp_utc=_normalize_text(
                                payload.get("source_keypoint_stale_timestamp_utc")
                            ),
                            source_keypoint_stale_json=_normalize_text(payload.get("source_keypoint_stale_json")),
                            rig_id=_normalize_text(payload.get("rig_id")),
                            camera_id=_normalize_text(payload.get("camera_id")),
                            arena_id=_normalize_text(payload.get("arena_id")),
                            dish_design=_normalize_text(payload.get("dish_design")),
                            canvas_name=_normalize_text(payload.get("canvas_name")),
                            protocol_name=_normalize_text(payload.get("protocol_name")),
                            genotype=_normalize_text(payload.get("genotype")),
                            dpf_at_acquisition=_as_int(payload.get("dpf_at_acquisition")),
                            profile_json=_normalize_text(payload.get("profile_json")),
                            zarr_mtime_ns=_as_int(payload.get("zarr_mtime_ns")),
                        )
                    print(
                        _format_status(
                            status=row_status,
                            dataset_id=dataset_id,
                            profile_run=str(profile_run),
                            zarr_path=zarr_path,
                        )
                    )
                else:
                    if next_status == "insert":
                        counts["would_insert"] += 1
                        row_status = "would_insert"
                    elif next_status == "update":
                        counts["would_update"] += 1
                        row_status = "would_update"
                    else:
                        counts["would_unchanged"] += 1
                        row_status = "would_unchanged"
                    print(
                        _format_status(
                            status=row_status,
                            dataset_id=dataset_id,
                            profile_run=str(profile_run),
                            zarr_path=zarr_path,
                        )
                    )
            except Exception as exc:
                counts["error"] += 1
                print(
                    _format_status(
                        status="error",
                        dataset_id=dataset_id,
                        profile_run=None,
                        zarr_path=zarr_path,
                        reason=str(exc),
                    )
                )

        mode = "apply" if args.apply else "dry-run"
        print(
            "Eye-mask profile registry sync: "
            f"mode={mode} "
            f"datasets={counts['datasets']} "
            f"inserted={counts['inserted']} "
            f"updated={counts['updated']} "
            f"unchanged={counts['unchanged']} "
            f"would_insert={counts['would_insert']} "
            f"would_update={counts['would_update']} "
            f"would_unchanged={counts['would_unchanged']} "
            f"missing_zarr={counts['missing_zarr']} "
            f"missing_profile={counts['missing_profile']} "
            f"errors={counts['error']}"
        )
        return 0 if counts["error"] == 0 else 1
    finally:
        registry.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
