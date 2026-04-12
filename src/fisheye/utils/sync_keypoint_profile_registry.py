#!/usr/bin/env python3
"""Sync latest keypoint profile summaries from Zarr into registry rows."""

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


def _to_json_text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _to_json_text(dict(value))
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
        return text or None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _to_kpt_shape_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        normalized: list[Any] = []
        for item in value:
            parsed = _as_int(item)
            normalized.append(int(parsed) if parsed is not None else item)
        return json.dumps(normalized, separators=(",", ":"), default=str)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
    return _normalize_text(value)


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
    runs_parent = _get_group(analysis, "keypoint_profile_runs")
    if runs_parent is None:
        return None, None, "analysis/keypoint_profile_runs missing"
    run_name = _select_latest_profile_run(runs_parent)
    if run_name is None:
        return None, None, "analysis/keypoint_profile_runs has no runs"
    run_group = _get_group(runs_parent, run_name)
    if run_group is None:
        return None, None, f"profile run missing: {run_name}"
    summary = _coerce_mapping(run_group.attrs.get("profile_summary"))
    if summary is None:
        return run_name, None, f"profile_summary missing or invalid on run: {run_name}"
    return run_name, summary, None


def _load_profile_run_attrs(root: zarr.Group, profile_run: Optional[str]) -> dict[str, Any]:
    if profile_run is None:
        return {}
    try:
        analysis = _get_group(root, "analysis")
        if analysis is None:
            return {}
        runs_parent = _get_group(analysis, "keypoint_profile_runs")
        if runs_parent is None:
            return {}
        run_group = _get_group(runs_parent, str(profile_run))
        if run_group is None:
            return {}
        return dict(run_group.attrs)
    except Exception:
        return {}


def _geometry_percentile(
    geometry: Mapping[str, Any],
    *,
    metric_name: str,
    percentile_key: str,
) -> Optional[float]:
    metric_payload = geometry.get(metric_name)
    if not isinstance(metric_payload, Mapping):
        return None
    stats_payload = metric_payload.get("stats")
    if not isinstance(stats_payload, Mapping):
        return None
    return _as_float(stats_payload.get(percentile_key))


def _build_profile_payload(
    *,
    dataset_id: str,
    fallback_recording_id: Optional[str],
    fallback_zarr_use: Optional[str],
    fallback_genotype: Optional[str],
    fallback_dpf_at_acquisition: Optional[int],
    profile_run: str,
    summary: Mapping[str, Any],
    run_attrs: Optional[Mapping[str, Any]],
    zarr_path: Path,
) -> dict[str, Any]:
    dataset = summary.get("dataset")
    source = summary.get("source")
    quality = summary.get("quality")
    geometry = summary.get("geometry")
    composition = summary.get("composition")

    dataset_map = dict(dataset) if isinstance(dataset, Mapping) else {}
    source_map = dict(source) if isinstance(source, Mapping) else {}
    quality_map = dict(quality) if isinstance(quality, Mapping) else {}
    geometry_map = dict(geometry) if isinstance(geometry, Mapping) else {}
    composition_map = dict(composition) if isinstance(composition, Mapping) else {}
    run_attrs_map = dict(run_attrs) if isinstance(run_attrs, Mapping) else {}

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except OSError:
        zarr_mtime_ns = None

    dpf_at_acquisition = _as_int(composition_map.get("dpf_at_acquisition"))
    if dpf_at_acquisition is None:
        dpf_at_acquisition = fallback_dpf_at_acquisition

    pose_schema_map = _coerce_mapping(
        run_attrs_map.get("source_pose_schema")
        if run_attrs_map.get("source_pose_schema") is not None
        else source_map.get("pose_schema")
    )
    heading_computation_map = _coerce_mapping(
        run_attrs_map.get("source_heading_computation")
        if run_attrs_map.get("source_heading_computation") is not None
        else source_map.get("heading_computation")
    )

    return {
        "dataset_id": dataset_id,
        "profile_run": profile_run,
        "recording_id": (
            _normalize_text(run_attrs_map.get("source_recording_id"))
            or _normalize_text(dataset_map.get("recording_id"))
            or fallback_recording_id
        ),
        "zarr_use": (
            _normalize_text(run_attrs_map.get("source_zarr_use"))
            or _normalize_text(dataset_map.get("zarr_use"))
            or fallback_zarr_use
        ),
        "keypoint_method": (
            _normalize_text(run_attrs_map.get("source_keypoint_method"))
            or _normalize_text(source_map.get("keypoint_method"))
        ),
        "source_keypoint_path": (
            _normalize_text(run_attrs_map.get("source_keypoint_path"))
            or _normalize_text(source_map.get("keypoint_path"))
        ),
        "source_keypoint_run": (
            _normalize_text(run_attrs_map.get("source_keypoint_run"))
            or _normalize_text(source_map.get("keypoint_run"))
        ),
        "skeleton_id": (
            _normalize_text(run_attrs_map.get("source_skeleton_id"))
            or _normalize_text(source_map.get("skeleton_id"))
        ),
        "kpt_shape": _to_kpt_shape_text(
            run_attrs_map.get("source_kpt_shape")
            if run_attrs_map.get("source_kpt_shape") is not None
            else source_map.get("kpt_shape")
        ),
        "pose_schema_name": (
            _normalize_text(run_attrs_map.get("source_pose_schema_name"))
            or _normalize_text(source_map.get("pose_schema_name"))
            or (_normalize_text(pose_schema_map.get("name")) if pose_schema_map is not None else None)
        ),
        "pose_schema_json": _to_json_text_or_none(pose_schema_map),
        "heading_computation_source": (
            _normalize_text(run_attrs_map.get("source_heading_computation_source"))
            or _normalize_text(source_map.get("heading_computation_source"))
        ),
        "heading_computation_json": _to_json_text_or_none(heading_computation_map),
        "profile_created_utc": (
            _normalize_text(run_attrs_map.get("created_at_utc"))
            or _normalize_text(summary.get("created_at_utc"))
        ),
        "rows_total": _as_int(quality_map.get("rows_total")),
        "rows_usable": _as_int(quality_map.get("rows_usable")),
        "usable_keypoints_total": _as_int(quality_map.get("usable_keypoints_total")),
        "usable_rate": _as_float(quality_map.get("usable_rate")),
        "confidence_valid_rate": _as_float(quality_map.get("confidence_valid_rate")),
        "geometry_valid_rate": _as_float(quality_map.get("geometry_valid_rate")),
        "triangle_area_p10": _geometry_percentile(
            geometry_map,
            metric_name="triangle_area",
            percentile_key="p10",
        ),
        "triangle_area_p50": _geometry_percentile(
            geometry_map,
            metric_name="triangle_area",
            percentile_key="p50",
        ),
        "triangle_area_p90": _geometry_percentile(
            geometry_map,
            metric_name="triangle_area",
            percentile_key="p90",
        ),
        "min_angle_p10": _geometry_percentile(
            geometry_map,
            metric_name="min_angle",
            percentile_key="p10",
        ),
        "min_angle_p50": _geometry_percentile(
            geometry_map,
            metric_name="min_angle",
            percentile_key="p50",
        ),
        "min_angle_p90": _geometry_percentile(
            geometry_map,
            metric_name="min_angle",
            percentile_key="p90",
        ),
        "heading_p10": _geometry_percentile(
            geometry_map,
            metric_name="heading",
            percentile_key="p10",
        ),
        "heading_p50": _geometry_percentile(
            geometry_map,
            metric_name="heading",
            percentile_key="p50",
        ),
        "heading_p90": _geometry_percentile(
            geometry_map,
            metric_name="heading",
            percentile_key="p90",
        ),
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
        description="Sync latest keypoint profile summaries from Zarr into keypoint_data_profile registry rows."
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
            "updated": 0,
            "would_upsert": 0,
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
                    run_attrs=_load_profile_run_attrs(root, profile_run),
                    zarr_path=zarr_path,
                )
                if args.apply:
                    registry.upsert_keypoint_data_profile(
                        dataset_id=str(payload["dataset_id"]),
                        profile_run=str(payload["profile_run"]),
                        recording_id=_normalize_text(payload.get("recording_id")),
                        zarr_use=_normalize_text(payload.get("zarr_use")),
                        keypoint_method=_normalize_text(payload.get("keypoint_method")),
                        source_keypoint_path=_normalize_text(payload.get("source_keypoint_path")),
                        source_keypoint_run=_normalize_text(payload.get("source_keypoint_run")),
                        skeleton_id=_normalize_text(payload.get("skeleton_id")),
                        kpt_shape=_normalize_text(payload.get("kpt_shape")),
                        profile_created_utc=_normalize_text(payload.get("profile_created_utc")),
                        rows_total=_as_int(payload.get("rows_total")),
                        rows_usable=_as_int(payload.get("rows_usable")),
                        usable_keypoints_total=_as_int(payload.get("usable_keypoints_total")),
                        usable_rate=_as_float(payload.get("usable_rate")),
                        confidence_valid_rate=_as_float(payload.get("confidence_valid_rate")),
                        geometry_valid_rate=_as_float(payload.get("geometry_valid_rate")),
                        triangle_area_p10=_as_float(payload.get("triangle_area_p10")),
                        triangle_area_p50=_as_float(payload.get("triangle_area_p50")),
                        triangle_area_p90=_as_float(payload.get("triangle_area_p90")),
                        min_angle_p10=_as_float(payload.get("min_angle_p10")),
                        min_angle_p50=_as_float(payload.get("min_angle_p50")),
                        min_angle_p90=_as_float(payload.get("min_angle_p90")),
                        heading_p10=_as_float(payload.get("heading_p10")),
                        heading_p50=_as_float(payload.get("heading_p50")),
                        heading_p90=_as_float(payload.get("heading_p90")),
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
                        pose_schema_name=_normalize_text(payload.get("pose_schema_name")),
                        pose_schema_json=_normalize_text(payload.get("pose_schema_json")),
                        heading_computation_source=_normalize_text(payload.get("heading_computation_source")),
                        heading_computation_json=_normalize_text(payload.get("heading_computation_json")),
                    )
                    counts["updated"] += 1
                    print(
                        _format_status(
                            status="updated",
                            dataset_id=dataset_id,
                            profile_run=str(profile_run),
                            zarr_path=zarr_path,
                        )
                    )
                else:
                    counts["would_upsert"] += 1
                    print(
                        _format_status(
                            status="would_upsert",
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
            "Keypoint profile registry sync: "
            f"mode={mode} "
            f"datasets={counts['datasets']} "
            f"updated={counts['updated']} "
            f"would_upsert={counts['would_upsert']} "
            f"missing_zarr={counts['missing_zarr']} "
            f"missing_profile={counts['missing_profile']} "
            f"errors={counts['error']}"
        )
        return 0 if counts["error"] == 0 else 1
    finally:
        registry.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
