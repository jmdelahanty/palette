"""Report readiness for using acquisition crop videos as ROI pixel providers."""

from __future__ import annotations

from fisheye.shared.json_safety import write_jsonl_atomic as _write_jsonl
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fisheye.registry.db import RegistryPaths
from fisheye.shared.crop_defaults import DEFAULT_ZEBRAFISH_CROP_SIZE_PX


SCHEMA_VERSION = "palette.acquisition_crop_video_roi_readiness_report.v1"
RUN_FAMILIES = (
    "crop_runs",
    "detect_runs",
    "refined_detect_runs",
    "keypoints_runs",
    "refined_keypoints_runs",
    "subject_mask_runs",
    "refined_subject_masks_runs",
)


def _registry_path_from_arg(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()


def _object_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE name = ? AND type IN ('table', 'view')
        LIMIT 1;
        """,
        (name,),
    ).fetchone()
    return row is not None


def _relation_columns(conn: sqlite3.Connection, relation: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({relation});").fetchall()
    return {str(row["name"]) for row in rows}


def _select_expr(columns: set[str], name: str) -> str:
    if name in columns:
        return name
    return f"NULL AS {name}"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _zarr_attrs(path: Path) -> dict[str, Any]:
    payload = _load_json(path / "zarr.json")
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _zarr_node_exists(path: Path) -> bool:
    return (path / "zarr.json").exists()


def _zarr_child_group_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    for child in sorted(path.iterdir(), key=lambda p: p.name):
        if child.is_dir() and (child / "zarr.json").exists():
            names.append(child.name)
    return names


def _infer_recording_dir(zarr_path: Path) -> Optional[Path]:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    if zarr_path.name.endswith(".zarr"):
        return zarr_path.parent
    return None


def _load_recording_manifest(recording_dir: Path) -> dict[str, Any]:
    manifest = _load_json(recording_dir / "recording_manifest.json")
    return manifest if isinstance(manifest, dict) else {}


def _manifest_stream(recording_dir: Path, stream_name: str) -> dict[str, Any]:
    manifest = _load_recording_manifest(recording_dir)
    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, dict):
        return {}
    streams = video_streams.get("streams")
    if not isinstance(streams, dict):
        return {}
    stream = streams.get(stream_name)
    return stream if isinstance(stream, dict) else {}


def _resolve_relative(recording_dir: Path, value: object) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = recording_dir / path
    return path


def _first_existing_or_first(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _resolve_crop_video_path(recording_dir: Path) -> Optional[Path]:
    stream = _manifest_stream(recording_dir, "crop")
    candidates: list[Path] = []
    for key in ("video", "video_path", "path"):
        resolved = _resolve_relative(recording_dir, stream.get(key))
        if resolved is not None:
            candidates.append(resolved)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    candidates.extend(sorted(crop_dir.glob("*_crop_external.mp4")))
    candidates.extend(sorted(crop_dir.glob("*.mp4")))
    return _first_existing_or_first(list(dict.fromkeys(candidates)))


def _resolve_crop_meta_path(recording_dir: Path) -> Optional[Path]:
    stream = _manifest_stream(recording_dir, "crop")
    candidates: list[Path] = []
    for key in ("metadata", "metadata_path", "crop_meta", "crop_meta_path"):
        resolved = _resolve_relative(recording_dir, stream.get(key))
        if resolved is not None:
            candidates.append(resolved)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    candidates.extend(sorted(crop_dir.glob("*_crop_meta.csv")))
    return _first_existing_or_first(list(dict.fromkeys(candidates)))


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_float_nan(value: object) -> float:
    parsed = _safe_float(value)
    return float(parsed) if parsed is not None else float("nan")


def _stream_dimensions_from_manifest(recording_dir: Path) -> tuple[Optional[int], Optional[int]]:
    stream = _manifest_stream(recording_dir, "crop")
    width = _safe_int(stream.get("width")) or _safe_int(stream.get("source_width"))
    height = _safe_int(stream.get("height")) or _safe_int(stream.get("source_height"))
    return width, height


def _ffprobe_dimensions(path: Path) -> tuple[Optional[int], Optional[int], dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,codec_name,pix_fmt,color_range,color_space,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return None, None, {"status": "ffprobe_missing"}
    if result.returncode != 0:
        return None, None, {"status": "ffprobe_failed", "error": (result.stderr or result.stdout).strip()}
    payload = _load_json_from_text(result.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        return None, None, {"status": "ffprobe_no_stream"}
    stream = streams[0]
    return _safe_int(stream.get("width")), _safe_int(stream.get("height")), {
        "status": "ok",
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "nb_frames": stream.get("nb_frames"),
    }


def _load_json_from_text(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _crop_meta_stats(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {
            "crop_meta_status": "missing_path",
            "crop_meta_exists": False,
        }
    if not path.exists():
        return {
            "crop_meta_status": "missing_file",
            "crop_meta_exists": False,
            "crop_meta_path": str(path),
        }
    try:
        rows = _read_crop_meta_rows(path)
    except Exception as exc:
        return {
            "crop_meta_status": "parse_failed",
            "crop_meta_exists": True,
            "crop_meta_path": str(path),
            "crop_meta_error": str(exc),
        }

    n = len(rows)
    has = np.asarray([bool(row["has_detection"]) for row in rows], dtype=bool)
    blank = np.asarray([bool(row["blank_frame"]) for row in rows], dtype=bool)
    crop_xywh = np.asarray([row["crop_xywh"] for row in rows], dtype=np.float64).reshape(-1, 4)
    finite_crop = np.isfinite(crop_xywh).all(axis=1) if crop_xywh.size else np.zeros((n,), dtype=bool)
    positive_crop = (crop_xywh[:, 2] > 0.0) & (crop_xywh[:, 3] > 0.0) if crop_xywh.size else np.zeros((n,), dtype=bool)
    valid_geometry = finite_crop & positive_crop
    usable = has & ~blank & valid_geometry
    widths = crop_xywh[:, 2] if crop_xywh.size else np.asarray([], dtype=np.float64)
    heights = crop_xywh[:, 3] if crop_xywh.size else np.asarray([], dtype=np.float64)
    return {
        "crop_meta_status": "ok",
        "crop_meta_exists": True,
        "crop_meta_path": str(path),
        "crop_meta_rows": n,
        "crop_meta_has_detection_rows": int(np.count_nonzero(has)),
        "crop_meta_blank_rows": int(np.count_nonzero(blank)),
        "crop_meta_no_detection_rows": int(n - np.count_nonzero(has)),
        "crop_meta_invalid_geometry_rows": int(n - np.count_nonzero(valid_geometry)),
        "crop_meta_usable_rows": int(np.count_nonzero(usable)),
        "crop_meta_usable_fraction": (float(np.count_nonzero(usable)) / float(n)) if n else None,
        "crop_meta_crop_width_min": _nan_safe_stat(widths, np.nanmin),
        "crop_meta_crop_width_median": _nan_safe_stat(widths, np.nanmedian),
        "crop_meta_crop_width_max": _nan_safe_stat(widths, np.nanmax),
        "crop_meta_crop_height_min": _nan_safe_stat(heights, np.nanmin),
        "crop_meta_crop_height_median": _nan_safe_stat(heights, np.nanmedian),
        "crop_meta_crop_height_max": _nan_safe_stat(heights, np.nanmax),
    }


def _read_crop_meta_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"recording_frame_id", "crop_x", "crop_y", "crop_w", "crop_h"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Crop metadata missing required columns: {missing}")
        for row_index, row in enumerate(reader):
            rows.append(
                {
                    "row_index": row_index,
                    "recording_frame_index": int(_safe_int(row.get("recording_frame_id")) or 0) - 1,
                    "crop_video_frame_index": int(_safe_int(row.get("crop_video_frame_index")) or row_index),
                    "local_frame_id": int(_safe_int(row.get("local_frame_id")) or row_index),
                    "has_detection": bool(_safe_int(row.get("has_detection")) or 0),
                    "blank_frame": bool(_safe_int(row.get("blank_frame")) or 0),
                    "crop_xywh": (
                        _safe_float_nan(row.get("crop_x")),
                        _safe_float_nan(row.get("crop_y")),
                        _safe_float_nan(row.get("crop_w")),
                        _safe_float_nan(row.get("crop_h")),
                    ),
                }
            )
    return rows


def _nan_safe_stat(values: np.ndarray, fn: Any) -> Optional[float]:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return None
    return float(fn(finite))


def _family_summary(zarr_path: Optional[Path], family: str) -> dict[str, Any]:
    if zarr_path is None:
        return {
            f"{family}_present": False,
            f"{family}_run_count": 0,
            f"{family}_latest": None,
            f"{family}_latest_any": None,
            f"{family}_latest_materialized": None,
        }
    parent = zarr_path / family
    attrs = _zarr_attrs(parent)
    children = _zarr_child_group_names(parent)
    return {
        f"{family}_present": _zarr_node_exists(parent),
        f"{family}_run_count": len(children),
        f"{family}_latest": attrs.get("latest"),
        f"{family}_latest_any": attrs.get("latest_any"),
        f"{family}_latest_materialized": attrs.get("latest_materialized"),
    }


def _analysis_acquisition_stream_summary(analysis_zarr: Optional[Path]) -> dict[str, Any]:
    if analysis_zarr is None:
        return {
            "analysis_acquisition_video_streams_present": False,
            "analysis_acquisition_crop_stream_present": False,
        }
    parent = analysis_zarr / "analysis" / "acquisition_video_streams"
    crop = parent / "streams" / "crop"
    return {
        "analysis_acquisition_video_streams_present": _zarr_node_exists(parent),
        "analysis_acquisition_crop_stream_present": _zarr_node_exists(crop),
        "analysis_acquisition_stream_names": _zarr_child_group_names(parent / "streams"),
    }


def _dataset_rows(
    conn: sqlite3.Connection,
    *,
    path_contains: Optional[str],
    recording_contains: Optional[str],
    dataset_contains: Optional[str],
    zarr_use: str,
    active_only: bool,
) -> list[sqlite3.Row]:
    if not _object_exists(conn, "datasets"):
        raise RuntimeError("registry is missing datasets table")
    columns = _relation_columns(conn, "datasets")
    fields = ["dataset_id", "recording_id", "zarr_path", "zarr_use", "status", "artifact_kind"]
    select_cols = [_select_expr(columns, field) for field in fields]
    sql = f"SELECT {', '.join(select_cols)} FROM datasets"
    clauses: list[str] = []
    params: list[object] = []
    if active_only and "status" in columns:
        clauses.append("status = 'active'")
    if zarr_use != "all" and "zarr_use" in columns:
        clauses.append("zarr_use = ?")
        params.append(zarr_use)
    if path_contains:
        clauses.append("COALESCE(zarr_path, '') LIKE ?")
        params.append(f"%{path_contains}%")
    if recording_contains:
        clauses.append("COALESCE(recording_id, '') LIKE ?")
        params.append(f"%{recording_contains}%")
    if dataset_contains:
        clauses.append("COALESCE(dataset_id, '') LIKE ?")
        params.append(f"%{dataset_contains}%")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY zarr_path, dataset_id"
    return list(conn.execute(sql, params))


def _row_path(row: Mapping[str, Any]) -> Optional[Path]:
    value = row.get("zarr_path")
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _is_analysis_row(row: Mapping[str, Any]) -> bool:
    zarr_use = str(row.get("zarr_use") or "").lower()
    path = str(row.get("zarr_path") or "")
    return zarr_use == "analysis" or path.endswith("_analysis.zarr")


def _is_training_row(row: Mapping[str, Any]) -> bool:
    zarr_use = str(row.get("zarr_use") or "").lower()
    path = str(row.get("zarr_path") or "")
    return zarr_use == "training" or path.endswith("_training.zarr")


def _group_rows_by_recording_dir(rows: Sequence[sqlite3.Row]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        record = dict(row)
        zarr_path = _row_path(record)
        recording_dir = _infer_recording_dir(zarr_path) if zarr_path is not None else None
        key = str(recording_dir) if recording_dir is not None else str(record.get("recording_id") or record.get("dataset_id"))
        groups.setdefault(key, []).append(record)
    return groups


def _pick_row(rows: Sequence[Mapping[str, Any]], predicate: Any) -> Optional[Mapping[str, Any]]:
    matches = [row for row in rows if predicate(row)]
    if not matches:
        return None
    return sorted(matches, key=lambda row: str(row.get("zarr_path") or ""))[0]


def _recommended_action(record: Mapping[str, Any]) -> str:
    if not record.get("analysis_zarr_path"):
        return "missing_analysis_zarr"
    if not record.get("crop_video_exists") or not record.get("crop_meta_exists"):
        return "missing_acquisition_crop_video_inputs"
    if record.get("crop_meta_status") != "ok":
        return "repair_or_parse_crop_metadata"
    if not record.get("crop_video_meets_min_size"):
        return "crop_video_too_small_for_current_policy"
    if int(record.get("analysis_crop_runs_run_count") or 0) == 0:
        return "build_analysis_acquisition_crop_run"
    if int(record.get("analysis_keypoints_runs_run_count") or 0) == 0:
        return "run_analysis_keypoints_from_roi_provider"
    if int(record.get("analysis_subject_mask_runs_run_count") or 0) == 0:
        return "run_analysis_subject_masks_from_roi_provider"
    return "analysis_provider_surfaces_present"


def _build_record(
    group_key: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    min_crop_size: int,
    probe_video: bool,
) -> dict[str, Any]:
    analysis = _pick_row(rows, _is_analysis_row)
    training = _pick_row(rows, _is_training_row)
    analysis_path = _row_path(analysis) if analysis is not None else None
    training_path = _row_path(training) if training is not None else None
    recording_dir = _infer_recording_dir(analysis_path or training_path) if (analysis_path or training_path) is not None else Path(group_key)

    crop_video = _resolve_crop_video_path(recording_dir) if recording_dir is not None else None
    crop_meta = _resolve_crop_meta_path(recording_dir) if recording_dir is not None else None
    manifest_width, manifest_height = _stream_dimensions_from_manifest(recording_dir)
    ffprobe_payload: dict[str, Any] = {"status": "not_requested"}
    ffprobe_width: Optional[int] = None
    ffprobe_height: Optional[int] = None
    if probe_video and crop_video is not None and crop_video.exists():
        ffprobe_width, ffprobe_height, ffprobe_payload = _ffprobe_dimensions(crop_video)
    width = ffprobe_width or manifest_width
    height = ffprobe_height or manifest_height
    meta_stats = _crop_meta_stats(crop_meta)

    record: dict[str, Any] = {
        "record_type": "acquisition_crop_video_roi_readiness",
        "recording_dir": str(recording_dir) if recording_dir is not None else group_key,
        "recording_name": recording_dir.name if recording_dir is not None else None,
        "analysis_dataset_id": analysis.get("dataset_id") if analysis is not None else None,
        "analysis_recording_id": analysis.get("recording_id") if analysis is not None else None,
        "analysis_zarr_path": str(analysis_path) if analysis_path is not None else None,
        "analysis_zarr_exists": bool(analysis_path and analysis_path.exists()),
        "training_dataset_id": training.get("dataset_id") if training is not None else None,
        "training_recording_id": training.get("recording_id") if training is not None else None,
        "training_zarr_path": str(training_path) if training_path is not None else None,
        "training_zarr_exists": bool(training_path and training_path.exists()),
        "dataset_row_count": len(rows),
        "crop_video_path": str(crop_video) if crop_video is not None else None,
        "crop_video_exists": bool(crop_video and crop_video.exists()),
        "crop_meta_path": str(crop_meta) if crop_meta is not None else None,
        "crop_width": width,
        "crop_height": height,
        "crop_video_meets_min_size": bool(width is not None and height is not None and width >= min_crop_size and height >= min_crop_size),
        "min_crop_size": int(min_crop_size),
        "crop_video_probe": ffprobe_payload,
    }
    record.update(meta_stats)
    record.update(_analysis_acquisition_stream_summary(analysis_path))
    for prefix, zarr_path in (("analysis", analysis_path), ("training", training_path)):
        for family in RUN_FAMILIES:
            summary = _family_summary(zarr_path, family)
            for key, value in summary.items():
                record[f"{prefix}_{key}"] = value
    record["offline_detection_surface_present"] = bool(
        int(record.get("analysis_detect_runs_run_count") or 0)
        or int(record.get("analysis_refined_detect_runs_run_count") or 0)
    )
    record["training_review_surfaces_present"] = bool(
        int(record.get("training_crop_runs_run_count") or 0)
        and int(record.get("training_refined_keypoints_runs_run_count") or 0)
        and int(record.get("training_refined_subject_masks_runs_run_count") or 0)
    )
    record["recommended_next_action"] = _recommended_action(record)
    return record


def build_acquisition_crop_video_roi_readiness_report(
    registry_path: Path,
    *,
    path_contains: Optional[str] = None,
    recording_contains: Optional[str] = None,
    dataset_contains: Optional[str] = None,
    zarr_use: str = "all",
    active_only: bool = True,
    min_crop_size: int = DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    probe_video: bool = False,
) -> dict[str, Any]:
    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = _dataset_rows(
            conn,
            path_contains=path_contains,
            recording_contains=recording_contains,
            dataset_contains=dataset_contains,
            zarr_use=zarr_use,
            active_only=active_only,
        )
    finally:
        conn.close()
    grouped = _group_rows_by_recording_dir(rows)
    records = [
        _build_record(key, group_rows, min_crop_size=min_crop_size, probe_video=probe_video)
        for key, group_rows in sorted(grouped.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_path": str(registry_path),
        "filters": {
            "path_contains": path_contains,
            "recording_contains": recording_contains,
            "dataset_contains": dataset_contains,
            "zarr_use": zarr_use,
            "active_only": active_only,
            "min_crop_size": int(min_crop_size),
            "probe_video": bool(probe_video),
        },
        "dataset_row_count": len(rows),
        "recording_count": len(records),
        "action_counts": dict(sorted(Counter(str(row["recommended_next_action"]) for row in records).items())),
        "crop_meta_status_counts": dict(sorted(Counter(str(row.get("crop_meta_status")) for row in records).items())),
        "records": records,
    }


def _format_counts(counts: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) if counts else "{}"


def print_text_report(report: Mapping[str, Any], *, limit: int = 40) -> None:
    print("acquisition_crop_video_roi_readiness_report")
    print(f"registry: {report['registry_path']}")
    print(f"filters: {report['filters']}")
    print(f"dataset_row_count: {report['dataset_row_count']}")
    print(f"recording_count: {report['recording_count']}")
    print(f"action_counts: {_format_counts(report['action_counts'])}")
    print(f"crop_meta_status_counts: {_format_counts(report['crop_meta_status_counts'])}")
    records = report.get("records")
    if isinstance(records, list) and records:
        print()
        print("sample_recordings:")
        for row in records[: max(int(limit), 0)]:
            if not isinstance(row, Mapping):
                continue
            print(
                "  "
                f"{row.get('recommended_next_action', '-'):<44} "
                f"{row.get('recording_name', '-') or '-'} "
                f"crop={row.get('crop_width')}x{row.get('crop_height')} "
                f"usable={row.get('crop_meta_usable_rows')} "
                f"analysis_crop_runs={row.get('analysis_crop_runs_run_count')} "
                f"training_crop_runs={row.get('training_crop_runs_run_count')}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only readiness report for using Orange/acquisition crop videos "
            "as ROI pixel providers for keypoints and subject masks."
        )
    )
    parser.add_argument("--registry", help="Registry SQLite path. Defaults to RegistryPaths.from_env(Path.cwd()).")
    parser.add_argument("--path-contains", help="Filter dataset zarr_path by substring.")
    parser.add_argument("--recording-contains", help="Filter dataset recording_id by substring.")
    parser.add_argument("--dataset-contains", help="Filter dataset_id by substring.")
    parser.add_argument(
        "--zarr-use",
        choices=("all", "analysis", "training"),
        default="all",
        help="Dataset zarr_use filter. Default pairs all analysis/training rows by recording dir.",
    )
    parser.add_argument("--include-inactive", action="store_true", help="Include non-active dataset rows.")
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
        help="Minimum crop video width/height considered usable for current RedScare-style models.",
    )
    parser.add_argument(
        "--probe-video",
        action="store_true",
        help="Run ffprobe on crop videos for dimensions/codec. Default uses recording_manifest metadata.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--output-json", type=Path, help="Write full JSON report to this path.")
    parser.add_argument("--output-jsonl", type=Path, help="Write one readiness row per recording.")
    parser.add_argument("--limit", type=int, default=40, help="Maximum sample rows in text mode.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_acquisition_crop_video_roi_readiness_report(
        _registry_path_from_arg(args.registry),
        path_contains=args.path_contains,
        recording_contains=args.recording_contains,
        dataset_contains=args.dataset_contains,
        zarr_use=args.zarr_use,
        active_only=not bool(args.include_inactive),
        min_crop_size=int(args.min_crop_size),
        probe_video=bool(args.probe_video),
    )
    if args.output_json:
        args.output_json.expanduser().resolve().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl.expanduser().resolve(), report["records"])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report, limit=int(args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
