#!/usr/bin/env python3
"""Generate a static HTML index for source-recording profile runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any, Iterable, Optional

from fisheye.registry.db import RegistryPaths

try:  # pragma: no cover - import fallback is environment-specific
    import numpy as np
    import zarr
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    zarr = None  # type: ignore[assignment]


@dataclass
class ProfileRow:
    profile_run: str
    profile_created_utc: Optional[str]
    updated_utc: Optional[str]
    metric_values: tuple[Any, ...]
    is_latest: bool = False
    artifact_png_paths: list[Path] = field(default_factory=list)


@dataclass
class SourceRecordingProfileEntry:
    dataset_id: str
    recording_id: Optional[str]
    zarr_use: Optional[str]
    zarr_path: str
    status: Optional[str]
    detection_profiles: list[ProfileRow] = field(default_factory=list)
    keypoint_profiles: list[ProfileRow] = field(default_factory=list)
    eye_mask_profiles: list[ProfileRow] = field(default_factory=list)


DETECTION_METRIC_HEADERS: tuple[str, ...] = (
    "frames_total",
    "frames_with_detections",
    "detections_total",
    "coverage_percent",
)
KEYPOINT_METRIC_HEADERS: tuple[str, ...] = (
    "rows_total",
    "rows_usable",
    "usable_rate",
)
EYE_MASK_METRIC_HEADERS: tuple[str, ...] = (
    "rows_total",
    "rows_usable",
    "usable_rate",
    "pair_success_rate",
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DETECTION_REFINED_PARENT_NAMES: tuple[str, ...] = ("refined_detect_runs", "refined_runs")
KEYPOINT_REFINED_PARENT_NAMES: tuple[str, ...] = ("refined_keypoints_runs", "keypoints_refined_runs")


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_mapping(value: object) -> Optional[dict[str, object]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    if hasattr(value, "items"):
        try:
            return dict(value)  # type: ignore[arg-type]
        except Exception:
            return None
    return None


def _read_zarr_attrs(zarr_json_path: Path) -> dict[str, object]:
    if not zarr_json_path.is_file():
        return {}
    try:
        payload = json.loads(zarr_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _table_or_view_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1;",
        (name,),
    ).fetchone()
    return row is not None


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "-"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._-") or "value"


def _href_for_path(path: Path, *, output_dir: Path) -> str:
    rel = os.path.relpath(str(path), str(output_dir))
    return rel.replace(os.sep, "/")


def _fetch_latest_runs(conn: sqlite3.Connection, latest_view: str) -> dict[str, str]:
    if not _table_or_view_exists(conn, latest_view):
        return {}
    rows = conn.execute(f"SELECT dataset_id, profile_run FROM {latest_view};").fetchall()
    return {str(row[0]): str(row[1]) for row in rows if row[0] is not None and row[1] is not None}


def _query_source_datasets(
    conn: sqlite3.Connection,
    *,
    zarr_use_filter: str,
) -> list[SourceRecordingProfileEntry]:
    sql = """
        SELECT dataset_id, recording_id, zarr_use, zarr_path, status
        FROM datasets
        WHERE artifact_kind = 'source_recording'
    """
    params: list[Any] = []
    if zarr_use_filter in {"training", "analysis"}:
        sql += " AND zarr_use = ?"
        params.append(zarr_use_filter)
    sql += " ORDER BY COALESCE(recording_id, ''), COALESCE(zarr_use, ''), dataset_id;"
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [
        SourceRecordingProfileEntry(
            dataset_id=str(row[0]),
            recording_id=_normalize_text(row[1]),
            zarr_use=_normalize_text(row[2]),
            zarr_path=str(row[3]),
            status=_normalize_text(row[4]),
        )
        for row in rows
    ]


def _append_profile_rows(
    conn: sqlite3.Connection,
    *,
    profile_table: str,
    latest_view: str,
    metric_columns: Iterable[str],
    out_map: dict[str, list[ProfileRow]],
    zarr_use_filter: str,
) -> None:
    if not _table_or_view_exists(conn, profile_table):
        return
    latest_by_dataset = _fetch_latest_runs(conn, latest_view)

    metrics_sql = ", ".join(metric_columns)
    sql = f"""
        SELECT p.dataset_id, p.profile_run, p.profile_created_utc, p.updated_utc, {metrics_sql}
        FROM {profile_table} AS p
        JOIN datasets AS d ON d.dataset_id = p.dataset_id
        WHERE d.artifact_kind = 'source_recording'
    """
    params: list[Any] = []
    if zarr_use_filter in {"training", "analysis"}:
        sql += " AND d.zarr_use = ?"
        params.append(zarr_use_filter)
    sql += " ORDER BY p.dataset_id, COALESCE(p.profile_created_utc, ''), p.profile_run;"

    rows = conn.execute(sql, tuple(params)).fetchall()
    for row in rows:
        dataset_id = str(row[0])
        profile_run = str(row[1])
        entry = ProfileRow(
            profile_run=profile_run,
            profile_created_utc=_normalize_text(row[2]),
            updated_utc=_normalize_text(row[3]),
            metric_values=tuple(row[idx] for idx in range(4, len(row))),
            is_latest=(latest_by_dataset.get(dataset_id) == profile_run),
        )
        out_map.setdefault(dataset_id, []).append(entry)


def _collect_png_artifact_names(visualizations_dir: Path) -> list[str]:
    artifact_names: list[str] = []
    if not visualizations_dir.is_dir():
        return artifact_names
    for child in sorted(visualizations_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "zarr.json").is_file():
            continue
        if not child.name.endswith("_png"):
            continue
        artifact_names.append(child.name)
    return artifact_names


def _split_path_parts(value: object) -> list[str]:
    text = _normalize_text(value)
    if text is None:
        return []
    return [part for part in text.strip("/").split("/") if part]


def _add_artifact_target(
    targets: list[tuple[str, str]],
    seen: set[tuple[str, str]],
    *,
    parent_path: str,
    run_name: Optional[str],
) -> None:
    resolved_run = _normalize_text(run_name)
    if resolved_run is None:
        return
    key = (parent_path.strip("/"), resolved_run)
    if key in seen:
        return
    seen.add(key)
    targets.append(key)


def _resolve_artifact_targets_for_profile_row(
    *,
    zarr_path: Path,
    parent_name: str,
    row: ProfileRow,
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    _add_artifact_target(
        targets,
        seen,
        parent_path=f"analysis/{parent_name}",
        run_name=row.profile_run,
    )
    if parent_name not in {"detection_profile_runs", "keypoint_profile_runs"}:
        return targets

    run_attrs = _read_zarr_attrs(zarr_path / "analysis" / parent_name / row.profile_run / "zarr.json")
    profile_summary = _coerce_mapping(run_attrs.get("profile_summary")) or {}
    source = _coerce_mapping(profile_summary.get("source")) or {}

    if parent_name == "detection_profile_runs":
        refined_run = _normalize_text(source.get("refined_run"))
        for refined_parent in DETECTION_REFINED_PARENT_NAMES:
            _add_artifact_target(
                targets,
                seen,
                parent_path=refined_parent,
                run_name=refined_run,
            )
        detection_path_parts = _split_path_parts(source.get("detection_path"))
        if len(detection_path_parts) >= 2 and detection_path_parts[0] in DETECTION_REFINED_PARENT_NAMES:
            _add_artifact_target(
                targets,
                seen,
                parent_path=detection_path_parts[0],
                run_name=detection_path_parts[1],
            )
    else:
        refined_run = _normalize_text(source.get("refined_run"))
        for refined_parent in KEYPOINT_REFINED_PARENT_NAMES:
            _add_artifact_target(
                targets,
                seen,
                parent_path=refined_parent,
                run_name=refined_run,
            )
        keypoint_path_parts = _split_path_parts(source.get("keypoint_path"))
        if len(keypoint_path_parts) >= 2 and keypoint_path_parts[0] in KEYPOINT_REFINED_PARENT_NAMES:
            _add_artifact_target(
                targets,
                seen,
                parent_path=keypoint_path_parts[0],
                run_name=keypoint_path_parts[1],
            )
    return targets


def _row_has_candidate_artifacts(
    *,
    zarr_path: Path,
    parent_name: str,
    row: ProfileRow,
) -> bool:
    targets = _resolve_artifact_targets_for_profile_row(
        zarr_path=zarr_path,
        parent_name=parent_name,
        row=row,
    )
    for parent_path, run_name in targets:
        visualizations_dir = zarr_path / parent_path / run_name / "visualizations"
        if _collect_png_artifact_names(visualizations_dir):
            return True
    return False


def _extract_profile_row_artifacts(
    *,
    root: Any,
    zarr_path: Path,
    parent_name: str,
    dataset_slug: str,
    row: ProfileRow,
    artifacts_dir: Path,
    overwrite: bool,
) -> int:
    targets = _resolve_artifact_targets_for_profile_row(
        zarr_path=zarr_path,
        parent_name=parent_name,
        row=row,
    )
    wrote = 0
    seen_arrays: set[str] = set()
    for source_parent_path, source_run in targets:
        visualizations_dir = zarr_path / source_parent_path / source_run / "visualizations"
        artifact_names = _collect_png_artifact_names(visualizations_dir)
        if not artifact_names:
            continue

        source_parent_slug = _safe_slug(source_parent_path.replace("/", "__"))
        source_run_slug = _safe_slug(source_run)
        for artifact_name in artifact_names:
            array_path = f"{source_parent_path}/{source_run}/visualizations/{artifact_name}"
            if array_path in seen_arrays:
                continue
            seen_arrays.add(array_path)

            artifact_slug = _safe_slug(artifact_name)
            out_name = f"{dataset_slug}__{source_parent_slug}__{source_run_slug}__{artifact_slug}.png"
            out_path = artifacts_dir / out_name
            if out_path.exists() and not overwrite:
                if out_path not in row.artifact_png_paths:
                    row.artifact_png_paths.append(out_path)
                continue

            try:
                data = np.asarray(root[array_path][:], dtype=np.uint8).tobytes()  # type: ignore[union-attr]
            except Exception:
                continue
            if not data.startswith(PNG_SIGNATURE):
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            if out_path not in row.artifact_png_paths:
                row.artifact_png_paths.append(out_path)
            wrote += 1
    return wrote


def enrich_entries_with_profile_artifacts(
    entries: list[SourceRecordingProfileEntry],
    *,
    artifacts_dir: Path,
    overwrite: bool,
) -> int:
    if zarr is None or np is None:
        return 0

    total_written = 0
    for entry in entries:
        zarr_path = Path(entry.zarr_path)
        if not zarr_path.is_dir():
            continue
        dataset_slug = _safe_slug(entry.dataset_id)
        needs_read = any(
            (
                _row_has_candidate_artifacts(
                    zarr_path=zarr_path,
                    parent_name="detection_profile_runs",
                    row=row,
                )
                for row in entry.detection_profiles
            )
        ) or any(
            (
                _row_has_candidate_artifacts(
                    zarr_path=zarr_path,
                    parent_name="keypoint_profile_runs",
                    row=row,
                )
                for row in entry.keypoint_profiles
            )
        ) or any(
            (
                _row_has_candidate_artifacts(
                    zarr_path=zarr_path,
                    parent_name="eye_mask_profile_runs",
                    row=row,
                )
                for row in entry.eye_mask_profiles
            )
        )
        if not needs_read:
            continue

        try:
            root = zarr.open_group(str(zarr_path), mode="r")
        except Exception:
            continue

        for row in entry.detection_profiles:
            total_written += _extract_profile_row_artifacts(
                root=root,
                zarr_path=zarr_path,
                parent_name="detection_profile_runs",
                dataset_slug=dataset_slug,
                row=row,
                artifacts_dir=artifacts_dir,
                overwrite=overwrite,
            )
        for row in entry.keypoint_profiles:
            total_written += _extract_profile_row_artifacts(
                root=root,
                zarr_path=zarr_path,
                parent_name="keypoint_profile_runs",
                dataset_slug=dataset_slug,
                row=row,
                artifacts_dir=artifacts_dir,
                overwrite=overwrite,
            )
        for row in entry.eye_mask_profiles:
            total_written += _extract_profile_row_artifacts(
                root=root,
                zarr_path=zarr_path,
                parent_name="eye_mask_profile_runs",
                dataset_slug=dataset_slug,
                row=row,
                artifacts_dir=artifacts_dir,
                overwrite=overwrite,
            )
    return total_written


def collect_source_recording_profile_entries(
    *,
    registry_path: Path,
    zarr_use_filter: str = "all",
) -> list[SourceRecordingProfileEntry]:
    conn = sqlite3.connect(str(registry_path))
    try:
        source_entries = _query_source_datasets(conn, zarr_use_filter=zarr_use_filter)
        by_dataset: dict[str, SourceRecordingProfileEntry] = {entry.dataset_id: entry for entry in source_entries}

        detect_rows: dict[str, list[ProfileRow]] = {}
        keypoint_rows: dict[str, list[ProfileRow]] = {}
        eye_mask_rows: dict[str, list[ProfileRow]] = {}

        _append_profile_rows(
            conn,
            profile_table="detection_data_profile",
            latest_view="detection_data_profile_latest",
            metric_columns=DETECTION_METRIC_HEADERS,
            out_map=detect_rows,
            zarr_use_filter=zarr_use_filter,
        )
        _append_profile_rows(
            conn,
            profile_table="keypoint_data_profile",
            latest_view="keypoint_data_profile_latest",
            metric_columns=KEYPOINT_METRIC_HEADERS,
            out_map=keypoint_rows,
            zarr_use_filter=zarr_use_filter,
        )
        _append_profile_rows(
            conn,
            profile_table="eye_mask_data_profile",
            latest_view="eye_mask_data_profile_latest",
            metric_columns=EYE_MASK_METRIC_HEADERS,
            out_map=eye_mask_rows,
            zarr_use_filter=zarr_use_filter,
        )

        for dataset_id, entry in by_dataset.items():
            entry.detection_profiles = detect_rows.get(dataset_id, [])
            entry.keypoint_profiles = keypoint_rows.get(dataset_id, [])
            entry.eye_mask_profiles = eye_mask_rows.get(dataset_id, [])

        return list(by_dataset.values())
    finally:
        conn.close()


def _render_profile_table(
    *,
    rows: list[ProfileRow],
    metric_headers: tuple[str, ...],
    output_html: Path,
    thumb_width: int,
) -> str:
    if not rows:
        return "<div class='empty'>No profile rows.</div>"
    output_dir = output_html.parent
    include_artifacts = any(bool(row.artifact_png_paths) for row in rows)
    parts: list[str] = []
    parts.append("<table>")
    parts.append("  <thead><tr><th>latest</th><th>profile_run</th><th>created</th><th>updated</th>")
    for header in metric_headers:
        parts.append(f"<th>{escape(header)}</th>")
    if include_artifacts:
        parts.append("<th>artifacts</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in rows:
        latest = "yes" if row.is_latest else ""
        parts.append("<tr>")
        parts.append(f"<td>{escape(latest)}</td>")
        parts.append(f"<td><code>{escape(row.profile_run)}</code></td>")
        parts.append(f"<td>{escape(row.profile_created_utc or '-')}</td>")
        parts.append(f"<td>{escape(row.updated_utc or '-')}</td>")
        for value in row.metric_values:
            parts.append(f"<td>{escape(_format_metric(value))}</td>")
        if include_artifacts:
            if row.artifact_png_paths:
                parts.append("<td><div class='artifact-strip'>")
                for artifact_path in row.artifact_png_paths:
                    href = _href_for_path(artifact_path, output_dir=output_dir)
                    name = artifact_path.name
                    parts.append(
                        f"<a href='{escape(href)}' target='_blank' rel='noopener'>"
                        f"<img src='{escape(href)}' alt='{escape(name)}' loading='lazy' width='{int(thumb_width)}' /></a>"
                    )
                parts.append("</div></td>")
            else:
                parts.append("<td>-</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_source_recording_profile_index_html(
    *,
    entries: list[SourceRecordingProfileEntry],
    registry_path: Path,
    zarr_use_filter: str,
    output_html: Path,
    title: str,
    thumb_width: int,
) -> str:
    total = len(entries)
    with_detect = sum(1 for entry in entries if entry.detection_profiles)
    with_keypoint = sum(1 for entry in entries if entry.keypoint_profiles)
    with_eye_mask = sum(1 for entry in entries if entry.eye_mask_profiles)
    generated_utc = datetime.now(timezone.utc).isoformat()

    parts: list[str] = []
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8' />")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1' />")
    parts.append(f"<title>{escape(title)}</title>")
    parts.append("<style>")
    parts.append(":root{--bg:#f6f7fb;--fg:#111827;--muted:#6b7280;--line:#d1d5db;--card:#fff;}")
    parts.append("body{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}")
    parts.append("main{max-width:1500px;margin:0 auto;padding:18px;}")
    parts.append("h1{margin:0 0 8px;font-size:1.5rem;} .sub{color:var(--muted);margin:0 0 12px;font-size:.92rem;}")
    parts.append(".toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0 16px;}")
    parts.append(".toolbar input{min-width:320px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;}")
    parts.append(".pill{border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 10px;color:var(--muted);font-size:.85rem;}")
    parts.append(".filters{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:8px 0 14px;}")
    parts.append(".filter-group{display:flex;gap:6px;flex-wrap:wrap;align-items:center;}")
    parts.append(".filter-label{font-size:.83rem;color:var(--muted);margin-right:2px;}")
    parts.append(".filter-chip{border:1px solid var(--line);background:#fff;color:var(--fg);border-radius:999px;padding:4px 10px;font-size:.82rem;cursor:pointer;}")
    parts.append(".filter-chip.active{background:#111827;color:#fff;border-color:#111827;}")
    parts.append("details.ds{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:10px 0;}")
    parts.append("details.ds>summary{cursor:pointer;padding:10px 12px;font-weight:600;}")
    parts.append("details.ds[open]>summary{border-bottom:1px solid var(--line);}")
    parts.append(".meta{padding:10px 12px;display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:.9rem;}")
    parts.append(".meta code{color:var(--fg);} .sections{padding:0 12px 12px;}")
    parts.append(".sec{margin:8px 0 12px;} .sec h4{margin:0 0 6px;font-size:.95rem;}")
    parts.append("table{border-collapse:collapse;width:100%;font-size:.86rem;background:#fff;}")
    parts.append("th,td{border:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top;}")
    parts.append("th{background:#f9fafb;} .empty{color:var(--muted);font-size:.88rem;}")
    parts.append(".hidden{display:none !important;}")
    parts.append(".artifact-strip{display:flex;gap:8px;flex-wrap:wrap;}")
    parts.append(".artifact-strip a{display:inline-block;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#fff;}")
    parts.append(".artifact-strip img{display:block;}")
    parts.append("</style></head><body><main>")
    parts.append(f"<h1>{escape(title)}</h1>")
    parts.append(
        "<p class='sub'>"
        f"registry: <code>{escape(str(registry_path))}</code> | "
        f"zarr_use filter: <code>{escape(zarr_use_filter)}</code> | "
        f"generated_utc: <code>{escape(generated_utc)}</code> | "
        f"output: <code>{escape(str(output_html))}</code>"
        "</p>"
    )
    parts.append("<div class='toolbar'>")
    parts.append("<input id='q' type='search' placeholder='Filter by dataset_id, recording_id, zarr path, run id...' />")
    parts.append(f"<span class='pill'>datasets: {total}</span>")
    parts.append(f"<span class='pill' id='visible-count'>visible: {total}</span>")
    parts.append(f"<span class='pill'>with detect profiles: {with_detect}</span>")
    parts.append(f"<span class='pill'>with keypoint profiles: {with_keypoint}</span>")
    parts.append(f"<span class='pill'>with eye-mask profiles: {with_eye_mask}</span>")
    parts.append("</div>")
    parts.append("<div class='filters'>")
    parts.append("<div class='filter-group'>")
    parts.append("<span class='filter-label'>Profile:</span>")
    parts.append("<button type='button' class='filter-chip active' data-filter-kind='profile' data-value='all'>all</button>")
    parts.append("<button type='button' class='filter-chip' data-filter-kind='profile' data-value='detect'>detect</button>")
    parts.append("<button type='button' class='filter-chip' data-filter-kind='profile' data-value='keypoint'>keypoint</button>")
    parts.append("<button type='button' class='filter-chip' data-filter-kind='profile' data-value='eye_mask'>eye-mask</button>")
    parts.append("</div>")
    parts.append("<div class='filter-group'>")
    parts.append("<span class='filter-label'>Zarr use:</span>")
    parts.append("<button type='button' class='filter-chip active' data-filter-kind='zarr-use' data-value='all'>all</button>")
    parts.append("<button type='button' class='filter-chip' data-filter-kind='zarr-use' data-value='training'>training</button>")
    parts.append("<button type='button' class='filter-chip' data-filter-kind='zarr-use' data-value='analysis'>analysis</button>")
    parts.append("<button type='button' class='filter-chip' data-filter-kind='zarr-use' data-value='unknown'>unknown</button>")
    parts.append("</div>")
    parts.append("</div>")

    for entry in entries:
        entry_zarr_use = (entry.zarr_use or "unknown").strip().lower() or "unknown"
        has_detect = "1" if entry.detection_profiles else "0"
        has_keypoint = "1" if entry.keypoint_profiles else "0"
        has_eye_mask = "1" if entry.eye_mask_profiles else "0"
        search_blob = " ".join(
            [
                entry.dataset_id,
                entry.recording_id or "",
                entry.zarr_use or "",
                entry.zarr_path,
                " ".join(row.profile_run for row in entry.detection_profiles),
                " ".join(row.profile_run for row in entry.keypoint_profiles),
                " ".join(row.profile_run for row in entry.eye_mask_profiles),
            ]
        ).lower()
        parts.append(
            f"<details class='ds' open "
            f"data-search='{escape(search_blob)}' "
            f"data-zarr-use='{escape(entry_zarr_use)}' "
            f"data-has-detect='{has_detect}' "
            f"data-has-keypoint='{has_keypoint}' "
            f"data-has-eye-mask='{has_eye_mask}'>"
            f"<summary><code>{escape(entry.dataset_id)}</code> "
            f"| rec={escape(entry.recording_id or '-')} "
            f"| use={escape(entry.zarr_use or '-')} "
            f"| detect={len(entry.detection_profiles)} "
            f"| keypoint={len(entry.keypoint_profiles)} "
            f"| eye_mask={len(entry.eye_mask_profiles)}</summary>"
        )
        parts.append("<div class='meta'>")
        parts.append(f"<span>status: <code>{escape(entry.status or '-')}</code></span>")
        parts.append(f"<span>recording_id: <code>{escape(entry.recording_id or '-')}</code></span>")
        parts.append(f"<span>zarr_use: <code>{escape(entry.zarr_use or '-')}</code></span>")
        parts.append(f"<span>zarr_path: <code>{escape(entry.zarr_path)}</code></span>")
        parts.append("</div>")
        parts.append("<div class='sections'>")
        parts.append("<div class='sec'><h4>Detection Profiles</h4>")
        parts.append(
            _render_profile_table(
                rows=entry.detection_profiles,
                metric_headers=DETECTION_METRIC_HEADERS,
                output_html=output_html,
                thumb_width=int(thumb_width),
            )
        )
        parts.append("</div>")
        parts.append("<div class='sec'><h4>Keypoint Profiles</h4>")
        parts.append(
            _render_profile_table(
                rows=entry.keypoint_profiles,
                metric_headers=KEYPOINT_METRIC_HEADERS,
                output_html=output_html,
                thumb_width=int(thumb_width),
            )
        )
        parts.append("</div>")
        parts.append("<div class='sec'><h4>Eye-Mask Profiles</h4>")
        parts.append(
            _render_profile_table(
                rows=entry.eye_mask_profiles,
                metric_headers=EYE_MASK_METRIC_HEADERS,
                output_html=output_html,
                thumb_width=int(thumb_width),
            )
        )
        parts.append("</div>")
        parts.append("</div></details>")

    parts.append("<script>")
    parts.append("const q=document.getElementById('q');")
    parts.append("const cards=Array.from(document.querySelectorAll('details.ds'));")
    parts.append("const chips=Array.from(document.querySelectorAll('.filter-chip'));")
    parts.append("const visibleCount=document.getElementById('visible-count');")
    parts.append("const state={text:'',profile:'all',zarrUse:'all'};")
    parts.append("const profileKey={detect:'hasDetect',keypoint:'hasKeypoint',eye_mask:'hasEyeMask'};")
    parts.append("function applyFilters(){let visible=0;")
    parts.append("for(const c of cards){")
    parts.append("const hay=(c.dataset.search||'');")
    parts.append("if(state.text!==''&&!hay.includes(state.text)){c.classList.add('hidden');continue;}")
    parts.append("const use=(c.dataset.zarrUse||'unknown');")
    parts.append("if(state.zarrUse!=='all'&&use!==state.zarrUse){c.classList.add('hidden');continue;}")
    parts.append("if(state.profile!=='all'){const key=profileKey[state.profile];if(!key||(c.dataset[key]||'0')!=='1'){c.classList.add('hidden');continue;}}")
    parts.append("c.classList.remove('hidden');visible+=1;}")
    parts.append("if(visibleCount){visibleCount.textContent=`visible: ${visible}`;}}")
    parts.append("function setActive(kind,value){for(const chip of chips){if(chip.dataset.filterKind===kind){chip.classList.toggle('active',chip.dataset.value===value);}}}")
    parts.append("q.addEventListener('input',()=>{state.text=q.value.toLowerCase().trim();applyFilters();});")
    parts.append("for(const chip of chips){chip.addEventListener('click',()=>{const kind=chip.dataset.filterKind||'';const value=chip.dataset.value||'all';")
    parts.append("if(kind==='profile'){state.profile=value;setActive(kind,value);applyFilters();}")
    parts.append("else if(kind==='zarr-use'){state.zarrUse=value;setActive(kind,value);applyFilters();}});}")
    parts.append("applyFilters();")
    parts.append("</script></main></body></html>")
    return "".join(parts) + "\n"


def _default_output_html(registry_path: Path) -> Path:
    stem = registry_path.stem or "registry"
    return Path("/tmp") / f"{stem}_source_recording_profiles_index.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry SQLite path (default: RegistryPaths.from_env(Path.cwd()).path).",
    )
    parser.add_argument(
        "--zarr-use",
        choices=("all", "training", "analysis"),
        default="all",
        help="Optional source dataset zarr_use filter.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path (default: /tmp/<registry-stem>_source_recording_profiles_index.html).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Source Recording Profile Runs Index",
        help="HTML page title.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=260,
        help="Artifact thumbnail width in pixels (default: 260).",
    )
    parser.add_argument(
        "--include-artifacts",
        dest="include_artifacts",
        action="store_true",
        help="Extract/render profile visual PNG artifacts in the HTML (default).",
    )
    parser.add_argument(
        "--no-include-artifacts",
        dest="include_artifacts",
        action="store_false",
        help="Do not extract/render artifact thumbnails.",
    )
    parser.set_defaults(include_artifacts=True)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory where extracted artifact PNGs are written "
        "(default: <output_html_stem>.artifacts next to output HTML).",
    )
    parser.add_argument(
        "--overwrite-artifacts",
        action="store_true",
        help="Overwrite previously extracted artifact PNG files.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML with xdg-open.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    registry_path = Path(args.registry) if args.registry is not None else RegistryPaths.from_env(Path.cwd()).path
    if not registry_path.exists():
        print(f"Source profile index failed: registry not found: {registry_path}")
        return 1
    output_html = Path(args.output_html) if args.output_html is not None else _default_output_html(registry_path)
    if int(args.thumb_width) <= 0:
        parser.error("--thumb-width must be > 0.")
    artifacts_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir is not None
        else output_html.with_name(f"{output_html.stem}.artifacts")
    )

    entries = collect_source_recording_profile_entries(
        registry_path=registry_path,
        zarr_use_filter=str(args.zarr_use),
    )
    artifacts_written = 0
    if bool(args.include_artifacts):
        artifacts_written = enrich_entries_with_profile_artifacts(
            entries=entries,
            artifacts_dir=artifacts_dir,
            overwrite=bool(args.overwrite_artifacts),
        )
    html = render_source_recording_profile_index_html(
        entries=entries,
        registry_path=registry_path,
        zarr_use_filter=str(args.zarr_use),
        output_html=output_html,
        title=str(args.title),
        thumb_width=int(args.thumb_width),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(
        "Source profile index: "
        f"datasets={len(entries)} output={output_html} "
        f"registry={registry_path} zarr_use={args.zarr_use} "
        f"artifacts={'on' if args.include_artifacts else 'off'} "
        f"artifacts_written={artifacts_written}"
    )
    if args.open:
        try:
            subprocess.run(["xdg-open", str(output_html)], check=False)
        except Exception as exc:
            print(f"Warning: failed to open HTML with xdg-open: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
