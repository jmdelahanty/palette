"""Read-only readiness audit for clipped collection ROI-cache model workflows."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq


SCHEMA_ID = "palette.clipped_collection_cache_readiness_audit.v1"
REQUIRED_FRAME_INDEX_COLUMNS = {
    "camera_serial",
    "clip_id",
    "clip_local_frame_index",
    "recording_frame_id",
}
CLIPPED_SOURCE_LAYOUT = "rolling_clips"


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _read_zarr_attrs(group_path: Path) -> dict[str, Any]:
    payload = _read_json(group_path / "zarr.json")
    attrs = payload.get("attributes")
    if attrs is None:
        attrs = payload.get("attrs")
    if not isinstance(attrs, Mapping):
        return {}
    return dict(attrs)


def _recording_root_from_zarr(zarr_path: Path) -> Path:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    return zarr_path.parent


def _path_payload(value: object) -> dict[str, Any]:
    text = "" if value is None else str(value).strip()
    if not text:
        return {"path": None, "exists": False, "stale_nvme1": False}
    path = Path(text).expanduser()
    return {
        "path": str(path),
        "exists": path.exists(),
        "stale_nvme1": str(path).startswith("/nvme1/"),
    }


def _resolve_collection_id(zarr_path: Path, explicit_collection_id: str | None) -> tuple[str | None, list[AuditIssue]]:
    issues: list[AuditIssue] = []
    if explicit_collection_id:
        return explicit_collection_id, issues
    refined_attrs_path = zarr_path / "refined_detect_runs" / "zarr.json"
    try:
        attrs = _read_zarr_attrs(zarr_path / "refined_detect_runs")
    except FileNotFoundError:
        issues.append(
            AuditIssue(
                "blocker",
                "missing_refined_detect_parent",
                "refined_detect_runs/zarr.json is missing and no --collection-id was provided.",
                str(refined_attrs_path),
            )
        )
        return None, issues
    collection_id = str(attrs.get("latest_collection") or "").strip()
    if not collection_id:
        issues.append(
            AuditIssue(
                "blocker",
                "missing_latest_collection",
                "refined_detect_runs.latest_collection is empty and no --collection-id was provided.",
                str(refined_attrs_path),
            )
        )
        return None, issues
    return collection_id, issues


def _load_collection(zarr_path: Path, collection_id: str) -> tuple[dict[str, Any] | None, list[AuditIssue]]:
    collection_group = zarr_path / "experiment_index" / "finalized_runs" / collection_id
    try:
        attrs = _read_zarr_attrs(collection_group)
    except FileNotFoundError:
        return None, [
            AuditIssue(
                "blocker",
                "missing_finalized_collection",
                f"Finalized collection not found: {collection_id}",
                str(collection_group / "zarr.json"),
            )
        ]
    selected_runs = attrs.get("selected_runs")
    if not isinstance(selected_runs, list) or not selected_runs:
        return attrs, [
            AuditIssue(
                "blocker",
                "empty_finalized_collection",
                f"Finalized collection has no selected_runs: {collection_id}",
                str(collection_group / "zarr.json"),
            )
        ]
    return attrs, []


def _audit_root_attrs(zarr_path: Path, recording_root: Path, attrs: Mapping[str, Any]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    if not str(attrs.get("recording_id") or "").strip():
        issues.append(AuditIssue("warning", "missing_recording_id_attr", "Zarr root is missing recording_id."))
    if str(attrs.get("source_layout") or "").strip() != CLIPPED_SOURCE_LAYOUT:
        issues.append(
            AuditIssue(
                "warning",
                "missing_or_nonclipped_source_layout",
                f"Zarr root source_layout is not {CLIPPED_SOURCE_LAYOUT!r}.",
                str(zarr_path / "zarr.json"),
            )
        )
    source_frame_index_attr = str(attrs.get("source_recording_frame_index_path") or "").strip()
    frame_index_attr = source_frame_index_attr or str(attrs.get("recording_frame_index_path") or "").strip()
    if not frame_index_attr:
        default_path = recording_root / "recording_frame_index.parquet"
        issues.append(
            AuditIssue(
                "warning",
                "missing_source_recording_frame_index_attr",
                "Zarr root is missing source_recording_frame_index_path; default recording-root sidecar will be checked.",
                str(default_path),
            )
        )
    elif frame_index_attr.startswith("/nvme1/"):
        issues.append(
            AuditIssue(
                "warning",
                "stale_source_recording_frame_index_attr",
                "Zarr root frame-index path points to /nvme1.",
                frame_index_attr,
            )
        )
    elif not source_frame_index_attr:
        issues.append(
            AuditIssue(
                "warning",
                "missing_source_recording_frame_index_alias",
                "Zarr root has recording_frame_index_path but is missing registry-readable source_recording_frame_index_path.",
                frame_index_attr,
            )
        )
    return issues


def _resolve_frame_index_path(
    recording_root: Path,
    attrs: Mapping[str, Any],
) -> Path:
    attr_value = str(
        attrs.get("source_recording_frame_index_path") or attrs.get("recording_frame_index_path") or ""
    ).strip()
    if attr_value:
        path = Path(attr_value).expanduser()
        if not path.is_absolute():
            path = recording_root / path
        return path
    return recording_root / "recording_frame_index.parquet"


def _audit_frame_index(path: Path, *, sample_video_paths: int) -> tuple[dict[str, Any], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "required_columns": sorted(REQUIRED_FRAME_INDEX_COLUMNS),
    }
    if not path.exists():
        issues.append(
            AuditIssue(
                "blocker",
                "missing_recording_frame_index",
                "recording_frame_index.parquet is required for clipped collection parent-frame mapping.",
                str(path),
            )
        )
        return payload, issues

    parquet_file = pq.ParquetFile(path)
    column_names = list(parquet_file.schema_arrow.names)
    missing = sorted(REQUIRED_FRAME_INDEX_COLUMNS.difference(column_names))
    payload.update(
        {
            "row_count": int(parquet_file.metadata.num_rows),
            "columns": column_names,
            "missing_required_columns": missing,
        }
    )
    if missing:
        issues.append(
            AuditIssue(
                "blocker",
                "recording_frame_index_missing_required_columns",
                f"recording_frame_index.parquet is missing required columns: {missing}",
                str(path),
            )
        )
        return payload, issues

    if "video_path" in column_names and sample_video_paths > 0 and parquet_file.num_row_groups > 0:
        table = parquet_file.read_row_group(0, columns=["video_path"])
        values = [str(item) for item in table["video_path"].to_pylist()[: int(sample_video_paths)]]
        sampled = [_path_payload(value) for value in values]
        payload["sampled_video_paths"] = sampled
        stale_count = sum(1 for item in sampled if item["stale_nvme1"])
        missing_count = sum(1 for item in sampled if item["path"] and not item["exists"])
        if stale_count:
            issues.append(
                AuditIssue(
                    "warning",
                    "sampled_frame_index_video_paths_stale",
                    f"{stale_count}/{len(sampled)} sampled frame-index video paths point to /nvme1.",
                    str(path),
                )
            )
        if missing_count:
            issues.append(
                AuditIssue(
                    "warning",
                    "sampled_frame_index_video_paths_missing",
                    f"{missing_count}/{len(sampled)} sampled frame-index video paths do not exist.",
                    str(path),
                )
            )
    return payload, issues


def _audit_selected_runs(selected_runs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    samples: list[dict[str, Any]] = []
    source_path_missing = 0
    source_path_stale = 0
    source_path_not_found = 0
    missing_refined_path = 0
    clip_ids: set[str] = set()
    work_units: set[str] = set()

    for index, row in enumerate(selected_runs):
        clip_id = str(row.get("clip_id") or "")
        work_unit_id = str(row.get("work_unit_id") or "")
        if clip_id:
            clip_ids.add(clip_id)
        if work_unit_id:
            work_units.add(work_unit_id)
        if not str(row.get("refined_group_path") or "").strip():
            missing_refined_path += 1
        source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
        source_payload = _path_payload(source.get("video_path"))
        if not source_payload["path"]:
            source_path_missing += 1
        elif source_payload["stale_nvme1"]:
            source_path_stale += 1
        elif not source_payload["exists"]:
            source_path_not_found += 1
        if index < 5:
            samples.append(
                {
                    "clip_id": clip_id or None,
                    "clip_index": row.get("clip_index"),
                    "work_unit_id": work_unit_id or None,
                    "refined_detect_run": row.get("refined_detect_run"),
                    "refined_group_path": row.get("refined_group_path"),
                    "source_video_path": source_payload,
                }
            )

    if missing_refined_path:
        issues.append(
            AuditIssue(
                "blocker",
                "selected_runs_missing_refined_group_path",
                f"{missing_refined_path} selected run(s) are missing refined_group_path.",
            )
        )
    if source_path_missing:
        issues.append(
            AuditIssue(
                "blocker",
                "selected_runs_missing_source_video_path",
                f"{source_path_missing} selected run(s) are missing source.video_path.",
            )
        )
    if source_path_not_found:
        issues.append(
            AuditIssue(
                "blocker",
                "selected_runs_source_video_path_not_found",
                f"{source_path_not_found} selected run source.video_path value(s) do not exist.",
            )
        )
    if source_path_stale:
        issues.append(
            AuditIssue(
                "warning",
                "selected_runs_source_video_path_stale",
                f"{source_path_stale} selected run source.video_path value(s) point to /nvme1.",
            )
        )
    return (
        {
            "selected_run_count": len(selected_runs),
            "clip_count": len(clip_ids),
            "work_unit_count": len(work_units),
            "missing_refined_group_path_count": missing_refined_path,
            "missing_source_video_path_count": source_path_missing,
            "stale_source_video_path_count": source_path_stale,
            "missing_source_video_file_count": source_path_not_found,
            "sample_selected_runs": samples,
        },
        issues,
    )


def _registry_rows(registry_path: Path, zarr_path: Path) -> tuple[list[dict[str, Any]], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    if not registry_path.exists():
        return [], [
            AuditIssue(
                "warning",
                "registry_not_found",
                "Registry path does not exist; skipping registry targeting checks.",
                str(registry_path),
            )
        ]
    conn = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT dataset_id, recording_id, zarr_use, source_layout,
                       source_recording_frame_index_path, source_frame_index_schema,
                       zarr_path
                FROM datasets
                WHERE zarr_path = ?
                ORDER BY dataset_id
                """,
                (str(zarr_path),),
            )
        ]
    finally:
        conn.close()
    if not rows:
        issues.append(
            AuditIssue(
                "warning",
                "registry_dataset_row_missing",
                "No datasets row matched this zarr_path exactly.",
                str(zarr_path),
            )
        )
    for row in rows:
        if str(row.get("source_layout") or "") != CLIPPED_SOURCE_LAYOUT:
            issues.append(
                AuditIssue(
                    "warning",
                    "registry_source_layout_not_clipped",
                    f"Registry dataset {row.get('dataset_id')} source_layout is not {CLIPPED_SOURCE_LAYOUT!r}.",
                )
            )
        frame_index = str(row.get("source_recording_frame_index_path") or "")
        if not frame_index:
            issues.append(
                AuditIssue(
                    "warning",
                    "registry_missing_source_recording_frame_index_path",
                    f"Registry dataset {row.get('dataset_id')} is missing source_recording_frame_index_path.",
                )
            )
        elif frame_index.startswith("/nvme1/"):
            issues.append(
                AuditIssue(
                    "warning",
                    "registry_stale_source_recording_frame_index_path",
                    f"Registry dataset {row.get('dataset_id')} source_recording_frame_index_path points to /nvme1.",
                    frame_index,
                )
            )
    return rows, issues


def audit_clipped_collection_cache_readiness(
    zarr_path: Path,
    *,
    collection_id: str | None = None,
    registry: Path | None = None,
    sample_video_paths: int = 5,
) -> dict[str, Any]:
    archive_path = zarr_path.expanduser().resolve()
    issues: list[AuditIssue] = []
    if not archive_path.exists():
        issues.append(AuditIssue("blocker", "zarr_path_not_found", "Zarr path does not exist.", str(archive_path)))
        return _finish_payload(archive_path, None, issues, {})

    recording_root = _recording_root_from_zarr(archive_path)
    try:
        root_attrs = _read_zarr_attrs(archive_path)
    except FileNotFoundError:
        issues.append(AuditIssue("blocker", "missing_root_zarr_json", "Root zarr.json is missing.", str(archive_path / "zarr.json")))
        return _finish_payload(archive_path, None, issues, {"recording_root": str(recording_root)})

    issues.extend(_audit_root_attrs(archive_path, recording_root, root_attrs))
    resolved_collection_id, collection_issues = _resolve_collection_id(archive_path, collection_id)
    issues.extend(collection_issues)

    collection_attrs: dict[str, Any] | None = None
    selected_runs: list[Mapping[str, Any]] = []
    if resolved_collection_id is not None:
        collection_attrs, collection_issues = _load_collection(archive_path, resolved_collection_id)
        issues.extend(collection_issues)
        if collection_attrs is not None and isinstance(collection_attrs.get("selected_runs"), list):
            selected_runs = [row for row in collection_attrs["selected_runs"] if isinstance(row, Mapping)]

    selected_summary: dict[str, Any] = {}
    if selected_runs:
        selected_summary, selected_issues = _audit_selected_runs(selected_runs)
        issues.extend(selected_issues)

    frame_index_path = _resolve_frame_index_path(recording_root, root_attrs)
    frame_index_payload, frame_index_issues = _audit_frame_index(
        frame_index_path,
        sample_video_paths=sample_video_paths,
    )
    issues.extend(frame_index_issues)

    registry_rows: list[dict[str, Any]] = []
    if registry is not None:
        registry_rows, registry_issues = _registry_rows(registry.expanduser().resolve(), archive_path)
        issues.extend(registry_issues)

    extra = {
        "recording_root": str(recording_root),
        "root_attrs": {
            "recording_id": root_attrs.get("recording_id"),
            "zarr_use": root_attrs.get("zarr_use"),
            "zarr_purpose": root_attrs.get("zarr_purpose"),
            "source_layout": root_attrs.get("source_layout"),
            "source_recording_frame_index_path": root_attrs.get("source_recording_frame_index_path"),
            "source_frame_index_schema": root_attrs.get("source_frame_index_schema"),
        },
        "collection": {
            "collection_id": resolved_collection_id,
            "path": (
                f"experiment_index/finalized_runs/{resolved_collection_id}"
                if resolved_collection_id
                else None
            ),
            "selected_run_count": len(selected_runs),
            "plan_path": collection_attrs.get("plan_path") if collection_attrs else None,
            **selected_summary,
        },
        "recording_frame_index": frame_index_payload,
        "registry": {
            "path": str(registry.expanduser().resolve()) if registry is not None else None,
            "matched_dataset_rows": registry_rows,
        },
    }
    return _finish_payload(archive_path, resolved_collection_id, issues, extra)


def _finish_payload(
    zarr_path: Path,
    collection_id: str | None,
    issues: Sequence[AuditIssue],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = [issue for issue in issues if issue.severity == "blocker"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    if blockers:
        status = "blocked"
    elif warnings:
        status = "warning"
    else:
        status = "ok"
    return {
        "schema_id": SCHEMA_ID,
        "status": status,
        "zarr_path": str(zarr_path),
        "collection_id": collection_id,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "issues": [asdict(issue) for issue in issues],
        **dict(extra),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit for clipped collection ROI-cache/keypoint/mask readiness."
    )
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path.")
    parser.add_argument("--collection-id", help="Finalized collection id; defaults to refined_detect_runs.latest_collection.")
    parser.add_argument("--registry", type=Path, help="Optional Palette registry sqlite path for dataset targeting checks.")
    parser.add_argument(
        "--sample-video-paths",
        type=int,
        default=5,
        help="Number of frame-index video_path values to sample from the first row group (default: 5).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = audit_clipped_collection_cache_readiness(
        args.zarr_path,
        collection_id=args.collection_id,
        registry=args.registry,
        sample_video_paths=max(0, int(args.sample_video_paths)),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"ok", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
