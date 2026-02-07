#!/usr/bin/env python3
"""Summarize training sets and latest runs from the registry."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fisheye.registry.db import Registry, RegistryPaths

try:  # optional rich dependency
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - optional
    Console = None
    Table = None


@dataclass
class TrainingRow:
    set_id: str
    name: Optional[str]
    created_utc: Optional[str]
    dataset_count: Optional[int]
    run_id: Optional[str]
    run_status: Optional[str]
    run_created_utc: Optional[str]
    model_path: Optional[str]
    metrics_path: Optional[str]
    manifest_path: Optional[str]
    config_path: Optional[str]
    onnx_path: Optional[str]
    trt_path: Optional[str]
    roi_count: Optional[int]
    metrics_summary: Optional[str]


@dataclass
class UnlinkedRunRow:
    run_id: str
    set_id: Optional[str]
    status: Optional[str]
    created_utc: Optional[str]
    model_path: Optional[str]
    metrics_path: Optional[str]
    manifest_path: Optional[str]
    config_path: Optional[str]
    metrics_summary: Optional[str]


def _status_text(value: Optional[bool]) -> str:
    if value is None:
        return "—"
    return "OK" if value else "MISS"


def _status_rich(value: Optional[bool]) -> str:
    if value is None:
        return "—"
    return "[chartreuse1]OK[/chartreuse1]" if value else "[red]MISS[/red]"


def _status_with_details(value: Optional[bool], details: Optional[str], *, rich: bool) -> str:
    if value is None:
        return "—"
    if not value:
        return "[red]MISS[/red]" if rich else "MISS"
    base = "[chartreuse1]OK[/chartreuse1]" if rich else "OK"
    if details:
        return f"{base} ({details})"
    return base


def _path_ok(path: Optional[str]) -> Optional[bool]:
    if not path:
        return None
    return Path(path).exists()


def _parse_count(payload: Optional[str]) -> Optional[int]:
    if not payload:
        return None
    try:
        items = json.loads(payload)
        if isinstance(items, list):
            return len(items)
    except Exception:
        return None
    return None


def _sum_manifest_rois(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return None
    total = 0
    found = False
    for item in datasets:
        if not isinstance(item, dict):
            continue
        value = item.get("total_bboxes")
        if isinstance(value, (int, float)):
            total += int(value)
            found = True
    return total if found else None


def _parse_json_dict(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _metric_value(payload: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _format_metric(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value:.3f}"


def _metrics_summary_from_json(payload: Optional[str]) -> Optional[str]:
    data = _parse_json_dict(payload)
    if not data:
        return None
    parts: List[str] = []
    map50 = _format_metric(_metric_value(data, ["mAP50", "map50"]))
    map50_95 = _format_metric(_metric_value(data, ["mAP50_95", "mAP50-95", "map50_95", "map50-95"]))
    precision = _format_metric(_metric_value(data, ["precision", "metrics/precision(B)"]))
    recall = _format_metric(_metric_value(data, ["recall", "metrics/recall(B)"]))
    if map50 is not None:
        parts.append(f"mAP50={map50}")
    if map50_95 is not None:
        parts.append(f"mAP50-95={map50_95}")
    if precision is not None:
        parts.append(f"P={precision}")
    if recall is not None:
        parts.append(f"R={recall}")
    if not parts:
        return None
    return ", ".join(parts)


def _fetch_latest_run(registry: Registry, set_id: str) -> Dict[str, Optional[str]]:
    row = registry.conn.execute(
        """
        SELECT run_id, status, created_utc, model_path, metrics_path, manifest_path, config_path, final_metrics_json
        FROM training_runs
        WHERE set_id = ?
        ORDER BY created_utc DESC, run_id DESC
        LIMIT 1;
        """,
        (set_id,),
    ).fetchone()
    if not row:
        return {}
    return {
        "run_id": row["run_id"],
        "run_status": row["status"],
        "run_created_utc": row["created_utc"],
        "model_path": row["model_path"],
        "metrics_path": row["metrics_path"],
        "manifest_path": row["manifest_path"],
        "config_path": row["config_path"],
        "final_metrics_json": row["final_metrics_json"],
    }


def _fetch_exports(registry: Registry, run_id: str) -> Dict[str, Optional[str]]:
    rows = registry.conn.execute(
        """
        SELECT export_type, path
        FROM model_exports
        WHERE run_id = ?;
        """,
        (run_id,),
    ).fetchall()
    exports = {str(row["export_type"]).lower(): row["path"] for row in rows}
    return {
        "onnx_path": exports.get("onnx"),
        "trt_path": exports.get("tensorrt"),
    }


def _load_rows(registry: Registry, set_filter: Optional[str], limit: Optional[int]) -> List[TrainingRow]:
    sql = [
        "SELECT set_id, name, dataset_ids_json, created_utc",
        "FROM training_sets",
    ]
    params: List[Any] = []
    if set_filter:
        sql.append("WHERE set_id = ?")
        params.append(set_filter)
    sql.append("ORDER BY created_utc DESC, set_id DESC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(limit)
    rows = registry.conn.execute(" ".join(sql), params).fetchall()

    results: List[TrainingRow] = []
    for row in rows:
        latest = _fetch_latest_run(registry, row["set_id"])
        exports = {}
        if latest.get("run_id"):
            exports = _fetch_exports(registry, latest["run_id"])
        roi_count = _sum_manifest_rois(latest.get("manifest_path"))
        results.append(
            TrainingRow(
                set_id=row["set_id"],
                name=row["name"],
                created_utc=row["created_utc"],
                dataset_count=_parse_count(row["dataset_ids_json"]),
                run_id=latest.get("run_id"),
                run_status=latest.get("run_status"),
                run_created_utc=latest.get("run_created_utc"),
                model_path=latest.get("model_path"),
                metrics_path=latest.get("metrics_path"),
                manifest_path=latest.get("manifest_path"),
                config_path=latest.get("config_path"),
                onnx_path=exports.get("onnx_path"),
                trt_path=exports.get("trt_path"),
                roi_count=roi_count,
                metrics_summary=_metrics_summary_from_json(latest.get("final_metrics_json")),
            )
        )
    return results


def _load_unlinked_runs(registry: Registry, limit: Optional[int]) -> List[UnlinkedRunRow]:
    sql = [
        "SELECT run_id, set_id, status, created_utc, model_path, metrics_path, manifest_path, config_path, final_metrics_json",
        "FROM training_runs",
        "WHERE set_id IS NULL OR TRIM(set_id) = '' OR set_id NOT IN (SELECT set_id FROM training_sets)",
        "ORDER BY created_utc DESC, run_id DESC",
    ]
    params: List[Any] = []
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(limit)
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        UnlinkedRunRow(
            run_id=row["run_id"],
            set_id=row["set_id"],
            status=row["status"],
            created_utc=row["created_utc"],
            model_path=row["model_path"],
            metrics_path=row["metrics_path"],
            manifest_path=row["manifest_path"],
            config_path=row["config_path"],
            metrics_summary=_metrics_summary_from_json(row["final_metrics_json"]),
        )
        for row in rows
    ]


def _format_time(value: Optional[str]) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--set-id", type=str, help="Filter to a specific training set id.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--hide-unlinked",
        action="store_true",
        help="Do not show runs that are not linked to a training_set row.",
    )
    parser.add_argument("--no-rich", action="store_true", help="Disable rich table output.")
    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    rows = _load_rows(registry, args.set_id, args.limit)
    unlinked_runs = _load_unlinked_runs(registry, args.limit) if not args.hide_unlinked else []
    registry.close()

    if not rows and not unlinked_runs:
        print("No training sets found.")
        return 1

    use_rich = not args.no_rich and Console is not None and Table is not None
    if use_rich:
        console = Console()
        if rows:
            table = Table(title="Training Registry Status", show_lines=False)
            table.add_column("Set ID", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Created")
            table.add_column("Datasets", justify="right")
            table.add_column("Manifest")
            table.add_column("Config")
            table.add_column("Latest Run")
            table.add_column("Status")
            table.add_column("Run Time")
            table.add_column("ROIs", justify="right")
            table.add_column("Model")
            table.add_column("Metrics")
            table.add_column("ONNX")
            table.add_column("TRT")
            for row in rows:
                run_id = row.run_id or "—"
                status = row.run_status or "—"
                if row.run_status == "failed":
                    run_id = f"[red]{run_id}[/red]"
                    status = "[red]failed[/red]"
                elif row.run_status == "in_progress":
                    run_id = f"[yellow]{run_id}[/yellow]"
                    status = "[yellow]in_progress[/yellow]"
                elif row.run_status == "success":
                    run_id = f"[chartreuse1]{run_id}[/chartreuse1]"
                    status = "[chartreuse1]success[/chartreuse1]"
                table.add_row(
                    row.set_id,
                    row.name or "—",
                    _format_time(row.created_utc),
                    str(row.dataset_count) if row.dataset_count is not None else "—",
                    _status_rich(_path_ok(row.manifest_path)),
                    _status_rich(_path_ok(row.config_path)),
                    run_id,
                    status,
                    _format_time(row.run_created_utc),
                    str(row.roi_count) if row.roi_count is not None else "—",
                    _status_rich(_path_ok(row.model_path)),
                    _status_with_details(_path_ok(row.metrics_path), row.metrics_summary, rich=True),
                    _status_rich(_path_ok(row.onnx_path)),
                    _status_rich(_path_ok(row.trt_path)),
                )
            console.print(table)

        if unlinked_runs:
            unlinked = Table(title="Unlinked Training Runs", show_lines=False)
            unlinked.add_column("Run ID", style="cyan")
            unlinked.add_column("Set ID")
            unlinked.add_column("Status")
            unlinked.add_column("Created")
            unlinked.add_column("Model")
            unlinked.add_column("Metrics")
            unlinked.add_column("Manifest")
            unlinked.add_column("Config")
            for run in unlinked_runs:
                status = run.status or "—"
                if run.status == "failed":
                    status = "[red]failed[/red]"
                elif run.status == "in_progress":
                    status = "[yellow]in_progress[/yellow]"
                elif run.status == "success":
                    status = "[chartreuse1]success[/chartreuse1]"
                unlinked.add_row(
                    run.run_id,
                    run.set_id or "—",
                    status,
                    _format_time(run.created_utc),
                    _status_rich(_path_ok(run.model_path)),
                    _status_with_details(_path_ok(run.metrics_path), run.metrics_summary, rich=True),
                    _status_rich(_path_ok(run.manifest_path)),
                    _status_rich(_path_ok(run.config_path)),
                )
            console.print(unlinked)
    else:
        for row in rows:
            print(row.set_id)
            print(f"  name: {row.name or '—'}")
            print(f"  created: {_format_time(row.created_utc)}")
            print(f"  datasets: {row.dataset_count if row.dataset_count is not None else '—'}")
            print(f"  latest_run: {row.run_id or '—'}")
            print(f"  status: {row.run_status or '—'}")
            print(f"  run_time: {_format_time(row.run_created_utc)}")
            print(f"  rois: {row.roi_count if row.roi_count is not None else '—'}")
            print(f"  model: {_status_text(_path_ok(row.model_path))}")
            print(f"  metrics: {_status_with_details(_path_ok(row.metrics_path), row.metrics_summary, rich=False)}")
            print(f"  manifest: {_status_text(_path_ok(row.manifest_path))}")
            print(f"  config: {_status_text(_path_ok(row.config_path))}")
            print(f"  onnx: {_status_text(_path_ok(row.onnx_path))}")
            print(f"  trt: {_status_text(_path_ok(row.trt_path))}")
        if unlinked_runs:
            print("\nUnlinked training runs:")
            for run in unlinked_runs:
                print(run.run_id)
                print(f"  set_id: {run.set_id or '—'}")
                print(f"  status: {run.status or '—'}")
                print(f"  created: {_format_time(run.created_utc)}")
                print(f"  model: {_status_text(_path_ok(run.model_path))}")
                print(
                    f"  metrics: {_status_with_details(_path_ok(run.metrics_path), run.metrics_summary, rich=False)}"
                )
                print(f"  manifest: {_status_text(_path_ok(run.manifest_path))}")
                print(f"  config: {_status_text(_path_ok(run.config_path))}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
