"""Manifest parsing and template helpers for web labeling CLI workflows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from .report_io import _csv_export_value


def _write_csv_manifest_template(path: Path, *, fieldnames: list[str], sample: dict[str, object], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV manifest template: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: _csv_export_value(sample.get(key)) for key in fieldnames})


def _write_manifest_templates_readme(
    path: Path,
    *,
    assignments_path: Path,
    tasks_path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest template README: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Palette labeling manifest templates",
        "",
        "Files:",
        f"- {assignments_path.name}: recording ownership plan",
        f"- {tasks_path.name}: task definition plan",
        "",
        "Recommended flow:",
        f"1. Edit {assignments_path.name} and {tasks_path.name} in a spreadsheet.",
        "2. Keep recording_id and task_id stable between dry-run and apply.",
        "3. Dry-run assignments:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-assignments --input {assignments_path.name} --assigned-by OPERATOR",
        "4. Apply assignments after review:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-assignments --input {assignments_path.name} --assigned-by OPERATOR --apply",
        "5. Dry-run tasks:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-tasks --input {tasks_path.name} --actor OPERATOR",
        "6. Apply tasks after review:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-tasks --input {tasks_path.name} --actor OPERATOR --apply",
        "",
        "CSV notes:",
        "- Fully blank trailing rows are ignored.",
        "- Partially filled rows fail validation.",
        "- Task scope_json cells must contain valid JSON, for example {\"frames\":[1,2,3]}.",
        "- CSV dry-run and apply output include source_line so spreadsheet rows can be traced.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_scope(value: str | None) -> object:
    if value is None or not str(value).strip():
        return {}
    raw = str(value).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        path = Path(raw).expanduser()
        return json.loads(path.read_text(encoding="utf-8"))


def _parse_csv_dict_rows(
    text: str,
    *,
    required_headers: Sequence[str] = (),
    any_header_groups: Sequence[Sequence[str]] = (),
    source: str = "CSV manifest",
) -> list[dict[str, object]]:
    reader = csv.DictReader(text.splitlines())
    fieldnames = [str(field or "").strip() for field in (reader.fieldnames or [])]
    fieldname_set = set(fieldnames)
    missing_headers = [header for header in required_headers if header not in fieldname_set]
    missing_groups = [
        " or ".join(group)
        for group in any_header_groups
        if not any(header in fieldname_set for header in group)
    ]
    if missing_headers or missing_groups:
        missing = missing_headers + missing_groups
        raise ValueError(f"{source} is missing required CSV column(s): {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for line_number, row in enumerate(reader, start=2):
        if not any(str(value or "").strip() for value in row.values()):
            continue
        parsed = dict(row)
        parsed["_source_line"] = line_number
        rows.append(parsed)
    return rows


def _parse_assignment_manifest(value: str) -> list[dict[str, object]]:
    path = Path(value).expanduser()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".csv":
        payload = _parse_csv_dict_rows(
            text,
            required_headers=("recording_id",),
            any_header_groups=(("assignee_user", "user"),),
            source="Assignment CSV manifest",
        )
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        rows = payload.get("assignments", [])
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Assignment manifest must be a CSV file, JSON list, JSONL file, or object with an assignments list.")
    parsed: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Assignment row {idx} must be an object.")
        recording_id = str(row.get("recording_id") or "").strip()
        assignee_user = str(row.get("assignee_user") or row.get("user") or "").strip()
        if not recording_id:
            raise ValueError(f"Assignment row {idx} is missing recording_id.")
        if not assignee_user:
            raise ValueError(f"Assignment row {idx} is missing assignee_user/user.")
        parsed.append(
            {
                "recording_id": recording_id,
                "assignee_user": assignee_user,
                "assigned_by": row.get("assigned_by"),
                "status": str(row.get("status") or "active"),
                "notes": row.get("notes"),
                **({"_source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
            }
        )
    return parsed


def _assignment_rows_for_apply(rows: Sequence[Mapping[str, object]], *, apply: bool) -> list[Mapping[str, object]]:
    if not apply:
        return list(rows)
    latest_by_recording: dict[str, Mapping[str, object]] = {}
    order: list[str] = []
    for row in rows:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        if recording_id not in latest_by_recording:
            order.append(recording_id)
        latest_by_recording[recording_id] = row
    return [latest_by_recording[recording_id] for recording_id in order]


def _parse_task_manifest(value: str) -> list[dict[str, object]]:
    path = Path(value).expanduser()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".csv":
        payload = _parse_csv_dict_rows(
            text,
            required_headers=("task_id", "recording_id", "workflow_kind"),
            source="Task CSV manifest",
        )
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        rows = payload.get("tasks", [])
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Task manifest must be a CSV file, JSON list, JSONL file, or object with a tasks list.")
    parsed: list[dict[str, object]] = []
    seen_task_ids: set[str] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Task row {idx} must be an object.")
        task_id = str(row.get("task_id") or "").strip()
        recording_id = str(row.get("recording_id") or "").strip()
        workflow_kind = str(row.get("workflow_kind") or "").strip()
        if not task_id:
            raise ValueError(f"Task row {idx} is missing task_id.")
        if not recording_id:
            raise ValueError(f"Task row {idx} is missing recording_id.")
        if not workflow_kind:
            raise ValueError(f"Task row {idx} is missing workflow_kind.")
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate task_id in task manifest: {task_id}")
        seen_task_ids.add(task_id)
        scope = row.get("scope", row.get("scope_json", {}))
        if isinstance(scope, str):
            scope = json.loads(scope) if scope.strip() else {}
        parsed.append(
            {
                "task_id": task_id,
                "recording_id": recording_id,
                "workflow_kind": workflow_kind,
                "dataset_id": row.get("dataset_id"),
                "zarr_use": row.get("zarr_use"),
                "stage_group": row.get("stage_group"),
                "run_name": row.get("run_name"),
                "component_name": row.get("component_name"),
                "title": row.get("title"),
                "scope": scope if scope is not None else {},
                "state": str(row.get("state") or "pending"),
                "priority": int(row.get("priority") or 0),
                "notes": row.get("notes"),
                **({"_source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
            }
        )
    return parsed
