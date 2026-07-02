#!/usr/bin/env python3
"""Read-only eye-mask severance coverage census.

This script checks whether recordings that may have legacy standalone
``refined_eye_masks_runs`` already have canonical subject-mask eye geometry in
either ``refined_subject_masks_runs`` or ``analysis/subject_shape_runs``. It
uses filesystem metadata only for Zarr inspection and opens the registry in
SQLite read-only mode.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_REGISTRY = Path("/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite")
DEFAULT_RECORDINGS_ROOT = Path("/groups/johnson/johnsonlab/jeremy/recordings")

REFINED_SUBJECT_ARRAYS = (
    "components/eye_left/geometry/ellipse_params",
    "components/eye_left/geometry/ellipse_success",
    "components/eye_right/geometry/ellipse_params",
    "components/eye_right/geometry/ellipse_success",
)
SUBJECT_SHAPE_ARRAYS = (
    "components/eye_left/ellipse_params",
    "components/eye_left/ellipse_success",
    "components/eye_right/ellipse_params",
    "components/eye_right/ellipse_success",
)


@dataclass
class RunPresence:
    present: bool
    run_names: list[str] = field(default_factory=list)


@dataclass
class RecordingCensus:
    recording_id: str
    zarr_paths: set[Path] = field(default_factory=set)
    registry_dataset_ids: set[str] = field(default_factory=set)
    registry_recording_ids: set[str] = field(default_factory=set)
    legacy_refined_eye: RunPresence = field(default_factory=lambda: RunPresence(False))
    refined_subject_eye: RunPresence = field(default_factory=lambda: RunPresence(False))
    subject_shape_eye: RunPresence = field(default_factory=lambda: RunPresence(False))
    missing_zarr_paths: set[Path] = field(default_factory=set)
    unreadable_zarr_paths: dict[Path, str] = field(default_factory=dict)

    @property
    def classification(self) -> str:
        if self.refined_subject_eye.present or self.subject_shape_eye.present:
            return "CONVERTED"
        if self.legacy_refined_eye.present:
            return "NEEDS-CONVERSION"
        return "NO-EYE-DATA"


def _array_shape(array_path: Path) -> Optional[tuple[int, ...]]:
    zarr_json = array_path / "zarr.json"
    if zarr_json.is_file():
        try:
            payload = json.loads(zarr_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        shape = payload.get("shape")
        if isinstance(shape, list):
            try:
                return tuple(int(value) for value in shape)
            except (TypeError, ValueError):
                return None

    zarray = array_path / ".zarray"
    if zarray.is_file():
        try:
            payload = json.loads(zarray.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        shape = payload.get("shape")
        if isinstance(shape, list):
            try:
                return tuple(int(value) for value in shape)
            except (TypeError, ValueError):
                return None

    return None


def _array_present_nonempty(run_group: Path, relative_path: str) -> bool:
    shape = _array_shape(run_group / relative_path)
    return bool(shape) and int(shape[0]) > 0


def _iter_run_groups(parent: Path) -> Iterable[Path]:
    if not parent.is_dir():
        return ()
    try:
        children = sorted(path for path in parent.iterdir() if path.is_dir())
    except OSError:
        return ()
    return tuple(children)


def _runs_with_arrays(zarr_path: Path, parent_path: str, required_arrays: tuple[str, ...]) -> RunPresence:
    parent = zarr_path / parent_path
    names: list[str] = []
    for run_group in _iter_run_groups(parent):
        if all(_array_present_nonempty(run_group, path) for path in required_arrays):
            names.append(run_group.name)
    return RunPresence(bool(names), names)


def _legacy_refined_eye_presence(zarr_path: Path) -> RunPresence:
    parent = zarr_path / "refined_eye_masks_runs"
    names: list[str] = []
    for run_group in _iter_run_groups(parent):
        if (
            _array_present_nonempty(run_group, "masks_roi")
            or _array_present_nonempty(run_group, "ellipse_params")
            or (run_group / "zarr.json").is_file()
            or (run_group / ".zgroup").is_file()
        ):
            names.append(run_group.name)
    return RunPresence(bool(names), names)


def _merge_presence(target: RunPresence, source: RunPresence) -> None:
    if source.present:
        target.present = True
    target.run_names = sorted(set(target.run_names).union(source.run_names))


def _has_recording_layout(zarr_path: Path) -> bool:
    return len(zarr_path.parts) >= 3 and zarr_path.parent.name == "zarr"


def _is_recording_zarr_path(zarr_path: Path) -> bool:
    return "recordings" in zarr_path.parts


def _recording_id_from_zarr_path(zarr_path: Path) -> str:
    if _has_recording_layout(zarr_path):
        return zarr_path.parent.parent.name
    name = zarr_path.name
    for suffix in ("_analysis.zarr", "_training.zarr", ".zarr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return zarr_path.stem


def _registry_zarr_rows(
    registry: Path,
    *,
    active_only: bool,
) -> tuple[list[tuple[str, Optional[str], Path]], int, int]:
    if not registry.exists():
        return [], 0, 0
    uri = f"file:{registry}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        where = "WHERE COALESCE(status, 'active') = 'active'" if active_only else ""
        rows = conn.execute(
            f"""
            SELECT dataset_id, recording_id, zarr_path
            FROM datasets
            {where}
            ORDER BY recording_id, dataset_id
            """
        ).fetchall()
    finally:
        conn.close()
    out: list[tuple[str, Optional[str], Path]] = []
    skipped_without_recording_context = 0
    skipped_non_recording_rows = 0
    for row in rows:
        zarr_text = str(row["zarr_path"] or "").strip()
        if not zarr_text:
            continue
        zarr_path = Path(zarr_text)
        if not _is_recording_zarr_path(zarr_path):
            skipped_non_recording_rows += 1
            continue
        recording_id = str(row["recording_id"] or "").strip() or None
        if recording_id is None and not _has_recording_layout(zarr_path):
            skipped_without_recording_context += 1
            continue
        out.append((str(row["dataset_id"] or ""), recording_id, zarr_path))
    return out, skipped_without_recording_context, skipped_non_recording_rows


def _discover_zarrs(root: Path, *, recursive: bool) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return []
    if root.suffix == ".zarr" and root.is_dir():
        return [root]
    if not recursive:
        out: list[Path] = []
        try:
            recording_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        except OSError:
            return []
        for recording_dir in recording_dirs:
            zarr_dir = recording_dir / "zarr"
            if not zarr_dir.is_dir():
                continue
            try:
                out.extend(sorted(path for path in zarr_dir.iterdir() if path.is_dir() and path.suffix == ".zarr"))
            except OSError:
                continue
        return out
    try:
        return sorted(path for path in root.rglob("*.zarr") if path.is_dir())
    except OSError:
        return []


def _inspect_zarr(recording: RecordingCensus, zarr_path: Path) -> None:
    recording.zarr_paths.add(zarr_path)
    if not zarr_path.exists():
        recording.missing_zarr_paths.add(zarr_path)
        return
    if not zarr_path.is_dir():
        recording.unreadable_zarr_paths[zarr_path] = "not a directory"
        return
    try:
        legacy = _legacy_refined_eye_presence(zarr_path)
        refined_subject = _runs_with_arrays(zarr_path, "refined_subject_masks_runs", REFINED_SUBJECT_ARRAYS)
        subject_shape = _runs_with_arrays(zarr_path, "analysis/subject_shape_runs", SUBJECT_SHAPE_ARRAYS)
    except OSError as exc:
        recording.unreadable_zarr_paths[zarr_path] = str(exc)
        return
    _merge_presence(recording.legacy_refined_eye, legacy)
    _merge_presence(recording.refined_subject_eye, refined_subject)
    _merge_presence(recording.subject_shape_eye, subject_shape)


def build_census(
    *,
    registry: Path,
    recordings_roots: list[Path],
    active_only: bool,
    recursive_recordings_root: bool,
) -> tuple[dict[str, RecordingCensus], dict[str, object]]:
    records: dict[str, RecordingCensus] = {}

    registry_rows, skipped_registry_rows, skipped_non_recording_rows = _registry_zarr_rows(
        registry,
        active_only=active_only,
    )
    for dataset_id, registry_recording_id, zarr_path in registry_rows:
        recording_id = _recording_id_from_zarr_path(zarr_path) if _has_recording_layout(zarr_path) else registry_recording_id
        if recording_id is None:
            continue
        record = records.setdefault(recording_id, RecordingCensus(recording_id=recording_id))
        if dataset_id:
            record.registry_dataset_ids.add(dataset_id)
        if registry_recording_id:
            record.registry_recording_ids.add(registry_recording_id)
        _inspect_zarr(record, zarr_path)

    discovered_paths: set[Path] = set()
    for root in recordings_roots:
        discovered_paths.update(_discover_zarrs(root, recursive=recursive_recordings_root))

    for zarr_path in sorted(discovered_paths):
        recording_id = _recording_id_from_zarr_path(zarr_path)
        record = records.setdefault(recording_id, RecordingCensus(recording_id=recording_id))
        _inspect_zarr(record, zarr_path)

    metadata = {
        "registry": str(registry),
        "registry_rows": len(registry_rows),
        "skipped_registry_rows_without_recording_context": skipped_registry_rows,
        "skipped_non_recording_registry_rows": skipped_non_recording_rows,
        "recordings_roots": [str(path) for path in recordings_roots],
        "recursive_recordings_root": recursive_recordings_root,
        "discovered_zarr_paths": len(discovered_paths),
        "active_only": active_only,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return records, metadata


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def _fmt_list(values: Iterable[str], *, limit: int = 3) -> str:
    items = sorted(str(value) for value in values if str(value))
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", +{len(items) - limit}"


def _fmt_paths(paths: Iterable[Path], *, limit: int = 2) -> str:
    items = sorted(str(path) for path in paths)
    if not items:
        return "-"
    if len(items) <= limit:
        return "<br>".join(f"`{item}`" for item in items)
    shown = "<br>".join(f"`{item}`" for item in items[:limit])
    return f"{shown}<br>+{len(items) - limit} more"


def render_markdown(records: dict[str, RecordingCensus], metadata: dict[str, object]) -> str:
    ordered = sorted(records.values(), key=lambda item: (item.classification, item.recording_id))
    counts: dict[str, int] = {"CONVERTED": 0, "NEEDS-CONVERSION": 0, "NO-EYE-DATA": 0}
    for record in records.values():
        counts[record.classification] += 1

    needs_conversion = [record for record in records.values() if record.classification == "NEEDS-CONVERSION"]
    missing_or_unreadable = [
        record
        for record in records.values()
        if record.missing_zarr_paths or record.unreadable_zarr_paths
    ]

    lines = [
        "# Eye-Mask Severance Coverage Census",
        "",
        f"Generated: `{metadata['generated_utc']}`",
        "",
        "This is a read-only census for the standalone eye-mask severance. A recording is",
        "`CONVERTED` when at least one inspected Zarr has native refined-subject eye geometry",
        "or subject-shape eye geometry. It is `NEEDS-CONVERSION` when legacy",
        "`refined_eye_masks_runs` are present but neither canonical geometry source is present.",
        "",
        "## Inputs",
        "",
        f"- Registry: `{metadata['registry']}` opened with SQLite `mode=ro`",
        f"- Registry rows inspected: `{metadata['registry_rows']}`",
        f"- Registry rows skipped without recording context: `{metadata['skipped_registry_rows_without_recording_context']}`",
        f"- Registry rows skipped outside recording paths: `{metadata['skipped_non_recording_registry_rows']}`",
        f"- Registry active-only filter: `{metadata['active_only']}`",
        f"- Recording roots: `{', '.join(metadata['recordings_roots'])}`",
        f"- Recursive recording-root scan: `{metadata['recursive_recordings_root']}`",
        f"- Zarr paths discovered from recording roots: `{metadata['discovered_zarr_paths']}`",
        "",
        "## Summary",
        "",
        f"- Recordings inspected: `{len(records)}`",
        f"- CONVERTED: `{counts['CONVERTED']}`",
        f"- NEEDS-CONVERSION: `{counts['NEEDS-CONVERSION']}`",
        f"- NO-EYE-DATA: `{counts['NO-EYE-DATA']}`",
        f"- Missing/unreadable Zarr path rows: `{len(missing_or_unreadable)}`",
        "",
    ]

    if needs_conversion:
        lines.extend(
            [
                "## Needs Conversion",
                "",
                "These recordings have legacy `refined_eye_masks_runs` but no detected canonical subject eye geometry.",
                "",
                "| recording_id | zarr_paths | legacy_runs |",
                "|---|---|---|",
            ]
        )
        for record in sorted(needs_conversion, key=lambda item: item.recording_id):
            lines.append(
                f"| `{record.recording_id}` | {_fmt_paths(record.zarr_paths)} | "
                f"{_fmt_list(record.legacy_refined_eye.run_names, limit=5)} |"
            )
        lines.append("")
    else:
        lines.extend(["## Needs Conversion", "", "No recordings require legacy-to-subject conversion in this census.", ""])

    if missing_or_unreadable:
        lines.extend(
            [
                "## Missing Or Unreadable Inputs",
                "",
                "| recording_id | missing_paths | unreadable_paths |",
                "|---|---|---|",
            ]
        )
        for record in sorted(missing_or_unreadable, key=lambda item: item.recording_id):
            unreadable = [f"{path}: {reason}" for path, reason in sorted(record.unreadable_zarr_paths.items())]
            lines.append(
                f"| `{record.recording_id}` | {_fmt_paths(record.missing_zarr_paths)} | "
                f"{_fmt_list(unreadable, limit=2)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Full Table",
            "",
            "| classification | recording_id | legacy_refined_eye | refined_subject_eye | subject_shape_eye | zarr_count | dataset_count | zarr_paths |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in ordered:
        lines.append(
            f"| {record.classification} | `{record.recording_id}` | "
            f"{_fmt_bool(record.legacy_refined_eye.present)} | "
            f"{_fmt_bool(record.refined_subject_eye.present)} | "
            f"{_fmt_bool(record.subject_shape_eye.present)} | "
            f"{len(record.zarr_paths)} | {len(record.registry_dataset_ids)} | {_fmt_paths(record.zarr_paths)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--recordings-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Recording root to scan for <recording>/zarr/*.zarr. "
            "Defaults to the active PRFS recordings root."
        ),
    )
    parser.add_argument(
        "--recursive-recordings-root",
        action="store_true",
        help="Use a full recursive *.zarr scan under each recording root.",
    )
    parser.add_argument("--include-inactive-registry-rows", action="store_true")
    parser.add_argument("--output-md", type=Path, help="Write the Markdown report to this path.")
    args = parser.parse_args()

    roots = args.recordings_root or [DEFAULT_RECORDINGS_ROOT]
    records, metadata = build_census(
        registry=args.registry,
        recordings_roots=roots,
        active_only=not args.include_inactive_registry_rows,
        recursive_recordings_root=bool(args.recursive_recordings_root),
    )
    report = render_markdown(records, metadata)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
