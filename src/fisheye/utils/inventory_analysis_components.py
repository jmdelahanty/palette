"""Build a read-only, schema-first inventory of derived analysis components.

The inventory reads Zarr v3 ``zarr.json`` metadata directly.  It never opens a
mutable Zarr group, writes archive metadata, or updates the Palette registry.
Physical run-family paths and declared schemas are reported separately so a
protocol-specific run name does not silently become a new semantic stage.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Callable


INVENTORY_SCHEMA_ID = "palette.analysis_component_inventory.v1"

_POINTER_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "palette_latest_complete_run",
    "palette_latest_pending_run",
)
_COMPLETION_STATUS_ATTR = "palette_run_completion_status"
_COMPLETION_CONTRACT_ATTR = "palette_run_completion_contract"
_ANALYSIS_SCOPE_NAMES = frozenset({"offline", "online"})
_SUMMARY_SOURCE_ATTRS = (
    "source_refs",
    "source_fingerprints",
    "source_fingerprint",
    "source_lineage_hash",
    "lineage_hash",
    "fingerprint_status",
    "source_detect_run",
    "source_refined_detect_run",
    "source_crop_run",
    "source_keypoint_run",
    "source_refined_keypoint_run",
    "source_subject_mask_run",
    "source_refined_subject_mask_run",
    "source_stimulus_run",
    "source_stimulus_epoch_run",
    "source_tracking_run",
    "source_track_kinematics_run",
    "source_chaser_distance_run",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return str(value)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"metadata is not a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _attrs(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("attributes")
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _contains_token(value: Any, tokens: Sequence[str]) -> bool:
    normalized = _normalize_token(str(value or ""))
    return bool(normalized) and any(token and token in normalized for token in tokens)


def _declares_contract(attrs: Mapping[str, Any]) -> bool:
    return any(
        attrs.get(key) not in (None, "")
        for key in (
            "schema_id",
            "method",
            _COMPLETION_STATUS_ATTR,
            _COMPLETION_CONTRACT_ATTR,
            "lineage_hash",
            "source_lineage_hash",
            "source_fingerprint",
        )
    )


def _source_summary(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(
        {
            key: attrs[key]
            for key in _SUMMARY_SOURCE_ATTRS
            if attrs.get(key) not in (None, "", {}, [])
        }
    )


def _pointer_summary(attrs: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(attrs[key])
        for key in _POINTER_ATTRS
        if attrs.get(key) not in (None, "")
    }


def _walk_analysis_groups(zarr_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    analysis_path = zarr_path / "analysis"
    metadata_path = analysis_path / "zarr.json"
    if not metadata_path.is_file():
        return [], [f"missing analysis group metadata: {metadata_path}"]

    groups: list[dict[str, Any]] = []
    errors: list[str] = []
    stack: list[tuple[Path, dict[str, Any]]] = []
    try:
        stack.append((analysis_path, _read_json_object(metadata_path)))
    except Exception as exc:
        return [], [f"failed to read {metadata_path}: {exc}"]

    while stack:
        node_path, payload = stack.pop()
        if payload.get("node_type") != "group":
            continue
        relative_path = node_path.relative_to(zarr_path).as_posix()
        groups.append(
            {
                "node_path": relative_path,
                "name": node_path.name,
                "attrs": _attrs(payload),
            }
        )
        try:
            children = sorted(
                (child for child in node_path.iterdir() if child.is_dir()),
                key=lambda child: child.name,
                reverse=True,
            )
        except OSError as exc:
            errors.append(f"failed to list {node_path}: {exc}")
            continue
        for child in children:
            child_metadata = child / "zarr.json"
            if not child_metadata.is_file():
                continue
            try:
                child_payload = _read_json_object(child_metadata)
            except Exception as exc:
                errors.append(f"failed to read {child_metadata}: {exc}")
                continue
            if child_payload.get("node_type") == "group":
                stack.append((child, child_payload))
    return sorted(groups, key=lambda item: str(item["node_path"])), errors


def _nearest_family(node_path: str, family_paths: Sequence[str]) -> str | None:
    candidates = [
        family
        for family in family_paths
        if node_path == family or node_path.startswith(f"{family}/")
    ]
    return max(candidates, key=lambda value: len(value.split("/")), default=None)


def _classify_groups(
    raw_groups: list[dict[str, Any]],
    *,
    branding_tokens: Sequence[str],
) -> list[dict[str, Any]]:
    family_paths = sorted(
        {
            str(group["node_path"])
            for group in raw_groups
            if str(group["name"]).endswith("_runs")
        }
    )
    by_path = {str(group["node_path"]): group for group in raw_groups}
    records: list[dict[str, Any]] = []

    for group in raw_groups:
        node_path = str(group["node_path"])
        attrs = dict(group.get("attrs") or {})
        family_path = _nearest_family(node_path, family_paths)
        role = "analysis_group"
        run_parent_path: str | None = None
        run_path: str | None = None
        component_name: str | None = None
        component_family_path: str | None = None

        if family_path is not None:
            if node_path == family_path:
                role = "run_family"
                run_parent_path = family_path
            else:
                family_parts = family_path.split("/")
                node_parts = node_path.split("/")
                tail = node_parts[len(family_parts) :]
                scope_len = 1 if tail and tail[0] in _ANALYSIS_SCOPE_NAMES else 0
                if scope_len and len(tail) == 1:
                    role = "run_scope"
                    run_parent_path = node_path
                elif len(tail) > scope_len:
                    run_parent_parts = family_parts + tail[:scope_len]
                    run_parent_path = "/".join(run_parent_parts)
                    run_path = "/".join(run_parent_parts + [tail[scope_len]])
                    if node_path == run_path:
                        role = "run"
                    else:
                        component_tail = node_parts[len(run_path.split("/")) :]
                        if component_tail:
                            component_name = component_tail[0]
                            component_family_path = (
                                f"{run_parent_path}/*/{component_name}"
                            )
                            role = (
                                "run_child_group"
                                if len(component_tail) == 1
                                else "component_group"
                            )

        parent_path = node_path.rsplit("/", 1)[0] if "/" in node_path else ""
        parent_attrs = dict((by_path.get(parent_path) or {}).get("attrs") or {})
        selected_by = sorted(
            key
            for key, value in _pointer_summary(parent_attrs).items()
            if value == str(group["name"])
        )
        schema_id = attrs.get("schema_id")
        method = attrs.get("method")
        branding_fields = []
        if _contains_token(schema_id, branding_tokens):
            branding_fields.append("schema_id")
        if _contains_token(method, branding_tokens):
            branding_fields.append("method")
        if _contains_token(node_path, branding_tokens):
            branding_fields.append("path")

        records.append(
            _json_ready(
                {
                    "node_path": node_path,
                    "name": str(group["name"]),
                    "role": role,
                    "run_family_path": family_path,
                    "run_parent_path": run_parent_path,
                    "run_path": run_path,
                    "component_name": component_name,
                    "component_family_path": component_family_path,
                    "schema_id": schema_id,
                    "schema_version": attrs.get("schema_version"),
                    "method": method,
                    "method_version": attrs.get("method_version"),
                    "completion_status": attrs.get(_COMPLETION_STATUS_ATTR),
                    "completion_contract": attrs.get(_COMPLETION_CONTRACT_ATTR),
                    "created_at_utc": attrs.get("created_at_utc")
                    or attrs.get("created_utc"),
                    "row_axis": attrs.get("row_axis"),
                    "declares_contract": _declares_contract(attrs),
                    "pointers": _pointer_summary(attrs),
                    "selected_by_parent_pointers": selected_by,
                    "source": _source_summary(attrs),
                    "protocol_branding_fields": branding_fields,
                }
            )
        )
    return records


def inventory_archive(
    zarr_path: Path,
    *,
    dataset_id: str | None = None,
    recording_id: str | None = None,
    branding_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    """Inventory one analysis Zarr without opening it through Zarr Python."""

    resolved = zarr_path.expanduser().resolve()
    normalized_tokens = tuple(
        sorted({token for value in branding_tokens if (token := _normalize_token(value))})
    )
    raw_groups, errors = _walk_analysis_groups(resolved)
    groups = _classify_groups(raw_groups, branding_tokens=normalized_tokens)
    root_attrs: dict[str, Any] = {}
    root_metadata = resolved / "zarr.json"
    if root_metadata.is_file():
        try:
            root_attrs = _attrs(_read_json_object(root_metadata))
        except Exception as exc:
            errors.append(f"failed to read {root_metadata}: {exc}")

    resolved_recording_id = recording_id or str(
        root_attrs.get("recording_id") or root_attrs.get("recording_name") or ""
    )
    resolved_dataset_id = dataset_id or str(root_attrs.get("dataset_id") or "")
    return {
        "dataset_id": resolved_dataset_id or None,
        "recording_id": resolved_recording_id or None,
        "zarr_path": str(resolved),
        "group_count": len(groups),
        "run_family_count": len(
            {group["node_path"] for group in groups if group["role"] == "run_family"}
        ),
        "groups": groups,
        "errors": sorted(errors),
    }


def _summary_rows(
    archives: Sequence[Mapping[str, Any]],
    *,
    key: str,
    predicate,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "recording_ids": set(),
            "node_paths": set(),
            "schema_ids": set(),
            "methods": set(),
            "completion_statuses": defaultdict(int),
            "node_count": 0,
            "contract_node_count": 0,
            "schema_protocol_branded": False,
            "method_protocol_branded": False,
            "path_protocol_branded": False,
        }
    )
    for archive in archives:
        recording_id = str(archive.get("recording_id") or archive.get("dataset_id") or "")
        for group in archive.get("groups") or []:
            if not isinstance(group, Mapping) or not predicate(group):
                continue
            value = group.get(key)
            if value in (None, ""):
                continue
            bucket = buckets[str(value)]
            bucket["recording_ids"].add(recording_id)
            bucket["node_paths"].add(str(group.get("node_path") or ""))
            if group.get("schema_id"):
                bucket["schema_ids"].add(str(group["schema_id"]))
            if group.get("method"):
                bucket["methods"].add(str(group["method"]))
            status = str(group.get("completion_status") or "unmarked")
            bucket["completion_statuses"][status] += 1
            bucket["node_count"] += 1
            bucket["contract_node_count"] += int(bool(group.get("declares_contract")))
            branding = set(group.get("protocol_branding_fields") or [])
            bucket["schema_protocol_branded"] |= "schema_id" in branding
            bucket["method_protocol_branded"] |= "method" in branding
            bucket["path_protocol_branded"] |= "path" in branding

    rows = []
    for value, bucket in buckets.items():
        rows.append(
            {
                key: value,
                "recording_count": len(bucket["recording_ids"]),
                "node_count": int(bucket["node_count"]),
                "contract_node_count": int(bucket["contract_node_count"]),
                "schema_ids": sorted(bucket["schema_ids"]),
                "methods": sorted(bucket["methods"]),
                "completion_statuses": dict(
                    sorted(bucket["completion_statuses"].items())
                ),
                "schema_protocol_branded": bool(bucket["schema_protocol_branded"]),
                "method_protocol_branded": bool(bucket["method_protocol_branded"]),
                "path_protocol_branded": bool(bucket["path_protocol_branded"]),
                "example_paths": sorted(bucket["node_paths"])[:3],
            }
        )
    return sorted(rows, key=lambda row: (-int(row["recording_count"]), str(row[key])))


def build_cohort_inventory(
    targets: Sequence[Mapping[str, Any]],
    *,
    branding_tokens: Sequence[str] = (),
    selection: Mapping[str, Any] | None = None,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Inventory and aggregate a collection of analysis-Zarr targets."""

    archives = []
    for index, target in enumerate(targets, start=1):
        if progress is not None:
            progress(index, len(targets), target)
        archives.append(
            inventory_archive(
                Path(str(target["zarr_path"])),
                dataset_id=str(target.get("dataset_id") or "") or None,
                recording_id=str(target.get("recording_id") or "") or None,
                branding_tokens=branding_tokens,
            )
        )
    family_rows = _summary_rows(
        archives,
        key="run_family_path",
        predicate=lambda group: group.get("role") == "run",
    )
    component_rows = _summary_rows(
        archives,
        key="component_family_path",
        predicate=lambda group: group.get("component_family_path") is not None,
    )
    schema_rows = _summary_rows(
        archives,
        key="schema_id",
        predicate=lambda group: group.get("schema_id") not in (None, ""),
    )
    summary = {
        "archive_count": len(archives),
        "archive_error_count": sum(bool(archive.get("errors")) for archive in archives),
        "group_count": sum(int(archive.get("group_count") or 0) for archive in archives),
        "run_family_count": len(family_rows),
        "component_family_count": len(component_rows),
        "declared_component_family_count": sum(
            int(row["contract_node_count"] > 0) for row in component_rows
        ),
        "schema_count": len(schema_rows),
        "protocol_branded_schema_count": sum(
            int(row["schema_protocol_branded"]) for row in schema_rows
        ),
        "protocol_branded_method_schema_count": sum(
            int(row["method_protocol_branded"]) for row in schema_rows
        ),
    }
    return _json_ready(
        {
            "schema_id": INVENTORY_SCHEMA_ID,
            "created_at_utc": _utc_now(),
            "selection": dict(selection or {}),
            "branding_tokens": sorted(
                {_normalize_token(value) for value in branding_tokens if value}
            ),
            "summary": summary,
            "run_families": family_rows,
            "component_families": component_rows,
            "schemas": schema_rows,
            "archives": archives,
        }
    )


def load_registry_targets(
    registry_path: Path,
    *,
    protocol_name: str | None = None,
    recording_like: str | None = None,
) -> list[dict[str, str]]:
    """Load canonical active analysis targets through a read-only connection."""

    if not protocol_name and not recording_like:
        raise ValueError("registry selection requires --protocol-name or --recording-like")
    resolved = registry_path.expanduser().resolve()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON;")
        clauses = ["dataset_status = 'active'", "zarr_use = 'analysis'"]
        parameters: list[str] = []
        if protocol_name:
            clauses.append("protocol_name = ?")
            parameters.append(str(protocol_name))
        if recording_like:
            clauses.append("recording_id LIKE ?")
            parameters.append(str(recording_like))
        rows = connection.execute(
            "SELECT dataset_id, recording_id, zarr_path "
            "FROM dataset_context_current WHERE "
            + " AND ".join(clauses)
            + " ORDER BY recording_id, dataset_id",
            parameters,
        ).fetchall()
    finally:
        connection.close()
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for dataset_id, recording_id, zarr_path in rows:
        path = str(Path(str(zarr_path)).expanduser().resolve())
        if path in seen:
            continue
        seen.add(path)
        targets.append(
            {
                "dataset_id": str(dataset_id),
                "recording_id": str(recording_id),
                "zarr_path": path,
            }
        )
    return targets


def _markdown_cell(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, Mapping):
        text = ", ".join(f"{key}:{item}" for key, item in value.items())
    else:
        text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in rows
    )
    return lines


def render_markdown(inventory: Mapping[str, Any]) -> str:
    """Render a compact cohort report from a JSON inventory payload."""

    summary = inventory.get("summary") or {}
    lines = [
        "# Analysis component inventory",
        "",
        f"Generated: `{inventory.get('created_at_utc')}`",
        "",
        "This is a read-only physical/schema inventory. It does not update the registry, "
        "select scientific authority, or define new semantic stages.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Measure", "Count"],
            ((key, value) for key, value in summary.items()),
        )
    )
    lines.extend(["", "## Physical run families", ""])
    lines.extend(
        _markdown_table(
            ["Run family", "Recordings", "Runs", "Schemas", "Methods", "Completion"],
            (
                (
                    row["run_family_path"],
                    row["recording_count"],
                    row["node_count"],
                    row["schema_ids"],
                    row["methods"],
                    row["completion_statuses"],
                )
                for row in inventory.get("run_families") or []
            ),
        )
    )
    lines.extend(["", "## Nested component families", ""])
    lines.extend(
        _markdown_table(
            [
                "Component family",
                "Recordings",
                "Contract nodes",
                "Schemas",
                "Schema branded",
                "Method branded",
            ],
            (
                (
                    row["component_family_path"],
                    row["recording_count"],
                    row["contract_node_count"],
                    row["schema_ids"],
                    row["schema_protocol_branded"],
                    row["method_protocol_branded"],
                )
                for row in inventory.get("component_families") or []
            ),
        )
    )
    lines.extend(["", "## Declared schemas", ""])
    lines.extend(
        _markdown_table(
            ["Schema", "Recordings", "Nodes", "Methods", "Protocol branded"],
            (
                (
                    row["schema_id"],
                    row["recording_count"],
                    row["node_count"],
                    row["methods"],
                    row["schema_protocol_branded"]
                    or row["method_protocol_branded"],
                )
                for row in inventory.get("schemas") or []
            ),
        )
    )
    errors = [
        (archive.get("recording_id"), error)
        for archive in inventory.get("archives") or []
        for error in archive.get("errors") or []
    ]
    lines.extend(["", "## Archive errors", ""])
    if errors:
        lines.extend(_markdown_table(["Recording", "Error"], errors))
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def _path_targets(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [{"zarr_path": str(path.expanduser().resolve())} for path in paths]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Analysis Zarr paths.")
    parser.add_argument("--registry", type=Path, help="Read-only Palette registry path.")
    parser.add_argument("--protocol-name", help="Exact protocol_name registry selector.")
    parser.add_argument("--recording-like", help="Optional recording_id LIKE selector.")
    parser.add_argument(
        "--branding-token",
        action="append",
        default=[],
        help="Token to flag in schema IDs, methods, and instance paths; repeatable.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one read-only scan progress line per archive to stderr.",
    )
    args = parser.parse_args(argv)

    if args.registry:
        targets = load_registry_targets(
            args.registry,
            protocol_name=args.protocol_name,
            recording_like=args.recording_like,
        )
        selection = {
            "registry_path": str(args.registry.expanduser().resolve()),
            "protocol_name": args.protocol_name,
            "recording_like": args.recording_like,
        }
    elif args.paths:
        targets = _path_targets(args.paths)
        selection = {"paths": [target["zarr_path"] for target in targets]}
    else:
        parser.error("provide paths or --registry with a cohort selector")

    branding_tokens = list(args.branding_token)
    if args.protocol_name and not branding_tokens:
        branding_tokens.append(args.protocol_name)
    inventory = build_cohort_inventory(
        targets,
        branding_tokens=branding_tokens,
        selection=selection,
        progress=(
            lambda index, total, target: print(
                f"[{index}/{total}] {target.get('recording_id') or target['zarr_path']}",
                file=sys.stderr,
                flush=True,
            )
            if args.progress
            else None
        ),
    )
    markdown = render_markdown(inventory)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    if not args.output_json and not args.output_markdown:
        print(markdown)
    if args.fail_on_errors and inventory["summary"]["archive_error_count"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
