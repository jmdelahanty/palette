"""Filesystem-only audit of Zarr metadata object counts.

This module intentionally reads ``zarr.json`` files directly instead of opening
Zarr stores. It is safe to use in environments where synchronous Zarr group
opening can hang or where stale consolidated metadata may hide recently written
groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ArchiveSummary:
    zarr_path: str
    zarr_json_count: int
    group_count: int
    array_count: int
    unknown_count: int
    invalid_json_count: int
    metadata_bytes: int
    top_family_prefix: str | None
    top_family_zarr_json_count: int


@dataclass(frozen=True)
class PrefixSummary:
    zarr_path: str
    prefix_kind: str
    prefix: str
    zarr_json_count: int
    group_count: int
    array_count: int
    unknown_count: int
    invalid_json_count: int
    metadata_bytes: int


def _node_path(zarr_path: Path, zarr_json: Path) -> str:
    parent = zarr_json.parent
    try:
        rel = parent.relative_to(zarr_path)
    except ValueError:
        return str(parent)
    return "" if str(rel) == "." else rel.as_posix()


def _read_node_type(path: Path) -> tuple[str, bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid", True
    if not isinstance(payload, Mapping):
        return "unknown", False
    value = payload.get("node_type")
    if value in {"group", "array"}:
        return str(value), False
    return "unknown", False


def _family_prefix(node_path: str) -> str:
    if not node_path:
        return "<root>"
    parts = node_path.split("/")
    if parts[0] == "analysis" and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def _component_prefix(node_path: str) -> str:
    if not node_path:
        return "<root>"
    parts = node_path.split("/")
    if parts[0] == "analysis" and len(parts) >= 3:
        if len(parts) >= 4 and parts[1] == "track_kinematics_runs":
            return "/".join(parts[:4])
        return "/".join(parts[:3])
    return "/".join(parts[: min(len(parts), 2)])


def _empty_counter() -> Counter[str]:
    return Counter(
        {
            "zarr_json_count": 0,
            "group_count": 0,
            "array_count": 0,
            "unknown_count": 0,
            "invalid_json_count": 0,
            "metadata_bytes": 0,
        }
    )


def _bump(counter: Counter[str], *, node_type: str, invalid: bool, metadata_bytes: int) -> None:
    counter["zarr_json_count"] += 1
    counter["metadata_bytes"] += int(metadata_bytes)
    if node_type == "group":
        counter["group_count"] += 1
    elif node_type == "array":
        counter["array_count"] += 1
    else:
        counter["unknown_count"] += 1
    if invalid:
        counter["invalid_json_count"] += 1


def _summary_from_counter(
    *,
    zarr_path: Path,
    prefix_kind: str,
    prefix: str,
    counter: Counter[str],
) -> PrefixSummary:
    return PrefixSummary(
        zarr_path=str(zarr_path),
        prefix_kind=prefix_kind,
        prefix=prefix,
        zarr_json_count=int(counter["zarr_json_count"]),
        group_count=int(counter["group_count"]),
        array_count=int(counter["array_count"]),
        unknown_count=int(counter["unknown_count"]),
        invalid_json_count=int(counter["invalid_json_count"]),
        metadata_bytes=int(counter["metadata_bytes"]),
    )


def audit_archive(zarr_path: Path) -> tuple[ArchiveSummary, list[PrefixSummary], list[PrefixSummary]]:
    """Audit one Zarr directory by scanning its ``zarr.json`` metadata files."""

    zarr_path = zarr_path.expanduser().resolve()
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Zarr path is not a directory: {zarr_path}")

    archive_counter = _empty_counter()
    family_counters: dict[str, Counter[str]] = defaultdict(_empty_counter)
    component_counters: dict[str, Counter[str]] = defaultdict(_empty_counter)

    for zarr_json in sorted(zarr_path.rglob("zarr.json")):
        try:
            metadata_bytes = zarr_json.stat().st_size
        except OSError:
            metadata_bytes = 0
        node_type, invalid = _read_node_type(zarr_json)
        node_path = _node_path(zarr_path, zarr_json)
        family = _family_prefix(node_path)
        component = _component_prefix(node_path)
        _bump(archive_counter, node_type=node_type, invalid=invalid, metadata_bytes=metadata_bytes)
        _bump(family_counters[family], node_type=node_type, invalid=invalid, metadata_bytes=metadata_bytes)
        _bump(component_counters[component], node_type=node_type, invalid=invalid, metadata_bytes=metadata_bytes)

    family_summaries = [
        _summary_from_counter(
            zarr_path=zarr_path,
            prefix_kind="family",
            prefix=prefix,
            counter=counter,
        )
        for prefix, counter in sorted(
            family_counters.items(),
            key=lambda item: (-item[1]["zarr_json_count"], item[0]),
        )
    ]
    component_summaries = [
        _summary_from_counter(
            zarr_path=zarr_path,
            prefix_kind="component",
            prefix=prefix,
            counter=counter,
        )
        for prefix, counter in sorted(
            component_counters.items(),
            key=lambda item: (-item[1]["zarr_json_count"], item[0]),
        )
    ]
    top_family = family_summaries[0] if family_summaries else None
    archive_summary = ArchiveSummary(
        zarr_path=str(zarr_path),
        zarr_json_count=int(archive_counter["zarr_json_count"]),
        group_count=int(archive_counter["group_count"]),
        array_count=int(archive_counter["array_count"]),
        unknown_count=int(archive_counter["unknown_count"]),
        invalid_json_count=int(archive_counter["invalid_json_count"]),
        metadata_bytes=int(archive_counter["metadata_bytes"]),
        top_family_prefix=top_family.prefix if top_family is not None else None,
        top_family_zarr_json_count=top_family.zarr_json_count if top_family is not None else 0,
    )
    return archive_summary, family_summaries, component_summaries


def discover_analysis_zarrs(roots: Iterable[Path], *, recursive: bool = True) -> list[Path]:
    """Discover ``*_analysis.zarr`` archives under recording roots."""

    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root.name.endswith("_analysis.zarr"):
            candidates: list[Path] = [root]
        elif not root.is_dir():
            candidates = []
        elif recursive:
            candidates = []
            for dirpath, dirnames, _filenames in os.walk(root):
                base = Path(dirpath)
                for dirname in list(dirnames):
                    candidate = base / dirname
                    if dirname.endswith(".zarr"):
                        if dirname.endswith("_analysis.zarr"):
                            candidates.append(candidate)
                        # Do not recurse into any Zarr store during discovery.
                        dirnames.remove(dirname)
        else:
            candidates = list(root.glob("*_analysis.zarr"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_dir():
                continue
            if not (resolved / "zarr.json").is_file():
                continue
            seen.add(resolved)
            found.append(resolved)
    return sorted(found)


def _merge_prefix_summaries(rows: Iterable[PrefixSummary], *, prefix_kind: str) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(_empty_counter)
    for row in rows:
        counter = counters[row.prefix]
        counter["zarr_json_count"] += row.zarr_json_count
        counter["group_count"] += row.group_count
        counter["array_count"] += row.array_count
        counter["unknown_count"] += row.unknown_count
        counter["invalid_json_count"] += row.invalid_json_count
        counter["metadata_bytes"] += row.metadata_bytes
    merged: list[dict[str, Any]] = []
    for prefix, counter in sorted(
        counters.items(),
        key=lambda item: (-item[1]["zarr_json_count"], item[0]),
    ):
        merged.append(
            {
                "prefix_kind": prefix_kind,
                "prefix": prefix,
                "zarr_json_count": int(counter["zarr_json_count"]),
                "group_count": int(counter["group_count"]),
                "array_count": int(counter["array_count"]),
                "unknown_count": int(counter["unknown_count"]),
                "invalid_json_count": int(counter["invalid_json_count"]),
                "metadata_bytes": int(counter["metadata_bytes"]),
            }
        )
    return merged


def build_report(zarr_paths: Sequence[Path]) -> dict[str, Any]:
    archive_rows: list[ArchiveSummary] = []
    family_rows: list[PrefixSummary] = []
    component_rows: list[PrefixSummary] = []
    for zarr_path in zarr_paths:
        archive, families, components = audit_archive(zarr_path)
        archive_rows.append(archive)
        family_rows.extend(families)
        component_rows.extend(components)

    totals = {
        "archive_count": len(archive_rows),
        "zarr_json_count": sum(row.zarr_json_count for row in archive_rows),
        "group_count": sum(row.group_count for row in archive_rows),
        "array_count": sum(row.array_count for row in archive_rows),
        "unknown_count": sum(row.unknown_count for row in archive_rows),
        "invalid_json_count": sum(row.invalid_json_count for row in archive_rows),
        "metadata_bytes": sum(row.metadata_bytes for row in archive_rows),
    }
    return {
        "totals": totals,
        "archives": [asdict(row) for row in sorted(archive_rows, key=lambda item: item.zarr_path)],
        "families": [asdict(row) for row in family_rows],
        "components": [asdict(row) for row in component_rows],
        "global_families": _merge_prefix_summaries(family_rows, prefix_kind="family"),
        "global_components": _merge_prefix_summaries(component_rows, prefix_kind="component"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def render_markdown(report: Mapping[str, Any], *, top_n: int = 10) -> str:
    totals = report["totals"]
    archives = sorted(
        report["archives"],
        key=lambda row: (-int(row["zarr_json_count"]), row["zarr_path"]),
    )
    lines = [
        "# Analysis Zarr Object Count Audit",
        "",
        "## Totals",
        "",
        f"- analysis stores: `{totals['archive_count']}`",
        f"- zarr.json files: `{totals['zarr_json_count']}`",
        f"- groups: `{totals['group_count']}`",
        f"- arrays: `{totals['array_count']}`",
        f"- unknown/invalid metadata nodes: `{totals['unknown_count']}` / `{totals['invalid_json_count']}`",
        f"- metadata bytes: `{totals['metadata_bytes']}`",
        "",
        "## Largest Archives",
        "",
    ]
    lines.extend(
        _markdown_table(
            archives[:top_n],
            [
                "zarr_path",
                "zarr_json_count",
                "group_count",
                "array_count",
                "top_family_prefix",
                "top_family_zarr_json_count",
            ],
        )
    )
    lines.extend(["", "## Top Families Across Archives", ""])
    lines.extend(
        _markdown_table(
            list(report["global_families"])[:top_n],
            ["prefix", "zarr_json_count", "group_count", "array_count", "metadata_bytes"],
        )
    )
    lines.extend(["", "## Top Components Across Archives", ""])
    lines.extend(
        _markdown_table(
            list(report["global_components"])[:top_n],
            ["prefix", "zarr_json_count", "group_count", "array_count", "metadata_bytes"],
        )
    )
    return "\n".join(lines) + "\n"


def write_outputs(report: Mapping[str, Any], output_dir: Path, *, top_n: int = 10) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "audit_summary.md").write_text(
        render_markdown(report, top_n=top_n),
        encoding="utf-8",
    )
    _write_csv(output_dir / "archive_summary.csv", report["archives"])
    _write_csv(output_dir / "family_summary.csv", report["families"])
    _write_csv(output_dir / "component_summary.csv", report["components"])
    _write_csv(output_dir / "global_family_summary.csv", report["global_families"])
    _write_csv(output_dir / "global_component_summary.csv", report["global_components"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Zarr zarr.json group/array object counts without opening Zarr stores.",
    )
    parser.add_argument("--zarr", action="append", type=Path, help="Analysis Zarr path. May be repeated.")
    parser.add_argument("--recordings-root", action="append", type=Path, help="Root to scan for *_analysis.zarr archives.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recursively scan recordings roots.")
    parser.add_argument("--limit", type=int, help="Limit discovered archives, useful for canaries.")
    parser.add_argument("--output-dir", type=Path, help="Write CSV, JSON, and Markdown diagnostics to this directory.")
    parser.add_argument("--top-n", type=int, default=10, help="Markdown/stdout top-N rows.")
    parser.add_argument("--format", choices=("table", "json", "markdown"), default="table")
    return parser


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    explicit = list(args.zarr or [])
    for path in explicit:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Analysis Zarr path is not a directory: {resolved}")
        paths.append(resolved)
    roots = list(args.recordings_root or [])
    if roots:
        paths.extend(discover_analysis_zarrs(roots, recursive=not bool(args.no_recursive)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if args.limit is not None and int(args.limit) >= 0:
        unique = unique[: int(args.limit)]
    if not unique:
        raise SystemExit("No analysis Zarr archives were provided or discovered.")
    return unique


def _print_table(report: Mapping[str, Any], *, top_n: int) -> None:
    totals = report["totals"]
    print(
        "archives\tzarr_json\tgroups\tarrays\tunknown\tinvalid\tmetadata_bytes"
    )
    print(
        f"{totals['archive_count']}\t{totals['zarr_json_count']}\t"
        f"{totals['group_count']}\t{totals['array_count']}\t"
        f"{totals['unknown_count']}\t{totals['invalid_json_count']}\t"
        f"{totals['metadata_bytes']}"
    )
    print()
    print("top_family\tzarr_json\tgroups\tarrays\tmetadata_bytes")
    for row in list(report["global_families"])[:top_n]:
        print(
            f"{row['prefix']}\t{row['zarr_json_count']}\t{row['group_count']}\t"
            f"{row['array_count']}\t{row['metadata_bytes']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    zarr_paths = _resolve_inputs(args)
    report = build_report(zarr_paths)
    if args.output_dir is not None:
        write_outputs(report, args.output_dir.expanduser().resolve(), top_n=int(args.top_n))
    if args.format == "json":
        print(json.dumps(report, sort_keys=True))
    elif args.format == "markdown":
        print(render_markdown(report, top_n=int(args.top_n)))
    else:
        _print_table(report, top_n=int(args.top_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
