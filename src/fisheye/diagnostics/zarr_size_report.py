# src/fisheye/diagnostics/zarr_size_report.py
"""Console report summarising on-disk sizes for each Zarr group/array.

Example::

    python -m fisheye.diagnostics.zarr_size_report /data/session.zarr

The script traverses the Zarr store using the Zarr API (no assumptions about the
filesystem layout) and prints a tree showing each group, array, its stored byte
count, and share of total size.  Use ``--max-depth`` to limit output and
``--sort`` to order children by size.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import zarr
from rich.console import Console


@dataclass
class Node:
    name: str
    kind: str  # "group" or "array"
    size: int
    path: str
    children: List["Node"] = field(default_factory=list)


def format_size(num_bytes: int) -> str:
    """Human readable byte count (e.g., 12.3 MB)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def build_tree(group: zarr.Group, name: str, path: str = "") -> Node:
    """Recursively build a tree of groups/arrays with their stored sizes."""
    node = Node(name=name or ".", kind="group", size=0, path=path)

    # Subgroups
    for key in group.group_keys():
        subgroup = group[key]
        child_path = f"{path}/{key}" if path else key
        child = build_tree(subgroup, key, child_path)
        node.children.append(child)
        node.size += child.size

    # Arrays
    for key in group.array_keys():
        array = group[key]
        stored = getattr(array, "nbytes_stored", None)
        if callable(stored):
            stored = stored()
        if stored is None or isinstance(stored, str):
            stored = getattr(array, "nbytes", 0)
        child_path = f"{path}/{key}" if path else key
        node.children.append(Node(name=key, kind="array", size=int(stored), path=child_path))
        node.size += int(stored)

    return node


def iter_sorted(children: Iterable[Node], sort_by_size: bool) -> List[Node]:
    if not sort_by_size:
        return list(children)
    return sorted(children, key=lambda item: item.size, reverse=True)


def print_tree(
    node: Node,
    *,
    console: Console,
    total_size: int,
    prefix: str = "",
    is_last: bool = True,
    depth: int = 0,
    max_depth: Optional[int] = None,
    sort_children: bool = True,
) -> None:
    """Pretty-print the report using box-drawing characters."""
    branch = ""
    child_prefix = prefix
    if depth > 0:
        branch = "└─ " if is_last else "├─ "
        child_prefix = prefix + ("   " if is_last else "│  ")

    label = "[G]" if node.kind == "group" else "[A]"
    percent = (node.size / total_size * 100) if total_size else 0.0
    console.print(
        f"{prefix}{branch}{label} {node.path or '.'} — {format_size(node.size)} ({percent:.2f}%)"
    )

    if max_depth is not None and depth >= max_depth:
        return

    children = iter_sorted(node.children, sort_children)
    for idx, child in enumerate(children):
        print_tree(
            child,
            console=console,
            total_size=total_size,
            prefix=child_prefix,
            is_last=(idx == len(children) - 1),
            depth=depth + 1,
            max_depth=max_depth,
            sort_children=sort_children,
        )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Report the on-disk size of every Zarr group/array.")
    parser.add_argument("zarr_path", help="Path to the Zarr store (directory or .zarr bundle).")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit tree expansion to this depth (0 = root only).",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Preserve original order instead of sorting children by size.",
    )
    args = parser.parse_args(argv)

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise SystemExit(f"Zarr path not found: {zarr_path}")

    try:
        root_group = zarr.open(zarr_path.as_posix(), mode="r")
    except Exception as exc:  # pragma: no cover - handled gracefully for CLI
        raise SystemExit(f"Failed to open Zarr archive: {exc}") from exc

    if not isinstance(root_group, zarr.Group):
        raise SystemExit(f"Path does not contain a Zarr group: {zarr_path}")

    tree = build_tree(root_group, name=zarr_path.name or ".")
    console = Console()
    console.print(f"[bold]Zarr size report for[/bold] {zarr_path}")
    console.print("-" * 80)
    print_tree(
        tree,
        console=console,
        total_size=tree.size,
        max_depth=args.max_depth,
        sort_children=not args.no_sort,
    )
    console.print("-" * 80)
    console.print(f"Total: {format_size(tree.size)}")


if __name__ == "__main__":  # pragma: no cover
    main()
