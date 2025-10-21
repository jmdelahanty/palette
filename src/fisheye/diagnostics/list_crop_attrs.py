"""Inspect crop run metadata inside a Palette Zarr archive.

This helper lists the attributes stored on the top-level ``crop_runs`` group
and each child crop run, making it easy to confirm whether ``.attrs`` have
been preserved during archive transfers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import zarr


def _sorted_group_items(group: zarr.Group) -> Iterable[Tuple[str, zarr.Group]]:
    """Yield (name, group) pairs for sub-groups, sorted lexicographically."""
    if not hasattr(group, "group_keys"):
        return []
    try:
        names = sorted(group.group_keys())
    except Exception:
        return []
    return [(name, group[name]) for name in names if isinstance(group[name], zarr.Group)]


def _attrs_to_dict(attrs: zarr.attrs.Attributes) -> Dict[str, object]:
    """Convert Zarr attributes to a plain dict for pretty-printing."""
    return {key: attrs[key] for key in attrs.keys()}


def inspect_crop_runs(zarr_path: Path, crop_group: str = "crop_runs") -> None:
    store = zarr.open(str(zarr_path), mode="r")
    if crop_group not in store:
        print(f"[!] Group '{crop_group}' not found in {zarr_path}")
        return

    crop_parent = store[crop_group]
    parent_attrs = _attrs_to_dict(crop_parent.attrs)
    print(f"[+] {crop_group} attrs: {json.dumps(parent_attrs, indent=2, sort_keys=True) or '{}'}")

    groups = list(_sorted_group_items(crop_parent))
    if not groups:
        print(f"[!] No crop run sub-groups found under '{crop_group}'.")
        return

    print(f"[+] Found {len(groups)} crop run(s):")
    for name, group in groups:
        attrs = _attrs_to_dict(group.attrs)
        print(f"  - {name}")
        if attrs:
            print(json.dumps(attrs, indent=4, sort_keys=True))
        else:
            print("    (no attrs stored)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--crop-group",
        default="crop_runs",
        help="Name of the crop runs group (default: crop_runs).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    inspect_crop_runs(args.zarr_path, args.crop_group)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
