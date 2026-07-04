#!/usr/bin/env python3
"""
Backfill camera metadata and dish design from the source H5 into existing Zarrs.

Defaults to a dry-run; pass --apply to write.
"""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import zarr

from fisheye.analysis import import_stimulus_to_zarr as stim_import


def _read_file_list(path: Path) -> List[Path]:
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _source_h5_from_zarr(root: zarr.Group) -> Optional[Path]:
    analysis = root.get("analysis")
    if analysis is None:
        return None
    stim_parent = analysis.get("stimulus_runs")
    if stim_parent is None:
        return None
    latest = stim_parent.attrs.get("latest")
    if not latest or latest not in stim_parent:
        return None
    stim_group = stim_parent[latest]
    source_h5 = stim_group.attrs.get("source_h5")
    if not source_h5:
        return None
    return Path(str(source_h5))


def _resolve_h5_path(
    root: zarr.Group,
    zarr_path: Path,
    *,
    explicit: Optional[Path],
    h5_root: Optional[Path],
) -> Optional[Path]:
    if explicit is not None:
        return explicit
    source_h5 = _source_h5_from_zarr(root)
    if source_h5 and source_h5.exists():
        return source_h5
    if h5_root is not None:
        candidate = h5_root / f"{zarr_path.stem}.h5"
        if candidate.exists():
            return candidate
    candidate = zarr_path.parent.parent / "raw" / f"{zarr_path.stem}.h5"
    if candidate.exists():
        return candidate
    return None


def _load_dish_design(h5: h5py.File) -> Optional[str]:
    arena_config = stim_import._read_h5_arena_config(h5)
    if not isinstance(arena_config, dict):
        return None
    value = arena_config.get("selected_dish_type_name")
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or Zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for Zarrs.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        help="Explicit H5 path (only valid with a single Zarr).",
    )
    parser.add_argument("--h5-root", type=Path, help="Root folder to search for <zarr>.h5 files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata fields.")
    parser.add_argument("--apply", action="store_true", help="Write changes to Zarrs.")

    args = parser.parse_args(argv)

    roots: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            roots.extend(_read_file_list(path))
    roots.extend(args.paths)
    if not roots:
        raise SystemExit("No paths provided.")

    if args.h5_path is not None and len(roots) > 1:
        raise SystemExit("--h5-path can only be used with a single Zarr path.")

    updated = 0
    for zarr_path in _iter_zarr(roots, args.recursive):
        mode = "a" if args.apply else "r"
        try:
            root = zarr.open(str(zarr_path), mode=mode)
        except Exception as exc:
            print(f"\n{zarr_path}\n  error: {exc}")
            continue

        h5_path = _resolve_h5_path(root, zarr_path, explicit=args.h5_path, h5_root=args.h5_root)
        if h5_path is None or not h5_path.exists():
            print(f"\n{zarr_path}\n  h5: missing")
            continue

        with h5py.File(h5_path, "r") as h5:
            camera_meta = stim_import._read_h5_camera_metadata(h5)
            dish_design = _load_dish_design(h5)

        analysis = root.require_group("analysis_metadata") if args.apply else root.get("analysis_metadata")
        camera_meta_existing = analysis.attrs.get("camera_metadata") if analysis is not None else None
        dish_existing = root.attrs.get("dish_design")

        write_camera = camera_meta is not None and (args.overwrite or not camera_meta_existing)
        write_dish = dish_design and (args.overwrite or not dish_existing)

        print(f"\n{zarr_path}")
        print(f"  h5: {h5_path}")
        if camera_meta is None:
            print("  camera_metadata: not found in H5")
        else:
            print(f"  camera_metadata: {'update' if write_camera else 'skip'}")
        if dish_design:
            print(f"  dish_design: {'update' if write_dish else 'skip'} ({dish_design})")
        else:
            print("  dish_design: not found in H5")

        if args.apply:
            if write_camera:
                analysis = root.require_group("analysis_metadata")
                analysis.attrs["camera_metadata"] = json.dumps(camera_meta, sort_keys=True)
                analysis.attrs["camera_config_hash"] = stim_import._camera_metadata_hash(camera_meta)
            if write_dish:
                root.attrs["dish_design"] = dish_design
            if write_camera or write_dish:
                updated += 1

    if args.apply:
        print(f"\nUpdated {updated} Zarr(s).")
    else:
        print("\nDry run complete. Use --apply to write changes.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
