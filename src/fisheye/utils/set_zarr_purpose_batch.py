#!/usr/bin/env python3
"""Batch-set root zarr_purpose on Zarr archives."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import zarr


@dataclass
class UpdateResult:
    zarr_path: Path
    status: str
    previous: Optional[str] = None
    updated: Optional[str] = None
    detail: Optional[str] = None
    dataset_id: Optional[str] = None


def _load_file_list(path: Path) -> List[Path]:
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _open_group(path: Path, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode)


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for *.zarr.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--purpose",
        default="training",
        choices=["training", "analysis", "production"],
        help="zarr_purpose value to set (default: training).",
    )
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run).")
    parser.add_argument(
        "--registry",
        type=Path,
        help="Registry path used with --rescan-registry (default: env-derived).",
    )
    parser.add_argument(
        "--rescan-registry",
        action="store_true",
        help="After apply, rescan changed zarrs into registry.",
    )
    args = parser.parse_args(argv)

    roots: List[Path] = []
    if args.file_list:
        for file_list in args.file_list:
            roots.extend(_load_file_list(file_list))
    roots.extend(args.paths)
    if not roots:
        roots = [Path("/nvme1/recordings")]

    zarr_paths = sorted({path.resolve() for path in _iter_zarr(roots, args.recursive)})
    if not zarr_paths:
        print("No Zarr paths found.")
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: target purpose='{args.purpose}'")
    print(f"Found {len(zarr_paths)} zarr path(s).")

    results: List[UpdateResult] = []
    changed_paths: List[Path] = []
    for zarr_path in zarr_paths:
        try:
            root = _open_group(zarr_path, "a" if args.apply else "r")
            previous = _normalize_text(root.attrs.get("zarr_purpose"))
            if previous == args.purpose:
                results.append(
                    UpdateResult(
                        zarr_path=zarr_path,
                        status="unchanged",
                        previous=previous,
                        updated=args.purpose,
                    )
                )
                continue

            if args.apply:
                attrs = dict(root.attrs)
                attrs["zarr_purpose"] = args.purpose
                root.attrs.put(attrs)
                changed_paths.append(zarr_path)
                status = "updated"
            else:
                status = "would_update"
            results.append(
                UpdateResult(
                    zarr_path=zarr_path,
                    status=status,
                    previous=previous,
                    updated=args.purpose,
                )
            )
        except Exception as exc:
            results.append(
                UpdateResult(
                    zarr_path=zarr_path,
                    status="error",
                    detail=str(exc),
                )
            )

    for result in results:
        if result.status == "error":
            print(f"error       {result.zarr_path} ({result.detail})")
            continue
        prev = result.previous if result.previous is not None else "None"
        print(f"{result.status:11s} {result.zarr_path} ({prev} -> {result.updated})")

    summary = {"updated": 0, "would_update": 0, "unchanged": 0, "error": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1

    rescanned = 0
    if args.apply and args.rescan_registry and changed_paths:
        from fisheye.registry.db import Registry, RegistryPaths

        registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        print(f"Rescanning {len(changed_paths)} changed zarr(s) into registry: {registry_path}")
        registry = Registry(registry_path)
        try:
            for zarr_path in changed_paths:
                try:
                    dataset_id = registry.scan_zarr(zarr_path)
                except Exception as exc:
                    print(f"rescan_error {zarr_path} ({exc})")
                    continue
                rescanned += 1
                print(f"rescanned   {zarr_path} ({dataset_id or 'no_dataset_id'})")
        finally:
            registry.close()

    print("\nSummary:")
    print(f"  updated: {summary.get('updated', 0)}")
    print(f"  would_update: {summary.get('would_update', 0)}")
    print(f"  unchanged: {summary.get('unchanged', 0)}")
    print(f"  errors: {summary.get('error', 0)}")
    if args.apply and args.rescan_registry:
        print(f"  rescanned: {rescanned}")

    if summary.get("error", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
