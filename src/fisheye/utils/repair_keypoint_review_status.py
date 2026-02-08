#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional
from collections.abc import Mapping

import numpy as np
import zarr


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def _pick_refined_group_name(root: zarr.Group) -> Optional[str]:
    if "refined_keypoints_runs" in root:
        return "refined_keypoints_runs"
    if "keypoints_refined_runs" in root:
        return "keypoints_refined_runs"
    return None


def _coerce_mapping(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.flat[0]
        else:
            try:
                return dict(value.tolist())  # type: ignore[arg-type]
            except Exception:
                return None
    if isinstance(value, Mapping):
        return dict(value)
    try:
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, (list, tuple)):
        try:
            return dict(value)
        except Exception:
            return None
    return None


def _iter_zarr(paths: Iterable[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_dir() and path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def _load_paths_file(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _load_group_attrs(group_path: Path) -> Dict[str, object]:
    zarr_json = group_path / "zarr.json"
    attrs: Dict[str, object] = {}
    if zarr_json.exists():
        try:
            data = json.loads(zarr_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        attrs_raw = data.get("attributes") if isinstance(data, dict) else None
        if isinstance(attrs_raw, dict):
            attrs = dict(attrs_raw)

    parent_zarr = group_path.parent / "zarr.json"
    if parent_zarr.exists():
        try:
            parent_data = json.loads(parent_zarr.read_text(encoding="utf-8"))
        except Exception:
            parent_data = {}
        meta = None
        if isinstance(parent_data, dict):
            meta = parent_data.get("consolidated_metadata", {}).get("metadata")
        if isinstance(meta, dict):
            entry = meta.get(group_path.name)
            if isinstance(entry, dict):
                child_attrs = entry.get("attributes")
                if isinstance(child_attrs, dict):
                    for key, value in child_attrs.items():
                        attrs.setdefault(key, value)
    return attrs


def _repair_one(path: Path, refined_run: Optional[str], dry_run: bool, no_latest: bool) -> Dict[str, int]:
    counts = {"checked": 0, "already_set": 0, "repaired": 0, "missing": 0}
    try:
        root = zarr.open_group(str(path), mode="r" if dry_run else "a", consolidated=False)
    except TypeError:
        root = zarr.open_group(str(path), mode="r" if dry_run else "a")

    refined_parent = _pick_refined_parent(root)
    refined_group_name = _pick_refined_group_name(root)
    if refined_parent is None or refined_group_name is None:
        return counts

    run_names = [str(refined_run)] if refined_run else list(refined_parent.group_keys())
    for run_name in run_names:
        if run_name not in refined_parent:
            continue
        counts["checked"] += 1
        run = refined_parent[run_name]
        existing = _coerce_mapping(run.attrs.get("keypoint_review_status"))
        if existing:
            counts["already_set"] += 1
            continue

        on_disk_attrs = _load_group_attrs(path / refined_group_name / run_name)
        recovered = _coerce_mapping(on_disk_attrs.get("keypoint_review_status"))
        if not recovered:
            counts["missing"] += 1
            continue

        if not dry_run:
            run_attrs = dict(run.attrs)
            run_attrs["keypoint_review_status"] = recovered
            run.attrs.put(run_attrs)
            if not no_latest:
                parent_attrs = dict(refined_parent.attrs)
                latest_refined = parent_attrs.get("latest")
                latest_status = parent_attrs.get("keypoint_review_status_latest")
                if latest_status is None or latest_status == "" or str(latest_refined) == run_name:
                    parent_attrs["keypoint_review_status_latest"] = run_name
                    refined_parent.attrs.put(parent_attrs)
        counts["repaired"] += 1

    return counts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing refined keypoint review statuses from on-disk zarr metadata."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Zarr path(s) or recording roots.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for Zarrs.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument("--refined-run", help="Only repair this refined run name.")
    parser.add_argument("--dry-run", action="store_true", help="Report repairs without writing.")
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update parent keypoint_review_status_latest when repairing.",
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero if unresolved missing rows remain.")
    args = parser.parse_args(argv)

    roots: list[Path] = []
    if args.file_list:
        for file_list in args.file_list:
            roots.extend(_load_paths_file(file_list))
    if args.paths:
        roots.extend(args.paths)
    if not roots:
        roots = [Path("/nvme1/recordings")]

    zarr_paths = sorted({p.resolve() for p in _iter_zarr(roots, args.recursive)})
    if not zarr_paths:
        print("No Zarr paths found.")
        return 1

    totals = {"checked": 0, "already_set": 0, "repaired": 0, "missing": 0}
    for zarr_path in zarr_paths:
        counts = _repair_one(zarr_path, args.refined_run, args.dry_run, args.no_latest)
        for key in totals:
            totals[key] += counts[key]
        if counts["repaired"] or counts["missing"]:
            action = "would_repair" if args.dry_run else "repaired"
            print(
                f"{zarr_path}: checked={counts['checked']} already_set={counts['already_set']} "
                f"{action}={counts['repaired']} missing={counts['missing']}"
            )

    print(
        f"Summary: checked={totals['checked']} already_set={totals['already_set']} "
        f"{'would_repair' if args.dry_run else 'repaired'}={totals['repaired']} missing={totals['missing']}"
    )

    if args.strict and totals["missing"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
