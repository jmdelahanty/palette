#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
from collections.abc import Mapping

import zarr
import numpy as np

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs, load_path_list


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


def _select_latest(parent: zarr.Group) -> Optional[str]:
    latest = parent.attrs.get("latest")
    if latest and latest in parent:
        return str(latest)
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        return None
    return sorted(names)[-1]


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

    if attrs:
        return attrs
    zattrs = group_path / ".zattrs"
    if zattrs.exists():
        try:
            data = json.loads(zattrs.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}
    return {}


def _pick_stats_block(summary: Dict[str, object]) -> Dict[str, object]:
    for key in ("postprocess", "refine"):
        block = summary.get(key)
        if isinstance(block, Mapping):
            return dict(block)
    return summary


def _summarize_stats(run: zarr.Group) -> Optional[Dict[str, object]]:
    summary = _coerce_mapping(run.attrs.get("summary_statistics"))
    if not summary:
        return None
    summary_block = _pick_stats_block(summary)
    total = summary_block.get("total_rois") or summary_block.get("total")
    refined_success = summary_block.get("refined_success")
    usable = summary_block.get("usable_keypoints") or summary_block.get("usable")
    success_rate = summary_block.get("success_rate_percent") or summary_block.get("pass_rate_percent")
    usable_rate = None
    try:
        total_val = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_val = None
    try:
        usable_val = float(usable) if usable is not None else None
    except (TypeError, ValueError):
        usable_val = None
    if total_val and usable_val is not None:
        usable_rate = usable_val / total_val * 100.0
    return {
        "total": total,
        "refined_success": refined_success,
        "usable": usable,
        "success_rate_percent": success_rate,
        "usable_rate_percent": usable_rate,
    }


def _show_one(path: Path, refined_run: Optional[str], as_json: bool, show_raw: bool) -> None:
    try:
        root = zarr.open_group(str(path), mode="r", consolidated=False)
    except TypeError:
        root = zarr.open_group(str(path), mode="r")
    refined_parent = _pick_refined_parent(root)
    if refined_parent is None:
        if as_json:
            print(json.dumps({"zarr": str(path), "status": "missing_refined_keypoints"}))
        else:
            print(f"{path}: missing refined_keypoints_runs")
        return

    latest = _select_latest(refined_parent)
    group_name = _pick_refined_group_name(root)
    status_latest = refined_parent.attrs.get("keypoint_review_status_latest")
    run_name = refined_run or latest
    if not run_name or run_name not in refined_parent:
        if as_json:
            print(
                json.dumps(
                    {
                        "zarr": str(path),
                        "status": "missing_run",
                        "refined_latest": latest,
                        "review_status_latest": status_latest,
                    }
                )
            )
        else:
            print(f"{path}: refined run not found (latest={latest})")
        return

    run = refined_parent[run_name]
    raw_review_status = run.attrs.get("keypoint_review_status")
    review_status = _coerce_mapping(raw_review_status)
    file_review_status = None
    if review_status is None and group_name:
        attrs = _load_group_attrs(Path(path) / group_name / run_name)
        file_review_status = attrs.get("keypoint_review_status")
        review_status = _coerce_mapping(file_review_status)
    stats = _summarize_stats(run)
    payload = {
        "zarr": str(path),
        "refined_latest": latest,
        "review_status_latest": status_latest,
        "refined_run": run_name,
        "review_status": review_status,
        "summary": stats,
    }
    if show_raw:
        payload["review_status_raw"] = repr(raw_review_status)
        payload["review_status_raw_type"] = str(type(raw_review_status))
        payload["review_status_file"] = repr(file_review_status)
        payload["review_status_file_type"] = str(type(file_review_status))

    if as_json:
        print(json.dumps(payload))
        return

    print(path)
    print(f"  refined_latest: {latest or 'none'}")
    print(f"  review_status_latest: {status_latest or 'none'}")
    print(f"  refined_run: {run_name}")
    print(f"  review_status: {review_status or '—'}")
    if show_raw:
        print(f"  review_status_raw: {repr(raw_review_status)}")
        print(f"  review_status_raw_type: {type(raw_review_status)}")
        print(f"  review_status_file: {repr(file_review_status)}")
        print(f"  review_status_file_type: {type(file_review_status)}")
    if stats:
        summary_parts = [
            f"total={stats.get('total')}",
            f"refined_success={stats.get('refined_success')}",
            f"success_rate_percent={stats.get('success_rate_percent')}",
            f"usable={stats.get('usable')}",
            f"usable_rate_percent={stats.get('usable_rate_percent')}",
        ]
        print(f"  summary: {', '.join(summary_parts)}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show keypoint_review_status and latest refined keypoint run metadata."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Zarr path(s) or recording roots.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for Zarrs.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument("--refined-run", help="Specific refined keypoint run to show.")
    parser.add_argument("--json", action="store_true", help="Emit JSON per Zarr.")
    parser.add_argument("--show-raw", action="store_true", help="Show raw review_status attr value and type.")
    args = parser.parse_args(argv)

    roots: list[Path] = []
    if args.file_list:
        for file_list in args.file_list:
            roots.extend(load_path_list(file_list))
    if args.paths:
        roots.extend(args.paths)
    if not roots:
        roots = [Path("/nvme1/recordings")]

    zarr_paths = list(iter_filesystem_zarrs(roots, args.recursive))
    if not zarr_paths:
        print("No Zarr paths found.")
        return 1

    for path in sorted({p.resolve() for p in zarr_paths}):
        _show_one(path, args.refined_run, args.json, args.show_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
