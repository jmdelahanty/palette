#!/usr/bin/env python3
"""Compare analysis and training Zarr roots for one recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import zarr


DEFAULT_ATTR_KEYS = (
    "zarr_purpose",
    "source_video",
    "source_video_path",
    "source_path",
    "width",
    "height",
    "fps",
    "total_frames",
    "n_frames",
    "duration_seconds",
    "video_codec",
    "video_pix_fmt",
    "analysis_archive_created_utc",
    "analysis_archive_updated_utc",
    "pipeline_type",
    "model_name",
)


def _as_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _resolve_paths(
    *,
    recording_dir: Optional[Path],
    analysis: Optional[Path],
    training: Optional[Path],
) -> tuple[Path, Path]:
    if analysis is not None and training is not None:
        return analysis.expanduser().resolve(), training.expanduser().resolve()

    if recording_dir is None:
        raise SystemExit("Provide --recording-dir, or both --analysis and --training.")

    rec = recording_dir.expanduser().resolve()
    name = rec.name
    zarr_dir = rec / "zarr"
    return (
        (analysis or (zarr_dir / f"{name}_analysis.zarr")).resolve(),
        (training or (zarr_dir / f"{name}_training.zarr")).resolve(),
    )


def _read_root(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "groups": [],
        "attr_count": 0,
        "attrs": {},
    }
    if not path.exists():
        return result

    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    result["groups"] = sorted(root.group_keys())
    result["attr_count"] = len(attrs)
    result["attrs"] = {str(k): _as_jsonable(v) for k, v in attrs.items()}
    return result


def _select_attrs(info: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    attrs = info.get("attrs", {})
    selected: dict[str, Any] = {}
    for key in keys:
        selected[key] = attrs.get(key)
    return selected


def _build_diff(
    analysis_info: dict[str, Any],
    training_info: dict[str, Any],
) -> dict[str, Any]:
    a_attrs: dict[str, Any] = analysis_info.get("attrs", {})
    t_attrs: dict[str, Any] = training_info.get("attrs", {})
    a_keys = set(a_attrs.keys())
    t_keys = set(t_attrs.keys())

    shared = sorted(a_keys & t_keys)
    value_diff: list[str] = [k for k in shared if a_attrs.get(k) != t_attrs.get(k)]

    return {
        "attr_keys_only_in_analysis": sorted(a_keys - t_keys),
        "attr_keys_only_in_training": sorted(t_keys - a_keys),
        "attr_keys_shared": shared,
        "attr_value_differences": value_diff,
        "attr_value_difference_count": len(value_diff),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, help="Recording directory containing zarr/*.zarr.")
    parser.add_argument("--analysis", type=Path, help="Explicit *_analysis.zarr path.")
    parser.add_argument("--training", type=Path, help="Explicit *_training.zarr path.")
    parser.add_argument(
        "--attr",
        action="append",
        default=[],
        help="Extra root attr key to print (can be provided multiple times).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    parser.add_argument(
        "--diff-limit",
        type=int,
        default=40,
        help="Max keys to print for each key-diff section in text mode.",
    )
    args = parser.parse_args(argv)

    analysis_path, training_path = _resolve_paths(
        recording_dir=args.recording_dir,
        analysis=args.analysis,
        training=args.training,
    )
    keys = list(DEFAULT_ATTR_KEYS) + list(args.attr or [])

    analysis_info = _read_root(analysis_path)
    training_info = _read_root(training_path)
    diff = _build_diff(analysis_info, training_info)

    payload = {
        "analysis": {
            "path": analysis_info["path"],
            "exists": analysis_info["exists"],
            "groups": analysis_info["groups"],
            "attr_count": analysis_info["attr_count"],
            "selected_attrs": _select_attrs(analysis_info, keys),
        },
        "training": {
            "path": training_info["path"],
            "exists": training_info["exists"],
            "groups": training_info["groups"],
            "attr_count": training_info["attr_count"],
            "selected_attrs": _select_attrs(training_info, keys),
        },
        "diff": diff,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"[analysis] {analysis_info['path']}")
    print(f"  exists: {analysis_info['exists']}")
    if analysis_info["exists"]:
        print(f"  groups: {analysis_info['groups']}")
        print(f"  attr_count: {analysis_info['attr_count']}")
        for key, value in payload["analysis"]["selected_attrs"].items():
            print(f"  {key}: {value}")

    print(f"[training] {training_info['path']}")
    print(f"  exists: {training_info['exists']}")
    if training_info["exists"]:
        print(f"  groups: {training_info['groups']}")
        print(f"  attr_count: {training_info['attr_count']}")
        for key, value in payload["training"]["selected_attrs"].items():
            print(f"  {key}: {value}")

    only_a = diff["attr_keys_only_in_analysis"]
    only_t = diff["attr_keys_only_in_training"]
    print(f"[diff] attr keys only in analysis: {len(only_a)}")
    print(f"  {only_a[: max(0, args.diff_limit)]}")
    print(f"[diff] attr keys only in training: {len(only_t)}")
    print(f"  {only_t[: max(0, args.diff_limit)]}")
    print(f"[diff] attr value differences on shared keys: {diff['attr_value_difference_count']}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
