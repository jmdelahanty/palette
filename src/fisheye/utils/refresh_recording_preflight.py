#!/usr/bin/env python3
"""Refresh manifest preflight diagnostics for already organized recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from fisheye.utils.organize_recordings import (
    RecordingPlan,
    _persist_preflight_to_manifest,
    _run_h5_diagnostics_for_plan,
    _run_video_diagnostics_for_plan,
)


def _iter_manifest_paths(paths: Sequence[Path], *, recursive: bool) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        candidates: list[Path] = []
        if path.is_file():
            candidates.append(path)
        elif path.exists():
            direct = path / "recording_manifest.json"
            if direct.is_file():
                candidates.append(direct)
            if recursive:
                candidates.extend(path.rglob("recording_manifest.json"))
            else:
                candidates.extend(path.glob("*/recording_manifest.json"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest root must be a JSON object: {path}")
    return payload


def _plan_from_manifest(manifest_path: Path) -> RecordingPlan:
    payload = _load_manifest(manifest_path)
    recording_dir = manifest_path.parent
    source_dir = Path(str(payload.get("source_dir") or recording_dir))
    return RecordingPlan(
        name=recording_dir.name,
        source_dir=source_dir,
        dest_dir=recording_dir,
        raw_files=[],
        cam_files=[],
        derived_files=[],
        camera_id=str(payload.get("camera_id")) if payload.get("camera_id") else None,
        meta={key: value for key, value in payload.items() if isinstance(key, str)},
    )


def refresh_manifest_preflight(
    manifest_path: Path,
    *,
    run_video: bool,
    run_h5: bool,
    apply: bool,
) -> tuple[str, Optional[str]]:
    if not run_video and not run_h5:
        return "skipped", "no diagnostics requested"
    try:
        plan = _plan_from_manifest(manifest_path)
    except Exception as exc:
        return "failed", str(exc)
    if not apply:
        return "planned", None

    video_result = _run_video_diagnostics_for_plan(plan, logger=None) if run_video else None
    h5_result = _run_h5_diagnostics_for_plan(plan, logger=None) if run_h5 else None
    warning = _persist_preflight_to_manifest(
        plan,
        video_result=video_result,
        h5_result=h5_result,
    )
    if warning:
        return "failed", warning
    return "updated", None


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Recording roots or recording_manifest.json files.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover recording manifests.")
    parser.add_argument("--run-video-diagnostics", action="store_true", default=True, help="Refresh video preflight.")
    parser.add_argument("--no-run-video-diagnostics", dest="run_video_diagnostics", action="store_false")
    parser.add_argument("--run-h5-diagnostics", action="store_true", default=True, help="Refresh H5 preflight.")
    parser.add_argument("--no-run-h5-diagnostics", dest="run_h5_diagnostics", action="store_false")
    parser.add_argument("--apply", action="store_true", help="Write preflight updates (default is dry-run).")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifests = list(_iter_manifest_paths(args.paths, recursive=bool(args.recursive)))
    if not manifests:
        raise SystemExit("No recording_manifest.json files found.")

    counts: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    for manifest_path in manifests:
        status, error = refresh_manifest_preflight(
            manifest_path,
            run_video=bool(args.run_video_diagnostics),
            run_h5=bool(args.run_h5_diagnostics),
            apply=bool(args.apply),
        )
        counts[status] = counts.get(status, 0) + 1
        if error:
            failures.append({"manifest": str(manifest_path), "error": error})
        if not args.json:
            action = status.upper()
            suffix = f": {error}" if error else ""
            print(f"{action} {manifest_path}{suffix}")

    summary = {
        "manifests": len(manifests),
        "apply": bool(args.apply),
        "run_video_diagnostics": bool(args.run_video_diagnostics),
        "run_h5_diagnostics": bool(args.run_h5_diagnostics),
        "counts": counts,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
