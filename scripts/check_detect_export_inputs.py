#!/usr/bin/env python3
"""Inspect detect export manifests for input shape/channel expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_channel_count(input_shape: object) -> int | None:
    if not isinstance(input_shape, list):
        return None
    if len(input_shape) < 2:
        return None
    return _as_int(input_shape[1])


def _manifest_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest JSON payload must be an object")

    export = payload.get("export")
    export = export if isinstance(export, dict) else {}

    onnx = payload.get("onnx")
    onnx = onnx if isinstance(onnx, dict) else {}

    engine = payload.get("engine")
    engine = engine if isinstance(engine, dict) else {}

    input_shape = export.get("input_shape")
    channel_count = _extract_channel_count(input_shape)
    kind = "engine" if engine else ("onnx" if onnx else "unknown")

    return {
        "manifest_path": str(path),
        "kind": kind,
        "onnx_path": onnx.get("path"),
        "engine_path": engine.get("path"),
        "imgsz": export.get("imgsz"),
        "input_shape": input_shape,
        "channel_count": channel_count,
        "is_rgb_3ch": channel_count == 3 if channel_count is not None else None,
    }


def _iter_manifest_paths(run_dir: Path) -> list[Path]:
    exports_dir = run_dir / "exports"
    if not exports_dir.exists():
        return []
    return sorted(exports_dir.rglob("*manifest*.json"))


def _print_human(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No export manifests found under run_dir/exports.")
        return
    print(f"Found {len(rows)} export manifest(s):")
    for row in rows:
        print()
        print(f"manifest: {row['manifest_path']}")
        print(f"  kind: {row['kind']}")
        if row.get("onnx_path"):
            print(f"  onnx_path: {row['onnx_path']}")
        if row.get("engine_path"):
            print(f"  engine_path: {row['engine_path']}")
        print(f"  imgsz: {row.get('imgsz')}")
        print(f"  input_shape: {row.get('input_shape')}")
        print(f"  channel_count: {row.get('channel_count')}")
        print(f"  is_rgb_3ch: {row.get('is_rgb_3ch')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Training run directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--require-rgb",
        action="store_true",
        help="Exit non-zero if any manifest does not report channel_count == 3.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    manifest_paths = _iter_manifest_paths(run_dir)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in manifest_paths:
        try:
            rows.append(_manifest_summary(path))
        except Exception as exc:
            errors.append({"manifest_path": str(path), "error": str(exc)})

    payload = {
        "run_dir": str(run_dir),
        "manifest_count": len(manifest_paths),
        "summaries": rows,
        "errors": errors,
    }

    failed_rgb = False
    if args.require_rgb:
        for row in rows:
            if row.get("channel_count") != 3:
                failed_rgb = True
                break
        if not rows:
            failed_rgb = True

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(rows)
        if errors:
            print()
            print("Manifest parse errors:")
            for item in errors:
                print(f"  - {item['manifest_path']}: {item['error']}")

    if args.require_rgb and failed_rgb:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
