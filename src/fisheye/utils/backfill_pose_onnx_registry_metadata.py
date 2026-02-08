#!/usr/bin/env python3
"""Backfill pose ONNX export metadata in registry from ONNX manifest files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fisheye.registry.db import Registry, RegistryPaths


def _coerce_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _find_manifest_for_run(onnx_path: Path, run_id: str) -> Optional[Path]:
    candidates = [
        onnx_path.with_suffix(".onnx.manifest.json"),
        onnx_path.parent / f"{run_id}.onnx.manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _collect_candidates(registry: Registry, run_id: Optional[str]) -> List[Dict[str, Any]]:
    sql = [
        "SELECT run_id, path, manifest_path",
        "FROM onnx_models",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    if run_id:
        sql.append("AND run_id = ?")
        params.append(run_id)
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [dict(row) for row in rows]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Registry SQLite path.")
    parser.add_argument("--run-id", type=str, help="Only backfill one run_id.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        rows = _collect_candidates(registry, args.run_id)
        if not rows:
            print("No ONNX model rows found.")
            return 0

        scanned = 0
        updated = 0
        skipped = 0
        errors = 0

        for row in rows:
            scanned += 1
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                skipped += 1
                print("skip      missing run_id")
                continue

            onnx_path_raw = row.get("path")
            if not onnx_path_raw:
                skipped += 1
                print(f"skip      {run_id} missing onnx path")
                continue

            onnx_path = Path(str(onnx_path_raw)).expanduser()
            manifest_path_raw = row.get("manifest_path")
            manifest_path = (
                Path(str(manifest_path_raw)).expanduser()
                if manifest_path_raw
                else _find_manifest_for_run(onnx_path, run_id)
            )
            if manifest_path is None or not manifest_path.exists():
                skipped += 1
                print(f"skip      {run_id} missing manifest")
                continue

            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors += 1
                print(f"error     {run_id} manifest parse failed: {exc}")
                continue

            export = _coerce_dict(payload.get("export"))
            onnx = _coerce_dict(payload.get("onnx"))
            build_env = _coerce_dict(payload.get("build_env"))
            metadata: Dict[str, Any] = {
                "sha256": onnx.get("sha256"),
                "manifest_sha256": None,
                "opset": export.get("opset"),
                "input_shape": export.get("input_shape"),
                "imgsz": export.get("imgsz"),
                "build_env": build_env,
            }

            if args.dry_run:
                updated += 1
                print(f"would_upd {run_id} manifest={manifest_path}")
                continue

            try:
                registry.record_model_export(
                    run_id=run_id,
                    export_type="onnx",
                    path=onnx_path,
                    manifest_path=manifest_path,
                    metadata=metadata,
                )
                updated += 1
                print(f"updated   {run_id} manifest={manifest_path}")
            except Exception as exc:
                errors += 1
                print(f"error     {run_id} registry update failed: {exc}")

        print("\nSummary:")
        print(f"  scanned: {scanned}")
        print(f"  updated: {updated}")
        print(f"  skipped: {skipped}")
        print(f"  errors: {errors}")
        return 2 if errors > 0 else 0
    finally:
        registry.close()


if __name__ == "__main__":
    raise SystemExit(main())
