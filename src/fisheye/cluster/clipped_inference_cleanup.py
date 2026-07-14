"""Fail-closed cleanup of NRS caches and mask packages after registry success."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_inference import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_PACKAGE_ROOT,
    PLAN_SCHEMA,
)
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.shared.batch_logging import utc_now


REPORT_SCHEMA = "palette.clipped_inference_nrs_cleanup.v1"


def _under(path: Path, root: Path, *, field: str) -> Path:
    resolved = path.expanduser().resolve()
    allowed = root.expanduser().resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"Refusing {field} cleanup outside {allowed}: {resolved}")
    if path.is_symlink():
        raise ValueError(f"Refusing symlink cleanup for {field}: {path}")
    return resolved


def cleanup(
    plan_path: Path,
    *,
    apply: bool,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    package_root: Path = DEFAULT_PACKAGE_ROOT,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported clipped inference plan: {plan_path}")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Clipped inference plan has no targets.")
    run_root = Path(str(payload["run_root"])).expanduser().resolve()
    registry_report_path = run_root / "registry" / "reconcile.json"
    if not registry_report_path.is_file():
        raise FileNotFoundError(f"Registry success report is missing: {registry_report_path}")
    registry_report = json.loads(registry_report_path.read_text(encoding="utf-8"))
    if (
        registry_report.get("status") != "ok"
        or registry_report.get("registry_integrity") != "ok"
        or int(registry_report.get("target_count") or -1) != len(targets)
    ):
        raise RuntimeError("Registry finalizer report is not a complete exact success.")

    planned: list[dict[str, str]] = []
    seen: set[Path] = set()
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("Clipped inference target must be an object.")
        for field, root in (("cache_dir", cache_root), ("package_dir", package_root)):
            raw = Path(str(target.get(field) or ""))
            path = _under(raw, root, field=field)
            if path in seen:
                raise ValueError(f"Duplicate cleanup directory in plan: {path}")
            seen.add(path)
            if not path.is_dir():
                raise FileNotFoundError(f"Planned cleanup directory is missing: {path}")
            planned.append(
                {
                    "target_id": str(target["target_id"]),
                    "kind": field,
                    "path": str(path),
                }
            )

    removed: list[dict[str, str]] = []
    if apply:
        for item in planned:
            path = Path(item["path"])
            shutil.rmtree(path)
            if path.exists():
                raise RuntimeError(f"Cleanup path still exists after removal: {path}")
            removed.append(dict(item))
    return {
        "schema": REPORT_SCHEMA,
        "status": "ok",
        "apply": bool(apply),
        "plan_path": str(plan_path),
        "registry_report": str(registry_report_path),
        "planned_count": len(planned),
        "removed_count": len(removed),
        "planned": planned,
        "removed": removed,
        "finished_at_utc": utc_now(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    report = cleanup(
        args.plan,
        apply=bool(args.apply),
        cache_root=args.cache_root,
        package_root=args.package_root,
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "cleanup"]
