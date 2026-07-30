"""Serially reconcile registry projections after every clipped target validates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_inference import SUPPORTED_PLAN_SCHEMAS
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.registry.db import Registry
from fisheye.shared.zarr_helpers import open_zarr_group_direct


REPORT_SCHEMA = "palette.clipped_inference_registry_finalizer.v1"


def finalize_registry(plan_path: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") not in SUPPORTED_PLAN_SCHEMAS
    ):
        raise ValueError(f"Unsupported clipped inference plan: {plan_path}")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Clipped inference plan has no targets.")
    run_root = Path(str(payload["run_root"])).expanduser().resolve()
    registry_path = Path(str(payload["registry"])).expanduser().resolve()

    validations: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("Clipped inference target must be an object.")
        safe = safe_component(str(target["target_id"]), default="target", max_length=56)
        validation_path = run_root / "validation" / f"{safe}.json"
        if not validation_path.is_file():
            raise FileNotFoundError(f"Target validation report is missing: {validation_path}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "ok" or validation.get("target_id") != target["target_id"]:
            raise RuntimeError(f"Target validation is not an exact success: {validation_path}")
        validations.append(validation)

    registry = Registry(registry_path)
    reconciled: list[dict[str, Any]] = []
    try:
        for target in targets:
            zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
            root = open_zarr_group_direct(zarr_path, mode="r")
            result = registry.reconcile_dataset_from_root(
                root,
                zarr_path,
                include_step_status=True,
            )
            reconciled.append(
                {
                    "target_id": str(target["target_id"]),
                    "analysis_zarr": str(zarr_path),
                    **result,
                }
            )
        integrity = str(registry.conn.execute("PRAGMA integrity_check;").fetchone()[0])
    finally:
        registry.close()
    if integrity != "ok":
        raise RuntimeError(f"Registry integrity check failed: {integrity}")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "ok",
        "plan_path": str(plan_path),
        "registry": str(registry_path),
        "registry_integrity": integrity,
        "target_count": len(reconciled),
        "validation_reports": [
            {
                "target_id": item["target_id"],
                "row_count": item["row_count"],
                "analysis_zarr": item["analysis_zarr"],
            }
            for item in validations
        ],
        "reconciled": reconciled,
    }
    if output_path is not None:
        write_json_snapshot(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    report = finalize_registry(args.plan, output_path=args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "finalize_registry"]
