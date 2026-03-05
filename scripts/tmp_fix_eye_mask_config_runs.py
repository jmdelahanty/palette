#!/usr/bin/env python3
"""Temporary helper: align eye-mask training config run names to merged Zarr latest runs.

Usage:
  scripts/py scripts/tmp_fix_eye_mask_config_runs.py /path/to/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
import zarr


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _latest_run_name(group: Any) -> str:
    if group is None:
        return ""
    attrs = getattr(group, "attrs", None)
    if attrs is None:
        return ""
    return _as_text(attrs.get("latest"))


def _resolve_runs(zarr_path: Path) -> tuple[str, str]:
    root = zarr.open_group(str(zarr_path), mode="r")
    crop_latest = _latest_run_name(root.get("crop_runs"))
    eye_latest = _latest_run_name(root.get("eye_masks_runs"))
    merged_run = crop_latest or eye_latest
    if not merged_run:
        raise RuntimeError(f"Could not resolve latest run under crop_runs/eye_masks_runs: {zarr_path}")
    return merged_run, (eye_latest or merged_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="Path to generated eye-mask training config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing.")
    args = parser.parse_args()

    config_path = args.config_path.expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Config root is not a mapping: {config_path}")

    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise SystemExit(f"No datasets mapping found in config: {config_path}")

    updates = 0
    for dataset_name, dataset_cfg in datasets.items():
        if not isinstance(dataset_cfg, dict):
            continue
        zarr_raw = dataset_cfg.get("zarr_path")
        if not zarr_raw:
            raise SystemExit(f"Dataset '{dataset_name}' missing zarr_path.")
        zarr_path = Path(str(zarr_raw)).expanduser().resolve()
        merged_run, mask_run = _resolve_runs(zarr_path)

        old_crop = _as_text(dataset_cfg.get("crop_run"))
        old_mask = _as_text(dataset_cfg.get("mask_run"))
        dataset_cfg["crop_run"] = merged_run
        dataset_cfg["mask_run"] = mask_run
        updates += 1
        print(
            f"{dataset_name}\tzarr={zarr_path}\t"
            f"crop_run:{old_crop or '-'}->{merged_run}\t"
            f"mask_run:{old_mask or '-'}->{mask_run}"
        )

    if args.dry_run:
        print(f"dry_run\tupdates={updates}\tconfig={config_path}")
        return 0

    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"updated\tupdates={updates}\tconfig={config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

