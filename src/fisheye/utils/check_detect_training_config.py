#!/usr/bin/env python3
"""Audit detect training configs for ignored or overridden training_params keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fisheye.training.config import DetectConfig, TrainingParams


KNOWN_AUGMENT_KEYS = {
    "mosaic",
    "close_mosaic",
    "auto_augment",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "erasing",
    "mixup",
    "copy_paste",
    "cutmix",
}


def _load_raw_training_params(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    training_params = raw.get("training_params") or {}
    if not isinstance(training_params, dict):
        raise ValueError("'training_params' must be a mapping in config YAML")
    return dict(training_params)


def audit_detect_training_config(config_path: Path) -> Dict[str, Any]:
    raw_training_params = _load_raw_training_params(config_path)
    parsed_config = DetectConfig.from_yaml(config_path)
    effective_training_params = parsed_config.training_params.model_dump(exclude_none=True)

    supported_keys = set(TrainingParams.model_fields.keys())
    raw_keys = set(raw_training_params.keys())
    ignored_keys = sorted(raw_keys - supported_keys)
    ignored_augment_keys = sorted(k for k in ignored_keys if k in KNOWN_AUGMENT_KEYS)

    return {
        "config_path": str(config_path),
        "raw_training_param_keys": sorted(raw_keys),
        "effective_training_params": effective_training_params,
        "supported_training_param_keys": sorted(supported_keys),
        "ignored_training_param_keys": ignored_keys,
        "ignored_augment_keys": ignored_augment_keys,
        "notes": {
            "optimizer_auto_may_override_lr0_momentum": (
                "lr0" in effective_training_params or "momentum" in effective_training_params
            ),
            "zarr_loader_uses_custom_pipeline": True,
        },
    }


def _print_text_report(summary: Dict[str, Any]) -> None:
    print("Detect Training Config Audit")
    print(f"Config: {summary['config_path']}")
    print()
    print("Effective training_params passed to YOLO:")
    print(json.dumps(summary["effective_training_params"], indent=2))
    print()

    ignored = summary["ignored_training_param_keys"]
    if ignored:
        print("Ignored training_params keys (not supported by training schema):")
        for key in ignored:
            print(f"- {key}")
    else:
        print("Ignored training_params keys: none")

    ignored_augment = summary["ignored_augment_keys"]
    if ignored_augment:
        print()
        print("Ignored augmentation keys:")
        for key in ignored_augment:
            print(f"- {key}")

    print()
    print("Notes:")
    if summary["notes"]["optimizer_auto_may_override_lr0_momentum"]:
        print("- Ultralytics defaults to optimizer=auto unless explicitly overridden in trainer code.")
        print("- With optimizer=auto, runtime may ignore configured lr0/momentum.")
    print("- Detect training uses a custom Zarr data pipeline, not image-folder loading.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="Path to detect training YAML.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary instead of human-readable text.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any training_params keys are ignored.",
    )
    args = parser.parse_args(argv)

    summary = audit_detect_training_config(args.config_path)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_text_report(summary)

    if args.strict and summary["ignored_training_param_keys"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
