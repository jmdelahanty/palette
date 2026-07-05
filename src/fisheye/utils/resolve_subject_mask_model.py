#!/usr/bin/env python3
"""Resolve subject-mask U-Net checkpoints from registry model metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.model_resolution import (
    SubjectMaskModelCandidate,
    build_resolution_payload,
    load_subject_mask_model_candidates,
    resolve_best_subject_mask_model,
)

__all__ = [
    "SubjectMaskModelCandidate",
    "build_resolution_payload",
    "load_subject_mask_model_candidates",
    "main",
    "resolve_best_subject_mask_model",
]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--coverage-class", default="dense_all_components")
    parser.add_argument("--component-coverage-key")
    parser.add_argument("--label-schema-id")
    parser.add_argument("--include-non-success", action="store_true")
    parser.add_argument("--allow-missing-path", action="store_true")
    parser.add_argument("--require-unique", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        best, candidates = resolve_best_subject_mask_model(
            registry,
            coverage_class=args.coverage_class,
            component_coverage_key=args.component_coverage_key,
            label_schema_id=args.label_schema_id,
            include_non_success=bool(args.include_non_success),
            require_existing_path=not bool(args.allow_missing_path),
            require_unique=bool(args.require_unique),
        )
    finally:
        registry.close()

    payload = build_resolution_payload(
        registry_path=registry_path,
        selected=best,
        candidates=candidates,
        top_k=int(args.top_k),
        parameters={
            "coverage_class": args.coverage_class,
            "component_coverage_key": args.component_coverage_key,
            "label_schema_id": args.label_schema_id,
            "include_non_success": bool(args.include_non_success),
            "require_existing_path": not bool(args.allow_missing_path),
            "require_unique": bool(args.require_unique),
        },
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("Recommended subject-mask model")
    print(f"  run_id: {best.run_id}")
    print(f"  set_id: {best.set_id or '-'}")
    print(f"  status: {best.status or '-'}")
    print(f"  label_schema_id: {best.label_schema_id or '-'}")
    print(f"  coverage_class: {best.coverage_class or '-'}")
    print(f"  component_coverage_key: {best.component_coverage_key or '-'}")
    print(f"  {best.best_metric_name or 'metric'}: {best.best_metric_value}")
    print(f"  best_epoch: {best.best_epoch}")
    print(f"  model_path: {best.model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
