#!/usr/bin/env python3
"""Build and validate one historical pose-model input contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from fisheye.shared.pose_model_input_contract import (
    build_historical_pose_model_input_contract,
    load_pose_model_input_contract,
)


def _write_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise FileExistsError(f"Refusing to replace a different contract: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-package-root", type=Path, required=True)
    parser.add_argument("--model-set-id", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument(
        "--weights-relative-path", type=Path, default=Path("weights/best.pt")
    )
    parser.add_argument("--training-manifest-relative-path", type=Path, required=True)
    parser.add_argument("--training-report-relative-path", type=Path, required=True)
    parser.add_argument(
        "--training-args-relative-path", type=Path, default=Path("args.yaml")
    )
    parser.add_argument("--model-stride", type=int, required=True)
    parser.add_argument(
        "--runtime-ultralytics-version",
        action="append",
        default=[],
        help=(
            "Explicitly reviewed runtime version whose maintained preprocessing "
            "must match the digest-bound reference; may be repeated."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_root = args.model_package_root.expanduser().resolve()
    document = build_historical_pose_model_input_contract(
        set_id=args.model_set_id,
        run_id=args.model_run_id,
        model_package_root=package_root,
        weights_relative_path=args.weights_relative_path,
        training_manifest_relative_path=args.training_manifest_relative_path,
        training_report_relative_path=args.training_report_relative_path,
        training_args_relative_path=args.training_args_relative_path,
        model_stride=int(args.model_stride),
        runtime_ultralytics_versions=tuple(args.runtime_ultralytics_version),
    )
    output = args.output.expanduser().resolve()
    if args.dry_run:
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0

    _write_atomic(output, document)
    binding = load_pose_model_input_contract(
        output,
        model_path=(package_root / args.weights_relative_path),
        expected_set_id=args.model_set_id,
        expected_run_id=args.model_run_id,
        expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "contract_sha256": binding.sha256,
                "payload_digest": binding.payload_digest,
                "training_source_shape_hw": list(binding.training_source_shape_hw),
                "network_shape_hw": list(binding.network_shape_hw),
                "model_stride": binding.model_stride,
                "input_mode": binding.input_mode,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
