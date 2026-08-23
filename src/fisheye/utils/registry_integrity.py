#!/usr/bin/env python3
"""Validate a Palette registry with the SQLite runtime used by Palette Python."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.registry.shadow_publish import validate_registry_sqlite
from fisheye.shared.json_safety import write_json_atomic

SCHEMA_ID = "palette.registry_integrity_validation"
SCHEMA_VERSION = 1


def build_validation_receipt(registry: str | Path) -> dict[str, object]:
    validation = validate_registry_sqlite(registry)
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "validation": validation.to_json(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional path for an atomic machine-readable validation receipt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_validation_receipt(args.registry)
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
