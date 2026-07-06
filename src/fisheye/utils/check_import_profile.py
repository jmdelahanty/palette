"""Read-only checker for Palette singleton import profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from fisheye.shared.import_profile_contract import classify_import_profile
from fisheye.shared.zarr_io import open_zarr_root


def _check_one(path: Path) -> dict[str, Any]:
    try:
        root = open_zarr_root(path, mode="r")
        report = classify_import_profile(root).to_dict()
        return {"zarr_path": str(path), **report}
    except Exception as exc:
        return {
            "zarr_path": str(path),
            "schema_id": "palette.import_profile_contract.v1",
            "profile": "open_failed",
            "status": "failed",
            "ok": False,
            "reason_codes": ["OPEN_FAILED"],
            "required_missing": [],
            "recommended_missing": [],
            "warnings": [str(exc)],
            "arrays_present": [],
            "attrs_observed": {},
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Palette singleton import profiles on one or more Zarr stores.",
    )
    parser.add_argument("zarr_path", nargs="+", type=Path, help="Palette .zarr path(s) to inspect.")
    parser.add_argument("--jsonl", action="store_true", help="Emit one JSON object per line.")
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit non-zero when any store is incomplete, unknown, or failed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = [_check_one(path.expanduser()) for path in args.zarr_path]

    if args.jsonl:
        for row in rows:
            print(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    else:
        print(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=True))

    if args.fail_on_incomplete and any(row.get("status") not in {"ok", "warning"} for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
