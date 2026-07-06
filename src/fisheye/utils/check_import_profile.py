"""Read-only checker for Palette singleton import profiles."""

from __future__ import annotations

import argparse
from collections import Counter
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


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "attrs_observed"}


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(str(reason) for reason in row.get("reason_codes", ()))
    return {
        "schema_id": "palette.import_profile_check_summary.v1",
        "total": len(rows),
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "status_counts": dict(sorted(Counter(str(row.get("status", "")) for row in rows).items())),
        "profile_counts": dict(sorted(Counter(str(row.get("profile", "")) for row in rows).items())),
        "reason_code_counts": dict(sorted(reason_counts.items())),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Palette singleton import profiles on one or more Zarr stores.",
    )
    parser.add_argument("zarr_path", nargs="+", type=Path, help="Palette .zarr path(s) to inspect.")
    parser.add_argument("--jsonl", action="store_true", help="Emit one JSON object per line.")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit full attrs_observed payloads from emitted per-Zarr rows.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Write aggregate profile/status/reason-code counts to this JSON file.",
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit non-zero when any store is incomplete, unknown, or failed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = [_check_one(path.expanduser()) for path in args.zarr_path]
    output_rows = [_compact_row(row) for row in rows] if args.compact else rows

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(_summary(rows), indent=2, sort_keys=True, ensure_ascii=True) + "\n")

    if args.jsonl:
        for row in output_rows:
            print(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    else:
        print(json.dumps(output_rows, indent=2, sort_keys=True, ensure_ascii=True))

    if args.fail_on_incomplete and any(row.get("status") not in {"ok", "warning"} for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
