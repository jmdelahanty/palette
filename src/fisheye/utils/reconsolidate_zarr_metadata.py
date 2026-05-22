from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fisheye.shared.zarr_helpers import reconsolidate_zarr_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Zarr consolidated metadata for a Palette archive or subtree. "
            "This is a finalization/compatibility step; readers must still tolerate stale metadata."
        )
    )
    parser.add_argument("zarr_path", help="Path to the Zarr archive root.")
    parser.add_argument(
        "--group-path",
        default=None,
        help="Optional group path to consolidate instead of the archive root.",
    )
    parser.add_argument(
        "--policy",
        default="manual",
        help="Policy/provenance label to write into metadata_consolidation_policy.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Exit zero even if consolidation fails; the JSON status will still be error.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--output-json", default=None, help="Optional path to write the JSON report.")
    return parser


def _print_text(report: dict[str, Any]) -> None:
    print("zarr_metadata_reconsolidation")
    print(f"status: {report.get('status')}")
    print(f"zarr_path: {report.get('zarr_path')}")
    print(f"group_path: {report.get('group_path') or '(root)'}")
    print(f"policy: {report.get('policy')}")
    print(f"consolidated_at_utc: {report.get('consolidated_at_utc')}")
    if report.get("error"):
        print(f"error: {report.get('error')}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = reconsolidate_zarr_metadata(
        args.zarr_path,
        group_path=args.group_path,
        policy=args.policy,
        fail_on_error=False,
    )
    if args.output_json:
        Path(args.output_json).expanduser().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    if report.get("status") != "ok" and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
