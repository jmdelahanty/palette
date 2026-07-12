"""CLI for validating immutable analytics exports before publication/use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.validation import validate_export_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--export-run-id", action="append", required=True)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = validate_export_runs(args.export_root, args.export_run_id)
    payload = {"status": "valid", "exports": reports}
    if args.json_output is not None:
        output = Path(args.json_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output)
    for report in reports:
        print(f"export_run_id\t{report['export_run_id']}")
        print(f"status\t{report['status']}")
        print(f"tables\t{report['table_count']}")
        print(f"parts\t{report['part_count']}")
        print(f"rows\t{report['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
