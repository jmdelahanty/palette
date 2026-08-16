"""Command-line interface for typed registry cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from fisheye.cohorts.registry import (
    CohortSelectionError,
    build_cohort_plan,
    coverage_report,
    freeze_cohort,
    validate_frozen_cohort,
    validate_frozen_cohort_registry_binding,
)
from fisheye.cohorts.spec import CohortSpecError, load_cohort_spec


def _write(payload: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, audit, and freeze typed registry-backed cohorts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "coverage", "freeze"):
        command = subparsers.add_parser(name)
        command.add_argument("--registry", type=Path, required=True)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--output", type=Path)
        if name == "freeze":
            command.add_argument(
                "--plan-output",
                type=Path,
                help="Optional immutable diagnostic plan written before freeze validation.",
            )
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--check-hash", action="store_true")
    validate.add_argument(
        "--registry",
        type=Path,
        help="Also require the manifest's registry UUID to match this registry.",
    )
    zarrs = subparsers.add_parser("zarr-list")
    zarrs.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"plan", "coverage", "freeze"}:
            spec = load_cohort_spec(args.spec)
        if args.command == "plan":
            _write(build_cohort_plan(args.registry, spec), args.output)
        elif args.command == "coverage":
            _write(coverage_report(args.registry, spec), args.output)
        elif args.command == "freeze":
            plan = build_cohort_plan(args.registry, spec)
            if args.plan_output is not None:
                _write(plan, args.plan_output)
            _write(freeze_cohort(plan), args.output)
        elif args.command == "validate":
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            errors = validate_frozen_cohort(payload, check_hash=bool(args.check_hash))
            if not errors and args.registry is not None:
                errors.extend(
                    validate_frozen_cohort_registry_binding(payload, args.registry)
                )
            if errors:
                raise CohortSelectionError(
                    "invalid frozen cohort: " + "; ".join(errors)
                )
            print(
                f"valid frozen cohort: {payload.get('cohort_id')} "
                f"({len(payload['members'])} member(s))"
            )
        elif args.command == "zarr-list":
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            errors = validate_frozen_cohort(payload, check_hash=True)
            if errors:
                raise CohortSelectionError(
                    "invalid frozen cohort: " + "; ".join(errors)
                )
            for member in payload.get("members", []):
                print(member["zarr_path"])
    except (
        CohortSelectionError,
        CohortSpecError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
