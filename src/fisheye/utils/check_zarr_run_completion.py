"""Check Palette Zarr run parents for unsafe incomplete latest pointers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import zarr

from fisheye.shared.zarr_run_completion import iter_run_parent_summaries


def _open_root(zarr_path: Path) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)


def build_completion_report(
    zarr_path: Path,
    *,
    strict_legacy: bool = False,
) -> dict[str, Any]:
    root = _open_root(zarr_path)
    parents = list(
        iter_run_parent_summaries(
            root,
            legacy_default=not strict_legacy,
        )
    )
    unsafe = [summary for summary in parents if summary["unsafe"]]
    pending = [
        summary
        for summary in parents
        if summary.get("latest_pending") or summary.get("incomplete_runs")
    ]
    return {
        "schema_version": "palette.zarr_run_completion_check.v1",
        "status": "unsafe" if unsafe else "ok",
        "zarr_path": str(zarr_path),
        "strict_legacy": bool(strict_legacy),
        "run_parent_count": len(parents),
        "unsafe_parent_count": len(unsafe),
        "pending_parent_count": len(pending),
        "unsafe_parents": unsafe,
        "pending_parents": pending,
        "run_parents": parents,
    }


def _print_report(report: dict[str, Any], *, include_ok: bool) -> None:
    print(f"zarr: {report['zarr_path']}")
    print(f"status: {report['status']}")
    print(f"run_parent_count: {report['run_parent_count']}")
    print(f"unsafe_parent_count: {report['unsafe_parent_count']}")
    print(f"pending_parent_count: {report['pending_parent_count']}")

    parents = report["run_parents"] if include_ok else (
        report["unsafe_parents"] or report["pending_parents"]
    )
    if not parents:
        return
    print()
    for summary in parents:
        label = "unsafe" if summary["unsafe"] else "pending" if (
            summary.get("latest_pending") or summary.get("incomplete_runs")
        ) else "ok"
        print(
            f"{label:>7}  {summary['parent_path'] or '<root>'} "
            f"latest={summary.get('latest') or '-'} "
            f"latest_complete={summary.get('latest_complete') or '-'} "
            f"latest_pending={summary.get('latest_pending') or '-'} "
            f"resolved={summary.get('resolved_latest_complete') or '-'}"
        )
        reasons = summary.get("unsafe_reasons") or []
        if reasons:
            print(f"         reasons={','.join(str(reason) for reason in reasons)}")
        incomplete = summary.get("incomplete_runs") or []
        if incomplete:
            print(f"         incomplete={','.join(str(name) for name in incomplete)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Palette Zarr run parents for completion-contract violations, "
            "especially attrs['latest'] pointing at incomplete opted-in runs."
        )
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument(
        "--strict-legacy",
        action="store_true",
        help="Treat legacy runs without completion attrs as incomplete.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report to stdout.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the full JSON report to this path.",
    )
    parser.add_argument(
        "--include-ok",
        action="store_true",
        help="In text mode, include all run parents instead of only unsafe/pending parents.",
    )
    parser.add_argument(
        "--fail-on-unsafe",
        action="store_true",
        help="Exit non-zero when any unsafe latest pointer is found.",
    )
    args = parser.parse_args(argv)

    report = build_completion_report(args.zarr_path, strict_legacy=args.strict_legacy)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report, include_ok=args.include_ok)
    if args.fail_on_unsafe and report["unsafe_parent_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
