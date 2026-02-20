"""Parse check_eye_mask_lineage JSONL logs and summarize failing zarrs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ParsedLog:
    path: Path
    run_id: Optional[str] = None
    run_start: Dict[str, object] = field(default_factory=dict)
    run_end: Dict[str, object] = field(default_factory=dict)
    failing_zarrs: List[str] = field(default_factory=list)
    passing_zarrs: List[str] = field(default_factory=list)
    filtered_zarrs: List[str] = field(default_factory=list)
    error_zarrs: List[str] = field(default_factory=list)
    invalid_rows: int = 0

    @property
    def has_issues(self) -> bool:
        return bool(self.failing_zarrs or self.error_zarrs)


def _default_log_dir() -> Path:
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "check_eye_mask_lineage"
    return Path("/nvme1/recordings/logs/check_eye_mask_lineage")


def _find_latest_log(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("check_eye_mask_lineage_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No check_eye_mask_lineage logs found in {log_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def _iter_json_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                yield {"_invalid_json": line}
                continue
            if isinstance(payload, dict):
                yield payload
            else:
                yield {"_invalid_json": line}


def _parse_log(path: Path) -> ParsedLog:
    parsed = ParsedLog(path=path)
    seen_fail: set[str] = set()
    seen_pass: set[str] = set()
    seen_filtered: set[str] = set()
    seen_error: set[str] = set()

    for row in _iter_json_rows(path):
        if "_invalid_json" in row:
            parsed.invalid_rows += 1
            continue
        event = str(row.get("event") or "")
        if parsed.run_id is None:
            run_id = row.get("run_id")
            if run_id is not None:
                parsed.run_id = str(run_id)

        if event == "run_start":
            parsed.run_start = dict(row)
            continue
        if event == "run_end":
            parsed.run_end = dict(row)
            continue
        if event == "zarr_checked":
            zarr = row.get("zarr")
            if zarr is None:
                continue
            zarr_text = str(zarr)
            issues = bool(row.get("issues"))
            if issues:
                if zarr_text not in seen_fail:
                    seen_fail.add(zarr_text)
                    parsed.failing_zarrs.append(zarr_text)
            else:
                if zarr_text not in seen_pass:
                    seen_pass.add(zarr_text)
                    parsed.passing_zarrs.append(zarr_text)
            continue
        if event == "zarr_filtered":
            zarr = row.get("zarr")
            if zarr is None:
                continue
            zarr_text = str(zarr)
            if zarr_text not in seen_filtered:
                seen_filtered.add(zarr_text)
                parsed.filtered_zarrs.append(zarr_text)
            continue
        if event == "zarr_error":
            zarr = row.get("zarr")
            if zarr is None:
                continue
            zarr_text = str(zarr)
            if zarr_text not in seen_error:
                seen_error.add(zarr_text)
                parsed.error_zarrs.append(zarr_text)

    return parsed


def _print_summary(parsed: ParsedLog, *, failures_only: bool) -> None:
    if failures_only:
        for zarr in parsed.failing_zarrs:
            print(zarr)
        return

    print(f"Log: {parsed.path}")
    if parsed.run_id:
        print(f"Run ID: {parsed.run_id}")
    if parsed.run_end:
        keys = (
            "zarr_scanned",
            "zarr_checked",
            "zarr_pass",
            "zarr_fail",
            "filtered_zarr_use",
            "checked_runs",
            "errors",
            "issues",
        )
        parts = [f"{key}={parsed.run_end.get(key)}" for key in keys if key in parsed.run_end]
        if parts:
            print("Run End: " + " ".join(parts))
    if parsed.invalid_rows:
        print(f"Invalid JSON rows: {parsed.invalid_rows}")

    print(f"Failing zarrs: {len(parsed.failing_zarrs)}")
    for zarr in parsed.failing_zarrs:
        print(f"  {zarr}")
    print(f"Error zarrs: {len(parsed.error_zarrs)}")
    for zarr in parsed.error_zarrs:
        print(f"  {zarr}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", nargs="?", type=Path, help="Path to a check_eye_mask_lineage JSONL log.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory used when --log_path is omitted (default: $PALETTE_LOG_ROOT/check_eye_mask_lineage or /nvme1/recordings/logs/check_eye_mask_lineage).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest log in --log-dir (default when log_path is omitted).",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Print only failing zarr paths (one per line).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when failing or error zarrs are present.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    log_path: Optional[Path] = args.log_path
    if log_path is None:
        log_dir = (args.log_dir or _default_log_dir()).expanduser()
        log_path = _find_latest_log(log_dir)
    elif args.latest:
        raise ValueError("--latest cannot be combined with an explicit log_path.")

    parsed = _parse_log(log_path.expanduser())
    _print_summary(parsed, failures_only=bool(args.failures_only))

    if args.strict and parsed.has_issues:
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
