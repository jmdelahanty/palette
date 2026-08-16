#!/usr/bin/env python3
"""Validate contract metadata headers and staleness windows."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

META_PATTERN = re.compile(r"<!--\s*contract-meta\s*\n(.*?)\n-->", re.DOTALL)
VALID_STATUSES = {"active", "draft", "superseded"}
VALID_IMPLEMENTATIONS = {"implemented", "partial", "specified-only"}


@dataclass
class Issue:
    path: str
    code: str
    message: str


def _parse_meta_block(text: str) -> dict[str, str] | None:
    match = META_PATTERN.search(text)
    if match is None:
        return None

    payload: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _collect_issues(
    path: Path,
    max_age_days: int,
    *,
    today: date | None = None,
) -> list[Issue]:
    text = path.read_text(encoding="utf-8")
    meta = _parse_meta_block(text)
    if meta is None:
        return []

    issues: list[Issue] = []
    for key in ("version", "status", "implementation"):
        if key not in meta:
            issues.append(Issue(path=str(path), code="missing_field", message=f"missing required field: {key}"))

    status = meta.get("status")
    if status is not None and status not in VALID_STATUSES:
        issues.append(
            Issue(
                path=str(path),
                code="invalid_status",
                message=f"invalid status '{status}', expected one of {sorted(VALID_STATUSES)}",
            )
        )

    implementation = meta.get("implementation")
    if implementation is not None and implementation not in VALID_IMPLEMENTATIONS:
        issues.append(
            Issue(
                path=str(path),
                code="invalid_implementation",
                message=(
                    f"invalid implementation '{implementation}', expected one of "
                    f"{sorted(VALID_IMPLEMENTATIONS)}"
                ),
            )
        )

    last_verified = meta.get("last_verified")
    if status == "active" and not last_verified:
        issues.append(
            Issue(
                path=str(path),
                code="missing_field",
                message="active contract is missing required field: last_verified",
            )
        )
    if last_verified:
        try:
            verified_date = date.fromisoformat(last_verified)
        except ValueError:
            issues.append(
                Issue(
                    path=str(path),
                    code="invalid_date",
                    message=f"invalid last_verified '{last_verified}', expected YYYY-MM-DD",
                )
            )
        else:
            reference_date = today or date.today()
            age_days = (reference_date - verified_date).days
            if status == "active" and age_days > max_age_days:
                issues.append(
                    Issue(
                        path=str(path),
                        code="stale",
                        message=f"last_verified is {age_days} day(s) old (max {max_age_days})",
                    )
                )

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check contract freshness metadata.")
    parser.add_argument("--max-age-days", type=int, default=90, help="Maximum allowed age for last_verified.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    contract_paths = sorted(Path("docs").glob("*contract*.md"))
    managed_paths = [
        path
        for path in contract_paths
        if _parse_meta_block(path.read_text(encoding="utf-8")) is not None
    ]
    issues: list[Issue] = []
    for path in managed_paths:
        issues.extend(_collect_issues(path, args.max_age_days))

    if args.json:
        payload: dict[str, Any] = {
            "checked": len(contract_paths),
            "managed": len(managed_paths),
            "unmanaged": len(contract_paths) - len(managed_paths),
            "max_age_days": int(args.max_age_days),
            "ok": len(issues) == 0,
            "issues": [asdict(issue) for issue in issues],
        }
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            f"Checked {len(managed_paths)} managed contract file(s); "
            f"ignored {len(contract_paths) - len(managed_paths)} without contract-meta."
        )
        if not issues:
            print("All contract metadata headers are present and fresh.")
        else:
            for issue in issues:
                print(f"- {issue.path}: [{issue.code}] {issue.message}")

    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
