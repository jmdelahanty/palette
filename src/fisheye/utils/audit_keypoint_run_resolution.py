#!/usr/bin/env python3
"""Audit keypoint selector vs. reviewed refined-run resolution across recordings."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr_paths
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import zarr

from fisheye.utils import prepare_keypoint_training_from_registry as prep_pose


ISSUE_STATUSES = {"switch_needed", "cross_method_review", "no_review_match", "error"}
DIVERGENCE_ISSUE_STATUSES = {"conflict", "attrs_missing_disk_present", "attrs_present_disk_missing"}


def _state_and_intended_use(review_status: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(review_status, dict):
        return None, None
    state = prep_pose._decode_attr(review_status.get("state"))
    intended_use = prep_pose._decode_attr(review_status.get("intended_use"))
    return state, intended_use


def _method_hint_for_selector(normalized_selector: Optional[str]) -> Optional[str]:
    if normalized_selector == "latest_traditional":
        return "traditional_pose"
    if normalized_selector == "latest_yolo":
        return "yolo_pose"
    return None


def _row_divergence_issues(row: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    resolved_divergence = row.get("resolved_review_divergence")
    if isinstance(resolved_divergence, str) and resolved_divergence in DIVERGENCE_ISSUE_STATUSES:
        issues.append(f"resolved:{resolved_divergence}")
    reviewed_divergence = row.get("reviewed_choice_review_divergence")
    if isinstance(reviewed_divergence, str) and reviewed_divergence in DIVERGENCE_ISSUE_STATUSES:
        issues.append(f"reviewed:{reviewed_divergence}")
    return issues


def audit_keypoint_resolution(
    paths: Iterable[Path],
    *,
    recursive: bool,
    selector: Optional[str],
    required_state: Optional[str],
    required_intended_use: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    zarr_paths = _iter_zarr_paths(paths, recursive=recursive)
    normalized_selector = prep_pose._normalize_keypoint_run_selector(selector)
    method_hint = _method_hint_for_selector(normalized_selector)

    for zarr_path in zarr_paths:
        payload: Dict[str, Any] = {
            "zarr_path": str(zarr_path),
            "selector_requested": selector,
            "selector_normalized": normalized_selector,
            "status": "error",
        }
        try:
            try:
                root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
            except TypeError:
                root = zarr.open_group(str(zarr_path), mode="r")
            keypoint_run_resolved, selector_resolved = prep_pose._resolve_keypoint_run(root, selector)
            payload["selector_resolved"] = selector_resolved
            payload["resolved_keypoint_run"] = keypoint_run_resolved

            keypoints_parent = root["keypoints_runs"]
            resolved_method = prep_pose._decode_attr(keypoints_parent[keypoint_run_resolved].attrs.get("method"))
            payload["resolved_method"] = resolved_method

            resolved_quality = prep_pose._resolve_refined_keypoint_quality(
                root,
                keypoint_run_resolved,
                zarr_path=zarr_path,
            )
            resolved_review = resolved_quality.get("keypoint_review_status")
            resolved_state, resolved_intended_use = _state_and_intended_use(resolved_review)
            payload["resolved_review_state"] = resolved_state
            payload["resolved_review_intended_use"] = resolved_intended_use
            payload["resolved_review_divergence"] = resolved_quality.get("keypoint_review_status_divergence")

            strict_choice = prep_pose._resolve_reviewed_keypoint_run(
                root,
                method_hint=method_hint,
                required_state=required_state,
                required_intended_use=required_intended_use,
                zarr_path=zarr_path,
            )
            relaxed_choice = None
            if strict_choice is None and method_hint is not None:
                relaxed_choice = prep_pose._resolve_reviewed_keypoint_run(
                    root,
                    method_hint=None,
                    required_state=required_state,
                    required_intended_use=required_intended_use,
                    zarr_path=zarr_path,
                )

            reviewed_choice = strict_choice or relaxed_choice
            if reviewed_choice is None:
                payload["status"] = "no_review_match"
                payload["reason"] = "No reviewed refined run satisfies the requested review constraints."
                rows.append(payload)
                continue

            reviewed_source = str(reviewed_choice["source_keypoint_run"])
            reviewed_refined_run = str(reviewed_choice["refined_keypoint_run"])
            reviewed_status = reviewed_choice.get("keypoint_review_status")
            reviewed_state, reviewed_intended_use = _state_and_intended_use(reviewed_status)

            payload["reviewed_source_keypoint_run"] = reviewed_source
            payload["reviewed_refined_run"] = reviewed_refined_run
            payload["reviewed_state"] = reviewed_state
            payload["reviewed_intended_use"] = reviewed_intended_use
            payload["reviewed_choice_review_divergence"] = reviewed_choice.get("keypoint_review_status_divergence")
            payload["reviewed_source_method"] = prep_pose._decode_attr(
                keypoints_parent[reviewed_source].attrs.get("method")
            )

            if strict_choice is not None:
                if reviewed_source == keypoint_run_resolved:
                    payload["status"] = "aligned"
                else:
                    payload["status"] = "switch_needed"
                    payload["reason"] = (
                        "Selector resolved to a run that differs from the reviewed run source for this method."
                    )
            else:
                payload["status"] = "cross_method_review"
                payload["reason"] = (
                    "No reviewed run matched the selector method; review exists only via another keypoint method."
                )

        except Exception as exc:
            payload["status"] = "error"
            payload["error"] = str(exc)

        rows.append(payload)

    return rows


def _print_text_report(rows: List[Dict[str, Any]], include_ok: bool) -> None:
    total = len(rows)
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    print("Keypoint Run Resolution Audit")
    print(f"Scanned archives: {total}")
    print("Status counts:")
    for status in sorted(status_counts):
        print(f"- {status}: {status_counts[status]}")
    divergence_issue_count = sum(1 for row in rows if _row_divergence_issues(row))
    print(f"- divergence_issues: {divergence_issue_count}")

    print()
    print("Findings:")
    printed = 0
    for row in rows:
        status = str(row.get("status", "unknown"))
        divergence_issues = _row_divergence_issues(row)
        if not include_ok and status not in ISSUE_STATUSES and not divergence_issues:
            continue
        printed += 1
        print(f"- [{status}] {row.get('zarr_path')}")
        if row.get("resolved_keypoint_run"):
            print(
                "  resolved: "
                f"{row.get('resolved_keypoint_run')} (method={row.get('resolved_method')}, "
                f"review={row.get('resolved_review_state')}/{row.get('resolved_review_intended_use')})"
            )
            if row.get("resolved_review_divergence"):
                print(f"  resolved_review_divergence: {row.get('resolved_review_divergence')}")
        if row.get("reviewed_source_keypoint_run"):
            print(
                "  reviewed: "
                f"{row.get('reviewed_source_keypoint_run')} (method={row.get('reviewed_source_method')}, "
                f"state={row.get('reviewed_state')}, use={row.get('reviewed_intended_use')}, "
                f"refined_run={row.get('reviewed_refined_run')})"
            )
            if row.get("reviewed_choice_review_divergence"):
                print(f"  reviewed_choice_review_divergence: {row.get('reviewed_choice_review_divergence')}")
        if divergence_issues:
            print(f"  divergence_issues: {', '.join(divergence_issues)}")
        if row.get("reason"):
            print(f"  reason: {row.get('reason')}")
        if row.get("error"):
            print(f"  error: {row.get('error')}")

    if printed == 0:
        print("- none")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Path(s) to .zarr archives or parent directories to scan.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for *.zarr under directory inputs.",
    )
    parser.add_argument(
        "--selector",
        default="latest_traditional",
        help="Keypoint run selector to audit (e.g. latest, latest_traditional, latest_yolo, explicit run).",
    )
    parser.add_argument(
        "--require-review-state",
        choices=["approved", "pending", "rejected", "needs_review"],
        default="approved",
        help="Required review state when resolving reviewed runs.",
    )
    parser.add_argument(
        "--require-review-intended-use",
        choices=["training", "full_recording"],
        default="training",
        help="Required review intended_use when resolving reviewed runs.",
    )
    parser.add_argument(
        "--include-ok",
        action="store_true",
        help="Include aligned rows in text output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any issue rows are detected.",
    )
    args = parser.parse_args(argv)

    rows = audit_keypoint_resolution(
        args.paths,
        recursive=args.recursive,
        selector=args.selector,
        required_state=args.require_review_state,
        required_intended_use=args.require_review_intended_use,
    )

    if args.json:
        status_issue_count = sum(1 for row in rows if row.get("status") in ISSUE_STATUSES)
        divergence_issue_count = sum(1 for row in rows if _row_divergence_issues(row))
        payload = {
            "summary": {
                "total": len(rows),
                "status_issues": status_issue_count,
                "divergence_issues": divergence_issue_count,
                "issues": status_issue_count + divergence_issue_count,
            },
            "rows": rows,
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_text_report(rows, include_ok=args.include_ok)

    if args.strict and any(
        row.get("status") in ISSUE_STATUSES or _row_divergence_issues(row) for row in rows
    ):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
