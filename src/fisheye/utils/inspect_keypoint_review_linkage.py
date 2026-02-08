#!/usr/bin/env python3
"""Inspect keypoint/review linkage for a single Zarr archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import zarr

from fisheye.utils import prepare_keypoint_training_from_registry as prep_pose


def _method_hint_for_selector(normalized_selector: Optional[str]) -> Optional[str]:
    if normalized_selector == "latest_traditional":
        return "traditional_pose"
    if normalized_selector == "latest_yolo":
        return "yolo_pose"
    return None


def _collect_keypoint_runs(root: zarr.Group) -> Dict[str, Any]:
    parent = root.get("keypoints_runs")
    if parent is None:
        return {"latest": None, "runs": []}

    rows: List[Dict[str, Any]] = []
    for run_name in parent.group_keys():
        run_group = parent[run_name]
        rows.append(
            {
                "run": str(run_name),
                "method": prep_pose._decode_attr(run_group.attrs.get("method")),
                "keypoints_timestamp_utc": prep_pose._decode_attr(run_group.attrs.get("keypoints_timestamp_utc")),
                "source_crop_run": prep_pose._decode_attr(run_group.attrs.get("source_crop_run")),
            }
        )

    rows.sort(key=lambda row: row["run"])
    return {
        "latest": prep_pose._decode_attr(parent.attrs.get("latest")),
        "runs": rows,
    }


def _collect_refined_runs(
    root: zarr.Group,
    keypoint_runs: Mapping[str, Dict[str, Any]],
    *,
    zarr_path: Path,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for group_name in ("refined_keypoints_runs", "keypoints_refined_runs"):
        parent = root.get(group_name)
        if parent is None:
            continue

        rows: List[Dict[str, Any]] = []
        for run_name in parent.group_keys():
            run_group = parent[run_name]
            source_run = prep_pose._decode_attr(run_group.attrs.get("source_keypoints_run"))
            review_sources = prep_pose._resolve_review_status_sources(
                run_group,
                zarr_path=zarr_path,
                refined_parent_name=group_name,
                refined_run_name=str(run_name),
            )
            review_status = review_sources.get("effective")
            if isinstance(review_status, Mapping):
                review_status = dict(review_status)
            else:
                review_status = None
            source_method = None
            source_exists = False
            if source_run and source_run in keypoint_runs:
                source_exists = True
                source_method = keypoint_runs[source_run].get("method")

            rows.append(
                {
                    "run": str(run_name),
                    "created_utc": prep_pose._decode_attr(
                        run_group.attrs.get("created_utc") or run_group.attrs.get("timestamp_utc")
                    ),
                    "source_keypoints_run": source_run,
                    "source_exists": source_exists,
                    "source_method": source_method,
                    "review_status": review_status,
                    "review_status_attrs": review_sources.get("attrs"),
                    "review_status_disk": review_sources.get("disk"),
                    "review_status_divergence": review_sources.get("divergence"),
                }
            )
        rows.sort(key=lambda row: row["run"])
        groups.append(
            {
                "group": group_name,
                "latest": prep_pose._decode_attr(parent.attrs.get("latest")),
                "runs": rows,
            }
        )
    return groups


def inspect_linkage(
    zarr_path: Path,
    *,
    selector: Optional[str],
    required_state: Optional[str],
    required_intended_use: Optional[str],
) -> Dict[str, Any]:
    try:
        root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
    except TypeError:
        root = zarr.open_group(str(zarr_path), mode="r")

    keypoint_info = _collect_keypoint_runs(root)
    keypoint_runs_by_name = {row["run"]: row for row in keypoint_info["runs"]}
    refined_groups = _collect_refined_runs(root, keypoint_runs_by_name, zarr_path=zarr_path)

    normalized_selector = prep_pose._normalize_keypoint_run_selector(selector)
    method_hint = _method_hint_for_selector(normalized_selector)

    payload: Dict[str, Any] = {
        "zarr_path": str(zarr_path),
        "selector_requested": selector,
        "selector_normalized": normalized_selector,
        "keypoints_runs": keypoint_info,
        "refined_groups": refined_groups,
    }

    try:
        resolved_run, selector_resolved = prep_pose._resolve_keypoint_run(root, selector)
        resolved_quality = prep_pose._resolve_refined_keypoint_quality(root, resolved_run, zarr_path=zarr_path)
        resolved_review = resolved_quality.get("keypoint_review_status")

        strict_reviewed = prep_pose._resolve_reviewed_keypoint_run(
            root,
            method_hint=method_hint,
            required_state=required_state,
            required_intended_use=required_intended_use,
            zarr_path=zarr_path,
        )
        relaxed_reviewed = None
        if strict_reviewed is None and method_hint is not None:
            relaxed_reviewed = prep_pose._resolve_reviewed_keypoint_run(
                root,
                method_hint=None,
                required_state=required_state,
                required_intended_use=required_intended_use,
                zarr_path=zarr_path,
            )

        payload["resolution"] = {
            "selector_resolved": selector_resolved,
            "resolved_keypoint_run": resolved_run,
            "resolved_method": keypoint_runs_by_name.get(resolved_run, {}).get("method"),
            "resolved_review_status": resolved_review,
            "resolved_review_divergence": resolved_quality.get("keypoint_review_status_divergence"),
            "strict_reviewed_choice": strict_reviewed,
            "relaxed_reviewed_choice": relaxed_reviewed,
        }
    except Exception as exc:
        payload["resolution_error"] = str(exc)

    issues: List[str] = []
    resolution = payload.get("resolution")
    if isinstance(resolution, Mapping):
        strict_choice = resolution.get("strict_reviewed_choice")
        relaxed_choice = resolution.get("relaxed_reviewed_choice")
        resolved_run = resolution.get("resolved_keypoint_run")
        if strict_choice is None and relaxed_choice is None:
            issues.append("no_review_match")
        elif strict_choice is None and relaxed_choice is not None:
            issues.append("cross_method_review_only")
        elif isinstance(strict_choice, Mapping) and strict_choice.get("source_keypoint_run") != resolved_run:
            issues.append("selector_resolved_differs_from_reviewed_source")

    for group in refined_groups:
        for run in group.get("runs", []):
            if not run.get("source_exists"):
                issues.append(f"orphan_refined_source:{group['group']}/{run.get('run')}")
            review_status = run.get("review_status")
            if review_status is None:
                issues.append(f"missing_review_status:{group['group']}/{run.get('run')}")
            divergence = run.get("review_status_divergence")
            if divergence in {"conflict", "attrs_missing_disk_present", "attrs_present_disk_missing"}:
                issues.append(f"review_status_divergence:{divergence}:{group['group']}/{run.get('run')}")

    payload["issues"] = sorted(set(issues))
    return payload


def _print_text_report(payload: Dict[str, Any]) -> None:
    print(f"Archive: {payload['zarr_path']}")
    print(f"Selector requested: {payload.get('selector_requested')}")
    print(f"Selector normalized: {payload.get('selector_normalized')}")
    print()

    keypoint_info = payload.get("keypoints_runs", {})
    print(f"keypoints_runs latest: {keypoint_info.get('latest')}")
    print("Keypoint runs:")
    runs = keypoint_info.get("runs", [])
    if not runs:
        print("- none")
    else:
        for row in runs:
            print(
                f"- {row.get('run')}: method={row.get('method')} "
                f"keypoints_timestamp_utc={row.get('keypoints_timestamp_utc')} "
                f"source_crop_run={row.get('source_crop_run')}"
            )
    print()

    print("Refined groups:")
    refined_groups = payload.get("refined_groups", [])
    if not refined_groups:
        print("- none")
    else:
        for group in refined_groups:
            print(f"- {group.get('group')} (latest={group.get('latest')})")
            for row in group.get("runs", []):
                review_status = row.get("review_status")
                if isinstance(review_status, Mapping):
                    review_state = prep_pose._decode_attr(review_status.get("state"))
                    review_use = prep_pose._decode_attr(review_status.get("intended_use"))
                    review_text = f"{review_state}/{review_use}"
                else:
                    review_text = "none"
                print(
                    f"  - {row.get('run')}: source={row.get('source_keypoints_run')} "
                    f"(exists={row.get('source_exists')}, method={row.get('source_method')}) "
                    f"created_utc={row.get('created_utc')} review={review_text} "
                    f"divergence={row.get('review_status_divergence')}"
                )
    print()

    if payload.get("resolution_error"):
        print(f"Resolution error: {payload['resolution_error']}")
    else:
        resolution = payload.get("resolution", {})
        print("Resolution:")
        print(
            f"- selector_resolved={resolution.get('selector_resolved')} "
            f"resolved_keypoint_run={resolution.get('resolved_keypoint_run')} "
            f"resolved_method={resolution.get('resolved_method')}"
        )
        print(f"- resolved_review_divergence={resolution.get('resolved_review_divergence')}")
        print(f"- strict_reviewed_choice={resolution.get('strict_reviewed_choice')}")
        print(f"- relaxed_reviewed_choice={resolution.get('relaxed_reviewed_choice')}")
    print()

    print("Issues:")
    issues = payload.get("issues", [])
    if issues:
        for issue in issues:
            print(f"- {issue}")
    else:
        print("- none")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to a .zarr archive.")
    parser.add_argument(
        "--selector",
        default="latest_traditional",
        help="Keypoint run selector to inspect (e.g. latest, latest_traditional, latest_yolo, explicit run).",
    )
    parser.add_argument(
        "--require-review-state",
        choices=["approved", "pending", "rejected", "needs_review"],
        default="approved",
        help="Required review state for reviewed-choice analysis.",
    )
    parser.add_argument(
        "--require-review-intended-use",
        choices=["training", "full_recording"],
        default="training",
        help="Required review intended_use for reviewed-choice analysis.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    payload = inspect_linkage(
        args.zarr_path.expanduser(),
        selector=args.selector,
        required_state=args.require_review_state,
        required_intended_use=args.require_review_intended_use,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_text_report(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
