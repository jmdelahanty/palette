#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels


@dataclass(frozen=True)
class AutoReviewPolicy:
    policy_id: str = "keypoint_auto_review_v1"
    policy_version: int = 1
    remaining_failures_max: int = 0
    refined_success_rate_min: float = 1.0
    usable_keypoints_rate_min: float = 0.999
    confidence_valid_rate_min: float = 0.999
    geometry_valid_rate_min: float = 0.999
    disqualifying_tags: tuple[str, ...] = ("detection_issue",)


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _coerce_mapping(value: object) -> Optional[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return value
    return None


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    parent = root.get("refined_keypoints_runs")
    if parent is not None:
        return parent
    return root.get("keypoints_refined_runs")


def _refined_run_source_keypoints_run(run_group: zarr.Group) -> Optional[str]:
    return _decode_text(run_group.attrs.get("source_keypoints_run")) or _decode_text(
        run_group.attrs.get("source_keypoint_run")
    )


def _select_refined_run(parent: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested in parent:
            return requested
        raise RuntimeError(f"Refined keypoint run '{requested}' not found.")
    latest = _decode_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        raise RuntimeError("No refined keypoint runs available.")
    return str(sorted(names)[-1])


def _select_refined_run_for_source(parent: zarr.Group, source_keypoints_run: str) -> str:
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    names = [str(name) for name in names]
    if not names:
        raise RuntimeError("No refined keypoint runs available.")

    latest = _decode_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        try:
            if _refined_run_source_keypoints_run(parent[latest]) == source_keypoints_run:
                return latest
        except Exception:
            pass

    matches: list[str] = []
    for run_name in sorted(names):
        try:
            if _refined_run_source_keypoints_run(parent[run_name]) == source_keypoints_run:
                matches.append(run_name)
        except Exception:
            continue
    if not matches:
        raise RuntimeError(
            f"No refined keypoint run found for source_keypoints_run '{source_keypoints_run}'."
        )
    return matches[-1]


def _count_truthy(arr: Optional[zarr.Array]) -> Optional[int]:
    if arr is None:
        return None
    data = np.asarray(arr[:], dtype=bool)
    return int(np.sum(data))


def _count_reason_tags(refined: zarr.Group) -> Dict[str, int]:
    reason_labels = read_reason_labels(refined)
    if reason_labels is None:
        return {}
    raw = np.asarray(reason_labels, dtype=object)
    counts: Dict[str, int] = {}
    for value in raw:
        if value is None:
            continue
        text = str(value)
        if not text:
            continue
        for tag in text.split("|"):
            normalized = tag.strip()
            if not normalized:
                continue
            counts[normalized] = int(counts.get(normalized, 0) + 1)
    return counts


def _summary_counts(refined: zarr.Group) -> Dict[str, object]:
    summary = _coerce_mapping(refined.attrs.get("summary_statistics")) or {}
    postprocess = _coerce_mapping(summary.get("postprocess")) or {}
    refine = _coerce_mapping(summary.get("refine")) or {}

    def _pick_int(*values: object) -> Optional[int]:
        for value in values:
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return None

    reason_counts = _coerce_mapping(postprocess.get("reason_counts")) or {}
    reason_counts_out: Dict[str, int] = {}
    for key, value in reason_counts.items():
        try:
            reason_counts_out[str(key)] = int(value)
        except Exception:
            continue

    return {
        "total_rois": _pick_int(
            postprocess.get("total_rois"),
            refine.get("total_rois"),
            summary.get("total_rois"),
        ),
        "refined_success": _pick_int(
            postprocess.get("refined_success"),
            refine.get("refined_success"),
            summary.get("refined_success"),
        ),
        "usable_keypoints": _pick_int(
            postprocess.get("usable_keypoints"),
            refine.get("usable_keypoints"),
            summary.get("usable_keypoints"),
        ),
        "confidence_valid": _pick_int(
            postprocess.get("confidence_valid"),
            refine.get("confidence_valid"),
            summary.get("confidence_valid"),
        ),
        "geometry_valid": _pick_int(
            postprocess.get("geometry_valid"),
            refine.get("geometry_valid"),
            summary.get("geometry_valid"),
        ),
        "remaining_failures": _pick_int(
            postprocess.get("remaining_failures"),
            refine.get("remaining_failures"),
            summary.get("remaining_failures"),
        ),
        "reason_counts": reason_counts_out,
    }


def _ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _evaluate_refined_run(
    refined: zarr.Group,
    *,
    refined_run: str,
    source_keypoints_run: Optional[str],
    policy: AutoReviewPolicy,
) -> Dict[str, object]:
    counts = _summary_counts(refined)

    total_rois = counts.get("total_rois")
    if not isinstance(total_rois, int):
        try:
            total_rois = int(refined["keypoints_roi"].shape[0])
        except Exception:
            total_rois = None

    refined_success = _count_truthy(refined.get("refined_success"))
    if refined_success is None and isinstance(counts.get("refined_success"), int):
        refined_success = int(counts["refined_success"])

    usable_keypoints = _count_truthy(refined.get("usable_keypoints"))
    if usable_keypoints is None and isinstance(counts.get("usable_keypoints"), int):
        usable_keypoints = int(counts["usable_keypoints"])

    confidence_valid = _count_truthy(refined.get("confidence_valid"))
    if confidence_valid is None and isinstance(counts.get("confidence_valid"), int):
        confidence_valid = int(counts["confidence_valid"])

    geometry_valid = _count_truthy(refined.get("geometry_valid"))
    if geometry_valid is None and isinstance(counts.get("geometry_valid"), int):
        geometry_valid = int(counts["geometry_valid"])

    remaining_failures = counts.get("remaining_failures")
    if not isinstance(remaining_failures, int):
        if isinstance(total_rois, int) and isinstance(refined_success, int):
            remaining_failures = max(0, total_rois - refined_success)
        else:
            remaining_failures = None

    reason_counts = _count_reason_tags(refined)
    if not reason_counts and isinstance(counts.get("reason_counts"), dict):
        reason_counts = {str(k): int(v) for k, v in counts["reason_counts"].items()}

    refined_success_rate = _ratio(
        refined_success if isinstance(refined_success, int) else None,
        total_rois if isinstance(total_rois, int) else None,
    )
    usable_rate = _ratio(
        usable_keypoints if isinstance(usable_keypoints, int) else None,
        total_rois if isinstance(total_rois, int) else None,
    )
    confidence_valid_rate = _ratio(
        confidence_valid if isinstance(confidence_valid, int) else None,
        total_rois if isinstance(total_rois, int) else None,
    )
    geometry_valid_rate = _ratio(
        geometry_valid if isinstance(geometry_valid, int) else None,
        total_rois if isinstance(total_rois, int) else None,
    )

    disqualifying_tag_counts = {
        tag: int(reason_counts.get(tag, 0))
        for tag in policy.disqualifying_tags
    }
    disqualifying_total = int(sum(disqualifying_tag_counts.values()))

    checks: Dict[str, Dict[str, object]] = {}

    rem_pass = isinstance(remaining_failures, int) and int(remaining_failures) <= int(policy.remaining_failures_max)
    checks["remaining_failures"] = {
        "observed": remaining_failures,
        "max": int(policy.remaining_failures_max),
        "pass": bool(rem_pass),
    }

    success_pass = (
        isinstance(refined_success_rate, float)
        and float(refined_success_rate) >= float(policy.refined_success_rate_min)
    )
    checks["refined_success_rate"] = {
        "observed": refined_success_rate,
        "min": float(policy.refined_success_rate_min),
        "pass": bool(success_pass),
    }

    usable_pass = isinstance(usable_rate, float) and float(usable_rate) >= float(policy.usable_keypoints_rate_min)
    checks["usable_keypoints_rate"] = {
        "observed": usable_rate,
        "min": float(policy.usable_keypoints_rate_min),
        "pass": bool(usable_pass),
    }

    confidence_pass = (
        isinstance(confidence_valid_rate, float)
        and float(confidence_valid_rate) >= float(policy.confidence_valid_rate_min)
    )
    checks["confidence_valid_rate"] = {
        "observed": confidence_valid_rate,
        "min": float(policy.confidence_valid_rate_min),
        "pass": bool(confidence_pass),
    }

    geometry_pass = (
        isinstance(geometry_valid_rate, float)
        and float(geometry_valid_rate) >= float(policy.geometry_valid_rate_min)
    )
    checks["geometry_valid_rate"] = {
        "observed": geometry_valid_rate,
        "min": float(policy.geometry_valid_rate_min),
        "pass": bool(geometry_pass),
    }

    tag_pass = disqualifying_total == 0
    checks["disqualifying_tags"] = {
        "observed_total": disqualifying_total,
        "tags": dict(disqualifying_tag_counts),
        "pass": bool(tag_pass),
    }

    passed = bool(all(bool(item.get("pass")) for item in checks.values()))

    evidence: Dict[str, object] = {
        "refined_run": refined_run,
        "source_keypoints_run": source_keypoints_run,
        "total_rois": total_rois,
        "remaining_failures": remaining_failures,
        "refined_success_rate": refined_success_rate,
        "usable_keypoints_rate": usable_rate,
        "confidence_valid_rate": confidence_valid_rate,
        "geometry_valid_rate": geometry_valid_rate,
        "reason_counts": reason_counts,
    }

    return {
        "passed": passed,
        "checks": checks,
        "evidence": evidence,
    }


def apply_auto_review(
    zarr_path: Path,
    *,
    refined_run: Optional[str] = None,
    source_keypoints_run: Optional[str] = None,
    policy: Optional[AutoReviewPolicy] = None,
    intended_use: str = "full_recording",
    reviewer: Optional[str] = None,
    notes: Optional[str] = None,
    state_on_pass: str = "approved",
    state_on_fail: str = "needs_review",
    overwrite_existing: bool = False,
    update_latest: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    if intended_use not in {"training", "full_recording"}:
        raise ValueError(f"Unsupported intended_use '{intended_use}'.")
    policy = policy or AutoReviewPolicy()

    mode = "r" if dry_run else "a"
    root = zarr.open_group(str(zarr_path), mode=mode)
    parent = _pick_refined_parent(root)
    if parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")
    if refined_run is None and source_keypoints_run is not None:
        selected_run = _select_refined_run_for_source(parent, source_keypoints_run)
    else:
        selected_run = _select_refined_run(parent, refined_run)
    refined_group = parent[selected_run]
    resolved_source_keypoints_run = _refined_run_source_keypoints_run(refined_group)

    existing_status = _coerce_mapping(refined_group.attrs.get("keypoint_review_status")) or {}
    existing_state = _decode_text(existing_status.get("state"))
    if existing_state and not overwrite_existing:
        existing_method = _decode_text(existing_status.get("method"))
        return {
            "zarr_path": str(zarr_path),
            "refined_run": selected_run,
            "source_keypoints_run": resolved_source_keypoints_run,
            "state": existing_state,
            "passed": None,
            "dry_run": bool(dry_run),
            "latest_updated": False,
            "skipped": True,
            "skip_reason": "existing_review_status",
            "existing_review_method": existing_method,
            "payload": existing_status,
        }

    evaluation = _evaluate_refined_run(
        refined_group,
        refined_run=selected_run,
        source_keypoints_run=resolved_source_keypoints_run,
        policy=policy,
    )
    passed = bool(evaluation["passed"])
    state = state_on_pass if passed else state_on_fail
    now_utc = datetime.now(timezone.utc).isoformat()
    policy_reviewer = reviewer or f"auto:{policy.policy_id}"
    if notes is None:
        notes = (
            "Auto-approved by keypoint policy."
            if passed
            else "Auto-review did not meet policy thresholds; manual review required."
        )

    payload: Dict[str, object] = {
        "state": state,
        "method": "algorithmic",
        "intended_use": intended_use,
        "timestamp_utc": now_utc,
        "timestamp": now_utc,
        "reviewer": policy_reviewer,
        "notes": notes,
        "auto_review": {
            "policy_id": policy.policy_id,
            "policy_version": int(policy.policy_version),
            "result": "approved" if passed else "needs_review",
            "applied_at_utc": now_utc,
            "thresholds": {
                "remaining_failures_max": int(policy.remaining_failures_max),
                "refined_success_rate_min": float(policy.refined_success_rate_min),
                "usable_keypoints_rate_min": float(policy.usable_keypoints_rate_min),
                "confidence_valid_rate_min": float(policy.confidence_valid_rate_min),
                "geometry_valid_rate_min": float(policy.geometry_valid_rate_min),
                "disqualifying_tags": list(policy.disqualifying_tags),
            },
            "checks": evaluation["checks"],
            "evidence": evaluation["evidence"],
            "drift": {
                "enabled": False,
                "status": "not_evaluated",
                "reference_profile_id": None,
                "max_zscore": None,
                "outlier_metric_count": None,
            },
        },
    }

    latest_updated = False
    if not dry_run:
        run_attrs = dict(refined_group.attrs)
        run_attrs["keypoint_review_status"] = payload
        refined_group.attrs.put(run_attrs)
        if update_latest:
            parent_attrs = dict(parent.attrs)
            parent_attrs["keypoint_review_status_latest"] = selected_run
            parent.attrs.put(parent_attrs)
            latest_updated = True

    return {
        "zarr_path": str(zarr_path),
        "refined_run": selected_run,
        "source_keypoints_run": resolved_source_keypoints_run,
        "state": state,
        "passed": passed,
        "dry_run": bool(dry_run),
        "latest_updated": latest_updated,
        "skipped": False,
        "payload": payload,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and write algorithmic keypoint review status for a refined keypoint run."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the recording .zarr directory.")
    parser.add_argument("--refined-run", help="Refined keypoint run name (default: latest).")
    parser.add_argument(
        "--source-keypoints-run",
        help="Resolve refined run by source_keypoints_run (used when --refined-run is omitted).",
    )
    parser.add_argument("--policy-id", default=AutoReviewPolicy.policy_id, help="Policy identifier.")
    parser.add_argument("--policy-version", type=int, default=AutoReviewPolicy.policy_version, help="Policy version.")
    parser.add_argument("--remaining-failures-max", type=int, default=AutoReviewPolicy.remaining_failures_max)
    parser.add_argument("--refined-success-rate-min", type=float, default=AutoReviewPolicy.refined_success_rate_min)
    parser.add_argument("--usable-rate-min", type=float, default=AutoReviewPolicy.usable_keypoints_rate_min)
    parser.add_argument("--confidence-valid-rate-min", type=float, default=AutoReviewPolicy.confidence_valid_rate_min)
    parser.add_argument("--geometry-valid-rate-min", type=float, default=AutoReviewPolicy.geometry_valid_rate_min)
    parser.add_argument(
        "--disqualifying-tag",
        action="append",
        default=None,
        help="Reason tag that forces needs_review (repeatable).",
    )
    parser.add_argument(
        "--intended-use",
        choices=["training", "full_recording"],
        default="full_recording",
        help="Review intended_use (default: full_recording).",
    )
    parser.add_argument(
        "--state-on-pass",
        choices=["approved", "pending", "rejected", "needs_review"],
        default="approved",
    )
    parser.add_argument(
        "--state-on-fail",
        choices=["approved", "pending", "rejected", "needs_review"],
        default="needs_review",
    )
    parser.add_argument("--reviewer", help="Reviewer identifier (default auto:<policy_id>).")
    parser.add_argument("--notes", help="Optional notes to include.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate but do not write attrs.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing keypoint_review_status if present.",
    )
    parser.add_argument("--no-latest", action="store_true", help="Do not update parent latest review pointer.")
    parser.add_argument("--json", action="store_true", help="Emit JSON result.")
    args = parser.parse_args(argv)

    tags = tuple(args.disqualifying_tag) if args.disqualifying_tag else AutoReviewPolicy.disqualifying_tags
    policy = AutoReviewPolicy(
        policy_id=str(args.policy_id),
        policy_version=int(args.policy_version),
        remaining_failures_max=int(args.remaining_failures_max),
        refined_success_rate_min=float(args.refined_success_rate_min),
        usable_keypoints_rate_min=float(args.usable_rate_min),
        confidence_valid_rate_min=float(args.confidence_valid_rate_min),
        geometry_valid_rate_min=float(args.geometry_valid_rate_min),
        disqualifying_tags=tags,
    )
    result = apply_auto_review(
        args.zarr_path,
        refined_run=args.refined_run,
        source_keypoints_run=args.source_keypoints_run,
        policy=policy,
        intended_use=str(args.intended_use),
        reviewer=args.reviewer,
        notes=args.notes,
        state_on_pass=str(args.state_on_pass),
        state_on_fail=str(args.state_on_fail),
        overwrite_existing=bool(args.overwrite_existing),
        update_latest=not bool(args.no_latest),
        dry_run=bool(args.dry_run),
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"zarr_path: {result['zarr_path']}")
        print(f"refined_run: {result['refined_run']}")
        print(f"state: {result['state']}")
        print(f"passed: {result['passed']}")
        print(f"dry_run: {result['dry_run']}")
        print(f"latest_updated: {result['latest_updated']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
