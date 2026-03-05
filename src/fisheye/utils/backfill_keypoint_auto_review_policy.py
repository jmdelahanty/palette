#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import zarr

from fisheye.utils.auto_keypoint_review import AutoReviewPolicy, apply_auto_review


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


@dataclass
class BackfillCounts:
    scanned: int = 0
    zarr_filtered: int = 0
    no_refined_runs: int = 0
    runs_seen: int = 0
    patched: int = 0
    created: int = 0
    unchanged: int = 0
    skipped_non_algorithmic: int = 0
    skipped_missing_status: int = 0
    errors: int = 0


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_mapping(value: object) -> Optional[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return value
    return None


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = _decode_text(root.attrs.get("zarr_purpose"))
    if purpose is not None:
        value = purpose.lower()
        if value in {"analysis", "training"}:
            return value
    use_attr = _decode_text(root.attrs.get("zarr_use"))
    if use_attr is not None:
        value = use_attr.lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        candidates: list[Path] = []
        if root.suffix == ".zarr" and (root.is_file() or root.is_dir()):
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(root.rglob("*.zarr"))
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def _iter_refined_runs(parent: zarr.Group, *, latest_only: bool) -> list[str]:
    if latest_only:
        latest = _decode_text(parent.attrs.get("latest"))
        if latest and latest in parent:
            return [latest]
    try:
        names = [str(name) for name in parent.group_keys()]
    except Exception:
        names = [str(name) for name in parent.keys()]
    return sorted(names)


def _patch_auto_review_metadata(
    status: Mapping[str, object],
    *,
    policy_id: str,
    policy_version: int,
    force: bool,
) -> tuple[dict[str, object], bool]:
    now_utc = datetime.now(timezone.utc).isoformat()
    out = dict(status)
    auto_review = dict(_coerce_mapping(out.get("auto_review")) or {})
    changed = False

    current_policy_id = _decode_text(auto_review.get("policy_id"))
    if force or current_policy_id is None:
        auto_review["policy_id"] = policy_id
        changed = True

    current_policy_version = _as_int(auto_review.get("policy_version"))
    if force or current_policy_version is None:
        auto_review["policy_version"] = int(policy_version)
        changed = True

    current_result = _decode_text(auto_review.get("result"))
    if force or current_result is None:
        state = _decode_text(out.get("state"))
        auto_review["result"] = "approved" if (state and state.lower() == "approved") else "needs_review"
        changed = True

    current_applied = _decode_text(auto_review.get("applied_at_utc"))
    if force or current_applied is None:
        timestamp_utc = _decode_text(out.get("timestamp_utc")) or _decode_text(out.get("timestamp"))
        auto_review["applied_at_utc"] = timestamp_utc or now_utc
        changed = True

    if changed:
        out["auto_review"] = auto_review
    return out, changed


def _build_policy(args: argparse.Namespace) -> AutoReviewPolicy:
    tags = tuple(args.disqualifying_tag) if args.disqualifying_tag else AutoReviewPolicy.disqualifying_tags
    return AutoReviewPolicy(
        policy_id=str(args.policy_id),
        policy_version=int(args.policy_version),
        remaining_failures_max=int(args.remaining_failures_max),
        refined_success_rate_min=float(args.refined_success_rate_min),
        usable_keypoints_rate_min=float(args.usable_rate_min),
        confidence_valid_rate_min=float(args.confidence_valid_rate_min),
        geometry_valid_rate_min=float(args.geometry_valid_rate_min),
        disqualifying_tags=tags,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill keypoint auto-review policy metadata on existing algorithmic review statuses, "
            "and optionally create missing algorithmic review statuses."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all refined runs (default: latest run only).",
    )
    parser.add_argument(
        "--write-missing",
        action="store_true",
        help="Create missing keypoint_review_status via policy evaluation.",
    )
    parser.add_argument("--policy-id", default=AutoReviewPolicy.policy_id, help="Auto-review policy identifier.")
    parser.add_argument("--policy-version", type=int, default=AutoReviewPolicy.policy_version, help="Auto-review policy version.")
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
        choices=("full_recording", "training"),
        default="full_recording",
        help="Intended-use for --write-missing created statuses.",
    )
    parser.add_argument("--reviewer", help="Optional reviewer for --write-missing.")
    parser.add_argument("--notes", help="Optional notes for --write-missing.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing policy_id/policy_version in auto_review metadata.",
    )
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    parser.add_argument("--json", action="store_true", help="Emit per-run JSON lines.")
    args = parser.parse_args(argv)

    policy = _build_policy(args)
    counts = BackfillCounts()
    roots = _resolve_roots(list(args.paths))
    latest_only = not bool(args.all_runs)

    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        counts.scanned += 1
        mode = "a" if args.apply else "r"
        try:
            root = zarr.open_group(str(zarr_path), mode=mode)
        except Exception as exc:
            counts.errors += 1
            if args.json:
                print(json.dumps({"zarr": str(zarr_path), "status": "error_open", "error": str(exc)}, sort_keys=True))
            else:
                print(f"error: {zarr_path}: {exc}")
            continue

        observed_use = _infer_zarr_use(root, zarr_path)
        if args.zarr_use != "any" and observed_use != args.zarr_use:
            counts.zarr_filtered += 1
            continue

        parent = _pick_refined_parent(root)
        if parent is None:
            counts.no_refined_runs += 1
            continue
        run_names = _iter_refined_runs(parent, latest_only=latest_only)
        if not run_names:
            counts.no_refined_runs += 1
            continue

        for run_name in run_names:
            counts.runs_seen += 1
            run_group = parent[run_name]
            status = _coerce_mapping(run_group.attrs.get("keypoint_review_status"))

            if status is None:
                if not args.write_missing:
                    counts.skipped_missing_status += 1
                    continue
                try:
                    result = apply_auto_review(
                        zarr_path,
                        refined_run=run_name,
                        policy=policy,
                        intended_use=str(args.intended_use),
                        reviewer=args.reviewer,
                        notes=args.notes,
                        dry_run=not bool(args.apply),
                        overwrite_existing=False,
                        update_latest=True,
                    )
                    counts.created += 1
                    if args.json:
                        print(
                            json.dumps(
                                {
                                    "zarr": str(zarr_path),
                                    "run": run_name,
                                    "status": "created" if args.apply else "would_create",
                                    "result_state": result.get("state"),
                                    "passed": result.get("passed"),
                                },
                                sort_keys=True,
                            )
                        )
                    else:
                        action = "created" if args.apply else "would_create"
                        print(f"{action}: {zarr_path} {run_name}")
                except Exception as exc:
                    counts.errors += 1
                    if args.json:
                        print(
                            json.dumps(
                                {
                                    "zarr": str(zarr_path),
                                    "run": run_name,
                                    "status": "error_create",
                                    "error": str(exc),
                                },
                                sort_keys=True,
                            )
                        )
                    else:
                        print(f"error: {zarr_path} {run_name}: {exc}")
                continue

            method = _decode_text(status.get("method"))
            if method is None or method.lower() != "algorithmic":
                counts.skipped_non_algorithmic += 1
                continue

            patched_status, changed = _patch_auto_review_metadata(
                status,
                policy_id=str(policy.policy_id),
                policy_version=int(policy.policy_version),
                force=bool(args.force),
            )
            if not changed:
                counts.unchanged += 1
                continue

            if args.apply:
                attrs = dict(run_group.attrs)
                attrs["keypoint_review_status"] = patched_status
                run_group.attrs.put(attrs)
            counts.patched += 1
            if args.json:
                print(
                    json.dumps(
                        {
                            "zarr": str(zarr_path),
                            "run": run_name,
                            "status": "patched" if args.apply else "would_patch",
                            "policy_id": policy.policy_id,
                            "policy_version": policy.policy_version,
                        },
                        sort_keys=True,
                    )
                )
            else:
                action = "patched" if args.apply else "would_patch"
                print(f"{action}: {zarr_path} {run_name}")

    mode = "Applied" if args.apply else "Dry-run"
    print(
        "Keypoint auto-review policy backfill: "
        f"mode={mode} "
        f"zarr_scanned={counts.scanned} "
        f"zarr_filtered={counts.zarr_filtered} "
        f"runs_seen={counts.runs_seen} "
        f"patched={counts.patched} "
        f"created={counts.created} "
        f"unchanged={counts.unchanged} "
        f"skipped_non_algorithmic={counts.skipped_non_algorithmic} "
        f"skipped_missing_status={counts.skipped_missing_status} "
        f"no_refined_runs={counts.no_refined_runs} "
        f"errors={counts.errors}"
    )
    return 0 if counts.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
