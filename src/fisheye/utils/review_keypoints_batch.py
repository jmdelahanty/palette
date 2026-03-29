#!/usr/bin/env python3
"""
Batch wrapper for keypoint review (manual/retune/audit) across many recordings.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels


@dataclass
class ReviewPlan:
    zarr_path: Path
    refined_run: Optional[str]
    status: str
    reason: Optional[str] = None
    review_state: Optional[str] = None


def _iter_zarr(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_dir() and path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def _load_paths_file(path: Path) -> List[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    items: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _latest_run(parent: zarr.Group) -> Optional[str]:
    latest = parent.attrs.get("latest")
    if latest and latest in parent:
        return str(latest)
    if hasattr(parent, "group_keys"):
        names = list(parent.group_keys())
    else:
        names = list(parent.keys())
    if not names:
        return None
    return sorted(names)[-1]


def _normalize_optional_text(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    return text


def _normalize_review_state(raw: object) -> Optional[str]:
    state = raw.get("state") if isinstance(raw, dict) else raw
    return _normalize_optional_text(state)


def _extract_pose_schema_name(raw: object) -> Optional[str]:
    if isinstance(raw, dict):
        name = raw.get("name")
        if name is not None:
            text = str(name).strip()
            return text or None
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _matches_review_state_filter(review_state: Optional[str], review_state_filter: str) -> bool:
    filt = _normalize_optional_text(review_state_filter) or "any"
    if filt in {"any", "all"}:
        return True
    if filt == "missing":
        return review_state is None
    if filt in {"not_approved", "not-approved"}:
        return review_state != "approved"
    return review_state == filt


def _relation_has_columns(conn: sqlite3.Connection, relation: str, required: tuple[str, ...]) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({relation});").fetchall()
    except sqlite3.Error:
        return False
    names = {str(row["name"]) for row in rows if row["name"] is not None}
    return all(name in names for name in required)


def _normalize_scope_roots(paths: List[Path]) -> List[Path]:
    roots: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = path.expanduser().resolve(strict=False)
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        roots.append(normalized)
    return roots


def _is_in_scope(target: Path, scope_roots: List[Path]) -> bool:
    if not scope_roots:
        return True
    normalized_target = target.expanduser().resolve(strict=False)
    for root in scope_roots:
        if root.suffix == ".zarr":
            if normalized_target == root:
                return True
            continue
        try:
            normalized_target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        raw = root.attrs.get(key)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _run_matches_pose_schema(refined_group: zarr.Group, pose_schema: Optional[str]) -> bool:
    wanted = _normalize_optional_text(pose_schema)
    if wanted is None:
        return True
    observed = _extract_pose_schema_name(refined_group.attrs.get("pose_schema"))
    return _normalize_optional_text(observed) == wanted


def _latest_matching_run(parent: zarr.Group, pose_schema: Optional[str]) -> Optional[str]:
    latest = parent.attrs.get("latest")
    if latest and latest in parent and _run_matches_pose_schema(parent[latest], pose_schema):
        return str(latest)

    if hasattr(parent, "group_keys"):
        names = list(parent.group_keys())
    else:
        names = list(parent.keys())
    if not names:
        return None
    matching = [str(name) for name in names if _run_matches_pose_schema(parent[str(name)], pose_schema)]
    if not matching:
        return None
    return sorted(matching)[-1]


def _extract_failures_from_refined_group(refined_group: zarr.Group) -> Optional[np.ndarray]:
    if "refined_success" in refined_group:
        success = np.asarray(refined_group["refined_success"][:], dtype=bool)
        failures = np.where(~success)[0].astype("i4", copy=False)
    elif "source_success" in refined_group:
        success = np.asarray(refined_group["source_success"][:], dtype=bool)
        failures = np.where(~success)[0].astype("i4", copy=False)
    elif "failure_indices" in refined_group:
        failures = np.asarray(refined_group["failure_indices"][:], dtype="i4")
    else:
        return None

    if failures.size == 0:
        return failures

    reason_vals = read_reason_labels(refined_group)
    if reason_vals is None:
        return failures
    reason_vals = np.asarray(reason_vals, dtype=object)
    if reason_vals.size == 0:
        return failures

    keep_mask = []
    for idx in failures:
        try:
            text = str(reason_vals[int(idx)]) if reason_vals[int(idx)] is not None else ""
        except Exception:
            text = ""
        keep_mask.append(
            "fish_present_no_keypoints" not in text
            and "detection_issue" not in text
        )
    return failures[np.array(keep_mask, dtype=bool)]


def _run_has_review_failures(zarr_path: Path, refined_run: str) -> Optional[bool]:
    if not zarr_path.exists():
        return None
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return None

    refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    if refined_parent is None or refined_run not in refined_parent:
        return None
    refined_group = refined_parent[refined_run]

    summary_stats = refined_group.attrs.get("summary_statistics")
    if isinstance(summary_stats, dict):
        postprocess = summary_stats.get("postprocess")
        if isinstance(postprocess, dict):
            remaining = postprocess.get("remaining_failures")
            if remaining is not None:
                try:
                    return int(remaining) > 0
                except Exception:
                    pass
        remaining = summary_stats.get("remaining_failures")
        if remaining is not None:
            try:
                return int(remaining) > 0
            except Exception:
                pass

    failures = _extract_failures_from_refined_group(refined_group)
    if failures is None:
        return None
    return bool(failures.size > 0)


def _zarr_run_matches_pose_schema(zarr_path: Path, refined_run: str, pose_schema: Optional[str]) -> Optional[bool]:
    if pose_schema is None:
        return True
    if not zarr_path.exists():
        return None
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return None
    refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    if refined_parent is None or refined_run not in refined_parent:
        return None
    return _run_matches_pose_schema(refined_parent[refined_run], pose_schema)


def _build_plans_from_registry(
    registry_path: Path,
    roots: List[Path],
    refined_run: Optional[str],
    zarr_use: str,
    pose_schema: Optional[str] = None,
    require_failures: bool = False,
    review_state_filter: str = "any",
) -> List[ReviewPlan]:
    scope_roots = _normalize_scope_roots(roots)
    plans: List[ReviewPlan] = []
    try:
        with sqlite3.connect(str(registry_path)) as conn:
            conn.row_factory = sqlite3.Row
            use_filter = _normalize_optional_text(zarr_use)
            if use_filter == "any":
                use_filter = None
            if refined_run:
                has_failure_columns = _relation_has_columns(
                    conn,
                    "keypoint_quality",
                    ("total_keypoints", "usable_keypoints"),
                )
                sql = """
                    SELECT
                        kq.dataset_id AS dataset_id,
                        d.zarr_path AS zarr_path,
                        kq.refined_run AS refined_run,
                        kq.review_state AS review_state,
                        d.zarr_use AS zarr_use
                    FROM keypoint_quality kq
                    JOIN datasets d ON d.dataset_id = kq.dataset_id
                    WHERE kq.refined_run = ?
                      AND (d.status IS NULL OR lower(d.status) != 'missing')
                """
                params: list[object] = [refined_run]
                if use_filter:
                    sql += " AND lower(COALESCE(d.zarr_use, '')) = ?"
                    params.append(use_filter)
                if require_failures and has_failure_columns:
                    sql += " AND COALESCE(kq.total_keypoints, 0) > COALESCE(kq.usable_keypoints, 0)"
                sql += ";"
                rows = conn.execute(
                    sql,
                    params,
                ).fetchall()
            else:
                pose_schema_filter = _normalize_optional_text(pose_schema)
                can_scan_all_runs = pose_schema_filter is not None and _relation_has_columns(
                    conn,
                    "keypoint_quality",
                    ("dataset_id", "refined_run", "review_state"),
                )
                if can_scan_all_runs:
                    has_failure_columns = _relation_has_columns(
                        conn,
                        "keypoint_quality",
                        ("total_keypoints", "usable_keypoints"),
                    )
                    filters = [
                        "d.zarr_path IS NOT NULL",
                        "TRIM(d.zarr_path) != ''",
                        "(d.status IS NULL OR lower(d.status) != 'missing')",
                    ]
                    params = []
                    if use_filter:
                        filters.append("lower(COALESCE(d.zarr_use, '')) = ?")
                        params.append(use_filter)
                    if require_failures and has_failure_columns:
                        filters.append("COALESCE(kq.total_keypoints, 0) > COALESCE(kq.usable_keypoints, 0)")
                    rows = conn.execute(
                        f"""
                        SELECT
                            kq.dataset_id AS dataset_id,
                            kq.refined_run AS refined_run,
                            kq.review_state AS review_state,
                            d.zarr_path AS zarr_path,
                            d.zarr_use AS zarr_use,
                            COALESCE(kq.review_timestamp_utc, kq.refined_created_utc, kq.quality_updated_utc, '') AS _sort_ts,
                            COALESCE(kq.refined_created_utc, '') AS _created_ts
                        FROM keypoint_quality kq
                        JOIN datasets d ON d.dataset_id = kq.dataset_id
                        WHERE {" AND ".join(filters)}
                        ORDER BY
                            kq.dataset_id,
                            _sort_ts DESC,
                            _created_ts DESC,
                            kq.refined_run DESC;
                        """,
                        params,
                    ).fetchall()
                else:
                    has_failure_columns = _relation_has_columns(
                        conn,
                        "keypoint_quality_current",
                        ("total_keypoints", "usable_keypoints"),
                    )
                    filters = [
                        "d.zarr_path IS NOT NULL",
                        "TRIM(d.zarr_path) != ''",
                        "(d.status IS NULL OR lower(d.status) != 'missing')",
                    ]
                    params = []
                    if use_filter:
                        filters.append("lower(COALESCE(d.zarr_use, '')) = ?")
                        params.append(use_filter)
                    if require_failures and has_failure_columns:
                        filters.append("COALESCE(kqc.total_keypoints, 0) > COALESCE(kqc.usable_keypoints, 0)")
                    rows = conn.execute(
                        f"""
                        WITH ranked AS (
                            SELECT
                                kqc.dataset_id AS dataset_id,
                                kqc.refined_run AS refined_run,
                                kqc.review_state AS review_state,
                                d.zarr_path AS zarr_path,
                                d.zarr_use AS zarr_use,
                                ROW_NUMBER() OVER (
                                    PARTITION BY kqc.dataset_id
                                    ORDER BY
                                        COALESCE(kqc.review_timestamp_utc, kqc.refined_created_utc, kqc.quality_updated_utc) DESC,
                                        COALESCE(kqc.refined_created_utc, '') DESC,
                                    kqc.refined_run DESC
                                ) AS _rn
                            FROM keypoint_quality_current kqc
                            JOIN datasets d ON d.dataset_id = kqc.dataset_id
                            WHERE {" AND ".join(filters)}
                        )
                        SELECT
                            dataset_id,
                            zarr_path,
                            refined_run,
                            review_state,
                            zarr_use
                        FROM ranked
                        WHERE _rn = 1;
                        """,
                        params,
                    ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to query registry keypoint quality rows: {exc}") from exc

    selected_dataset_ids: set[str] = set()
    for row in rows:
        dataset_id = str(row["dataset_id"]) if row["dataset_id"] is not None else ""
        raw_zarr_path = row["zarr_path"]
        raw_refined_run = row["refined_run"]
        review_state = _normalize_review_state(row["review_state"])
        if not _matches_review_state_filter(review_state, review_state_filter):
            continue

        if raw_zarr_path is None:
            plans.append(
                ReviewPlan(
                    zarr_path=Path("<unknown>"),
                    refined_run=str(raw_refined_run) if raw_refined_run is not None else None,
                    review_state=review_state,
                    status="error",
                    reason="missing zarr_path in registry row",
                )
            )
            continue
        if raw_refined_run is None:
            plans.append(
                ReviewPlan(
                    zarr_path=Path(str(raw_zarr_path)).expanduser(),
                    refined_run=None,
                    review_state=review_state,
                    status="error",
                    reason="missing refined_run in registry row",
                )
            )
            continue

        zarr_path = Path(str(raw_zarr_path)).expanduser()
        if pose_schema is not None:
            matches_schema = _zarr_run_matches_pose_schema(zarr_path, str(raw_refined_run), pose_schema)
            if matches_schema is False:
                continue
            if matches_schema is None:
                plans.append(
                    ReviewPlan(
                        zarr_path=zarr_path,
                        refined_run=str(raw_refined_run),
                        review_state=review_state,
                        status="error",
                        reason="unable to verify pose_schema on refined run",
                    )
                )
                continue
        if not _is_in_scope(zarr_path, scope_roots):
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=str(raw_refined_run),
                    review_state=review_state,
                    status="filtered",
                    reason="outside requested scope",
                )
            )
            continue

        observed_use = _normalize_optional_text(row["zarr_use"])
        if zarr_use != "any" and observed_use != zarr_use:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=str(raw_refined_run),
                    review_state=review_state,
                    status="filtered",
                    reason=f"zarr_use={observed_use or 'unknown'}",
                )
            )
            continue
        if dataset_id and dataset_id in selected_dataset_ids:
            continue

        plans.append(
            ReviewPlan(
                zarr_path=zarr_path,
                refined_run=str(raw_refined_run),
                review_state=review_state,
                status="ok",
            )
        )
        if dataset_id:
            selected_dataset_ids.add(dataset_id)
        if require_failures:
            has_failures = _run_has_review_failures(zarr_path, str(raw_refined_run))
            if has_failures is False:
                plans.pop()
                if dataset_id:
                    selected_dataset_ids.discard(dataset_id)
    return sorted(plans, key=lambda p: str(p.zarr_path))


def _build_plans(
    roots: List[Path],
    recursive: bool,
    refined_run: Optional[str],
    zarr_use: str,
    pose_schema: Optional[str] = None,
    review_state_filter: str = "any",
) -> List[ReviewPlan]:
    plans: List[ReviewPlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    status="error",
                    reason=str(exc),
                )
            )
            continue

        observed_use = _infer_zarr_use(root, zarr_path)
        if zarr_use != "any" and observed_use != zarr_use:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    review_state=None,
                    status="filtered",
                    reason=f"zarr_use={observed_use or 'unknown'}",
                )
            )
            continue

        refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
        if refined_parent is None:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    review_state=None,
                    status="missing",
                    reason="no refined_keypoints_runs",
                )
            )
            continue

        run_name = refined_run or _latest_matching_run(refined_parent, pose_schema)
        if not run_name:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    review_state=None,
                    status="missing",
                    reason=(
                        f"no refined keypoints runs matching pose_schema={pose_schema}"
                        if pose_schema is not None
                        else "no refined keypoints runs"
                    ),
                )
            )
            continue
        if run_name not in refined_parent:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=run_name,
                    review_state=None,
                    status="missing",
                    reason="refined keypoints run not found",
                )
            )
            continue

        run_group = refined_parent[run_name]
        if refined_run and not _run_matches_pose_schema(run_group, pose_schema):
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=run_name,
                    review_state=None,
                    status="filtered",
                    reason=f"pose_schema={_extract_pose_schema_name(run_group.attrs.get('pose_schema')) or 'unknown'}",
                )
            )
            continue
        review_state = _normalize_review_state(run_group.attrs.get("keypoint_review_status"))
        if not _matches_review_state_filter(review_state, review_state_filter):
            continue
        plans.append(
            ReviewPlan(
                zarr_path=zarr_path,
                refined_run=run_name,
                review_state=review_state,
                status="ok",
            )
        )
    return sorted(plans, key=lambda p: str(p.zarr_path))


def _prompt_continue() -> bool:
    resp = input("Press Enter for next, or type 'q' to quit: ").strip().lower()
    return resp != "q"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch keypoint review (manual/retune/audit) for refined keypoint runs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or Zarr paths to scan.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "Optional registry sqlite path. When provided, candidates are selected from "
            "registry views/tables (no zarr filesystem scan)."
        ),
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help="Fail if registry-based selection cannot be used.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for Zarrs under each root.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--refined-run",
        type=str,
        help="Specific refined keypoint run name to use (default: latest per Zarr).",
    )
    parser.add_argument(
        "--pose-schema",
        type=str,
        help="Select the latest refined run matching this pose schema name (for example: traditional_v2).",
    )
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument(
        "--review-state-filter",
        default="any",
        choices=["any", "approved", "pending", "rejected", "needs_review", "missing", "not_approved"],
        help="Filter candidate runs by review state before launching review.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--retune", action="store_true", help="Run failure retune UI.")
    mode.add_argument("--manual", action="store_true", help="Run manual correction UI.")
    mode.add_argument("--audit", action="store_true", help="Recompute postprocess summary only.")
    parser.add_argument(
        "--crop-run",
        type=str,
        help="Crop run to source ROI images from (manual mode only).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review all keypoints (manual mode only; default is failures only).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        help=(
            "Frame indices to review (manual mode only). Accepts comma/space-separated "
            "indices or a path to a JSON/text list. JSON mappings of zarr->frames are supported."
        ),
    )
    parser.add_argument(
        "--apply-batch-size",
        type=int,
        default=128,
        help="Batch size for retune apply.",
    )
    parser.add_argument(
        "--apply-workers",
        type=int,
        default=4,
        help="Worker threads for retune apply.",
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to set when approving in manual review.",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default=None,
        choices=["training", "full_recording"],
        help="Intended use label (default: infer from existing status or zarr use).",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER).")
    parser.add_argument("--review-notes", help="Optional review notes.")
    parser.add_argument(
        "--frame-flag-file",
        default="keypoint_frame_flags.json",
        help="JSON file to append flagged frames (manual/retune modes).",
    )
    parser.add_argument(
        "--detect-flag-file",
        default="retune_flags.txt",
        help="Text file to append recordings flagged for detection retune.",
    )
    parser.add_argument(
        "--detect-frame-flag-file",
        default="retune_frame_flags.json",
        help="JSON file to append detection frames flagged for retune.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index within the review list (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of recordings to review.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List candidates and exit without opening the review tool.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Advance without prompting between recordings.",
    )
    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    roots: List[Path] = []
    if args.paths:
        roots.extend(args.paths)
    if file_list_paths:
        roots.extend(file_list_paths)
    if not roots:
        env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
        roots = [Path(env_root)] if env_root else [Path("/nvme1/recordings")]

    if args.registry_only and not args.registry:
        print("--registry-only requires --registry.")
        return 2

    plans: List[ReviewPlan]
    review_state_filter = str(args.review_state_filter)
    if args.registry:
        require_failures = bool(args.retune or (args.manual and not args.all))
        try:
            plans = _build_plans_from_registry(
                Path(args.registry),
                roots=roots,
                refined_run=args.refined_run,
                pose_schema=args.pose_schema,
                zarr_use=str(args.zarr_use),
                require_failures=require_failures,
                review_state_filter=review_state_filter,
            )
        except RuntimeError as exc:
            if args.registry_only:
                print(str(exc))
                return 1
            print(f"{exc}; falling back to filesystem scan.")
            plans = _build_plans(
                roots,
                recursive=bool(args.recursive),
                refined_run=args.refined_run,
                pose_schema=args.pose_schema,
                zarr_use=str(args.zarr_use),
                review_state_filter=review_state_filter,
            )
    else:
        plans = _build_plans(
            roots,
            recursive=bool(args.recursive),
            refined_run=args.refined_run,
            pose_schema=args.pose_schema,
            zarr_use=str(args.zarr_use),
            review_state_filter=review_state_filter,
        )

    if not plans:
        print("No recordings found to review.")
        return 0

    start = max(args.start, 0)
    if start >= len(plans):
        print(f"--start {start} >= total {len(plans)}")
        return 1

    end = len(plans)
    if args.limit is not None:
        end = min(end, start + max(args.limit, 0))

    print(f"Review list: {len(plans)} total; showing {start}–{end - 1}")
    for idx, plan in enumerate(plans):
        marker = "→" if start <= idx < end else " "
        status = plan.status
        reason = f" ({plan.reason})" if plan.reason else ""
        run = plan.refined_run or "none"
        print(f"{marker} [{idx:03d}] {plan.zarr_path} | refined={run} | {status}{reason}")

    if args.list:
        return 0

    for idx in range(start, end):
        plan = plans[idx]
        if plan.status != "ok":
            print(f"\n[{idx + 1}/{end}] Skipping {plan.zarr_path} ({plan.status})")
            continue
        print(f"\n[{idx + 1}/{end}] Opening keypoint review for {plan.zarr_path}")
        cmd = [
            sys.executable,
            "-m",
            "fisheye.tune.keypoint_review",
            str(plan.zarr_path),
        ]
        if args.retune:
            cmd.append("--retune")
        elif args.manual:
            cmd.append("--manual")
        else:
            cmd.append("--audit")

        if plan.refined_run:
            cmd.extend(["--refined-run", plan.refined_run])
        if args.crop_run and args.manual:
            cmd.extend(["--crop-run", args.crop_run])
        if args.retune:
            cmd.extend(["--apply-batch-size", str(args.apply_batch_size)])
            cmd.extend(["--apply-workers", str(args.apply_workers)])

        if args.manual:
            if args.all:
                cmd.append("--all")
            cmd.extend(["--review-state", args.review_state])
            cmd.extend(["--review-method", args.review_method])
            if args.review_intended_use is not None:
                cmd.extend(["--review-intended-use", args.review_intended_use])
            if args.reviewer:
                cmd.extend(["--reviewer", args.reviewer])
            if args.review_notes:
                cmd.extend(["--review-notes", args.review_notes])
        if args.frames and (args.manual or args.retune):
            cmd.extend(["--frames", args.frames])
        if args.frame_flag_file and (args.manual or args.retune):
            cmd.extend(["--frame-flag-file", args.frame_flag_file])
        if args.detect_flag_file and (args.manual or args.retune):
            cmd.extend(["--detect-flag-file", args.detect_flag_file])
        if args.detect_frame_flag_file and (args.manual or args.retune):
            cmd.extend(["--detect-frame-flag-file", args.detect_frame_flag_file])

        env = None
        if args.registry:
            env = os.environ.copy()
            env["PALETTE_REGISTRY_PATH"] = str(Path(args.registry).expanduser().resolve())
        subprocess.run(cmd, check=False, env=env)
        if idx < end - 1 and not args.no_prompt:
            if not _prompt_continue():
                break

    print("Review complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
