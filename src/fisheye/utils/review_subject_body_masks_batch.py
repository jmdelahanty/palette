#!/usr/bin/env python3
"""
Batch wrapper for subject-body review across many recordings.

Opens refined_subject_mask_review per selected Zarr/source-refined pair.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import zarr


REVIEW_STATES = {"approved", "pending", "rejected", "needs_review"}
COMPONENT_NAME = "subject_body"
REFINED_STAGE = "refined_subject_masks_runs"
SOURCE_STAGE = "subject_mask_runs"
SAM_BODY_RUN_SEMANTICS = "sam_body_mask_inference"


@dataclass
class ReviewPlan:
    zarr_path: Path
    subject_run: Optional[str]
    refined_run: Optional[str]
    review_state: Optional[str]
    status: str
    reason: Optional[str] = None
    stage_label: str = REFINED_STAGE
    roi_indices: Optional[list[int]] = None

    @property
    def opens_new_refined_run(self) -> bool:
        return self.refined_run is None


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
    return sorted(str(name) for name in names)[-1]


def _normalize_review_state(raw: object) -> Optional[str]:
    state = raw.get("state") if isinstance(raw, Mapping) else raw
    return _normalize_optional_text(state)


def _optional_text(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return text


def _normalize_optional_text(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    return text


def _is_group_like(value: object) -> bool:
    return hasattr(value, "attrs") and hasattr(value, "get")


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


def _normalize_stale_roi_indices(raw: object, *, total_rois: Optional[int] = None) -> list[int]:
    if raw is None:
        values: list[object] = []
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]

    seen: set[int] = set()
    normalized: list[int] = []
    for value in values:
        try:
            roi_idx = int(value)
        except (TypeError, ValueError):
            continue
        if roi_idx < 0:
            continue
        if total_rois is not None and roi_idx >= int(total_rois):
            continue
        if roi_idx in seen:
            continue
        seen.add(roi_idx)
        normalized.append(roi_idx)
    return normalized


def _pending_stale_roi_indices(run_group: object, component_name: str) -> list[int]:
    if not _is_group_like(run_group):
        return []
    masks_arr = run_group.get("masks_roi")
    total_rois = int(masks_arr.shape[0]) if masks_arr is not None and hasattr(masks_arr, "shape") else None
    components_parent = run_group.get("components")
    if not _is_group_like(components_parent):
        return []
    component_group = components_parent.get(str(component_name))
    if not _is_group_like(component_group):
        return []
    raw = component_group.attrs.get("source_update_pending_rows")
    return _normalize_stale_roi_indices(raw, total_rois=total_rois)


def _status_matches_filter(
    review_state: Optional[str],
    status_filter: str,
    *,
    stale_roi_indices: Optional[list[int]] = None,
) -> bool:
    if status_filter == "any":
        return True
    if status_filter == "missing":
        return review_state is None
    if status_filter == "stale":
        return bool(stale_roi_indices)
    return review_state == status_filter


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


def _component_labels(run_group: zarr.Group) -> tuple[list[str], list[bool]]:
    labels_raw = run_group.attrs.get("mask_labels")
    labels = [str(item) for item in labels_raw] if isinstance(labels_raw, (list, tuple)) else []

    available_flags: list[bool] = []
    available_channels = run_group.get("available_channels")
    if available_channels is not None:
        try:
            available_flags = np.asarray(available_channels[:], dtype=bool).astype(bool).tolist()
        except Exception:
            available_flags = []
    if labels:
        if not available_flags:
            available_flags = [True] * len(labels)
        if len(available_flags) < len(labels):
            available_flags.extend([False] * (len(labels) - len(available_flags)))
        available_flags = available_flags[: len(labels)]
    return labels, available_flags


def _run_has_component(run_group: zarr.Group, component_name: str) -> bool:
    labels, flags = _component_labels(run_group)
    if not labels:
        return False
    for label, flag in zip(labels, flags):
        if str(label) == component_name:
            return bool(flag)
    return False


def _run_semantics_matches(run_group: zarr.Group, run_semantics: Optional[str]) -> bool:
    normalized = _normalize_optional_text(run_semantics)
    if normalized is None:
        return True
    observed = _normalize_optional_text(run_group.attrs.get("run_semantics"))
    return observed == normalized


def _latest_run_with_component(
    parent: zarr.Group,
    component_name: str,
    *,
    run_semantics: Optional[str] = None,
) -> Optional[str]:
    latest = _latest_run(parent)
    if (
        latest
        and latest in parent
        and _run_has_component(parent[latest], component_name)
        and _run_semantics_matches(parent[latest], run_semantics)
    ):
        return str(latest)
    names = sorted((str(name) for name in parent.keys()), reverse=True)
    for name in names:
        if (
            name in parent
            and _run_has_component(parent[name], component_name)
            and _run_semantics_matches(parent[name], run_semantics)
        ):
            return name
    return None


def _resolve_refined_review_state(run_group: zarr.Group, component_name: str) -> Optional[str]:
    payload = dict(run_group.attrs.get("component_review_statuses") or {}).get(component_name)
    return _normalize_review_state(payload)


def _resolve_subject_run_name(
    root: zarr.Group,
    *,
    explicit_subject_run: Optional[str],
    subject_run_semantics: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    parent = root.get(SOURCE_STAGE)
    if not isinstance(parent, zarr.Group):
        return None, f"no {SOURCE_STAGE}"
    run_name = explicit_subject_run or _latest_run_with_component(
        parent,
        COMPONENT_NAME,
        run_semantics=subject_run_semantics,
    )
    if not run_name:
        reason = f"no {SOURCE_STAGE} with {COMPONENT_NAME}"
        normalized_semantics = _normalize_optional_text(subject_run_semantics)
        if normalized_semantics:
            reason = (
                f"no {SOURCE_STAGE} with {COMPONENT_NAME} "
                f"and run_semantics={normalized_semantics}"
            )
        if explicit_subject_run:
            reason = f"{SOURCE_STAGE}/{explicit_subject_run} not found or lacks {COMPONENT_NAME}"
        return None, reason
    if run_name not in parent:
        return None, f"{SOURCE_STAGE}/{run_name} not found"
    run_group = parent[run_name]
    if not _run_has_component(run_group, COMPONENT_NAME):
        return None, f"{SOURCE_STAGE}/{run_name} lacks {COMPONENT_NAME}"
    if not _run_semantics_matches(run_group, subject_run_semantics):
        normalized_semantics = _normalize_optional_text(subject_run_semantics)
        return None, f"{SOURCE_STAGE}/{run_name} run_semantics != {normalized_semantics}"
    return str(run_name), None


def _resolve_refined_run_name(
    root: zarr.Group,
    *,
    explicit_refined_run: Optional[str],
    subject_run: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    parent = root.get(REFINED_STAGE)
    if not isinstance(parent, zarr.Group):
        return None, None
    if explicit_refined_run:
        if explicit_refined_run not in parent:
            return None, f"{REFINED_STAGE}/{explicit_refined_run} not found"
        run_group = parent[explicit_refined_run]
        if not _run_has_component(run_group, COMPONENT_NAME):
            return None, f"{REFINED_STAGE}/{explicit_refined_run} lacks {COMPONENT_NAME}"
        return str(explicit_refined_run), None

    preferred = _latest_run_with_component(parent, COMPONENT_NAME)
    if not preferred:
        return None, None
    if subject_run:
        names = sorted((str(name) for name in parent.keys()), reverse=True)
        for name in names:
            run_group = parent[name]
            if not _run_has_component(run_group, COMPONENT_NAME):
                continue
            source_subject_run = str(run_group.attrs.get("source_subject_mask_run") or "").strip() or None
            if source_subject_run == subject_run:
                return name, None
        return None, None
    return preferred, None


def _build_plans_from_registry(
    registry_path: Path,
    roots: List[Path],
    subject_run: Optional[str],
    refined_run: Optional[str],
    status_filter: str,
    zarr_use: str,
) -> List[ReviewPlan]:
    if status_filter == "stale":
        raise RuntimeError(
            "Registry-backed stale subject-body review is not supported because row-level "
            "source_update_pending_rows metadata is not present in the registry view."
        )
    scope_roots = _normalize_scope_roots(roots)
    plans: List[ReviewPlan] = []
    try:
        with sqlite3.connect(str(registry_path)) as conn:
            conn.row_factory = sqlite3.Row
            if refined_run:
                rows = conn.execute(
                    """
                    SELECT
                        d.zarr_path AS zarr_path,
                        smcq.stage_group AS stage_group,
                        smcq.run_name AS run_name,
                        smcq.review_state AS review_state,
                        smcq.zarr_use AS zarr_use,
                        smcq.available AS available,
                        smcq.source_subject_mask_run AS source_subject_mask_run
                    FROM subject_mask_component_quality smcq
                    JOIN datasets d ON d.dataset_id = smcq.dataset_id
                    WHERE smcq.stage_group = ?
                      AND smcq.component_name = ?
                      AND smcq.run_name = ?
                      AND (d.status IS NULL OR lower(d.status) != 'missing');
                    """,
                    (REFINED_STAGE, COMPONENT_NAME, refined_run),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        zarr_path,
                        stage_group,
                        run_name,
                        review_state,
                        zarr_use,
                        available,
                        source_subject_mask_run
                    FROM subject_mask_component_quality_overview
                    WHERE component_name = ?
                      AND stage_group IN (?, ?);
                    """,
                    (COMPONENT_NAME, REFINED_STAGE, SOURCE_STAGE),
                ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to query registry subject-body rows: {exc}") from exc

    if refined_run:
        for row in rows:
            raw_zarr_path = row["zarr_path"]
            if raw_zarr_path is None:
                plans.append(
                    ReviewPlan(
                        zarr_path=Path("<unknown>"),
                        subject_run=subject_run,
                        refined_run=refined_run,
                        review_state=_normalize_review_state(row["review_state"]),
                        status="error",
                        reason="missing zarr_path in registry row",
                    )
                )
                continue
            zarr_path = Path(str(raw_zarr_path)).expanduser()
            if not _is_in_scope(zarr_path, scope_roots):
                plans.append(
                    ReviewPlan(
                        zarr_path=zarr_path,
                        subject_run=subject_run,
                        refined_run=refined_run,
                        review_state=_normalize_review_state(row["review_state"]),
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
                        subject_run=subject_run,
                        refined_run=refined_run,
                        review_state=_normalize_review_state(row["review_state"]),
                        status="filtered",
                        reason=f"zarr_use={observed_use or 'unknown'}",
                    )
                )
                continue

            if not bool(row["available"]):
                plans.append(
                    ReviewPlan(
                        zarr_path=zarr_path,
                        subject_run=subject_run,
                        refined_run=refined_run,
                        review_state=_normalize_review_state(row["review_state"]),
                        status="missing",
                        reason=f"{REFINED_STAGE}/{refined_run} lacks {COMPONENT_NAME}",
                    )
                )
                continue

            resolved_subject_run = subject_run or _optional_text(row["source_subject_mask_run"])
            review_state = _normalize_review_state(row["review_state"])
            if not _status_matches_filter(review_state, status_filter):
                plans.append(
                    ReviewPlan(
                        zarr_path=zarr_path,
                        subject_run=resolved_subject_run,
                        refined_run=refined_run,
                        review_state=review_state,
                        status="filtered",
                        reason=f"review_state={review_state or 'missing'}",
                    )
                )
                continue

            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=resolved_subject_run,
                    refined_run=refined_run,
                    review_state=review_state,
                    status="ok",
                    stage_label=REFINED_STAGE,
                )
            )
        return sorted(plans, key=lambda p: str(p.zarr_path))

    grouped: dict[str, dict[str, sqlite3.Row]] = {}
    for row in rows:
        raw_zarr_path = row["zarr_path"]
        if raw_zarr_path is None:
            continue
        zarr_path = Path(str(raw_zarr_path)).expanduser()
        key = str(zarr_path.resolve(strict=False))
        stage_group = str(row["stage_group"] or "")
        existing = grouped.setdefault(key, {})
        if stage_group not in existing:
            existing[stage_group] = row

    for key in sorted(grouped):
        row_map = grouped[key]
        zarr_path = Path(str((row_map.get(REFINED_STAGE) or row_map.get(SOURCE_STAGE))["zarr_path"])).expanduser()
        if not _is_in_scope(zarr_path, scope_roots):
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=subject_run,
                    refined_run=None,
                    review_state=None,
                    status="filtered",
                    reason="outside requested scope",
                )
            )
            continue

        observed_use = _normalize_optional_text(
            (row_map.get(REFINED_STAGE) or row_map.get(SOURCE_STAGE))["zarr_use"]
        )
        if zarr_use != "any" and observed_use != zarr_use:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=subject_run,
                    refined_run=None,
                    review_state=None,
                    status="filtered",
                    reason=f"zarr_use={observed_use or 'unknown'}",
                )
            )
            continue

        refined_row = row_map.get(REFINED_STAGE)
        source_row = row_map.get(SOURCE_STAGE)
        if subject_run:
            if refined_row is not None:
                refined_source = _optional_text(refined_row["source_subject_mask_run"])
                if refined_source != subject_run:
                    refined_row = None
            if source_row is not None and _optional_text(source_row["run_name"]) != subject_run:
                source_row = None

        chosen_row = refined_row or source_row
        if chosen_row is None:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=subject_run,
                    refined_run=None,
                    review_state=None,
                    status="missing",
                    reason=f"no {COMPONENT_NAME} review source matching requested runs",
                )
            )
            continue

        if not bool(chosen_row["available"]):
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=subject_run
                    or _optional_text(chosen_row["source_subject_mask_run"])
                    or _optional_text(chosen_row["run_name"]),
                    refined_run=None,
                    review_state=None,
                    status="missing",
                    reason=f"{chosen_row['stage_group']}/{chosen_row['run_name']} lacks {COMPONENT_NAME}",
                )
            )
            continue

        chosen_stage = str(chosen_row["stage_group"] or "")
        chosen_subject_run = (
            subject_run
            or _optional_text(chosen_row["source_subject_mask_run"])
            or (_optional_text(chosen_row["run_name"]) if chosen_stage == SOURCE_STAGE else None)
        )
        chosen_refined_run = _optional_text(chosen_row["run_name"]) if chosen_stage == REFINED_STAGE else None
        review_state = _normalize_review_state(chosen_row["review_state"]) if chosen_stage == REFINED_STAGE else None

        if not _status_matches_filter(review_state, status_filter):
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=chosen_subject_run,
                    refined_run=chosen_refined_run,
                    review_state=review_state,
                    status="filtered",
                    reason=f"review_state={review_state or 'missing'}",
                    stage_label=chosen_stage if chosen_stage == REFINED_STAGE else f"{SOURCE_STAGE} -> new refined run",
                )
            )
            continue

        plans.append(
            ReviewPlan(
                zarr_path=zarr_path,
                subject_run=chosen_subject_run,
                refined_run=chosen_refined_run,
                review_state=review_state,
                status="ok",
                stage_label=chosen_stage if chosen_stage == REFINED_STAGE else f"{SOURCE_STAGE} -> new refined run",
            )
        )

    return plans


def _build_plans(
    roots: List[Path],
    recursive: bool,
    subject_run: Optional[str],
    subject_run_semantics: Optional[str],
    refined_run: Optional[str],
    status_filter: str,
    zarr_use: str,
) -> List[ReviewPlan]:
    plans: List[ReviewPlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        except Exception as exc:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=subject_run,
                    refined_run=refined_run,
                    review_state=None,
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
                    subject_run=subject_run,
                    refined_run=refined_run,
                    review_state=None,
                    status="filtered",
                    reason=f"zarr_use={observed_use or 'unknown'}",
                )
            )
            continue

        resolved_subject_run, subject_reason = _resolve_subject_run_name(
            root,
            explicit_subject_run=subject_run,
            subject_run_semantics=subject_run_semantics,
        )
        if resolved_subject_run is None:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=subject_run,
                    refined_run=refined_run,
                    review_state=None,
                    status="missing",
                    reason=subject_reason,
                )
            )
            continue

        resolved_refined_run, refined_reason = _resolve_refined_run_name(
            root,
            explicit_refined_run=refined_run,
            subject_run=resolved_subject_run,
        )
        if refined_run is not None and resolved_refined_run is None:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=resolved_subject_run,
                    refined_run=refined_run,
                    review_state=None,
                    status="missing",
                    reason=refined_reason,
                )
            )
            continue

        if resolved_refined_run is None:
            review_state = None
            stage_label = f"{SOURCE_STAGE} -> new refined run"
            stale_roi_indices: list[int] = []
        else:
            refined_group = root[REFINED_STAGE][resolved_refined_run]
            review_state = _resolve_refined_review_state(refined_group, COMPONENT_NAME)
            stage_label = REFINED_STAGE
            stale_roi_indices = _pending_stale_roi_indices(refined_group, COMPONENT_NAME)

        if not _status_matches_filter(
            review_state,
            status_filter,
            stale_roi_indices=stale_roi_indices,
        ):
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    subject_run=resolved_subject_run,
                    refined_run=resolved_refined_run,
                    review_state=review_state,
                    status="filtered",
                    reason=(
                        f"stale_rows={len(stale_roi_indices)}"
                        if status_filter == "stale"
                        else f"review_state={review_state or 'missing'}"
                    ),
                    stage_label=stage_label,
                )
            )
            continue

        plans.append(
            ReviewPlan(
                zarr_path=zarr_path,
                subject_run=resolved_subject_run,
                refined_run=resolved_refined_run,
                review_state=review_state,
                status="ok",
                stage_label=stage_label,
                roi_indices=stale_roi_indices or None,
            )
        )
    return sorted(plans, key=lambda p: str(p.zarr_path))


def _prompt_continue() -> bool:
    resp = input("Press Enter for next, or type 'q' to quit: ").strip().lower()
    return resp != "q"


def _plan_counts(plans: List[ReviewPlan]) -> dict[str, int]:
    counts = {"ok": 0, "missing": 0, "error": 0, "filtered": 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
    return counts


def _viewer_cmd(args: argparse.Namespace, plan: ReviewPlan) -> List[str]:
    display_scale = max(1, int(args.scale_percent)) / 100.0
    cmd: List[str] = [
        sys.executable,
        "-m",
        "fisheye.tune.refined_subject_mask_review",
        str(plan.zarr_path),
        "--components",
        COMPONENT_NAME,
        "--component",
        COMPONENT_NAME,
        "--edit-padding",
        str(args.padding),
        "--display-scale",
        f"{display_scale:g}",
        "--edit-zoom",
        str(args.edit_zoom),
        "--approve-state",
        str(args.review_state),
        "--review-method",
        str(args.review_method),
        "--review-intended-use",
        str(args.review_intended_use),
    ]
    if plan.subject_run:
        cmd.extend(["--subject-run", str(plan.subject_run)])
    if plan.refined_run:
        cmd.extend(["--refined-run", str(plan.refined_run)])
    if plan.roi_indices:
        cmd.extend(["--roi-index", str(plan.roi_indices[0])])
    if args.crop_run:
        cmd.extend(["--crop-run", str(args.crop_run)])
    if args.reviewer:
        cmd.extend(["--reviewer", str(args.reviewer)])
    if args.review_notes:
        cmd.extend(["--review-notes", str(args.review_notes)])
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch review subject-body masks. Opens the refined subject-mask reviewer "
            "on an existing refined_subject_masks run when present, or starts a new "
            "refined run from subject_mask_runs when review is missing."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths to scan.")
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "Optional registry sqlite path. When provided, candidates are selected from "
            "registry views (no zarr filesystem scan)."
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarrs.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument(
        "--status",
        "--review-state-filter",
        dest="status",
        choices=["missing", "approved", "pending", "rejected", "needs_review", "stale", "any"],
        default="missing",
        help="Select subject-body review candidates by component review state (default: missing).",
    )
    parser.add_argument(
        "--subject-run",
        type=str,
        help="Specific subject_mask_runs/<run> source to review (default: latest with subject_body).",
    )
    parser.add_argument(
        "--subject-run-semantics",
        type=str,
        help=(
            "Filter source subject_mask_runs by run_semantics before choosing the latest body run "
            "(for example: sam_body_mask_inference)."
        ),
    )
    parser.add_argument(
        "--sam",
        action="store_true",
        help="Shortcut for --subject-run-semantics sam_body_mask_inference.",
    )
    parser.add_argument(
        "--refined-run",
        type=str,
        help="Specific refined_subject_masks_runs/<run> to reopen (default: latest with subject_body).",
    )
    parser.add_argument("--crop-run", type=str, help="Crop run override for viewer.")
    parser.add_argument("--padding", type=int, default=18, help="Edit patch padding in pixels.")
    parser.add_argument("--scale-percent", type=int, default=220, help="Main-window scale percent.")
    parser.add_argument("--edit-zoom", type=int, default=8, help="Edit-patch zoom.")
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=sorted(REVIEW_STATES),
        help="Review state written by reviewer when pressing 'a'.",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method written by reviewer when pressing 'a'.",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Review intended_use written by reviewer when pressing 'a'.",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER in reviewer).")
    parser.add_argument("--review-notes", help="Optional review notes for reviewer.")
    parser.add_argument("--start", type=int, default=0, help="Start index within selected plans.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected plans to run.")
    parser.add_argument("--list", action="store_true", help="List candidates and exit.")
    parser.add_argument("--no-prompt", action="store_true", help="Do not prompt between recordings.")
    args = parser.parse_args(argv)

    if args.sam and args.subject_run_semantics:
        parser.error("--sam cannot be combined with --subject-run-semantics.")
    subject_run_semantics = str(args.subject_run_semantics) if args.subject_run_semantics else None
    if args.sam:
        subject_run_semantics = SAM_BODY_RUN_SEMANTICS

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    explicit_paths: List[Path] = []
    if args.paths:
        explicit_paths.extend(args.paths)
    if file_list_paths:
        explicit_paths.extend(file_list_paths)
    if not explicit_paths:
        explicit_paths = [Path("/nvme1/recordings")]

    if args.registry and str(args.status) == "stale":
        print(
            "Ignoring --registry for --status stale: row-level stale subject-body metadata "
            "is only available from the refined zarr attrs."
        )
        plans = _build_plans(
            explicit_paths,
            recursive=bool(args.recursive),
            subject_run=args.subject_run,
            subject_run_semantics=subject_run_semantics,
            refined_run=args.refined_run,
            status_filter=str(args.status),
            zarr_use=str(args.zarr_use),
        )
    elif args.registry and subject_run_semantics:
        print(
            "Ignoring --registry for --subject-run-semantics: source run_semantics filtering "
            "is resolved from subject_mask_runs attrs."
        )
        plans = _build_plans(
            explicit_paths,
            recursive=bool(args.recursive),
            subject_run=args.subject_run,
            subject_run_semantics=subject_run_semantics,
            refined_run=args.refined_run,
            status_filter=str(args.status),
            zarr_use=str(args.zarr_use),
        )
    elif args.registry:
        plans = _build_plans_from_registry(
            Path(args.registry),
            roots=explicit_paths,
            subject_run=args.subject_run,
            refined_run=args.refined_run,
            status_filter=str(args.status),
            zarr_use=str(args.zarr_use),
        )
    else:
        plans = _build_plans(
            explicit_paths,
            recursive=bool(args.recursive),
            subject_run=args.subject_run,
            subject_run_semantics=subject_run_semantics,
            refined_run=args.refined_run,
            status_filter=str(args.status),
            zarr_use=str(args.zarr_use),
        )
    counts = _plan_counts(plans)
    selected = [plan for plan in plans if plan.status == "ok"]

    start = max(0, int(args.start))
    if start:
        selected = selected[start:]
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]

    print(
        "Subject-body batch review plans: "
        f"ok={counts['ok']} missing={counts['missing']} filtered={counts['filtered']} errors={counts['error']}"
    )

    if args.list:
        for plan in selected:
            print(
                f"{plan.zarr_path}\n"
                f"  subject_run={plan.subject_run or 'latest'}\n"
                f"  refined_run={plan.refined_run or '(new refined run)'}\n"
                f"  stage={plan.stage_label}\n"
                f"  review_state={plan.review_state or 'missing'}\n"
                f"  roi_indices={','.join(str(idx) for idx in (plan.roi_indices or [])) or '(all)'}"
            )
        if not selected:
            print("No matching review candidates.")
        return 0

    if not selected:
        print("No matching review candidates.")
        return 0

    failures = 0
    for idx, plan in enumerate(selected, start=1):
        print(f"\n[{idx}/{len(selected)}] Reviewing: {plan.zarr_path}")
        cmd = _viewer_cmd(args, plan)
        rc = subprocess.call(cmd)
        if rc != 0:
            failures += 1
            print(f"Viewer exited with status {rc}.")
        if idx < len(selected) and not args.no_prompt and not _prompt_continue():
            break
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
