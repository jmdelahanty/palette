"""Operator command: upgrade a refined subject-mask run from cheap to full metrics.

Browser (and admin-on-behalf) mask Apply runs a full mask-local QC refresh,
but refuses a run that declares ``component_metric_level: cheap`` rather than
silently change that declared level mid-Apply. Runs finalized with cheap
metrics (shape-QC metrics deferred) can therefore never finish an Apply.

This command performs the same full refresh deliberately, once, before any
Apply: ``finalize_subject_masks.refresh_refined_subject_mask_metrics_run`` with
the parameters Apply QC uses, under the refined-run write lock, and then the
same row validation Apply QC applies. Mask pixels and the edit revision are
not changed (both are checked). The browser QC policy stamp is left to the
next Apply, which binds it to that Apply's edit revision.

Dry run by default. With --execute it refuses while any labeling task on this
run has an open editor session, and copies the run group to a backup first.

    scripts/py -m fisheye.labeling.upgrade_mask_run_metrics \\
        --store ~/.palette/labeling_work.sqlite --zarr ARCHIVE.zarr --refined-run RUN [--execute]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

DEFAULT_BACKUP_DIR = Path("/groups/johnson/johnsonlab/jeremy/palette_backups/mask_run_metric_upgrades")
ALREADY_FULL_REFUSAL = "the run already declares full (or no) component metrics"


class MetricUpgradeRefused(RuntimeError):
    """The run is not in a state where its metrics may be upgraded."""


def _masks_sha256(run: zarr.Group) -> str:
    masks = run["masks_roi"]
    digest = hashlib.sha256()
    digest.update(json.dumps([list(masks.shape), str(masks.dtype)]).encode())
    step = max(1, int(masks.chunks[0]) if masks.chunks else 64)
    for start in range(0, int(masks.shape[0]), step):
        digest.update(np.ascontiguousarray(masks[start:start + step]).tobytes())
    return digest.hexdigest()


def _run_tasks(store_path: Path, zarr_path: Path, refined_run: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tasks = []
        for row in conn.execute("SELECT task_id, workflow_kind, component_name, state, scope_json FROM labeling_tasks;"):
            scope = json.loads(row["scope_json"] or "{}")
            path = str(scope.get("zarr_path") or "")
            if not path or Path(path).expanduser().resolve() != zarr_path:
                continue
            if row["workflow_kind"] == "subject_mask_component" and scope.get("refined_run") not in (None, refined_run):
                continue
            open_sessions = conn.execute(
                """
                SELECT COUNT(*) FROM labeling_sessions WHERE task_id = ? AND closed_at_utc IS NULL
                  AND expires_at_utc > strftime('%Y-%m-%dT%H:%M:%fZ', 'now');
                """,
                (row["task_id"],),
            ).fetchone()[0]
            pending_effects = [
                str(r[0]) for r in conn.execute(
                    """
                    SELECT apply_id FROM labeling_checkpoint_apply_receipts
                    WHERE task_id = ? AND state = 'applied' AND secondary_effects_state != 'complete';
                    """,
                    (row["task_id"],),
                )
            ]
            tasks.append({
                "task_id": row["task_id"], "workflow_kind": row["workflow_kind"],
                "component_name": row["component_name"], "state": row["state"],
                "open_sessions": int(open_sessions), "pending_effect_apply_ids": pending_effects,
            })
        return tasks
    finally:
        conn.close()


def upgrade_plan(store_path: Path, zarr_path: Path, refined_run: str) -> dict[str, object]:
    zarr_path = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root[f"refined_subject_masks_runs/{refined_run}"]
    labels = [str(v) for v in (run.attrs.get("mask_labels") or [])]
    components = run.get("components")
    component_levels = {
        name: (components[name].attrs.get("component_metric_level") if components is not None and name in components else None)
        for name in labels
    }
    run_level = run.attrs.get("component_metric_level")
    tasks = _run_tasks(Path(store_path).expanduser().resolve(), zarr_path, refined_run)
    refusals = []
    if run_level in (None, "full") and all(level in (None, "full") for level in component_levels.values()):
        refusals.append(ALREADY_FULL_REFUSAL)
    if "masks_roi" not in run:
        refusals.append("the run has no dense masks_roi")
    if not labels:
        refusals.append("the run declares no mask_labels")
    busy = [t["task_id"] for t in tasks if t["open_sessions"]]
    if busy:
        refusals.append(f"open editor sessions on task(s) {busy}; labelers may be working")
    return {
        "zarr_path": str(zarr_path),
        "refined_run": refined_run,
        "run_metric_level": run_level,
        "component_metric_levels": component_levels,
        "mask_labels": labels,
        "rows": int(run["masks_roi"].shape[0]) if "masks_roi" in run else 0,
        "edit_revision": run.attrs.get("edit_revision"),
        "metrics_stale": run.attrs.get("metrics_stale"),
        "tasks": tasks,
        "refusals": refusals,
        "ok": not refusals,
    }


def upgrade_to_full(
    store_path: Path,
    zarr_path: Path,
    refined_run: str,
    *,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
) -> dict[str, object]:
    from fisheye.labeling.web_subject_mask_apply_qc import (
        QC_ROW_CHUNK,
        _EYE_COMPONENTS,
        _validate_metric_and_contour_rows,
    )
    from fisheye.refinement import finalize_subject_masks as finalizer
    from fisheye.shared.detect_reason_codec import read_reason_labels
    from fisheye.shared.refined_subject_mask_mutation import resolve_mutable_refined_subject_mask_run
    from fisheye.tune import refined_subject_mask_review as review_mod

    zarr_path = Path(zarr_path).expanduser().resolve()
    plan = upgrade_plan(store_path, zarr_path, refined_run)
    if not plan["ok"]:
        raise MetricUpgradeRefused("; ".join(plan["refusals"]))
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined_run):
        root = review_mod.open_zarr_root(zarr_path, mode="a")
        run = resolve_mutable_refined_subject_mask_run(root, refined_run)
        revision_before = run.attrs.get("edit_revision")
        masks_before = _masks_sha256(run)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = Path(backup_dir).expanduser() / zarr_path.stem / f"{refined_run}_{stamp}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(zarr_path / "refined_subject_masks_runs" / refined_run, backup)
        names = tuple(plan["mask_labels"])
        original_reasons = {name: read_reason_labels(run["components"][name]) for name in names}
        # Fail closed, as Apply QC does: stale until the refresh is verified.
        run.attrs.update({"metrics_stale": True, "contours_stale": True})
        summary = finalizer.refresh_refined_subject_mask_metrics_run(
            root,
            refined_run=refined_run,
            components=None,
            metric_level="full",
            chunk_size=QC_ROW_CHUNK,
            refresh_reason_tags=True,
            write_eye_geometry=_EYE_COMPONENTS.issubset(names),
            write_component_contours=True,
        )
        run = resolve_mutable_refined_subject_mask_run(root, refined_run)
        if run.attrs.get("edit_revision") != revision_before:
            raise RuntimeError("Refined mask edit revision changed during the metric upgrade.")
        if _masks_sha256(run) != masks_before:
            raise RuntimeError("Mask pixels changed during the metric upgrade.")
        if tuple(summary.get("components") or ()) != names:
            raise RuntimeError("The metric upgrade did not refresh every component.")
        _validate_metric_and_contour_rows(run, names, original_reasons)
        run.attrs["metrics_stale"] = False
        run.attrs["contours_stale"] = False
        run.attrs["component_metric_level_upgrade"] = {
            "from": plan["run_metric_level"],
            "to": "full",
            "upgraded_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "fisheye.labeling.upgrade_mask_run_metrics",
            "backup_path": str(backup),
            "masks_roi_sha256": masks_before,
            "edit_revision": revision_before,
        }
    return {
        "plan": plan,
        "backup_path": str(backup),
        "masks_roi_sha256": masks_before,
        "edit_revision": revision_before,
        "run_metric_level_after": "full",
        "review_counts": summary.get("review_counts"),
        "duration_seconds": summary.get("duration_seconds"),
    }


def ensure_full_metrics_for_tasks(
    store_path: Path,
    tasks: Sequence[Mapping[str, object] | None],
    *,
    backup_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Give the runs behind newly created subject-mask tasks full metrics.

    Mask Apply requires full component metrics, so task creation upgrades
    each distinct target run that declares cheap ones (with this module's
    checks and backup). A run that cannot be upgraded now (for instance, a
    labeler has a sibling task on it open) is reported, not failed, so the
    caller can warn; the operator command upgrades it later.
    """

    runs: dict[tuple[str, str], list[str]] = {}
    for task in tasks:
        if not task or task.get("workflow_kind") != "subject_mask_component" or task.get("state") == "complete":
            continue
        scope = task.get("scope") if isinstance(task.get("scope"), Mapping) else {}
        zarr_path, refined_run = scope.get("zarr_path"), scope.get("refined_run") or task.get("run_name")
        if zarr_path and refined_run:
            runs.setdefault((str(zarr_path), str(refined_run)), []).append(str(task.get("task_id")))
    results = []
    for (zarr_path, refined_run), task_ids in runs.items():
        result: dict[str, object] = {"zarr_path": zarr_path, "refined_run": refined_run, "task_ids": task_ids}
        try:
            plan = upgrade_plan(store_path, Path(zarr_path), refined_run)
            if ALREADY_FULL_REFUSAL in plan["refusals"]:
                result["status"] = "already_full"
            elif not plan["ok"]:
                result.update(status="not_upgraded", refusals=plan["refusals"])
            else:
                done = upgrade_to_full(store_path, Path(zarr_path), refined_run, backup_dir=backup_dir or DEFAULT_BACKUP_DIR)
                result.update(status="upgraded", backup_path=done["backup_path"])
        except Exception as exc:  # noqa: BLE001 - reported to the caller as a warning
            result.update(status="not_upgraded", refusals=[f"{type(exc).__name__}: {exc}"])
        results.append(result)
    return results


def metric_upgrade_warnings(results: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Warnings for task-creation reports: runs left without full metrics."""

    return [
        {
            "code": "subject_mask_task_run_metrics_not_full",
            "zarr_path": r["zarr_path"],
            "refined_run": r["refined_run"],
            "task_ids": r["task_ids"],
            "refusals": r.get("refusals"),
            "details": (
                "Mask Apply on these tasks will refuse until the run has full metrics: run "
                "scripts/py -m fisheye.labeling.upgrade_mask_run_metrics --execute once no one has it open."
            ),
        }
        for r in results
        if r.get("status") == "not_upgraded"
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True, help="Labeling store whose tasks use this run.")
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        plan = upgrade_plan(args.store, args.zarr, args.refined_run)
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, default=str))
        return 0 if plan["ok"] else 2
    try:
        report = upgrade_to_full(args.store, args.zarr, args.refined_run, backup_dir=args.backup_dir)
    except MetricUpgradeRefused as exc:
        print(json.dumps({"ok": False, "refused": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
