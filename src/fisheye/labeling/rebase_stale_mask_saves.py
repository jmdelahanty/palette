"""Operator command: re-save mask checkpoints made stale by another component's Apply.

A refined subject-mask run has one ``edit_revision`` for all its components,
and mask Apply treats every checkpoint saved at an older revision as stale.
So applying one component (say eye_left) makes the saved, unapplied edits of
the run's other components unusable, even though their pixels were never
touched; the editor's only remedy is re-saving every row by hand.

This command re-saves such checkpoints for the labeler, but only with proof
that the component's pixels are unchanged since the save was made: a backup
copy of the run at the checkpoints' revision (``--base-backup``) must hold the
same bytes, row for row, as the run now. Each row is re-saved through the
mask editor's own /nav and /save routes in a private session as the task's
assignee, with exactly the saved mask, so the checkpoint is replaced in place
at the current revision and stays the labeler's. Before re-saving, a
``rebase_stale_mask_saves`` event records the operator, the reason
``rebased_unchanged_base`` and, per row, the original checkpoint id, saved
time and revision and the pixel digest that was checked.

Dry run by default. --execute refuses when the task is open, an Apply is
unfinished, the backup's revision differs from the checkpoints', row
identities differ, or any row's component pixels changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import zarr

from fisheye.labeling.admin_apply_on_behalf import _request, private_assignee_session
from fisheye.labeling.assignment_store import LabelingStore
from fisheye.labeling.store_backup import DEFAULT_BACKUP_DIR, backup_labeling_store

REBASE_EVENT = "rebase_stale_mask_saves"
REBASE_REASON = "rebased_unchanged_base"
REBASE_CLIENT_LABEL = "rebase_stale_mask_saves"
_ROW_IDENTITY_ARRAYS = ("source_crop_row_ids", "source_refined_row_ids", "source_detect_row_index", "instance_key", "frame_indices")


class RebaseRefused(RuntimeError):
    """The checkpoints cannot be proven safe to re-save."""


def _revision(group) -> int:
    value = group.attrs.get("edit_revision")
    return int(value) if isinstance(value, int) else 0


def _digest(array: np.ndarray) -> str:
    values = (np.asarray(array) > 0).astype(np.uint8)
    return hashlib.sha256(json.dumps(list(values.shape)).encode() + values.tobytes()).hexdigest()


def rebase_plan(store_path: Path, task_id: str, base_backup: Path) -> dict[str, object]:
    """Read-only check of which stale checkpoints may be re-saved, with refusals."""

    conn = sqlite3.connect(f"file:{Path(store_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        task = conn.execute(
            """
            SELECT t.*, a.assignee_user, a.status AS assignment_status FROM labeling_tasks t
            LEFT JOIN recording_assignments a ON a.recording_id = t.recording_id WHERE t.task_id = ?;
            """,
            (task_id,),
        ).fetchone()
        if task is None:
            raise RebaseRefused(f"Unknown task_id: {task_id}")
        scope = json.loads(task["scope_json"] or "{}")
        refusals: list[str] = []
        if task["workflow_kind"] != "subject_mask_component":
            raise RebaseRefused("Only subject-mask tasks are supported.")
        open_sessions = conn.execute(
            """
            SELECT COUNT(*) FROM labeling_sessions WHERE task_id = ? AND closed_at_utc IS NULL
              AND expires_at_utc > strftime('%Y-%m-%dT%H:%M:%fZ', 'now');
            """,
            (task_id,),
        ).fetchone()[0]
        if open_sessions:
            refusals.append(f"{open_sessions} open editor session(s) on this task")
        unfinished = conn.execute(
            """
            SELECT COUNT(*) FROM labeling_checkpoint_apply_receipts
            WHERE task_id = ? AND (state != 'applied' OR secondary_effects_state != 'complete');
            """,
            (task_id,),
        ).fetchone()[0]
        if unfinished:
            refusals.append("an Apply is unfinished on this task")
        checkpoints = [dict(r) for r in conn.execute(
            """
            SELECT checkpoint_id, roi_idx, component_name, target_edit_revision, updated_at_utc, payload_json
            FROM labeling_session_checkpoints WHERE task_id = ? AND state = 'active' ORDER BY roi_idx;
            """,
            (task_id,),
        )]
        saved_at = {
            json.loads(r["target_json"] or "{}").get("checkpoint_id"): r["created_at_utc"]
            for r in conn.execute(
                """
                SELECT created_at_utc, target_json FROM labeling_task_events
                WHERE task_id = ? AND event_type = 'checkpoint_subject_mask_roi' ORDER BY created_at_utc;
                """,
                (task_id,),
            )
        }
    finally:
        conn.close()

    run_path = f"refined_subject_masks_runs/{scope.get('refined_run')}"
    current = zarr.open_group(str(scope.get("zarr_path")), mode="r", use_consolidated=False)[run_path]
    base = zarr.open_group(str(base_backup), mode="r", use_consolidated=False)
    labels = [str(v) for v in current.attrs.get("mask_labels") or []]
    component = str(task["component_name"])
    if labels != [str(v) for v in base.attrs.get("mask_labels") or []] or component not in labels:
        raise RebaseRefused("The base backup's mask labels differ from the run's, or lack this component.")
    comp = labels.index(component)
    current_revision, base_revision = _revision(current), _revision(base)
    if tuple(current["masks_roi"].shape) != tuple(base["masks_roi"].shape):
        refusals.append("the base backup's masks_roi shape differs from the run's")
    for name in _ROW_IDENTITY_ARRAYS:
        if (name in current) != (name in base) or (
            name in current and not np.array_equal(np.asarray(current[name][:]), np.asarray(base[name][:]))
        ):
            refusals.append(f"row identity array {name} differs between the run and the base backup")

    stale, conflicts, other_revision = [], [], []
    for checkpoint in checkpoints:
        revision = int(checkpoint["target_edit_revision"] or 0)
        if revision == current_revision:
            continue
        if revision != base_revision:
            other_revision.append(int(checkpoint["roi_idx"]))
            continue
        roi = int(checkpoint["roi_idx"])
        base_sha = _digest(base["masks_roi"][roi, comp])
        current_sha = _digest(current["masks_roi"][roi, comp])
        row = {
            "roi_idx": roi,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "original_target_edit_revision": revision,
            "original_updated_at_utc": checkpoint["updated_at_utc"],
            "original_saved_at_utc": saved_at.get(checkpoint["checkpoint_id"]),
            "base_component_sha256": base_sha,
            "current_component_sha256": current_sha,
        }
        (stale if base_sha == current_sha else conflicts).append(row)
    if conflicts:
        refusals.append(f"{len(conflicts)} row(s) whose {component} pixels changed since the save; not re-saving")
    if other_revision:
        refusals.append(f"{len(other_revision)} stale row(s) at a revision the base backup does not match")
    if not stale and not conflicts:
        refusals.append("no stale checkpoints to re-save")
    return {
        "task_id": task_id,
        "recording_id": task["recording_id"],
        "component_name": component,
        "assignee": task["assignee_user"],
        "zarr_path": scope.get("zarr_path"),
        "refined_run": scope.get("refined_run"),
        "base_backup": str(base_backup),
        "base_revision": base_revision,
        "current_revision": current_revision,
        "stale_rows": stale,
        "conflict_rows": conflicts,
        "refusals": refusals,
        "ok": not refusals,
    }


def rebase_stale_saves(
    store_path: Path,
    task_id: str,
    base_backup: Path,
    *,
    actor: str,
    backup_dir: Path | None = None,
    skip_backup: bool = False,
) -> dict[str, object]:
    plan = rebase_plan(store_path, task_id, base_backup)
    if not plan["ok"]:
        raise RebaseRefused("; ".join(plan["refusals"]))
    store = LabelingStore(store_path)
    try:
        backup = None if skip_backup else backup_labeling_store(
            store_path, backup_dir or DEFAULT_BACKUP_DIR, label=Path(store_path).stem
        )
        rows = plan["stale_rows"]
        audit = {
            "reason": REBASE_REASON,
            "base_backup": plan["base_backup"],
            "from_revision": plan["base_revision"],
            "to_revision": plan["current_revision"],
            "store_backup_path": (backup or {}).get("backup_path"),
            "rows": rows,
        }
        store.record_event(
            task_id=task_id, recording_id=str(plan["recording_id"]), user=actor,
            event_type=REBASE_EVENT, target={"labeler": plan["assignee"], "row_count": len(rows)}, before=audit,
        )
        masks = {
            r["checkpoint_id"]: json.loads(r["payload_json"])["mask"]
            for r in store.conn.execute(
                "SELECT checkpoint_id, payload_json FROM labeling_session_checkpoints WHERE task_id = ? AND state = 'active';",
                (task_id,),
            )
        }
        failures = []
        with private_assignee_session(
            store, store_path, task_id, assignee=str(plan["assignee"]), actor=actor, client_label=REBASE_CLIENT_LABEL,
        ) as (base_url, _session_id):
            for row in rows:
                status, nav = _request(base_url, "/nav", {"roi_idx": row["roi_idx"]})
                if status != 200 or not nav.get("ok"):
                    failures.append({"roi_idx": row["roi_idx"], "step": "nav", "error": nav.get("details") or nav.get("error")})
                    continue
                status, saved = _request(base_url, "/save", {
                    "mask": masks[row["checkpoint_id"]], "target_token": nav["state"]["target_token"], "advance": False,
                })
                if status != 200 or not saved.get("ok"):
                    failures.append({"roi_idx": row["roi_idx"], "step": "save", "error": saved.get("details") or saved.get("error")})
        after = {
            int(r["roi_idx"]): dict(r) for r in store.conn.execute(
                """
                SELECT roi_idx, checkpoint_id, user, target_edit_revision, payload_json FROM labeling_session_checkpoints
                WHERE task_id = ? AND state = 'active';
                """,
                (task_id,),
            )
        }
        verified = 0
        for row in rows:
            now = after.get(int(row["roi_idx"]))
            if (now and int(now["target_edit_revision"] or 0) == plan["current_revision"]
                    and now["user"] == plan["assignee"]
                    and json.loads(now["payload_json"])["mask"] == masks[row["checkpoint_id"]]):
                verified += 1
        outcome = {"ok": not failures and verified == len(rows), "rebased_rows": verified, "row_count": len(rows), "failures": failures}
        store.record_event(
            task_id=task_id, recording_id=str(plan["recording_id"]), user=actor,
            event_type=REBASE_EVENT + "_result", target={"labeler": plan["assignee"]}, after=outcome,
        )
        return {"plan": {k: v for k, v in plan.items() if k != "stale_rows"}, "backup": backup, "outcome": outcome}
    finally:
        store.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-backup", type=Path, required=True, help="Copy of the run at the checkpoints' revision.")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        plan = rebase_plan(args.store, args.task_id, args.base_backup)
        summary = {k: v for k, v in plan.items() if k not in ("stale_rows", "conflict_rows")}
        summary.update(stale_row_count=len(plan["stale_rows"]), conflict_row_count=len(plan["conflict_rows"]))
        print(json.dumps({"dry_run": True, "plan": summary}, indent=2, default=str))
        return 0 if plan["ok"] else 2
    try:
        report = rebase_stale_saves(args.store, args.task_id, args.base_backup, actor=args.actor)
    except RebaseRefused as exc:
        print(json.dumps({"ok": False, "refused": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["outcome"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
